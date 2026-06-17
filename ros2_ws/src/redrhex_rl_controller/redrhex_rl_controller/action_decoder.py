from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .redrhex_contract import (
    ABAD_JOINT_NAMES,
    ABAD_POS_SCALE,
    INIT_ABAD_POS_RAD,
    INIT_MAIN_DRIVE_POS_RAD,
    MAIN_DRIVE_JOINT_NAMES,
    MAIN_DRIVE_RESIDUAL_SCALE,
    MAIN_DRIVE_VEL_SCALE,
    MOTOR_JOINT_NAMES,
    STANCE_VELOCITY,
    SWING_VELOCITY,
    command_mode,
)


@dataclass
class DecodedCommand:
    joint_names: list[str]
    target_position_rad: np.ndarray
    target_velocity_rad_s: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    effort_limit_nm: np.ndarray
    safe_action: np.ndarray
    mode: int = 1
    debug: dict[str, object] = field(default_factory=dict)


class ActionDecoder:
    """Decode policy action to high-level motor targets.

    This mirrors the IsaacLab action order and mode gating, then applies a
    bench-safe envelope for the first real robot tests.
    """

    def __init__(
        self,
        action_clip: float = 1.0,
        main_drive_vel_limit_rad_s: float = 1.0,
        main_drive_slew_rate_rad_s2: float = 4.0,
        abad_pos_limit_rad: float = 0.15,
        abad_slew_rate_rad_s: float = 1.0,
        init_stand_max_main_drive_vel_rad_s: float = 0.5,
        enable_forward_bias: bool = True,
        main_sign_correction: Sequence[float] | None = None,
        abad_sign_correction: Sequence[float] | None = None,
        main_zero_offset_rad: Sequence[float] | None = None,
        abad_zero_offset_rad: Sequence[float] | None = None,
    ) -> None:
        self.action_clip = float(action_clip)
        self.main_limit = float(main_drive_vel_limit_rad_s)
        self.main_slew = float(main_drive_slew_rate_rad_s2)
        self.abad_limit = float(abad_pos_limit_rad)
        self.abad_slew = float(abad_slew_rate_rad_s)
        self.init_stand_max_main_drive_vel_rad_s = float(init_stand_max_main_drive_vel_rad_s)
        self.enable_forward_bias = bool(enable_forward_bias)
        self.main_sign = np.asarray(main_sign_correction if main_sign_correction is not None else [1.0] * 6, dtype=np.float64)
        self.abad_sign = np.asarray(abad_sign_correction if abad_sign_correction is not None else [1.0] * 6, dtype=np.float64)
        self.main_zero = np.asarray(main_zero_offset_rad if main_zero_offset_rad is not None else [0.0] * 6, dtype=np.float64)
        self.abad_zero = np.asarray(abad_zero_offset_rad if abad_zero_offset_rad is not None else [0.0] * 6, dtype=np.float64)
        self.last_main_vel = np.zeros(6, dtype=np.float64)
        self.last_abad_pos = np.asarray(INIT_ABAD_POS_RAD, dtype=np.float64)
        self.lateral_phase = 0.0

    def reset(self) -> None:
        self.last_main_vel[:] = 0.0
        self.last_abad_pos[:] = np.asarray(INIT_ABAD_POS_RAD, dtype=np.float64)
        self.lateral_phase = 0.0

    @staticmethod
    def _slew(current: np.ndarray, target: np.ndarray, max_rate: float, dt: float) -> np.ndarray:
        step = max(float(max_rate) * max(float(dt), 1e-4), 0.0)
        return current + np.clip(target - current, -step, step)

    def decode(self, raw_action: Sequence[float], command: Sequence[float], dt: float) -> DecodedCommand:
        action = np.asarray(raw_action, dtype=np.float64).reshape(12)
        if not np.isfinite(action).all():
            raise ValueError("Raw action contains NaN or Inf")
        action = np.clip(action, -self.action_clip, self.action_clip)

        vx, vy, wz = [float(v) for v in command]
        mode_name = command_mode(vx, vy, wz)
        main_actions = action[:6].copy()
        abad_actions = action[6:12].copy()

        if mode_name == "LAT":
            main_actions[:] = 0.0
        if mode_name == "FWD":
            abad_actions[:] = 0.0

        main_vel = main_actions * MAIN_DRIVE_VEL_SCALE * MAIN_DRIVE_RESIDUAL_SCALE

        if self.enable_forward_bias and mode_name in ("FWD", "DIAG"):
            vx_norm = np.clip(vx / 0.45, -1.0, 1.0)
            # IsaacLab forward bias follows stance/swing CPG; on the bench profile
            # we use a very small mean bias and let slew/PWM limits protect hardware.
            main_vel += np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]) * STANCE_VELOCITY * vx_norm
        if mode_name == "YAW":
            wz_norm = np.clip(wz / 1.0, -1.0, 1.0)
            yaw_body_pattern = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float64)
            main_vel += yaw_body_pattern * 0.5 * wz_norm
        if mode_name == "LAT":
            self.lateral_phase = (self.lateral_phase + 2.0 * math.pi * 0.5 * float(dt)) % (2.0 * math.pi)
            main_vel += 0.2 * np.sin(self.lateral_phase) * np.array([-1.0, 1.0, 1.0, -1.0, 1.0, -1.0])
        else:
            self.lateral_phase = 0.0

        main_vel = np.clip(main_vel * self.main_sign, -self.main_limit, self.main_limit)
        main_vel = self._slew(self.last_main_vel, main_vel, self.main_slew, dt)
        self.last_main_vel = main_vel.copy()

        abad_pos = abad_actions * ABAD_POS_SCALE
        if mode_name == "LAT":
            lateral_dir = 1.0 if vy >= 0.0 else -1.0
            phase_sin = math.sin(self.lateral_phase)
            lateral_abad = np.array([-lateral_dir, -lateral_dir, -lateral_dir, lateral_dir, lateral_dir, lateral_dir]) * phase_sin * 0.30
            abad_pos = 0.7 * lateral_abad + 0.3 * abad_pos
        abad_pos = np.clip(abad_pos * self.abad_sign + self.abad_zero, -self.abad_limit, self.abad_limit)
        abad_pos = self._slew(self.last_abad_pos, abad_pos, self.abad_slew, dt)
        self.last_abad_pos = abad_pos.copy()

        pos = np.full(12, np.nan, dtype=np.float64)
        vel = np.zeros(12, dtype=np.float64)
        pos[:6] = np.asarray(INIT_MAIN_DRIVE_POS_RAD, dtype=np.float64) + self.main_zero
        vel[:6] = main_vel
        pos[6:] = abad_pos

        safe_action = np.concatenate([main_actions, abad_actions]).astype(np.float32)
        return DecodedCommand(
            joint_names=list(MOTOR_JOINT_NAMES),
            target_position_rad=pos,
            target_velocity_rad_s=vel,
            kp=np.array([0.0] * 6 + [2.0] * 6, dtype=np.float64),
            kd=np.array([0.0] * 6 + [0.1] * 6, dtype=np.float64),
            effort_limit_nm=np.array([0.0] * 12, dtype=np.float64),
            safe_action=safe_action,
            mode=1,
            debug={"mode_name": mode_name, "raw_action_max_abs": float(np.max(np.abs(action)))},
        )

    def disabled_command(self) -> DecodedCommand:
        pos = np.asarray(INIT_MAIN_DRIVE_POS_RAD + INIT_ABAD_POS_RAD, dtype=np.float64)
        return DecodedCommand(
            joint_names=list(MOTOR_JOINT_NAMES),
            target_position_rad=pos,
            target_velocity_rad_s=np.zeros(12, dtype=np.float64),
            kp=np.zeros(12, dtype=np.float64),
            kd=np.zeros(12, dtype=np.float64),
            effort_limit_nm=np.zeros(12, dtype=np.float64),
            safe_action=np.zeros(12, dtype=np.float32),
            mode=0,
        )

    def init_stand_command(self, current_main_pos: Sequence[float], dt: float) -> DecodedCommand:
        current = np.asarray(current_main_pos, dtype=np.float64).reshape(6)
        target = np.asarray(INIT_MAIN_DRIVE_POS_RAD, dtype=np.float64)
        err = np.arctan2(np.sin(target - current), np.cos(target - current))
        main_vel = np.clip(1.2 * err, -self.init_stand_max_main_drive_vel_rad_s, self.init_stand_max_main_drive_vel_rad_s)
        main_vel = self._slew(self.last_main_vel, main_vel, self.main_slew, dt)
        self.last_main_vel = main_vel.copy()
        pos = np.asarray(INIT_MAIN_DRIVE_POS_RAD + INIT_ABAD_POS_RAD, dtype=np.float64)
        vel = np.zeros(12, dtype=np.float64)
        vel[:6] = main_vel
        return DecodedCommand(
            joint_names=list(MOTOR_JOINT_NAMES),
            target_position_rad=pos,
            target_velocity_rad_s=vel,
            kp=np.array([0.0] * 6 + [2.0] * 6, dtype=np.float64),
            kd=np.array([0.0] * 6 + [0.1] * 6, dtype=np.float64),
            effort_limit_nm=np.zeros(12, dtype=np.float64),
            safe_action=np.zeros(12, dtype=np.float32),
            mode=2,
            debug={"mode_name": "INIT_STAND", "pose_error_max_rad": float(np.max(np.abs(err)))},
        )
