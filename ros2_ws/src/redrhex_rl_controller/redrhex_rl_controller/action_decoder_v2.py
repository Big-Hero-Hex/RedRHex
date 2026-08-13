"""ROS motor-command adapter around the shared V2 residual-CPG decoder."""

from __future__ import annotations

import math

import numpy as np

from redrhex_policy_io.action import decode_forward_residual_action_v2
from redrhex_policy_io.contracts import ForwardResidualActionContractV2

from .action_decoder import DecodedMotorCommand


class ForwardResidualActionDecoderV2:
    """Apply only safety-tightening limits around versioned bundle semantics."""

    MODE_DISABLED = 0
    MODE_MIXED_POSITION_VELOCITY = 1
    MODE_INIT_STAND = 2
    MODE_PROTECTIVE_STOP = 255

    def __init__(
        self,
        contract: ForwardResidualActionContractV2,
        config: dict | None = None,
    ) -> None:
        self.contract = contract.validate()
        cfg = config or {}
        self.main_velocity_limit = float(
            cfg.get("main_drive_vel_limit_rad_s", self.contract.main_velocity_limit_rad_s)
        )
        self.action_clip = float(cfg.get("action_clip", self.contract.action_clip))
        self.main_slew_rate = float(cfg.get("main_drive_slew_rate_rad_s2", 120.0))
        self.init_main_position = np.asarray(
            cfg.get(
                "init_main_drive_pos",
                [math.pi / 4.0] * 3 + [-math.pi / 4.0] * 3,
            ),
            dtype=np.float64,
        )
        self.init_position_gain = float(cfg.get("init_stand_main_drive_position_gain", 3.0))
        self.init_velocity_limit = float(cfg.get("init_stand_max_main_drive_vel_rad_s", 1.5))
        self.main_kp = np.asarray(cfg.get("main_drive_kp", [0.0] * 6), dtype=np.float64)
        self.main_kd = np.asarray(cfg.get("main_drive_kd", [50.0] * 6), dtype=np.float64)
        self.abad_kp = np.asarray(cfg.get("abad_kp", [40.0] * 6), dtype=np.float64)
        self.abad_kd = np.asarray(cfg.get("abad_kd", [4.0] * 6), dtype=np.float64)
        self.main_effort = np.asarray(cfg.get("main_drive_effort_limit_nm", [100.0] * 6), dtype=np.float64)
        self.abad_effort = np.asarray(cfg.get("abad_effort_limit_nm", [8.0] * 6), dtype=np.float64)
        self._validate_safety_limits()
        self.reset()

    @property
    def decoder_sha256(self) -> str:
        return self.contract.decoder_sha256

    def _validate_safety_limits(self) -> None:
        if not 0.0 < self.main_velocity_limit <= self.contract.main_velocity_limit_rad_s:
            raise ValueError("hardware main velocity limit may only tighten the V2 action contract")
        if not 0.0 < self.action_clip <= self.contract.action_clip:
            raise ValueError("hardware action clip may only tighten the V2 action contract")
        for name, value in (
            ("main_drive_slew_rate_rad_s2", self.main_slew_rate),
            ("init_stand_main_drive_position_gain", self.init_position_gain),
            ("init_stand_max_main_drive_vel_rad_s", self.init_velocity_limit),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        vectors = {
            "init_main_drive_pos": self.init_main_position,
            "main_drive_kp": self.main_kp,
            "main_drive_kd": self.main_kd,
            "abad_kp": self.abad_kp,
            "abad_kd": self.abad_kd,
            "main_drive_effort_limit_nm": self.main_effort,
            "abad_effort_limit_nm": self.abad_effort,
        }
        for name, values in vectors.items():
            if values.shape != (6,) or not np.isfinite(values).all():
                raise ValueError(f"{name} must be finite with length 6")

    def reset(self, gait_phase_rad: float = 0.0) -> None:
        self.gait_phase_rad = float(gait_phase_rad) % (2.0 * math.pi)
        self._previous_target_velocity = np.zeros(6, dtype=np.float64)

    def _pack(
        self,
        *,
        main_position: np.ndarray,
        main_velocity: np.ndarray,
        abad_position: np.ndarray,
        safe_action: np.ndarray,
        enable: bool,
        mode: int,
    ) -> DecodedMotorCommand:
        return DecodedMotorCommand(
            joint_names=list(self.contract.MAIN_JOINT_ORDER + self.contract.ABAD_JOINT_ORDER),
            target_position_rad=[float(value) for value in np.concatenate((main_position, abad_position))],
            target_velocity_rad_s=[float(value) for value in np.concatenate((main_velocity, np.zeros(6)))],
            kp=[float(value) for value in np.concatenate((self.main_kp, self.abad_kp))],
            kd=[float(value) for value in np.concatenate((self.main_kd, self.abad_kd))],
            effort_limit_nm=[float(value) for value in np.concatenate((self.main_effort, self.abad_effort))],
            enable=bool(enable),
            mode=int(mode),
            safe_action=safe_action.astype(np.float32, copy=True),
            target_main_drive_velocity=main_velocity.astype(np.float64, copy=True),
            target_abad_position=abad_position.astype(np.float64, copy=True),
        )

    def disabled_command(self) -> DecodedMotorCommand:
        return self._pack(
            main_position=np.zeros(6),
            main_velocity=np.zeros(6),
            abad_position=np.asarray(self.contract.abad_neutral_position_rad),
            safe_action=np.zeros(12),
            enable=False,
            mode=self.MODE_DISABLED,
        )

    def init_stand_command(
        self,
        current_main_position: np.ndarray,
        *,
        enable: bool = True,
    ) -> DecodedMotorCommand:
        position = np.asarray(current_main_position, dtype=np.float64).reshape(6)
        error = np.arctan2(np.sin(self.init_main_position - position), np.cos(self.init_main_position - position))
        velocity = np.clip(
            self.init_position_gain * error,
            -self.init_velocity_limit,
            self.init_velocity_limit,
        )
        return self._pack(
            main_position=position,
            main_velocity=velocity,
            abad_position=np.asarray(self.contract.abad_neutral_position_rad),
            safe_action=np.zeros(12),
            enable=enable,
            mode=self.MODE_INIT_STAND,
        )

    def protective_stop_command(
        self,
        current_main_position: np.ndarray,
        current_abad_position: np.ndarray,
    ) -> DecodedMotorCommand:
        return self._pack(
            main_position=np.asarray(current_main_position, dtype=np.float64).reshape(6),
            main_velocity=np.zeros(6),
            abad_position=np.asarray(current_abad_position, dtype=np.float64).reshape(6),
            safe_action=np.zeros(12),
            enable=False,
            mode=self.MODE_PROTECTIVE_STOP,
        )

    def decode(
        self,
        action: np.ndarray,
        main_position_rad: np.ndarray,
        command: np.ndarray,
        dt_s: float,
    ) -> DecodedMotorCommand:
        dt = float(dt_s)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_s must be positive and finite")
        self.gait_phase_rad = (
            self.gait_phase_rad
            + 2.0 * math.pi * self.contract.NOMINAL_GAIT_FREQUENCY_HZ * dt
        ) % (2.0 * math.pi)
        constrained_action = np.clip(
            np.asarray(action, dtype=np.float64).reshape(12),
            -self.action_clip,
            self.action_clip,
        )
        decoded = decode_forward_residual_action_v2(
            constrained_action,
            np.asarray(command, dtype=np.float64).reshape(3),
            self.gait_phase_rad,
            np.asarray(main_position_rad, dtype=np.float64).reshape(6),
            contract=self.contract,
        )
        velocity = np.asarray(decoded.target_main_velocity_rad_s, dtype=np.float64)
        maximum_delta = self.main_slew_rate * dt
        velocity = np.clip(
            velocity,
            self._previous_target_velocity - maximum_delta,
            self._previous_target_velocity + maximum_delta,
        )
        velocity = np.clip(velocity, -self.main_velocity_limit, self.main_velocity_limit)
        self._previous_target_velocity = velocity.copy()
        safe_action = np.asarray(decoded.safe_action, dtype=np.float32)
        safe_action[6:12] = 0.0
        abad = np.asarray(decoded.target_abad_position_rad, dtype=np.float64)
        return self._pack(
            main_position=np.asarray(main_position_rad, dtype=np.float64).reshape(6),
            main_velocity=velocity,
            abad_position=abad,
            safe_action=safe_action,
            enable=True,
            mode=self.MODE_MIXED_POSITION_VELOCITY,
        )
