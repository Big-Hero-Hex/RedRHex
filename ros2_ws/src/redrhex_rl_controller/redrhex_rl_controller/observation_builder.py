from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .redrhex_contract import (
    ABAD_JOINT_NAMES,
    ABAD_POS_SCALE,
    BASE_GAIT_ANGULAR_VEL,
    BASE_GAIT_FREQUENCY_HZ,
    COMMAND_LIMITS,
    EXPECTED_ACTION_DIM,
    EXPECTED_OBS_DIM,
    MAIN_DRIVE_JOINT_NAMES,
)


@dataclass
class BuilderStatus:
    ready: bool
    reasons: list[str] = field(default_factory=list)


def _quat_to_matrix_xyzw(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = [float(v) for v in q]
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-9:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def roll_pitch_from_quat_xyzw(q: Sequence[float]) -> tuple[float, float]:
    x, y, z, w = [float(v) for v in q]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    return roll, pitch


class ObservationBuilder:
    """Builds the exact 56-D policy observation from ROS-style state."""

    def __init__(
        self,
        command_limits: Mapping[str, float] | None = None,
        cmd_timeout_s: float = 0.25,
        base_lin_vel_source: str = "zero",
        abad_feedback_source: str = "commanded",
    ) -> None:
        self.command_limits = dict(COMMAND_LIMITS)
        if command_limits:
            self.command_limits.update({k: v for k, v in command_limits.items() if v is not None})
        self.cmd_timeout_s = float(cmd_timeout_s)
        self.base_lin_vel_source = base_lin_vel_source
        self.abad_feedback_source = abad_feedback_source
        self.reset()

    def reset(self) -> None:
        self.imu_quat_xyzw: np.ndarray | None = None
        self.imu_ang_vel = np.zeros(3, dtype=np.float64)
        self.base_lin_vel = np.zeros(3, dtype=np.float64)
        self.main_pos = np.zeros(6, dtype=np.float64)
        self.main_vel = np.zeros(6, dtype=np.float64)
        self.abad_pos = np.zeros(6, dtype=np.float64)
        self.abad_vel = np.zeros(6, dtype=np.float64)
        self.main_present = [False] * 6
        self.abad_present = [False] * 6
        self.cmd = np.zeros(3, dtype=np.float64)
        self.last_actions = np.zeros(EXPECTED_ACTION_DIM, dtype=np.float32)
        self.gait_phase = 0.0
        self.last_update_time = time.monotonic()
        self.last_cmd_time = 0.0
        self.last_imu_time = 0.0
        self.last_joint_time = 0.0

    def update_imu(self, quat_xyzw: Sequence[float], angular_velocity: Sequence[float], stamp_s: float | None = None) -> None:
        self.imu_quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
        self.imu_ang_vel = np.asarray(angular_velocity, dtype=np.float64)
        self.last_imu_time = stamp_s if stamp_s is not None else time.monotonic()

    def update_joint_states(
        self,
        names: Sequence[str],
        positions: Sequence[float],
        velocities: Sequence[float] | None = None,
        stamp_s: float | None = None,
    ) -> None:
        vel = velocities if velocities is not None and len(velocities) == len(names) else [0.0] * len(names)
        pos_map = {name: float(positions[i]) for i, name in enumerate(names)}
        vel_map = {name: float(vel[i]) for i, name in enumerate(names)}
        for i, name in enumerate(MAIN_DRIVE_JOINT_NAMES):
            if name in pos_map:
                self.main_pos[i] = pos_map[name]
                self.main_vel[i] = vel_map.get(name, 0.0)
                self.main_present[i] = True
        if self.abad_feedback_source == "joint_states":
            for i, name in enumerate(ABAD_JOINT_NAMES):
                if name in pos_map:
                    self.abad_pos[i] = pos_map[name]
                    self.abad_vel[i] = vel_map.get(name, 0.0)
                    self.abad_present[i] = True
        self.last_joint_time = stamp_s if stamp_s is not None else time.monotonic()

    def update_command(self, vx: float, vy: float, wz: float, stamp_s: float | None = None) -> None:
        cmd = np.array([vx, vy, wz], dtype=np.float64)
        cmd[0] = np.clip(cmd[0], self.command_limits["vx_min"], self.command_limits["vx_max"])
        cmd[1] = np.clip(cmd[1], self.command_limits["vy_min"], self.command_limits["vy_max"])
        cmd[2] = np.clip(cmd[2], self.command_limits["wz_min"], self.command_limits["wz_max"])
        self.cmd = cmd
        self.last_cmd_time = stamp_s if stamp_s is not None else time.monotonic()

    def update_odom_body_velocity(self, lin_vel_body: Sequence[float]) -> None:
        self.base_lin_vel = np.asarray(lin_vel_body, dtype=np.float64)

    def update_commanded_abad(self, abad_pos: Sequence[float], dt: float) -> None:
        new_pos = np.asarray(abad_pos, dtype=np.float64)
        dt = max(float(dt), 1e-4)
        self.abad_vel = (new_pos - self.abad_pos) / dt
        self.abad_pos = new_pos
        self.abad_present = [True] * 6

    def update_last_actions(self, action: Sequence[float]) -> None:
        self.last_actions = np.asarray(action, dtype=np.float32).reshape(EXPECTED_ACTION_DIM)

    def advance_phase(self, dt: float) -> None:
        self.gait_phase = (self.gait_phase + 2.0 * math.pi * BASE_GAIT_FREQUENCY_HZ * float(dt)) % (2.0 * math.pi)

    def status(self, now_s: float | None = None, sensor_timeout_s: float = 0.10) -> BuilderStatus:
        now = now_s if now_s is not None else time.monotonic()
        reasons: list[str] = []
        if self.imu_quat_xyzw is None:
            reasons.append("waiting IMU")
        elif now - self.last_imu_time > sensor_timeout_s:
            reasons.append("IMU timeout")
        if not all(self.main_present):
            missing = [MAIN_DRIVE_JOINT_NAMES[i] for i, ok in enumerate(self.main_present) if not ok]
            reasons.append(f"missing main encoders: {missing}")
        elif now - self.last_joint_time > sensor_timeout_s:
            reasons.append("joint_states timeout")
        if self.abad_feedback_source == "joint_states" and not all(self.abad_present):
            reasons.append("missing ABAD feedback")
        return BuilderStatus(ready=not reasons, reasons=reasons)

    def build(self, now_s: float | None = None) -> np.ndarray:
        now = now_s if now_s is not None else time.monotonic()
        if self.last_cmd_time <= 0.0 or now - self.last_cmd_time > self.cmd_timeout_s:
            cmd = np.zeros(3, dtype=np.float64)
        else:
            cmd = self.cmd.copy()

        quat = self.imu_quat_xyzw if self.imu_quat_xyzw is not None else np.array([0.0, 0.0, 0.0, 1.0])
        rot_body_to_world = _quat_to_matrix_xyzw(quat)
        projected_gravity = rot_body_to_world.T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)

        if self.base_lin_vel_source == "zero":
            base_lin_vel = np.zeros(3, dtype=np.float64)
        else:
            base_lin_vel = self.base_lin_vel.copy()

        obs = np.concatenate(
            [
                base_lin_vel,
                self.imu_ang_vel,
                projected_gravity,
                np.sin(self.main_pos),
                np.cos(self.main_pos),
                self.main_vel / BASE_GAIT_ANGULAR_VEL,
                self.abad_pos / ABAD_POS_SCALE,
                self.abad_vel,
                cmd,
                np.array([math.sin(self.gait_phase), math.cos(self.gait_phase)], dtype=np.float64),
                self.last_actions.astype(np.float64),
            ]
        ).astype(np.float32)

        if obs.shape != (EXPECTED_OBS_DIM,):
            raise ValueError(f"Observation dim mismatch: {obs.shape}")
        if not np.isfinite(obs).all():
            raise ValueError("Observation contains NaN or Inf")
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        obs = np.clip(obs, -100.0, 100.0).astype(np.float32)
        return obs
