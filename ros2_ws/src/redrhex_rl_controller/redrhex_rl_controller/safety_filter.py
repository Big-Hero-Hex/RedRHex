from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .action_decoder import DecodedCommand
from .observation_builder import roll_pitch_from_quat_xyzw


@dataclass
class SafetyStatus:
    ok: bool
    reasons: list[str] = field(default_factory=list)


class SafetyFilter:
    def __init__(
        self,
        max_abs_roll_rad: float = 0.7,
        max_abs_pitch_rad: float = 0.7,
        sensor_timeout_s: float = 0.10,
        heartbeat_timeout_s: float = 0.10,
        action_clip: float = 1.0,
        main_drive_vel_limit_rad_s: float = 1.0,
        abad_pos_limit_rad: float = 0.15,
        max_motor_temperature_c: float = 70.0,
        max_motor_current_a: float = 3.0,
        control_deadline_factor: float = 2.0,
    ) -> None:
        self.max_abs_roll_rad = float(max_abs_roll_rad)
        self.max_abs_pitch_rad = float(max_abs_pitch_rad)
        self.sensor_timeout_s = float(sensor_timeout_s)
        self.heartbeat_timeout_s = float(heartbeat_timeout_s)
        self.action_clip = float(action_clip)
        self.main_drive_vel_limit_rad_s = float(main_drive_vel_limit_rad_s)
        self.abad_pos_limit_rad = float(abad_pos_limit_rad)
        self.max_motor_temperature_c = float(max_motor_temperature_c)
        self.max_motor_current_a = float(max_motor_current_a)
        self.control_deadline_factor = float(control_deadline_factor)
        self.estop = True
        self.last_loop_dt = 0.0
        self.motor_currents: list[float] = []
        self.motor_temps: list[float] = []
        self.motor_faults: list[bool] = []

    def update_estop(self, active: bool) -> None:
        self.estop = bool(active)

    def update_motor_state(
        self,
        currents: Sequence[float] | None = None,
        temperatures: Sequence[float] | None = None,
        faults: Sequence[bool] | None = None,
    ) -> None:
        self.motor_currents = list(currents or [])
        self.motor_temps = list(temperatures or [])
        self.motor_faults = list(faults or [])

    def check(
        self,
        observation: np.ndarray | None,
        raw_action: np.ndarray | None,
        decoded: DecodedCommand | None,
        imu_quat_xyzw: Sequence[float] | None,
        last_imu_time: float,
        last_joint_time: float,
        last_heartbeat_time: float,
        now_s: float | None = None,
        control_dt: float = 0.008,
        require_heartbeat: bool = False,
    ) -> SafetyStatus:
        now = now_s if now_s is not None else time.monotonic()
        reasons: list[str] = []
        if self.estop:
            reasons.append("E-stop active")
        if now - last_imu_time > self.sensor_timeout_s:
            reasons.append("IMU timeout")
        if now - last_joint_time > self.sensor_timeout_s:
            reasons.append("joint_states timeout")
        if require_heartbeat and now - last_heartbeat_time > self.heartbeat_timeout_s:
            reasons.append("lowlevel heartbeat timeout")
        if observation is not None and not np.isfinite(observation).all():
            reasons.append("observation NaN/Inf")
        if raw_action is not None:
            if not np.isfinite(raw_action).all():
                reasons.append("ONNX action NaN/Inf")
            if np.max(np.abs(raw_action)) > self.action_clip + 1e-3:
                reasons.append("raw action magnitude too large")
        if decoded is not None:
            if np.max(np.abs(decoded.target_velocity_rad_s[:6])) > self.main_drive_vel_limit_rad_s + 1e-6:
                reasons.append("main drive velocity target exceeds safety limit")
            if np.max(np.abs(decoded.target_position_rad[6:])) > self.abad_pos_limit_rad + 1e-6:
                reasons.append("ABAD target exceeds safety limit")
        if imu_quat_xyzw is not None:
            roll, pitch = roll_pitch_from_quat_xyzw(imu_quat_xyzw)
            if abs(roll) > self.max_abs_roll_rad:
                reasons.append(f"roll too large: {roll:.3f} rad")
            if abs(pitch) > self.max_abs_pitch_rad:
                reasons.append(f"pitch too large: {pitch:.3f} rad")
        if self.motor_currents and max(abs(v) for v in self.motor_currents) > self.max_motor_current_a:
            reasons.append("motor current too high")
        if self.motor_temps and max(self.motor_temps) > self.max_motor_temperature_c:
            reasons.append("motor temperature too high")
        if any(self.motor_faults):
            reasons.append("motor fault flag")
        if self.last_loop_dt > control_dt * self.control_deadline_factor:
            reasons.append(f"control loop deadline miss: {self.last_loop_dt:.4f}s")
        return SafetyStatus(ok=not reasons, reasons=reasons)
