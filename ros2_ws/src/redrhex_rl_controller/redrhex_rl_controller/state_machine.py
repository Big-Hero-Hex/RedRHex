from __future__ import annotations

from enum import Enum


class RedRhexState(str, Enum):
    BOOT = "BOOT"
    SENSOR_CHECK = "SENSOR_CHECK"
    MOTOR_IDLE = "MOTOR_IDLE"
    INIT_STAND = "INIT_STAND"
    WARMUP = "WARMUP"
    POLICY_READY = "POLICY_READY"
    POLICY_RUN = "POLICY_RUN"
    PROTECTIVE_STOP = "PROTECTIVE_STOP"
    FALL_DETECTED = "FALL_DETECTED"
    RECOVER = "RECOVER"


class RedRhexStateMachine:
    def __init__(self) -> None:
        self.state = RedRhexState.BOOT
        self.reason = "node boot"
        self.stand_complete = False

    def force_stop(self, reason: str) -> None:
        self.state = RedRhexState.PROTECTIVE_STOP
        self.reason = reason

    def recover_to_sensor_check(self) -> None:
        self.state = RedRhexState.SENSOR_CHECK
        self.reason = "manual recover"
        self.stand_complete = False

    def update(
        self,
        sensors_ready: bool,
        safety_ok: bool,
        enable_motors: bool,
        enable_policy: bool,
        init_stand_complete: bool,
        fall_detected: bool = False,
        safety_reason: str = "",
    ) -> RedRhexState:
        if fall_detected:
            self.state = RedRhexState.FALL_DETECTED
            self.reason = "fall detected"
            return self.state
        if not safety_ok:
            self.state = RedRhexState.PROTECTIVE_STOP
            self.reason = safety_reason or "safety not ok"
            return self.state
        if self.state == RedRhexState.BOOT:
            self.state = RedRhexState.SENSOR_CHECK
            self.reason = "boot complete"
        if not sensors_ready:
            self.state = RedRhexState.SENSOR_CHECK
            self.reason = "waiting sensors"
            return self.state
        if not enable_motors:
            self.state = RedRhexState.MOTOR_IDLE
            self.reason = "motor output disabled"
            return self.state
        self.stand_complete = self.stand_complete or init_stand_complete
        if not self.stand_complete:
            self.state = RedRhexState.INIT_STAND
            self.reason = "moving to init stand"
            return self.state
        if not enable_policy:
            self.state = RedRhexState.POLICY_READY
            self.reason = "policy ready, waiting enable_policy"
            return self.state
        self.state = RedRhexState.POLICY_RUN
        self.reason = "policy running"
        return self.state
