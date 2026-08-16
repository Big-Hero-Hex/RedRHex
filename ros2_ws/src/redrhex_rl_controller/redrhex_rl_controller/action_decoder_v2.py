"""ROS motor-command adapter around the shared V2 residual-CPG decoder."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from redrhex_policy_io.action import decode_forward_residual_action_v2
from redrhex_policy_io.contracts import ForwardResidualActionContractV2

from .action_decoder import DecodedMotorCommand


@dataclass(frozen=True)
class ActionTargetStatusV2:
    """Separate the bundle contract target from hardware-only tightening."""

    raw_contract_target_main_drive_velocity: np.ndarray
    action_clipped_contract_target_main_drive_velocity: np.ndarray
    hardware_slew_target_main_drive_velocity: np.ndarray
    hardware_target_main_drive_velocity: np.ndarray
    action_warmup_scale: float
    contract_slew_rate_rad_s2: float | None
    hardware_slew_rate_rad_s2: float
    hardware_action_clip_applied: bool
    hardware_slew_applied: bool
    hardware_velocity_limit_applied: bool

    @property
    def hardware_tightening_applied(self) -> bool:
        return bool(
            self.hardware_action_clip_applied
            or self.hardware_slew_applied
            or self.hardware_velocity_limit_applied
        )


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
                self.contract.initial_main_position_rad,
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
        if not np.allclose(
            self.init_main_position,
            np.asarray(self.contract.initial_main_position_rad),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("init_main_drive_pos must exactly match the V2 action contract")

    def reset(self, gait_phase_rad: float = 0.0) -> None:
        self._gait_phase_origin_rad = float(gait_phase_rad) % (2.0 * math.pi)
        self.gait_phase_rad = self._gait_phase_origin_rad
        self.contract_control_step = 0
        self._previous_target_velocity = np.zeros(6, dtype=np.float64)
        self.last_target_status: ActionTargetStatusV2 | None = None

    def _advance_contract_clock(
        self,
        dt_s: float,
        *,
        command_vx_m_s: float,
        warmup_scale: float,
    ) -> None:
        if not math.isfinite(float(dt_s)) or float(dt_s) <= 0.0:
            raise ValueError("contract clock dt must be positive and finite")
        vx = float(command_vx_m_s)
        warmup = float(warmup_scale)
        if not math.isfinite(vx) or not math.isfinite(warmup):
            raise ValueError("contract clock command and warmup must be finite")
        if vx <= self.contract.FORWARD_ACTIVE_MIN_M_S:
            self.contract_control_step = 0
            return
        command_scale = float(
            np.clip(vx / self.contract.forward_command_reference_m_s, 0.0, 1.0)
        )
        self.gait_phase_rad = (
            self.gait_phase_rad
            + 2.0
            * math.pi
            * self.contract.NOMINAL_GAIT_FREQUENCY_HZ
            / self.contract.policy_rate_hz
            * command_scale
            * warmup
        ) % (2.0 * math.pi)
        self.contract_control_step += 1

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
        advance_contract_clock: bool = True,
    ) -> DecodedMotorCommand:
        position = np.asarray(current_main_position, dtype=np.float64).reshape(6)
        error = np.arctan2(np.sin(self.init_main_position - position), np.cos(self.init_main_position - position))
        velocity = np.clip(
            self.init_position_gain * error,
            -self.init_velocity_limit,
            self.init_velocity_limit,
        )
        result = self._pack(
            main_position=position,
            main_velocity=velocity,
            abad_position=np.asarray(self.contract.abad_neutral_position_rad),
            safe_action=np.zeros(12),
            enable=enable,
            mode=self.MODE_INIT_STAND,
        )
        # INIT_STAND is a zero-motion state.  It must not consume the
        # command-relative gait ramp or advance the hidden CPG phase.
        if advance_contract_clock:
            self._advance_contract_clock(
                1.0 / self.contract.policy_rate_hz,
                command_vx_m_s=0.0,
                warmup_scale=0.0,
            )
        return result

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
        self.last_target_status = None
        original_action = np.asarray(action, dtype=np.float64).reshape(12)
        raw_decoded = decode_forward_residual_action_v2(
            original_action,
            np.asarray(command, dtype=np.float64).reshape(3),
            self.gait_phase_rad,
            np.asarray(main_position_rad, dtype=np.float64).reshape(6),
            control_step=self.contract_control_step,
            contract=self.contract,
        )
        constrained_action = np.clip(
            original_action,
            -self.action_clip,
            self.action_clip,
        )
        if np.array_equal(constrained_action, original_action):
            decoded = raw_decoded
        else:
            decoded = decode_forward_residual_action_v2(
                constrained_action,
                np.asarray(command, dtype=np.float64).reshape(3),
                self.gait_phase_rad,
                np.asarray(main_position_rad, dtype=np.float64).reshape(6),
                control_step=self.contract_control_step,
                contract=self.contract,
            )
        raw_velocity = np.asarray(
            raw_decoded.target_main_velocity_rad_s,
            dtype=np.float64,
        )
        action_clipped_velocity = np.asarray(
            decoded.target_main_velocity_rad_s,
            dtype=np.float64,
        )
        maximum_delta = self.main_slew_rate * dt
        slew_velocity = np.clip(
            action_clipped_velocity,
            self._previous_target_velocity - maximum_delta,
            self._previous_target_velocity + maximum_delta,
        )
        velocity = np.clip(
            slew_velocity,
            -self.main_velocity_limit,
            self.main_velocity_limit,
        )
        self._previous_target_velocity = velocity.copy()
        self.last_target_status = ActionTargetStatusV2(
            raw_contract_target_main_drive_velocity=raw_velocity.copy(),
            action_clipped_contract_target_main_drive_velocity=action_clipped_velocity.copy(),
            hardware_slew_target_main_drive_velocity=slew_velocity.copy(),
            hardware_target_main_drive_velocity=velocity.copy(),
            action_warmup_scale=float(raw_decoded.action_warmup_scale),
            contract_slew_rate_rad_s2=None,
            hardware_slew_rate_rad_s2=self.main_slew_rate,
            hardware_action_clip_applied=not np.array_equal(
                np.asarray(raw_decoded.safe_action),
                np.asarray(decoded.safe_action),
            ),
            hardware_slew_applied=not np.allclose(
                slew_velocity,
                action_clipped_velocity,
                rtol=0.0,
                atol=1.0e-12,
            ),
            hardware_velocity_limit_applied=not np.allclose(
                velocity,
                slew_velocity,
                rtol=0.0,
                atol=1.0e-12,
            ),
        )
        safe_action = np.asarray(decoded.safe_action, dtype=np.float32)
        safe_action[6:12] = 0.0
        abad = np.asarray(decoded.target_abad_position_rad, dtype=np.float64)
        result = self._pack(
            main_position=np.asarray(main_position_rad, dtype=np.float64).reshape(6),
            main_velocity=velocity,
            abad_position=abad,
            safe_action=safe_action,
            enable=True,
            mode=self.MODE_MIXED_POSITION_VELOCITY,
        )
        self._advance_contract_clock(
            dt,
            command_vx_m_s=float(np.asarray(command, dtype=np.float64).reshape(3)[0]),
            warmup_scale=float(raw_decoded.action_warmup_scale),
        )
        return result
