"""Dependency-free deployment gates for the Sensor-Only V2 ROS route."""

from __future__ import annotations

from dataclasses import dataclass
import math


def action_target_envelope_matches_v2(
    *,
    configured_action_clip: float,
    configured_main_velocity_limit_rad_s: float,
    contract_action_clip: float,
    contract_main_velocity_limit_rad_s: float,
) -> bool:
    """Return true only when static hardware limits preserve bundle targets."""

    values = (
        configured_action_clip,
        configured_main_velocity_limit_rad_s,
        contract_action_clip,
        contract_main_velocity_limit_rad_s,
    )
    return bool(
        all(math.isfinite(float(value)) and float(value) > 0.0 for value in values)
        and math.isclose(
            float(configured_action_clip),
            float(contract_action_clip),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and math.isclose(
            float(configured_main_velocity_limit_rad_s),
            float(contract_main_velocity_limit_rad_s),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    )


@dataclass(frozen=True)
class DeploymentGuardV2:
    """Keep hardware authorization separate from runtime enable latches."""

    allow_motor_enable: bool
    calibration_hardware_ready: bool
    action_target_envelope_compatible: bool = True

    @property
    def hardware_authorized(self) -> bool:
        return bool(
            self.allow_motor_enable
            and self.calibration_hardware_ready
            and self.action_target_envelope_compatible
        )

    def motor_output_allowed(
        self,
        *,
        requested: bool,
        state_allowed: bool,
        estop: bool,
        runtime_action_target_compatible: bool = True,
    ) -> bool:
        return bool(
            requested
            and state_allowed
            and not estop
            and runtime_action_target_compatible
            and self.hardware_authorized
        )


def warmup_complete_v2(
    *,
    elapsed_s: float,
    minimum_duration_s: float,
    history_ready: bool,
    require_history_ready: bool = True,
) -> bool:
    """Require chronological real history in addition to elapsed warmup time."""

    duration_ready = float(elapsed_s) >= float(minimum_duration_s)
    return bool(duration_ready and (history_ready or not require_history_ready))
