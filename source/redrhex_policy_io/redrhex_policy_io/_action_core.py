"""Array-backend-neutral math for the Sensor-V2 action contract."""

from __future__ import annotations

import math
from typing import Any

from .contracts import ForwardResidualActionContractV2


def decode_forward_residual_arrays_v2(
    values: Any,
    command: Any,
    gait_phase: Any,
    main_position: Any,
    warmup_scale: Any,
    *,
    contract: ForwardResidualActionContractV2,
    namespace: Any,
    action_mask: Any,
    phase_offsets: Any,
    direction: Any,
    main_output_sign: Any,
    abad_neutral: Any,
    abad_output_sign: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    """Run the one authoritative elementwise decoder on NumPy or Torch arrays.

    Shape, finiteness, and command validation stay in the public backend
    adapters.  Keeping all gait, sign, residual, warmup, and target-limit math
    here prevents the simulator and deployment implementations from drifting.
    """

    safe_action = namespace.clip(values, -contract.action_clip, contract.action_clip)
    safe_action = safe_action * action_mask

    # ``gait_phase`` is a uniform *time* clock.  A RHex C-leg does not move
    # uniformly in angle: the 60-degree stance arc consumes 65% of the cycle
    # and the 300-degree recovery arc consumes the remaining 35%.  Mapping the
    # time fraction into leg angle here is part of the serialized action
    # contract and is shared by NumPy, Torch, simulator, and ROS.
    time_phase = namespace.remainder(
        gait_phase[..., None] + phase_offsets,
        2.0 * math.pi,
    )
    time_fraction = time_phase / (2.0 * math.pi)
    in_stance = time_fraction < contract.STANCE_DUTY_CYCLE
    stance_start = contract.STANCE_PHASE_START_RAD % (2.0 * math.pi)
    stance_arc = (
        contract.STANCE_PHASE_END_RAD - contract.STANCE_PHASE_START_RAD
    ) % (2.0 * math.pi)
    swing_arc = 2.0 * math.pi - stance_arc
    desired_unwrapped = namespace.where(
        in_stance,
        stance_start
        + stance_arc * time_fraction / contract.STANCE_DUTY_CYCLE,
        stance_start
        + stance_arc
        + swing_arc
        * (time_fraction - contract.STANCE_DUTY_CYCLE)
        / (1.0 - contract.STANCE_DUTY_CYCLE),
    )
    desired_phase = namespace.remainder(desired_unwrapped, 2.0 * math.pi)
    base_angular_velocity = 2.0 * math.pi * contract.NOMINAL_GAIT_FREQUENCY_HZ
    nominal_profile = namespace.where(
        in_stance,
        namespace.zeros_like(desired_phase)
        + base_angular_velocity * contract.STANCE_VELOCITY_RATIO,
        namespace.zeros_like(desired_phase)
        + base_angular_velocity * contract.SWING_VELOCITY_RATIO,
    )

    effective_position = main_position * direction
    phase_delta = effective_position - desired_phase
    # ``arctan2`` is the stable Array API spelling shared by NumPy and Torch.
    # Some Isaac environments intentionally omit NumPy's historical ``atan2``
    # alias, so using the alias makes an otherwise backend-neutral decoder fail.
    phase_error = namespace.arctan2(namespace.sin(phase_delta), namespace.cos(phase_delta))
    phase_correction = namespace.clip(
        -contract.phase_lock_gain * phase_error,
        -contract.phase_correction_limit_rad_s,
        contract.phase_correction_limit_rad_s,
    )
    vx_scale = namespace.clip(
        command[..., 0] / contract.forward_command_reference_m_s,
        0.0,
        1.0,
    )
    active = command[..., 0] > contract.FORWARD_ACTIVE_MIN_M_S
    nominal_unwarmed = (
        (nominal_profile + phase_correction)
        * direction
        * vx_scale[..., None]
        * contract.forward_bias_scale
        * active[..., None]
    )
    residual_unwarmed = (
        safe_action[..., :6] * contract.main_residual_scale_rad_s * active[..., None]
    )
    residual_floor = (
        namespace.zeros_like(nominal_unwarmed) + contract.RESIDUAL_CAP_MIN_RAD_S
    )
    residual_cap = namespace.maximum(
        namespace.abs(nominal_unwarmed) * contract.residual_cap_ratio,
        residual_floor,
    )
    residual_unwarmed = namespace.clip(
        residual_unwarmed,
        -residual_cap,
        residual_cap,
    )

    nominal = nominal_unwarmed * warmup_scale[..., None]
    residual = residual_unwarmed * warmup_scale[..., None]
    target = namespace.clip(
        nominal + residual,
        -contract.main_velocity_limit_rad_s,
        contract.main_velocity_limit_rad_s,
    )
    target = target * main_output_sign
    abad = namespace.zeros_like(main_position) + abad_neutral * abad_output_sign
    return safe_action, nominal, residual, target, abad
