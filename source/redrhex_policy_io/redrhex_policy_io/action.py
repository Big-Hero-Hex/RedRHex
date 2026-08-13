"""Pure reference implementation of the V2 forward residual decoder."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .contracts import ContractError, ForwardResidualActionContractV2
from .preprocessing import wrap_angle


@dataclass(frozen=True)
class DecodedForwardActionV2:
    safe_action: tuple[float, ...]
    nominal_main_velocity_rad_s: tuple[float, ...]
    residual_main_velocity_rad_s: tuple[float, ...]
    target_main_velocity_rad_s: tuple[float, ...]
    target_abad_position_rad: tuple[float, ...]


def _six(value: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (6,) or not np.isfinite(result).all():
        raise ContractError(f"{name} must be finite with shape (6,)")
    return result


def _in_stance(phase: np.ndarray, contract: ForwardResidualActionContractV2) -> np.ndarray:
    start = contract.STANCE_PHASE_START_RAD
    end = contract.STANCE_PHASE_END_RAD
    wrapped_start = start + 2.0 * math.pi if start < 0.0 else start
    return np.logical_or(phase >= wrapped_start, phase < end)


def decode_forward_residual_action_v2(
    action: Sequence[float],
    command: Sequence[float],
    gait_phase_rad: float,
    main_position_rad: Sequence[float],
    *,
    contract: ForwardResidualActionContractV2 | None = None,
) -> DecodedForwardActionV2:
    """Decode one strict-forward action in canonical joint order.

    The procedural gait clock is an input to this decoder, never to the actor.
    ``action[6:12]`` is discarded and the returned ABAD target is always the
    versioned neutral pose.
    """

    contract = contract or ForwardResidualActionContractV2()
    values = np.asarray(action, dtype=np.float64)
    command_array = np.asarray(command, dtype=np.float64)
    if values.shape != (12,) or not np.isfinite(values).all():
        raise ContractError("action must be finite with shape (12,)")
    if command_array.shape != (3,) or not np.isfinite(command_array).all():
        raise ContractError("command must be finite with shape (3,)")
    if abs(command_array[1]) > 0.08 or abs(command_array[2]) > 0.10:
        raise ContractError("V2 forward decoder rejects lateral and yaw commands")
    phase = float(gait_phase_rad)
    if not math.isfinite(phase):
        raise ContractError("gait_phase_rad must be finite")
    position = _six(main_position_rad, "main_position_rad")

    safe_action = np.clip(values, -contract.action_clip, contract.action_clip)
    safe_action[6:] = 0.0
    phase_offsets = np.zeros(6, dtype=np.float64)
    phase_offsets[list(contract.TRIPOD_B)] = contract.TRIPOD_PHASE_OFFSET_RAD
    desired_phase = np.mod(phase + phase_offsets, 2.0 * math.pi)
    base_angular_velocity = 2.0 * math.pi * contract.NOMINAL_GAIT_FREQUENCY_HZ
    nominal_profile = np.where(
        _in_stance(desired_phase, contract),
        base_angular_velocity * contract.STANCE_VELOCITY_RATIO,
        base_angular_velocity * contract.SWING_VELOCITY_RATIO,
    )
    phase_error = np.asarray(wrap_angle(position - desired_phase), dtype=np.float64)
    phase_correction = np.clip(
        -contract.phase_lock_gain * phase_error,
        -contract.phase_correction_limit_rad_s,
        contract.phase_correction_limit_rad_s,
    )
    vx_scale = np.clip(
        command_array[0] / contract.forward_command_reference_m_s, -1.0, 1.0
    )
    active = float(command_array[0] > 0.10)
    nominal = (
        (nominal_profile + phase_correction)
        * np.asarray(contract.LEG_DIRECTION_MULTIPLIER)
        * vx_scale
        * contract.forward_bias_scale
        * active
    )
    residual = safe_action[:6] * contract.main_residual_scale_rad_s * active
    residual_cap = np.maximum(np.abs(nominal) * contract.residual_cap_ratio, 0.08)
    residual = np.clip(residual, -residual_cap, residual_cap)
    target = np.clip(
        nominal + residual,
        -contract.main_velocity_limit_rad_s,
        contract.main_velocity_limit_rad_s,
    )
    target *= np.asarray(contract.main_output_sign)
    abad = np.asarray(contract.abad_neutral_position_rad) * np.asarray(
        contract.abad_output_sign
    )
    return DecodedForwardActionV2(
        safe_action=tuple(float(item) for item in safe_action),
        nominal_main_velocity_rad_s=tuple(float(item) for item in nominal),
        residual_main_velocity_rad_s=tuple(float(item) for item in residual),
        target_main_velocity_rad_s=tuple(float(item) for item in target),
        target_abad_position_rad=tuple(float(item) for item in abad),
    )
