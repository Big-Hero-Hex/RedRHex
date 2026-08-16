"""Pure reference implementation of the V2 forward residual decoder."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ._action_core import decode_forward_residual_arrays_v2
from .contracts import ContractError, ForwardResidualActionContractV2


@dataclass(frozen=True)
class DecodedForwardActionV2:
    safe_action: tuple[float, ...]
    nominal_main_velocity_rad_s: tuple[float, ...]
    residual_main_velocity_rad_s: tuple[float, ...]
    target_main_velocity_rad_s: tuple[float, ...]
    target_abad_position_rad: tuple[float, ...]
    action_warmup_scale: float


def _six(value: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (6,) or not np.isfinite(result).all():
        raise ContractError(f"{name} must be finite with shape (6,)")
    return result


def _warmup_scale(control_step: int | None, contract: ForwardResidualActionContractV2) -> float:
    if control_step is None or contract.action_warmup_steps == 0:
        return 1.0
    step = float(control_step)
    if not math.isfinite(step) or step < 0.0 or not step.is_integer():
        raise ContractError("control_step must be a non-negative integer")
    return float(np.clip((step + 1.0) / contract.action_warmup_steps, 0.0, 1.0))


def decode_forward_residual_action_v2(
    action: Sequence[float],
    command: Sequence[float],
    gait_phase_rad: float,
    main_position_rad: Sequence[float],
    *,
    control_step: int | None = None,
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
    if (
        abs(command_array[1]) > contract.MAX_ABS_LATERAL_COMMAND_M_S
        or abs(command_array[2]) > contract.MAX_ABS_YAW_COMMAND_RAD_S
    ):
        raise ContractError("V2 forward decoder rejects lateral and yaw commands")
    phase = float(gait_phase_rad)
    if not math.isfinite(phase):
        raise ContractError("gait_phase_rad must be finite")
    position = _six(main_position_rad, "main_position_rad")

    warmup_scale = _warmup_scale(control_step, contract)
    action_mask = np.asarray((1.0,) * 6 + (0.0,) * 6, dtype=np.float64)
    phase_offsets = np.zeros(6, dtype=np.float64)
    phase_offsets[list(contract.TRIPOD_B)] = contract.TRIPOD_PHASE_OFFSET_RAD
    safe_action, nominal, residual, target, abad = decode_forward_residual_arrays_v2(
        values,
        command_array,
        np.asarray(phase, dtype=np.float64),
        position,
        np.asarray(warmup_scale, dtype=np.float64),
        contract=contract,
        namespace=np,
        action_mask=action_mask,
        phase_offsets=phase_offsets,
        direction=np.asarray(contract.LEG_DIRECTION_MULTIPLIER, dtype=np.float64),
        main_output_sign=np.asarray(contract.main_output_sign, dtype=np.float64),
        abad_neutral=np.asarray(contract.abad_neutral_position_rad, dtype=np.float64),
        abad_output_sign=np.asarray(contract.abad_output_sign, dtype=np.float64),
    )
    return DecodedForwardActionV2(
        safe_action=tuple(float(item) for item in safe_action),
        nominal_main_velocity_rad_s=tuple(float(item) for item in nominal),
        residual_main_velocity_rad_s=tuple(float(item) for item in residual),
        target_main_velocity_rad_s=tuple(float(item) for item in target),
        target_abad_position_rad=tuple(float(item) for item in abad),
        action_warmup_scale=warmup_scale,
    )
