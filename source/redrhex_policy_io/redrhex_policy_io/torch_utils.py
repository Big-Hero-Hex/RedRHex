"""Lazy Torch parity helpers; importing redrhex_policy_io never requires Torch."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any, Sequence

from ._action_core import decode_forward_residual_arrays_v2
from .contracts import ContractError, ForwardResidualActionContractV2


def _torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError("Torch helpers require the optional torch package") from exc


def wrap_angle_torch(angle: Any) -> Any:
    torch = _torch()
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def wrapped_velocity_torch(current_position_rad: Any, previous_position_rad: Any, dt_s: Any) -> Any:
    if current_position_rad.shape != previous_position_rad.shape:
        raise ContractError("current and previous Torch positions must have identical shape")
    if current_position_rad.ndim < 1:
        raise ContractError("Torch positions must be non-scalar")
    return wrap_angle_torch(current_position_rad - previous_position_rad) / dt_s


@dataclass(frozen=True)
class DecodedForwardActionV2Torch:
    """Batched Torch result used by the Sensor-V2 simulator adapter."""

    safe_action: Any
    nominal_main_velocity_rad_s: Any
    residual_main_velocity_rad_s: Any
    target_main_velocity_rad_s: Any
    target_abad_position_rad: Any
    action_warmup_scale: Any


def _prefix_tensor(
    value: Any,
    prefix: Any,
    *,
    dtype: Any,
    device: Any,
    name: str,
) -> Any:
    torch = _torch()
    result = torch.as_tensor(value, dtype=dtype, device=device)
    if result.ndim == 0:
        return result.expand(prefix) if len(prefix) else result
    if result.shape != prefix:
        raise ContractError(f"{name} must be scalar or have shape {tuple(prefix)}")
    return result


def decode_forward_residual_action_v2_torch(
    action: Any,
    command: Any,
    gait_phase_rad: Any,
    main_position_rad: Any,
    *,
    control_step: Any = None,
    contract: ForwardResidualActionContractV2 | None = None,
) -> DecodedForwardActionV2Torch:
    """Decode batched simulator actions with the exact NumPy/ROS V2 math."""

    torch = _torch()
    contract = contract or ForwardResidualActionContractV2()
    values = torch.as_tensor(action)
    if not values.is_floating_point():
        values = values.to(dtype=torch.float32)
    command_tensor = torch.as_tensor(command, dtype=values.dtype, device=values.device)
    position = torch.as_tensor(main_position_rad, dtype=values.dtype, device=values.device)
    if values.ndim < 1 or values.shape[-1] != 12:
        raise ContractError("Torch action must end in dimension 12")
    prefix = values.shape[:-1]
    if command_tensor.shape != (*prefix, 3):
        raise ContractError(f"Torch command must have shape {tuple(prefix)} + (3,)")
    if position.shape != (*prefix, 6):
        raise ContractError(f"Torch main_position_rad must have shape {tuple(prefix)} + (6,)")
    if not bool(torch.isfinite(values).all()):
        raise ContractError("Torch action contains NaN or Inf")
    if not bool(torch.isfinite(command_tensor).all()):
        raise ContractError("Torch command contains NaN or Inf")
    if not bool(torch.isfinite(position).all()):
        raise ContractError("Torch main_position_rad contains NaN or Inf")
    if bool(
        torch.any(
            torch.abs(command_tensor[..., 1])
            > contract.MAX_ABS_LATERAL_COMMAND_M_S
        )
    ) or bool(
        torch.any(
            torch.abs(command_tensor[..., 2])
            > contract.MAX_ABS_YAW_COMMAND_RAD_S
        )
    ):
        raise ContractError("V2 forward decoder rejects lateral and yaw commands")

    phase = _prefix_tensor(
        gait_phase_rad,
        prefix,
        dtype=values.dtype,
        device=values.device,
        name="gait_phase_rad",
    )
    if not bool(torch.isfinite(phase).all()):
        raise ContractError("gait_phase_rad must be finite")
    if control_step is None or contract.action_warmup_steps == 0:
        warmup_scale = torch.ones(prefix or (), dtype=values.dtype, device=values.device)
    else:
        step = _prefix_tensor(
            control_step,
            prefix,
            dtype=values.dtype,
            device=values.device,
            name="control_step",
        )
        if (
            not bool(torch.isfinite(step).all())
            or bool(torch.any(step < 0.0))
            or not bool(torch.equal(step, torch.round(step)))
        ):
            raise ContractError("control_step must contain non-negative integers")
        warmup_scale = torch.clamp(
            (step + 1.0) / float(contract.action_warmup_steps),
            min=0.0,
            max=1.0,
        )

    def constant(items: Sequence[float]) -> Any:
        return torch.as_tensor(items, dtype=values.dtype, device=values.device)

    phase_offsets = torch.zeros(6, dtype=values.dtype, device=values.device)
    phase_offsets[list(contract.TRIPOD_B)] = contract.TRIPOD_PHASE_OFFSET_RAD
    decoded = decode_forward_residual_arrays_v2(
        values,
        command_tensor,
        phase,
        position,
        warmup_scale,
        contract=contract,
        namespace=torch,
        action_mask=constant((1.0,) * 6 + (0.0,) * 6),
        phase_offsets=phase_offsets,
        direction=constant(contract.LEG_DIRECTION_MULTIPLIER),
        main_output_sign=constant(contract.main_output_sign),
        abad_neutral=constant(contract.abad_neutral_position_rad),
        abad_output_sign=constant(contract.abad_output_sign),
    )
    return DecodedForwardActionV2Torch(*decoded, action_warmup_scale=warmup_scale)


def _quat_rotate_wxyz_torch(quaternion: Any, vector: Any) -> Any:
    torch = _torch()
    if quaternion.shape[-1] != 4 or vector.shape[-1] != 3:
        raise ContractError("quaternion/vector final dimensions must be 4/3")
    q_vec = quaternion[..., 1:]
    while q_vec.ndim < vector.ndim:
        q_vec = q_vec.unsqueeze(-2)
    q_w = quaternion[..., :1]
    while q_w.ndim < vector.ndim:
        q_w = q_w.unsqueeze(-2)
    uv = torch.linalg.cross(q_vec.expand_as(vector), vector, dim=-1)
    uuv = torch.linalg.cross(q_vec.expand_as(vector), uv, dim=-1)
    return vector + 2.0 * (q_w * uv + uuv)


def transform_imu_vector_torch(vector_imu: Any, imu_to_body_wxyz: Any | Sequence[float]) -> Any:
    torch = _torch()
    quaternion = torch.as_tensor(
        imu_to_body_wxyz, dtype=vector_imu.dtype, device=vector_imu.device
    )
    if quaternion.shape[-1] != 4:
        raise ContractError("imu_to_body_wxyz final dimension must be 4")
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    if not bool(torch.allclose(norm, torch.ones_like(norm), rtol=0.0, atol=1.0e-6)):
        raise ContractError("imu_to_body_wxyz must be a unit quaternion")
    return _quat_rotate_wxyz_torch(quaternion, vector_imu)


def projected_gravity_from_quaternion_torch(
    orientation_wxyz: Any,
    *,
    imu_to_body_wxyz: Any | Sequence[float] | None = None,
) -> Any:
    """Pure batched math for validated simulator quaternions (body/IMU -> world)."""

    torch = _torch()
    if orientation_wxyz.shape[-1] != 4:
        raise ContractError("orientation_wxyz final dimension must be 4")
    norm = torch.linalg.vector_norm(orientation_wxyz, dim=-1, keepdim=True)
    if bool(torch.any(norm <= 0.0)):
        raise ContractError("orientation_wxyz contains a zero quaternion")
    orientation = orientation_wxyz / norm
    inverse = orientation.clone()
    inverse[..., 1:] *= -1.0
    gravity_world = torch.zeros(
        (*orientation.shape[:-1], 3), dtype=orientation.dtype, device=orientation.device
    )
    gravity_world[..., 2] = -1.0
    gravity_imu = _quat_rotate_wxyz_torch(inverse, gravity_world)
    if imu_to_body_wxyz is None:
        return gravity_imu
    return transform_imu_vector_torch(gravity_imu, imu_to_body_wxyz)


def build_sensor_frame_torch(
    body_gyro_rad_s: Any,
    projected_gravity: Any,
    main_position_rad: Any,
    main_velocity_rad_s: Any,
    abad_position_rad: Any,
    abad_velocity_rad_s: Any,
    *,
    abad_neutral_position_rad: Any | Sequence[float] | None = None,
) -> Any:
    """Compose batched ``[..., 36]`` frames with the NumPy contract ordering."""

    torch = _torch()
    expected = (
        (body_gyro_rad_s, 3, "body_gyro_rad_s"),
        (projected_gravity, 3, "projected_gravity"),
        (main_position_rad, 6, "main_position_rad"),
        (main_velocity_rad_s, 6, "main_velocity_rad_s"),
        (abad_position_rad, 6, "abad_position_rad"),
        (abad_velocity_rad_s, 6, "abad_velocity_rad_s"),
    )
    prefix = body_gyro_rad_s.shape[:-1]
    for tensor, final_dim, name in expected:
        if tensor.shape[:-1] != prefix or tensor.shape[-1] != final_dim:
            raise ContractError(f"{name} has incompatible shape {tuple(tensor.shape)}")
    if abad_neutral_position_rad is None:
        neutral = torch.zeros_like(abad_position_rad)
    else:
        neutral = torch.as_tensor(
            abad_neutral_position_rad,
            dtype=abad_position_rad.dtype,
            device=abad_position_rad.device,
        )
        if neutral.shape == (6,):
            neutral = neutral.expand_as(abad_position_rad)
        elif neutral.shape != abad_position_rad.shape:
            raise ContractError("abad_neutral_position_rad has incompatible shape")
    return torch.cat(
        (
            body_gyro_rad_s,
            projected_gravity,
            torch.sin(main_position_rad),
            torch.cos(main_position_rad),
            main_velocity_rad_s,
            abad_position_rad - neutral,
            abad_velocity_rad_s,
        ),
        dim=-1,
    )


class BatchedCausalGyroAccelAttitudeV2:
    """Stateful batched Torch equivalent of CausalGyroAccelAttitudeV2.

    ``projected_gravity`` is a unit world-gravity direction expressed in each
    policy body frame, with shape ``[num_envs, 3]``.
    """

    def __init__(
        self,
        num_envs: int,
        *,
        device: Any = None,
        dtype: Any = None,
        correction_gain: float = 0.02,
        gravity_magnitude_m_s2: float = 9.80665,
        accel_magnitude_tolerance_ratio: float = 0.25,
        initial_projected_gravity: Sequence[float] = (0.0, 0.0, -1.0),
    ) -> None:
        torch = _torch()
        if isinstance(num_envs, bool) or int(num_envs) < 1:
            raise ContractError("num_envs must be positive")
        self.num_envs = int(num_envs)
        self.correction_gain = float(correction_gain)
        self.gravity_magnitude_m_s2 = float(gravity_magnitude_m_s2)
        self.accel_magnitude_tolerance_ratio = float(accel_magnitude_tolerance_ratio)
        if not 0.0 <= self.correction_gain <= 1.0:
            raise ContractError("correction_gain must be in [0, 1]")
        if self.gravity_magnitude_m_s2 <= 0.0 or self.accel_magnitude_tolerance_ratio < 0.0:
            raise ContractError("invalid gravity estimator parameters")
        resolved_dtype = dtype if dtype is not None else torch.float32
        initial = torch.as_tensor(initial_projected_gravity, dtype=resolved_dtype, device=device)
        if initial.shape != (3,) or not bool(torch.isfinite(initial).all()):
            raise ContractError("initial_projected_gravity must be a finite 3-vector")
        initial_norm = torch.linalg.vector_norm(initial)
        if float(initial_norm) <= 1.0e-12:
            raise ContractError("initial_projected_gravity must be non-zero")
        self._initial = initial / initial_norm
        self.projected_gravity = self._initial.repeat(self.num_envs, 1)

    def reset(self, env_ids: Any = None, projected_gravity: Any = None) -> None:
        torch = _torch()
        if env_ids is None:
            env_ids = torch.arange(
                self.num_envs, device=self.projected_gravity.device, dtype=torch.long
            )
        else:
            env_ids = torch.as_tensor(
                env_ids, device=self.projected_gravity.device, dtype=torch.long
            )
        if projected_gravity is None:
            values = self._initial.expand(env_ids.numel(), 3)
        else:
            values = torch.as_tensor(
                projected_gravity,
                device=self.projected_gravity.device,
                dtype=self.projected_gravity.dtype,
            )
            if values.shape == (3,):
                values = values.expand(env_ids.numel(), 3)
            if values.shape != (env_ids.numel(), 3):
                raise ContractError("reset projected_gravity has incompatible shape")
            norms = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
            if bool(torch.any(norms <= 1.0e-12)):
                raise ContractError("reset projected_gravity contains a zero vector")
            values = values / norms
        self.projected_gravity[env_ids] = values

    def update(self, body_gyro_rad_s: Any, body_linear_accel_m_s2: Any, dt_s: Any) -> Any:
        torch = _torch()
        expected = (self.num_envs, 3)
        if body_gyro_rad_s.shape != expected or body_linear_accel_m_s2.shape != expected:
            raise ContractError(f"gyro and acceleration must have shape {expected}")
        if not bool(torch.isfinite(body_gyro_rad_s).all()) or not bool(
            torch.isfinite(body_linear_accel_m_s2).all()
        ):
            raise ContractError("gyro or acceleration contains NaN or Inf")
        dt = torch.as_tensor(
            dt_s, dtype=self.projected_gravity.dtype, device=self.projected_gravity.device
        )
        if dt.ndim == 0:
            if float(dt) <= 0.0:
                raise ContractError("dt_s must be positive")
        elif dt.shape == (self.num_envs,):
            if bool(torch.any(dt <= 0.0)):
                raise ContractError("dt_s must be positive")
            dt = dt.unsqueeze(-1)
        elif dt.shape != (self.num_envs, 1):
            raise ContractError("dt_s must be scalar, [num_envs], or [num_envs, 1]")
        predicted = self.projected_gravity - torch.linalg.cross(
            body_gyro_rad_s, self.projected_gravity, dim=-1
        ) * dt
        predicted = predicted / torch.linalg.vector_norm(predicted, dim=-1, keepdim=True).clamp_min(
            1.0e-12
        )
        accel_norm = torch.linalg.vector_norm(body_linear_accel_m_s2, dim=-1, keepdim=True)
        if bool(torch.any(accel_norm <= 1.0e-12)):
            raise ContractError("acceleration contains a zero vector")
        relative_error = torch.abs(accel_norm - self.gravity_magnitude_m_s2) / self.gravity_magnitude_m_s2
        use_correction = relative_error <= self.accel_magnitude_tolerance_ratio
        measured = -body_linear_accel_m_s2 / accel_norm
        corrected = (1.0 - self.correction_gain) * predicted + self.correction_gain * measured
        corrected = corrected / torch.linalg.vector_norm(corrected, dim=-1, keepdim=True).clamp_min(
            1.0e-12
        )
        self.projected_gravity = torch.where(use_correction, corrected, predicted)
        return self.projected_gravity
