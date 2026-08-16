# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evidence-gated sensor perturbations for the Sensor V2 observation path.

This module intentionally depends only on PyTorch.  It can therefore be tested
without importing Isaac Lab, and it keeps packet-state semantics separate from
the simulator environment.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch


FloatRange = tuple[float, float]
IntRange = tuple[int, int]


def real_history_sample_mask_v2(
    accepted_sample: torch.Tensor,
    encoder_initialized: torch.Tensor,
) -> torch.Tensor:
    """Exclude the first accepted encoder event used to prime differences."""

    if accepted_sample.shape != encoder_initialized.shape:
        raise ValueError("accepted sample and encoder initialization masks must match")
    return accepted_sample & encoder_initialized


def _validated_float_range(
    name: str,
    value: FloatRange,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-value range")
    low, high = float(value[0]), float(value[1])
    bounds = torch.tensor((low, high), dtype=torch.float64)
    if not bool(torch.isfinite(bounds).all()):
        raise ValueError(f"{name} must contain finite values")
    if low > high:
        raise ValueError(f"{name} lower bound must not exceed its upper bound")
    if minimum is not None and low < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and high > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return low, high


def _validated_int_range(name: str, value: IntRange) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-value integer range")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{name} must contain integers")
    low, high = int(value[0]), int(value[1])
    if low > high:
        raise ValueError(f"{name} lower bound must not exceed its upper bound")
    return low, high


@dataclass(frozen=True)
class SensorDomainRandomizationV2Config:
    """Neutral-by-default sensor perturbation ranges.

    A non-neutral range is rejected unless ``evidence`` names the measured or
    otherwise reviewed source that justified it.  This deliberately does not
    provide guessed production ranges.
    """

    evidence: str = ""

    gyro_noise_std_range_rad_s: FloatRange = (0.0, 0.0)
    gyro_bias_range_rad_s: FloatRange = (0.0, 0.0)
    gyro_drift_std_range_rad_s_sqrt_s: FloatRange = (0.0, 0.0)
    gyro_filter_time_constant_range_s: FloatRange = (0.0, 0.0)
    gyro_latency_steps_range: IntRange = (0, 0)
    gyro_latency_jitter_steps_range: IntRange = (0, 0)

    imu_mount_roll_range_rad: FloatRange = (0.0, 0.0)
    imu_mount_pitch_range_rad: FloatRange = (0.0, 0.0)
    imu_mount_yaw_range_rad: FloatRange = (0.0, 0.0)

    encoder_zero_offset_range_rad: FloatRange = (0.0, 0.0)
    encoder_noise_std_range_rad: FloatRange = (0.0, 0.0)
    encoder_quantization_range_rad: FloatRange = (0.0, 0.0)
    encoder_latency_steps_range: IntRange = (0, 0)
    encoder_stale_probability_range: FloatRange = (0.0, 0.0)
    encoder_dropout_probability_range: FloatRange = (0.0, 0.0)

    accel_noise_std_range_m_s2: FloatRange = (0.0, 0.0)
    accel_bias_range_m_s2: FloatRange = (0.0, 0.0)

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, str):
            raise ValueError("evidence must be a string")

        nonnegative_ranges = (
            "gyro_noise_std_range_rad_s",
            "gyro_drift_std_range_rad_s_sqrt_s",
            "gyro_filter_time_constant_range_s",
            "encoder_noise_std_range_rad",
            "encoder_quantization_range_rad",
            "accel_noise_std_range_m_s2",
        )
        signed_ranges = (
            "gyro_bias_range_rad_s",
            "imu_mount_roll_range_rad",
            "imu_mount_pitch_range_rad",
            "imu_mount_yaw_range_rad",
            "encoder_zero_offset_range_rad",
            "accel_bias_range_m_s2",
        )
        probability_ranges = (
            "encoder_stale_probability_range",
            "encoder_dropout_probability_range",
        )
        for name in nonnegative_ranges:
            _validated_float_range(name, getattr(self, name), minimum=0.0)
        for name in signed_ranges:
            _validated_float_range(name, getattr(self, name))
        for name in probability_ranges:
            _validated_float_range(name, getattr(self, name), minimum=0.0, maximum=1.0)

        gyro_latency = _validated_int_range(
            "gyro_latency_steps_range", self.gyro_latency_steps_range
        )
        gyro_jitter = _validated_int_range(
            "gyro_latency_jitter_steps_range", self.gyro_latency_jitter_steps_range
        )
        encoder_latency = _validated_int_range(
            "encoder_latency_steps_range", self.encoder_latency_steps_range
        )
        if gyro_latency[0] < 0 or encoder_latency[0] < 0:
            raise ValueError("base sensor latency must be non-negative")
        if gyro_latency[0] + gyro_jitter[0] < 0:
            raise ValueError("gyro latency plus its minimum jitter must be non-negative")

        if self.enabled and not self.evidence.strip():
            raise ValueError(
                "non-neutral Sensor V2 domain randomization requires non-empty evidence"
            )

    @property
    def enabled(self) -> bool:
        for item in fields(self):
            if item.name == "evidence":
                continue
            value = getattr(self, item.name)
            if any(float(bound) != 0.0 for bound in value):
                return True
        return False

    @classmethod
    def from_namespace(cls, namespace: Any) -> SensorDomainRandomizationV2Config:
        """Read ``sensor_dr_<field>`` attributes from an environment config."""

        values: dict[str, Any] = {}
        for item in fields(cls):
            attribute = f"sensor_dr_{item.name}"
            if hasattr(namespace, attribute):
                values[item.name] = getattr(namespace, attribute)
        return cls(**values)


@dataclass(frozen=True)
class SensorEventV2:
    gyro_body: torch.Tensor
    main_position: torch.Tensor
    abad_position: torch.Tensor
    gravity_body: torch.Tensor | None
    specific_force_body: torch.Tensor | None
    accepted_sample: torch.Tensor
    encoder_stale: torch.Tensor
    encoder_dropout: torch.Tensor


class SensorDomainRandomizerV2:
    """Stateful, vectorized sensor event perturbation with per-environment resets."""

    def __init__(
        self,
        config: SensorDomainRandomizationV2Config,
        *,
        num_envs: int,
        num_main_joints: int,
        num_abad_joints: int,
        sample_hz: float,
        device: str | torch.device,
        seed: int,
    ) -> None:
        if num_envs <= 0 or num_main_joints <= 0 or num_abad_joints <= 0:
            raise ValueError("sensor randomizer dimensions must be positive")
        if not torch.isfinite(torch.tensor(float(sample_hz))) or float(sample_hz) <= 0.0:
            raise ValueError("sample_hz must be finite and positive")

        self.config = config
        self.num_envs = int(num_envs)
        self.num_main_joints = int(num_main_joints)
        self.num_abad_joints = int(num_abad_joints)
        self.num_encoders = self.num_main_joints + self.num_abad_joints
        self.sample_hz = float(sample_hz)
        self.device = torch.device(device)
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(int(seed) % (2**63 - 1))

        vector_shape = (self.num_envs, 3)
        encoder_shape = (self.num_envs, self.num_encoders)
        scalar_shape = (self.num_envs,)
        self.gyro_noise_std_rad_s = torch.zeros(scalar_shape, device=self.device)
        self.gyro_bias_rad_s = torch.zeros(vector_shape, device=self.device)
        self.gyro_drift_std_rad_s_sqrt_s = torch.zeros(scalar_shape, device=self.device)
        self.gyro_drift_rad_s = torch.zeros(vector_shape, device=self.device)
        self.gyro_filter_time_constant_s = torch.zeros(scalar_shape, device=self.device)
        self.gyro_latency_steps = torch.zeros(scalar_shape, dtype=torch.long, device=self.device)
        self.imu_mount_rpy_rad = torch.zeros(vector_shape, device=self.device)

        self.encoder_zero_offset_rad = torch.zeros(encoder_shape, device=self.device)
        self.encoder_noise_std_rad = torch.zeros(scalar_shape, device=self.device)
        self.encoder_quantization_rad = torch.zeros(scalar_shape, device=self.device)
        self.encoder_latency_steps = torch.zeros(
            scalar_shape, dtype=torch.long, device=self.device
        )
        self.encoder_stale_probability = torch.zeros(scalar_shape, device=self.device)
        self.encoder_dropout_probability = torch.zeros(scalar_shape, device=self.device)

        self.accel_noise_std_m_s2 = torch.zeros(scalar_shape, device=self.device)
        self.accel_bias_m_s2 = torch.zeros(vector_shape, device=self.device)

        max_gyro_delay = (
            int(config.gyro_latency_steps_range[1])
            + int(config.gyro_latency_jitter_steps_range[1])
        )
        max_encoder_delay = int(config.encoder_latency_steps_range[1])
        self._gyro_queue = torch.zeros(
            self.num_envs, max(max_gyro_delay, 0) + 1, 3, device=self.device
        )
        self._encoder_queue = torch.zeros(
            self.num_envs,
            max(max_encoder_delay, 0) + 1,
            self.num_encoders,
            device=self.device,
        )
        self._gyro_queue_initialized = torch.zeros(
            scalar_shape, dtype=torch.bool, device=self.device
        )
        self._encoder_queue_initialized = torch.zeros(
            scalar_shape, dtype=torch.bool, device=self.device
        )
        self._gyro_filter_state = torch.zeros(vector_shape, device=self.device)
        self._gyro_filter_initialized = torch.zeros(
            scalar_shape, dtype=torch.bool, device=self.device
        )
        self._last_encoder = torch.zeros(encoder_shape, device=self.device)
        self._last_encoder_initialized = torch.zeros(
            scalar_shape, dtype=torch.bool, device=self.device
        )
        self.encoder_stale_count = torch.zeros(scalar_shape, dtype=torch.long, device=self.device)
        self.encoder_dropout_count = torch.zeros(
            scalar_shape, dtype=torch.long, device=self.device
        )

        self._gyro_noise_enabled = self._range_nonzero(config.gyro_noise_std_range_rad_s)
        self._gyro_drift_enabled = self._range_nonzero(
            config.gyro_drift_std_range_rad_s_sqrt_s
        )
        self._gyro_filter_enabled = self._range_nonzero(
            config.gyro_filter_time_constant_range_s
        )
        self._mount_enabled = any(
            self._range_nonzero(value)
            for value in (
                config.imu_mount_roll_range_rad,
                config.imu_mount_pitch_range_rad,
                config.imu_mount_yaw_range_rad,
            )
        )
        self._encoder_noise_enabled = self._range_nonzero(
            config.encoder_noise_std_range_rad
        )
        self._encoder_quantization_enabled = self._range_nonzero(
            config.encoder_quantization_range_rad
        )
        self._encoder_stale_enabled = self._range_nonzero(
            config.encoder_stale_probability_range
        )
        self._encoder_dropout_enabled = self._range_nonzero(
            config.encoder_dropout_probability_range
        )
        self._accel_noise_enabled = self._range_nonzero(config.accel_noise_std_range_m_s2)

        self.reset(torch.arange(self.num_envs, device=self.device))

    @staticmethod
    def _range_nonzero(value: FloatRange | IntRange) -> bool:
        return any(float(bound) != 0.0 for bound in value)

    def _uniform(
        self, value_range: FloatRange, shape: tuple[int, ...], *, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        low, high = float(value_range[0]), float(value_range[1])
        if low == high:
            return torch.full(shape, low, dtype=dtype, device=self.device)
        return torch.rand(shape, generator=self._generator, dtype=dtype, device=self.device) * (
            high - low
        ) + low

    def _uniform_int(self, value_range: IntRange, shape: tuple[int, ...]) -> torch.Tensor:
        low, high = int(value_range[0]), int(value_range[1])
        if low == high:
            return torch.full(shape, low, dtype=torch.long, device=self.device)
        return torch.randint(
            low,
            high + 1,
            shape,
            generator=self._generator,
            dtype=torch.long,
            device=self.device,
        )

    def reset(self, env_ids: torch.Tensor | list[int] | tuple[int, ...]) -> None:
        """Clear packet state and resample parameters only for ``env_ids``."""

        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        if ids.numel() == 0:
            return
        if bool(((ids < 0) | (ids >= self.num_envs)).any()):
            raise IndexError("sensor randomizer reset contains an out-of-range environment id")
        count = int(ids.numel())

        self.gyro_noise_std_rad_s[ids] = self._uniform(
            self.config.gyro_noise_std_range_rad_s, (count,)
        )
        self.gyro_bias_rad_s[ids] = self._uniform(
            self.config.gyro_bias_range_rad_s, (count, 3)
        )
        self.gyro_drift_std_rad_s_sqrt_s[ids] = self._uniform(
            self.config.gyro_drift_std_range_rad_s_sqrt_s, (count,)
        )
        self.gyro_drift_rad_s[ids] = 0.0
        self.gyro_filter_time_constant_s[ids] = self._uniform(
            self.config.gyro_filter_time_constant_range_s, (count,)
        )
        self.gyro_latency_steps[ids] = self._uniform_int(
            self.config.gyro_latency_steps_range, (count,)
        )
        self.imu_mount_rpy_rad[ids, 0] = self._uniform(
            self.config.imu_mount_roll_range_rad, (count,)
        )
        self.imu_mount_rpy_rad[ids, 1] = self._uniform(
            self.config.imu_mount_pitch_range_rad, (count,)
        )
        self.imu_mount_rpy_rad[ids, 2] = self._uniform(
            self.config.imu_mount_yaw_range_rad, (count,)
        )

        self.encoder_zero_offset_rad[ids] = self._uniform(
            self.config.encoder_zero_offset_range_rad, (count, self.num_encoders)
        )
        self.encoder_noise_std_rad[ids] = self._uniform(
            self.config.encoder_noise_std_range_rad, (count,)
        )
        self.encoder_quantization_rad[ids] = self._uniform(
            self.config.encoder_quantization_range_rad, (count,)
        )
        self.encoder_latency_steps[ids] = self._uniform_int(
            self.config.encoder_latency_steps_range, (count,)
        )
        self.encoder_stale_probability[ids] = self._uniform(
            self.config.encoder_stale_probability_range, (count,)
        )
        self.encoder_dropout_probability[ids] = self._uniform(
            self.config.encoder_dropout_probability_range, (count,)
        )

        self.accel_noise_std_m_s2[ids] = self._uniform(
            self.config.accel_noise_std_range_m_s2, (count,)
        )
        self.accel_bias_m_s2[ids] = self._uniform(
            self.config.accel_bias_range_m_s2, (count, 3)
        )

        self._gyro_queue[ids] = 0.0
        self._encoder_queue[ids] = 0.0
        self._gyro_queue_initialized[ids] = False
        self._encoder_queue_initialized[ids] = False
        self._gyro_filter_state[ids] = 0.0
        self._gyro_filter_initialized[ids] = False
        self._last_encoder[ids] = 0.0
        self._last_encoder_initialized[ids] = False
        self.encoder_stale_count[ids] = 0
        self.encoder_dropout_count[ids] = 0

    def _sampled_parameter_views(self) -> dict[str, torch.Tensor]:
        return {
            "gyro_noise_std_rad_s": self.gyro_noise_std_rad_s,
            "gyro_bias_rad_s": self.gyro_bias_rad_s,
            "gyro_drift_std_rad_s_sqrt_s": self.gyro_drift_std_rad_s_sqrt_s,
            "gyro_filter_time_constant_s": self.gyro_filter_time_constant_s,
            "gyro_latency_steps": self.gyro_latency_steps,
            "imu_mount_rpy_rad": self.imu_mount_rpy_rad,
            "encoder_zero_offset_rad": self.encoder_zero_offset_rad,
            "encoder_noise_std_rad": self.encoder_noise_std_rad,
            "encoder_quantization_rad": self.encoder_quantization_rad,
            "encoder_latency_steps": self.encoder_latency_steps,
            "encoder_stale_probability": self.encoder_stale_probability,
            "encoder_dropout_probability": self.encoder_dropout_probability,
            "accel_noise_std_m_s2": self.accel_noise_std_m_s2,
            "accel_bias_m_s2": self.accel_bias_m_s2,
        }

    def sampled_parameters(self) -> dict[str, torch.Tensor]:
        """Return detached copies suitable for diagnostics or experiment capture."""

        return {
            name: value.detach().clone() for name, value in self._sampled_parameter_views().items()
        }

    def runtime_counters(self) -> dict[str, torch.Tensor]:
        return {
            "encoder_stale_count": self.encoder_stale_count.detach().clone(),
            "encoder_dropout_count": self.encoder_dropout_count.detach().clone(),
        }

    def sampled_statistics(
        self, env_ids: torch.Tensor | list[int] | tuple[int, ...] | None = None
    ) -> dict[str, float]:
        """Return finite min/mean/max statistics for experiment logging."""

        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None
            else torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        )
        if ids.numel() == 0:
            return {}
        statistics: dict[str, float] = {}
        for name, values in self._sampled_parameter_views().items():
            selected = values[ids].float().reshape(-1)
            statistics[f"{name}_min"] = float(selected.min().item())
            statistics[f"{name}_mean"] = float(selected.mean().item())
            statistics[f"{name}_max"] = float(selected.max().item())
        return statistics

    @staticmethod
    def _rotation_matrix_from_rpy(rpy: torch.Tensor) -> torch.Tensor:
        """Build the active Rz-Ry-Rx perturbation applied to reported IMU vectors."""

        roll, pitch, yaw = rpy.unbind(dim=-1)
        cr, sr = torch.cos(roll), torch.sin(roll)
        cp, sp = torch.cos(pitch), torch.sin(pitch)
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        return torch.stack(
            (
                cy * cp,
                cy * sp * sr - sy * cr,
                cy * sp * cr + sy * sr,
                sy * cp,
                sy * sp * sr + cy * cr,
                sy * sp * cr - cy * sr,
                -sp,
                cp * sr,
                cp * cr,
            ),
            dim=-1,
        ).reshape(-1, 3, 3)

    def _apply_mount(self, vectors: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        if not self._mount_enabled:
            return vectors
        rotation = self._rotation_matrix_from_rpy(self.imu_mount_rpy_rad[ids])
        return torch.bmm(rotation, vectors.unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def _check_shape(name: str, value: torch.Tensor, expected: tuple[int, int]) -> None:
        if tuple(value.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains NaN or Inf")

    def _push_and_delay(
        self,
        queue: torch.Tensor,
        initialized: torch.Tensor,
        current: torch.Tensor,
        ids: torch.Tensor,
        delay: torch.Tensor,
    ) -> torch.Tensor:
        shifted = torch.roll(queue[ids], shifts=1, dims=1)
        shifted[:, 0] = current
        first = ~initialized[ids]
        if bool(first.any()):
            shifted[first] = current[first].unsqueeze(1).expand(-1, shifted.shape[1], -1)
        queue[ids] = shifted
        initialized[ids] = True
        rows = torch.arange(ids.numel(), device=self.device)
        return shifted[rows, delay]

    def process(
        self,
        *,
        gyro_body: torch.Tensor,
        main_position: torch.Tensor,
        abad_position: torch.Tensor,
        new_sample: torch.Tensor,
        gravity_body: torch.Tensor | None = None,
        specific_force_body: torch.Tensor | None = None,
    ) -> SensorEventV2:
        """Perturb a causal event before feature differencing and history insertion."""

        self._check_shape("gyro_body", gyro_body, (self.num_envs, 3))
        self._check_shape(
            "main_position", main_position, (self.num_envs, self.num_main_joints)
        )
        self._check_shape(
            "abad_position", abad_position, (self.num_envs, self.num_abad_joints)
        )
        if tuple(new_sample.shape) != (self.num_envs,) or new_sample.dtype != torch.bool:
            raise ValueError("new_sample must be a boolean tensor with one value per environment")
        if gravity_body is not None:
            self._check_shape("gravity_body", gravity_body, (self.num_envs, 3))
        if specific_force_body is not None:
            self._check_shape("specific_force_body", specific_force_body, (self.num_envs, 3))

        output_gyro = gyro_body.clone()
        output_main = main_position.clone()
        output_abad = abad_position.clone()
        output_gravity = gravity_body.clone() if gravity_body is not None else None
        output_specific_force = (
            specific_force_body.clone() if specific_force_body is not None else None
        )
        accepted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        stale = torch.zeros_like(accepted)
        dropout = torch.zeros_like(accepted)
        ids = torch.nonzero(new_sample, as_tuple=False).flatten()
        if ids.numel() == 0:
            return SensorEventV2(
                output_gyro,
                output_main,
                output_abad,
                output_gravity,
                output_specific_force,
                accepted,
                stale,
                dropout,
            )

        count = int(ids.numel())
        current_gyro = self._apply_mount(gyro_body[ids], ids)
        if self._gyro_drift_enabled:
            drift_increment = torch.randn(
                (count, 3), generator=self._generator, device=self.device
            ) * self.gyro_drift_std_rad_s_sqrt_s[ids].unsqueeze(-1) * (
                self.sample_hz ** -0.5
            )
            self.gyro_drift_rad_s[ids] += drift_increment
        current_gyro = current_gyro + self.gyro_bias_rad_s[ids] + self.gyro_drift_rad_s[ids]
        if self._gyro_noise_enabled:
            current_gyro = current_gyro + torch.randn(
                (count, 3), generator=self._generator, device=self.device
            ) * self.gyro_noise_std_rad_s[ids].unsqueeze(-1)
        if self._gyro_filter_enabled:
            tau = self.gyro_filter_time_constant_s[ids].unsqueeze(-1)
            dt = 1.0 / self.sample_hz
            alpha = dt / (tau + dt)
            previous = self._gyro_filter_state[ids]
            filtered = alpha * current_gyro + (1.0 - alpha) * previous
            filtered = torch.where(
                self._gyro_filter_initialized[ids].unsqueeze(-1), filtered, current_gyro
            )
            self._gyro_filter_state[ids] = filtered
            self._gyro_filter_initialized[ids] = True
            current_gyro = filtered

        jitter = self._uniform_int(
            self.config.gyro_latency_jitter_steps_range, (count,)
        )
        gyro_delay = torch.clamp(self.gyro_latency_steps[ids] + jitter, min=0)
        output_gyro[ids] = self._push_and_delay(
            self._gyro_queue,
            self._gyro_queue_initialized,
            current_gyro,
            ids,
            gyro_delay,
        )

        encoder = torch.cat((main_position[ids], abad_position[ids]), dim=-1)
        encoder = encoder + self.encoder_zero_offset_rad[ids]
        if self._encoder_noise_enabled:
            encoder = encoder + torch.randn(
                (count, self.num_encoders), generator=self._generator, device=self.device
            ) * self.encoder_noise_std_rad[ids].unsqueeze(-1)
        if self._encoder_quantization_enabled:
            quantum = self.encoder_quantization_rad[ids].unsqueeze(-1)
            safe_quantum = torch.where(quantum > 0.0, quantum, torch.ones_like(quantum))
            quantized = torch.round(encoder / safe_quantum) * safe_quantum
            encoder = torch.where(quantum > 0.0, quantized, encoder)
        delayed_encoder = self._push_and_delay(
            self._encoder_queue,
            self._encoder_queue_initialized,
            encoder,
            ids,
            self.encoder_latency_steps[ids],
        )

        if self._encoder_stale_enabled:
            stale_requested = torch.rand(
                (count,), generator=self._generator, device=self.device
            ) < self.encoder_stale_probability[ids]
        else:
            stale_requested = torch.zeros(count, dtype=torch.bool, device=self.device)
        stale_local = stale_requested & self._last_encoder_initialized[ids]
        delivered_encoder = torch.where(
            stale_local.unsqueeze(-1), self._last_encoder[ids], delayed_encoder
        )

        if self._encoder_dropout_enabled:
            dropout_local = torch.rand(
                (count,), generator=self._generator, device=self.device
            ) < self.encoder_dropout_probability[ids]
        else:
            dropout_local = torch.zeros(count, dtype=torch.bool, device=self.device)
        stale_local &= ~dropout_local
        has_previous = self._last_encoder_initialized[ids]
        encoder_output = torch.where(
            (dropout_local & has_previous).unsqueeze(-1),
            self._last_encoder[ids],
            delivered_encoder,
        )
        # Deployment treats replayed/stale encoder packets as a freshness
        # failure, not a new physical sample.  Keep the held value observable
        # for diagnostics, but never advance finite differences or history.
        accepted_local = ~(dropout_local | stale_local)
        if bool(accepted_local.any()):
            accepted_ids = ids[accepted_local]
            self._last_encoder[accepted_ids] = delayed_encoder[accepted_local]
            self._last_encoder_initialized[accepted_ids] = True

        output_main[ids] = encoder_output[:, : self.num_main_joints]
        output_abad[ids] = encoder_output[:, self.num_main_joints :]
        accepted[ids] = accepted_local
        stale[ids] = stale_local
        dropout[ids] = dropout_local
        self.encoder_stale_count[ids] += stale_local.long()
        self.encoder_dropout_count[ids] += dropout_local.long()

        if output_gravity is not None:
            output_gravity[ids] = self._apply_mount(gravity_body[ids], ids)
        if output_specific_force is not None:
            perturbed_accel = self._apply_mount(specific_force_body[ids], ids)
            perturbed_accel = perturbed_accel + self.accel_bias_m_s2[ids]
            if self._accel_noise_enabled:
                perturbed_accel = perturbed_accel + torch.randn(
                    (count, 3), generator=self._generator, device=self.device
                ) * self.accel_noise_std_m_s2[ids].unsqueeze(-1)
            output_specific_force[ids] = perturbed_accel

        outputs = (output_gyro, output_main, output_abad)
        optional_outputs = tuple(
            value for value in (output_gravity, output_specific_force) if value is not None
        )
        if any(not bool(torch.isfinite(value).all()) for value in outputs + optional_outputs):
            raise RuntimeError("Sensor V2 domain randomization produced NaN or Inf")
        return SensorEventV2(
            output_gyro,
            output_main,
            output_abad,
            output_gravity,
            output_specific_force,
            accepted,
            stale,
            dropout,
        )
