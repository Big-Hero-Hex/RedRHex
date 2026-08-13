"""Sensor-only V2 policy networks.

This module intentionally depends only on PyTorch.  Isaac Lab and rsl-rl
adapters live outside the model so the deployment graph can be tested without
starting a simulator.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


SENSOR_FRAME_DIM_V2 = 36
SENSOR_HISTORY_LENGTH_V2 = 60
COMMAND_DIM_V2 = 3
MAIN_ACTION_DIM_V2 = 6
ACTION_DIM_V2 = 12
LATENT_DIM_V2 = 64


def _require_last_dim(tensor: torch.Tensor, expected: int, name: str) -> None:
    if tensor.ndim < 2 or tensor.shape[-1] != expected:
        raise ValueError(f"{name} must end in dimension {expected}; got {tuple(tensor.shape)}")


class FeaturewiseNormalizerV2(nn.Module):
    """Featurewise population-statistics normalizer with serializable state."""

    def __init__(self, num_features: int, epsilon: float = 1.0e-5) -> None:
        super().__init__()
        if num_features <= 0:
            raise ValueError("num_features must be positive")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        self.num_features = int(num_features)
        self.epsilon = float(epsilon)
        self.register_buffer("mean", torch.zeros(num_features))
        self.register_buffer("variance", torch.ones(num_features))
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        """Merge a batch into running population mean and variance."""

        _require_last_dim(values, self.num_features, "values")
        flat = values.detach().reshape(-1, self.num_features).to(
            device=self.mean.device, dtype=self.mean.dtype
        )
        if flat.shape[0] == 0:
            return

        batch_count = torch.tensor(float(flat.shape[0]), device=self.count.device, dtype=self.count.dtype)
        batch_mean = flat.mean(dim=0)
        batch_variance = flat.var(dim=0, unbiased=False)
        if self.count.item() == 0.0:
            self.mean.copy_(batch_mean)
            self.variance.copy_(batch_variance)
            self.count.copy_(batch_count)
            return

        old_count = self.count.to(dtype=self.mean.dtype)
        new_count = batch_count.to(dtype=self.mean.dtype)
        total_count = old_count + new_count
        delta = batch_mean - self.mean
        merged_mean = self.mean + delta * (new_count / total_count)
        old_m2 = self.variance * old_count
        batch_m2 = batch_variance * new_count
        merged_m2 = old_m2 + batch_m2 + delta.square() * old_count * new_count / total_count
        self.mean.copy_(merged_mean)
        self.variance.copy_(merged_m2 / total_count)
        self.count.copy_(batch_count + self.count)

    @torch.no_grad()
    def set_statistics(
        self,
        mean: torch.Tensor,
        standard_deviation: torch.Tensor,
        *,
        count: int | float = 1,
    ) -> None:
        """Set validated statistics, typically from a training checkpoint."""

        mean = torch.as_tensor(mean, device=self.mean.device, dtype=self.mean.dtype)
        standard_deviation = torch.as_tensor(
            standard_deviation, device=self.mean.device, dtype=self.mean.dtype
        )
        if mean.shape != self.mean.shape or standard_deviation.shape != self.mean.shape:
            raise ValueError(
                "normalizer statistics must have shape "
                f"{tuple(self.mean.shape)}; got {tuple(mean.shape)} and {tuple(standard_deviation.shape)}"
            )
        if not torch.isfinite(mean).all() or not torch.isfinite(standard_deviation).all():
            raise ValueError("normalizer statistics must be finite")
        if torch.any(standard_deviation <= 0.0):
            raise ValueError("normalizer standard deviations must be positive")
        if count <= 0:
            raise ValueError("normalizer count must be positive")
        self.mean.copy_(mean)
        self.variance.copy_(standard_deviation.square())
        self.count.fill_(float(count))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        _require_last_dim(values, self.num_features, "values")
        standard_deviation = torch.sqrt(torch.clamp(self.variance, min=self.epsilon))
        return (values - self.mean) / standard_deviation


class CausalResidualBlockV2(nn.Module):
    """One-convolution residual block with left-only temporal padding."""

    def __init__(self, width: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        if kernel_size <= 0 or dilation <= 0:
            raise ValueError("kernel_size and dilation must be positive")
        self.left_padding = (kernel_size - 1) * dilation
        self.convolution = nn.Conv1d(
            width,
            width,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )
        self.activation = nn.ELU()
        # LayerNorm operates over channels at each time step, so it does not
        # couple past outputs to future samples.
        self.normalization = nn.LayerNorm(width)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        residual = sequence
        convolved = self.convolution(F.pad(sequence, (self.left_padding, 0)))
        convolved = self.activation(convolved)
        combined = (residual + convolved).transpose(1, 2)
        return self.normalization(combined).transpose(1, 2)


class CausalTCNEncoderV2(nn.Module):
    """Four-block causal TCN with an exact 61-frame receptive field."""

    kernel_size = 5
    dilations = (1, 2, 4, 8)
    receptive_field = 1 + (kernel_size - 1) * sum(dilations)

    def __init__(self, input_dim: int = SENSOR_FRAME_DIM_V2, width: int = LATENT_DIM_V2) -> None:
        super().__init__()
        if input_dim <= 0 or width <= 0:
            raise ValueError("input_dim and width must be positive")
        self.input_dim = int(input_dim)
        self.width = int(width)
        self.input_projection = nn.Linear(input_dim, width)
        self.blocks = nn.ModuleList(
            CausalResidualBlockV2(width, self.kernel_size, dilation) for dilation in self.dilations
        )

    def forward_sequence(self, sensor_history: torch.Tensor) -> torch.Tensor:
        if sensor_history.ndim != 3 or sensor_history.shape[-1] != self.input_dim:
            raise ValueError(
                f"sensor_history must have shape [batch,time,{self.input_dim}]; "
                f"got {tuple(sensor_history.shape)}"
            )
        encoded = self.input_projection(sensor_history).transpose(1, 2)
        for block in self.blocks:
            encoded = block(encoded)
        return encoded.transpose(1, 2)

    def forward(self, sensor_history: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(sensor_history)[:, -1, :]


class SensorStudentCoreV2(nn.Module):
    """Sensor-history actor with base-velocity and next-frame auxiliary heads."""

    def __init__(self) -> None:
        super().__init__()
        self.sensor_normalizer = FeaturewiseNormalizerV2(SENSOR_FRAME_DIM_V2)
        self.command_normalizer = FeaturewiseNormalizerV2(COMMAND_DIM_V2)
        self.encoder = CausalTCNEncoderV2()
        actor_input_dim = LATENT_DIM_V2 + COMMAND_DIM_V2
        self.actor_head = nn.Sequential(
            nn.Linear(actor_input_dim, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, MAIN_ACTION_DIM_V2),
            nn.Tanh(),
        )
        self.base_velocity_head = nn.Sequential(
            nn.Linear(LATENT_DIM_V2, 128),
            nn.ELU(),
            nn.Linear(128, 3),
        )
        self.next_frame_head = nn.Sequential(
            nn.Linear(actor_input_dim, 128),
            nn.ELU(),
            nn.Linear(128, SENSOR_FRAME_DIM_V2),
        )

    @torch.no_grad()
    def update_normalization(self, sensor_history: torch.Tensor, command: torch.Tensor) -> None:
        self._validate_inputs(sensor_history, command)
        self.sensor_normalizer.update(sensor_history)
        self.command_normalizer.update(command)

    @staticmethod
    def _validate_inputs(sensor_history: torch.Tensor, command: torch.Tensor) -> None:
        if sensor_history.ndim != 3 or sensor_history.shape[1:] != (
            SENSOR_HISTORY_LENGTH_V2,
            SENSOR_FRAME_DIM_V2,
        ):
            raise ValueError(
                "sensor_history must have shape "
                f"[batch,{SENSOR_HISTORY_LENGTH_V2},{SENSOR_FRAME_DIM_V2}]; "
                f"got {tuple(sensor_history.shape)}"
            )
        if command.ndim != 2 or command.shape[-1] != COMMAND_DIM_V2:
            raise ValueError(
                f"command must have shape [batch,{COMMAND_DIM_V2}]; got {tuple(command.shape)}"
            )
        if command.shape[0] != sensor_history.shape[0]:
            raise ValueError("sensor_history and command batch dimensions must match")

    def encode(self, sensor_history: torch.Tensor, command: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(sensor_history, command)
        normalized_history = self.sensor_normalizer(sensor_history)
        normalized_command = self.command_normalizer(command)
        latent = self.encoder(normalized_history)
        return latent, normalized_command

    def forward(
        self, sensor_history: torch.Tensor, command: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent, normalized_command = self.encode(sensor_history, command)
        actor_input = torch.cat((latent, normalized_command), dim=-1)
        main_residuals = self.actor_head(actor_input)
        neutral_abad = torch.zeros_like(main_residuals)
        actions = torch.cat((main_residuals, neutral_abad), dim=-1)
        base_velocity_estimate = self.base_velocity_head(latent)
        next_sensor_frame = self.next_frame_head(actor_input)
        return actions, base_velocity_estimate, next_sensor_frame

    def act(self, sensor_history: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        return self(sensor_history, command)[0]


class RolloutActionsV2(NamedTuple):
    teacher: torch.Tensor
    student: torch.Tensor
    executed: torch.Tensor
    base_velocity_estimate: torch.Tensor
    next_sensor_frame: torch.Tensor


class SensorStudentTeacherV2(nn.Module):
    """Frozen teacher and sensor-only student with rollout action mixing."""

    def __init__(self, teacher: nn.Module, student: SensorStudentCoreV2 | None = None) -> None:
        super().__init__()
        self.teacher = teacher
        self.student = student if student is not None else SensorStudentCoreV2()
        self.teacher.requires_grad_(False)
        self.teacher.eval()

    def train(self, mode: bool = True) -> "SensorStudentTeacherV2":
        super().train(mode)
        self.teacher.eval()
        return self

    @torch.no_grad()
    def teacher_actions(self, privileged_observation: torch.Tensor) -> torch.Tensor:
        if hasattr(self.teacher, "act_inference"):
            actions = self.teacher.act_inference(privileged_observation)
        else:
            actions = self.teacher(privileged_observation)
        if isinstance(actions, (tuple, list)):
            actions = actions[0]
        if not isinstance(actions, torch.Tensor):
            raise TypeError("teacher must return a tensor or a tuple whose first item is a tensor")
        _require_last_dim(actions, ACTION_DIM_V2, "teacher actions")
        return actions

    def rollout(
        self,
        sensor_history: torch.Tensor,
        command: torch.Tensor,
        privileged_observation: torch.Tensor,
        *,
        beta: float,
        noise_std: float,
        generator: torch.Generator | None = None,
    ) -> RolloutActionsV2:
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        if noise_std < 0.0:
            raise ValueError("noise_std must be non-negative")
        teacher_actions = self.teacher_actions(privileged_observation)
        student_actions, velocity, next_frame = self.student(sensor_history, command)
        if teacher_actions.shape != student_actions.shape:
            raise ValueError(
                f"teacher/student action shapes differ: {teacher_actions.shape} and {student_actions.shape}"
            )
        if noise_std:
            noise = torch.randn(
                student_actions.shape,
                device=student_actions.device,
                dtype=student_actions.dtype,
                generator=generator,
            ) * noise_std
        else:
            noise = torch.zeros_like(student_actions)
        executed = torch.clamp(beta * teacher_actions + (1.0 - beta) * student_actions + noise, -1.0, 1.0)
        # Strict forward V2 never executes learned/noisy ABAD outputs.
        executed = torch.cat((executed[..., :MAIN_ACTION_DIM_V2], torch.zeros_like(executed[..., 6:12])), dim=-1)
        return RolloutActionsV2(teacher_actions, student_actions, executed, velocity, next_frame)
