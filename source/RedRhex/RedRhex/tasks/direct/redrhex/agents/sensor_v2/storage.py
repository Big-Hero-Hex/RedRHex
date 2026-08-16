"""Rollout storage for V2 sensor distillation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import torch

from .models import ACTION_DIM_V2, COMMAND_DIM_V2, SENSOR_FRAME_DIM_V2, SENSOR_HISTORY_LENGTH_V2


def clone_observations_v2(
    observations: Mapping[str, torch.Tensor],
    names: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    """Take ownership of rollout observations before the environment mutates live buffers."""

    return {name: observations[name].detach().clone() for name in names}


def causal_transition_mask_v2(
    dones: torch.Tensor,
    next_history_ready: torch.Tensor,
) -> torch.Tensor:
    """Return transitions whose causal next observation may be bootstrapped."""

    if dones.shape != next_history_ready.shape:
        raise ValueError(
            "dones and next_history_ready must have identical shapes; "
            f"got {tuple(dones.shape)} and {tuple(next_history_ready.shape)}"
        )
    return ~dones.bool() & next_history_ready.bool()


def has_trainable_history_v2(history_ready: torch.Tensor) -> bool:
    """Return whether this tick contains at least one causally valid actor row."""

    if history_ready.ndim != 1:
        raise ValueError(
            "history_ready must have shape [num_envs]; "
            f"got {tuple(history_ready.shape)}"
        )
    return bool(history_ready.bool().any())


def causal_gae_step_v2(
    *,
    reward: torch.Tensor,
    value: torch.Tensor,
    following_value: torch.Tensor,
    following_advantage: torch.Tensor,
    current_ready: torch.Tensor,
    transition_valid: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute one GAE step without crossing an invalid next-history boundary."""

    tensors = {
        "value": value,
        "following_value": following_value,
        "following_advantage": following_advantage,
        "current_ready": current_ready,
        "transition_valid": transition_valid,
    }
    mismatches = {
        name: tuple(tensor.shape)
        for name, tensor in tensors.items()
        if tensor.shape != reward.shape
    }
    if mismatches:
        raise ValueError(
            f"GAE tensors must match reward shape {tuple(reward.shape)}; got {mismatches}"
        )
    valid = current_ready.to(dtype=value.dtype)
    continuation = transition_valid.to(dtype=value.dtype)
    delta = (reward + gamma * continuation * following_value - value) * valid
    advantage = (
        delta + gamma * lam * continuation * following_advantage
    ) * valid
    returns = torch.where(current_ready.bool(), advantage + value, value)
    return returns, advantage


@dataclass(frozen=True)
class SensorDistillationBatchV2:
    sensor_history: torch.Tensor
    command: torch.Tensor
    teacher_actions: torch.Tensor
    student_actions: torch.Tensor
    executed_actions: torch.Tensor
    base_velocity_target: torch.Tensor
    next_sensor_frame_target: torch.Tensor
    terminal: torch.Tensor

    def validate(self) -> None:
        batch = self.sensor_history.shape[0]
        expected = {
            "sensor_history": (batch, SENSOR_HISTORY_LENGTH_V2, SENSOR_FRAME_DIM_V2),
            "command": (batch, COMMAND_DIM_V2),
            "teacher_actions": (batch, ACTION_DIM_V2),
            "student_actions": (batch, ACTION_DIM_V2),
            "executed_actions": (batch, ACTION_DIM_V2),
            "base_velocity_target": (batch, 3),
            "next_sensor_frame_target": (batch, SENSOR_FRAME_DIM_V2),
        }
        for name, shape in expected.items():
            actual = tuple(getattr(self, name).shape)
            if actual != shape:
                raise ValueError(f"{name} must have shape {shape}; got {actual}")
        if tuple(self.terminal.shape) not in ((batch,), (batch, 1)):
            raise ValueError(f"terminal must have shape ({batch},) or ({batch},1); got {self.terminal.shape}")


class SensorDistillationStorageV2:
    """Fixed-capacity tensor storage preserving all three rollout action streams."""

    def __init__(self, capacity: int, *, device: torch.device | str = "cpu") -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.sensor_history = torch.empty(capacity, SENSOR_HISTORY_LENGTH_V2, SENSOR_FRAME_DIM_V2, device=device)
        self.command = torch.empty(capacity, COMMAND_DIM_V2, device=device)
        self.teacher_actions = torch.empty(capacity, ACTION_DIM_V2, device=device)
        self.student_actions = torch.empty(capacity, ACTION_DIM_V2, device=device)
        self.executed_actions = torch.empty(capacity, ACTION_DIM_V2, device=device)
        self.base_velocity_target = torch.empty(capacity, 3, device=device)
        self.next_sensor_frame_target = torch.empty(capacity, SENSOR_FRAME_DIM_V2, device=device)
        self.terminal = torch.empty(capacity, 1, dtype=torch.bool, device=device)
        self.size = 0

    def clear(self) -> None:
        self.size = 0

    @torch.no_grad()
    def add(self, batch: SensorDistillationBatchV2) -> None:
        batch.validate()
        count = batch.sensor_history.shape[0]
        if self.size + count > self.capacity:
            raise OverflowError(
                f"distillation storage capacity {self.capacity} exceeded by {self.size + count} samples"
            )
        destination = slice(self.size, self.size + count)
        for name in (
            "sensor_history",
            "command",
            "teacher_actions",
            "student_actions",
            "executed_actions",
            "base_velocity_target",
            "next_sensor_frame_target",
        ):
            getattr(self, name)[destination].copy_(getattr(batch, name).to(self.device))
        self.terminal[destination].copy_(batch.terminal.reshape(count, 1).to(self.device, dtype=torch.bool))
        self.size += count

    def batch(self) -> SensorDistillationBatchV2:
        if self.size == 0:
            raise RuntimeError("distillation storage is empty")
        valid = slice(0, self.size)
        return SensorDistillationBatchV2(
            sensor_history=self.sensor_history[valid],
            command=self.command[valid],
            teacher_actions=self.teacher_actions[valid],
            student_actions=self.student_actions[valid],
            executed_actions=self.executed_actions[valid],
            base_velocity_target=self.base_velocity_target[valid],
            next_sensor_frame_target=self.next_sensor_frame_target[valid],
            terminal=self.terminal[valid],
        )

    def mini_batches(
        self,
        num_mini_batches: int,
        *,
        max_batch_size: int | None = None,
        shuffle: bool = True,
    ) -> Iterator[SensorDistillationBatchV2]:
        """Yield a complete partition without materializing the full TCN graph at once."""

        if num_mini_batches <= 0:
            raise ValueError("num_mini_batches must be positive")
        if max_batch_size is not None and max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        batch = self.batch()
        required_for_cap = (
            (self.size + max_batch_size - 1) // max_batch_size
            if max_batch_size is not None
            else 1
        )
        partition_count = min(max(num_mini_batches, required_for_cap), self.size)
        indices = (
            torch.randperm(self.size, device=self.device)
            if shuffle
            else torch.arange(self.size, device=self.device)
        )
        for index in torch.tensor_split(indices, partition_count):
            yield SensorDistillationBatchV2(
                sensor_history=batch.sensor_history[index],
                command=batch.command[index],
                teacher_actions=batch.teacher_actions[index],
                student_actions=batch.student_actions[index],
                executed_actions=batch.executed_actions[index],
                base_velocity_target=batch.base_velocity_target[index],
                next_sensor_frame_target=batch.next_sensor_frame_target[index],
                terminal=batch.terminal[index],
            )
