"""Rollout storage for V2 sensor distillation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .models import ACTION_DIM_V2, COMMAND_DIM_V2, SENSOR_FRAME_DIM_V2, SENSOR_HISTORY_LENGTH_V2


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
