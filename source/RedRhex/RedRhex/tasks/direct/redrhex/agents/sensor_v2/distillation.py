"""Losses and optimization for sensor-only teacher/student distillation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .models import MAIN_ACTION_DIM_V2, SensorStudentCoreV2
from .storage import SensorDistillationBatchV2


@dataclass(frozen=True)
class SensorDistillationLossWeightsV2:
    """Approved strict-forward F2 loss weights."""

    main_drive: float = 1.0
    forward_abad: float = 0.0
    base_velocity: float = 0.5
    next_frame: float = 0.1
    latent_regularization: float = 1.0e-4
    contact: float = 0.0

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if value < 0.0:
                raise ValueError(f"{name} weight must be non-negative")
        if self.contact != 0.0:
            raise ValueError("contact supervision is unavailable in V2 and its weight must remain zero")


def _masked_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Return masked Huber without producing NaN for an all-invalid batch."""

    per_element = F.huber_loss(prediction, target, reduction="none")
    per_sample = per_element.mean(dim=-1)
    mask = valid.reshape(-1).to(dtype=per_sample.dtype, device=per_sample.device)
    return (per_sample * mask).sum() / mask.sum().clamp_min(1.0)


class SensorDistillationV2:
    """Pure-PyTorch F2 optimizer with split, inspectable loss terms."""

    def __init__(
        self,
        student: SensorStudentCoreV2,
        *,
        learning_rate: float = 1.0e-3,
        max_grad_norm: float = 1.0,
        weights: SensorDistillationLossWeightsV2 | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        self.student = student
        self.weights = weights or SensorDistillationLossWeightsV2()
        self.max_grad_norm = float(max_grad_norm)
        self.optimizer = optimizer or torch.optim.Adam(student.parameters(), lr=learning_rate)

    def loss(self, batch: SensorDistillationBatchV2) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch.validate()
        actions, base_velocity, next_frame = self.student(batch.sensor_history, batch.command)
        latent, _ = self.student.encode(batch.sensor_history, batch.command)
        teacher_actions = batch.teacher_actions.detach()
        base_velocity_target = batch.base_velocity_target.detach()
        next_frame_target = batch.next_sensor_frame_target.detach()

        main_loss = F.huber_loss(
            actions[..., :MAIN_ACTION_DIM_V2],
            teacher_actions[..., :MAIN_ACTION_DIM_V2],
        )
        abad_loss = F.huber_loss(
            actions[..., MAIN_ACTION_DIM_V2:],
            teacher_actions[..., MAIN_ACTION_DIM_V2:],
        )
        velocity_loss = F.huber_loss(base_velocity, base_velocity_target)
        dynamics_loss = _masked_huber(next_frame, next_frame_target, ~batch.terminal.bool())
        latent_loss = latent.square().mean()
        total = (
            self.weights.main_drive * main_loss
            + self.weights.forward_abad * abad_loss
            + self.weights.base_velocity * velocity_loss
            + self.weights.next_frame * dynamics_loss
            + self.weights.latent_regularization * latent_loss
        )

        with torch.no_grad():
            disagreement = (actions - teacher_actions).abs()
            velocity_error = base_velocity - base_velocity_target
            metrics = {
                "loss/total": total.detach(),
                "loss/main_drive_huber": main_loss.detach(),
                "loss/forward_abad_huber": abad_loss.detach(),
                "loss/base_velocity_huber": velocity_loss.detach(),
                "loss/next_frame_huber": dynamics_loss.detach(),
                "loss/latent_regularization": latent_loss.detach(),
                "mae/main_drive": disagreement[..., :MAIN_ACTION_DIM_V2].mean(),
                "mae/forward_abad": disagreement[..., MAIN_ACTION_DIM_V2:].mean(),
                "mae/base_velocity": (base_velocity - base_velocity_target).abs().mean(),
                "rmse/base_velocity_x": velocity_error[..., 0].square().mean().sqrt(),
                "rmse/base_velocity_y": velocity_error[..., 1].square().mean().sqrt(),
                "rmse/base_velocity_z": velocity_error[..., 2].square().mean().sqrt(),
                "mae/next_frame": (next_frame - next_frame_target).abs().mean(),
                "rollout/teacher_student_disagreement": disagreement.mean(),
                "rollout/teacher_saturation": (teacher_actions.abs() >= 1.0).float().mean(),
                "rollout/student_saturation": (actions.abs() >= 1.0).float().mean(),
                "rollout/executed_saturation": (batch.executed_actions.abs() >= 1.0).float().mean(),
            }
        return total, metrics

    def update(self, batch: SensorDistillationBatchV2) -> dict[str, float]:
        self.optimizer.zero_grad(set_to_none=True)
        total, metrics = self.loss(batch)
        if not torch.isfinite(total):
            raise FloatingPointError("non-finite V2 distillation loss")
        total.backward()
        gradient_norm = nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
        if not torch.isfinite(gradient_norm):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("non-finite V2 distillation gradient")
        self.optimizer.step()
        result = {name: float(value.item()) for name, value in metrics.items()}
        result["optimizer/gradient_norm"] = float(gradient_norm.item())
        return result
