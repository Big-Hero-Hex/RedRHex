"""Asymmetric actor/critic and focused PPO update for Sensor V2."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from .distillation import SensorDistillationLossWeightsV2, _masked_huber
from .models import (
    ACTION_DIM_V2,
    MAIN_ACTION_DIM_V2,
    SensorStudentCoreV2,
    pad_main_actions_v2,
    strict_forward_actions_v2,
)


INITIAL_MAIN_ACTION_NOISE_STD_V2 = 0.05
MAX_INITIAL_MAIN_ACTION_NOISE_STD_V2 = 0.10


def _observation(observations: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
    try:
        return observations[name]
    except KeyError as exc:
        raise KeyError(f"V2 observation group {name!r} is required") from exc


class SensorActorCriticV2(nn.Module):
    """Sensor-only actor paired with a physically privileged critic."""

    is_recurrent = False

    def __init__(
        self,
        critic_observation_dim: int,
        *,
        student: SensorStudentCoreV2 | None = None,
        critic_hidden_dims: tuple[int, ...] = (256, 128, 128),
        init_noise_std: float = INITIAL_MAIN_ACTION_NOISE_STD_V2,
    ) -> None:
        super().__init__()
        if critic_observation_dim <= 0:
            raise ValueError("critic_observation_dim must be positive")
        if not 0.0 < init_noise_std <= MAX_INITIAL_MAIN_ACTION_NOISE_STD_V2:
            raise ValueError(
                "init_noise_std must be positive and at most "
                f"{MAX_INITIAL_MAIN_ACTION_NOISE_STD_V2} for strict-forward Sensor PPO V2"
            )
        self.initial_noise_std = float(init_noise_std)
        self.actor = student if student is not None else SensorStudentCoreV2()
        layers: list[nn.Module] = []
        input_dim = critic_observation_dim
        for output_dim in critic_hidden_dims:
            layers.extend((nn.Linear(input_dim, output_dim), nn.ELU()))
            input_dim = output_dim
        layers.append(nn.Linear(input_dim, 1))
        self.critic = nn.Sequential(*layers)
        self.std = nn.Parameter(torch.full((MAIN_ACTION_DIM_V2,), self.initial_noise_std))
        self.distribution: Normal | None = None

    @staticmethod
    def _actor_inputs(observations: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        history_name = "sensor_history_v2" if "sensor_history_v2" in observations else "sensor_history"
        command_name = "command_v2" if "command_v2" in observations else "command"
        return _observation(observations, history_name), _observation(observations, command_name)

    def update_distribution(self, observations: Mapping[str, torch.Tensor]) -> None:
        history, command = self._actor_inputs(observations)
        mean = self.actor.act(history, command)[..., :MAIN_ACTION_DIM_V2]
        # A small positive floor prevents an invalid distribution if optimizer
        # momentum briefly drives a learned standard deviation through zero.
        self.distribution = Normal(mean, self.main_action_std.expand_as(mean))

    def act(self, observations: Mapping[str, torch.Tensor], **_: object) -> torch.Tensor:
        self.update_distribution(observations)
        assert self.distribution is not None
        return pad_main_actions_v2(self.distribution.sample())

    def act_inference(self, observations: Mapping[str, torch.Tensor]) -> torch.Tensor:
        history, command = self._actor_inputs(observations)
        return self.actor.act(history, command)

    def evaluate(self, observations: Mapping[str, torch.Tensor], **_: object) -> torch.Tensor:
        privileged = _observation(observations, "critic_privileged_v2")
        return self.critic(privileged)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("call act or update_distribution before requesting action log probability")
        if actions.ndim < 2 or actions.shape[-1] != ACTION_DIM_V2:
            raise ValueError(f"actions must end in dimension {ACTION_DIM_V2}; got {tuple(actions.shape)}")
        return self.distribution.log_prob(actions[..., :MAIN_ACTION_DIM_V2]).sum(dim=-1)

    @property
    def action_mean(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("action distribution has not been initialized")
        return pad_main_actions_v2(self.distribution.mean)

    @property
    def action_std(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("action distribution has not been initialized")
        return pad_main_actions_v2(self.distribution.stddev)

    @property
    def main_action_std(self) -> torch.Tensor:
        return self.std.abs().clamp_min(1.0e-6)

    @property
    def entropy(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("action distribution has not been initialized")
        return self.distribution.entropy().sum(dim=-1)

    @torch.no_grad()
    def update_normalization(self, observations: Mapping[str, torch.Tensor]) -> None:
        history, command = self._actor_inputs(observations)
        self.actor.update_normalization(history, command)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        del dones

    def bootstrap_distilled_actor(self, distilled_student: SensorStudentCoreV2) -> None:
        """Strictly copy a distilled actor while leaving critic/noise untouched."""

        source = distilled_student.state_dict()
        self.actor.load_state_dict(source, strict=True)
        copied = self.actor.state_dict()
        if source.keys() != copied.keys() or any(
            not torch.equal(source[name].detach().cpu(), copied[name].detach().cpu()) for name in source
        ):
            raise RuntimeError("distilled actor bootstrap failed exact equality verification")


@dataclass(frozen=True)
class SensorPPOBatchV2:
    observations: Mapping[str, torch.Tensor]
    actions: torch.Tensor
    old_action_log_probability: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    teacher_actions: torch.Tensor
    base_velocity_target: torch.Tensor
    next_sensor_frame_target: torch.Tensor
    terminal: torch.Tensor


class SensorPPOV2:
    """One-minibatch standard PPO objective plus annealed V2 supervision."""

    def __init__(
        self,
        policy: SensorActorCriticV2,
        *,
        learning_rate: float = 3.0e-4,
        clip_param: float = 0.2,
        value_loss_coefficient: float = 1.0,
        entropy_coefficient: float = 0.003,
        max_grad_norm: float = 1.0,
        auxiliary_weights: SensorDistillationLossWeightsV2 | None = None,
    ) -> None:
        self.policy = policy
        self.clip_param = float(clip_param)
        self.value_loss_coefficient = float(value_loss_coefficient)
        self.entropy_coefficient = float(entropy_coefficient)
        self.max_grad_norm = float(max_grad_norm)
        self.auxiliary_weights = auxiliary_weights or SensorDistillationLossWeightsV2()
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    def loss(
        self,
        batch: SensorPPOBatchV2,
        *,
        teacher_bc_coefficient: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if teacher_bc_coefficient < 0.0:
            raise ValueError("teacher_bc_coefficient must be non-negative")
        self.policy.update_distribution(batch.observations)
        action_log_probability = self.policy.get_actions_log_prob(batch.actions)
        value = self.policy.evaluate(batch.observations)
        entropy = self.policy.entropy
        old_log_probability = batch.old_action_log_probability.reshape_as(action_log_probability)
        advantage = batch.advantages.reshape_as(action_log_probability)
        ratio = torch.exp(action_log_probability - old_log_probability)
        surrogate = -advantage * ratio
        clipped_surrogate = -advantage * ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)
        surrogate_loss = torch.maximum(surrogate, clipped_surrogate).mean()

        returns = batch.returns.reshape_as(value)
        old_values = batch.old_values.reshape_as(value)
        clipped_value = old_values + (value - old_values).clamp(-self.clip_param, self.clip_param)
        value_loss = torch.maximum((value - returns).square(), (clipped_value - returns).square()).mean()

        history, command = self.policy._actor_inputs(batch.observations)
        predicted_actions, velocity, next_frame = self.policy.actor(history, command)
        teacher_actions = strict_forward_actions_v2(batch.teacher_actions.detach())
        bc_main = F.huber_loss(predicted_actions[..., :MAIN_ACTION_DIM_V2], teacher_actions[..., :6])
        bc_abad = F.huber_loss(predicted_actions[..., 6:12], teacher_actions[..., 6:12])
        velocity_loss = F.huber_loss(velocity, batch.base_velocity_target.detach())
        dynamics_loss = _masked_huber(next_frame, batch.next_sensor_frame_target.detach(), ~batch.terminal.bool())
        latent, _ = self.policy.actor.encode(history, command)
        latent_loss = latent.square().mean()

        total = (
            surrogate_loss
            + self.value_loss_coefficient * value_loss
            - self.entropy_coefficient * entropy.mean()
            + teacher_bc_coefficient
            * (
                self.auxiliary_weights.main_drive * bc_main
                + self.auxiliary_weights.forward_abad * bc_abad
            )
            + self.auxiliary_weights.base_velocity * velocity_loss
            + self.auxiliary_weights.next_frame * dynamics_loss
            + self.auxiliary_weights.latent_regularization * latent_loss
        )
        metrics = {
            "loss/total": total.detach(),
            "loss/surrogate": surrogate_loss.detach(),
            "loss/value": value_loss.detach(),
            "loss/teacher_bc_main": bc_main.detach(),
            "loss/teacher_bc_abad": bc_abad.detach(),
            "loss/base_velocity": velocity_loss.detach(),
            "loss/next_frame": dynamics_loss.detach(),
            "loss/latent_regularization": latent_loss.detach(),
            "loss/teacher_bc_coefficient": torch.as_tensor(
                teacher_bc_coefficient, device=total.device, dtype=total.dtype
            ),
            "policy/entropy": entropy.mean().detach(),
            "rmse/base_velocity_x": (
                velocity[..., 0] - batch.base_velocity_target[..., 0]
            ).square().mean().sqrt().detach(),
            "rmse/base_velocity_y": (
                velocity[..., 1] - batch.base_velocity_target[..., 1]
            ).square().mean().sqrt().detach(),
            "rmse/base_velocity_z": (
                velocity[..., 2] - batch.base_velocity_target[..., 2]
            ).square().mean().sqrt().detach(),
        }
        return total, metrics

    def update(self, batch: SensorPPOBatchV2, *, teacher_bc_coefficient: float) -> dict[str, float]:
        self.optimizer.zero_grad(set_to_none=True)
        total, metrics = self.loss(batch, teacher_bc_coefficient=teacher_bc_coefficient)
        if not torch.isfinite(total):
            raise FloatingPointError("non-finite Sensor PPO V2 loss")
        total.backward()
        gradient_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        if not torch.isfinite(gradient_norm):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("non-finite Sensor PPO V2 gradient")
        self.optimizer.step()
        result = {name: float(value.item()) for name, value in metrics.items()}
        result["optimizer/gradient_norm"] = float(gradient_norm.item())
        return result
