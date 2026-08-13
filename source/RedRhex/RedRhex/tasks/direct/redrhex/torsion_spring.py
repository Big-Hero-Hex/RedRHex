"""Passive linear torsion-spring calculations shared by RedRHex backends."""

from __future__ import annotations

import math

import torch


SPRING_BACKENDS = ("explicit", "native")


def reorder_joint_parameters(
    values: torch.Tensor,
    canonical_joint_names: list[str] | tuple[str, ...],
    target_joint_names: list[str] | tuple[str, ...],
) -> torch.Tensor:
    """Reorder a final joint dimension from canonical to target name order."""
    canonical = list(canonical_joint_names)
    target = list(target_joint_names)
    if (
        len(canonical) != len(target)
        or len(set(canonical)) != len(canonical)
        or set(canonical) != set(target)
        or values.shape[-1] != len(canonical)
    ):
        raise ValueError("canonical and target spring orders must contain the same unique joints")
    canonical_index = {name: index for index, name in enumerate(canonical)}
    indices = torch.tensor(
        [canonical_index[name] for name in target],
        device=values.device,
        dtype=torch.long,
    )
    return values.index_select(-1, indices)


def unwrap_continuous_position(
    position: torch.Tensor,
    previous_wrapped_position: torch.Tensor,
    previous_unwrapped_position: torch.Tensor,
    predicted_delta: torch.Tensor | None = None,
) -> torch.Tensor:
    """Accumulate continuous displacement, using velocity to resolve full turns."""

    wrapped_delta = torch.remainder(
        position - previous_wrapped_position + math.pi, 2.0 * math.pi
    ) - math.pi
    if predicted_delta is not None:
        turns = torch.round(
            (predicted_delta - wrapped_delta) / (2.0 * math.pi)
        )
        wrapped_delta = wrapped_delta + turns * (2.0 * math.pi)
    return previous_unwrapped_position + wrapped_delta


def unwrap_ambiguity_mask(velocity: torch.Tensor, dt_s: float) -> torch.Tensor:
    """Flag samples whose travel can exceed the uniquely observable half turn."""

    if dt_s <= 0.0 or not math.isfinite(float(dt_s)):
        raise ValueError("spring unwrap dt must be positive and finite")
    return torch.abs(velocity) * float(dt_s) >= math.pi


def actuator_gains(backend: str, stiffness: float, damping: float) -> tuple[float, float]:
    """Return PhysX drive gains for the selected spring backend."""
    if backend == "explicit":
        return 0.0, 0.0
    if backend == "native":
        return float(stiffness), float(damping)
    raise ValueError(
        f"unsupported spring backend {backend!r}; expected one of {SPRING_BACKENDS}"
    )


def restoring_torque(
    position: torch.Tensor,
    velocity: torch.Tensor,
    rest_position: torch.Tensor,
    stiffness: torch.Tensor,
    damping: torch.Tensor,
) -> torch.Tensor:
    """Return passive restoring torque without wrapping continuous-joint angles."""
    return -stiffness * (position - rest_position) - damping * velocity


def potential_energy(
    position: torch.Tensor,
    rest_position: torch.Tensor,
    stiffness: torch.Tensor,
) -> torch.Tensor:
    """Return elastic potential energy per spring joint."""
    return 0.5 * stiffness * torch.square(position - rest_position)


def mechanical_power(torque: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
    """Return mechanical power delivered by each spring to the articulation."""
    return torque * velocity


def trapezoidal_energy_increment(
    previous_power: torch.Tensor,
    current_power: torch.Tensor,
    dt_s: float,
) -> torch.Tensor:
    """Integrate one power interval with the trapezoidal rule."""
    if dt_s <= 0.0 or not math.isfinite(float(dt_s)):
        raise ValueError("spring energy integration dt must be positive and finite")
    return 0.5 * (previous_power + current_power) * float(dt_s)


def energy_work_residual(
    current_energy: torch.Tensor,
    reference_energy: torch.Tensor,
    cumulative_work: torch.Tensor,
    cumulative_dissipation: torch.Tensor,
) -> torch.Tensor:
    """Return signed ΔU + spring work + dissipated-energy residual."""
    return (
        current_energy
        - reference_energy
        + cumulative_work
        + cumulative_dissipation
    )
