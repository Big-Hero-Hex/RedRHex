"""Passive linear torsion-spring calculations shared by RedRHex backends."""

from __future__ import annotations

import torch


SPRING_BACKENDS = ("explicit", "native")


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
