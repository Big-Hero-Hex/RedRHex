"""Annealing schedules used by Sensor-Only Distillation V2."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RolloutMixtureScheduleV2:
    """Teacher/noise schedule with a deterministic student-only final phase."""

    total_updates: int
    anneal_fraction: float = 0.70
    initial_noise_std: float = 0.05

    def __post_init__(self) -> None:
        if self.total_updates <= 0:
            raise ValueError("total_updates must be positive")
        if not 0.0 < self.anneal_fraction <= 1.0:
            raise ValueError("anneal_fraction must be in (0, 1]")
        if self.initial_noise_std < 0.0:
            raise ValueError("initial_noise_std must be non-negative")

    @property
    def anneal_updates(self) -> float:
        return self.total_updates * self.anneal_fraction

    def value(self, update: int) -> tuple[float, float]:
        if update < 0:
            raise ValueError("update must be non-negative")
        progress = min(float(update) / self.anneal_updates, 1.0)
        beta = 1.0 - progress
        noise_std = self.initial_noise_std * (1.0 - progress)
        return beta, noise_std


@dataclass(frozen=True)
class LinearWeightScheduleV2:
    """Linearly anneal a loss coefficient to zero."""

    initial_weight: float
    total_updates: int
    anneal_fraction: float

    def __post_init__(self) -> None:
        if self.initial_weight < 0.0:
            raise ValueError("initial_weight must be non-negative")
        if self.total_updates <= 0:
            raise ValueError("total_updates must be positive")
        if not 0.0 < self.anneal_fraction <= 1.0:
            raise ValueError("anneal_fraction must be in (0, 1]")

    def value(self, update: int) -> float:
        if update < 0:
            raise ValueError("update must be non-negative")
        progress = min(float(update) / (self.total_updates * self.anneal_fraction), 1.0)
        return self.initial_weight * (1.0 - progress)
