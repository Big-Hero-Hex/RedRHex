"""Dependency-light derivation of simulator values serialized into V2 bundles."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def causal_attitude_parameters_v2(
    *,
    correction_gain: float,
    accel_norm_gate_m_s2: Sequence[float],
    gravity_vector_m_s2: Sequence[float],
) -> tuple[tuple[str, float], ...]:
    """Translate simulator settings into the keys consumed by SensorFrameBuilderV2."""

    gain = float(correction_gain)
    gate = tuple(float(value) for value in accel_norm_gate_m_s2)
    gravity = tuple(float(value) for value in gravity_vector_m_s2)
    if not 0.0 <= gain <= 1.0:
        raise ValueError("sensor gravity correction gain must be in [0, 1]")
    if len(gate) != 2 or len(gravity) != 3:
        raise ValueError("acceleration gate and gravity vector must have lengths 2 and 3")
    if not all(math.isfinite(value) for value in (*gate, *gravity)):
        raise ValueError("acceleration gate and gravity vector must be finite")
    gravity_magnitude = math.sqrt(sum(value * value for value in gravity))
    low, high = gate
    if gravity_magnitude <= 0.0 or not 0.0 < low <= gravity_magnitude <= high:
        raise ValueError("acceleration gate must contain the configured gravity magnitude")
    tolerance_ratio = max(gravity_magnitude - low, high - gravity_magnitude) / gravity_magnitude
    return (
        ("accel_correction_gain", gain),
        ("accel_magnitude_tolerance_ratio", tolerance_ratio),
        ("gravity_magnitude_m_s2", gravity_magnitude),
    )


def rest_projected_gravity_v2(reference: object | None) -> tuple[float, float, float]:
    """Return one validated unit rest-gravity direction from an environment buffer."""

    if reference is None:
        return (0.0, 0.0, -1.0)
    values = torch.as_tensor(reference).detach().to(device="cpu", dtype=torch.float64)
    if values.numel() == 0 or values.shape[-1:] != (3,):
        raise ValueError("reference_projected_gravity must end in dimension 3")
    rows = values.reshape(-1, 3)
    if not torch.isfinite(rows).all():
        raise ValueError("reference_projected_gravity must be finite")
    norms = torch.linalg.vector_norm(rows, dim=-1, keepdim=True)
    if torch.any(norms <= 1.0e-12):
        raise ValueError("reference_projected_gravity must be non-zero")
    unit_rows = rows / norms
    if not torch.allclose(unit_rows, unit_rows[:1].expand_as(unit_rows), atol=1.0e-6, rtol=0.0):
        raise ValueError("reference_projected_gravity must agree across environments")
    return tuple(float(value) for value in unit_rows[0].tolist())


def environment_rest_projected_gravity_v2(env: object) -> tuple[float, float, float]:
    """Resolve the reference buffer through the RSL-RL wrapper when present."""

    unwrapped = getattr(env, "unwrapped", env)
    return rest_projected_gravity_v2(getattr(unwrapped, "reference_projected_gravity", None))
