"""Tensor-only calibrated ABAD target mapping shared by training and playback."""

from __future__ import annotations

import torch


def map_abad_targets(
    requested: torch.Tensor,
    *,
    scale: torch.Tensor,
    offset: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Map measured targets, then clamp them to the physical joint range."""

    if any(
        value.shape != requested.shape for value in (scale, offset, lower, upper)
    ):
        raise ValueError("ABAD target mapping tensors must match the requested shape")
    if not all(
        torch.isfinite(value).all()
        for value in (requested, scale, offset, lower, upper)
    ):
        raise ValueError("ABAD target mapping tensors must be finite")
    if torch.any(scale <= 0.0):
        raise ValueError("ABAD target mapping scale must be positive")
    if torch.any(lower >= upper):
        raise ValueError("ABAD target mapping bounds must satisfy lower < upper")
    mapped = scale * requested + offset
    return torch.maximum(torch.minimum(mapped, upper), lower)
