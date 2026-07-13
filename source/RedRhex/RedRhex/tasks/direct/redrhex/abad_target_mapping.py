"""Tensor-only calibrated ABAD target mapping shared by training and playback."""

from __future__ import annotations

import torch


def map_abad_targets(
    requested: torch.Tensor,
    *,
    scale: torch.Tensor,
    offset: torch.Tensor,
) -> torch.Tensor:
    """Return actual targets from ``actual = scale * requested + offset``."""

    if scale.shape != requested.shape or offset.shape != requested.shape:
        raise ValueError("ABAD target mapping tensors must match the requested shape")
    if not torch.isfinite(requested).all() or not torch.isfinite(scale).all() or not torch.isfinite(offset).all():
        raise ValueError("ABAD target mapping tensors must be finite")
    if torch.any(scale <= 0.0):
        raise ValueError("ABAD target mapping scale must be positive")
    return scale * requested + offset
