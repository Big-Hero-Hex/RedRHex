"""Small tensor-only command delay primitive shared by training and playback."""

from __future__ import annotations

import torch


def advance_target_delay(
    drive_history: torch.Tensor,
    abad_history: torch.Tensor,
    cursor: int,
    requested_drive: torch.Tensor,
    requested_abad: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Store current requests and return targets requested exactly N calls earlier."""

    history_steps = int(drive_history.shape[1])
    if history_steps < 1 or int(abad_history.shape[1]) != history_steps:
        raise ValueError("target delay histories must have the same positive length")
    if cursor < 0 or cursor >= history_steps:
        raise ValueError("target delay cursor is out of range")
    if drive_history[:, cursor].shape != requested_drive.shape:
        raise ValueError("main-drive target delay shape mismatch")
    if abad_history[:, cursor].shape != requested_abad.shape:
        raise ValueError("ABAD target delay shape mismatch")

    applied_drive = drive_history[:, cursor].clone()
    applied_abad = abad_history[:, cursor].clone()
    drive_history[:, cursor].copy_(requested_drive)
    abad_history[:, cursor].copy_(requested_abad)
    return applied_drive, applied_abad, (cursor + 1) % history_steps
