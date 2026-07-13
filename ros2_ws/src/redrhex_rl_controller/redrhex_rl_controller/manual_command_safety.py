"""ROS-independent selection and finalization helpers for manual commands."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar


T = TypeVar("T")
_ALL_DISABLED = (False, False, False, False, False, False)
_ALL_ENABLED = (True, True, True, True, True, True)


def selection_for_mode(
    mode: str,
    *,
    index: int,
    enabled: bool,
) -> tuple[tuple[bool, bool, bool, bool, bool, bool], bool]:
    if not enabled or mode == "disable":
        return _ALL_DISABLED, False
    if mode == "init-stand":
        return _ALL_DISABLED, True
    if mode == "single-main-velocity":
        if not 0 <= index < 6:
            raise ValueError("--index must be 0..5 for single-main-velocity")
        return tuple(position == index for position in range(6)), False
    if mode == "all-main-velocity":
        return _ALL_ENABLED, False
    if mode == "single-abad":
        if not 0 <= index < 6:
            raise ValueError("--index must be 0..5 for single-abad")
        return _ALL_DISABLED, True
    if mode == "all-abad":
        return _ALL_DISABLED, True
    raise ValueError(f"Unsupported mode {mode}")


def run_with_terminal_disable(
    operation: Callable[[], T],
    disable_once: Callable[[], None],
    *,
    repeats: int,
) -> T:
    if repeats < 2:
        raise ValueError("terminal disable repeats must be at least 2")
    try:
        return operation()
    finally:
        for _ in range(repeats):
            disable_once()
