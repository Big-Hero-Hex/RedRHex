"""Controlled process signals that leave ROS publishers valid during unwinding."""

from __future__ import annotations

import signal
from collections.abc import Mapping
from types import FrameType


class ControlledSignalInterrupt(KeyboardInterrupt):
    """A process stop request raised before the caller shuts down rclpy."""

    def __init__(self, signum: int) -> None:
        self.signum = int(signum)
        super().__init__(signal.Signals(self.signum).name)


def _raise_controlled_interrupt(signum: int, _frame: FrameType | None) -> None:
    raise ControlledSignalInterrupt(signum)


def install_controlled_signal_handlers() -> dict[int, object]:
    """Replace SIGINT/SIGTERM with an exception; the caller owns ROS shutdown."""

    previous: dict[int, object] = {}
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, _raise_controlled_interrupt)
    except BaseException:
        restore_signal_handlers(previous)
        raise
    return previous


def restore_signal_handlers(previous: Mapping[int, object]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)
