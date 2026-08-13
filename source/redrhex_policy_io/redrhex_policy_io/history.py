"""Strict oldest-to-newest sensor history buffering for V2."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .contracts import ContractError, StudentObservationContractV2


class SensorHistoryBufferV2:
    """A 60x36 ring buffer that never exposes zero-padded history as ready."""

    def __init__(self, contract: StudentObservationContractV2) -> None:
        if not isinstance(contract, StudentObservationContractV2):
            raise ContractError("contract must be StudentObservationContractV2")
        self.contract = contract.validate()
        self._data = np.zeros(
            (contract.history_length, contract.sensor_frame_dim), dtype=np.float32
        )
        self.reset()

    def reset(self) -> None:
        self._data.fill(0.0)
        self._next_index = 0
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    @property
    def ready(self) -> bool:
        return self._count == self.contract.history_length

    @property
    def frames_needed(self) -> int:
        return self.contract.history_length - self._count

    def append(self, frame: Sequence[float] | np.ndarray) -> None:
        values = np.asarray(frame, dtype=np.float32)
        expected = (self.contract.sensor_frame_dim,)
        if values.shape != expected or not np.isfinite(values).all():
            raise ContractError(f"sensor frame must be finite with shape {expected}")
        self._data[self._next_index] = values
        self._next_index = (self._next_index + 1) % self.contract.history_length
        self._count = min(self._count + 1, self.contract.history_length)

    def warmup(self, frames: Iterable[Sequence[float] | np.ndarray]) -> None:
        """Reset and append real chronological frames; never replicate or pad."""

        self.reset()
        for frame in frames:
            self.append(frame)

    def array(self, *, require_ready: bool = True) -> np.ndarray:
        if require_ready and not self.ready:
            raise ContractError(
                f"history is not ready: {self._count}/{self.contract.history_length} valid frames"
            )
        if self._count == 0:
            return np.empty((0, self.contract.sensor_frame_dim), dtype=np.float32)
        if self._count < self.contract.history_length:
            return self._data[: self._count].copy()
        indices = np.arange(self._next_index, self._next_index + self.contract.history_length)
        indices %= self.contract.history_length
        return self._data[indices].copy()
