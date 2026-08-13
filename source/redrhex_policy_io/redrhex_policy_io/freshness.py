"""Per-channel monotonicity, validity, and freshness gates for V2 events."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from .contracts import ContractError


@dataclass(frozen=True)
class ChannelFreshnessReportV2:
    ready: bool
    missing: tuple[str, ...]
    invalid: tuple[str, ...]
    stale: tuple[str, ...]


class ChannelFreshnessTrackerV2:
    def __init__(
        self,
        required_channels: Sequence[str],
        *,
        max_age_s: float | Mapping[str, float],
    ) -> None:
        if isinstance(required_channels, (str, bytes)) or not required_channels:
            raise ContractError("required_channels must be a non-empty sequence")
        self.required_channels = tuple(str(name) for name in required_channels)
        if any(not name for name in self.required_channels) or len(set(self.required_channels)) != len(
            self.required_channels
        ):
            raise ContractError("required_channels must be non-empty and unique")
        if isinstance(max_age_s, Mapping):
            if set(max_age_s) != set(self.required_channels):
                raise ContractError("max_age_s mapping must cover required channels exactly")
            ages = {name: float(max_age_s[name]) for name in self.required_channels}
        else:
            ages = {name: float(max_age_s) for name in self.required_channels}
        if any(not math.isfinite(value) or value <= 0.0 for value in ages.values()):
            raise ContractError("max ages must be positive and finite")
        self.max_age_s = ages
        self.reset()

    def reset(self) -> None:
        self._timestamps: dict[str, float] = {}
        self._valid: dict[str, bool] = {}

    def update(self, channel: str, source_timestamp_s: float, *, valid: bool = True) -> None:
        if channel not in self.max_age_s:
            raise ContractError(f"unknown channel {channel!r}")
        timestamp = float(source_timestamp_s)
        if not math.isfinite(timestamp):
            raise ContractError("source timestamp must be finite")
        previous = self._timestamps.get(channel)
        if previous is not None and timestamp <= previous:
            relation = "repeated" if timestamp == previous else "out-of-order"
            raise ContractError(f"{channel} source timestamp is {relation}")
        self._timestamps[channel] = timestamp
        self._valid[channel] = bool(valid)

    def invalidate(self, channel: str) -> None:
        if channel not in self.max_age_s:
            raise ContractError(f"unknown channel {channel!r}")
        self._valid[channel] = False

    def report(self, now_s: float) -> ChannelFreshnessReportV2:
        now = float(now_s)
        if not math.isfinite(now):
            raise ContractError("now_s must be finite")
        missing = tuple(name for name in self.required_channels if name not in self._timestamps)
        invalid = tuple(
            name
            for name in self.required_channels
            if name in self._timestamps and not self._valid.get(name, False)
        )
        stale = tuple(
            name
            for name in self.required_channels
            if name in self._timestamps
            and (now < self._timestamps[name] or now - self._timestamps[name] > self.max_age_s[name])
        )
        return ChannelFreshnessReportV2(
            ready=not missing and not invalid and not stale,
            missing=missing,
            invalid=invalid,
            stale=stale,
        )
