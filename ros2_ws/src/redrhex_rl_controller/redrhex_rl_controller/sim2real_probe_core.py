"""ROS-independent schedule and fail-closed execution for the bench probe."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


RATE_HZ = 60.0
REPEATS = 3
COMMAND_SPEED_RAD_S = 0.25
NEUTRAL_DURATION_S = 0.5
DRIVE_DURATION_S = 1.0
COAST_DURATION_S = 1.0
INPUT_FRESHNESS_TIMEOUT_S = 0.25
TERMINAL_DISABLE_PACKETS = 5
TERMINAL_DISABLE_PERIOD_S = 0.02
SCENARIO_SCHEMA_VERSION = 1

_COMMAND_SEGMENTS = (
    {"duration_s": NEUTRAL_DURATION_S, "value": 0.0, "label": "neutral_before_positive"},
    {"duration_s": DRIVE_DURATION_S, "value": COMMAND_SPEED_RAD_S, "label": "drive_positive"},
    {"duration_s": COAST_DURATION_S, "value": 0.0, "label": "coast_positive"},
    {"duration_s": NEUTRAL_DURATION_S, "value": 0.0, "label": "neutral_between"},
    {"duration_s": DRIVE_DURATION_S, "value": -COMMAND_SPEED_RAD_S, "label": "drive_negative"},
    {"duration_s": COAST_DURATION_S, "value": 0.0, "label": "coast_negative"},
    {"duration_s": NEUTRAL_DURATION_S, "value": 0.0, "label": "neutral_finish"},
)


def _sha256_json(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_ALL_MAIN_DISABLED = (False, False, False, False, False, False)
_ALL_TARGETS_ZERO = (0.0,) * 12


class ProbeAbort(RuntimeError):
    """The probe stopped because a fail-closed condition was observed."""


@dataclass(frozen=True)
class ProbeCommand:
    enable: bool
    main_drive_enable: tuple[bool, bool, bool, bool, bool, bool]
    abad_output_enable: bool
    target_velocity_rad_s: tuple[float, ...]


@dataclass(frozen=True)
class ProbeTick:
    tick_index: int
    elapsed_s: float
    repetition: int
    segment_index: int
    segment_tick: int
    segment: str
    command: ProbeCommand


@dataclass(frozen=True)
class SafetySnapshot:
    command_subscriber_count: int
    heartbeat_value: bool | None
    heartbeat_received_at: float | None
    joint_state_received_at: float | None
    estop_value: bool | None


def _validate_main_index(main_index: object) -> int:
    if isinstance(main_index, bool) or not isinstance(main_index, int) or not 0 <= main_index < 6:
        raise ValueError("main index must be 0..5")
    return main_index


def scenario_id(main_index: object) -> str:
    selected = _validate_main_index(main_index)
    return f"suspended-main-{selected}-step-coast"


def scenario_spec(main_index: object) -> dict[str, Any]:
    selected = _validate_main_index(main_index)
    return {
        "schema_version": SCENARIO_SCHEMA_VERSION,
        "scenario_id": scenario_id(selected),
        "name": f"Suspended main {selected} step and coast",
        "description": (
            f"Three fixed low-energy repetitions on suspended main_{selected}, "
            "with globally disabled coast and neutral segments."
        ),
        "subsystem": "main_drive",
        "experiment_kind": "step_coast",
        "joint": f"main_{selected}",
        "command_segments": [dict(segment) for segment in _COMMAND_SEGMENTS],
        "repeats": REPEATS,
        "required_channels": ["command", "position"],
        "time_bases": {"command": "command_time_s", "position": "position_time_s"},
        "split": "holdout" if selected == 5 else "calibration",
        "scene_mode": "fixed_base",
        "safety_class": "low_energy",
    }


def scenario_sha256(main_index: object) -> str:
    return _sha256_json(scenario_spec(main_index))


def _command(main_index: int, *, enabled: bool, velocity: float) -> ProbeCommand:
    if not enabled:
        return ProbeCommand(False, _ALL_MAIN_DISABLED, False, _ALL_TARGETS_ZERO)
    mask = tuple(index == main_index for index in range(6))
    targets = [0.0] * 12
    targets[main_index] = float(velocity)
    return ProbeCommand(True, mask, False, tuple(targets))


def terminal_command() -> ProbeCommand:
    return ProbeCommand(False, _ALL_MAIN_DISABLED, False, _ALL_TARGETS_ZERO)


def build_schedule(main_index: object) -> tuple[ProbeTick, ...]:
    """Build the only operator-reviewed schedule; no caller parameters tune it."""

    selected = _validate_main_index(main_index)
    spec = scenario_spec(selected)
    result: list[ProbeTick] = []
    tick_index = 0
    for repetition in range(1, REPEATS + 1):
        for segment_index, segment in enumerate(spec["command_segments"]):
            duration_s = float(segment["duration_s"])
            exact_ticks = duration_s * RATE_HZ
            segment_ticks = int(round(exact_ticks))
            if not math.isclose(exact_ticks, segment_ticks, rel_tol=0.0, abs_tol=1.0e-12):
                raise RuntimeError("fixed probe segment is not aligned to 60 Hz")
            label = str(segment["label"])
            enabled = label in {"drive_positive", "drive_negative"}
            velocity = float(segment["value"])
            if enabled and abs(velocity) > COMMAND_SPEED_RAD_S:
                raise RuntimeError("fixed probe scenario exceeds its command speed cap")
            command = _command(selected, enabled=enabled, velocity=velocity)
            for segment_tick in range(segment_ticks):
                result.append(
                    ProbeTick(
                        tick_index=tick_index,
                        elapsed_s=tick_index / RATE_HZ,
                        repetition=repetition,
                        segment_index=segment_index,
                        segment_tick=segment_tick,
                        segment=label,
                        command=command,
                    )
                )
                tick_index += 1
    return tuple(result)


def build_preview(main_index: object) -> dict[str, Any]:
    selected = _validate_main_index(main_index)
    schedule = build_schedule(selected)
    return {
        "schema_version": 1,
        "actuation": False,
        "scenario_id": scenario_id(selected),
        "scenario_schema_version": SCENARIO_SCHEMA_VERSION,
        "scenario_sha256": scenario_sha256(selected),
        "main_index": selected,
        "rate_hz": RATE_HZ,
        "repeats": REPEATS,
        "ticks": len(schedule),
        "duration_s": len(schedule) / RATE_HZ,
        "command_speed_cap_rad_s": COMMAND_SPEED_RAD_S,
        "terminal_disable_packets": TERMINAL_DISABLE_PACKETS,
        "segments": [dict(segment) for segment in scenario_spec(selected)["command_segments"]],
        "safety": {
            "command_subscriber_required": True,
            "heartbeat_true_max_age_s": INPUT_FRESHNESS_TIMEOUT_S,
            "joint_states_max_age_s": INPUT_FRESHNESS_TIMEOUT_S,
            "explicit_estop_false_required": True,
        },
    }


def _is_recent(received_at: float | None, now: float) -> bool:
    if received_at is None or not math.isfinite(float(received_at)):
        return False
    age = now - float(received_at)
    return 0.0 <= age <= INPUT_FRESHNESS_TIMEOUT_S


def safety_failure(snapshot: SafetySnapshot, *, now: float) -> str | None:
    if snapshot.estop_value is not False:
        return "E-stop must be explicitly false"
    if snapshot.command_subscriber_count < 1:
        return "motor command subscriber is not visible"
    if snapshot.heartbeat_value is not True or not _is_recent(snapshot.heartbeat_received_at, now):
        return "low-level heartbeat is false, missing, stale, or has invalid callback time"
    if not _is_recent(snapshot.joint_state_received_at, now):
        return "joint_states callback is missing, stale, or has invalid callback time"
    return None


class ProbeRunner:
    """Execute the fixed schedule using injected transport and monotonic timing."""

    def __init__(
        self,
        *,
        publish_command: Callable[[ProbeCommand], None],
        publish_event: Callable[[dict[str, Any]], None],
        safety_snapshot: Callable[[], SafetySnapshot],
        monotonic: Callable[[], float],
        wait_until: Callable[[float], None],
        poll: Callable[[], None],
        terminal_pause: Callable[[], None],
    ) -> None:
        self._publish_command = publish_command
        self._publish_event = publish_event
        self._safety_snapshot = safety_snapshot
        self._monotonic = monotonic
        self._wait_until = wait_until
        self._poll = poll
        self._terminal_pause = terminal_pause
        self._abort_reason: str | None = None
        self._abort_event_sent = False
        self._start_time: float | None = None
        self._main_index: int | None = None

    def _event(self, event: str, **values: Any) -> dict[str, Any]:
        selected = self._main_index
        payload = {
            "schema_version": 1,
            "event": event,
            "scenario_id": scenario_id(selected) if selected is not None else None,
            "scenario_schema_version": SCENARIO_SCHEMA_VERSION,
            "scenario_sha256": scenario_sha256(selected) if selected is not None else None,
            "main_index": selected,
        }
        payload.update(values)
        return payload

    def _emit_abort(self, reason: str) -> None:
        if self._abort_event_sent:
            return
        self._abort_event_sent = True
        try:
            self._publish_event(self._event("abort", reason=reason))
        except BaseException:
            pass

    def _terminal_burst(self) -> None:
        disabled = terminal_command()
        for packet in range(TERMINAL_DISABLE_PACKETS):
            try:
                self._publish_command(disabled)
            except BaseException:
                pass
            if packet + 1 < TERMINAL_DISABLE_PACKETS:
                try:
                    self._terminal_pause()
                except BaseException:
                    pass

    def request_abort(self, reason: str, *, immediate: bool) -> None:
        if self._abort_reason is None:
            self._abort_reason = str(reason)
            self._emit_abort(self._abort_reason)
        if immediate:
            self._terminal_burst()

    def _raise_if_aborted(self) -> None:
        if self._abort_reason is not None:
            raise ProbeAbort(self._abort_reason)

    def bind(self, main_index: object) -> None:
        selected = _validate_main_index(main_index)
        if self._main_index is not None and self._main_index != selected:
            raise ValueError("probe runner is already bound to another main index")
        self._main_index = selected

    def run(self, main_index: object) -> None:
        self.bind(main_index)
        selected = self._main_index
        assert selected is not None
        schedule = build_schedule(selected)
        self._start_time = self._monotonic()
        last_repetition: int | None = None
        last_segment: tuple[int, int] | None = None
        try:
            self._publish_event(
                self._event(
                    "scenario",
                    rate_hz=RATE_HZ,
                    repeats=REPEATS,
                    ticks=len(schedule),
                    duration_s=len(schedule) / RATE_HZ,
                )
            )
            for tick in schedule:
                self._wait_until(self._start_time + tick.elapsed_s)
                self._poll()
                self._raise_if_aborted()
                if tick.repetition != last_repetition:
                    self._publish_event(self._event("repetition", repetition=tick.repetition))
                    last_repetition = tick.repetition
                segment_key = (tick.repetition, tick.segment_index)
                if segment_key != last_segment:
                    self._publish_event(
                        self._event(
                            "segment",
                            repetition=tick.repetition,
                            segment_index=tick.segment_index,
                            segment=tick.segment,
                            tick_index=tick.tick_index,
                        )
                    )
                    last_segment = segment_key
                failure = safety_failure(self._safety_snapshot(), now=self._monotonic())
                if failure is not None:
                    self.request_abort(failure, immediate=True)
                    raise ProbeAbort(failure)
                self._publish_command(tick.command)
            self._wait_until(self._start_time + len(schedule) / RATE_HZ)
            self._poll()
            self._raise_if_aborted()
            self._publish_event(self._event("complete", ticks=len(schedule)))
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                reason = "SIGINT"
            elif isinstance(exc, ProbeAbort):
                reason = str(exc)
            else:
                reason = f"exception:{type(exc).__name__}: {exc}"
            self.request_abort(reason, immediate=False)
            raise
        finally:
            self._terminal_burst()
