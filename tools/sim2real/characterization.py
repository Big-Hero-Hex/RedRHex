from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .contracts import ContractError, ScenarioSpecV1


PHYSICS_DT = 1.0 / 120.0
SUPPORTED_MODES = ("fixed-base", "free-root", "contact")
DEFAULT_AUDIT_STEPS = 240
EXPECTED_FOOT_BODY_NAMES = (
    "left_feet_1",
    "left_feet_2",
    "left_feet_3",
    "right_feet_1",
    "right_feet_2",
    "right_feet_3",
)

_WORLD_FRAME_CHANNELS = frozenset(
    {
        "root_position",
        "root_quaternion",
        "root_linear_velocity",
        "root_angular_velocity",
        "contact_force_w",
        "contact_force_n",
        "body_contact_force_w",
        "body_contact_force_n",
    }
)


@dataclass(frozen=True)
class RunRequest:
    mode: str
    steps: int
    physics_dt: float
    require_contact: bool


@dataclass(frozen=True)
class ScheduledCommand:
    value: float
    label: str
    actuator_enabled: bool
    repeat_index: int


@dataclass(frozen=True)
class ReplaySchedule:
    schedule: tuple[ScheduledCommand, ...]
    trace_sha256: str


def characterization_channel_metadata(
    scenario: ScenarioSpecV1,
    channels: set[str] | tuple[str, ...] | list[str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Return explicit physical units and frames for simulator trace channels."""

    command_unit = "rad" if scenario.experiment_kind == "abad_static" else "rad/s"
    known_units = {
        "requested_command": command_unit,
        "applied_command": command_unit,
        "joint_position": "rad",
        "joint_velocity": "rad/s",
        "joint_effort_estimate": "N*m",
        "root_position": "m",
        "root_quaternion": "1",
        "root_linear_velocity": "m/s",
        "root_angular_velocity": "rad/s",
        "contact_force_w": "N",
        "contact_force_n": "N",
        "command": command_unit,
        "position": "rad",
        "audit_value": "kg",
        "repeat_index": "1",
        "settled": "1",
        "body_contact_force_w": "N",
        "body_contact_force_n": "N",
    }
    units = {name: known_units.get(name, "unspecified") for name in channels}
    frames = {name: "joint_order" for name in channels}
    for name in _WORLD_FRAME_CHANNELS.intersection(frames):
        frames[name] = "world"
    for name in ("command", "position"):
        if name in frames:
            frames[name] = scenario.joint
    for name in ("repeat_index", "settled"):
        if name in frames:
            frames[name] = "scalar"
    if "audit_value" in frames:
        frames["audit_value"] = "scalar"
    return units, frames


def validate_run_request(
    *,
    mode: str,
    steps: int,
    physics_dt: float = PHYSICS_DT,
    require_contact: bool = False,
) -> RunRequest:
    if mode not in SUPPORTED_MODES:
        raise ContractError(f"unsupported characterization mode: {mode}")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ContractError("steps must be a positive integer")
    if isinstance(physics_dt, bool) or not isinstance(physics_dt, (int, float)):
        raise ContractError("physics_dt must be numeric")
    physics_dt = float(physics_dt)
    if not math.isfinite(physics_dt) or physics_dt <= 0.0:
        raise ContractError("physics_dt must be positive and finite")
    if require_contact and mode == "fixed-base":
        raise ContractError("contact validation requires free-root or contact mode")
    return RunRequest(mode, steps, physics_dt, bool(require_contact))


def simulation_times(steps: int, physics_dt: float = PHYSICS_DT) -> np.ndarray:
    request = validate_run_request(mode="free-root", steps=steps, physics_dt=physics_dt)
    return np.arange(1, request.steps + 1, dtype=np.float64) * request.physics_dt


def scenario_step_count(
    scenario: ScenarioSpecV1,
    physics_dt: float = PHYSICS_DT,
) -> int:
    """Return the complete declarative scenario length in physics frames."""

    validate_run_request(mode="free-root", steps=1, physics_dt=physics_dt)
    total_duration = sum(float(segment["duration_s"]) for segment in scenario.command_segments)
    exact_steps = total_duration * int(scenario.repeats) / float(physics_dt)
    rounded_steps = int(round(exact_steps))
    if rounded_steps < 1 or not math.isclose(exact_steps, rounded_steps, rel_tol=0.0, abs_tol=1.0e-9):
        raise ContractError(
            f"scenario {scenario.scenario_id} duration does not align to the {1.0 / physics_dt:g} Hz physics clock"
        )
    return rounded_steps


def resolve_scenario_steps(
    scenario: ScenarioSpecV1,
    requested_steps: int | None,
    physics_dt: float = PHYSICS_DT,
) -> int:
    """Resolve an optional CLI length without permitting partial command experiments."""

    if requested_steps is None:
        if scenario.experiment_kind == "audit":
            return DEFAULT_AUDIT_STEPS
        return scenario_step_count(scenario, physics_dt)
    validate_run_request(mode="free-root", steps=requested_steps, physics_dt=physics_dt)
    if scenario.experiment_kind != "audit":
        expected = scenario_step_count(scenario, physics_dt)
        if requested_steps != expected:
            raise ContractError(
                f"scenario {scenario.scenario_id} requires exactly {expected} physics steps; "
                f"received {requested_steps}"
            )
    return requested_steps


def validate_scenario_mode(scenario: ScenarioSpecV1, mode: str) -> None:
    """Reject scene modes that invalidate the declared physical experiment."""

    if mode not in SUPPORTED_MODES:
        raise ContractError(f"unsupported characterization mode: {mode}")
    if scenario.scene_mode == "manual":
        raise ContractError(f"manual scenario cannot be run in Isaac: {scenario.scenario_id}")
    if scenario.scene_mode == "audit":
        return
    allowed = {
        "fixed_base": {"fixed-base"},
        "free_root": {"free-root", "contact"},
    }.get(scenario.scene_mode, set())
    if mode not in allowed:
        choices = ", ".join(sorted(allowed)) or "manual acquisition"
        raise ContractError(
            f"scenario {scenario.scenario_id} requires scene mode {choices}; received {mode}"
        )


def validate_simulated_experiment(scenario: ScenarioSpecV1) -> None:
    """Refuse scenarios whose required physical stimulus is not implemented."""

    if scenario.experiment_kind == "friction":
        raise ContractError(
            "friction requires measured pull-force data; run-sim does not implement a controlled pull. "
            "Use an audit contact/settle probe to validate sensors and apply measured friction directly."
        )


def scenario_schedule(
    scenario: ScenarioSpecV1,
    steps: int,
    physics_dt: float = PHYSICS_DT,
) -> tuple[ScheduledCommand, ...]:
    request = validate_run_request(mode="free-root", steps=steps, physics_dt=physics_dt)
    resolve_scenario_steps(scenario, request.steps, request.physics_dt)
    segments = scenario.command_segments
    cycle_duration = sum(float(segment["duration_s"]) for segment in segments)
    total_duration = cycle_duration * scenario.repeats
    cumulative = np.cumsum([float(segment["duration_s"]) for segment in segments])
    result: list[ScheduledCommand] = []
    for index in range(request.steps):
        sample_time = (index + 0.5) * request.physics_dt
        bounded_time = min(sample_time, max(total_duration - np.finfo(float).eps, 0.0))
        repeat_index = min(int(bounded_time / cycle_duration), scenario.repeats - 1)
        cycle_time = bounded_time - repeat_index * cycle_duration
        segment_index = int(np.searchsorted(cumulative, cycle_time, side="right"))
        segment_index = min(segment_index, len(segments) - 1)
        segment = segments[segment_index]
        label = str(segment.get("label", f"segment_{segment_index}"))
        actuator_enabled = (
            label in {"drive_positive", "drive_negative"}
            if scenario.experiment_kind == "step_coast"
            else "coast" not in label.lower()
        )
        result.append(
            ScheduledCommand(
                value=float(segment["value"]),
                label=label,
                actuator_enabled=actuator_enabled,
                repeat_index=repeat_index,
            )
        )
    return tuple(result)


def measurement_annotations(
    schedule: tuple[ScheduledCommand, ...],
    *,
    settled_fraction: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Annotate repeat identity and the settled tail of each command segment."""

    fraction = float(settled_fraction)
    if not schedule:
        raise ContractError("measurement annotation schedule must not be empty")
    if not math.isfinite(fraction) or fraction <= 0.0 or fraction > 1.0:
        raise ContractError("settled_fraction must lie in (0, 1]")
    repeats = np.asarray([item.repeat_index for item in schedule], dtype=np.float64)
    settled = np.zeros(len(schedule), dtype=np.float64)
    start = 0
    while start < len(schedule):
        current = schedule[start]
        end = start + 1
        while end < len(schedule):
            item = schedule[end]
            if (
                item.repeat_index != current.repeat_index
                or item.label != current.label
                or item.value != current.value
            ):
                break
            end += 1
        tail = max(1, int(math.ceil((end - start) * fraction)))
        settled[end - tail : end] = 1.0
        start = end
    return repeats, settled


def apply_schedule_delay(
    schedule: tuple[ScheduledCommand, ...],
    *,
    delay_steps: int,
) -> tuple[ScheduledCommand, ...]:
    """Delay every command transition by an exact number of physics frames."""

    if isinstance(delay_steps, bool) or not isinstance(delay_steps, int) or delay_steps < 0:
        raise ContractError("delay_steps must be a non-negative integer")
    if not schedule:
        raise ContractError("command schedule must not be empty")
    if delay_steps == 0:
        return schedule
    pending = [schedule[0]] * delay_steps
    applied: list[ScheduledCommand] = []
    for requested in schedule:
        applied.append(pending.pop(0))
        pending.append(requested)
    return tuple(applied)


def load_replay_schedule(
    value: str | Path,
    scenario: ScenarioSpecV1,
    *,
    steps: int,
    physics_dt: float = PHYSICS_DT,
) -> ReplaySchedule:
    """Load and zero-order-hold one verified real command trace onto physics steps."""

    from .traces import load_trace

    resolve_scenario_steps(scenario, steps, physics_dt)
    if "command" not in scenario.time_bases:
        raise ContractError("replay scenario does not declare a command channel")
    command_unit = "rad" if scenario.experiment_kind == "abad_static" else "rad/s"
    loaded = load_trace(
        value,
        scenario=scenario,
        expected_units={"command": command_unit},
        expected_frames={"command": scenario.joint},
    )
    if loaded.manifest.source != "real":
        raise ContractError("replay trace must have source='real'")

    time_name = loaded.manifest.time_bases.get("command")
    if time_name is None:
        raise ContractError("replay trace has no command time base")
    command_time = np.asarray(loaded.arrays[time_name], dtype=np.float64)
    command = np.asarray(loaded.arrays["command"], dtype=np.float64)
    if command.ndim != 1:
        raise ContractError("isolated replay command must be one-dimensional")

    duration_s = float(steps) * float(physics_dt)
    constants = loaded.manifest.metadata.get("calibration_constants", {})
    evidence = constants.get("probe_event_evidence") if isinstance(constants, dict) else None
    time_origin_s = 0.0
    coverage_end_s = float(command_time[-1])
    if evidence is not None:
        if not isinstance(evidence, dict):
            raise ContractError("probe replay coverage evidence must be an object")
        try:
            time_origin_s = float(evidence["scenario_receive_time_s"])
            coverage_end_s = float(evidence["complete_receive_time_s"])
            complete_ticks = int(evidence["complete_ticks"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError("probe replay coverage evidence is incomplete") from exc
        expected_ticks = int(round(duration_s * 60.0))
        if complete_ticks != expected_ticks:
            raise ContractError("probe replay coverage tick count does not match scenario")
        if not np.isfinite([time_origin_s, coverage_end_s]).all():
            raise ContractError("probe replay coverage times must be finite")

    relative_time = command_time - time_origin_s
    relative_coverage_end = coverage_end_s - time_origin_s
    tolerance = max(1.0e-9, physics_dt * 1.0e-6)
    if relative_time[0] > tolerance:
        raise ContractError("replay command trace does not cover scenario start")
    if relative_coverage_end + tolerance < duration_s:
        raise ContractError("replay command trace does not cover complete scenario duration")

    sample_time = (np.arange(steps, dtype=np.float64) + 0.5) * physics_dt
    sample_indices = np.searchsorted(relative_time, sample_time, side="right") - 1
    if np.any(sample_indices < 0):
        raise ContractError("replay command trace has no sample at scenario start")
    replay_values = command[sample_indices]
    nominal = scenario_schedule(scenario, steps, physics_dt)
    schedule: list[ScheduledCommand] = []
    for item, value in zip(nominal, replay_values, strict=True):
        actuator_enabled = item.actuator_enabled
        if scenario.experiment_kind == "step_coast":
            actuator_enabled = not math.isclose(float(value), 0.0, abs_tol=1.0e-12)
        schedule.append(
            ScheduledCommand(
                value=float(value),
                label=item.label,
                actuator_enabled=actuator_enabled,
                repeat_index=item.repeat_index,
            )
        )
    return ReplaySchedule(
        schedule=tuple(schedule),
        trace_sha256=loaded.manifest.provenance["trace_sha256"],
    )


def validate_contact_probe(
    resolved_body_names: list[str] | tuple[str, ...],
    observed_forces_n: np.ndarray,
    *,
    threshold_n: float,
) -> dict[str, object]:
    names = [str(name) for name in resolved_body_names if str(name)]
    if not names:
        raise ContractError("contact sensor resolved no robot bodies")
    if not math.isfinite(float(threshold_n)) or threshold_n <= 0.0:
        raise ContractError("contact threshold must be positive and finite")
    forces = np.asarray(observed_forces_n, dtype=np.float64)
    finite = forces[np.isfinite(forces)]
    maximum = float(np.max(finite)) if finite.size else 0.0
    if maximum <= threshold_n:
        raise ContractError(
            f"contact probe observed no measurable impulse above {threshold_n:g} N "
            f"(maximum {maximum:g} N)"
        )
    return {"resolved_body_names": names, "max_force_n": maximum, "threshold_n": float(threshold_n)}


def validate_foot_contact_probe(
    resolved_body_names: list[str] | tuple[str, ...],
    observed_forces_n: np.ndarray,
    *,
    threshold_n: float,
) -> dict[str, object]:
    """Require a sensor scoped to exactly the six terminal feet."""

    names = tuple(str(name) for name in resolved_body_names)
    if len(names) != len(EXPECTED_FOOT_BODY_NAMES) or set(names) != set(EXPECTED_FOOT_BODY_NAMES):
        raise ContractError(
            "foot contact sensor must resolve exactly the six terminal feet; "
            f"resolved {list(names)}"
        )
    forces = np.asarray(observed_forces_n, dtype=np.float64)
    if forces.ndim < 1 or forces.shape[-1] != len(names):
        raise ContractError("foot contact force layout does not match resolved foot bodies")
    return validate_contact_probe(list(names), forces, threshold_n=threshold_n)


def requires_contact_probe(
    scenario: ScenarioSpecV1,
    *,
    mode: str,
    explicit: bool,
) -> bool:
    if mode not in SUPPORTED_MODES:
        raise ContractError(f"unsupported characterization mode: {mode}")
    return bool(explicit or mode == "contact" or scenario.experiment_kind == "friction")
