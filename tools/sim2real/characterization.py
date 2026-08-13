from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import ContractError, ScenarioSpecV1


PHYSICS_DT = 1.0 / 120.0
REPLAY_STATIONARY_WINDOW_S = 0.4
REPLAY_JOINT_STATIONARITY_LIMIT_RAD_S = 0.05
REPLAY_IMU_STATIONARITY_LIMIT_RAD_S = 0.1
REPLAY_FIXTURE_ORIENTATION_TOLERANCE_RAD = math.radians(5.0)
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
    initial_state: "ReplayInitialState"
    initial_state_sha256: str


@dataclass(frozen=True)
class ReplayInitialState:
    joint_order: tuple[str, ...]
    position_source_channel: str
    position_rad: tuple[float, ...]
    velocity_rad_s: tuple[float, ...]
    velocity_source: str
    velocity_window_start_s: float
    velocity_window_end_s: float
    velocity_stationarity_limit_rad_s: float
    fixture_mode: str
    fixture_frame: str
    root_pose_source: str
    fixture_id: str
    fixture_sha256: str
    root_orientation_wxyz: tuple[float, ...]
    imu_orientation_source_channel: str
    measured_imu_orientation_xyzw: tuple[float, ...]
    expected_imu_orientation_xyzw: tuple[float, ...]
    imu_orientation_error_rad: float
    imu_orientation_tolerance_rad: float
    imu_angular_velocity_source_channel: str
    max_imu_angular_speed_rad_s: float
    imu_stationarity_limit_rad_s: float
    sample_time_s: float
    scenario_time_s: float
    sample_offset_s: float
    schema_version: int = 2

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "joint_order": list(self.joint_order),
            "position_source_channel": self.position_source_channel,
            "position_rad": list(self.position_rad),
            "velocity_rad_s": list(self.velocity_rad_s),
            "velocity_source": self.velocity_source,
            "velocity_window_start_s": self.velocity_window_start_s,
            "velocity_window_end_s": self.velocity_window_end_s,
            "velocity_stationarity_limit_rad_s": self.velocity_stationarity_limit_rad_s,
            "fixture_mode": self.fixture_mode,
            "fixture_frame": self.fixture_frame,
            "root_pose_source": self.root_pose_source,
            "fixture_id": self.fixture_id,
            "fixture_sha256": self.fixture_sha256,
            "root_orientation_wxyz": list(self.root_orientation_wxyz),
            "imu_orientation_source_channel": self.imu_orientation_source_channel,
            "measured_imu_orientation_xyzw": list(self.measured_imu_orientation_xyzw),
            "expected_imu_orientation_xyzw": list(self.expected_imu_orientation_xyzw),
            "imu_orientation_error_rad": self.imu_orientation_error_rad,
            "imu_orientation_tolerance_rad": self.imu_orientation_tolerance_rad,
            "imu_angular_velocity_source_channel": self.imu_angular_velocity_source_channel,
            "max_imu_angular_speed_rad_s": self.max_imu_angular_speed_rad_s,
            "imu_stationarity_limit_rad_s": self.imu_stationarity_limit_rad_s,
            "sample_time_s": self.sample_time_s,
            "scenario_time_s": self.scenario_time_s,
            "sample_offset_s": self.sample_offset_s,
        }


def characterization_channel_metadata(
    scenario: ScenarioSpecV1,
    channels: set[str] | tuple[str, ...] | list[str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Return explicit physical units and frames for simulator trace channels."""

    command_unit = (
        "rad"
        if scenario.experiment_kind in {"abad_static", "spring_release"}
        else "rad/s"
    )
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
        "spring_deflection": "rad",
        "spring_model_torque": "N*m",
        "spring_applied_torque_estimate": "N*m",
        "spring_potential_energy": "J",
        "spring_mechanical_power": "W",
        "spring_passivity_residual": "J",
        "spring_release_start": "1",
        "spring_fixture_position_error": "rad",
        "spring_fixture_velocity": "rad/s",
        "spring_unwrap_ambiguous": "1",
        "spring_pre_step_time_s": "s",
        "sim_time_s": "s",
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
    for name in (
        "spring_deflection",
        "spring_model_torque",
        "spring_applied_torque_estimate",
        "spring_potential_energy",
        "spring_mechanical_power",
    ):
        if name in frames:
            frames[name] = "damper_order"
    for name in (
        "spring_passivity_residual",
        "spring_release_start",
        "spring_fixture_position_error",
        "spring_fixture_velocity",
        "spring_unwrap_ambiguous",
    ):
        if name in frames:
            frames[name] = "scalar"
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


def _replay_initial_state(
    loaded: Any,
    scenario: ScenarioSpecV1,
    *,
    scenario_time_s: float,
) -> tuple[ReplayInitialState, str]:
    from .traces import sha256_json

    constants = loaded.manifest.metadata.get("calibration_constants", {})
    declaration = (
        constants.get("replay_initial_state")
        if isinstance(constants, Mapping)
        else None
    )
    declared_hash = (
        constants.get("replay_initial_state_sha256")
        if isinstance(constants, Mapping)
        else None
    )
    if not isinstance(declaration, Mapping) or not isinstance(declared_hash, str):
        raise ContractError("replay trace is missing its initial state declaration and hash")
    expected_fields = {
        "schema_version",
        "joint_order",
        "position_source_channel",
        "position_rad",
        "velocity_rad_s",
        "velocity_source",
        "velocity_window_start_s",
        "velocity_window_end_s",
        "velocity_stationarity_limit_rad_s",
        "fixture_mode",
        "fixture_frame",
        "root_pose_source",
        "fixture_id",
        "fixture_sha256",
        "root_orientation_wxyz",
        "imu_orientation_source_channel",
        "measured_imu_orientation_xyzw",
        "expected_imu_orientation_xyzw",
        "imu_orientation_error_rad",
        "imu_orientation_tolerance_rad",
        "imu_angular_velocity_source_channel",
        "max_imu_angular_speed_rad_s",
        "imu_stationarity_limit_rad_s",
        "sample_time_s",
        "scenario_time_s",
        "sample_offset_s",
    }
    if set(declaration) != expected_fields:
        raise ContractError("replay initial state declaration has missing or unknown fields")
    payload = dict(declaration)
    if sha256_json(payload) != declared_hash:
        raise ContractError("replay initial state hash does not match its declaration")
    if payload["schema_version"] != 2 or isinstance(payload["schema_version"], bool):
        raise ContractError("replay initial state schema version must be 2")
    expected_joint_order = tuple(f"main_{index}" for index in range(6))
    raw_joint_order = payload["joint_order"]
    if not isinstance(raw_joint_order, list) or tuple(raw_joint_order) != expected_joint_order:
        raise ContractError("replay initial state must bind all six canonical main joints")
    if scenario.joint not in expected_joint_order:
        raise ContractError("replay initial state scenario does not select a main joint")
    if payload["position_source_channel"] != "main_joint_position_canonical":
        raise ContractError("replay initial state must use canonical main-joint positions")
    if payload["fixture_mode"] != scenario.scene_mode or scenario.scene_mode != "fixed_base":
        raise ContractError("replay initial state fixture mode does not match the scenario")
    if payload["fixture_frame"] != "world":
        raise ContractError("replay initial state fixture frame must be world")
    if payload["root_pose_source"] != "reviewed_fixture":
        raise ContractError("replay initial state root pose source is unsupported")
    if payload["velocity_source"] != "stationary_window_linear_fit":
        raise ContractError("replay initial state velocity source is unsupported")
    if payload["imu_orientation_source_channel"] != "imu_orientation_xyzw":
        raise ContractError("replay initial state IMU orientation source is unsupported")
    if payload["imu_angular_velocity_source_channel"] != "imu_angular_velocity":
        raise ContractError(
            "replay initial state IMU angular-velocity source is unsupported"
        )

    def finite_number(field: str) -> float:
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractError(f"replay initial state {field} must be numeric")
        result = float(value)
        if not math.isfinite(result):
            raise ContractError(f"replay initial state {field} must be finite")
        return result

    def finite_vector(field: str) -> tuple[float, ...]:
        value = payload[field]
        if not isinstance(value, list) or len(value) != len(expected_joint_order):
            raise ContractError(
                f"replay initial state {field} must contain six numeric values"
            )
        result: list[float] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ContractError(
                    f"replay initial state {field} must contain six numeric values"
                )
            result.append(float(item))
        if not np.isfinite(result).all():
            raise ContractError(f"replay initial state {field} must be finite")
        return tuple(result)

    def finite_quaternion(field: str) -> tuple[float, ...]:
        value = payload[field]
        if not isinstance(value, list) or len(value) != 4:
            raise ContractError(
                f"replay initial state {field} must contain four numeric values"
            )
        result = np.asarray(value, dtype=np.float64)
        if not np.isfinite(result).all():
            raise ContractError(f"replay initial state {field} must be finite")
        norm = float(np.linalg.norm(result))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-6):
            raise ContractError(f"replay initial state {field} must be normalized")
        return tuple(float(item) for item in result)

    def quaternion_error_rad(first: np.ndarray, second: np.ndarray) -> float:
        return 2.0 * math.acos(
            float(np.clip(abs(float(np.dot(first, second))), 0.0, 1.0))
        )

    position_rad = finite_vector("position_rad")
    velocity_rad_s = finite_vector("velocity_rad_s")
    root_orientation_wxyz = finite_quaternion("root_orientation_wxyz")
    measured_imu_orientation_xyzw = finite_quaternion(
        "measured_imu_orientation_xyzw"
    )
    expected_imu_orientation_xyzw = finite_quaternion(
        "expected_imu_orientation_xyzw"
    )
    velocity_window_start_s = finite_number("velocity_window_start_s")
    velocity_window_end_s = finite_number("velocity_window_end_s")
    velocity_limit = finite_number("velocity_stationarity_limit_rad_s")
    imu_orientation_error = finite_number("imu_orientation_error_rad")
    imu_orientation_tolerance = finite_number("imu_orientation_tolerance_rad")
    max_imu_speed = finite_number("max_imu_angular_speed_rad_s")
    imu_speed_limit = finite_number("imu_stationarity_limit_rad_s")
    sample_time_s = finite_number("sample_time_s")
    declared_scenario_time_s = finite_number("scenario_time_s")
    sample_offset_s = finite_number("sample_offset_s")
    tolerance = 1.0e-9
    if not math.isclose(
        velocity_window_start_s, scenario_time_s, rel_tol=0.0, abs_tol=tolerance
    ) or not math.isclose(
        velocity_window_end_s,
        scenario_time_s + REPLAY_STATIONARY_WINDOW_S,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise ContractError("replay stationary window must be the reviewed initial window")
    first_segment = scenario.command_segments[0]
    if (
        not math.isclose(float(first_segment["value"]), 0.0, abs_tol=tolerance)
        or float(first_segment["duration_s"]) + tolerance < REPLAY_STATIONARY_WINDOW_S
    ):
        raise ContractError("replay scenario has no reviewed initial neutral window")
    if not math.isclose(
        velocity_limit,
        REPLAY_JOINT_STATIONARITY_LIMIT_RAD_S,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise ContractError("replay joint stationarity limit is unsupported")
    if not math.isclose(
        imu_speed_limit,
        REPLAY_IMU_STATIONARITY_LIMIT_RAD_S,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise ContractError("replay IMU stationarity limit is unsupported")
    if not math.isclose(
        imu_orientation_tolerance,
        REPLAY_FIXTURE_ORIENTATION_TOLERANCE_RAD,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise ContractError("replay fixture orientation tolerance is unsupported")
    if not math.isclose(
        declared_scenario_time_s,
        scenario_time_s,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise ContractError("replay initial state scenario time does not match command coverage")
    if not math.isclose(
        sample_offset_s,
        sample_time_s - scenario_time_s,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise ContractError("replay initial state sample offset is inconsistent")

    source_channel = str(payload["position_source_channel"])
    time_name = loaded.manifest.time_bases.get(source_channel)
    if time_name is None:
        raise ContractError("replay trace has no canonical position time base")
    position_time = np.asarray(loaded.arrays[time_name], dtype=np.float64)
    position = np.asarray(loaded.arrays[source_channel], dtype=np.float64)
    if position.ndim != 2 or position.shape[1] != len(expected_joint_order):
        raise ContractError("replay canonical position channel must have shape (samples, 6)")
    distances = np.abs(position_time - scenario_time_s)
    nearest_distance = float(np.min(distances))
    if nearest_distance > 1.0 / 60.0 + tolerance:
        raise ContractError("replay initial state has no sample at the scenario start")
    nearest = np.flatnonzero(
        np.isclose(distances, nearest_distance, rtol=0.0, atol=tolerance)
    )
    if nearest.size != 1:
        raise ContractError("replay initial state sample is ambiguous at the scenario start")
    sample_index = int(nearest[0])
    if not math.isclose(
        sample_time_s,
        float(position_time[sample_index]),
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise ContractError("replay initial state does not select the nearest position sample")
    if not np.allclose(
        np.asarray(position_rad),
        position[sample_index],
        rtol=0.0,
        atol=tolerance,
    ):
        raise ContractError("replay initial state position does not match the verified trace")

    stationary_mask = (position_time >= velocity_window_start_s - tolerance) & (
        position_time <= velocity_window_end_s + tolerance
    )
    if int(np.count_nonzero(stationary_mask)) < 3:
        raise ContractError("replay stationary window requires at least three position samples")
    fit_time = position_time[stationary_mask]
    centered_time = fit_time - float(np.mean(fit_time))
    denominator = float(np.dot(centered_time, centered_time))
    if denominator <= 0.0:
        raise ContractError("replay stationary position samples have no time span")
    fit_position = position[stationary_mask]
    fitted_velocity = np.sum(
        centered_time[:, None] * fit_position, axis=0
    ) / denominator
    if float(np.max(np.abs(fitted_velocity))) > velocity_limit + 1.0e-9:
        raise ContractError("replay initial joint state is not stationary")
    if not np.allclose(
        np.asarray(velocity_rad_s), fitted_velocity, rtol=0.0, atol=1.0e-8
    ):
        raise ContractError(
            "replay initial velocity does not match the verified stationary window"
        )

    fixture_id = payload["fixture_id"]
    fixture_sha256 = payload["fixture_sha256"]
    if not isinstance(fixture_id, str) or not fixture_id.strip():
        raise ContractError("replay fixture_id must be a non-empty string")
    if (
        not isinstance(fixture_sha256, str)
        or len(fixture_sha256) != 64
        or any(character not in "0123456789abcdef" for character in fixture_sha256)
    ):
        raise ContractError("replay fixture_sha256 must be a SHA-256 digest")
    fixture_payload = {
        "schema_version": 1,
        "fixture_id": fixture_id,
        "scene_mode": payload["fixture_mode"],
        "fixture_frame": payload["fixture_frame"],
        "root_orientation_wxyz": list(root_orientation_wxyz),
        "expected_imu_orientation_xyzw": list(expected_imu_orientation_xyzw),
    }
    if sha256_json(fixture_payload) != fixture_sha256:
        raise ContractError("replay fixture hash does not match its reviewed contents")

    imu_orientation_channel = str(payload["imu_orientation_source_channel"])
    imu_orientation_time_name = loaded.manifest.time_bases.get(
        imu_orientation_channel
    )
    if imu_orientation_time_name is None:
        raise ContractError("replay trace has no IMU orientation time base")
    imu_time = np.asarray(
        loaded.arrays[imu_orientation_time_name], dtype=np.float64
    )
    imu_orientation = np.asarray(
        loaded.arrays[imu_orientation_channel], dtype=np.float64
    )
    if imu_orientation.ndim != 2 or imu_orientation.shape[1] != 4:
        raise ContractError("replay IMU orientation channel must have shape (samples, 4)")
    imu_mask = (imu_time >= velocity_window_start_s - tolerance) & (
        imu_time <= velocity_window_end_s + tolerance
    )
    if int(np.count_nonzero(imu_mask)) < 3:
        raise ContractError("replay stationary window requires at least three IMU samples")
    quaternions = np.array(imu_orientation[imu_mask], copy=True)
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(norms <= 0.0):
        raise ContractError("replay IMU trace contains a zero quaternion")
    quaternions /= norms[:, None]
    reference = quaternions[0]
    quaternions[np.sum(quaternions * reference, axis=1) < 0.0] *= -1.0
    mean_quaternion = np.mean(quaternions, axis=0)
    mean_norm = float(np.linalg.norm(mean_quaternion))
    if mean_norm <= 0.0:
        raise ContractError("replay IMU orientation average is ambiguous")
    mean_quaternion /= mean_norm
    declared_measured = np.asarray(measured_imu_orientation_xyzw)
    if quaternion_error_rad(mean_quaternion, declared_measured) > 1.0e-7:
        raise ContractError("replay measured IMU orientation does not match the trace")
    expected_imu = np.asarray(expected_imu_orientation_xyzw)
    computed_orientation_error = quaternion_error_rad(mean_quaternion, expected_imu)
    if not math.isclose(
        imu_orientation_error,
        computed_orientation_error,
        rel_tol=0.0,
        abs_tol=1.0e-7,
    ):
        raise ContractError("replay IMU orientation error is inconsistent")
    if computed_orientation_error > imu_orientation_tolerance + 1.0e-9:
        raise ContractError("replay IMU orientation does not match the reviewed fixture")

    imu_velocity_channel = str(payload["imu_angular_velocity_source_channel"])
    imu_velocity_time_name = loaded.manifest.time_bases.get(imu_velocity_channel)
    if imu_velocity_time_name is None:
        raise ContractError("replay trace has no IMU angular-velocity time base")
    imu_velocity_time = np.asarray(
        loaded.arrays[imu_velocity_time_name], dtype=np.float64
    )
    imu_velocity = np.asarray(loaded.arrays[imu_velocity_channel], dtype=np.float64)
    if imu_velocity.ndim != 2 or imu_velocity.shape[1] != 3:
        raise ContractError(
            "replay IMU angular-velocity channel must have shape (samples, 3)"
        )
    imu_velocity_mask = (
        imu_velocity_time >= velocity_window_start_s - tolerance
    ) & (imu_velocity_time <= velocity_window_end_s + tolerance)
    if int(np.count_nonzero(imu_velocity_mask)) < 3:
        raise ContractError(
            "replay stationary window requires at least three IMU angular-velocity samples"
        )
    computed_max_imu_speed = float(
        np.max(np.linalg.norm(imu_velocity[imu_velocity_mask], axis=1))
    )
    if not math.isclose(
        max_imu_speed, computed_max_imu_speed, rel_tol=0.0, abs_tol=1.0e-8
    ):
        raise ContractError("replay maximum IMU angular speed is inconsistent")
    if computed_max_imu_speed > imu_speed_limit + 1.0e-9:
        raise ContractError("replay fixture is not stationary according to the IMU")
    return (
        ReplayInitialState(
            joint_order=expected_joint_order,
            position_source_channel=source_channel,
            position_rad=position_rad,
            velocity_rad_s=velocity_rad_s,
            velocity_source=str(payload["velocity_source"]),
            velocity_window_start_s=velocity_window_start_s,
            velocity_window_end_s=velocity_window_end_s,
            velocity_stationarity_limit_rad_s=velocity_limit,
            fixture_mode=str(payload["fixture_mode"]),
            fixture_frame=str(payload["fixture_frame"]),
            root_pose_source=str(payload["root_pose_source"]),
            fixture_id=fixture_id,
            fixture_sha256=fixture_sha256,
            root_orientation_wxyz=root_orientation_wxyz,
            imu_orientation_source_channel=imu_orientation_channel,
            measured_imu_orientation_xyzw=measured_imu_orientation_xyzw,
            expected_imu_orientation_xyzw=expected_imu_orientation_xyzw,
            imu_orientation_error_rad=imu_orientation_error,
            imu_orientation_tolerance_rad=imu_orientation_tolerance,
            imu_angular_velocity_source_channel=imu_velocity_channel,
            max_imu_angular_speed_rad_s=max_imu_speed,
            imu_stationarity_limit_rad_s=imu_speed_limit,
            sample_time_s=sample_time_s,
            scenario_time_s=declared_scenario_time_s,
            sample_offset_s=sample_offset_s,
        ),
        declared_hash,
    )


def load_replay_schedule(
    value: str | Path,
    scenario: ScenarioSpecV1,
    *,
    steps: int,
    physics_dt: float = PHYSICS_DT,
) -> ReplaySchedule:
    """Load and zero-order-hold one verified real command trace onto physics steps."""

    from .provenance import validate_real_trace_provenance
    from .traces import load_trace

    resolve_scenario_steps(scenario, steps, physics_dt)
    if "command" not in scenario.time_bases:
        raise ContractError("replay scenario does not declare a command channel")
    command_unit = (
        "rad"
        if scenario.experiment_kind in {"abad_static", "spring_release"}
        else "rad/s"
    )
    loaded = load_trace(
        value,
        scenario=scenario,
        expected_units={"command": command_unit, "position": "rad"},
        expected_frames={"command": scenario.joint, "position": scenario.joint},
        require_managed_dataset=True,
    )
    validate_real_trace_provenance(
        loaded, scenario, require_all_main_positions=True
    )

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

    initial_state, initial_state_sha256 = _replay_initial_state(
        loaded,
        scenario,
        scenario_time_s=time_origin_s,
    )

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
        initial_state=initial_state,
        initial_state_sha256=initial_state_sha256,
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
    return bool(
        explicit
        or mode == "contact"
        or scenario.experiment_kind in {"friction", "static_settle"}
    )
