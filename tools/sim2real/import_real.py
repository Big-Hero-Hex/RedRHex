from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import (
    CalibrationProfileV1,
    ContractError,
    ScenarioSpecV1,
    TraceManifestV1,
)
from .scenarios import load_scenario
from .traces import sha256_json, write_trace


_BAG_LATENCY_CLOCK = "bag_receive_time"


def validate_latency_clock(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError("latency clock must be a non-empty name")
    return value


def resolve_latency_clock(source_path: str | Path, value: str | None) -> str:
    source = Path(source_path)
    if source.is_file() and source.suffix == ".npz":
        if value is None:
            raise ContractError("numeric NPZ import requires an explicit latency clock")
        return validate_latency_clock(value)
    if value is None:
        return _BAG_LATENCY_CLOCK
    if value != _BAG_LATENCY_CLOCK:
        raise ContractError(
            'rosbag latency clock must be exactly "bag_receive_time"; '
            "bag extraction uses the SequentialReader receive timestamp"
        )
    return _BAG_LATENCY_CLOCK


def _load_numeric_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError) as exc:
        raise ContractError(f"cannot import numeric NPZ {path}: {exc}") from exc


_LEG_ORDER = ("l1", "l2", "l3", "r1", "r2", "r3")
_JOINT_TO_LEG = {
    "main_0": "r1",
    "main_1": "r2",
    "main_2": "r3",
    "main_3": "l1",
    "main_4": "l2",
    "main_5": "l3",
}
_ENCODER_SIGN = {"l1": -1.0, "l2": -1.0, "l3": -1.0, "r1": 1.0, "r2": 1.0, "r3": 1.0}
_POSITIVE_DIRECTION = {"l1": True, "l2": True, "l3": True, "r1": False, "r2": False, "r3": False}
_COUNTS_PER_REV = 54984.83
_MAX_PWM = 500.0
_MAIN_DRIVE_EXPERIMENTS = frozenset({"step", "coast", "step_coast"})
_PROBE_EVENT_TOPIC = "/redrhex/sim2real_probe/events"
_PROBE_RATE_HZ = 60.0
_PROBE_RECEIVE_JITTER_BOUND_S = 1.0 / _PROBE_RATE_HZ


def _rosbag_dependencies():
    try:
        rosbag2_py = importlib.import_module("rosbag2_py")
        deserialize_message = importlib.import_module(
            "rclpy.serialization"
        ).deserialize_message
        get_message = importlib.import_module(
            "rosidl_runtime_py.utilities"
        ).get_message
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "rosbag2_py, rclpy, and rosidl_runtime_py are required to import a rosbag2 directory; "
            "install the matching ROS 2 Python environment"
        ) from exc
    return rosbag2_py, deserialize_message, get_message


def _field(message: Any, name: str, topic: str) -> Any:
    if not hasattr(message, name):
        raise ContractError(f"{topic} message has unknown schema: missing field {name}")
    return getattr(message, name)


def _leg(message: Any, name: str, fields: tuple[str, ...], topic: str) -> Any:
    value = _field(message, name, topic)
    for field_name in fields:
        _field(value, field_name, f"{topic}.{name}")
    return value


def _vector(message: Any, name: str, topic: str) -> list[float]:
    value = _field(message, name, topic)
    return [
        float(_field(value, axis, f"{topic}.{name}"))
        for axis in ("x", "y", "z")
    ]


def _probe_event(message: Any) -> dict[str, Any]:
    raw = _field(message, "data", _PROBE_EVENT_TOPIC)
    if not isinstance(raw, str):
        raise ContractError("probe event data must be a JSON string")

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            raw,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid probe event JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError("probe event JSON must be an object")
    return payload


def _event_int(payload: Mapping[str, Any], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"probe event {field} must be an integer")
    return value


def _event_number(payload: Mapping[str, Any], field: str) -> float:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"probe event {field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"probe event {field} must be finite")
    return result


def _validate_probe_events(
    samples: list[tuple[float, dict[str, Any]]],
    scenario: ScenarioSpecV1,
) -> tuple[dict[str, Any], list[float], list[float]]:
    if not samples:
        raise ContractError("rosbag contains no probe events")
    if any(event.get("event") == "abort" for _, event in samples):
        raise ContractError("probe event stream contains an abort")

    try:
        main_index = int(scenario.joint.removeprefix("main_"))
    except ValueError as exc:
        raise ContractError(f"probe scenario has invalid main joint {scenario.joint}") from exc
    digest = sha256_json(scenario.to_dict())
    for _, event in samples:
        if _event_int(event, "schema_version") != 1:
            raise ContractError("probe event schema version mismatch")
        if event.get("scenario_id") != scenario.scenario_id:
            raise ContractError("probe event scenario id mismatch")
        if _event_int(event, "scenario_schema_version") != scenario.schema_version:
            raise ContractError("probe event scenario schema version mismatch")
        if event.get("scenario_sha256") != digest:
            raise ContractError("probe event scenario hash mismatch")
        if _event_int(event, "main_index") != main_index:
            raise ContractError("probe event main index mismatch")
        if event.get("abad_output_enable") is not False:
            raise ContractError("probe event does not prove ABAD output remained disabled")

    exact_ticks_per_cycle = sum(
        float(segment["duration_s"]) * _PROBE_RATE_HZ
        for segment in scenario.command_segments
    )
    ticks_per_cycle = int(round(exact_ticks_per_cycle))
    if not math.isclose(exact_ticks_per_cycle, ticks_per_cycle, abs_tol=1.0e-12):
        raise ContractError("probe scenario segments are not aligned to 60 Hz")
    expected_ticks = ticks_per_cycle * scenario.repeats
    expected_duration_s = expected_ticks / _PROBE_RATE_HZ

    expected: list[tuple[str, int | None, int | None, str | None, int | None]] = [
        ("scenario", None, None, None, None)
    ]
    tick_index = 0
    for repetition in range(1, scenario.repeats + 1):
        expected.append(("repetition", repetition, None, None, tick_index))
        for segment_index, segment in enumerate(scenario.command_segments):
            expected.append(
                (
                    "segment",
                    repetition,
                    segment_index,
                    str(segment.get("label", f"segment_{segment_index}")),
                    tick_index,
                )
            )
            tick_index += int(round(float(segment["duration_s"]) * _PROBE_RATE_HZ))
    expected.append(("complete", None, None, None, expected_ticks))
    observed_names = [event.get("event") for _, event in samples]
    expected_names = [item[0] for item in expected]
    if observed_names != expected_names:
        raise ContractError(
            "probe events do not contain one ordered scenario, all repetitions/segments, "
            "and one complete marker"
        )

    scenario_event = samples[0][1]
    scenario_receive_time_s = float(samples[0][0])
    if not math.isclose(
        _event_number(scenario_event, "rate_hz"), _PROBE_RATE_HZ, abs_tol=1.0e-12
    ):
        raise ContractError("probe event rate mismatch")
    if _event_int(scenario_event, "repeats") != scenario.repeats:
        raise ContractError("probe event repeat count mismatch")
    if _event_int(scenario_event, "ticks") != expected_ticks:
        raise ContractError("probe event tick count mismatch")
    if not math.isclose(
        _event_number(scenario_event, "duration_s"),
        expected_duration_s,
        abs_tol=1.0e-12,
    ):
        raise ContractError("probe event duration mismatch")

    command_times_s: list[float] = []
    command_values: list[float] = []
    for (receive_time_s, event), (_, repetition, segment_index, segment, expected_tick) in zip(
        samples[1:-1], expected[1:-1], strict=True
    ):
        if _event_int(event, "repetition") != repetition:
            raise ContractError("probe event repetition order mismatch")
        if event["event"] == "segment":
            if _event_int(event, "segment_index") != segment_index:
                raise ContractError("probe event segment index mismatch")
            if event.get("segment") != segment:
                raise ContractError("probe event segment label mismatch")
            if _event_int(event, "tick_index") != expected_tick:
                raise ContractError("probe event segment tick mismatch")
            command_times_s.append(receive_time_s)
            command_values.append(
                float(scenario.command_segments[int(segment_index)]["value"])
            )
        scheduled = _event_number(event, "scheduled_elapsed_s")
        if not math.isclose(
            scheduled, float(expected_tick) / _PROBE_RATE_HZ, abs_tol=1.0e-12
        ):
            raise ContractError("probe event scheduled time mismatch")
        actual = _event_number(event, "actual_elapsed_s")
        lateness = _event_number(event, "lateness_s")
        if lateness < 0.0 or lateness >= 1.0 / _PROBE_RATE_HZ:
            raise ContractError("probe event lateness is outside the reviewed bound")
        if not math.isclose(actual, scheduled + lateness, abs_tol=1.0e-9):
            raise ContractError("probe event actual time is inconsistent with lateness")
        receive_elapsed_s = float(receive_time_s) - scenario_receive_time_s
        if not math.isclose(
            receive_elapsed_s,
            actual,
            rel_tol=0.0,
            abs_tol=_PROBE_RECEIVE_JITTER_BOUND_S,
        ):
            raise ContractError(
                "probe event receive time is inconsistent with its actual elapsed time"
            )

    complete = samples[-1][1]
    if _event_int(complete, "ticks") != expected_ticks:
        raise ContractError("probe complete tick count mismatch")
    if not math.isclose(
        _event_number(complete, "scheduled_elapsed_s"),
        expected_duration_s,
        abs_tol=1.0e-12,
    ):
        raise ContractError("probe complete scheduled time mismatch")
    complete_actual = _event_number(complete, "actual_elapsed_s")
    complete_lateness = _event_number(complete, "lateness_s")
    if complete_lateness < 0.0 or complete_lateness >= 1.0 / _PROBE_RATE_HZ:
        raise ContractError("probe complete lateness is outside the reviewed bound")
    if not math.isclose(
        complete_actual,
        expected_duration_s + complete_lateness,
        abs_tol=1.0e-9,
    ):
        raise ContractError("probe complete time is inconsistent with lateness")
    complete_receive_elapsed_s = float(samples[-1][0]) - scenario_receive_time_s
    if not math.isclose(
        complete_receive_elapsed_s,
        complete_actual,
        rel_tol=0.0,
        abs_tol=_PROBE_RECEIVE_JITTER_BOUND_S,
    ):
        raise ContractError(
            "probe complete receive time is inconsistent with its actual elapsed time"
        )

    receive_times_s = np.asarray([receive_time_s for receive_time_s, _ in samples])
    if len(receive_times_s) > 1 and not np.all(np.diff(receive_times_s) >= 0.0):
        raise ContractError("probe event receive timestamps must be monotonic")
    return (
        {
            "scenario_sha256": digest,
            "scenario_receive_time_s": samples[0][0],
            "complete_receive_time_s": samples[-1][0],
            "repetition_count": scenario.repeats,
            "segment_count": scenario.repeats * len(scenario.command_segments),
            "complete_ticks": expected_ticks,
            "receive_duration_s": complete_receive_elapsed_s,
            "receive_jitter_bound_s": _PROBE_RECEIVE_JITTER_BOUND_S,
            "abad_output_disabled_verified": True,
        },
        command_times_s,
        command_values,
    )


def _load_rosbag(
    path: Path,
    scenario: ScenarioSpecV1,
    profile: CalibrationProfileV1 | None,
) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, Any]]:
    rosbag2_py, deserialize_message, get_message = _rosbag_dependencies()
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id=""),
        rosbag2_py.ConverterOptions(
            input_serialization_format="",
            output_serialization_format="",
        ),
    )
    topic_types = {
        item.name: item.type for item in reader.get_all_topics_and_types()
    }
    recognized = {
        "/motor/command",
        "/motor/state",
        "/imu/data",
        "/power/state",
        _PROBE_EVENT_TOPIC,
    }
    message_types = {
        topic: get_message(type_name)
        for topic, type_name in topic_types.items()
        if topic in recognized
    }
    is_main_drive_experiment = scenario.experiment_kind in _MAIN_DRIVE_EXPERIMENTS
    is_bound_probe = scenario.experiment_kind == "step_coast"
    if is_main_drive_experiment:
        missing_topics = {"/motor/command", "/motor/state"} - set(message_types)
        if missing_topics:
            raise ContractError(
                "rosbag missing required topics: " + ", ".join(sorted(missing_topics))
            )
    if is_bound_probe and _PROBE_EVENT_TOPIC not in message_types:
        raise ContractError(
            f"rosbag missing required probe events topic {_PROBE_EVENT_TOPIC}"
        )
    selected_leg = _JOINT_TO_LEG.get(scenario.joint)
    if selected_leg is None and is_main_drive_experiment:
        raise ContractError(f"unsupported main-drive joint for rosbag import: {scenario.joint}")
    hardware = profile.hardware_mapping if profile is not None else {}

    def calibrated(field: str, fallback: float) -> tuple[float, bool]:
        mapping = hardware.get(field, {})
        if isinstance(mapping, Mapping) and scenario.joint in mapping:
            return float(mapping[scenario.joint]), True
        return fallback, False

    counts_per_rev, has_counts = calibrated("encoder_counts_per_rev", _COUNTS_PER_REV)
    encoder_zero, has_zero = calibrated("encoder_zero_count", 0.0)
    encoder_sign, has_sign = calibrated(
        "encoder_sign", _ENCODER_SIGN.get(selected_leg, 1.0)
    )
    joint_direction, has_joint_direction = calibrated("joint_direction", 1.0)
    pwm_scale, has_pwm_scale = calibrated("pwm_scale", 1.0 / _MAX_PWM)
    pwm_cap, has_pwm_cap = calibrated("pwm_cap", 1.0)
    fully_profiled = all(
        (
            has_counts,
            has_zero,
            has_sign,
            has_joint_direction,
            has_pwm_scale,
            has_pwm_cap,
        )
    )
    position_fully_profiled = all((has_counts, has_zero, has_sign))
    command_fully_profiled = all((has_joint_direction, has_pwm_scale, has_pwm_cap))

    def mapping_source(measured: bool) -> str:
        if profile is None:
            return "provisional_repository_defaults"
        if measured:
            return f"profile:{profile.profile_id}"
        return f"profile:{profile.profile_id}:with_provisional_fallbacks"

    times: dict[str, list[float]] = {
        "command_time_s": [],
        "motor_command_time_s": [],
        "position_time_s": [],
        "imu_time_s": [],
        "power_time_s": [],
    }
    values: dict[str, list[Any]] = {
        "command": [],
        "position": [],
        "motor_command_pwm_raw": [],
        "motor_command_canonical": [],
        "motor_state_encoder_raw": [],
        "imu_acceleration": [],
        "imu_angular_velocity": [],
        "imu_orientation_xyzw": [],
        "power_voltage": [],
        "power_current": [],
    }
    probe_events: list[tuple[float, dict[str, Any]]] = []
    earliest: float | None = None
    selected_leg_observed_enabled = False
    while reader.has_next():
        topic, serialized, timestamp_ns = reader.read_next()
        if topic not in message_types:
            continue
        if topic == _PROBE_EVENT_TOPIC and not is_bound_probe:
            continue
        timestamp_s = float(timestamp_ns) * 1e-9
        earliest = timestamp_s if earliest is None else min(earliest, timestamp_s)
        message = deserialize_message(serialized, message_types[topic])
        if topic == _PROBE_EVENT_TOPIC:
            probe_events.append((timestamp_s, _probe_event(message)))
        elif topic == "/motor/command":
            legs = [
                _leg(message, name, ("enable", "direction", "voltage"), topic)
                for name in _LEG_ORDER
            ]
            if is_main_drive_experiment:
                enabled_legs = tuple(
                    name
                    for name, leg in zip(_LEG_ORDER, legs, strict=True)
                    if bool(leg.enable)
                )
                unexpected = tuple(name for name in enabled_legs if name != selected_leg)
                if unexpected:
                    raise ContractError(
                        "raw main-drive command enables "
                        f"{', '.join(unexpected)} but scenario {scenario.scenario_id} "
                        f"binds {scenario.joint} to {selected_leg}"
                    )
                selected_leg_observed_enabled |= selected_leg in enabled_legs
            signed_pwm = []
            for name, leg in zip(_LEG_ORDER, legs, strict=True):
                voltage = float(leg.voltage) if bool(leg.enable) else 0.0
                sign = 1.0 if bool(leg.direction) == _POSITIVE_DIRECTION[name] else -1.0
                signed_pwm.append(sign * voltage)
            times["motor_command_time_s"].append(timestamp_s)
            values["motor_command_pwm_raw"].append(signed_pwm)
            selected_index = _LEG_ORDER.index(selected_leg)
            scaled = signed_pwm[selected_index] * pwm_scale * joint_direction
            mapped_command = max(-pwm_cap, min(pwm_cap, scaled))
            values["motor_command_canonical"].append(mapped_command)
            if not is_bound_probe:
                times["command_time_s"].append(timestamp_s)
                values["command"].append(mapped_command)
        elif topic == "/motor/state":
            legs = [
                _leg(message, name, ("position",), topic) for name in _LEG_ORDER
            ]
            encoder = [float(leg.position) for leg in legs]
            times["position_time_s"].append(timestamp_s)
            values["motor_state_encoder_raw"].append(encoder)
            selected_index = _LEG_ORDER.index(selected_leg)
            position_rad = (
                (encoder[selected_index] - encoder_zero)
                * encoder_sign
                * 2.0
                * math.pi
                / counts_per_rev
            )
            values["position"].append(position_rad)
        elif topic == "/imu/data":
            orientation = _field(message, "orientation", topic)
            times["imu_time_s"].append(timestamp_s)
            values["imu_acceleration"].append(
                _vector(message, "linear_acceleration", topic)
            )
            values["imu_angular_velocity"].append(
                _vector(message, "angular_velocity", topic)
            )
            values["imu_orientation_xyzw"].append(
                [
                    float(_field(orientation, axis, f"{topic}.orientation"))
                    for axis in ("x", "y", "z", "w")
                ]
            )
        else:
            voltage = [float(_field(message, f"v_{index}", topic)) for index in range(8)]
            current = [float(_field(message, f"i_{index}", topic)) for index in range(8)]
            times["power_time_s"].append(timestamp_s)
            values["power_voltage"].append(voltage)
            values["power_current"].append(current)
    if earliest is None:
        raise ContractError("rosbag contains no recognized messages")
    if is_main_drive_experiment and not selected_leg_observed_enabled:
        raise ContractError(
            "raw main-drive command never enabled the scenario joint "
            f"{scenario.joint} ({selected_leg})"
        )
    probe_evidence: dict[str, Any] | None = None
    replay_initial_state: dict[str, Any] | None = None
    if is_bound_probe:
        probe_evidence, event_command_times, event_commands = _validate_probe_events(
            probe_events, scenario
        )
        scenario_start = float(probe_evidence["scenario_receive_time_s"])
        if not times["motor_command_time_s"] or scenario_start >= times["motor_command_time_s"][0]:
            raise ContractError(
                "probe scenario marker must precede the first raw motor command"
            )
        # The raw bridge intentionally suppresses globally-disabled packets. The
        # authenticated segment markers are therefore the authoritative requested
        # command timeline; raw PWM remains on its own untouched clock for mapping.
        times["command_time_s"] = event_command_times
        values["command"] = event_commands
        if not times["position_time_s"]:
            raise ContractError("probe replay initial state has no position samples")
        position_time = np.asarray(times["position_time_s"], dtype=np.float64)
        position = np.asarray(values["position"], dtype=np.float64)
        distances = np.abs(position_time - scenario_start)
        nearest_distance = float(np.min(distances))
        tolerance = 1.0e-9
        if nearest_distance > 1.0 / _PROBE_RATE_HZ + tolerance:
            raise ContractError(
                "probe replay initial state has no position sample at scenario start"
            )
        nearest = np.flatnonzero(
            np.isclose(distances, nearest_distance, rtol=0.0, atol=tolerance)
        )
        if nearest.size != 1:
            raise ContractError(
                "probe replay initial state position sample is ambiguous at scenario start"
            )
        sample_index = int(nearest[0])
        sample_time_s = float(position_time[sample_index]) - earliest
        scenario_time_s = scenario_start - earliest
        replay_initial_state = {
            "schema_version": 1,
            "joint": scenario.joint,
            "source_channel": "position",
            "position_rad": float(position[sample_index]),
            "sample_time_s": sample_time_s,
            "scenario_time_s": scenario_time_s,
            "sample_offset_s": sample_time_s - scenario_time_s,
        }
    arrays: dict[str, np.ndarray] = {}
    for time_name, samples in times.items():
        if samples:
            arrays[time_name] = np.asarray(samples, dtype=float) - earliest
    for channel, samples in values.items():
        if samples:
            arrays[channel] = np.asarray(samples, dtype=float)
    extra_time_bases: dict[str, str] = {}
    for channel, time_name in (
        ("motor_command_pwm_raw", "motor_command_time_s"),
        ("motor_command_canonical", "motor_command_time_s"),
        ("motor_state_encoder_raw", "position_time_s"),
        ("imu_acceleration", "imu_time_s"),
        ("imu_angular_velocity", "imu_time_s"),
        ("imu_orientation_xyzw", "imu_time_s"),
        ("power_voltage", "power_time_s"),
        ("power_current", "power_time_s"),
    ):
        if channel in arrays:
            extra_time_bases[channel] = time_name
    constants = {
        "calibration_source": mapping_source(fully_profiled),
        "position_mapping_source": mapping_source(position_fully_profiled),
        "requested_command_source": (
            f"authenticated_probe_events:{probe_evidence['scenario_sha256']}"
            if probe_evidence is not None
            else mapping_source(command_fully_profiled)
        ),
        "encoder_counts_per_rev": counts_per_rev,
        "encoder_zero_count": encoder_zero,
        "encoder_sign": encoder_sign,
        "joint_direction": joint_direction,
        "pwm_scale": pwm_scale,
        "pwm_cap": pwm_cap,
        "motor_command_conversion_unit": (
            "rad/s"
            if all((has_joint_direction, has_pwm_scale, has_pwm_cap))
            else "normalized"
        ),
        "selected_leg": selected_leg,
        "raw_enabled_leg_binding_verified": bool(is_main_drive_experiment),
        "positive_direction_bit": _POSITIVE_DIRECTION.get(selected_leg),
    }
    if probe_evidence is not None:
        relative_evidence = dict(probe_evidence)
        relative_evidence["scenario_receive_time_s"] -= earliest
        relative_evidence["complete_receive_time_s"] -= earliest
        constants["probe_event_evidence"] = relative_evidence
    if replay_initial_state is not None:
        constants["replay_initial_state"] = replay_initial_state
        constants["replay_initial_state_sha256"] = sha256_json(replay_initial_state)
    return arrays, extra_time_bases, constants


def import_real_trace(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    scenario: ScenarioSpecV1 | str | Path,
    source_kind: str = "real",
    units: Mapping[str, str] | None = None,
    frames: Mapping[str, str] | None = None,
    latency_clock: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    time_bases: Mapping[str, str] | None = None,
    profile: CalibrationProfileV1 | None = None,
) -> TraceManifestV1:
    source = Path(source_path)
    spec = scenario if isinstance(scenario, ScenarioSpecV1) else load_scenario(scenario)
    clock = resolve_latency_clock(source, latency_clock)
    if source.is_file() and source.suffix == ".npz":
        arrays = _load_numeric_npz(source)
        extracted_time_bases: dict[str, str] = {}
        calibration_constants: dict[str, Any] = {}
    else:
        arrays, extracted_time_bases, calibration_constants = _load_rosbag(
            source, spec, profile
        )
    details = dict(metadata or {})
    details["units"] = dict(units or {name: "unspecified" for name in spec.required_channels})
    details["frames"] = dict(frames or {name: "unspecified" for name in spec.required_channels})
    if "motor_command_pwm_raw" in arrays:
        details["units"].setdefault("motor_command_pwm_raw", "raw_pwm")
        details["frames"].setdefault("motor_command_pwm_raw", "rinbo_leg_order")
    if "motor_command_canonical" in arrays:
        details["units"].setdefault(
            "motor_command_canonical",
            str(calibration_constants["motor_command_conversion_unit"]),
        )
        details["frames"].setdefault("motor_command_canonical", spec.joint)
    details.setdefault("joint_order", [] if spec.joint in {"all", "root"} else [spec.joint])
    details["clock"] = {
        "source": clock,
        "timestamp_semantics": "relative_monotonic",
        "time_unit": "s",
    }
    details.setdefault("git_sha", None)
    details.setdefault("asset_sha256", None)
    details.setdefault("config_sha256", None)
    caller_constants = dict(details.get("calibration_constants", {}))
    collisions = set(caller_constants).intersection(calibration_constants)
    if collisions:
        raise ContractError(
            "metadata.calibration_constants conflicts with extracted constants: "
            + ", ".join(sorted(collisions))
        )
    details["calibration_constants"] = {
        **caller_constants,
        **calibration_constants,
    }
    merged_time_bases = {**extracted_time_bases, **dict(time_bases or {})}
    return write_trace(
        output_dir,
        arrays,
        scenario=spec,
        source=source_kind,
        source_path=source,
        metadata=details,
        time_bases=merged_time_bases,
        profile=profile,
    )
