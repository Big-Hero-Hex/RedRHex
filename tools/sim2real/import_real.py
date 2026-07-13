from __future__ import annotations

import importlib
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
from .traces import write_trace


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
    recognized = {"/motor/command", "/motor/state", "/imu/data", "/power/state"}
    message_types = {
        topic: get_message(type_name)
        for topic, type_name in topic_types.items()
        if topic in recognized
    }
    if scenario.experiment_kind in {"step", "coast"}:
        missing_topics = {"/motor/command", "/motor/state"} - set(message_types)
        if missing_topics:
            raise ContractError(
                "rosbag missing required topics: " + ", ".join(sorted(missing_topics))
            )
    selected_leg = _JOINT_TO_LEG.get(scenario.joint)
    if selected_leg is None and scenario.experiment_kind in {"step", "coast"}:
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
    pwm_scale, has_pwm_scale = calibrated("pwm_scale", 1.0 / _MAX_PWM)
    pwm_cap, has_pwm_cap = calibrated("pwm_cap", 1.0)
    fully_profiled = all((has_counts, has_zero, has_sign, has_pwm_scale, has_pwm_cap))

    times: dict[str, list[float]] = {
        "command_time_s": [],
        "position_time_s": [],
        "imu_time_s": [],
        "power_time_s": [],
    }
    values: dict[str, list[Any]] = {
        "command": [],
        "position": [],
        "motor_command_pwm_raw": [],
        "motor_state_encoder_raw": [],
        "imu_acceleration": [],
        "imu_angular_velocity": [],
        "imu_orientation_xyzw": [],
        "power_voltage": [],
        "power_current": [],
    }
    earliest: float | None = None
    while reader.has_next():
        topic, serialized, timestamp_ns = reader.read_next()
        if topic not in message_types:
            continue
        timestamp_s = float(timestamp_ns) * 1e-9
        earliest = timestamp_s if earliest is None else min(earliest, timestamp_s)
        message = deserialize_message(serialized, message_types[topic])
        if topic == "/motor/command":
            legs = [
                _leg(message, name, ("enable", "direction", "voltage"), topic)
                for name in _LEG_ORDER
            ]
            signed_pwm = []
            for name, leg in zip(_LEG_ORDER, legs, strict=True):
                voltage = float(leg.voltage) if bool(leg.enable) else 0.0
                sign = 1.0 if bool(leg.direction) == _POSITIVE_DIRECTION[name] else -1.0
                signed_pwm.append(sign * voltage)
            times["command_time_s"].append(timestamp_s)
            values["motor_command_pwm_raw"].append(signed_pwm)
            selected_index = _LEG_ORDER.index(selected_leg)
            scaled = signed_pwm[selected_index] * pwm_scale
            values["command"].append(max(-pwm_cap, min(pwm_cap, scaled)))
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
    arrays: dict[str, np.ndarray] = {}
    for time_name, samples in times.items():
        if samples:
            arrays[time_name] = np.asarray(samples, dtype=float) - earliest
    for channel, samples in values.items():
        if samples:
            arrays[channel] = np.asarray(samples, dtype=float)
    extra_time_bases: dict[str, str] = {}
    for channel, time_name in (
        ("motor_command_pwm_raw", "command_time_s"),
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
        "calibration_source": (
            f"profile:{profile.profile_id}"
            if profile is not None and fully_profiled
            else (
                f"profile:{profile.profile_id}:with_provisional_fallbacks"
                if profile is not None
                else "provisional_repository_defaults"
            )
        ),
        "encoder_counts_per_rev": counts_per_rev,
        "encoder_zero_count": encoder_zero,
        "encoder_sign": encoder_sign,
        "pwm_scale": pwm_scale,
        "pwm_cap": pwm_cap,
        "selected_leg": selected_leg,
        "positive_direction_bit": _POSITIVE_DIRECTION.get(selected_leg),
    }
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
