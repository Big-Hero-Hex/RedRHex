#!/usr/bin/env python3
"""Import a strict Sensor-V2 rosbag2 capture into the canonical replay NPZ.

ROS dependencies are loaded only by :func:`read_sensor_v2_rosbag`, keeping the
conversion and receipt validators usable in dependency-light tests.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_IO_ROOT = REPO_ROOT / "source" / "redrhex_policy_io"
if str(POLICY_IO_ROOT) not in sys.path:
    sys.path.insert(0, str(POLICY_IO_ROOT))

from redrhex_policy_io import (  # noqa: E402
    ContractError,
    ForwardResidualActionContractV2,
    StudentObservationContractV2,
)


IMPORT_RECEIPT_SCHEMA_V2 = "redrhex.sensor-v2-rosbag-import.v1"
CAPTURE_ATTESTATION_SCHEMA_V2 = "redrhex.sensor-v2-capture-attestation.v1"
PHYSICAL_CAPTURE_DECLARATION_V2 = "physical_hardware"
SAMPLE_RATE_HZ_V2 = 60.0
MAX_PERIOD_ERROR_RATIO_V2 = 0.25
MAX_IMU_JOINT_SKEW_S_V2 = 0.5 / SAMPLE_RATE_HZ_V2
MAX_COMMAND_AGE_S_V2 = 0.25
MIN_SOURCE_SAMPLES_V2 = 61
JOINT_ORDER_V2 = (
    ForwardResidualActionContractV2.MAIN_JOINT_ORDER
    + ForwardResidualActionContractV2.ABAD_JOINT_ORDER
)
REQUIRED_TOPIC_TYPES_V2 = {
    "/joint_states": "sensor_msgs/msg/JointState",
    "/imu/data": "sensor_msgs/msg/Imu",
    "/cmd_vel": "geometry_msgs/msg/Twist",
    "/redrhex/joint_feedback_status_v2": "diagnostic_msgs/msg/DiagnosticArray",
}
TIMESTAMP_SEMANTICS_V2 = {
    "/joint_states": "header_source_stamp",
    "/imu/data": "header_source_stamp",
    "/redrhex/joint_feedback_status_v2": "header_source_stamp",
    "/cmd_vel": "bag_receive_time_unstamped_message",
    "canonical_timestamp_s": "/joint_states.header.source_stamp",
}


@dataclass(frozen=True)
class ValidatedSensorV2ImportReceipt:
    receipt_path: Path
    receipt_sha256: str
    source_bag_path: Path
    source_bag_sha256: str
    trace_path: Path
    trace_sha256: str
    capture_attestation_path: Path
    capture_attestation_sha256: str
    capture_attestation: dict[str, Any]
    payload: dict[str, Any]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_hash_kind(path: Path) -> str:
    return "sha256-file-v1" if path.is_file() else "sha256-directory-manifest-v1"


def sha256_path_v2(path: str | Path) -> str:
    """Hash a file or a rosbag directory using a deterministic manifest."""

    source = Path(path).expanduser().resolve()
    if source.is_file():
        return _file_sha256(source)
    if not source.is_dir():
        raise OSError(f"artifact path does not exist: {source}")
    entries_on_disk = sorted(source.rglob("*"))
    symlinks = [item for item in entries_on_disk if item.is_symlink()]
    if symlinks:
        raise OSError(f"artifact directory contains a symlink: {symlinks[0]}")
    files = [item for item in entries_on_disk if item.is_file()]
    if not files:
        raise OSError(f"artifact directory contains no files: {source}")
    entries: list[dict[str, Any]] = []
    for item in files:
        entries.append(
            {
                "path": item.relative_to(source).as_posix(),
                "sha256": _file_sha256(item),
                "size": item.stat().st_size,
            }
        )
    payload = json.dumps(
        {
            "schema": "redrhex.directory-sha256.v1",
            "files": entries,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _resolve_record_path(receipt_path: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{label} path must be a non-empty string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = receipt_path.parent / path
    return path.resolve()


def _load_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ContractError(f"cannot read Sensor-V2 import receipt {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError("Sensor-V2 import receipt must be a JSON object")
    return payload


def validate_sensor_v2_capture_attestation(
    attestation_path: str | Path,
    *,
    expected_attestation_sha256: str,
    source_bag_path: str | Path,
    source_bag_sha256: str,
) -> tuple[Path, str, dict[str, Any]]:
    """Validate the operator/capture-system declaration bound to a rosbag."""

    path = Path(attestation_path).expanduser().resolve()
    expected_hash = _require_sha256(
        expected_attestation_sha256,
        "Sensor-V2 capture attestation sha256",
    )
    actual_hash = sha256_path_v2(path)
    if actual_hash != expected_hash:
        raise ContractError("Sensor-V2 capture attestation sha256 mismatch")
    payload = _load_json(path)
    required = {
        "schema",
        "source_recorder_id",
        "operator_id",
        "capture_declaration",
        "attested_at_utc",
        "observation_contract_sha256",
        "attitude_mode",
        "runtime_calibration_sha256",
        "source_bag_sha256",
        "source_bag_hash_kind",
        "topics",
    }
    if set(payload) != required:
        raise ContractError("Sensor-V2 capture attestation fields changed or are incomplete")
    if payload["schema"] != CAPTURE_ATTESTATION_SCHEMA_V2:
        raise ContractError("unsupported Sensor-V2 capture attestation schema")
    for name in ("source_recorder_id", "operator_id"):
        if not isinstance(payload[name], str) or not payload[name].strip():
            raise ContractError(f"Sensor-V2 capture attestation {name} is missing")
    if payload["capture_declaration"] != PHYSICAL_CAPTURE_DECLARATION_V2:
        raise ContractError(
            "Sensor-V2 real replay requires a physical_hardware capture declaration"
        )
    _require_sha256(
        payload["observation_contract_sha256"],
        "Sensor-V2 capture observation contract sha256",
    )
    if payload["attitude_mode"] not in {"causal_gyro_accel", "validated_quaternion"}:
        raise ContractError("Sensor-V2 capture attestation attitude mode is invalid")
    _require_sha256(
        payload["runtime_calibration_sha256"],
        "Sensor-V2 capture runtime calibration sha256",
    )
    attested_at = payload["attested_at_utc"]
    if not isinstance(attested_at, str) or not attested_at.endswith("Z"):
        raise ContractError("Sensor-V2 capture attestation requires a UTC timestamp")
    try:
        parsed_time = datetime.fromisoformat(attested_at[:-1] + "+00:00")
    except ValueError as exc:
        raise ContractError("Sensor-V2 capture attestation UTC timestamp is invalid") from exc
    if parsed_time.utcoffset() is None or parsed_time.utcoffset().total_seconds() != 0.0:
        raise ContractError("Sensor-V2 capture attestation timestamp must be UTC")
    source = Path(source_bag_path).expanduser().resolve()
    expected_source_hash = _require_sha256(
        source_bag_sha256,
        "Sensor-V2 source rosbag sha256",
    )
    if payload["source_bag_sha256"] != expected_source_hash:
        raise ContractError("Sensor-V2 capture attestation binds a different rosbag hash")
    if payload["source_bag_hash_kind"] != _path_hash_kind(source):
        raise ContractError("Sensor-V2 capture attestation rosbag hash kind changed")
    if payload["topics"] != REQUIRED_TOPIC_TYPES_V2:
        raise ContractError("Sensor-V2 capture attestation topic schema changed")
    if sha256_path_v2(source) != expected_source_hash:
        raise ContractError("Sensor-V2 source rosbag changed after capture attestation")
    return path, actual_hash, payload


def validate_sensor_v2_import_receipt(
    receipt_path: str | Path,
    *,
    expected_receipt_sha256: str,
    expected_trace_path: str | Path | None = None,
) -> ValidatedSensorV2ImportReceipt:
    """Rehash a receipt, its canonical trace, and the original rosbag source."""

    receipt = Path(receipt_path).expanduser().resolve()
    expected_receipt_hash = _require_sha256(
        expected_receipt_sha256,
        "Sensor-V2 import receipt sha256",
    )
    actual_receipt_hash = sha256_path_v2(receipt)
    if actual_receipt_hash != expected_receipt_hash:
        raise ContractError(
            "Sensor-V2 import receipt sha256 mismatch: "
            f"expected {expected_receipt_hash}, got {actual_receipt_hash}"
        )
    payload = _load_json(receipt)
    required_keys = {
        "schema",
        "source_bag",
        "output_trace",
        "topics",
        "joint_order",
        "imu_frame_id",
        "observation_contract_sha256",
        "attitude_mode",
        "sample_rate_hz",
        "sample_count",
        "max_period_error_ratio",
        "max_imu_joint_skew_s",
        "observed_max_imu_joint_skew_s",
        "timestamp_semantics",
        "capture_attestation",
    }
    if set(payload) != required_keys:
        raise ContractError("Sensor-V2 import receipt fields changed or are incomplete")
    if payload["schema"] != IMPORT_RECEIPT_SCHEMA_V2:
        raise ContractError("unsupported Sensor-V2 import receipt schema")
    if payload["joint_order"] != list(JOINT_ORDER_V2):
        raise ContractError("Sensor-V2 import receipt joint order is not canonical")
    if payload["timestamp_semantics"] != TIMESTAMP_SEMANTICS_V2:
        raise ContractError("Sensor-V2 import receipt timestamp semantics changed")
    if payload["sample_rate_hz"] != SAMPLE_RATE_HZ_V2:
        raise ContractError("Sensor-V2 import receipt sample rate must be 60 Hz")
    if payload["max_period_error_ratio"] != MAX_PERIOD_ERROR_RATIO_V2:
        raise ContractError("Sensor-V2 import receipt cadence bound changed")
    if payload["max_imu_joint_skew_s"] != MAX_IMU_JOINT_SKEW_S_V2:
        raise ContractError("Sensor-V2 import receipt source-skew bound changed")
    if (
        isinstance(payload["sample_count"], bool)
        or not isinstance(payload["sample_count"], int)
        or payload["sample_count"] < MIN_SOURCE_SAMPLES_V2
    ):
        raise ContractError("Sensor-V2 import receipt has too few source samples")
    observed_skew = float(payload["observed_max_imu_joint_skew_s"])
    if (
        not math.isfinite(observed_skew)
        or observed_skew < 0.0
        or observed_skew > MAX_IMU_JOINT_SKEW_S_V2 + 1.0e-12
    ):
        raise ContractError("Sensor-V2 import receipt reports invalid source skew")
    imu_frame_id = payload["imu_frame_id"]
    if not isinstance(imu_frame_id, str) or not imu_frame_id:
        raise ContractError("Sensor-V2 import receipt has no IMU frame")
    observation_contract_sha256 = _require_sha256(
        payload["observation_contract_sha256"],
        "Sensor-V2 receipt observation contract sha256",
    )
    attitude_mode = payload["attitude_mode"]
    if attitude_mode not in {"causal_gyro_accel", "validated_quaternion"}:
        raise ContractError("Sensor-V2 import receipt attitude mode is invalid")

    topics = payload["topics"]
    if not isinstance(topics, Mapping) or set(topics) != set(REQUIRED_TOPIC_TYPES_V2):
        raise ContractError("Sensor-V2 import receipt lacks the four required topics")
    for topic, expected_type in REQUIRED_TOPIC_TYPES_V2.items():
        record = topics[topic]
        if not isinstance(record, Mapping) or set(record) != {"type", "message_count"}:
            raise ContractError(f"Sensor-V2 receipt topic record is invalid: {topic}")
        if record["type"] != expected_type:
            raise ContractError(f"Sensor-V2 receipt topic type changed: {topic}")
        count = record["message_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ContractError(f"Sensor-V2 receipt topic is empty: {topic}")
    for topic in (
        "/joint_states",
        "/imu/data",
        "/redrhex/joint_feedback_status_v2",
    ):
        if topics[topic]["message_count"] != payload["sample_count"]:
            raise ContractError(f"Sensor-V2 receipt sensor count disagrees: {topic}")

    source_record = payload["source_bag"]
    trace_record = payload["output_trace"]
    attestation_record = payload["capture_attestation"]
    if not isinstance(source_record, Mapping) or set(source_record) != {
        "path",
        "sha256",
        "hash_kind",
    }:
        raise ContractError("Sensor-V2 receipt source_bag record is invalid")
    if not isinstance(trace_record, Mapping) or set(trace_record) != {"path", "sha256"}:
        raise ContractError("Sensor-V2 receipt output_trace record is invalid")
    if not isinstance(attestation_record, Mapping) or set(attestation_record) != {
        "path",
        "sha256",
        "schema",
        "source_recorder_id",
        "operator_id",
        "capture_declaration",
        "attested_at_utc",
        "observation_contract_sha256",
        "attitude_mode",
        "runtime_calibration_sha256",
    }:
        raise ContractError("Sensor-V2 receipt capture_attestation record is invalid")
    source_path = _resolve_record_path(receipt, source_record["path"], "source_bag")
    trace_path = _resolve_record_path(receipt, trace_record["path"], "output_trace")
    if source_record["hash_kind"] != _path_hash_kind(source_path):
        raise ContractError("Sensor-V2 receipt source_bag hash kind disagrees with its path")
    source_sha256 = _require_sha256(source_record["sha256"], "source_bag sha256")
    trace_sha256 = _require_sha256(trace_record["sha256"], "output_trace sha256")
    if sha256_path_v2(source_path) != source_sha256:
        raise ContractError("Sensor-V2 source rosbag sha256 mismatch")
    if sha256_path_v2(trace_path) != trace_sha256:
        raise ContractError("Sensor-V2 canonical trace sha256 mismatch")
    if expected_trace_path is not None and trace_path != Path(expected_trace_path).expanduser().resolve():
        raise ContractError("Sensor-V2 receipt path does not identify the replay input trace")
    expected_trace_arrays = {
        "timestamp_s",
        "imu_source_timestamp_s",
        "joint_validity_timestamp_s",
        "imu_gyro_rad_s",
        "imu_linear_accel_m_s2",
        "imu_orientation_xyzw",
        "imu_orientation_covariance",
        "imu_frame_id",
        "main_position_rad",
        "abad_position_rad",
        "command",
    }
    try:
        with np.load(trace_path, allow_pickle=False) as archive:
            if set(archive.files) != expected_trace_arrays:
                raise ContractError("Sensor-V2 canonical trace arrays changed")
            sample_count = int(payload["sample_count"])
            expected_shapes = {
                "timestamp_s": (sample_count,),
                "imu_source_timestamp_s": (sample_count,),
                "joint_validity_timestamp_s": (sample_count,),
                "imu_gyro_rad_s": (sample_count, 3),
                "imu_linear_accel_m_s2": (sample_count, 3),
                "imu_orientation_xyzw": (sample_count, 4),
                "imu_orientation_covariance": (sample_count, 9),
                "main_position_rad": (sample_count, 6),
                "abad_position_rad": (sample_count, 6),
                "command": (sample_count, 3),
            }
            for name, shape in expected_shapes.items():
                values = np.asarray(archive[name])
                if values.shape != shape or not np.isfinite(values).all():
                    raise ContractError(
                        f"Sensor-V2 canonical trace array is invalid: {name}"
                    )
            trace_joint_times = np.asarray(archive["timestamp_s"], dtype=np.float64)
            trace_imu_times = np.asarray(
                archive["imu_source_timestamp_s"],
                dtype=np.float64,
            )
            trace_validity_times = np.asarray(
                archive["joint_validity_timestamp_s"],
                dtype=np.float64,
            )
            if float(np.min(trace_joint_times)) <= 0.0:
                raise ContractError(
                    "Sensor-V2 canonical trace source stamps must be positive"
                )
            _validate_cadence(trace_joint_times, "canonical /joint_states")
            _validate_cadence(trace_imu_times, "canonical /imu/data")
            _validate_cadence(
                trace_validity_times,
                "canonical joint validity",
            )
            trace_skew = np.abs(trace_imu_times - trace_joint_times)
            if (
                float(np.max(trace_skew))
                > MAX_IMU_JOINT_SKEW_S_V2 + 1.0e-12
                or not math.isclose(
                    float(np.max(trace_skew)),
                    observed_skew,
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                )
            ):
                raise ContractError(
                    "Sensor-V2 canonical trace source skew disagrees with receipt"
                )
            covariance = np.asarray(
                archive["imu_orientation_covariance"],
                dtype=np.float64,
            )
            quaternion = np.asarray(
                archive["imu_orientation_xyzw"],
                dtype=np.float64,
            )
            quaternion_norm = np.linalg.norm(quaternion, axis=1)
            unavailable = covariance[:, 0] == -1.0
            known_covariance = (
                (covariance[:, 0] >= 0.0)
                & np.any(covariance != 0.0, axis=1)
            )
            valid_quaternion = np.abs(quaternion_norm - 1.0) <= 0.02
            if attitude_mode == "validated_quaternion" and (
                np.any(~known_covariance) or np.any(~valid_quaternion)
            ):
                raise ContractError(
                    "Sensor-V2 canonical trace contains unknown IMU covariance"
                )
            if attitude_mode == "causal_gyro_accel" and np.any(
                ~(unavailable | (known_covariance & valid_quaternion))
            ):
                raise ContractError(
                    "Sensor-V2 causal trace has malformed orientation availability"
                )
            if str(np.asarray(archive["imu_frame_id"]).reshape(()).item()) != imu_frame_id:
                raise ContractError(
                    "Sensor-V2 canonical trace IMU frame disagrees with receipt"
                )
    except (OSError, ValueError) as exc:
        raise ContractError(f"cannot validate Sensor-V2 canonical trace: {exc}") from exc
    attestation_path = _resolve_record_path(
        receipt,
        attestation_record["path"],
        "capture_attestation",
    )
    try:
        (
            attestation_path,
            attestation_sha256,
            attestation,
        ) = validate_sensor_v2_capture_attestation(
            attestation_path,
            expected_attestation_sha256=attestation_record["sha256"],
            source_bag_path=source_path,
            source_bag_sha256=source_sha256,
        )
    except (OSError, ValueError) as exc:
        raise ContractError(f"invalid Sensor-V2 capture attestation: {exc}") from exc
    for name in (
        "schema",
        "source_recorder_id",
        "operator_id",
        "capture_declaration",
        "attested_at_utc",
        "observation_contract_sha256",
        "attitude_mode",
        "runtime_calibration_sha256",
    ):
        if attestation_record[name] != attestation[name]:
            raise ContractError(
                f"Sensor-V2 receipt disagrees with capture attestation: {name}"
            )
    if (
        attestation["observation_contract_sha256"]
        != observation_contract_sha256
        or attestation["attitude_mode"] != attitude_mode
    ):
        raise ContractError(
            "Sensor-V2 receipt contract binding disagrees with capture attestation"
        )
    return ValidatedSensorV2ImportReceipt(
        receipt_path=receipt,
        receipt_sha256=actual_receipt_hash,
        source_bag_path=source_path,
        source_bag_sha256=source_sha256,
        trace_path=trace_path,
        trace_sha256=trace_sha256,
        capture_attestation_path=attestation_path,
        capture_attestation_sha256=attestation_sha256,
        capture_attestation=attestation,
        payload=payload,
    )


def _header_stamp_s(message: object, topic: str) -> float:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    sec = getattr(stamp, "sec", None)
    nanosec = getattr(stamp, "nanosec", None)
    if (
        isinstance(sec, bool)
        or not isinstance(sec, int)
        or isinstance(nanosec, bool)
        or not isinstance(nanosec, int)
        or nanosec < 0
        or nanosec >= 1_000_000_000
    ):
        raise ContractError(f"{topic} requires an integer ROS header source stamp")
    value = float(sec) + float(nanosec) * 1.0e-9
    if not math.isfinite(value) or value <= 0.0:
        raise ContractError(f"{topic} requires a positive header source stamp")
    return value


def _bag_time_s(timestamp_ns: object, topic: str) -> float:
    if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int):
        raise ContractError(f"{topic} bag receive timestamp must be integer nanoseconds")
    value = float(timestamp_ns) * 1.0e-9
    if not math.isfinite(value) or value <= 0.0:
        raise ContractError(f"{topic} bag receive timestamp must be positive")
    return value


def _vector3(message: object, field: str, topic: str) -> np.ndarray:
    vector = getattr(message, field, None)
    values = np.asarray(
        [getattr(vector, axis, np.nan) for axis in ("x", "y", "z")],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ContractError(f"{topic}.{field} must be a finite three-vector")
    return values


def _joint_sample(message: object) -> tuple[float, np.ndarray]:
    stamp = _header_stamp_s(message, "/joint_states")
    names = [str(value) for value in getattr(message, "name", [])]
    if len(names) != 12 or len(set(names)) != 12 or set(names) != set(JOINT_ORDER_V2):
        raise ContractError("/joint_states must contain exactly the 12 canonical V2 joint names")
    positions = np.asarray(getattr(message, "position", []), dtype=np.float64)
    if positions.shape != (12,) or not np.isfinite(positions).all():
        raise ContractError("/joint_states must contain 12 finite measured positions")
    by_name = dict(zip(names, positions, strict=True))
    return stamp, np.asarray([by_name[name] for name in JOINT_ORDER_V2], dtype=np.float64)


def _imu_sample(
    message: object,
    expected_imu_frame_id: str,
    attitude_mode: str,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stamp = _header_stamp_s(message, "/imu/data")
    frame_id = str(getattr(getattr(message, "header", None), "frame_id", ""))
    if frame_id != expected_imu_frame_id:
        raise ContractError(
            f"/imu/data frame {frame_id!r} != expected {expected_imu_frame_id!r}"
        )
    gyro = _vector3(message, "angular_velocity", "/imu/data")
    acceleration = _vector3(message, "linear_acceleration", "/imu/data")
    orientation = getattr(message, "orientation", None)
    quaternion = np.asarray(
        [getattr(orientation, axis, np.nan) for axis in ("x", "y", "z", "w")],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(quaternion))
    covariance = np.asarray(
        getattr(message, "orientation_covariance", []),
        dtype=np.float64,
    )
    finite_quaternion = bool(
        quaternion.shape == (4,)
        and np.isfinite(quaternion).all()
    )
    valid_quaternion = finite_quaternion and abs(norm - 1.0) <= 0.02
    valid_covariance = bool(
        covariance.shape == (9,)
        and np.isfinite(covariance).all()
        and covariance[0] >= 0.0
        and np.any(covariance != 0.0)
    )
    orientation_unavailable = bool(
        covariance.shape == (9,)
        and np.isfinite(covariance).all()
        and covariance[0] == -1.0
        and finite_quaternion
    )
    if attitude_mode == "validated_quaternion" and not (
        valid_quaternion and valid_covariance
    ):
        raise ContractError("/imu/data orientation covariance is unknown or invalid")
    if attitude_mode == "causal_gyro_accel" and not (
        orientation_unavailable or (valid_quaternion and valid_covariance)
    ):
        raise ContractError(
            "/imu/data causal orientation must be valid or explicitly unavailable"
        )
    normalized = quaternion if orientation_unavailable else quaternion / norm
    return stamp, gyro, acceleration, normalized, covariance


def _validity_sample(message: object) -> tuple[float, tuple[bool, ...]]:
    stamp = _header_stamp_s(message, "/redrhex/joint_feedback_status_v2")
    validity: dict[str, bool] = {}
    for status in getattr(message, "status", []):
        fields = {
            str(getattr(item, "key", "")): str(getattr(item, "value", ""))
            for item in getattr(status, "values", [])
        }
        name = fields.get("joint_name")
        if name not in JOINT_ORDER_V2 or name in validity:
            raise ContractError("joint validity diagnostics contain unknown or duplicate names")
        expected_status_name = f"redrhex_joint_feedback_v2/{name}"
        if str(getattr(status, "name", "")) != expected_status_name:
            raise ContractError(f"joint validity diagnostic name changed for {name}")
        valid_text = fields.get("valid", "").strip().lower()
        valid = valid_text == "true" and int(getattr(status, "level", -1)) == 0
        validity[name] = valid
    if set(validity) != set(JOINT_ORDER_V2):
        raise ContractError("joint validity diagnostics must cover all 12 canonical joints")
    ordered = tuple(validity[name] for name in JOINT_ORDER_V2)
    if not all(ordered):
        raise ContractError("joint validity diagnostics contain invalid or unverified feedback")
    return stamp, ordered


def _command_sample(message: object) -> np.ndarray:
    linear = getattr(message, "linear", None)
    angular = getattr(message, "angular", None)
    command = np.asarray(
        [
            getattr(linear, "x", np.nan),
            getattr(linear, "y", np.nan),
            getattr(angular, "z", np.nan),
        ],
        dtype=np.float64,
    )
    if not np.isfinite(command).all():
        raise ContractError("/cmd_vel must contain finite vx, vy, and wz")
    return command


def _validate_strictly_increasing(times: np.ndarray, label: str) -> None:
    if times.ndim != 1 or times.size == 0 or not np.isfinite(times).all():
        raise ContractError(f"{label} timestamps are empty or invalid")
    if np.any(np.diff(times) <= 0.0):
        raise ContractError(f"{label} timestamps must be strictly increasing")


def _validate_cadence(times: np.ndarray, label: str) -> dict[str, float]:
    _validate_strictly_increasing(times, label)
    if times.size < MIN_SOURCE_SAMPLES_V2:
        raise ContractError(
            f"{label} requires at least {MIN_SOURCE_SAMPLES_V2} source samples"
        )
    periods = np.diff(times)
    expected = 1.0 / SAMPLE_RATE_HZ_V2
    error = np.abs(periods - expected) / expected
    if float(np.max(error)) > MAX_PERIOD_ERROR_RATIO_V2 + 1.0e-9:
        raise ContractError(f"{label} source cadence violates the fixed 60 Hz contract")
    return {
        "period_min_s": float(np.min(periods)),
        "period_max_s": float(np.max(periods)),
        "max_period_error_ratio": float(np.max(error)),
    }


def convert_sensor_v2_rosbag_records(
    records: Iterable[tuple[str, object, int]],
    *,
    topic_types: Mapping[str, str],
    observation_contract: StudentObservationContractV2,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Validate deserialized records and synchronize them without ROS imports."""

    contract = observation_contract.validate()
    if (
        contract.sample_rate_hz != SAMPLE_RATE_HZ_V2
        or contract.history_length != 60
    ):
        raise ContractError("rosbag importer requires the fixed 60 Hz / 60-frame V2 contract")
    expected_imu_frame_id = contract.imu_frame_id
    for topic, expected_type in REQUIRED_TOPIC_TYPES_V2.items():
        if topic_types.get(topic) != expected_type:
            raise ContractError(
                f"rosbag missing required topic/type {topic}: {expected_type}"
            )

    joints: list[tuple[float, float, np.ndarray]] = []
    imus: list[tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    validities: list[tuple[float, float, tuple[bool, ...]]] = []
    commands: list[tuple[float, np.ndarray]] = []
    topic_counts = {topic: 0 for topic in REQUIRED_TOPIC_TYPES_V2}
    for topic, message, timestamp_ns in records:
        if topic not in REQUIRED_TOPIC_TYPES_V2:
            continue
        bag_time = _bag_time_s(timestamp_ns, topic)
        topic_counts[topic] += 1
        if topic == "/joint_states":
            stamp, positions = _joint_sample(message)
            joints.append((stamp, bag_time, positions))
        elif topic == "/imu/data":
            stamp, gyro, accel, quaternion, covariance = _imu_sample(
                message,
                expected_imu_frame_id,
                contract.attitude_mode,
            )
            imus.append((stamp, bag_time, gyro, accel, quaternion, covariance))
        elif topic == "/redrhex/joint_feedback_status_v2":
            stamp, validity = _validity_sample(message)
            validities.append((stamp, bag_time, validity))
        else:
            commands.append((bag_time, _command_sample(message)))
    empty_topics = [topic for topic, count in topic_counts.items() if count == 0]
    if empty_topics:
        raise ContractError("rosbag required topics are empty: " + ", ".join(empty_topics))
    if not (len(joints) == len(imus) == len(validities)):
        raise ContractError("rosbag IMU/joint/validity sample counts must match exactly")

    joint_times = np.asarray([sample[0] for sample in joints], dtype=np.float64)
    imu_times = np.asarray([sample[0] for sample in imus], dtype=np.float64)
    validity_times = np.asarray([sample[0] for sample in validities], dtype=np.float64)
    command_times = np.asarray([sample[0] for sample in commands], dtype=np.float64)
    cadence = {
        "/joint_states": _validate_cadence(joint_times, "/joint_states"),
        "/imu/data": _validate_cadence(imu_times, "/imu/data"),
        "/redrhex/joint_feedback_status_v2": _validate_cadence(
            validity_times,
            "/redrhex/joint_feedback_status_v2",
        ),
    }
    _validate_strictly_increasing(command_times, "/cmd_vel bag receive")
    for label, values in (
        ("/joint_states bag receive", np.asarray([item[1] for item in joints])),
        ("/imu/data bag receive", np.asarray([item[1] for item in imus])),
        (
            "/redrhex/joint_feedback_status_v2 bag receive",
            np.asarray([item[1] for item in validities]),
        ),
    ):
        _validate_strictly_increasing(values, label)

    imu_skew = np.abs(imu_times - joint_times)
    validity_skew = np.abs(validity_times - joint_times)
    if float(np.max(imu_skew)) > MAX_IMU_JOINT_SKEW_S_V2 + 1.0e-12:
        raise ContractError("rosbag IMU/joint source skew exceeds the Sensor-V2 bound")
    if float(np.max(validity_skew)) > MAX_IMU_JOINT_SKEW_S_V2 + 1.0e-12:
        raise ContractError("rosbag validity/joint source skew exceeds the Sensor-V2 bound")

    synchronized_commands: list[np.ndarray] = []
    command_index = -1
    for _, joint_bag_time, _ in joints:
        while (
            command_index + 1 < len(commands)
            and commands[command_index + 1][0] <= joint_bag_time + 1.0e-9
        ):
            command_index += 1
        if command_index < 0:
            raise ContractError("rosbag has no /cmd_vel at or before a sensor generation")
        command_age = joint_bag_time - commands[command_index][0]
        if command_age < -1.0e-9 or command_age > MAX_COMMAND_AGE_S_V2 + 1.0e-9:
            raise ContractError("rosbag /cmd_vel is stale for a sensor generation")
        synchronized_commands.append(commands[command_index][1])

    positions = np.stack([item[2] for item in joints])
    arrays = {
        "timestamp_s": joint_times,
        "imu_source_timestamp_s": imu_times,
        "joint_validity_timestamp_s": validity_times,
        "imu_gyro_rad_s": np.stack([item[2] for item in imus]),
        "imu_linear_accel_m_s2": np.stack([item[3] for item in imus]),
        "imu_orientation_xyzw": np.stack([item[4] for item in imus]),
        "imu_orientation_covariance": np.stack([item[5] for item in imus]),
        "imu_frame_id": np.asarray(expected_imu_frame_id),
        "main_position_rad": positions[:, :6],
        "abad_position_rad": positions[:, 6:],
        "command": np.stack(synchronized_commands),
    }
    details = {
        "sample_count": int(joint_times.size),
        "topic_counts": topic_counts,
        "cadence": cadence,
        "observed_max_imu_joint_skew_s": float(np.max(imu_skew)),
        "observed_max_validity_joint_skew_s": float(np.max(validity_skew)),
        "observation_contract_sha256": contract.sha256,
        "attitude_mode": contract.attitude_mode,
    }
    return arrays, details


def _rosbag_dependencies() -> tuple[object, object, object]:
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
            "rosbag2_py, rclpy, and rosidl_runtime_py are required for Sensor-V2 "
            "rosbag import; source the matching ROS 2 environment"
        ) from exc
    return rosbag2_py, deserialize_message, get_message


def read_sensor_v2_rosbag(
    source_bag: str | Path,
) -> tuple[dict[str, str], list[tuple[str, object, int]]]:
    """Deserialize only the four evidence-bearing Sensor-V2 topics."""

    source = Path(source_bag).expanduser().resolve()
    rosbag2_py, deserialize_message, get_message = _rosbag_dependencies()
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(source), storage_id=""),
        rosbag2_py.ConverterOptions(
            input_serialization_format="",
            output_serialization_format="",
        ),
    )
    topic_types = {
        item.name: item.type for item in reader.get_all_topics_and_types()
    }
    for topic, expected_type in REQUIRED_TOPIC_TYPES_V2.items():
        if topic_types.get(topic) != expected_type:
            raise ContractError(
                f"rosbag missing required topic/type {topic}: {expected_type}"
            )
    message_types = {
        topic: get_message(topic_types[topic]) for topic in REQUIRED_TOPIC_TYPES_V2
    }
    records: list[tuple[str, object, int]] = []
    while reader.has_next():
        topic, serialized, timestamp_ns = reader.read_next()
        if topic in message_types:
            records.append(
                (
                    topic,
                    deserialize_message(serialized, message_types[topic]),
                    int(timestamp_ns),
                )
            )
    return topic_types, records


def _write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_sensor_v2_import_artifacts(
    source_bag: str | Path,
    output_trace: str | Path,
    receipt_path: str | Path,
    *,
    capture_attestation_path: str | Path,
    capture_attestation_sha256: str,
    topic_types: Mapping[str, str],
    records: Iterable[tuple[str, object, int]],
    observation_contract: StudentObservationContractV2,
) -> dict[str, Any]:
    """Convert validated records and atomically write the NPZ plus receipt."""

    source = Path(source_bag).expanduser().resolve()
    output = Path(output_trace).expanduser().resolve()
    receipt = Path(receipt_path).expanduser().resolve()
    attestation_input = Path(capture_attestation_path).expanduser().resolve()
    contract = observation_contract.validate()
    if output.suffix != ".npz":
        raise ContractError("Sensor-V2 canonical output trace must use the .npz suffix")
    if len({source, output, receipt, attestation_input}) != 4:
        raise ContractError(
            "Sensor-V2 source, output, receipt, and attestation paths must differ"
        )
    if source.is_dir() and (
        output.is_relative_to(source)
        or receipt.is_relative_to(source)
        or attestation_input.is_relative_to(source)
    ):
        raise ContractError(
            "Sensor-V2 outputs and capture attestation must stay outside the source rosbag"
        )
    source_sha256_before = sha256_path_v2(source)
    (
        attestation_path,
        attestation_sha256,
        attestation,
    ) = validate_sensor_v2_capture_attestation(
        attestation_input,
        expected_attestation_sha256=capture_attestation_sha256,
        source_bag_path=source,
        source_bag_sha256=source_sha256_before,
    )
    if (
        attestation["observation_contract_sha256"] != contract.sha256
        or attestation["attitude_mode"] != contract.attitude_mode
    ):
        raise ContractError(
            "capture attestation does not bind the requested observation contract"
        )
    arrays, details = convert_sensor_v2_rosbag_records(
        records,
        topic_types=topic_types,
        observation_contract=contract,
    )
    if sha256_path_v2(source) != source_sha256_before:
        raise ContractError("source rosbag changed during Sensor-V2 import")
    if sha256_path_v2(attestation_path) != attestation_sha256:
        raise ContractError("capture attestation changed during Sensor-V2 import")
    _write_npz_atomic(output, arrays)
    output_sha256 = sha256_path_v2(output)
    payload: dict[str, Any] = {
        "schema": IMPORT_RECEIPT_SCHEMA_V2,
        "source_bag": {
            "path": str(source),
            "sha256": source_sha256_before,
            "hash_kind": _path_hash_kind(source),
        },
        "output_trace": {
            "path": str(output),
            "sha256": output_sha256,
        },
        "topics": {
            topic: {
                "type": REQUIRED_TOPIC_TYPES_V2[topic],
                "message_count": int(details["topic_counts"][topic]),
            }
            for topic in REQUIRED_TOPIC_TYPES_V2
        },
        "joint_order": list(JOINT_ORDER_V2),
        "imu_frame_id": contract.imu_frame_id,
        "observation_contract_sha256": contract.sha256,
        "attitude_mode": contract.attitude_mode,
        "sample_rate_hz": SAMPLE_RATE_HZ_V2,
        "sample_count": int(details["sample_count"]),
        "max_period_error_ratio": MAX_PERIOD_ERROR_RATIO_V2,
        "max_imu_joint_skew_s": MAX_IMU_JOINT_SKEW_S_V2,
        "observed_max_imu_joint_skew_s": details[
            "observed_max_imu_joint_skew_s"
        ],
        "timestamp_semantics": dict(TIMESTAMP_SEMANTICS_V2),
        "capture_attestation": {
            "path": str(attestation_path),
            "sha256": attestation_sha256,
            "schema": attestation["schema"],
            "source_recorder_id": attestation["source_recorder_id"],
            "operator_id": attestation["operator_id"],
            "capture_declaration": attestation["capture_declaration"],
            "attested_at_utc": attestation["attested_at_utc"],
            "observation_contract_sha256": attestation[
                "observation_contract_sha256"
            ],
            "attitude_mode": attestation["attitude_mode"],
            "runtime_calibration_sha256": attestation[
                "runtime_calibration_sha256"
            ],
        },
    }
    _write_json_atomic(receipt, payload)
    return payload


def import_sensor_v2_rosbag(
    source_bag: str | Path,
    output_trace: str | Path,
    receipt_path: str | Path,
    *,
    observation_contract: StudentObservationContractV2,
    capture_attestation_path: str | Path,
    capture_attestation_sha256: str,
) -> dict[str, Any]:
    topic_types, records = read_sensor_v2_rosbag(source_bag)
    return write_sensor_v2_import_artifacts(
        source_bag,
        output_trace,
        receipt_path,
        capture_attestation_path=capture_attestation_path,
        capture_attestation_sha256=capture_attestation_sha256,
        topic_types=topic_types,
        records=records,
        observation_contract=observation_contract,
    )


def _load_observation_contract(
    path: Path,
    expected_sha256: str,
) -> StudentObservationContractV2:
    try:
        contract = StudentObservationContractV2.from_dict(_load_json(path))
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(f"invalid Sensor-V2 observation contract: {exc}") from exc
    expected = _require_sha256(
        expected_sha256,
        "Sensor-V2 observation contract sha256",
    )
    if contract.sha256 != expected:
        raise ContractError("Sensor-V2 observation contract sha256 mismatch")
    return contract


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_bag", type=Path)
    parser.add_argument("output_trace", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--observation-contract", type=Path, required=True)
    parser.add_argument("--observation-contract-sha256", required=True)
    parser.add_argument("--capture-attestation", type=Path, required=True)
    parser.add_argument("--capture-attestation-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    observation_contract = _load_observation_contract(
        args.observation_contract,
        args.observation_contract_sha256,
    )
    receipt = import_sensor_v2_rosbag(
        args.source_bag,
        args.output_trace,
        args.receipt,
        observation_contract=observation_contract,
        capture_attestation_path=args.capture_attestation,
        capture_attestation_sha256=args.capture_attestation_sha256,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, OSError, RuntimeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
