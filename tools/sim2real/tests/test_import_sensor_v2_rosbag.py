from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from redrhex_policy_io import ContractError, StudentObservationContractV2
from tools.sim2real.import_sensor_v2_rosbag import (
    CAPTURE_ATTESTATION_SCHEMA_V2,
    IMPORT_RECEIPT_SCHEMA_V2,
    JOINT_ORDER_V2,
    REQUIRED_TOPIC_TYPES_V2,
    convert_sensor_v2_rosbag_records,
    sha256_path_v2,
    validate_sensor_v2_import_receipt,
    write_sensor_v2_import_artifacts,
)


_VALIDATED_CONTRACT = StudentObservationContractV2.validated_quaternion(
    imu_frame_id="imu_link"
)
_CAUSAL_CONTRACT = StudentObservationContractV2.causal_gyro_accel(
    imu_frame_id="imu_link"
)


def _stamp(value_s: float) -> SimpleNamespace:
    sec = int(value_s)
    nanosec = int(round((value_s - sec) * 1.0e9))
    if nanosec == 1_000_000_000:
        sec += 1
        nanosec = 0
    return SimpleNamespace(sec=sec, nanosec=nanosec)


def _header(value_s: float, frame_id: str = "") -> SimpleNamespace:
    return SimpleNamespace(stamp=_stamp(value_s), frame_id=frame_id)


def _joint_message(value_s: float, sample: int) -> SimpleNamespace:
    names = list(reversed(JOINT_ORDER_V2))
    by_name = {name: float(index + sample) for index, name in enumerate(JOINT_ORDER_V2)}
    return SimpleNamespace(
        header=_header(value_s, "redrhex_base"),
        name=names,
        position=[by_name[name] for name in names],
        velocity=[],
    )


def _imu_message(
    value_s: float,
    *,
    frame_id: str = "imu_link",
    covariance: list[float] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        header=_header(value_s, frame_id),
        angular_velocity=SimpleNamespace(x=0.1, y=0.2, z=0.3),
        linear_acceleration=SimpleNamespace(x=0.0, y=0.0, z=9.80665),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        orientation_covariance=(
            covariance
            if covariance is not None
            else [1.0e-6, 0.0, 0.0, 0.0, 1.0e-6, 0.0, 0.0, 0.0, 1.0e-6]
        ),
    )


def _validity_message(value_s: float, *, invalid_name: str | None = None) -> SimpleNamespace:
    statuses = []
    for name in JOINT_ORDER_V2:
        valid = name != invalid_name
        statuses.append(
            SimpleNamespace(
                name=f"redrhex_joint_feedback_v2/{name}",
                level=0 if valid else 2,
                values=[
                    SimpleNamespace(key="joint_name", value=name),
                    SimpleNamespace(key="valid", value=str(valid).lower()),
                ],
            )
        )
    return SimpleNamespace(header=_header(value_s), status=statuses)


def _command_message(sample: int) -> SimpleNamespace:
    return SimpleNamespace(
        linear=SimpleNamespace(x=0.2 + sample * 0.001, y=0.0, z=99.0),
        angular=SimpleNamespace(x=99.0, y=99.0, z=0.0),
    )


def _records(
    *,
    period_s: float = 1.0 / 60.0,
    imu_skew_s: float = 0.001,
) -> list[tuple[str, object, int]]:
    result: list[tuple[str, object, int]] = []
    start = 1_000.0
    for index in range(61):
        source_time = start + index * period_s
        bag_time = source_time + 0.010
        result.extend(
            (
                (
                    "/cmd_vel",
                    _command_message(index),
                    int(round((bag_time - 0.001) * 1.0e9)),
                ),
                (
                    "/joint_states",
                    _joint_message(source_time, index),
                    int(round(bag_time * 1.0e9)),
                ),
                (
                    "/imu/data",
                    _imu_message(source_time + imu_skew_s),
                    int(round((bag_time + 0.001) * 1.0e9)),
                ),
                (
                    "/redrhex/joint_feedback_status_v2",
                    _validity_message(source_time),
                    int(round((bag_time + 0.002) * 1.0e9)),
                ),
            )
        )
    return result


def _capture_attestation(
    bag: Path,
    contract: StudentObservationContractV2 = _VALIDATED_CONTRACT,
) -> tuple[Path, str]:
    """Write a structural fixture; this does not represent project hardware evidence."""

    path = bag.parent / "capture-attestation.json"
    payload = {
        "schema": CAPTURE_ATTESTATION_SCHEMA_V2,
        "source_recorder_id": "ros2bag/redrhex-v2-recorder",
        "operator_id": "unit-test-structural-fixture",
        "capture_declaration": "physical_hardware",
        "attested_at_utc": "2026-08-16T00:00:00Z",
        "observation_contract_sha256": contract.sha256,
        "attitude_mode": contract.attitude_mode,
        "runtime_calibration_sha256": "a" * 64,
        "source_bag_sha256": sha256_path_v2(bag),
        "source_bag_hash_kind": (
            "sha256-file-v1" if bag.is_file() else "sha256-directory-manifest-v1"
        ),
        "topics": dict(REQUIRED_TOPIC_TYPES_V2),
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path, sha256_path_v2(path)


def test_conversion_uses_source_stamps_and_canonical_joint_order() -> None:
    arrays, details = convert_sensor_v2_rosbag_records(
        _records(),
        topic_types=REQUIRED_TOPIC_TYPES_V2,
        observation_contract=_VALIDATED_CONTRACT,
    )

    assert arrays["timestamp_s"].shape == (61,)
    assert arrays["main_position_rad"].shape == (61, 6)
    assert arrays["abad_position_rad"].shape == (61, 6)
    np.testing.assert_array_equal(arrays["main_position_rad"][0], np.arange(6))
    np.testing.assert_array_equal(arrays["abad_position_rad"][0], np.arange(6, 12))
    assert arrays["imu_frame_id"].item() == "imu_link"
    assert arrays["command"][0, 0] == pytest.approx(0.2)
    assert details["sample_count"] == 61
    assert details["observed_max_imu_joint_skew_s"] == pytest.approx(0.001)


def test_writer_emits_rehashable_trace_and_capture_attestation_receipt(tmp_path: Path) -> None:
    bag = tmp_path / "capture"
    bag.mkdir()
    (bag / "metadata.yaml").write_text("rosbag2_bagfile_information: {}\n", encoding="utf-8")
    (bag / "capture.db3").write_bytes(b"mock rosbag fixture; not hardware evidence")
    output = tmp_path / "canonical.npz"
    receipt_path = tmp_path / "canonical.receipt.json"
    attestation_path, attestation_sha256 = _capture_attestation(bag)

    receipt = write_sensor_v2_import_artifacts(
        bag,
        output,
        receipt_path,
        capture_attestation_path=attestation_path,
        capture_attestation_sha256=attestation_sha256,
        topic_types=REQUIRED_TOPIC_TYPES_V2,
        records=_records(),
        observation_contract=_VALIDATED_CONTRACT,
    )
    receipt_sha256 = sha256_path_v2(receipt_path)
    validated = validate_sensor_v2_import_receipt(
        receipt_path,
        expected_receipt_sha256=receipt_sha256,
        expected_trace_path=output,
    )

    assert receipt["schema"] == IMPORT_RECEIPT_SCHEMA_V2
    assert receipt["capture_attestation"]["sha256"] == attestation_sha256
    assert receipt["capture_attestation"]["runtime_calibration_sha256"] == "a" * 64
    assert receipt["source_bag"]["sha256"] == sha256_path_v2(bag)
    assert receipt["output_trace"]["sha256"] == sha256_path_v2(output)
    assert receipt["joint_order"] == list(JOINT_ORDER_V2)
    assert set(receipt["topics"]) == set(REQUIRED_TOPIC_TYPES_V2)
    assert validated.trace_path == output.resolve()
    assert validated.source_bag_path == bag.resolve()

    (bag / "capture.db3").write_bytes(b"tampered")
    with pytest.raises(ContractError, match="source rosbag sha256 mismatch"):
        validate_sensor_v2_import_receipt(
            receipt_path,
            expected_receipt_sha256=receipt_sha256,
            expected_trace_path=output,
        )


def test_writer_rejects_synthetic_or_missing_capture_attestation(tmp_path: Path) -> None:
    bag = tmp_path / "capture.db3"
    bag.write_bytes(b"mock rosbag")
    attestation_path, _ = _capture_attestation(bag)
    payload = json.loads(attestation_path.read_text(encoding="utf-8"))
    payload["capture_declaration"] = "synthetic_fixture"
    attestation_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ContractError, match="physical_hardware"):
        write_sensor_v2_import_artifacts(
            bag,
            tmp_path / "trace.npz",
            tmp_path / "receipt.json",
            capture_attestation_path=attestation_path,
            capture_attestation_sha256=sha256_path_v2(attestation_path),
            topic_types=REQUIRED_TOPIC_TYPES_V2,
            records=_records(),
            observation_contract=_VALIDATED_CONTRACT,
        )


def test_conversion_rejects_missing_topic_or_unverified_joint() -> None:
    topic_types = dict(REQUIRED_TOPIC_TYPES_V2)
    topic_types.pop("/cmd_vel")
    with pytest.raises(ContractError, match="missing required topic/type /cmd_vel"):
        convert_sensor_v2_rosbag_records(
            _records(),
            topic_types=topic_types,
            observation_contract=_VALIDATED_CONTRACT,
        )

    records = _records()
    for index, (topic, message, timestamp_ns) in enumerate(records):
        if topic == "/redrhex/joint_feedback_status_v2":
            records[index] = (
                topic,
                _validity_message(
                    1_000.0,
                    invalid_name=JOINT_ORDER_V2[-1],
                ),
                timestamp_ns,
            )
            break
    with pytest.raises(ContractError, match="invalid or unverified feedback"):
        convert_sensor_v2_rosbag_records(
            records,
            topic_types=REQUIRED_TOPIC_TYPES_V2,
            observation_contract=_VALIDATED_CONTRACT,
        )


def test_conversion_rejects_noncanonical_names_and_missing_source_stamp() -> None:
    records = _records()
    for index, (topic, message, timestamp_ns) in enumerate(records):
        if topic == "/joint_states":
            message.name[-1] = "not_a_sensor_v2_joint"
            records[index] = (topic, message, timestamp_ns)
            break
    with pytest.raises(ContractError, match="12 canonical V2 joint names"):
        convert_sensor_v2_rosbag_records(
            records,
            topic_types=REQUIRED_TOPIC_TYPES_V2,
            observation_contract=_VALIDATED_CONTRACT,
        )

    records = _records()
    for index, (topic, message, timestamp_ns) in enumerate(records):
        if topic == "/imu/data":
            message.header.stamp = _stamp(0.0)
            records[index] = (topic, message, timestamp_ns)
            break
    with pytest.raises(ContractError, match="positive header source stamp"):
        convert_sensor_v2_rosbag_records(
            records,
            topic_types=REQUIRED_TOPIC_TYPES_V2,
            observation_contract=_VALIDATED_CONTRACT,
        )


@pytest.mark.parametrize(
    ("period_s", "imu_skew_s", "match"),
    (
        (1.0 / 30.0, 0.001, "cadence"),
        (1.0 / 60.0, 0.010, "IMU/joint source skew"),
    ),
)
def test_conversion_rejects_bad_cadence_or_source_skew(
    period_s: float,
    imu_skew_s: float,
    match: str,
) -> None:
    with pytest.raises(ContractError, match=match):
        convert_sensor_v2_rosbag_records(
            _records(period_s=period_s, imu_skew_s=imu_skew_s),
            topic_types=REQUIRED_TOPIC_TYPES_V2,
            observation_contract=_VALIDATED_CONTRACT,
        )


def test_conversion_rejects_wrong_imu_frame_and_unknown_covariance() -> None:
    records = _records()
    for index, (topic, message, timestamp_ns) in enumerate(records):
        if topic == "/imu/data":
            records[index] = (
                topic,
                _imu_message(1_000.001, frame_id="wrong_imu"),
                timestamp_ns,
            )
            break
    with pytest.raises(ContractError, match="frame"):
        convert_sensor_v2_rosbag_records(
            records,
            topic_types=REQUIRED_TOPIC_TYPES_V2,
            observation_contract=_VALIDATED_CONTRACT,
        )


def test_causal_contract_accepts_explicitly_unavailable_orientation_only(
    tmp_path: Path,
) -> None:
    records = _records()
    for topic, message, _ in records:
        if topic == "/imu/data":
            message.orientation = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0)
            message.orientation_covariance = [-1.0] + [0.0] * 8

    arrays, details = convert_sensor_v2_rosbag_records(
        records,
        topic_types=REQUIRED_TOPIC_TYPES_V2,
        observation_contract=_CAUSAL_CONTRACT,
    )

    assert details["attitude_mode"] == "causal_gyro_accel"
    assert np.all(arrays["imu_orientation_covariance"][:, 0] == -1.0)
    bag = tmp_path / "causal-capture.db3"
    bag.write_bytes(b"structural causal rosbag fixture")
    attestation_path, attestation_sha256 = _capture_attestation(
        bag,
        _CAUSAL_CONTRACT,
    )
    trace_path = tmp_path / "causal-trace.npz"
    receipt_path = tmp_path / "causal-receipt.json"
    receipt = write_sensor_v2_import_artifacts(
        bag,
        trace_path,
        receipt_path,
        capture_attestation_path=attestation_path,
        capture_attestation_sha256=attestation_sha256,
        topic_types=REQUIRED_TOPIC_TYPES_V2,
        records=records,
        observation_contract=_CAUSAL_CONTRACT,
    )
    validated = validate_sensor_v2_import_receipt(
        receipt_path,
        expected_receipt_sha256=sha256_path_v2(receipt_path),
        expected_trace_path=trace_path,
    )
    assert receipt["attitude_mode"] == "causal_gyro_accel"
    assert validated.payload["observation_contract_sha256"] == _CAUSAL_CONTRACT.sha256
    with pytest.raises(ContractError, match="covariance is unknown"):
        convert_sensor_v2_rosbag_records(
            records,
            topic_types=REQUIRED_TOPIC_TYPES_V2,
            observation_contract=_VALIDATED_CONTRACT,
        )

    records = _records()
    for index, (topic, message, timestamp_ns) in enumerate(records):
        if topic == "/imu/data":
            records[index] = (
                topic,
                _imu_message(1_000.001, covariance=[0.0] * 9),
                timestamp_ns,
            )
            break
    with pytest.raises(ContractError, match="covariance is unknown"):
        convert_sensor_v2_rosbag_records(
            records,
            topic_types=REQUIRED_TOPIC_TYPES_V2,
            observation_contract=_VALIDATED_CONTRACT,
        )
