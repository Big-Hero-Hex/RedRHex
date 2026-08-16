from __future__ import annotations

from dataclasses import dataclass
import json
from types import SimpleNamespace
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
POLICY_IO_ROOT = REPO_ROOT / "source" / "redrhex_policy_io"
for root in (POLICY_IO_ROOT, REPO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from redrhex_policy_io import (  # noqa: E402
    ContractError,
    ForwardResidualActionContractV2,
    SensorCalibrationProfileV2,
    StudentObservationContractV2,
)
from tools.sim2real import replay_student_observation_v2 as replay_module  # noqa: E402
from tools.sim2real.import_sensor_v2_rosbag import (  # noqa: E402
    CAPTURE_ATTESTATION_SCHEMA_V2,
    JOINT_ORDER_V2,
    MAX_IMU_JOINT_SKEW_S_V2,
    MAX_PERIOD_ERROR_RATIO_V2,
    REQUIRED_TOPIC_TYPES_V2,
    SAMPLE_RATE_HZ_V2,
    TIMESTAMP_SEMANTICS_V2,
    sha256_path_v2,
)


replay_arrays = replay_module.replay_arrays


def _trace(count: int = 61) -> dict[str, np.ndarray]:
    timestamps = np.arange(count, dtype=np.float64) / 60.0
    phase = timestamps[:, None] * np.arange(1.0, 7.0)[None, :]
    return {
        "timestamp_s": timestamps,
        "imu_gyro_rad_s": np.zeros((count, 3)),
        "imu_linear_accel_m_s2": np.tile([0.0, 0.0, 9.80665], (count, 1)),
        "main_position_rad": phase,
        "abad_position_rad": phase * 0.01,
        "command": np.asarray([0.3, 0.0, 0.0]),
    }


@dataclass
class _Runner:
    observation_contract: StudentObservationContractV2
    action_contract: ForwardResidualActionContractV2

    @property
    def metadata(self) -> dict[str, str]:
        return {
            "calibration_sha256": "a" * 64,
            "training_calibration_sha256": "b" * 64,
            "checkpoint_sha256": "c" * 64,
            "architecture_sha256": "d" * 64,
            "config_sha256": "e" * 64,
            "canonical_config_sha256": "f" * 64,
            "training_seed": "42",
        }

    @property
    def runtime_calibration_sha256(self) -> str:
        return self.metadata["calibration_sha256"]

    @property
    def training_calibration_sha256(self) -> str:
        return self.metadata["training_calibration_sha256"]

    @property
    def checkpoint_sha256(self) -> str:
        return self.metadata["checkpoint_sha256"]

    def run(self, sensor_history: np.ndarray, command: np.ndarray) -> object:
        assert sensor_history.shape == (60, 36)
        assert command.shape == (3,)
        return SimpleNamespace(
            actions=np.concatenate(
                (0.5 * np.tanh(sensor_history[-1, 18:24]), np.zeros(6))
            ),
            base_velocity_estimate=np.asarray([command[0], 0.0, 0.0]),
        )


@dataclass
class _SaturatedRunner(_Runner):
    def run(self, sensor_history: np.ndarray, command: np.ndarray) -> object:
        return SimpleNamespace(
            actions=np.concatenate((np.ones(6), np.zeros(6))),
            base_velocity_estimate=np.asarray([command[0], 0.0, 0.0]),
        )


def _ready_calibration(
    contract: StudentObservationContractV2,
    action_contract: ForwardResidualActionContractV2,
) -> SensorCalibrationProfileV2:
    return SensorCalibrationProfileV2(
        profile_id="test-hardware-ready",
        observation_contract_sha256=contract.sha256,
        action_contract_sha256=action_contract.sha256,
        attitude_mode=contract.attitude_mode,
        imu_frame_id=contract.imu_frame_id,
        main_counts_per_rad=(1000.0,) * 6,
        abad_counts_per_rad=(1000.0,) * 6,
        main_encoder_evidence=("fixture",) * 6,
        abad_encoder_evidence=("fixture",) * 6,
        imu_mount_evidence="fixture",
        rest_gravity_evidence="fixture",
    )


def test_replay_builds_only_real_full_histories_and_policy_outputs() -> None:
    contract = StudentObservationContractV2.causal_gyro_accel()
    action_contract = ForwardResidualActionContractV2()
    outputs, summary = replay_arrays(
        _trace(),
        contract=contract,
        action_contract=action_contract,
        runner=_Runner(contract, action_contract),
    )

    assert outputs["sensor_frames"].shape == (60, 36)
    assert outputs["sensor_histories"].shape == (1, 60, 36)
    np.testing.assert_array_equal(
        outputs["sensor_histories"][0], outputs["sensor_frames"]
    )
    assert outputs["sensor_frame_timestamp_s"][0] == pytest.approx(1.0 / 60.0)
    assert np.any(outputs["sensor_frames"][0, 18:24] != 0.0)
    assert outputs["actions"].shape == (1, 12)
    assert outputs["base_velocity_estimate"].shape == (1, 3)
    assert outputs[
        "raw_contract_target_main_drive_velocity_rad_s"
    ].shape == (1, 6)
    assert outputs["hardware_target_main_drive_velocity_rad_s"].shape == (1, 6)
    assert np.max(np.abs(outputs["actions"][:, 6:])) == 0.0
    assert summary["status"] == "passed"
    assert summary["velocity_baseline_samples"] == 1
    assert summary["sensor_frame_count"] == 60
    assert summary["history_ready_count"] == 1
    assert summary["policy"]["inference_count"] == 1
    assert summary["policy"]["main_action_saturation_gate_passed"] is True
    assert summary["calibration_sha256"] == "a" * 64
    assert summary["runtime_calibration_sha256"] == "a" * 64
    assert summary["training_calibration_sha256"] == "b" * 64
    assert summary["checkpoint_sha256"] == "c" * 64
    assert summary["architecture_sha256"] == "d" * 64
    assert summary["config_sha256"] == "e" * 64
    assert summary["canonical_config_sha256"] == "f" * 64
    assert summary["training_seed"] == 42
    tightening = summary["hardware_target_tightening"]
    assert tightening["deployment_main_velocity_limit_rad_s"] == 9.0
    assert tightening["bundle_main_velocity_limit_rad_s"] == 15.0
    assert tightening["deployment_config"]["sha256"] == sha256_path_v2(
        tightening["deployment_config"]["path"]
    )
    assert tightening["max_total_tightening_fraction"] == 0.0
    assert tightening["required_for_real_replay"] is False


def test_replay_fails_high_main_action_saturation() -> None:
    contract = StudentObservationContractV2.causal_gyro_accel()
    action_contract = ForwardResidualActionContractV2()
    _, summary = replay_arrays(
        _trace(),
        contract=contract,
        action_contract=action_contract,
        runner=_SaturatedRunner(contract, action_contract),
        max_main_action_saturation_fraction=0.05,
    )

    assert summary["status"] == "failed"
    assert summary["policy"]["main_action_saturation_fraction"] == 1.0
    assert summary["policy"]["main_action_saturation_gate_passed"] is False
    assert "saturation fraction" in summary["failure_reasons"][0]


def test_replay_rejects_invalid_saturation_limit() -> None:
    with pytest.raises(ContractError, match="saturation_fraction"):
        replay_arrays(
            _trace(),
            contract=StudentObservationContractV2.causal_gyro_accel(),
            max_main_action_saturation_fraction=1.1,
        )


def test_real_replay_fails_any_deployment_target_divergence() -> None:
    contract = StudentObservationContractV2.causal_gyro_accel()
    action_contract = ForwardResidualActionContractV2()
    calibration = _ready_calibration(contract, action_contract)

    @dataclass
    class HardwareBoundSaturatedRunner(_SaturatedRunner):
        @property
        def metadata(self) -> dict[str, str]:
            return {
                **super().metadata,
                "calibration_sha256": calibration.sha256,
            }

    outputs, summary = replay_arrays(
        _trace(121),
        contract=contract,
        action_contract=action_contract,
        calibration=calibration,
        runner=HardwareBoundSaturatedRunner(contract, action_contract),
        trace_kind="real",
    )

    tightening = summary["hardware_target_tightening"]
    assert summary["status"] == "failed"
    assert tightening["gate_passed"] is False
    assert tightening["total"]["tightening_fraction"] > 0.0
    assert tightening["slew_rate"]["tightened_target_count"] > 0
    assert tightening["velocity_limit"]["tightened_target_count"] > 0
    assert np.max(
        np.abs(
            outputs["raw_contract_target_main_drive_velocity_rad_s"]
            - outputs["hardware_target_main_drive_velocity_rad_s"]
        )
    ) == pytest.approx(tightening["total"]["max_abs_delta_rad_s"])
    assert "hardware target differs" in summary["failure_reasons"][-1]


def test_replay_reports_ranked_feature_domain_shift() -> None:
    contract = StudentObservationContractV2.causal_gyro_accel()
    baseline, _ = replay_arrays(_trace(), contract=contract)
    shifted_trace = _trace()
    shifted_trace["imu_gyro_rad_s"][:, 0] = 1.0
    _, summary = replay_arrays(
        shifted_trace,
        contract=contract,
        reference_frames=baseline["sensor_frames"],
    )

    assert summary["domain_shift_ranked"][0]["feature"] == "body_gyro"
    assert summary["domain_shift_ranked"][0]["max_standardized_mean_shift"] > 1.0


def test_real_replay_requires_all_hardware_calibration_evidence() -> None:
    contract = StudentObservationContractV2.causal_gyro_accel()
    action_contract = ForwardResidualActionContractV2()
    provisional = SensorCalibrationProfileV2.provisional(contract, action_contract)
    with pytest.raises(ContractError, match="hardware calibration is incomplete"):
        replay_arrays(
            _trace(),
            contract=contract,
            action_contract=action_contract,
            calibration=provisional,
            trace_kind="real",
        )

    outputs, summary = replay_arrays(
        _trace(),
        contract=contract,
        action_contract=action_contract,
        calibration=_ready_calibration(contract, action_contract),
        trace_kind="real",
    )
    assert outputs["sensor_histories"].shape[0] == 1
    assert summary["trace_kind"] == "real"
    assert summary["runtime_calibration_sha256"] == summary["calibration_sha256"]
    assert summary["training_calibration_sha256"] is None


def test_replay_rejects_runner_runtime_calibration_provenance_mismatch() -> None:
    contract = StudentObservationContractV2.causal_gyro_accel()
    action_contract = ForwardResidualActionContractV2()
    with pytest.raises(ContractError, match="runtime calibration does not match"):
        replay_arrays(
            _trace(),
            contract=contract,
            action_contract=action_contract,
            calibration=_ready_calibration(contract, action_contract),
            runner=_Runner(contract, action_contract),
            trace_kind="real",
        )


def test_sim_validated_quaternion_replay_uses_simulator_rest_evidence() -> None:
    contract = StudentObservationContractV2.validated_quaternion()
    trace = _trace()
    trace.pop("imu_linear_accel_m_s2")
    trace["imu_orientation_xyzw"] = np.tile([0.0, 0.0, 0.0, 1.0], (61, 1))
    trace["imu_orientation_covariance"] = np.tile(
        np.eye(3, dtype=np.float64).reshape(9) * 1.0e-8,
        (61, 1),
    )
    trace["imu_frame_id"] = np.asarray(contract.imu_frame_id)

    outputs, summary = replay_arrays(trace, contract=contract, trace_kind="sim")

    assert outputs["sensor_histories"].shape == (1, 60, 36)
    assert summary["status"] == "passed"


@pytest.mark.parametrize("fault", ("nonmonotonic", "wrong_rate", "nan"))
def test_replay_fails_closed_on_bad_sensor_trace(fault: str) -> None:
    trace = _trace()
    if fault == "nonmonotonic":
        trace["timestamp_s"][20] = trace["timestamp_s"][19]
        match = "strictly increasing"
    elif fault == "wrong_rate":
        trace["timestamp_s"][20:] += 0.02
        match = "sample cadence"
    else:
        trace["main_position_rad"][20, 3] = np.nan
        match = "NaN or Inf"
    with pytest.raises(ContractError, match=match):
        replay_arrays(
            trace,
            contract=StudentObservationContractV2.causal_gyro_accel(),
        )


def test_validated_quaternion_replay_rejects_unknown_covariance() -> None:
    contract = StudentObservationContractV2.validated_quaternion()
    trace = _trace()
    trace.pop("imu_linear_accel_m_s2")
    trace["imu_orientation_xyzw"] = np.tile([0.0, 0.0, 0.0, 1.0], (61, 1))
    trace["imu_orientation_covariance"] = np.zeros(9)
    trace["imu_frame_id"] = np.asarray(contract.imu_frame_id)
    action_contract = ForwardResidualActionContractV2()
    calibration = _ready_calibration(contract, action_contract)
    with pytest.raises(ContractError, match="covariance is unknown"):
        replay_arrays(
            trace,
            contract=contract,
            action_contract=action_contract,
            calibration=calibration,
            trace_kind="real",
        )


def _write_import_receipt(
    tmp_path: Path,
    trace_path: Path,
    *,
    imu_frame_id: str,
    runtime_calibration_sha256: str,
) -> tuple[Path, str, Path, Path]:
    source_bag = tmp_path / "capture.db3"
    source_bag.write_bytes(b"mock rosbag; this fixture is not hardware evidence")
    attestation_path = tmp_path / "capture-attestation.json"
    attestation = {
        "schema": CAPTURE_ATTESTATION_SCHEMA_V2,
        "source_recorder_id": "ros2bag/redrhex-v2-recorder",
        "operator_id": "unit-test-structural-fixture",
        "capture_declaration": "physical_hardware",
        "attested_at_utc": "2026-08-16T00:00:00Z",
        "observation_contract_sha256": (
            StudentObservationContractV2.causal_gyro_accel().sha256
        ),
        "attitude_mode": "causal_gyro_accel",
        "runtime_calibration_sha256": runtime_calibration_sha256,
        "source_bag_sha256": sha256_path_v2(source_bag),
        "source_bag_hash_kind": "sha256-file-v1",
        "topics": dict(REQUIRED_TOPIC_TYPES_V2),
    }
    attestation_path.write_text(
        json.dumps(attestation, sort_keys=True),
        encoding="utf-8",
    )
    receipt_path = tmp_path / "capture.receipt.json"
    receipt = {
        "schema": "redrhex.sensor-v2-rosbag-import.v1",
        "source_bag": {
            "path": str(source_bag.resolve()),
            "sha256": sha256_path_v2(source_bag),
            "hash_kind": "sha256-file-v1",
        },
        "output_trace": {
            "path": str(trace_path.resolve()),
            "sha256": sha256_path_v2(trace_path),
        },
        "topics": {
            topic: {
                "type": message_type,
                "message_count": 61 if topic != "/cmd_vel" else 1,
            }
            for topic, message_type in REQUIRED_TOPIC_TYPES_V2.items()
        },
        "joint_order": list(JOINT_ORDER_V2),
        "imu_frame_id": imu_frame_id,
        "observation_contract_sha256": attestation[
            "observation_contract_sha256"
        ],
        "attitude_mode": attestation["attitude_mode"],
        "sample_rate_hz": SAMPLE_RATE_HZ_V2,
        "sample_count": 61,
        "max_period_error_ratio": MAX_PERIOD_ERROR_RATIO_V2,
        "max_imu_joint_skew_s": MAX_IMU_JOINT_SKEW_S_V2,
        "observed_max_imu_joint_skew_s": 0.001,
        "timestamp_semantics": dict(TIMESTAMP_SEMANTICS_V2),
        "capture_attestation": {
            "path": str(attestation_path.resolve()),
            "sha256": sha256_path_v2(attestation_path),
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
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    return receipt_path, sha256_path_v2(receipt_path), source_bag, attestation_path


def test_real_cli_requires_receipt_and_binds_every_replay_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = StudentObservationContractV2.causal_gyro_accel()
    action_contract = ForwardResidualActionContractV2()
    calibration = _ready_calibration(contract, action_contract)
    trace = _trace()
    trace["timestamp_s"] = trace["timestamp_s"] + 1_000.0
    trace["imu_source_timestamp_s"] = trace["timestamp_s"] + 0.001
    trace["joint_validity_timestamp_s"] = trace["timestamp_s"].copy()
    trace["imu_frame_id"] = np.asarray(contract.imu_frame_id)
    trace["imu_orientation_xyzw"] = np.tile([0.0, 0.0, 0.0, 1.0], (61, 1))
    trace["imu_orientation_covariance"] = np.tile(
        np.eye(3, dtype=np.float64).reshape(9) * 1.0e-6,
        (61, 1),
    )
    trace["command"] = np.tile(np.asarray([0.3, 0.0, 0.0]), (61, 1))
    trace_path = tmp_path / "canonical.npz"
    np.savez_compressed(trace_path, **trace)
    receipt_path, receipt_sha256, source_bag, attestation_path = _write_import_receipt(
        tmp_path,
        trace_path,
        imu_frame_id=contract.imu_frame_id,
        runtime_calibration_sha256=calibration.sha256,
    )
    onnx_path = tmp_path / "policy.onnx"
    sidecar_path = tmp_path / "policy.onnx.json"
    onnx_path.write_bytes(b"mock ONNX; runner is dependency-light fixture")
    sidecar_path.write_text("{}", encoding="utf-8")

    class BoundRunner(_Runner):
        @property
        def metadata(self) -> dict[str, str]:
            return {
                "calibration_sha256": calibration.sha256,
                "training_calibration_sha256": "b" * 64,
                "checkpoint_sha256": "c" * 64,
                "architecture_sha256": "d" * 64,
                "config_sha256": "e" * 64,
            }

    monkeypatch.setattr(
        replay_module,
        "_load_bundle",
        lambda *_args, **_kwargs: (
            BoundRunner(contract, action_contract),
            contract,
            action_contract,
            calibration,
        ),
    )
    output_npz = tmp_path / "replayed.npz"
    output_json = tmp_path / "replayed.json"
    result = replay_module.main(
        [
            str(trace_path),
            "--onnx",
            str(onnx_path),
            "--sidecar",
            str(sidecar_path),
            "--trace-kind",
            "real",
            "--import-receipt",
            str(receipt_path),
            "--import-receipt-sha256",
            receipt_sha256,
            "--output-npz",
            str(output_npz),
            "--output-json",
            str(output_json),
        ]
    )

    assert result == 0
    summary = json.loads(output_json.read_text(encoding="utf-8"))
    sources = summary["source_artifacts"]
    assert set(sources) == {
        "source_bag",
        "import_receipt",
        "capture_attestation",
        "input_trace",
        "onnx",
        "sidecar",
        "hardware_config",
        "output_npz",
    }
    for record in sources.values():
        assert sha256_path_v2(record["path"]) == record["sha256"]
    assert sources["source_bag"]["path"] == str(source_bag.resolve())
    assert sources["capture_attestation"]["path"] == str(attestation_path.resolve())
    assert sources["input_trace"]["sha256"] == sha256_path_v2(trace_path)
    assert sources["output_npz"]["sha256"] == sha256_path_v2(output_npz)

    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation["runtime_calibration_sha256"] = "f" * 64
    attestation_path.write_text(
        json.dumps(attestation, sort_keys=True),
        encoding="utf-8",
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["capture_attestation"]["sha256"] = sha256_path_v2(attestation_path)
    receipt["capture_attestation"]["runtime_calibration_sha256"] = "f" * 64
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    with pytest.raises(ContractError, match="attestation runtime calibration"):
        replay_module.main(
            [
                str(trace_path),
                "--onnx",
                str(onnx_path),
                "--sidecar",
                str(sidecar_path),
                "--trace-kind",
                "real",
                "--import-receipt",
                str(receipt_path),
                "--import-receipt-sha256",
                sha256_path_v2(receipt_path),
                "--output-npz",
                str(output_npz),
                "--output-json",
                str(output_json),
            ]
        )


def test_real_cli_rejects_unreceipted_relabel(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="requires --import-receipt"):
        replay_module.main(
            [
                str(tmp_path / "arbitrary.npz"),
                "--onnx",
                str(tmp_path / "policy.onnx"),
                "--trace-kind",
                "real",
                "--output-npz",
                str(tmp_path / "output.npz"),
                "--output-json",
                str(tmp_path / "output.json"),
            ]
        )
