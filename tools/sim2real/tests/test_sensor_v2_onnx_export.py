from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
POLICY_IO_ROOT = REPO_ROOT / "source" / "redrhex_policy_io"
AGENTS_ROOT = (
    REPO_ROOT
    / "source"
    / "RedRhex"
    / "RedRhex"
    / "tasks"
    / "direct"
    / "redrhex"
    / "agents"
)
for root in (POLICY_IO_ROOT, AGENTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from redrhex_policy_io import (  # noqa: E402
    ContractError,
    ForwardResidualActionContractV2,
    SensorCalibrationProfileV2,
    StudentObservationContractV2,
)

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from sensor_v2.export import (  # noqa: E402
    BundleMetadataV2,
    BundleRecordsV2,
    export_runner_policy_bundle_v2,
    export_sensor_policy_onnx_v2,
    validate_sensor_policy_onnx_parity_v2,
)
from sensor_v2.checkpoint import (  # noqa: E402
    CheckpointManifestV2,
    architecture_hash_v2,
    canonical_hash_v2,
    file_sha256_v2,
)
from sensor_v2.models import SensorStudentCoreV2  # noqa: E402


def _metadata() -> BundleMetadataV2:
    values = iter("123456789")
    hashes = {name: next(values) * 64 for name in (
        "contract_sha256",
        "action_contract_sha256",
        "calibration_sha256",
        "training_calibration_sha256",
        "checkpoint_sha256",
        "feature_layout_sha256",
        "architecture_sha256",
        "config_sha256",
        "canonical_config_sha256",
    )}
    return BundleMetadataV2(
        **hashes,
        training_seed=42,
        checkpoint_kind="student_distilled_v2",
        stage="F2",
    )


def test_export_runs_random_and_recorded_torch_onnx_parity_gate(tmp_path: Path) -> None:
    torch.manual_seed(7)
    model = SensorStudentCoreV2().eval()
    path = tmp_path / "policy.onnx"
    sidecar = export_sensor_policy_onnx_v2(
        model,
        path,
        metadata=_metadata(),
        records=BundleRecordsV2(
            contract={"fixture": "contract"},
            action_contract={"fixture": "action"},
            calibration={"fixture": "calibration"},
            training_calibration={"fixture": "training-calibration"},
            checkpoint={"fixture": "checkpoint"},
            feature_layout={"fixture": "features"},
        ),
    )

    assert path.is_file()
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["torch_onnx_parity"]["status"] == "passed"
    assert payload["torch_onnx_parity"]["sample_count"] == 4

    recorded_history = np.linspace(-1.0, 1.0, 60 * 36, dtype=np.float32).reshape(1, 60, 36)
    recorded_command = np.asarray([[0.35, 0.0, 0.0]], dtype=np.float32)
    report = validate_sensor_policy_onnx_parity_v2(
        model,
        path,
        sensor_histories=recorded_history,
        commands=recorded_command,
        random_sample_count=1,
    )
    assert report.sample_count == 2
    assert report.action_max_abs_error <= report.absolute_tolerance
    assert report.velocity_max_abs_error <= report.absolute_tolerance


def _runtime_calibration(
    observation: StudentObservationContractV2,
    action: ForwardResidualActionContractV2,
    **overrides: object,
) -> SensorCalibrationProfileV2:
    values = {
        "profile_id": "measured-runtime",
        "observation_contract_sha256": observation.sha256,
        "action_contract_sha256": action.sha256,
        "attitude_mode": observation.attitude_mode,
        "imu_frame_id": observation.imu_frame_id,
        "imu_to_body_wxyz": observation.imu_to_body_wxyz,
        "main_counts_per_rad": (1000.0,) * 6,
        "abad_counts_per_rad": (1000.0,) * 6,
        "main_encoder_evidence": ("bench",) * 6,
        "abad_encoder_evidence": ("bench",) * 6,
        "imu_mount_evidence": "bench",
        "rest_gravity_evidence": "bench",
    }
    values.update(overrides)
    return SensorCalibrationProfileV2(**values).validate(
        require_hardware_ready=True
    )


def _production_export_fixture(tmp_path: Path) -> SimpleNamespace:
    model = SensorStudentCoreV2().eval()
    observation = StudentObservationContractV2.causal_gyro_accel()
    action = ForwardResidualActionContractV2()
    training = SensorCalibrationProfileV2.provisional(
        observation,
        action,
        profile_id="training-provisional",
    )
    runtime = _runtime_calibration(observation, action)
    feature_layout = {
        "features": [item.to_dict() for item in observation.FEATURE_LAYOUT]
    }
    manifest = CheckpointManifestV2(
        kind="student_distilled_v2",
        stage="distillation_f2",
        observation_contract_id=observation.contract_id,
        contract_hash=observation.sha256,
        action_contract_id=action.contract_id,
        action_contract_hash=action.sha256,
        calibration_hash=training.sha256,
        architecture_hash=architecture_hash_v2(model),
        config_hash=canonical_hash_v2({"fixture": "config"}),
        canonical_config_hash=canonical_hash_v2({"fixture": "canonical-config"}),
        training_seed=42,
        action_order=action.MAIN_JOINT_ORDER + action.ABAD_JOINT_ORDER,
        package_versions={"python": "test"},
    )
    records = BundleRecordsV2(
        contract=observation.to_dict(include_sha256=True),
        action_contract=action.to_dict(include_sha256=True),
        calibration=training.to_dict(include_sha256=True),
        training_calibration=training.to_dict(include_sha256=True),
        checkpoint=manifest.to_dict(),
        feature_layout=feature_layout,
        versions=manifest.package_versions,
    )
    checkpoint = tmp_path / "student.pt"
    checkpoint.write_bytes(b"checkpoint fixture")
    return SimpleNamespace(
        model=model,
        observation=observation,
        action=action,
        training=training,
        runtime=runtime,
        manifest=manifest,
        checkpoint=checkpoint,
        runner=SimpleNamespace(
            checkpoint_manifest=manifest,
            bundle_records=records,
            get_exportable_actor=lambda: model,
        ),
    )


def test_production_export_separates_training_and_runtime_calibration_lineage(
    tmp_path: Path,
) -> None:
    fixture = _production_export_fixture(tmp_path)
    recorded_history = np.zeros((1, 60, 36), dtype=np.float32)
    recorded_command = np.asarray([[0.3, 0.0, 0.0]], dtype=np.float32)

    onnx_path, sidecar_path = export_runner_policy_bundle_v2(
        fixture.runner,
        fixture.model,
        fixture.checkpoint,
        tmp_path / "export",
        parity_sensor_histories=recorded_history,
        parity_commands=recorded_command,
        parity_input_sha256="9" * 64,
        runtime_calibration=fixture.runtime,
    )

    assert onnx_path.is_file()
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    metadata = payload["metadata"]
    embedded = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    ).get_modelmeta().custom_metadata_map
    assert embedded == metadata
    assert metadata["calibration_sha256"] == fixture.runtime.sha256
    assert metadata["training_calibration_sha256"] == fixture.training.sha256
    assert metadata["checkpoint_sha256"] == file_sha256_v2(fixture.checkpoint)
    assert metadata["architecture_sha256"] == fixture.manifest.architecture_hash
    assert metadata["config_sha256"] == fixture.manifest.config_hash
    assert (
        metadata["canonical_config_sha256"]
        == fixture.manifest.canonical_config_hash
    )
    assert metadata["training_seed"] == str(fixture.manifest.training_seed)
    assert payload["calibration"]["sha256"] == fixture.runtime.sha256
    assert payload["calibration"]["hardware_ready"] is True
    assert payload["training_calibration"]["sha256"] == fixture.training.sha256
    assert payload["training_calibration"]["hardware_ready"] is False
    assert payload["checkpoint"]["calibration_hash"] == fixture.training.sha256


def test_production_export_rejects_runtime_sensor_frame_change(
    tmp_path: Path,
) -> None:
    fixture = _production_export_fixture(tmp_path)
    wrong_runtime = _runtime_calibration(
        fixture.observation,
        fixture.action,
        imu_to_body_wxyz=(0.0, 1.0, 0.0, 0.0),
    )

    with pytest.raises(ContractError, match="imu_to_body_wxyz"):
        export_runner_policy_bundle_v2(
            fixture.runner,
            fixture.model,
            fixture.checkpoint,
            tmp_path / "export",
            parity_sensor_histories=np.zeros((1, 60, 36), dtype=np.float32),
            parity_commands=np.zeros((1, 3), dtype=np.float32),
            parity_input_sha256="9" * 64,
            runtime_calibration=wrong_runtime,
        )
