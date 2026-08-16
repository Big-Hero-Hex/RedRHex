from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from redrhex_policy_io import (
    ForwardResidualActionContractV2,
    SensorCalibrationProfileV2,
    StudentObservationContractV2,
    canonical_sha256,
)
from tools.sim2real.generate_sensor_v2_promotion_gates import (
    PromotionGateGenerationError,
    generate_promotion_gates,
)
from sensor_v2.checkpoint import (
    CheckpointManifestV2,
    architecture_hash_v2,
    save_checkpoint_v2,
)
from sensor_v2.export import (
    BundleMetadataV2,
    BundleRecordsV2,
    export_sensor_policy_onnx_v2,
)
from sensor_v2.models import SensorStudentCoreV2
from sensor_v2.ppo import SensorActorCriticV2


PARITY_INPUT_NAME = "recorded_parity_input.npz"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ready_calibration(
    observation: StudentObservationContractV2,
    action: ForwardResidualActionContractV2,
) -> SensorCalibrationProfileV2:
    return SensorCalibrationProfileV2(
        profile_id="measured-hardware-v2",
        observation_contract_sha256=observation.sha256,
        action_contract_sha256=action.sha256,
        attitude_mode=observation.attitude_mode,
        imu_frame_id=observation.imu_frame_id,
        imu_to_body_wxyz=observation.imu_to_body_wxyz,
        main_counts_per_rad=(1000.0,) * 6,
        abad_counts_per_rad=(1000.0,) * 6,
        main_encoder_evidence=("measured-main",) * 6,
        abad_encoder_evidence=("measured-abad",) * 6,
        imu_mount_evidence="measured-mount",
        rest_gravity_evidence="measured-rest-gravity",
    )


def _bundle(
    tmp_path: Path,
    *,
    runtime_ready: bool = True,
    stage: str = "ppo_f4",
    metadata_stage: str | None = None,
) -> tuple[Path, Path, Path]:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    tmp_path.mkdir(parents=True, exist_ok=True)

    observation = StudentObservationContractV2.validated_quaternion()
    action = ForwardResidualActionContractV2()
    training = SensorCalibrationProfileV2.provisional(observation, action)
    runtime = _ready_calibration(observation, action) if runtime_ready else training
    torch.manual_seed(11)
    actor = SensorStudentCoreV2().eval()
    policy = SensorActorCriticV2(
        critic_observation_dim=7,
        student=actor,
    ).eval()
    architecture_sha256 = architecture_hash_v2(policy)
    config_sha256 = "f" * 64
    canonical_config_sha256 = "e" * 64
    manifest = CheckpointManifestV2(
        kind="student_ppo_v2",
        stage=stage,
        observation_contract_id=observation.contract_id,
        contract_hash=observation.sha256,
        action_contract_id=action.contract_id,
        action_contract_hash=action.sha256,
        calibration_hash=training.sha256,
        architecture_hash=architecture_sha256,
        config_hash=config_sha256,
        canonical_config_hash=canonical_config_sha256,
        training_seed=42,
        action_order=action.MAIN_JOINT_ORDER + action.ABAD_JOINT_ORDER,
    )
    checkpoint_path = tmp_path / "model_1.pt"
    save_checkpoint_v2(
        checkpoint_path,
        manifest=manifest,
        model=policy,
        update=0,
    )
    parity_input_path = tmp_path / PARITY_INPUT_NAME
    histories = np.linspace(
        -1.0,
        1.0,
        2 * 60 * 36,
        dtype=np.float32,
    ).reshape(2, 60, 36)
    commands = np.asarray([[0.25, 0.0, 0.0], [0.4, 0.0, 0.0]], dtype=np.float32)
    np.savez_compressed(
        parity_input_path,
        sensor_histories=histories,
        command=commands,
    )
    feature_layout = {"features": observation.to_dict()["feature_layout"]}
    metadata = BundleMetadataV2(
        contract_sha256=observation.sha256,
        action_contract_sha256=action.sha256,
        calibration_sha256=runtime.sha256,
        training_calibration_sha256=training.sha256,
        checkpoint_sha256=_sha256(checkpoint_path),
        feature_layout_sha256=canonical_sha256(feature_layout),
        architecture_sha256=architecture_sha256,
        config_sha256=config_sha256,
        canonical_config_sha256=canonical_config_sha256,
        training_seed=42,
        checkpoint_kind="student_ppo_v2",
        stage=stage if metadata_stage is None else metadata_stage,
    )
    onnx_path = tmp_path / "policy_sensor_v2.onnx"
    sidecar_path = export_sensor_policy_onnx_v2(
        actor,
        onnx_path,
        metadata=metadata,
        records=BundleRecordsV2(
            contract=observation.to_dict(include_sha256=True),
            action_contract=action.to_dict(include_sha256=True),
            calibration=runtime.to_dict(include_sha256=True),
            training_calibration=training.to_dict(include_sha256=True),
            checkpoint=manifest.to_dict(),
            feature_layout=feature_layout,
        ),
        parity_sensor_histories=histories,
        parity_commands=commands,
        parity_input_sha256=_sha256(parity_input_path),
    )
    return onnx_path, sidecar_path, checkpoint_path


def test_generator_emits_canonical_hash_bound_gate_artifacts(tmp_path: Path) -> None:
    onnx_path, sidecar_path, checkpoint_path = _bundle(tmp_path)
    parity_input_path = tmp_path / PARITY_INPUT_NAME

    result = generate_promotion_gates(
        onnx_path=onnx_path,
        sidecar_path=sidecar_path,
        checkpoint_path=checkpoint_path,
        parity_input_path=parity_input_path,
        parity_input_sha256=_sha256(parity_input_path),
        output_dir=tmp_path / "gates",
    )

    assert set(result) == {
        "no_privileged_leak",
        "torch_onnx_parity",
        "contract_provenance",
    }
    for gate, record in result.items():
        path = Path(record["path"])
        assert _sha256(path) == record["sha256"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["gate"] == gate
        assert payload["status"] == "PASS"
        assert payload["provenance"]["checkpoint_sha256"] == _sha256(checkpoint_path)
        assert payload["provenance"]["checkpoint_stage"] == "ppo_f4"
    provenance = json.loads(
        Path(result["contract_provenance"]["path"]).read_text(encoding="utf-8")
    )
    assert {
        check["name"] for check in provenance["checks"]
    } >= {"runtime_calibration_lineage", "checkpoint_manifest_binding"}
    assert (
        provenance["provenance"]["canonical_config_sha256"]
        == "e" * 64
    )
    assert provenance["provenance"]["training_seed"] == 42
    parity = json.loads(
        Path(result["torch_onnx_parity"]["path"]).read_text(encoding="utf-8")
    )
    assert parity["schema"] == "redrhex.torch-onnx-parity-gate.v2"
    assert parity["torch_onnx_parity"]["random_sample_count"] == 4
    assert parity["torch_onnx_parity"]["recorded_sample_count"] == 2
    assert parity["recorded_parity_input"]["sha256"] == _sha256(parity_input_path)
    assert set(parity["source_artifacts"]) == {
        "onnx",
        "sidecar",
        "checkpoint",
        "parity_input",
    }


def test_generator_rejects_provisional_runtime_or_changed_checkpoint(tmp_path: Path) -> None:
    onnx_path, sidecar_path, checkpoint_path = _bundle(tmp_path, runtime_ready=False)
    parity_input_path = tmp_path / PARITY_INPUT_NAME
    with pytest.raises(PromotionGateGenerationError, match="hardware calibration is incomplete"):
        generate_promotion_gates(
            onnx_path=onnx_path,
            sidecar_path=sidecar_path,
            checkpoint_path=checkpoint_path,
            parity_input_path=parity_input_path,
            parity_input_sha256=_sha256(parity_input_path),
            output_dir=tmp_path / "gates-provisional",
        )

    onnx_path, sidecar_path, checkpoint_path = _bundle(tmp_path / "changed")
    parity_input_path = checkpoint_path.parent / PARITY_INPUT_NAME
    checkpoint_path.write_bytes(b"changed-after-export")
    with pytest.raises(PromotionGateGenerationError, match="metadata hash bindings disagree"):
        generate_promotion_gates(
            onnx_path=onnx_path,
            sidecar_path=sidecar_path,
            checkpoint_path=checkpoint_path,
            parity_input_path=parity_input_path,
            parity_input_sha256=_sha256(parity_input_path),
            output_dir=tmp_path / "gates-changed",
        )


@pytest.mark.parametrize("stage", ("ppo_f3", "robustness_f4"))
def test_generator_rejects_noncanonical_checkpoint_stage(
    tmp_path: Path,
    stage: str,
) -> None:
    onnx_path, sidecar_path, checkpoint_path = _bundle(tmp_path, stage=stage)
    parity_input_path = tmp_path / PARITY_INPUT_NAME

    with pytest.raises(PromotionGateGenerationError, match="stage 'ppo_f4'"):
        generate_promotion_gates(
            onnx_path=onnx_path,
            sidecar_path=sidecar_path,
            checkpoint_path=checkpoint_path,
            parity_input_path=parity_input_path,
            parity_input_sha256=_sha256(parity_input_path),
            output_dir=tmp_path / "gates-wrong-stage",
        )


def test_generator_rejects_f3_checkpoint_behind_f4_metadata(tmp_path: Path) -> None:
    onnx_path, sidecar_path, checkpoint_path = _bundle(
        tmp_path,
        stage="ppo_f3",
        metadata_stage="ppo_f4",
    )
    parity_input_path = tmp_path / PARITY_INPUT_NAME

    with pytest.raises(
        PromotionGateGenerationError,
        match="checkpoint manifest stage 'ppo_f4'",
    ):
        generate_promotion_gates(
            onnx_path=onnx_path,
            sidecar_path=sidecar_path,
            checkpoint_path=checkpoint_path,
            parity_input_path=parity_input_path,
            parity_input_sha256=_sha256(parity_input_path),
            output_dir=tmp_path / "gates-f3-checkpoint",
        )


def test_generator_rejects_changed_or_misdeclared_parity_input(tmp_path: Path) -> None:
    onnx_path, sidecar_path, checkpoint_path = _bundle(tmp_path)
    parity_input_path = tmp_path / PARITY_INPUT_NAME
    original_sha256 = _sha256(parity_input_path)
    parity_input_path.write_bytes(parity_input_path.read_bytes() + b"changed")

    with pytest.raises(PromotionGateGenerationError, match="sha256 mismatch"):
        generate_promotion_gates(
            onnx_path=onnx_path,
            sidecar_path=sidecar_path,
            checkpoint_path=checkpoint_path,
            parity_input_path=parity_input_path,
            parity_input_sha256=original_sha256,
            output_dir=tmp_path / "gates-changed-input",
        )

    with pytest.raises(PromotionGateGenerationError, match="sha256 mismatch"):
        generate_promotion_gates(
            onnx_path=onnx_path,
            sidecar_path=sidecar_path,
            checkpoint_path=checkpoint_path,
            parity_input_path=parity_input_path,
            parity_input_sha256="0" * 64,
            output_dir=tmp_path / "gates-wrong-input-hash",
        )


def test_generator_reruns_checkpoint_actor_against_onnx(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    onnx_path, sidecar_path, checkpoint_path = _bundle(tmp_path)
    parity_input_path = tmp_path / PARITY_INPUT_NAME

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["model_state_dict"]["actor.actor_head.4.bias"] += 0.5
    torch.save(checkpoint, checkpoint_path)
    changed_checkpoint_sha256 = _sha256(checkpoint_path)

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["metadata"]["checkpoint_sha256"] = changed_checkpoint_sha256
    sidecar_path.write_text(json.dumps(sidecar, sort_keys=True), encoding="utf-8")
    graph = onnx.load(onnx_path)
    for item in graph.metadata_props:
        if item.key == "checkpoint_sha256":
            item.value = changed_checkpoint_sha256
            break
    else:  # pragma: no cover - _bundle always writes this mandatory key
        raise AssertionError("checkpoint_sha256 metadata is missing")
    onnx.save(graph, onnx_path)

    with pytest.raises(
        PromotionGateGenerationError,
        match="canonical Sensor V2 Torch/ONNX parity failed",
    ):
        generate_promotion_gates(
            onnx_path=onnx_path,
            sidecar_path=sidecar_path,
            checkpoint_path=checkpoint_path,
            parity_input_path=parity_input_path,
            parity_input_sha256=_sha256(parity_input_path),
            output_dir=tmp_path / "gates-rebound-checkpoint",
        )
