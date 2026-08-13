"""Fixed-shape, two-input ONNX bundle exporter for Sensor V2."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .checkpoint import CHECKPOINT_FORMAT_V2, CheckpointManifestV2, canonical_hash_v2, file_sha256_v2
from .models import COMMAND_DIM_V2, SENSOR_FRAME_DIM_V2, SENSOR_HISTORY_LENGTH_V2, SensorStudentCoreV2


BUNDLE_SCHEMA_V2 = "redrhex.sensor-policy-bundle.v2"
CONTRACT_ID_V2 = "redrhex.student-observation.v2"
ONNX_INPUT_SHAPES_V2 = {
    "sensor_history": [1, SENSOR_HISTORY_LENGTH_V2, SENSOR_FRAME_DIM_V2],
    "command": [1, COMMAND_DIM_V2],
}
ONNX_OUTPUT_SHAPES_V2 = {"actions": [1, 12], "base_velocity_estimate": [1, 3]}
REQUIRED_METADATA_KEYS_V2 = (
    "bundle_schema",
    "bundle_version",
    "contract_id",
    "contract_sha256",
    "action_contract_sha256",
    "calibration_sha256",
    "checkpoint_sha256",
    "feature_layout_sha256",
    "contact_supervision",
)


def _sha256(name: str, value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


@dataclass(frozen=True)
class BundleMetadataV2:
    contract_sha256: str
    action_contract_sha256: str
    calibration_sha256: str
    checkpoint_sha256: str
    feature_layout_sha256: str
    checkpoint_kind: str
    stage: str
    architecture_sha256: str
    config_sha256: str
    bundle_schema: str = BUNDLE_SCHEMA_V2
    bundle_version: str = "2"
    contract_id: str = CONTRACT_ID_V2
    contact_supervision: str = "disabled"

    def __post_init__(self) -> None:
        if self.bundle_schema != BUNDLE_SCHEMA_V2 or self.bundle_version != "2":
            raise ValueError("unsupported Sensor V2 bundle format or version")
        if self.contract_id != CONTRACT_ID_V2:
            raise ValueError(f"Sensor V2 contract_id must be {CONTRACT_ID_V2!r}")
        if self.contact_supervision != "disabled":
            raise ValueError("contact supervision must remain disabled")
        for name in (
            "contract_sha256",
            "action_contract_sha256",
            "calibration_sha256",
            "checkpoint_sha256",
            "architecture_sha256",
            "config_sha256",
            "feature_layout_sha256",
        ):
            _sha256(name, getattr(self, name))
        if self.checkpoint_kind not in ("student_distilled_v2", "student_ppo_v2"):
            raise ValueError("deployable checkpoint_kind must be student_distilled_v2 or student_ppo_v2")
        if not self.stage:
            raise ValueError("bundle stage must not be empty")

    def embedded(self) -> dict[str, str]:
        values = asdict(self)
        # ROS requires the fixed subset above; architecture/config/kind/stage
        # stay embedded too so lineage is inspectable without the sidecar.
        keys = REQUIRED_METADATA_KEYS_V2 + (
            "architecture_sha256",
            "config_sha256",
            "checkpoint_kind",
            "stage",
        )
        return {key: str(values[key]) for key in keys}


@dataclass(frozen=True)
class BundleRecordsV2:
    """Verbatim records whose hashes are carried in the metadata."""

    contract: dict[str, Any] | None = None
    action_contract: dict[str, Any] | None = None
    calibration: dict[str, Any] | None = None
    checkpoint: dict[str, Any] | None = None
    feature_layout: dict[str, Any] | None = None
    versions: dict[str, str] = field(default_factory=dict)

    def require_deployable(self) -> None:
        missing = [
            name
            for name in ("contract", "action_contract", "calibration", "checkpoint", "feature_layout")
            if getattr(self, name) is None
        ]
        if missing:
            raise ValueError(f"deployable Sensor V2 bundle is missing records: {missing}")


class _DeploymentGraphV2(nn.Module):
    def __init__(self, model: SensorStudentCoreV2) -> None:
        super().__init__()
        self.model = model

    def forward(self, sensor_history: torch.Tensor, command: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        actions, base_velocity_estimate, _ = self.model(sensor_history, command)
        return actions, base_velocity_estimate


def export_sensor_policy_onnx_v2(
    model: SensorStudentCoreV2,
    output_path: str | Path,
    *,
    metadata: BundleMetadataV2,
    records: BundleRecordsV2 | None = None,
    sidecar_path: str | Path | None = None,
    opset_version: int = 18,
) -> Path:
    """Export fixed-batch ONNX and a matching, hash-bearing JSON sidecar."""

    output_path = Path(output_path)
    sidecar_path = (
        Path(sidecar_path)
        if sidecar_path is not None
        else output_path.with_suffix(output_path.suffix + ".json")
    )
    graph = _DeploymentGraphV2(model).eval()
    first_parameter = next(model.parameters())
    history = torch.zeros(
        ONNX_INPUT_SHAPES_V2["sensor_history"],
        device=first_parameter.device,
        dtype=first_parameter.dtype,
    )
    command = torch.zeros(
        ONNX_INPUT_SHAPES_V2["command"],
        device=first_parameter.device,
        dtype=first_parameter.dtype,
    )
    torch.onnx.export(
        graph,
        (history, command),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=list(ONNX_INPUT_SHAPES_V2),
        output_names=list(ONNX_OUTPUT_SHAPES_V2),
        dynamic_axes=None,
        dynamo=False,
    )

    try:
        import onnx
    except ImportError as exc:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("onnx is required to finalize Sensor V2 metadata") from exc
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    del onnx_model.metadata_props[:]
    for key, value in metadata.embedded().items():
        item = onnx_model.metadata_props.add()
        item.key = key
        item.value = value
    onnx.save(onnx_model, output_path)

    records = records or BundleRecordsV2()
    records.require_deployable()
    sidecar = {
        "metadata": metadata.embedded(),
        "io": {
            "inputs": ONNX_INPUT_SHAPES_V2,
            "outputs": ONNX_OUTPUT_SHAPES_V2,
        },
        "checkpoint_format": CHECKPOINT_FORMAT_V2,
        "contract": records.contract,
        "action_contract": records.action_contract,
        "calibration": records.calibration,
        "checkpoint": records.checkpoint,
        "feature_layout": records.feature_layout,
        "versions": records.versions,
    }
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return sidecar_path


def export_runner_policy_bundle_v2(
    runner: Any,
    policy_nn: nn.Module,
    resume_path: str | Path,
    export_model_dir: str | Path,
) -> tuple[Path, Path]:
    """Export a loaded V2 runner through the fixed deployment contract."""

    actor = runner.get_exportable_actor() if hasattr(runner, "get_exportable_actor") else policy_nn
    if not isinstance(actor, SensorStudentCoreV2):
        candidate = getattr(actor, "actor", None) or getattr(actor, "student", None)
        if not isinstance(candidate, SensorStudentCoreV2):
            raise TypeError("V2 export requires a SensorStudentCoreV2 actor")
        actor = candidate
    manifest = getattr(runner, "checkpoint_manifest", None)
    if not isinstance(manifest, CheckpointManifestV2):
        raise ValueError("V2 runner has no loaded CheckpointManifestV2")
    resume_path = Path(resume_path)
    output_directory = Path(export_model_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "policy_sensor_v2.onnx"
    sidecar_path = output_path.with_suffix(output_path.suffix + ".json")
    bundle_records = getattr(runner, "bundle_records", None)
    if isinstance(bundle_records, BundleRecordsV2):
        records = bundle_records
    elif isinstance(bundle_records, dict):
        records = BundleRecordsV2(**bundle_records)
    else:
        raise ValueError("V2 runner has no complete bundle_records for deployment export")
    records.require_deployable()
    assert records.contract is not None
    assert records.action_contract is not None
    assert records.calibration is not None
    assert records.feature_layout is not None
    contract_record = dict(records.contract)
    action_record = dict(records.action_contract)
    calibration_record = dict(records.calibration)
    contract_record.pop("sha256", None)
    action_record.pop("sha256", None)
    calibration_record.pop("sha256", None)
    if canonical_hash_v2(contract_record) != manifest.contract_hash:
        raise ValueError("runner observation-contract record disagrees with checkpoint manifest")
    if canonical_hash_v2(action_record) != manifest.action_contract_hash:
        raise ValueError("runner action-contract record disagrees with checkpoint manifest")
    if canonical_hash_v2(calibration_record) != manifest.calibration_hash:
        raise ValueError("runner calibration record disagrees with checkpoint manifest")
    feature_layout_sha256 = canonical_hash_v2(records.feature_layout)
    metadata = BundleMetadataV2(
        contract_sha256=manifest.contract_hash,
        action_contract_sha256=manifest.action_contract_hash,
        calibration_sha256=manifest.calibration_hash,
        checkpoint_sha256=file_sha256_v2(resume_path),
        architecture_sha256=manifest.architecture_hash,
        config_sha256=manifest.config_hash,
        feature_layout_sha256=feature_layout_sha256,
        checkpoint_kind=manifest.kind,
        stage=manifest.stage,
        contract_id=manifest.observation_contract_id,
    )
    export_sensor_policy_onnx_v2(
        actor,
        output_path,
        metadata=metadata,
        records=BundleRecordsV2(
            contract=records.contract,
            action_contract=records.action_contract,
            calibration=records.calibration,
            checkpoint=manifest.to_dict(),
            feature_layout=records.feature_layout,
            versions=records.versions or manifest.package_versions,
        ),
        sidecar_path=sidecar_path,
    )
    return output_path, sidecar_path
