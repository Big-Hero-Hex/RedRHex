"""Fixed-shape, two-input ONNX bundle exporter for Sensor V2."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from redrhex_policy_io import (
    ForwardResidualActionContractV2,
    SensorCalibrationProfileV2,
    StudentObservationContractV2,
    validate_calibration_lineage_v2,
)

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
    "training_calibration_sha256",
    "checkpoint_sha256",
    "feature_layout_sha256",
    "contact_supervision",
)
ONNX_PARITY_ATOL_V2 = 2.0e-5
ONNX_PARITY_RTOL_V2 = 2.0e-5


def _sha256(name: str, value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


@dataclass(frozen=True)
class BundleMetadataV2:
    contract_sha256: str
    action_contract_sha256: str
    calibration_sha256: str
    training_calibration_sha256: str
    checkpoint_sha256: str
    feature_layout_sha256: str
    checkpoint_kind: str
    stage: str
    architecture_sha256: str
    config_sha256: str
    canonical_config_sha256: str
    training_seed: int
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
            "training_calibration_sha256",
            "checkpoint_sha256",
            "architecture_sha256",
            "config_sha256",
            "canonical_config_sha256",
            "feature_layout_sha256",
        ):
            _sha256(name, getattr(self, name))
        if self.checkpoint_kind not in ("student_distilled_v2", "student_ppo_v2"):
            raise ValueError("deployable checkpoint_kind must be student_distilled_v2 or student_ppo_v2")
        if not self.stage:
            raise ValueError("bundle stage must not be empty")
        if (
            isinstance(self.training_seed, bool)
            or not isinstance(self.training_seed, int)
            or self.training_seed < 0
        ):
            raise ValueError("training_seed must be a non-negative integer")

    def embedded(self) -> dict[str, str]:
        values = asdict(self)
        # ROS requires the fixed subset above; architecture/config/kind/stage
        # stay embedded too so lineage is inspectable without the sidecar.
        keys = REQUIRED_METADATA_KEYS_V2 + (
            "architecture_sha256",
            "config_sha256",
            "canonical_config_sha256",
            "training_seed",
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
    training_calibration: dict[str, Any] | None = None
    checkpoint: dict[str, Any] | None = None
    feature_layout: dict[str, Any] | None = None
    versions: dict[str, str] = field(default_factory=dict)

    def require_deployable(self) -> None:
        missing = [
            name
            for name in (
                "contract",
                "action_contract",
                "calibration",
                "training_calibration",
                "checkpoint",
                "feature_layout",
            )
            if getattr(self, name) is None
        ]
        if missing:
            raise ValueError(f"deployable Sensor V2 bundle is missing records: {missing}")


@dataclass(frozen=True)
class ONNXParityReportV2:
    sample_count: int
    random_sample_count: int
    recorded_sample_count: int
    action_max_abs_error: float
    velocity_max_abs_error: float
    absolute_tolerance: float
    relative_tolerance: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "status": "passed",
            "sample_count": self.sample_count,
            "random_sample_count": self.random_sample_count,
            "recorded_sample_count": self.recorded_sample_count,
            "action_max_abs_error": self.action_max_abs_error,
            "velocity_max_abs_error": self.velocity_max_abs_error,
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": self.relative_tolerance,
        }


class _DeploymentGraphV2(nn.Module):
    def __init__(self, model: SensorStudentCoreV2) -> None:
        super().__init__()
        self.model = model

    def forward(self, sensor_history: torch.Tensor, command: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        actions, base_velocity_estimate, _ = self.model(sensor_history, command)
        return actions, base_velocity_estimate


def validate_sensor_policy_onnx_parity_v2(
    model: SensorStudentCoreV2,
    onnx_path: str | Path,
    *,
    sensor_histories: torch.Tensor | np.ndarray | None = None,
    commands: torch.Tensor | np.ndarray | None = None,
    random_sample_count: int = 4,
    seed: int = 0,
    absolute_tolerance: float = ONNX_PARITY_ATOL_V2,
    relative_tolerance: float = ONNX_PARITY_RTOL_V2,
) -> ONNXParityReportV2:
    """Fail unless ONNX Runtime matches Torch on randomized/recorded histories."""

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required for the Sensor V2 export parity gate") from exc
    if random_sample_count <= 0:
        raise ValueError("random_sample_count must be positive")
    first_parameter = next(model.parameters())
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    random_history = torch.randn(
        (random_sample_count, SENSOR_HISTORY_LENGTH_V2, SENSOR_FRAME_DIM_V2),
        generator=generator,
        dtype=torch.float32,
    )
    random_command = torch.empty(
        (random_sample_count, COMMAND_DIM_V2), dtype=torch.float32
    ).uniform_(-1.0, 1.0, generator=generator)

    if (sensor_histories is None) != (commands is None):
        raise ValueError("sensor_histories and commands must be supplied together")
    recorded_sample_count = 0
    if sensor_histories is not None:
        recorded_history = torch.as_tensor(sensor_histories, dtype=torch.float32)
        recorded_command = torch.as_tensor(commands, dtype=torch.float32)
        if recorded_history.ndim == 2:
            recorded_history = recorded_history.unsqueeze(0)
        if recorded_command.ndim == 1:
            recorded_command = recorded_command.unsqueeze(0)
        expected_history_tail = (SENSOR_HISTORY_LENGTH_V2, SENSOR_FRAME_DIM_V2)
        if tuple(recorded_history.shape[1:]) != expected_history_tail:
            raise ValueError(
                f"recorded sensor_histories must have shape (N, {expected_history_tail[0]}, "
                f"{expected_history_tail[1]})"
            )
        if tuple(recorded_command.shape) != (recorded_history.shape[0], COMMAND_DIM_V2):
            raise ValueError("recorded commands must have shape (N, 3)")
        if not torch.isfinite(recorded_history).all() or not torch.isfinite(recorded_command).all():
            raise ValueError("recorded parity inputs contain NaN or Inf")
        if recorded_history.shape[0] == 0:
            raise ValueError("recorded parity inputs must contain at least one sample")
        recorded_sample_count = int(recorded_history.shape[0])
        histories = torch.cat((random_history, recorded_history.cpu()), dim=0)
        command_values = torch.cat((random_command, recorded_command.cpu()), dim=0)
    else:
        histories = random_history
        command_values = random_command

    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    if [item.name for item in session.get_inputs()] != list(ONNX_INPUT_SHAPES_V2):
        raise RuntimeError("exported Sensor V2 ONNX input names changed")
    if [item.name for item in session.get_outputs()] != list(ONNX_OUTPUT_SHAPES_V2):
        raise RuntimeError("exported Sensor V2 ONNX output names changed")

    graph = _DeploymentGraphV2(model).eval()
    action_error = 0.0
    velocity_error = 0.0
    with torch.inference_mode():
        for history, command in zip(histories, command_values, strict=True):
            torch_actions, torch_velocity = graph(
                history.unsqueeze(0).to(device=first_parameter.device, dtype=first_parameter.dtype),
                command.unsqueeze(0).to(device=first_parameter.device, dtype=first_parameter.dtype),
            )
            onnx_actions, onnx_velocity = session.run(
                list(ONNX_OUTPUT_SHAPES_V2),
                {
                    "sensor_history": history.unsqueeze(0).numpy(),
                    "command": command.unsqueeze(0).numpy(),
                },
            )
            torch_actions_np = torch_actions.detach().cpu().numpy()
            torch_velocity_np = torch_velocity.detach().cpu().numpy()
            action_error = max(
                action_error, float(np.max(np.abs(torch_actions_np - onnx_actions)))
            )
            velocity_error = max(
                velocity_error, float(np.max(np.abs(torch_velocity_np - onnx_velocity)))
            )
            if not np.allclose(
                torch_actions_np,
                onnx_actions,
                rtol=relative_tolerance,
                atol=absolute_tolerance,
            ):
                raise RuntimeError(
                    f"Sensor V2 Torch/ONNX action parity failed: max error {action_error:.8g}"
                )
            if not np.allclose(
                torch_velocity_np,
                onnx_velocity,
                rtol=relative_tolerance,
                atol=absolute_tolerance,
            ):
                raise RuntimeError(
                    "Sensor V2 Torch/ONNX velocity-estimator parity failed: "
                    f"max error {velocity_error:.8g}"
                )
    return ONNXParityReportV2(
        sample_count=int(histories.shape[0]),
        random_sample_count=int(random_sample_count),
        recorded_sample_count=recorded_sample_count,
        action_max_abs_error=action_error,
        velocity_max_abs_error=velocity_error,
        absolute_tolerance=float(absolute_tolerance),
        relative_tolerance=float(relative_tolerance),
    )


def export_sensor_policy_onnx_v2(
    model: SensorStudentCoreV2,
    output_path: str | Path,
    *,
    metadata: BundleMetadataV2,
    records: BundleRecordsV2 | None = None,
    sidecar_path: str | Path | None = None,
    opset_version: int = 18,
    parity_sensor_histories: torch.Tensor | np.ndarray | None = None,
    parity_commands: torch.Tensor | np.ndarray | None = None,
    parity_input_sha256: str | None = None,
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

    try:
        parity = validate_sensor_policy_onnx_parity_v2(
            model,
            output_path,
            sensor_histories=parity_sensor_histories,
            commands=parity_commands,
        )
    except BaseException:
        output_path.unlink(missing_ok=True)
        sidecar_path.unlink(missing_ok=True)
        raise

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
        "training_calibration": records.training_calibration,
        "checkpoint": records.checkpoint,
        "feature_layout": records.feature_layout,
        "versions": records.versions,
        "torch_onnx_parity": parity.to_dict(),
        "recorded_parity_input": (
            None
            if parity_input_sha256 is None
            else {"sha256": _sha256("parity_input_sha256", parity_input_sha256)}
        ),
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
    *,
    parity_sensor_histories: torch.Tensor | np.ndarray,
    parity_commands: torch.Tensor | np.ndarray,
    parity_input_sha256: str,
    runtime_calibration: SensorCalibrationProfileV2 | None = None,
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
    parity_history_array = np.asarray(parity_sensor_histories)
    parity_command_array = np.asarray(parity_commands)
    if parity_history_array.shape[0] == 0 or parity_command_array.shape[0] == 0:
        raise ValueError("V2 deployment export requires non-empty recorded parity inputs")
    _sha256("parity_input_sha256", parity_input_sha256)
    assert records.contract is not None
    assert records.action_contract is not None
    assert records.calibration is not None
    assert records.training_calibration is not None
    assert records.feature_layout is not None
    observation_contract = StudentObservationContractV2.from_dict(records.contract)
    action_contract = ForwardResidualActionContractV2.from_dict(
        records.action_contract
    )
    checkpoint_calibration = SensorCalibrationProfileV2.from_dict(
        records.calibration
    )
    training_calibration = SensorCalibrationProfileV2.from_dict(
        records.training_calibration
    )
    if observation_contract.sha256 != manifest.contract_hash:
        raise ValueError("runner observation-contract record disagrees with checkpoint manifest")
    if action_contract.sha256 != manifest.action_contract_hash:
        raise ValueError("runner action-contract record disagrees with checkpoint manifest")
    if (
        manifest.observation_contract_id != observation_contract.contract_id
        or manifest.action_contract_id != action_contract.contract_id
    ):
        raise ValueError("runner contract IDs disagree with checkpoint manifest")
    if checkpoint_calibration.sha256 != training_calibration.sha256:
        raise ValueError(
            "runner calibration and training-calibration records disagree"
        )
    if training_calibration.sha256 != manifest.calibration_hash:
        raise ValueError("runner calibration record disagrees with checkpoint manifest")
    deployment_calibration = runtime_calibration or training_calibration
    training_calibration, deployment_calibration = validate_calibration_lineage_v2(
        training_calibration,
        deployment_calibration,
        observation_contract=observation_contract,
        action_contract=action_contract,
        require_runtime_hardware_ready=runtime_calibration is not None,
    )
    feature_layout_sha256 = canonical_hash_v2(records.feature_layout)
    metadata = BundleMetadataV2(
        contract_sha256=manifest.contract_hash,
        action_contract_sha256=manifest.action_contract_hash,
        calibration_sha256=deployment_calibration.sha256,
        training_calibration_sha256=manifest.calibration_hash,
        checkpoint_sha256=file_sha256_v2(resume_path),
        architecture_sha256=manifest.architecture_hash,
        config_sha256=manifest.config_hash,
        canonical_config_sha256=manifest.canonical_config_hash,
        training_seed=manifest.training_seed,
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
            calibration=deployment_calibration.to_dict(include_sha256=True),
            training_calibration=training_calibration.to_dict(include_sha256=True),
            checkpoint=manifest.to_dict(),
            feature_layout=records.feature_layout,
            versions=records.versions or manifest.package_versions,
        ),
        sidecar_path=sidecar_path,
        parity_sensor_histories=parity_history_array,
        parity_commands=parity_command_array,
        parity_input_sha256=parity_input_sha256,
    )
    return output_path, sidecar_path
