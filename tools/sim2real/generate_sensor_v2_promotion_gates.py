"""Generate source-verifiable Sensor V2 promotion gates.

This command intentionally emits only PASS artifacts.  Any missing bundle,
checkpoint, ONNX, recorded parity input, or hardware-calibration evidence raises
before an artifact is written, so an operator cannot turn a partial development
export into promotion evidence by relabeling a generic JSON result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_IO_SOURCE = REPO_ROOT / "source" / "redrhex_policy_io"
SENSOR_V2_AGENTS_SOURCE = (
    REPO_ROOT
    / "source"
    / "RedRhex"
    / "RedRhex"
    / "tasks"
    / "direct"
    / "redrhex"
    / "agents"
)
for source_root in (POLICY_IO_SOURCE, SENSOR_V2_AGENTS_SOURCE):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from redrhex_policy_io import (  # noqa: E402
    ForwardResidualActionContractV2,
    SensorCalibrationProfileV2,
    StudentObservationContractV2,
    canonical_sha256,
)
from sensor_v2.checkpoint import (  # noqa: E402
    CHECKPOINT_FORMAT_V2,
    CheckpointManifestV2,
    architecture_hash_v2,
    canonical_hash_v2,
)
from sensor_v2.export import (  # noqa: E402
    ONNX_PARITY_ATOL_V2,
    ONNX_PARITY_RTOL_V2,
    validate_sensor_policy_onnx_parity_v2,
)
from sensor_v2.models import SensorStudentCoreV2  # noqa: E402


NO_LEAK_SCHEMA = "redrhex.no-privileged-leak-gate.v2"
PROVENANCE_SCHEMA = "redrhex.contract-provenance-gate.v2"
PARITY_SCHEMA = "redrhex.torch-onnx-parity-gate.v2"
PARITY_RANDOM_SAMPLE_COUNT = 4
PARITY_RANDOM_SEED = 0
PROMOTABLE_CHECKPOINT_STAGE = "ppo_f4"
EXPECTED_INPUTS = {"sensor_history": [1, 60, 36], "command": [1, 3]}
EXPECTED_OUTPUTS = {"actions": [1, 12], "base_velocity_estimate": [1, 3]}
FORBIDDEN_ACTOR_INPUTS = {
    "true_base_velocity",
    "odometry",
    "gait_clock",
    "previous_action",
    "commanded_abad",
    "internal_controller_targets",
    "linear_acceleration",
}


class PromotionGateGenerationError(ValueError):
    """Raised when a bundle cannot support canonical promotion evidence."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PromotionGateGenerationError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _require_non_negative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PromotionGateGenerationError(f"{label} must be a non-negative integer")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PromotionGateGenerationError(f"{label} must be a JSON object")
    return value


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PromotionGateGenerationError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PromotionGateGenerationError(f"{label} root must be a JSON object")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _onnx_metadata(path: Path) -> dict[str, str]:
    try:
        import onnx
    except ImportError as exc:
        raise PromotionGateGenerationError(
            "canonical gate generation requires the onnx package"
        ) from exc
    try:
        model = onnx.load(path)
        onnx.checker.check_model(model)
    except Exception as exc:  # onnx exposes multiple checker/parser exception types
        raise PromotionGateGenerationError(f"invalid ONNX graph {path}: {exc}") from exc
    result: dict[str, str] = {}
    for item in model.metadata_props:
        if item.key in result:
            raise PromotionGateGenerationError(
                f"ONNX graph has duplicate metadata key {item.key!r}"
            )
        result[item.key] = item.value
    return result


def _checkpoint_actor(
    checkpoint_path: Path,
    sidecar_checkpoint: Mapping[str, Any],
) -> tuple[SensorStudentCoreV2, CheckpointManifestV2]:
    """Strictly recover the deployable actor from a real V2 PPO checkpoint."""

    try:
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise PromotionGateGenerationError(
            f"cannot safely load Sensor V2 checkpoint {checkpoint_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping) or payload.get("format") != CHECKPOINT_FORMAT_V2:
        raise PromotionGateGenerationError("checkpoint file is not redrhex.sensor-training.v2")
    required = {
        "manifest",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "update",
    }
    missing = required - set(payload)
    if missing:
        raise PromotionGateGenerationError(
            f"checkpoint file is missing fields: {sorted(missing)}"
        )
    try:
        embedded_manifest = CheckpointManifestV2.from_dict(
            dict(_mapping(payload.get("manifest"), "checkpoint embedded manifest"))
        )
        advertised_manifest = CheckpointManifestV2.from_dict(dict(sidecar_checkpoint))
    except (TypeError, ValueError) as exc:
        raise PromotionGateGenerationError(f"invalid checkpoint manifest: {exc}") from exc
    if canonical_hash_v2(embedded_manifest.to_dict()) != canonical_hash_v2(
        advertised_manifest.to_dict()
    ):
        raise PromotionGateGenerationError(
            "checkpoint embedded manifest differs from the bundle sidecar manifest"
        )
    if payload.get("update") != embedded_manifest.iteration:
        raise PromotionGateGenerationError(
            "checkpoint update differs from its embedded manifest iteration"
        )
    if (payload.get("scheduler_state_dict") is not None) != bool(
        embedded_manifest.scheduler_state_present
    ):
        raise PromotionGateGenerationError(
            "checkpoint scheduler state differs from its embedded manifest"
        )

    state = _mapping(payload.get("model_state_dict"), "checkpoint model_state_dict")
    if not state:
        raise PromotionGateGenerationError("checkpoint model_state_dict is empty")
    architecture_schema: list[dict[str, Any]] = []
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise PromotionGateGenerationError(
                "checkpoint model_state_dict must contain only named tensors"
            )
        architecture_schema.append(
            {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
        )
    actual_architecture_hash = canonical_hash_v2(architecture_schema)
    if actual_architecture_hash != embedded_manifest.architecture_hash:
        raise PromotionGateGenerationError(
            "checkpoint tensor schema differs from its architecture hash"
        )
    if embedded_manifest.kind != "student_ppo_v2":
        raise PromotionGateGenerationError(
            "canonical promotion parity requires a student_ppo_v2 checkpoint"
        )
    if embedded_manifest.stage != PROMOTABLE_CHECKPOINT_STAGE:
        raise PromotionGateGenerationError(
            "canonical promotion parity requires checkpoint stage "
            f"{PROMOTABLE_CHECKPOINT_STAGE!r}"
        )

    actor = SensorStudentCoreV2().cpu().eval()
    expected_actor_state = actor.state_dict()
    actor_state = {
        name.removeprefix("actor."): value
        for name, value in state.items()
        if name.startswith("actor.")
    }
    if set(actor_state) != set(expected_actor_state):
        missing_actor = sorted(set(expected_actor_state) - set(actor_state))
        unexpected_actor = sorted(set(actor_state) - set(expected_actor_state))
        raise PromotionGateGenerationError(
            "checkpoint does not contain one complete SensorStudentCoreV2 actor: "
            f"missing={missing_actor}, unexpected={unexpected_actor}"
        )
    incompatible_actor_tensors = sorted(
        name
        for name, value in actor_state.items()
        if value.shape != expected_actor_state[name].shape
        or value.dtype != expected_actor_state[name].dtype
    )
    if incompatible_actor_tensors:
        raise PromotionGateGenerationError(
            "checkpoint actor tensor shape/dtype differs from SensorStudentCoreV2: "
            + ", ".join(incompatible_actor_tensors)
        )
    if not all(bool(torch.isfinite(value).all()) for value in actor_state.values()):
        raise PromotionGateGenerationError("checkpoint actor contains NaN or Inf")
    try:
        actor.load_state_dict(actor_state, strict=True)
    except (RuntimeError, ValueError) as exc:
        raise PromotionGateGenerationError(
            f"cannot strictly load checkpoint actor state: {exc}"
        ) from exc
    # The actor-only architecture is separately useful when diagnosing an
    # otherwise correctly bound full actor-critic checkpoint.
    if architecture_hash_v2(actor) != canonical_hash_v2(
        [
            {
                "name": name.removeprefix("actor."),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            for name, value in state.items()
            if name.startswith("actor.")
        ]
    ):
        raise PromotionGateGenerationError("checkpoint actor tensor order is non-canonical")
    return actor, embedded_manifest


def _load_parity_input(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            if "sensor_histories" not in payload.files or "command" not in payload.files:
                raise PromotionGateGenerationError(
                    "recorded parity NPZ requires sensor_histories and command arrays"
                )
            histories = np.asarray(payload["sensor_histories"], dtype=np.float32)
            commands = np.asarray(payload["command"], dtype=np.float32)
    except PromotionGateGenerationError:
        raise
    except (OSError, ValueError) as exc:
        raise PromotionGateGenerationError(
            f"cannot read recorded parity NPZ {path}: {exc}"
        ) from exc
    if histories.ndim != 3 or histories.shape[1:] != (60, 36):
        raise PromotionGateGenerationError(
            "recorded parity sensor_histories must have shape [N,60,36]"
        )
    if commands.shape != (histories.shape[0], 3) or histories.shape[0] == 0:
        raise PromotionGateGenerationError(
            "recorded parity command must have matching non-empty shape [N,3]"
        )
    if not np.isfinite(histories).all() or not np.isfinite(commands).all():
        raise PromotionGateGenerationError("recorded parity input contains NaN or Inf")
    source_sample_count = int(histories.shape[0])
    # Mirror scripts/rsl_rl/play.py exactly: a long real replay is represented
    # by 64 evenly-spaced samples, while the hash still binds the complete NPZ.
    if source_sample_count > 64:
        indices = np.linspace(0, source_sample_count - 1, 64, dtype=np.int64)
        histories = histories[indices]
        commands = commands[indices]
    return histories, commands, source_sample_count


def _validate_promotion_bundle_details(
    *,
    onnx_path: Path,
    sidecar_path: Path,
    checkpoint_path: Path,
) -> tuple[dict[str, str], dict[str, Any], SensorStudentCoreV2]:
    for path, label in (
        (onnx_path, "ONNX graph"),
        (sidecar_path, "bundle sidecar"),
        (checkpoint_path, "checkpoint"),
    ):
        if not path.is_file():
            raise PromotionGateGenerationError(f"{label} does not exist: {path}")

    sidecar = _load_json(sidecar_path, "bundle sidecar")
    metadata = dict(_mapping(sidecar.get("metadata"), "bundle metadata"))
    if metadata.get("bundle_schema") != "redrhex.sensor-policy-bundle.v2":
        raise PromotionGateGenerationError("bundle schema is not Sensor V2")
    if metadata.get("bundle_version") != "2":
        raise PromotionGateGenerationError("bundle version must be 2")
    if metadata.get("checkpoint_kind") != "student_ppo_v2":
        raise PromotionGateGenerationError(
            "promotion gates require a student_ppo_v2 deployment candidate"
        )
    if metadata.get("stage") != PROMOTABLE_CHECKPOINT_STAGE:
        raise PromotionGateGenerationError(
            "promotion gates require bundle metadata stage "
            f"{PROMOTABLE_CHECKPOINT_STAGE!r}"
        )
    if metadata.get("contact_supervision") != "disabled":
        raise PromotionGateGenerationError(
            "contact supervision cannot be promoted without validated labels"
        )
    embedded_metadata = _onnx_metadata(onnx_path)
    if embedded_metadata != metadata:
        raise PromotionGateGenerationError(
            "ONNX embedded metadata and bundle sidecar metadata differ"
        )

    io = _mapping(sidecar.get("io"), "bundle io")
    if io.get("inputs") != EXPECTED_INPUTS or io.get("outputs") != EXPECTED_OUTPUTS:
        raise PromotionGateGenerationError(
            "deployable actor must have exactly history+command inputs and fixed V2 outputs"
        )

    try:
        observation = StudentObservationContractV2.from_dict(
            _mapping(sidecar.get("contract"), "observation contract")
        )
        action = ForwardResidualActionContractV2.from_dict(
            _mapping(sidecar.get("action_contract"), "action contract")
        )
        runtime_calibration = SensorCalibrationProfileV2.from_dict(
            _mapping(sidecar.get("calibration"), "runtime calibration")
        ).validate(require_hardware_ready=True)
        training_calibration = SensorCalibrationProfileV2.from_dict(
            _mapping(sidecar.get("training_calibration"), "training calibration")
        ).validate(require_hardware_ready=False)
    except (TypeError, ValueError) as exc:
        raise PromotionGateGenerationError(f"invalid Sensor V2 bundle record: {exc}") from exc

    checkpoint = dict(_mapping(sidecar.get("checkpoint"), "checkpoint manifest"))
    if checkpoint.get("stage") != PROMOTABLE_CHECKPOINT_STAGE:
        raise PromotionGateGenerationError(
            "promotion gates require checkpoint manifest stage "
            f"{PROMOTABLE_CHECKPOINT_STAGE!r}"
        )
    feature_layout = dict(_mapping(sidecar.get("feature_layout"), "feature layout"))
    training_seed = _require_non_negative_int(
        checkpoint.get("training_seed"), "checkpoint training_seed"
    )
    bindings = {
        "observation_contract_sha256": observation.sha256,
        "action_contract_sha256": action.sha256,
        "runtime_calibration_sha256": runtime_calibration.sha256,
        "training_calibration_sha256": training_calibration.sha256,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "architecture_sha256": _require_sha256(
            checkpoint.get("architecture_hash"), "checkpoint architecture_hash"
        ),
        "config_sha256": _require_sha256(
            checkpoint.get("config_hash"), "checkpoint config_hash"
        ),
        "canonical_config_sha256": _require_sha256(
            checkpoint.get("canonical_config_hash"),
            "checkpoint canonical_config_hash",
        ),
    }
    metadata_expected = {
        "contract_id": observation.contract_id,
        "contract_sha256": bindings["observation_contract_sha256"],
        "action_contract_sha256": bindings["action_contract_sha256"],
        "calibration_sha256": bindings["runtime_calibration_sha256"],
        "training_calibration_sha256": bindings["training_calibration_sha256"],
        "checkpoint_sha256": bindings["checkpoint_sha256"],
        "architecture_sha256": bindings["architecture_sha256"],
        "config_sha256": bindings["config_sha256"],
        "canonical_config_sha256": bindings["canonical_config_sha256"],
        "training_seed": str(training_seed),
        "feature_layout_sha256": canonical_sha256(feature_layout),
    }
    mismatches = {
        name: {"expected": expected, "actual": metadata.get(name)}
        for name, expected in metadata_expected.items()
        if metadata.get(name) != expected
    }
    if mismatches:
        raise PromotionGateGenerationError(
            f"bundle metadata hash bindings disagree: {mismatches}"
        )
    checkpoint_expected = {
        "format": "redrhex.sensor-training.v2",
        "schema_version": 2,
        "kind": metadata["checkpoint_kind"],
        "stage": metadata["stage"],
        "action_order": list(action.MAIN_JOINT_ORDER + action.ABAD_JOINT_ORDER),
        "observation_contract_id": observation.contract_id,
        "contract_hash": bindings["observation_contract_sha256"],
        "action_contract_id": action.contract_id,
        "action_contract_hash": bindings["action_contract_sha256"],
        "calibration_hash": bindings["training_calibration_sha256"],
        "architecture_hash": bindings["architecture_sha256"],
        "config_hash": bindings["config_sha256"],
        "canonical_config_hash": bindings["canonical_config_sha256"],
        "training_seed": training_seed,
        "sensor_frame_dim": 36,
        "history_length": 60,
        "command_dim": 3,
        "action_dim": 12,
        "latent_dim": 64,
    }
    checkpoint_mismatches = {
        name: {"expected": expected, "actual": checkpoint.get(name)}
        for name, expected in checkpoint_expected.items()
        if checkpoint.get(name) != expected
    }
    if checkpoint_mismatches:
        raise PromotionGateGenerationError(
            f"checkpoint manifest binding disagrees: {checkpoint_mismatches}"
        )
    if runtime_calibration.observation_contract_sha256 != observation.sha256:
        raise PromotionGateGenerationError(
            "runtime calibration references a different observation contract"
        )
    if runtime_calibration.action_contract_sha256 != action.sha256:
        raise PromotionGateGenerationError(
            "runtime calibration references a different action contract"
        )
    if training_calibration.observation_contract_sha256 != observation.sha256:
        raise PromotionGateGenerationError(
            "training calibration references a different observation contract"
        )
    if training_calibration.action_contract_sha256 != action.sha256:
        raise PromotionGateGenerationError(
            "training calibration references a different action contract"
        )
    if (
        runtime_calibration.attitude_mode != training_calibration.attitude_mode
        or runtime_calibration.imu_frame_id != training_calibration.imu_frame_id
    ):
        raise PromotionGateGenerationError(
            "runtime calibration changes the checkpoint attitude/IMU-frame contract"
        )
    if feature_layout != {"features": observation.to_dict()["feature_layout"]}:
        raise PromotionGateGenerationError(
            "feature-layout record differs from the observation contract"
        )
    if sidecar.get("checkpoint_format") != "redrhex.sensor-training.v2":
        raise PromotionGateGenerationError("bundle checkpoint format is invalid")
    actor, embedded_manifest = _checkpoint_actor(checkpoint_path, checkpoint)
    return bindings, {
        "sidecar_sha256": _sha256(sidecar_path),
        "onnx_sha256": _sha256(onnx_path),
        "checkpoint_manifest_sha256": canonical_hash_v2(
            embedded_manifest.to_dict()
        ),
        "runtime_calibration_profile_id": runtime_calibration.profile_id,
        "training_calibration_profile_id": training_calibration.profile_id,
        "training_seed": training_seed,
        "checkpoint_stage": embedded_manifest.stage,
    }, actor


def validate_promotion_bundle(
    *,
    onnx_path: Path,
    sidecar_path: Path,
    checkpoint_path: Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Validate bundle hashes, contracts, manifests, and actual checkpoint state."""

    bindings, provenance, _ = _validate_promotion_bundle_details(
        onnx_path=onnx_path,
        sidecar_path=sidecar_path,
        checkpoint_path=checkpoint_path,
    )
    return bindings, provenance


def validate_torch_onnx_parity_bundle(
    *,
    onnx_path: Path,
    sidecar_path: Path,
    checkpoint_path: Path,
    parity_input_path: Path,
    parity_input_sha256: str,
) -> dict[str, Any]:
    """Re-run canonical Torch/ORT parity from hash-bound source artifacts."""

    parity_input_sha256 = _require_sha256(
        parity_input_sha256, "recorded parity input sha256"
    )
    if not parity_input_path.is_file():
        raise PromotionGateGenerationError(
            f"recorded parity input does not exist: {parity_input_path}"
        )
    actual_parity_sha256 = _sha256(parity_input_path)
    if actual_parity_sha256 != parity_input_sha256:
        raise PromotionGateGenerationError(
            "recorded parity input sha256 mismatch: "
            f"expected {parity_input_sha256}, got {actual_parity_sha256}"
        )
    bindings, provenance, actor = _validate_promotion_bundle_details(
        onnx_path=onnx_path,
        sidecar_path=sidecar_path,
        checkpoint_path=checkpoint_path,
    )
    sidecar = _load_json(sidecar_path, "bundle sidecar")
    recorded = _mapping(
        sidecar.get("recorded_parity_input"), "bundle recorded_parity_input"
    )
    sidecar_parity_sha256 = _require_sha256(
        recorded.get("sha256"), "bundle recorded parity input sha256"
    )
    if sidecar_parity_sha256 != actual_parity_sha256:
        raise PromotionGateGenerationError(
            "bundle recorded parity input hash differs from the supplied NPZ"
        )
    histories, commands, source_sample_count = _load_parity_input(parity_input_path)
    try:
        parity = validate_sensor_policy_onnx_parity_v2(
            actor,
            onnx_path,
            sensor_histories=histories,
            commands=commands,
            random_sample_count=PARITY_RANDOM_SAMPLE_COUNT,
            seed=PARITY_RANDOM_SEED,
            absolute_tolerance=ONNX_PARITY_ATOL_V2,
            relative_tolerance=ONNX_PARITY_RTOL_V2,
        )
    except (RuntimeError, ValueError) as exc:
        raise PromotionGateGenerationError(
            f"canonical Sensor V2 Torch/ONNX parity failed: {exc}"
        ) from exc
    report = parity.to_dict()
    report.pop("status", None)
    return {
        "bindings": bindings,
        "provenance": {
            **provenance,
            "parity_input_sha256": actual_parity_sha256,
        },
        "metadata": dict(_mapping(sidecar.get("metadata"), "bundle metadata")),
        "torch_onnx_parity": report,
        "recorded_parity_input": {
            "path": str(parity_input_path),
            "sha256": actual_parity_sha256,
            "source_sample_count": source_sample_count,
            "evaluated_sample_count": int(histories.shape[0]),
        },
        "source_artifacts": {
            "onnx": {"path": str(onnx_path), "sha256": _sha256(onnx_path)},
            "sidecar": {
                "path": str(sidecar_path),
                "sha256": _sha256(sidecar_path),
            },
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": _sha256(checkpoint_path),
            },
            "parity_input": {
                "path": str(parity_input_path),
                "sha256": actual_parity_sha256,
            },
        },
    }


def generate_promotion_gates(
    *,
    onnx_path: str | Path,
    sidecar_path: str | Path,
    checkpoint_path: str | Path,
    parity_input_path: str | Path,
    parity_input_sha256: str,
    output_dir: str | Path,
) -> dict[str, dict[str, str]]:
    """Validate one deployment candidate and atomically emit its gate files."""

    onnx_path = Path(onnx_path).resolve()
    sidecar_path = Path(sidecar_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    parity_input_path = Path(parity_input_path).resolve()
    output_dir = Path(output_dir).resolve()
    parity_result = validate_torch_onnx_parity_bundle(
        onnx_path=onnx_path,
        sidecar_path=sidecar_path,
        checkpoint_path=checkpoint_path,
        parity_input_path=parity_input_path,
        parity_input_sha256=parity_input_sha256,
    )
    bindings = parity_result["bindings"]
    source = {
        name: value
        for name, value in parity_result["provenance"].items()
        if name != "parity_input_sha256"
    }
    common = {
        "schema_version": 2,
        "status": "PASS",
        "provenance": {**bindings, **source},
        "source_artifacts": {
            "onnx": {"path": str(onnx_path), "sha256": _sha256(onnx_path)},
            "sidecar": {
                "path": str(sidecar_path),
                "sha256": _sha256(sidecar_path),
            },
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": _sha256(checkpoint_path),
            },
        },
    }
    no_leak = {
        **common,
        "schema": NO_LEAK_SCHEMA,
        "gate": "no_privileged_leak",
        "checks": [
            {"name": "actor_inputs_exact", "status": "PASS"},
            {"name": "forbidden_features_absent", "status": "PASS"},
            {"name": "command_separate", "status": "PASS"},
            {"name": "privileged_groups_training_only", "status": "PASS"},
        ],
        "actor_input_names": list(EXPECTED_INPUTS),
        "forbidden_actor_inputs": sorted(FORBIDDEN_ACTOR_INPUTS),
    }
    provenance = {
        **common,
        "schema": PROVENANCE_SCHEMA,
        "gate": "contract_provenance",
        "checks": [
            {"name": "observation_contract_hash", "status": "PASS"},
            {"name": "action_contract_hash", "status": "PASS"},
            {"name": "calibration_hash", "status": "PASS"},
            {"name": "runtime_calibration_lineage", "status": "PASS"},
            {"name": "checkpoint_manifest_binding", "status": "PASS"},
            {"name": "architecture_config_binding", "status": "PASS"},
        ],
    }
    parity = {
        "schema_version": 2,
        "schema": PARITY_SCHEMA,
        "gate": "torch_onnx_parity",
        "status": "PASS",
        "provenance": {
            **parity_result["bindings"],
            **parity_result["provenance"],
        },
        "metadata": parity_result["metadata"],
        "checks": [
            {"name": "checkpoint_state_loaded_strictly", "status": "PASS"},
            {"name": "random_history_parity", "status": "PASS"},
            {"name": "recorded_history_parity", "status": "PASS"},
            {"name": "source_artifact_hashes", "status": "PASS"},
        ],
        "verifier": {
            "random_seed": PARITY_RANDOM_SEED,
            "random_sample_count": PARITY_RANDOM_SAMPLE_COUNT,
            "absolute_tolerance": ONNX_PARITY_ATOL_V2,
            "relative_tolerance": ONNX_PARITY_RTOL_V2,
        },
        "torch_onnx_parity": parity_result["torch_onnx_parity"],
        "recorded_parity_input": parity_result["recorded_parity_input"],
        "source_artifacts": parity_result["source_artifacts"],
    }
    payloads = {
        "no_privileged_leak": (output_dir / "no_privileged_leak_v2.json", no_leak),
        "torch_onnx_parity": (output_dir / "torch_onnx_parity_v2.json", parity),
        "contract_provenance": (output_dir / "contract_provenance_v2.json", provenance),
    }
    result: dict[str, dict[str, str]] = {}
    for gate, (path, payload) in payloads.items():
        _atomic_json(path, payload)
        result[gate] = {"path": str(path), "sha256": _sha256(path)}
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parity-input", type=Path, required=True)
    parser.add_argument("--parity-input-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = generate_promotion_gates(
        onnx_path=args.onnx,
        sidecar_path=args.sidecar,
        checkpoint_path=args.checkpoint,
        parity_input_path=args.parity_input,
        parity_input_sha256=args.parity_input_sha256,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
