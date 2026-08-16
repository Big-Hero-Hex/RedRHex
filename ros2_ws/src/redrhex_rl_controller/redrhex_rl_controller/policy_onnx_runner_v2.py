"""Strict two-input ONNX Runtime wrapper for sensor-only policy bundles."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from redrhex_policy_io.contracts import (
    ForwardResidualActionContractV2,
    SensorCalibrationProfileV2,
    StudentObservationContractV2,
    canonical_sha256,
    validate_calibration_lineage_v2,
)


BUNDLE_SCHEMA_V2 = "redrhex.sensor-policy-bundle.v2"
BUNDLE_VERSION_V2 = "2"
CONTRACT_ID_V2 = "redrhex.student-observation.v2"
ACTION_CONTRACT_ID_V2 = "redrhex.forward-residual-action.v2"
CHECKPOINT_FORMAT_V2 = "redrhex.sensor-training.v2"
DEPLOYABLE_CHECKPOINT_KINDS_V2 = ("student_distilled_v2", "student_ppo_v2")
ALL_CHECKPOINT_KINDS_V2 = ("teacher_v2", *DEPLOYABLE_CHECKPOINT_KINDS_V2)
ONNX_FLOAT32_TYPE_V2 = "tensor(float)"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
INPUT_SHAPES_V2 = {
    "sensor_history": [1, 60, 36],
    "command": [1, 3],
}
OUTPUT_SHAPES_V2 = {
    "actions": [1, 12],
    "base_velocity_estimate": [1, 3],
}
OPTIONAL_OUTPUT_SHAPES_V2 = {"contact_belief": [1, 6]}
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
    "architecture_sha256",
    "config_sha256",
    "canonical_config_sha256",
    "training_seed",
    "checkpoint_kind",
    "stage",
)
_HASH_METADATA_KEYS_V2 = (
    "contract_sha256",
    "action_contract_sha256",
    "calibration_sha256",
    "training_calibration_sha256",
    "checkpoint_sha256",
    "feature_layout_sha256",
    "architecture_sha256",
    "config_sha256",
    "canonical_config_sha256",
)
_CHECKPOINT_MANIFEST_KEYS_V2 = {
    "kind",
    "stage",
    "observation_contract_id",
    "contract_hash",
    "action_contract_id",
    "action_contract_hash",
    "calibration_hash",
    "architecture_hash",
    "config_hash",
    "canonical_config_hash",
    "training_seed",
    "action_order",
    "iteration",
    "scheduler_state_present",
    "source_checkpoint_hash",
    "source_checkpoint_kind",
    "package_versions",
    "sensor_frame_dim",
    "history_length",
    "command_dim",
    "action_dim",
    "latent_dim",
    "schema_version",
    "format",
}


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"V2 {name} must be a lowercase SHA-256 digest string")
    digest = value
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError(f"V2 {name} must be a lowercase SHA-256 digest")
    return digest


@dataclass(frozen=True)
class SensorPolicyOutputsV2:
    actions: np.ndarray
    base_velocity_estimate: np.ndarray
    contact_belief: np.ndarray | None = None


@dataclass(frozen=True)
class ONNXIOInfoV2:
    inputs: dict[str, list[Any]]
    outputs: dict[str, list[Any]]
    providers: list[str]
    metadata: dict[str, str]
    sidecar_path: str


class SensorPolicyONNXRunnerV2:
    """Fail-closed loader for a hash-bound Sensor-Only Distillation V2 bundle.

    Unlike the compatibility V1 runner, this class never guesses the first
    graph input/output and never accepts dynamic or alternate dimensions.
    Embedded ONNX metadata and the JSON sidecar must agree exactly.
    """

    def __init__(
        self,
        onnx_path: str,
        *,
        sidecar_path: str | None = None,
        expected_contract_sha256: str,
        expected_action_contract_sha256: str,
        expected_calibration_sha256: str,
        expected_checkpoint_sha256: str | None = None,
        require_hardware_ready: bool = False,
        use_cuda: bool = False,
        use_tensorrt: bool = False,
        session_factory: Callable[..., object] | None = None,
    ) -> None:
        self.onnx_path = str(Path(onnx_path).expanduser())
        if not isinstance(require_hardware_ready, bool):
            raise TypeError("require_hardware_ready must be a boolean")
        self.require_hardware_ready = require_hardware_ready
        model_path = Path(self.onnx_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"V2 policy ONNX not found: {model_path}")
        self.sidecar_path = str(
            Path(sidecar_path).expanduser()
            if sidecar_path is not None
            else model_path.with_suffix(model_path.suffix + ".json")
        )
        sidecar_file = Path(self.sidecar_path)
        if not sidecar_file.is_file():
            raise FileNotFoundError(f"V2 policy sidecar not found: {sidecar_file}")

        if session_factory is None:
            try:
                import onnxruntime as ort
            except Exception as exc:  # pragma: no cover - deployment dependency
                raise RuntimeError(
                    "onnxruntime is required for V2 deployment; install a CPU or GPU build."
                ) from exc
            available = ort.get_available_providers()
            providers: list[str] = []
            if use_tensorrt and "TensorrtExecutionProvider" in available:
                providers.append("TensorrtExecutionProvider")
            if use_cuda and "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_factory = lambda path, selected_providers: ort.InferenceSession(  # noqa: E731
                path, sess_options=options, providers=selected_providers
            )
        else:
            providers = ["CPUExecutionProvider"]

        self.session = session_factory(self.onnx_path, providers)
        self._inputs = {item.name: item for item in self.session.get_inputs()}
        self._outputs = {item.name: item for item in self.session.get_outputs()}
        self._validate_io()

        model_metadata = self.session.get_modelmeta().custom_metadata_map
        if not isinstance(model_metadata, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in model_metadata.items()
        ):
            raise ValueError("V2 ONNX custom metadata must map strings to strings")
        self.metadata = dict(model_metadata)
        self.sidecar = self._load_sidecar(sidecar_file)
        self._validate_bundle_metadata(
            expected_contract_sha256=expected_contract_sha256,
            expected_action_contract_sha256=expected_action_contract_sha256,
            expected_calibration_sha256=expected_calibration_sha256,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
        )
        (
            self.observation_contract,
            self.action_contract,
            self.calibration_profile,
            self.checkpoint_manifest,
        ) = self._validate_sidecar_records()

    @staticmethod
    def _shape(node: object) -> list[Any]:
        return list(getattr(node, "shape", []))

    @staticmethod
    def _dtype(node: object) -> str:
        return str(getattr(node, "type", ""))

    @staticmethod
    def _load_sidecar(path: Path) -> dict[str, object]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid V2 policy sidecar {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("V2 policy sidecar root must be an object")
        if not isinstance(data.get("metadata"), dict):
            raise ValueError("V2 policy sidecar must contain a metadata object")
        if not isinstance(data.get("io"), dict):
            raise ValueError("V2 policy sidecar must contain an io object")
        return data

    def _validate_io(self) -> None:
        if set(self._inputs) != set(INPUT_SHAPES_V2):
            raise ValueError(
                f"V2 ONNX inputs must be exactly {sorted(INPUT_SHAPES_V2)}, got {sorted(self._inputs)}"
            )
        output_names = set(self._outputs)
        required_outputs = set(OUTPUT_SHAPES_V2)
        if not required_outputs.issubset(output_names):
            raise ValueError(
                f"V2 ONNX outputs must include {sorted(required_outputs)}, got {sorted(output_names)}"
            )
        unexpected = output_names - required_outputs - set(OPTIONAL_OUTPUT_SHAPES_V2)
        if unexpected:
            raise ValueError(f"V2 ONNX has unexpected outputs: {sorted(unexpected)}")
        for name, shape in INPUT_SHAPES_V2.items():
            actual = self._shape(self._inputs[name])
            if actual != shape:
                raise ValueError(f"V2 ONNX input {name} shape {actual} != {shape}")
            if self._dtype(self._inputs[name]) != ONNX_FLOAT32_TYPE_V2:
                raise ValueError(f"V2 ONNX input {name} must use float32 tensors")
        for name, shape in OUTPUT_SHAPES_V2.items():
            actual = self._shape(self._outputs[name])
            if actual != shape:
                raise ValueError(f"V2 ONNX output {name} shape {actual} != {shape}")
            if self._dtype(self._outputs[name]) != ONNX_FLOAT32_TYPE_V2:
                raise ValueError(f"V2 ONNX output {name} must use float32 tensors")
        if "contact_belief" in self._outputs:
            actual = self._shape(self._outputs["contact_belief"])
            if actual != OPTIONAL_OUTPUT_SHAPES_V2["contact_belief"]:
                raise ValueError(
                    f"V2 ONNX output contact_belief shape {actual} != "
                    f"{OPTIONAL_OUTPUT_SHAPES_V2['contact_belief']}"
                )
            if self._dtype(self._outputs["contact_belief"]) != ONNX_FLOAT32_TYPE_V2:
                raise ValueError("V2 ONNX output contact_belief must use float32 tensors")

    def _validate_bundle_metadata(
        self,
        *,
        expected_contract_sha256: str,
        expected_action_contract_sha256: str,
        expected_calibration_sha256: str,
        expected_checkpoint_sha256: str | None,
    ) -> None:
        missing = [key for key in REQUIRED_METADATA_KEYS_V2 if not self.metadata.get(key)]
        if missing:
            raise ValueError(f"V2 ONNX metadata missing keys: {missing}")
        unexpected = set(self.metadata) - set(REQUIRED_METADATA_KEYS_V2)
        if unexpected:
            raise ValueError(f"V2 ONNX metadata has unexpected keys: {sorted(unexpected)}")
        sidecar_metadata = self.sidecar["metadata"]
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in sidecar_metadata.items()
        ):
            raise ValueError("V2 sidecar metadata must map strings to strings")
        if set(sidecar_metadata) != set(REQUIRED_METADATA_KEYS_V2):
            raise ValueError("V2 sidecar metadata keys do not match bundle version 2")
        for key in REQUIRED_METADATA_KEYS_V2:
            if sidecar_metadata.get(key) != self.metadata[key]:
                raise ValueError(f"V2 sidecar/ONNX metadata disagreement for {key}")

        for key in _HASH_METADATA_KEYS_V2:
            _require_sha256(self.metadata[key], key)
        for name, value in (
            ("expected_contract_sha256", expected_contract_sha256),
            ("expected_action_contract_sha256", expected_action_contract_sha256),
            ("expected_calibration_sha256", expected_calibration_sha256),
        ):
            _require_sha256(value, name)
        if expected_checkpoint_sha256 is not None:
            _require_sha256(
                expected_checkpoint_sha256,
                "expected_checkpoint_sha256",
            )

        expected = {
            "bundle_schema": BUNDLE_SCHEMA_V2,
            "bundle_version": BUNDLE_VERSION_V2,
            "contract_id": CONTRACT_ID_V2,
            "contract_sha256": str(expected_contract_sha256),
            "action_contract_sha256": str(expected_action_contract_sha256),
            "calibration_sha256": str(expected_calibration_sha256),
            "contact_supervision": "disabled",
        }
        if expected_checkpoint_sha256 is not None:
            expected["checkpoint_sha256"] = str(expected_checkpoint_sha256)
        for key, value in expected.items():
            if self.metadata[key] != value:
                raise ValueError(
                    f"V2 bundle metadata mismatch for {key}: {self.metadata[key]!r} != {value!r}"
                )
        if "contact_belief" in self._outputs:
            raise ValueError("contact_belief output is forbidden while contact supervision is disabled")
        if self.metadata["checkpoint_kind"] not in DEPLOYABLE_CHECKPOINT_KINDS_V2:
            raise ValueError("V2 bundle checkpoint kind is not deployable")
        if not self.metadata["stage"].strip():
            raise ValueError("V2 bundle stage must not be empty")
        if not re.fullmatch(r"0|[1-9][0-9]*", self.metadata["training_seed"]):
            raise ValueError("V2 bundle training_seed must be a non-negative integer")

        expected_io = {
            "inputs": INPUT_SHAPES_V2,
            "outputs": OUTPUT_SHAPES_V2,
        }
        if self.sidecar["io"] != expected_io:
            raise ValueError("V2 sidecar IO declaration does not match the fixed deployment contract")

    def _validate_sidecar_records(
        self,
    ) -> tuple[
        StudentObservationContractV2,
        ForwardResidualActionContractV2,
        SensorCalibrationProfileV2,
        dict[str, object],
    ]:
        required_records = (
            "contract",
            "action_contract",
            "calibration",
            "training_calibration",
            "checkpoint",
            "feature_layout",
        )
        missing = [name for name in required_records if not isinstance(self.sidecar.get(name), dict)]
        if missing:
            raise ValueError(f"V2 sidecar missing hash-bound records: {missing}")
        try:
            contract = StudentObservationContractV2.from_dict(self.sidecar["contract"])
            action_contract = ForwardResidualActionContractV2.from_dict(
                self.sidecar["action_contract"]
            )
            calibration = SensorCalibrationProfileV2.from_dict(self.sidecar["calibration"])
            training_calibration = SensorCalibrationProfileV2.from_dict(
                self.sidecar["training_calibration"]
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid V2 sidecar contract/calibration record: {exc}") from exc
        if contract.contract_id != CONTRACT_ID_V2:
            raise ValueError("V2 observation contract record has the wrong contract ID")
        if action_contract.contract_id != ACTION_CONTRACT_ID_V2:
            raise ValueError("V2 action contract record has the wrong contract ID")
        record_hashes = {
            "contract_sha256": contract.sha256,
            "action_contract_sha256": action_contract.sha256,
            "calibration_sha256": calibration.sha256,
            "training_calibration_sha256": training_calibration.sha256,
            "feature_layout_sha256": canonical_sha256(self.sidecar["feature_layout"]),
        }
        for name, digest in record_hashes.items():
            if self.metadata[name] != digest:
                raise ValueError(f"V2 {name} does not match its sidecar record")
        training_calibration, calibration = validate_calibration_lineage_v2(
            training_calibration,
            calibration,
            observation_contract=contract,
            action_contract=action_contract,
            require_runtime_hardware_ready=self.require_hardware_ready,
        )
        checkpoint = self._validate_checkpoint_manifest(
            self.sidecar["checkpoint"],
            contract=contract,
            action_contract=action_contract,
            training_calibration=training_calibration,
        )
        self.training_calibration_profile = training_calibration
        self.training_calibration_sha256 = training_calibration.sha256
        self.runtime_calibration_sha256 = calibration.sha256
        self.checkpoint_sha256 = self.metadata["checkpoint_sha256"]
        return contract, action_contract, calibration, checkpoint

    def _validate_checkpoint_manifest(
        self,
        payload: object,
        *,
        contract: StudentObservationContractV2,
        action_contract: ForwardResidualActionContractV2,
        training_calibration: SensorCalibrationProfileV2,
    ) -> dict[str, object]:
        if not isinstance(payload, dict):
            raise ValueError("V2 checkpoint manifest must be an object")
        if set(payload) != _CHECKPOINT_MANIFEST_KEYS_V2:
            missing = sorted(_CHECKPOINT_MANIFEST_KEYS_V2 - set(payload))
            unexpected = sorted(set(payload) - _CHECKPOINT_MANIFEST_KEYS_V2)
            raise ValueError(
                "V2 checkpoint manifest keys changed: "
                f"missing={missing}, unexpected={unexpected}"
            )
        if self.sidecar.get("checkpoint_format") != CHECKPOINT_FORMAT_V2:
            raise ValueError("V2 sidecar checkpoint format is missing or unsupported")
        if payload["format"] != CHECKPOINT_FORMAT_V2 or payload["schema_version"] != 2:
            raise ValueError("V2 checkpoint manifest format or schema version is unsupported")
        if payload["kind"] not in DEPLOYABLE_CHECKPOINT_KINDS_V2:
            raise ValueError("V2 checkpoint manifest kind is not deployable")
        if not isinstance(payload["stage"], str) or not payload["stage"].strip():
            raise ValueError("V2 checkpoint manifest stage must not be empty")
        for key in (
            "contract_hash",
            "action_contract_hash",
            "calibration_hash",
            "architecture_hash",
            "config_hash",
            "canonical_config_hash",
        ):
            _require_sha256(payload[key], f"checkpoint.{key}")
        source_hash = payload["source_checkpoint_hash"]
        source_kind = payload["source_checkpoint_kind"]
        if (source_hash is None) != (source_kind is None):
            raise ValueError("V2 checkpoint source hash and kind must be set together")
        if source_hash is not None:
            _require_sha256(source_hash, "checkpoint.source_checkpoint_hash")
            if source_kind not in ALL_CHECKPOINT_KINDS_V2:
                raise ValueError("V2 checkpoint source kind is unsupported")
        if (
            isinstance(payload["iteration"], bool)
            or not isinstance(payload["iteration"], int)
            or payload["iteration"] < 0
        ):
            raise ValueError("V2 checkpoint iteration must be a non-negative integer")
        if not isinstance(payload["scheduler_state_present"], bool):
            raise ValueError("V2 checkpoint scheduler_state_present must be boolean")
        if (
            isinstance(payload["training_seed"], bool)
            or not isinstance(payload["training_seed"], int)
            or payload["training_seed"] < 0
        ):
            raise ValueError("V2 checkpoint training_seed must be a non-negative integer")
        versions = payload["package_versions"]
        if not isinstance(versions, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in versions.items()
        ):
            raise ValueError("V2 checkpoint package_versions must map strings to strings")

        expected_dimensions = {
            "sensor_frame_dim": 36,
            "history_length": 60,
            "command_dim": 3,
            "action_dim": 12,
            "latent_dim": 64,
        }
        for key, expected_value in expected_dimensions.items():
            if payload[key] != expected_value:
                raise ValueError(
                    f"V2 checkpoint {key} {payload[key]!r} != {expected_value!r}"
                )
        expected_action_order = list(
            action_contract.MAIN_JOINT_ORDER + action_contract.ABAD_JOINT_ORDER
        )
        if payload["action_order"] != expected_action_order:
            raise ValueError("V2 checkpoint action order disagrees with action contract")

        expected_bindings = {
            "kind": self.metadata["checkpoint_kind"],
            "stage": self.metadata["stage"],
            "observation_contract_id": CONTRACT_ID_V2,
            "contract_hash": contract.sha256,
            "action_contract_id": ACTION_CONTRACT_ID_V2,
            "action_contract_hash": action_contract.sha256,
            "calibration_hash": training_calibration.sha256,
            "architecture_hash": self.metadata["architecture_sha256"],
            "config_hash": self.metadata["config_sha256"],
            "canonical_config_hash": self.metadata["canonical_config_sha256"],
            "training_seed": int(self.metadata["training_seed"]),
        }
        mismatches = [
            key
            for key, expected_value in expected_bindings.items()
            if payload[key] != expected_value
        ]
        if mismatches:
            raise ValueError(
                "V2 checkpoint manifest disagrees with bundle metadata/records: "
                + ", ".join(mismatches)
            )
        return dict(payload)

    @property
    def io_info(self) -> ONNXIOInfoV2:
        return ONNXIOInfoV2(
            inputs={name: self._shape(node) for name, node in self._inputs.items()},
            outputs={name: self._shape(node) for name, node in self._outputs.items()},
            providers=list(self.session.get_providers()),
            metadata=dict(self.metadata),
            sidecar_path=self.sidecar_path,
        )

    def run(self, sensor_history: np.ndarray, command: np.ndarray) -> SensorPolicyOutputsV2:
        history = np.asarray(sensor_history, dtype=np.float32)
        cmd = np.asarray(command, dtype=np.float32)
        if history.shape == (60, 36):
            history = history.reshape(1, 60, 36)
        if cmd.shape == (3,):
            cmd = cmd.reshape(1, 3)
        if list(history.shape) != INPUT_SHAPES_V2["sensor_history"]:
            raise ValueError(f"sensor_history shape {history.shape} != (1, 60, 36)")
        if list(cmd.shape) != INPUT_SHAPES_V2["command"]:
            raise ValueError(f"command shape {cmd.shape} != (1, 3)")
        if not np.isfinite(history).all() or not np.isfinite(cmd).all():
            raise ValueError("V2 policy inputs contain NaN or Inf")

        output_names = list(OUTPUT_SHAPES_V2)
        raw = self.session.run(
            output_names,
            {"sensor_history": history, "command": cmd},
        )
        values = dict(zip(output_names, raw, strict=True))
        actions = np.asarray(values["actions"])
        velocity = np.asarray(values["base_velocity_estimate"])
        if actions.dtype != np.float32 or velocity.dtype != np.float32:
            raise ValueError("V2 policy outputs must be float32 tensors")
        if list(actions.shape) != OUTPUT_SHAPES_V2["actions"]:
            raise ValueError(f"V2 actions shape {actions.shape} != (1, 12)")
        if list(velocity.shape) != OUTPUT_SHAPES_V2["base_velocity_estimate"]:
            raise ValueError(f"V2 base velocity shape {velocity.shape} != (1, 3)")
        if not np.isfinite(actions).all() or not np.isfinite(velocity).all():
            raise ValueError("V2 policy outputs contain NaN or Inf")
        # The strict-forward action contract makes ABAD neutrality observable at
        # this boundary even if a malformed graph emits non-zero values.
        actions = actions.copy()
        actions[:, 6:12] = 0.0
        return SensorPolicyOutputsV2(
            actions=actions.reshape(12),
            base_velocity_estimate=velocity.reshape(3),
        )
