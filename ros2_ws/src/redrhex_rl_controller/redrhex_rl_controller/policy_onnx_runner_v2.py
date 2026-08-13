"""Strict two-input ONNX Runtime wrapper for sensor-only policy bundles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from redrhex_policy_io.contracts import (
    ForwardResidualActionContractV2,
    SensorCalibrationProfileV2,
    StudentObservationContractV2,
    canonical_sha256,
)


BUNDLE_SCHEMA_V2 = "redrhex.sensor-policy-bundle.v2"
CONTRACT_ID_V2 = "redrhex.student-observation.v2"
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
    "checkpoint_sha256",
    "feature_layout_sha256",
    "contact_supervision",
)


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
        use_cuda: bool = False,
        use_tensorrt: bool = False,
        session_factory: Callable[..., object] | None = None,
    ) -> None:
        self.onnx_path = str(Path(onnx_path).expanduser())
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
        self.metadata = {str(key): str(value) for key, value in model_metadata.items()}
        self.sidecar = self._load_sidecar(sidecar_file)
        self._validate_bundle_metadata(
            expected_contract_sha256=expected_contract_sha256,
            expected_action_contract_sha256=expected_action_contract_sha256,
            expected_calibration_sha256=expected_calibration_sha256,
        )
        self.observation_contract, self.action_contract, self.calibration_profile = (
            self._validate_sidecar_records()
        )

    @staticmethod
    def _shape(node: object) -> list[Any]:
        return list(getattr(node, "shape", []))

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
        for name, shape in OUTPUT_SHAPES_V2.items():
            actual = self._shape(self._outputs[name])
            if actual != shape:
                raise ValueError(f"V2 ONNX output {name} shape {actual} != {shape}")
        if "contact_belief" in self._outputs:
            actual = self._shape(self._outputs["contact_belief"])
            if actual != OPTIONAL_OUTPUT_SHAPES_V2["contact_belief"]:
                raise ValueError(
                    f"V2 ONNX output contact_belief shape {actual} != "
                    f"{OPTIONAL_OUTPUT_SHAPES_V2['contact_belief']}"
                )

    def _validate_bundle_metadata(
        self,
        *,
        expected_contract_sha256: str,
        expected_action_contract_sha256: str,
        expected_calibration_sha256: str,
    ) -> None:
        missing = [key for key in REQUIRED_METADATA_KEYS_V2 if not self.metadata.get(key)]
        if missing:
            raise ValueError(f"V2 ONNX metadata missing keys: {missing}")
        sidecar_metadata = {
            str(key): str(value) for key, value in self.sidecar["metadata"].items()
        }
        for key in REQUIRED_METADATA_KEYS_V2:
            if sidecar_metadata.get(key) != self.metadata[key]:
                raise ValueError(f"V2 sidecar/ONNX metadata disagreement for {key}")

        expected = {
            "bundle_schema": BUNDLE_SCHEMA_V2,
            "contract_id": CONTRACT_ID_V2,
            "contract_sha256": str(expected_contract_sha256),
            "action_contract_sha256": str(expected_action_contract_sha256),
            "calibration_sha256": str(expected_calibration_sha256),
            "contact_supervision": "disabled",
        }
        for key, value in expected.items():
            if self.metadata[key] != value:
                raise ValueError(
                    f"V2 bundle metadata mismatch for {key}: {self.metadata[key]!r} != {value!r}"
                )
        if "contact_belief" in self._outputs:
            raise ValueError("contact_belief output is forbidden while contact supervision is disabled")

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
    ]:
        required_records = ("contract", "action_contract", "calibration", "checkpoint", "feature_layout")
        missing = [name for name in required_records if not isinstance(self.sidecar.get(name), dict)]
        if missing:
            raise ValueError(f"V2 sidecar missing hash-bound records: {missing}")
        try:
            contract = StudentObservationContractV2.from_dict(self.sidecar["contract"])
            action_contract = ForwardResidualActionContractV2.from_dict(
                self.sidecar["action_contract"]
            )
            calibration = SensorCalibrationProfileV2.from_dict(self.sidecar["calibration"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid V2 sidecar contract/calibration record: {exc}") from exc
        record_hashes = {
            "contract_sha256": contract.sha256,
            "action_contract_sha256": action_contract.sha256,
            "calibration_sha256": calibration.sha256,
            "feature_layout_sha256": canonical_sha256(self.sidecar["feature_layout"]),
        }
        for name, digest in record_hashes.items():
            if self.metadata[name] != digest:
                raise ValueError(f"V2 {name} does not match its sidecar record")
        if calibration.observation_contract_sha256 != contract.sha256:
            raise ValueError("V2 calibration references a different observation contract")
        if calibration.action_contract_sha256 != action_contract.sha256:
            raise ValueError("V2 calibration references a different action contract")
        if calibration.attitude_mode != contract.attitude_mode:
            raise ValueError("V2 calibration attitude mode disagrees with observation contract")
        if calibration.imu_frame_id != contract.imu_frame_id:
            raise ValueError("V2 calibration IMU frame disagrees with observation contract")
        return contract, action_contract, calibration

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
        actions = np.asarray(values["actions"], dtype=np.float32)
        velocity = np.asarray(values["base_velocity_estimate"], dtype=np.float32)
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
