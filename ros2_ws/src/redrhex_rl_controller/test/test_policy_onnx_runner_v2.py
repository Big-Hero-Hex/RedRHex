from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from redrhex_policy_io.contracts import (
    ForwardResidualActionContractV2,
    SensorCalibrationProfileV2,
    StudentObservationContractV2,
    canonical_sha256,
)

from redrhex_rl_controller.deployment_route import (
    LEGACY_CONTRACT_ID,
    SENSOR_ONLY_CONTRACT_ID_V2,
    resolve_deployment_route,
)
from redrhex_rl_controller.policy_onnx_runner_v2 import (
    BUNDLE_SCHEMA_V2,
    SensorPolicyONNXRunnerV2,
)
from redrhex_rl_controller.preflight_check_v2 import validate_v2_config


OBSERVATION_CONTRACT = StudentObservationContractV2.validated_quaternion(
    imu_frame_id="imu_link",
)
ACTION_CONTRACT = ForwardResidualActionContractV2()
CALIBRATION = SensorCalibrationProfileV2.provisional(
    OBSERVATION_CONTRACT,
    ACTION_CONTRACT,
)
FEATURE_LAYOUT = {"features": [item.to_dict() for item in OBSERVATION_CONTRACT.FEATURE_LAYOUT]}
HASHES = {
    "contract_sha256": OBSERVATION_CONTRACT.sha256,
    "action_contract_sha256": ACTION_CONTRACT.sha256,
    "calibration_sha256": CALIBRATION.sha256,
    "training_calibration_sha256": CALIBRATION.sha256,
    "checkpoint_sha256": "4" * 64,
    "feature_layout_sha256": canonical_sha256(FEATURE_LAYOUT),
    "architecture_sha256": "5" * 64,
    "config_sha256": "6" * 64,
    "canonical_config_sha256": "8" * 64,
}


def _hardware_runtime_calibration(**overrides) -> SensorCalibrationProfileV2:
    values = {
        "profile_id": "measured-runtime",
        "observation_contract_sha256": OBSERVATION_CONTRACT.sha256,
        "action_contract_sha256": ACTION_CONTRACT.sha256,
        "attitude_mode": OBSERVATION_CONTRACT.attitude_mode,
        "imu_frame_id": OBSERVATION_CONTRACT.imu_frame_id,
        "imu_to_body_wxyz": OBSERVATION_CONTRACT.imu_to_body_wxyz,
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


class _Session:
    def __init__(
        self,
        metadata=None,
        *,
        command_shape=None,
        outputs=None,
        input_type="tensor(float)",
        output_type="tensor(float)",
        output_dtype=np.float32,
    ):
        self.metadata = metadata or _metadata()
        self.inputs = [
            SimpleNamespace(
                name="sensor_history",
                shape=[1, 60, 36],
                type=input_type,
            ),
            SimpleNamespace(
                name="command",
                shape=command_shape or [1, 3],
                type=input_type,
            ),
        ]
        output_shapes = outputs or {
            "actions": [1, 12],
            "base_velocity_estimate": [1, 3],
        }
        self.outputs = [
            SimpleNamespace(name=name, shape=shape, type=output_type)
            for name, shape in output_shapes.items()
        ]
        self.output_dtype = output_dtype
        self.last_feed = None

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def get_modelmeta(self):
        return SimpleNamespace(custom_metadata_map=self.metadata)

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, output_names, feed):
        self.last_feed = feed
        assert output_names == ["actions", "base_velocity_estimate"]
        action = np.arange(12, dtype=self.output_dtype).reshape(1, 12)
        velocity = np.array([[0.1, 0.2, 0.3]], dtype=self.output_dtype)
        return [action, velocity]


def _metadata():
    return {
        "bundle_schema": BUNDLE_SCHEMA_V2,
        "bundle_version": "2",
        "contract_id": SENSOR_ONLY_CONTRACT_ID_V2,
        **HASHES,
        "contact_supervision": "disabled",
        "checkpoint_kind": "student_distilled_v2",
        "stage": "distillation_f2",
        "training_seed": "42",
    }


def _checkpoint_manifest():
    return {
        "kind": "student_distilled_v2",
        "stage": "distillation_f2",
        "observation_contract_id": OBSERVATION_CONTRACT.contract_id,
        "contract_hash": OBSERVATION_CONTRACT.sha256,
        "action_contract_id": ACTION_CONTRACT.contract_id,
        "action_contract_hash": ACTION_CONTRACT.sha256,
        "calibration_hash": CALIBRATION.sha256,
        "architecture_hash": HASHES["architecture_sha256"],
        "config_hash": HASHES["config_sha256"],
        "canonical_config_hash": HASHES["canonical_config_sha256"],
        "training_seed": 42,
        "action_order": list(
            ACTION_CONTRACT.MAIN_JOINT_ORDER + ACTION_CONTRACT.ABAD_JOINT_ORDER
        ),
        "iteration": 7,
        "scheduler_state_present": False,
        "source_checkpoint_hash": "7" * 64,
        "source_checkpoint_kind": "teacher_v2",
        "package_versions": {"python": "3.11"},
        "sensor_frame_dim": 36,
        "history_length": 60,
        "command_dim": 3,
        "action_dim": 12,
        "latent_dim": 64,
        "schema_version": 2,
        "format": "redrhex.sensor-training.v2",
    }


def _sidecar(metadata=None, *, calibration=CALIBRATION):
    return {
        "metadata": metadata or _metadata(),
        "io": {
            "inputs": {"sensor_history": [1, 60, 36], "command": [1, 3]},
            "outputs": {"actions": [1, 12], "base_velocity_estimate": [1, 3]},
        },
        "contract": OBSERVATION_CONTRACT.to_dict(include_sha256=True),
        "action_contract": ACTION_CONTRACT.to_dict(include_sha256=True),
        "calibration": calibration.to_dict(include_sha256=True),
        "training_calibration": CALIBRATION.to_dict(include_sha256=True),
        "checkpoint_format": "redrhex.sensor-training.v2",
        "checkpoint": _checkpoint_manifest(),
        "feature_layout": FEATURE_LAYOUT,
        "versions": {"python": "3.11"},
    }


def _make_runner(tmp_path, session, sidecar=None, **kwargs):
    model = tmp_path / "policy.onnx"
    model.write_bytes(b"test-placeholder")
    sidecar_path = tmp_path / "policy.onnx.json"
    sidecar_path.write_text(json.dumps(sidecar or _sidecar()), encoding="utf-8")
    expected_checkpoint_sha256 = kwargs.pop(
        "expected_checkpoint_sha256",
        HASHES["checkpoint_sha256"],
    )
    expected_calibration_sha256 = kwargs.pop(
        "expected_calibration_sha256",
        HASHES["calibration_sha256"],
    )
    return SensorPolicyONNXRunnerV2(
        str(model),
        expected_contract_sha256=HASHES["contract_sha256"],
        expected_action_contract_sha256=HASHES["action_contract_sha256"],
        expected_calibration_sha256=expected_calibration_sha256,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        session_factory=lambda _path, _providers: session,
        **kwargs,
    )


def test_runner_requires_fixed_names_shapes_and_hash_agreement(tmp_path):
    runner = _make_runner(tmp_path, _Session())
    result = runner.run(np.zeros((60, 36)), np.zeros(3))
    assert result.actions.shape == (12,)
    assert result.base_velocity_estimate.shape == (3,)
    assert np.array_equal(result.actions[6:12], np.zeros(6, dtype=np.float32))


def test_runner_rejects_wrong_fixed_shape(tmp_path):
    with pytest.raises(ValueError, match="command shape"):
        _make_runner(tmp_path, _Session(command_shape=[1, 4]))


def test_runner_requires_exact_float32_graph_and_runtime_outputs(tmp_path):
    with pytest.raises(ValueError, match="input sensor_history must use float32"):
        _make_runner(tmp_path, _Session(input_type="tensor(double)"))
    with pytest.raises(ValueError, match="output actions must use float32"):
        _make_runner(tmp_path, _Session(output_type="tensor(double)"))
    runner = _make_runner(tmp_path, _Session(output_dtype=np.float64))
    with pytest.raises(ValueError, match="outputs must be float32"):
        runner.run(np.zeros((60, 36)), np.zeros(3))


def test_runner_requires_exact_bundle_version_and_checkpoint_pin(tmp_path):
    metadata = _metadata()
    metadata["bundle_version"] = "3"
    with pytest.raises(ValueError, match="bundle_version"):
        _make_runner(
            tmp_path,
            _Session(metadata=metadata),
            sidecar=_sidecar(metadata=metadata),
        )

    with pytest.raises(ValueError, match="checkpoint_sha256"):
        _make_runner(
            tmp_path,
            _Session(),
            expected_checkpoint_sha256="8" * 64,
        )

    sidecar = _sidecar()
    sidecar["metadata"]["bundle_version"] = 2
    with pytest.raises(ValueError, match="metadata must map strings to strings"):
        _make_runner(tmp_path, _Session(), sidecar=sidecar)


def test_runner_binds_checkpoint_manifest_to_bundle_metadata(tmp_path):
    sidecar = _sidecar()
    sidecar["checkpoint"]["stage"] = "ppo_f3"
    with pytest.raises(ValueError, match="manifest disagrees"):
        _make_runner(tmp_path, _Session(), sidecar=sidecar)

    sidecar = _sidecar()
    sidecar["checkpoint"]["architecture_hash"] = "8" * 64
    with pytest.raises(ValueError, match="architecture_hash"):
        _make_runner(tmp_path, _Session(), sidecar=sidecar)

    sidecar = _sidecar()
    sidecar["checkpoint"].pop("config_hash")
    with pytest.raises(ValueError, match="manifest keys changed"):
        _make_runner(tmp_path, _Session(), sidecar=sidecar)

    sidecar = _sidecar()
    sidecar["checkpoint"].pop("canonical_config_hash")
    with pytest.raises(ValueError, match="manifest keys changed"):
        _make_runner(tmp_path, _Session(), sidecar=sidecar)

    sidecar = _sidecar()
    sidecar["checkpoint"]["canonical_config_hash"] = "9" * 64
    with pytest.raises(ValueError, match="canonical_config_hash"):
        _make_runner(tmp_path, _Session(), sidecar=sidecar)

    sidecar = _sidecar()
    sidecar["checkpoint"]["training_seed"] = 43
    with pytest.raises(ValueError, match="training_seed"):
        _make_runner(tmp_path, _Session(), sidecar=sidecar)


@pytest.mark.parametrize("value", ("-1", "01", "seed-42"))
def test_runner_rejects_noncanonical_training_seed_metadata(tmp_path, value):
    metadata = _metadata()
    metadata["training_seed"] = value
    with pytest.raises(ValueError, match="training_seed"):
        _make_runner(
            tmp_path,
            _Session(metadata=metadata),
            sidecar=_sidecar(metadata=metadata),
        )


def test_runner_hardware_ready_option_rejects_provisional_calibration(tmp_path):
    _make_runner(tmp_path, _Session(), require_hardware_ready=False)
    with pytest.raises(ValueError, match="hardware calibration is incomplete"):
        _make_runner(tmp_path, _Session(), require_hardware_ready=True)
    with pytest.raises(TypeError, match="must be a boolean"):
        _make_runner(tmp_path, _Session(), require_hardware_ready="false")


def test_runner_accepts_hardware_runtime_calibration_with_training_lineage(tmp_path):
    runtime = _hardware_runtime_calibration()
    metadata = _metadata()
    metadata["calibration_sha256"] = runtime.sha256
    runner = _make_runner(
        tmp_path,
        _Session(metadata=metadata),
        sidecar=_sidecar(metadata=metadata, calibration=runtime),
        require_hardware_ready=True,
        expected_calibration_sha256=runtime.sha256,
    )

    assert runner.calibration_profile.sha256 == runtime.sha256
    assert runner.runtime_calibration_sha256 == runtime.sha256
    assert runner.training_calibration_sha256 == CALIBRATION.sha256
    assert runner.checkpoint_manifest["calibration_hash"] == CALIBRATION.sha256


@pytest.mark.parametrize(
    ("override", "match"),
    (
        ({"imu_frame_id": "other_imu"}, "imu_frame_id"),
        ({"attitude_mode": "causal_gyro_accel"}, "attitude_mode"),
        ({"imu_to_body_wxyz": (0.0, 1.0, 0.0, 0.0)}, "imu_to_body_wxyz"),
    ),
)
def test_runner_rejects_runtime_calibration_that_changes_sensor_frame(
    tmp_path,
    override,
    match,
):
    runtime = _hardware_runtime_calibration(**override)
    metadata = _metadata()
    metadata["calibration_sha256"] = runtime.sha256
    with pytest.raises(ValueError, match=match):
        _make_runner(
            tmp_path,
            _Session(metadata=metadata),
            sidecar=_sidecar(metadata=metadata, calibration=runtime),
            require_hardware_ready=True,
            expected_calibration_sha256=runtime.sha256,
        )


def test_runner_rejects_checkpoint_bound_to_runtime_instead_of_training(tmp_path):
    runtime = _hardware_runtime_calibration()
    metadata = _metadata()
    metadata["calibration_sha256"] = runtime.sha256
    sidecar = _sidecar(metadata=metadata, calibration=runtime)
    sidecar["checkpoint"]["calibration_hash"] = runtime.sha256
    with pytest.raises(ValueError, match="calibration_hash"):
        _make_runner(
            tmp_path,
            _Session(metadata=metadata),
            sidecar=sidecar,
            expected_calibration_sha256=runtime.sha256,
        )


def test_runner_rejects_sidecar_metadata_disagreement(tmp_path):
    sidecar = _sidecar()
    sidecar["metadata"]["contract_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="disagreement"):
        _make_runner(tmp_path, _Session(), sidecar=sidecar)


def test_runner_rejects_disabled_contact_output(tmp_path):
    session = _Session(
        outputs={
            "actions": [1, 12],
            "base_velocity_estimate": [1, 3],
            "contact_belief": [1, 6],
        }
    )
    with pytest.raises(ValueError, match="contact_belief"):
        _make_runner(tmp_path, session)


def test_deployment_route_requires_exact_contract_id():
    assert resolve_deployment_route(LEGACY_CONTRACT_ID).contract_id == LEGACY_CONTRACT_ID
    assert resolve_deployment_route(SENSOR_ONLY_CONTRACT_ID_V2).contract_id == SENSOR_ONLY_CONTRACT_ID_V2
    with pytest.raises(ValueError, match="unsupported"):
        resolve_deployment_route("v2")


def test_default_v2_yaml_is_disabled_and_reports_evidence_blockers():
    import yaml
    from pathlib import Path

    config_path = Path(__file__).resolve().parents[1] / "config/redrhex_policy_sensor_v2.yaml"
    params = yaml.safe_load(config_path.read_text(encoding="utf-8"))[
        "redrhex_rl_controller_v2"
    ]["ros__parameters"]
    checks, blockers = validate_v2_config(params)
    by_name = {check["name"]: check for check in checks}
    assert by_name["exact_v2_route"]["ok"] is True
    assert by_name["fixed_sensor_layout"]["ok"] is True
    assert by_name["twelve_measured_joint_order"]["ok"] is True
    assert by_name["disabled_on_start"]["ok"] is True
    assert by_name["attitude_evidence"]["ok"] is False
    assert by_name["twelve_encoder_calibration_evidence"]["ok"] is False
    assert blockers
