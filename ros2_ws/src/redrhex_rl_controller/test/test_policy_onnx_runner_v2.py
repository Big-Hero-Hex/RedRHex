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
    "checkpoint_sha256": "4" * 64,
    "feature_layout_sha256": canonical_sha256(FEATURE_LAYOUT),
}


class _Session:
    def __init__(self, metadata=None, *, command_shape=None, outputs=None):
        self.metadata = metadata or _metadata()
        self.inputs = [
            SimpleNamespace(name="sensor_history", shape=[1, 60, 36]),
            SimpleNamespace(name="command", shape=command_shape or [1, 3]),
        ]
        output_shapes = outputs or {
            "actions": [1, 12],
            "base_velocity_estimate": [1, 3],
        }
        self.outputs = [SimpleNamespace(name=name, shape=shape) for name, shape in output_shapes.items()]
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
        action = np.arange(12, dtype=np.float32).reshape(1, 12)
        velocity = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        return [action, velocity]


def _metadata():
    return {
        "bundle_schema": BUNDLE_SCHEMA_V2,
        "bundle_version": "2",
        "contract_id": SENSOR_ONLY_CONTRACT_ID_V2,
        **HASHES,
        "contact_supervision": "disabled",
    }


def _sidecar(metadata=None):
    return {
        "metadata": metadata or _metadata(),
        "io": {
            "inputs": {"sensor_history": [1, 60, 36], "command": [1, 3]},
            "outputs": {"actions": [1, 12], "base_velocity_estimate": [1, 3]},
        },
        "contract": OBSERVATION_CONTRACT.to_dict(include_sha256=True),
        "action_contract": ACTION_CONTRACT.to_dict(include_sha256=True),
        "calibration": CALIBRATION.to_dict(include_sha256=True),
        "checkpoint": {"kind": "student_distilled_v2"},
        "feature_layout": FEATURE_LAYOUT,
    }


def _make_runner(tmp_path, session, sidecar=None):
    model = tmp_path / "policy.onnx"
    model.write_bytes(b"test-placeholder")
    sidecar_path = tmp_path / "policy.onnx.json"
    sidecar_path.write_text(json.dumps(sidecar or _sidecar()), encoding="utf-8")
    return SensorPolicyONNXRunnerV2(
        str(model),
        expected_contract_sha256=HASHES["contract_sha256"],
        expected_action_contract_sha256=HASHES["action_contract_sha256"],
        expected_calibration_sha256=HASHES["calibration_sha256"],
        session_factory=lambda _path, _providers: session,
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
