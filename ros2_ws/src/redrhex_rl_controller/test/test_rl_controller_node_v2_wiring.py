from __future__ import annotations

import ast
from pathlib import Path

from redrhex_rl_controller.deployment_guard_v2 import (
    DeploymentGuardV2,
    action_target_envelope_matches_v2,
    warmup_complete_v2,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
NODE_PATH = PACKAGE_ROOT / "redrhex_rl_controller/rl_controller_node_v2.py"
LAUNCH_PATH = PACKAGE_ROOT / "launch/redrhex_policy_sensor_v2.launch.py"
CONFIG_PATH = PACKAGE_ROOT / "config/redrhex_policy_sensor_v2.yaml"
SETUP_PATH = PACKAGE_ROOT / "setup.py"
PREFLIGHT_PATH = PACKAGE_ROOT / "redrhex_rl_controller/preflight_check_v2.py"


def test_v2_node_is_an_explicit_entrypoint_with_a_separate_launch_route() -> None:
    setup_source = SETUP_PATH.read_text(encoding="utf-8")
    launch_source = LAUNCH_PATH.read_text(encoding="utf-8")
    ast.parse(NODE_PATH.read_text(encoding="utf-8"))
    ast.parse(launch_source)

    assert (
        "rl_controller_node_v2 = redrhex_rl_controller.rl_controller_node_v2:main"
        in setup_source
    )
    assert 'executable="rl_controller_node_v2"' in launch_source
    assert 'name="redrhex_rl_controller_v2"' in launch_source
    assert '"redrhex_policy_sensor_v2.yaml"' in launch_source
    assert '"lowlevel_bridge_sensor_v2.yaml"' in launch_source
    assert "fake_sensor_node" not in launch_source


def test_v2_node_wires_only_physical_actor_sources_and_source_stamps() -> None:
    source = NODE_PATH.read_text(encoding="utf-8")
    for component in (
        "SensorObservationBuilderV2",
        "SensorPolicyONNXRunnerV2",
        "ForwardResidualActionDecoderV2",
        "RedRhexStateMachine",
        "SafetyFilter",
    ):
        assert component in source
    for topic in (
        '"/imu/data"',
        '"/joint_states"',
        '"/redrhex/joint_feedback_status_v2"',
    ):
        assert topic in source
    assert "self.observation_builder.update_imu(msg)" in source
    assert "self.observation_builder.update_joint_state(msg)" in source
    assert "update_joint_validity_diagnostic(msg)" in source
    assert "Odometry" not in source
    assert '"/odom"' not in source
    assert "commanded_abad" not in source
    assert "base_lin_vel" not in source


def test_v2_startup_is_disabled_and_history_is_a_state_transition_gate() -> None:
    import yaml

    params = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "redrhex_rl_controller_v2"
    ]["ros__parameters"]
    assert params["state_machine"]["enable_policy_on_start"] is False
    assert params["state_machine"]["enable_motor_output_on_start"] is False
    assert params["state_machine"]["require_history_ready"] is True
    assert params["observation"]["max_sensor_source_skew_s"] <= 1.0 / 60.0
    assert params["observation"]["max_history_period_error_ratio"] == 0.25
    node_source = NODE_PATH.read_text(encoding="utf-8")
    assert '"max_sensor_source_skew_s"' in node_source
    assert '"max_history_period_error_ratio"' in node_source
    assert warmup_complete_v2(
        elapsed_s=2.0,
        minimum_duration_s=1.0,
        history_ready=False,
    ) is False
    assert warmup_complete_v2(
        elapsed_s=2.0,
        minimum_duration_s=1.0,
        history_ready=True,
    ) is True


def test_motor_enable_requires_explicit_gate_and_hardware_ready_calibration() -> None:
    assert DeploymentGuardV2(False, False).hardware_authorized is False
    assert DeploymentGuardV2(True, False).hardware_authorized is False
    assert DeploymentGuardV2(False, True).hardware_authorized is False
    guard = DeploymentGuardV2(True, True)
    assert guard.hardware_authorized is True
    assert guard.motor_output_allowed(
        requested=True,
        state_allowed=True,
        estop=False,
    ) is True
    assert guard.motor_output_allowed(
        requested=True,
        state_allowed=True,
        estop=True,
    ) is False
    assert DeploymentGuardV2(True, True, False).hardware_authorized is False
    assert guard.motor_output_allowed(
        requested=True,
        state_allowed=True,
        estop=False,
        runtime_action_target_compatible=False,
    ) is False

    assert action_target_envelope_matches_v2(
        configured_action_clip=1.0,
        configured_main_velocity_limit_rad_s=15.0,
        contract_action_clip=1.0,
        contract_main_velocity_limit_rad_s=15.0,
    ) is True
    assert action_target_envelope_matches_v2(
        configured_action_clip=1.0,
        configured_main_velocity_limit_rad_s=9.0,
        contract_action_clip=1.0,
        contract_main_velocity_limit_rad_s=15.0,
    ) is False

    node_source = NODE_PATH.read_text(encoding="utf-8")
    preflight_source = PREFLIGHT_PATH.read_text(encoding="utf-8")
    assert "calibration_profile.hardware_ready" in node_source
    assert "bundle_calibration_hardware_ready" in preflight_source
    assert "hardware action tightening changed the V2 bundle target" in node_source
    assert "action_target_envelope_matches_bundle" in preflight_source


def test_default_hardware_limit_is_explicitly_incompatible_with_v2_contract() -> None:
    import yaml

    params = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "redrhex_rl_controller_v2"
    ]["ros__parameters"]
    assert params["safety"]["main_drive_vel_limit_rad_s"] == 9.0
    assert action_target_envelope_matches_v2(
        configured_action_clip=params["safety"]["action_clip"],
        configured_main_velocity_limit_rad_s=params["safety"][
            "main_drive_vel_limit_rad_s"
        ],
        contract_action_clip=1.0,
        contract_main_velocity_limit_rad_s=15.0,
    ) is False
