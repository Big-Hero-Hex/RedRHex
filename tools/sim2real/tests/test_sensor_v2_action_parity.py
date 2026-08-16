from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
for package_root in (
    REPO_ROOT / "source" / "redrhex_policy_io",
    REPO_ROOT / "ros2_ws" / "src" / "redrhex_rl_controller",
):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from redrhex_policy_io import (  # noqa: E402
    ForwardResidualActionContractV2,
    decode_forward_residual_action_v2,
    decode_forward_residual_action_v2_torch,
)
from redrhex_rl_controller.action_decoder_v2 import (  # noqa: E402
    ForwardResidualActionDecoderV2,
)


def _load_sim_contract_factory():
    path = (
        REPO_ROOT
        / "source"
        / "RedRhex"
        / "RedRhex"
        / "tasks"
        / "direct"
        / "redrhex"
        / "sensor_v2_action.py"
    )
    spec = importlib.util.spec_from_file_location("redrhex_sensor_v2_action_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.forward_residual_action_contract_v2_from_config


def _numpy_batch(
    actions: np.ndarray,
    commands: np.ndarray,
    phases: np.ndarray,
    positions: np.ndarray,
    steps: np.ndarray,
    contract: ForwardResidualActionContractV2,
) -> dict[str, np.ndarray]:
    decoded = [
        decode_forward_residual_action_v2(
            actions[index],
            commands[index],
            float(phases[index]),
            positions[index],
            control_step=int(steps[index]),
            contract=contract,
        )
        for index in range(actions.shape[0])
    ]
    return {
        "safe": np.asarray([item.safe_action for item in decoded]),
        "nominal": np.asarray([item.nominal_main_velocity_rad_s for item in decoded]),
        "residual": np.asarray([item.residual_main_velocity_rad_s for item in decoded]),
        "main": np.asarray([item.target_main_velocity_rad_s for item in decoded]),
        "abad": np.asarray([item.target_abad_position_rad for item in decoded]),
        "warmup": np.asarray([item.action_warmup_scale for item in decoded]),
    }


def test_simulator_and_bundle_use_one_fail_closed_contract_factory() -> None:
    build_contract = _load_sim_contract_factory()
    defaults = ForwardResidualActionContractV2()
    initial_joint_positions = {
        **dict(zip(defaults.MAIN_JOINT_ORDER, defaults.initial_main_position_rad)),
        **dict(zip(defaults.ABAD_JOINT_ORDER, defaults.abad_neutral_position_rad)),
    }
    cfg = SimpleNamespace(
        action_space=12,
        main_drive_joint_names=list(defaults.MAIN_JOINT_ORDER),
        abad_joint_names=list(defaults.ABAD_JOINT_ORDER),
        tripod_a_leg_indices=list(defaults.TRIPOD_A),
        tripod_b_leg_indices=list(defaults.TRIPOD_B),
        leg_direction_multiplier=list(defaults.LEG_DIRECTION_MULTIPLIER),
        base_gait_frequency=defaults.NOMINAL_GAIT_FREQUENCY_HZ,
        stance_phase_start=defaults.STANCE_PHASE_START_RAD,
        stance_phase_end=defaults.STANCE_PHASE_END_RAD,
        stance_duty_cycle=defaults.STANCE_DUTY_CYCLE,
        stance_velocity_ratio=defaults.STANCE_VELOCITY_RATIO,
        swing_velocity_ratio=defaults.SWING_VELOCITY_RATIO,
        tripod_phase_offset=defaults.TRIPOD_PHASE_OFFSET_RAD,
        strict_forward_residual_actions=True,
        stage_drive_vel_scale=[8.0],
        main_drive_vel_scale=7.0,
        stage_forward_policy_drive_residual_scale=[0.1],
        main_drive_residual_scale=0.2,
        drive_bias_vx_ref=0.45,
        stage_forward_bias_scale=[1.0],
        forward_drive_action_scale=0.5,
        forward_phase_lock_gain=1.2,
        stage_forward_residual_cap_ratio=[0.26],
        forward_residual_cap_ratio=0.2,
        stage_action_warmup_steps=[30],
        robot_cfg=SimpleNamespace(
            init_state=SimpleNamespace(joint_pos=initial_joint_positions)
        ),
    )
    contract = build_contract(cfg)
    assert contract.main_residual_scale_rad_s == pytest.approx(0.8)
    assert contract.action_warmup_steps == 30
    assert contract.main_velocity_limit_rad_s == pytest.approx(15.0)
    assert contract.to_dict()["target_layers"]["raw_contract_target_slew_rate_rad_s2"] is None

    cfg.leg_direction_multiplier[0] = 1.0
    with pytest.raises(ValueError, match="leg_direction_multiplier"):
        build_contract(cfg)


def test_numpy_and_torch_sim_adapters_match_asymmetric_vector_contract() -> None:
    contract = ForwardResidualActionContractV2(
        main_residual_scale_rad_s=1.1,
        forward_bias_scale=1.25,
        residual_cap_ratio=0.31,
        action_clip=0.7,
        action_warmup_steps=4,
        main_velocity_limit_rad_s=2.4,
        abad_neutral_position_rad=(-0.12, -0.08, -0.03, 0.04, 0.09, 0.13),
        main_output_sign=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
        abad_output_sign=(1.0, -1.0, 1.0, -1.0, 1.0, -1.0),
    )
    actions = np.asarray(
        [
            np.linspace(-1.4, 1.4, 12),
            np.linspace(1.2, -1.2, 12),
            [0.7, -0.7, 0.5, -0.5, 0.2, -0.2, 1, 1, 1, 1, 1, 1],
            np.zeros(12),
        ],
        dtype=np.float64,
    )
    commands = np.asarray(
        [[0.45, 0.0, 0.0], [0.22, 0.0, 0.0], [0.70, 0.0, 0.0], [0.10, 0.0, 0.0]],
        dtype=np.float64,
    )
    phases = np.asarray([0.0, math.pi / 6.0, math.pi, 5.9], dtype=np.float64)
    positions = np.asarray(
        [
            [-0.1, -2.8, -3.2, 0.2, 3.0, 0.4],
            [-1.1, -0.2, -4.7, 2.0, 0.3, 3.1],
            [0.7, -3.0, 1.2, -0.4, 4.0, -2.2],
            [-2.0, -1.0, -0.2, 0.4, 1.1, 2.5],
        ],
        dtype=np.float64,
    )
    steps = np.asarray([0, 1, 3, 9], dtype=np.int64)

    expected = _numpy_batch(actions, commands, phases, positions, steps, contract)
    actual = decode_forward_residual_action_v2_torch(
        torch.from_numpy(actions),
        torch.from_numpy(commands),
        torch.from_numpy(phases),
        torch.from_numpy(positions),
        control_step=torch.from_numpy(steps),
        contract=contract,
    )
    fields = {
        "safe": actual.safe_action,
        "nominal": actual.nominal_main_velocity_rad_s,
        "residual": actual.residual_main_velocity_rad_s,
        "main": actual.target_main_velocity_rad_s,
        "abad": actual.target_abad_position_rad,
        "warmup": actual.action_warmup_scale,
    }
    for name, tensor in fields.items():
        np.testing.assert_allclose(tensor.cpu().numpy(), expected[name], rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_array_equal(actual.safe_action[:, 6:].cpu().numpy(), np.zeros((4, 6)))
    np.testing.assert_array_less(
        np.abs(actual.target_main_velocity_rad_s.cpu().numpy()),
        contract.main_velocity_limit_rad_s + 1.0e-12,
    )


def test_short_trace_matches_numpy_torch_and_ros_raw_contract_target() -> None:
    contract = ForwardResidualActionContractV2(
        action_warmup_steps=4,
        main_velocity_limit_rad_s=15.0,
    )
    decoder = ForwardResidualActionDecoderV2(
        contract,
        {"main_drive_slew_rate_rad_s2": 1.0e6},
    )
    position = np.asarray([-0.3, -2.7, -3.0, 0.4, 2.9, 0.7], dtype=np.float64)
    command = np.asarray([0.45, 0.0, 0.0], dtype=np.float64)
    dt = 1.0 / contract.policy_rate_hz

    # INIT_STAND is outside the motion-relative CPG lifecycle.  It must not
    # consume startup ramp samples or run the reference ahead of the legs.
    decoder.init_stand_command(position)
    decoder.init_stand_command(position)
    assert decoder.contract_control_step == 0
    assert decoder.gait_phase_rad == pytest.approx(0.0)

    observed_warmup: list[float] = []
    for trace_index in range(5):
        action = np.linspace(-0.8, 0.8, 12) * (trace_index + 1) / 5.0
        phase = decoder.gait_phase_rad
        step = decoder.contract_control_step
        expected = decode_forward_residual_action_v2(
            action,
            command,
            phase,
            position,
            control_step=step,
            contract=contract,
        )
        torch_result = decode_forward_residual_action_v2_torch(
            torch.from_numpy(action),
            torch.from_numpy(command),
            torch.tensor(phase, dtype=torch.float64),
            torch.from_numpy(position),
            control_step=torch.tensor(step),
            contract=contract,
        )
        command_result = decoder.decode(action, position, command, dt)
        status = decoder.last_target_status
        assert status is not None
        expected_main = np.asarray(expected.target_main_velocity_rad_s)
        np.testing.assert_allclose(
            torch_result.target_main_velocity_rad_s.numpy(),
            expected_main,
            rtol=1.0e-10,
            atol=1.0e-10,
        )
        np.testing.assert_allclose(
            status.raw_contract_target_main_drive_velocity,
            expected_main,
            rtol=1.0e-10,
            atol=1.0e-10,
        )
        np.testing.assert_allclose(command_result.target_main_drive_velocity, expected_main)
        assert status.hardware_tightening_applied is False
        observed_warmup.append(status.action_warmup_scale)
        position = position + expected_main * dt

    assert observed_warmup == pytest.approx([0.25, 0.5, 0.75, 1.0, 1.0])


def test_time_warped_reference_has_65_percent_duty_and_continuous_support() -> None:
    contract = ForwardResidualActionContractV2(action_warmup_steps=0)
    samples = 6000
    phases = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    offsets = np.zeros(6)
    offsets[list(contract.TRIPOD_B)] = contract.TRIPOD_PHASE_OFFSET_RAD
    fraction = np.mod(phases[:, None] + offsets, 2.0 * math.pi) / (2.0 * math.pi)
    stance = fraction < contract.STANCE_DUTY_CYCLE
    tripod_a = np.all(stance[:, list(contract.TRIPOD_A)], axis=1)
    tripod_b = np.all(stance[:, list(contract.TRIPOD_B)], axis=1)

    assert np.mean(tripod_a) == pytest.approx(0.65, abs=1.0 / samples)
    assert np.mean(tripod_b) == pytest.approx(0.65, abs=1.0 / samples)
    assert np.mean(tripod_a & tripod_b) == pytest.approx(0.30, abs=1.0 / samples)
    assert np.all(tripod_a | tripod_b)
    assert contract.command_scaled_cycle_steps(0.22) == 121
    assert contract.command_scaled_cycle_steps(0.35) == 76
    assert contract.command_scaled_cycle_steps(0.42) == 67


def test_ros_reports_hardware_slew_separately_from_raw_parity_target() -> None:
    contract = ForwardResidualActionContractV2(
        action_warmup_steps=0,
        main_velocity_limit_rad_s=9.0,
    )
    decoder = ForwardResidualActionDecoderV2(
        contract,
        {
            "action_clip": 0.25,
            "main_drive_vel_limit_rad_s": 0.4,
            "main_drive_slew_rate_rad_s2": 0.6,
        },
    )
    result = decoder.decode(
        np.ones(12),
        np.zeros(6),
        np.asarray([0.45, 0.0, 0.0]),
        1.0 / 60.0,
    )
    status = decoder.last_target_status
    assert status is not None
    assert status.contract_slew_rate_rad_s2 is None
    assert status.hardware_slew_rate_rad_s2 == pytest.approx(0.6)
    assert status.hardware_action_clip_applied is True
    assert status.hardware_slew_applied is True
    assert status.hardware_tightening_applied is True
    assert np.max(np.abs(status.raw_contract_target_main_drive_velocity)) > 0.01
    np.testing.assert_allclose(
        result.target_main_drive_velocity,
        status.hardware_target_main_drive_velocity,
    )
    assert np.max(np.abs(result.target_main_drive_velocity)) <= 0.0100001
