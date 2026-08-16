from __future__ import annotations

from dataclasses import replace
import math
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
POLICY_IO_ROOT = REPO_ROOT / "source" / "redrhex_policy_io"
AGENTS_ROOT = (
    REPO_ROOT
    / "source"
    / "RedRhex"
    / "RedRhex"
    / "tasks"
    / "direct"
    / "redrhex"
    / "agents"
)
for root in (POLICY_IO_ROOT, AGENTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from redrhex_policy_io import (  # noqa: E402
    ForwardResidualActionContractV2,
    SensorCalibrationProfileV2,
    SensorFrameBuilderV2,
    SensorHistoryBufferV2,
    StudentObservationContractV2,
    build_sensor_frame_numpy,
    decode_forward_residual_action_v2,
    transform_imu_vector,
    validate_calibration_lineage_v2,
    wrap_angle,
    wrapped_velocity,
)


torch = pytest.importorskip("torch")

from redrhex_policy_io import build_sensor_frame_torch  # noqa: E402
from sensor_v2.checkpoint import CheckpointIntentV2, load_checkpoint_v2  # noqa: E402
from sensor_v2.models import CausalTCNEncoderV2, SensorStudentCoreV2  # noqa: E402


def test_encoder_wrap_and_wrapped_velocity_are_continuous() -> None:
    assert wrap_angle(math.pi) == pytest.approx(-math.pi)
    velocity = wrapped_velocity(
        np.asarray([-math.pi + 0.1]),
        np.asarray([math.pi - 0.1]),
        0.1,
    )
    np.testing.assert_allclose(velocity, [2.0], atol=1.0e-12)


def test_frame_builder_wraps_main_but_not_bounded_abad_velocity() -> None:
    contract = StudentObservationContractV2.causal_gyro_accel()
    builder = SensorFrameBuilderV2(contract)
    common = {
        "imu_gyro_rad_s": [0.0, 0.0, 0.0],
        "imu_linear_accel_m_s2": [0.0, 0.0, 9.80665],
    }
    builder.build(
        timestamp_s=1.0,
        main_position_rad=[math.pi - 0.1] * 6,
        abad_position_rad=[3.0] * 6,
        **common,
    )
    frame = builder.build(
        timestamp_s=1.1,
        main_position_rad=[-math.pi + 0.1] * 6,
        abad_position_rad=[-3.0] * 6,
        **common,
    )
    np.testing.assert_allclose(frame[18:24], [2.0] * 6, atol=1.0e-5)
    np.testing.assert_allclose(frame[30:36], [-60.0] * 6, atol=1.0e-5)


def test_imu_mount_rotation_is_policy_body_frame() -> None:
    half = math.sqrt(0.5)
    rotated = transform_imu_vector([1.0, 0.0, 0.0], [half, 0.0, 0.0, half])
    np.testing.assert_allclose(rotated, [0.0, 1.0, 0.0], atol=1.0e-7)


def test_sensor_layout_history_and_forbidden_inputs_are_explicit() -> None:
    contract = StudentObservationContractV2.causal_gyro_accel()
    assert contract.sensor_frame_dim == 36
    assert contract.history_length == 60
    assert contract.feature_slices == {
        "body_gyro": slice(0, 3),
        "projected_gravity": slice(3, 6),
        "main_position_sin": slice(6, 12),
        "main_position_cos": slice(12, 18),
        "main_velocity": slice(18, 24),
        "abad_position": slice(24, 30),
        "abad_velocity": slice(30, 36),
    }
    forbidden = set(contract.to_dict()["forbidden_actor_inputs"])
    assert {
        "true_base_velocity",
        "odometry",
        "gait_clock",
        "previous_action",
        "commanded_abad",
        "internal_controller_targets",
    }.issubset(forbidden)
    assert all(spec.actor_allowed and not spec.privileged_only for spec in contract.FEATURE_LAYOUT)


def test_history_is_oldest_to_newest_and_never_ready_when_zero_padded() -> None:
    contract = StudentObservationContractV2.causal_gyro_accel()
    history = SensorHistoryBufferV2(contract)
    for index in range(59):
        history.append(np.full(36, index, dtype=np.float32))
    assert not history.ready
    with pytest.raises(ValueError, match="history is not ready"):
        history.array()
    history.append(np.full(36, 59, dtype=np.float32))
    assert history.ready
    np.testing.assert_array_equal(history.array()[:, 0], np.arange(60))
    history.append(np.full(36, 60, dtype=np.float32))
    np.testing.assert_array_equal(history.array()[:, 0], np.arange(1, 61))


def test_numpy_and_torch_frame_preprocessing_match() -> None:
    rng = np.random.default_rng(42)
    values = [rng.normal(size=(4, width)) for width in (3, 3, 6, 6, 6, 6)]
    gravity = values[1]
    gravity /= np.linalg.norm(gravity, axis=1, keepdims=True)
    numpy_frames = np.stack(
        [
            build_sensor_frame_numpy(*(value[row] for value in values))
            for row in range(4)
        ]
    )
    torch_frames = build_sensor_frame_torch(
        *(torch.from_numpy(value) for value in values)
    )
    np.testing.assert_allclose(numpy_frames, torch_frames.numpy(), rtol=1.0e-6, atol=1.0e-6)


def test_contract_round_trip_and_calibration_require_all_twelve_scales() -> None:
    observation = StudentObservationContractV2.causal_gyro_accel()
    action = ForwardResidualActionContractV2()
    assert StudentObservationContractV2.from_dict(
        observation.to_dict(include_sha256=True)
    ) == observation
    provisional = SensorCalibrationProfileV2.provisional(observation, action)
    assert not provisional.hardware_ready
    assert "main_encoder_0_counts_per_rad" in provisional.readiness_blockers
    assert "abad_encoder_0_counts_per_rad" in provisional.readiness_blockers


def test_training_and_runtime_calibration_share_sensor_contract_lineage() -> None:
    observation = StudentObservationContractV2.causal_gyro_accel()
    action = ForwardResidualActionContractV2()
    training = SensorCalibrationProfileV2.provisional(observation, action)
    runtime = SensorCalibrationProfileV2(
        profile_id="measured-runtime",
        observation_contract_sha256=observation.sha256,
        action_contract_sha256=action.sha256,
        attitude_mode=observation.attitude_mode,
        imu_frame_id=observation.imu_frame_id,
        imu_to_body_wxyz=observation.imu_to_body_wxyz,
        main_counts_per_rad=(1000.0,) * 6,
        abad_counts_per_rad=(1000.0,) * 6,
        main_encoder_evidence=("fixture",) * 6,
        abad_encoder_evidence=("fixture",) * 6,
        imu_mount_evidence="fixture",
        rest_gravity_evidence="fixture",
    )
    validated_training, validated_runtime = validate_calibration_lineage_v2(
        training,
        runtime,
        observation_contract=observation,
        action_contract=action,
        require_runtime_hardware_ready=True,
    )
    assert validated_training.sha256 == training.sha256
    assert validated_runtime.sha256 == runtime.sha256

    incompatible = replace(
        runtime,
        imu_to_body_wxyz=(0.0, 1.0, 0.0, 0.0),
    )
    with pytest.raises(ValueError, match="imu_to_body_wxyz"):
        validate_calibration_lineage_v2(
            training,
            incompatible,
            observation_contract=observation,
            action_contract=action,
            require_runtime_hardware_ready=True,
        )


def test_v2_checkpoint_loader_rejects_legacy_shape_only_payload(tmp_path: Path) -> None:
    path = tmp_path / "legacy.pt"
    torch.save({"model_state_dict": SensorStudentCoreV2().state_dict()}, path)
    with pytest.raises(ValueError, match="not a Sensor V2 checkpoint"):
        load_checkpoint_v2(
            path,
            model=SensorStudentCoreV2(),
            intent=CheckpointIntentV2.INFERENCE,
        )


def test_tcn_and_auxiliary_heads_have_deployment_shapes() -> None:
    history = torch.randn(2, 60, 36)
    command = torch.randn(2, 3)
    encoder = CausalTCNEncoderV2()
    assert encoder(history).shape == (2, 64)
    student = SensorStudentCoreV2()
    actions, velocity, next_frame = student(history, command)
    assert actions.shape == (2, 12)
    assert velocity.shape == (2, 3)
    assert next_frame.shape == (2, 36)
    assert torch.count_nonzero(actions[:, 6:]) == 0


def test_forward_decoder_uses_effective_leg_angle_before_phase_lock() -> None:
    contract = ForwardResidualActionContractV2()
    phase = 0.7
    offsets = np.zeros(6)
    offsets[list(contract.TRIPOD_B)] = contract.TRIPOD_PHASE_OFFSET_RAD
    time_fraction = np.mod(phase + offsets, 2.0 * math.pi) / (2.0 * math.pi)
    in_stance = time_fraction < contract.STANCE_DUTY_CYCLE
    stance_start = contract.STANCE_PHASE_START_RAD % (2.0 * math.pi)
    stance_arc = (
        contract.STANCE_PHASE_END_RAD - contract.STANCE_PHASE_START_RAD
    ) % (2.0 * math.pi)
    desired = np.mod(
        np.where(
            in_stance,
            stance_start
            + stance_arc * time_fraction / contract.STANCE_DUTY_CYCLE,
            stance_start
            + stance_arc
            + (2.0 * math.pi - stance_arc)
            * (time_fraction - contract.STANCE_DUTY_CYCLE)
            / (1.0 - contract.STANCE_DUTY_CYCLE),
        ),
        2.0 * math.pi,
    )
    direction = np.asarray(contract.LEG_DIRECTION_MULTIPLIER)
    measured = desired / direction
    decoded = decode_forward_residual_action_v2(
        np.zeros(12),
        [contract.forward_command_reference_m_s, 0.0, 0.0],
        phase,
        measured,
        contract=contract,
    )
    desired_profile = np.where(
        in_stance,
        2.0 * math.pi * contract.NOMINAL_GAIT_FREQUENCY_HZ * contract.STANCE_VELOCITY_RATIO,
        2.0 * math.pi * contract.NOMINAL_GAIT_FREQUENCY_HZ * contract.SWING_VELOCITY_RATIO,
    )
    np.testing.assert_allclose(
        decoded.nominal_main_velocity_rad_s,
        desired_profile * direction,
        atol=1.0e-12,
    )
