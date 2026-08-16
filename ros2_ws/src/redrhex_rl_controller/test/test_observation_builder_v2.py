from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from redrhex_rl_controller.observation_builder_v2 import (
    ABAD_JOINT_NAMES_V2,
    HISTORY_LENGTH_V2,
    MAIN_JOINT_NAMES_V2,
    SensorObservationBuilderV2,
)


def _config(mode="validated_quaternion"):
    return {
        "attitude_mode": mode,
        "imu_frame_id": "imu_link",
        "imu_mount_calibration_verified": True,
        "rest_gravity_verified": True,
        "expected_rest_projected_gravity": [0.0, 0.0, -1.0],
        "imu_mount_rpy_deg": [0.0, 0.0, 0.0],
        "require_joint_validity": True,
    }


def _header(stamp, frame_id=""):
    sec = int(stamp)
    nanosec = int(round((stamp - sec) * 1e9))
    return SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nanosec), frame_id=frame_id)


def _imu(stamp, *, frame_id="imu_link", covariance=None):
    return SimpleNamespace(
        header=_header(stamp, frame_id),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        orientation_covariance=[0.01] + [0.0] * 8 if covariance is None else covariance,
        angular_velocity=SimpleNamespace(x=0.1, y=0.2, z=0.3),
        linear_acceleration=SimpleNamespace(x=0.0, y=0.0, z=9.81),
    )


def _joints(stamp, main_position=0.0, include_abad=True):
    names = list(MAIN_JOINT_NAMES_V2)
    positions = [main_position + index * 0.01 for index in range(6)]
    if include_abad:
        names += list(ABAD_JOINT_NAMES_V2)
        positions += [0.02 * index for index in range(6)]
    return SimpleNamespace(
        header=_header(stamp, "redrhex_policy_body"),
        name=names,
        position=positions,
        velocity=[0.0] * len(names),
    )


def _mark_valid(builder, stamp):
    assert builder.update_joint_validity({name: True for name in builder.all_joint_names}, stamp)


def test_strict_frame_contains_only_36_physical_feedback_features():
    builder = SensorObservationBuilderV2(_config())
    assert builder.update_imu(_imu(1.0))
    assert builder.update_joint_state(_joints(1.0))
    _mark_valid(builder, 1.0)
    frame = builder.build_sensor_frame(1.01)
    assert frame.shape == (36,)
    assert frame[:3] == pytest.approx([0.1, 0.2, 0.3])
    assert frame[3:6] == pytest.approx([0.0, 0.0, -1.0])
    assert frame[6:12] == pytest.approx(np.sin(np.arange(6) * 0.01))
    assert frame[24:30] == pytest.approx(np.arange(6) * 0.02)


def test_history_is_unpadded_oldest_to_newest_and_requires_new_sensor_events():
    builder = SensorObservationBuilderV2(_config())
    first_frame = None
    last_frame = None
    for index in range(HISTORY_LENGTH_V2):
        stamp = 1.0 + index / 60.0
        assert builder.update_imu(_imu(stamp))
        assert builder.update_joint_state(_joints(stamp, main_position=index * 0.001))
        _mark_valid(builder, stamp)
        frame = builder.append_sensor_frame(stamp)
        first_frame = frame.copy() if first_frame is None else first_frame
        last_frame = frame.copy()
    inputs = builder.policy_inputs(1.0 + 59.0 / 60.0)
    assert inputs.sensor_history.shape == (60, 36)
    assert np.array_equal(inputs.sensor_history[0], first_frame)
    assert np.array_equal(inputs.sensor_history[-1], last_frame)
    with pytest.raises(RuntimeError, match="repeated"):
        builder.append_sensor_frame(2.0)
    assert builder.history_ready is False


def test_missing_abad_or_invalid_channel_blocks_and_resets_history():
    builder = SensorObservationBuilderV2(_config())
    assert builder.update_imu(_imu(1.0))
    assert builder.update_joint_state(_joints(1.0, include_abad=False))
    _mark_valid(builder, 1.0)
    status = builder.status(1.01)
    assert status.ok is False
    assert any("missing measured joint" in reason for reason in status.reasons)

    assert builder.update_joint_state(_joints(1.02))
    validity = {name: True for name in builder.all_joint_names}
    validity[ABAD_JOINT_NAMES_V2[2]] = False
    assert builder.update_joint_validity(validity, 1.02) is False
    assert builder.history_ready is False


def test_stale_dropout_and_out_of_order_data_clear_history():
    builder = SensorObservationBuilderV2(_config())
    assert builder.update_imu(_imu(1.0))
    assert builder.update_joint_state(_joints(1.0))
    _mark_valid(builder, 1.0)
    builder.append_sensor_frame(1.0)
    assert builder.history_size == 1
    assert builder.status(1.2).ok is False
    assert builder.history_size == 0

    assert builder.update_imu(_imu(2.0))
    assert builder.update_joint_state(_joints(2.0))
    _mark_valid(builder, 2.0)
    builder.prime_velocity_baseline()
    next_stamp = 2.0 + 1.0 / 60.0
    assert builder.update_imu(_imu(next_stamp))
    assert builder.update_joint_state(_joints(next_stamp))
    _mark_valid(builder, next_stamp)
    builder.append_sensor_frame(next_stamp)
    assert builder.update_imu(_imu(1.9)) is False
    assert builder.history_size == 0


def test_validated_quaternion_has_no_covariance_or_frame_fallback():
    builder = SensorObservationBuilderV2(_config())
    assert builder.update_imu(_imu(1.0, covariance=[-1.0] + [0.0] * 8)) is False
    assert builder.update_imu(_imu(1.1, frame_id="wrong_frame")) is False
    assert builder.status(1.1).ok is False


def test_validated_quaternion_rejects_all_zero_unknown_covariance():
    builder = SensorObservationBuilderV2(_config())
    assert builder.update_imu(_imu(1.0, covariance=[0.0] * 9)) is False
    assert builder.history_ready is False


def test_runtime_can_prime_velocity_without_adding_a_fake_history_frame():
    builder = SensorObservationBuilderV2(_config())
    assert builder.update_imu(_imu(1.0))
    assert builder.update_joint_state(_joints(1.0))
    _mark_valid(builder, 1.0)
    assert builder.has_complete_new_sensor_generation is True
    builder.prime_velocity_baseline()
    assert builder.history_size == 0
    assert builder.velocity_baseline_required is False

    assert builder.update_imu(_imu(1.0 + 1.0 / 60.0))
    assert builder.has_complete_new_sensor_generation is False
    assert builder.update_joint_state(
        _joints(1.0 + 1.0 / 60.0, main_position=0.01)
    )
    _mark_valid(builder, 1.0 + 1.0 / 60.0)
    assert builder.has_complete_new_sensor_generation is True
    frame = builder.append_sensor_frame(builder.latest_sensor_source_time_s)
    assert builder.history_size == 1
    assert frame[18:24] == pytest.approx([0.6] * 6)


def test_imu_joint_source_skew_rejects_generation_and_resets_history():
    config = _config()
    config["max_sensor_source_skew_s"] = 0.5 / 60.0
    builder = SensorObservationBuilderV2(config)
    assert builder.update_imu(_imu(1.0))
    assert builder.update_joint_state(_joints(1.0))
    _mark_valid(builder, 1.0)
    builder.append_sensor_frame(1.0)
    assert builder.history_size == 1

    imu_stamp = 1.0 + 1.0 / 60.0
    joint_stamp = imu_stamp + 0.01
    assert builder.update_imu(_imu(imu_stamp))
    assert builder.update_joint_state(_joints(joint_stamp, main_position=0.01))
    _mark_valid(builder, joint_stamp)
    with pytest.raises(RuntimeError, match="source skew"):
        builder.append_sensor_frame(builder.latest_sensor_source_time_s)
    assert builder.history_size == 0
    assert builder.velocity_baseline_required is True


def test_partial_generation_waits_without_triggering_false_source_skew():
    builder = SensorObservationBuilderV2(_config())
    assert builder.update_imu(_imu(1.0))
    assert builder.update_joint_state(_joints(1.0))
    _mark_valid(builder, 1.0)
    builder.append_sensor_frame(1.0)

    next_stamp = 1.0 + 1.0 / 60.0
    assert builder.update_imu(_imu(next_stamp))
    assert builder.has_complete_new_sensor_generation is False
    assert builder.status(next_stamp).ok is True
    assert builder.history_size == 1

    assert builder.update_joint_state(_joints(next_stamp, main_position=0.01))
    _mark_valid(builder, next_stamp)
    builder.append_sensor_frame(next_stamp)
    assert builder.history_size == 2


def test_30_hz_generation_cadence_cannot_fill_60_hz_history():
    config = _config()
    config["max_history_period_error_ratio"] = 0.25
    builder = SensorObservationBuilderV2(config)
    assert builder.update_imu(_imu(1.0))
    assert builder.update_joint_state(_joints(1.0))
    _mark_valid(builder, 1.0)
    builder.prime_velocity_baseline()

    slow_stamp = 1.0 + 1.0 / 30.0
    assert builder.update_imu(_imu(slow_stamp))
    assert builder.update_joint_state(_joints(slow_stamp, main_position=0.01))
    _mark_valid(builder, slow_stamp)
    with pytest.raises(RuntimeError, match="source cadence"):
        builder.append_sensor_frame(builder.latest_sensor_source_time_s)
    assert builder.history_size == 0
    assert builder.velocity_baseline_required is True


def test_causal_mode_uses_gyro_accel_without_quaternion_fallback():
    config = _config("causal_gyro_accel")
    config["rest_gravity_verified"] = False
    builder = SensorObservationBuilderV2(config)
    message = _imu(1.0, covariance=[-1.0] + [0.0] * 8)
    message.orientation = SimpleNamespace(x=float("nan"), y=0.0, z=0.0, w=0.0)
    assert builder.update_imu(message) is True
    assert builder.projected_gravity_body() == pytest.approx([0.0, 0.0, -1.0])


def test_bridge_validity_diagnostic_requires_every_canonical_joint():
    builder = SensorObservationBuilderV2(_config())
    statuses = [
        SimpleNamespace(
            values=[
                SimpleNamespace(key="joint_name", value=name),
                SimpleNamespace(key="valid", value="true"),
            ]
        )
        for name in builder.all_joint_names
    ]
    message = SimpleNamespace(header=_header(1.0), status=statuses)
    assert builder.update_joint_validity_diagnostic(message) is True
    message.header = _header(1.1)
    message.status = statuses[:-1]
    assert builder.update_joint_validity_diagnostic(message) is False
