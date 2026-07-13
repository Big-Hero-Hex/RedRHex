from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.sim2real.characterization import (
    load_replay_schedule,
    scenario_schedule,
    scenario_step_count,
)
from tools.sim2real.contracts import CalibrationProfileV1, ContractError
from tools.sim2real.import_real import derive_replay_initial_state
from tools.sim2real.scenarios import load_scenario
from tools.sim2real.traces import sha256_file, sha256_json, sha256_path, write_trace


def _profile() -> CalibrationProfileV1:
    joints = [f"main_{index}" for index in range(6)]
    return CalibrationProfileV1.from_dict(
        {
            "schema_version": 1,
            "profile_id": "all-main-encoder-audit",
            "hardware_mapping": {
                "encoder_counts_per_rev": {joint: 54984.83 for joint in joints},
                "encoder_zero_count": {joint: 0.0 for joint in joints},
                "encoder_sign": {joint: 1 for joint in joints},
                "joint_direction": {"main_0": 1},
                "pwm_scale": {"main_0": 1.0 / 120.0},
                "pwm_cap": {"main_0": 500.0 / 120.0},
            },
            "sensor_timing": {},
            "simulation_physics": {},
        }
    )


def _managed_replay(tmp_path: Path, *, provisional: bool = False) -> Path:
    scenario = load_scenario("main-step")
    profile = _profile()
    steps = scenario_step_count(scenario)
    duration_s = steps / 120.0
    command_time = np.arange(0.0, duration_s + 1.0e-12, 1.0 / 60.0)
    schedule = scenario_schedule(scenario, steps)
    command = np.asarray(
        [schedule[min(int(value * 120.0), steps - 1)].value for value in command_time]
    )
    position_time = np.array([0.0, 0.1, 0.2, 0.3, 0.4, duration_s])
    initial_positions = np.array([0.10, 0.20, 0.30, -0.10, -0.20, -0.30])
    fitted_velocity = np.array([0.001, -0.002, 0.0, 0.003, -0.001, 0.002])
    positions = initial_positions + position_time[:, None] * fitted_velocity
    imu_time = position_time.copy()
    imu_orientation_xyzw = np.tile(
        np.array([0.7071067811865475, 0.0, 0.0, 0.7071067811865476]),
        (imu_time.size, 1),
    )
    imu_angular_velocity = np.zeros((imu_time.size, 3))
    fixture = {
        "schema_version": 1,
        "fixture_id": "suspended-level-v1",
        "scene_mode": "fixed_base",
        "fixture_frame": "world",
        "root_orientation_wxyz": [
            0.7071067811865476,
            0.7071067811865475,
            0.0,
            0.0,
        ],
        "expected_imu_orientation_xyzw": [
            0.7071067811865475,
            0.0,
            0.0,
            0.7071067811865476,
        ],
    }
    state = {
        "schema_version": 2,
        "joint_order": [f"main_{index}" for index in range(6)],
        "position_source_channel": "main_joint_position_canonical",
        "position_rad": positions[0].tolist(),
        "velocity_rad_s": fitted_velocity.tolist(),
        "velocity_source": "stationary_window_linear_fit",
        "velocity_window_start_s": 0.0,
        "velocity_window_end_s": 0.4,
        "velocity_stationarity_limit_rad_s": 0.05,
        "fixture_mode": "fixed_base",
        "fixture_frame": "world",
        "root_pose_source": "reviewed_fixture",
        "fixture_id": fixture["fixture_id"],
        "fixture_sha256": sha256_json(fixture),
        "root_orientation_wxyz": fixture["root_orientation_wxyz"],
        "imu_orientation_source_channel": "imu_orientation_xyzw",
        "measured_imu_orientation_xyzw": fixture[
            "expected_imu_orientation_xyzw"
        ],
        "expected_imu_orientation_xyzw": fixture[
            "expected_imu_orientation_xyzw"
        ],
        "imu_orientation_error_rad": 0.0,
        "imu_orientation_tolerance_rad": np.deg2rad(5.0),
        "imu_angular_velocity_source_channel": "imu_angular_velocity",
        "max_imu_angular_speed_rad_s": 0.0,
        "imu_stationarity_limit_rad_s": 0.1,
        "sample_time_s": 0.0,
        "scenario_time_s": 0.0,
        "sample_offset_s": 0.0,
    }
    dataset = tmp_path / "datasets" / "sim2real" / "replay-data"
    raw = dataset / "raw" / "run.raw"
    episode = dataset / "episodes" / "run-1"
    raw.parent.mkdir(parents=True)
    episode.parent.mkdir(parents=True)
    raw.write_bytes(b"immutable rosbag evidence")
    source = "provisional_repository_defaults" if provisional else f"profile:{profile.profile_id}"
    manifest = write_trace(
        episode,
        {
            "command_time_s": command_time,
            "command": command,
            "position_time_s": position_time,
            "position": positions[:, 0],
            "main_joint_position_canonical": positions,
            "imu_time_s": imu_time,
            "imu_orientation_xyzw": imu_orientation_xyzw,
            "imu_angular_velocity": imu_angular_velocity,
        },
        scenario=scenario,
        source="real",
        source_path=raw,
        profile=profile,
        metadata={
            "units": {
                "command": "rad/s",
                "position": "rad",
                "main_joint_position_canonical": "rad",
                "imu_orientation_xyzw": "quaternion_xyzw",
                "imu_angular_velocity": "rad/s",
            },
            "frames": {
                "command": "main_0",
                "position": "main_0",
                "main_joint_position_canonical": "canonical_main_joint_order",
                "imu_orientation_xyzw": "imu_mount",
                "imu_angular_velocity": "imu_mount",
            },
            "calibration_constants": {
                "position_mapping_source": source,
                "requested_command_source": source,
                "all_main_position_mapping_source": source,
                "replay_initial_state": state,
                "replay_initial_state_sha256": sha256_json(state),
            },
        },
        time_bases={
            "main_joint_position_canonical": "position_time_s",
            "imu_orientation_xyzw": "imu_time_s",
            "imu_angular_velocity": "imu_time_s",
        },
    )
    dataset_manifest = {
        "schema_version": 1,
        "dataset_id": "replay-data",
        "raw": [{"path": "raw/run.raw", "sha256": sha256_path(raw)}],
        "episodes": [
            {
                "episode_id": "run-1",
                "scenario_id": scenario.scenario_id,
                "path": "episodes/run-1",
                "trace_sha256": manifest.provenance["trace_sha256"],
                "metadata_sha256": sha256_file(episode / "metadata.json"),
                "raw_path": "raw/run.raw",
            }
        ],
    }
    (dataset / "manifest.json").write_text(
        json.dumps(dataset_manifest), encoding="utf-8"
    )
    return episode


def _derived_state_inputs() -> dict:
    scenario = load_scenario("main-step")
    time = np.array([10.0, 10.1, 10.2, 10.3, 10.4])
    initial = np.array([0.10, 0.20, 0.30, -0.10, -0.20, -0.30])
    velocity = np.array([0.001, -0.002, 0.0, 0.003, -0.001, 0.002])
    imu_orientation = np.tile(
        np.array([0.7071067811865475, 0.0, 0.0, 0.7071067811865476]),
        (time.size, 1),
    )
    return {
        "position_time_s": time,
        "canonical_position_rad": initial + (time - 10.0)[:, None] * velocity,
        "imu_time_s": time,
        "imu_orientation_xyzw": imu_orientation,
        "imu_angular_velocity_rad_s": np.zeros((time.size, 3)),
        "scenario_start_s": 10.0,
        "time_origin_s": 9.5,
        "scenario": scenario,
        "fixture": {
            "schema_version": 1,
            "fixture_id": "suspended-level-v1",
            "scene_mode": "fixed_base",
            "fixture_frame": "world",
            "root_orientation_wxyz": [
                0.7071067811865476,
                0.7071067811865475,
                0.0,
                0.0,
            ],
            "expected_imu_orientation_xyzw": imu_orientation[0].tolist(),
        },
    }


def test_import_derives_velocity_and_fixture_from_stationary_sensor_window() -> None:
    state = derive_replay_initial_state(**_derived_state_inputs())

    assert state["schema_version"] == 2
    assert state["velocity_rad_s"] == pytest.approx(
        [0.001, -0.002, 0.0, 0.003, -0.001, 0.002]
    )
    assert state["scenario_time_s"] == pytest.approx(0.5)
    assert state["root_pose_source"] == "reviewed_fixture"
    assert len(state["fixture_sha256"]) == 64


@pytest.mark.parametrize("failure", ["joint_motion", "imu_motion", "fixture_pose"])
def test_import_rejects_unverified_replay_initial_state(failure: str) -> None:
    inputs = _derived_state_inputs()
    if failure == "joint_motion":
        inputs["canonical_position_rad"][:, 0] += np.linspace(0.0, 0.1, 5)
        expected = "joint state is not stationary"
    elif failure == "imu_motion":
        inputs["imu_angular_velocity_rad_s"][:, 2] = 0.2
        expected = "not stationary according to the IMU"
    else:
        inputs["fixture"] = {
            **inputs["fixture"],
            "expected_imu_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
        expected = "orientation does not match"

    with pytest.raises(ContractError, match=expected):
        derive_replay_initial_state(**inputs)


def test_replay_binds_all_observable_main_joints_and_fixture(tmp_path: Path) -> None:
    scenario = load_scenario("main-step")
    replay = load_replay_schedule(
        _managed_replay(tmp_path),
        scenario,
        steps=scenario_step_count(scenario),
    )

    assert replay.initial_state.joint_order == tuple(
        f"main_{index}" for index in range(6)
    )
    assert replay.initial_state.position_rad == pytest.approx(
        (0.10, 0.20, 0.30, -0.10, -0.20, -0.30)
    )
    assert replay.initial_state.velocity_rad_s == pytest.approx(
        (0.001, -0.002, 0.0, 0.003, -0.001, 0.002)
    )
    assert replay.initial_state.fixture_mode == "fixed_base"
    assert replay.initial_state.fixture_frame == "world"
    assert replay.initial_state.root_orientation_wxyz == pytest.approx(
        (0.7071067811865476, 0.7071067811865475, 0.0, 0.0)
    )


def _rewrite_initial_state(episode: Path, update) -> None:
    metadata_path = episode / "metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    constants = payload["metadata"]["calibration_constants"]
    state = constants["replay_initial_state"]
    update(state)
    constants["replay_initial_state_sha256"] = sha256_json(state)
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    dataset_manifest_path = episode.parents[1] / "manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    dataset_manifest["episodes"][0]["metadata_sha256"] = sha256_file(metadata_path)
    dataset_manifest_path.write_text(json.dumps(dataset_manifest), encoding="utf-8")


def test_replay_rejects_velocity_not_derived_from_stationary_window(
    tmp_path: Path,
) -> None:
    scenario = load_scenario("main-step")
    episode = _managed_replay(tmp_path)
    _rewrite_initial_state(episode, lambda state: state.__setitem__("velocity_rad_s", [0.0] * 6))

    with pytest.raises(ContractError, match="velocity.*stationary window"):
        load_replay_schedule(
            episode, scenario, steps=scenario_step_count(scenario)
        )


def test_replay_rejects_fixture_orientation_not_bound_to_imu_trace(
    tmp_path: Path,
) -> None:
    scenario = load_scenario("main-step")
    episode = _managed_replay(tmp_path)
    _rewrite_initial_state(
        episode,
        lambda state: state.__setitem__(
            "measured_imu_orientation_xyzw", [0.0, 0.0, 0.0, 1.0]
        ),
    )

    with pytest.raises(ContractError, match="IMU orientation.*trace"):
        load_replay_schedule(
            episode, scenario, steps=scenario_step_count(scenario)
        )


def test_replay_requires_managed_dataset_and_measured_mapping(tmp_path: Path) -> None:
    scenario = load_scenario("main-step")
    managed = _managed_replay(tmp_path)
    standalone = tmp_path / "standalone"
    standalone.mkdir()
    for name in ("trace.npz", "metadata.json"):
        (standalone / name).write_bytes((managed / name).read_bytes())

    with pytest.raises(ContractError, match="managed dataset"):
        load_replay_schedule(
            standalone, scenario, steps=scenario_step_count(scenario)
        )

    with pytest.raises(ContractError, match="mapping provenance"):
        load_replay_schedule(
            _managed_replay(tmp_path / "provisional", provisional=True),
            scenario,
            steps=scenario_step_count(scenario),
        )
