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
    position_time = np.array([0.0, duration_s])
    positions = np.array(
        [
            [0.10, 0.20, 0.30, -0.10, -0.20, -0.30],
            [0.11, 0.20, 0.30, -0.10, -0.20, -0.30],
        ]
    )
    state = {
        "schema_version": 1,
        "joint_order": [f"main_{index}" for index in range(6)],
        "position_source_channel": "main_joint_position_canonical",
        "position_rad": positions[0].tolist(),
        "velocity_rad_s": [0.0] * 6,
        "velocity_source": "reviewed_initial_neutral",
        "fixture_mode": "fixed_base",
        "fixture_frame": "world",
        "root_pose_source": "production_asset_default",
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
            },
            "frames": {
                "command": "main_0",
                "position": "main_0",
                "main_joint_position_canonical": "canonical_main_joint_order",
            },
            "calibration_constants": {
                "position_mapping_source": source,
                "requested_command_source": source,
                "all_main_position_mapping_source": source,
                "replay_initial_state": state,
                "replay_initial_state_sha256": sha256_json(state),
            },
        },
        time_bases={"main_joint_position_canonical": "position_time_s"},
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
    assert replay.initial_state.velocity_rad_s == (0.0,) * 6
    assert replay.initial_state.fixture_mode == "fixed_base"
    assert replay.initial_state.fixture_frame == "world"


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
