from __future__ import annotations

import copy
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tools.sim2real.cli import main
from tools.sim2real.compare import compare_traces
from tools.sim2real.contracts import CalibrationProfileV1, ContractError, load_profile
from tools.sim2real.metrics import compute_subsystem_metrics
from tools.sim2real.profile_measurements import apply_measurements_to_profile
from tools.sim2real.promotion import (
    _validate_changed_field_evidence,
    _validate_measurement_sources,
    evaluate_promotion,
)
from tools.sim2real.scenarios import load_scenario
from tools.sim2real.traces import (
    load_trace,
    sha256_file,
    sha256_json,
    sha256_path,
    write_trace,
)


def _profile(profile_id: str, damping: float) -> CalibrationProfileV1:
    joints = [f"main_{index}" for index in range(6)]
    return CalibrationProfileV1.from_dict(
        {
            "schema_version": 1,
            "profile_id": profile_id,
            "hardware_mapping": {
                "encoder_counts_per_rev": {joint: 54984.83 for joint in joints},
                "encoder_zero_count": {joint: 0.0 for joint in joints},
                "encoder_sign": {joint: 1 for joint in joints},
                "joint_direction": {joint: 1 for joint in joints},
                "pwm_scale": {joint: 1.0 / 120.0 for joint in joints},
                "pwm_cap": {joint: 500.0 / 120.0 for joint in joints},
            },
            "sensor_timing": {},
            "simulation_physics": {"main_drive": {"damping": damping}},
        }
    )


def _response_trace(
    directory: Path,
    *,
    scenario_id: str,
    source: str,
    speed_scale: float,
    profile: CalibrationProfileV1 | None = None,
    load_coordinate: str | None = None,
    runtime_provenance: dict[str, str] | None = None,
    replay_ready: bool = False,
) -> None:
    scenario = load_scenario(scenario_id)
    scenario_hash = sha256_json(scenario.to_dict())
    dt = 0.05
    cycle_s = sum(float(segment["duration_s"]) for segment in scenario.command_segments)
    duration_s = cycle_s * scenario.repeats
    time_s = np.arange(0.0, duration_s + dt * 0.5, dt)
    command = np.zeros_like(time_s)
    velocity = np.zeros_like(time_s)
    cumulative = np.cumsum([float(item["duration_s"]) for item in scenario.command_segments])
    for index, sample_time in enumerate(time_s):
        local = min(sample_time, duration_s - np.finfo(float).eps) % cycle_s
        segment_index = min(int(np.searchsorted(cumulative, local, side="right")), len(cumulative) - 1)
        command[index] = float(scenario.command_segments[segment_index]["value"])
        # The reviewed step/coast sequence uses fixed boundaries within each 5.5 s cycle.
        if 0.6 <= local < 1.5:
            velocity[index] = 2.0 * speed_scale
        elif 1.5 <= local < 1.9:
            velocity[index] = 2.0 * speed_scale * (1.9 - local) / 0.4
        elif 3.1 <= local < 4.0:
            velocity[index] = -1.5 * speed_scale
        elif 4.0 <= local < 4.4:
            velocity[index] = -1.5 * speed_scale * (4.4 - local) / 0.4
    position = np.cumsum(velocity) * dt
    metadata = {
        "units": {"command": "rad/s", "position": "rad"},
        "frames": {"command": scenario.joint, "position": scenario.joint},
        "joint_order": [scenario.joint],
        "clock": {
            "source": "test",
            "timestamp_semantics": "relative_monotonic",
            "time_unit": "s",
        },
        "git_sha": None,
        "asset_sha256": None,
        "config_sha256": None,
        **dict(runtime_provenance or {}),
        "calibration_constants": {
            "position_mapping_source": (
                f"profile:{profile.profile_id}" if source == "real" and profile else "synthetic"
            ),
            "requested_command_source": (
                f"authenticated_probe_events:{scenario_hash}"
                if source == "real"
                else "synthetic"
            ),
            "probe_event_evidence": {
                "scenario_sha256": scenario_hash,
                "repetition_count": 3,
                "segment_count": 21,
                "complete_ticks": 990,
                "abad_output_disabled_verified": True,
                "receive_duration_s": duration_s,
                "receive_jitter_bound_s": 1.0 / 60.0,
            },
            **(
                {"condition_coordinates": {"load": load_coordinate}}
                if load_coordinate is not None
                else {}
            ),
        },
    }
    source_path = None
    if source == "real":
        if directory.parent.name == "episodes":
            dataset_root = directory.parent.parent
            (dataset_root / "raw").mkdir(parents=True, exist_ok=True)
            source_path = dataset_root / "raw" / f"{directory.name}.raw"
        else:
            source_path = directory.parent / f"{directory.name}.raw"
        source_path.write_bytes(f"raw:{directory.name}".encode())
    arrays = {
        "command_time_s": time_s,
        "command": command,
        "position_time_s": time_s,
        "position": position,
    }
    time_bases = None
    if replay_ready:
        if source != "real" or profile is None:
            raise AssertionError("replay-ready fixture requires a profiled real trace")
        selected_index = int(scenario.joint.removeprefix("main_"))
        canonical_position = np.tile(
            np.array([0.0, 0.2, 0.3, -0.1, -0.2, -0.3]),
            (time_s.size, 1),
        )
        canonical_position[:, selected_index] += position
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
        initial_state = {
            "schema_version": 2,
            "joint_order": [f"main_{index}" for index in range(6)],
            "position_source_channel": "main_joint_position_canonical",
            "position_rad": canonical_position[0].tolist(),
            "velocity_rad_s": [0.0] * 6,
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
            "imu_orientation_tolerance_rad": math.radians(5.0),
            "imu_angular_velocity_source_channel": "imu_angular_velocity",
            "max_imu_angular_speed_rad_s": 0.0,
            "imu_stationarity_limit_rad_s": 0.1,
            "sample_time_s": 0.0,
            "scenario_time_s": 0.0,
            "sample_offset_s": 0.0,
        }
        arrays.update(
            {
                "main_joint_position_canonical": canonical_position,
                "imu_time_s": time_s,
                "imu_orientation_xyzw": np.tile(
                    np.asarray(fixture["expected_imu_orientation_xyzw"]),
                    (time_s.size, 1),
                ),
                "imu_angular_velocity": np.zeros((time_s.size, 3)),
            }
        )
        metadata["units"].update(
            {
                "main_joint_position_canonical": "rad",
                "imu_orientation_xyzw": "quaternion_xyzw",
                "imu_angular_velocity": "rad/s",
            }
        )
        metadata["frames"].update(
            {
                "main_joint_position_canonical": "canonical_main_joint_order",
                "imu_orientation_xyzw": "imu_mount",
                "imu_angular_velocity": "imu_mount",
            }
        )
        metadata["calibration_constants"].update(
            {
                "all_main_position_mapping_source": f"profile:{profile.profile_id}",
                "replay_initial_state": initial_state,
                "replay_initial_state_sha256": sha256_json(initial_state),
            }
        )
        metadata["calibration_constants"]["probe_event_evidence"].update(
            {
                "scenario_receive_time_s": 0.0,
                "complete_receive_time_s": duration_s,
            }
        )
        time_bases = {
            "main_joint_position_canonical": "position_time_s",
            "imu_orientation_xyzw": "imu_time_s",
            "imu_angular_velocity": "imu_time_s",
        }
    write_trace(
        directory,
        arrays,
        scenario=scenario,
        source=source,
        source_path=source_path,
        profile=profile,
        metadata=metadata,
        time_bases=time_bases,
    )
    if source == "real" and directory.parent.name == "episodes":
        dataset_root = directory.parent.parent
        raw_relative = source_path.relative_to(dataset_root).as_posix()
        manifest = {
            "schema_version": 1,
            "dataset_id": dataset_root.name,
            "raw": [{"path": raw_relative, "sha256": sha256_path(source_path)}],
            "episodes": [
                {
                    "episode_id": directory.name,
                    "scenario_id": scenario.scenario_id,
                    "path": directory.relative_to(dataset_root).as_posix(),
                    "trace_sha256": load_trace(directory).manifest.provenance[
                        "trace_sha256"
                    ],
                    "metadata_sha256": sha256_file(directory / "metadata.json"),
                    "raw_path": raw_relative,
                }
            ],
        }
        _write_json(dataset_root / "manifest.json", manifest)


def _write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return sha256_file(path)


def _trace_binding(path: Path, root: Path) -> dict[str, str]:
    trace = load_trace(path)
    return {
        "path": path.relative_to(root).as_posix(),
        "trace_sha256": trace.manifest.provenance["trace_sha256"],
        "metadata_sha256": trace.metadata_sha256,
    }


def _write_managed_trace(
    root: Path,
    *,
    dataset_id: str,
    episode_id: str,
    scenario_id: str,
    arrays: dict[str, np.ndarray],
    metadata: dict,
    profile: CalibrationProfileV1 | None = None,
) -> object:
    scenario = load_scenario(scenario_id)
    dataset = root / "datasets" / "sim2real" / dataset_id
    raw = dataset / "raw" / f"{episode_id}.bin"
    episode = dataset / "episodes" / episode_id
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(f"immutable:{dataset_id}:{episode_id}".encode())
    manifest = write_trace(
        episode,
        arrays,
        scenario=scenario,
        source="real",
        source_path=raw,
        profile=profile,
        metadata=metadata,
    )
    dataset_manifest = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "raw": [
            {
                "path": raw.relative_to(dataset).as_posix(),
                "sha256": sha256_path(raw),
            }
        ],
        "episodes": [
            {
                "episode_id": episode_id,
                "scenario_id": scenario_id,
                "path": episode.relative_to(dataset).as_posix(),
                "trace_sha256": manifest.provenance["trace_sha256"],
                "metadata_sha256": sha256_file(episode / "metadata.json"),
                "raw_path": raw.relative_to(dataset).as_posix(),
            }
        ],
    }
    _write_json(dataset / "manifest.json", dataset_manifest)
    return load_trace(episode, scenario=scenario, require_managed_dataset=True)


def _real_binding(trace, root: Path) -> dict[str, str]:
    assert trace.dataset is not None
    return {
        "dataset_id": trace.dataset.dataset_id,
        "episode_id": trace.dataset.episode_id,
        **_trace_binding(trace.directory, root),
    }


def _direct_measurement_metadata(
    scenario_id: str, *, load_coordinate: str | None = None
) -> dict[str, object]:
    scenario = load_scenario(scenario_id)
    if scenario.experiment_kind == "abad_static":
        units = {
            "command": "rad",
            "position": "rad",
            "repeat_index": "1",
            "settled": "1",
        }
        frames = {name: scenario.joint for name in units}
    elif scenario.experiment_kind == "friction":
        units = {
            "breakaway_force": "N",
            "static_normal_load": "N",
            "static_repeat_index": "1",
            "dynamic_pull_force": "N",
            "dynamic_normal_load": "N",
            "dynamic_speed": "m/s",
            "dynamic_repeat_index": "1",
        }
        frames = {name: f"{scenario.joint}/ground" for name in units}
    elif scenario.experiment_kind == "static_settle":
        units = {
            "root_position": "m",
            "contact_force_n": "N",
            "repeat_index": "1",
            "settled": "1",
        }
        frames = {
            "root_position": "world",
            "contact_force_n": "feet/ground",
            "repeat_index": "annotation",
            "settled": "annotation",
        }
    else:  # pragma: no cover - test helper guard
        raise AssertionError(scenario.experiment_kind)
    calibration_constants: dict[str, object] = {}
    if load_coordinate is not None:
        calibration_constants["condition_coordinates"] = {"load": load_coordinate}
    return {
        "units": units,
        "frames": frames,
        "joint_order": [scenario.joint],
        "calibration_constants": calibration_constants,
    }


def _abad_arrays(
    scale: float, offset: float, *, command_level: float = 0.15
) -> dict[str, np.ndarray]:
    command = np.tile(np.array([-command_level, 0.0, command_level]), 3)
    repeats = np.repeat(np.arange(3), 3)
    time_s = np.arange(command.size, dtype=float) * 0.1
    return {
        "command_time_s": time_s,
        "command": command,
        "position_time_s": time_s,
        "position": scale * command + offset,
        "repeat_index": repeats,
        "settled": np.ones(command.size),
    }


def _friction_arrays() -> dict[str, np.ndarray]:
    static_time = np.arange(3, dtype=float) * 0.1
    dynamic_time = np.arange(9, dtype=float) * 0.1
    return {
        "static_time_s": static_time,
        "breakaway_force": np.array([6.0, 6.1, 5.9]),
        "static_normal_load": np.full(3, 10.0),
        "static_repeat_index": np.arange(3),
        "dynamic_time_s": dynamic_time,
        "dynamic_pull_force": np.tile(np.array([4.0, 4.0, 4.0]), 3),
        "dynamic_normal_load": np.full(9, 10.0),
        "dynamic_speed": np.full(9, 0.05),
        "dynamic_repeat_index": np.repeat(np.arange(3), 3),
    }


def _settle_arrays(root_height_m: float) -> dict[str, np.ndarray]:
    repeat_index = np.repeat(np.arange(3), 3)
    time_s = np.arange(repeat_index.size, dtype=float) * 0.1
    root_position = np.zeros((time_s.size, 3))
    root_position[:, 2] = root_height_m
    return {
        "sim_time_s": time_s,
        "root_position": root_position,
        "contact_force_n": np.full((time_s.size, 6), 16.35),
        "repeat_index": repeat_index,
        "settled": np.ones(time_s.size),
    }


def _audit_evidence(
    root: Path, candidate: CalibrationProfileV1
) -> dict[str, object]:
    scenario = load_scenario("audit")
    run = root / "runtime-audit-run"
    physical_path = root / "physical-audit.json"
    if (run / "trace.npz").is_file():
        return {
            "runtime_trace": _trace_binding(run, root),
            "runtime_audit": {
                "path": (run / "runtime_audit.json").relative_to(root).as_posix(),
                "sha256": sha256_file(run / "runtime_audit.json"),
            },
            "physical_measurements": {
                "path": physical_path.relative_to(root).as_posix(),
                "sha256": sha256_file(physical_path),
            },
        }
    time_s = np.array([0.0, 0.05, 0.1])
    runtime_joint_names = [
        "Revolute_15",
        "Revolute_7",
        "Revolute_12",
        "Revolute_18",
        "Revolute_23",
        "Revolute_24",
        "Revolute_14",
        "Revolute_6",
        "Revolute_11",
        "Revolute_17",
        "Revolute_22",
        "Revolute_21",
        "Revolute_5",
        "Revolute_8",
        "Revolute_13",
        "Revolute_25",
        "Revolute_26",
        "Revolute_27",
    ]
    canonical_joint_names = [
        *(f"main_{index}" for index in range(6)),
        *(f"abad_{index}" for index in range(6)),
        *(f"damper_{index}" for index in range(6)),
    ]
    collision_body_names = [
        "base_link",
        "left_feet_1",
        "left_feet_2",
        "left_feet_3",
        "right_feet_1",
        "right_feet_2",
        "right_feet_3",
    ]
    candidate_mass = candidate.simulation_physics.get("mass", {})
    absolute_mass = "target_total_mass_kg" in candidate_mass
    target_mass = (
        float(candidate_mass["target_total_mass_kg"]) if absolute_mass else 10.0
    )
    target_xy = (
        list(candidate_mass["reference_planar_com_xy_m"])
        if absolute_mass
        else [0.1, -0.05]
    )
    mass_application = (
        {
            "mode": "absolute",
            "uniform_scale": 1.0,
            "reference_total_mass_kg": target_mass,
            "reference_whole_com_root_m": [*target_xy, 0.0],
            "reference_pose": {
                "joint_position_rad": dict(
                    candidate_mass["reference_joint_position_rad"]
                ),
                "root_orientation_xyzw": list(
                    candidate_mass["reference_root_orientation_xyzw"]
                ),
            },
            "target_total_mass_kg": target_mass,
            "target_planar_com_xy_m": target_xy,
            "achieved_total_mass_kg": target_mass,
            "achieved_whole_com_root_m": [*target_xy, 0.0],
            "root_com_xyz_m": [*target_xy, 0.0],
            "total_mass_kg": target_mass,
        }
        if absolute_mass
        else None
    )
    runtime_audit = {
        "schema_version": 2,
        "mode": "contact",
        "physics_dt_s": 1.0 / 120.0,
        "num_envs": 1,
        "joint_names": runtime_joint_names,
        "body_names": collision_body_names,
        "body_properties": {
            "mass_kg": [
                [0.4 * target_mass, *([0.1 * target_mass] * 6)]
            ],
            "total_mass_kg": target_mass,
            "inertia_kg_m2_matrix": [
                [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]] * 7
            ],
            "com_pose_xyz_xyzw": [
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]] * 7
            ],
            "aggregate_com_body_m": [*target_xy, 0.0],
        },
        "mass_profile_application": mass_application,
        "joint_geometry": [
            {
                "canonical_joint": canonical,
                "runtime_joint": runtime,
                "articulation_index": index,
                "axis": "Z",
                "range_kind": "continuous",
                "lower_limit_rad": None,
                "upper_limit_rad": None,
            }
            for index, (canonical, runtime) in enumerate(
                zip(canonical_joint_names, runtime_joint_names, strict=True)
            )
        ],
        "collision_geometry": [
            {
                "prim_path": f"/World/envs/env_0/Robot/test_7/{name}/collisions",
                "type_name": "Xform",
                "has_collision_api": True,
                "is_geometry": False,
            }
            for name in collision_body_names
        ],
        "contact_sensors": {
            "foot": {
                "body_names": [
                    "left_feet_1",
                    "left_feet_2",
                    "left_feet_3",
                    "right_feet_1",
                    "right_feet_2",
                    "right_feet_3",
                ],
                "body_count": 6,
            }
        },
    }
    write_trace(
        run,
        {
            "audit_time_s": time_s,
            "audit_value": np.full(3, target_mass),
            "sim_time_s": time_s,
            "contact_force_n": np.array(
                [[0.0] * 6, [0.0] * 6, [1.0] * 6], dtype=float
            ),
        },
        scenario=scenario,
        source="sim",
        time_bases={
            "audit_value": "audit_time_s",
            "contact_force_n": "sim_time_s",
        },
        metadata={
            "units": {"audit_value": "kg", "contact_force_n": "N"},
            "frames": {"audit_value": "scalar", "contact_force_n": "world"},
            "calibration_constants": {
                "runtime_audit_sha256": sha256_json(runtime_audit),
            },
        },
    )
    runtime_audit_hash = _write_json(run / "runtime_audit.json", runtime_audit)
    physical_hash = _write_json(
        physical_path,
        {
            "schema_version": 2,
            "units": {
                "encoder_position": "rad",
                "main_command": "rad/s",
                "imu_gyro": "rad/s",
                "scale_mass": "kg",
                "load_force": "N",
            },
            "frames": {
                "encoder_position": "canonical_joint",
                "imu_gravity": "imu_mount",
                "contact_force": "world",
            },
            "joint_geometry": [
                {
                    "canonical_joint": canonical,
                    "runtime_joint": runtime,
                    "expected_axis": "Z",
                    "range_kind": "continuous",
                    "mechanical_min_rad": None,
                    "mechanical_max_rad": None,
                    "range_uncertainty_rad": 0.01,
                }
                for canonical, runtime in zip(
                    canonical_joint_names, runtime_joint_names, strict=True
                )
            ],
            "encoder_observations": [
                {
                    "joint": f"main_{index}",
                    "raw_start_count": 0.0,
                    "raw_end_count": 54984.83 / 4.0,
                    "physical_delta_rad": np.pi / 2.0,
                    "observed_counts_per_rev": 54984.83,
                    "counts_per_rev_uncertainty": 1.0,
                    "observed_zero_count": 0.0,
                    "zero_count_uncertainty": 1.0,
                    "angle_uncertainty_rad": 0.01,
                }
                for index in range(6)
            ],
            "mass_measurements_kg": [
                target_mass - 0.1,
                target_mass,
                target_mass + 0.1,
            ],
            "mass_instrument_uncertainty_kg": 0.25,
            "planar_com_measurements_m": [
                target_xy,
                [target_xy[0] + 0.001, target_xy[1] + 0.001],
                [target_xy[0] - 0.001, target_xy[1] - 0.001],
            ],
            "com_instrument_uncertainty_m": 0.005,
            "collision_body_names": collision_body_names,
            "imu_rest_orientations": [
                {
                    "label": "upright",
                    "measured_gravity": [0.0, 0.0, -1.0],
                    "expected_gravity": [0.0, 0.0, -1.0],
                },
                {
                    "label": "left_side",
                    "measured_gravity": [0.0, -1.0, 0.0],
                    "expected_gravity": [0.0, -1.0, 0.0],
                },
            ],
        },
    )
    return {
        "runtime_trace": _trace_binding(run, root),
        "runtime_audit": {
            "path": (run / "runtime_audit.json").relative_to(root).as_posix(),
            "sha256": runtime_audit_hash,
        },
        "physical_measurements": {
            "path": physical_path.relative_to(root).as_posix(),
            "sha256": physical_hash,
        },
    }


def _sweep_evidence(
    root: Path,
    *,
    baseline: CalibrationProfileV1,
    candidate: CalibrationProfileV1,
    real_holdout: Path,
    sim_scale: float,
) -> tuple[dict[str, object], Path]:
    from tools.sim2real.sweep_runner import execute_sweep

    scenario = load_scenario("suspended-main-5-step-coast")
    sweep_root = root / "sweep-main-held"
    runtime = {
        "git_sha": "1" * 40,
        "asset_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "redrhex_module_path": "/repo/source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py",
        "redrhex_module_sha256": "b" * 64,
        "isaaclab_version": "0.54.2",
        "isaacsim_version": "5.1.0-test",
        "characterization_runner_sha256": "c" * 64,
        "sweep_runner_sha256": "d" * 64,
        "runtime_bundle_sha256": "e" * 64,
    }

    def argument(command: list[str], flag: str) -> str:
        return command[command.index(flag) + 1]

    def run_process(command, **_kwargs):
        command = list(command)
        output = Path(argument(command, "--output"))
        profile = load_profile(argument(command, "--physics-profile"))
        _response_trace(
            output,
            scenario_id=scenario.scenario_id,
            source="sim",
            speed_scale=sim_scale,
            profile=profile,
            load_coordinate="holdout-load",
            runtime_provenance=runtime,
        )
        trace = load_trace(output)
        _write_json(output / "runtime_audit.json", {})
        _write_json(
            output / "results.json",
            {
                "schema_version": 1,
                "scenario_id": scenario.scenario_id,
                "mode": "fixed-base",
                "steps": 1980,
                "physics_dt_s": 1.0 / 120.0,
                "trace_sha256": trace.manifest.provenance["trace_sha256"],
                "profile_id": profile.profile_id,
                "runtime_audit": "runtime_audit.json",
            },
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    audit_root = root
    audit_artifact = _audit_evidence(audit_root, baseline)
    result = execute_sweep(
        output=sweep_root,
        scenario=scenario,
        base_profile=baseline,
        candidates=[candidate],
        sweep_mode="one-factor",
        scene_mode="fixed-base",
        headless=True,
        seed=17,
        device="cpu",
        provenance={},
        provenance_provider=lambda: runtime,
        command_prefix=("/opt/isaaclab/isaaclab.sh", "-p", "-m", "tools.sim2real"),
        real_trace=real_holdout,
        known_load_trace=None,
        audit_artifact=audit_artifact,
        audit_artifact_root=audit_root,
        run_process=run_process,
    )
    run_output = sweep_root / result["candidates"][0]["run_output"]
    return (
        {
            "path": sweep_root.relative_to(root).as_posix(),
            "results_sha256": sha256_file(sweep_root / "results.json"),
        },
        run_output,
    )


def _fixture(root: Path, *, sim_scale: float = 1.02) -> tuple[CalibrationProfileV1, dict]:
    baseline = _profile("baseline", 0.1)
    candidate = _profile("candidate", 0.2)
    baseline_path = root / "baseline.json"
    _write_json(baseline_path, baseline.to_dict())
    calibration = (
        root
        / "datasets"
        / "sim2real"
        / "main-cal-dataset"
        / "episodes"
        / "main-cal-real"
    )
    holdout = (
        root
        / "datasets"
        / "sim2real"
        / "main-held-dataset"
        / "episodes"
        / "main-held-real"
    )
    _response_trace(
        calibration,
        scenario_id="suspended-main-0-step-coast",
        source="real",
        speed_scale=0.9,
        profile=candidate,
        load_coordinate="suspended-unloaded",
    )
    _response_trace(
        holdout,
        scenario_id="suspended-main-5-step-coast",
        source="real",
        speed_scale=1.0,
        profile=candidate,
        load_coordinate="holdout-load",
        replay_ready=True,
    )
    real_trace = load_trace(holdout)
    sweep_binding, simulated = _sweep_evidence(
        root,
        baseline=baseline,
        candidate=candidate,
        real_holdout=holdout,
        sim_scale=sim_scale,
    )
    sim_trace = load_trace(simulated)
    metric_units = {
        **{
            f"step.{direction}.{metric}": unit
            for direction in ("positive", "negative")
            for metric, unit in (
                ("onset_delay_s", "s"),
                ("steady_speed_rad_s", "rad/s"),
                ("rise_time_s", "s"),
                ("overshoot_ratio", "1"),
            )
        },
        **{
            f"coast.{direction}.{metric}": unit
            for direction in ("positive", "negative")
            for metric, unit in (
                ("coast_time_s", "s"),
                ("pre_coast_speed_rad_s", "rad/s"),
            )
        },
    }
    evidence = {
        "schema_version": 1,
        "candidate_profile_sha256": sha256_json(candidate.to_dict()),
        "baseline_profile": {
            "path": "baseline.json",
            "sha256": sha256_file(baseline_path),
        },
        "audit_artifact": _audit_evidence(root, candidate),
        "conditions": [
            {
                "condition_id": "main-cal",
                "subsystem": "main_drive",
                "role": "calibration",
                "real_episodes": [
                    {
                        "episode_id": "main-cal-real",
                        "dataset_id": "main-cal-dataset",
                        "path": calibration.relative_to(root).as_posix(),
                        "trace_sha256": load_trace(calibration).manifest.provenance[
                            "trace_sha256"
                        ],
                        "metadata_sha256": load_trace(calibration).metadata_sha256,
                    }
                ],
                "metrics": {},
            },
            {
                "condition_id": "main-held",
                "subsystem": "main_drive",
                "role": "holdout",
                "held_out_by": ["leg"],
                "real_episodes": [
                    {
                        "episode_id": "main-held-real",
                        "dataset_id": "main-held-dataset",
                        "path": holdout.relative_to(root).as_posix(),
                        "trace_sha256": real_trace.manifest.provenance["trace_sha256"],
                        "metadata_sha256": real_trace.metadata_sha256,
                    }
                ],
                "sim_artifact": {
                    "path": simulated.relative_to(root).as_posix(),
                    "trace_sha256": sim_trace.manifest.provenance["trace_sha256"],
                    "metadata_sha256": sim_trace.metadata_sha256,
                },
                "metrics": {
                    metric_path: {
                        "unit": unit,
                        "instrument_uncertainty": 0.15,
                    }
                    for metric_path, unit in metric_units.items()
                },
            },
        ],
        "actuator_sweeps": {
            "main_drive": [sweep_binding]
        },
    }
    return candidate, evidence


def test_promotion_requires_every_mandatory_heldout_metric(tmp_path: Path) -> None:
    profile, evidence = _fixture(tmp_path)
    evidence["conditions"][1]["metrics"].pop("coast.negative.coast_time_s")

    with pytest.raises(ContractError, match="mandatory held-out metrics.*coast.negative"):
        evaluate_promotion(profile, evidence, artifact_root=tmp_path)


@pytest.mark.parametrize(
    ("section", "value", "expected"),
    [
        ("abad", {"damping": 1.0}, "simulation_physics.abad"),
        ("rigid_body", {"linear_damping": 0.1}, "simulation_physics.rigid_body"),
        ("damper", {"damping": 1.0}, "simulation_physics.damper"),
        ("ground", {"restitution": 0.1}, "ground.restitution"),
        ("main_drive", {"damping": 1.5, "stiffness": 1.0}, "main_drive.stiffness"),
        (
            "passive_spring",
            {"damper_0": {"stiffness": 10.0, "damping": 1.0}},
            "passive_spring.damper_0.damping",
        ),
        ("joint_friction", {"abad_0": 0.1}, "joint_friction.abad_0"),
        ("mass", {"scale": 1.1}, "mass.scale"),
    ],
)
def test_promotion_rejects_unidentifiable_profile_changes(
    tmp_path: Path, section: str, value: dict[str, object], expected: str
) -> None:
    candidate, evidence = _fixture(tmp_path)
    payload = candidate.to_dict()
    payload["simulation_physics"][section] = value
    changed = CalibrationProfileV1.from_dict(payload)
    evidence["candidate_profile_sha256"] = sha256_json(changed.to_dict())

    with pytest.raises(ContractError, match=expected):
        evaluate_promotion(changed, evidence, artifact_root=tmp_path)


def test_promotion_resolves_artifacts_and_derives_repetitions_metrics_and_fitted_subsystems(
    tmp_path: Path,
) -> None:
    profile, evidence = _fixture(tmp_path)

    result = evaluate_promotion(profile, evidence, artifact_root=tmp_path)

    assert result["eligible_for_review"] is True
    assert result["promotion_requires_reviewed_config_change"] is True
    assert set(result["subsystems"]) == {"main_drive"}
    assert result["subsystems"]["main_drive"]["pass"] is True
    condition = result["subsystems"]["main_drive"]["holdout_conditions"][0]
    assert condition["real_repetition_count"] == 3
    metric = condition["metrics"]["step.positive.steady_speed_rad_s"]
    assert metric["absolute_error"] <= metric["tolerance"]
    assert "score" not in str(result).lower()
    assert result["evidence_sha256"] == sha256_json(evidence)
    assert result["audit"]["checks"] == {
        "units_pass": True,
        "frames_pass": True,
        "joint_order_pass": True,
        "joint_axis_pass": True,
        "encoder_scale_zero_pass": True,
        "joint_sign_pass": True,
        "mechanical_range_pass": True,
        "mass_pass": True,
        "mass_profile_application_pass": True,
        "inertia_com_pass": True,
        "planar_com_pass": True,
        "collision_geometry_pass": True,
        "imu_mount_pass": True,
        "contact_sensor_pass": True,
    }


@pytest.mark.parametrize(
    ("mutate", "failed_check"),
    [
        (lambda audit: audit["joint_geometry"].reverse(), "joint_order_pass"),
        (
            lambda audit: audit["joint_geometry"][0].__setitem__(
                "expected_axis", "X"
            ),
            "joint_axis_pass",
        ),
        (
            lambda audit: audit["encoder_observations"][0].__setitem__(
                "observed_counts_per_rev", 50000.0
            ),
            "encoder_scale_zero_pass",
        ),
        (
            lambda audit: audit["encoder_observations"][0].__setitem__(
                "raw_end_count", -54984.83 / 4.0
            ),
            "joint_sign_pass",
        ),
        (
            lambda audit: audit["joint_geometry"][0].update(
                {
                    "range_kind": "limited",
                    "mechanical_min_rad": -1.0,
                    "mechanical_max_rad": 1.0,
                }
            ),
            "mechanical_range_pass",
        ),
        (
            lambda audit: audit.__setitem__(
                "planar_com_measurements_m", [[1.0, 1.0]] * 3
            ),
            "planar_com_pass",
        ),
        (
            lambda audit: audit.__setitem__(
                "collision_body_names", audit["collision_body_names"][:-1]
            ),
            "collision_geometry_pass",
        ),
    ],
    ids=(
        "joint-order",
        "joint-axis",
        "encoder-scale-zero",
        "encoder-sign",
        "mechanical-range",
        "planar-com",
        "collision-geometry",
    ),
)
def test_geometry_audit_derives_explicit_failures_from_measurements(
    tmp_path: Path, mutate, failed_check: str
) -> None:
    profile, evidence = _fixture(tmp_path)
    physical_binding = evidence["audit_artifact"]["physical_measurements"]
    physical_path = tmp_path / physical_binding["path"]
    physical = json.loads(physical_path.read_text())
    mutate(physical)
    physical_binding["sha256"] = _write_json(physical_path, physical)
    evidence["actuator_sweeps"]["main_drive"] = []

    result = evaluate_promotion(profile, evidence, artifact_root=tmp_path)

    assert result["audit"]["checks"][failed_check] is False
    assert any(f"audit.{failed_check}" in item for item in result["failures"])


def test_failed_audit_rejects_stale_actuator_sweep_bindings(tmp_path: Path) -> None:
    profile, evidence = _fixture(tmp_path)
    physical_binding = evidence["audit_artifact"]["physical_measurements"]
    physical_path = tmp_path / physical_binding["path"]
    physical = json.loads(physical_path.read_text())
    physical["mass_measurements_kg"] = [12.0, 12.0, 12.0]
    physical_binding["sha256"] = _write_json(physical_path, physical)

    with pytest.raises(ContractError, match="failed pre-fit audit.*actuator sweep"):
        evaluate_promotion(profile, evidence, artifact_root=tmp_path)


def test_runtime_audit_json_must_be_content_bound_to_its_trace(tmp_path: Path) -> None:
    profile, evidence = _fixture(tmp_path)
    runtime_binding = evidence["audit_artifact"]["runtime_audit"]
    runtime_path = tmp_path / runtime_binding["path"]
    runtime = json.loads(runtime_path.read_text())
    runtime["body_properties"]["total_mass_kg"] = 11.0
    runtime_binding["sha256"] = _write_json(runtime_path, runtime)

    with pytest.raises(ContractError, match="not bound to its audit trace"):
        evaluate_promotion(profile, evidence, artifact_root=tmp_path)


def test_promotion_rejects_real_trace_outside_managed_dataset(tmp_path: Path) -> None:
    profile, evidence = _fixture(tmp_path)
    managed = tmp_path / evidence["conditions"][0]["real_episodes"][0]["path"]
    standalone = tmp_path / "standalone-real"
    shutil.copytree(managed, standalone)
    evidence["conditions"][0]["real_episodes"][0]["path"] = "standalone-real"

    with pytest.raises(ContractError, match="managed dataset"):
        evaluate_promotion(profile, evidence, artifact_root=tmp_path)


def test_promotion_rejects_managed_real_trace_with_provisional_mapping(
    tmp_path: Path,
) -> None:
    profile, evidence = _fixture(tmp_path)
    binding = evidence["conditions"][0]["real_episodes"][0]
    episode = tmp_path / binding["path"]
    metadata_path = episode / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["metadata"]["calibration_constants"][
        "position_mapping_source"
    ] = "provisional_repository_defaults"
    metadata_sha = _write_json(metadata_path, metadata)
    dataset_manifest_path = episode.parent.parent / "manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text())
    dataset_manifest["episodes"][0]["metadata_sha256"] = metadata_sha
    _write_json(dataset_manifest_path, dataset_manifest)
    binding["metadata_sha256"] = metadata_sha

    with pytest.raises(ContractError, match="provisional.*hardware mapping"):
        evaluate_promotion(profile, evidence, artifact_root=tmp_path)


def test_each_claimed_holdout_coordinate_must_differ_from_every_calibration(
    tmp_path: Path,
) -> None:
    profile, evidence = _fixture(tmp_path)
    second = (
        tmp_path
        / "datasets"
        / "sim2real"
        / "main-second-dataset"
        / "episodes"
        / "main-second-real"
    )
    _response_trace(
        second,
        scenario_id="suspended-main-1-step-coast",
        source="real",
        speed_scale=0.95,
        profile=profile,
        load_coordinate="holdout-load",
    )
    loaded = load_trace(second, require_managed_dataset=True)
    evidence["conditions"].insert(
        1,
        {
            "condition_id": "main-second-cal",
            "subsystem": "main_drive",
            "role": "calibration",
            "real_episodes": [
                {
                    "dataset_id": "main-second-dataset",
                    "episode_id": "main-second-real",
                    "path": second.relative_to(tmp_path).as_posix(),
                    "trace_sha256": loaded.manifest.provenance["trace_sha256"],
                    "metadata_sha256": loaded.metadata_sha256,
                }
            ],
            "metrics": {},
        },
    )
    evidence["conditions"][2]["held_out_by"] = ["load"]

    with pytest.raises(ContractError, match="held-out dimension load"):
        evaluate_promotion(profile, evidence, artifact_root=tmp_path)


def test_promotion_rejects_caller_authored_boolean_audit_assertions(
    tmp_path: Path,
) -> None:
    profile, evidence = _fixture(tmp_path)
    boolean_path = tmp_path / "boolean-audit.json"
    boolean_hash = _write_json(
        boolean_path,
        {
            "schema_version": 1,
            "checks": {field: True for field in (
                "units_pass",
                "frames_pass",
                "joint_sign_pass",
                "mass_pass",
                "imu_mount_pass",
                "contact_sensor_pass",
            )},
        },
    )
    evidence["audit_artifact"] = {
        "path": "boolean-audit.json",
        "sha256": boolean_hash,
    }

    with pytest.raises(ContractError, match="runtime_trace|physical_measurements"):
        evaluate_promotion(profile, evidence, artifact_root=tmp_path)


def test_promotion_verifies_every_sweep_candidate_not_an_evidence_subset(
    tmp_path: Path,
) -> None:
    profile, evidence = _fixture(tmp_path)
    sweep_binding = evidence["actuator_sweeps"]["main_drive"][0]
    results_path = tmp_path / sweep_binding["path"] / "results.json"
    results = json.loads(results_path.read_text())
    results["candidates"].append(
        {
            **results["candidates"][0],
            "index": 2,
            "status_file": "statuses/0002.json",
        }
    )
    results["candidate_count"] = 2
    results["counts"]["completed"] = 2
    sweep_binding["results_sha256"] = _write_json(results_path, results)

    with pytest.raises(ContractError, match="index and results disagree|candidate status"):
        evaluate_promotion(profile, evidence, artifact_root=tmp_path)


def test_promotion_recomputes_sweep_comparison_and_scenario(tmp_path: Path) -> None:
    profile, evidence = _fixture(tmp_path)
    sweep_root = tmp_path / evidence["actuator_sweeps"]["main_drive"][0]["path"]
    results = json.loads((sweep_root / "results.json").read_text())
    comparison_path = sweep_root / results["candidates"][0]["comparison"]
    comparison = json.loads(comparison_path.read_text())
    comparison["subsystems"]["main_drive"]["delta"] = {"fabricated": 0.0}
    _write_json(comparison_path, comparison)

    with pytest.raises(ContractError, match="comparison hash mismatch"):
        evaluate_promotion(profile, evidence, artifact_root=tmp_path)

    profile, evidence = _fixture(tmp_path / "scenario-case")
    sweep_root = (
        tmp_path
        / "scenario-case"
        / evidence["actuator_sweeps"]["main_drive"][0]["path"]
    )
    scenario_path = sweep_root / "scenario.json"
    scenario = json.loads(scenario_path.read_text())
    scenario["joint"] = "main_4"
    _write_json(scenario_path, scenario)

    with pytest.raises(ContractError, match="scenario snapshot"):
        evaluate_promotion(profile, evidence, artifact_root=tmp_path / "scenario-case")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda evidence: evidence["conditions"][0]["real_episodes"][0].__setitem__(
                "path", "missing"
            ),
            "missing|does not exist",
        ),
        (
            lambda evidence: evidence["conditions"][1]["sim_artifact"].__setitem__(
                "trace_sha256", "0" * 64
            ),
            "trace hash mismatch",
        ),
        (
            lambda evidence: evidence["conditions"][1]["real_episodes"][0].update(
                {
                    "path": evidence["conditions"][0]["real_episodes"][0]["path"],
                    "trace_sha256": evidence["conditions"][0]["real_episodes"][0][
                        "trace_sha256"
                    ],
                }
            ),
            "metadata hash mismatch|calibration and holdout.*disjoint",
        ),
        (
            lambda evidence: evidence["conditions"][1].__setitem__(
                "held_out_by", ["direction"]
            ),
            "held-out dimension direction",
        ),
    ],
    ids=("missing-artifact", "wrong-hash", "reused-trace", "fake-coordinate"),
)
def test_promotion_rejects_unbound_or_fake_artifact_claims(
    tmp_path: Path, mutation, message: str
) -> None:
    profile, evidence = _fixture(tmp_path)
    mutation(evidence)

    with pytest.raises(ContractError, match=message):
        evaluate_promotion(profile, evidence, artifact_root=tmp_path)


def test_model_envelope_fails_the_affected_subsystem(
    tmp_path: Path,
) -> None:
    profile, evidence = _fixture(tmp_path, sim_scale=1.3)

    result = evaluate_promotion(profile, evidence, artifact_root=tmp_path)

    assert result["eligible_for_review"] is False
    main = result["subsystems"]["main_drive"]
    assert main["pass"] is False
    assert main["actuator_model_mismatch"] is True
    assert any("outside" in reason for reason in main["failures"])


def test_missing_holdout_fails_the_derived_fitted_subsystem(tmp_path: Path) -> None:
    profile, evidence = _fixture(tmp_path)
    evidence["conditions"] = [evidence["conditions"][0]]
    evidence["actuator_sweeps"]["main_drive"] = []

    result = evaluate_promotion(profile, evidence, artifact_root=tmp_path)

    assert result["eligible_for_review"] is False
    assert result["subsystems"]["main_drive"]["pass"] is False
    assert any("missing a holdout" in reason for reason in result["failures"])


def test_effort_limit_change_requires_known_load_calibration_condition(
    tmp_path: Path,
) -> None:
    profile, evidence = _fixture(tmp_path)
    baseline_path = tmp_path / evidence["baseline_profile"]["path"]
    baseline = json.loads(baseline_path.read_text())
    baseline["simulation_physics"]["main_drive"]["effort_limit"] = 1.0
    evidence["baseline_profile"]["sha256"] = _write_json(baseline_path, baseline)

    result = evaluate_promotion(profile, evidence, artifact_root=tmp_path)

    assert result["eligible_for_review"] is False
    assert any("known-load" in reason for reason in result["failures"])


def test_validate_promotion_cli_resolves_paths_relative_to_evidence(
    tmp_path: Path, capsys
) -> None:
    profile, evidence = _fixture(tmp_path)
    profile_path = tmp_path / "candidate.json"
    evidence_path = tmp_path / "evidence.json"
    output_path = tmp_path / "report.json"
    _write_json(profile_path, profile.to_dict())
    _write_json(evidence_path, evidence)

    code = main(
        [
            "validate-promotion",
            str(profile_path),
            str(evidence_path),
            "--output",
            str(output_path),
        ]
    )

    emitted = json.loads(capsys.readouterr().out)
    assert code == 0
    assert emitted == json.loads(output_path.read_text())
    assert emitted["eligible_for_review"] is True


def _direct_measurement_fixture(
    root: Path, *, subsystem: str
) -> tuple[CalibrationProfileV1, dict[str, object]]:
    main_joints = [f"main_{index}" for index in range(6)]
    baseline = CalibrationProfileV1.from_dict(
        {
            "schema_version": 1,
            "profile_id": "baseline",
            "hardware_mapping": {
                "encoder_counts_per_rev": {
                    joint: 54984.83 for joint in main_joints
                },
                "encoder_zero_count": {joint: 0.0 for joint in main_joints},
                "encoder_sign": {joint: 1 for joint in main_joints},
            },
            "sensor_timing": {},
            "simulation_physics": {
                "ground": {"static_friction": 0.5, "dynamic_friction": 0.3}
            },
        }
    )
    baseline_path = root / "baseline.json"
    _write_json(baseline_path, baseline.to_dict())

    if subsystem == "abad":
        calibration = _write_managed_trace(
            root,
            dataset_id="abad-cal-data",
            episode_id="abad-cal-episode",
            scenario_id="abad-static",
            arrays=_abad_arrays(1.1, 0.02),
            metadata=_direct_measurement_metadata("abad-static"),
        )
        candidate = apply_measurements_to_profile(
            baseline,
            profile_id="abad-candidate",
            trace_paths=[calibration.directory],
        )
        holdout = _write_managed_trace(
            root,
            dataset_id="abad-holdout-data",
            episode_id="abad-holdout-episode",
            scenario_id="abad-static-holdout",
            arrays=_abad_arrays(1.1, 0.02, command_level=0.1),
            metadata=_direct_measurement_metadata("abad-static-holdout"),
            profile=candidate,
        )
        sim_path = root / "abad-holdout-sim"
        write_trace(
            sim_path,
            _abad_arrays(1.1, 0.02, command_level=0.1),
            scenario=load_scenario("abad-static-holdout"),
            source="sim",
            profile=candidate,
            metadata=_direct_measurement_metadata("abad-static-holdout"),
        )
        metric_units = {
            "aggregate.target_scale": "1",
            "aggregate.target_offset_rad": "rad",
            "aggregate.fit_rmse_rad": "rad",
        }
        uncertainty = 0.01
        held_out_by = ["command_level"]
    elif subsystem == "contact":
        calibration = _write_managed_trace(
            root,
            dataset_id="friction-cal-data",
            episode_id="friction-cal-episode",
            scenario_id="friction",
            arrays=_friction_arrays(),
            metadata=_direct_measurement_metadata(
                "friction", load_coordinate="pull-block"
            ),
        )
        candidate = apply_measurements_to_profile(
            baseline,
            profile_id="contact-candidate",
            trace_paths=[calibration.directory],
        )
        holdout = _write_managed_trace(
            root,
            dataset_id="contact-holdout-data",
            episode_id="contact-holdout-episode",
            scenario_id="contact-static-settle",
            arrays=_settle_arrays(0.2),
            metadata=_direct_measurement_metadata(
                "contact-static-settle", load_coordinate="full-robot"
            ),
            profile=candidate,
        )
        sim_path = root / "contact-holdout-sim"
        write_trace(
            sim_path,
            _settle_arrays(0.2),
            scenario=load_scenario("contact-static-settle"),
            source="sim",
            profile=candidate,
            metadata=_direct_measurement_metadata(
                "contact-static-settle", load_coordinate="full-robot"
            ),
        )
        metric_units = {
            "settled.root_height_m": "m",
            "settled.contact_force_n": "N",
        }
        uncertainty = 0.005
        held_out_by = ["load"]
    else:  # pragma: no cover - test helper guard
        raise AssertionError(subsystem)

    sim_trace = load_trace(sim_path)
    evidence: dict[str, object] = {
        "schema_version": 1,
        "candidate_profile_sha256": sha256_json(candidate.to_dict()),
        "baseline_profile": {
            "path": baseline_path.relative_to(root).as_posix(),
            "sha256": sha256_file(baseline_path),
        },
        "audit_artifact": _audit_evidence(root, candidate),
        "conditions": [
            {
                "condition_id": f"{subsystem}-cal",
                "subsystem": subsystem,
                "role": "calibration",
                "real_episodes": [_real_binding(calibration, root)],
                "metrics": {},
            },
            {
                "condition_id": f"{subsystem}-held",
                "subsystem": subsystem,
                "role": "holdout",
                "held_out_by": held_out_by,
                "real_episodes": [_real_binding(holdout, root)],
                "sim_artifact": _trace_binding(sim_trace.directory, root),
                "metrics": {
                    metric_path: {
                        "unit": unit,
                        "instrument_uncertainty": uncertainty,
                    }
                    for metric_path, unit in metric_units.items()
                },
            },
        ],
        "actuator_sweeps": {},
    }
    return candidate, evidence


@pytest.mark.parametrize("subsystem", ["abad", "contact"])
def test_promotion_accepts_authenticated_direct_measurement_and_distinct_holdout(
    tmp_path: Path, subsystem: str
) -> None:
    candidate, evidence = _direct_measurement_fixture(tmp_path, subsystem=subsystem)

    result = evaluate_promotion(candidate, evidence, artifact_root=tmp_path)

    assert result["eligible_for_review"] is True
    assert result["subsystems"][subsystem]["pass"] is True
    assert result["measurement_sources"]["pass"] is True


def test_promotion_rejects_candidate_measurement_source_not_bound_as_calibration(
    tmp_path: Path,
) -> None:
    candidate, evidence = _direct_measurement_fixture(tmp_path, subsystem="abad")
    alternate = _write_managed_trace(
        tmp_path,
        dataset_id="alternate-cal-data",
        episode_id="alternate-cal-episode",
        scenario_id="abad-static",
        arrays=_abad_arrays(1.11, 0.02),
        metadata=_direct_measurement_metadata("abad-static"),
    )
    evidence["conditions"][0]["real_episodes"] = [
        _real_binding(alternate, tmp_path)
    ]

    with pytest.raises(ContractError, match="measurement source.*calibration"):
        evaluate_promotion(candidate, evidence, artifact_root=tmp_path)


def _manual_route_fixture(
    root: Path, *, subsystem: str
) -> tuple[CalibrationProfileV1, dict[str, object]]:
    measured_mapping = {
        "encoder_counts_per_rev": {
            f"main_{index}": 54984.83 for index in range(6)
        },
        "encoder_zero_count": {f"main_{index}": 0.0 for index in range(6)},
        "encoder_sign": {f"main_{index}": 1 for index in range(6)},
    }
    if subsystem == "spring":
        baseline = CalibrationProfileV1.from_dict(
            {
                "schema_version": 1,
                "profile_id": "spring-baseline",
                "hardware_mapping": measured_mapping,
                "sensor_timing": {},
                "simulation_physics": {
                    "passive_spring": {"damper_0": {"stiffness": 8.0}}
                },
            }
        )
        candidate_id = "spring-candidate"
        calibration_scenario = "spring"
        holdout_scenario = "spring-holdout"
        time_s = np.arange(9, dtype=float) * 0.1
        angle = np.tile(np.array([0.0, 0.1, 0.2]), 3)
        arrays = {
            "load_force_time_s": time_s,
            "load_force": angle * 100.0,
            "lever_arm_time_s": time_s,
            "lever_arm": np.full(9, 0.1),
            "angle_time_s": time_s,
            "angle": angle,
            "repeat_index": np.repeat(np.arange(3), 3),
        }
        metadata = {
            "units": {
                "load_force": "N",
                "lever_arm": "m",
                "angle": "rad",
                "repeat_index": "1",
            },
            "frames": {name: "damper_0" for name in (
                "load_force",
                "lever_arm",
                "angle",
                "repeat_index",
            )},
        }
    elif subsystem == "rigid_body":
        baseline = CalibrationProfileV1.from_dict(
            {
                "schema_version": 1,
                "profile_id": "mass-baseline",
                "hardware_mapping": measured_mapping,
                "sensor_timing": {},
                "simulation_physics": {},
            }
        )
        candidate_id = "mass-candidate"
        calibration_scenario = "mass-com"
        holdout_scenario = "mass-com-holdout"
        arrays = {
            "scale_time_s": np.arange(3, dtype=float),
            "scale_mass": np.array([9.9, 10.0, 10.1]),
            "repeat_index": np.arange(3),
            "support_force_time_s": np.arange(3, dtype=float),
            "support_force": np.array([[40.0, 30.0, 30.0]] * 3),
            "support_position_time_s": np.arange(3, dtype=float),
            "support_position": np.array(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
            ),
        }
        metadata = {
            "units": {
                "scale_mass": "kg",
                "support_force": "N",
                "support_position": "m",
                "repeat_index": "1",
            },
            "frames": {name: "root" for name in (
                "scale_mass",
                "support_force",
                "support_position",
                "repeat_index",
            )},
        }
    else:  # pragma: no cover - test helper guard
        raise AssertionError(subsystem)

    baseline_path = root / "manual-baseline.json"
    _write_json(baseline_path, baseline.to_dict())
    calibration_metadata = copy.deepcopy(metadata)
    calibration_metadata["calibration_constants"] = {
        "condition_coordinates": {"load": "calibration-load"}
    }
    holdout_metadata = copy.deepcopy(metadata)
    holdout_metadata["calibration_constants"] = {
        "condition_coordinates": {"load": "held-out-load"}
    }
    if subsystem == "rigid_body":
        reference_pose = {
            "reference_joint_position_rad": {
                f"{group}_{index}": 0.0
                for group in ("main", "abad", "damper")
                for index in range(6)
            },
            "reference_root_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
        calibration_metadata["calibration_constants"].update(reference_pose)
        holdout_metadata["calibration_constants"].update(reference_pose)
    calibration = _write_managed_trace(
        root,
        dataset_id=f"{subsystem}-cal-data",
        episode_id=f"{subsystem}-cal-episode",
        scenario_id=calibration_scenario,
        arrays=arrays,
        metadata=calibration_metadata,
    )
    candidate = apply_measurements_to_profile(
        baseline,
        profile_id=candidate_id,
        trace_paths=[calibration.directory],
    )
    holdout = _write_managed_trace(
        root,
        dataset_id=f"{subsystem}-held-data",
        episode_id=f"{subsystem}-held-episode",
        scenario_id=holdout_scenario,
        arrays=arrays,
        metadata=holdout_metadata,
    )
    evidence: dict[str, object] = {
        "schema_version": 1,
        "candidate_profile_sha256": sha256_json(candidate.to_dict()),
        "baseline_profile": {
            "path": baseline_path.relative_to(root).as_posix(),
            "sha256": sha256_file(baseline_path),
        },
        "audit_artifact": _audit_evidence(root, candidate),
        "conditions": [
            {
                "condition_id": f"{subsystem}-cal",
                "subsystem": subsystem,
                "role": "calibration",
                "real_episodes": [_real_binding(calibration, root)],
                "metrics": {},
            },
            {
                "condition_id": f"{subsystem}-held",
                "subsystem": subsystem,
                "role": "holdout",
                "held_out_by": ["load"],
                "real_episodes": [_real_binding(holdout, root)],
                "metrics": (
                    {
                        "stiffness_nm_per_rad": {
                            "unit": "N*m/rad",
                            "instrument_uncertainty": 0.1,
                        }
                    }
                    if subsystem == "spring"
                    else {
                        "mass_kg": {
                            "unit": "kg",
                            "instrument_uncertainty": 0.2,
                        },
                        "com_x_m": {
                            "unit": "m",
                            "instrument_uncertainty": 0.01,
                        },
                        "com_y_m": {
                            "unit": "m",
                            "instrument_uncertainty": 0.01,
                        },
                    }
                ),
            },
        ],
        "actuator_sweeps": {},
    }
    return candidate, evidence


@pytest.mark.parametrize("subsystem", ["spring", "rigid_body"])
def test_direct_manual_subsystem_route_uses_independent_holdout_measurement(
    tmp_path: Path, subsystem: str
) -> None:
    candidate, evidence = _manual_route_fixture(tmp_path, subsystem=subsystem)

    result = evaluate_promotion(candidate, evidence, artifact_root=tmp_path)

    assert result["eligible_for_review"] is True
    assert result["subsystems"][subsystem]["pass"] is True
    assert result["subsystems"][subsystem]["holdout_conditions"][0]["metrics"]


def test_mass_com_holdout_must_use_candidate_reference_pose(tmp_path: Path) -> None:
    candidate, evidence = _manual_route_fixture(tmp_path, subsystem="rigid_body")
    binding = evidence["conditions"][1]["real_episodes"][0]
    episode = tmp_path / binding["path"]
    metadata_path = episode / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["metadata"]["calibration_constants"]["reference_joint_position_rad"][
        "main_0"
    ] = 0.1
    metadata_sha = _write_json(metadata_path, metadata)
    dataset_manifest_path = episode.parent.parent / "manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text())
    dataset_manifest["episodes"][0]["metadata_sha256"] = metadata_sha
    _write_json(dataset_manifest_path, dataset_manifest)
    binding["metadata_sha256"] = metadata_sha

    with pytest.raises(ContractError, match="mass-com holdout reference pose"):
        evaluate_promotion(candidate, evidence, artifact_root=tmp_path)


def _known_load_promotion_fixture(
    root: Path,
) -> tuple[CalibrationProfileV1, dict[str, object]]:
    joints = [f"main_{index}" for index in range(6)]
    baseline = CalibrationProfileV1.from_dict(
        {
            "schema_version": 1,
            "profile_id": "effort-baseline",
            "hardware_mapping": {
                "encoder_counts_per_rev": {joint: 54984.83 for joint in joints},
                "encoder_zero_count": {joint: 0.0 for joint in joints},
                "encoder_sign": {joint: 1 for joint in joints},
            },
            "sensor_timing": {},
            "simulation_physics": {"main_drive": {"effort_limit": 1.0}},
        }
    )
    baseline_path = root / "effort-baseline.json"
    _write_json(baseline_path, baseline.to_dict())
    time_s = np.arange(6, dtype=float)
    arrays = {
        "load_force_time_s": time_s,
        "load_force": np.array([20.0, 20.0, 21.0, 21.0, 19.0, 19.0]),
        "lever_arm_time_s": time_s,
        "lever_arm": np.full(6, 0.1),
        "command_time_s": time_s,
        "command": np.tile(np.array([0.25, -0.25]), 3),
        "direction_time_s": time_s,
        "direction": np.tile(np.array([1.0, -1.0]), 3),
        "saturation_confirmed": np.ones(6),
        "repeat_index": np.repeat(np.arange(3), 2),
    }
    units = {
        "load_force": "N",
        "lever_arm": "m",
        "command": "normalized",
        "direction": "1",
        "saturation_confirmed": "1",
        "repeat_index": "1",
    }
    calibration = _write_managed_trace(
        root,
        dataset_id="effort-cal-data",
        episode_id="effort-cal-episode",
        scenario_id="manual-load",
        arrays=arrays,
        metadata={
            "units": units,
            "frames": {name: "main_0" for name in units},
            "calibration_constants": {
                "condition_coordinates": {"load": "calibration-gauge"}
            },
        },
    )
    candidate = apply_measurements_to_profile(
        baseline,
        profile_id="effort-candidate",
        trace_paths=[calibration.directory],
    )
    holdout = _write_managed_trace(
        root,
        dataset_id="effort-held-data",
        episode_id="effort-held-episode",
        scenario_id="manual-load-holdout",
        arrays=arrays,
        metadata={
            "units": units,
            "frames": {name: "main_0" for name in units},
            "calibration_constants": {
                "condition_coordinates": {"load": "held-out-weight"}
            },
        },
    )
    evidence: dict[str, object] = {
        "schema_version": 1,
        "candidate_profile_sha256": sha256_json(candidate.to_dict()),
        "baseline_profile": {
            "path": baseline_path.relative_to(root).as_posix(),
            "sha256": sha256_file(baseline_path),
        },
        "audit_artifact": _audit_evidence(root, candidate),
        "conditions": [
            {
                "condition_id": "effort-cal",
                "subsystem": "main_drive",
                "role": "calibration",
                "real_episodes": [_real_binding(calibration, root)],
                "metrics": {},
            },
            {
                "condition_id": "effort-held",
                "subsystem": "main_drive",
                "role": "holdout",
                "held_out_by": ["load"],
                "real_episodes": [_real_binding(holdout, root)],
                "metrics": {
                    direction: {
                        "unit": "N*m",
                        "instrument_uncertainty": 0.1,
                    }
                    for direction in (
                        "positive_torque_nm",
                        "negative_torque_nm",
                    )
                },
            },
        ],
        "actuator_sweeps": {"main_drive": []},
    }
    return candidate, evidence


def test_effort_candidate_itself_must_match_confirmed_known_load_envelope(
    tmp_path: Path,
) -> None:
    candidate, evidence = _known_load_promotion_fixture(tmp_path)
    passed = evaluate_promotion(candidate, evidence, artifact_root=tmp_path)
    assert passed["eligible_for_review"] is True

    payload = candidate.to_dict()
    payload["profile_id"] = "effort-mismatch"
    payload["simulation_physics"]["main_drive"]["effort_limit"] = 4.0
    mismatch = CalibrationProfileV1.from_dict(payload)
    evidence["candidate_profile_sha256"] = sha256_json(mismatch.to_dict())
    with pytest.raises(ContractError, match="does not match its calibration measurement"):
        evaluate_promotion(mismatch, evidence, artifact_root=tmp_path)


@pytest.mark.parametrize(
    ("direct_case", "field_case"),
    [
        ("abad", "target_scale"),
        ("abad", "target_offset"),
        ("contact", "static_friction"),
        ("contact", "dynamic_friction"),
        ("spring", "stiffness"),
        ("rigid_body", "total_mass"),
        ("rigid_body", "com_x"),
        ("rigid_body", "com_y"),
        ("rigid_body", "joint_reference"),
        ("rigid_body", "root_reference"),
        ("effort_limit", "effort_limit"),
    ],
)
def test_direct_measurement_source_binds_candidate_value_to_calibration_trace(
    tmp_path: Path,
    direct_case: str,
    field_case: str,
) -> None:
    if direct_case in {"abad", "contact"}:
        candidate, evidence = _direct_measurement_fixture(
            tmp_path, subsystem=direct_case
        )
    elif direct_case in {"spring", "rigid_body"}:
        candidate, evidence = _manual_route_fixture(
            tmp_path, subsystem=direct_case
        )
    else:
        candidate, evidence = _known_load_promotion_fixture(tmp_path)

    payload = candidate.to_dict()
    if field_case == "target_scale":
        payload["hardware_mapping"]["abad_target_scale"]["abad_0"] += 0.01
    elif field_case == "target_offset":
        payload["hardware_mapping"]["abad_target_offset_rad"]["abad_0"] += 0.01
    elif field_case in {"static_friction", "dynamic_friction"}:
        payload["simulation_physics"]["ground"][field_case] += 0.01
    elif field_case == "stiffness":
        payload["simulation_physics"]["passive_spring"]["damper_0"][
            "stiffness"
        ] += 0.05
    elif field_case == "total_mass":
        payload["simulation_physics"]["mass"]["target_total_mass_kg"] += 0.05
    elif field_case in {"com_x", "com_y"}:
        index = 0 if field_case == "com_x" else 1
        payload["simulation_physics"]["mass"]["reference_planar_com_xy_m"][
            index
        ] += 0.01
    elif field_case == "joint_reference":
        payload["simulation_physics"]["mass"]["reference_joint_position_rad"][
            "main_0"
        ] += 0.01
    elif field_case == "root_reference":
        orientation = payload["simulation_physics"]["mass"][
            "reference_root_orientation_xyzw"
        ]
        payload["simulation_physics"]["mass"][
            "reference_root_orientation_xyzw"
        ] = [-value for value in orientation]
    else:
        payload["simulation_physics"]["main_drive"]["effort_limit"] += 0.05
    tampered = CalibrationProfileV1.from_dict(payload)

    baseline = load_profile(tmp_path / evidence["baseline_profile"]["path"])
    calibration_binding = evidence["conditions"][0]["real_episodes"][0]
    calibration_trace = load_trace(
        tmp_path / calibration_binding["path"], require_managed_dataset=True
    )
    scenario = load_scenario(calibration_trace.manifest.scenario_id)
    conditions = {
        "calibration": {
            "role": "calibration",
            "scenario": scenario,
            "real_traces": [calibration_trace],
        }
    }

    with pytest.raises(ContractError, match="does not match its calibration measurement"):
        _validate_measurement_sources(baseline, tampered, conditions)


def _tamper_bound_episode_metadata(
    root: Path,
    binding: dict[str, str],
    *,
    section: str,
    channel: str,
    value: str,
) -> None:
    episode = root / binding["path"]
    metadata_path = episode / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["metadata"][section][channel] = value
    metadata_sha = _write_json(metadata_path, metadata)

    dataset_manifest_path = episode.parent.parent / "manifest.json"
    dataset_manifest = json.loads(
        dataset_manifest_path.read_text(encoding="utf-8")
    )
    matching = [
        item
        for item in dataset_manifest["episodes"]
        if item["episode_id"] == binding["episode_id"]
    ]
    assert len(matching) == 1
    matching[0]["metadata_sha256"] = metadata_sha
    _write_json(dataset_manifest_path, dataset_manifest)
    binding["metadata_sha256"] = metadata_sha


@pytest.mark.parametrize(
    ("condition_index", "section", "channel", "value", "message"),
    [
        (0, "units", "load_force", "lbf", "unit.*load_force"),
        (1, "frames", "load_force", "force_gauge", "frame.*load_force"),
    ],
    ids=("calibration-unit-lbf", "holdout-wrong-frame"),
)
def test_promotion_authenticates_expected_metadata_for_every_condition_trace(
    tmp_path: Path,
    condition_index: int,
    section: str,
    channel: str,
    value: str,
    message: str,
) -> None:
    candidate, evidence = _known_load_promotion_fixture(tmp_path)
    binding = evidence["conditions"][condition_index]["real_episodes"][0]
    _tamper_bound_episode_metadata(
        tmp_path,
        binding,
        section=section,
        channel=channel,
        value=value,
    )

    with pytest.raises(ContractError, match=message):
        evaluate_promotion(candidate, evidence, artifact_root=tmp_path)


@pytest.mark.parametrize(
    ("profile_section", "field", "value", "message"),
    [
        ("simulation_physics", "damping", 999.0, "main-drive response evidence"),
        ("hardware_mapping", "gear_ratio", 999.0, "unidentifiable profile change"),
        (
            "hardware_mapping",
            "pwm_scale",
            0.01,
            "mapping-specific measured/source evidence",
        ),
        (
            "hardware_mapping",
            "pwm_cap",
            4.0,
            "mapping-specific measured/source evidence",
        ),
    ],
    ids=(
        "manual-load-does-not-cover-damping",
        "manual-load-does-not-cover-gear-ratio",
        "manual-load-does-not-cover-pwm-scale",
        "manual-load-does-not-cover-pwm-cap",
    ),
)
def test_manual_effort_evidence_cannot_validate_unrelated_profile_fields(
    tmp_path: Path,
    profile_section: str,
    field: str,
    value: float,
    message: str,
) -> None:
    candidate, evidence = _known_load_promotion_fixture(tmp_path)
    payload = candidate.to_dict()
    payload["profile_id"] = f"unrelated-{field}"
    if profile_section == "simulation_physics":
        payload[profile_section].setdefault("main_drive", {})[field] = value
    else:
        payload[profile_section].setdefault(field, {})["main_0"] = value
    unrelated = CalibrationProfileV1.from_dict(payload)
    evidence["candidate_profile_sha256"] = sha256_json(unrelated.to_dict())

    with pytest.raises(ContractError, match=message):
        evaluate_promotion(unrelated, evidence, artifact_root=tmp_path)


def test_velocity_limit_change_requires_conditions_that_reach_saturation() -> None:
    baseline = CalibrationProfileV1.from_dict(
        {
            "schema_version": 1,
            "profile_id": "velocity-baseline",
            "hardware_mapping": {},
            "sensor_timing": {},
            "simulation_physics": {"main_drive": {"velocity_limit": 2.0}},
        }
    )
    candidate = CalibrationProfileV1.from_dict(
        {
            "schema_version": 1,
            "profile_id": "velocity-candidate",
            "hardware_mapping": {},
            "sensor_timing": {},
            "simulation_physics": {"main_drive": {"velocity_limit": 1.0}},
        }
    )
    conditions = {
        "ordinary-cal": {
            "role": "calibration",
            "scenario": load_scenario("suspended-main-0-step-coast"),
            "real_traces": [],
        },
        "ordinary-held": {
            "role": "holdout",
            "scenario": load_scenario("suspended-main-5-step-coast"),
            "real_traces": [],
        },
    }

    with pytest.raises(ContractError, match="velocity-limit.*saturation"):
        _validate_changed_field_evidence(baseline, candidate, conditions)


def test_independently_evidenced_effort_and_response_changes_can_be_combined() -> None:
    baseline = CalibrationProfileV1.from_dict(
        {
            "schema_version": 1,
            "profile_id": "combined-baseline",
            "hardware_mapping": {},
            "sensor_timing": {},
            "simulation_physics": {
                "main_drive": {"damping": 0.1, "effort_limit": 1.0}
            },
        }
    )
    candidate = CalibrationProfileV1.from_dict(
        {
            "schema_version": 1,
            "profile_id": "combined-candidate",
            "hardware_mapping": {},
            "sensor_timing": {},
            "simulation_physics": {
                "main_drive": {"damping": 0.2, "effort_limit": 2.0}
            },
        }
    )
    conditions = {
        "response-cal": {
            "role": "calibration",
            "scenario": load_scenario("suspended-main-0-step-coast"),
            "real_traces": [],
        },
        "response-held": {
            "role": "holdout",
            "scenario": load_scenario("suspended-main-5-step-coast"),
            "real_traces": [],
        },
        "effort-cal": {
            "role": "calibration",
            "scenario": load_scenario("manual-load"),
            "real_traces": [],
        },
        "effort-held": {
            "role": "holdout",
            "scenario": load_scenario("manual-load-holdout"),
            "real_traces": [],
        },
    }

    _validate_changed_field_evidence(baseline, candidate, conditions)
