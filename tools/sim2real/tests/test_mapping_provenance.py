from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.sim2real.characterization import scenario_schedule, scenario_step_count
from tools.sim2real.compare import compare_traces
from tools.sim2real.contracts import CalibrationProfileV1, ContractError
from tools.sim2real.dataset import import_real_dataset
from tools.sim2real.provenance import validate_real_trace_provenance
from tools.sim2real.scenarios import load_scenario
from tools.sim2real.traces import load_trace, sha256_json, write_trace


def _profile(
    *,
    encoder_joints: tuple[str, ...] = ("main_0",),
    command_mapping: bool = True,
) -> CalibrationProfileV1:
    hardware_mapping: dict[str, dict[str, float | int]] = {
        "encoder_counts_per_rev": {joint: 54984.83 for joint in encoder_joints},
        "encoder_zero_count": {joint: 0.0 for joint in encoder_joints},
        "encoder_sign": {joint: 1 for joint in encoder_joints},
    }
    if command_mapping:
        hardware_mapping.update(
            {
                "joint_direction": {"main_0": 1},
                "pwm_scale": {"main_0": 1.0 / 120.0},
                "pwm_cap": {"main_0": 500.0 / 120.0},
            }
        )
    return CalibrationProfileV1.from_dict(
        {
            "schema_version": 1,
            "profile_id": "measured-main-mapping",
            "hardware_mapping": hardware_mapping,
            "sensor_timing": {},
            "simulation_physics": {},
        }
    )


def _arrays(scenario_id: str = "main-step") -> dict[str, np.ndarray]:
    scenario = load_scenario(scenario_id)
    steps = scenario_step_count(scenario)
    time_s = np.arange(steps, dtype=np.float64) / 120.0
    command = np.asarray(
        [item.value for item in scenario_schedule(scenario, steps)],
        dtype=np.float64,
    )
    return {
        "command_time_s": time_s,
        "command": command,
        "position_time_s": time_s.copy(),
        "position": np.cumsum(command) / 120.0,
    }


def _write_real(
    output: Path,
    profile: CalibrationProfileV1,
    *,
    scenario_id: str = "main-step",
    constants: dict[str, object] | None = None,
) -> Path:
    scenario = load_scenario(scenario_id)
    raw = output.parent / f"{output.name}.raw"
    raw.write_bytes(b"immutable real mapping evidence")
    source = f"profile:{profile.profile_id}"
    write_trace(
        output,
        _arrays(scenario_id),
        scenario=scenario,
        source="real",
        source_path=raw,
        profile=profile,
        metadata={
            "units": {"command": "rad/s", "position": "rad"},
            "frames": {"command": scenario.joint, "position": scenario.joint},
            "calibration_constants": constants
            or {
                "position_mapping_source": source,
                "requested_command_source": source,
            },
        },
    )
    return output


def _write_sim(output: Path, *, scenario_id: str = "main-step") -> Path:
    scenario = load_scenario(scenario_id)
    write_trace(
        output,
        _arrays(scenario_id),
        scenario=scenario,
        source="sim",
        metadata={
            "units": {"command": "rad/s", "position": "rad"},
            "frames": {"command": scenario.joint, "position": scenario.joint},
        },
    )
    return output


def test_profiled_trace_records_content_hashed_mapping_snapshot(tmp_path: Path) -> None:
    profile = _profile()
    real = _write_real(tmp_path / "real", profile)
    loaded = load_trace(real)

    snapshot = loaded.manifest.metadata["calibration_constants"][
        "hardware_mapping_snapshot"
    ]
    assert snapshot == {
        "schema_version": 1,
        "profile_id": profile.profile_id,
        "profile_sha256": sha256_json(profile.to_dict()),
        "hardware_mapping": profile.hardware_mapping,
    }
    assert loaded.manifest.provenance["hardware_mapping_sha256"] == sha256_json(
        snapshot
    )
    assert compare_traces(real, _write_sim(tmp_path / "sim"))["scenario_id"] == (
        "main-step"
    )


def test_fabricated_source_labels_cannot_hide_empty_mapping(tmp_path: Path) -> None:
    empty = CalibrationProfileV1.from_dict(
        {
            "schema_version": 1,
            "profile_id": "empty-mapping",
            "hardware_mapping": {},
            "sensor_timing": {},
            "simulation_physics": {},
        }
    )
    source = f"profile:{empty.profile_id}"
    real = _write_real(
        tmp_path / "real",
        empty,
        constants={
            "position_mapping_source": source,
            "requested_command_source": source,
        },
    )

    with pytest.raises(ContractError, match="encoder_counts_per_rev.*main_0"):
        compare_traces(real, _write_sim(tmp_path / "sim"))


def test_mapping_snapshot_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    real = _write_real(tmp_path / "real", _profile())
    metadata_path = real / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["metadata"]["calibration_constants"]["hardware_mapping_snapshot"][
        "hardware_mapping"
    ]["pwm_scale"]["main_0"] = 99.0
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ContractError, match="mapping snapshot hash"):
        compare_traces(real, _write_sim(tmp_path / "sim"))


def test_profile_command_source_requires_selected_joint_pwm_mapping(
    tmp_path: Path,
) -> None:
    payload = _profile(command_mapping=False).to_dict()
    payload["hardware_mapping"]["joint_direction"] = {"main_0": 1}
    profile = CalibrationProfileV1.from_dict(payload)
    real = _write_real(tmp_path / "real", profile)

    with pytest.raises(ContractError, match="pwm_scale.*main_0"):
        compare_traces(real, _write_sim(tmp_path / "sim"))


def test_public_import_accepts_manual_load_without_encoder_provenance(
    tmp_path: Path,
) -> None:
    source = tmp_path / "known-load.npz"
    time_s = np.arange(3, dtype=float)
    np.savez(
        source,
        load_force_time_s=time_s,
        load_force=np.array([20.0, 21.0, 19.0]),
        lever_arm_time_s=time_s,
        lever_arm=np.full(3, 0.1),
        command_time_s=time_s,
        command=np.full(3, 0.25),
        direction_time_s=time_s,
        direction=np.ones(3),
        repeat_index=np.arange(3),
    )
    units = {
        "load_force": "N",
        "lever_arm": "m",
        "command": "normalized",
        "direction": "1",
        "repeat_index": "1",
    }
    imported = import_real_dataset(
        source,
        tmp_path / "managed",
        dataset_id="known-load",
        episode_id="load-1",
        scenario="manual-load",
        units=units,
        frames={name: "main_0" for name in units},
        latency_clock="operator_monotonic",
        profile=_profile(),
    )

    trace = load_trace(imported.episode, require_managed_dataset=True)
    validate_real_trace_provenance(trace, load_scenario("manual-load"))
    assert "position_mapping_source" not in trace.manifest.metadata["calibration_constants"]


def test_replay_mapping_requires_all_six_encoder_snapshots(tmp_path: Path) -> None:
    scenario = load_scenario("suspended-main-0-step-coast")
    joints = tuple(f"main_{index}" for index in range(6))
    profile = _profile(encoder_joints=joints)
    source = f"profile:{profile.profile_id}"
    valid = _write_real(
        tmp_path / "valid",
        profile,
        scenario_id=scenario.scenario_id,
        constants={
            "position_mapping_source": source,
            "requested_command_source": source,
            "all_main_position_mapping_source": source,
        },
    )

    validate_real_trace_provenance(
        load_trace(valid, scenario=scenario),
        scenario,
        require_all_main_positions=True,
    )

    incomplete = _profile(encoder_joints=joints[:-1])
    incomplete_source = f"profile:{incomplete.profile_id}"
    invalid = _write_real(
        tmp_path / "invalid",
        incomplete,
        scenario_id=scenario.scenario_id,
        constants={
            "position_mapping_source": incomplete_source,
            "requested_command_source": incomplete_source,
            "all_main_position_mapping_source": incomplete_source,
        },
    )
    with pytest.raises(ContractError, match="encoder_counts_per_rev.*main_5"):
        validate_real_trace_provenance(
            load_trace(invalid, scenario=scenario),
            scenario,
            require_all_main_positions=True,
        )
