from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tools.sim2real.contracts import CalibrationProfileV1, ContractError
from tools.sim2real.scenarios import load_scenario
from tools.sim2real.traces import sha256_file, sha256_json, write_trace


def _baseline() -> CalibrationProfileV1:
    return CalibrationProfileV1.from_dict(
        {
            "schema_version": 1,
            "profile_id": "baseline",
            "description": "preserve me",
            "hardware_mapping": {"pwm_cap": {"main_0": 4.1667}},
            "sensor_timing": {"aggregate_command_delay_s": 0.02},
            "simulation_physics": {
                "main_drive": {"damping": 1.2},
                "ground": {"restitution": 0.1},
            },
            "measurement_sources": {"mass": "c" * 64},
        }
    )


def _abad_arrays(*, repeat_count: int = 3) -> dict[str, np.ndarray]:
    command = np.tile(np.array([-0.2, 0.0, 0.2]), repeat_count)
    time_s = np.arange(command.size, dtype=float) * 0.1
    return {
        "command_time_s": time_s,
        "command": command,
        "position_time_s": time_s.copy(),
        "position": 1.25 * command - 0.03,
        "repeat_index": np.repeat(np.arange(repeat_count), 3),
        "settled": np.ones(command.size),
    }


def _friction_arrays() -> dict[str, np.ndarray]:
    return {
        "static_time_s": np.arange(3, dtype=float),
        "breakaway_force": np.array([20.0, 21.0, 19.0]),
        "static_normal_load": np.full(3, 100.0),
        "static_repeat_index": np.arange(3),
        "dynamic_time_s": np.arange(9, dtype=float) * 0.1,
        "dynamic_pull_force": np.full(9, 15.0),
        "dynamic_normal_load": np.full(9, 100.0),
        "dynamic_speed": np.full(9, 0.05),
        "dynamic_repeat_index": np.repeat(np.arange(3), 3),
    }


def _units_and_frames(scenario_id: str) -> tuple[dict[str, str], dict[str, str]]:
    if scenario_id == "abad-static":
        units = {
            "command": "rad",
            "position": "rad",
            "repeat_index": "1",
            "settled": "1",
        }
        frames = {name: "abad_0" for name in units}
        return units, frames
    if scenario_id == "friction":
        units = {
            "breakaway_force": "N",
            "static_normal_load": "N",
            "static_repeat_index": "1",
            "dynamic_pull_force": "N",
            "dynamic_normal_load": "N",
            "dynamic_speed": "m/s",
            "dynamic_repeat_index": "1",
        }
        frames = {name: "foot_0/ground" for name in units}
        return units, frames
    raise AssertionError(f"unsupported fixture scenario: {scenario_id}")


def _managed_trace(
    tmp_path: Path,
    *,
    scenario_id: str,
    episode_id: str,
    arrays: dict[str, np.ndarray],
    source: str = "real",
    units: dict[str, str] | None = None,
    frames: dict[str, str] | None = None,
) -> tuple[Path, dict[str, Any]]:
    dataset_id = "bench-20260713"
    dataset = tmp_path / "datasets" / "sim2real" / dataset_id
    raw = dataset / "raw" / f"{episode_id}.raw"
    episode = dataset / "episodes" / episode_id
    raw.parent.mkdir(parents=True, exist_ok=True)
    episode.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(f"immutable raw source for {episode_id}".encode())
    canonical_units, canonical_frames = _units_and_frames(scenario_id)
    manifest = write_trace(
        episode,
        arrays,
        scenario=load_scenario(scenario_id),
        source=source,
        source_path=raw,
        metadata={
            "units": units or canonical_units,
            "frames": frames or canonical_frames,
            "joint_order": [load_scenario(scenario_id).joint],
        },
    )
    dataset_manifest_path = dataset / "manifest.json"
    if dataset_manifest_path.exists():
        dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    else:
        dataset_manifest = {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "raw": [],
            "episodes": [],
        }
    raw_relative = raw.relative_to(dataset).as_posix()
    dataset_manifest["raw"].append(
        {"path": raw_relative, "sha256": manifest.provenance["source_sha256"]}
    )
    episode_record = {
        "episode_id": episode_id,
        "scenario_id": scenario_id,
        "path": episode.relative_to(dataset).as_posix(),
        "trace_sha256": manifest.provenance["trace_sha256"],
        "metadata_sha256": sha256_file(episode / "metadata.json"),
        "raw_path": raw_relative,
    }
    dataset_manifest["episodes"].append(episode_record)
    dataset_manifest_path.write_text(
        json.dumps(dataset_manifest, sort_keys=True), encoding="utf-8"
    )
    return episode, episode_record


def test_verified_measurement_traces_update_profile_and_record_managed_sources(
    tmp_path: Path,
) -> None:
    from tools.sim2real.profile_measurements import apply_measurements_to_profile

    abad, abad_record = _managed_trace(
        tmp_path,
        scenario_id="abad-static",
        episode_id="abad-run-1",
        arrays=_abad_arrays(),
    )
    friction, friction_record = _managed_trace(
        tmp_path,
        scenario_id="friction",
        episode_id="friction-run-1",
        arrays=_friction_arrays(),
    )

    candidate = apply_measurements_to_profile(
        _baseline(),
        profile_id="measured-candidate",
        trace_paths=[abad, friction],
    )

    assert candidate.description == "preserve me"
    assert candidate.hardware_mapping["pwm_cap"] == {"main_0": 4.1667}
    assert candidate.hardware_mapping["abad_target_scale"] == {
        "abad_0": pytest.approx(1.25)
    }
    assert candidate.hardware_mapping["abad_target_offset_rad"] == {
        "abad_0": pytest.approx(-0.03)
    }
    assert candidate.simulation_physics["main_drive"] == {"damping": 1.2}
    assert candidate.simulation_physics["ground"] == {
        "restitution": 0.1,
        "static_friction": pytest.approx(0.2),
        "dynamic_friction": pytest.approx(0.15),
    }
    assert candidate.measurement_sources["mass"] == "c" * 64
    assert candidate.measurement_sources["abad_target:abad_0"] == {
        "trace_sha256": abad_record["trace_sha256"],
        "metadata_sha256": abad_record["metadata_sha256"],
        "scenario_id": "abad-static",
        "scenario_sha256": sha256_json(load_scenario("abad-static").to_dict()),
        "source": "real",
        "metric_kind": "abad_static_mapping",
        "frame": "abad_0",
        "repeat_count": 3,
        "dataset_id": "bench-20260713",
        "episode_id": "abad-run-1",
    }
    assert candidate.measurement_sources["ground_friction"] == {
        "trace_sha256": friction_record["trace_sha256"],
        "metadata_sha256": friction_record["metadata_sha256"],
        "scenario_id": "friction",
        "scenario_sha256": sha256_json(load_scenario("friction").to_dict()),
        "source": "real",
        "metric_kind": "ground_friction",
        "frame": "foot_0/ground",
        "repeat_count": 3,
        "dataset_id": "bench-20260713",
        "episode_id": "friction-run-1",
    }


def test_measurement_api_cannot_accept_caller_metrics_or_hashes() -> None:
    from tools.sim2real.profile_measurements import apply_measurements_to_profile

    parameters = inspect.signature(apply_measurements_to_profile).parameters
    assert "abad_metrics" not in parameters
    assert "friction_metrics" not in parameters
    assert not any(name.endswith("trace_sha256") for name in parameters)


@pytest.mark.parametrize("failure", ["unmanaged", "sim", "units", "frames", "repeats"])
def test_measurement_profile_rejects_unverified_or_incompatible_evidence(
    tmp_path: Path,
    failure: str,
) -> None:
    from tools.sim2real.profile_measurements import apply_measurements_to_profile

    units, frames = _units_and_frames("abad-static")
    arrays = _abad_arrays(repeat_count=2 if failure == "repeats" else 3)
    if failure == "units":
        units = {**units, "position": "deg"}
    if failure == "frames":
        frames = {**frames, "position": "phone"}
    trace, _ = _managed_trace(
        tmp_path,
        scenario_id="abad-static",
        episode_id=f"abad-{failure}",
        arrays=arrays,
        source="sim" if failure == "sim" else "real",
        units=units,
        frames=frames,
    )
    if failure == "unmanaged":
        unmanaged = tmp_path / "unmanaged"
        unmanaged.mkdir()
        for name in ("trace.npz", "metadata.json"):
            (unmanaged / name).write_bytes((trace / name).read_bytes())
        trace = unmanaged

    with pytest.raises(ContractError):
        apply_measurements_to_profile(
            _baseline(), profile_id="invalid-evidence", trace_paths=[trace]
        )
