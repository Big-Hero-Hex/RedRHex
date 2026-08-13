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


def _mass_com_arrays() -> dict[str, np.ndarray]:
    return {
        "scale_time_s": np.arange(3, dtype=float),
        "scale_mass": np.array([9.9, 10.0, 10.1]),
        "support_force_time_s": np.arange(3, dtype=float),
        "support_force": np.array([[40.0, 30.0, 30.0]] * 3),
        "support_position_time_s": np.arange(3, dtype=float),
        "support_position": np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        ),
        "repeat_index": np.arange(3),
    }


def _spring_arrays() -> dict[str, np.ndarray]:
    angle = np.tile(np.array([0.0, 0.1, 0.2]), 3)
    time_s = np.arange(angle.size, dtype=float) * 0.1
    return {
        "angle_time_s": time_s,
        "angle": angle,
        "load_force_time_s": time_s.copy(),
        "load_force": angle * 100.0,
        "lever_arm_time_s": time_s.copy(),
        "lever_arm": np.full(angle.size, 0.1),
        "repeat_index": np.repeat(np.arange(3), 3),
    }


def _torsion_spring_arrays(
    *, envelope_fractions: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)
) -> dict[str, np.ndarray]:
    maximum_deflection = 0.5
    levels = np.asarray(envelope_fractions) * maximum_deflection
    one_repeat = np.concatenate(
        (-levels, levels, -levels[::-1], levels[::-1])
    )
    angle = np.tile(one_repeat, 3)
    time_s = np.arange(angle.size, dtype=float) * 0.1
    torque = angle * 10.0
    return {
        "angle_time_s": time_s,
        "angle": angle,
        "load_force_time_s": time_s.copy(),
        "load_force": np.abs(torque) / 0.1,
        "lever_arm_time_s": time_s.copy(),
        "lever_arm": np.full(angle.size, 0.1),
        "torque_direction": np.sign(torque),
        "sweep_branch": np.tile(
            np.repeat([1.0, -1.0], 2 * len(levels)), 3
        ),
        "repeat_index": np.repeat(np.arange(3), one_repeat.size),
    }


def _torsion_spring_constants(
    *, aliases: list[str] | None = None
) -> dict[str, Any]:
    constants: dict[str, Any] = {
        "rest_position_rad": 0.0,
        "mechanical_owner_approval": {
            "owner": "mechanical-owner",
            "fixture_id": "torsion-spring-bench-v1",
            "maximum_safe_deflection_rad": 0.5,
        },
    }
    if aliases is not None:
        constants["applies_to_spring_aliases"] = aliases
    return constants


def _known_load_arrays() -> dict[str, np.ndarray]:
    time_s = np.arange(6, dtype=float)
    return {
        "load_force_time_s": time_s,
        "load_force": np.array([20.0, 20.0, 21.0, 21.0, 19.0, 19.0]),
        "lever_arm_time_s": time_s.copy(),
        "lever_arm": np.full(6, 0.1),
        "command_time_s": time_s.copy(),
        "command": np.full(6, 0.25),
        "direction_time_s": time_s.copy(),
        "direction": np.tile(np.array([1.0, -1.0]), 3),
        "saturation_confirmed": np.ones(6),
        "repeat_index": np.repeat(np.arange(3), 2),
    }


def _mass_reference_pose() -> dict[str, Any]:
    return {
        "reference_joint_position_rad": {
            f"{group}_{index}": 0.0
            for group in ("main", "abad", "damper")
            for index in range(6)
        },
        "reference_root_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
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
    if scenario_id == "mass-com":
        units = {
            "scale_mass": "kg",
            "support_force": "N",
            "support_position": "m",
            "repeat_index": "1",
        }
        return units, {name: "root" for name in units}
    if scenario_id == "spring":
        units = {
            "load_force": "N",
            "lever_arm": "m",
            "angle": "rad",
            "repeat_index": "1",
        }
        return units, {name: "damper_0" for name in units}
    if scenario_id in {"torsion-spring", "torsion-spring-holdout"}:
        units = {
            "load_force": "N",
            "lever_arm": "m",
            "angle": "rad",
            "torque_direction": "1",
            "sweep_branch": "1",
            "repeat_index": "1",
        }
        return units, {name: "damper_0" for name in units}
    if scenario_id == "manual-load":
        units = {
            "load_force": "N",
            "lever_arm": "m",
            "command": "normalized",
            "direction": "1",
            "saturation_confirmed": "1",
            "repeat_index": "1",
        }
        return units, {name: "main_0" for name in units}
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
    calibration_constants: dict[str, Any] | None = None,
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
            "calibration_constants": calibration_constants or {},
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


def test_direct_mass_spring_and_known_load_measurements_update_profile(
    tmp_path: Path,
) -> None:
    from tools.sim2real.profile_measurements import apply_measurements_to_profile

    mass, mass_record = _managed_trace(
        tmp_path,
        scenario_id="mass-com",
        episode_id="mass-run-1",
        arrays=_mass_com_arrays(),
        calibration_constants=_mass_reference_pose(),
    )
    spring, spring_record = _managed_trace(
        tmp_path,
        scenario_id="spring",
        episode_id="spring-run-1",
        arrays=_spring_arrays(),
    )
    load, load_record = _managed_trace(
        tmp_path,
        scenario_id="manual-load",
        episode_id="load-run-1",
        arrays=_known_load_arrays(),
    )

    candidate = apply_measurements_to_profile(
        _baseline(),
        profile_id="static-candidate",
        trace_paths=[mass, spring, load],
    )

    mass_profile = candidate.simulation_physics["mass"]
    assert mass_profile["target_total_mass_kg"] == pytest.approx(10.0)
    assert mass_profile["reference_planar_com_xy_m"] == pytest.approx([0.3, 0.3])
    assert set(mass_profile["reference_joint_position_rad"]) == {
        f"{group}_{index}"
        for group in ("main", "abad", "damper")
        for index in range(6)
    }
    assert set(mass_profile["reference_joint_position_rad"].values()) == {0.0}
    assert mass_profile["reference_root_orientation_xyzw"] == [0.0, 0.0, 0.0, 1.0]
    assert candidate.simulation_physics["passive_spring"]["damper_0"] == {
        "stiffness": pytest.approx(10.0)
    }
    assert candidate.simulation_physics["main_drive"]["effort_limit"] == pytest.approx(
        2.0
    )
    expected = {
        "mass_com": (mass_record, "mass-com", "mass_com", "root"),
        "passive_spring:damper_0": (
            spring_record,
            "spring",
            "torsional_spring",
            "damper_0",
        ),
        "main_drive_effort_limit:main_0": (
            load_record,
            "manual-load",
            "torque_saturation",
            "main_0",
        ),
    }
    for key, (record, scenario_id, metric_kind, frame) in expected.items():
        assert candidate.measurement_sources[key] == {
            "trace_sha256": record["trace_sha256"],
            "metadata_sha256": record["metadata_sha256"],
            "scenario_id": scenario_id,
            "scenario_sha256": sha256_json(load_scenario(scenario_id).to_dict()),
            "source": "real",
            "metric_kind": metric_kind,
            "frame": frame,
            "repeat_count": 3,
            "dataset_id": "bench-20260713",
            "episode_id": record["episode_id"],
        }


def test_representative_torsion_spring_measurement_propagates_to_all_six_aliases(
    tmp_path: Path,
) -> None:
    from tools.sim2real.profile_measurements import apply_measurements_to_profile

    aliases = [f"damper_{index}" for index in range(6)]
    spring, spring_record = _managed_trace(
        tmp_path,
        scenario_id="torsion-spring",
        episode_id="representative-spring-run",
        arrays=_torsion_spring_arrays(),
        calibration_constants=_torsion_spring_constants(aliases=aliases),
    )
    baseline_payload = _baseline().to_dict()
    baseline_payload["simulation_physics"]["damper"] = {
        "stiffness": 190.0,
        "damping": 19.0,
    }
    baseline_payload["simulation_physics"]["passive_spring"] = {
        alias: {"stiffness": 180.0, "damping": 18.0} for alias in aliases
    }

    candidate = apply_measurements_to_profile(
        CalibrationProfileV1.from_dict(baseline_payload),
        profile_id="representative-spring-candidate",
        trace_paths=[spring],
    )

    assert candidate.simulation_physics["passive_spring"] == {
        alias: {
            "stiffness": pytest.approx(10.0),
            "damping": 0.0,
            **({"rest_position_rad": 0.0} if alias == "damper_0" else {}),
        }
        for alias in aliases
    }
    assert set(candidate.measurement_sources) >= {"passive_spring:damper_0"}
    assert candidate.measurement_sources["passive_spring:damper_0"] == {
        "trace_sha256": spring_record["trace_sha256"],
        "metadata_sha256": spring_record["metadata_sha256"],
        "scenario_id": "torsion-spring",
        "scenario_sha256": sha256_json(load_scenario("torsion-spring").to_dict()),
        "source": "real",
        "metric_kind": "torsional_spring",
        "frame": "damper_0",
        "repeat_count": 3,
        "dataset_id": "bench-20260713",
        "episode_id": "representative-spring-run",
        "applies_to": aliases,
        "rest_position_rad": 0.0,
        "episode_path": str(spring.resolve()),
    }


@pytest.mark.parametrize(
    "aliases",
    [
        [f"damper_{index}" for index in range(5)],
        ["damper_0", "damper_1", "damper_2", "damper_3", "damper_4", "damper_4"],
        ["damper_0", "damper_1", "damper_2", "damper_3", "damper_4", "damper_6"],
        [f"damper_{index}" for index in reversed(range(6))],
    ],
    ids=("partial", "duplicate", "unknown", "out-of-order"),
)
def test_representative_torsion_spring_rejects_invalid_alias_declarations(
    tmp_path: Path, aliases: list[str]
) -> None:
    from tools.sim2real.profile_measurements import apply_measurements_to_profile

    spring, _ = _managed_trace(
        tmp_path,
        scenario_id="torsion-spring",
        episode_id="invalid-representative-spring",
        arrays=_torsion_spring_arrays(),
        calibration_constants=_torsion_spring_constants(aliases=aliases),
    )

    with pytest.raises(ContractError, match="applies_to_spring_aliases"):
        apply_measurements_to_profile(
            _baseline(), profile_id="invalid-spring", trace_paths=[spring]
        )


def test_representative_torsion_spring_rejects_a_nonlinear_calibration(
    tmp_path: Path,
) -> None:
    from tools.sim2real.profile_measurements import apply_measurements_to_profile

    arrays = _torsion_spring_arrays()
    arrays["load_force"] = np.full(arrays["load_force"].shape, 5.0)
    spring, _ = _managed_trace(
        tmp_path,
        scenario_id="torsion-spring",
        episode_id="nonlinear-representative-spring",
        arrays=arrays,
        calibration_constants=_torsion_spring_constants(
            aliases=[f"damper_{index}" for index in range(6)]
        ),
    )

    with pytest.raises(ContractError, match="nonlinear or hysteretic"):
        apply_measurements_to_profile(
            _baseline(), profile_id="nonlinear-spring", trace_paths=[spring]
        )


def test_representative_spring_requires_mechanical_owner_fixture_approval(
    tmp_path: Path,
) -> None:
    from tools.sim2real.profile_measurements import apply_measurements_to_profile

    spring, _ = _managed_trace(
        tmp_path,
        scenario_id="torsion-spring",
        episode_id="unapproved-representative-spring",
        arrays=_torsion_spring_arrays(),
        calibration_constants={
            "rest_position_rad": 0.0,
            "applies_to_spring_aliases": [f"damper_{index}" for index in range(6)],
        },
    )

    with pytest.raises(ContractError, match="mechanical owner approval"):
        apply_measurements_to_profile(
            _baseline(), profile_id="unapproved-spring", trace_paths=[spring]
        )


def test_representative_spring_rejects_samples_outside_the_approved_envelope(
    tmp_path: Path,
) -> None:
    from tools.sim2real.profile_measurements import apply_measurements_to_profile

    aliases = [f"damper_{index}" for index in range(6)]
    constants = _torsion_spring_constants(aliases=aliases)
    constants["mechanical_owner_approval"]["maximum_safe_deflection_rad"] = 0.3
    spring, _ = _managed_trace(
        tmp_path,
        scenario_id="torsion-spring",
        episode_id="out-of-envelope-spring",
        arrays=_torsion_spring_arrays(),
        calibration_constants=constants,
    )

    with pytest.raises(ContractError, match="approved.*deflection"):
        apply_measurements_to_profile(
            _baseline(), profile_id="out-of-envelope", trace_paths=[spring]
        )


def test_representative_spring_requires_the_reviewed_calibration_envelope_levels(
    tmp_path: Path,
) -> None:
    from tools.sim2real.profile_measurements import apply_measurements_to_profile

    aliases = [f"damper_{index}" for index in range(6)]
    spring, _ = _managed_trace(
        tmp_path,
        scenario_id="torsion-spring",
        episode_id="wrong-calibration-levels",
        arrays=_torsion_spring_arrays(envelope_fractions=(0.3, 0.5, 0.7)),
        calibration_constants=_torsion_spring_constants(aliases=aliases),
    )

    with pytest.raises(ContractError, match="20/40/60/80"):
        apply_measurements_to_profile(
            _baseline(), profile_id="wrong-calibration-levels", trace_paths=[spring]
        )


@pytest.mark.parametrize(
    "approval_envelope",
    [
        {"maximum_safe_load_n": 50.0},
        {"maximum_safe_torque_nm": 5.0},
    ],
    ids=("load", "torque"),
)
def test_representative_spring_accepts_owner_approved_load_or_torque_envelopes(
    tmp_path: Path, approval_envelope: dict[str, float]
) -> None:
    from tools.sim2real.profile_measurements import apply_measurements_to_profile

    aliases = [f"damper_{index}" for index in range(6)]
    constants = _torsion_spring_constants(aliases=aliases)
    constants["mechanical_owner_approval"] = {
        "owner": "mechanical-owner",
        "fixture_id": "torsion-spring-bench-v1",
        **approval_envelope,
    }
    spring, _ = _managed_trace(
        tmp_path,
        scenario_id="torsion-spring",
        episode_id=f"approved-{next(iter(approval_envelope))}",
        arrays=_torsion_spring_arrays(),
        calibration_constants=constants,
    )

    candidate = apply_measurements_to_profile(
        _baseline(), profile_id="alternate-safe-envelope", trace_paths=[spring]
    )

    assert candidate.simulation_physics["passive_spring"]["damper_0"][
        "stiffness"
    ] == pytest.approx(10.0)


def test_verified_calibration_and_holdout_report_real_world_spring_quality(
    tmp_path: Path,
) -> None:
    from tools.sim2real.profile_measurements import (
        apply_measurements_to_profile,
        evaluate_torsional_spring_quality,
    )

    aliases = [f"damper_{index}" for index in range(6)]
    calibration, calibration_record = _managed_trace(
        tmp_path,
        scenario_id="torsion-spring",
        episode_id="quality-calibration",
        arrays=_torsion_spring_arrays(),
        calibration_constants=_torsion_spring_constants(aliases=aliases),
    )
    holdout, holdout_record = _managed_trace(
        tmp_path,
        scenario_id="torsion-spring-holdout",
        episode_id="quality-holdout",
        arrays=_torsion_spring_arrays(envelope_fractions=(0.3, 0.5, 0.7)),
        calibration_constants=_torsion_spring_constants(),
    )

    report = evaluate_torsional_spring_quality(calibration, holdout)

    assert report["quality"]["accepted"] is True
    assert report["holdout"]["rmse_full_scale_ratio"] == pytest.approx(0.0)
    assert report["provenance"] == {
        "calibration_trace_sha256": calibration_record["trace_sha256"],
        "calibration_metadata_sha256": calibration_record["metadata_sha256"],
        "holdout_trace_sha256": holdout_record["trace_sha256"],
        "holdout_metadata_sha256": holdout_record["metadata_sha256"],
    }
    candidate = apply_measurements_to_profile(
        _baseline(),
        profile_id="accepted-representative-spring",
        trace_paths=[calibration, holdout],
    )
    assert candidate.measurement_sources["passive_spring:damper_0"][
        "quality_validation"
    ] == {
        "accepted": True,
        "gates": {
            "r_squared": True,
            "heldout_rmse": True,
            "stiffness_cv": True,
            "hysteresis": True,
            "neutral_model_heldout_rmse": True,
        },
        "calibration_trace_sha256": calibration_record["trace_sha256"],
        "calibration_metadata_sha256": calibration_record["metadata_sha256"],
        "holdout_trace_sha256": holdout_record["trace_sha256"],
        "holdout_metadata_sha256": holdout_record["metadata_sha256"],
        "holdout_scenario_id": "torsion-spring-holdout",
        "holdout_scenario_sha256": sha256_json(
            load_scenario("torsion-spring-holdout").to_dict()
        ),
        "source": "real",
        "dataset_id": "bench-20260713",
        "episode_id": "quality-holdout",
        "episode_path": str(holdout.resolve()),
    }
    from tools.sim2real.profile_measurements import verify_representative_spring_source

    verified = verify_representative_spring_source(
        candidate.measurement_sources["passive_spring:damper_0"]
    )
    assert verified["quality"]["accepted"] is True


def test_real_world_spring_quality_rejects_a_bad_heldout_prediction(
    tmp_path: Path,
) -> None:
    from tools.sim2real.profile_measurements import (
        apply_measurements_to_profile,
        evaluate_torsional_spring_quality,
    )

    aliases = [f"damper_{index}" for index in range(6)]
    calibration, _ = _managed_trace(
        tmp_path,
        scenario_id="torsion-spring",
        episode_id="bad-holdout-calibration",
        arrays=_torsion_spring_arrays(),
        calibration_constants=_torsion_spring_constants(aliases=aliases),
    )
    holdout_arrays = _torsion_spring_arrays(
        envelope_fractions=(0.3, 0.5, 0.7)
    )
    holdout_arrays["load_force"] *= 1.2
    holdout, _ = _managed_trace(
        tmp_path,
        scenario_id="torsion-spring-holdout",
        episode_id="bad-holdout",
        arrays=holdout_arrays,
        calibration_constants=_torsion_spring_constants(),
    )

    report = evaluate_torsional_spring_quality(calibration, holdout)

    assert report["quality"]["accepted"] is False
    assert report["quality"]["gates"]["heldout_rmse"] is False
    assert report["quality"]["gates"]["neutral_model_heldout_rmse"] is False
    with pytest.raises(ContractError, match="heldout_rmse"):
        apply_measurements_to_profile(
            _baseline(),
            profile_id="rejected-spring-model",
            trace_paths=[calibration, holdout],
        )


def test_mass_profile_measurement_requires_recorded_reference_pose(
    tmp_path: Path,
) -> None:
    from tools.sim2real.profile_measurements import apply_measurements_to_profile

    mass, _ = _managed_trace(
        tmp_path,
        scenario_id="mass-com",
        episode_id="mass-missing-pose",
        arrays=_mass_com_arrays(),
    )

    with pytest.raises(ContractError, match="mass-com.*reference pose"):
        apply_measurements_to_profile(
            _baseline(), profile_id="unsafe-mass", trace_paths=[mass]
        )


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
