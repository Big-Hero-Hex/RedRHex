from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.sim2real.contracts import (
    CalibrationProfileV1,
    ContractError,
    ScenarioSpecV1,
    TraceManifestV1,
)
from tools.sim2real.scenarios import load_scenario, list_scenarios


def test_versioned_contracts_round_trip() -> None:
    scenario = ScenarioSpecV1.from_dict(
        {
            "schema_version": 1,
            "scenario_id": "main-step",
            "name": "Main drive step",
            "subsystem": "main_drive",
            "experiment_kind": "step",
            "joint": "main_0",
            "command_segments": [
                {"duration_s": 0.25, "value": 0.0},
                {"duration_s": 0.75, "value": 1.0},
            ],
            "repeats": 3,
            "required_channels": ["command", "position"],
            "time_bases": {"command": "command_time_s", "position": "position_time_s"},
            "split": "calibration",
            "scene_mode": "fixed_base",
            "safety_class": "low_energy",
        }
    )
    manifest = TraceManifestV1.from_dict(
        {
            "schema_version": 1,
            "scenario_id": scenario.scenario_id,
            "source": "real",
            "trace_file": "trace.npz",
            "channels": ["command_time_s", "command", "position_time_s", "position"],
            "time_bases": scenario.time_bases,
            "sample_counts": {"command": 2, "position": 3},
            "provenance": {"trace_sha256": "0" * 64, "scenario_sha256": "1" * 64},
            "metadata": {
                "units": {"command": "normalized", "position": "rad"},
                "frames": {"command": "actuator", "position": "joint"},
                "joint_order": ["main_0"],
                "clock": {
                    "source": "independent",
                    "timestamp_semantics": "relative_monotonic",
                    "time_unit": "s"
                },
                "scenario_schema_version": 1,
                "scenario_sha256": "1" * 64,
                "git_sha": None,
                "asset_sha256": None,
                "config_sha256": None,
                "calibration_constants": {},
                "raw_data_sha256": "2" * 64,
            },
        }
    )
    profile = CalibrationProfileV1.from_dict(
        {
            "schema_version": 1,
            "profile_id": "baseline",
            "hardware_mapping": {
                "joint_direction": {"main_0": 1},
                "joint_offset_rad": {"main_0": 0.0},
                "gear_ratio": {"main_0": 1.0},
            },
            "sensor_timing": {"command_delay_s": 0.01, "sample_period_s": 0.005},
            "simulation_physics": {
                "main_drive": {"damping": 0.2, "stiffness": 0.0},
                "ground": {"static_friction": 0.8, "dynamic_friction": 0.7},
            },
        }
    )

    assert ScenarioSpecV1.from_dict(scenario.to_dict()) == scenario
    assert TraceManifestV1.from_dict(manifest.to_dict()) == manifest
    assert CalibrationProfileV1.from_dict(profile.to_dict()) == profile


@pytest.mark.parametrize(
    ("factory", "payload"),
    [
        (ScenarioSpecV1.from_dict, {"schema_version": 2}),
        (TraceManifestV1.from_dict, {"schema_version": "1"}),
        (
            CalibrationProfileV1.from_dict,
            {
                "schema_version": 1,
                "profile_id": "bad",
                "hardware_mapping": {},
                "sensor_timing": {},
                "simulation_physics": {"main_drive": {"damping": float("nan")}},
            },
        ),
    ],
)
def test_contracts_reject_invalid_versions_and_non_finite_values(factory, payload) -> None:
    with pytest.raises(ContractError):
        factory(payload)


def test_reviewed_scenario_catalog_is_complete_and_valid() -> None:
    expected = {
        "audit",
        "main-step",
        "main-coast",
        "suspended-main-0-step-coast",
        "suspended-main-1-step-coast",
        "suspended-main-2-step-coast",
        "suspended-main-3-step-coast",
        "suspended-main-4-step-coast",
        "suspended-main-5-step-coast",
        "manual-load",
        "manual-load-holdout",
        "mass-com",
        "mass-com-holdout",
        "abad-static",
        "abad-static-holdout",
        "spring",
        "spring-holdout",
        "friction",
        "contact-static-settle",
    }

    summaries = list_scenarios()

    assert {item["scenario_id"] for item in summaries} == expected
    for item in summaries:
        scenario = load_scenario(item["scenario_id"])
        assert scenario.required_channels
        assert set(scenario.required_channels) == set(scenario.time_bases)
        assert scenario.split in {"calibration", "holdout"}
        assert scenario.command_segments

    for main_index in range(6):
        scenario = load_scenario(f"suspended-main-{main_index}-step-coast")
        expected_split = "holdout" if main_index == 5 else "calibration"
        assert scenario.joint == f"main_{main_index}"
        assert scenario.split == expected_split


def test_profile_rejects_unknown_or_physically_invalid_fields() -> None:
    base = {
        "schema_version": 1,
        "profile_id": "bad",
        "hardware_mapping": {},
        "sensor_timing": {},
        "simulation_physics": {},
    }
    with pytest.raises(ContractError, match="unknown"):
        CalibrationProfileV1.from_dict({**base, "mystery": {}})
    with pytest.raises(ContractError, match="non-negative"):
        CalibrationProfileV1.from_dict(
            {**base, "simulation_physics": {"main_drive": {"damping": -0.1}}}
        )


def test_manifest_requires_explicit_units_frames_and_provenance_metadata() -> None:
    payload = {
        "schema_version": 1,
        "scenario_id": "main-step",
        "source": "real",
        "trace_file": "trace.npz",
        "channels": ["time_s", "position"],
        "time_bases": {"position": "time_s"},
        "sample_counts": {"position": 2},
        "provenance": {"trace_sha256": "0" * 64},
        "metadata": {},
    }

    with pytest.raises(ContractError, match="metadata"):
        TraceManifestV1.from_dict(payload)


def test_profile_models_hardware_timing_friction_and_passive_spring() -> None:
    profile = CalibrationProfileV1.from_dict(
        {
            "schema_version": 1,
            "profile_id": "measured",
            "hardware_mapping": {
                "encoder_counts_per_rev": {"main_0": 54984.83},
                "encoder_zero_count": {"main_0": 32.5},
                "encoder_sign": {"main_0": -1},
                "pwm_scale": {"main_0": 0.8},
                "pwm_cap": {"main_0": 0.6},
            },
            "sensor_timing": {
                "measured_state_rate_hz": 200.0,
                "velocity_filter_alpha": 0.2,
                "aggregate_command_delay_s": 0.015,
            },
            "simulation_physics": {
                "joint_dynamic_friction": {"main_0": 0.03},
                "joint_viscous_friction": {"main_0": 0.01},
                "passive_spring": {
                    "damper_0": {"stiffness": 12.0, "rest_position_rad": 0.1}
                },
            },
        }
    )

    assert profile.hardware_mapping["encoder_sign"]["main_0"] == -1
    assert profile.sensor_timing["measured_state_rate_hz"] == 200.0
    assert profile.simulation_physics["passive_spring"]["damper_0"]["stiffness"] == 12.0


def test_profile_models_measured_abad_target_mapping_and_source_hashes() -> None:
    abad_source = {
        "trace_sha256": "a" * 64,
        "metadata_sha256": "b" * 64,
        "scenario_id": "abad-static",
        "scenario_sha256": "c" * 64,
        "source": "real",
        "metric_kind": "abad_static_mapping",
        "frame": "abad_0",
        "repeat_count": 3,
        "dataset_id": "bench-20260713",
        "episode_id": "abad-0-run-1",
    }
    friction_source = {
        "trace_sha256": "d" * 64,
        "metadata_sha256": "e" * 64,
        "scenario_id": "friction",
        "scenario_sha256": "f" * 64,
        "source": "real",
        "metric_kind": "ground_friction",
        "frame": "foot_0/ground",
        "repeat_count": 3,
        "dataset_id": "bench-20260713",
        "episode_id": "friction-run-1",
    }
    profile = CalibrationProfileV1.from_dict(
        {
            "schema_version": 1,
            "profile_id": "abad-measured",
            "hardware_mapping": {
                "abad_target_scale": {"abad_0": 1.1},
                "abad_target_offset_rad": {"abad_0": -0.02},
            },
            "sensor_timing": {},
            "simulation_physics": {},
            "measurement_sources": {
                "abad_target:abad_0": abad_source,
                "ground_friction": friction_source,
            },
        }
    )

    assert profile.hardware_mapping["abad_target_scale"]["abad_0"] == 1.1
    assert profile.hardware_mapping["abad_target_offset_rad"]["abad_0"] == -0.02
    assert profile.measurement_sources == {
        "abad_target:abad_0": abad_source,
        "ground_friction": friction_source,
    }


@pytest.mark.parametrize(
    "payload",
    [
        {
            "hardware_mapping": {"abad_target_scale": {"abad_0": 0.0}},
            "measurement_sources": {},
        },
        {
            "hardware_mapping": {},
            "measurement_sources": {"abad_target:abad_0": "a" * 64},
        },
    ],
    ids=("nonpositive-abad-scale", "legacy-abad-source-without-semantics"),
)
def test_profile_rejects_invalid_abad_mapping_or_measurement_source(payload) -> None:
    with pytest.raises(ContractError):
        CalibrationProfileV1.from_dict(
            {
                "schema_version": 1,
                "profile_id": "bad-measurement",
                "hardware_mapping": payload["hardware_mapping"],
                "sensor_timing": {},
                "simulation_physics": {},
                "measurement_sources": payload["measurement_sources"],
            }
        )


def test_profile_rejects_measurement_source_with_wrong_scenario_semantics() -> None:
    source = {
        "trace_sha256": "a" * 64,
        "metadata_sha256": "b" * 64,
        "scenario_id": "friction",
        "scenario_sha256": "c" * 64,
        "source": "real",
        "metric_kind": "abad_static_mapping",
        "frame": "abad_0",
        "repeat_count": 3,
        "dataset_id": "bench-20260713",
        "episode_id": "wrong-scenario",
    }

    with pytest.raises(ContractError, match="scenario_id"):
        CalibrationProfileV1.from_dict(
            {
                "schema_version": 1,
                "profile_id": "wrong-source-semantics",
                "hardware_mapping": {},
                "sensor_timing": {},
                "simulation_physics": {},
                "measurement_sources": {"abad_target:abad_0": source},
            }
        )


@pytest.mark.parametrize("pwm_cap", [1e-12, 0.5, 4.1667, 50.0])
def test_profile_accepts_canonical_velocity_cap_rad_s(pwm_cap: float) -> None:
    profile = CalibrationProfileV1.from_dict(
        {
            "schema_version": 1,
            "profile_id": "valid-cap",
            "hardware_mapping": {"pwm_cap": {"main_0": pwm_cap}},
            "sensor_timing": {},
            "simulation_physics": {},
        }
    )

    assert profile.hardware_mapping["pwm_cap"]["main_0"] == pwm_cap


@pytest.mark.parametrize("pwm_cap", [0.0, -0.1, 50.000001, 500.0])
def test_profile_rejects_canonical_velocity_cap_outside_physical_range(
    pwm_cap: float,
) -> None:
    with pytest.raises(ContractError, match="pwm_cap"):
        CalibrationProfileV1.from_dict(
            {
                "schema_version": 1,
                "profile_id": "invalid-cap",
                "hardware_mapping": {"pwm_cap": {"main_0": pwm_cap}},
                "sensor_timing": {},
                "simulation_physics": {},
            }
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("abad_target_scale", 0.49),
        ("abad_target_scale", 1.51),
        ("abad_target_offset_rad", -0.351),
        ("abad_target_offset_rad", 0.351),
    ],
)
def test_profile_rejects_abad_corrections_outside_conservative_range(
    field: str,
    value: float,
) -> None:
    with pytest.raises(ContractError, match=field):
        CalibrationProfileV1.from_dict(
            {
                "schema_version": 1,
                "profile_id": "unsafe-abad-correction",
                "hardware_mapping": {field: {"abad_0": value}},
                "sensor_timing": {},
                "simulation_physics": {},
            }
        )


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("hardware_mapping", "encoder_counts_per_rev", {"main_0": 0}),
        ("sensor_timing", "velocity_filter_alpha", 1.1),
        ("simulation_physics", "joint_viscous_friction", {"main_0": -0.1}),
    ],
)
def test_profile_rejects_impossible_explicit_calibration_values(section, field, value) -> None:
    payload = {
        "schema_version": 1,
        "profile_id": "bad",
        "hardware_mapping": {},
        "sensor_timing": {},
        "simulation_physics": {},
    }
    payload[section] = {field: value}

    with pytest.raises(ContractError):
        CalibrationProfileV1.from_dict(payload)


def test_scenario_loader_rejects_an_unreviewed_external_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version": 9}), encoding="utf-8")

    with pytest.raises(ContractError):
        load_scenario(path)


def test_reviewed_scenarios_use_safe_commands_and_observable_channels() -> None:
    main = load_scenario("main-step")
    assert [segment["value"] for segment in main.command_segments] == [
        0.0,
        0.25,
        0.0,
        -0.25,
        0.0,
    ]
    assert main.repeats == 3

    assert set(load_scenario("abad-static").required_channels) == {
        "command",
        "position",
        "repeat_index",
        "settled",
    }
    assert set(load_scenario("mass-com").required_channels) == {
        "scale_mass",
        "support_force",
        "support_position",
        "repeat_index",
    }
    assert set(load_scenario("manual-load").required_channels) == {
        "load_force",
        "lever_arm",
        "command",
        "direction",
        "repeat_index",
    }
    assert set(load_scenario("friction").required_channels) == {
        "breakaway_force",
        "static_normal_load",
        "static_repeat_index",
        "dynamic_pull_force",
        "dynamic_normal_load",
        "dynamic_speed",
        "dynamic_repeat_index",
    }
    assert load_scenario("friction").scene_mode == "manual"
    assert set(load_scenario("spring").required_channels) == {
        "load_force",
        "lever_arm",
        "angle",
        "repeat_index",
    }
    assert load_scenario("mass-com").split == "calibration"
    assert load_scenario("mass-com-holdout").split == "holdout"
    assert load_scenario("spring").split == "calibration"
    assert load_scenario("spring-holdout").split == "holdout"
    assert load_scenario("manual-load").split == "calibration"
    assert load_scenario("manual-load-holdout").split == "holdout"
