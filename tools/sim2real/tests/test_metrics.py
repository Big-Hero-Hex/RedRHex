from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tools.sim2real.compare import compare_traces
from tools.sim2real.contracts import CalibrationProfileV1, ContractError, ScenarioSpecV1
from tools.sim2real.metrics import (
    abad_static_mapping_metrics,
    bidirectional_coast_metrics,
    bidirectional_step_metrics,
    coast_response_metrics,
    compute_subsystem_metrics,
    friction_metrics,
    mass_com_metrics,
    position_derived_velocity,
    stiffness_metrics,
    static_settle_metrics,
    torsional_spring_metrics,
    torsional_spring_holdout_metrics,
    torsional_spring_quality_gates,
    step_response_metrics,
    torque_saturation_metrics,
    variation_metrics,
)
from tools.sim2real.scenarios import load_scenario
from tools.sim2real.traces import load_trace, write_trace


def test_position_derived_velocity_uses_nonuniform_sample_times() -> None:
    time_s = np.array([0.0, 0.1, 0.4, 1.0])
    position = 3.0 * time_s + 2.0

    velocity = position_derived_velocity(time_s, position)

    np.testing.assert_allclose(velocity, 3.0, atol=1e-12)


def test_step_and_coast_metrics_cover_drive_response() -> None:
    time_s = np.arange(0.0, 3.01, 0.05)
    command = np.where((time_s >= 0.5) & (time_s < 2.0), 0.25, 0.0)
    velocity = np.zeros_like(time_s)
    ramp = (time_s >= 0.7) & (time_s < 1.1)
    velocity[ramp] = (time_s[ramp] - 0.7) / 0.4 * 2.0
    velocity[(time_s >= 1.1) & (time_s < 2.0)] = 2.0
    velocity[np.argmin(abs(time_s - 1.2))] = 2.4
    coast = (time_s >= 2.0) & (time_s < 2.5)
    velocity[coast] = 2.0 * (2.5 - time_s[coast]) / 0.5
    position = np.cumsum(velocity) * 0.05

    step = step_response_metrics(time_s, command, time_s, position)
    coast_result = coast_response_metrics(time_s, command, time_s, position)

    assert step["onset_delay_s"] == pytest.approx(0.2, abs=0.08)
    assert step["steady_speed_rad_s"] == pytest.approx(2.0, rel=0.12)
    assert step["rise_time_s"] == pytest.approx(0.32, abs=0.12)
    assert step["overshoot_ratio"] == pytest.approx(0.2, abs=0.12)
    assert coast_result["coast_time_s"] == pytest.approx(0.45, abs=0.1)


def test_static_measurement_metrics() -> None:
    stiffness = stiffness_metrics(
        force=np.array([0.0, 10.0, 20.0]),
        displacement=np.array([0.0, 0.01, 0.02]),
    )
    torque = torque_saturation_metrics(
        load_force=np.array([20.0, 20.0, 21.0, 21.0, 19.0, 19.0]),
        lever_arm=np.full(6, 0.1),
        command=np.full(6, 0.25),
        direction=np.tile(np.array([1.0, -1.0]), 3),
        saturation_confirmed=np.ones(6),
        repeat_index=np.repeat(np.arange(3), 2),
        expected_repeats=3,
    )
    mass_com = mass_com_metrics(
        scale_mass=np.array([9.9, 10.0, 10.1]),
        support_force=np.array([[40.0, 30.0, 30.0]] * 3),
        support_position=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        repeat_index=np.arange(3),
        expected_repeats=3,
    )

    assert stiffness["stiffness_n_per_m"] == pytest.approx(1000.0)
    assert torque["torque_saturation_nm"] == pytest.approx(2.0)
    assert torque["repeat_count"] == 3
    assert mass_com["mass_kg"] == pytest.approx(10.0)
    assert mass_com["com_m"] == pytest.approx([0.3, 0.3])
    assert mass_com["com_x_m"] == pytest.approx(0.3)
    assert mass_com["com_y_m"] == pytest.approx(0.3)
    assert mass_com["repeat_count"] == 3
    assert mass_com["com_m_std"] == pytest.approx([0.0, 0.0])
    assert mass_com["com_x_m_std"] == pytest.approx(0.0)
    assert mass_com["com_y_m_std"] == pytest.approx(0.0)
    np.testing.assert_allclose(
        [item["com_m"] for item in mass_com["repeats"]], [[0.3, 0.3]] * 3
    )


def test_mass_com_reports_planar_repeat_variation() -> None:
    result = mass_com_metrics(
        scale_mass=np.array([10.0, 10.1, 9.9]),
        support_force=np.array(
            [
                [40.0, 40.0, 20.0],
                [35.0, 45.0, 20.0],
                [45.0, 35.0, 20.0],
            ]
        ),
        support_position=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        repeat_index=np.arange(3),
        expected_repeats=3,
    )

    np.testing.assert_allclose(result["com_m"], [0.4, 0.2])
    np.testing.assert_allclose(
        result["com_m_std"],
        np.std([[0.4, 0.2], [0.45, 0.2], [0.35, 0.2]], axis=0),
    )
    assert result["repeats"][1]["com_m"] == pytest.approx([0.45, 0.2])


def test_mass_com_rejects_collinear_or_insufficient_planar_supports() -> None:
    with pytest.raises(ContractError, match="three non-collinear"):
        mass_com_metrics(
            scale_mass=np.array([10.0, 10.0, 10.0]),
            support_force=np.array([[50.0, 50.0]] * 3),
            support_position=np.array([[0.0, 0.0], [1.0, 0.0]]),
            repeat_index=np.arange(3),
            expected_repeats=3,
        )


def test_known_load_requires_confirmed_saturation_in_both_directions() -> None:
    common = {
        "load_force": np.array([20.0, 20.0]),
        "lever_arm": np.array([0.1, 0.1]),
        "command": np.array([0.25, -0.25]),
    }
    with pytest.raises(ContractError, match="every sample.*confirm"):
        torque_saturation_metrics(
            **common,
            direction=np.array([1.0, -1.0]),
            saturation_confirmed=np.array([1.0, 0.0]),
        )
    with pytest.raises(ContractError, match="positive and negative"):
        torque_saturation_metrics(
            **common,
            direction=np.ones(2),
            saturation_confirmed=np.ones(2),
        )


def test_abad_static_metrics_fit_only_settled_samples_and_report_repeat_variation() -> None:
    command = np.tile(np.array([-0.2, 0.0, 0.2, 0.1]), 3)
    repeat_index = np.repeat(np.arange(3), 4)
    settled = np.tile(np.array([1.0, 1.0, 1.0, 0.0]), 3)
    repeat_scales = np.repeat(np.array([1.2, 1.25, 1.3]), 4)
    measured = repeat_scales * command - 0.03
    measured[settled == 0.0] = 99.0

    result = abad_static_mapping_metrics(
        command,
        measured,
        repeat_index=repeat_index,
        settled=settled,
        expected_repeats=3,
        frame="abad_0",
    )

    assert result["schema_version"] == 1
    assert result["metric_kind"] == "abad_static_mapping"
    assert result["repeat_count"] == 3
    assert result["frame"] == "abad_0"
    assert result["units"] == {
        "target_scale": "1",
        "target_offset_rad": "rad",
        "fit_rmse_rad": "rad",
    }
    assert result["aggregate"]["target_scale"] == pytest.approx(1.25)
    assert result["aggregate"]["target_offset_rad"] == pytest.approx(-0.03)
    assert result["aggregate"]["pose_count"] == 3
    assert result["repeat_variation"]["target_scale_mean"] == pytest.approx(1.25)
    assert result["repeat_variation"]["target_scale_std"] == pytest.approx(
        np.std([1.2, 1.25, 1.3])
    )
    assert result["repeat_variation"]["target_scale_count"] == 3
    assert [item["repeat_index"] for item in result["repeats"]] == [0, 1, 2]


def test_abad_static_metrics_reject_unsettled_or_two_pose_repeats() -> None:
    with pytest.raises(ContractError, match="settled"):
        abad_static_mapping_metrics(
            [-0.2, 0.0, 0.2] * 3,
            [-0.2, 0.0, 0.2] * 3,
            repeat_index=np.repeat(np.arange(3), 3),
            settled=np.zeros(9),
            expected_repeats=3,
            frame="abad_0",
        )

    with pytest.raises(ContractError, match="three distinct"):
        abad_static_mapping_metrics(
            [-0.2, -0.2, 0.2] * 3,
            [-0.2, -0.2, 0.2] * 3,
            repeat_index=np.repeat(np.arange(3), 3),
            settled=np.ones(9),
            expected_repeats=3,
            frame="abad_0",
        )


def _write_drive_trace(
    directory: Path,
    *,
    position_scale: float,
    source: str,
    source_path: Path | None = None,
    position_unit: str = "rad",
    scenario: ScenarioSpecV1 | None = None,
    calibration_source: str | None = None,
):
    scenario = scenario or load_scenario("main-step")
    time_s = np.arange(0.0, 3.01, 0.05)
    command = np.where((time_s >= 0.5) & (time_s < 2.0), 0.25, 0.0)
    velocity = np.where((time_s >= 0.7) & (time_s < 2.0), 2.0, 0.0)
    position = np.cumsum(velocity) * 0.05 * position_scale
    profile = None
    constants = (
        {} if calibration_source is None else {"calibration_source": calibration_source}
    )
    if source == "real" and calibration_source is None:
        profile = CalibrationProfileV1.from_dict(
            {
                "schema_version": 1,
                "profile_id": "measured-main-0",
                "hardware_mapping": {
                    "encoder_counts_per_rev": {"main_0": 54984.83},
                    "encoder_zero_count": {"main_0": 0.0},
                    "encoder_sign": {"main_0": 1.0},
                    "joint_direction": {"main_0": 1.0},
                    "pwm_scale": {"main_0": 0.002},
                    "pwm_cap": {"main_0": 1.0},
                },
                "sensor_timing": {},
                "simulation_physics": {},
            }
        )
        source_name = f"profile:{profile.profile_id}"
        constants = {
            "position_mapping_source": source_name,
            "requested_command_source": source_name,
        }
    metadata = {
        "units": {"command": "normalized", "position": position_unit},
        "frames": {"command": "actuator", "position": "main_0"},
        "joint_order": ["main_0"],
        "clock": {
            "source": "test",
            "timestamp_semantics": "relative_monotonic",
            "time_unit": "s",
        },
        "git_sha": None,
        "asset_sha256": None,
        "config_sha256": None,
        "calibration_constants": constants,
    }
    if source == "real":
        assert source_path is not None
        source_path.write_bytes(b"real trace fixture")
    write_trace(
        directory,
        {
            "command_time_s": time_s,
            "command": command,
            "position_time_s": time_s,
            "position": position,
        },
        scenario=scenario,
        source=source,
        source_path=source_path,
        profile=profile,
        metadata=metadata,
    )
    return load_trace(directory, scenario=scenario)


def test_comparison_keeps_subsystems_separate_without_a_global_score(tmp_path: Path) -> None:
    real = _write_drive_trace(
        tmp_path / "real",
        position_scale=1.0,
        source="real",
        source_path=tmp_path / "real-source.npz",
    )
    sim = _write_drive_trace(
        tmp_path / "sim", position_scale=0.8, source="sim"
    )

    result = compare_traces(real, sim, scenario=load_scenario("main-step"))

    assert set(result) == {
        "schema_version",
        "scenario_id",
        "delta_convention",
        "subsystems",
    }
    assert result["delta_convention"] == "sim_minus_real"
    assert set(result["subsystems"]) == {"main_drive"}
    assert set(result["subsystems"]["main_drive"]) == {"real", "sim", "delta"}
    assert result["subsystems"]["main_drive"]["delta"]["positive"][
        "steady_speed_rad_s"
    ] < 0.0
    assert "score" not in str(result).lower()


@pytest.mark.parametrize("input_kind", ["path", "loaded"])
def test_comparison_rejects_selected_scenario_id_mismatch(
    tmp_path: Path, input_kind: str
) -> None:
    recorded_scenario = load_scenario("main-coast")
    real_loaded = _write_drive_trace(
        tmp_path / "real",
        position_scale=1.0,
        source="real",
        source_path=tmp_path / "real-source.npz",
        scenario=recorded_scenario,
    )
    sim_loaded = _write_drive_trace(
        tmp_path / "sim",
        position_scale=0.8,
        source="sim",
        scenario=recorded_scenario,
    )
    real = real_loaded.directory if input_kind == "path" else real_loaded
    sim = sim_loaded.directory if input_kind == "path" else sim_loaded

    with pytest.raises(ContractError, match="scenario id mismatch"):
        compare_traces(real, sim, scenario=load_scenario("main-step"))


@pytest.mark.parametrize("input_kind", ["path", "loaded"])
def test_comparison_rejects_selected_scenario_hash_mismatch(
    tmp_path: Path, input_kind: str
) -> None:
    canonical = load_scenario("main-step")
    modified_payload = canonical.to_dict()
    modified_payload["description"] = "Locally modified scenario contract."
    recorded_scenario = ScenarioSpecV1.from_dict(modified_payload)
    real_loaded = _write_drive_trace(
        tmp_path / "real",
        position_scale=1.0,
        source="real",
        source_path=tmp_path / "real-source.npz",
        scenario=recorded_scenario,
    )
    sim_loaded = _write_drive_trace(
        tmp_path / "sim",
        position_scale=0.8,
        source="sim",
        scenario=recorded_scenario,
    )
    real = real_loaded.directory if input_kind == "path" else real_loaded
    sim = sim_loaded.directory if input_kind == "path" else sim_loaded

    with pytest.raises(ContractError, match="scenario hash mismatch"):
        compare_traces(real, sim, scenario=canonical)


def test_comparison_rejects_unit_or_frame_mismatch(tmp_path: Path) -> None:
    real = _write_drive_trace(
        tmp_path / "real",
        position_scale=1.0,
        source="real",
        source_path=tmp_path / "real-source.npz",
    )
    sim = _write_drive_trace(
        tmp_path / "sim",
        position_scale=1.0,
        source="sim",
        position_unit="degree",
    )

    with pytest.raises(ContractError, match="unit mismatch"):
        compare_traces(real, sim, scenario=load_scenario("main-step"))

    with pytest.raises(ContractError, match="expected unit"):
        load_trace(tmp_path / "sim", expected_units={"position": "rad"})


@pytest.mark.parametrize(
    "calibration_source",
    [
        "provisional_repository_defaults",
        "profile:partial:with_provisional_fallbacks",
    ],
)
def test_comparison_rejects_provisional_real_hardware_mapping(
    tmp_path: Path, calibration_source: str
) -> None:
    real = _write_drive_trace(
        tmp_path / "real",
        position_scale=1.0,
        source="real",
        source_path=tmp_path / "real-source.npz",
        calibration_source=calibration_source,
    )
    sim = _write_drive_trace(
        tmp_path / "sim", position_scale=1.0, source="sim"
    )

    with pytest.raises(ContractError, match="provisional.*hardware mapping"):
        compare_traces(real, sim, scenario=load_scenario("main-step"))


@pytest.mark.parametrize("input_kind", ["path", "loaded"])
def test_comparison_rejects_swapped_real_and_sim_sources(
    tmp_path: Path, input_kind: str
) -> None:
    real_loaded = _write_drive_trace(
        tmp_path / "real-position", position_scale=1.0, source="sim"
    )
    sim_loaded = _write_drive_trace(
        tmp_path / "sim-position",
        position_scale=0.8,
        source="real",
        source_path=tmp_path / "sim-position-source.npz",
    )
    real = real_loaded.directory if input_kind == "path" else real_loaded
    sim = sim_loaded.directory if input_kind == "path" else sim_loaded

    with pytest.raises(ContractError, match='real trace must have source "real"'):
        compare_traces(real, sim, scenario=load_scenario("main-step"))


@pytest.mark.parametrize("input_kind", ["path", "loaded"])
@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("sim", 'real trace must have source "real"'),
        ("real", 'sim trace must have source "sim"'),
    ],
)
def test_comparison_rejects_same_source_inputs(
    tmp_path: Path, input_kind: str, source: str, message: str
) -> None:
    left = _write_drive_trace(
        tmp_path / "left",
        position_scale=1.0,
        source=source,
        source_path=tmp_path / "left-source.npz" if source == "real" else None,
    )
    right = _write_drive_trace(
        tmp_path / "right",
        position_scale=0.8,
        source=source,
        source_path=tmp_path / "right-source.npz" if source == "real" else None,
    )
    real = left.directory if input_kind == "path" else left
    sim = right.directory if input_kind == "path" else right

    with pytest.raises(ContractError, match=message):
        compare_traces(real, sim, scenario=load_scenario("main-step"))


def test_bidirectional_drive_metrics_do_not_drop_reverse_segments() -> None:
    time_s = np.arange(0.0, 5.01, 0.05)
    command = np.zeros_like(time_s)
    command[(time_s >= 0.5) & (time_s < 1.75)] = 0.25
    command[(time_s >= 3.0) & (time_s < 4.25)] = -0.25
    velocity = np.zeros_like(time_s)
    velocity[(time_s >= 0.7) & (time_s < 1.75)] = 2.0
    positive_coast = (time_s >= 1.75) & (time_s < 2.25)
    velocity[positive_coast] = 2.0 * (2.25 - time_s[positive_coast]) / 0.5
    velocity[(time_s >= 3.2) & (time_s < 4.25)] = -1.5
    negative_coast = (time_s >= 4.25) & (time_s < 4.75)
    velocity[negative_coast] = -1.5 * (4.75 - time_s[negative_coast]) / 0.5
    position = np.cumsum(velocity) * 0.05

    steps = bidirectional_step_metrics(time_s, command, time_s, position)
    coasts = bidirectional_coast_metrics(time_s, command, time_s, position)

    assert set(steps) == {"positive", "negative"}
    assert set(coasts) == {"positive", "negative"}
    assert steps["positive"]["steady_speed_rad_s"] > steps["negative"][
        "steady_speed_rad_s"
    ]


def test_combined_step_coast_scenario_reports_metric_families_separately(
    tmp_path: Path,
) -> None:
    payload = load_scenario("main-coast").to_dict()
    payload["scenario_id"] = "combined-response"
    payload["experiment_kind"] = "step_coast"
    scenario = ScenarioSpecV1.from_dict(payload)
    time_s = np.arange(0.0, 5.01, 0.05)
    command = np.zeros_like(time_s)
    command[(time_s >= 0.5) & (time_s < 1.75)] = 0.25
    command[(time_s >= 3.0) & (time_s < 4.25)] = -0.25
    velocity = np.zeros_like(time_s)
    velocity[(time_s >= 0.7) & (time_s < 1.75)] = 2.0
    positive_coast = (time_s >= 1.75) & (time_s < 2.25)
    velocity[positive_coast] = 2.0 * (2.25 - time_s[positive_coast]) / 0.5
    velocity[(time_s >= 3.2) & (time_s < 4.25)] = -1.5
    negative_coast = (time_s >= 4.25) & (time_s < 4.75)
    velocity[negative_coast] = -1.5 * (4.75 - time_s[negative_coast]) / 0.5
    position = np.cumsum(velocity) * 0.05
    write_trace(
        tmp_path / "combined",
        {
            "command_time_s": time_s,
            "command": command,
            "position_time_s": time_s,
            "position": position,
        },
        scenario=scenario,
        source="sim",
    )

    metrics = compute_subsystem_metrics(
        scenario, load_trace(tmp_path / "combined", scenario=scenario)
    )

    assert set(metrics) == {"step", "coast"}
    assert set(metrics["step"]) == {"positive", "negative"}
    assert set(metrics["coast"]) == {"positive", "negative"}


def test_torsional_spring_and_variation_metrics() -> None:
    angles = np.tile(np.array([0.0, 0.1, 0.2]), 3)
    spring = torsional_spring_metrics(
        angle_rad=angles,
        load_force=np.tile(np.array([0.0, 10.0, 20.0]), 3),
        lever_arm_m=np.full(angles.size, 0.1),
        repeat_index=np.repeat(np.arange(3), 3),
        expected_repeats=3,
    )
    variation = variation_metrics(
        np.array([1.0, 2.0, 3.0]), metric_name="steady_speed_rad_s"
    )

    assert spring["stiffness_nm_per_rad"] == pytest.approx(10.0)
    assert spring["repeat_count"] == 3
    assert variation == {
        "steady_speed_rad_s_mean": pytest.approx(2.0),
        "steady_speed_rad_s_std": pytest.approx(np.std([1.0, 2.0, 3.0])),
        "steady_speed_rad_s_count": 3,
    }


def test_signed_torsional_spring_metrics_report_linearity_repeatability_and_hysteresis() -> None:
    angles: list[float] = []
    torque: list[float] = []
    branches: list[float] = []
    repeats: list[int] = []
    for repeat_index, stiffness in enumerate((9.8, 10.0, 10.2)):
        for branch, offset in ((1.0, 0.05), (-1.0, -0.05)):
            for angle in (-0.4, -0.2, 0.2, 0.4):
                angles.append(angle)
                torque.append(stiffness * angle + offset)
                branches.append(branch)
                repeats.append(repeat_index)
    torque_array = np.asarray(torque)

    result = torsional_spring_metrics(
        angle_rad=np.asarray(angles),
        load_force=np.abs(torque_array) / 0.1,
        lever_arm_m=np.full(len(angles), 0.1),
        torque_direction=np.sign(torque_array),
        sweep_branch=np.asarray(branches),
        repeat_index=np.asarray(repeats),
        expected_repeats=3,
        rest_position_rad=0.0,
    )

    assert result["stiffness_nm_per_rad"] == pytest.approx(10.0)
    assert result["r_squared"] > 0.998
    assert result["stiffness_cv"] == pytest.approx(
        np.std([9.8, 10.0, 10.2]) / 10.0
    )
    assert result["hysteresis_width_nm"] == pytest.approx(0.1)
    assert result["hysteresis_full_scale_ratio"] < 0.03
    assert result["neutral_stiffness_nm_per_rad"] == pytest.approx(10.0)
    assert result["neutral_fit_rmse_full_scale_ratio"] < 0.02


def test_torsional_spring_holdout_metrics_and_quality_gates_use_calibration_model() -> None:
    calibration = {
        "r_squared": 0.995,
        "stiffness_cv": 0.02,
        "hysteresis_full_scale_ratio": 0.04,
        "stiffness_nm_per_rad": 10.0,
        "torque_intercept_nm": 0.0,
        "neutral_stiffness_nm_per_rad": 10.0,
    }
    angle = np.tile(np.array([-0.3, 0.3]), 3)
    torque = angle * 10.0
    holdout = torsional_spring_holdout_metrics(
        angle_rad=angle,
        load_force=np.abs(torque) / 0.1,
        lever_arm_m=np.full(angle.size, 0.1),
        torque_direction=np.sign(torque),
        calibration_metrics=calibration,
        rest_position_rad=0.0,
    )
    quality = torsional_spring_quality_gates(calibration, holdout)

    assert holdout["torque_rmse_nm"] == pytest.approx(0.0)
    assert holdout["rmse_full_scale_ratio"] == pytest.approx(0.0)
    assert holdout["neutral_model_rmse_full_scale_ratio"] == pytest.approx(0.0)
    assert quality["accepted"] is True
    assert set(quality["gates"]) == {
        "r_squared",
        "heldout_rmse",
        "stiffness_cv",
        "hysteresis",
        "neutral_model_heldout_rmse",
    }


def test_compute_metrics_uses_managed_signed_torsion_spring_annotations(
    tmp_path: Path,
) -> None:
    scenario = load_scenario("torsion-spring")
    angle = np.tile(np.array([-0.4, -0.2, 0.2, 0.4, 0.4, 0.2, -0.2, -0.4]), 3)
    time_s = np.arange(angle.size, dtype=float) * 0.1
    torque = angle * 10.0
    write_trace(
        tmp_path / "torsion-spring",
        {
            "angle_time_s": time_s,
            "angle": angle,
            "load_force_time_s": time_s,
            "load_force": np.abs(torque) / 0.1,
            "lever_arm_time_s": time_s,
            "lever_arm": np.full(angle.size, 0.1),
            "torque_direction": np.sign(torque),
            "sweep_branch": np.tile(np.repeat([1.0, -1.0], 4), 3),
            "repeat_index": np.repeat(np.arange(3), 8),
        },
        scenario=scenario,
        source="sim",
        metadata={"calibration_constants": {"rest_position_rad": 0.0}},
    )

    result = compute_subsystem_metrics(
        scenario, load_trace(tmp_path / "torsion-spring", scenario=scenario)
    )

    assert result["r_squared"] == pytest.approx(1.0)
    assert result["hysteresis_full_scale_ratio"] == pytest.approx(0.0)
    assert result["neutral_stiffness_nm_per_rad"] == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("torque_direction", "sweep_branch", "message"),
    [
        ([1.0, 0.0], [1.0, -1.0], "torque_direction"),
        ([1.0, -1.0], [1.0, 2.0], "sweep_branch"),
    ],
)
def test_torsional_spring_annotations_require_signed_unit_encodings(
    torque_direction: list[float], sweep_branch: list[float], message: str
) -> None:
    with pytest.raises(ContractError, match=message):
        torsional_spring_metrics(
            angle_rad=[-0.1, 0.1],
            load_force=[1.0, 1.0],
            lever_arm_m=[0.1, 0.1],
            torque_direction=torque_direction,
            sweep_branch=sweep_branch,
        )


def test_torsional_spring_fit_rejects_non_restoring_torque_direction() -> None:
    with pytest.raises(ContractError, match="positive restoring stiffness"):
        torsional_spring_metrics(
            angle_rad=[-0.2, -0.1, 0.1, 0.2],
            load_force=[2.0, 1.0, 1.0, 2.0],
            lever_arm_m=[0.1, 0.1, 0.1, 0.1],
            torque_direction=[1.0, 1.0, -1.0, -1.0],
        )


def test_legacy_unsigned_spring_fit_preserves_its_historical_sign_convention() -> None:
    result = torsional_spring_metrics(
        angle_rad=[-0.1, 0.1],
        load_force=[1.0, -1.0],
        lever_arm_m=[0.1, 0.1],
    )

    assert result["stiffness_nm_per_rad"] == pytest.approx(-1.0)


def test_legacy_unsigned_spring_fit_still_accepts_a_flat_trace() -> None:
    result = torsional_spring_metrics(
        angle_rad=[-0.1, 0.1],
        load_force=[0.0, 0.0],
        lever_arm_m=[0.1, 0.1],
    )

    assert result == {
        "stiffness_nm_per_rad": pytest.approx(0.0),
        "torque_intercept_nm": pytest.approx(0.0),
    }


def test_friction_metrics_report_breakaway_and_constant_speed_repeat_variation() -> None:
    dynamic_repeat = np.repeat(np.arange(3), 3)
    result = friction_metrics(
        breakaway_force=np.array([20.0, 22.0, 18.0]),
        static_normal_load=np.full(3, 100.0),
        static_repeat_index=np.arange(3),
        dynamic_pull_force=np.repeat(np.array([15.0, 16.0, 14.0]), 3),
        dynamic_normal_load=np.full(9, 100.0),
        dynamic_speed=np.tile(np.array([0.049, 0.05, 0.051]), 3),
        dynamic_repeat_index=dynamic_repeat,
        expected_repeats=3,
        frame="foot_0/ground",
        max_dynamic_speed_m_s=0.1,
    )

    assert result["schema_version"] == 1
    assert result["metric_kind"] == "ground_friction"
    assert result["frame"] == "foot_0/ground"
    assert result["units"] == {
        "coefficient": "1",
        "force": "N",
        "speed": "m/s",
    }
    assert result["static"]["coefficient_mean"] == pytest.approx(0.2)
    assert result["static"]["coefficient_std"] == pytest.approx(np.std([0.2, 0.22, 0.18]))
    assert result["static"]["coefficient_count"] == 3
    assert result["dynamic"]["coefficient_mean"] == pytest.approx(0.15)
    assert result["dynamic"]["coefficient_std"] == pytest.approx(np.std([0.15, 0.16, 0.14]))
    assert result["dynamic"]["coefficient_count"] == 3
    assert result["repeat_count"] == 3
    assert [item["repeat_index"] for item in result["dynamic"]["repeats"]] == [0, 1, 2]


def test_friction_metrics_reject_ambiguous_thresholds_and_nonconstant_speed() -> None:
    common = {
        "static_normal_load": np.full(3, 100.0),
        "static_repeat_index": np.arange(3),
        "dynamic_pull_force": np.full(9, 15.0),
        "dynamic_normal_load": np.full(9, 100.0),
        "dynamic_repeat_index": np.repeat(np.arange(3), 3),
        "expected_repeats": 3,
        "frame": "foot_0/ground",
        "max_dynamic_speed_m_s": 0.1,
    }
    with pytest.raises(ContractError, match="one breakaway threshold"):
        friction_metrics(
            breakaway_force=np.array([10.0, 20.0, 22.0, 18.0]),
            static_normal_load=np.full(4, 100.0),
            static_repeat_index=np.array([0, 0, 1, 2]),
            dynamic_speed=np.full(9, 0.05),
            **{
                key: value
                for key, value in common.items()
                if key not in {"static_normal_load", "static_repeat_index"}
            },
        )

    with pytest.raises(ContractError, match="constant speed"):
        friction_metrics(
            breakaway_force=np.array([20.0, 22.0, 18.0]),
            dynamic_speed=np.tile(np.array([0.02, 0.05, 0.08]), 3),
            **common,
        )


def test_static_settle_metrics_report_repeat_aware_height_and_compression() -> None:
    repeat_index = np.repeat(np.arange(3), 4)
    settled = np.tile(np.array([0.0, 0.0, 1.0, 1.0]), 3)
    root_height = np.repeat(np.array([0.20, 0.21, 0.19]), 4)
    root_position = np.column_stack(
        (np.zeros(root_height.size), np.zeros(root_height.size), root_height)
    )
    foot_force = np.repeat(np.array([100.0, 105.0, 95.0]), 4)
    contact_force = np.column_stack((foot_force * 0.5, foot_force * 0.5))

    result = static_settle_metrics(
        root_position,
        contact_force,
        repeat_index=repeat_index,
        settled=settled,
        expected_repeats=3,
    )

    assert result["schema_version"] == 1
    assert result["metric_kind"] == "contact_static_settle"
    assert result["repeat_count"] == 3
    assert result["settled"]["root_height_m"] == pytest.approx(0.20)
    assert result["settled"]["root_height_m_std"] == pytest.approx(
        np.std([0.20, 0.21, 0.19])
    )
    assert result["settled"]["contact_force_n"] == pytest.approx(100.0)
    assert result["settled"]["contact_force_n_std"] == pytest.approx(
        np.std([100.0, 105.0, 95.0])
    )
    assert result["settled"]["repeat_count"] == 3


def test_contact_static_settle_scenario_metrics_use_annotations(tmp_path: Path) -> None:
    scenario = load_scenario("contact-static-settle")
    time_s = np.arange(12, dtype=float)
    repeat_index = np.repeat(np.arange(3), 4)
    settled = np.tile(np.array([0.0, 0.0, 1.0, 1.0]), 3)
    height = np.repeat(np.array([0.20, 0.21, 0.19]), 4)
    write_trace(
        tmp_path / "settle",
        {
            "sim_time_s": time_s,
            "root_position": np.column_stack((np.zeros(12), np.zeros(12), height)),
            "contact_force_n": np.column_stack((np.full(12, 50.0), np.full(12, 50.0))),
            "repeat_index": repeat_index,
            "settled": settled,
        },
        scenario=scenario,
        source="sim",
    )

    result = compute_subsystem_metrics(
        scenario, load_trace(tmp_path / "settle", scenario=scenario)
    )

    assert result["settled"]["root_height_m"] == pytest.approx(0.20)
    assert result["settled"]["repeat_count"] == 3


def test_scenario_metrics_include_abad_mapping_and_dynamic_friction(
    tmp_path: Path,
) -> None:
    abad = load_scenario("abad-static")
    command = np.tile(np.array([-0.15, 0.0, 0.15]), 3)
    sample_time = np.arange(command.size, dtype=float)
    write_trace(
        tmp_path / "abad",
        {
            "command_time_s": sample_time,
            "command": command,
            "position_time_s": sample_time,
            "position": 0.8 * command + 0.02,
            "repeat_index": np.repeat(np.arange(3), 3),
            "settled": np.ones(command.size),
        },
        scenario=abad,
        source="sim",
    )
    friction = load_scenario("friction")
    write_trace(
        tmp_path / "friction",
        {
            "static_time_s": np.array([0.0, 1.0, 2.0]),
            "breakaway_force": np.array([20.0, 21.0, 19.0]),
            "static_normal_load": np.array([100.0, 100.0, 100.0]),
            "static_repeat_index": np.arange(3),
            "dynamic_time_s": np.arange(9, dtype=float),
            "dynamic_pull_force": np.repeat(np.array([15.0, 16.0, 14.0]), 3),
            "dynamic_normal_load": np.full(9, 100.0),
            "dynamic_speed": np.full(9, 0.05),
            "dynamic_repeat_index": np.repeat(np.arange(3), 3),
        },
        scenario=friction,
        source="sim",
    )

    abad_result = compute_subsystem_metrics(abad, load_trace(tmp_path / "abad"))
    friction_result = compute_subsystem_metrics(
        friction, load_trace(tmp_path / "friction")
    )

    assert abad_result["aggregate"]["target_scale"] == pytest.approx(0.8)
    assert abad_result["aggregate"]["target_offset_rad"] == pytest.approx(0.02)
    assert abad_result["repeat_variation"]["target_scale_count"] == 3
    assert friction_result["static"]["coefficient_mean"] == pytest.approx(0.2)
    assert friction_result["dynamic"]["coefficient_mean"] == pytest.approx(0.15)


@pytest.mark.parametrize(
    ("load_force_time_s", "other_time_s"),
    [
        (np.array([10.0, 11.0]), np.array([0.0, 1.0])),
        (np.array([0.0, 1.0]), np.array([0.5, 1.5])),
    ],
    ids=("disjoint", "partial-overlap"),
)
def test_metrics_reject_interpolation_without_full_clock_coverage(
    tmp_path: Path,
    load_force_time_s: np.ndarray,
    other_time_s: np.ndarray,
) -> None:
    scenario = load_scenario("manual-load")
    episode = tmp_path / "episode"
    write_trace(
        episode,
        {
            "load_force_time_s": load_force_time_s,
            "load_force": np.array([10.0, 20.0]),
            "lever_arm_time_s": other_time_s,
            "lever_arm": np.array([0.1, 0.2]),
            "command_time_s": other_time_s,
            "command": np.array([0.1, 0.2]),
            "direction_time_s": other_time_s,
            "direction": np.array([1.0, 1.0]),
            "saturation_confirmed": np.ones(2),
            "repeat_index": np.array([0.0, 1.0]),
        },
        scenario=scenario,
        source="sim",
    )
    trace = load_trace(episode, scenario=scenario)

    with pytest.raises(ContractError, match="lever_arm.*load_force.*clock"):
        compute_subsystem_metrics(scenario, trace)
