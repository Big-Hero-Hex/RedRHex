from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tools.sim2real.compare import compare_traces
from tools.sim2real.contracts import ContractError, ScenarioSpecV1
from tools.sim2real.metrics import (
    bidirectional_coast_metrics,
    bidirectional_step_metrics,
    coast_response_metrics,
    compute_subsystem_metrics,
    friction_metrics,
    mass_com_metrics,
    position_derived_velocity,
    stiffness_metrics,
    torsional_spring_metrics,
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
        load_force=np.array([10.0, 20.0, 30.0]),
        lever_arm=np.array([0.1, 0.1, 0.1]),
        command=np.array([0.1, 0.2, 0.25]),
        direction=np.array([1.0, 1.0, -1.0]),
    )
    mass_com = mass_com_metrics(
        scale_mass=np.array([9.9, 10.0, 10.1]),
        support_force=np.array([[60.0, 40.0], [60.0, 40.0]]),
        support_position=np.array([0.0, 1.0]),
    )
    friction = friction_metrics(
        pull_force=np.array([19.0, 20.0, 21.0]),
        normal_load=np.array([100.0, 100.0, 100.0]),
    )

    assert stiffness["stiffness_n_per_m"] == pytest.approx(1000.0)
    assert torque["torque_saturation_nm"] == pytest.approx(3.0)
    assert mass_com == {"mass_kg": pytest.approx(10.0), "com_m": pytest.approx(0.4)}
    assert friction["static_friction_coefficient"] == pytest.approx(0.2)
    assert friction_metrics(incline_angle_rad=np.array([np.arctan(0.5)]))[
        "static_friction_coefficient"
    ] == pytest.approx(0.5)


def _write_drive_trace(
    directory: Path,
    *,
    position_scale: float,
    source: str,
    source_path: Path | None = None,
    position_unit: str = "rad",
    scenario: ScenarioSpecV1 | None = None,
):
    scenario = scenario or load_scenario("main-step")
    time_s = np.arange(0.0, 3.01, 0.05)
    command = np.where((time_s >= 0.5) & (time_s < 2.0), 0.25, 0.0)
    velocity = np.where((time_s >= 0.7) & (time_s < 2.0), 2.0, 0.0)
    position = np.cumsum(velocity) * 0.05 * position_scale
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
        "calibration_constants": {},
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


def test_torsional_spring_dynamic_friction_and_variation_metrics() -> None:
    spring = torsional_spring_metrics(
        angle_rad=np.array([0.0, 0.1, 0.2]),
        load_force=np.array([0.0, 10.0, 20.0]),
        lever_arm_m=np.array([0.1, 0.1, 0.1]),
    )
    friction = friction_metrics(
        pull_force=np.array([20.0, 20.0]),
        normal_load=np.array([100.0, 100.0]),
        dynamic_pull_force=np.array([15.0, 16.0]),
    )
    variation = variation_metrics(
        np.array([1.0, 2.0, 3.0]), metric_name="steady_speed_rad_s"
    )

    assert spring["stiffness_nm_per_rad"] == pytest.approx(10.0)
    assert friction["dynamic_friction_coefficient"] == pytest.approx(0.155)
    assert variation == {
        "steady_speed_rad_s_mean": pytest.approx(2.0),
        "steady_speed_rad_s_std": pytest.approx(np.std([1.0, 2.0, 3.0])),
        "steady_speed_rad_s_count": 3,
    }


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
        },
        scenario=scenario,
        source="sim",
    )
    trace = load_trace(episode, scenario=scenario)

    with pytest.raises(ContractError, match="lever_arm.*load_force.*clock"):
        compute_subsystem_metrics(scenario, trace)
