from __future__ import annotations

import math

import numpy as np
import pytest

from tools.sim2real.contracts import ContractError, ScenarioSpecV1
from tools.sim2real.scenarios import load_scenario


def test_simulation_times_have_one_sample_per_120_hz_physics_step() -> None:
    from tools.sim2real.characterization import simulation_times

    times = simulation_times(240, 1.0 / 120.0)

    assert times.shape == (240,)
    np.testing.assert_allclose(np.diff(times), 1.0 / 120.0, rtol=0.0, atol=1.0e-12)
    assert times[0] == pytest.approx(1.0 / 120.0)
    assert times[-1] == pytest.approx(2.0)


@pytest.mark.parametrize("steps", [0, -1, True])
def test_simulation_times_reject_invalid_step_counts(steps: object) -> None:
    from tools.sim2real.characterization import simulation_times

    with pytest.raises(ContractError):
        simulation_times(steps, 1.0 / 120.0)  # type: ignore[arg-type]


def test_scenario_schedule_repeats_segments_deterministically() -> None:
    from tools.sim2real.characterization import scenario_schedule, scenario_step_count

    scenario = load_scenario("main-step")
    steps = scenario_step_count(scenario, 1.0 / 120.0)
    schedule = scenario_schedule(scenario, steps, 1.0 / 120.0)

    assert steps == 1260
    assert len(schedule) == 1260
    assert schedule[0].value == 0.0
    assert schedule[59].value == pytest.approx(0.0)
    assert schedule[0].label == "settle"
    assert schedule[-1].repeat_index == 2


def test_command_scenario_schedule_rejects_partial_or_extended_runs() -> None:
    from tools.sim2real.characterization import scenario_schedule

    scenario = load_scenario("main-step")
    with pytest.raises(ContractError, match="requires exactly 1260"):
        scenario_schedule(scenario, 180, 1.0 / 120.0)
    with pytest.raises(ContractError, match="requires exactly 1260"):
        scenario_schedule(scenario, 1261, 1.0 / 120.0)


def test_scenario_schedule_uses_segment_labels_for_coast_disable() -> None:
    from tools.sim2real.characterization import scenario_schedule, scenario_step_count

    scenario = load_scenario("main-coast")
    schedule = scenario_schedule(
        scenario, scenario_step_count(scenario, 1.0 / 120.0), 1.0 / 120.0
    )

    coast = schedule[151]
    assert coast.value == 0.0
    assert coast.label == "coast"
    assert coast.actuator_enabled is False


def test_disabled_neutral_and_coast_segments_match_fail_safe_hardware_probe() -> None:
    from tools.sim2real.characterization import scenario_schedule, scenario_step_count

    payload = load_scenario("main-coast").to_dict()
    payload["scenario_id"] = "combined-response"
    payload["experiment_kind"] = "step_coast"
    payload["command_segments"] = [
        {"duration_s": 0.5, "value": 0.0, "label": "neutral_before_positive"},
        {"duration_s": 1.0, "value": 0.25, "label": "drive_positive"},
        {"duration_s": 1.0, "value": 0.0, "label": "coast_positive"},
    ]
    payload["repeats"] = 1
    scenario = ScenarioSpecV1.from_dict(payload)
    schedule = scenario_schedule(
        scenario, scenario_step_count(scenario, 1.0 / 120.0), 1.0 / 120.0
    )

    assert all(not item.actuator_enabled for item in schedule if "neutral" in item.label)
    assert all(item.actuator_enabled for item in schedule if "drive" in item.label)
    assert all(not item.actuator_enabled for item in schedule if "coast" in item.label)


def test_characterization_trace_metadata_uses_physical_units_and_selected_joint_frame() -> None:
    from tools.sim2real.characterization import characterization_channel_metadata

    channels = {
        "requested_command",
        "applied_command",
        "command",
        "position",
        "root_position",
    }
    units, frames = characterization_channel_metadata(load_scenario("main-step"), channels)

    assert units["requested_command"] == "rad/s"
    assert units["applied_command"] == "rad/s"
    assert units["command"] == "rad/s"
    assert units["position"] == "rad"
    assert frames["command"] == "main_0"
    assert frames["position"] == "main_0"
    assert frames["root_position"] == "world"

    abad_units, abad_frames = characterization_channel_metadata(
        load_scenario("abad-static"), {"command", "position"}
    )
    assert abad_units == {"command": "rad", "position": "rad"}
    assert abad_frames == {"command": "abad_0", "position": "abad_0"}


def test_contact_probe_requires_resolved_bodies_and_measurable_force() -> None:
    from tools.sim2real.characterization import validate_contact_probe

    with pytest.raises(ContractError, match="resolved no robot bodies"):
        validate_contact_probe([], np.array([1.0]), threshold_n=0.05)
    with pytest.raises(ContractError, match="no measurable impulse"):
        validate_contact_probe(["foot"], np.array([0.0, 0.01]), threshold_n=0.05)

    result = validate_contact_probe(
        ["foot"], np.array([0.0, 0.051, math.nan]), threshold_n=0.05
    )
    assert result["max_force_n"] == pytest.approx(0.051)
    assert result["resolved_body_names"] == ["foot"]


def test_foot_contact_probe_rejects_chassis_or_incomplete_body_sets() -> None:
    from tools.sim2real.characterization import (
        EXPECTED_FOOT_BODY_NAMES,
        validate_foot_contact_probe,
    )

    forces = np.ones((3, len(EXPECTED_FOOT_BODY_NAMES)))
    with pytest.raises(ContractError, match="exactly the six terminal feet"):
        validate_foot_contact_probe(["base_link", *EXPECTED_FOOT_BODY_NAMES], forces, threshold_n=0.05)
    with pytest.raises(ContractError, match="exactly the six terminal feet"):
        validate_foot_contact_probe(list(EXPECTED_FOOT_BODY_NAMES[:-1]), forces[:, :-1], threshold_n=0.05)

    result = validate_foot_contact_probe(
        list(reversed(EXPECTED_FOOT_BODY_NAMES)), forces, threshold_n=0.05
    )
    assert set(result["resolved_body_names"]) == set(EXPECTED_FOOT_BODY_NAMES)


@pytest.mark.parametrize("mode", ["fixed-base", "free-root", "contact"])
def test_supported_characterization_modes(mode: str) -> None:
    from tools.sim2real.characterization import validate_run_request

    request = validate_run_request(mode=mode, steps=1, physics_dt=1.0 / 120.0)
    assert request.mode == mode
    assert request.steps == 1


def test_contact_requirement_rejects_fixed_base_mode() -> None:
    from tools.sim2real.characterization import validate_run_request

    with pytest.raises(ContractError, match="free-root or contact"):
        validate_run_request(
            mode="fixed-base", steps=1, physics_dt=1.0 / 120.0, require_contact=True
        )


def test_friction_scenario_always_requires_a_valid_contact_probe() -> None:
    from tools.sim2real.characterization import requires_contact_probe

    assert requires_contact_probe(load_scenario("friction"), mode="free-root", explicit=False)
    assert requires_contact_probe(load_scenario("audit"), mode="contact", explicit=False)
    assert requires_contact_probe(load_scenario("audit"), mode="free-root", explicit=True)
    assert not requires_contact_probe(load_scenario("audit"), mode="free-root", explicit=False)


@pytest.mark.parametrize(
    ("scenario_id", "mode"),
    [("main-step", "fixed-base"), ("friction", "free-root"), ("friction", "contact")],
)
def test_scenario_mode_accepts_only_scientifically_compatible_modes(
    scenario_id: str, mode: str
) -> None:
    from tools.sim2real.characterization import validate_scenario_mode

    validate_scenario_mode(load_scenario(scenario_id), mode)


@pytest.mark.parametrize(
    ("scenario_id", "mode"),
    [("main-step", "free-root"), ("main-step", "contact"), ("friction", "fixed-base")],
)
def test_scenario_mode_rejects_incompatible_modes(scenario_id: str, mode: str) -> None:
    from tools.sim2real.characterization import validate_scenario_mode

    with pytest.raises(ContractError, match="requires"):
        validate_scenario_mode(load_scenario(scenario_id), mode)


@pytest.mark.parametrize("mode", ["fixed-base", "free-root", "contact"])
def test_audit_scenario_allows_explicit_audit_modes(mode: str) -> None:
    from tools.sim2real.characterization import validate_scenario_mode

    validate_scenario_mode(load_scenario("audit"), mode)


def test_simulation_scope_refuses_to_invent_a_friction_pull_measurement() -> None:
    from tools.sim2real.characterization import validate_simulated_experiment

    validate_simulated_experiment(load_scenario("audit"))
    with pytest.raises(ContractError, match="controlled pull"):
        validate_simulated_experiment(load_scenario("friction"))
