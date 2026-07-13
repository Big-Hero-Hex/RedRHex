from __future__ import annotations

import json
import math
from pathlib import Path

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
        load_scenario("abad-static"),
        {"command", "position", "repeat_index", "settled"},
    )
    assert abad_units == {
        "command": "rad",
        "position": "rad",
        "repeat_index": "1",
        "settled": "1",
    }
    assert abad_frames == {
        "command": "abad_0",
        "position": "abad_0",
        "repeat_index": "scalar",
        "settled": "scalar",
    }


def test_abad_measurement_annotations_mark_settled_tail_of_every_pose_and_repeat() -> None:
    from tools.sim2real.characterization import (
        measurement_annotations,
        scenario_schedule,
        scenario_step_count,
    )

    scenario = load_scenario("abad-static")
    schedule = scenario_schedule(
        scenario, scenario_step_count(scenario, 1.0 / 120.0), 1.0 / 120.0
    )
    repeat_index, settled = measurement_annotations(schedule, settled_fraction=0.25)

    assert repeat_index.shape == settled.shape == (540,)
    assert set(np.unique(repeat_index)) == {0.0, 1.0, 2.0}
    assert int(np.count_nonzero(settled)) == 3 * 3 * 15
    assert not np.any(settled[:45])
    assert np.all(settled[45:60])
    assert not np.any(settled[60:105])
    assert np.all(settled[105:120])


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


def test_contact_probe_requirement_remains_explicit_for_simulated_audits() -> None:
    from tools.sim2real.characterization import requires_contact_probe

    assert requires_contact_probe(load_scenario("audit"), mode="contact", explicit=False)
    assert requires_contact_probe(load_scenario("audit"), mode="free-root", explicit=True)
    assert not requires_contact_probe(load_scenario("audit"), mode="free-root", explicit=False)


@pytest.mark.parametrize(
    ("scenario_id", "mode"),
    [("main-step", "fixed-base")],
)
def test_scenario_mode_accepts_only_scientifically_compatible_modes(
    scenario_id: str, mode: str
) -> None:
    from tools.sim2real.characterization import validate_scenario_mode

    validate_scenario_mode(load_scenario(scenario_id), mode)


@pytest.mark.parametrize(
    ("scenario_id", "mode"),
    [
        ("main-step", "free-root"),
        ("main-step", "contact"),
        ("friction", "fixed-base"),
        ("friction", "free-root"),
        ("friction", "contact"),
    ],
)
def test_scenario_mode_rejects_incompatible_modes(scenario_id: str, mode: str) -> None:
    from tools.sim2real.characterization import validate_scenario_mode

    with pytest.raises(ContractError, match="requires|manual scenario"):
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


def test_command_delay_shifts_requested_schedule_by_exact_physics_steps() -> None:
    from tools.sim2real.characterization import apply_schedule_delay, scenario_schedule

    scenario = load_scenario("main-step")
    requested = scenario_schedule(scenario, 1260, 1.0 / 120.0)

    applied = apply_schedule_delay(requested, delay_steps=3)

    assert requested[60].value == pytest.approx(0.25)
    assert [item.value for item in applied[60:63]] == [0.0, 0.0, 0.0]
    assert applied[63].value == pytest.approx(0.25)
    assert apply_schedule_delay(requested, delay_steps=0) == requested


def _write_replay_trace(
    tmp_path: Path,
    *,
    scenario_id: str = "main-step",
    unit: str = "rad/s",
    frame: str | None = None,
    end_time_s: float | None = None,
    source: str = "real",
    calibration_constants: dict[str, object] | None = None,
    initial_position_rad: float = 0.125,
    declare_initial_state: bool = True,
    ambiguous_position: bool = False,
) -> Path:
    from tools.sim2real.characterization import scenario_schedule, scenario_step_count
    from tools.sim2real.traces import write_trace

    scenario = load_scenario(scenario_id)
    steps = scenario_step_count(scenario, 1.0 / 120.0)
    duration_s = steps / 120.0
    final_time = duration_s if end_time_s is None else end_time_s
    command_time = np.arange(0.0, final_time + 1.0e-12, 1.0 / 60.0)
    nominal = scenario_schedule(scenario, steps, 1.0 / 120.0)
    command = np.asarray(
        [nominal[min(int(time_s * 120.0), steps - 1)].value for time_s in command_time],
        dtype=np.float64,
    )
    raw = tmp_path / f"{scenario_id}-raw.bin"
    raw.write_bytes(b"immutable raw replay source")
    output = tmp_path / f"{scenario_id}-episode"
    constants = dict(calibration_constants or {})
    initial_state = {
        "schema_version": 1,
        "joint": scenario.joint,
        "source_channel": "position",
        "position_rad": initial_position_rad,
        "sample_time_s": 0.0,
        "scenario_time_s": 0.0,
        "sample_offset_s": 0.0,
    }
    if declare_initial_state:
        from tools.sim2real.traces import sha256_json

        constants["replay_initial_state"] = initial_state
        constants["replay_initial_state_sha256"] = sha256_json(initial_state)
    position = (
        np.full((2, 2), initial_position_rad, dtype=np.float64)
        if ambiguous_position
        else np.full(2, initial_position_rad, dtype=np.float64)
    )
    metadata = {
        "units": {"command": unit, "position": "rad"},
        "frames": {
            "command": frame or scenario.joint,
            "position": scenario.joint,
        },
        "calibration_constants": constants,
    }
    write_trace(
        output,
        {
            "command_time_s": command_time,
            "command": command,
            "position_time_s": np.asarray([0.0, final_time], dtype=np.float64),
            "position": position,
        },
        scenario=scenario,
        source=source,
        source_path=raw,
        metadata=metadata,
    )
    return output


def test_real_trace_replay_verifies_provenance_and_resamples_exact_scenario(
    tmp_path: Path,
) -> None:
    from tools.sim2real.characterization import load_replay_schedule

    trace = _write_replay_trace(tmp_path)
    replay = load_replay_schedule(
        trace,
        load_scenario("main-step"),
        steps=1260,
        physics_dt=1.0 / 120.0,
    )

    assert len(replay.schedule) == 1260
    assert replay.schedule[59].value == pytest.approx(0.0)
    assert replay.schedule[60].value == pytest.approx(0.25)
    assert len(replay.trace_sha256) == 64
    assert replay.initial_state.position_rad == pytest.approx(0.125)
    assert replay.initial_state.joint == "main_0"
    assert len(replay.initial_state_sha256) == 64


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"declare_initial_state": False}, "initial state declaration"),
        ({"ambiguous_position": True}, "one-dimensional"),
    ],
    ids=("missing-declaration", "ambiguous-position-channel"),
)
def test_real_trace_replay_rejects_missing_or_ambiguous_initial_state(
    tmp_path: Path,
    override: dict[str, object],
    expected: str,
) -> None:
    from tools.sim2real.characterization import load_replay_schedule

    trace = _write_replay_trace(tmp_path, **override)

    with pytest.raises(ContractError, match=expected):
        load_replay_schedule(
            trace,
            load_scenario("main-step"),
            steps=1260,
            physics_dt=1.0 / 120.0,
        )


def test_real_trace_replay_rejects_initial_state_hash_mismatch(tmp_path: Path) -> None:
    from tools.sim2real.characterization import load_replay_schedule

    trace = _write_replay_trace(tmp_path)
    metadata_path = trace / "metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["metadata"]["calibration_constants"]["replay_initial_state"][
        "position_rad"
    ] = 0.5
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ContractError, match="initial state hash"):
        load_replay_schedule(
            trace,
            load_scenario("main-step"),
            steps=1260,
            physics_dt=1.0 / 120.0,
        )


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"unit": "normalized_pwm"}, "unit"),
        ({"frame": "main_1"}, "frame"),
        ({"end_time_s": 1.0}, "cover"),
        ({"source": "sim"}, "real"),
    ],
)
def test_real_trace_replay_rejects_incompatible_or_incomplete_trace(
    tmp_path: Path, override: dict[str, object], expected: str
) -> None:
    from tools.sim2real.characterization import load_replay_schedule

    trace = _write_replay_trace(tmp_path, **override)
    with pytest.raises(ContractError, match=expected):
        load_replay_schedule(
            trace,
            load_scenario("main-step"),
            steps=1260,
            physics_dt=1.0 / 120.0,
        )


def test_real_trace_replay_rejects_scenario_id_or_hash_mismatch(tmp_path: Path) -> None:
    from tools.sim2real.characterization import load_replay_schedule, scenario_step_count

    trace = _write_replay_trace(tmp_path)
    other = load_scenario("main-coast")
    with pytest.raises(ContractError, match="scenario"):
        load_replay_schedule(
            trace,
            other,
            steps=scenario_step_count(other),
            physics_dt=1.0 / 120.0,
        )


def test_bound_probe_completion_evidence_covers_suppressed_final_neutral(
    tmp_path: Path,
) -> None:
    from tools.sim2real.characterization import load_replay_schedule, scenario_step_count

    scenario = load_scenario("suspended-main-0-step-coast")
    duration_s = scenario_step_count(scenario) / 120.0
    trace = _write_replay_trace(
        tmp_path,
        scenario_id=scenario.scenario_id,
        end_time_s=duration_s - 0.5,
        calibration_constants={
            "probe_event_evidence": {
                "scenario_receive_time_s": 0.0,
                "complete_receive_time_s": duration_s,
                "complete_ticks": int(duration_s * 60.0),
            }
        },
    )

    replay = load_replay_schedule(
        trace,
        scenario,
        steps=scenario_step_count(scenario),
        physics_dt=1.0 / 120.0,
    )

    assert len(replay.schedule) == scenario_step_count(scenario)
    assert replay.schedule[-1].value == pytest.approx(0.0)
