from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tools.sim2real.characterization import scenario_schedule, scenario_step_count
from tools.sim2real.compare import _delta, compare_traces
from tools.sim2real.contracts import CalibrationProfileV1, ContractError
from tools.sim2real.scenarios import load_scenario
from tools.sim2real.traces import sha256_json, write_trace


def _profile(*, command_mapping: bool) -> CalibrationProfileV1:
    hardware_mapping: dict[str, dict[str, float]] = {
        "encoder_counts_per_rev": {"main_0": 54984.83},
        "encoder_zero_count": {"main_0": 0.0},
        "encoder_sign": {"main_0": 1.0},
    }
    if command_mapping:
        hardware_mapping.update(
            {
                "joint_direction": {"main_0": 1.0},
                "pwm_scale": {"main_0": 0.002},
                "pwm_cap": {"main_0": 1.0},
            }
        )
    return CalibrationProfileV1.from_dict(
        {
            "schema_version": 1,
            "profile_id": "measured-main-0",
            "hardware_mapping": hardware_mapping,
            "sensor_timing": {},
            "simulation_physics": {},
        }
    )


def _write_pair(
    tmp_path: Path,
    *,
    scenario_id: str = "main-step",
    real_constants: dict[str, object] | None = None,
    profile: CalibrationProfileV1 | None = None,
) -> tuple[Path, Path]:
    scenario = load_scenario(scenario_id)
    steps = scenario_step_count(scenario)
    duration_s = steps / 120.0
    time_s = np.arange(0.0, duration_s + 1.0e-12, 1.0 / 60.0)
    nominal = scenario_schedule(scenario, steps)
    command = np.asarray(
        [nominal[min(int(value * 120.0), steps - 1)].value for value in time_s],
        dtype=np.float64,
    )
    position = np.cumsum(command) / 60.0
    arrays = {
        "command_time_s": time_s,
        "command": command,
        "position_time_s": time_s,
        "position": position,
    }
    metadata = {
        "units": {"command": "rad/s", "position": "rad"},
        "frames": {"command": scenario.joint, "position": scenario.joint},
        "joint_order": [scenario.joint],
        "calibration_constants": dict(real_constants or {}),
    }
    raw = tmp_path / "raw.bin"
    raw.write_bytes(b"verified real source")
    real = tmp_path / "real"
    sim = tmp_path / "sim"
    write_trace(
        real,
        arrays,
        scenario=scenario,
        source="real",
        source_path=raw,
        profile=profile,
        metadata=metadata,
    )
    write_trace(
        sim,
        arrays,
        scenario=scenario,
        source="sim",
        metadata={
            **metadata,
            "calibration_constants": {},
        },
    )
    return real, sim


def test_metric_delta_handles_nested_repeat_records_without_losing_identity() -> None:
    real = {
        "aggregate": {"fit_rmse_rad": 0.10},
        "repeats": [
            {"repeat_index": 0.0, "fit_rmse_rad": 0.08},
            {"repeat_index": 1.0, "fit_rmse_rad": 0.12},
        ],
    }
    sim = {
        "aggregate": {"fit_rmse_rad": 0.13},
        "repeats": [
            {"repeat_index": 0.0, "fit_rmse_rad": 0.10},
            {"repeat_index": 1.0, "fit_rmse_rad": 0.16},
        ],
    }

    delta = _delta(real, sim)

    assert delta["aggregate"]["fit_rmse_rad"] == pytest.approx(0.03)
    assert delta["repeats"] == [
        {"repeat_index": 0.0, "fit_rmse_rad": pytest.approx(0.02)},
        {"repeat_index": 1.0, "fit_rmse_rad": pytest.approx(0.04)},
    ]


@pytest.mark.parametrize(
    "constants",
    [
        {},
        {"calibration_source": "fabricated-measurement"},
    ],
    ids=("missing", "unrecognized-label"),
)
def test_main_drive_comparison_fails_closed_without_mapping_provenance(
    tmp_path: Path, constants: dict[str, object]
) -> None:
    real, sim = _write_pair(tmp_path, real_constants=constants)

    with pytest.raises(ContractError, match="mapping provenance"):
        compare_traces(real, sim, scenario=load_scenario("main-step"))


def test_main_drive_comparison_accepts_two_profile_bound_mappings(tmp_path: Path) -> None:
    profile = _profile(command_mapping=True)
    source = f"profile:{profile.profile_id}"
    real, sim = _write_pair(
        tmp_path,
        profile=profile,
        real_constants={
            "position_mapping_source": source,
            "requested_command_source": source,
        },
    )

    result = compare_traces(real, sim, scenario=load_scenario("main-step"))

    assert result["scenario_id"] == "main-step"


def test_probe_event_commands_are_distinct_from_profile_position_mapping(
    tmp_path: Path,
) -> None:
    scenario = load_scenario("suspended-main-0-step-coast")
    profile = _profile(command_mapping=False)
    scenario_hash = sha256_json(scenario.to_dict())
    duration_s = scenario_step_count(scenario) / 120.0
    real, sim = _write_pair(
        tmp_path,
        scenario_id=scenario.scenario_id,
        profile=profile,
        real_constants={
            "position_mapping_source": f"profile:{profile.profile_id}",
            "requested_command_source": f"authenticated_probe_events:{scenario_hash}",
            "probe_event_evidence": {
                "scenario_sha256": scenario_hash,
                "repetition_count": scenario.repeats,
                "segment_count": scenario.repeats * len(scenario.command_segments),
                "complete_ticks": round(duration_s * 60.0),
                "receive_duration_s": duration_s,
                "receive_jitter_bound_s": 1.0 / 60.0,
                "abad_output_disabled_verified": True,
            },
        },
    )

    result = compare_traces(real, sim, scenario=scenario)

    assert result["scenario_id"] == scenario.scenario_id
