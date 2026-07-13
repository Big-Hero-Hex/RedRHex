from __future__ import annotations

import numpy as np
import pytest

from tools.sim2real.contracts import CalibrationProfileV1, ContractError
from tools.sim2real.metrics import abad_static_mapping_metrics, friction_metrics


def _baseline() -> CalibrationProfileV1:
    return CalibrationProfileV1.from_dict(
        {
            "schema_version": 1,
            "profile_id": "baseline",
            "description": "preserve me",
            "hardware_mapping": {"pwm_cap": {"main_0": 0.5}},
            "sensor_timing": {"aggregate_command_delay_s": 0.02},
            "simulation_physics": {
                "main_drive": {"damping": 1.2},
                "ground": {"restitution": 0.1},
            },
            "measurement_sources": {"mass": "c" * 64},
        }
    )


def _abad_metrics():
    command = np.tile(np.array([-0.2, 0.0, 0.2]), 3)
    return abad_static_mapping_metrics(
        command,
        1.25 * command - 0.03,
        repeat_index=np.repeat(np.arange(3), 3),
        settled=np.ones(9),
        expected_repeats=3,
        frame="abad_0",
    )


def _friction_metrics():
    return friction_metrics(
        breakaway_force=np.array([20.0, 21.0, 19.0]),
        static_normal_load=np.full(3, 100.0),
        static_repeat_index=np.arange(3),
        dynamic_pull_force=np.full(9, 15.0),
        dynamic_normal_load=np.full(9, 100.0),
        dynamic_speed=np.full(9, 0.05),
        dynamic_repeat_index=np.repeat(np.arange(3), 3),
        expected_repeats=3,
        frame="foot_0/ground",
        max_dynamic_speed_m_s=0.1,
    )


def test_measurements_update_candidate_profile_without_erasing_unrelated_values() -> None:
    from tools.sim2real.profile_measurements import apply_measurements_to_profile

    candidate = apply_measurements_to_profile(
        _baseline(),
        profile_id="measured-candidate",
        abad_metrics=_abad_metrics(),
        abad_trace_sha256="a" * 64,
        friction_metrics=_friction_metrics(),
        friction_trace_sha256="b" * 64,
    )

    assert candidate.profile_id == "measured-candidate"
    assert candidate.description == "preserve me"
    assert candidate.hardware_mapping["pwm_cap"] == {"main_0": 0.5}
    assert candidate.hardware_mapping["abad_target_scale"] == {"abad_0": pytest.approx(1.25)}
    assert candidate.hardware_mapping["abad_target_offset_rad"] == {
        "abad_0": pytest.approx(-0.03)
    }
    assert candidate.sensor_timing == {"aggregate_command_delay_s": 0.02}
    assert candidate.simulation_physics["main_drive"] == {"damping": 1.2}
    assert candidate.simulation_physics["ground"] == {
        "restitution": 0.1,
        "static_friction": pytest.approx(0.2),
        "dynamic_friction": pytest.approx(0.15),
    }
    assert candidate.measurement_sources == {
        "mass": "c" * 64,
        "abad_target:abad_0": "a" * 64,
        "ground_friction": "b" * 64,
    }


def test_measurement_profile_update_requires_matching_trace_hash_and_metric_contract() -> None:
    from tools.sim2real.profile_measurements import apply_measurements_to_profile

    with pytest.raises(ContractError, match="trace SHA"):
        apply_measurements_to_profile(
            _baseline(),
            profile_id="missing-source",
            abad_metrics=_abad_metrics(),
        )

    metrics = _abad_metrics()
    metrics["metric_kind"] = "ground_friction"
    with pytest.raises(ContractError, match="ABAD metric contract"):
        apply_measurements_to_profile(
            _baseline(),
            profile_id="wrong-kind",
            abad_metrics=metrics,
            abad_trace_sha256="a" * 64,
        )
