from __future__ import annotations

import inspect

import pytest

from tools.sim2real.contracts import CalibrationProfileV1, ContractError
from tools.sim2real.scenarios import load_scenario
from tools.sim2real.sweep import (
    candidate_cache_key,
    generate_coarse_grid_candidates,
    generate_one_factor_candidates,
)


def _profile(profile_id: str = "baseline") -> CalibrationProfileV1:
    return CalibrationProfileV1.from_dict(
        {
            "schema_version": 1,
            "profile_id": profile_id,
            "hardware_mapping": {},
            "sensor_timing": {"aggregate_command_delay_s": 0.01},
            "simulation_physics": {
                "main_drive": {"damping": 0.2},
                "ground": {"static_friction": 0.8, "dynamic_friction": 0.7},
            },
        }
    )


def test_one_factor_candidates_are_deterministic_and_change_one_value() -> None:
    first = generate_one_factor_candidates(
        _profile(),
        {
            "simulation_physics.main_drive.damping": [0.3, 0.2],
            "sensor_timing.aggregate_command_delay_s": [0.02, 0.01],
        },
    )
    second = generate_one_factor_candidates(
        _profile(),
        {
            "sensor_timing.aggregate_command_delay_s": [0.01, 0.02],
            "simulation_physics.main_drive.damping": [0.2, 0.3],
        },
    )

    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]
    assert len(first) == 2
    assert first[0].profile_id == "baseline-one-factor-0001"


def test_coarse_grid_is_bounded_and_order_independent() -> None:
    space = {
        "simulation_physics.main_drive.damping": [0.3, 0.1],
        "sensor_timing.aggregate_command_delay_s": [0.02, 0.01],
    }

    candidates = generate_coarse_grid_candidates(_profile(), space, max_candidates=4)

    assert len(candidates) == 4
    assert candidates[0].simulation_physics["main_drive"]["damping"] == 0.1
    assert candidates[0].sensor_timing["aggregate_command_delay_s"] == 0.01
    with pytest.raises(ContractError, match="max_candidates"):
        generate_coarse_grid_candidates(_profile(), space, max_candidates=3)

    with pytest.raises(ContractError, match="two parameter"):
        generate_coarse_grid_candidates(
            _profile(),
            {
                **space,
                "simulation_physics.ground.static_friction": [0.7, 0.8],
            },
        )


def test_cache_key_tracks_physics_scenario_and_provenance_not_profile_label() -> None:
    scenario = load_scenario("main-step")
    key = candidate_cache_key(_profile("first"), scenario, provenance={"asset": "abc"})

    assert key == candidate_cache_key(
        _profile("renamed"), scenario, provenance={"asset": "abc"}
    )
    assert key != candidate_cache_key(
        _profile("first"), scenario, provenance={"asset": "changed"}
    )
    assert key != candidate_cache_key(_profile("first"), load_scenario("main-coast"))


def test_sweep_has_no_optimizer_dependency() -> None:
    import tools.sim2real.sweep as sweep

    assert "optuna" not in inspect.getsource(sweep).lower()
