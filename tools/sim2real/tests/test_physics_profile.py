from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.sim2real.contracts import CalibrationProfileV1, ContractError


def _profile() -> CalibrationProfileV1:
    return CalibrationProfileV1.from_dict(
        {
            "schema_version": 1,
            "profile_id": "candidate-a",
            "hardware_mapping": {"pwm_scale": {"Revolute_15": 0.2}},
            "sensor_timing": {
                "aggregate_command_delay_s": 0.025,
            },
            "simulation_physics": {
                "rigid_body": {"linear_damping": 0.11, "angular_damping": 0.22},
                "main_drive": {
                    "stiffness": 0.0,
                    "damping": 1.5,
                    "effort_limit": 12.0,
                    "velocity_limit": 14.0,
                    "armature": 0.02,
                    "friction": 0.03,
                },
                "abad": {"stiffness": 42.0, "damping": 4.2},
                "damper": {"stiffness": 190.0, "damping": 19.0},
                "joint_friction": {"Revolute_15": 0.04},
                "joint_dynamic_friction": {"Revolute_15": 0.03},
                "joint_viscous_friction": {"Revolute_15": 0.02},
                "passive_spring": {
                    "Revolute_5": {
                        "stiffness": 180.0,
                        "damping": 18.0,
                        "rest_position_rad": 0.7,
                    }
                },
                "mass": {"scale": 1.1, "added_mass_kg": 0.5, "com_offset_m": [0.01, 0.0, -0.02]},
                "ground": {"static_friction": 0.9, "dynamic_friction": 0.8, "restitution": 0.1},
            },
        }
    )


def _fake_env_cfg() -> SimpleNamespace:
    material_a = SimpleNamespace(static_friction=1.2, dynamic_friction=1.0, restitution=0.0)
    material_b = SimpleNamespace(static_friction=1.2, dynamic_friction=1.0, restitution=0.0)
    actuators = {
        name: SimpleNamespace(
            stiffness=0.0,
            damping=0.0,
            effort_limit_sim=1.0,
            velocity_limit_sim=1.0,
            armature=0.0,
            friction=0.0,
        )
        for name in ("main_drive", "abad", "damper")
    }
    robot_cfg = SimpleNamespace(
        spawn=SimpleNamespace(
            rigid_props=SimpleNamespace(linear_damping=0.0, angular_damping=0.0)
        ),
        actuators=actuators,
        init_state=SimpleNamespace(joint_pos={"Revolute_5": 0.0}),
    )
    return SimpleNamespace(
        robot_cfg=robot_cfg,
        sim=SimpleNamespace(physics_material=material_a, dt=1.0 / 120.0),
        terrain=SimpleNamespace(physics_material=material_b),
        damper_stiffness=200.0,
        damper_damping=20.0,
        sim2real_command_delay_steps=0,
    )


def test_profile_application_updates_only_explicit_candidate_config() -> None:
    from tools.sim2real.physics_profile import apply_profile_to_config

    original = _fake_env_cfg()
    untouched = copy.deepcopy(original)
    candidate = copy.deepcopy(original)

    summary = apply_profile_to_config(candidate, _profile())

    assert original.robot_cfg.spawn.rigid_props.linear_damping == untouched.robot_cfg.spawn.rigid_props.linear_damping
    assert candidate.robot_cfg.spawn.rigid_props.linear_damping == 0.11
    assert candidate.robot_cfg.spawn.rigid_props.angular_damping == 0.22
    main = candidate.robot_cfg.actuators["main_drive"]
    assert main.damping == 1.5
    assert main.effort_limit_sim == 12.0
    assert main.velocity_limit_sim == 14.0
    assert candidate.robot_cfg.init_state.joint_pos["Revolute_5"] == 0.7
    assert candidate.sim.physics_material.static_friction == 0.9
    assert candidate.terrain.physics_material.dynamic_friction == 0.8
    assert summary["profile_id"] == "candidate-a"
    assert summary["sensor_timing"]["aggregate_command_delay_s"] == 0.025
    assert summary["sensor_timing"]["command_delay_steps"] == 3
    assert summary["sensor_timing"]["effective_command_delay_s"] == pytest.approx(0.025)
    assert candidate.sim2real_command_delay_steps == 3


def test_none_profile_is_a_noop() -> None:
    from tools.sim2real.physics_profile import apply_profile_to_config

    cfg = _fake_env_cfg()
    before = copy.deepcopy(cfg)
    assert apply_profile_to_config(cfg, None) is None
    assert cfg.robot_cfg.spawn.rigid_props.linear_damping == before.robot_cfg.spawn.rigid_props.linear_damping


@pytest.mark.parametrize(
    "field",
    [
        "command_delay_s",
        "sensor_delay_s",
        "sample_period_s",
        "measured_state_rate_hz",
        "velocity_filter_alpha",
        "position_noise_std_rad",
        "velocity_filter_window_s",
    ],
)
def test_simulation_profile_rejects_timing_fields_without_a_physical_effect(field: str) -> None:
    from tools.sim2real.physics_profile import apply_profile_to_config

    payload = _profile().to_dict()
    payload["sensor_timing"] = {field: 60.0 if field == "measured_state_rate_hz" else 0.01}
    profile = CalibrationProfileV1.from_dict(payload)
    cfg = _fake_env_cfg()
    before = copy.deepcopy(cfg)

    with pytest.raises(ContractError, match="unsupported sensor_timing"):
        apply_profile_to_config(cfg, profile)

    assert cfg.robot_cfg.spawn.rigid_props.linear_damping == before.robot_cfg.spawn.rigid_props.linear_damping


def test_positive_substep_delay_is_not_silently_rounded_to_zero() -> None:
    from tools.sim2real.physics_profile import apply_profile_to_config

    payload = _profile().to_dict()
    payload["sensor_timing"] = {"aggregate_command_delay_s": 0.001}
    cfg = _fake_env_cfg()

    summary = apply_profile_to_config(cfg, CalibrationProfileV1.from_dict(payload))

    assert cfg.sim2real_command_delay_steps == 1
    assert summary["sensor_timing"]["effective_command_delay_s"] == pytest.approx(1.0 / 120.0)


def test_mass_correction_scales_mass_and_inertia_and_offsets_root_com() -> None:
    from tools.sim2real.physics_profile import corrected_mass_properties

    masses = np.array([[2.0, 3.0]])
    inertias = np.ones((1, 2, 9))
    coms = np.zeros((1, 2, 7))
    coms[..., 6] = 1.0

    corrected = corrected_mass_properties(
        masses,
        inertias,
        coms,
        {"scale": 1.1, "added_mass_kg": 0.5, "com_offset_m": [0.01, 0.0, -0.02]},
    )

    np.testing.assert_allclose(corrected.masses, [[2.7, 3.3]])
    np.testing.assert_allclose(corrected.inertias, inertias * 1.1)
    np.testing.assert_allclose(corrected.coms[0, 0, :3], [0.01, 0.0, -0.02])
    np.testing.assert_allclose(corrected.coms[0, 1, :3], [0.0, 0.0, 0.0])


def test_load_optional_profile_preserves_default_none(tmp_path: Path) -> None:
    from tools.sim2real.physics_profile import load_optional_profile

    assert load_optional_profile(None) is None

    path = tmp_path / "candidate.json"
    import json

    path.write_text(json.dumps(_profile().to_dict()), encoding="utf-8")
    assert load_optional_profile(path).profile_id == "candidate-a"


def test_runtime_mass_profile_replaces_domain_randomization_baseline() -> None:
    import torch

    from tools.sim2real.isaac_profile import apply_profile_to_runtime_env

    class FakeView:
        def __init__(self) -> None:
            self.masses = torch.tensor([[99.0, 99.0]])
            self.inertias = torch.full((1, 2, 9), 99.0)
            self.coms = torch.zeros((1, 2, 7))

        def get_masses(self):
            return self.masses

        def get_inertias(self):
            return self.inertias

        def get_coms(self):
            return self.coms

        def set_masses(self, values, _indices):
            self.masses = values.clone()

        def set_inertias(self, values, _indices):
            self.inertias = values.clone()

        def set_coms(self, values, _indices):
            self.coms = values.clone()

    profile_data = _profile().to_dict()
    profile_data["simulation_physics"] = {
        "mass": {"scale": 2.0, "added_mass_kg": 1.0, "com_offset_m": [0.0, 0.0, 0.0]}
    }
    profile = CalibrationProfileV1.from_dict(profile_data)
    view = FakeView()
    data = SimpleNamespace(
        default_mass=torch.tensor([[2.0, 3.0]]),
        default_inertia=torch.ones((1, 2, 9)),
    )
    robot = SimpleNamespace(
        root_physx_view=view,
        data=data,
        num_instances=1,
        device="cpu",
        joint_names=[],
    )
    unwrapped = SimpleNamespace(
        robot=robot,
        _default_body_masses=torch.tensor([[2.0, 3.0]]),
        _robot_mass=5.0,
    )

    apply_profile_to_runtime_env(SimpleNamespace(unwrapped=unwrapped), profile)

    expected = torch.tensor([[5.0, 6.0]])
    torch.testing.assert_close(view.masses, expected)
    torch.testing.assert_close(data.default_mass, expected)
    torch.testing.assert_close(unwrapped._default_body_masses, expected)
    assert unwrapped._robot_mass == 11.0
