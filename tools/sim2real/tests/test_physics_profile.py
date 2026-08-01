from __future__ import annotations

import copy
import json
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
            "hardware_mapping": {
                "pwm_scale": {"Revolute_15": 0.2},
                "abad_target_scale": {"abad_0": 1.2, "abad_5": 0.9},
                "abad_target_offset_rad": {"abad_0": -0.03, "abad_5": 0.04},
            },
            "sensor_timing": {
                "aggregate_command_delay_s": 0.025,
                "measured_state_rate_hz": 60.0,
                "velocity_filter_alpha": 0.35,
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
                "abad": {
                    "stiffness": 42.0,
                    "damping": 4.2,
                    "effort_limit": 7.5,
                },
                "damper": {"stiffness": 190.0, "damping": 19.0},
                "joint_friction": {"main_0": 0.04},
                "joint_dynamic_friction": {"main_0": 0.03},
                "joint_viscous_friction": {"main_0": 0.02},
                "passive_spring": {
                    "damper_0": {
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
    material_a = SimpleNamespace(
        static_friction=1.2,
        dynamic_friction=1.0,
        restitution=0.0,
        friction_combine_mode="multiply",
    )
    material_b = SimpleNamespace(
        static_friction=1.2,
        dynamic_friction=1.0,
        restitution=0.0,
        friction_combine_mode="multiply",
    )
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
        spring_backend="explicit",
        spring_stiffness_nm_per_rad=(200.0,) * 6,
        spring_damping_nm_s_per_rad=(20.0,) * 6,
        sim2real_command_delay_steps=0,
        sim2real_abad_target_scale=(1.0,) * 6,
        sim2real_abad_target_offset_rad=(0.0,) * 6,
        main_drive_joint_names=[
            "Revolute_15",
            "Revolute_7",
            "Revolute_12",
            "Revolute_18",
            "Revolute_23",
            "Revolute_24",
        ],
        abad_joint_names=[
            "Revolute_14",
            "Revolute_6",
            "Revolute_11",
            "Revolute_17",
            "Revolute_22",
            "Revolute_21",
        ],
        damper_joint_names=[
            "Revolute_5",
            "Revolute_8",
            "Revolute_13",
            "Revolute_25",
            "Revolute_26",
            "Revolute_27",
        ],
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
    assert candidate.main_drive_torque_estimate_damping == 1.5
    assert candidate.main_drive_torque_estimate_limit == 12.0
    assert candidate.abad_torque_estimate_stiffness == 42.0
    assert candidate.abad_torque_estimate_damping == 4.2
    assert candidate.abad_torque_estimate_limit == 7.5
    assert candidate.damper_stiffness == 190.0
    assert candidate.damper_damping == 19.0
    assert candidate.spring_stiffness_nm_per_rad == (
        180.0,
        190.0,
        190.0,
        190.0,
        190.0,
        190.0,
    )
    assert candidate.spring_damping_nm_s_per_rad == (
        18.0,
        19.0,
        19.0,
        19.0,
        19.0,
        19.0,
    )
    assert candidate.robot_cfg.actuators["damper"].stiffness == 0.0
    assert candidate.robot_cfg.actuators["damper"].damping == 0.0
    assert candidate.robot_cfg.init_state.joint_pos["Revolute_5"] == 0.7
    assert candidate.sim.physics_material.static_friction == 0.9
    assert candidate.terrain.physics_material.dynamic_friction == 0.8
    assert candidate.sim.physics_material.friction_combine_mode == "max"
    assert candidate.terrain.physics_material.friction_combine_mode == "max"
    assert summary["profile_id"] == "candidate-a"
    assert summary["sensor_timing"]["aggregate_command_delay_s"] == 0.025
    assert summary["sensor_timing"]["command_delay_steps"] == 3
    assert summary["sensor_timing"]["effective_command_delay_s"] == pytest.approx(0.025)
    assert summary["sensor_timing"]["inactive_metadata_fields"] == [
        "measured_state_rate_hz",
        "velocity_filter_alpha",
    ]
    assert candidate.sim2real_command_delay_steps == 3
    assert candidate.sim2real_abad_target_scale == (1.2, 1.0, 1.0, 1.0, 1.0, 0.9)
    assert candidate.sim2real_abad_target_offset_rad == (-0.03, 0.0, 0.0, 0.0, 0.0, 0.04)


def test_native_profile_resolves_same_effective_springs_but_configures_physx_gains() -> None:
    from tools.sim2real.physics_profile import apply_profile_to_config

    candidate = _fake_env_cfg()
    candidate.spring_backend = "native"

    summary = apply_profile_to_config(candidate, _profile())

    assert candidate.spring_stiffness_nm_per_rad == (
        180.0,
        190.0,
        190.0,
        190.0,
        190.0,
        190.0,
    )
    assert candidate.spring_damping_nm_s_per_rad == (
        18.0,
        19.0,
        19.0,
        19.0,
        19.0,
        19.0,
    )
    assert candidate.robot_cfg.actuators["damper"].stiffness == 190.0
    assert candidate.robot_cfg.actuators["damper"].damping == 19.0
    assert summary["spring_backend"] == "native"


def test_scalar_abad_mapping_uses_actual_equals_scale_times_requested_plus_offset() -> None:
    from tools.sim2real.physics_profile import apply_abad_target_mapping

    profile = _profile()

    assert apply_abad_target_mapping(
        0.2, profile, joint="abad_0", minimum_rad=-0.5, maximum_rad=0.5
    ) == pytest.approx(0.21)
    assert apply_abad_target_mapping(
        0.2, profile, joint="abad_1", minimum_rad=-0.5, maximum_rad=0.5
    ) == pytest.approx(0.2)
    assert apply_abad_target_mapping(
        0.2, None, joint="abad_0", minimum_rad=-0.5, maximum_rad=0.5
    ) == pytest.approx(0.2)
    assert apply_abad_target_mapping(
        0.5, profile, joint="abad_0", minimum_rad=-0.4, maximum_rad=0.4
    ) == pytest.approx(0.4)


def test_none_profile_is_a_noop() -> None:
    from tools.sim2real.physics_profile import apply_profile_to_config

    cfg = _fake_env_cfg()
    before = copy.deepcopy(cfg)
    assert apply_profile_to_config(cfg, None) is None
    assert cfg.robot_cfg.spawn.rigid_props.linear_damping == before.robot_cfg.spawn.rigid_props.linear_damping


def test_restitution_only_profile_does_not_change_friction_combine_mode() -> None:
    from tools.sim2real.physics_profile import apply_profile_to_config

    payload = _profile().to_dict()
    payload["simulation_physics"] = {"ground": {"restitution": 0.2}}
    cfg = _fake_env_cfg()

    apply_profile_to_config(cfg, CalibrationProfileV1.from_dict(payload))

    assert cfg.sim.physics_material.friction_combine_mode == "multiply"
    assert cfg.terrain.physics_material.friction_combine_mode == "multiply"


@pytest.mark.parametrize(
    "field",
    [
        "command_delay_s",
        "sensor_delay_s",
        "sample_period_s",
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


def test_measured_sensor_processing_loads_as_explicitly_inactive_metadata() -> None:
    from tools.sim2real.physics_profile import apply_profile_to_config

    payload = _profile().to_dict()
    payload["sensor_timing"] = {
        "aggregate_command_delay_s": 0.01,
        "measured_state_rate_hz": 60.0,
        "velocity_filter_alpha": 0.35,
    }
    cfg = _fake_env_cfg()

    summary = apply_profile_to_config(cfg, CalibrationProfileV1.from_dict(payload))

    assert cfg.sim2real_command_delay_steps == 1
    assert summary["sensor_timing"]["measured_state_rate_hz"] == 60.0
    assert summary["sensor_timing"]["velocity_filter_alpha"] == 0.35
    assert summary["sensor_timing"]["inactive_metadata_fields"] == [
        "measured_state_rate_hz",
        "velocity_filter_alpha",
    ]


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


def test_training_profile_snapshot_records_canonical_profile_and_applications(
    tmp_path: Path,
) -> None:
    from tools.sim2real.physics_profile import write_training_profile_snapshot
    from tools.sim2real.traces import sha256_json

    profile = _profile()
    result = write_training_profile_snapshot(
        tmp_path,
        profile,
        config_application={"profile_id": profile.profile_id, "phase": "config"},
        runtime_application={"profile_id": profile.profile_id, "phase": "runtime"},
    )

    profile_path = tmp_path / "params" / "physics_profile.json"
    metadata_path = tmp_path / "params" / "physics_profile_metadata.json"
    assert json.loads(profile_path.read_text(encoding="utf-8")) == profile.to_dict()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata == {
        "schema_version": 1,
        "profile_id": profile.profile_id,
        "profile_sha256": sha256_json(profile.to_dict()),
        "config_application": {"profile_id": profile.profile_id, "phase": "config"},
        "runtime_application": {"profile_id": profile.profile_id, "phase": "runtime"},
    }
    assert result == metadata


def test_training_profile_snapshot_none_preserves_default_behavior(tmp_path: Path) -> None:
    from tools.sim2real.physics_profile import write_training_profile_snapshot

    assert write_training_profile_snapshot(
        tmp_path,
        None,
        config_application=None,
        runtime_application=None,
    ) is None
    assert not (tmp_path / "params").exists()


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


def _absolute_mass_profile(
    *,
    target_mass_kg: float = 10.0,
    target_com_xy_m: tuple[float, float] = (0.8, 0.2),
) -> CalibrationProfileV1:
    payload = _profile().to_dict()
    payload["simulation_physics"] = {
        "mass": {
            "target_total_mass_kg": target_mass_kg,
            "reference_planar_com_xy_m": list(target_com_xy_m),
            "reference_joint_position_rad": {
                f"{group}_{index}": 0.0
                for group in ("main", "abad", "damper")
                for index in range(6)
            },
            "reference_root_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
    }
    return CalibrationProfileV1.from_dict(payload)


def _absolute_mass_runtime(
    *, state_device: str = "cpu"
) -> tuple[SimpleNamespace, object]:
    import torch

    cfg = _fake_env_cfg()
    runtime_names = (
        cfg.main_drive_joint_names
        + cfg.abad_joint_names
        + cfg.damper_joint_names
    )

    class FakeView:
        def __init__(self) -> None:
            self.masses = torch.tensor([[2.0, 3.0]])
            self.inertias = torch.ones((1, 2, 9))
            self.coms = torch.zeros((1, 2, 7))
            self.coms[..., 6] = 1.0

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

    view = FakeView()
    data = SimpleNamespace(
        default_mass=view.masses.clone(),
        default_inertia=view.inertias.clone(),
        joint_pos=torch.zeros((1, 18), device=state_device),
        root_pos_w=torch.tensor([[10.0, 0.0, 0.0]], device=state_device),
        root_quat_w=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]], device=state_device
        ),
        body_com_pos_w=torch.tensor(
            [[[10.0, 0.0, 0.0], [11.0, 0.0, 0.0]]], device=state_device
        ),
    )
    robot = SimpleNamespace(
        root_physx_view=view,
        data=data,
        num_instances=1,
        device=state_device,
        joint_names=runtime_names,
    )
    unwrapped = SimpleNamespace(
        robot=robot,
        cfg=cfg,
        _default_body_masses=view.masses.clone(),
        _robot_mass=5.0,
    )
    return unwrapped, view


def test_absolute_mass_profile_hits_measured_mass_and_planar_com() -> None:
    import torch

    from tools.sim2real.isaac_profile import apply_profile_to_runtime_env

    unwrapped, view = _absolute_mass_runtime()

    summary = apply_profile_to_runtime_env(
        SimpleNamespace(unwrapped=unwrapped), _absolute_mass_profile()
    )

    torch.testing.assert_close(view.masses, torch.tensor([[4.0, 6.0]]))
    torch.testing.assert_close(view.inertias, torch.full((1, 2, 9), 2.0))
    torch.testing.assert_close(view.coms[0, 0, :3], torch.tensor([0.5, 0.5, 0.0]))
    assert summary["mass"]["mode"] == "absolute"
    assert summary["mass"]["reference_total_mass_kg"] == pytest.approx(5.0)
    assert summary["mass"]["reference_whole_com_root_m"] == pytest.approx(
        [0.6, 0.0, 0.0]
    )
    assert summary["mass"]["achieved_total_mass_kg"] == pytest.approx(10.0)
    assert summary["mass"]["achieved_whole_com_root_m"][:2] == pytest.approx(
        [0.8, 0.2]
    )
    assert unwrapped._robot_mass == pytest.approx(10.0)


def test_absolute_mass_supports_cpu_physx_properties_with_cuda_state() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from tools.sim2real.isaac_profile import apply_profile_to_runtime_env

    unwrapped, view = _absolute_mass_runtime(state_device="cuda")

    summary = apply_profile_to_runtime_env(
        SimpleNamespace(unwrapped=unwrapped), _absolute_mass_profile()
    )

    assert view.masses.device.type == "cpu"
    assert summary["mass"]["achieved_total_mass_kg"] == pytest.approx(10.0)
    assert summary["mass"]["achieved_whole_com_root_m"][:2] == pytest.approx(
        [0.8, 0.2]
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda runtime: runtime.robot.data.joint_pos.__setitem__((0, 0), 0.1), "joint pose"),
        (
            lambda runtime: setattr(
                runtime.robot.data,
                "root_quat_w",
                __import__("torch").tensor([[0.0, 1.0, 0.0, 0.0]]),
            ),
            "root orientation",
        ),
        (
            lambda runtime: delattr(runtime.robot.data, "body_com_pos_w"),
            "body_com_pos_w",
        ),
    ],
    ids=("joint-pose-mismatch", "orientation-mismatch", "missing-body-com-state"),
)
def test_absolute_mass_profile_fails_closed_without_matching_reference_state(
    mutation, message: str
) -> None:
    from tools.sim2real.isaac_profile import apply_profile_to_runtime_env

    unwrapped, _view = _absolute_mass_runtime()
    mutation(unwrapped)

    with pytest.raises(ContractError, match=message):
        apply_profile_to_runtime_env(
            SimpleNamespace(unwrapped=unwrapped), _absolute_mass_profile()
        )


def test_runtime_profile_overrides_unknown_robot_material_for_measured_pair_friction() -> None:
    import torch

    from tools.sim2real.isaac_profile import apply_profile_to_runtime_env

    class FakeView:
        def __init__(self) -> None:
            self.materials = torch.tensor(
                [[[1.7, 1.4, 0.3], [0.2, 0.1, 0.4], [3.0, 2.0, 0.5]]]
            )

        def get_material_properties(self):
            return self.materials

        def set_material_properties(self, values, _indices):
            self.materials = values.clone()

    payload = _profile().to_dict()
    payload["simulation_physics"] = {
        "ground": {"static_friction": 0.9, "dynamic_friction": 0.8}
    }
    profile = CalibrationProfileV1.from_dict(payload)
    view = FakeView()
    robot = SimpleNamespace(
        root_physx_view=view,
        data=SimpleNamespace(),
        num_instances=1,
        device="cpu",
        joint_names=[],
    )

    summary = apply_profile_to_runtime_env(SimpleNamespace(robot=robot), profile)

    torch.testing.assert_close(view.materials[..., 0], torch.full((1, 3), 0.9))
    torch.testing.assert_close(view.materials[..., 1], torch.full((1, 3), 0.8))
    torch.testing.assert_close(
        view.materials[..., 2], torch.tensor([[0.3, 0.4, 0.5]])
    )
    assert summary["contact_material"] == {
        "friction_combine_mode": "max",
        "robot_shape_count": 3,
        "static_friction": 0.9,
        "dynamic_friction": 0.8,
    }
    # Both pair materials are explicitly 0.9/0.8 and use max-combine, so the
    # effective coefficients are the measured values (never their square).
    assert max(float(view.materials[0, 0, 0]), 0.9) == pytest.approx(0.9)
    assert max(float(view.materials[0, 0, 1]), 0.8) == pytest.approx(0.8)


def test_runtime_profile_resolves_canonical_joint_aliases_to_articulation_indices() -> None:
    import torch

    from tools.sim2real.isaac_profile import apply_profile_to_runtime_env

    payload = _profile().to_dict()
    payload["simulation_physics"] = {
        "joint_friction": {"main_0": 0.4, "abad_1": 0.5},
        "joint_dynamic_friction": {"damper_0": 0.3},
        "joint_viscous_friction": {"main_0": 0.2},
        "passive_spring": {
            "damper_0": {"stiffness": 18.0, "damping": 1.8}
        },
    }
    profile = CalibrationProfileV1.from_dict(payload)
    cfg = _fake_env_cfg()
    runtime_names = [
        "Revolute_8",
        "Revolute_6",
        "Revolute_15",
        "Revolute_5",
    ]
    data = SimpleNamespace(
        joint_friction_coeff=torch.zeros((1, 4)),
        joint_dynamic_friction_coeff=torch.zeros((1, 4)),
        joint_viscous_friction_coeff=torch.zeros((1, 4)),
        joint_stiffness=torch.zeros((1, 4)),
        joint_damping=torch.zeros((1, 4)),
    )

    class FakeRobot:
        num_instances = 1
        device = "cpu"
        joint_names = runtime_names

        def __init__(self) -> None:
            self.data = data

        def write_joint_friction_coefficient_to_sim(self, static, dynamic, viscous):
            self.static = static
            self.dynamic = dynamic
            self.viscous = viscous

        def write_joint_stiffness_to_sim(self, stiffness):
            self.stiffness = stiffness

        def write_joint_damping_to_sim(self, damping):
            self.damping = damping

    robot = FakeRobot()

    summary = apply_profile_to_runtime_env(
        SimpleNamespace(robot=robot, cfg=cfg), profile
    )

    assert robot.static[0, runtime_names.index("Revolute_15")] == pytest.approx(0.4)
    assert robot.static[0, runtime_names.index("Revolute_6")] == pytest.approx(0.5)
    assert robot.dynamic[0, runtime_names.index("Revolute_5")] == pytest.approx(0.3)
    assert robot.viscous[0, runtime_names.index("Revolute_15")] == pytest.approx(0.2)
    assert robot.stiffness[0, runtime_names.index("Revolute_5")] == pytest.approx(18.0)
    assert robot.damping[0, runtime_names.index("Revolute_5")] == pytest.approx(1.8)
    assert summary["friction_joints"] == ["abad_1", "damper_0", "main_0"]
    assert summary["passive_spring_joints"] == ["damper_0"]


def test_runtime_passive_springs_synchronize_per_joint_energy_bookkeeping() -> None:
    import torch

    from tools.sim2real.isaac_profile import apply_profile_to_runtime_env

    payload = _profile().to_dict()
    payload["simulation_physics"] = {
        "passive_spring": {
            "damper_0": {"stiffness": 18.0, "damping": 1.8},
            "damper_5": {"stiffness": 27.0, "damping": 2.7},
        }
    }
    profile = CalibrationProfileV1.from_dict(payload)
    cfg = _fake_env_cfg()
    runtime_names = (
        cfg.main_drive_joint_names
        + cfg.abad_joint_names
        + cfg.damper_joint_names
    )
    data = SimpleNamespace(
        joint_stiffness=torch.full((2, 18), 200.0),
        joint_damping=torch.full((2, 18), 20.0),
    )

    class FakeRobot:
        num_instances = 2
        device = "cpu"
        joint_names = runtime_names

        def __init__(self) -> None:
            self.data = data

        def write_joint_stiffness_to_sim(self, stiffness):
            self.data.joint_stiffness = stiffness.clone()

        def write_joint_damping_to_sim(self, damping):
            self.data.joint_damping = damping.clone()

    robot = FakeRobot()
    unwrapped = SimpleNamespace(
        robot=robot,
        cfg=cfg,
        _spring_k=200.0,
        _spring_d=20.0,
    )

    summary = apply_profile_to_runtime_env(
        SimpleNamespace(unwrapped=unwrapped), profile
    )

    assert unwrapped._spring_k.shape == (2, 6)
    assert unwrapped._spring_d.shape == (2, 6)
    assert unwrapped._spring_k[0].tolist() == pytest.approx(
        [18.0, 200.0, 200.0, 200.0, 200.0, 27.0]
    )
    assert unwrapped._spring_d[1].tolist() == pytest.approx(
        [1.8, 20.0, 20.0, 20.0, 20.0, 2.7]
    )
    assert summary["energy_bookkeeping"]["passive_spring_per_joint"] is True


@pytest.mark.parametrize("backend", ["explicit", "native"])
def test_runtime_passive_spring_profile_is_backend_aware(backend: str) -> None:
    import torch

    from tools.sim2real.isaac_profile import apply_profile_to_runtime_env

    payload = _profile().to_dict()
    payload["simulation_physics"] = {
        "passive_spring": {
            "damper_0": {"stiffness": 18.0, "damping": 1.8},
            "damper_5": {"stiffness": 27.0, "damping": 2.7},
        }
    }
    profile = CalibrationProfileV1.from_dict(payload)
    cfg = _fake_env_cfg()
    runtime_names = (
        cfg.main_drive_joint_names
        + cfg.abad_joint_names
        + cfg.damper_joint_names
    )
    data = SimpleNamespace(
        joint_stiffness=torch.full((2, 18), 99.0),
        joint_damping=torch.full((2, 18), 9.0),
        default_joint_stiffness=torch.full((2, 18), 99.0),
        default_joint_damping=torch.full((2, 18), 9.0),
    )

    class FakeRobot:
        num_instances = 2
        device = "cpu"
        joint_names = runtime_names

        def __init__(self) -> None:
            self.data = data

        def write_joint_stiffness_to_sim(self, stiffness):
            self.data.joint_stiffness = stiffness.clone()

        def write_joint_damping_to_sim(self, damping):
            self.data.joint_damping = damping.clone()

    robot = FakeRobot()
    unwrapped = SimpleNamespace(
        robot=robot,
        cfg=cfg,
        _spring_backend=backend,
        _spring_k=torch.full((2, 6), 190.0),
        _spring_d=torch.full((2, 6), 19.0),
    )

    summary = apply_profile_to_runtime_env(
        SimpleNamespace(unwrapped=unwrapped), profile
    )

    expected_stiffness = torch.tensor(
        [[18.0, 190.0, 190.0, 190.0, 190.0, 27.0]]
    ).expand(2, -1)
    expected_damping = torch.tensor(
        [[1.8, 19.0, 19.0, 19.0, 19.0, 2.7]]
    ).expand(2, -1)
    torch.testing.assert_close(unwrapped._spring_k, expected_stiffness)
    torch.testing.assert_close(unwrapped._spring_d, expected_damping)
    damper_indices = [runtime_names.index(name) for name in cfg.damper_joint_names]
    if backend == "explicit":
        torch.testing.assert_close(
            robot.data.joint_stiffness[:, damper_indices],
            torch.zeros_like(expected_stiffness),
        )
        torch.testing.assert_close(
            robot.data.joint_damping[:, damper_indices],
            torch.zeros_like(expected_damping),
        )
    else:
        torch.testing.assert_close(
            robot.data.joint_stiffness[:, damper_indices], expected_stiffness
        )
        torch.testing.assert_close(
            robot.data.joint_damping[:, damper_indices], expected_damping
        )
    assert summary["spring_backend"] == backend
    assert summary["energy_bookkeeping"]["stiffness"] == pytest.approx(
        expected_stiffness[0].tolist()
    )


def test_profile_application_rejects_duplicate_runtime_joint_mapping() -> None:
    from tools.sim2real.physics_profile import apply_profile_to_config

    cfg = _fake_env_cfg()
    cfg.damper_joint_names[0] = cfg.main_drive_joint_names[0]

    with pytest.raises(ContractError, match="duplicate runtime joint"):
        apply_profile_to_config(cfg, _profile())


def test_runtime_profile_rejects_missing_alias_mapping() -> None:
    import torch

    from tools.sim2real.isaac_profile import apply_profile_to_runtime_env

    payload = _profile().to_dict()
    payload["simulation_physics"] = {"joint_friction": {"main_0": 0.4}}
    profile = CalibrationProfileV1.from_dict(payload)
    robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_friction_coeff=torch.zeros((1, 1)),
            joint_dynamic_friction_coeff=torch.zeros((1, 1)),
            joint_viscous_friction_coeff=torch.zeros((1, 1)),
        ),
        joint_names=["Revolute_15"],
        num_instances=1,
        device="cpu",
    )

    with pytest.raises(ContractError, match="ordered main drive joint names"):
        apply_profile_to_runtime_env(SimpleNamespace(robot=robot), profile)
