from __future__ import annotations

from typing import Any

import torch

from .contracts import CalibrationProfileV1, ContractError
from .physics_profile import corrected_mass_properties


def runtime_robot(env: object) -> object:
    unwrapped = getattr(env, "unwrapped", env)
    robot = getattr(unwrapped, "robot", None)
    if robot is not None:
        return robot
    scene = getattr(unwrapped, "scene", None)
    if scene is None:
        raise ContractError("runtime environment has no robot or scene")
    articulations = getattr(scene, "articulations", None)
    if isinstance(articulations, dict) and "robot" in articulations:
        return articulations["robot"]
    try:
        return scene["robot"]
    except (KeyError, TypeError) as exc:
        raise ContractError("runtime scene has no robot articulation") from exc


def _indices(robot: object) -> torch.Tensor:
    count = int(getattr(robot, "num_instances"))
    return torch.arange(count, dtype=torch.int64, device="cpu")


def _apply_mass(robot: object, correction: dict[str, Any]) -> dict[str, Any] | None:
    if not correction:
        return None
    view = robot.root_physx_view
    runtime_masses = view.get_masses().clone()
    runtime_inertias = view.get_inertias().clone()
    coms = view.get_coms().clone()
    default_masses = getattr(robot.data, "default_mass", None)
    default_inertias = getattr(robot.data, "default_inertia", None)
    masses = (
        default_masses.to(device=runtime_masses.device, dtype=runtime_masses.dtype).clone()
        if isinstance(default_masses, torch.Tensor) and default_masses.shape == runtime_masses.shape
        else runtime_masses
    )
    inertias = (
        default_inertias.to(device=runtime_inertias.device, dtype=runtime_inertias.dtype).clone()
        if isinstance(default_inertias, torch.Tensor) and default_inertias.shape == runtime_inertias.shape
        else runtime_inertias
    )
    corrected = corrected_mass_properties(
        masses.detach().cpu().numpy(),
        inertias.detach().cpu().numpy(),
        coms.detach().cpu().numpy(),
        correction,
    )
    new_masses = torch.as_tensor(corrected.masses, dtype=masses.dtype, device=masses.device)
    new_inertias = torch.as_tensor(corrected.inertias, dtype=inertias.dtype, device=inertias.device)
    new_coms = torch.as_tensor(corrected.coms, dtype=coms.dtype, device=coms.device)
    env_ids = _indices(robot)
    view.set_masses(new_masses, env_ids)
    view.set_inertias(new_inertias, env_ids)
    view.set_coms(new_coms, env_ids)
    robot.data.default_mass = new_masses.to(robot.device).clone()
    robot.data.default_inertia = new_inertias.to(robot.device).clone()
    return {
        "total_mass_kg": float(new_masses[0].sum().item()),
        "root_com_xyz_m": new_coms[0, 0, :3].detach().cpu().tolist(),
    }


def _joint_tensor(robot: object, attribute: str) -> torch.Tensor:
    value = getattr(robot.data, attribute, None)
    if value is None:
        raise ContractError(f"runtime articulation does not expose {attribute}")
    return value.clone()


def _assign_named(values: torch.Tensor, robot: object, overrides: dict[str, Any]) -> None:
    joint_lookup = {name: index for index, name in enumerate(robot.joint_names)}
    for name, value in overrides.items():
        if name not in joint_lookup:
            raise ContractError(f"profile names unknown runtime joint: {name}")
        values[:, joint_lookup[name]] = float(value)


def _apply_joint_friction(robot: object, physics: dict[str, Any]) -> list[str]:
    static_values = physics.get("joint_friction", {})
    dynamic_values = physics.get("joint_dynamic_friction", {})
    viscous_values = physics.get("joint_viscous_friction", {})
    if not (static_values or dynamic_values or viscous_values):
        return []
    static = _joint_tensor(robot, "joint_friction_coeff")
    dynamic = _joint_tensor(robot, "joint_dynamic_friction_coeff")
    viscous = _joint_tensor(robot, "joint_viscous_friction_coeff")
    _assign_named(static, robot, static_values)
    _assign_named(dynamic, robot, dynamic_values)
    _assign_named(viscous, robot, viscous_values)
    robot.write_joint_friction_coefficient_to_sim(static, dynamic, viscous)
    return sorted(set(static_values) | set(dynamic_values) | set(viscous_values))


def _apply_passive_springs(robot: object, springs: dict[str, Any]) -> list[str]:
    if not springs:
        return []
    stiffness = _joint_tensor(robot, "joint_stiffness")
    damping = _joint_tensor(robot, "joint_damping")
    _assign_named(
        stiffness,
        robot,
        {name: spring["stiffness"] for name, spring in springs.items()},
    )
    _assign_named(
        damping,
        robot,
        {name: spring["damping"] for name, spring in springs.items() if "damping" in spring},
    )
    robot.write_joint_stiffness_to_sim(stiffness)
    robot.write_joint_damping_to_sim(damping)
    robot.data.default_joint_stiffness = stiffness.clone()
    robot.data.default_joint_damping = damping.clone()
    return sorted(springs)


def _apply_contact_material(
    robot: object,
    ground: dict[str, Any],
) -> dict[str, Any] | None:
    friction = {
        name: float(ground[name])
        for name in ("static_friction", "dynamic_friction")
        if name in ground
    }
    if not friction:
        return None
    view = robot.root_physx_view
    try:
        materials = view.get_material_properties().clone()
    except AttributeError as exc:
        raise ContractError(
            "runtime articulation does not expose collision material properties"
        ) from exc
    if materials.ndim != 3 or materials.shape[0] != int(robot.num_instances) or materials.shape[2] < 3:
        raise ContractError(
            "runtime collision materials must have shape (environment, shape, 3)"
        )
    if "static_friction" in friction:
        materials[..., 0] = friction["static_friction"]
    if "dynamic_friction" in friction:
        materials[..., 1] = friction["dynamic_friction"]
    view.set_material_properties(materials, _indices(robot))
    return {
        "friction_combine_mode": "max",
        "robot_shape_count": int(materials.shape[1]),
        **friction,
    }


def apply_profile_to_runtime_env(
    env: object,
    profile: CalibrationProfileV1 | None,
) -> dict[str, Any] | None:
    """Apply runtime-only properties after an explicitly profiled env is created."""

    if profile is None:
        return None
    profile = profile.validate()
    unwrapped = getattr(env, "unwrapped", env)
    robot = runtime_robot(env)
    physics = profile.simulation_physics
    mass_summary = _apply_mass(robot, physics.get("mass", {}))
    if mass_summary is not None:
        # RedrhexEnv's physical domain randomization restores this baseline on
        # later resets. Replace it with the calibrated baseline so DR composes
        # around the selected profile instead of erasing it.
        if hasattr(unwrapped, "_default_body_masses"):
            unwrapped._default_body_masses = robot.root_physx_view.get_masses().clone()
        if hasattr(unwrapped, "_robot_mass"):
            unwrapped._robot_mass = float(mass_summary["total_mass_kg"])
    return {
        "schema_version": 1,
        "profile_id": profile.profile_id,
        "mass": mass_summary,
        "contact_material": _apply_contact_material(
            robot, physics.get("ground", {})
        ),
        "friction_joints": _apply_joint_friction(robot, physics),
        "passive_spring_joints": _apply_passive_springs(
            robot, physics.get("passive_spring", {})
        ),
    }
