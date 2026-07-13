from __future__ import annotations

from typing import Any

import torch

from .contracts import CalibrationProfileV1, ContractError
from .physics_profile import canonical_joint_name_map, corrected_mass_properties


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


def _quat_rotate_wxyz(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    if quaternion.ndim != 2 or quaternion.shape[1] != 4:
        raise ContractError("runtime root_quat_w must have shape (environment, 4)")
    if vector.ndim != 2 or vector.shape != (quaternion.shape[0], 3):
        raise ContractError("runtime vector does not match root quaternion layout")
    vector_part = quaternion[:, 1:]
    uv = torch.linalg.cross(vector_part, vector, dim=-1)
    uuv = torch.linalg.cross(vector_part, uv, dim=-1)
    return vector + 2.0 * (quaternion[:, :1] * uv + uuv)


def _quat_rotate_inverse_wxyz(
    quaternion: torch.Tensor, vector: torch.Tensor
) -> torch.Tensor:
    conjugate = quaternion.clone()
    conjugate[:, 1:] *= -1.0
    return _quat_rotate_wxyz(conjugate, vector)


def _runtime_mass_tensors(
    robot: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    view = robot.root_physx_view
    try:
        runtime_masses = view.get_masses().clone()
        runtime_inertias = view.get_inertias().clone()
        coms = view.get_coms().clone()
    except AttributeError as exc:
        raise ContractError(
            "runtime articulation does not expose complete mass properties"
        ) from exc
    if runtime_masses.ndim != 2 or runtime_masses.shape[1] < 1:
        raise ContractError("runtime masses must have shape (environment, body)")
    if runtime_inertias.shape[:2] != runtime_masses.shape:
        raise ContractError("runtime inertia body layout does not match masses")
    if coms.shape[:2] != runtime_masses.shape or coms.shape[-1] < 3:
        raise ContractError("runtime CoM body layout does not match masses")
    if not all(
        bool(torch.isfinite(value).all())
        for value in (runtime_masses, runtime_inertias, coms)
    ):
        raise ContractError("runtime mass properties must be finite")
    if bool((runtime_masses <= 0.0).any()):
        raise ContractError("runtime body masses must be positive")
    return runtime_masses, runtime_inertias, coms


def _whole_com_root(
    robot: object,
    masses: torch.Tensor,
    *,
    body_com_pos_w: torch.Tensor | None = None,
) -> torch.Tensor:
    data = robot.data
    body_positions = (
        getattr(data, "body_com_pos_w", None)
        if body_com_pos_w is None
        else body_com_pos_w
    )
    root_positions = getattr(data, "root_pos_w", None)
    root_quaternions = getattr(data, "root_quat_w", None)
    if not isinstance(body_positions, torch.Tensor):
        raise ContractError("runtime articulation does not expose body_com_pos_w")
    if not isinstance(root_positions, torch.Tensor):
        raise ContractError("runtime articulation does not expose root_pos_w")
    if not isinstance(root_quaternions, torch.Tensor):
        raise ContractError("runtime articulation does not expose root_quat_w")
    if body_positions.shape != (*masses.shape, 3):
        raise ContractError("runtime body_com_pos_w layout does not match body masses")
    if root_positions.shape != (masses.shape[0], 3):
        raise ContractError("runtime root_pos_w layout does not match environments")
    if root_quaternions.shape != (masses.shape[0], 4):
        raise ContractError("runtime root_quat_w layout does not match environments")
    if not all(
        bool(torch.isfinite(value).all())
        for value in (body_positions, root_positions, root_quaternions)
    ):
        raise ContractError("runtime reference pose must be finite")
    quaternion_norm = torch.linalg.vector_norm(root_quaternions, dim=-1)
    if not torch.allclose(
        quaternion_norm,
        torch.ones_like(quaternion_norm),
        atol=1.0e-5,
        rtol=0.0,
    ):
        raise ContractError("runtime root orientation is not normalized")
    state_masses = masses.to(device=body_positions.device, dtype=body_positions.dtype)
    total_mass = state_masses.sum(dim=1)
    aggregate_world = (
        body_positions * state_masses.unsqueeze(-1)
    ).sum(dim=1) / total_mass.unsqueeze(-1)
    return _quat_rotate_inverse_wxyz(
        root_quaternions,
        aggregate_world - root_positions,
    )


def _verify_absolute_reference_pose(
    robot: object,
    correction: dict[str, Any],
    joint_aliases: dict[str, str],
) -> dict[str, Any]:
    joint_position = getattr(robot.data, "joint_pos", None)
    root_quaternion = getattr(robot.data, "root_quat_w", None)
    if not isinstance(joint_position, torch.Tensor) or joint_position.ndim != 2:
        raise ContractError("runtime articulation does not expose a joint pose matrix")
    if not isinstance(root_quaternion, torch.Tensor):
        raise ContractError("runtime articulation does not expose root orientation")
    joint_lookup = {name: index for index, name in enumerate(robot.joint_names)}
    canonical_names = sorted(correction["reference_joint_position_rad"])
    try:
        runtime_indices = [joint_lookup[joint_aliases[name]] for name in canonical_names]
    except KeyError as exc:
        raise ContractError(
            "absolute mass reference maps to a joint absent from the articulation"
        ) from exc
    actual_joint_position = joint_position[:, runtime_indices]
    expected_joint_position = torch.as_tensor(
        [correction["reference_joint_position_rad"][name] for name in canonical_names],
        dtype=actual_joint_position.dtype,
        device=actual_joint_position.device,
    ).unsqueeze(0)
    if not torch.allclose(
        actual_joint_position,
        expected_joint_position.expand_as(actual_joint_position),
        atol=1.0e-4,
        rtol=0.0,
    ):
        max_error = float(
            torch.max(torch.abs(actual_joint_position - expected_joint_position)).item()
        )
        raise ContractError(
            "runtime joint pose does not match absolute mass reference "
            f"(maximum error {max_error:.6g} rad)"
        )

    xyzw = correction["reference_root_orientation_xyzw"]
    expected_root = torch.as_tensor(
        [xyzw[3], xyzw[0], xyzw[1], xyzw[2]],
        dtype=root_quaternion.dtype,
        device=root_quaternion.device,
    ).unsqueeze(0)
    normalized_root = root_quaternion / torch.linalg.vector_norm(
        root_quaternion, dim=-1, keepdim=True
    )
    orientation_dot = torch.abs(
        (normalized_root * expected_root).sum(dim=-1)
    ).clamp(max=1.0)
    orientation_error = 2.0 * torch.acos(orientation_dot)
    if bool((orientation_error > 1.0e-4).any()):
        raise ContractError(
            "runtime root orientation does not match absolute mass reference "
            f"(maximum error {float(orientation_error.max().item()):.6g} rad)"
        )
    return {
        "joint_position_rad": {
            name: float(expected_joint_position[0, index].item())
            for index, name in enumerate(canonical_names)
        },
        "root_orientation_xyzw": [float(value) for value in xyzw],
    }


def _apply_absolute_mass(
    robot: object,
    correction: dict[str, Any],
    joint_aliases: dict[str, str],
) -> dict[str, Any]:
    view = robot.root_physx_view
    masses, inertias, local_coms = _runtime_mass_tensors(robot)
    reference_pose = _verify_absolute_reference_pose(robot, correction, joint_aliases)
    reference_whole_com = _whole_com_root(robot, masses)
    reference_total = masses.sum(dim=1)
    target_total = float(correction["target_total_mass_kg"])
    target_xy = torch.as_tensor(
        correction["reference_planar_com_xy_m"],
        dtype=reference_whole_com.dtype,
        device=reference_whole_com.device,
    )
    scale = target_total / reference_total
    new_masses = masses * scale.unsqueeze(-1)
    new_inertias = inertias * scale.reshape(-1, 1, 1)

    planar_delta = target_xy.unsqueeze(0) - reference_whole_com[:, :2]
    root_mass = new_masses[:, 0]
    if bool((root_mass <= 0.0).any()):
        raise ContractError("absolute mass correction requires a positive root mass")
    root_com_delta = torch.zeros(
        (masses.shape[0], 3), dtype=local_coms.dtype, device=local_coms.device
    )
    root_com_delta[:, :2] = planar_delta.to(local_coms) * (
        target_total / root_mass
    ).unsqueeze(-1)
    new_coms = local_coms.clone()
    new_coms[:, 0, :3] += root_com_delta

    env_ids = _indices(robot)
    view.set_masses(new_masses, env_ids)
    view.set_inertias(new_inertias, env_ids)
    view.set_coms(new_coms, env_ids)
    applied_masses, applied_inertias, applied_coms = _runtime_mass_tensors(robot)
    if not torch.allclose(applied_masses, new_masses, atol=1.0e-6, rtol=1.0e-6):
        raise ContractError("runtime rejected absolute body mass update")
    if not torch.allclose(applied_inertias, new_inertias, atol=1.0e-6, rtol=1.0e-6):
        raise ContractError("runtime rejected uniformly scaled body inertias")
    if not torch.allclose(applied_coms, new_coms, atol=1.0e-6, rtol=1.0e-6):
        raise ContractError("runtime rejected root local CoM update")

    body_positions = robot.data.body_com_pos_w.clone()
    actual_root_delta = applied_coms[:, 0, :3] - local_coms[:, 0, :3]
    root_quaternion = robot.data.root_quat_w
    body_positions[:, 0, :] += _quat_rotate_wxyz(
        root_quaternion,
        actual_root_delta.to(
            device=root_quaternion.device, dtype=root_quaternion.dtype
        ),
    )
    achieved_whole_com = _whole_com_root(
        robot, applied_masses, body_com_pos_w=body_positions
    )
    achieved_total = applied_masses.sum(dim=1)
    if not torch.allclose(
        achieved_total,
        torch.full_like(achieved_total, target_total),
        atol=1.0e-5,
        rtol=1.0e-6,
    ):
        raise ContractError("absolute mass correction did not achieve target total mass")
    if not torch.allclose(
        achieved_whole_com[:, :2],
        target_xy.unsqueeze(0).expand_as(achieved_whole_com[:, :2]),
        atol=1.0e-5,
        rtol=0.0,
    ):
        raise ContractError("absolute mass correction did not achieve target planar CoM")

    robot.data.default_mass = applied_masses.to(robot.device).clone()
    robot.data.default_inertia = applied_inertias.to(robot.device).clone()
    return {
        "mode": "absolute",
        "uniform_scale": float(scale[0].item()),
        "reference_total_mass_kg": float(reference_total[0].item()),
        "reference_whole_com_root_m": reference_whole_com[0].detach().cpu().tolist(),
        "reference_pose": reference_pose,
        "target_total_mass_kg": target_total,
        "target_planar_com_xy_m": [float(value) for value in target_xy.tolist()],
        "achieved_total_mass_kg": float(achieved_total[0].item()),
        "achieved_whole_com_root_m": achieved_whole_com[0].detach().cpu().tolist(),
        "root_com_xyz_m": applied_coms[0, 0, :3].detach().cpu().tolist(),
        "total_mass_kg": float(achieved_total[0].item()),
    }


def _apply_legacy_mass(robot: object, correction: dict[str, Any]) -> dict[str, Any]:
    if not correction:
        raise ContractError("legacy mass correction cannot be empty")
    view = robot.root_physx_view
    runtime_masses, runtime_inertias, coms = _runtime_mass_tensors(robot)
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
        "mode": "legacy",
        "total_mass_kg": float(new_masses[0].sum().item()),
        "root_com_xyz_m": new_coms[0, 0, :3].detach().cpu().tolist(),
    }


def _apply_mass(
    robot: object,
    correction: dict[str, Any],
    joint_aliases: dict[str, str],
) -> dict[str, Any] | None:
    if not correction:
        return None
    if "target_total_mass_kg" in correction:
        return _apply_absolute_mass(robot, correction, joint_aliases)
    return _apply_legacy_mass(robot, correction)


def _joint_tensor(robot: object, attribute: str) -> torch.Tensor:
    value = getattr(robot.data, attribute, None)
    if value is None:
        raise ContractError(f"runtime articulation does not expose {attribute}")
    return value.clone()


def _assign_named(
    values: torch.Tensor,
    robot: object,
    overrides: dict[str, Any],
    joint_aliases: dict[str, str],
) -> None:
    joint_lookup = {name: index for index, name in enumerate(robot.joint_names)}
    for alias, value in overrides.items():
        name = joint_aliases[alias]
        if name not in joint_lookup:
            raise ContractError(
                f"canonical joint {alias} maps to absent runtime joint: {name}"
            )
        values[:, joint_lookup[name]] = float(value)


def _apply_joint_friction(
    robot: object,
    physics: dict[str, Any],
    joint_aliases: dict[str, str],
) -> list[str]:
    static_values = physics.get("joint_friction", {})
    dynamic_values = physics.get("joint_dynamic_friction", {})
    viscous_values = physics.get("joint_viscous_friction", {})
    if not (static_values or dynamic_values or viscous_values):
        return []
    static = _joint_tensor(robot, "joint_friction_coeff")
    dynamic = _joint_tensor(robot, "joint_dynamic_friction_coeff")
    viscous = _joint_tensor(robot, "joint_viscous_friction_coeff")
    _assign_named(static, robot, static_values, joint_aliases)
    _assign_named(dynamic, robot, dynamic_values, joint_aliases)
    _assign_named(viscous, robot, viscous_values, joint_aliases)
    robot.write_joint_friction_coefficient_to_sim(static, dynamic, viscous)
    return sorted(set(static_values) | set(dynamic_values) | set(viscous_values))


def _apply_passive_springs(
    robot: object,
    springs: dict[str, Any],
    joint_aliases: dict[str, str],
) -> list[str]:
    if not springs:
        return []
    stiffness = _joint_tensor(robot, "joint_stiffness")
    damping = _joint_tensor(robot, "joint_damping")
    _assign_named(
        stiffness,
        robot,
        {name: spring["stiffness"] for name, spring in springs.items()},
        joint_aliases,
    )
    _assign_named(
        damping,
        robot,
        {name: spring["damping"] for name, spring in springs.items() if "damping" in spring},
        joint_aliases,
    )
    robot.write_joint_stiffness_to_sim(stiffness)
    robot.write_joint_damping_to_sim(damping)
    robot.data.default_joint_stiffness = stiffness.clone()
    robot.data.default_joint_damping = damping.clone()
    return sorted(springs)


def _synchronize_spring_bookkeeping(
    unwrapped: object,
    robot: object,
    joint_aliases: dict[str, str],
    applied_joints: list[str],
) -> dict[str, Any] | None:
    """Make per-leg energy diagnostics use the same spring values as PhysX."""

    if not applied_joints or not any(
        hasattr(unwrapped, field) for field in ("_spring_k", "_spring_d")
    ):
        return None
    runtime_lookup = {name: index for index, name in enumerate(robot.joint_names)}
    try:
        indices = [
            runtime_lookup[joint_aliases[f"damper_{index}"]]
            for index in range(6)
        ]
    except KeyError as exc:
        raise ContractError(
            "spring energy bookkeeping requires all six runtime damper joints"
        ) from exc
    stiffness = _joint_tensor(robot, "default_joint_stiffness")[:, indices]
    damping = _joint_tensor(robot, "default_joint_damping")[:, indices]
    if hasattr(unwrapped, "_spring_k"):
        unwrapped._spring_k = stiffness.clone()
    if hasattr(unwrapped, "_spring_d"):
        unwrapped._spring_d = damping.clone()
    return {
        "passive_spring_per_joint": True,
        "joint_order": [f"damper_{index}" for index in range(6)],
        "stiffness": stiffness[0].detach().cpu().tolist(),
        "damping": damping[0].detach().cpu().tolist(),
    }


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
    has_per_joint_physics = any(
        physics.get(section)
        for section in (
            "joint_friction",
            "joint_dynamic_friction",
            "joint_viscous_friction",
            "passive_spring",
        )
    )
    mass = physics.get("mass", {})
    needs_joint_aliases = has_per_joint_physics or "target_total_mass_kg" in mass
    joint_aliases = canonical_joint_name_map(
        getattr(unwrapped, "cfg", None)
    ) if needs_joint_aliases else {}
    mass_summary = _apply_mass(robot, mass, joint_aliases)
    if mass_summary is not None:
        # RedrhexEnv's physical domain randomization restores this baseline on
        # later resets. Replace it with the calibrated baseline so DR composes
        # around the selected profile instead of erasing it.
        if hasattr(unwrapped, "_default_body_masses"):
            unwrapped._default_body_masses = robot.root_physx_view.get_masses().clone()
        if hasattr(unwrapped, "_robot_mass"):
            unwrapped._robot_mass = float(mass_summary["total_mass_kg"])
    passive_spring_joints = _apply_passive_springs(
        robot, physics.get("passive_spring", {}), joint_aliases
    )
    return {
        "schema_version": 1,
        "profile_id": profile.profile_id,
        "mass": mass_summary,
        "contact_material": _apply_contact_material(
            robot, physics.get("ground", {})
        ),
        "friction_joints": _apply_joint_friction(robot, physics, joint_aliases),
        "passive_spring_joints": passive_spring_joints,
        "energy_bookkeeping": _synchronize_spring_bookkeeping(
            unwrapped, robot, joint_aliases, passive_spring_joints
        ),
    }
