from __future__ import annotations

import argparse
import copy
import json
import math
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from .repo_binding import assert_redrhex_module_source, bind_redrhex_source


bind_redrhex_source()

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from RedRhex.tasks.direct.redrhex import redrhex_env_cfg as _redrhex_env_cfg
from RedRhex.tasks.direct.redrhex.redrhex_env_cfg import REDRHEX_CFG, RedrhexEnvCfg
from RedRhex.tasks.direct.redrhex.torsion_spring import (
    mechanical_power,
    potential_energy,
    reorder_joint_parameters,
    restoring_torque,
    unwrap_ambiguity_mask,
    unwrap_continuous_position,
)


assert_redrhex_module_source(_redrhex_env_cfg)

from .characterization import (
    apply_schedule_delay,
    characterization_channel_metadata,
    load_replay_schedule,
    measurement_annotations,
    requires_contact_probe,
    resolve_scenario_steps,
    scenario_schedule,
    validate_foot_contact_probe,
    validate_run_request,
    validate_scenario_mode,
    validate_simulated_experiment,
)
from .contracts import CalibrationProfileV1, ContractError
from .isaac_profile import apply_profile_to_runtime_env
from .physics_profile import (
    apply_abad_target_mapping,
    apply_profile_to_config,
    load_optional_profile,
)
from .runtime_provenance import production_runtime_provenance
from .scenarios import load_scenario
from .traces import sha256_json, write_trace


@configclass
class CharacterizationSceneCfg(InteractiveSceneCfg):
    """One production-asset scene with a real PhysX contact reporter."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75)),
    )
    robot = REDRHEX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    foot_contact_sensor = ContactSensorCfg(
        prim_path=(
            "{ENV_REGEX_NS}/Robot/test_7/"
            "(left_feet_[1-3]|right_feet_[1-3])"
        ),
        update_period=0.0,
        history_length=3,
        track_air_time=True,
    )
    body_contact_sensor = ContactSensorCfg(
        prim_path=(
            "{ENV_REGEX_NS}/Robot/test_7/"
            "(base_link|top_bottom_connector_[1-7]|motor_holder_[1-7]|left_feet_connect_[1-7])"
        ),
        update_period=0.0,
        history_length=3,
        track_air_time=True,
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _to_list(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _to_list(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _to_list(value.tolist())
    if isinstance(value, list):
        return [_to_list(item) for item in value]
    if isinstance(value, tuple):
        return [_to_list(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        if np.isnan(value):
            return "NaN"
        return "Infinity" if value > 0.0 else "-Infinity"
    if isinstance(value, slice):
        return {"start": value.start, "stop": value.stop, "step": value.step}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _collision_geometry(scene: InteractiveScene) -> list[dict[str, Any]]:
    prefix = "/World/envs/env_0/Robot"
    geometries: list[dict[str, Any]] = []
    root = scene.stage.GetPrimAtPath(prefix)
    pending = [root] if root.IsValid() else []
    while pending:
        prim = pending.pop()
        pending.extend(prim.GetFilteredChildren(Usd.TraverseInstanceProxies()))
        path = prim.GetPath().pathString
        has_collision_api = prim.HasAPI(UsdPhysics.CollisionAPI)
        is_geometry = prim.IsA(UsdGeom.Gprim)
        if path.startswith(prefix) and (has_collision_api or is_geometry):
            geometries.append(
                {
                    "prim_path": path,
                    "type_name": prim.GetTypeName(),
                    "has_collision_api": bool(has_collision_api),
                    "is_geometry": bool(is_geometry),
                }
            )
    return sorted(geometries, key=lambda item: item["prim_path"])


def _runtime_joint_geometry(
    robot: Any,
    scene: InteractiveScene,
    env_cfg: RedrhexEnvCfg,
) -> list[dict[str, Any]]:
    prefix = "/World/envs/env_0/Robot"
    axes: dict[str, str] = {}
    root = scene.stage.GetPrimAtPath(prefix)
    pending = [root] if root.IsValid() else []
    while pending:
        prim = pending.pop()
        pending.extend(prim.GetFilteredChildren(Usd.TraverseInstanceProxies()))
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        name = prim.GetName()
        axis = str(UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get()).upper()
        if name in axes and axes[name] != axis:
            raise ContractError(f"runtime joint {name} has conflicting USD axes")
        axes[name] = axis

    runtime_names = list(robot.joint_names)
    groups = (
        ("main", list(env_cfg.main_drive_joint_names)),
        ("abad", list(env_cfg.abad_joint_names)),
        ("damper", list(env_cfg.damper_joint_names)),
    )
    if any(len(names) != 6 or len(set(names)) != 6 for _, names in groups):
        raise ContractError("production joint groups must each contain six unique names")
    limits = robot.data.joint_pos_limits[0]
    records: list[dict[str, Any]] = []
    for group, names in groups:
        for canonical_index, runtime_name in enumerate(names):
            try:
                articulation_index = runtime_names.index(runtime_name)
            except ValueError as exc:
                raise ContractError(
                    f"configured joint {runtime_name} is absent from the articulation"
                ) from exc
            axis = axes.get(runtime_name)
            if axis not in {"X", "Y", "Z"}:
                raise ContractError(
                    f"runtime joint {runtime_name} does not expose a resolved USD axis"
                )
            lower = float(limits[articulation_index, 0].item())
            upper = float(limits[articulation_index, 1].item())
            continuous = (
                not math.isfinite(lower)
                or not math.isfinite(upper)
                or lower <= -1.0e20
                or upper >= 1.0e20
            )
            records.append(
                {
                    "canonical_joint": f"{group}_{canonical_index}",
                    "runtime_joint": runtime_name,
                    "articulation_index": articulation_index,
                    "axis": axis,
                    "range_kind": "continuous" if continuous else "limited",
                    "lower_limit_rad": None if continuous else lower,
                    "upper_limit_rad": None if continuous else upper,
                }
            )
    return records


def _runtime_audit(
    robot: Any,
    foot_contact_sensor: Any,
    body_contact_sensor: Any,
    scene: InteractiveScene,
    env_cfg: RedrhexEnvCfg,
    *,
    mode: str,
    physics_dt: float,
    spring_runtime: SimpleNamespace,
    runtime_profile_application: Mapping[str, Any] | None,
) -> dict[str, Any]:
    masses = robot.root_physx_view.get_masses()
    inertias = robot.root_physx_view.get_inertias()
    coms = robot.root_physx_view.get_coms()
    total_mass = masses[0].sum()
    body_masses = masses[0].to(robot.data.body_com_pos_w.device)
    aggregate_com_world = (
        robot.data.body_com_pos_w[0] * body_masses.unsqueeze(-1)
    ).sum(dim=0) / body_masses.sum()
    aggregate_com_body = quat_apply_inverse(
        robot.data.root_quat_w[0].unsqueeze(0),
        (aggregate_com_world - robot.data.root_pos_w[0]).unsqueeze(0),
    )[0]
    mass_profile_application = (
        runtime_profile_application.get("mass")
        if runtime_profile_application is not None
        else None
    )
    if (
        isinstance(mass_profile_application, Mapping)
        and mass_profile_application.get("mode") == "absolute"
    ):
        # Mass/CoM setters take effect immediately, while Isaac's cached body
        # CoM world-state may not refresh until the next scene update. The
        # application result is read back from PhysX and analytically verified,
        # so use its effective value in this pre-experiment audit.
        aggregate_com_body = torch.as_tensor(
            mass_profile_application["achieved_whole_com_root_m"],
            dtype=aggregate_com_body.dtype,
            device=aggregate_com_body.device,
        )
    actuator_audit: dict[str, Any] = {}
    for name, actuator in robot.actuators.items():
        actuator_audit[name] = {
            "class": type(actuator).__name__,
            "joint_names": list(actuator.joint_names),
            "joint_indices": _to_list(actuator.joint_indices),
            "stiffness": _to_list(actuator.stiffness),
            "damping": _to_list(actuator.damping),
            "armature": _to_list(actuator.armature),
            "friction": _to_list(actuator.friction),
            "effort_limit": _to_list(actuator.effort_limit_sim),
            "velocity_limit": _to_list(actuator.velocity_limit_sim),
        }
    data = robot.data
    joint_properties = {
        name: _to_list(getattr(data, name))
        for name in (
            "joint_pos_limits",
            "joint_stiffness",
            "joint_damping",
            "joint_armature",
            "joint_friction_coeff",
            "joint_dynamic_friction_coeff",
            "joint_viscous_friction_coeff",
            "joint_effort_limits",
            "joint_vel_limits",
        )
    }
    ground = env_cfg.terrain.physics_material
    return {
        "schema_version": 2,
        "mode": mode,
        "physics_dt_s": physics_dt,
        "num_envs": 1,
        "is_fixed_base": bool(robot.is_fixed_base),
        "joint_names": list(robot.joint_names),
        "body_names": list(robot.body_names),
        "body_properties": {
            "mass_kg": _to_list(masses),
            "total_mass_kg": float(total_mass.item()),
            "inertia_kg_m2_matrix": _to_list(inertias),
            "com_pose_xyz_xyzw": _to_list(coms),
            "aggregate_com_body_m": _to_list(aggregate_com_body),
        },
        "mass_profile_application": copy.deepcopy(mass_profile_application),
        "joint_geometry": _runtime_joint_geometry(robot, scene, env_cfg),
        "joint_properties": joint_properties,
        "effort_trace_semantics": (
            "Isaac Lab implicit-PD estimate, not a measured PhysX joint effort; "
            "disabled coast-drive entries are forced to zero."
        ),
        "actuators": actuator_audit,
        "torsion_springs": {
            "backend": spring_runtime._spring_backend,
            "joint_order": [f"damper_{index}" for index in range(6)],
            "runtime_joint_names": [
                robot.joint_names[index] for index in spring_runtime.damper_indices
            ],
            "stiffness_nm_per_rad": _to_list(spring_runtime._spring_k[0]),
            "damping_nm_s_per_rad": _to_list(spring_runtime._spring_d[0]),
            "neutral_angle_rad": _to_list(spring_runtime._spring_rest_pos[0]),
            "locked_joint_names": list(
                getattr(spring_runtime, "fixture_locked_joint_names", [])
            ),
            "fixture_limit_half_width_rad": getattr(
                spring_runtime, "fixture_limit_half_width_rad", None
            ),
            "fixture_selected_limit_rad": getattr(
                spring_runtime, "fixture_selected_limit_rad", None
            ),
            "fixture_selected_range_kind": getattr(
                spring_runtime, "fixture_selected_range_kind", None
            ),
            "fixture_constraint_kind": getattr(
                spring_runtime, "fixture_constraint_kind", None
            ),
        },
        "collision_geometry": _collision_geometry(scene),
        "contact_sensors": {
            "foot": {
                "prim_path": foot_contact_sensor.cfg.prim_path,
                "body_names": list(foot_contact_sensor.body_names),
                "body_count": int(foot_contact_sensor.num_bodies),
            },
            "body": {
                "prim_path": body_contact_sensor.cfg.prim_path,
                "body_names": list(body_contact_sensor.body_names),
                "body_count": int(body_contact_sensor.num_bodies),
            },
        },
        "ground_material": {
            "friction_combine_mode": ground.friction_combine_mode,
            "restitution_combine_mode": ground.restitution_combine_mode,
            "static_friction": float(ground.static_friction),
            "dynamic_friction": float(ground.dynamic_friction),
            "restitution": float(ground.restitution),
        },
    }


def _selected_joint_name(scenario: Any, env_cfg: RedrhexEnvCfg) -> str | None:
    selectors = {
        "main": list(env_cfg.main_drive_joint_names),
        "abad": list(env_cfg.abad_joint_names),
        "damper": list(env_cfg.damper_joint_names),
    }
    if scenario.joint in {"all", "root"} or scenario.joint.startswith("foot_"):
        return None
    try:
        group, raw_index = scenario.joint.rsplit("_", 1)
        return selectors[group][int(raw_index)]
    except (KeyError, ValueError, IndexError) as exc:
        raise ContractError(f"scenario selects unknown simulation joint: {scenario.joint}") from exc


def _resolve_selected_joint(scenario: Any, env_cfg: RedrhexEnvCfg, robot: Any) -> int | None:
    joint_name = _selected_joint_name(scenario, env_cfg)
    if joint_name is None:
        return None
    try:
        return list(robot.joint_names).index(joint_name)
    except ValueError as exc:
        raise ContractError(f"scenario joint is absent from articulation: {joint_name}") from exc


def _author_spring_release_fixture(
    scene: InteractiveScene,
    env_cfg: RedrhexEnvCfg,
    selected_joint_name: str,
) -> None:
    """Lock every non-tested joint while leaving the tested spring continuous."""

    prefix = "/World/envs/env_0/Robot"
    sim_utils.make_uninstanceable(prefix, stage=scene.stage)
    all_joint_names = (
        list(env_cfg.main_drive_joint_names)
        + list(env_cfg.abad_joint_names)
        + list(env_cfg.damper_joint_names)
    )
    if len(all_joint_names) != 18 or len(set(all_joint_names)) != 18:
        raise ContractError("spring fixture requires 18 unique production joints")
    if selected_joint_name not in env_cfg.damper_joint_names:
        raise ContractError("spring fixture must select a damper joint")
    initial_positions = env_cfg.robot_cfg.init_state.joint_pos
    half_width_deg = math.degrees(1.0e-6)
    authored: set[str] = set()
    fixture_scope = "/World/envs/env_0/SpringReleaseFixture"
    UsdGeom.Scope.Define(scene.stage, fixture_scope)
    root = scene.stage.GetPrimAtPath(prefix)
    pending = [root] if root.IsValid() else []
    while pending:
        prim = pending.pop()
        pending.extend(prim.GetFilteredChildren(Usd.TraverseInstanceProxies()))
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        name = prim.GetName()
        if name not in all_joint_names:
            continue
        if name in authored:
            raise ContractError(f"spring fixture found duplicate USD joint {name}")
        if name == selected_joint_name:
            authored.add(name)
            continue
        center_deg = math.degrees(float(initial_positions.get(name, 0.0)))
        lower_deg = center_deg - half_width_deg
        upper_deg = center_deg + half_width_deg
        joint = UsdPhysics.RevoluteJoint(prim)
        joint.CreateLowerLimitAttr().Set(lower_deg)
        joint.CreateUpperLimitAttr().Set(upper_deg)
        source_joint = UsdPhysics.Joint(prim)
        fixed_joint = UsdPhysics.FixedJoint.Define(
            scene.stage, f"{fixture_scope}/{name}_lock"
        )
        fixed_joint.CreateBody0Rel().SetTargets(
            source_joint.GetBody0Rel().GetTargets()
        )
        fixed_joint.CreateBody1Rel().SetTargets(
            source_joint.GetBody1Rel().GetTargets()
        )
        fixed_joint.CreateLocalPos0Attr().Set(
            source_joint.GetLocalPos0Attr().Get()
        )
        axis_name = str(joint.GetAxisAttr().Get()).upper()
        axis = {
            "X": Gf.Vec3d(1.0, 0.0, 0.0),
            "Y": Gf.Vec3d(0.0, 1.0, 0.0),
            "Z": Gf.Vec3d(0.0, 0.0, 1.0),
        }.get(axis_name)
        if axis is None:
            raise ContractError(f"spring fixture joint {name} has invalid axis")
        initial_rotation = Gf.Quatf(Gf.Rotation(axis, center_deg).GetQuat())
        fixed_joint.CreateLocalRot0Attr().Set(
            source_joint.GetLocalRot0Attr().Get() * initial_rotation
        )
        fixed_joint.CreateLocalPos1Attr().Set(
            source_joint.GetLocalPos1Attr().Get()
        )
        fixed_joint.CreateLocalRot1Attr().Set(
            source_joint.GetLocalRot1Attr().Get()
        )
        fixed_joint.CreateExcludeFromArticulationAttr().Set(True)
        authored.add(name)
    if authored != set(all_joint_names):
        missing = ", ".join(sorted(set(all_joint_names) - authored))
        raise ContractError(f"spring fixture could not author USD joints: {missing}")


def _resolve_main_joint_indices(env_cfg: RedrhexEnvCfg, robot: Any) -> list[int]:
    names = list(env_cfg.main_drive_joint_names)
    if len(names) != 6 or len(set(names)) != 6:
        raise ContractError("production configuration must expose six ordered main joints")
    try:
        indices = [list(robot.joint_names).index(name) for name in names]
    except ValueError as exc:
        raise ContractError("replay main joint is absent from the runtime articulation") from exc
    return indices


def _configure_runner_torsion_springs(
    robot: Any,
    env_cfg: RedrhexEnvCfg,
    spring_backend: str,
) -> SimpleNamespace:
    """Configure the raw characterization articulation like the production env."""

    if spring_backend not in {"explicit", "native"}:
        raise ContractError(f"unsupported torsion-spring backend: {spring_backend}")
    damper_names = list(env_cfg.damper_joint_names)
    if len(damper_names) != 6 or len(set(damper_names)) != 6:
        raise ContractError("torsion-spring characterization requires six ordered joints")
    try:
        damper_indices = [list(robot.joint_names).index(name) for name in damper_names]
    except ValueError as exc:
        raise ContractError(
            "torsion-spring joint is absent from the runtime articulation"
        ) from exc

    dtype = robot.data.joint_pos.dtype
    device = robot.data.joint_pos.device
    stiffness = torch.as_tensor(
        env_cfg.spring_stiffness_nm_per_rad, dtype=dtype, device=device
    ).unsqueeze(0)
    damping = torch.as_tensor(
        env_cfg.spring_damping_nm_s_per_rad, dtype=dtype, device=device
    ).unsqueeze(0)
    if tuple(stiffness.shape) != (1, 6) or tuple(damping.shape) != (1, 6):
        raise ContractError("torsion-spring configuration must contain six parameters")
    rest_position = robot.data.default_joint_pos[:, damper_indices].clone()

    joint_stiffness = robot.data.joint_stiffness.clone()
    joint_damping = robot.data.joint_damping.clone()
    joint_stiffness[:, damper_indices] = (
        stiffness if spring_backend == "native" else 0.0
    )
    joint_damping[:, damper_indices] = (
        damping if spring_backend == "native" else 0.0
    )
    robot.write_joint_stiffness_to_sim(joint_stiffness)
    robot.write_joint_damping_to_sim(joint_damping)
    robot.data.default_joint_stiffness = joint_stiffness.clone()
    robot.data.default_joint_damping = joint_damping.clone()
    actuators = getattr(robot, "actuators", {})
    if isinstance(actuators, dict) and "damper" in actuators:
        actuator = actuators["damper"]
        actuators["damper"].stiffness = reorder_joint_parameters(
            joint_stiffness[:, damper_indices],
            damper_names,
            actuator.joint_names,
        ).clone()
        actuators["damper"].damping = reorder_joint_parameters(
            joint_damping[:, damper_indices],
            damper_names,
            actuator.joint_names,
        ).clone()

    zero = torch.zeros_like(rest_position)
    robot.set_joint_position_target(rest_position, joint_ids=damper_indices)
    robot.set_joint_velocity_target(zero, joint_ids=damper_indices)
    robot.set_joint_effort_target(zero, joint_ids=damper_indices)
    return SimpleNamespace(
        robot=robot,
        cfg=env_cfg,
        _spring_backend=spring_backend,
        _spring_k=stiffness.clone(),
        _spring_d=damping.clone(),
        _spring_rest_pos=rest_position,
        _spring_wrapped_pos=robot.data.joint_pos[:, damper_indices].clone(),
        _spring_unwrapped_pos=robot.data.joint_pos[:, damper_indices].clone(),
        _spring_unwrap_velocity=robot.data.joint_vel[:, damper_indices].clone(),
        _spring_unwrap_ambiguous=torch.zeros_like(rest_position, dtype=torch.bool),
        damper_indices=damper_indices,
    )


def _verify_spring_release_fixture(
    robot: Any,
    selected_joint: int,
    spring_runtime: SimpleNamespace,
) -> None:
    """Verify and record the pre-authored isolated release fixture."""

    if selected_joint not in spring_runtime.damper_indices:
        raise ContractError("spring release fixture must select a damper joint")
    locked_joint_ids = [
        index for index in range(len(robot.joint_names)) if index != selected_joint
    ]
    positions = robot.data.default_joint_pos
    half_width_rad = 1.0e-6
    expected_limits = torch.stack(
        (positions - half_width_rad, positions + half_width_rad), dim=-1
    )
    applied_limits = robot.root_physx_view.get_dof_limits().to(
        device=expected_limits.device, dtype=expected_limits.dtype
    )
    if not torch.allclose(
        applied_limits[:, locked_joint_ids],
        expected_limits[:, locked_joint_ids],
        atol=1.0e-5,
        rtol=0.0,
    ):
        maximum_error = float(
            torch.max(
                torch.abs(
                    applied_limits[:, locked_joint_ids]
                    - expected_limits[:, locked_joint_ids]
                )
            ).item()
        )
        raise ContractError(
            "PhysX did not instantiate the isolated spring release fixture "
            f"(maximum limit error {maximum_error:.6g} rad)"
        )
    selected_limits = applied_limits[:, selected_joint]
    selected_continuous = bool(
        torch.all(selected_limits[:, 0] <= -1.0e20)
        and torch.all(selected_limits[:, 1] >= 1.0e20)
    )
    if not selected_continuous:
        raise ContractError(
            "spring release selected joint must remain continuous without hard stops"
        )
    spring_runtime.fixture_locked_joint_ids = locked_joint_ids
    spring_runtime.fixture_locked_joint_names = [
        robot.joint_names[index] for index in locked_joint_ids
    ]
    spring_runtime.fixture_limit_half_width_rad = half_width_rad
    spring_runtime.fixture_selected_limit_rad = [None, None]
    spring_runtime.fixture_selected_range_kind = "continuous"
    spring_runtime.fixture_constraint_kind = "excluded_fixed_joint"


def _update_runner_unwrapped_position(
    spring_runtime: SimpleNamespace,
    physics_dt: float,
) -> torch.Tensor:
    position = spring_runtime.robot.data.joint_pos[:, spring_runtime.damper_indices]
    velocity = spring_runtime.robot.data.joint_vel[:, spring_runtime.damper_indices]
    predicted_delta = 0.5 * (
        spring_runtime._spring_unwrap_velocity + velocity
    ) * float(physics_dt)
    candidate = unwrap_continuous_position(
        position,
        spring_runtime._spring_wrapped_pos,
        spring_runtime._spring_unwrapped_pos,
        predicted_delta=predicted_delta,
    )
    changed = position != spring_runtime._spring_wrapped_pos
    unwrapped = torch.where(
        changed, candidate, spring_runtime._spring_unwrapped_pos
    )
    maximum_velocity = torch.maximum(
        torch.abs(spring_runtime._spring_unwrap_velocity), torch.abs(velocity)
    )
    spring_runtime._spring_unwrap_ambiguous.copy_(
        unwrap_ambiguity_mask(maximum_velocity, float(physics_dt))
    )
    spring_runtime._spring_wrapped_pos.copy_(position)
    spring_runtime._spring_unwrapped_pos.copy_(unwrapped)
    spring_runtime._spring_unwrap_velocity.copy_(velocity)
    return spring_runtime._spring_unwrapped_pos


def _apply_runner_explicit_spring(
    spring_runtime: SimpleNamespace,
    physics_dt: float,
) -> torch.Tensor:
    """Return the shared spring law and apply it only for the explicit backend."""

    robot = spring_runtime.robot
    indices = spring_runtime.damper_indices
    spring_position = _update_runner_unwrapped_position(spring_runtime, physics_dt)
    spring_velocity = robot.data.joint_vel[:, indices]
    torque = restoring_torque(
        spring_position,
        spring_velocity,
        spring_runtime._spring_rest_pos,
        spring_runtime._spring_k,
        spring_runtime._spring_d,
    )
    if spring_runtime._spring_backend == "explicit":
        robot.set_joint_effort_target(torque, joint_ids=indices)
    return torque


def _articulation_kinetic_energy(robot: Any) -> torch.Tensor:
    """Return rigid-body kinetic energy for each fixed-base release instance."""

    linear_velocity = robot.data.body_com_lin_vel_w
    angular_velocity_w = robot.data.body_com_ang_vel_w
    body_quaternion_w = robot.data.body_quat_w
    masses = robot.root_physx_view.get_masses().to(
        device=linear_velocity.device, dtype=linear_velocity.dtype
    )
    inertias = robot.root_physx_view.get_inertias().to(
        device=linear_velocity.device, dtype=linear_velocity.dtype
    )
    inertia_matrix = inertias.reshape(*inertias.shape[:2], 3, 3)
    angular_velocity_body = quat_apply_inverse(
        body_quaternion_w.reshape(-1, 4),
        angular_velocity_w.reshape(-1, 3),
    ).reshape_as(angular_velocity_w)
    translational = 0.5 * masses * torch.square(linear_velocity).sum(dim=-1)
    rotational = 0.5 * torch.einsum(
        "ebi,ebij,ebj->eb",
        angular_velocity_body,
        inertia_matrix,
        angular_velocity_body,
    )
    return (translational + rotational).sum(dim=1)


def _required_aliases(
    scenario: Any,
    traces: dict[str, np.ndarray],
    time_bases: dict[str, str],
    *,
    selected_joint: int | None,
    total_mass_kg: float,
    requested_schedule: tuple[Any, ...],
) -> None:
    steps = traces["sim_time_s"].shape[0]
    sim_time = traces["sim_time_s"]
    selected = selected_joint if selected_joint is not None else 0
    repeat_index, settled = measurement_annotations(requested_schedule)
    aliases: dict[str, np.ndarray] = {
        "command": traces["requested_command"][:, selected],
        "position": traces["joint_position"][:, selected],
        "repeat_index": repeat_index,
        "settled": settled,
        "audit_value": np.full(steps, total_mass_kg, dtype=np.float64),
    }
    for channel in scenario.required_channels:
        if channel in aliases:
            time_name = scenario.time_bases[channel]
            traces[channel] = np.asarray(aliases[channel], dtype=np.float64)
            traces[time_name] = sim_time.copy()
            time_bases[channel] = time_name
        elif channel not in traces:
            raise ContractError(
                f"scenario channel {channel} requires manual measurement and cannot be generated by run-sim"
            )
        elif time_bases.get(channel) != scenario.time_bases[channel]:
            raise ContractError(
                f"scenario channel {channel} requires time base {scenario.time_bases[channel]}"
            )


def _trace_metadata(
    time_bases: Mapping[str, str],
    robot: Any,
    scenario: Any,
    env_cfg: RedrhexEnvCfg,
    profile: CalibrationProfileV1 | None,
    profile_application: Mapping[str, Any] | None,
    audit: Mapping[str, Any],
    *,
    mode: str,
    physics_dt: float,
    spring_backend: str,
    spring_calibration_status: str,
    profile_sha256: str | None,
    seed: int,
    replay_trace_sha256: str | None,
    replay_initial_state: Mapping[str, Any] | None,
    replay_initial_state_sha256: str | None,
) -> dict[str, Any]:
    units, frames = characterization_channel_metadata(scenario, set(time_bases))
    runtime_provenance = production_runtime_provenance()
    return {
        "units": {name: units[name] for name in time_bases},
        "frames": frames,
        "joint_order": list(robot.joint_names),
        "clock": {
            "source": "isaac_physics_step",
            "timestamp_semantics": (
                "channel_time_bases_pre_and_post_step_relative_monotonic"
            ),
            "time_unit": "s",
        },
        "scenario_schema_version": scenario.schema_version,
        "git_sha": runtime_provenance["git_sha"],
        "asset_sha256": runtime_provenance["asset_sha256"],
        "config_sha256": runtime_provenance["config_sha256"],
        "redrhex_module_path": runtime_provenance["redrhex_module_path"],
        "redrhex_module_sha256": runtime_provenance["redrhex_module_sha256"],
        "isaaclab_version": runtime_provenance["isaaclab_version"],
        "isaacsim_version": runtime_provenance["isaacsim_version"],
        "characterization_runner_sha256": runtime_provenance["characterization_runner_sha256"],
        "torsion_spring_model_sha256": runtime_provenance[
            "torsion_spring_model_sha256"
        ],
        "runtime_bundle_sha256": runtime_provenance["runtime_bundle_sha256"],
        "spring_backend": spring_backend,
        "calibration_status": spring_calibration_status,
        "profile_sha256": profile_sha256,
        "seed": seed,
        "calibration_constants": {
            "profile_id": profile.profile_id if profile is not None else None,
            "runtime_audit_sha256": sha256_json(audit),
            "sensor_timing": (
                profile_application["sensor_timing"]
                if profile_application is not None
                else {}
            ),
            "hardware_mapping": profile.hardware_mapping if profile is not None else {},
            "physics_dt_s": physics_dt,
            "seed": seed,
            "spring_backend": spring_backend,
            "spring_calibration_status": spring_calibration_status,
            "mode": mode,
            "fixed_base": bool(audit["is_fixed_base"]),
            "replay_trace_sha256": replay_trace_sha256,
            "replay_initial_state": replay_initial_state,
            "replay_initial_state_sha256": replay_initial_state_sha256,
        },
        "raw_data_sha256": None,
    }


def run_characterization(args: argparse.Namespace) -> dict[str, Any]:
    """Run exactly ``steps`` physics frames and emit one numeric calibration trace."""

    output = Path(args.output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ContractError(f"output already exists: {output}")
    scenario = load_scenario(args.scenario)
    validate_scenario_mode(scenario, args.mode)
    validate_simulated_experiment(scenario)
    physics_hz = int(getattr(args, "physics_hz", 120))
    if physics_hz not in {120, 240}:
        raise ContractError("spring characterization supports 120 or 240 Hz")
    requested_physics_dt = 1.0 / physics_hz
    steps = resolve_scenario_steps(scenario, args.steps, requested_physics_dt)
    request = validate_run_request(
        mode=args.mode,
        steps=steps,
        physics_dt=requested_physics_dt,
        require_contact=args.require_contact,
    )
    physics_dt = request.physics_dt
    profile = load_optional_profile(args.physics_profile)
    replay_trace = getattr(args, "replay_trace", None)

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    env_cfg = RedrhexEnvCfg()
    env_cfg.spring_backend = str(args.spring_backend)
    profile_application = apply_profile_to_config(env_cfg, profile)
    spring_backend = str(env_cfg.spring_backend)
    spring_calibration_status = (
        "calibrated"
        if bool(getattr(env_cfg, "spring_calibrated", False))
        else "uncalibrated"
    )
    profile_sha256 = sha256_json(profile.to_dict()) if profile is not None else None
    env_cfg.robot_cfg.spawn.articulation_props.fix_root_link = request.mode == "fixed-base"

    requested_schedule = scenario_schedule(scenario, request.steps, physics_dt)
    replay = None
    replay_trace_sha256 = None
    replay_initial_state = None
    replay_initial_state_sha256 = None
    if replay_trace is not None:
        replay = load_replay_schedule(
            replay_trace,
            scenario,
            steps=request.steps,
            physics_dt=physics_dt,
        )
        requested_schedule = replay.schedule
        replay_trace_sha256 = replay.trace_sha256
        replay_initial_state = replay.initial_state.to_dict()
        replay_initial_state_sha256 = replay.initial_state_sha256
    delay_steps = (
        int(profile_application["sensor_timing"]["command_delay_steps"])
        if profile_application is not None
        else 0
    )
    applied_schedule = apply_schedule_delay(
        requested_schedule,
        delay_steps=delay_steps,
    )

    sim_cfg = copy.deepcopy(env_cfg.sim)
    sim_cfg.dt = physics_dt
    sim_cfg.render_interval = 1
    sim_cfg.device = str(args.device)
    if scenario.experiment_kind == "spring_release":
        sim_cfg.gravity = (0.0, 0.0, 0.0)
    sim = sim_utils.SimulationContext(sim_cfg)

    scene_cfg = CharacterizationSceneCfg(num_envs=1, env_spacing=env_cfg.scene.env_spacing)
    scene_cfg.robot = copy.deepcopy(env_cfg.robot_cfg).replace(prim_path="{ENV_REGEX_NS}/Robot")
    scene_cfg.ground.spawn.physics_material = copy.deepcopy(env_cfg.terrain.physics_material)
    scene = InteractiveScene(scene_cfg)
    selected_joint_name = _selected_joint_name(scenario, env_cfg)
    if scenario.experiment_kind == "spring_release":
        if selected_joint_name is None:
            raise ContractError("spring_release requires one selected damper joint")
        _author_spring_release_fixture(scene, env_cfg, selected_joint_name)
    sim.reset()

    robot = scene["robot"]
    foot_contact_sensor = scene["foot_contact_sensor"]
    body_contact_sensor = scene["body_contact_sensor"]
    selected_joint = _resolve_selected_joint(scenario, env_cfg, robot)
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    if replay is not None:
        root_state[:, 3:7] = torch.as_tensor(
            replay.initial_state.root_orientation_wxyz,
            dtype=root_state.dtype,
            device=root_state.device,
        ).unsqueeze(0)
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    initial_joint_position = robot.data.default_joint_pos.clone()
    initial_joint_velocity = robot.data.default_joint_vel.clone()
    if replay is not None:
        main_joint_indices = _resolve_main_joint_indices(env_cfg, robot)
        replay_position = torch.as_tensor(
            replay.initial_state.position_rad,
            dtype=initial_joint_position.dtype,
            device=initial_joint_position.device,
        ).unsqueeze(0)
        replay_velocity = torch.as_tensor(
            replay.initial_state.velocity_rad_s,
            dtype=initial_joint_velocity.dtype,
            device=initial_joint_velocity.device,
        ).unsqueeze(0)
        initial_joint_position[:, main_joint_indices] = replay_position
        initial_joint_velocity[:, main_joint_indices] = replay_velocity
    robot.write_joint_state_to_sim(
        initial_joint_position, initial_joint_velocity
    )
    scene.reset()
    spring_runtime = _configure_runner_torsion_springs(
        robot, env_cfg, spring_backend
    )
    runtime_profile_application = apply_profile_to_runtime_env(
        spring_runtime, profile
    )
    if scenario.experiment_kind == "spring_release":
        if selected_joint is None:
            raise ContractError("spring_release requires one selected damper joint")
        _verify_spring_release_fixture(robot, selected_joint, spring_runtime)

    # Record the effective PhysX state before the experiment can alter it.
    audit = _runtime_audit(
        robot,
        foot_contact_sensor,
        body_contact_sensor,
        scene,
        env_cfg,
        mode=request.mode,
        physics_dt=physics_dt,
        spring_runtime=spring_runtime,
        runtime_profile_application=runtime_profile_application,
    )

    default_position = initial_joint_position.clone()
    zero_velocity = torch.zeros_like(robot.data.default_joint_vel)
    original_selected_damping = (
        float(robot.data.joint_damping[0, selected_joint].item())
        if selected_joint is not None
        else 0.0
    )
    original_selected_stiffness = (
        float(robot.data.joint_stiffness[0, selected_joint].item())
        if selected_joint is not None
        else 0.0
    )
    was_enabled = True

    requested_log: list[np.ndarray] = []
    applied_log: list[np.ndarray] = []
    joint_position_log: list[np.ndarray] = []
    joint_velocity_log: list[np.ndarray] = []
    joint_effort_estimate_log: list[np.ndarray] = []
    root_position_log: list[np.ndarray] = []
    root_quaternion_log: list[np.ndarray] = []
    root_linear_velocity_log: list[np.ndarray] = []
    root_angular_velocity_log: list[np.ndarray] = []
    foot_contact_force_log: list[np.ndarray] = []
    body_contact_force_log: list[np.ndarray] = []
    spring_deflection_log: list[np.ndarray] = []
    spring_model_torque_log: list[np.ndarray] = []
    spring_applied_torque_estimate_log: list[np.ndarray] = []
    spring_potential_energy_log: list[np.ndarray] = []
    spring_mechanical_power_log: list[np.ndarray] = []
    spring_passivity_residual_log: list[np.ndarray] = []
    spring_release_start_log: list[float] = []
    spring_fixture_position_error_log: list[float] = []
    spring_fixture_velocity_log: list[float] = []
    spring_unwrap_ambiguous_log: list[float] = []
    initial_spring_energy = potential_energy(
        robot.data.joint_pos[:, spring_runtime.damper_indices],
        spring_runtime._spring_rest_pos,
        spring_runtime._spring_k,
    ).sum(dim=1)
    spring_energy_reference = initial_spring_energy.clone()
    previous_release_key: tuple[int, str, float] | None = None

    for step_index in range(request.steps):
        requested_scheduled = requested_schedule[step_index]
        applied_scheduled = applied_schedule[step_index]
        release_key = (
            int(applied_scheduled.repeat_index),
            str(applied_scheduled.label),
            float(applied_scheduled.value),
        )
        release_start = bool(
            scenario.experiment_kind == "spring_release"
            and release_key != previous_release_key
        )
        if release_start:
            if selected_joint is None:
                raise ContractError("spring_release requires one selected damper joint")
            reset_position = default_position.clone()
            reset_velocity = torch.zeros_like(robot.data.joint_vel)
            spring_slot = spring_runtime.damper_indices.index(selected_joint)
            reset_position[:, selected_joint] = (
                spring_runtime._spring_rest_pos[:, spring_slot]
                + float(applied_scheduled.value)
            )
            reset_velocity[:, selected_joint] = 0.0
            robot.write_joint_state_to_sim(reset_position, reset_velocity)
            reset_spring_position = robot.data.joint_pos[
                :, spring_runtime.damper_indices
            ]
            spring_runtime._spring_wrapped_pos.copy_(reset_spring_position)
            spring_runtime._spring_unwrapped_pos.copy_(reset_spring_position)
            spring_runtime._spring_unwrap_velocity.copy_(
                robot.data.joint_vel[:, spring_runtime.damper_indices]
            )
            spring_runtime._spring_unwrap_ambiguous.zero_()
            spring_energy_reference = potential_energy(
                spring_runtime._spring_unwrapped_pos,
                spring_runtime._spring_rest_pos,
                spring_runtime._spring_k,
            ).sum(dim=1)
            previous_release_key = release_key
        position_target = default_position.clone()
        velocity_target = zero_velocity.clone()
        requested = torch.zeros_like(zero_velocity)
        if selected_joint is not None:
            requested[:, selected_joint] = requested_scheduled.value
            if scenario.experiment_kind == "spring_release":
                pass
            elif scenario.experiment_kind == "abad_static":
                abad_rest_rad = float(default_position[0, selected_joint].item())
                abad_limit_rad = float(env_cfg.abad_pos_limit_rad)
                position_target[:, selected_joint] = apply_abad_target_mapping(
                    applied_scheduled.value,
                    profile,
                    joint=scenario.joint,
                    minimum_rad=abad_rest_rad - abad_limit_rad,
                    maximum_rad=abad_rest_rad + abad_limit_rad,
                )
            else:
                velocity_target[:, selected_joint] = applied_scheduled.value
            if applied_scheduled.actuator_enabled != was_enabled:
                robot.write_joint_stiffness_to_sim(
                    original_selected_stiffness if applied_scheduled.actuator_enabled else 0.0,
                    joint_ids=[selected_joint],
                )
                robot.write_joint_damping_to_sim(
                    original_selected_damping if applied_scheduled.actuator_enabled else 0.0,
                    joint_ids=[selected_joint],
                )
                was_enabled = applied_scheduled.actuator_enabled

        robot.set_joint_position_target(position_target)
        robot.set_joint_velocity_target(velocity_target)
        spring_velocity_pre = robot.data.joint_vel[:, spring_runtime.damper_indices].clone()
        spring_torque_pre = _apply_runner_explicit_spring(
            spring_runtime, physics_dt
        )
        spring_position_pre = spring_runtime._spring_unwrapped_pos.clone()
        spring_energy_pre = potential_energy(
            spring_position_pre,
            spring_runtime._spring_rest_pos,
            spring_runtime._spring_k,
        )
        spring_power_pre = mechanical_power(spring_torque_pre, spring_velocity_pre)
        scene.write_data_to_sim()
        spring_applied_torque_pre = robot.data.applied_torque[
            :, spring_runtime.damper_indices
        ].clone()
        if selected_joint is not None and not applied_scheduled.actuator_enabled:
            spring_slot = spring_runtime.damper_indices.index(selected_joint)
            spring_applied_torque_pre[:, spring_slot] = 0.0
        sim.step(render=not bool(args.headless))
        scene.update(physics_dt)

        spring_position_post = _update_runner_unwrapped_position(
            spring_runtime, physics_dt
        )
        spring_energy_post = potential_energy(
            spring_position_post,
            spring_runtime._spring_rest_pos,
            spring_runtime._spring_k,
        ).sum(dim=1)
        spring_system_energy = spring_energy_post + _articulation_kinetic_energy(robot)
        spring_passivity_residual = (
            spring_system_energy - spring_energy_reference
        )

        applied = position_target if scenario.experiment_kind == "abad_static" else velocity_target
        requested_log.append(requested[0].detach().cpu().numpy().copy())
        applied_log.append(applied[0].detach().cpu().numpy().copy())
        joint_position_log.append(robot.data.joint_pos[0].detach().cpu().numpy().copy())
        joint_velocity_log.append(robot.data.joint_vel[0].detach().cpu().numpy().copy())
        effort_estimate = robot.data.applied_torque[0].detach().cpu().numpy().copy()
        if selected_joint is not None and not applied_scheduled.actuator_enabled:
            effort_estimate[selected_joint] = 0.0
        joint_effort_estimate_log.append(effort_estimate)
        root_position_log.append(robot.data.root_pos_w[0].detach().cpu().numpy().copy())
        root_quaternion_log.append(robot.data.root_quat_w[0].detach().cpu().numpy().copy())
        root_linear_velocity_log.append(robot.data.root_lin_vel_w[0].detach().cpu().numpy().copy())
        root_angular_velocity_log.append(robot.data.root_ang_vel_w[0].detach().cpu().numpy().copy())
        foot_contact_force_log.append(
            foot_contact_sensor.data.net_forces_w[0].detach().cpu().numpy().copy()
        )
        body_contact_force_log.append(
            body_contact_sensor.data.net_forces_w[0].detach().cpu().numpy().copy()
        )
        spring_deflection_log.append(
            (
                spring_position_pre[0] - spring_runtime._spring_rest_pos[0]
            ).detach().cpu().numpy().copy()
        )
        spring_model_torque_log.append(
            spring_torque_pre[0].detach().cpu().numpy().copy()
        )
        spring_applied_torque_estimate_log.append(
            spring_applied_torque_pre[0].detach().cpu().numpy().copy()
        )
        spring_potential_energy_log.append(
            spring_energy_pre[0].detach().cpu().numpy().copy()
        )
        spring_mechanical_power_log.append(
            spring_power_pre[0].detach().cpu().numpy().copy()
        )
        spring_passivity_residual_log.append(
            spring_passivity_residual.detach().cpu().numpy().copy()
        )
        spring_release_start_log.append(1.0 if release_start else 0.0)
        spring_unwrap_ambiguous_log.append(
            float(spring_runtime._spring_unwrap_ambiguous.any().item())
        )
        if scenario.experiment_kind == "spring_release":
            locked_ids = spring_runtime.fixture_locked_joint_ids
            locked_position = robot.data.joint_pos[:, locked_ids]
            locked_default = default_position[:, locked_ids]
            locked_position_error = torch.remainder(
                locked_position - locked_default + math.pi,
                2.0 * math.pi,
            ) - math.pi
            spring_fixture_position_error_log.append(
                float(torch.max(torch.abs(locked_position_error)).item())
            )
            spring_fixture_velocity_log.append(
                float(
                    torch.max(
                        torch.abs(robot.data.joint_vel[:, locked_ids])
                    ).item()
                )
            )
        else:
            spring_fixture_position_error_log.append(0.0)
            spring_fixture_velocity_log.append(0.0)

    sim_time = np.arange(1, request.steps + 1, dtype=np.float64) * physics_dt
    spring_pre_step_time = np.arange(request.steps, dtype=np.float64) * physics_dt
    foot_contact_force = np.asarray(foot_contact_force_log, dtype=np.float64)
    body_contact_force = np.asarray(body_contact_force_log, dtype=np.float64)
    traces: dict[str, np.ndarray] = {
        "sim_time_s": sim_time,
        "spring_pre_step_time_s": spring_pre_step_time,
        "requested_command": np.asarray(requested_log, dtype=np.float64),
        "applied_command": np.asarray(applied_log, dtype=np.float64),
        "joint_position": np.asarray(joint_position_log, dtype=np.float64),
        "joint_velocity": np.asarray(joint_velocity_log, dtype=np.float64),
        "joint_effort_estimate": np.asarray(joint_effort_estimate_log, dtype=np.float64),
        "root_position": np.asarray(root_position_log, dtype=np.float64),
        "root_quaternion": np.asarray(root_quaternion_log, dtype=np.float64),
        "root_linear_velocity": np.asarray(root_linear_velocity_log, dtype=np.float64),
        "root_angular_velocity": np.asarray(root_angular_velocity_log, dtype=np.float64),
        "contact_force_w": foot_contact_force,
        "contact_force_n": np.linalg.norm(foot_contact_force, axis=-1),
        "body_contact_force_w": body_contact_force,
        "body_contact_force_n": np.linalg.norm(body_contact_force, axis=-1),
        "spring_deflection": np.asarray(spring_deflection_log, dtype=np.float64),
        "spring_model_torque": np.asarray(spring_model_torque_log, dtype=np.float64),
        "spring_applied_torque_estimate": np.asarray(
            spring_applied_torque_estimate_log, dtype=np.float64
        ),
        "spring_potential_energy": np.asarray(
            spring_potential_energy_log, dtype=np.float64
        ),
        "spring_mechanical_power": np.asarray(
            spring_mechanical_power_log, dtype=np.float64
        ),
        "spring_passivity_residual": np.asarray(
            spring_passivity_residual_log, dtype=np.float64
        ),
        "spring_release_start": np.asarray(
            spring_release_start_log, dtype=np.float64
        ),
        "spring_fixture_position_error": np.asarray(
            spring_fixture_position_error_log, dtype=np.float64
        ),
        "spring_fixture_velocity": np.asarray(
            spring_fixture_velocity_log, dtype=np.float64
        ),
        "spring_unwrap_ambiguous": np.asarray(
            spring_unwrap_ambiguous_log, dtype=np.float64
        ),
    }
    time_bases = {
        name: "sim_time_s"
        for name in traces
        if name not in {"sim_time_s", "spring_pre_step_time_s"}
    }
    for name in (
        "spring_deflection",
        "spring_model_torque",
        "spring_applied_torque_estimate",
        "spring_potential_energy",
        "spring_mechanical_power",
        "spring_release_start",
    ):
        time_bases[name] = "spring_pre_step_time_s"

    _required_aliases(
        scenario,
        traces,
        time_bases,
        selected_joint=selected_joint,
        total_mass_kg=float(audit["body_properties"]["total_mass_kg"]),
        requested_schedule=requested_schedule,
    )
    contact_required = requires_contact_probe(
        scenario, mode=request.mode, explicit=bool(args.require_contact)
    )
    contact_validation = None
    if contact_required:
        contact_validation = validate_foot_contact_probe(
            list(foot_contact_sensor.body_names),
            traces["contact_force_n"],
            threshold_n=float(args.contact_threshold_n),
        )

    metadata = _trace_metadata(
        time_bases,
        robot,
        scenario,
        env_cfg,
        profile,
        profile_application,
        audit,
        mode=request.mode,
        physics_dt=physics_dt,
        spring_backend=spring_backend,
        spring_calibration_status=spring_calibration_status,
        profile_sha256=profile_sha256,
        seed=int(args.seed),
        replay_trace_sha256=replay_trace_sha256,
        replay_initial_state=replay_initial_state,
        replay_initial_state_sha256=replay_initial_state_sha256,
    )
    manifest = write_trace(
        output,
        traces,
        scenario=scenario,
        source="sim",
        profile=profile,
        metadata=metadata,
        time_bases=time_bases,
    )
    _atomic_json(output / "runtime_audit.json", audit)
    result = {
        "schema_version": 1,
        "scenario_id": scenario.scenario_id,
        "mode": request.mode,
        "steps": request.steps,
        "physics_dt_s": request.physics_dt,
        "trace_sha256": manifest.provenance["trace_sha256"],
        "profile_id": profile.profile_id if profile is not None else None,
        "profile_sha256": profile_sha256,
        "spring_backend": spring_backend,
        "calibration_status": spring_calibration_status,
        "seed": int(args.seed),
        "effective_spring_parameters": audit["torsion_springs"],
        "command_delay_steps": delay_steps,
        "effective_command_delay_s": delay_steps * physics_dt,
        "replay_trace_sha256": replay_trace_sha256,
        "replay_initial_state": replay_initial_state,
        "replay_initial_state_sha256": replay_initial_state_sha256,
        "contact_validation": contact_validation,
        "runtime_audit": "runtime_audit.json",
        "runtime_audit_sha256": sha256_json(audit),
    }
    _atomic_json(output / "results.json", result)
    sim._app_control_on_stop_handle = None
    sim.stop()
    sim.clear()
    sim.clear_all_callbacks()
    sim.clear_instance()
    return result
