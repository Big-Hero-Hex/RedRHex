from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from pxr import Usd, UsdGeom, UsdPhysics

from RedRhex.tasks.direct.redrhex.redrhex_env_cfg import REDRHEX_CFG, RedrhexEnvCfg

from .characterization import (
    PHYSICS_DT,
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
from .scenarios import load_scenario
from .traces import sha256_file, write_trace


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


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


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


def _runtime_audit(
    robot: Any,
    foot_contact_sensor: Any,
    body_contact_sensor: Any,
    scene: InteractiveScene,
    env_cfg: RedrhexEnvCfg,
    *,
    mode: str,
) -> dict[str, Any]:
    masses = robot.root_physx_view.get_masses()
    inertias = robot.root_physx_view.get_inertias()
    coms = robot.root_physx_view.get_coms()
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
        "schema_version": 1,
        "mode": mode,
        "physics_dt_s": PHYSICS_DT,
        "num_envs": 1,
        "is_fixed_base": bool(robot.is_fixed_base),
        "joint_names": list(robot.joint_names),
        "body_names": list(robot.body_names),
        "body_properties": {
            "mass_kg": _to_list(masses),
            "total_mass_kg": float(masses[0].sum().item()),
            "inertia_kg_m2_matrix": _to_list(inertias),
            "com_pose_xyz_xyzw": _to_list(coms),
        },
        "joint_properties": joint_properties,
        "effort_trace_semantics": (
            "Isaac Lab implicit-PD estimate, not a measured PhysX joint effort; "
            "disabled coast-drive entries are forced to zero."
        ),
        "actuators": actuator_audit,
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


def _resolve_selected_joint(scenario: Any, env_cfg: RedrhexEnvCfg, robot: Any) -> int | None:
    selectors = {
        "main": list(env_cfg.main_drive_joint_names),
        "abad": list(env_cfg.abad_joint_names),
        "damper": list(env_cfg.damper_joint_names),
    }
    if scenario.joint in {"all", "root"} or scenario.joint.startswith("foot_"):
        return None
    try:
        group, raw_index = scenario.joint.rsplit("_", 1)
        joint_name = selectors[group][int(raw_index)]
        return list(robot.joint_names).index(joint_name)
    except (KeyError, ValueError, IndexError) as exc:
        raise ContractError(f"scenario selects unknown simulation joint: {scenario.joint}") from exc


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
        if channel not in aliases:
            raise ContractError(
                f"scenario channel {channel} requires manual measurement and cannot be generated by run-sim"
            )
        time_name = scenario.time_bases[channel]
        traces[channel] = np.asarray(aliases[channel], dtype=np.float64)
        traces[time_name] = sim_time.copy()
        time_bases[channel] = time_name


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
    replay_trace_sha256: str | None,
    replay_initial_state: Mapping[str, Any] | None,
    replay_initial_state_sha256: str | None,
) -> dict[str, Any]:
    units, frames = characterization_channel_metadata(scenario, set(time_bases))

    asset_path = Path(env_cfg.robot_cfg.spawn.usd_path)
    cfg_path = Path(inspect.getsourcefile(RedrhexEnvCfg) or "")
    return {
        "units": {name: units[name] for name in time_bases},
        "frames": frames,
        "joint_order": list(robot.joint_names),
        "clock": {
            "source": "isaac_physics_step",
            "timestamp_semantics": "post_step_relative_monotonic",
            "time_unit": "s",
        },
        "scenario_schema_version": scenario.schema_version,
        "git_sha": _git_sha(),
        "asset_sha256": sha256_file(asset_path),
        "config_sha256": sha256_file(cfg_path),
        "calibration_constants": {
            "profile_id": profile.profile_id if profile is not None else None,
            "sensor_timing": (
                profile_application["sensor_timing"]
                if profile_application is not None
                else {}
            ),
            "hardware_mapping": profile.hardware_mapping if profile is not None else {},
            "physics_dt_s": PHYSICS_DT,
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
    steps = resolve_scenario_steps(scenario, args.steps, PHYSICS_DT)
    request = validate_run_request(
        mode=args.mode,
        steps=steps,
        physics_dt=PHYSICS_DT,
        require_contact=args.require_contact,
    )
    physics_dt = request.physics_dt
    profile = load_optional_profile(args.physics_profile)
    replay_trace = getattr(args, "replay_trace", None)

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    env_cfg = RedrhexEnvCfg()
    profile_application = apply_profile_to_config(env_cfg, profile)
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
    sim = sim_utils.SimulationContext(sim_cfg)

    scene_cfg = CharacterizationSceneCfg(num_envs=1, env_spacing=env_cfg.scene.env_spacing)
    scene_cfg.robot = copy.deepcopy(env_cfg.robot_cfg).replace(prim_path="{ENV_REGEX_NS}/Robot")
    scene_cfg.ground.spawn.physics_material = copy.deepcopy(env_cfg.terrain.physics_material)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot = scene["robot"]
    foot_contact_sensor = scene["foot_contact_sensor"]
    body_contact_sensor = scene["body_contact_sensor"]
    selected_joint = _resolve_selected_joint(scenario, env_cfg, robot)
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    initial_joint_position = robot.data.default_joint_pos.clone()
    if replay is not None:
        if selected_joint is None:
            raise ContractError("replay initial state requires one selected joint")
        initial_joint_position[:, selected_joint] = replay.initial_state.position_rad
    robot.write_joint_state_to_sim(
        initial_joint_position, robot.data.default_joint_vel.clone()
    )
    scene.reset()
    apply_profile_to_runtime_env(SimpleNamespace(robot=robot), profile)

    # Record the effective PhysX state before the experiment can alter it.
    audit = _runtime_audit(
        robot,
        foot_contact_sensor,
        body_contact_sensor,
        scene,
        env_cfg,
        mode=request.mode,
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

    for step_index in range(request.steps):
        requested_scheduled = requested_schedule[step_index]
        applied_scheduled = applied_schedule[step_index]
        position_target = default_position.clone()
        velocity_target = zero_velocity.clone()
        requested = torch.zeros_like(zero_velocity)
        if selected_joint is not None:
            requested[:, selected_joint] = requested_scheduled.value
            if scenario.experiment_kind == "abad_static":
                position_target[:, selected_joint] = apply_abad_target_mapping(
                    applied_scheduled.value,
                    profile,
                    joint=scenario.joint,
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
        scene.write_data_to_sim()
        sim.step(render=not bool(args.headless))
        scene.update(physics_dt)

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

    sim_time = np.arange(1, request.steps + 1, dtype=np.float64) * physics_dt
    foot_contact_force = np.asarray(foot_contact_force_log, dtype=np.float64)
    body_contact_force = np.asarray(body_contact_force_log, dtype=np.float64)
    traces: dict[str, np.ndarray] = {
        "sim_time_s": sim_time,
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
    }
    time_bases = {
        name: "sim_time_s"
        for name in traces
        if name != "sim_time_s"
    }

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
        "command_delay_steps": delay_steps,
        "effective_command_delay_s": delay_steps * physics_dt,
        "replay_trace_sha256": replay_trace_sha256,
        "replay_initial_state": replay_initial_state,
        "replay_initial_state_sha256": replay_initial_state_sha256,
        "contact_validation": contact_validation,
        "runtime_audit": "runtime_audit.json",
    }
    _atomic_json(output / "results.json", result)
    sim._app_control_on_stop_handle = None
    sim.stop()
    sim.clear()
    sim.clear_all_callbacks()
    sim.clear_instance()
    return result
