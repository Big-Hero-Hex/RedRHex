from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import CalibrationProfileV1, ContractError, load_profile


@dataclass(frozen=True)
class CorrectedMassProperties:
    masses: np.ndarray
    inertias: np.ndarray
    coms: np.ndarray


def load_optional_profile(path: str | Path | None) -> CalibrationProfileV1 | None:
    return None if path is None else load_profile(path)


def _robot_cfg(env_cfg: object) -> object:
    robot = getattr(env_cfg, "robot_cfg", None)
    if robot is None:
        scene = getattr(env_cfg, "scene", None)
        robot = getattr(scene, "robot", None) if scene is not None else None
    if robot is None:
        raise ContractError("environment configuration has no robot_cfg or scene.robot")
    return robot


def _set_fields(target: object, values: Mapping[str, Any], field_map: Mapping[str, str] | None = None) -> None:
    mapping = dict(field_map or {})
    for name, value in values.items():
        attribute = mapping.get(name, name)
        if not hasattr(target, attribute):
            raise ContractError(f"configuration target does not support {attribute}")
        setattr(target, attribute, copy.deepcopy(value))


def _apply_sensor_timing(
    env_cfg: object,
    timing: Mapping[str, float],
) -> dict[str, float | int]:
    supported = {"aggregate_command_delay_s"}
    unsupported = set(timing) - supported
    if unsupported:
        raise ContractError(
            "unsupported sensor_timing fields for simulation: "
            + ", ".join(sorted(unsupported))
        )
    sim = getattr(env_cfg, "sim", None)
    physics_dt = float(getattr(sim, "dt", 0.0))
    if not math_is_positive(physics_dt):
        raise ContractError("environment configuration has no positive finite sim.dt")
    if not hasattr(env_cfg, "sim2real_command_delay_steps"):
        raise ContractError("environment configuration does not support sim2real command delay")

    requested_delay_s = float(timing.get("aggregate_command_delay_s", 0.0))
    if requested_delay_s > 0.0:
        delay_steps = max(1, int(math.floor(requested_delay_s / physics_dt + 0.5)))
    else:
        delay_steps = 0
    env_cfg.sim2real_command_delay_steps = delay_steps
    return {
        **dict(timing),
        "command_delay_steps": delay_steps,
        "effective_command_delay_s": delay_steps * physics_dt,
    }


def apply_abad_target_mapping(
    requested_rad: float,
    profile: CalibrationProfileV1 | None,
    *,
    joint: str,
) -> float:
    """Apply the measured relation ``actual = scale * requested + offset``."""

    if profile is None:
        return float(requested_rad)
    hardware = profile.validate().hardware_mapping
    scale = float(hardware.get("abad_target_scale", {}).get(joint, 1.0))
    offset = float(hardware.get("abad_target_offset_rad", {}).get(joint, 0.0))
    return scale * float(requested_rad) + offset


def _apply_abad_mapping_config(
    env_cfg: object,
    hardware: Mapping[str, Any],
) -> None:
    joint_names = getattr(env_cfg, "abad_joint_names", None)
    if not isinstance(joint_names, (list, tuple)) or not joint_names:
        raise ContractError("environment configuration has no ordered ABAD joint names")
    if not hasattr(env_cfg, "sim2real_abad_target_scale") or not hasattr(
        env_cfg, "sim2real_abad_target_offset_rad"
    ):
        raise ContractError("environment configuration does not support ABAD target mapping")
    aliases = tuple(f"abad_{index}" for index in range(len(joint_names)))
    scale_values = hardware.get("abad_target_scale", {})
    offset_values = hardware.get("abad_target_offset_rad", {})
    unknown = (set(scale_values) | set(offset_values)) - set(aliases)
    if unknown:
        raise ContractError(
            "ABAD target mapping uses non-canonical joints: " + ", ".join(sorted(unknown))
        )
    env_cfg.sim2real_abad_target_scale = tuple(
        float(scale_values.get(alias, 1.0)) for alias in aliases
    )
    env_cfg.sim2real_abad_target_offset_rad = tuple(
        float(offset_values.get(alias, 0.0)) for alias in aliases
    )


def apply_profile_to_config(
    env_cfg: object,
    profile: CalibrationProfileV1 | None,
) -> dict[str, Any] | None:
    """Apply an explicitly selected candidate to one config instance.

    This function never discovers or loads a default profile. Runtime-only mass and
    per-joint friction values are returned in the summary for ``isaac_profile``.
    """

    if profile is None:
        return None
    profile = profile.validate()
    timing_summary = _apply_sensor_timing(env_cfg, profile.sensor_timing)
    _apply_abad_mapping_config(env_cfg, profile.hardware_mapping)
    physics = profile.simulation_physics
    robot = _robot_cfg(env_cfg)

    rigid = physics.get("rigid_body", {})
    if rigid:
        rigid_props = getattr(getattr(robot, "spawn", None), "rigid_props", None)
        if rigid_props is None:
            raise ContractError("robot configuration has no spawn.rigid_props")
        _set_fields(rigid_props, rigid)

    actuators = getattr(robot, "actuators", None)
    if not isinstance(actuators, Mapping):
        raise ContractError("robot configuration has no actuator mapping")
    actuator_field_map = {
        "effort_limit": "effort_limit_sim",
        "velocity_limit": "velocity_limit_sim",
    }
    for section_name in ("main_drive", "abad", "damper"):
        values = physics.get(section_name, {})
        if not values:
            continue
        if section_name not in actuators:
            raise ContractError(f"robot configuration has no {section_name} actuator")
        _set_fields(actuators[section_name], values, actuator_field_map)

    springs = physics.get("passive_spring", {})
    init_state = getattr(robot, "init_state", None)
    joint_pos = getattr(init_state, "joint_pos", None) if init_state is not None else None
    for joint_name, spring in springs.items():
        if "rest_position_rad" in spring:
            if not isinstance(joint_pos, dict) or joint_name not in joint_pos:
                raise ContractError(f"passive spring joint is absent from init state: {joint_name}")
            joint_pos[joint_name] = float(spring["rest_position_rad"])

    damper_values = physics.get("damper", {})
    if "stiffness" in damper_values and hasattr(env_cfg, "damper_stiffness"):
        env_cfg.damper_stiffness = float(damper_values["stiffness"])
    if "damping" in damper_values and hasattr(env_cfg, "damper_damping"):
        env_cfg.damper_damping = float(damper_values["damping"])

    ground = physics.get("ground", {})
    if ground:
        material_targets: list[object] = []
        sim_material = getattr(getattr(env_cfg, "sim", None), "physics_material", None)
        terrain_material = getattr(getattr(env_cfg, "terrain", None), "physics_material", None)
        for material in (sim_material, terrain_material):
            if material is not None and all(material is not item for item in material_targets):
                material_targets.append(material)
        if not material_targets:
            raise ContractError("environment configuration has no ground material")
        for material in material_targets:
            _set_fields(material, ground)

    return {
        "schema_version": 1,
        "profile_id": profile.profile_id,
        "hardware_mapping": copy.deepcopy(profile.hardware_mapping),
        "sensor_timing": timing_summary,
        "runtime_physics": {
            name: copy.deepcopy(physics.get(name, {}))
            for name in (
                "joint_friction",
                "joint_dynamic_friction",
                "joint_viscous_friction",
                "passive_spring",
                "mass",
            )
        },
    }


def corrected_mass_properties(
    masses: np.ndarray,
    inertias: np.ndarray,
    coms: np.ndarray,
    correction: Mapping[str, Any],
) -> CorrectedMassProperties:
    clean_masses = np.asarray(masses, dtype=np.float64).copy()
    clean_inertias = np.asarray(inertias, dtype=np.float64).copy()
    clean_coms = np.asarray(coms, dtype=np.float64).copy()
    if clean_masses.ndim != 2 or clean_masses.shape[1] < 1:
        raise ContractError("masses must have shape (environment, body)")
    if clean_inertias.shape[:2] != clean_masses.shape or clean_coms.shape[:2] != clean_masses.shape:
        raise ContractError("mass, inertia, and CoM body layouts must match")
    if clean_coms.shape[-1] < 3:
        raise ContractError("CoM poses must contain xyz coordinates")
    if not all(np.isfinite(value).all() for value in (clean_masses, clean_inertias, clean_coms)):
        raise ContractError("mass properties must be finite")

    scale = float(correction.get("scale", 1.0))
    added_mass = float(correction.get("added_mass_kg", 0.0))
    if not math_is_positive(scale) or added_mass < 0.0 or not np.isfinite(added_mass):
        raise ContractError("mass correction contains an invalid scale or added mass")
    clean_masses *= scale
    clean_inertias *= scale
    clean_masses[:, 0] += added_mass
    offset = correction.get("com_offset_m", [0.0, 0.0, 0.0])
    if len(offset) != 3 or not np.isfinite(np.asarray(offset, dtype=np.float64)).all():
        raise ContractError("mass correction CoM offset must contain three finite values")
    clean_coms[:, 0, :3] += np.asarray(offset, dtype=np.float64)
    if np.any(clean_masses <= 0.0):
        raise ContractError("corrected body masses must remain positive")
    return CorrectedMassProperties(clean_masses, clean_inertias, clean_coms)


def math_is_positive(value: float) -> bool:
    return bool(np.isfinite(value) and value > 0.0)
