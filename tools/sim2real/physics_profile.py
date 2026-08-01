from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import CalibrationProfileV1, ContractError, load_profile
from .traces import _atomic_json, sha256_json


@dataclass(frozen=True)
class CorrectedMassProperties:
    masses: np.ndarray
    inertias: np.ndarray
    coms: np.ndarray


def load_optional_profile(path: str | Path | None) -> CalibrationProfileV1 | None:
    return None if path is None else load_profile(path)


def write_training_profile_snapshot(
    log_dir: str | Path,
    profile: CalibrationProfileV1 | None,
    *,
    config_application: Mapping[str, Any] | None,
    runtime_application: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Persist the explicit profile and both application phases with one hash."""

    if profile is None:
        return None
    clean_profile = profile.validate()
    if not isinstance(config_application, Mapping) or not isinstance(
        runtime_application, Mapping
    ):
        raise ContractError(
            "profile snapshot requires config and runtime application summaries"
        )
    params = Path(log_dir) / "params"
    params.mkdir(parents=True, exist_ok=True)
    payload = clean_profile.to_dict()
    metadata = {
        "schema_version": 1,
        "profile_id": clean_profile.profile_id,
        "profile_sha256": sha256_json(payload),
        "config_application": dict(config_application),
        "runtime_application": dict(runtime_application),
    }
    _atomic_json(params / "physics_profile.json", payload)
    _atomic_json(params / "physics_profile_metadata.json", metadata)
    return metadata


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
) -> dict[str, Any]:
    applied = {"aggregate_command_delay_s"}
    inactive_metadata = {"measured_state_rate_hz", "velocity_filter_alpha"}
    supported = applied | inactive_metadata
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
        "inactive_metadata_fields": sorted(set(timing) & inactive_metadata),
    }


def canonical_joint_name_map(env_cfg: object) -> dict[str, str]:
    """Resolve stable profile aliases through the production joint ordering."""

    groups = (
        ("main", "main_drive_joint_names", "main drive"),
        ("abad", "abad_joint_names", "ABAD"),
        ("damper", "damper_joint_names", "damper"),
    )
    aliases: dict[str, str] = {}
    runtime_names: list[str] = []
    for prefix, attribute, label in groups:
        names = getattr(env_cfg, attribute, None)
        if (
            not isinstance(names, (list, tuple))
            or len(names) != 6
            or any(not isinstance(name, str) or not name for name in names)
        ):
            raise ContractError(
                f"environment configuration has no six ordered {label} joint names"
            )
        for index, name in enumerate(names):
            aliases[f"{prefix}_{index}"] = name
            runtime_names.append(name)
    duplicates = sorted(
        {name for name in runtime_names if runtime_names.count(name) > 1}
    )
    if duplicates:
        raise ContractError(
            "environment configuration maps more than one canonical alias to a "
            "duplicate runtime joint: "
            + ", ".join(duplicates)
        )
    return aliases


def apply_abad_target_mapping(
    requested_rad: float,
    profile: CalibrationProfileV1 | None,
    *,
    joint: str,
    minimum_rad: float,
    maximum_rad: float,
) -> float:
    """Apply the measured relation and final physical target clamp."""

    lower = float(minimum_rad)
    upper = float(maximum_rad)
    requested = float(requested_rad)
    if not all(np.isfinite(value) for value in (requested, lower, upper)) or lower >= upper:
        raise ContractError("ABAD target mapping requires finite ordered physical bounds")
    scale = 1.0
    offset = 0.0
    if profile is not None:
        hardware = profile.validate().hardware_mapping
        scale = float(hardware.get("abad_target_scale", {}).get(joint, 1.0))
        offset = float(hardware.get("abad_target_offset_rad", {}).get(joint, 0.0))
    return min(max(scale * requested + offset, lower), upper)


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
    per_joint_sections = (
        "joint_friction",
        "joint_dynamic_friction",
        "joint_viscous_friction",
        "passive_spring",
    )
    joint_aliases = (
        canonical_joint_name_map(env_cfg)
        if any(physics.get(section) for section in per_joint_sections)
        else {}
    )

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
    spring_backend = str(getattr(env_cfg, "spring_backend", "explicit"))
    if spring_backend not in ("explicit", "native"):
        raise ContractError(
            "environment spring_backend must be 'explicit' or 'native'"
        )
    for section_name in ("main_drive", "abad", "damper"):
        values = physics.get(section_name, {})
        if not values:
            continue
        if section_name not in actuators:
            raise ContractError(f"robot configuration has no {section_name} actuator")
        actuator_values = dict(values)
        if section_name == "damper":
            # Spring stiffness/damping are resolved below as physical parameters.
            # Explicit mode must never accidentally retain a PhysX drive gain.
            actuator_values.pop("stiffness", None)
            actuator_values.pop("damping", None)
        _set_fields(actuators[section_name], actuator_values, actuator_field_map)
        energy_proxy_fields = {
            "main_drive": {
                "damping": "main_drive_torque_estimate_damping",
                "effort_limit": "main_drive_torque_estimate_limit",
            },
            "abad": {
                "stiffness": "abad_torque_estimate_stiffness",
                "damping": "abad_torque_estimate_damping",
                "effort_limit": "abad_torque_estimate_limit",
            },
        }.get(section_name, {})
        for profile_field, config_field in energy_proxy_fields.items():
            if profile_field in values:
                setattr(env_cfg, config_field, float(values[profile_field]))

    springs = physics.get("passive_spring", {})
    init_state = getattr(robot, "init_state", None)
    joint_pos = getattr(init_state, "joint_pos", None) if init_state is not None else None
    for joint_alias, spring in springs.items():
        if "rest_position_rad" in spring:
            joint_name = joint_aliases[joint_alias]
            if not isinstance(joint_pos, dict) or joint_name not in joint_pos:
                raise ContractError(
                    "passive spring joint is absent from init state: "
                    f"{joint_alias} ({joint_name})"
                )
            joint_pos[joint_name] = float(spring["rest_position_rad"])

    damper_values = physics.get("damper", {})
    if "stiffness" in damper_values and hasattr(env_cfg, "damper_stiffness"):
        env_cfg.damper_stiffness = float(damper_values["stiffness"])
    if "damping" in damper_values and hasattr(env_cfg, "damper_damping"):
        env_cfg.damper_damping = float(damper_values["damping"])

    damper_names = tuple(getattr(env_cfg, "damper_joint_names", ()))
    if damper_names:
        spring_count = len(damper_names)
        existing_stiffness = tuple(
            getattr(
                env_cfg,
                "spring_stiffness_nm_per_rad",
                (float(getattr(env_cfg, "damper_stiffness", 200.0)),)
                * spring_count,
            )
        )
        existing_damping = tuple(
            getattr(
                env_cfg,
                "spring_damping_nm_s_per_rad",
                (float(getattr(env_cfg, "damper_damping", 0.0)),)
                * spring_count,
            )
        )
        if len(existing_stiffness) != spring_count or len(existing_damping) != spring_count:
            raise ContractError(
                "torsion-spring parameter lengths must match damper_joint_names"
            )
        if "stiffness" in damper_values:
            stiffness_values = [float(damper_values["stiffness"])] * spring_count
        else:
            stiffness_values = [float(value) for value in existing_stiffness]
        if "damping" in damper_values:
            damping_values = [float(damper_values["damping"])] * spring_count
        else:
            damping_values = [float(value) for value in existing_damping]
        for index in range(spring_count):
            spring = springs.get(f"damper_{index}", {})
            if "stiffness" in spring:
                stiffness_values[index] = float(spring["stiffness"])
            if "damping" in spring:
                damping_values[index] = float(spring["damping"])
        env_cfg.spring_stiffness_nm_per_rad = tuple(stiffness_values)
        env_cfg.spring_damping_nm_s_per_rad = tuple(damping_values)

        damper_actuator = actuators.get("damper")
        if damper_actuator is None:
            raise ContractError("robot configuration has no damper actuator")
        if hasattr(damper_actuator, "stiffness"):
            damper_actuator.stiffness = (
                0.0
                if spring_backend == "explicit"
                else float(damper_values.get("stiffness", stiffness_values[0]))
            )
        if hasattr(damper_actuator, "damping"):
            damper_actuator.damping = (
                0.0
                if spring_backend == "explicit"
                else float(damper_values.get("damping", damping_values[0]))
            )

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
        has_friction = any(
            name in ground for name in ("static_friction", "dynamic_friction")
        )
        if has_friction and any(
            not hasattr(material, "friction_combine_mode") for material in material_targets
        ):
            raise ContractError("ground material does not expose a friction combine mode")
        for material in material_targets:
            _set_fields(material, ground)
            # Runtime overrides every robot collision shape to the same measured
            # coefficient. Max-combine then makes the effective pair value equal
            # the measurement regardless of the material embedded in the USD.
            if has_friction:
                material.friction_combine_mode = "max"

    return {
        "schema_version": 1,
        "profile_id": profile.profile_id,
        "spring_backend": spring_backend,
        "effective_springs": {
            "joint_order": [f"damper_{index}" for index in range(len(damper_names))],
            "stiffness": list(getattr(env_cfg, "spring_stiffness_nm_per_rad", ())),
            "damping": list(getattr(env_cfg, "spring_damping_nm_s_per_rad", ())),
        },
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
