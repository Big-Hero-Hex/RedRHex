from __future__ import annotations

import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np

from .compare import compare_traces
from .contracts import CalibrationProfileV1, ContractError, load_profile
from .characterization import (
    EXPECTED_FOOT_BODY_NAMES,
    PHYSICS_DT,
    load_replay_schedule,
    scenario_step_count,
)
from .metrics import compute_subsystem_metrics
from .provenance import validate_real_trace_provenance
from .scenarios import load_scenario
from .sweep import candidate_cache_key, validate_sweep_candidates
from .traces import LoadedTrace, load_trace, sha256_file, sha256_json


_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_-]*")
_METRIC_PATH = re.compile(r"[a-z][a-z0-9_.-]*")
_AUDIT_FIELDS = {
    "units_pass",
    "frames_pass",
    "joint_order_pass",
    "joint_axis_pass",
    "encoder_scale_zero_pass",
    "joint_sign_pass",
    "mechanical_range_pass",
    "mass_pass",
    "mass_profile_application_pass",
    "inertia_com_pass",
    "planar_com_pass",
    "collision_geometry_pass",
    "imu_mount_pass",
    "contact_sensor_pass",
}
_EXPECTED_AUDIT_UNITS = {
    "encoder_position": "rad",
    "main_command": "rad/s",
    "imu_gyro": "rad/s",
    "scale_mass": "kg",
    "load_force": "N",
}
_EXPECTED_AUDIT_FRAMES = {
    "encoder_position": "canonical_joint",
    "imu_gravity": "imu_mount",
    "contact_force": "world",
}
_HELD_OUT_DIMENSIONS = {"leg", "direction", "command_level", "load"}
_CANONICAL_JOINTS = tuple(
    [*(f"main_{index}" for index in range(6))]
    + [*(f"abad_{index}" for index in range(6))]
    + [*(f"damper_{index}" for index in range(6))]
)
_MAIN_JOINTS = tuple(f"main_{index}" for index in range(6))


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be an object")
    return value


def _exact_fields(
    value: Mapping[str, Any],
    *,
    name: str,
    required: set[str],
    optional: set[str] = frozenset(),
) -> None:
    missing = required - set(value)
    if missing:
        raise ContractError(f"{name} missing fields: {', '.join(sorted(missing))}")
    unknown = set(value) - required - optional
    if unknown:
        label = (
            "unknown validation evidence fields"
            if name == "validation evidence"
            else f"unknown {name} fields"
        )
        raise ContractError(f"{label}: {', '.join(sorted(unknown))}")


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ContractError(f"{name} must be an identifier")
    return value


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ContractError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _number(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "finite and non-negative" if nonnegative else "finite"
        raise ContractError(f"{name} must be {qualifier}")
    return result


def _json(path: Path, name: str) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot load {name} {path}: {exc}") from exc
    return _mapping(payload, name)


def load_validation_evidence(path: str | Path) -> Mapping[str, Any]:
    return _json(Path(path), "validation evidence")


def _artifact(root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{name} path must be non-empty")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ContractError(f"{name} path must be a safe relative path")
    resolved = root.joinpath(*relative.parts).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ContractError(f"{name} path escapes the artifact root") from exc
    if not resolved.exists():
        raise ContractError(f"{name} artifact does not exist: {value}")
    return resolved


def _file_binding(root: Path, value: Any, name: str) -> tuple[Path, Mapping[str, Any]]:
    binding = _mapping(value, name)
    _exact_fields(binding, name=name, required={"path", "sha256"})
    path = _artifact(root, binding["path"], name)
    expected = _sha(binding["sha256"], f"{name} sha256")
    actual = sha256_file(path)
    if actual != expected:
        raise ContractError(f"{name} hash mismatch")
    return path, binding


def _trace_binding(
    root: Path,
    value: Any,
    name: str,
    *,
    source: str,
    scenario: Any | None = None,
    profile: CalibrationProfileV1 | None = None,
) -> tuple[LoadedTrace, Mapping[str, Any]]:
    binding = _mapping(value, name)
    required = {"path", "trace_sha256", "metadata_sha256"}
    optional: set[str] = set()
    if source == "real":
        required.update({"dataset_id", "episode_id"})
    _exact_fields(binding, name=name, required=required, optional=optional)
    path = _artifact(root, binding["path"], name)
    metadata_sha256 = _sha(binding["metadata_sha256"], f"{name} metadata_sha256")
    loaded = load_trace(
        path,
        scenario=scenario,
        profile=profile,
        require_managed_dataset=source == "real",
        expected_metadata_sha256=metadata_sha256,
    )
    expected = _sha(binding["trace_sha256"], f"{name} trace_sha256")
    actual = loaded.manifest.provenance["trace_sha256"]
    if actual != expected:
        raise ContractError(f"{name} trace hash mismatch")
    if loaded.manifest.source != source:
        raise ContractError(f"{name} must have source={source!r}")
    if source == "real":
        assert loaded.dataset is not None
        if loaded.dataset.dataset_id != _identifier(
            binding["dataset_id"], f"{name} dataset_id"
        ):
            raise ContractError(f"{name} dataset identity mismatch")
        if loaded.dataset.episode_id != _identifier(
            binding["episode_id"], f"{name} episode_id"
        ):
            raise ContractError(f"{name} episode identity mismatch")
    return loaded, binding


def _expected_condition_metadata(
    scenario: Any,
) -> tuple[dict[str, str], dict[str, str]]:
    """Return the reviewed unit/frame contract for a runnable condition."""

    kind = scenario.experiment_kind
    if kind in {"step", "coast", "step_coast"}:
        units = {"command": "rad/s", "position": "rad"}
        frames = {name: scenario.joint for name in units}
    elif kind == "abad_static":
        units = {
            "command": "rad",
            "position": "rad",
            "repeat_index": "1",
            "settled": "1",
        }
        frames = {name: scenario.joint for name in units}
    elif kind == "friction":
        units = {
            "breakaway_force": "N",
            "static_normal_load": "N",
            "static_repeat_index": "1",
            "dynamic_pull_force": "N",
            "dynamic_normal_load": "N",
            "dynamic_speed": "m/s",
            "dynamic_repeat_index": "1",
        }
        frames = {name: f"{scenario.joint}/ground" for name in units}
    elif kind == "static_settle":
        units = {
            "root_position": "m",
            "contact_force_n": "N",
            "repeat_index": "1",
            "settled": "1",
        }
        frames = {
            "root_position": "world",
            "contact_force_n": "feet/ground",
            "repeat_index": "annotation",
            "settled": "annotation",
        }
    elif kind == "mass_com":
        units = {
            "scale_mass": "kg",
            "support_force": "N",
            "support_position": "m",
            "repeat_index": "1",
        }
        frames = {name: "root" for name in units}
    elif kind == "spring":
        units = {
            "load_force": "N",
            "lever_arm": "m",
            "angle": "rad",
            "repeat_index": "1",
        }
        frames = {name: scenario.joint for name in units}
    elif kind == "manual_load":
        units = {
            "load_force": "N",
            "lever_arm": "m",
            "command": "normalized",
            "direction": "1",
            "saturation_confirmed": "1",
            "repeat_index": "1",
        }
        frames = {name: scenario.joint for name in units}
    else:
        raise ContractError(
            f"scenario {scenario.scenario_id} has no reviewed trace metadata contract"
        )
    missing = set(scenario.required_channels) - set(units)
    if missing:
        raise ContractError(
            f"scenario {scenario.scenario_id} has no reviewed metadata for: "
            + ", ".join(sorted(missing))
        )
    return units, frames


def _validate_condition_trace_metadata(loaded: LoadedTrace, scenario: Any) -> None:
    expected_units, expected_frames = _expected_condition_metadata(scenario)
    actual_units = loaded.manifest.metadata["units"]
    actual_frames = loaded.manifest.metadata["frames"]
    for channel in scenario.required_channels:
        expected_unit = expected_units[channel]
        actual_unit = actual_units.get(channel)
        if actual_unit != expected_unit:
            raise ContractError(
                f"expected unit for {channel} is {expected_unit}, got {actual_unit}"
            )
        expected_frame = expected_frames[channel]
        actual_frame = actual_frames.get(channel)
        if actual_frame != expected_frame:
            raise ContractError(
                f"expected frame for {channel} is {expected_frame}, got {actual_frame}"
            )


def _changed_subsystems(
    baseline: CalibrationProfileV1, candidate: CalibrationProfileV1
) -> set[str]:
    result: set[str] = set()
    before_hardware = baseline.hardware_mapping
    after_hardware = candidate.hardware_mapping
    for field in set(before_hardware) | set(after_hardware):
        before = before_hardware.get(field, {})
        after = after_hardware.get(field, {})
        if before == after:
            continue
        joints = set(before) | set(after) if isinstance(before, Mapping) and isinstance(after, Mapping) else set()
        for joint in joints:
            if isinstance(before, Mapping) and isinstance(after, Mapping) and before.get(joint) == after.get(joint):
                continue
            if str(joint).startswith("main_"):
                result.add("main_drive")
            elif str(joint).startswith("abad_"):
                result.add("abad")
            elif str(joint).startswith("damper_"):
                result.add("spring")
            else:
                result.add("hardware_mapping")
    if baseline.sensor_timing != candidate.sensor_timing:
        result.add("timing")
    aliases = {
        "rigid_body": "rigid_body",
        "mass": "rigid_body",
        "main_drive": "main_drive",
        "abad": "abad",
        "damper": "spring",
        "passive_spring": "spring",
        "ground": "contact",
    }
    before_physics = baseline.simulation_physics
    after_physics = candidate.simulation_physics
    for section in set(before_physics) | set(after_physics):
        if before_physics.get(section) == after_physics.get(section):
            continue
        if section in {"joint_friction", "joint_dynamic_friction", "joint_viscous_friction"}:
            values: dict[str, Any] = {}
            values.update(before_physics.get(section, {}))
            values.update(after_physics.get(section, {}))
            for joint in values:
                result.add(
                    "main_drive"
                    if str(joint).startswith("main_")
                    else "abad"
                    if str(joint).startswith("abad_")
                    else "spring"
                )
        else:
            result.add(aliases.get(section, section))
    return result


def _validate_identifiable_changes(
    baseline: CalibrationProfileV1, candidate: CalibrationProfileV1
) -> None:
    """Reject parameters that the implemented experiments cannot identify."""

    before = baseline.simulation_physics
    after = candidate.simulation_physics

    def changed(section: str, field: str | None = None) -> bool:
        old = before.get(section, {})
        new = after.get(section, {})
        if field is None:
            return old != new
        old_value = old.get(field) if isinstance(old, Mapping) else None
        new_value = new.get(field) if isinstance(new, Mapping) else None
        return old_value != new_value

    unsupported: list[str] = []
    if changed("abad"):
        unsupported.append("simulation_physics.abad")
    if changed("rigid_body"):
        unsupported.append("simulation_physics.rigid_body")
    if changed("damper"):
        unsupported.append("simulation_physics.damper")
    if changed("ground", "restitution"):
        unsupported.append("simulation_physics.ground.restitution")
    if changed("main_drive", "stiffness"):
        unsupported.append("simulation_physics.main_drive.stiffness")

    old_springs = before.get("passive_spring", {})
    new_springs = after.get("passive_spring", {})
    if isinstance(old_springs, Mapping) and isinstance(new_springs, Mapping):
        for joint in set(old_springs) | set(new_springs):
            old = old_springs.get(joint, {})
            new = new_springs.get(joint, {})
            for field in {"damping", "rest_position_rad"}:
                if (
                    isinstance(old, Mapping)
                    and isinstance(new, Mapping)
                    and old.get(field) != new.get(field)
                ):
                    unsupported.append(
                        f"simulation_physics.passive_spring.{joint}.{field}"
                    )

    for section in (
        "joint_friction",
        "joint_dynamic_friction",
        "joint_viscous_friction",
    ):
        old_values = before.get(section, {})
        new_values = after.get(section, {})
        if not isinstance(old_values, Mapping) or not isinstance(new_values, Mapping):
            continue
        for joint in set(old_values) | set(new_values):
            if old_values.get(joint) != new_values.get(joint) and str(joint).startswith(
                ("abad_", "damper_")
            ):
                unsupported.append(f"simulation_physics.{section}.{joint}")

    supported_hardware = {
        "encoder_counts_per_rev",
        "encoder_zero_count",
        "encoder_sign",
        "joint_direction",
        "abad_target_scale",
        "abad_target_offset_rad",
    }
    for field in set(baseline.hardware_mapping) | set(candidate.hardware_mapping):
        if (
            baseline.hardware_mapping.get(field)
            != candidate.hardware_mapping.get(field)
            and field not in supported_hardware
        ):
            if field in {"pwm_scale", "pwm_cap"}:
                unsupported.append(
                    f"hardware_mapping.{field} requires mapping-specific "
                    "measured/source evidence"
                )
            else:
                unsupported.append(f"hardware_mapping.{field}")

    supported_timing = {"aggregate_command_delay_s"}
    for field in set(baseline.sensor_timing) | set(candidate.sensor_timing):
        if (
            baseline.sensor_timing.get(field) != candidate.sensor_timing.get(field)
            and field not in supported_timing
        ):
            unsupported.append(f"sensor_timing.{field}")

    old_mass = before.get("mass", {})
    new_mass = after.get("mass", {})
    for field in {"scale", "added_mass_kg", "com_offset_m"}:
        if (
            isinstance(old_mass, Mapping)
            and isinstance(new_mass, Mapping)
            and old_mass.get(field) != new_mass.get(field)
        ):
            unsupported.append(f"simulation_physics.mass.{field}")

    if unsupported:
        raise ContractError(
            "unidentifiable profile change: " + ", ".join(sorted(unsupported))
        )


def _profile_field_changed(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any], field: str
) -> bool:
    return baseline.get(field) != candidate.get(field)


def _validate_changed_field_evidence(
    baseline: CalibrationProfileV1,
    candidate: CalibrationProfileV1,
    conditions: Mapping[str, Mapping[str, Any]],
) -> None:
    """Require evidence for each changed field, not just its broad subsystem."""

    before_main = baseline.simulation_physics.get("main_drive", {})
    after_main = candidate.simulation_physics.get("main_drive", {})
    velocity_limit_changed = False
    response_changed = False
    if isinstance(before_main, Mapping) and isinstance(after_main, Mapping):
        velocity_limit_changed = (
            before_main.get("velocity_limit") != after_main.get("velocity_limit")
        )
        response_changed = any(
            before_main.get(field) != after_main.get(field)
            for field in {"damping", "velocity_limit", "armature", "friction"}
        )

    for section in (
        "joint_friction",
        "joint_dynamic_friction",
        "joint_viscous_friction",
    ):
        before = baseline.simulation_physics.get(section, {})
        after = candidate.simulation_physics.get(section, {})
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            continue
        response_changed |= any(
            str(joint).startswith("main_")
            and before.get(joint) != after.get(joint)
            for joint in set(before) | set(after)
        )

    mapping_changed = any(
        _profile_field_changed(
            baseline.hardware_mapping, candidate.hardware_mapping, field
        )
        for field in ("joint_direction", "pwm_scale", "pwm_cap")
    )
    response_changed |= mapping_changed
    response_changed |= (
        baseline.sensor_timing.get("aggregate_command_delay_s")
        != candidate.sensor_timing.get("aggregate_command_delay_s")
    )
    if not response_changed:
        return

    response_conditions = [
        internal
        for internal in conditions.values()
        if internal["scenario"].experiment_kind in {"step", "coast", "step_coast"}
        and internal["scenario"].scene_mode != "manual"
    ]
    roles = {internal["role"] for internal in response_conditions}
    if velocity_limit_changed and roles == {"calibration", "holdout"}:
        new_limit = after_main.get("velocity_limit")
        if isinstance(new_limit, bool) or not isinstance(new_limit, (int, float)):
            raise ContractError(
                "velocity-limit changes require a finite saturation target"
            )
        limit = float(new_limit)
        saturation_roles = {
            internal["role"]
            for internal in response_conditions
            if max(
                abs(float(segment["value"]))
                for segment in internal["scenario"].command_segments
            )
            >= limit
        }
        if saturation_roles != {"calibration", "holdout"}:
            raise ContractError(
                "velocity-limit changes require calibration and holdout commands "
                "that demonstrably excite saturation"
            )
    if not response_conditions:
        raise ContractError(
            "changed actuator/timing fields require independent calibration and "
            "held-out main-drive response evidence"
        )
    if roles != {"calibration", "holdout"}:
        return

    if mapping_changed:
        candidate_hash = sha256_json(candidate.to_dict())
        for internal in response_conditions:
            if not any(
                trace.manifest.provenance.get("profile_sha256") == candidate_hash
                for trace in internal["real_traces"]
            ):
                raise ContractError(
                    "changed main-drive command mapping requires candidate-bound "
                    "response evidence"
                )


def _scenario_supports(subsystem: str, scenario_subsystem: str) -> bool:
    supported = {
        "main_drive": {"main_drive"},
        "timing": {"main_drive"},
        "abad": {"abad"},
        "spring": {"spring"},
        "contact": {"friction"},
        "rigid_body": {"mass_com", "audit"},
    }
    return scenario_subsystem in supported.get(subsystem, {subsystem})


def _mandatory_holdout_metrics(
    subsystem: str, scenario: Any
) -> dict[str, str]:
    """Return the complete reviewed metric contract for one runnable holdout."""

    kind = scenario.experiment_kind
    if subsystem in {"main_drive", "timing"}:
        result: dict[str, str] = {}
        if kind in {"step", "step_coast"}:
            step_metrics = (
                (("onset_delay_s", "s"),)
                if subsystem == "timing"
                else (
                    ("onset_delay_s", "s"),
                    ("steady_speed_rad_s", "rad/s"),
                    ("rise_time_s", "s"),
                    ("overshoot_ratio", "1"),
                )
            )
            prefix = "step." if kind == "step_coast" else ""
            for direction in ("positive", "negative"):
                for metric, unit in step_metrics:
                    result[f"{prefix}{direction}.{metric}"] = unit
        if subsystem == "main_drive" and kind in {"coast", "step_coast"}:
            prefix = "coast." if kind == "step_coast" else ""
            for direction in ("positive", "negative"):
                result[f"{prefix}{direction}.coast_time_s"] = "s"
                result[f"{prefix}{direction}.pre_coast_speed_rad_s"] = "rad/s"
        if subsystem == "main_drive" and kind == "manual_load":
            return {
                "positive_torque_nm": "N*m",
                "negative_torque_nm": "N*m",
            }
        if result:
            return result
    elif subsystem == "abad" and kind == "abad_static":
        return {
            "aggregate.target_scale": "1",
            "aggregate.target_offset_rad": "rad",
            "aggregate.fit_rmse_rad": "rad",
        }
    elif subsystem == "contact" and kind == "static_settle":
        return {
            "settled.root_height_m": "m",
            "settled.contact_force_n": "N",
        }
    elif subsystem == "rigid_body" and kind == "mass_com":
        return {"mass_kg": "kg", "com_x_m": "m", "com_y_m": "m"}
    elif subsystem == "spring" and kind == "spring":
        return {"stiffness_nm_per_rad": "N*m/rad"}
    raise ContractError(
        f"no mandatory held-out metric contract for {subsystem}.{scenario.scenario_id}"
    )


def _direct_profile_value(
    candidate: CalibrationProfileV1,
    subsystem: str,
    scenario: Any,
    metric_path: str,
) -> float:
    physics = candidate.simulation_physics
    if subsystem == "rigid_body" and scenario.experiment_kind == "mass_com":
        mass = _mapping(physics.get("mass", {}), "candidate mass profile")
        if metric_path == "mass_kg":
            return _number(
                mass.get("target_total_mass_kg"), "candidate target_total_mass_kg"
            )
        planar = mass.get("reference_planar_com_xy_m")
        if not isinstance(planar, list) or len(planar) != 2:
            raise ContractError(
                "candidate mass profile has no reference_planar_com_xy_m target"
            )
        index = {"com_x_m": 0, "com_y_m": 1}.get(metric_path)
        if index is not None:
            return _number(planar[index], f"candidate {metric_path}")
    if subsystem == "spring" and scenario.experiment_kind == "spring":
        springs = _mapping(
            physics.get("passive_spring", {}), "candidate passive_spring profile"
        )
        spring = _mapping(
            springs.get(scenario.joint), f"candidate spring {scenario.joint}"
        )
        if metric_path == "stiffness_nm_per_rad":
            return _number(spring.get("stiffness"), "candidate spring stiffness")
    if subsystem == "main_drive" and scenario.experiment_kind == "manual_load":
        if metric_path in {"positive_torque_nm", "negative_torque_nm"}:
            actuator = _mapping(
                physics.get("main_drive", {}), "candidate main_drive profile"
            )
            return _number(
                actuator.get("effort_limit"), "candidate main_drive effort_limit"
            )
    raise ContractError(
        f"no direct profile target for {subsystem}.{scenario.scenario_id}.{metric_path}"
    )


def _validate_direct_holdout_context(
    candidate: CalibrationProfileV1,
    scenario: Any,
    traces: list[LoadedTrace],
) -> None:
    if scenario.experiment_kind != "mass_com":
        return
    mass = _mapping(
        candidate.simulation_physics.get("mass", {}), "candidate mass profile"
    )
    expected_joints = mass.get("reference_joint_position_rad")
    expected_orientation = mass.get("reference_root_orientation_xyzw")
    for trace in traces:
        constants = _mapping(
            trace.manifest.metadata.get("calibration_constants", {}),
            "mass-com holdout calibration_constants",
        )
        if (
            constants.get("reference_joint_position_rad") != expected_joints
            or constants.get("reference_root_orientation_xyzw")
            != expected_orientation
        ):
            raise ContractError(
                "mass-com holdout reference pose must match the candidate profile"
            )


def _required_measurement_source_keys(
    baseline: CalibrationProfileV1,
    candidate: CalibrationProfileV1,
) -> set[str]:
    """Return profile fields that must remain bound to direct real measurements."""

    required: set[str] = set()
    for field in ("abad_target_scale", "abad_target_offset_rad"):
        before = baseline.hardware_mapping.get(field, {})
        after = candidate.hardware_mapping.get(field, {})
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            continue
        for joint in set(before) | set(after):
            if before.get(joint) != after.get(joint):
                required.add(f"abad_target:{joint}")
    before_ground = baseline.simulation_physics.get("ground", {})
    after_ground = candidate.simulation_physics.get("ground", {})
    if isinstance(before_ground, Mapping) and isinstance(after_ground, Mapping):
        if any(
            before_ground.get(field) != after_ground.get(field)
            for field in ("static_friction", "dynamic_friction")
        ):
            required.add("ground_friction")
    before_mass = baseline.simulation_physics.get("mass", {})
    after_mass = candidate.simulation_physics.get("mass", {})
    absolute_mass_fields = {
        "target_total_mass_kg",
        "reference_planar_com_xy_m",
        "reference_joint_position_rad",
        "reference_root_orientation_xyzw",
    }
    if isinstance(before_mass, Mapping) and isinstance(after_mass, Mapping) and any(
        before_mass.get(field) != after_mass.get(field)
        for field in absolute_mass_fields
    ):
        required.add("mass_com")
    before_springs = baseline.simulation_physics.get("passive_spring", {})
    after_springs = candidate.simulation_physics.get("passive_spring", {})
    if isinstance(before_springs, Mapping) and isinstance(after_springs, Mapping):
        for joint in set(before_springs) | set(after_springs):
            old = before_springs.get(joint, {})
            new = after_springs.get(joint, {})
            if (
                isinstance(old, Mapping)
                and isinstance(new, Mapping)
                and old.get("stiffness") != new.get("stiffness")
            ):
                required.add(f"passive_spring:{joint}")
    after_main_drive = candidate.simulation_physics.get("main_drive", {})
    after_effort_limit = (
        after_main_drive.get("effort_limit")
        if isinstance(after_main_drive, Mapping)
        else None
    )
    if _main_effort_limit_changed(baseline, candidate) and after_effort_limit is not None:
        effort_sources = [
            key
            for key in candidate.measurement_sources
            if key.startswith("main_drive_effort_limit:")
        ]
        if len(effort_sources) != 1:
            raise ContractError(
                "main-drive effort-limit fitting requires exactly one measured joint source"
            )
        required.add(effort_sources[0])
    return required


def _main_effort_limit_changed(
    baseline: CalibrationProfileV1, candidate: CalibrationProfileV1
) -> bool:
    before = baseline.simulation_physics.get("main_drive", {})
    after = candidate.simulation_physics.get("main_drive", {})
    before_value = before.get("effort_limit") if isinstance(before, Mapping) else None
    after_value = after.get("effort_limit") if isinstance(after, Mapping) else None
    return before_value != after_value


def _require_calibration_value(actual: Any, expected: Any, field: str) -> None:
    measured = _number(expected, f"calibration measurement {field}")
    configured = _number(actual, f"candidate {field}")
    if not math.isclose(configured, measured, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise ContractError(
            f"candidate {field} does not match its calibration measurement"
        )


def _validate_direct_measurement_value(
    candidate: CalibrationProfileV1,
    key: str,
    trace: LoadedTrace,
    metrics: Mapping[str, Any],
) -> None:
    """Bind directly applied profile fields to their immutable calibration trace."""

    if key.startswith("abad_target:"):
        joint = key.removeprefix("abad_target:")
        aggregate = _mapping(metrics.get("aggregate"), "ABAD calibration metrics")
        for field, metric in (
            ("abad_target_scale", "target_scale"),
            ("abad_target_offset_rad", "target_offset_rad"),
        ):
            values = _mapping(
                candidate.hardware_mapping.get(field), f"candidate {field}"
            )
            _require_calibration_value(
                values.get(joint),
                aggregate.get(metric),
                f"hardware_mapping.{field}.{joint}",
            )
        return

    physics = candidate.simulation_physics
    if key == "ground_friction":
        ground = _mapping(physics.get("ground"), "candidate ground profile")
        for field, section in (
            ("static_friction", "static"),
            ("dynamic_friction", "dynamic"),
        ):
            measured = _mapping(
                metrics.get(section), f"ground calibration {section} metrics"
            )
            _require_calibration_value(
                ground.get(field),
                measured.get("coefficient_mean"),
                f"simulation_physics.ground.{field}",
            )
        return

    if key == "mass_com":
        mass = _mapping(physics.get("mass"), "candidate mass profile")
        _require_calibration_value(
            mass.get("target_total_mass_kg"),
            metrics.get("mass_kg"),
            "simulation_physics.mass.target_total_mass_kg",
        )
        planar = mass.get("reference_planar_com_xy_m")
        if not isinstance(planar, list) or len(planar) != 2:
            raise ContractError(
                "candidate simulation_physics.mass.reference_planar_com_xy_m "
                "does not match its calibration measurement"
            )
        for index, metric in enumerate(("com_x_m", "com_y_m")):
            _require_calibration_value(
                planar[index],
                metrics.get(metric),
                f"simulation_physics.mass.reference_planar_com_xy_m[{index}]",
            )
        constants = _mapping(
            trace.manifest.metadata.get("calibration_constants"),
            "mass calibration constants",
        )
        for field in (
            "reference_joint_position_rad",
            "reference_root_orientation_xyzw",
        ):
            if mass.get(field) != constants.get(field):
                raise ContractError(
                    f"candidate simulation_physics.mass.{field} does not match "
                    "its calibration measurement"
                )
        return

    if key.startswith("passive_spring:"):
        joint = key.removeprefix("passive_spring:")
        springs = _mapping(
            physics.get("passive_spring"), "candidate passive_spring profile"
        )
        spring = _mapping(springs.get(joint), f"candidate passive spring {joint}")
        _require_calibration_value(
            spring.get("stiffness"),
            metrics.get("stiffness_nm_per_rad"),
            f"simulation_physics.passive_spring.{joint}.stiffness",
        )
        return

    if key.startswith("main_drive_effort_limit:"):
        actuator = _mapping(
            physics.get("main_drive"), "candidate main_drive profile"
        )
        _require_calibration_value(
            actuator.get("effort_limit"),
            metrics.get("torque_saturation_nm"),
            "simulation_physics.main_drive.effort_limit",
        )
        return

    raise ContractError(f"measurement source {key} has no direct profile binding")


def _validate_measurement_sources(
    baseline: CalibrationProfileV1,
    candidate: CalibrationProfileV1,
    conditions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Cross-check candidate source records against calibration-role episodes."""

    required = _required_measurement_source_keys(baseline, candidate)
    bindings: dict[str, Any] = {}
    for key in sorted(required):
        raw_source = candidate.measurement_sources.get(key)
        if not isinstance(raw_source, Mapping):
            raise ContractError(
                f"measurement source {key} is required for the fitted profile field"
            )
        matches: list[tuple[str, LoadedTrace, Mapping[str, Any]]] = []
        for condition_id, internal in conditions.items():
            if internal["role"] != "calibration":
                continue
            scenario = internal["scenario"]
            for trace in internal["real_traces"]:
                dataset = trace.dataset
                if dataset is None:  # pragma: no cover - real binding already guards this
                    continue
                metrics = compute_subsystem_metrics(scenario, trace)
                frame = metrics.get("frame")
                expected = {
                    "trace_sha256": trace.manifest.provenance["trace_sha256"],
                    "metadata_sha256": trace.metadata_sha256,
                    "scenario_id": scenario.scenario_id,
                    "scenario_sha256": trace.manifest.provenance["scenario_sha256"],
                    "source": trace.manifest.source,
                    "metric_kind": metrics.get("metric_kind"),
                    "frame": frame,
                    "repeat_count": _repeat_count(metrics, trace),
                    "dataset_id": dataset.dataset_id,
                    "episode_id": dataset.episode_id,
                }
                if dict(raw_source) == expected:
                    matches.append((condition_id, trace, metrics))
        if len(matches) != 1:
            raise ContractError(
                f"measurement source {key} must bind exactly one calibration-role real episode"
            )
        condition_id, trace, metrics = matches[0]
        _validate_direct_measurement_value(candidate, key, trace, metrics)
        bindings[key] = {
            "condition_id": condition_id,
            "dataset_id": trace.dataset.dataset_id,
            "episode_id": trace.dataset.episode_id,
            "trace_sha256": trace.manifest.provenance["trace_sha256"],
            "metadata_sha256": trace.metadata_sha256,
        }
    return {"pass": True, "required": sorted(required), "bindings": bindings}


def _condition_coordinates(trace: LoadedTrace, scenario: Any) -> dict[str, Any]:
    values = [float(segment["value"]) for segment in scenario.command_segments]
    directions = sorted({"positive" if value > 0.0 else "negative" for value in values if value})
    levels = sorted({abs(value) for value in values if value})
    constants = trace.manifest.metadata.get("calibration_constants", {})
    supplied = constants.get("condition_coordinates", {}) if isinstance(constants, Mapping) else {}
    if supplied and not isinstance(supplied, Mapping):
        raise ContractError("condition_coordinates metadata must be an object")
    return {
        "leg": scenario.joint,
        "direction": directions,
        "command_level": levels,
        "load": supplied.get("load") if isinstance(supplied, Mapping) else None,
    }


def _lookup(metrics: Mapping[str, Any], path: str) -> tuple[float, Mapping[str, Any], str]:
    if not isinstance(path, str) or not _METRIC_PATH.fullmatch(path):
        raise ContractError(f"invalid metric path {path!r}")
    current: Any = metrics
    parts = path.split(".")
    for part in parts[:-1]:
        if not isinstance(current, Mapping) or part not in current:
            raise ContractError(f"metric path does not exist: {path}")
        current = current[part]
    leaf = parts[-1]
    if not isinstance(current, Mapping) or leaf not in current:
        raise ContractError(f"metric path does not exist: {path}")
    return _number(current[leaf], f"metric {path}"), current, leaf


def _repeat_count(metrics: Mapping[str, Any], trace: LoadedTrace) -> int:
    counts: list[int] = []

    def visit(value: Any) -> None:
        if not isinstance(value, Mapping):
            return
        for key, item in value.items():
            if key == "repeat_count" and isinstance(item, (int, float)) and not isinstance(item, bool):
                numeric = float(item)
                if numeric.is_integer() and numeric >= 1:
                    counts.append(int(numeric))
            elif isinstance(item, Mapping):
                visit(item)

    visit(metrics)
    constants = trace.manifest.metadata.get("calibration_constants", {})
    evidence = constants.get("probe_event_evidence") if isinstance(constants, Mapping) else None
    if isinstance(evidence, Mapping):
        count = evidence.get("repetition_count")
        if isinstance(count, int) and not isinstance(count, bool) and count >= 1:
            counts.append(count)
    return min(counts) if counts else 0


def _observation(metrics: Mapping[str, Any], path: str, fallback_count: int) -> tuple[float, float, int]:
    mean, parent, leaf = _lookup(metrics, path)
    std_raw = parent.get(f"{leaf}_std", 0.0)
    std = _number(std_raw, f"metric {path} std", nonnegative=True)
    count_raw = parent.get("repeat_count", parent.get(f"{leaf}_count", fallback_count))
    if isinstance(count_raw, bool) or not isinstance(count_raw, (int, float)):
        raise ContractError(f"metric {path} repeat count is invalid")
    count_float = float(count_raw)
    if not count_float.is_integer() or count_float < 1:
        raise ContractError(f"metric {path} repeat count is invalid")
    return mean, std, int(count_float)


def _pool(observations: list[tuple[float, float, int]]) -> tuple[float, float, int]:
    count = sum(item[2] for item in observations)
    if count < 1:
        raise ContractError("cannot pool an empty metric observation")
    mean = sum(item[0] * item[2] for item in observations) / count
    variance = sum(
        item[2] * (item[1] ** 2 + (item[0] - mean) ** 2)
        for item in observations
    ) / count
    return float(mean), float(math.sqrt(max(0.0, variance))), count


def _numeric_vector(value: Any, name: str, *, length: int) -> np.ndarray:
    if not isinstance(value, list) or len(value) != length:
        raise ContractError(f"{name} must contain exactly {length} numeric values")
    result = np.asarray([_number(item, name) for item in value], dtype=float)
    return result


def _numeric_array(value: Any, name: str) -> np.ndarray:
    def clean(raw: Any, path: str) -> Any:
        if isinstance(raw, list):
            return [clean(item, f"{path}[{index}]") for index, item in enumerate(raw)]
        return _number(raw, path)

    if not isinstance(value, list):
        raise ContractError(f"{name} must be a numeric array")
    try:
        result = np.asarray(clean(value, name), dtype=float)
    except ValueError as exc:
        raise ContractError(f"{name} must be a rectangular numeric array") from exc
    return result


def _string_array(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ContractError(f"{name} must be a non-empty array of strings")
    if len(set(value)) != len(value):
        raise ContractError(f"{name} must contain unique values")
    return list(value)


def _range(
    record: Mapping[str, Any],
    *,
    name: str,
    lower_field: str,
    upper_field: str,
) -> tuple[str, float | None, float | None]:
    kind = record.get("range_kind")
    if kind not in {"continuous", "limited"}:
        raise ContractError(f"{name} range_kind must be continuous or limited")
    lower_raw = record.get(lower_field)
    upper_raw = record.get(upper_field)
    if kind == "continuous":
        if lower_raw is not None or upper_raw is not None:
            raise ContractError(f"{name} continuous range must use null limits")
        return kind, None, None
    lower = _number(lower_raw, f"{name} {lower_field}")
    upper = _number(upper_raw, f"{name} {upper_field}")
    if lower >= upper:
        raise ContractError(f"{name} lower limit must be smaller than its upper limit")
    return kind, lower, upper


def _derive_audit(
    root: Path,
    value: Any,
    candidate: CalibrationProfileV1,
) -> dict[str, Any]:
    binding = _mapping(value, "audit_artifact")
    _exact_fields(
        binding,
        name="audit_artifact",
        required={"runtime_trace", "runtime_audit", "physical_measurements"},
    )
    audit_scenario = load_scenario("audit")
    runtime_trace, _ = _trace_binding(
        root,
        binding["runtime_trace"],
        "audit runtime_trace",
        source="sim",
        scenario=audit_scenario,
    )
    runtime_path, runtime_binding = _file_binding(
        root, binding["runtime_audit"], "runtime audit"
    )
    if runtime_path != runtime_trace.directory / "runtime_audit.json":
        raise ContractError("runtime audit must be the bound runtime_trace sibling")
    runtime = _json(runtime_path, "runtime audit")
    if runtime.get("schema_version") != 2 or isinstance(runtime.get("schema_version"), bool):
        raise ContractError("runtime audit schema_version must be 2")
    trace_constants = _mapping(
        runtime_trace.manifest.metadata.get("calibration_constants"),
        "audit trace calibration_constants",
    )
    trace_runtime_hash = _sha(
        trace_constants.get("runtime_audit_sha256"),
        "audit trace runtime_audit_sha256",
    )
    if trace_runtime_hash != sha256_json(runtime):
        raise ContractError("runtime audit JSON is not bound to its audit trace")
    if runtime.get("num_envs") != 1 or isinstance(runtime.get("num_envs"), bool):
        raise ContractError("runtime audit must describe one environment")
    physics_dt = _number(runtime.get("physics_dt_s"), "runtime audit physics_dt_s")
    if not math.isclose(physics_dt, PHYSICS_DT, rel_tol=0.0, abs_tol=1.0e-12):
        raise ContractError("runtime audit physics_dt_s must be 1/120 s")

    runtime_joint_names = _string_array(
        runtime.get("joint_names"), "runtime audit joint_names"
    )
    runtime_geometry_raw = runtime.get("joint_geometry")
    if not isinstance(runtime_geometry_raw, list) or len(runtime_geometry_raw) != len(
        _CANONICAL_JOINTS
    ):
        raise ContractError("runtime audit joint_geometry must contain all 18 joints")
    runtime_geometry: list[dict[str, Any]] = []
    for index, raw in enumerate(runtime_geometry_raw):
        record = _mapping(raw, f"runtime joint_geometry[{index}]")
        _exact_fields(
            record,
            name="runtime joint geometry",
            required={
                "canonical_joint",
                "runtime_joint",
                "articulation_index",
                "axis",
                "range_kind",
                "lower_limit_rad",
                "upper_limit_rad",
            },
        )
        canonical = _identifier(
            record["canonical_joint"], "runtime joint canonical_joint"
        )
        runtime_name = record["runtime_joint"]
        if not isinstance(runtime_name, str) or not runtime_name:
            raise ContractError("runtime joint runtime_joint must be a non-empty string")
        articulation_index = record["articulation_index"]
        if (
            isinstance(articulation_index, bool)
            or not isinstance(articulation_index, int)
            or not 0 <= articulation_index < len(runtime_joint_names)
        ):
            raise ContractError("runtime joint articulation_index is invalid")
        axis = record["axis"]
        if axis not in {"X", "Y", "Z"}:
            raise ContractError("runtime joint axis must be X, Y, or Z")
        kind, lower, upper = _range(
            record,
            name=f"runtime joint {canonical}",
            lower_field="lower_limit_rad",
            upper_field="upper_limit_rad",
        )
        runtime_geometry.append(
            {
                "canonical_joint": canonical,
                "runtime_joint": runtime_name,
                "articulation_index": articulation_index,
                "axis": axis,
                "range_kind": kind,
                "lower_limit_rad": lower,
                "upper_limit_rad": upper,
            }
        )
    runtime_mapping_valid = (
        tuple(record["canonical_joint"] for record in runtime_geometry)
        == _CANONICAL_JOINTS
        and len({record["runtime_joint"] for record in runtime_geometry})
        == len(_CANONICAL_JOINTS)
        and len({record["articulation_index"] for record in runtime_geometry})
        == len(_CANONICAL_JOINTS)
        and all(
            runtime_joint_names[record["articulation_index"]]
            == record["runtime_joint"]
            for record in runtime_geometry
        )
    )

    body_names = _string_array(runtime.get("body_names"), "runtime audit body_names")
    bodies = _mapping(runtime.get("body_properties"), "runtime audit body_properties")
    runtime_mass = _number(bodies.get("total_mass_kg"), "runtime audit total_mass_kg")
    if runtime_mass <= 0.0:
        raise ContractError("runtime audit total_mass_kg must be positive")
    masses = _numeric_array(bodies.get("mass_kg"), "runtime audit mass_kg")
    inertias = _numeric_array(
        bodies.get("inertia_kg_m2_matrix"), "runtime audit inertia_kg_m2_matrix"
    )
    com_poses = _numeric_array(
        bodies.get("com_pose_xyz_xyzw"), "runtime audit com_pose_xyz_xyzw"
    )
    aggregate_com = _numeric_vector(
        bodies.get("aggregate_com_body_m"),
        "runtime audit aggregate_com_body_m",
        length=3,
    )
    candidate_mass = candidate.simulation_physics.get("mass", {})
    absolute_mass = isinstance(candidate_mass, Mapping) and {
        "target_total_mass_kg",
        "reference_planar_com_xy_m",
        "reference_joint_position_rad",
        "reference_root_orientation_xyzw",
    }.issubset(candidate_mass)
    mass_application = runtime.get("mass_profile_application")
    if absolute_mass:
        application = _mapping(
            mass_application, "runtime audit mass_profile_application"
        )
        target_total = _number(
            application.get("target_total_mass_kg"),
            "runtime mass application target_total_mass_kg",
        )
        achieved_total = _number(
            application.get("achieved_total_mass_kg"),
            "runtime mass application achieved_total_mass_kg",
        )
        target_xy = _numeric_vector(
            application.get("target_planar_com_xy_m"),
            "runtime mass application target_planar_com_xy_m",
            length=2,
        )
        achieved_com = _numeric_vector(
            application.get("achieved_whole_com_root_m"),
            "runtime mass application achieved_whole_com_root_m",
            length=3,
        )
        reference_pose = _mapping(
            application.get("reference_pose"),
            "runtime mass application reference_pose",
        )
        mass_profile_application_pass = bool(
            application.get("mode") == "absolute"
            and math.isclose(
                target_total,
                float(candidate_mass["target_total_mass_kg"]),
                rel_tol=0.0,
                abs_tol=1.0e-6,
            )
            and np.allclose(
                target_xy,
                candidate_mass["reference_planar_com_xy_m"],
                rtol=0.0,
                atol=1.0e-6,
            )
            and math.isclose(
                achieved_total, runtime_mass, rel_tol=0.0, abs_tol=1.0e-6
            )
            and np.allclose(
                achieved_com[:2], aggregate_com[:2], rtol=0.0, atol=1.0e-6
            )
            and math.isclose(
                achieved_total, target_total, rel_tol=0.0, abs_tol=1.0e-6
            )
            and np.allclose(achieved_com[:2], target_xy, rtol=0.0, atol=1.0e-6)
            and reference_pose.get("joint_position_rad")
            == candidate_mass["reference_joint_position_rad"]
            and reference_pose.get("root_orientation_xyzw")
            == candidate_mass["reference_root_orientation_xyzw"]
        )
    else:
        mass_profile_application_pass = mass_application is None
    body_count = len(body_names)
    inertia_com_pass = (
        masses.shape == (1, body_count)
        and inertias.shape == (1, body_count, 9)
        and com_poses.shape == (1, body_count, 7)
    )
    if inertia_com_pass:
        inertia_matrices = inertias[0].reshape(body_count, 3, 3)
        symmetric = np.allclose(
            inertia_matrices,
            np.swapaxes(inertia_matrices, 1, 2),
            rtol=0.0,
            atol=1.0e-9,
        )
        positive_definite = all(
            float(np.min(np.linalg.eigvalsh(matrix))) > 0.0
            for matrix in inertia_matrices
        )
        quaternion_norms = np.linalg.norm(com_poses[0, :, 3:7], axis=1)
        inertia_com_pass = bool(
            np.all(masses > 0.0)
            and math.isclose(
                float(np.sum(masses)), runtime_mass, rel_tol=0.0, abs_tol=1.0e-6
            )
            and symmetric
            and positive_definite
            and np.allclose(quaternion_norms, 1.0, rtol=0.0, atol=1.0e-6)
        )

    audit_value = runtime_trace.arrays.get("audit_value")
    if (
        audit_value is None
        or audit_value.size == 0
        or not np.allclose(audit_value, runtime_mass, rtol=0.0, atol=1.0e-6)
    ):
        raise ContractError("audit trace audit_value does not match runtime total mass")

    collision_raw = runtime.get("collision_geometry")
    if not isinstance(collision_raw, list) or not collision_raw:
        raise ContractError("runtime audit collision_geometry must be non-empty")
    collision_paths: list[str] = []
    for index, raw in enumerate(collision_raw):
        collision = _mapping(raw, f"runtime collision_geometry[{index}]")
        path = collision.get("prim_path")
        if not isinstance(path, str) or not path.startswith("/"):
            raise ContractError("runtime collision prim_path must be absolute")
        has_api = collision.get("has_collision_api")
        if not isinstance(has_api, bool):
            raise ContractError("runtime collision has_collision_api must be boolean")
        if has_api:
            collision_paths.append(path)

    sensors = _mapping(runtime.get("contact_sensors"), "runtime audit contact_sensors")
    foot_sensor = _mapping(sensors.get("foot"), "runtime audit foot contact sensor")
    foot_names = foot_sensor.get("body_names")
    foot_count = foot_sensor.get("body_count")
    if not isinstance(foot_names, list) or not all(isinstance(item, str) for item in foot_names):
        raise ContractError("runtime audit foot body_names must be an array of strings")
    if isinstance(foot_count, bool) or not isinstance(foot_count, int):
        raise ContractError("runtime audit foot body_count must be an integer")

    physical_path, physical_binding = _file_binding(
        root, binding["physical_measurements"], "physical audit measurements"
    )
    physical = _json(physical_path, "physical audit measurements")
    _exact_fields(
        physical,
        name="physical audit measurements",
        required={
            "schema_version",
            "units",
            "frames",
            "joint_geometry",
            "encoder_observations",
            "mass_measurements_kg",
            "mass_instrument_uncertainty_kg",
            "planar_com_measurements_m",
            "com_instrument_uncertainty_m",
            "collision_body_names",
            "imu_rest_orientations",
        },
    )
    if physical["schema_version"] != 2 or isinstance(physical["schema_version"], bool):
        raise ContractError("physical audit schema_version must be 2")
    units = _mapping(physical["units"], "physical audit units")
    frames = _mapping(physical["frames"], "physical audit frames")
    units_pass = units == _EXPECTED_AUDIT_UNITS
    frames_pass = frames == _EXPECTED_AUDIT_FRAMES

    physical_geometry_raw = physical["joint_geometry"]
    if not isinstance(physical_geometry_raw, list) or len(physical_geometry_raw) != len(
        _CANONICAL_JOINTS
    ):
        raise ContractError("physical joint_geometry must contain all 18 joints")
    physical_geometry: list[dict[str, Any]] = []
    for index, raw in enumerate(physical_geometry_raw):
        record = _mapping(raw, f"physical joint_geometry[{index}]")
        _exact_fields(
            record,
            name="physical joint geometry",
            required={
                "canonical_joint",
                "runtime_joint",
                "expected_axis",
                "range_kind",
                "mechanical_min_rad",
                "mechanical_max_rad",
                "range_uncertainty_rad",
            },
        )
        canonical = _identifier(
            record["canonical_joint"], "physical joint canonical_joint"
        )
        runtime_name = record["runtime_joint"]
        if not isinstance(runtime_name, str) or not runtime_name:
            raise ContractError("physical joint runtime_joint must be a non-empty string")
        axis = record["expected_axis"]
        if axis not in {"X", "Y", "Z"}:
            raise ContractError("physical joint expected_axis must be X, Y, or Z")
        kind, lower, upper = _range(
            record,
            name=f"physical joint {canonical}",
            lower_field="mechanical_min_rad",
            upper_field="mechanical_max_rad",
        )
        physical_geometry.append(
            {
                "canonical_joint": canonical,
                "runtime_joint": runtime_name,
                "expected_axis": axis,
                "range_kind": kind,
                "mechanical_min_rad": lower,
                "mechanical_max_rad": upper,
                "range_uncertainty_rad": _number(
                    record["range_uncertainty_rad"],
                    f"physical joint {canonical} range_uncertainty_rad",
                    nonnegative=True,
                ),
            }
        )
    physical_order_valid = (
        tuple(record["canonical_joint"] for record in physical_geometry)
        == _CANONICAL_JOINTS
        and len({record["runtime_joint"] for record in physical_geometry})
        == len(_CANONICAL_JOINTS)
    )
    joint_order_pass = runtime_mapping_valid and physical_order_valid and all(
        physical_record["runtime_joint"] == runtime_record["runtime_joint"]
        for physical_record, runtime_record in zip(
            physical_geometry, runtime_geometry, strict=True
        )
    )
    joint_axis_pass = joint_order_pass and all(
        physical_record["expected_axis"] == runtime_record["axis"]
        for physical_record, runtime_record in zip(
            physical_geometry, runtime_geometry, strict=True
        )
    )
    range_results: list[bool] = []
    for physical_record, runtime_record in zip(
        physical_geometry, runtime_geometry, strict=True
    ):
        same_kind = physical_record["range_kind"] == runtime_record["range_kind"]
        if not same_kind:
            range_results.append(False)
        elif physical_record["range_kind"] == "continuous":
            range_results.append(True)
        else:
            tolerance = physical_record["range_uncertainty_rad"]
            range_results.append(
                abs(
                    physical_record["mechanical_min_rad"]
                    - runtime_record["lower_limit_rad"]
                )
                <= tolerance
                and abs(
                    physical_record["mechanical_max_rad"]
                    - runtime_record["upper_limit_rad"]
                )
                <= tolerance
            )
    mechanical_range_pass = joint_order_pass and all(range_results)

    raw_encoders = physical["encoder_observations"]
    if not isinstance(raw_encoders, list) or len(raw_encoders) != len(_MAIN_JOINTS):
        raise ContractError("encoder_observations must contain all six main joints")
    counts_mapping = candidate.hardware_mapping.get("encoder_counts_per_rev", {})
    zero_mapping = candidate.hardware_mapping.get("encoder_zero_count", {})
    sign_mapping = candidate.hardware_mapping.get("encoder_sign", {})
    mapping_complete = all(
        isinstance(mapping, Mapping) and set(mapping) == set(_MAIN_JOINTS)
        for mapping in (counts_mapping, zero_mapping, sign_mapping)
    )
    encoder_scale_zero_results: list[bool] = []
    joint_sign_results: list[bool] = []
    observed_encoder_order: list[str] = []
    for index, raw in enumerate(raw_encoders):
        observation = _mapping(raw, f"encoder_observations[{index}]")
        _exact_fields(
            observation,
            name="encoder observation",
            required={
                "joint",
                "raw_start_count",
                "raw_end_count",
                "physical_delta_rad",
                "observed_counts_per_rev",
                "counts_per_rev_uncertainty",
                "observed_zero_count",
                "zero_count_uncertainty",
                "angle_uncertainty_rad",
            },
        )
        joint = _identifier(observation["joint"], "encoder observation joint")
        observed_encoder_order.append(joint)
        raw_start = _number(observation["raw_start_count"], "raw_start_count")
        raw_end = _number(observation["raw_end_count"], "raw_end_count")
        physical_delta = _number(
            observation["physical_delta_rad"], "physical_delta_rad"
        )
        observed_counts = _number(
            observation["observed_counts_per_rev"],
            "observed_counts_per_rev",
        )
        if observed_counts <= 0.0:
            raise ContractError("observed_counts_per_rev must be positive")
        counts_uncertainty = _number(
            observation["counts_per_rev_uncertainty"],
            "counts_per_rev_uncertainty",
            nonnegative=True,
        )
        observed_zero = _number(
            observation["observed_zero_count"], "observed_zero_count"
        )
        zero_uncertainty = _number(
            observation["zero_count_uncertainty"],
            "zero_count_uncertainty",
            nonnegative=True,
        )
        angle_uncertainty = _number(
            observation["angle_uncertainty_rad"],
            "angle_uncertainty_rad",
            nonnegative=True,
        )
        if mapping_complete and joint in _MAIN_JOINTS:
            candidate_counts = float(counts_mapping[joint])
            candidate_zero = float(zero_mapping[joint])
            candidate_sign = int(sign_mapping[joint])
            mapped_magnitude = (
                abs(raw_end - raw_start) * 2.0 * math.pi / candidate_counts
            )
            encoder_scale_zero_results.append(
                abs(candidate_counts - observed_counts) <= counts_uncertainty
                and abs(candidate_zero - observed_zero) <= zero_uncertainty
                and abs(mapped_magnitude - abs(physical_delta)) <= angle_uncertainty
            )
            joint_sign_results.append(
                raw_end != raw_start
                and physical_delta != 0.0
                and (raw_end - raw_start) * candidate_sign * physical_delta > 0.0
            )
        else:
            encoder_scale_zero_results.append(False)
            joint_sign_results.append(False)
    encoder_order_valid = tuple(observed_encoder_order) == _MAIN_JOINTS
    encoder_scale_zero_pass = (
        mapping_complete and encoder_order_valid and all(encoder_scale_zero_results)
    )
    joint_sign_pass = mapping_complete and encoder_order_valid and all(joint_sign_results)

    raw_masses = physical["mass_measurements_kg"]
    if not isinstance(raw_masses, list) or len(raw_masses) < 3:
        raise ContractError("mass_measurements_kg requires at least three measurements")
    masses = np.asarray(
        [_number(item, "mass_measurements_kg") for item in raw_masses], dtype=float
    )
    if np.any(masses <= 0.0):
        raise ContractError("mass_measurements_kg must be positive")
    mass_uncertainty = _number(
        physical["mass_instrument_uncertainty_kg"],
        "mass_instrument_uncertainty_kg",
        nonnegative=True,
    )
    mass_mean = float(np.mean(masses))
    mass_std = float(np.std(masses))
    mass_tolerance = max(mass_uncertainty, 2.0 * mass_std)
    mass_error = abs(runtime_mass - mass_mean)
    mass_pass = mass_error <= mass_tolerance

    planar_com = _numeric_array(
        physical["planar_com_measurements_m"], "planar_com_measurements_m"
    )
    if planar_com.ndim != 2 or planar_com.shape[0] < 3 or planar_com.shape[1] != 2:
        raise ContractError(
            "planar_com_measurements_m requires at least three [x, y] measurements"
        )
    com_uncertainty = _number(
        physical["com_instrument_uncertainty_m"],
        "com_instrument_uncertainty_m",
        nonnegative=True,
    )
    com_mean = np.mean(planar_com, axis=0)
    com_std = np.std(planar_com, axis=0)
    com_tolerance = np.maximum(com_uncertainty, 2.0 * com_std)
    com_error = np.abs(aggregate_com[:2] - com_mean)
    planar_com_pass = bool(np.all(com_error <= com_tolerance))

    expected_collision_bodies = _string_array(
        physical["collision_body_names"], "collision_body_names"
    )
    mandatory_collision_bodies = {"base_link", *EXPECTED_FOOT_BODY_NAMES}
    collision_geometry_pass = (
        set(expected_collision_bodies) == mandatory_collision_bodies
        and set(expected_collision_bodies).issubset(body_names)
        and all(
            any(body_name in PurePosixPath(path).parts for path in collision_paths)
            for body_name in expected_collision_bodies
        )
    )

    orientations = physical["imu_rest_orientations"]
    if not isinstance(orientations, list) or len(orientations) < 2:
        raise ContractError("imu_rest_orientations requires at least two known poses")
    imu_errors: list[float] = []
    labels: set[str] = set()
    for index, raw in enumerate(orientations):
        orientation = _mapping(raw, f"imu_rest_orientations[{index}]")
        _exact_fields(
            orientation,
            name="IMU rest orientation",
            required={"label", "measured_gravity", "expected_gravity"},
        )
        label = _identifier(orientation["label"], "IMU rest orientation label")
        if label in labels:
            raise ContractError("IMU rest orientation labels must be unique")
        labels.add(label)
        measured = _numeric_vector(
            orientation["measured_gravity"], "measured_gravity", length=3
        )
        expected = _numeric_vector(
            orientation["expected_gravity"], "expected_gravity", length=3
        )
        measured_norm = float(np.linalg.norm(measured))
        expected_norm = float(np.linalg.norm(expected))
        if measured_norm <= 0.0 or expected_norm <= 0.0:
            raise ContractError("IMU gravity vectors must be nonzero")
        cosine = float(np.dot(measured, expected) / (measured_norm * expected_norm))
        imu_errors.append(math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0)))))
    imu_mount_pass = max(imu_errors) <= 5.0

    contact = runtime_trace.arrays.get("contact_force_n")
    max_contact_force = (
        float(np.max(contact)) if contact is not None and contact.size else 0.0
    )
    contact_sensor_pass = (
        foot_count == len(EXPECTED_FOOT_BODY_NAMES)
        and set(foot_names) == set(EXPECTED_FOOT_BODY_NAMES)
        and max_contact_force > 0.05
    )
    checks = {
        "units_pass": units_pass,
        "frames_pass": frames_pass,
        "joint_order_pass": joint_order_pass,
        "joint_axis_pass": joint_axis_pass,
        "encoder_scale_zero_pass": encoder_scale_zero_pass,
        "joint_sign_pass": joint_sign_pass,
        "mechanical_range_pass": mechanical_range_pass,
        "mass_pass": mass_pass,
        "mass_profile_application_pass": mass_profile_application_pass,
        "inertia_com_pass": inertia_com_pass,
        "planar_com_pass": planar_com_pass,
        "collision_geometry_pass": collision_geometry_pass,
        "imu_mount_pass": imu_mount_pass,
        "contact_sensor_pass": contact_sensor_pass,
    }
    return {
        "checks": checks,
        "runtime_trace_sha256": runtime_trace.manifest.provenance["trace_sha256"],
        "runtime_metadata_sha256": runtime_trace.metadata_sha256,
        "runtime_audit_sha256": runtime_binding["sha256"],
        "runtime_audit_payload_sha256": trace_runtime_hash,
        "physical_measurements_sha256": physical_binding["sha256"],
        "joint_geometry": {
            "canonical_order": list(_CANONICAL_JOINTS),
            "runtime_joint_order": [
                record["runtime_joint"] for record in runtime_geometry
            ],
        },
        "encoder_mapping": {
            "observed_joint_order": observed_encoder_order,
        },
        "mass": {
            "runtime_kg": runtime_mass,
            "real_mean_kg": mass_mean,
            "real_std_kg": mass_std,
            "instrument_uncertainty_kg": mass_uncertainty,
            "tolerance_kg": mass_tolerance,
            "absolute_error_kg": mass_error,
        },
        "planar_com": {
            "runtime_xy_m": aggregate_com[:2].tolist(),
            "real_mean_xy_m": com_mean.tolist(),
            "real_std_xy_m": com_std.tolist(),
            "instrument_uncertainty_m": com_uncertainty,
            "tolerance_xy_m": com_tolerance.tolist(),
            "absolute_error_xy_m": com_error.tolist(),
        },
        "imu_max_error_deg": max(imu_errors),
        "contact_max_force_n": max_contact_force,
    }


def _verify_sweep_for_holdout(
    root: Path,
    raw_binding: Any,
    *,
    holdout: Mapping[str, Any],
    internal: Mapping[str, Any],
    baseline_profile: CalibrationProfileV1,
    effort_limit_changed: bool,
    known_load_traces: list[LoadedTrace],
    audit_artifact_sha256: str,
    audit_report_sha256: str,
) -> bool:
    binding = _mapping(raw_binding, "actuator sweep")
    _exact_fields(
        binding,
        name="actuator sweep",
        required={"path", "results_sha256"},
    )
    sweep_root = _artifact(root, binding["path"], "actuator sweep")
    if not sweep_root.is_dir():
        raise ContractError("actuator sweep path must name a directory")
    results_path = sweep_root / "results.json"
    if not results_path.is_file():
        raise ContractError("actuator sweep results.json is missing")
    expected_results_hash = _sha(
        binding["results_sha256"], "actuator sweep results_sha256"
    )
    if sha256_file(results_path) != expected_results_hash:
        raise ContractError("actuator sweep results hash mismatch")
    results = _json(results_path, "actuator sweep results")
    required_result_fields = {
        "schema_version",
        "sweep_sha256",
        "sweep_mode",
        "scenario_id",
        "scene_mode",
        "provenance_sha256",
        "candidate_count",
        "candidates",
        "counts",
    }
    _exact_fields(results, name="actuator sweep results", required=required_result_fields)
    if results["schema_version"] != 1 or isinstance(results["schema_version"], bool):
        raise ContractError("actuator sweep results schema_version must be 1")
    scenario = internal["scenario"]
    if results["scenario_id"] != scenario.scenario_id:
        raise ContractError("actuator sweep scenario does not match its holdout")
    scenario_snapshot = load_scenario(sweep_root / "scenario.json")
    if scenario_snapshot.to_dict() != scenario.to_dict():
        raise ContractError("actuator sweep scenario snapshot does not match its holdout")
    if results["sweep_mode"] not in {"one-factor", "coarse-grid"}:
        raise ContractError("actuator sweep mode is unsupported")
    expected_scene_mode = {
        "fixed_base": "fixed-base",
        "free_root": "free-root",
    }.get(scenario.scene_mode)
    if results["scene_mode"] != expected_scene_mode:
        raise ContractError("actuator sweep scene mode does not match its holdout")

    provenance = _json(sweep_root / "provenance.json", "actuator sweep provenance")
    if sha256_json(provenance) != _sha(
        results["provenance_sha256"], "actuator sweep provenance_sha256"
    ):
        raise ContractError("actuator sweep provenance hash mismatch")
    required_provenance = {
        "git_sha",
        "asset_sha256",
        "config_sha256",
        "redrhex_module_path",
        "redrhex_module_sha256",
        "isaaclab_version",
        "isaacsim_version",
        "characterization_runner_sha256",
        "sweep_runner_sha256",
        "runtime_bundle_sha256",
        "real_trace_sha256",
        "real_metadata_sha256",
        "replay_initial_state_sha256",
        "known_load_trace_sha256",
        "known_load_metadata_sha256",
        "audit_artifact_sha256",
        "audit_report_sha256",
        "scene_mode",
        "headless",
        "seed",
        "device",
        "sweep_runner_schema_version",
    }
    if set(provenance) != required_provenance:
        raise ContractError("actuator sweep provenance has missing or unknown fields")
    for field in (
        "asset_sha256",
        "config_sha256",
        "redrhex_module_sha256",
        "characterization_runner_sha256",
        "sweep_runner_sha256",
        "runtime_bundle_sha256",
        "replay_initial_state_sha256",
        "audit_artifact_sha256",
        "audit_report_sha256",
    ):
        _sha(provenance[field], f"actuator sweep provenance {field}")
    if provenance["redrhex_module_sha256"] != provenance["config_sha256"]:
        raise ContractError("actuator sweep RedRhex module/config hash mismatch")
    if not isinstance(provenance["redrhex_module_path"], str) or not Path(
        provenance["redrhex_module_path"]
    ).is_absolute():
        raise ContractError("actuator sweep RedRhex module path must be absolute")
    for field in ("isaaclab_version", "isaacsim_version"):
        if not isinstance(provenance[field], str) or not provenance[field]:
            raise ContractError(f"actuator sweep provenance {field} is invalid")
    if provenance["audit_artifact_sha256"] != audit_artifact_sha256:
        raise ContractError("actuator sweep pre-fit audit artifact mismatch")
    if provenance["audit_report_sha256"] != audit_report_sha256:
        raise ContractError("actuator sweep pre-fit audit report mismatch")
    git_sha = provenance["git_sha"]
    if not isinstance(git_sha, str) or len(git_sha) not in {40, 64}:
        raise ContractError("actuator sweep provenance git_sha is invalid")
    real_traces = internal["real_traces"]
    if len(real_traces) != 1:
        raise ContractError("an actuator sweep must bind exactly one real holdout episode")
    real_trace = real_traces[0]
    if provenance["real_trace_sha256"] != real_trace.manifest.provenance["trace_sha256"]:
        raise ContractError("actuator sweep real reference trace mismatch")
    if provenance["real_metadata_sha256"] != real_trace.metadata_sha256:
        raise ContractError("actuator sweep real reference metadata mismatch")
    replay = load_replay_schedule(
        real_trace.directory,
        scenario,
        steps=scenario_step_count(scenario),
    )
    if provenance["replay_initial_state_sha256"] != replay.initial_state_sha256:
        raise ContractError("actuator sweep replay initial-state provenance mismatch")
    if provenance["scene_mode"] != results["scene_mode"]:
        raise ContractError("actuator sweep scene mode provenance mismatch")
    known_load_identity = (
        provenance["known_load_trace_sha256"],
        provenance["known_load_metadata_sha256"],
    )
    if effort_limit_changed and known_load_traces:
        expected_known_loads = {
            (trace.manifest.provenance["trace_sha256"], trace.metadata_sha256)
            for trace in known_load_traces
        }
        if known_load_identity not in expected_known_loads:
            raise ContractError(
                "actuator sweep known-load provenance does not match a calibration condition"
            )
    elif not effort_limit_changed and any(value is not None for value in known_load_identity):
        raise ContractError(
            "actuator sweep must not claim unused known-load provenance"
        )

    raw_candidates = results["candidates"]
    candidate_count = results["candidate_count"]
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count < 1
        or not isinstance(raw_candidates, list)
        or len(raw_candidates) != candidate_count
    ):
        raise ContractError("actuator sweep candidate_count is invalid")
    counts = _mapping(results["counts"], "actuator sweep counts")
    expected_count_fields = {"cached", "completed", "failed", "generated", "pending"}
    if set(counts) != expected_count_fields:
        raise ContractError("actuator sweep counts have missing or unknown fields")
    actual_counts = {
        name: sum(
            isinstance(item, Mapping) and item.get("status") == name
            for item in raw_candidates
        )
        for name in expected_count_fields
    }
    if dict(counts) != actual_counts:
        raise ContractError("actuator sweep counts do not match candidate statuses")
    if any(
        not isinstance(item, Mapping)
        or item.get("status") not in {"completed", "cached"}
        for item in raw_candidates
    ):
        raise ContractError("actuator sweep must complete every bounded candidate")
    index_payload = _json(sweep_root / "index.json", "actuator sweep index")
    expected_index = dict(results)
    expected_index.pop("counts")
    if dict(index_payload) != expected_index:
        raise ContractError("actuator sweep index and results disagree")

    candidate_profiles: list[CalibrationProfileV1] = []
    candidate_inside = False
    holdout_sim_hash = holdout["sim_trace_sha256"]
    observed_sim_hashes: set[str] = set()
    for offset, raw_candidate in enumerate(raw_candidates, start=1):
        item = _mapping(raw_candidate, f"actuator sweep candidate {offset}")
        if item.get("index") != offset:
            raise ContractError("actuator sweep candidate indices must be contiguous")
        status_path = _artifact(sweep_root, item.get("status_file"), "candidate status")
        status = _json(status_path, "candidate status")
        if status.get("schema_version") != 1:
            raise ContractError("candidate status schema_version must be 1")
        for field, value in item.items():
            if field == "status_file":
                continue
            if status.get(field) != value:
                raise ContractError(f"candidate status field {field} disagrees with results")
        profile_path = _artifact(sweep_root, item.get("profile"), "candidate profile")
        candidate_profile = load_profile(profile_path)
        profile_sha = sha256_json(candidate_profile.to_dict())
        if profile_sha != _sha(item.get("profile_sha256"), "candidate profile_sha256"):
            raise ContractError("candidate profile hash mismatch")
        candidate_profiles.append(candidate_profile)
        if item.get("cache_key") != candidate_cache_key(
            candidate_profile, scenario, provenance=provenance
        ):
            raise ContractError("candidate cache key mismatch")
        run_output = _artifact(sweep_root, item.get("run_output"), "candidate run output")
        if not run_output.is_dir():
            raise ContractError("candidate run output must be a directory")
        sim = load_trace(
            run_output,
            scenario=scenario,
            profile=candidate_profile,
            expected_metadata_sha256=_sha(
                item.get("metadata_sha256"), "candidate metadata_sha256"
            ),
        )
        if sim.manifest.source != "sim":
            raise ContractError("candidate run trace must have source='sim'")
        sim_hash = sim.manifest.provenance["trace_sha256"]
        if sim_hash != _sha(item.get("trace_sha256"), "candidate trace_sha256"):
            raise ContractError("candidate trace hash mismatch")
        observed_sim_hashes.add(sim_hash)
        for field in (
            "git_sha",
            "asset_sha256",
            "config_sha256",
            "redrhex_module_path",
            "redrhex_module_sha256",
            "isaaclab_version",
            "isaacsim_version",
            "characterization_runner_sha256",
            "runtime_bundle_sha256",
        ):
            if sim.manifest.metadata.get(field) != provenance[field]:
                raise ContractError(f"candidate runtime provenance {field} mismatch")
        run_results = _json(run_output / "results.json", "candidate results")
        expected_run = {
            "schema_version": 1,
            "scenario_id": scenario.scenario_id,
            "mode": results["scene_mode"],
            "trace_sha256": sim_hash,
            "profile_id": candidate_profile.profile_id,
        }
        if any(run_results.get(field) != value for field, value in expected_run.items()):
            raise ContractError("candidate results do not match the sweep artifact")
        runtime_audit_name = run_results.get("runtime_audit")
        runtime_audit = _artifact(run_output, runtime_audit_name, "candidate runtime audit")
        _json(runtime_audit, "candidate runtime audit")
        comparison_path = _artifact(
            sweep_root, item.get("comparison"), "candidate comparison"
        )
        if comparison_path.parent != run_output:
            raise ContractError("candidate comparison must belong to its run output")
        comparison = _json(comparison_path, "candidate comparison")
        if sha256_json(comparison) != _sha(
            item.get("comparison_sha256"), "candidate comparison_sha256"
        ):
            raise ContractError("candidate comparison hash mismatch")
        recomputed = compare_traces(real_trace, sim, scenario=scenario)
        expected_comparison = {
            **recomputed,
            "real_trace_sha256": real_trace.manifest.provenance["trace_sha256"],
            "sim_trace_sha256": sim_hash,
        }
        if dict(comparison) != expected_comparison:
            raise ContractError("candidate comparison does not match bound traces")
        if item.get("metrics") != recomputed["subsystems"]:
            raise ContractError("candidate metrics do not match its comparison")
        passes = True
        metrics = compute_subsystem_metrics(scenario, sim)
        for metric_path, expected in holdout["metrics"].items():
            value, _, _ = _lookup(metrics, metric_path)
            if abs(value - expected["real_mean"]) > expected["tolerance"]:
                passes = False
        candidate_inside |= passes
    if holdout_sim_hash not in observed_sim_hashes:
        raise ContractError("holdout simulator artifact is absent from its complete sweep")
    validate_sweep_candidates(
        baseline_profile,
        candidate_profiles,
        scenario,
        sweep_mode=results["sweep_mode"],
    )
    expected_sweep_sha = sha256_json(
        {
            "schema_version": 1,
            "sweep_mode": results["sweep_mode"],
            "scenario": scenario.to_dict(),
            "candidate_profiles": [profile.to_dict() for profile in candidate_profiles],
            "provenance": dict(provenance),
        }
    )
    if _sha(results["sweep_sha256"], "actuator sweep_sha256") != expected_sweep_sha:
        raise ContractError("actuator sweep identity hash mismatch")
    return candidate_inside


def evaluate_promotion(
    profile: CalibrationProfileV1,
    evidence: Mapping[str, Any],
    *,
    artifact_root: str | Path,
) -> dict[str, Any]:
    """Resolve and evaluate immutable calibration artifacts without promoting them."""

    candidate = profile.validate()
    data = _mapping(evidence, "validation evidence")
    _exact_fields(
        data,
        name="validation evidence",
        required={
            "schema_version",
            "candidate_profile_sha256",
            "baseline_profile",
            "audit_artifact",
            "conditions",
            "actuator_sweeps",
        },
    )
    if data["schema_version"] != 1 or isinstance(data["schema_version"], bool):
        raise ContractError("validation evidence schema_version must be 1")
    expected_profile_hash = sha256_json(candidate.to_dict())
    if _sha(data["candidate_profile_sha256"], "candidate profile hash") != expected_profile_hash:
        raise ContractError("candidate profile hash mismatch")
    root = Path(artifact_root).resolve()
    if not root.is_dir():
        raise ContractError("artifact_root must be an existing directory")

    baseline_path, _ = _file_binding(root, data["baseline_profile"], "baseline profile")
    baseline = load_profile(baseline_path)
    _validate_identifiable_changes(baseline, candidate)
    fitted = _changed_subsystems(baseline, candidate)
    effort_limit_changed = _main_effort_limit_changed(baseline, candidate)
    if not fitted:
        raise ContractError("candidate profile has no fitted subsystem changes from baseline")

    audit_report = _derive_audit(root, data["audit_artifact"], candidate)
    audit = audit_report["checks"]
    global_failures = [
        f"audit.{field} failed" for field in sorted(_AUDIT_FIELDS) if not audit[field]
    ]

    raw_conditions = data["conditions"]
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise ContractError("conditions must be a non-empty array")
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        subsystem: {"calibration": [], "holdout": []} for subsystem in fitted
    }
    episode_ids: set[str] = set()
    condition_ids: set[str] = set()
    real_source_roles: dict[str, str] = {}
    condition_internal: dict[str, dict[str, Any]] = {}

    for index, raw in enumerate(raw_conditions):
        condition = _mapping(raw, f"conditions[{index}]")
        _exact_fields(
            condition,
            name="condition",
            required={"condition_id", "subsystem", "role", "real_episodes", "metrics"},
            optional={"held_out_by", "sim_artifact"},
        )
        condition_id = _identifier(condition["condition_id"], "condition_id")
        if condition_id in condition_ids:
            raise ContractError("condition_id values must be unique")
        condition_ids.add(condition_id)
        subsystem = _identifier(condition["subsystem"], "condition subsystem")
        if subsystem not in grouped:
            raise ContractError(f"condition references a subsystem not changed by the candidate: {subsystem}")
        role = condition["role"]
        if role not in {"calibration", "holdout"}:
            raise ContractError("condition role must be calibration or holdout")
        raw_episodes = condition["real_episodes"]
        if not isinstance(raw_episodes, list) or not raw_episodes:
            raise ContractError(f"condition {condition_id} real_episodes must be non-empty")

        loaded_real: list[LoadedTrace] = []
        coordinates: dict[str, Any] | None = None
        scenario = None
        repetitions = 0
        for episode_index, raw_episode in enumerate(raw_episodes):
            episode = _mapping(raw_episode, f"condition {condition_id} episode")
            episode_id = _identifier(episode.get("episode_id"), "episode_id")
            if episode_id in episode_ids:
                raise ContractError("episode_id values must be unique")
            episode_ids.add(episode_id)
            loaded, _ = _trace_binding(
                root,
                episode,
                f"condition {condition_id} real episode {episode_index}",
                source="real",
            )
            source_hash = loaded.manifest.provenance["source_sha256"]
            previous_role = real_source_roles.get(source_hash)
            if previous_role is not None and previous_role != role:
                raise ContractError("calibration and holdout real artifacts must be disjoint")
            if previous_role is not None:
                raise ContractError("real raw sources must be unique across conditions")
            real_source_roles[source_hash] = role
            episode_scenario = load_scenario(loaded.manifest.scenario_id)
            _validate_condition_trace_metadata(loaded, episode_scenario)
            validate_real_trace_provenance(loaded, episode_scenario)
            if episode_scenario.split != role:
                raise ContractError(
                    f"condition {condition_id} role does not match scenario split"
                )
            if not _scenario_supports(subsystem, episode_scenario.subsystem):
                raise ContractError(
                    f"condition {condition_id} scenario does not measure {subsystem}"
                )
            if scenario is None:
                scenario = episode_scenario
            elif scenario.to_dict() != episode_scenario.to_dict():
                raise ContractError("all real episodes in one condition must share a scenario")
            episode_coordinates = _condition_coordinates(loaded, episode_scenario)
            if coordinates is None:
                coordinates = episode_coordinates
            elif coordinates != episode_coordinates:
                raise ContractError("all real episodes in one condition must share coordinates")
            episode_metrics = compute_subsystem_metrics(episode_scenario, loaded)
            episode_repetitions = _repeat_count(episode_metrics, loaded)
            if episode_repetitions < 1:
                raise ContractError(
                    f"condition {condition_id} trace does not expose authenticated repetitions"
                )
            repetitions += episode_repetitions
            loaded_real.append(loaded)
        assert scenario is not None and coordinates is not None

        subsystem_failures: list[str] = []
        if repetitions < 3:
            subsystem_failures.append(
                f"{subsystem}.{condition_id} requires at least three real repetitions"
            )
        raw_metrics = _mapping(condition["metrics"], f"condition {condition_id} metrics")
        clean_condition: dict[str, Any] = {
            "condition_id": condition_id,
            "scenario_id": scenario.scenario_id,
            "coordinates": coordinates,
            "real_trace_sha256": [
                trace.manifest.provenance["trace_sha256"] for trace in loaded_real
            ],
            "real_metadata_sha256": [trace.metadata_sha256 for trace in loaded_real],
            "real_repetition_count": repetitions,
            "metrics": {},
        }
        sim_trace: LoadedTrace | None = None
        if role == "calibration":
            if "held_out_by" in condition or "sim_artifact" in condition:
                raise ContractError("calibration condition cannot claim holdout fields")
            if raw_metrics:
                raise ContractError("calibration condition metrics must be empty")
        else:
            held_out_by = condition.get("held_out_by")
            if not isinstance(held_out_by, list) or not held_out_by:
                raise ContractError("holdout condition held_out_by must be non-empty")
            if any(item not in _HELD_OUT_DIMENSIONS for item in held_out_by):
                raise ContractError("held_out_by contains an unsupported condition dimension")
            if len(held_out_by) != len(set(held_out_by)):
                raise ContractError("held_out_by values must be unique")
            clean_condition["held_out_by"] = list(held_out_by)
            if scenario.scene_mode == "manual":
                if "sim_artifact" in condition:
                    raise ContractError(
                        "manual holdout cannot claim an Isaac simulator artifact"
                    )
                _validate_direct_holdout_context(candidate, scenario, loaded_real)
                if not raw_metrics:
                    raise ContractError("manual holdout metrics must be non-empty")
                mandatory_metrics = _mandatory_holdout_metrics(subsystem, scenario)
                supplied_metrics = set(raw_metrics)
                missing_metrics = set(mandatory_metrics) - supplied_metrics
                if missing_metrics:
                    raise ContractError(
                        "mandatory held-out metrics are missing: "
                        + ", ".join(sorted(missing_metrics))
                    )
                unsupported_metrics = supplied_metrics - set(mandatory_metrics)
                if unsupported_metrics:
                    raise ContractError(
                        "unsupported held-out metrics: "
                        + ", ".join(sorted(unsupported_metrics))
                    )
                real_metric_sets = [
                    compute_subsystem_metrics(scenario, trace)
                    for trace in loaded_real
                ]
                for metric_path, raw_metric in sorted(raw_metrics.items()):
                    metric = _mapping(raw_metric, f"metric {metric_path}")
                    _exact_fields(
                        metric,
                        name=f"metric {metric_path}",
                        required={"unit", "instrument_uncertainty"},
                    )
                    unit = metric["unit"]
                    expected_unit = mandatory_metrics[metric_path]
                    if unit != expected_unit:
                        raise ContractError(
                            f"metric {metric_path} unit must be {expected_unit}, got {unit}"
                        )
                    uncertainty = _number(
                        metric["instrument_uncertainty"],
                        f"metric {metric_path}.instrument_uncertainty",
                        nonnegative=True,
                    )
                    observations = [
                        _observation(
                            metrics, metric_path, _repeat_count(metrics, trace)
                        )
                        for metrics, trace in zip(
                            real_metric_sets, loaded_real, strict=True
                        )
                    ]
                    real_mean, real_std, real_count = _pool(observations)
                    profile_value = _direct_profile_value(
                        candidate, subsystem, scenario, metric_path
                    )
                    tolerance = max(uncertainty, 2.0 * real_std)
                    error = abs(profile_value - real_mean)
                    passed = error <= tolerance
                    clean_condition["metrics"][metric_path] = {
                        "unit": unit,
                        "real_mean": real_mean,
                        "real_std": real_std,
                        "real_count": real_count,
                        "instrument_uncertainty": uncertainty,
                        "profile_value": profile_value,
                        "tolerance": tolerance,
                        "absolute_error": error,
                        "pass": passed,
                    }
                    if not passed:
                        subsystem_failures.append(
                            f"{subsystem}.{condition_id}.{metric_path} is outside "
                            "its held-out direct-measurement envelope"
                        )
            else:
                sim_trace, _ = _trace_binding(
                    root,
                    condition.get("sim_artifact"),
                    f"condition {condition_id} simulator artifact",
                    source="sim",
                    scenario=scenario,
                    profile=candidate,
                )
                _validate_condition_trace_metadata(sim_trace, scenario)
                if _condition_coordinates(sim_trace, scenario) != coordinates:
                    raise ContractError("simulator artifact coordinates do not match real holdout")
                if not raw_metrics:
                    raise ContractError("holdout condition metrics must be non-empty")
                sim_metrics = compute_subsystem_metrics(scenario, sim_trace)
                real_metric_sets = [
                    compute_subsystem_metrics(scenario, trace) for trace in loaded_real
                ]
                mandatory_metrics = _mandatory_holdout_metrics(subsystem, scenario)
                supplied_metrics = set(raw_metrics)
                missing_metrics = set(mandatory_metrics) - supplied_metrics
                if missing_metrics:
                    raise ContractError(
                        "mandatory held-out metrics are missing: "
                        + ", ".join(sorted(missing_metrics))
                    )
                unsupported_metrics = supplied_metrics - set(mandatory_metrics)
                if unsupported_metrics:
                    raise ContractError(
                        "unsupported held-out metrics: "
                        + ", ".join(sorted(unsupported_metrics))
                    )
                for metric_path, raw_metric in sorted(raw_metrics.items()):
                    metric = _mapping(raw_metric, f"metric {metric_path}")
                    _exact_fields(
                        metric,
                        name=f"metric {metric_path}",
                        required={"unit", "instrument_uncertainty"},
                    )
                    unit = metric["unit"]
                    if not isinstance(unit, str) or not unit.strip():
                        raise ContractError(f"metric {metric_path} unit must be non-empty")
                    expected_unit = mandatory_metrics[metric_path]
                    if unit != expected_unit:
                        raise ContractError(
                            f"metric {metric_path} unit must be {expected_unit}, got {unit}"
                        )
                    uncertainty = _number(
                        metric["instrument_uncertainty"],
                        f"metric {metric_path}.instrument_uncertainty",
                        nonnegative=True,
                    )
                    observations = [
                        _observation(metrics, metric_path, _repeat_count(metrics, trace))
                        for metrics, trace in zip(
                            real_metric_sets, loaded_real, strict=True
                        )
                    ]
                    real_mean, real_std, real_count = _pool(observations)
                    sim_value, _, _ = _lookup(sim_metrics, metric_path)
                    tolerance = max(uncertainty, 2.0 * real_std)
                    error = abs(sim_value - real_mean)
                    passed = error <= tolerance
                    clean_condition["metrics"][metric_path] = {
                        "unit": unit,
                        "real_mean": real_mean,
                        "real_std": real_std,
                        "real_count": real_count,
                        "instrument_uncertainty": uncertainty,
                        "sim_value": sim_value,
                        "tolerance": tolerance,
                        "absolute_error": error,
                        "pass": passed,
                    }
                    if not passed:
                        subsystem_failures.append(
                            f"{subsystem}.{condition_id}.{metric_path} is outside its held-out envelope"
                        )
                clean_condition["sim_trace_sha256"] = sim_trace.manifest.provenance[
                    "trace_sha256"
                ]
        clean_condition["failures"] = subsystem_failures
        grouped[subsystem][role].append(clean_condition)
        condition_internal[condition_id] = {
            "subsystem": subsystem,
            "role": role,
            "coordinates": coordinates,
            "held_out_by": list(condition.get("held_out_by", [])),
            "scenario": scenario,
            "metrics": clean_condition["metrics"],
            "real_traces": loaded_real,
        }

    for condition_id, internal in condition_internal.items():
        if internal["role"] != "holdout":
            continue
        calibrations = grouped[internal["subsystem"]]["calibration"]
        if not calibrations:
            continue
        for dimension in internal["held_out_by"]:
            held_value = internal["coordinates"].get(dimension)
            if held_value is None or any(
                item["coordinates"].get(dimension) == held_value for item in calibrations
            ):
                raise ContractError(
                    f"held-out dimension {dimension} does not differ from calibration"
                )

    _validate_changed_field_evidence(baseline, candidate, condition_internal)
    measurement_source_report = _validate_measurement_sources(
        baseline, candidate, condition_internal
    )
    known_load_traces = [
        trace
        for internal in condition_internal.values()
        if internal["role"] == "calibration"
        and internal["scenario"].experiment_kind == "manual_load"
        for trace in internal["real_traces"]
    ]

    raw_sweeps = _mapping(data["actuator_sweeps"], "actuator_sweeps")
    if set(raw_sweeps) - fitted:
        raise ContractError("actuator_sweeps references a subsystem not changed by the candidate")
    for subsystem, sweeps in raw_sweeps.items():
        if not isinstance(sweeps, list):
            raise ContractError(f"{subsystem} actuator sweeps must be an array")
    actuator_mismatch: dict[str, bool] = {subsystem: False for subsystem in fitted}
    if global_failures:
        stale_sweeps = sorted(
            subsystem for subsystem, sweeps in raw_sweeps.items() if sweeps
        )
        if stale_sweeps:
            raise ContractError(
                "failed pre-fit audit cannot reference non-empty actuator sweep "
                "bindings: " + ", ".join(stale_sweeps)
            )
    elif "main_drive" in fitted:
        holdouts = [
            condition
            for condition in grouped["main_drive"]["holdout"]
            if condition_internal[condition["condition_id"]]["scenario"].scene_mode
            != "manual"
        ]
        sweeps = raw_sweeps.get("main_drive")
        if not isinstance(sweeps, list):
            raise ContractError("main_drive actuator sweeps must be an array")
        if len(sweeps) != len(holdouts):
            raise ContractError("every main_drive holdout requires exactly one actuator sweep")
        if effort_limit_changed and not known_load_traces:
            # The result is already ineligible below. A response sweep cannot
            # substitute for the missing direct effort-saturation measurement.
            sweeps = []
            holdouts = []
        inside_by_holdout: list[bool] = []
        for sweep, holdout in zip(sweeps, holdouts, strict=True):
            internal = condition_internal[holdout["condition_id"]]
            inside_by_holdout.append(
                _verify_sweep_for_holdout(
                    root,
                    sweep,
                    holdout=holdout,
                    internal=internal,
                    baseline_profile=baseline,
                    effort_limit_changed=effort_limit_changed,
                    known_load_traces=known_load_traces,
                    audit_artifact_sha256=sha256_json(data["audit_artifact"]),
                    audit_report_sha256=sha256_json(audit_report),
                )
            )
        actuator_mismatch["main_drive"] = bool(holdouts) and not all(inside_by_holdout)

    subsystem_results: dict[str, Any] = {}
    all_failures = list(global_failures)
    for subsystem in sorted(fitted):
        local_failures = list(global_failures)
        calibration = grouped[subsystem]["calibration"]
        holdout = grouped[subsystem]["holdout"]
        if not calibration:
            local_failures.append(f"{subsystem} is missing a calibration condition")
        if not holdout:
            local_failures.append(f"{subsystem} is missing a holdout condition")
        if subsystem == "main_drive" and effort_limit_changed and not known_load_traces:
            local_failures.append(
                "main_drive effort-limit fitting requires a managed known-load calibration condition"
            )
        for condition in calibration + holdout:
            local_failures.extend(condition["failures"])
        mismatch = actuator_mismatch[subsystem]
        if mismatch:
            local_failures.append(
                f"{subsystem} actuator-model mismatch: bounded candidates miss the holdout envelope"
            )
        all_failures.extend(reason for reason in local_failures if reason not in all_failures)
        subsystem_results[subsystem] = {
            "pass": not local_failures,
            "failures": local_failures,
            "actuator_model_mismatch": mismatch,
            "calibration_conditions": calibration,
            "holdout_conditions": holdout,
        }

    return {
        "schema_version": 1,
        "profile_id": candidate.profile_id,
        "candidate_profile_sha256": expected_profile_hash,
        "baseline_profile_sha256": sha256_json(baseline.to_dict()),
        "evidence_sha256": sha256_json(data),
        "eligible_for_review": not all_failures,
        "promotion_requires_reviewed_config_change": True,
        "audit": audit_report,
        "measurement_sources": measurement_source_report,
        "derived_fitted_subsystems": sorted(fitted),
        "subsystems": subsystem_results,
        "failures": all_failures,
    }
