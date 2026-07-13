from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping


SCHEMA_VERSION = 1
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ABAD_JOINT_RE = re.compile(r"^abad_[0-5]$")
_CANONICAL_JOINT_RE = re.compile(r"^(?:main|abad|damper)_[0-5]$")
_DAMPER_JOINT_RE = re.compile(r"^damper_[0-5]$")
# 50 rad/s is about 477 rpm: well above the reviewed 4.17 rad/s bridge cap,
# while still rejecting normalized/raw-PWM unit mistakes as physical profiles.
_MAX_CANONICAL_PWM_CAP_RAD_S = 50.0
# Static angle calibration is a correction near identity, not an alternate
# kinematic model. Larger deviations require a geometry/encoder audit first.
_MIN_ABAD_TARGET_SCALE = 0.5
_MAX_ABAD_TARGET_SCALE = 1.5
_MAX_ABAD_TARGET_OFFSET_RAD = 0.35
_LEGACY_MASS_FIELDS = {"scale", "added_mass_kg", "com_offset_m"}
_ABSOLUTE_MASS_FIELDS = {
    "target_total_mass_kg",
    "reference_planar_com_xy_m",
    "reference_joint_position_rad",
    "reference_root_orientation_xyzw",
}
_CANONICAL_JOINTS = {
    f"{group}_{index}"
    for group in ("main", "abad", "damper")
    for index in range(6)
}


class ContractError(ValueError):
    """Raised when versioned calibration data violates its contract."""


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{field_name} must be a JSON object")
    return dict(value)


def _keys(data: Mapping[str, Any], *, required: set[str], optional: set[str] = set()) -> None:
    missing = required - set(data)
    if missing:
        raise ContractError(f"missing required fields: {', '.join(sorted(missing))}")
    unknown = set(data) - required - optional
    if unknown:
        raise ContractError(f"unknown fields: {', '.join(sorted(unknown))}")


def _version(data: Mapping[str, Any]) -> None:
    if data.get("schema_version") != SCHEMA_VERSION or isinstance(
        data.get("schema_version"), bool
    ):
        raise ContractError(f"schema_version must be integer {SCHEMA_VERSION}")


def _identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise ContractError(f"{field_name} must be a lowercase identifier")
    return value


def _string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field_name} must be a non-empty string")
    return value


def _number(value: Any, field_name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{field_name} must be finite")
    if minimum is not None and result < minimum:
        raise ContractError(f"{field_name} must be non-negative")
    return result


def _positive(value: Any, field_name: str) -> float:
    result = _number(value, field_name)
    if result <= 0.0:
        raise ContractError(f"{field_name} must be positive")
    return result


def _string_map(value: Any, field_name: str) -> dict[str, Any]:
    result = _mapping(value, field_name)
    for key in result:
        _string(key, field_name)
    return result


@dataclass(frozen=True)
class ScenarioSpecV1:
    scenario_id: str
    name: str
    subsystem: str
    experiment_kind: str
    joint: str
    command_segments: tuple[dict[str, Any], ...]
    repeats: int
    required_channels: tuple[str, ...]
    time_bases: dict[str, str]
    split: str
    scene_mode: str
    safety_class: str
    description: str = ""
    schema_version: int = field(default=SCHEMA_VERSION, init=False)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ScenarioSpecV1":
        data = _mapping(payload, "scenario")
        _version(data)
        required = {
            "schema_version",
            "scenario_id",
            "name",
            "subsystem",
            "experiment_kind",
            "joint",
            "command_segments",
            "repeats",
            "required_channels",
            "time_bases",
            "split",
            "scene_mode",
            "safety_class",
        }
        _keys(data, required=required, optional={"description"})
        scenario_id = _identifier(data["scenario_id"], "scenario_id")
        name = _string(data["name"], "name")
        subsystem = _identifier(data["subsystem"], "subsystem")
        experiment_kind = _identifier(data["experiment_kind"], "experiment_kind")
        joint = _string(data["joint"], "joint")

        raw_segments = data["command_segments"]
        if not isinstance(raw_segments, list) or not raw_segments:
            raise ContractError("command_segments must be a non-empty array")
        segments: list[dict[str, Any]] = []
        for index, raw in enumerate(raw_segments):
            segment = _mapping(raw, f"command_segments[{index}]")
            _keys(segment, required={"duration_s", "value"}, optional={"label"})
            clean: dict[str, Any] = {
                "duration_s": _positive(segment["duration_s"], f"command_segments[{index}].duration_s"),
                "value": _number(segment["value"], f"command_segments[{index}].value"),
            }
            if "label" in segment:
                clean["label"] = _string(segment["label"], f"command_segments[{index}].label")
            segments.append(clean)

        repeats = data["repeats"]
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
            raise ContractError("repeats must be a positive integer")
        raw_channels = data["required_channels"]
        if not isinstance(raw_channels, list) or not raw_channels:
            raise ContractError("required_channels must be a non-empty array")
        channels = tuple(_string(item, "required_channels") for item in raw_channels)
        if len(set(channels)) != len(channels):
            raise ContractError("required_channels must be unique")
        raw_time_bases = _string_map(data["time_bases"], "time_bases")
        time_bases = {str(key): _string(value, f"time_bases.{key}") for key, value in raw_time_bases.items()}
        if set(time_bases) != set(channels):
            raise ContractError("time_bases must map every required channel exactly once")
        split = data["split"]
        if split not in {"calibration", "holdout"}:
            raise ContractError("split must be calibration or holdout")
        scene_mode = data["scene_mode"]
        if scene_mode not in {"fixed_base", "free_root", "manual", "audit"}:
            raise ContractError("unsupported scene_mode")
        safety_class = _identifier(data["safety_class"], "safety_class")
        description = data.get("description", "")
        if not isinstance(description, str):
            raise ContractError("description must be a string")
        return cls(
            scenario_id=scenario_id,
            name=name,
            subsystem=subsystem,
            experiment_kind=experiment_kind,
            joint=joint,
            command_segments=tuple(segments),
            repeats=repeats,
            required_channels=channels,
            time_bases=time_bases,
            split=split,
            scene_mode=scene_mode,
            safety_class=safety_class,
            description=description,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "name": self.name,
            "description": self.description,
            "subsystem": self.subsystem,
            "experiment_kind": self.experiment_kind,
            "joint": self.joint,
            "command_segments": copy.deepcopy(list(self.command_segments)),
            "repeats": self.repeats,
            "required_channels": list(self.required_channels),
            "time_bases": dict(self.time_bases),
            "split": self.split,
            "scene_mode": self.scene_mode,
            "safety_class": self.safety_class,
        }

    def validate(self) -> "ScenarioSpecV1":
        return self.from_dict(self.to_dict())


@dataclass(frozen=True)
class TraceManifestV1:
    scenario_id: str
    source: str
    trace_file: str
    channels: tuple[str, ...]
    time_bases: dict[str, str]
    sample_counts: dict[str, int]
    provenance: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = field(default=SCHEMA_VERSION, init=False)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TraceManifestV1":
        data = _mapping(payload, "manifest")
        _version(data)
        _keys(
            data,
            required={
                "schema_version",
                "scenario_id",
                "source",
                "trace_file",
                "channels",
                "time_bases",
                "sample_counts",
                "provenance",
            },
            optional={"metadata"},
        )
        scenario_id = _identifier(data["scenario_id"], "scenario_id")
        source = data["source"]
        if source not in {"real", "sim", "derived"}:
            raise ContractError("source must be real, sim, or derived")
        trace_file = _string(data["trace_file"], "trace_file")
        if Path(trace_file).name != trace_file or not trace_file.endswith(".npz"):
            raise ContractError("trace_file must be a local .npz filename")
        raw_channels = data["channels"]
        if not isinstance(raw_channels, list) or not raw_channels:
            raise ContractError("channels must be a non-empty array")
        channels = tuple(_string(item, "channels") for item in raw_channels)
        if len(channels) != len(set(channels)):
            raise ContractError("channels must be unique")
        raw_time_bases = _string_map(data["time_bases"], "time_bases")
        time_bases = {key: _string(value, f"time_bases.{key}") for key, value in raw_time_bases.items()}
        if not set(time_bases).issubset(channels) or not set(time_bases.values()).issubset(channels):
            raise ContractError("time_bases must reference declared channels")
        raw_counts = _string_map(data["sample_counts"], "sample_counts")
        counts: dict[str, int] = {}
        for key, value in raw_counts.items():
            if key not in time_bases:
                raise ContractError(f"sample_counts contains unknown data channel {key}")
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ContractError(f"sample_counts.{key} must be a positive integer")
            counts[key] = value
        if set(counts) != set(time_bases):
            raise ContractError("sample_counts must cover every data channel")
        raw_provenance = _string_map(data["provenance"], "provenance")
        provenance: dict[str, str] = {}
        for key, value in raw_provenance.items():
            if not key.endswith("_sha256") or not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
                raise ContractError(f"provenance.{key} must be a lowercase SHA-256 digest")
            provenance[key] = value
        if "trace_sha256" not in provenance:
            raise ContractError("provenance.trace_sha256 is required")
        metadata = _manifest_metadata(
            data.get("metadata", {}),
            source=source,
            data_channels=set(time_bases),
            provenance=provenance,
        )
        return cls(
            scenario_id=scenario_id,
            source=source,
            trace_file=trace_file,
            channels=channels,
            time_bases=time_bases,
            sample_counts=counts,
            provenance=provenance,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "source": self.source,
            "trace_file": self.trace_file,
            "channels": list(self.channels),
            "time_bases": dict(self.time_bases),
            "sample_counts": dict(self.sample_counts),
            "provenance": dict(self.provenance),
            "metadata": copy.deepcopy(self.metadata),
        }

    def validate(self) -> "TraceManifestV1":
        return self.from_dict(self.to_dict())


_HARDWARE_FIELDS = {
    "abad_target_offset_rad",
    "abad_target_scale",
    "joint_direction",
    "joint_offset_rad",
    "gear_ratio",
    "actuator_id",
    "encoder_counts_per_rev",
    "encoder_zero_count",
    "encoder_sign",
    "pwm_scale",
    "pwm_cap",
}
_TIMING_FIELDS = {
    "command_delay_s",
    "aggregate_command_delay_s",
    "sensor_delay_s",
    "sample_period_s",
    "measured_state_rate_hz",
    "velocity_filter_alpha",
    "position_noise_std_rad",
    "velocity_filter_window_s",
}
_PHYSICS_SECTIONS = {
    "rigid_body",
    "main_drive",
    "abad",
    "damper",
    "joint_friction",
    "joint_dynamic_friction",
    "joint_viscous_friction",
    "passive_spring",
    "mass",
    "ground",
}
_ACTUATOR_FIELDS = {"stiffness", "damping", "effort_limit", "velocity_limit", "armature", "friction"}


def _clean_profile_section(raw: Any, section_name: str, allowed: set[str]) -> dict[str, Any]:
    section = _mapping(raw, section_name)
    unknown = set(section) - allowed
    if unknown:
        raise ContractError(f"unknown {section_name} fields: {', '.join(sorted(unknown))}")
    return section


def _validate_json(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        _number(value, path)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json(item, f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError(f"{path} keys must be strings")
            _validate_json(item, f"{path}.{key}")
        return
    raise ContractError(f"{path} is not JSON-compatible")


def _manifest_metadata(
    value: Any,
    *,
    source: str,
    data_channels: set[str],
    provenance: Mapping[str, str],
) -> dict[str, Any]:
    metadata = _mapping(value, "metadata")
    required = {
        "units",
        "frames",
        "joint_order",
        "clock",
        "scenario_schema_version",
        "scenario_sha256",
        "git_sha",
        "asset_sha256",
        "config_sha256",
        "calibration_constants",
        "raw_data_sha256",
    }
    missing = required - set(metadata)
    if missing:
        raise ContractError(f"metadata missing required fields: {', '.join(sorted(missing))}")
    units = _string_map(metadata["units"], "metadata.units")
    frames = _string_map(metadata["frames"], "metadata.frames")
    if set(units) != data_channels or set(frames) != data_channels:
        raise ContractError("metadata units and frames must cover every data channel")
    for name, unit in units.items():
        _string(unit, f"metadata.units.{name}")
    for name, frame_name in frames.items():
        _string(frame_name, f"metadata.frames.{name}")
    joint_order = metadata["joint_order"]
    if not isinstance(joint_order, list) or any(
        not isinstance(name, str) or not name for name in joint_order
    ):
        raise ContractError("metadata.joint_order must be an array of names")
    if len(joint_order) != len(set(joint_order)):
        raise ContractError("metadata.joint_order must be unique")
    clock = _mapping(metadata["clock"], "metadata.clock")
    _keys(
        clock,
        required={"source", "timestamp_semantics", "time_unit"},
    )
    for key, item in clock.items():
        _string(item, f"metadata.clock.{key}")
    scenario_version = metadata["scenario_schema_version"]
    if scenario_version not in {None, SCHEMA_VERSION}:
        raise ContractError("metadata.scenario_schema_version must be 1 or null")

    def optional_sha(key: str) -> str | None:
        digest = metadata[key]
        if digest is not None and (
            not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest)
        ):
            raise ContractError(f"metadata.{key} must be a SHA-256 digest or null")
        return digest

    scenario_hash = optional_sha("scenario_sha256")
    asset_hash = optional_sha("asset_sha256")
    config_hash = optional_sha("config_sha256")
    raw_hash = optional_sha("raw_data_sha256")
    if provenance.get("scenario_sha256") not in {None, scenario_hash}:
        raise ContractError("metadata scenario hash does not match provenance")
    if provenance.get("source_sha256") not in {None, raw_hash}:
        raise ContractError("metadata raw-data hash does not match provenance")
    if source == "real" and raw_hash is None:
        raise ContractError("metadata.raw_data_sha256 is required for real traces")
    git_sha = metadata["git_sha"]
    if git_sha is not None and (not isinstance(git_sha, str) or not git_sha):
        raise ContractError("metadata.git_sha must be a string or null")
    calibration = _mapping(metadata["calibration_constants"], "metadata.calibration_constants")
    _validate_json(calibration, "metadata.calibration_constants")
    _validate_json(metadata, "metadata")
    clean = copy.deepcopy(metadata)
    clean["units"] = dict(units)
    clean["frames"] = dict(frames)
    clean["joint_order"] = list(joint_order)
    clean["clock"] = dict(clock)
    clean["scenario_sha256"] = scenario_hash
    clean["asset_sha256"] = asset_hash
    clean["config_sha256"] = config_hash
    clean["raw_data_sha256"] = raw_hash
    clean["calibration_constants"] = calibration
    return clean


@dataclass(frozen=True)
class CalibrationProfileV1:
    profile_id: str
    hardware_mapping: dict[str, Any]
    sensor_timing: dict[str, float]
    simulation_physics: dict[str, Any]
    description: str = ""
    measurement_sources: dict[str, Any] = field(default_factory=dict)
    schema_version: int = field(default=SCHEMA_VERSION, init=False)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CalibrationProfileV1":
        data = _mapping(payload, "profile")
        _version(data)
        _keys(
            data,
            required={
                "schema_version",
                "profile_id",
                "hardware_mapping",
                "sensor_timing",
                "simulation_physics",
            },
            optional={"description", "measurement_sources"},
        )
        profile_id = _identifier(data["profile_id"], "profile_id")
        hardware = _clean_profile_section(data["hardware_mapping"], "hardware_mapping", _HARDWARE_FIELDS)
        clean_hardware: dict[str, Any] = {}
        for key, raw_values in hardware.items():
            values = _string_map(raw_values, f"hardware_mapping.{key}")
            clean: dict[str, Any] = {}
            for joint, value in values.items():
                path = f"hardware_mapping.{key}.{joint}"
                if key in {"joint_direction", "encoder_sign"}:
                    if value not in {-1, 1} or isinstance(value, bool):
                        raise ContractError(f"{path} must be -1 or 1")
                    clean[joint] = int(value)
                elif key in {"gear_ratio", "pwm_scale"}:
                    clean[joint] = _positive(value, path)
                elif key == "abad_target_scale":
                    if not _ABAD_JOINT_RE.fullmatch(joint):
                        raise ContractError(f"{path} must name abad_0..abad_5")
                    scale = _positive(value, path)
                    if not _MIN_ABAD_TARGET_SCALE <= scale <= _MAX_ABAD_TARGET_SCALE:
                        raise ContractError(
                            f"{path} must be between {_MIN_ABAD_TARGET_SCALE} and "
                            f"{_MAX_ABAD_TARGET_SCALE}"
                        )
                    clean[joint] = scale
                elif key == "pwm_cap":
                    cap = _positive(value, path)
                    if cap > _MAX_CANONICAL_PWM_CAP_RAD_S:
                        raise ContractError(
                            f"{path} is a canonical velocity cap in rad/s and must be at "
                            f"most {_MAX_CANONICAL_PWM_CAP_RAD_S}"
                        )
                    clean[joint] = cap
                elif key == "actuator_id":
                    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                        raise ContractError(f"{path} must be a non-negative integer")
                    clean[joint] = value
                elif key == "encoder_counts_per_rev":
                    clean[joint] = _positive(value, path)
                elif key == "encoder_zero_count":
                    clean[joint] = _number(value, path, minimum=0.0)
                else:
                    number = _number(value, path)
                    if key == "abad_target_offset_rad":
                        if not _ABAD_JOINT_RE.fullmatch(joint):
                            raise ContractError(f"{path} must name abad_0..abad_5")
                        if abs(number) > _MAX_ABAD_TARGET_OFFSET_RAD:
                            raise ContractError(
                                f"{path} must be within "
                                f"+/-{_MAX_ABAD_TARGET_OFFSET_RAD} rad"
                            )
                    clean[joint] = number
            clean_hardware[key] = clean

        timing = _clean_profile_section(data["sensor_timing"], "sensor_timing", _TIMING_FIELDS)
        clean_timing: dict[str, float] = {}
        for key, value in timing.items():
            if key in {"sample_period_s", "measured_state_rate_hz"}:
                clean_timing[key] = _positive(value, f"sensor_timing.{key}")
            else:
                clean_timing[key] = _number(value, f"sensor_timing.{key}", minimum=0.0)
            if key == "velocity_filter_alpha" and clean_timing[key] > 1.0:
                raise ContractError("sensor_timing.velocity_filter_alpha must be at most 1")

        physics = _clean_profile_section(data["simulation_physics"], "simulation_physics", _PHYSICS_SECTIONS)
        clean_physics: dict[str, Any] = {}
        for section_name, raw_section in physics.items():
            section = _mapping(raw_section, f"simulation_physics.{section_name}")
            if section_name in {"main_drive", "abad", "damper"}:
                unknown = set(section) - _ACTUATOR_FIELDS
                if unknown:
                    raise ContractError(f"unknown simulation_physics.{section_name} fields: {', '.join(sorted(unknown))}")
                clean_physics[section_name] = {
                    key: _number(value, f"simulation_physics.{section_name}.{key}", minimum=0.0)
                    for key, value in section.items()
                }
            elif section_name == "rigid_body":
                unknown = set(section) - {"linear_damping", "angular_damping"}
                if unknown:
                    raise ContractError(f"unknown simulation_physics.rigid_body fields: {', '.join(sorted(unknown))}")
                clean_physics[section_name] = {
                    key: _number(value, f"simulation_physics.rigid_body.{key}", minimum=0.0)
                    for key, value in section.items()
                }
            elif section_name in {
                "joint_friction",
                "joint_dynamic_friction",
                "joint_viscous_friction",
            }:
                friction: dict[str, float] = {}
                for joint, value in _string_map(
                    section, f"simulation_physics.{section_name}"
                ).items():
                    if not _CANONICAL_JOINT_RE.fullmatch(joint):
                        raise ContractError(
                            f"simulation_physics.{section_name}.{joint} must name a "
                            "canonical joint main_0..main_5, abad_0..abad_5, or "
                            "damper_0..damper_5"
                        )
                    friction[joint] = _number(
                        value,
                        f"simulation_physics.{section_name}.{joint}",
                        minimum=0.0,
                    )
                clean_physics[section_name] = friction
            elif section_name == "passive_spring":
                springs: dict[str, dict[str, float]] = {}
                for joint, raw_spring in _string_map(
                    section, "simulation_physics.passive_spring"
                ).items():
                    if not _DAMPER_JOINT_RE.fullmatch(joint):
                        raise ContractError(
                            f"simulation_physics.passive_spring.{joint} must name a "
                            "canonical joint damper_0..damper_5"
                        )
                    spring = _mapping(
                        raw_spring, f"simulation_physics.passive_spring.{joint}"
                    )
                    unknown = set(spring) - {
                        "stiffness",
                        "damping",
                        "rest_position_rad",
                    }
                    if unknown:
                        raise ContractError(
                            "unknown simulation_physics.passive_spring fields: "
                            + ", ".join(sorted(unknown))
                        )
                    if "stiffness" not in spring:
                        raise ContractError(
                            f"simulation_physics.passive_spring.{joint}.stiffness is required"
                        )
                    clean_spring = {
                        "stiffness": _number(
                            spring["stiffness"],
                            f"simulation_physics.passive_spring.{joint}.stiffness",
                            minimum=0.0,
                        )
                    }
                    if "damping" in spring:
                        clean_spring["damping"] = _number(
                            spring["damping"],
                            f"simulation_physics.passive_spring.{joint}.damping",
                            minimum=0.0,
                        )
                    if "rest_position_rad" in spring:
                        clean_spring["rest_position_rad"] = _number(
                            spring["rest_position_rad"],
                            f"simulation_physics.passive_spring.{joint}.rest_position_rad",
                        )
                    springs[joint] = clean_spring
                clean_physics[section_name] = springs
            elif section_name == "mass":
                unknown = set(section) - _LEGACY_MASS_FIELDS - _ABSOLUTE_MASS_FIELDS
                if unknown:
                    raise ContractError(f"unknown simulation_physics.mass fields: {', '.join(sorted(unknown))}")
                legacy_fields = set(section) & _LEGACY_MASS_FIELDS
                absolute_fields = set(section) & _ABSOLUTE_MASS_FIELDS
                if legacy_fields and absolute_fields:
                    raise ContractError(
                        "simulation_physics.mass cannot mix legacy relative and "
                        "absolute mass fields"
                    )
                clean_mass: dict[str, Any] = {}
                if absolute_fields:
                    missing = _ABSOLUTE_MASS_FIELDS - absolute_fields
                    if missing:
                        raise ContractError(
                            "simulation_physics.mass absolute mode is missing fields: "
                            + ", ".join(sorted(missing))
                        )
                    clean_mass["target_total_mass_kg"] = _positive(
                        section["target_total_mass_kg"],
                        "simulation_physics.mass.target_total_mass_kg",
                    )
                    planar_com = section["reference_planar_com_xy_m"]
                    if not isinstance(planar_com, list) or len(planar_com) != 2:
                        raise ContractError(
                            "simulation_physics.mass.reference_planar_com_xy_m must "
                            "contain two values"
                        )
                    clean_mass["reference_planar_com_xy_m"] = [
                        _number(
                            value,
                            "simulation_physics.mass.reference_planar_com_xy_m"
                            f"[{index}]",
                        )
                        for index, value in enumerate(planar_com)
                    ]
                    reference_joints = _string_map(
                        section["reference_joint_position_rad"],
                        "simulation_physics.mass.reference_joint_position_rad",
                    )
                    if set(reference_joints) != _CANONICAL_JOINTS:
                        missing_joints = _CANONICAL_JOINTS - set(reference_joints)
                        unknown_joints = set(reference_joints) - _CANONICAL_JOINTS
                        details = []
                        if missing_joints:
                            details.append("missing " + ", ".join(sorted(missing_joints)))
                        if unknown_joints:
                            details.append("unknown " + ", ".join(sorted(unknown_joints)))
                        raise ContractError(
                            "simulation_physics.mass.reference_joint_position_rad "
                            "must contain all 18 canonical joints (" + "; ".join(details) + ")"
                        )
                    clean_mass["reference_joint_position_rad"] = {
                        joint: _number(
                            reference_joints[joint],
                            "simulation_physics.mass.reference_joint_position_rad."
                            f"{joint}",
                        )
                        for joint in sorted(reference_joints)
                    }
                    orientation = section["reference_root_orientation_xyzw"]
                    if not isinstance(orientation, list) or len(orientation) != 4:
                        raise ContractError(
                            "simulation_physics.mass.reference_root_orientation_xyzw "
                            "must contain four values"
                        )
                    clean_orientation = [
                        _number(
                            value,
                            "simulation_physics.mass.reference_root_orientation_xyzw"
                            f"[{index}]",
                        )
                        for index, value in enumerate(orientation)
                    ]
                    norm = math.sqrt(sum(value * value for value in clean_orientation))
                    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-6):
                        raise ContractError(
                            "simulation_physics.mass.reference_root_orientation_xyzw "
                            "must be normalized"
                        )
                    clean_mass["reference_root_orientation_xyzw"] = clean_orientation
                else:
                    if "scale" in section:
                        clean_mass["scale"] = _positive(section["scale"], "simulation_physics.mass.scale")
                    if "added_mass_kg" in section:
                        clean_mass["added_mass_kg"] = _number(
                            section["added_mass_kg"], "simulation_physics.mass.added_mass_kg", minimum=0.0
                        )
                    if "com_offset_m" in section:
                        offset = section["com_offset_m"]
                        if not isinstance(offset, list) or len(offset) != 3:
                            raise ContractError("simulation_physics.mass.com_offset_m must contain three values")
                        clean_mass["com_offset_m"] = [
                            _number(value, f"simulation_physics.mass.com_offset_m[{index}]")
                            for index, value in enumerate(offset)
                        ]
                clean_physics[section_name] = clean_mass
            else:
                allowed = {"static_friction", "dynamic_friction", "restitution"}
                unknown = set(section) - allowed
                if unknown:
                    raise ContractError(f"unknown simulation_physics.ground fields: {', '.join(sorted(unknown))}")
                clean_ground = {
                    key: _number(value, f"simulation_physics.ground.{key}", minimum=0.0)
                    for key, value in section.items()
                }
                friction_fields = {"static_friction", "dynamic_friction"}
                present_friction = set(clean_ground).intersection(friction_fields)
                if present_friction and present_friction != friction_fields:
                    raise ContractError(
                        "simulation_physics.ground static_friction and "
                        "dynamic_friction must be provided together"
                    )
                if clean_ground.get("restitution", 0.0) > 1.0:
                    raise ContractError("simulation_physics.ground.restitution must be at most 1")
                if clean_ground.get("dynamic_friction", 0.0) > clean_ground.get(
                    "static_friction", float("inf")
                ):
                    raise ContractError("dynamic_friction cannot exceed static_friction")
                clean_physics[section_name] = clean_ground
        description = data.get("description", "")
        if not isinstance(description, str):
            raise ContractError("description must be a string")
        raw_sources = _string_map(data.get("measurement_sources", {}), "measurement_sources")
        measurement_sources: dict[str, Any] = {}
        source_fields = {
            "trace_sha256",
            "metadata_sha256",
            "scenario_id",
            "scenario_sha256",
            "source",
            "metric_kind",
            "frame",
            "repeat_count",
            "dataset_id",
            "episode_id",
        }
        for name, raw_source in raw_sources.items():
            is_abad = name.startswith("abad_target:")
            is_ground = name == "ground_friction"
            is_mass = name == "mass_com"
            is_spring = name.startswith("passive_spring:")
            is_effort = name.startswith("main_drive_effort_limit:")
            if not (is_abad or is_ground or is_mass or is_spring or is_effort):
                if not isinstance(raw_source, str) or not _SHA256_RE.fullmatch(raw_source):
                    raise ContractError(
                        f"measurement_sources.{name} must be a lowercase SHA-256 digest"
                    )
                measurement_sources[name] = raw_source
                continue

            source_record = _mapping(raw_source, f"measurement_sources.{name}")
            _keys(source_record, required=source_fields)
            for field_name in ("trace_sha256", "metadata_sha256", "scenario_sha256"):
                digest = source_record[field_name]
                if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
                    raise ContractError(
                        f"measurement_sources.{name}.{field_name} must be a lowercase SHA-256 digest"
                    )
            scenario_id = _identifier(
                source_record["scenario_id"], f"measurement_sources.{name}.scenario_id"
            )
            _identifier(source_record["dataset_id"], f"measurement_sources.{name}.dataset_id")
            _identifier(source_record["episode_id"], f"measurement_sources.{name}.episode_id")
            if source_record["source"] != "real":
                raise ContractError(f"measurement_sources.{name}.source must be real")
            repeat_count = source_record["repeat_count"]
            if isinstance(repeat_count, bool) or not isinstance(repeat_count, int) or repeat_count < 3:
                raise ContractError(
                    f"measurement_sources.{name}.repeat_count must be an integer of at least 3"
                )
            if is_abad:
                expected_kind, expected_scenario = "abad_static_mapping", "abad-static"
            elif is_ground:
                expected_kind, expected_scenario = "ground_friction", "friction"
            elif is_mass:
                expected_kind, expected_scenario = "mass_com", "mass-com"
            elif is_spring:
                expected_kind, expected_scenario = "torsional_spring", "spring"
            else:
                expected_kind, expected_scenario = "torque_saturation", "manual-load"
            if scenario_id != expected_scenario:
                raise ContractError(
                    f"measurement_sources.{name}.scenario_id must be {expected_scenario}"
                )
            if source_record["metric_kind"] != expected_kind:
                raise ContractError(
                    f"measurement_sources.{name}.metric_kind must be {expected_kind}"
                )
            frame = _string(source_record["frame"], f"measurement_sources.{name}.frame")
            if is_abad:
                joint = name.removeprefix("abad_target:")
                if not _ABAD_JOINT_RE.fullmatch(joint) or frame != joint:
                    raise ContractError(
                        f"measurement_sources.{name}.frame must match its canonical ABAD joint"
                    )
            elif is_ground and not frame.endswith("/ground"):
                raise ContractError(
                    "measurement_sources.ground_friction.frame must describe a ground pair"
                )
            elif is_mass and frame != "root":
                raise ContractError("measurement_sources.mass_com.frame must be root")
            elif is_spring:
                joint = name.removeprefix("passive_spring:")
                if not _DAMPER_JOINT_RE.fullmatch(joint) or frame != joint:
                    raise ContractError(
                        f"measurement_sources.{name}.frame must match its canonical damper joint"
                    )
            elif is_effort:
                joint = name.removeprefix("main_drive_effort_limit:")
                if not re.fullmatch(r"main_[0-5]", joint) or frame != joint:
                    raise ContractError(
                        f"measurement_sources.{name}.frame must match its canonical main joint"
                    )
            measurement_sources[name] = copy.deepcopy(source_record)
        return cls(
            profile_id,
            clean_hardware,
            clean_timing,
            clean_physics,
            description,
            measurement_sources,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "description": self.description,
            "hardware_mapping": copy.deepcopy(self.hardware_mapping),
            "sensor_timing": dict(self.sensor_timing),
            "simulation_physics": copy.deepcopy(self.simulation_physics),
            "measurement_sources": copy.deepcopy(self.measurement_sources),
        }

    def validate(self) -> "CalibrationProfileV1":
        return self.from_dict(self.to_dict())


def _load(path: str | Path, factory: Callable[[Mapping[str, Any]], Any]) -> Any:
    file_path = Path(path)
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {file_path}: {exc}") from exc
    return factory(payload)


def load_profile(path: str | Path) -> CalibrationProfileV1:
    return _load(path, CalibrationProfileV1.from_dict)


def validate_profile(path: str | Path) -> CalibrationProfileV1:
    return load_profile(path)


def load_manifest(path: str | Path) -> TraceManifestV1:
    return _load(path, TraceManifestV1.from_dict)
