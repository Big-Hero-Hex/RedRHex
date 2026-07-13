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
    measurement_sources: dict[str, str] = field(default_factory=dict)
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
                elif key in {"gear_ratio", "pwm_scale", "abad_target_scale"}:
                    clean[joint] = _positive(value, path)
                elif key == "pwm_cap":
                    cap = _positive(value, path)
                    if cap > 1.0:
                        raise ContractError(f"{path} must be at most 1")
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
                    clean[joint] = _number(value, path)
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
                clean_physics[section_name] = {
                    joint: _number(
                        value,
                        f"simulation_physics.{section_name}.{joint}",
                        minimum=0.0,
                    )
                    for joint, value in _string_map(
                        section, f"simulation_physics.{section_name}"
                    ).items()
                }
            elif section_name == "passive_spring":
                springs: dict[str, dict[str, float]] = {}
                for joint, raw_spring in _string_map(
                    section, "simulation_physics.passive_spring"
                ).items():
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
                unknown = set(section) - {"scale", "added_mass_kg", "com_offset_m"}
                if unknown:
                    raise ContractError(f"unknown simulation_physics.mass fields: {', '.join(sorted(unknown))}")
                clean_mass: dict[str, Any] = {}
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
        raw_sources = _string_map(
            data.get("measurement_sources", {}), "measurement_sources"
        )
        measurement_sources: dict[str, str] = {}
        for name, digest in raw_sources.items():
            if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
                raise ContractError(
                    f"measurement_sources.{name} must be a lowercase SHA-256 digest"
                )
            measurement_sources[name] = digest
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
            "measurement_sources": dict(self.measurement_sources),
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
