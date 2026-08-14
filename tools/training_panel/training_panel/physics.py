from __future__ import annotations

import copy
import json
import math
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.sim2real.contracts import CalibrationProfileV1, ContractError


_SLUG_RE = re.compile(r"[^a-z0-9]+")
_LEG_LABELS = ("Right front", "Right middle", "Right rear", "Left front", "Left middle", "Left rear")


@dataclass(frozen=True)
class PhysicsField:
    key: str
    label: str
    category: str
    unit: str
    description: str
    default: float | None = None
    step: float = 0.01
    minimum: float | None = 0.0
    maximum: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "category": self.category,
            "type": "float",
            "unit": self.unit,
            "description": self.description,
            "default": self.default,
            "step": self.step,
            "min": self.minimum,
            "max": self.maximum,
        }


def _actuator_fields(section: str, label: str, defaults: tuple[float, float, float, float]) -> list[PhysicsField]:
    stiffness, damping, effort, velocity = defaults
    prefix = f"simulation_physics.{section}"
    category = f"{label} actuator"
    return [
        PhysicsField(f"{prefix}.stiffness", "Stiffness", category, "N·m/rad", "Position-error gain used by the implicit actuator.", stiffness, 0.1),
        PhysicsField(f"{prefix}.damping", "Damping", category, "N·m·s/rad", "Velocity-error gain and viscous damping used by the implicit actuator.", damping, 0.1),
        PhysicsField(f"{prefix}.effort_limit", "Effort limit", category, "N·m", "Maximum simulated joint effort.", effort, 0.1),
        PhysicsField(f"{prefix}.velocity_limit", "Velocity limit", category, "rad/s", "Maximum simulated joint speed.", velocity, 0.1),
        PhysicsField(f"{prefix}.armature", "Armature", category, "kg·m²", "Additional reflected rotor inertia.", None, 0.001),
        PhysicsField(f"{prefix}.friction", "Actuator friction", category, "N·m", "Actuator-level friction term.", None, 0.001),
    ]


def _torsion_spring_actuator_fields() -> list[PhysicsField]:
    """Expose legacy uniform aliases without presenting them as a second PD drive."""

    prefix = "simulation_physics.damper"
    category = "Torsion spring compatibility"
    return [
        PhysicsField(f"{prefix}.stiffness", "Uniform spring stiffness (legacy)", category, "N·m/rad", "Compatibility shortcut applied to all six torsion-spring stiffness values.", 200.0, 0.1),
        PhysicsField(f"{prefix}.damping", "Uniform spring damping (legacy)", category, "N·m·s/rad", "Compatibility shortcut applied to all six torsion-spring damping values; zero is the repository default.", 0.0, 0.1),
        PhysicsField(f"{prefix}.effort_limit", "Effort limit", category, "N·m", "Nonbinding actuator-path effort limit; this does not clip the torsion-spring law.", 1.0e9, 0.1),
        PhysicsField(f"{prefix}.velocity_limit", "Velocity limit", category, "rad/s", "Nonbinding actuator-path velocity limit; this is not a spring-joint brake.", 1.0e6, 0.1),
        PhysicsField(f"{prefix}.armature", "Armature", category, "kg·m²", "Additional joint-space inertia; set only from measured or reviewed physical evidence.", None, 0.001),
        PhysicsField(f"{prefix}.friction", "Actuator friction", category, "N·m", "Actuator-level friction term.", None, 0.001),
    ]


def _joint_fields(group: str, group_label: str) -> list[PhysicsField]:
    fields: list[PhysicsField] = []
    for index, leg in enumerate(_LEG_LABELS):
        joint = f"{group}_{index}"
        category = f"{group_label} joint dynamics"
        fields.extend(
            [
                PhysicsField(f"simulation_physics.joint_friction.{joint}", f"{leg} static friction", category, "N·m", "Per-joint static friction coefficient.", None, 0.001),
                PhysicsField(f"simulation_physics.joint_dynamic_friction.{joint}", f"{leg} dynamic friction", category, "N·m", "Per-joint dynamic friction coefficient.", None, 0.001),
                PhysicsField(f"simulation_physics.joint_viscous_friction.{joint}", f"{leg} viscous friction", category, "N·m·s/rad", "Per-joint velocity-proportional friction coefficient.", None, 0.001),
            ]
        )
    return fields


def _passive_spring_fields() -> list[PhysicsField]:
    fields: list[PhysicsField] = []
    default_rest = (math.pi / 4, math.pi / 4, -math.pi / 4, math.pi / 4, math.pi / 4, math.pi / 4)
    for index, leg in enumerate(_LEG_LABELS):
        prefix = f"simulation_physics.passive_spring.damper_{index}"
        fields.extend(
            [
                PhysicsField(f"{prefix}.stiffness", f"{leg} spring stiffness", "Torsion springs", "N·m/rad", "Torsion-spring stiffness for the stable damper alias.", 200.0, 0.1),
                PhysicsField(f"{prefix}.damping", f"{leg} spring damping", "Torsion springs", "N·m·s/rad", "Torsion-spring damping; zero is the uncalibrated repository default.", 0.0, 0.1),
                PhysicsField(f"{prefix}.rest_position_rad", f"{leg} rest position", "Torsion springs", "rad", "Torsion-spring neutral angle for the stable damper alias.", default_rest[index], 0.001, -math.pi, math.pi),
            ]
        )
    return fields


def _abad_mapping_fields() -> list[PhysicsField]:
    fields: list[PhysicsField] = []
    for index, leg in enumerate(_LEG_LABELS):
        fields.extend(
            [
                PhysicsField(f"hardware_mapping.abad_target_scale.abad_{index}", f"{leg} target scale", "ABAD calibration", "ratio", "Measured ABAD target scale in actual = scale × requested + offset.", 1.0, 0.001, 0.5, 1.5),
                PhysicsField(f"hardware_mapping.abad_target_offset_rad.abad_{index}", f"{leg} target offset", "ABAD calibration", "rad", "Measured ABAD target offset.", 0.0, 0.001, -0.35, 0.35),
            ]
        )
    return fields


FIELD_SCHEMA: tuple[PhysicsField, ...] = tuple(
    [
        PhysicsField("simulation_physics.rigid_body.linear_damping", "Linear damping", "Rigid body", "1/s", "Damping applied to robot linear motion.", 0.05, 0.01),
        PhysicsField("simulation_physics.rigid_body.angular_damping", "Angular damping", "Rigid body", "1/s", "Damping applied to robot angular motion.", 0.10, 0.01),
        PhysicsField("simulation_physics.mass.scale", "Mass scale", "Mass and center of mass", "ratio", "Uniformly scales every body mass and inertia at runtime.", 1.0, 0.01, 0.000001),
        PhysicsField("simulation_physics.mass.added_mass_kg", "Added root mass", "Mass and center of mass", "kg", "Mass added to the root body after uniform scaling.", 0.0, 0.01),
        PhysicsField("simulation_physics.mass.com_offset_m.0", "Root CoM X offset", "Mass and center of mass", "m", "Root-body center-of-mass offset along X.", 0.0, 0.001, None),
        PhysicsField("simulation_physics.mass.com_offset_m.1", "Root CoM Y offset", "Mass and center of mass", "m", "Root-body center-of-mass offset along Y.", 0.0, 0.001, None),
        PhysicsField("simulation_physics.mass.com_offset_m.2", "Root CoM Z offset", "Mass and center of mass", "m", "Root-body center-of-mass offset along Z.", 0.0, 0.001, None),
        PhysicsField("simulation_physics.ground.static_friction", "Static friction", "Contact material", "coefficient", "Robot/ground static friction coefficient.", 1.2, 0.01),
        PhysicsField("simulation_physics.ground.dynamic_friction", "Dynamic friction", "Contact material", "coefficient", "Robot/ground dynamic friction coefficient; cannot exceed static friction.", 1.0, 0.01),
        PhysicsField("simulation_physics.ground.restitution", "Restitution", "Contact material", "coefficient", "Collision bounce coefficient from 0 to 1.", 0.0, 0.01, 0.0, 1.0),
        PhysicsField("sensor_timing.aggregate_command_delay_s", "Command delay", "Timing", "s", "Aggregate actuator-command delay, quantized to physics steps.", 0.0, 0.001),
    ]
    + _actuator_fields("main_drive", "Main drive", (0.0, 1.0, 15.0, 15.0))
    + _actuator_fields("abad", "ABAD", (40.0, 4.0, 8.0, 5.0))
    + _torsion_spring_actuator_fields()
    + _joint_fields("main", "Main drive")
    + _joint_fields("abad", "ABAD")
    + _joint_fields("damper", "Torsion spring")
    + _passive_spring_fields()
    + _abad_mapping_fields()
)
FIELD_MAP = {field.key: field for field in FIELD_SCHEMA}

BUILT_IN_PHYSICS_PRESETS: tuple[dict[str, Any], ...] = (
    {
        "id": "baseline",
        "name": "Baseline",
        "description": "Inherit the repository and USD physical defaults without a candidate profile.",
        "built_in": True,
        "values": {},
    },
)


def _slugify(name: str) -> str:
    return _SLUG_RE.sub("-", name.lower().strip()).strip("-") or "physics-profile"


def normalize_physics_values(values: dict[str, Any] | None) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for raw_key, raw_value in (values or {}).items():
        key = str(raw_key)
        field = FIELD_MAP.get(key)
        if field is None:
            raise ValueError(f"Unknown physics field: {key}")
        if raw_value in (None, ""):
            continue
        if isinstance(raw_value, bool):
            raise ValueError(f"Physics field {key} must be numeric")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Physics field {key} must be numeric") from exc
        if not math.isfinite(value):
            raise ValueError(f"Physics field {key} must be finite")
        if field.minimum is not None and value < field.minimum:
            raise ValueError(f"Physics field {key} must be at least {field.minimum}")
        if field.maximum is not None and value > field.maximum:
            raise ValueError(f"Physics field {key} must be at most {field.maximum}")
        normalized[key] = value
    if normalized:
        values_to_profile("panel-validation", normalized)
    return normalized


def _set_nested(root: dict[str, Any], parts: list[str], value: float) -> None:
    target = root
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


def values_to_profile(profile_id: str, values: dict[str, Any], description: str = "") -> CalibrationProfileV1:
    normalized = {
        str(key): float(value)
        for key, value in values.items()
        if str(key) in FIELD_MAP and value not in (None, "")
    }
    ground_static = "simulation_physics.ground.static_friction"
    ground_dynamic = "simulation_physics.ground.dynamic_friction"
    if ground_static in normalized or ground_dynamic in normalized:
        normalized.setdefault(ground_static, float(FIELD_MAP[ground_static].default))
        normalized.setdefault(ground_dynamic, float(FIELD_MAP[ground_dynamic].default))
    for index in range(6):
        spring_prefix = f"simulation_physics.passive_spring.damper_{index}"
        spring_keys = [key for key in normalized if key.startswith(f"{spring_prefix}.")]
        if spring_keys:
            stiffness_key = f"{spring_prefix}.stiffness"
            normalized.setdefault(stiffness_key, float(FIELD_MAP[stiffness_key].default))
    payload: dict[str, Any] = {
        "schema_version": 1,
        "profile_id": profile_id,
        "description": description,
        "hardware_mapping": {},
        "sensor_timing": {},
        "simulation_physics": {},
    }
    com_values: dict[int, float] = {}
    for key, value in normalized.items():
        if key.startswith("simulation_physics.mass.com_offset_m."):
            com_values[int(key.rsplit(".", 1)[1])] = value
            continue
        _set_nested(payload, key.split("."), value)
    if com_values:
        mass = payload["simulation_physics"].setdefault("mass", {})
        mass["com_offset_m"] = [com_values.get(index, 0.0) for index in range(3)]
    try:
        return CalibrationProfileV1.from_dict(payload)
    except ContractError as exc:
        raise ValueError(str(exc)) from exc


def write_physics_profile(path: Path, profile_id: str, description: str, values: dict[str, Any]) -> Path | None:
    normalized = normalize_physics_values(values)
    if not normalized:
        if path.exists():
            path.unlink()
        return None
    profile = values_to_profile(_slugify(profile_id), normalized, description)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def physics_catalog() -> dict[str, Any]:
    return {
        "field_schema": [field.to_dict() for field in FIELD_SCHEMA],
        "field_count": len(FIELD_SCHEMA),
        "mode": "sparse-calibration-profile",
        "contract": "CalibrationProfileV1",
    }


class PhysicsPresetStore:
    def __init__(self, preset_file: Path) -> None:
        self._file = preset_file
        self._lock = threading.Lock()
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        if self._file.exists():
            try:
                data = json.loads(self._file.read_text(encoding="utf-8"))
                custom = []
                for preset in data.get("presets", []):
                    if preset.get("id") == "baseline":
                        continue
                    custom.append({
                        "id": str(preset["id"]),
                        "name": str(preset["name"]),
                        "description": str(preset.get("description") or ""),
                        "built_in": False,
                        "values": normalize_physics_values(dict(preset.get("values") or {})),
                    })
                active = str(data.get("active_preset_id") or "baseline")
                ids = {"baseline", *(preset["id"] for preset in custom)}
                return {
                    "active_preset_id": active if active in ids else "baseline",
                    "presets": [copy.deepcopy(BUILT_IN_PHYSICS_PRESETS[0]), *custom],
                }
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass
        return {
            "active_preset_id": "baseline",
            "presets": [copy.deepcopy(BUILT_IN_PHYSICS_PRESETS[0])],
        }

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        self._file.write_text(json.dumps(self._data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def list_presets(self) -> list[dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._data["presets"])

    def get_preset(self, preset_id: str) -> dict[str, Any] | None:
        with self._lock:
            for preset in self._data["presets"]:
                if preset["id"] == preset_id:
                    return copy.deepcopy(preset)
        return None

    def create_preset(self, name: str, description: str, values: dict[str, Any]) -> dict[str, Any]:
        if not name.strip():
            raise ValueError("name is required")
        normalized = normalize_physics_values(values)
        with self._lock:
            slug = _slugify(name)
            existing = {preset["id"] for preset in self._data["presets"]}
            base, counter = slug, 2
            while slug in existing:
                slug = f"{base}-{counter}"
                counter += 1
            preset = {
                "id": slug,
                "name": name.strip(),
                "description": description.strip(),
                "built_in": False,
                "values": normalized,
            }
            self._data["presets"].append(preset)
            self._save()
            return copy.deepcopy(preset)

    def update_preset(self, preset_id: str, **updates: Any) -> dict[str, Any]:
        with self._lock:
            for preset in self._data["presets"]:
                if preset["id"] != preset_id:
                    continue
                if preset.get("built_in"):
                    raise ValueError(f"Cannot modify built-in physics preset '{preset_id}'")
                if "name" in updates:
                    name = str(updates["name"]).strip()
                    if not name:
                        raise ValueError("name is required")
                    preset["name"] = name
                if "description" in updates:
                    preset["description"] = str(updates["description"]).strip()
                if "values" in updates:
                    preset["values"] = normalize_physics_values(dict(updates["values"]))
                self._save()
                return copy.deepcopy(preset)
        raise KeyError(f"Physics preset '{preset_id}' not found")

    def delete_preset(self, preset_id: str) -> bool:
        with self._lock:
            for preset in self._data["presets"]:
                if preset["id"] != preset_id:
                    continue
                if preset.get("built_in"):
                    raise ValueError(f"Cannot delete built-in physics preset '{preset_id}'")
                self._data["presets"] = [item for item in self._data["presets"] if item["id"] != preset_id]
                if self._data["active_preset_id"] == preset_id:
                    self._data["active_preset_id"] = "baseline"
                self._save()
                return True
        return False

    def get_active_preset_id(self) -> str:
        with self._lock:
            return str(self._data.get("active_preset_id") or "baseline")

    def set_active_preset(self, preset_id: str) -> None:
        with self._lock:
            if preset_id not in {preset["id"] for preset in self._data["presets"]}:
                raise KeyError(f"Physics preset '{preset_id}' not found")
            self._data["active_preset_id"] = preset_id
            self._save()
