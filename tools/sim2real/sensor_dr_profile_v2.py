"""Hashable, evidence-bound Sensor V2 robustness profiles.

The profile contains ranges, not sampled values.  It is intentionally separate
from the Isaac environment so profiles can be reviewed and validated in CI
without importing Isaac Lab.  No non-neutral default is provided here: ranges
must come from a measured or explicitly reviewed evidence artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


PROFILE_SCHEMA_V2 = "redrhex.sensor-dr-profile.v2"
PROFILE_PURPOSES_V2 = frozenset({"training_curriculum", "held_out_evaluation"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_SENSOR_FLOAT_RANGES = frozenset(
    {
        "sensor_dr_gyro_noise_std_range_rad_s",
        "sensor_dr_gyro_bias_range_rad_s",
        "sensor_dr_gyro_drift_std_range_rad_s_sqrt_s",
        "sensor_dr_gyro_filter_time_constant_range_s",
        "sensor_dr_imu_mount_roll_range_rad",
        "sensor_dr_imu_mount_pitch_range_rad",
        "sensor_dr_imu_mount_yaw_range_rad",
        "sensor_dr_encoder_zero_offset_range_rad",
        "sensor_dr_encoder_noise_std_range_rad",
        "sensor_dr_encoder_quantization_range_rad",
        "sensor_dr_encoder_stale_probability_range",
        "sensor_dr_encoder_dropout_probability_range",
        "sensor_dr_accel_noise_std_range_m_s2",
        "sensor_dr_accel_bias_range_m_s2",
    }
)
_SENSOR_INT_RANGES = frozenset(
    {
        "sensor_dr_gyro_latency_steps_range",
        "sensor_dr_gyro_latency_jitter_steps_range",
        "sensor_dr_encoder_latency_steps_range",
    }
)
_PHYSICAL_FLOAT_RANGES = frozenset(
    {
        "dr_main_actuator_strength_range",
        "dr_abad_actuator_strength_range",
        "dr_friction_range",
        "dr_mass_range",
    }
)
_BOOLEAN_PARAMETERS = frozenset(
    {
        "domain_randomization_enable",
        "dr_try_physical_material_randomization",
        "dr_randomize_actuator_strength",
        "dr_randomize_friction",
        "dr_randomize_mass",
    }
)
_INTEGER_PARAMETERS = frozenset({"sim2real_command_delay_steps"})
PHYSICAL_PROFILE_PARAMETERS_V2 = (
    _PHYSICAL_FLOAT_RANGES | _BOOLEAN_PARAMETERS | _INTEGER_PARAMETERS
)

_NOISE_PARAMETERS = frozenset(
    {
        "sensor_dr_gyro_noise_std_range_rad_s",
        "sensor_dr_gyro_bias_range_rad_s",
        "sensor_dr_gyro_drift_std_range_rad_s_sqrt_s",
        "sensor_dr_gyro_filter_time_constant_range_s",
        "sensor_dr_imu_mount_roll_range_rad",
        "sensor_dr_imu_mount_pitch_range_rad",
        "sensor_dr_imu_mount_yaw_range_rad",
        "sensor_dr_encoder_zero_offset_range_rad",
        "sensor_dr_encoder_noise_std_range_rad",
        "sensor_dr_encoder_quantization_range_rad",
        "sensor_dr_accel_noise_std_range_m_s2",
        "sensor_dr_accel_bias_range_m_s2",
    }
)
_LATENCY_PARAMETERS = frozenset(
    {
        "sensor_dr_gyro_latency_steps_range",
        "sensor_dr_gyro_latency_jitter_steps_range",
        "sensor_dr_encoder_latency_steps_range",
        "sensor_dr_encoder_stale_probability_range",
        "sensor_dr_encoder_dropout_probability_range",
        "sim2real_command_delay_steps",
    }
)
ACTIVE_CATEGORY_PARAMETERS_V2 = {
    "noise": _NOISE_PARAMETERS,
    "latency": _LATENCY_PARAMETERS,
    "actuator": frozenset(
        {
            "dr_main_actuator_strength_range",
            "dr_abad_actuator_strength_range",
        }
    ),
    "friction": frozenset({"dr_friction_range"}),
    "mass": frozenset({"dr_mass_range"}),
}
ALLOWED_PARAMETERS_V2 = (
    _SENSOR_FLOAT_RANGES
    | _SENSOR_INT_RANGES
    | _PHYSICAL_FLOAT_RANGES
    | _BOOLEAN_PARAMETERS
    | _INTEGER_PARAMETERS
)

_NONNEGATIVE_RANGES = frozenset(
    {
        "sensor_dr_gyro_noise_std_range_rad_s",
        "sensor_dr_gyro_drift_std_range_rad_s_sqrt_s",
        "sensor_dr_gyro_filter_time_constant_range_s",
        "sensor_dr_encoder_noise_std_range_rad",
        "sensor_dr_encoder_quantization_range_rad",
        "sensor_dr_accel_noise_std_range_m_s2",
        "dr_main_actuator_strength_range",
        "dr_abad_actuator_strength_range",
        "dr_friction_range",
        "dr_mass_range",
    }
)
_PROBABILITY_RANGES = frozenset(
    {
        "sensor_dr_encoder_stale_probability_range",
        "sensor_dr_encoder_dropout_probability_range",
    }
)


class SensorDrProfileErrorV2(ValueError):
    """Raised when a robustness profile is ambiguous or unauditable."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_range(name: str, raw: Any, *, integer: bool) -> tuple[float, float] | tuple[int, int]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise SensorDrProfileErrorV2(f"{name} must be a two-value range")
    if integer:
        if any(isinstance(value, bool) or not isinstance(value, int) for value in raw):
            raise SensorDrProfileErrorV2(f"{name} must contain two integers")
        values: tuple[int, int] = (int(raw[0]), int(raw[1]))
    else:
        values = (float(raw[0]), float(raw[1]))
        if not all(math.isfinite(value) for value in values):
            raise SensorDrProfileErrorV2(f"{name} contains NaN or Inf")
    if values[0] > values[1]:
        raise SensorDrProfileErrorV2(f"{name} lower bound exceeds upper bound")
    if name in _NONNEGATIVE_RANGES and values[0] < 0:
        raise SensorDrProfileErrorV2(f"{name} must be non-negative")
    if name in _PROBABILITY_RANGES and not (0.0 <= values[0] <= values[1] <= 1.0):
        raise SensorDrProfileErrorV2(f"{name} must stay within [0, 1]")
    if name in {
        "sensor_dr_gyro_latency_steps_range",
        "sensor_dr_encoder_latency_steps_range",
    } and values[0] < 0:
        raise SensorDrProfileErrorV2(f"{name} must be non-negative")
    return values


@dataclass(frozen=True)
class EvidenceReferenceV2:
    artifact: str
    sha256: str
    note: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "EvidenceReferenceV2":
        required = {"artifact", "sha256", "note"}
        if set(raw) != required:
            raise SensorDrProfileErrorV2(
                "each evidence record requires exactly artifact, sha256, and note"
            )
        artifact = str(raw["artifact"]).strip()
        digest = str(raw["sha256"]).strip()
        note = str(raw["note"]).strip()
        if not artifact or not note or not _SHA256.fullmatch(digest):
            raise SensorDrProfileErrorV2(
                "evidence artifact/note must be non-empty and sha256 must be lowercase hex"
            )
        return cls(artifact=artifact, sha256=digest, note=note)


@dataclass(frozen=True)
class SensorDrProfileV2:
    profile_id: str
    purpose: str
    evidence: tuple[EvidenceReferenceV2, ...]
    parameters: Mapping[str, Any]
    schema: str = PROFILE_SCHEMA_V2

    def __post_init__(self) -> None:
        if self.schema != PROFILE_SCHEMA_V2:
            raise SensorDrProfileErrorV2(f"schema must be {PROFILE_SCHEMA_V2}")
        if not self.profile_id.strip():
            raise SensorDrProfileErrorV2("profile_id must not be empty")
        if self.purpose not in PROFILE_PURPOSES_V2:
            raise SensorDrProfileErrorV2(
                f"purpose must be one of {sorted(PROFILE_PURPOSES_V2)}"
            )
        if not self.evidence:
            raise SensorDrProfileErrorV2(
                "non-neutral robustness profiles require at least one evidence record"
            )
        unknown = set(self.parameters) - ALLOWED_PARAMETERS_V2
        if unknown:
            raise SensorDrProfileErrorV2(
                f"unsupported Sensor V2 profile parameters: {sorted(unknown)}"
            )
        if not self.parameters:
            raise SensorDrProfileErrorV2("profile parameters must not be empty")
        normalized: dict[str, Any] = {}
        for name, raw in self.parameters.items():
            if name in _BOOLEAN_PARAMETERS:
                if not isinstance(raw, bool):
                    raise SensorDrProfileErrorV2(f"{name} must be boolean")
                normalized[name] = raw
            elif name in _INTEGER_PARAMETERS:
                if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
                    raise SensorDrProfileErrorV2(f"{name} must be a non-negative integer")
                normalized[name] = int(raw)
            elif name in _SENSOR_INT_RANGES:
                normalized[name] = _finite_range(name, raw, integer=True)
            else:
                normalized[name] = _finite_range(name, raw, integer=False)
        sensor_active = any(
            name in (_SENSOR_FLOAT_RANGES | _SENSOR_INT_RANGES)
            and isinstance(value, tuple)
            and any(float(item) != 0.0 for item in value)
            for name, value in normalized.items()
        )
        command_delay_active = int(normalized.get("sim2real_command_delay_steps", 0)) > 0
        physical_requirements = {
            "dr_main_actuator_strength_range": "dr_randomize_actuator_strength",
            "dr_abad_actuator_strength_range": "dr_randomize_actuator_strength",
            "dr_friction_range": "dr_randomize_friction",
            "dr_mass_range": "dr_randomize_mass",
        }
        enabled_range_requirements = {
            "dr_randomize_actuator_strength": {
                "dr_main_actuator_strength_range",
                "dr_abad_actuator_strength_range",
            },
            "dr_randomize_friction": {"dr_friction_range"},
            "dr_randomize_mass": {"dr_mass_range"},
        }
        enabled_flags = {
            name for name in enabled_range_requirements if normalized.get(name) is True
        }
        if enabled_flags and normalized.get("domain_randomization_enable") is not True:
            raise SensorDrProfileErrorV2(
                "enabled physical DR flags require domain_randomization_enable=true"
            )
        if normalized.get("domain_randomization_enable") is True and not enabled_flags:
            raise SensorDrProfileErrorV2(
                "domain_randomization_enable=true requires an explicit physical DR flag"
            )
        physical_material_flags = {
            "dr_randomize_friction",
            "dr_randomize_mass",
        } & enabled_flags
        if (
            physical_material_flags
            and normalized.get("dr_try_physical_material_randomization") is not True
        ):
            raise SensorDrProfileErrorV2(
                "friction/mass DR requires "
                "dr_try_physical_material_randomization=true; controller-target "
                "fallbacks are not physical F4/F5 evidence"
            )
        for flag_name in enabled_flags:
            missing_ranges = enabled_range_requirements[flag_name] - set(normalized)
            if missing_ranges:
                raise SensorDrProfileErrorV2(
                    f"{flag_name}=true requires profile-bound ranges: {sorted(missing_ranges)}"
                )
        physical_active = False
        for range_name, flag_name in physical_requirements.items():
            if range_name not in normalized:
                continue
            value = normalized[range_name]
            assert isinstance(value, tuple)
            range_active = tuple(float(item) for item in value) != (1.0, 1.0)
            if range_active and (
                normalized.get("domain_randomization_enable") is not True
                or normalized.get(flag_name) is not True
            ):
                raise SensorDrProfileErrorV2(
                    f"active {range_name} requires domain_randomization_enable=true "
                    f"and {flag_name}=true in the same profile"
                )
            physical_active |= range_active
        if enabled_flags and not physical_active:
            raise SensorDrProfileErrorV2(
                "enabled physical DR ranges are neutral and do not perturb the domain"
            )
        if not (sensor_active or command_delay_active or physical_active):
            raise SensorDrProfileErrorV2("profile is neutral and cannot provide F4/F5 evidence")
        object.__setattr__(self, "parameters", normalized)

    @property
    def active_categories(self) -> frozenset[str]:
        """Return semantic perturbation categories used by F4/F5 gates."""

        active: set[str] = set()
        sensor_active_names = {
            name
            for name, value in self.parameters.items()
            if name in (_SENSOR_FLOAT_RANGES | _SENSOR_INT_RANGES)
            and isinstance(value, tuple)
            and any(float(item) != 0.0 for item in value)
        }
        if sensor_active_names:
            active.add("sensor")
        if sensor_active_names & _NOISE_PARAMETERS:
            active.add("noise")
        if (
            sensor_active_names & _LATENCY_PARAMETERS
            or int(self.parameters.get("sim2real_command_delay_steps", 0)) > 0
        ):
            active.add("latency")
        strength_active = any(
            name in self.parameters
            and tuple(float(item) for item in self.parameters[name]) != (1.0, 1.0)
            for name in (
                "dr_main_actuator_strength_range",
                "dr_abad_actuator_strength_range",
            )
        )
        if strength_active:
            active.add("actuator")
        if (
            "dr_friction_range" in self.parameters
            and tuple(float(item) for item in self.parameters["dr_friction_range"])
            != (1.0, 1.0)
        ):
            active.add("friction")
        if (
            "dr_mass_range" in self.parameters
            and tuple(float(item) for item in self.parameters["dr_mass_range"])
            != (1.0, 1.0)
        ):
            active.add("mass")
        return frozenset(active)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SensorDrProfileV2":
        required = {"schema", "profile_id", "purpose", "evidence", "parameters"}
        if set(raw) != required:
            missing = required - set(raw)
            unknown = set(raw) - required
            raise SensorDrProfileErrorV2(
                f"profile keys mismatch; missing={sorted(missing)} unknown={sorted(unknown)}"
            )
        evidence_raw = raw["evidence"]
        if not isinstance(evidence_raw, list) or not all(
            isinstance(item, Mapping) for item in evidence_raw
        ):
            raise SensorDrProfileErrorV2("evidence must be a list of objects")
        parameters = raw["parameters"]
        if not isinstance(parameters, Mapping):
            raise SensorDrProfileErrorV2("parameters must be an object")
        return cls(
            schema=str(raw["schema"]),
            profile_id=str(raw["profile_id"]),
            purpose=str(raw["purpose"]),
            evidence=tuple(EvidenceReferenceV2.from_mapping(item) for item in evidence_raw),
            parameters=dict(parameters),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "profile_id": self.profile_id,
            "purpose": self.purpose,
            "evidence": [
                {"artifact": item.artifact, "sha256": item.sha256, "note": item.note}
                for item in self.evidence
            ],
            "parameters": {
                name: list(value) if isinstance(value, tuple) else value
                for name, value in sorted(self.parameters.items())
            },
        }


def load_sensor_dr_profile_v2(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_purpose: str | None = None,
) -> tuple[SensorDrProfileV2, str]:
    profile_path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SensorDrProfileErrorV2(f"cannot read Sensor V2 profile {profile_path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise SensorDrProfileErrorV2("Sensor V2 profile must contain a JSON object")
    digest = file_sha256(profile_path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise SensorDrProfileErrorV2(
            f"Sensor V2 profile SHA-256 mismatch: expected {expected_sha256}, got {digest}"
        )
    profile = SensorDrProfileV2.from_mapping(raw)
    for reference in profile.evidence:
        artifact = Path(reference.artifact).expanduser()
        if not artifact.is_absolute():
            artifact = profile_path.parent / artifact
        artifact = artifact.resolve()
        if not artifact.is_file():
            raise SensorDrProfileErrorV2(
                f"Sensor V2 evidence artifact does not exist: {artifact}"
            )
        actual_evidence_sha256 = file_sha256(artifact)
        if actual_evidence_sha256 != reference.sha256:
            raise SensorDrProfileErrorV2(
                "Sensor V2 evidence SHA-256 mismatch for "
                f"{artifact}: expected {reference.sha256}, got {actual_evidence_sha256}"
            )
    if expected_purpose is not None and profile.purpose != expected_purpose:
        raise SensorDrProfileErrorV2(
            f"profile purpose {profile.purpose!r} does not match {expected_purpose!r}"
        )
    return profile, digest


def apply_sensor_dr_profile_v2(env_cfg: Any, profile: SensorDrProfileV2, digest: str) -> None:
    """Apply only validated allowlisted fields to an Isaac environment config."""

    for name, value in profile.parameters.items():
        if not hasattr(env_cfg, name):
            raise SensorDrProfileErrorV2(f"environment config has no profile field {name!r}")
        setattr(env_cfg, name, value)
    if profile.active_categories & {"actuator", "friction", "mass"}:
        if not hasattr(env_cfg, "curriculum_stage_scales"):
            raise SensorDrProfileErrorV2(
                "environment config has no physical DR curriculum scale"
            )
        # A reviewed profile already contains the curriculum/evaluation range.
        # Do not silently contract it through ForwardFast's legacy 0.05 scale.
        env_cfg.curriculum_stage_scales = [1.0]
        env_cfg.sensor_dr_physical_stage_scale = 1.0
    env_cfg.sensor_dr_require_physical_material_writes = bool(
        profile.active_categories & {"friction", "mass"}
    )
    env_cfg.sensor_dr_evidence = (
        f"profile={profile.profile_id};sha256={digest};"
        + ",".join(reference.sha256 for reference in profile.evidence)
    )
    env_cfg.sensor_dr_profile_id = profile.profile_id
    env_cfg.sensor_dr_profile_sha256 = digest
    env_cfg.sensor_dr_profile_purpose = profile.purpose
