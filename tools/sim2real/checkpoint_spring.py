from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from .contracts import load_profile
from .profile_measurements import verify_representative_spring_source
from .traces import sha256_json


_BACKENDS = {"explicit", "native"}
_CALIBRATION_STATUSES = {"calibrated", "uncalibrated"}
_SPRING_ALIASES = tuple(f"damper_{index}" for index in range(6))


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _load_spring_metadata(path: Path) -> Mapping[str, Any]:
    import yaml

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"torsion-spring metadata is invalid: {exc}") from exc
    metadata = _mapping(payload, "torsion-spring metadata")
    backend = metadata.get("spring_backend")
    status = metadata.get("calibration_status")
    if backend not in _BACKENDS or status not in _CALIBRATION_STATUSES:
        raise ValueError(
            "torsion-spring metadata requires a valid spring_backend and calibration_status"
        )
    return metadata


def _load_profile_binding(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"calibrated checkpoint profile metadata is invalid: {exc}") from exc
    metadata = _mapping(payload, "calibrated checkpoint profile metadata")
    if metadata.get("schema_version") != 1 or isinstance(metadata.get("schema_version"), bool):
        raise ValueError("calibrated checkpoint profile metadata has unsupported schema_version")
    profile_id = metadata.get("profile_id")
    profile_sha256 = metadata.get("profile_sha256")
    if not isinstance(profile_id, str) or not profile_id:
        raise ValueError("calibrated checkpoint profile metadata has invalid profile_id")
    if (
        not isinstance(profile_sha256, str)
        or len(profile_sha256) != 64
        or any(character not in "0123456789abcdef" for character in profile_sha256)
    ):
        raise ValueError("calibrated checkpoint profile metadata has invalid profile_sha256")
    return metadata


def validate_checkpoint_spring_evaluation(
    run_dir: str | Path,
    *,
    selected_backend: str,
    selected_profile_id: str | None,
    selected_profile_sha256: str | None,
) -> str:
    """Reject physics changes when evaluating a calibrated spring checkpoint."""

    if selected_backend not in _BACKENDS:
        raise ValueError(f"unsupported selected spring backend: {selected_backend}")
    params = Path(run_dir) / "params"
    spring_path = params / "torsion_spring.yaml"
    if not spring_path.is_file():
        return "uncalibrated"

    spring = _load_spring_metadata(spring_path)
    status = str(spring["calibration_status"])
    if status == "uncalibrated":
        return status

    checkpoint_backend = str(spring["spring_backend"])
    if selected_backend != checkpoint_backend:
        raise ValueError(
            "calibrated checkpoint spring backend mismatch: "
            f"expected {checkpoint_backend}, got {selected_backend}"
        )
    if selected_profile_id is None or selected_profile_sha256 is None:
        raise ValueError("calibrated checkpoint evaluation requires --physics-profile")

    binding = _load_profile_binding(params / "physics_profile_metadata.json")
    if selected_profile_id != binding["profile_id"]:
        raise ValueError(
            "calibrated checkpoint profile_id mismatch: "
            f"expected {binding['profile_id']}, got {selected_profile_id}"
        )
    if selected_profile_sha256 != binding["profile_sha256"]:
        raise ValueError(
            "calibrated checkpoint profile_sha256 mismatch: "
            f"expected {binding['profile_sha256']}, got {selected_profile_sha256}"
        )
    return status


def _six_finite_numbers(value: Any, label: str) -> tuple[float, ...]:
    if (
        not isinstance(value, list)
        or len(value) != 6
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in value
        )
    ):
        raise ValueError(f"torsion-spring deployment {label} must contain six finite numbers")
    return tuple(float(item) for item in value)


def validate_checkpoint_spring_deployment(run_dir: str | Path) -> dict[str, Any]:
    """Fail closed unless a checkpoint retains authenticated spring evidence."""

    params = Path(run_dir) / "params"
    spring_path = params / "torsion_spring.yaml"
    if not spring_path.is_file():
        raise ValueError("deployment requires a calibrated torsion-spring checkpoint")
    spring = _load_spring_metadata(spring_path)
    if spring["calibration_status"] != "calibrated":
        raise ValueError("deployment requires a calibrated torsion-spring checkpoint")

    binding = _load_profile_binding(params / "physics_profile_metadata.json")
    profile = load_profile(params / "physics_profile.json")
    profile_payload = profile.to_dict()
    profile_sha256 = sha256_json(profile_payload)
    if (
        binding["profile_id"] != profile.profile_id
        or binding["profile_sha256"] != profile_sha256
        or spring.get("profile_id") != profile.profile_id
        or spring.get("profile_sha256") != profile_sha256
    ):
        raise ValueError("calibrated checkpoint spring profile binding mismatch")

    source = profile.measurement_sources.get("passive_spring:damper_0")
    if not isinstance(source, Mapping):
        raise ValueError("calibrated checkpoint omits representative spring evidence")
    verified = verify_representative_spring_source(source)
    quality = verified.get("quality") if isinstance(verified, Mapping) else None
    calibration = verified.get("calibration") if isinstance(verified, Mapping) else None
    if (
        not isinstance(quality, Mapping)
        or quality.get("accepted") is not True
        or not isinstance(calibration, Mapping)
    ):
        raise ValueError("calibrated checkpoint spring evidence did not pass quality gates")
    if source.get("applies_to") != list(_SPRING_ALIASES):
        raise ValueError("calibrated checkpoint spring evidence does not cover all six joints")

    fitted_stiffness = float(calibration["neutral_stiffness_nm_per_rad"])
    fitted_rest = float(calibration["rest_position_rad"])
    if not math.isfinite(fitted_stiffness) or fitted_stiffness < 0.0 or not math.isfinite(fitted_rest):
        raise ValueError("calibrated checkpoint spring fit is invalid")
    passive_springs = profile.simulation_physics.get("passive_spring")
    if not isinstance(passive_springs, Mapping):
        raise ValueError("calibrated checkpoint profile omits six passive springs")
    for alias in _SPRING_ALIASES:
        entry = passive_springs.get(alias)
        if not isinstance(entry, Mapping):
            raise ValueError("calibrated checkpoint profile omits six passive springs")
        if not math.isclose(
            float(entry.get("stiffness", math.nan)),
            fitted_stiffness,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("calibrated checkpoint profile stiffness no longer matches evidence")
        if float(entry.get("damping", math.nan)) != 0.0:
            raise ValueError("calibrated checkpoint profile damping must remain zero")

    aliases = spring.get("joint_aliases")
    if aliases != list(_SPRING_ALIASES):
        raise ValueError("torsion-spring deployment joint alias order is invalid")
    stiffness = _six_finite_numbers(
        spring.get("stiffness_nm_per_rad"), "stiffness"
    )
    damping = _six_finite_numbers(
        spring.get("damping_nm_s_per_rad"), "damping"
    )
    neutral = _six_finite_numbers(spring.get("neutral_angle_rad"), "neutral angles")
    if any(
        not math.isclose(value, fitted_stiffness, rel_tol=0.0, abs_tol=1.0e-12)
        for value in stiffness
    ):
        raise ValueError("torsion-spring deployment stiffness does not match evidence")
    if any(value != 0.0 for value in damping):
        raise ValueError("torsion-spring deployment damping must remain zero")
    if not math.isclose(neutral[0], fitted_rest, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("torsion-spring deployment neutral angle does not match evidence")

    return {
        "spring_backend": str(spring["spring_backend"]),
        "calibration_status": "calibrated",
        "profile_id": profile.profile_id,
        "profile_sha256": profile_sha256,
        "stiffness_nm_per_rad": list(stiffness),
        "damping_nm_s_per_rad": list(damping),
        "neutral_angle_rad": list(neutral),
    }
