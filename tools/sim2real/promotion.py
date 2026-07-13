from __future__ import annotations

import math
import json
import re
from pathlib import Path
from typing import Any, Mapping

from .contracts import CalibrationProfileV1, ContractError
from .traces import sha256_json


_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_-]*")
_AUDIT_FIELDS = {
    "units_pass",
    "frames_pass",
    "joint_sign_pass",
    "mass_pass",
    "imu_mount_pass",
    "contact_sensor_pass",
}
_HELD_OUT_DIMENSIONS = {"leg", "direction", "command_level", "load"}


def load_validation_evidence(path: str | Path) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key}")
            result[key] = value
        return result

    source = Path(path)
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot load validation evidence {source}: {exc}") from exc
    return _mapping(payload, "validation evidence")


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
        label = "unknown validation evidence fields" if name == "validation evidence" else f"unknown {name} fields"
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


def _episodes(value: Any, condition_id: str, seen: set[str]) -> tuple[list[dict[str, Any]], int]:
    if not isinstance(value, list) or not value:
        raise ContractError(f"condition {condition_id} real_episodes must be non-empty")
    clean: list[dict[str, Any]] = []
    repetitions = 0
    for index, raw in enumerate(value):
        episode = _mapping(raw, f"condition {condition_id} real_episodes[{index}]")
        _exact_fields(
            episode,
            name=f"condition {condition_id} real episode",
            required={"episode_id", "trace_sha256", "repetition_count"},
        )
        episode_id = _identifier(episode["episode_id"], "episode_id")
        if episode_id in seen:
            raise ContractError("real episode_id values must be unique")
        seen.add(episode_id)
        count = episode["repetition_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ContractError("repetition_count must be a positive integer")
        repetitions += count
        clean.append(
            {
                "episode_id": episode_id,
                "trace_sha256": _sha(episode["trace_sha256"], "trace_sha256"),
                "repetition_count": count,
            }
        )
    return clean, repetitions


def _metric_result(value: Any, *, name: str) -> dict[str, Any]:
    metric = _mapping(value, f"metric {name}")
    _exact_fields(
        metric,
        name=f"metric {name}",
        required={
            "unit",
            "real_mean",
            "real_std",
            "instrument_uncertainty",
            "sim_value",
        },
    )
    unit = metric["unit"]
    if not isinstance(unit, str) or not unit.strip():
        raise ContractError(f"metric {name} unit must be non-empty")
    real_mean = _number(metric["real_mean"], f"metric {name}.real_mean")
    real_std = _number(metric["real_std"], f"metric {name}.real_std", nonnegative=True)
    uncertainty = _number(
        metric["instrument_uncertainty"],
        f"metric {name}.instrument_uncertainty",
        nonnegative=True,
    )
    sim_value = _number(metric["sim_value"], f"metric {name}.sim_value")
    tolerance = max(uncertainty, 2.0 * real_std)
    error = abs(sim_value - real_mean)
    return {
        "unit": unit,
        "real_mean": real_mean,
        "real_std": real_std,
        "instrument_uncertainty": uncertainty,
        "sim_value": sim_value,
        "tolerance": tolerance,
        "absolute_error": error,
        "pass": error <= tolerance,
    }


def evaluate_promotion(
    profile: CalibrationProfileV1,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate promotion evidence without mutating any training configuration."""

    candidate = profile.validate()
    data = _mapping(evidence, "validation evidence")
    _exact_fields(
        data,
        name="validation evidence",
        required={
            "schema_version",
            "candidate_profile_sha256",
            "audit",
            "fitted_subsystems",
            "conditions",
            "actuator_model_envelope",
        },
    )
    if data["schema_version"] != 1 or isinstance(data["schema_version"], bool):
        raise ContractError("validation evidence schema_version must be 1")
    expected_profile_hash = sha256_json(candidate.to_dict())
    if _sha(data["candidate_profile_sha256"], "candidate profile hash") != expected_profile_hash:
        raise ContractError("candidate profile hash mismatch")

    audit = _mapping(data["audit"], "audit")
    _exact_fields(audit, name="audit", required=_AUDIT_FIELDS)
    if any(not isinstance(audit[field], bool) for field in _AUDIT_FIELDS):
        raise ContractError("audit fields must be booleans")

    raw_subsystems = data["fitted_subsystems"]
    if not isinstance(raw_subsystems, list) or not raw_subsystems:
        raise ContractError("fitted_subsystems must be a non-empty array")
    fitted = [_identifier(item, "fitted_subsystems") for item in raw_subsystems]
    if len(fitted) != len(set(fitted)):
        raise ContractError("fitted_subsystems must be unique")

    raw_conditions = data["conditions"]
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise ContractError("conditions must be a non-empty array")
    condition_ids: set[str] = set()
    episode_ids: set[str] = set()
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        subsystem: {"calibration": [], "holdout": []} for subsystem in fitted
    }
    failures = [f"audit.{field} failed" for field in sorted(_AUDIT_FIELDS) if not audit[field]]

    for index, raw in enumerate(raw_conditions):
        condition = _mapping(raw, f"conditions[{index}]")
        _exact_fields(
            condition,
            name="condition",
            required={"condition_id", "subsystem", "role", "real_episodes", "metrics"},
            optional={"held_out_by", "sim_trace_sha256"},
        )
        condition_id = _identifier(condition["condition_id"], "condition_id")
        if condition_id in condition_ids:
            raise ContractError("condition_id values must be unique")
        condition_ids.add(condition_id)
        subsystem = _identifier(condition["subsystem"], "condition subsystem")
        if subsystem not in grouped:
            raise ContractError(f"condition references unfitted subsystem {subsystem}")
        role = condition["role"]
        if role not in {"calibration", "holdout"}:
            raise ContractError("condition role must be calibration or holdout")
        episodes, repetitions = _episodes(condition["real_episodes"], condition_id, episode_ids)
        if repetitions < 3:
            failures.append(
                f"{subsystem}.{condition_id} requires at least three real repetitions"
            )
        raw_metrics = _mapping(condition["metrics"], f"condition {condition_id} metrics")
        clean_condition: dict[str, Any] = {
            "condition_id": condition_id,
            "real_episodes": episodes,
            "real_repetition_count": repetitions,
            "metrics": {},
        }
        if role == "calibration":
            if "held_out_by" in condition or "sim_trace_sha256" in condition:
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
            clean_condition["sim_trace_sha256"] = _sha(
                condition.get("sim_trace_sha256"), "sim_trace_sha256"
            )
            if not raw_metrics:
                raise ContractError("holdout condition metrics must be non-empty")
            for metric_name, metric in sorted(raw_metrics.items()):
                name = _identifier(metric_name, "metric name")
                result = _metric_result(metric, name=name)
                clean_condition["metrics"][name] = result
                if not result["pass"]:
                    failures.append(
                        f"{subsystem}.{condition_id}.{name} is outside its held-out envelope"
                    )
        grouped[subsystem][role].append(clean_condition)

    envelope = _mapping(data["actuator_model_envelope"], "actuator_model_envelope")
    if set(envelope) - set(fitted):
        raise ContractError("actuator_model_envelope references an unfitted subsystem")
    if "main_drive" in fitted and "main_drive" not in envelope:
        raise ContractError("main_drive actuator_model_envelope result is required")
    for subsystem, status in envelope.items():
        if status not in {"inside", "mismatch"}:
            raise ContractError("actuator model envelope must be inside or mismatch")

    subsystem_results: dict[str, Any] = {}
    for subsystem in fitted:
        calibration = grouped[subsystem]["calibration"]
        holdout = grouped[subsystem]["holdout"]
        if not calibration:
            failures.append(f"{subsystem} is missing a calibration condition")
        if not holdout:
            failures.append(f"{subsystem} is missing a holdout condition")
        mismatch = envelope.get(subsystem) == "mismatch"
        if mismatch:
            failures.append(
                f"{subsystem} actuator-model mismatch: bounded candidates miss the holdout envelope"
            )
        local_prefix = f"{subsystem}."
        subsystem_results[subsystem] = {
            "pass": not any(reason.startswith(local_prefix) for reason in failures),
            "actuator_model_mismatch": mismatch,
            "calibration_conditions": calibration,
            "holdout_conditions": holdout,
        }

    return {
        "schema_version": 1,
        "profile_id": candidate.profile_id,
        "candidate_profile_sha256": expected_profile_hash,
        "eligible_for_review": not failures,
        "promotion_requires_reviewed_config_change": True,
        "audit": dict(audit),
        "subsystems": subsystem_results,
        "failures": failures,
    }
