from __future__ import annotations

import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np

from .contracts import CalibrationProfileV1, ContractError, load_profile
from .metrics import compute_subsystem_metrics
from .scenarios import load_scenario
from .traces import LoadedTrace, load_trace, sha256_file, sha256_json


_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_-]*")
_METRIC_PATH = re.compile(r"[a-z][a-z0-9_.-]*")
_AUDIT_FIELDS = {
    "units_pass",
    "frames_pass",
    "joint_sign_pass",
    "mass_pass",
    "imu_mount_pass",
    "contact_sensor_pass",
}
_HELD_OUT_DIMENSIONS = {"leg", "direction", "command_level", "load"}


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
    required = {"path", "trace_sha256"}
    optional = {"episode_id"}
    _exact_fields(binding, name=name, required=required, optional=optional)
    path = _artifact(root, binding["path"], name)
    loaded = load_trace(path, scenario=scenario, profile=profile)
    expected = _sha(binding["trace_sha256"], f"{name} trace_sha256")
    actual = loaded.manifest.provenance["trace_sha256"]
    if actual != expected:
        raise ContractError(f"{name} trace hash mismatch")
    if loaded.manifest.source != source:
        raise ContractError(f"{name} must have source={source!r}")
    return loaded, binding


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
    fitted = _changed_subsystems(baseline, candidate)
    if not fitted:
        raise ContractError("candidate profile has no fitted subsystem changes from baseline")

    audit_path, _ = _file_binding(root, data["audit_artifact"], "audit artifact")
    audit_payload = _json(audit_path, "audit artifact")
    _exact_fields(audit_payload, name="audit artifact", required={"schema_version", "checks"})
    if audit_payload["schema_version"] != 1 or isinstance(audit_payload["schema_version"], bool):
        raise ContractError("audit artifact schema_version must be 1")
    audit = _mapping(audit_payload["checks"], "audit checks")
    _exact_fields(audit, name="audit checks", required=_AUDIT_FIELDS)
    if any(not isinstance(audit[field], bool) for field in _AUDIT_FIELDS):
        raise ContractError("audit checks must be booleans")
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
    real_hash_roles: dict[str, str] = {}
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
            trace_hash = loaded.manifest.provenance["trace_sha256"]
            previous_role = real_hash_roles.get(trace_hash)
            if previous_role is not None and previous_role != role:
                raise ContractError("calibration and holdout real artifacts must be disjoint")
            if previous_role is not None:
                raise ContractError("real trace hashes must be unique across conditions")
            real_hash_roles[trace_hash] = role
            episode_scenario = load_scenario(loaded.manifest.scenario_id)
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
            sim_trace, _ = _trace_binding(
                root,
                condition.get("sim_artifact"),
                f"condition {condition_id} simulator artifact",
                source="sim",
                scenario=scenario,
                profile=candidate,
            )
            if _condition_coordinates(sim_trace, scenario) != coordinates:
                raise ContractError("simulator artifact coordinates do not match real holdout")
            if not raw_metrics:
                raise ContractError("holdout condition metrics must be non-empty")
            sim_metrics = compute_subsystem_metrics(scenario, sim_trace)
            real_metric_sets = [compute_subsystem_metrics(scenario, trace) for trace in loaded_real]
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
                uncertainty = _number(
                    metric["instrument_uncertainty"],
                    f"metric {metric_path}.instrument_uncertainty",
                    nonnegative=True,
                )
                observations = [
                    _observation(metrics, metric_path, _repeat_count(metrics, trace))
                    for metrics, trace in zip(real_metric_sets, loaded_real, strict=True)
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
            clean_condition["held_out_by"] = list(held_out_by)
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
        }

    for condition_id, internal in condition_internal.items():
        if internal["role"] != "holdout":
            continue
        calibrations = grouped[internal["subsystem"]]["calibration"]
        if not calibrations:
            continue
        for dimension in internal["held_out_by"]:
            held_value = internal["coordinates"].get(dimension)
            if held_value is None or all(
                item["coordinates"].get(dimension) == held_value for item in calibrations
            ):
                raise ContractError(
                    f"held-out dimension {dimension} does not differ from calibration"
                )

    raw_sweeps = _mapping(data["actuator_sweeps"], "actuator_sweeps")
    if set(raw_sweeps) - fitted:
        raise ContractError("actuator_sweeps references a subsystem not changed by the candidate")
    actuator_mismatch: dict[str, bool] = {subsystem: False for subsystem in fitted}
    if "main_drive" in fitted:
        sweep = _mapping(raw_sweeps.get("main_drive"), "main_drive actuator sweep")
        _exact_fields(
            sweep,
            name="main_drive actuator sweep",
            required={"results_path", "results_sha256", "candidate_artifacts"},
        )
        results_path = _artifact(root, sweep["results_path"], "actuator sweep results")
        if sha256_file(results_path) != _sha(sweep["results_sha256"], "sweep results hash"):
            raise ContractError("actuator sweep results hash mismatch")
        sweep_results = _json(results_path, "actuator sweep results")
        if sweep_results.get("schema_version") != 1:
            raise ContractError("actuator sweep results schema_version must be 1")
        completed_hashes = {
            item.get("trace_sha256")
            for item in sweep_results.get("candidates", [])
            if isinstance(item, Mapping) and item.get("status") in {"completed", "cached"}
        }
        raw_candidates = sweep["candidate_artifacts"]
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise ContractError("actuator sweep candidate_artifacts must be non-empty")
        holdouts = grouped["main_drive"]["holdout"]
        candidate_inside = False
        for index, raw_candidate in enumerate(raw_candidates):
            if not holdouts:
                break
            scenario = load_scenario(holdouts[0]["scenario_id"])
            trace, _ = _trace_binding(
                root,
                raw_candidate,
                f"actuator sweep candidate {index}",
                source="sim",
                scenario=scenario,
            )
            trace_hash = trace.manifest.provenance["trace_sha256"]
            if trace_hash not in completed_hashes:
                raise ContractError("actuator sweep candidate is absent from completed results")
            metrics = compute_subsystem_metrics(scenario, trace)
            candidate_passes = True
            for condition in holdouts:
                for metric_path, expected in condition["metrics"].items():
                    value, _, _ = _lookup(metrics, metric_path)
                    if abs(value - expected["real_mean"]) > expected["tolerance"]:
                        candidate_passes = False
            candidate_inside |= candidate_passes
        actuator_mismatch["main_drive"] = not candidate_inside

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
        "audit": dict(audit),
        "derived_fitted_subsystems": sorted(fitted),
        "subsystems": subsystem_results,
        "failures": all_failures,
    }
