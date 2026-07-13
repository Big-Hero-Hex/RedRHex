from __future__ import annotations

import copy
import itertools
import math
from typing import Any, Mapping

from .contracts import CalibrationProfileV1, ContractError, ScenarioSpecV1
from .traces import sha256_json


_SWEEP_MODES = {"one-factor", "coarse-grid"}
_MAIN_ACTUATOR_FIELDS = {
    "damping",
    "effort_limit",
    "velocity_limit",
    "armature",
    "friction",
}


def _space(search_space: Mapping[str, list[float]]) -> list[tuple[str, list[float]]]:
    if not isinstance(search_space, Mapping) or not search_space:
        raise ContractError("search_space must be a non-empty mapping")
    result: list[tuple[str, list[float]]] = []
    for path in sorted(search_space):
        if not isinstance(path, str) or path.count(".") < 1:
            raise ContractError("search-space keys must be dotted profile paths")
        raw_values = search_space[path]
        if not isinstance(raw_values, list) or not raw_values:
            raise ContractError(f"search-space values for {path} must be a non-empty array")
        values: list[float] = []
        for raw in raw_values:
            if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
                raise ContractError(f"search-space values for {path} must be finite numbers")
            values.append(float(raw))
        result.append((path, sorted(set(values))))
    return result


def _get(payload: Mapping[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ContractError(f"profile path does not exist: {path}")
        value = value[part]
    return value


def _set(payload: dict[str, Any], path: str, value: float) -> None:
    parts = path.split(".")
    target: dict[str, Any] = payload
    for part in parts[:-1]:
        child = target.get(part)
        if not isinstance(child, dict):
            raise ContractError(f"profile path does not exist: {path}")
        target = child
    if parts[-1] not in target:
        raise ContractError(f"profile path does not exist: {path}")
    target[parts[-1]] = value


def _changed_paths(before: Any, after: Any, prefix: str) -> set[str]:
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        result: set[str] = set()
        for key in set(before) | set(after):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in before or key not in after:
                value = after.get(key, before.get(key))
                if isinstance(value, Mapping):
                    result.update(_changed_paths({}, value, path))
                else:
                    result.add(path)
            else:
                result.update(_changed_paths(before[key], after[key], path))
        return result
    return set() if before == after else {prefix}


def profile_parameter_changes(
    base: CalibrationProfileV1,
    candidate: CalibrationProfileV1,
) -> set[str]:
    """Return behavior-bearing candidate paths, excluding labels and prose."""

    result: set[str] = set()
    for field in (
        "hardware_mapping",
        "sensor_timing",
        "simulation_physics",
        "measurement_sources",
    ):
        result.update(
            _changed_paths(getattr(base, field), getattr(candidate, field), field)
        )
    return result


def _identifiable_sweep_paths(scenario: ScenarioSpecV1) -> set[str]:
    if scenario.subsystem != "main_drive" or scenario.experiment_kind not in {
        "step",
        "coast",
        "step_coast",
    }:
        return set()
    allowed = {
        f"simulation_physics.main_drive.{field}"
        for field in _MAIN_ACTUATOR_FIELDS
    }
    for section in (
        "joint_friction",
        "joint_dynamic_friction",
        "joint_viscous_friction",
    ):
        allowed.add(f"simulation_physics.{section}.{scenario.joint}")
    return allowed


def validate_sweep_candidates(
    base: CalibrationProfileV1,
    candidates: list[CalibrationProfileV1] | tuple[CalibrationProfileV1, ...],
    scenario: ScenarioSpecV1,
    *,
    sweep_mode: str,
) -> list[set[str]]:
    """Fail closed on compensating or scenario-unidentifiable candidate changes."""

    if sweep_mode not in _SWEEP_MODES:
        raise ContractError(f"unsupported sweep_mode: {sweep_mode}")
    allowed = _identifiable_sweep_paths(scenario)
    changes_by_candidate: list[set[str]] = []
    for index, candidate in enumerate(candidates, start=1):
        changes = profile_parameter_changes(base, candidate)
        unsupported = sorted(changes - allowed)
        if unsupported:
            raise ContractError(
                f"candidate {index} change is not identifiable by {scenario.scenario_id}: "
                + ", ".join(unsupported)
            )
        if sweep_mode == "one-factor" and len(changes) != 1:
            raise ContractError(
                f"one-factor candidate {index} must change exactly one parameter"
            )
        if sweep_mode == "coarse-grid" and len(changes) > 2:
            raise ContractError(
                f"coarse-grid candidate {index} may change at most two parameters"
            )
        changes_by_candidate.append(changes)
    if sweep_mode == "coarse-grid" and len(set().union(*changes_by_candidate)) > 2:
        raise ContractError("coarse-grid sweep may vary at most two parameter paths")
    return changes_by_candidate


def _candidate(
    base: CalibrationProfileV1,
    changes: Mapping[str, float],
    *,
    mode: str,
    index: int,
) -> CalibrationProfileV1:
    payload = copy.deepcopy(base.to_dict())
    payload["profile_id"] = f"{base.profile_id}-{mode}-{index:04d}"
    for path, value in changes.items():
        _set(payload, path, value)
    return CalibrationProfileV1.from_dict(payload)


def generate_one_factor_candidates(
    base: CalibrationProfileV1,
    search_space: Mapping[str, list[float]],
    *,
    max_candidates: int = 256,
) -> list[CalibrationProfileV1]:
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or max_candidates < 1:
        raise ContractError("max_candidates must be a positive integer")
    payload = base.to_dict()
    changes: list[tuple[str, float]] = []
    for path, values in _space(search_space):
        current = float(_get(payload, path))
        changes.extend((path, value) for value in values if value != current)
    if len(changes) > max_candidates:
        raise ContractError(
            f"one-factor sweep has {len(changes)} candidates, "
            f"exceeding max_candidates={max_candidates}"
        )
    return [
        _candidate(base, {path: value}, mode="one-factor", index=index)
        for index, (path, value) in enumerate(changes, start=1)
    ]


def generate_coarse_grid_candidates(
    base: CalibrationProfileV1,
    search_space: Mapping[str, list[float]],
    *,
    max_candidates: int = 256,
) -> list[CalibrationProfileV1]:
    dimensions = _space(search_space)
    if len(dimensions) > 2:
        raise ContractError("coarse grids may vary at most two parameter paths")
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or max_candidates < 1:
        raise ContractError("max_candidates must be a positive integer")
    count = math.prod(len(values) for _, values in dimensions)
    if count > max_candidates:
        raise ContractError(f"grid has {count} candidates, exceeding max_candidates={max_candidates}")
    paths = [path for path, _ in dimensions]
    combinations = itertools.product(*(values for _, values in dimensions))
    return [
        _candidate(
            base,
            dict(zip(paths, values, strict=True)),
            mode="coarse-grid",
            index=index,
        )
        for index, values in enumerate(combinations, start=1)
    ]


def candidate_cache_key(
    profile: CalibrationProfileV1,
    scenario: ScenarioSpecV1,
    *,
    provenance: Mapping[str, Any] | None = None,
) -> str:
    return sha256_json(
        {
            "schema_version": 1,
            "hardware_mapping": profile.hardware_mapping,
            "sensor_timing": profile.sensor_timing,
            "simulation_physics": profile.simulation_physics,
            "scenario": scenario.to_dict(),
            "provenance": dict(provenance or {}),
        }
    )


one_factor_candidates = generate_one_factor_candidates
coarse_grid_candidates = generate_coarse_grid_candidates
cache_key = candidate_cache_key
