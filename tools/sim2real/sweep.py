from __future__ import annotations

import copy
import itertools
import math
from typing import Any, Mapping

from .contracts import CalibrationProfileV1, ContractError, ScenarioSpecV1
from .traces import sha256_json


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
) -> list[CalibrationProfileV1]:
    payload = base.to_dict()
    changes: list[tuple[str, float]] = []
    for path, values in _space(search_space):
        current = float(_get(payload, path))
        changes.extend((path, value) for value in values if value != current)
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
