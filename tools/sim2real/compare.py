from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .contracts import ContractError, ScenarioSpecV1
from .metrics import compute_subsystem_metrics
from .scenarios import load_scenario
from .traces import LoadedTrace, load_trace


def _loaded(value: LoadedTrace | str | Path) -> LoadedTrace:
    return value if isinstance(value, LoadedTrace) else load_trace(value)


def _compatible(real: LoadedTrace, sim: LoadedTrace, scenario: ScenarioSpecV1) -> None:
    if real.manifest.scenario_id != sim.manifest.scenario_id:
        raise ContractError("scenario id mismatch")
    for field, label in (("units", "unit"), ("frames", "frame")):
        real_values = real.manifest.metadata[field]
        sim_values = sim.manifest.metadata[field]
        for channel in scenario.required_channels:
            if real_values[channel] != sim_values[channel]:
                raise ContractError(
                    f"{label} mismatch for {channel}: {real_values[channel]} != {sim_values[channel]}"
                )


def _delta(real: dict[str, Any], sim: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in sorted(set(real) & set(sim)):
        real_value = real[key]
        sim_value = sim[key]
        if isinstance(real_value, (int, float)) and isinstance(sim_value, (int, float)):
            result[key] = float(sim_value - real_value)
        elif isinstance(real_value, dict) and isinstance(sim_value, dict):
            result[key] = _delta(real_value, sim_value)
        elif isinstance(real_value, list) and isinstance(sim_value, list):
            left = np.asarray(real_value, dtype=float)
            right = np.asarray(sim_value, dtype=float)
            if left.shape == right.shape:
                result[key] = (right - left).tolist()
    return result


def compare_traces(
    real: LoadedTrace | str | Path,
    sim: LoadedTrace | str | Path,
    *,
    scenario: ScenarioSpecV1 | str | Path | None = None,
) -> dict[str, Any]:
    real_trace = _loaded(real)
    sim_trace = _loaded(sim)
    spec = (
        scenario
        if isinstance(scenario, ScenarioSpecV1)
        else load_scenario(scenario or real_trace.manifest.scenario_id)
    )
    _compatible(real_trace, sim_trace, spec)
    real_metrics = compute_subsystem_metrics(spec, real_trace)
    sim_metrics = compute_subsystem_metrics(spec, sim_trace)
    return {
        "schema_version": 1,
        "scenario_id": spec.scenario_id,
        "subsystems": {
            spec.subsystem: {
                "real": real_metrics,
                "sim": sim_metrics,
                "delta": _delta(real_metrics, sim_metrics),
            }
        },
    }
