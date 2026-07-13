from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .contracts import ContractError, ScenarioSpecV1
from .metrics import compute_subsystem_metrics
from .provenance import validate_real_trace_provenance
from .scenarios import load_scenario
from .traces import LoadedTrace, load_trace, sha256_json


def _loaded(
    value: LoadedTrace | str | Path, scenario: ScenarioSpecV1
) -> LoadedTrace:
    if not isinstance(value, LoadedTrace):
        return load_trace(value, scenario=scenario)
    if value.manifest.scenario_id != scenario.scenario_id:
        raise ContractError("scenario id mismatch")
    if value.manifest.provenance.get("scenario_sha256") != sha256_json(
        scenario.to_dict()
    ):
        raise ContractError("scenario hash mismatch")
    return value


def _compatible(real: LoadedTrace, sim: LoadedTrace, scenario: ScenarioSpecV1) -> None:
    validate_real_trace_provenance(real, scenario)
    if sim.manifest.source != "sim":
        raise ContractError('sim trace must have source "sim"')
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
            if len(real_value) != len(sim_value):
                continue
            if all(isinstance(item, dict) for item in real_value) and all(
                isinstance(item, dict) for item in sim_value
            ):
                records: list[dict[str, Any]] = []
                for real_record, sim_record in zip(
                    real_value, sim_value, strict=True
                ):
                    record = _delta(real_record, sim_record)
                    for identity in ("repeat_index", "direction", "label"):
                        if (
                            identity in real_record
                            and real_record.get(identity) == sim_record.get(identity)
                        ):
                            record[identity] = real_record[identity]
                    records.append(record)
                result[key] = records
                continue
            try:
                left = np.asarray(real_value, dtype=float)
                right = np.asarray(sim_value, dtype=float)
            except (TypeError, ValueError):
                continue
            if left.shape == right.shape:
                result[key] = (right - left).tolist()
    return result


def compare_traces(
    real: LoadedTrace | str | Path,
    sim: LoadedTrace | str | Path,
    *,
    scenario: ScenarioSpecV1 | str | Path | None = None,
) -> dict[str, Any]:
    spec = (
        scenario
        if isinstance(scenario, ScenarioSpecV1)
        else load_scenario(
            scenario
            or (
                real.manifest.scenario_id
                if isinstance(real, LoadedTrace)
                else load_trace(real).manifest.scenario_id
            )
        )
    )
    real_trace = _loaded(real, spec)
    sim_trace = _loaded(sim, spec)
    _compatible(real_trace, sim_trace, spec)
    real_metrics = compute_subsystem_metrics(spec, real_trace)
    sim_metrics = compute_subsystem_metrics(spec, sim_trace)
    return {
        "schema_version": 1,
        "scenario_id": spec.scenario_id,
        "delta_convention": "sim_minus_real",
        "subsystems": {
            spec.subsystem: {
                "real": real_metrics,
                "sim": sim_metrics,
                "delta": _delta(real_metrics, sim_metrics),
            }
        },
    }
