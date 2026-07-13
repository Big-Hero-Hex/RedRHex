from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import ContractError, ScenarioSpecV1
from .metrics import compute_subsystem_metrics
from .scenarios import load_scenario
from .traces import LoadedTrace, load_trace, sha256_json


_PROFILE_SOURCE_PREFIX = "profile:"
_PROBE_SOURCE_PREFIX = "authenticated_probe_events:"


def _measured_profile_source(value: Any) -> str | None:
    if not isinstance(value, str) or not value.startswith(_PROFILE_SOURCE_PREFIX):
        return None
    profile_id = value.removeprefix(_PROFILE_SOURCE_PREFIX)
    if not profile_id or ":" in profile_id:
        return None
    return profile_id


def _validate_main_drive_mapping_provenance(
    real: LoadedTrace,
    scenario: ScenarioSpecV1,
) -> None:
    constants = real.manifest.metadata.get("calibration_constants", {})
    if not isinstance(constants, Mapping):
        raise ContractError("real main-drive trace has invalid hardware mapping provenance")

    position_source = constants.get("position_mapping_source")
    position_profile_id = _measured_profile_source(position_source)
    if (
        position_profile_id is None
        or "profile_sha256" not in real.manifest.provenance
    ):
        raise ContractError(
            "real main-drive trace uses provisional or missing hardware mapping provenance "
            "for encoder position"
        )

    command_source = constants.get("requested_command_source")
    command_profile_id = _measured_profile_source(command_source)
    if command_profile_id is not None:
        if command_profile_id != position_profile_id:
            raise ContractError(
                "real main-drive trace hardware mapping provenance names different profiles"
            )
        return

    scenario_hash = sha256_json(scenario.to_dict())
    if command_source != f"{_PROBE_SOURCE_PREFIX}{scenario_hash}":
        raise ContractError(
            "real main-drive trace uses provisional or missing hardware mapping provenance "
            "for requested command"
        )
    if scenario.experiment_kind != "step_coast":
        raise ContractError(
            "authenticated probe-event command provenance is only valid for step_coast scenarios"
        )
    evidence = constants.get("probe_event_evidence")
    if not isinstance(evidence, Mapping):
        raise ContractError("authenticated probe-event mapping provenance is missing evidence")

    expected_duration_s = sum(
        float(segment["duration_s"]) for segment in scenario.command_segments
    ) * scenario.repeats
    expected_ticks = int(round(expected_duration_s * 60.0))
    expected_segment_count = scenario.repeats * len(scenario.command_segments)
    try:
        valid_evidence = (
            evidence["scenario_sha256"] == scenario_hash
            and evidence["abad_output_disabled_verified"] is True
            and int(evidence["repetition_count"]) == scenario.repeats
            and int(evidence["segment_count"]) == expected_segment_count
            and int(evidence["complete_ticks"]) == expected_ticks
            and math.isclose(
                float(evidence["receive_duration_s"]),
                expected_duration_s,
                rel_tol=0.0,
                abs_tol=float(evidence["receive_jitter_bound_s"]),
            )
            and 0.0 < float(evidence["receive_jitter_bound_s"]) <= 1.0 / 60.0
        )
    except (KeyError, TypeError, ValueError):
        valid_evidence = False
    if not valid_evidence:
        raise ContractError("authenticated probe-event mapping provenance evidence is invalid")


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
    if real.manifest.source != "real":
        raise ContractError('real trace must have source "real"')
    if sim.manifest.source != "sim":
        raise ContractError('sim trace must have source "sim"')
    if real.manifest.scenario_id != sim.manifest.scenario_id:
        raise ContractError("scenario id mismatch")
    if scenario.subsystem == "main_drive":
        _validate_main_drive_mapping_provenance(real, scenario)
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
