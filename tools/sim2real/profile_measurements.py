from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Sequence

from .contracts import CalibrationProfileV1, ContractError, ScenarioSpecV1
from .metrics import compute_subsystem_metrics
from .provenance import validate_real_trace_provenance
from .scenarios import load_scenario
from .traces import LoadedTrace, load_trace


def _expected_metadata(
    scenario: ScenarioSpecV1,
) -> tuple[dict[str, str], dict[str, str]]:
    if scenario.experiment_kind == "abad_static":
        units = {
            "command": "rad",
            "position": "rad",
            "repeat_index": "1",
            "settled": "1",
        }
        return units, {name: scenario.joint for name in units}
    if scenario.experiment_kind == "friction":
        units = {
            "breakaway_force": "N",
            "static_normal_load": "N",
            "static_repeat_index": "1",
            "dynamic_pull_force": "N",
            "dynamic_normal_load": "N",
            "dynamic_speed": "m/s",
            "dynamic_repeat_index": "1",
        }
        frame = f"{scenario.joint}/ground"
        return units, {name: frame for name in units}
    raise ContractError(
        f"scenario {scenario.scenario_id} is not a direct profile measurement"
    )


def _load_measurement_trace(path: str | Path) -> tuple[ScenarioSpecV1, LoadedTrace]:
    discovered = load_trace(path, require_managed_dataset=True)
    if discovered.manifest.scenario_id not in {"abad-static", "friction"}:
        raise ContractError(
            f"scenario {discovered.manifest.scenario_id} is not a direct profile measurement"
        )
    scenario = load_scenario(discovered.manifest.scenario_id)
    units, frames = _expected_metadata(scenario)
    trace = load_trace(
        path,
        scenario=scenario,
        require_managed_dataset=True,
        expected_metadata_sha256=discovered.metadata_sha256,
        expected_units=units,
        expected_frames=frames,
    )
    validate_real_trace_provenance(trace, scenario)
    if trace.dataset is None:  # pragma: no cover - guarded by load_trace
        raise ContractError("measurement trace has no managed dataset identity")
    return scenario, trace


def _source_record(
    scenario: ScenarioSpecV1,
    trace: LoadedTrace,
    *,
    metric_kind: str,
    frame: str,
    repeat_count: int,
) -> dict[str, Any]:
    dataset = trace.dataset
    if dataset is None:  # pragma: no cover - guarded by _load_measurement_trace
        raise ContractError("measurement trace has no managed dataset identity")
    return {
        "trace_sha256": trace.manifest.provenance["trace_sha256"],
        "metadata_sha256": trace.metadata_sha256,
        "scenario_id": scenario.scenario_id,
        "scenario_sha256": trace.manifest.provenance["scenario_sha256"],
        "source": trace.manifest.source,
        "metric_kind": metric_kind,
        "frame": frame,
        "repeat_count": repeat_count,
        "dataset_id": dataset.dataset_id,
        "episode_id": dataset.episode_id,
    }


def apply_measurements_to_profile(
    baseline: CalibrationProfileV1,
    *,
    profile_id: str,
    trace_paths: Sequence[str | Path],
) -> CalibrationProfileV1:
    """Build a candidate only from verified, managed real measurement traces.

    Metrics and source digests are recomputed from immutable dataset episodes;
    callers cannot supply either result directly.
    """

    source = baseline.validate()
    if isinstance(trace_paths, (str, Path)) or not trace_paths:
        raise ContractError("trace_paths must contain at least one trace artifact")

    payload = copy.deepcopy(source.to_dict())
    payload["profile_id"] = profile_id
    hardware = payload["hardware_mapping"]
    physics = payload["simulation_physics"]
    sources = payload["measurement_sources"]
    applied_keys: set[str] = set()

    for path in trace_paths:
        scenario, trace = _load_measurement_trace(path)
        metrics = compute_subsystem_metrics(scenario, trace)
        if scenario.experiment_kind == "abad_static":
            key = f"abad_target:{scenario.joint}"
            if key in applied_keys:
                raise ContractError(f"duplicate measurement source for {key}")
            repeat_count = len(metrics["repeats"])
            if repeat_count != scenario.repeats:
                raise ContractError("ABAD metric repeat count does not match its scenario")
            hardware.setdefault("abad_target_scale", {})[scenario.joint] = metrics[
                "aggregate"
            ]["target_scale"]
            hardware.setdefault("abad_target_offset_rad", {})[scenario.joint] = metrics[
                "aggregate"
            ]["target_offset_rad"]
            sources[key] = _source_record(
                scenario,
                trace,
                metric_kind="abad_static_mapping",
                frame=scenario.joint,
                repeat_count=repeat_count,
            )
        elif scenario.experiment_kind == "friction":
            key = "ground_friction"
            if key in applied_keys:
                raise ContractError("duplicate measurement source for ground_friction")
            static_count = int(metrics["static"]["coefficient_count"])
            dynamic_count = int(metrics["dynamic"]["coefficient_count"])
            if static_count != scenario.repeats or dynamic_count != scenario.repeats:
                raise ContractError("friction metric repeat count does not match its scenario")
            ground = physics.setdefault("ground", {})
            ground["static_friction"] = metrics["static"]["coefficient_mean"]
            ground["dynamic_friction"] = metrics["dynamic"]["coefficient_mean"]
            sources[key] = _source_record(
                scenario,
                trace,
                metric_kind="ground_friction",
                frame=f"{scenario.joint}/ground",
                repeat_count=static_count,
            )
        else:  # pragma: no cover - guarded by _load_measurement_trace
            raise ContractError(f"unsupported measurement scenario: {scenario.scenario_id}")
        applied_keys.add(key)

    return CalibrationProfileV1.from_dict(payload)
