from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import CalibrationProfileV1, ContractError, ScenarioSpecV1
from .metrics import (
    compute_subsystem_metrics,
    torsional_spring_holdout_trace_metrics,
    torsional_spring_quality_gates,
)
from .provenance import validate_real_trace_provenance
from .scenarios import load_scenario
from .traces import LoadedTrace, load_trace


_CANONICAL_JOINTS = {
    f"{group}_{index}"
    for group in ("main", "abad", "damper")
    for index in range(6)
}
TORSION_SPRING_ALIASES = tuple(f"damper_{index}" for index in range(6))
_TORSION_SPRING_PROTOCOL_LEVELS = {
    "torsion-spring": ((0.2, 0.4, 0.6, 0.8), "20/40/60/80%"),
    "torsion-spring-holdout": ((0.3, 0.5, 0.7), "30/50/70%"),
}


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
    if scenario.experiment_kind == "mass_com":
        units = {
            "scale_mass": "kg",
            "support_force": "N",
            "support_position": "m",
            "repeat_index": "1",
        }
        return units, {name: "root" for name in units}
    if scenario.experiment_kind == "spring":
        units = {
            "load_force": "N",
            "lever_arm": "m",
            "angle": "rad",
            "repeat_index": "1",
        }
        if "torque_direction" in scenario.required_channels:
            units["torque_direction"] = "1"
        if "sweep_branch" in scenario.required_channels:
            units["sweep_branch"] = "1"
        return units, {name: scenario.joint for name in units}
    if scenario.experiment_kind == "manual_load":
        units = {
            "load_force": "N",
            "lever_arm": "m",
            "command": "normalized",
            "direction": "1",
            "saturation_confirmed": "1",
            "repeat_index": "1",
        }
        return units, {name: scenario.joint for name in units}
    raise ContractError(
        f"scenario {scenario.scenario_id} is not a direct profile measurement"
    )


def _load_measurement_trace(path: str | Path) -> tuple[ScenarioSpecV1, LoadedTrace]:
    discovered = load_trace(path, require_managed_dataset=True)
    if discovered.manifest.scenario_id not in {
        "abad-static",
        "friction",
        "mass-com",
        "spring",
        "torsion-spring",
        "torsion-spring-holdout",
        "manual-load",
    }:
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
    applies_to: Sequence[str] | None = None,
    rest_position_rad: float | None = None,
) -> dict[str, Any]:
    dataset = trace.dataset
    if dataset is None:  # pragma: no cover - guarded by _load_measurement_trace
        raise ContractError("measurement trace has no managed dataset identity")
    record = {
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
    if applies_to is not None:
        record["applies_to"] = list(applies_to)
        record["episode_path"] = str(trace.directory.resolve())
    if rest_position_rad is not None:
        record["rest_position_rad"] = float(rest_position_rad)
    return record


def representative_spring_aliases(
    scenario: ScenarioSpecV1,
    trace: LoadedTrace,
) -> tuple[str, ...]:
    """Resolve and strictly validate a representative spring declaration."""

    if scenario.scenario_id != "torsion-spring":
        return (scenario.joint,)
    _spring_mechanical_approval(trace)
    constants = trace.manifest.metadata.get("calibration_constants", {})
    raw_aliases = (
        constants.get("applies_to_spring_aliases")
        if isinstance(constants, Mapping)
        else None
    )
    if not isinstance(raw_aliases, list) or raw_aliases != list(TORSION_SPRING_ALIASES):
        raise ContractError(
            "calibration_constants.applies_to_spring_aliases must contain exactly "
            "damper_0 through damper_5 in canonical order"
        )
    if scenario.joint != TORSION_SPRING_ALIASES[0]:
        raise ContractError("representative torsion-spring scenario must measure damper_0")
    return TORSION_SPRING_ALIASES


def _spring_mechanical_approval(trace: LoadedTrace) -> dict[str, Any]:
    constants = trace.manifest.metadata.get("calibration_constants", {})
    approval = (
        constants.get("mechanical_owner_approval")
        if isinstance(constants, Mapping)
        else None
    )
    identity_fields = {"owner", "fixture_id"}
    envelope_fields = {
        "maximum_safe_deflection_rad",
        "maximum_safe_load_n",
        "maximum_safe_torque_nm",
    }
    selected_envelopes = (
        set(approval).intersection(envelope_fields)
        if isinstance(approval, Mapping)
        else set()
    )
    if (
        not isinstance(approval, Mapping)
        or set(approval) != identity_fields | selected_envelopes
        or len(selected_envelopes) != 1
    ):
        raise ContractError(
            "torsion-spring measurement requires mechanical owner approval of "
            "the fixture and exactly one maximum safe deflection, load, or torque envelope"
        )
    if any(
        not isinstance(approval[field], str) or not approval[field].strip()
        for field in ("owner", "fixture_id")
    ):
        raise ContractError(
            "torsion-spring mechanical owner approval must identify owner and fixture"
        )
    envelope_field = next(iter(selected_envelopes))
    maximum = approval[envelope_field]
    if (
        isinstance(maximum, bool)
        or not isinstance(maximum, (int, float))
        or not math.isfinite(float(maximum))
        or float(maximum) <= 0.0
    ):
        raise ContractError(
            "torsion-spring mechanical owner approval requires a positive safe envelope"
        )
    rest = constants.get("rest_position_rad")
    if (
        isinstance(rest, bool)
        or not isinstance(rest, (int, float))
        or not math.isfinite(float(rest))
    ):
        raise ContractError("torsion-spring rest_position_rad must be finite")
    if envelope_field == "maximum_safe_deflection_rad":
        envelope_values = np.abs(
            np.asarray(trace.arrays["angle"], dtype=float) - float(rest)
        )
    elif envelope_field == "maximum_safe_load_n":
        envelope_values = np.abs(_spring_channel_on_angle_clock(trace, "load_force"))
    else:
        envelope_values = np.abs(
            _spring_channel_on_angle_clock(trace, "load_force")
            * _spring_channel_on_angle_clock(trace, "lever_arm")
        )
    if float(np.max(envelope_values)) > float(maximum) + 1.0e-12:
        raise ContractError(
            f"torsion-spring sample exceeds the approved {envelope_field} envelope"
        )
    _validate_spring_protocol_levels(
        trace,
        envelope_fractions=envelope_values / float(maximum),
    )
    return dict(approval)


def _spring_channel_on_angle_clock(
    trace: LoadedTrace, channel: str
) -> np.ndarray:
    arrays = trace.arrays
    time_bases = trace.manifest.time_bases
    source_clock = time_bases[channel]
    target_clock = time_bases["angle"]
    values = np.asarray(arrays[channel], dtype=float)
    if source_clock == target_clock:
        return values
    source_time = np.asarray(arrays[source_clock], dtype=float)
    target_time = np.asarray(arrays[target_clock], dtype=float)
    if target_time[0] < source_time[0] or target_time[-1] > source_time[-1]:
        raise ContractError(
            f"torsion-spring {channel} clock does not cover the angle clock"
        )
    return np.interp(target_time, source_time, values)


def _validate_spring_protocol_levels(
    trace: LoadedTrace,
    *,
    envelope_fractions: np.ndarray,
) -> None:
    protocol = _TORSION_SPRING_PROTOCOL_LEVELS.get(trace.manifest.scenario_id)
    if protocol is None:
        return
    expected_fractions, label = protocol
    branches = np.asarray(trace.arrays["sweep_branch"], dtype=float)
    directions = np.asarray(trace.arrays["torque_direction"], dtype=float)
    repeats = np.asarray(trace.arrays["repeat_index"], dtype=float)
    fractions = np.asarray(envelope_fractions, dtype=float)
    tolerance = 0.02

    for repeat in np.unique(repeats):
        for branch in (-1.0, 1.0):
            for direction in (-1.0, 1.0):
                selected = (
                    (repeats == repeat)
                    & (branches == branch)
                    & (directions == direction)
                )
                observed = fractions[selected]
                if observed.size == 0 or any(
                    not np.any(np.isclose(observed, expected, atol=tolerance, rtol=0.0))
                    for expected in expected_fractions
                ) or any(
                    not any(
                        abs(float(value) - expected) <= tolerance
                        for expected in expected_fractions
                    )
                    for value in observed
                ):
                    raise ContractError(
                        f"torsion-spring {trace.manifest.scenario_id} requires "
                        f"{label} approved-envelope levels in every signed sweep branch"
                    )
                increments = np.diff(observed)
                if (branch > 0.0 and np.any(increments < -tolerance)) or (
                    branch < 0.0 and np.any(increments > tolerance)
                ):
                    raise ContractError(
                        "torsion-spring sweep_branch must distinguish ordered loading "
                        "and unloading samples"
                    )


def evaluate_torsional_spring_quality(
    calibration_trace_path: str | Path,
    holdout_trace_path: str | Path,
) -> dict[str, Any]:
    """Evaluate the linear real-world model from two immutable managed episodes."""

    calibration_scenario, calibration_trace = _load_measurement_trace(
        calibration_trace_path
    )
    if calibration_scenario.scenario_id != "torsion-spring":
        raise ContractError("spring quality calibration must use torsion-spring")
    representative_spring_aliases(calibration_scenario, calibration_trace)

    discovered = load_trace(holdout_trace_path, require_managed_dataset=True)
    if discovered.manifest.scenario_id != "torsion-spring-holdout":
        raise ContractError(
            "spring quality holdout must use torsion-spring-holdout"
        )
    holdout_scenario = load_scenario("torsion-spring-holdout")
    units, frames = _expected_metadata(holdout_scenario)
    holdout_trace = load_trace(
        holdout_trace_path,
        scenario=holdout_scenario,
        require_managed_dataset=True,
        expected_metadata_sha256=discovered.metadata_sha256,
        expected_units=units,
        expected_frames=frames,
    )
    validate_real_trace_provenance(holdout_trace, holdout_scenario)
    calibration_approval = _spring_mechanical_approval(calibration_trace)
    holdout_approval = _spring_mechanical_approval(holdout_trace)
    if holdout_approval != calibration_approval:
        raise ContractError(
            "torsion-spring calibration and holdout must use the same approved fixture"
        )

    calibration = compute_subsystem_metrics(
        calibration_scenario, calibration_trace
    )
    # Compute the intrinsic holdout fit too, so malformed branch/repeat annotations
    # fail before the cross-model prediction is evaluated.
    compute_subsystem_metrics(holdout_scenario, holdout_trace)
    if not math.isclose(
        float(calibration["rest_position_rad"]),
        float(
            holdout_trace.manifest.metadata["calibration_constants"][
                "rest_position_rad"
            ]
        ),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ContractError(
            "torsion-spring calibration and holdout rest positions must match"
        )
    holdout = torsional_spring_holdout_trace_metrics(calibration, holdout_trace)
    return {
        "schema_version": 1,
        "calibration": calibration,
        "holdout": holdout,
        "quality": torsional_spring_quality_gates(calibration, holdout),
        "provenance": {
            "calibration_trace_sha256": calibration_trace.manifest.provenance[
                "trace_sha256"
            ],
            "calibration_metadata_sha256": calibration_trace.metadata_sha256,
            "holdout_trace_sha256": holdout_trace.manifest.provenance[
                "trace_sha256"
            ],
            "holdout_metadata_sha256": holdout_trace.metadata_sha256,
        },
    }


def _spring_quality_validation_record(
    report: Mapping[str, Any],
    calibration_trace: LoadedTrace,
    holdout_trace: LoadedTrace,
) -> dict[str, Any]:
    """Bind a quality decision to the exact immutable calibration/holdout pair."""

    calibration_dataset = calibration_trace.dataset
    holdout_dataset = holdout_trace.dataset
    if calibration_dataset is None or holdout_dataset is None:
        raise ContractError("spring quality evidence must use managed dataset episodes")
    provenance = report["provenance"]
    if (
        provenance["calibration_trace_sha256"]
        != calibration_trace.manifest.provenance["trace_sha256"]
        or provenance["calibration_metadata_sha256"]
        != calibration_trace.metadata_sha256
        or provenance["holdout_trace_sha256"]
        != holdout_trace.manifest.provenance["trace_sha256"]
        or provenance["holdout_metadata_sha256"] != holdout_trace.metadata_sha256
    ):
        raise ContractError("spring quality report provenance does not bind its traces")
    return {
        "accepted": bool(report["quality"]["accepted"]),
        "gates": dict(report["quality"]["gates"]),
        "calibration_trace_sha256": provenance["calibration_trace_sha256"],
        "calibration_metadata_sha256": provenance["calibration_metadata_sha256"],
        "holdout_trace_sha256": provenance["holdout_trace_sha256"],
        "holdout_metadata_sha256": provenance["holdout_metadata_sha256"],
        "holdout_scenario_id": holdout_trace.manifest.scenario_id,
        "holdout_scenario_sha256": holdout_trace.manifest.provenance[
            "scenario_sha256"
        ],
        "source": holdout_trace.manifest.source,
        "dataset_id": holdout_dataset.dataset_id,
        "episode_id": holdout_dataset.episode_id,
        "episode_path": str(holdout_trace.directory.resolve()),
    }


def verify_representative_spring_source(
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute a representative fit and holdout decision from its exact files."""

    calibration_path = source.get("episode_path")
    quality = source.get("quality_validation")
    if not isinstance(calibration_path, str) or not isinstance(quality, Mapping):
        raise ContractError(
            "representative torsion-spring source has no file-backed holdout evidence"
        )
    holdout_path = quality.get("episode_path")
    if not isinstance(holdout_path, str):
        raise ContractError(
            "representative torsion-spring holdout has no immutable episode path"
        )
    calibration_scenario, calibration_trace = _load_measurement_trace(
        calibration_path
    )
    holdout_scenario, holdout_trace = _load_measurement_trace(holdout_path)
    if calibration_scenario.scenario_id != "torsion-spring" or (
        holdout_scenario.scenario_id != "torsion-spring-holdout"
    ):
        raise ContractError("representative spring evidence uses the wrong scenarios")
    calibration_metrics = compute_subsystem_metrics(
        calibration_scenario, calibration_trace
    )
    aliases = representative_spring_aliases(
        calibration_scenario, calibration_trace
    )
    report = evaluate_torsional_spring_quality(calibration_path, holdout_path)
    expected = _source_record(
        calibration_scenario,
        calibration_trace,
        metric_kind="torsional_spring",
        frame=calibration_scenario.joint,
        repeat_count=int(calibration_metrics["repeat_count"]),
        applies_to=aliases,
        rest_position_rad=float(calibration_metrics["rest_position_rad"]),
    )
    expected["quality_validation"] = _spring_quality_validation_record(
        report, calibration_trace, holdout_trace
    )
    if dict(source) != expected:
        raise ContractError(
            "representative torsion-spring source does not match its immutable episodes"
        )
    return report


def _mass_reference_pose(trace: LoadedTrace) -> tuple[dict[str, Any], list[Any]]:
    constants = trace.manifest.metadata.get("calibration_constants", {})
    if not isinstance(constants, Mapping):  # pragma: no cover - trace contract guards this
        raise ContractError("mass-com calibration constants must be an object")
    joints = constants.get("reference_joint_position_rad")
    orientation = constants.get("reference_root_orientation_xyzw")
    if (
        not isinstance(joints, Mapping)
        or set(joints) != _CANONICAL_JOINTS
        or not isinstance(orientation, list)
        or len(orientation) != 4
    ):
        raise ContractError(
            "mass-com measurement must record its complete reference pose in "
            "metadata.calibration_constants"
        )
    return dict(joints), list(orientation)


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
    representative_calibration: tuple[str | Path, LoadedTrace] | None = None
    representative_holdout: tuple[str | Path, LoadedTrace] | None = None

    for path in trace_paths:
        scenario, trace = _load_measurement_trace(path)
        if scenario.scenario_id == "torsion-spring-holdout":
            if representative_holdout is not None:
                raise ContractError("duplicate torsion-spring holdout evidence")
            representative_holdout = (path, trace)
            continue
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
        elif scenario.experiment_kind == "mass_com":
            key = "mass_com"
            if key in applied_keys:
                raise ContractError("duplicate measurement source for mass_com")
            repeat_count = int(metrics["repeat_count"])
            if repeat_count != scenario.repeats:
                raise ContractError("mass/CoM repeat count does not match its scenario")
            mass = physics.setdefault("mass", {})
            legacy_fields = {"scale", "added_mass_kg", "com_offset_m"} & set(mass)
            if legacy_fields:
                raise ContractError(
                    "direct mass/CoM measurement cannot mix with legacy mass correction fields"
                )
            reference_joints, reference_orientation = _mass_reference_pose(trace)
            mass.update(
                {
                    "target_total_mass_kg": metrics["mass_kg"],
                    "reference_planar_com_xy_m": [
                        metrics["com_x_m"],
                        metrics["com_y_m"],
                    ],
                    "reference_joint_position_rad": reference_joints,
                    "reference_root_orientation_xyzw": reference_orientation,
                }
            )
            sources[key] = _source_record(
                scenario,
                trace,
                metric_kind="mass_com",
                frame="root",
                repeat_count=repeat_count,
            )
        elif scenario.experiment_kind == "spring":
            key = f"passive_spring:{scenario.joint}"
            if key in applied_keys:
                raise ContractError(f"duplicate measurement source for {key}")
            repeat_count = int(metrics["repeat_count"])
            if repeat_count != scenario.repeats:
                raise ContractError("spring repeat count does not match its scenario")
            applies_to = representative_spring_aliases(scenario, trace)
            if scenario.scenario_id == "torsion-spring":
                representative_calibration = (path, trace)
                failed = []
                if metrics["r_squared"] < 0.98:
                    failed.append("R²")
                if metrics["stiffness_cv"] > 0.05:
                    failed.append("stiffness CV")
                if metrics["hysteresis_full_scale_ratio"] > 0.10:
                    failed.append("hysteresis")
                if failed:
                    raise ContractError(
                        "representative torsion-spring calibration requires a nonlinear "
                        "or hysteretic model; failed: " + ", ".join(failed)
                    )
            springs = physics.setdefault("passive_spring", {})
            fitted_stiffness = metrics[
                "neutral_stiffness_nm_per_rad"
                if scenario.scenario_id == "torsion-spring"
                else "stiffness_nm_per_rad"
            ]
            for joint_alias in applies_to:
                spring = springs.setdefault(joint_alias, {})
                spring["stiffness"] = fitted_stiffness
                if scenario.scenario_id == "torsion-spring":
                    # Static testing does not identify dynamic damping. Keep the
                    # reviewed physical assumption explicit and prevent legacy
                    # high-gain damper values from leaking into this profile.
                    spring["damping"] = 0.0
                    if joint_alias == TORSION_SPRING_ALIASES[0]:
                        spring["rest_position_rad"] = float(
                            metrics["rest_position_rad"]
                        )
            sources[key] = _source_record(
                scenario,
                trace,
                metric_kind="torsional_spring",
                frame=scenario.joint,
                repeat_count=repeat_count,
                applies_to=(
                    applies_to if scenario.scenario_id == "torsion-spring" else None
                ),
                rest_position_rad=(
                    metrics["rest_position_rad"]
                    if scenario.scenario_id == "torsion-spring"
                    else None
                ),
            )
        elif scenario.experiment_kind == "manual_load":
            key = f"main_drive_effort_limit:{scenario.joint}"
            if key in applied_keys:
                raise ContractError(f"duplicate measurement source for {key}")
            repeat_count = int(metrics["repeat_count"])
            if repeat_count != scenario.repeats:
                raise ContractError("known-load repeat count does not match its scenario")
            physics.setdefault("main_drive", {})["effort_limit"] = metrics[
                "torque_saturation_nm"
            ]
            sources[key] = _source_record(
                scenario,
                trace,
                metric_kind="torque_saturation",
                frame=scenario.joint,
                repeat_count=repeat_count,
            )
        else:  # pragma: no cover - guarded by _load_measurement_trace
            raise ContractError(f"unsupported measurement scenario: {scenario.scenario_id}")
        applied_keys.add(key)

    if representative_holdout is not None and representative_calibration is None:
        raise ContractError(
            "torsion-spring holdout requires its calibration trace in the same profile fit"
        )
    if representative_calibration is not None and representative_holdout is not None:
        calibration_path, calibration_trace = representative_calibration
        holdout_path, holdout_trace = representative_holdout
        report = evaluate_torsional_spring_quality(calibration_path, holdout_path)
        if not report["quality"]["accepted"]:
            failed = [
                name
                for name, passed in report["quality"]["gates"].items()
                if not passed
            ]
            raise ContractError(
                "representative torsion-spring quality gates failed: "
                + ", ".join(failed)
            )
        sources["passive_spring:damper_0"]["quality_validation"] = (
            _spring_quality_validation_record(
                report, calibration_trace, holdout_trace
            )
        )

    return CalibrationProfileV1.from_dict(payload)
