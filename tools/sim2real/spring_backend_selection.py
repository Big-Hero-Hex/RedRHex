from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import ContractError
from .traces import load_trace, sha256_file, sha256_json


BACKENDS = ("explicit", "native")
PHYSICS_HZ = (120, 240)
STATIC_TORQUE_RMSE_LIMIT = 0.01
ENERGY_CREATION_LIMIT = 0.02
ENERGY_WORK_RESIDUAL_LIMIT = 0.02
PEAK_TIMESTEP_DIFFERENCE_LIMIT = 0.02
RUNAWAY_AMPLITUDE_LIMIT = 1.05
EXPLICIT_TIE_BREAK_FRACTION = 0.10
FIXTURE_POSITION_ERROR_LIMIT_RAD = 1.0e-4
FIXTURE_VELOCITY_LIMIT_RAD_S = 1.0e-4
FIXTURE_SETTLED_FRACTION = 0.80

_REQUIRED_CHANNELS = (
    "spring_deflection",
    "spring_model_torque",
    "spring_applied_torque_estimate",
    "spring_potential_energy",
    "spring_mechanical_power",
    "spring_passivity_residual",
    "spring_release_start",
    "spring_fixture_position_error",
    "spring_fixture_velocity",
    "spring_unwrap_ambiguous",
)


def _matrix(
    arrays: Mapping[str, np.ndarray], name: str, *, samples: int | None = None
) -> np.ndarray:
    if name not in arrays:
        raise ContractError(f"spring release trace is missing {name}")
    value = np.asarray(arrays[name], dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 6:
        raise ContractError(f"{name} must have shape (sample, 6)")
    if samples is not None and value.shape[0] != samples:
        raise ContractError(f"{name} sample count does not match spring_deflection")
    if not np.isfinite(value).all():
        raise ContractError(f"{name} contains a non-finite value")
    return value


def _vector(
    arrays: Mapping[str, np.ndarray], name: str, *, samples: int
) -> np.ndarray:
    if name not in arrays:
        raise ContractError(f"spring release trace is missing {name}")
    value = np.asarray(arrays[name], dtype=np.float64)
    if value.ndim == 2 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != 1 or value.shape[0] != samples:
        raise ContractError(f"{name} must have shape (sample,) or (sample, 1)")
    if not np.isfinite(value).all():
        raise ContractError(f"{name} contains a non-finite value")
    return value


def _first_rebound_peak_with_status(deflection: np.ndarray) -> tuple[float, bool]:
    """Estimate the first completed opposite-sign rebound extremum."""

    values = np.asarray(deflection, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ContractError("spring rebound deflection must be a finite vector")
    initial = float(values[0])
    if abs(initial) <= 1.0e-12:
        raise ContractError("spring rebound deflection must begin away from neutral")
    opposite = values * initial < 0.0
    oriented = -math.copysign(1.0, initial) * values
    peak_index: int | None = None
    for index in np.flatnonzero(opposite):
        index = int(index)
        if index <= 0 or index >= values.size - 1:
            continue
        center = float(oriented[index])
        before = float(oriented[index - 1])
        after = float(oriented[index + 1])
        if center >= before and center >= after and (
            center > before or center > after
        ):
            peak_index = index
            break
    if peak_index is None:
        return 0.0, False
    peak = float(oriented[peak_index])
    y0, y1, y2 = oriented[peak_index - 1 : peak_index + 2]
    denominator = float(y0 - 2.0 * y1 + y2)
    if abs(denominator) > 1.0e-15:
        offset = float(0.5 * (y0 - y2) / denominator)
        if abs(offset) <= 1.0:
            vertex = float(y1 - 0.25 * (y0 - y2) * offset)
            peak = max(peak, vertex)
    return peak, True


def _first_rebound_peak(deflection: np.ndarray) -> float:
    """Return the interpolated peak, or zero when no completed rebound exists."""

    peak, _ = _first_rebound_peak_with_status(deflection)
    return peak


def evaluate_release_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    backend: str,
    physics_dt_s: float,
    calibration_status: str,
    profile_sha256: str | None,
) -> dict[str, Any]:
    """Evaluate one deterministic signed release against the physics gates."""

    if backend not in BACKENDS:
        raise ContractError(f"unsupported spring backend: {backend}")
    if calibration_status not in {"calibrated", "uncalibrated"}:
        raise ContractError("spring calibration status is invalid")
    if not math.isfinite(float(physics_dt_s)) or float(physics_dt_s) <= 0.0:
        raise ContractError("spring release physics dt must be positive and finite")
    physics_hz = int(round(1.0 / float(physics_dt_s)))
    if physics_hz not in PHYSICS_HZ or not math.isclose(
        float(physics_dt_s), 1.0 / physics_hz, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ContractError("spring release must use exactly 120 or 240 Hz")

    deflection = _matrix(arrays, "spring_deflection")
    samples = deflection.shape[0]
    model_torque = _matrix(arrays, "spring_model_torque", samples=samples)
    applied_torque = _matrix(
        arrays, "spring_applied_torque_estimate", samples=samples
    )
    energy = _matrix(arrays, "spring_potential_energy", samples=samples)
    mechanical_power = _matrix(
        arrays, "spring_mechanical_power", samples=samples
    )
    passivity_residual = _vector(
        arrays, "spring_passivity_residual", samples=samples
    )
    release_start_raw = _vector(arrays, "spring_release_start", samples=samples)
    fixture_position_error = _vector(
        arrays, "spring_fixture_position_error", samples=samples
    )
    fixture_velocity = _vector(arrays, "spring_fixture_velocity", samples=samples)
    unwrap_ambiguous = _vector(
        arrays, "spring_unwrap_ambiguous", samples=samples
    )
    if np.any(fixture_position_error < 0.0) or np.any(fixture_velocity < 0.0):
        raise ContractError("spring fixture error channels must be non-negative")
    if not np.isin(release_start_raw, (0.0, 1.0)).all():
        raise ContractError("spring_release_start must contain only zero or one")
    if not np.isin(unwrap_ambiguous, (0.0, 1.0)).all():
        raise ContractError("spring_unwrap_ambiguous must contain only zero or one")
    release_starts = np.flatnonzero(release_start_raw == 1.0)
    if release_starts.size < 2 or release_starts[0] != 0:
        raise ContractError("spring release trace must begin with at least two releases")

    tested_at_start = np.abs(deflection[release_starts]) > 1.0e-8
    if not np.all(np.sum(tested_at_start, axis=1) == 1):
        raise ContractError("every spring release must test exactly one spring joint")
    tested = np.zeros_like(deflection, dtype=bool)
    tested[release_starts] = tested_at_start
    torque_error = applied_torque[tested] - model_torque[tested]
    full_scale_torque = float(np.max(np.abs(model_torque[tested])))
    if full_scale_torque <= 0.0:
        raise ContractError("spring release fitted law has zero full-scale torque")
    torque_rmse = float(np.sqrt(np.mean(np.square(torque_error))))
    torque_rmse_fraction = torque_rmse / full_scale_torque
    restoring_sign_correct = bool(
        np.all(model_torque[tested] * deflection[tested] < 0.0)
        and np.all(applied_torque[tested] * deflection[tested] < 0.0)
    )

    release_peaks: list[float] = []
    completed_rebounds: list[bool] = []
    amplitude_ratios: list[float] = []
    energy_creation_ratios: list[float] = []
    energy_drift_ratios: list[float] = []
    spring_work_residual_ratios: list[float] = []
    fixture_settled_position_errors: list[float] = []
    for release_index, start in enumerate(release_starts):
        stop = (
            int(release_starts[release_index + 1])
            if release_index + 1 < release_starts.size
            else samples
        )
        tested_joint = int(np.flatnonzero(tested[start])[0])
        initial_amplitude = float(abs(deflection[start, tested_joint]))
        initial_energy = float(np.sum(energy[start]))
        if initial_amplitude <= 0.0 or initial_energy <= 0.0:
            raise ContractError("every spring release must begin with positive energy")
        release_deflection = deflection[start:stop]
        peak = float(np.max(np.abs(release_deflection)))
        dynamic_deflection = release_deflection[1:]
        if dynamic_deflection.shape[0] < 1:
            raise ContractError("every spring release must include a post-release sample")
        rebound_peak, rebound_completed = _first_rebound_peak_with_status(
            release_deflection[:, tested_joint]
        )
        release_peaks.append(rebound_peak)
        completed_rebounds.append(rebound_completed)
        amplitude_ratios.append(peak / initial_amplitude)
        signed_energy_drift = passivity_residual[start:stop]
        energy_creation_ratios.append(
            float(np.max(np.maximum(signed_energy_drift, 0.0))) / initial_energy
        )
        energy_drift_ratios.append(
            float(np.max(np.abs(signed_energy_drift))) / initial_energy
        )

        release_power = np.sum(mechanical_power[start:stop], axis=1)
        cumulative_work = np.zeros_like(release_power)
        cumulative_work[1:] = np.cumsum(
            0.5 * (release_power[:-1] + release_power[1:]) * float(physics_dt_s)
        )
        release_energy = np.sum(energy[start:stop], axis=1)
        work_residual = release_energy - release_energy[0] + cumulative_work
        spring_work_residual_ratios.append(
            float(np.max(np.abs(work_residual))) / initial_energy
        )

        # Fixed-joint projection can correct generalized coordinates on the
        # first release frames. Velocity is checked throughout; position is
        # checked over the settled final 20% of every release segment.
        settled_start = start + max(
            1, int(math.floor((stop - start) * FIXTURE_SETTLED_FRACTION))
        )
        if settled_start >= stop:
            settled_start = stop - 1
        fixture_settled_position_errors.append(
            float(np.max(fixture_position_error[settled_start:stop]))
        )

    energy_creation_fraction = max(energy_creation_ratios)
    energy_drift_fraction = max(energy_drift_ratios)
    spring_work_residual_fraction = max(spring_work_residual_ratios)
    energy_work_residual_fraction = max(
        energy_drift_fraction, spring_work_residual_fraction
    )
    runaway_ratio = max(amplitude_ratios)
    fixture_position_error_max = max(fixture_settled_position_errors)
    fixture_velocity_max = float(np.max(fixture_velocity))
    unwrap_ambiguity_count = int(np.count_nonzero(unwrap_ambiguous))
    gates = {
        "static_torque_rmse": torque_rmse_fraction <= STATIC_TORQUE_RMSE_LIMIT,
        "restoring_sign": restoring_sign_correct,
        "finite": True,
        "runaway": runaway_ratio <= RUNAWAY_AMPLITUDE_LIMIT,
        "energy_creation": energy_creation_fraction <= ENERGY_CREATION_LIMIT,
        "energy_work_residual": (
            energy_work_residual_fraction <= ENERGY_WORK_RESIDUAL_LIMIT
        ),
        "fixture_position": (
            fixture_position_error_max <= FIXTURE_POSITION_ERROR_LIMIT_RAD
        ),
        "fixture_velocity": fixture_velocity_max <= FIXTURE_VELOCITY_LIMIT_RAD_S,
        "unwrap_unambiguous": unwrap_ambiguity_count == 0,
        "completed_rebound": all(completed_rebounds),
    }
    return {
        "backend": backend,
        "physics_hz": physics_hz,
        "physics_dt_s": float(physics_dt_s),
        "calibration_status": calibration_status,
        "profile_sha256": profile_sha256,
        "static_torque_rmse_nm": torque_rmse,
        "static_torque_full_scale_nm": full_scale_torque,
        "static_torque_rmse_fraction": torque_rmse_fraction,
        "restoring_sign_correct": restoring_sign_correct,
        "release_peak_angles_rad": release_peaks,
        "maximum_amplitude_ratio": runaway_ratio,
        "energy_creation_fraction": energy_creation_fraction,
        "energy_drift_fraction": energy_drift_fraction,
        "spring_work_residual_fraction": spring_work_residual_fraction,
        "energy_work_residual_fraction": energy_work_residual_fraction,
        "fixture_settled_position_error_max_rad": fixture_position_error_max,
        "fixture_velocity_max_rad_s": fixture_velocity_max,
        "unwrap_ambiguity_count": unwrap_ambiguity_count,
        "completed_rebound_count": int(sum(completed_rebounds)),
        "release_count": int(len(completed_rebounds)),
        "gates": gates,
        "passed": all(gates.values()),
    }


def load_release_report(value: str | Path) -> dict[str, Any]:
    """Load one hash-verified simulator run and evaluate its spring channels."""

    directory = Path(value)
    loaded = load_trace(directory, scenario="spring-release")
    if loaded.manifest.source != "sim":
        raise ContractError("spring backend selection requires simulator traces")
    results_path = directory / "results.json"
    try:
        results = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"spring release results are invalid: {exc}") from exc
    if not isinstance(results, Mapping):
        raise ContractError("spring release results must be an object")
    metadata = loaded.manifest.metadata
    for field in ("spring_backend", "calibration_status", "profile_sha256", "seed"):
        if results.get(field) != metadata.get(field):
            raise ContractError(f"spring release {field} metadata mismatch")
    constants = metadata.get("calibration_constants")
    if not isinstance(constants, Mapping):
        raise ContractError("spring release metadata omits calibration constants")
    physics_dt = results.get("physics_dt_s")
    if isinstance(physics_dt, bool) or not isinstance(physics_dt, (int, float)):
        raise ContractError("spring release results omit physics_dt_s")
    if constants.get("physics_dt_s") != physics_dt:
        raise ContractError("spring release physics dt metadata mismatch")
    seed = results.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ContractError("spring release seed must be a non-negative integer")
    if constants.get("seed") != seed:
        raise ContractError("spring release seed calibration metadata mismatch")

    audit_name = results.get("runtime_audit")
    if audit_name != "runtime_audit.json":
        raise ContractError("spring release runtime audit path is invalid")
    audit_path = directory / audit_name
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"spring release runtime audit is invalid: {exc}") from exc
    if not isinstance(audit, Mapping):
        raise ContractError("spring release runtime audit must be an object")
    audit_sha256 = sha256_json(audit)
    if (
        results.get("runtime_audit_sha256") != audit_sha256
        or constants.get("runtime_audit_sha256") != audit_sha256
    ):
        raise ContractError("spring release runtime audit hash mismatch")
    if audit.get("physics_dt_s") != physics_dt:
        raise ContractError("spring release runtime audit physics dt mismatch")
    if audit.get("mode") != "fixed-base" or audit.get("is_fixed_base") is not True:
        raise ContractError("spring release runtime audit is not fixed-base")

    parameters = audit.get("torsion_springs")
    if not isinstance(parameters, Mapping):
        raise ContractError("spring release runtime audit omits torsion springs")
    result_parameters = results.get("effective_spring_parameters")
    if not isinstance(result_parameters, Mapping) or sha256_json(
        result_parameters
    ) != sha256_json(parameters):
        raise ContractError("spring release effective spring parameters mismatch")
    runtime_joint_names = parameters.get("runtime_joint_names")
    locked_joint_names = parameters.get("locked_joint_names")
    joint_names = audit.get("joint_names")
    if not all(
        isinstance(names, list) and all(isinstance(name, str) for name in names)
        for names in (runtime_joint_names, locked_joint_names, joint_names)
    ):
        raise ContractError("spring release fixture joint names are invalid")
    if len(runtime_joint_names) != 6 or len(set(runtime_joint_names)) != 6:
        raise ContractError("spring release runtime spring order is invalid")
    expected_locked = set(joint_names) - {runtime_joint_names[0]}
    if len(joint_names) != 18 or set(locked_joint_names) != expected_locked:
        raise ContractError("spring release fixture did not lock every non-tested joint")
    fixture_half_width = parameters.get("fixture_limit_half_width_rad")
    if (
        isinstance(fixture_half_width, bool)
        or not isinstance(fixture_half_width, (int, float))
        or not 0.0 < float(fixture_half_width) <= 1.0e-5
    ):
        raise ContractError("spring release fixture limit is invalid")
    selected_limit = parameters.get("fixture_selected_limit_rad")
    if (
        not isinstance(selected_limit, list)
        or len(selected_limit) != 2
        or selected_limit != [None, None]
        or parameters.get("fixture_selected_range_kind") != "continuous"
    ):
        raise ContractError("spring release selected joint is not continuous")
    if parameters.get("fixture_constraint_kind") != "excluded_fixed_joint":
        raise ContractError("spring release fixed-joint fixture is invalid")

    report = evaluate_release_arrays(
        loaded.arrays,
        backend=str(results["spring_backend"]),
        physics_dt_s=float(physics_dt),
        calibration_status=str(results["calibration_status"]),
        profile_sha256=results.get("profile_sha256"),
    )
    comparable_parameters = copy.deepcopy(dict(parameters))
    comparable_parameters.pop("backend", None)
    runtime_identity_fields = (
        "git_sha",
        "asset_sha256",
        "config_sha256",
        "redrhex_module_sha256",
        "isaaclab_version",
        "isaacsim_version",
        "characterization_runner_sha256",
        "torsion_spring_model_sha256",
        "runtime_bundle_sha256",
    )
    runtime_identity = {field: metadata.get(field) for field in runtime_identity_fields}
    if any(value is None for value in runtime_identity.values()):
        raise ContractError("spring release runtime identity is incomplete")
    report.update(
        {
            "directory": str(directory.resolve()),
            "trace_sha256": loaded.manifest.provenance["trace_sha256"],
            "metadata_sha256": loaded.metadata_sha256,
            "results_sha256": sha256_file(results_path),
            "runtime_audit_sha256": audit_sha256,
            "effective_parameter_sha256": sha256_json(comparable_parameters),
            "runtime_identity_sha256": sha256_json(runtime_identity),
            "seed": seed,
        }
    )
    return report


def select_spring_backend(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the 120/240 Hz gates and deterministic production tie-break."""

    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for raw in reports:
        report = dict(raw)
        key = (str(report.get("backend")), int(report.get("physics_hz", 0)))
        if key in indexed:
            raise ContractError(f"duplicate spring backend report: {key}")
        indexed[key] = report
    expected = {(backend, hz) for backend in BACKENDS for hz in PHYSICS_HZ}
    if set(indexed) != expected:
        raise ContractError(
            "spring backend selection requires exactly one explicit/native report "
            "at both 120 and 240 Hz"
        )

    parameter_hashes = [
        report.get("effective_parameter_sha256") for report in indexed.values()
    ]
    parameters_match = all(isinstance(value, str) for value in parameter_hashes) and len(
        set(parameter_hashes)
    ) == 1
    seeds = [report.get("seed") for report in indexed.values()]
    seeds_match = all(
        isinstance(value, int) and not isinstance(value, bool) for value in seeds
    ) and len(set(seeds)) == 1
    runtime_identities = [
        report.get("runtime_identity_sha256") for report in indexed.values()
    ]
    runtime_identity_matches = all(
        isinstance(value, str) for value in runtime_identities
    ) and len(set(runtime_identities)) == 1
    profile_hashes = {report.get("profile_sha256") for report in indexed.values()}
    profiles_match = len(profile_hashes) == 1
    calibrated = all(
        report.get("calibration_status") == "calibrated"
        and isinstance(report.get("profile_sha256"), str)
        for report in indexed.values()
    )

    backend_summaries: dict[str, Any] = {}
    for backend in BACKENDS:
        low = indexed[(backend, 120)]
        high = indexed[(backend, 240)]
        low_peaks = np.asarray(low["release_peak_angles_rad"], dtype=np.float64)
        high_peaks = np.asarray(high["release_peak_angles_rad"], dtype=np.float64)
        if low_peaks.shape != high_peaks.shape or low_peaks.size < 1:
            raise ContractError("120/240 Hz release peak layouts do not match")
        denominator = np.maximum(np.maximum(np.abs(low_peaks), np.abs(high_peaks)), 1.0e-12)
        peak_difference = float(np.max(np.abs(low_peaks - high_peaks) / denominator))
        timestep_gate = peak_difference <= PEAK_TIMESTEP_DIFFERENCE_LIMIT
        passed = bool(low.get("passed")) and bool(high.get("passed")) and timestep_gate
        backend_summaries[backend] = {
            "passed": passed,
            "energy_creation_fraction": max(
                float(low["energy_creation_fraction"]),
                float(high["energy_creation_fraction"]),
            ),
            "energy_work_residual_fraction": max(
                float(low["energy_work_residual_fraction"]),
                float(high["energy_work_residual_fraction"]),
            ),
            "peak_angle_difference_fraction": peak_difference,
            "timestep_gate": timestep_gate,
            "runs": {"120": low, "240": high},
        }

    physics_passed = (
        parameters_match
        and profiles_match
        and seeds_match
        and runtime_identity_matches
        and all(summary["passed"] for summary in backend_summaries.values())
    )
    selected_backend: str | None = None
    if not calibrated:
        status = "blocked_uncalibrated"
    elif not physics_passed:
        status = "blocked_physics_validation"
    else:
        explicit_residual = backend_summaries["explicit"][
            "energy_work_residual_fraction"
        ]
        native_residual = backend_summaries["native"][
            "energy_work_residual_fraction"
        ]
        within_ten_percent = math.isclose(
            explicit_residual,
            native_residual,
            rel_tol=EXPLICIT_TIE_BREAK_FRACTION,
            abs_tol=0.0,
        )
        selected_backend = (
            "explicit"
            if within_ten_percent or explicit_residual < native_residual
            else "native"
        )
        status = "selected"

    return {
        "schema_version": 1,
        "status": status,
        "eligible": status == "selected",
        "selected_backend": selected_backend,
        "calibrated": calibrated,
        "profiles_match": profiles_match,
        "effective_parameters_match": parameters_match,
        "seeds_match": seeds_match,
        "seed": seeds[0] if seeds_match else None,
        "runtime_identity_matches": runtime_identity_matches,
        "physics_passed": physics_passed,
        "thresholds": {
            "static_torque_rmse_fraction": STATIC_TORQUE_RMSE_LIMIT,
            "energy_creation_fraction": ENERGY_CREATION_LIMIT,
            "energy_work_residual_fraction": ENERGY_WORK_RESIDUAL_LIMIT,
            "peak_timestep_difference_fraction": PEAK_TIMESTEP_DIFFERENCE_LIMIT,
            "explicit_tie_break_fraction": EXPLICIT_TIE_BREAK_FRACTION,
            "fixture_position_error_rad": FIXTURE_POSITION_ERROR_LIMIT_RAD,
            "fixture_velocity_rad_s": FIXTURE_VELOCITY_LIMIT_RAD_S,
            "fixture_settled_fraction": FIXTURE_SETTLED_FRACTION,
        },
        "backends": backend_summaries,
    }


def evaluate_backend_runs(
    *,
    explicit_120: str | Path,
    explicit_240: str | Path,
    native_120: str | Path,
    native_240: str | Path,
) -> dict[str, Any]:
    return select_spring_backend(
        [
            load_release_report(explicit_120),
            load_release_report(explicit_240),
            load_release_report(native_120),
            load_release_report(native_240),
        ]
    )
