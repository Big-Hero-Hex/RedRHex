from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .contracts import ContractError, ScenarioSpecV1
from .traces import LoadedTrace


def _series(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size < 1 or not np.isfinite(array).all():
        raise ContractError(f"{name} must be a finite one-dimensional series")
    return array


def _time(value: Any, name: str) -> np.ndarray:
    result = _series(value, name)
    if result.size < 2 or not np.all(np.diff(result) > 0.0):
        raise ContractError(f"{name} must be strictly increasing with at least two samples")
    return result


def position_derived_velocity(
    time_s: Any,
    position: Any,
    *,
    smoothing_window: int = 1,
) -> np.ndarray:
    """Derive offline velocity using central differences on the recorded time base."""

    time = _time(time_s, "time_s")
    values = np.asarray(position, dtype=float)
    if values.ndim < 1 or values.shape[0] != time.size or not np.isfinite(values).all():
        raise ContractError("position shape must match time_s and contain finite values")
    edge_order = 2 if time.size >= 3 else 1
    velocity = np.gradient(values, time, axis=0, edge_order=edge_order)
    if isinstance(smoothing_window, bool) or not isinstance(smoothing_window, int) or smoothing_window < 1:
        raise ContractError("smoothing_window must be a positive integer")
    if smoothing_window > 1:
        kernel = np.ones(smoothing_window, dtype=float) / smoothing_window
        if velocity.ndim == 1:
            velocity = np.convolve(velocity, kernel, mode="same")
        else:
            velocity = np.apply_along_axis(
                lambda column: np.convolve(column, kernel, mode="same"), 0, velocity
            )
    return velocity


def _command_transitions(command: np.ndarray) -> np.ndarray:
    tolerance = max(float(np.max(np.abs(command))) * 1e-9, 1e-12)
    return np.flatnonzero(np.abs(np.diff(command)) > tolerance) + 1


def step_response_metrics(
    command_time_s: Any,
    command: Any,
    position_time_s: Any,
    position: Any,
) -> dict[str, float]:
    command_time = _time(command_time_s, "command_time_s")
    commands = _series(command, "command")
    position_time = _time(position_time_s, "position_time_s")
    positions = _series(position, "position")
    if commands.size != command_time.size or positions.size != position_time.size:
        raise ContractError("command and position shapes must match their time bases")
    transitions = _command_transitions(commands)
    starts = [index for index in transitions if abs(commands[index]) > abs(commands[index - 1])]
    if not starts:
        raise ContractError("step response requires a command step away from neutral")
    start_index = starts[0]
    start_time = float(command_time[start_index])
    later = transitions[transitions > start_index]
    end_time = float(command_time[later[0]]) if later.size else float(position_time[-1])
    active = (position_time >= start_time) & (position_time < end_time)
    if np.count_nonzero(active) < 4:
        raise ContractError("step response interval has too few position samples")
    velocity = position_derived_velocity(position_time, positions)
    direction = 1.0 if commands[start_index] - commands[start_index - 1] > 0.0 else -1.0
    signed_velocity = direction * velocity
    active_indices = np.flatnonzero(active)
    steady_start = active_indices[int(0.75 * (active_indices.size - 1))]
    steady = float(np.median(signed_velocity[steady_start : active_indices[-1] + 1]))
    if steady <= 0.0:
        raise ContractError("step response has no positive steady speed in the command direction")
    active_velocity = signed_velocity[active_indices]
    active_time = position_time[active_indices]
    onset_candidates = np.flatnonzero(active_velocity >= 0.05 * steady)
    low_candidates = np.flatnonzero(active_velocity >= 0.10 * steady)
    high_candidates = np.flatnonzero(active_velocity >= 0.90 * steady)
    if not onset_candidates.size or not low_candidates.size or not high_candidates.size:
        raise ContractError("step response does not cross onset/rise thresholds")
    onset_time = float(active_time[onset_candidates[0]])
    low_time = float(active_time[low_candidates[0]])
    high_after_low = high_candidates[high_candidates >= low_candidates[0]]
    if not high_after_low.size:
        raise ContractError("step response does not reach its 90% rise threshold")
    high_time = float(active_time[high_after_low[0]])
    overshoot = max(0.0, float(np.max(active_velocity) / steady - 1.0))
    return {
        "onset_delay_s": max(0.0, onset_time - start_time),
        "steady_speed_rad_s": steady,
        "rise_time_s": max(0.0, high_time - low_time),
        "overshoot_ratio": overshoot,
    }


def coast_response_metrics(
    command_time_s: Any,
    command: Any,
    position_time_s: Any,
    position: Any,
) -> dict[str, float]:
    command_time = _time(command_time_s, "command_time_s")
    commands = _series(command, "command")
    position_time = _time(position_time_s, "position_time_s")
    positions = _series(position, "position")
    if commands.size != command_time.size or positions.size != position_time.size:
        raise ContractError("command and position shapes must match their time bases")
    transitions = _command_transitions(commands)
    stops = [index for index in transitions if abs(commands[index]) < abs(commands[index - 1])]
    if not stops:
        raise ContractError("coast response requires a transition toward neutral")
    stop_time = float(command_time[stops[0]])
    velocity = np.abs(position_derived_velocity(position_time, positions))
    before = np.flatnonzero(position_time < stop_time)
    after = np.flatnonzero(position_time >= stop_time)
    if before.size < 2 or after.size < 2:
        raise ContractError("coast response has too few samples")
    steady_samples = before[max(0, int(before.size * 0.75)) :]
    steady_speed = float(np.median(velocity[steady_samples]))
    threshold = max(steady_speed * 0.10, 1e-9)
    stopped = after[velocity[after] <= threshold]
    coast_time = float(position_time[stopped[0]] - stop_time) if stopped.size else float("nan")
    if not np.isfinite(coast_time):
        raise ContractError("velocity did not settle before the trace ended")
    return {"coast_time_s": max(0.0, coast_time), "pre_coast_speed_rad_s": steady_speed}


def _directional_metrics(
    command_time_s: Any,
    command: Any,
    position_time_s: Any,
    position: Any,
    *,
    event: str,
) -> dict[str, dict[str, float]]:
    command_time = _time(command_time_s, "command_time_s")
    commands = _series(command, "command")
    position_time = _time(position_time_s, "position_time_s")
    positions = _series(position, "position")
    transitions = _command_transitions(commands)
    if event == "step":
        selected = [index for index in transitions if abs(commands[index]) > abs(commands[index - 1])]
        calculator = step_response_metrics
    else:
        selected = [index for index in transitions if abs(commands[index]) < abs(commands[index - 1])]
        calculator = coast_response_metrics
    grouped: dict[str, list[dict[str, float]]] = {"positive": [], "negative": []}
    for index in selected:
        if event == "step":
            direction_value = commands[index]
            left = max(0, index - 1)
        else:
            direction_value = commands[index - 1]
            earlier = transitions[transitions < index]
            left = max(0, int(earlier[-1]) - 1) if earlier.size else 0
        later = transitions[transitions > index]
        right = int(later[0]) if later.size else commands.size - 1
        command_slice = slice(left, min(commands.size, right + 1))
        start_time = command_time[left]
        end_time = command_time[right] if later.size else position_time[-1]
        position_mask = (position_time >= start_time) & (position_time <= end_time)
        if np.count_nonzero(position_mask) < 4:
            continue
        metrics = calculator(
            command_time[command_slice],
            commands[command_slice],
            position_time[position_mask],
            positions[position_mask],
        )
        grouped["positive" if direction_value > 0.0 else "negative"].append(metrics)
    result: dict[str, dict[str, float]] = {}
    for direction, repetitions in grouped.items():
        if not repetitions:
            continue
        if len(repetitions) == 1:
            result[direction] = repetitions[0]
            continue
        summary: dict[str, float] = {"repeat_count": float(len(repetitions))}
        for key in repetitions[0]:
            values = np.asarray([item[key] for item in repetitions], dtype=float)
            summary[key] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
        result[direction] = summary
    if not result:
        raise ContractError(f"no valid {event} response segments found")
    return result


def bidirectional_step_metrics(
    command_time_s: Any,
    command: Any,
    position_time_s: Any,
    position: Any,
) -> dict[str, dict[str, float]]:
    return _directional_metrics(
        command_time_s, command, position_time_s, position, event="step"
    )


def bidirectional_coast_metrics(
    command_time_s: Any,
    command: Any,
    position_time_s: Any,
    position: Any,
) -> dict[str, dict[str, float]]:
    return _directional_metrics(
        command_time_s, command, position_time_s, position, event="coast"
    )


def stiffness_metrics(force: Any, displacement: Any) -> dict[str, float]:
    forces = _series(force, "force")
    displacements = _series(displacement, "displacement")
    if forces.size != displacements.size or forces.size < 2:
        raise ContractError("force and displacement must have matching multi-sample shapes")
    if float(np.ptp(displacements)) <= 0.0:
        raise ContractError("displacement must vary to estimate stiffness")
    slope, intercept = np.polyfit(displacements, forces, 1)
    return {"stiffness_n_per_m": float(slope), "force_intercept_n": float(intercept)}


def torsional_spring_metrics(
    *,
    angle_rad: Any,
    load_force: Any,
    lever_arm_m: Any,
    torque_direction: Any | None = None,
    sweep_branch: Any | None = None,
    repeat_index: Any | None = None,
    expected_repeats: int | None = None,
    rest_position_rad: float | None = None,
) -> dict[str, Any]:
    angle = _series(angle_rad, "angle_rad")
    force = _series(load_force, "load_force")
    arm = _series(lever_arm_m, "lever_arm_m")
    if len({angle.size, force.size, arm.size}) != 1 or angle.size < 2:
        raise ContractError("spring angle, load force, and lever arm must have matching shapes")
    if np.any(arm < 0.0) or float(np.ptp(angle)) <= 0.0:
        raise ContractError("spring lever arm must be non-negative and angle must vary")
    if torque_direction is None:
        torque = force * arm
        directions = None
    else:
        if np.any(force < 0.0):
            raise ContractError("spring load_force must be non-negative when torque_direction is recorded")
        directions = _signed_unit_series(torque_direction, "torque_direction")
        if directions.size != angle.size:
            raise ContractError("spring torque_direction must align with angle samples")
        torque = force * arm * directions
    branches = None
    if sweep_branch is not None:
        branches = _signed_unit_series(sweep_branch, "sweep_branch")
        if branches.size != angle.size:
            raise ContractError("spring sweep_branch must align with angle samples")
    slope, intercept = np.polyfit(angle, torque, 1)
    if directions is not None and slope <= 0.0:
        raise ContractError(
            "spring signed measurements must identify a positive restoring stiffness"
        )
    result: dict[str, Any] = {
        "stiffness_nm_per_rad": float(slope),
        "torque_intercept_nm": float(intercept),
    }
    enhanced_metrics = (
        directions is not None
        or branches is not None
        or rest_position_rad is not None
    )
    full_scale: float | None = None
    if enhanced_metrics:
        fitted = slope * angle + intercept
        residual = torque - fitted
        centered = torque - float(np.mean(torque))
        total_variation = float(np.dot(centered, centered))
        if total_variation <= 0.0:
            raise ContractError("spring torque must vary to evaluate linearity")
        full_scale = float(np.max(np.abs(torque)))
        if full_scale <= 0.0:
            raise ContractError("spring torque must have a non-zero full scale")
        result.update(
            {
                "r_squared": float(
                    1.0 - np.dot(residual, residual) / total_variation
                ),
                "fit_rmse_nm": float(np.sqrt(np.mean(np.square(residual)))),
                "full_scale_torque_nm": full_scale,
            }
        )
    if rest_position_rad is not None:
        assert full_scale is not None
        rest = float(rest_position_rad)
        if not np.isfinite(rest):
            raise ContractError("spring rest_position_rad must be finite")
        deflection = angle - rest
        denominator = float(np.dot(deflection, deflection))
        if denominator <= 0.0:
            raise ContractError("spring angle must vary from its configured rest position")
        neutral_stiffness = float(np.dot(deflection, torque) / denominator)
        neutral_residual = torque - neutral_stiffness * deflection
        result.update(
            {
                "rest_position_rad": rest,
                "neutral_stiffness_nm_per_rad": neutral_stiffness,
                "neutral_fit_rmse_nm": float(
                    np.sqrt(np.mean(np.square(neutral_residual)))
                ),
                "neutral_fit_rmse_full_scale_ratio": float(
                    np.sqrt(np.mean(np.square(neutral_residual))) / full_scale
                ),
            }
        )
    if repeat_index is not None or expected_repeats is not None:
        repeat_ids, repeats = _validated_repeats(
            repeat_index, angle.size, expected_repeats, "spring"
        )
        repeat_results: list[dict[str, Any]] = []
        for repeat_id in repeat_ids:
            selected = repeats == repeat_id
            if np.count_nonzero(selected) < 2 or float(np.ptp(angle[selected])) <= 0.0:
                raise ContractError("each spring repeat requires varying multi-sample angles")
            repeat_slope, repeat_intercept = np.polyfit(
                angle[selected], torque[selected], 1
            )
            repeat_results.append(
                {
                    "repeat_index": int(repeat_id),
                    "stiffness_nm_per_rad": float(repeat_slope),
                    "torque_intercept_nm": float(repeat_intercept),
                }
            )
        stiffness = np.asarray(
            [item["stiffness_nm_per_rad"] for item in repeat_results]
        )
        result.update(
            {
                "repeat_count": int(repeat_ids.size),
                "stiffness_nm_per_rad_std": float(np.std(stiffness)),
                "repeats": repeat_results,
            }
        )
        if enhanced_metrics:
            mean_stiffness = float(np.mean(stiffness))
            if mean_stiffness == 0.0:
                raise ContractError("spring repeat mean stiffness must be non-zero")
            result["stiffness_cv"] = float(
                np.std(stiffness) / abs(mean_stiffness)
            )
        if directions is not None and branches is not None:
            for repeat_id in repeat_ids:
                selected = repeats == repeat_id
                if set(np.unique(directions[selected])) != {-1.0, 1.0}:
                    raise ContractError("each spring repeat requires both torque directions")
                if set(np.unique(branches[selected])) != {-1.0, 1.0}:
                    raise ContractError("each spring repeat requires loading and unloading branches")
    if branches is not None:
        assert full_scale is not None
        branch_models: dict[float, tuple[float, float, float, float]] = {}
        for branch in (-1.0, 1.0):
            selected = branches == branch
            if np.count_nonzero(selected) < 2 or float(np.ptp(angle[selected])) <= 0.0:
                raise ContractError("each spring sweep branch requires varying multi-sample angles")
            branch_slope, branch_intercept = np.polyfit(angle[selected], torque[selected], 1)
            branch_models[branch] = (
                float(branch_slope),
                float(branch_intercept),
                float(np.min(angle[selected])),
                float(np.max(angle[selected])),
            )
        lower = max(branch_models[-1.0][2], branch_models[1.0][2])
        upper = min(branch_models[-1.0][3], branch_models[1.0][3])
        if lower >= upper:
            raise ContractError("spring loading and unloading angle ranges must overlap")
        differences = [
            abs(
                branch_models[1.0][0] * value
                + branch_models[1.0][1]
                - branch_models[-1.0][0] * value
                - branch_models[-1.0][1]
            )
            for value in (lower, upper)
        ]
        hysteresis = float(max(differences))
        result.update(
            {
                "hysteresis_width_nm": hysteresis,
                "hysteresis_full_scale_ratio": hysteresis / full_scale,
            }
        )
    return result


def _signed_unit_series(value: Any, name: str) -> np.ndarray:
    result = _series(value, name)
    if np.any((result != -1.0) & (result != 1.0)):
        raise ContractError(f"{name} must contain only -1 or 1")
    return result


def torsional_spring_holdout_metrics(
    *,
    angle_rad: Any,
    load_force: Any,
    lever_arm_m: Any,
    torque_direction: Any,
    calibration_metrics: Mapping[str, Any],
    rest_position_rad: float,
) -> dict[str, float]:
    """Evaluate held-out signed torques against both calibrated linear models."""

    angle = _series(angle_rad, "angle_rad")
    force = _series(load_force, "load_force")
    arm = _series(lever_arm_m, "lever_arm_m")
    direction = _signed_unit_series(torque_direction, "torque_direction")
    if len({angle.size, force.size, arm.size, direction.size}) != 1:
        raise ContractError("spring holdout channels must have matching shapes")
    if np.any(force < 0.0) or np.any(arm < 0.0):
        raise ContractError("spring holdout force and lever arm must be non-negative")
    torque = force * arm * direction
    full_scale = float(np.max(np.abs(torque)))
    if full_scale <= 0.0:
        raise ContractError("spring holdout torque must have a non-zero full scale")

    stiffness = float(calibration_metrics["stiffness_nm_per_rad"])
    intercept = float(calibration_metrics["torque_intercept_nm"])
    neutral_stiffness = float(calibration_metrics["neutral_stiffness_nm_per_rad"])
    rest = float(rest_position_rad)
    if not np.isfinite([stiffness, intercept, neutral_stiffness, rest]).all():
        raise ContractError("spring calibration model must contain finite values")
    residual = torque - (stiffness * angle + intercept)
    neutral_residual = torque - neutral_stiffness * (angle - rest)
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    neutral_rmse = float(np.sqrt(np.mean(np.square(neutral_residual))))
    return {
        "torque_rmse_nm": rmse,
        "full_scale_torque_nm": full_scale,
        "rmse_full_scale_ratio": rmse / full_scale,
        "neutral_model_torque_rmse_nm": neutral_rmse,
        "neutral_model_rmse_full_scale_ratio": neutral_rmse / full_scale,
    }


def torsional_spring_quality_gates(
    calibration_metrics: Mapping[str, Any],
    holdout_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the reviewed acceptance thresholds for the linear spring model."""

    checks = {
        "r_squared": float(calibration_metrics["r_squared"]) >= 0.98,
        "heldout_rmse": float(holdout_metrics["rmse_full_scale_ratio"]) <= 0.05,
        "stiffness_cv": float(calibration_metrics["stiffness_cv"]) <= 0.05,
        "hysteresis": float(calibration_metrics["hysteresis_full_scale_ratio"]) <= 0.10,
        "neutral_model_heldout_rmse": float(
            holdout_metrics["neutral_model_rmse_full_scale_ratio"]
        )
        <= 0.05,
    }
    return {"accepted": all(checks.values()), "gates": checks}


def torsional_spring_holdout_trace_metrics(
    calibration_metrics: Mapping[str, Any],
    trace: LoadedTrace,
) -> dict[str, float]:
    """Evaluate an authenticated holdout trace on its native channel clocks."""

    arrays = trace.arrays
    time_bases = trace.manifest.time_bases
    required = {"angle", "load_force", "lever_arm", "torque_direction"}
    if not required.issubset(arrays):
        raise ContractError("torsion-spring holdout is missing signed measurement channels")
    if time_bases["torque_direction"] != time_bases["angle"]:
        raise ContractError("spring torque_direction must use the angle clock")
    constants = trace.manifest.metadata.get("calibration_constants", {})
    rest_position = (
        constants.get("rest_position_rad")
        if isinstance(constants, Mapping)
        else None
    )
    if rest_position is None:
        raise ContractError(
            "torsion-spring holdout requires calibration_constants.rest_position_rad"
        )
    return torsional_spring_holdout_metrics(
        angle_rad=arrays["angle"],
        load_force=_interpolate(trace, "load_force", "angle"),
        lever_arm_m=_interpolate(trace, "lever_arm", "angle"),
        torque_direction=arrays["torque_direction"],
        calibration_metrics=calibration_metrics,
        rest_position_rad=float(rest_position),
    )


def _integer_series(value: Any, name: str) -> np.ndarray:
    series = _series(value, name)
    rounded = np.rint(series)
    if np.any(series != rounded) or np.any(rounded < 0.0):
        raise ContractError(f"{name} must contain non-negative integers")
    return rounded.astype(np.int64)


def _validated_repeats(
    repeat_index: Any,
    sample_count: int,
    expected_repeats: int | None,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    if (
        isinstance(expected_repeats, bool)
        or not isinstance(expected_repeats, int)
        or expected_repeats < 1
    ):
        raise ContractError(f"{label} expected_repeats must be a positive integer")
    repeats = _integer_series(repeat_index, f"{label} repeat_index")
    if repeats.size != sample_count:
        raise ContractError(f"{label} repeat_index must align with its native samples")
    repeat_ids = np.unique(repeats)
    if repeat_ids.size != expected_repeats:
        raise ContractError(
            f"{label} trace requires exactly {expected_repeats} repeats"
        )
    return repeat_ids, repeats


def _abad_fit(command: np.ndarray, measured: np.ndarray) -> dict[str, float | int]:
    if np.unique(command).size < 3:
        raise ContractError("each ABAD repeat must contain at least three distinct poses")
    scale, offset = np.polyfit(command, measured, 1)
    residual = measured - (scale * command + offset)
    return {
        "target_scale": float(scale),
        "target_offset_rad": float(offset),
        "fit_rmse_rad": float(np.sqrt(np.mean(np.square(residual)))),
        "pose_count": int(np.unique(command).size),
        "sample_count": int(command.size),
    }


def abad_static_mapping_metrics(
    command: Any,
    position: Any,
    *,
    repeat_index: Any,
    settled: Any,
    expected_repeats: int,
    frame: str,
) -> dict[str, Any]:
    """Fit ``actual = scale * requested + offset`` from settled static repeats."""

    targets = _series(command, "command")
    measured = _series(position, "position")
    repeats = _integer_series(repeat_index, "repeat_index")
    settled_values = _integer_series(settled, "settled")
    if len({targets.size, measured.size, repeats.size, settled_values.size}) != 1:
        raise ContractError("ABAD measurement channels must have matching shapes")
    if np.any((settled_values != 0) & (settled_values != 1)):
        raise ContractError("settled must contain only 0 or 1")
    if (
        isinstance(expected_repeats, bool)
        or not isinstance(expected_repeats, int)
        or expected_repeats < 1
    ):
        raise ContractError("expected_repeats must be a positive integer")
    if not isinstance(frame, str) or not frame.strip():
        raise ContractError("ABAD frame must be non-empty")

    selected = settled_values == 1
    if not np.any(selected):
        raise ContractError("ABAD trace contains no settled measurements")
    targets = targets[selected]
    measured = measured[selected]
    repeats = repeats[selected]
    repeat_ids = np.unique(repeats)
    if repeat_ids.size != expected_repeats:
        raise ContractError(
            f"ABAD trace requires exactly {expected_repeats} settled repeats"
        )

    repeat_results: list[dict[str, Any]] = []
    for repeat_id in repeat_ids:
        in_repeat = repeats == repeat_id
        repeat_results.append(
            {
                "repeat_index": int(repeat_id),
                **_abad_fit(targets[in_repeat], measured[in_repeat]),
            }
        )
    aggregate = _abad_fit(targets, measured)
    variation: dict[str, float | int] = {}
    for name in ("target_scale", "target_offset_rad", "fit_rmse_rad"):
        values = np.asarray([item[name] for item in repeat_results], dtype=float)
        variation[f"{name}_mean"] = float(np.mean(values))
        variation[f"{name}_std"] = float(np.std(values))
        variation[f"{name}_count"] = int(values.size)
    return {
        "schema_version": 1,
        "metric_kind": "abad_static_mapping",
        "repeat_count": int(repeat_ids.size),
        "equation": "actual_rad = target_scale * requested_rad + target_offset_rad",
        "frame": frame,
        "units": {
            "target_scale": "1",
            "target_offset_rad": "rad",
            "fit_rmse_rad": "rad",
        },
        "aggregate": aggregate,
        "repeat_variation": variation,
        "repeats": repeat_results,
    }


def variation_metrics(values: Any, *, metric_name: str) -> dict[str, float | int]:
    samples = _series(values, metric_name)
    if not metric_name or not isinstance(metric_name, str):
        raise ContractError("metric_name must be a non-empty string")
    return {
        f"{metric_name}_mean": float(np.mean(samples)),
        f"{metric_name}_std": float(np.std(samples)),
        f"{metric_name}_count": int(samples.size),
    }


def torque_saturation_metrics(
    load_force: Any,
    lever_arm: Any,
    command: Any,
    direction: Any,
    saturation_confirmed: Any,
    *,
    repeat_index: Any | None = None,
    expected_repeats: int | None = None,
) -> dict[str, Any]:
    force = _series(load_force, "load_force")
    arm = _series(lever_arm, "lever_arm")
    pwm = _series(command, "command")
    sign = _series(direction, "direction")
    confirmed = _integer_series(saturation_confirmed, "saturation_confirmed")
    if len({force.size, arm.size, pwm.size, sign.size, confirmed.size}) != 1:
        raise ContractError("manual-load channels must have matching shapes")
    if np.any(arm < 0.0) or np.any(np.abs(sign) != 1.0):
        raise ContractError("lever arms must be non-negative and direction must be -1 or 1")
    if np.any(confirmed != 1):
        raise ContractError(
            "manual-load effort fitting requires every sample to confirm current-limited saturation"
        )
    torque = np.abs(force * arm)
    result = {
        "max_command": float(np.max(np.abs(pwm))),
    }
    if repeat_index is not None or expected_repeats is not None:
        repeat_ids, repeats = _validated_repeats(
            repeat_index, force.size, expected_repeats, "manual-load"
        )
        repeat_results: list[dict[str, Any]] = []
        positive: list[float] = []
        negative: list[float] = []
        for repeat_id in repeat_ids:
            selected = repeats == repeat_id
            repeat_signs = set(sign[selected].tolist())
            if repeat_signs != {-1.0, 1.0}:
                raise ContractError(
                    "each manual-load repeat must contain confirmed positive and negative saturation"
                )
            positive_value = float(np.max(torque[selected & (sign > 0.0)]))
            negative_value = float(np.max(torque[selected & (sign < 0.0)]))
            positive.append(positive_value)
            negative.append(negative_value)
            repeat_results.append(
                {
                    "repeat_index": int(repeat_id),
                    "positive_torque_nm": positive_value,
                    "negative_torque_nm": negative_value,
                    "torque_saturation_nm": 0.5 * (
                        positive_value + negative_value
                    ),
                }
            )
        positive_values = np.asarray(positive)
        negative_values = np.asarray(negative)
        combined = np.concatenate((positive_values, negative_values))
        result.update(
            {
                "metric_kind": "torque_saturation",
                "repeat_count": int(repeat_ids.size),
                "torque_saturation_nm": float(np.mean(combined)),
                "torque_saturation_nm_std": float(np.std(combined)),
                "positive_torque_nm": float(np.mean(positive_values)),
                "positive_torque_nm_std": float(np.std(positive_values)),
                "negative_torque_nm": float(np.mean(negative_values)),
                "negative_torque_nm_std": float(np.std(negative_values)),
                "repeats": repeat_results,
            }
        )
    else:
        if set(sign.tolist()) != {-1.0, 1.0}:
            raise ContractError(
                "manual-load measurement must contain confirmed positive and negative saturation"
            )
        positive_value = float(np.max(torque[sign > 0.0]))
        negative_value = float(np.max(torque[sign < 0.0]))
        result.update(
            {
                "metric_kind": "torque_saturation",
                "torque_saturation_nm": 0.5 * (positive_value + negative_value),
                "positive_torque_nm": positive_value,
                "negative_torque_nm": negative_value,
            }
        )
    return result


def mass_com_metrics(
    scale_mass: Any,
    support_force: Any,
    support_position: Any,
    *,
    repeat_index: Any | None = None,
    expected_repeats: int | None = None,
) -> dict[str, Any]:
    masses = _series(scale_mass, "scale_mass")
    forces = np.asarray(support_force, dtype=float)
    positions = np.asarray(support_position, dtype=float)
    if (
        positions.ndim != 2
        or positions.shape[0] < 3
        or positions.shape[1] != 2
        or np.linalg.matrix_rank(positions[1:] - positions[0]) < 2
    ):
        raise ContractError(
            "mass-com requires at least three non-collinear planar support positions"
        )
    if forces.ndim == 1:
        representative_force = forces
    elif forces.ndim == 2:
        representative_force = np.median(forces, axis=0)
    else:
        raise ContractError("support_force must be one- or two-dimensional")
    if not np.isfinite(representative_force).all() or np.any(representative_force < 0.0):
        raise ContractError("support_force must be finite and non-negative")
    if positions.shape[0] != representative_force.size or not np.isfinite(positions).all():
        raise ContractError("support_position shape must match the support count")
    def weighted_com(force_vector: np.ndarray) -> np.ndarray:
        total_force = float(np.sum(force_vector))
        if total_force <= 0.0:
            raise ContractError("support force sum must be positive")
        return np.asarray(
            np.sum(
                positions
                * force_vector.reshape(
                    (-1,) + (1,) * (positions.ndim - 1)
                ),
                axis=0,
            )
            / total_force
        )

    com = weighted_com(representative_force)
    clean_com: Any = float(com) if np.ndim(com) == 0 else np.asarray(com).tolist()
    result = {
        "mass_kg": float(np.median(masses)),
        "com_m": clean_com,
        "com_x_m": float(com[0]),
        "com_y_m": float(com[1]),
    }
    if repeat_index is not None or expected_repeats is not None:
        repeat_ids, repeats = _validated_repeats(
            repeat_index, masses.size, expected_repeats, "mass-com"
        )
        if forces.ndim != 2 or forces.shape[0] != masses.size:
            raise ContractError(
                "repeat-aware mass-com support_force must contain one row per scale sample"
            )
        per_repeat_mass = np.asarray(
            [float(np.median(masses[repeats == repeat_id])) for repeat_id in repeat_ids]
        )
        per_repeat_com = np.asarray(
            [
                weighted_com(np.median(forces[repeats == repeat_id], axis=0))
                for repeat_id in repeat_ids
            ]
        )
        result.update(
            {
                "repeat_count": int(repeat_ids.size),
                "mass_kg_std": float(np.std(per_repeat_mass)),
                "com_m_std": (
                    float(np.std(per_repeat_com))
                    if per_repeat_com.ndim == 1
                    else np.std(per_repeat_com, axis=0).tolist()
                ),
                "com_x_m_std": float(np.std(per_repeat_com[:, 0])),
                "com_y_m_std": float(np.std(per_repeat_com[:, 1])),
                "repeats": [
                    {
                        "repeat_index": int(repeat_id),
                        "mass_kg": float(mass),
                        "com_m": (
                            float(repeat_com)
                            if np.ndim(repeat_com) == 0
                            else np.asarray(repeat_com).tolist()
                        ),
                    }
                    for repeat_id, mass, repeat_com in zip(
                        repeat_ids, per_repeat_mass, per_repeat_com, strict=True
                    )
                ],
            }
        )
    return result


def friction_metrics(
    *,
    breakaway_force: Any,
    static_normal_load: Any,
    static_repeat_index: Any,
    dynamic_pull_force: Any,
    dynamic_normal_load: Any,
    dynamic_speed: Any,
    dynamic_repeat_index: Any,
    expected_repeats: int,
    frame: str,
    max_dynamic_speed_m_s: float,
    max_speed_cv: float = 0.1,
) -> dict[str, Any]:
    """Identify static breakaway and dynamic constant-speed friction by repeat."""

    static_force = _series(breakaway_force, "breakaway_force")
    static_normal = _series(static_normal_load, "static_normal_load")
    static_repeats = _integer_series(static_repeat_index, "static_repeat_index")
    dynamic_force = _series(dynamic_pull_force, "dynamic_pull_force")
    dynamic_normal = _series(dynamic_normal_load, "dynamic_normal_load")
    speed = np.abs(_series(dynamic_speed, "dynamic_speed"))
    dynamic_repeats = _integer_series(dynamic_repeat_index, "dynamic_repeat_index")
    if len({static_force.size, static_normal.size, static_repeats.size}) != 1:
        raise ContractError("static friction channels must have matching shapes")
    if len({dynamic_force.size, dynamic_normal.size, speed.size, dynamic_repeats.size}) != 1:
        raise ContractError("dynamic friction channels must have matching shapes")
    if np.any(static_normal <= 0.0) or np.any(dynamic_normal <= 0.0):
        raise ContractError("friction normal loads must be positive")
    if (
        isinstance(expected_repeats, bool)
        or not isinstance(expected_repeats, int)
        or expected_repeats < 1
    ):
        raise ContractError("expected_repeats must be a positive integer")
    if not isinstance(frame, str) or not frame.strip():
        raise ContractError("friction frame must be non-empty")
    max_speed = float(max_dynamic_speed_m_s)
    speed_cv_limit = float(max_speed_cv)
    if not np.isfinite(max_speed) or max_speed <= 0.0:
        raise ContractError("maximum dynamic speed must be positive and finite")
    if not np.isfinite(speed_cv_limit) or speed_cv_limit < 0.0:
        raise ContractError("speed CV limit must be finite and non-negative")

    static_ids = np.unique(static_repeats)
    dynamic_ids = np.unique(dynamic_repeats)
    if static_ids.size != expected_repeats or not np.array_equal(static_ids, dynamic_ids):
        raise ContractError("static and dynamic friction must cover the same expected repeats")

    static_results: list[dict[str, Any]] = []
    dynamic_results: list[dict[str, Any]] = []
    for repeat_id in static_ids:
        static_mask = static_repeats == repeat_id
        if np.count_nonzero(static_mask) != 1:
            raise ContractError("each static repeat must contain one breakaway threshold")
        static_coefficient = float(
            np.abs(static_force[static_mask][0]) / static_normal[static_mask][0]
        )
        static_results.append(
            {
                "repeat_index": int(repeat_id),
                "coefficient": static_coefficient,
                "breakaway_force_n": float(np.abs(static_force[static_mask][0])),
                "normal_load_n": float(static_normal[static_mask][0]),
            }
        )

        dynamic_mask = dynamic_repeats == repeat_id
        if np.count_nonzero(dynamic_mask) < 2:
            raise ContractError("each dynamic repeat requires at least two samples")
        repeat_speed = speed[dynamic_mask]
        mean_speed = float(np.mean(repeat_speed))
        speed_std = float(np.std(repeat_speed))
        if mean_speed <= 0.0 or float(np.max(repeat_speed)) > max_speed:
            raise ContractError("dynamic pull speed must be nonzero and remain slow")
        speed_cv = speed_std / mean_speed
        if speed_cv > speed_cv_limit:
            raise ContractError("dynamic pull must maintain constant speed within the CV limit")
        coefficient = float(
            np.median(np.abs(dynamic_force[dynamic_mask]) / dynamic_normal[dynamic_mask])
        )
        dynamic_results.append(
            {
                "repeat_index": int(repeat_id),
                "coefficient": coefficient,
                "sample_count": int(np.count_nonzero(dynamic_mask)),
                "speed_mean_m_s": mean_speed,
                "speed_std_m_s": speed_std,
                "speed_cv": speed_cv,
            }
        )

    def summarize(repeats: list[dict[str, Any]], method: str) -> dict[str, Any]:
        coefficients = np.asarray([item["coefficient"] for item in repeats], dtype=float)
        return {
            "method": method,
            "coefficient_mean": float(np.mean(coefficients)),
            "coefficient_std": float(np.std(coefficients)),
            "coefficient_count": int(coefficients.size),
            "repeats": repeats,
        }

    return {
        "schema_version": 1,
        "metric_kind": "ground_friction",
        "repeat_count": int(static_ids.size),
        "frame": frame,
        "units": {"coefficient": "1", "force": "N", "speed": "m/s"},
        "static": summarize(static_results, "breakaway_force"),
        "dynamic": summarize(dynamic_results, "constant_speed_pull"),
    }


def static_settle_metrics(
    root_position: Any,
    contact_force_n: Any,
    *,
    repeat_index: Any,
    settled: Any,
    expected_repeats: int,
) -> dict[str, Any]:
    """Summarize settled root height and total foot force by repeat."""

    position = np.asarray(root_position, dtype=float)
    force = np.asarray(contact_force_n, dtype=float)
    repeats = _integer_series(repeat_index, "repeat_index")
    settled_values = _integer_series(settled, "settled")
    if position.ndim != 2 or position.shape[1] < 3:
        raise ContractError("root_position must contain xyz samples")
    if force.ndim < 1 or force.shape[0] != position.shape[0]:
        raise ContractError("contact_force_n must align with root_position")
    if repeats.size != position.shape[0] or settled_values.size != position.shape[0]:
        raise ContractError("static-settle annotations must align with root_position")
    if not np.isfinite(position).all() or not np.isfinite(force).all():
        raise ContractError("static-settle samples must be finite")
    if np.any(force < 0.0):
        raise ContractError("contact_force_n must be non-negative")
    if np.any((settled_values != 0) & (settled_values != 1)):
        raise ContractError("settled must contain only 0 or 1")
    if (
        isinstance(expected_repeats, bool)
        or not isinstance(expected_repeats, int)
        or expected_repeats < 1
    ):
        raise ContractError("expected_repeats must be a positive integer")
    repeat_ids = np.unique(repeats)
    if repeat_ids.size != expected_repeats:
        raise ContractError(
            f"static-settle trace requires exactly {expected_repeats} repeats"
        )
    total_force = force if force.ndim == 1 else np.sum(force, axis=tuple(range(1, force.ndim)))
    repeat_results: list[dict[str, Any]] = []
    for repeat_id in repeat_ids:
        selected = (repeats == repeat_id) & (settled_values == 1)
        if not np.any(selected):
            raise ContractError("each static-settle repeat requires settled samples")
        repeat_results.append(
            {
                "repeat_index": int(repeat_id),
                "root_height_m": float(np.mean(position[selected, 2])),
                "contact_force_n": float(np.mean(total_force[selected])),
                "sample_count": int(np.count_nonzero(selected)),
            }
        )
    heights = np.asarray([item["root_height_m"] for item in repeat_results])
    forces = np.asarray([item["contact_force_n"] for item in repeat_results])
    return {
        "schema_version": 1,
        "metric_kind": "contact_static_settle",
        "repeat_count": int(repeat_ids.size),
        "frames": {"root_height_m": "world", "contact_force_n": "feet/ground"},
        "units": {"root_height_m": "m", "contact_force_n": "N"},
        "settled": {
            "root_height_m": float(np.mean(heights)),
            "root_height_m_std": float(np.std(heights)),
            "contact_force_n": float(np.mean(forces)),
            "contact_force_n_std": float(np.std(forces)),
            "repeat_count": int(repeat_ids.size),
        },
        "repeats": repeat_results,
    }


def _interpolate(trace: LoadedTrace, source: str, target: str) -> np.ndarray:
    source_time = trace.arrays[trace.manifest.time_bases[source]]
    target_time = trace.arrays[trace.manifest.time_bases[target]]
    values = trace.arrays[source]
    if values.ndim != 1:
        raise ContractError(f"cannot interpolate multidimensional channel {source}")
    if source_time[0] > target_time[0] or source_time[-1] < target_time[-1]:
        raise ContractError(f"{source} clock does not cover {target} clock")
    return np.interp(target_time, source_time, values)


def compute_subsystem_metrics(
    scenario: ScenarioSpecV1,
    trace: LoadedTrace,
) -> dict[str, Any]:
    arrays = trace.arrays
    time_bases = trace.manifest.time_bases
    kind = scenario.experiment_kind
    if kind == "step":
        return bidirectional_step_metrics(
            arrays[time_bases["command"]],
            arrays["command"],
            arrays[time_bases["position"]],
            arrays["position"],
        )
    if kind == "coast":
        return bidirectional_coast_metrics(
            arrays[time_bases["command"]],
            arrays["command"],
            arrays[time_bases["position"]],
            arrays["position"],
        )
    if kind == "step_coast":
        arguments = (
            arrays[time_bases["command"]],
            arrays["command"],
            arrays[time_bases["position"]],
            arrays["position"],
        )
        return {
            "step": bidirectional_step_metrics(*arguments),
            "coast": bidirectional_coast_metrics(*arguments),
        }
    if kind == "manual_load":
        target = "load_force"
        if time_bases["repeat_index"] != time_bases[target]:
            raise ContractError(
                "manual-load repeat_index must use the load-force clock"
            )
        result = torque_saturation_metrics(
            arrays[target],
            _interpolate(trace, "lever_arm", target),
            _interpolate(trace, "command", target),
            _interpolate(trace, "direction", target),
            arrays["saturation_confirmed"],
            repeat_index=arrays["repeat_index"],
            expected_repeats=scenario.repeats,
        )
        return {"schema_version": 1, "frame": scenario.joint, **result}
    if kind == "mass_com":
        if time_bases["repeat_index"] != time_bases["scale_mass"]:
            raise ContractError("mass-com repeat_index must use the scale clock")
        result = mass_com_metrics(
            arrays["scale_mass"],
            arrays["support_force"],
            arrays["support_position"],
            repeat_index=arrays["repeat_index"],
            expected_repeats=scenario.repeats,
        )
        return {
            "schema_version": 1,
            "metric_kind": "mass_com",
            "frame": "root",
            **result,
        }
    if kind == "abad_static":
        command = _interpolate(trace, "command", "position")
        position_clock = time_bases["position"]
        if time_bases["repeat_index"] != position_clock or time_bases["settled"] != position_clock:
            raise ContractError("ABAD repeat_index and settled annotations must use the position clock")
        return abad_static_mapping_metrics(
            command,
            arrays["position"],
            repeat_index=arrays["repeat_index"],
            settled=arrays["settled"],
            expected_repeats=scenario.repeats,
            frame=scenario.joint,
        )
    if kind == "spring":
        if time_bases["repeat_index"] != time_bases["angle"]:
            raise ContractError("spring repeat_index must use the angle clock")
        signed_channels = {"torque_direction", "sweep_branch"}
        present_signed = signed_channels.intersection(arrays)
        if present_signed and present_signed != signed_channels:
            raise ContractError(
                "spring torque_direction and sweep_branch must be recorded together"
            )
        if any(time_bases[name] != time_bases["angle"] for name in present_signed):
            raise ContractError("spring signed annotations must use the angle clock")
        constants = trace.manifest.metadata.get("calibration_constants", {})
        rest_position = (
            constants.get("rest_position_rad")
            if isinstance(constants, Mapping)
            else None
        )
        if scenario.scenario_id.startswith("torsion-spring") and rest_position is None:
            raise ContractError(
                "managed torsion-spring traces require calibration_constants.rest_position_rad"
            )
        result = torsional_spring_metrics(
            angle_rad=arrays["angle"],
            load_force=_interpolate(trace, "load_force", "angle"),
            lever_arm_m=_interpolate(trace, "lever_arm", "angle"),
            torque_direction=arrays.get("torque_direction"),
            sweep_branch=arrays.get("sweep_branch"),
            repeat_index=arrays["repeat_index"],
            expected_repeats=scenario.repeats,
            rest_position_rad=rest_position,
        )
        return {
            "schema_version": 1,
            "metric_kind": "torsional_spring",
            "frame": scenario.joint,
            **result,
        }
    if kind == "friction":
        static_clock = time_bases["breakaway_force"]
        dynamic_clock = time_bases["dynamic_pull_force"]
        if any(
            time_bases[name] != static_clock
            for name in ("static_normal_load", "static_repeat_index")
        ):
            raise ContractError("static friction annotations must use the breakaway clock")
        if any(
            time_bases[name] != dynamic_clock
            for name in ("dynamic_normal_load", "dynamic_speed", "dynamic_repeat_index")
        ):
            raise ContractError("dynamic friction channels must use one native pull clock")
        return friction_metrics(
            breakaway_force=arrays["breakaway_force"],
            static_normal_load=arrays["static_normal_load"],
            static_repeat_index=arrays["static_repeat_index"],
            dynamic_pull_force=arrays["dynamic_pull_force"],
            dynamic_normal_load=arrays["dynamic_normal_load"],
            dynamic_speed=arrays["dynamic_speed"],
            dynamic_repeat_index=arrays["dynamic_repeat_index"],
            expected_repeats=scenario.repeats,
            frame=f"{scenario.joint}/ground",
            max_dynamic_speed_m_s=max(
                abs(float(segment["value"])) for segment in scenario.command_segments
            ),
        )
    if kind == "static_settle":
        clock = time_bases["root_position"]
        if any(
            time_bases[name] != clock
            for name in ("contact_force_n", "repeat_index", "settled")
        ):
            raise ContractError("static-settle channels must use one native physics clock")
        return static_settle_metrics(
            arrays["root_position"],
            arrays["contact_force_n"],
            repeat_index=arrays["repeat_index"],
            settled=arrays["settled"],
            expected_repeats=scenario.repeats,
        )
    if kind == "audit":
        return {"sample_count": int(arrays[scenario.required_channels[0]].shape[0])}
    raise ContractError(f"unsupported experiment kind: {kind}")
