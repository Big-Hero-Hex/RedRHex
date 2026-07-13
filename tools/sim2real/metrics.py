from __future__ import annotations

from typing import Any

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
) -> dict[str, float]:
    angle = _series(angle_rad, "angle_rad")
    force = _series(load_force, "load_force")
    arm = _series(lever_arm_m, "lever_arm_m")
    if len({angle.size, force.size, arm.size}) != 1 or angle.size < 2:
        raise ContractError("spring angle, load force, and lever arm must have matching shapes")
    if np.any(arm < 0.0) or float(np.ptp(angle)) <= 0.0:
        raise ContractError("spring lever arm must be non-negative and angle must vary")
    torque = force * arm
    slope, intercept = np.polyfit(angle, torque, 1)
    return {
        "stiffness_nm_per_rad": float(slope),
        "torque_intercept_nm": float(intercept),
    }


def abad_static_mapping_metrics(command: Any, position: Any) -> dict[str, float | int]:
    """Fit the observable static ABAD target mapping, without claiming dynamics."""

    targets = _series(command, "command")
    measured = _series(position, "position")
    if targets.size != measured.size or targets.size < 3:
        raise ContractError("ABAD command and position must have at least three matching samples")
    if float(np.ptp(targets)) <= 0.0:
        raise ContractError("ABAD command must vary to identify target scale and offset")
    scale, offset = np.polyfit(targets, measured, 1)
    residual = measured - (scale * targets + offset)
    return {
        "target_scale": float(scale),
        "target_offset_rad": float(offset),
        "fit_rmse_rad": float(np.sqrt(np.mean(np.square(residual)))),
        "sample_count": int(targets.size),
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
) -> dict[str, float]:
    force = _series(load_force, "load_force")
    arm = _series(lever_arm, "lever_arm")
    pwm = _series(command, "command")
    sign = _series(direction, "direction")
    if len({force.size, arm.size, pwm.size, sign.size}) != 1:
        raise ContractError("manual-load channels must have matching shapes")
    if np.any(arm < 0.0) or np.any(np.abs(sign) != 1.0):
        raise ContractError("lever arms must be non-negative and direction must be -1 or 1")
    torque = np.abs(force * arm)
    result = {
        "torque_saturation_nm": float(np.max(torque)),
        "max_command": float(np.max(np.abs(pwm))),
    }
    for value, label in ((1.0, "positive_torque_nm"), (-1.0, "negative_torque_nm")):
        selected = torque[sign == value]
        if selected.size:
            result[label] = float(np.max(selected))
    return result


def mass_com_metrics(
    scale_mass: Any,
    support_force: Any,
    support_position: Any,
) -> dict[str, Any]:
    masses = _series(scale_mass, "scale_mass")
    forces = np.asarray(support_force, dtype=float)
    positions = np.asarray(support_position, dtype=float)
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
    total_force = float(np.sum(representative_force))
    if total_force <= 0.0:
        raise ContractError("support force sum must be positive")
    com = np.sum(
        positions * representative_force.reshape((-1,) + (1,) * (positions.ndim - 1)),
        axis=0,
    ) / total_force
    clean_com: Any = float(com) if np.ndim(com) == 0 else np.asarray(com).tolist()
    return {"mass_kg": float(np.median(masses)), "com_m": clean_com}


def friction_metrics(
    *,
    pull_force: Any | None = None,
    normal_load: Any | None = None,
    incline_angle_rad: Any | None = None,
    dynamic_pull_force: Any | None = None,
) -> dict[str, float]:
    result: dict[str, float] = {}
    if incline_angle_rad is not None:
        angles = _series(incline_angle_rad, "incline_angle_rad")
        result["static_friction_coefficient"] = float(np.tan(np.median(np.abs(angles))))
    elif pull_force is not None and normal_load is not None:
        pull = _series(pull_force, "pull_force")
        normal = _series(normal_load, "normal_load")
        if pull.size != normal.size or np.any(normal <= 0.0):
            raise ContractError("pull_force and positive normal_load must have matching shapes")
        result["static_friction_coefficient"] = float(np.median(np.abs(pull) / normal))
    else:
        raise ContractError("friction requires incline angle or pull force plus normal load")
    if dynamic_pull_force is not None:
        if normal_load is None:
            raise ContractError("dynamic friction requires normal_load")
        dynamic = _series(dynamic_pull_force, "dynamic_pull_force")
        normal = _series(normal_load, "normal_load")
        if dynamic.size != normal.size or np.any(normal <= 0.0):
            raise ContractError("dynamic pull and normal load must have matching shapes")
        result["dynamic_friction_coefficient"] = float(np.median(np.abs(dynamic) / normal))
    return result


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
        return torque_saturation_metrics(
            arrays[target],
            _interpolate(trace, "lever_arm", target),
            _interpolate(trace, "command", target),
            _interpolate(trace, "direction", target),
        )
    if kind == "mass_com":
        return mass_com_metrics(arrays["scale_mass"], arrays["support_force"], arrays["support_position"])
    if kind == "abad_static":
        command = _interpolate(trace, "command", "position")
        return abad_static_mapping_metrics(command, arrays["position"])
    if kind == "spring":
        return torsional_spring_metrics(
            angle_rad=arrays["angle"],
            load_force=_interpolate(trace, "load_force", "angle"),
            lever_arm_m=_interpolate(trace, "lever_arm", "angle"),
        )
    if kind == "friction":
        return friction_metrics(
            pull_force=arrays["pull_force"],
            normal_load=_interpolate(trace, "normal_load", "pull_force"),
            dynamic_pull_force=_interpolate(
                trace, "dynamic_pull_force", "pull_force"
            ),
        )
    if kind == "audit":
        return {"sample_count": int(arrays[scenario.required_channels[0]].shape[0])}
    raise ContractError(f"unsupported experiment kind: {kind}")
