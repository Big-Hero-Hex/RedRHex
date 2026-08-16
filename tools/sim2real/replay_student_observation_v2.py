#!/usr/bin/env python3
"""Replay synchronized IMU/encoder samples through the Sensor-Only V2 pipe.

The input is a canonical ``.npz`` trace sampled at the policy rate.  Encoder
positions must already be calibrated radians in the policy joint order.  Raw
ROS messages should first be converted by the hardware bridge/importer so this
gate never guesses joint names, signs, zero offsets, or clock alignment.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Protocol

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_IO_ROOT = REPO_ROOT / "source" / "redrhex_policy_io"
ROS_CONTROLLER_ROOT = REPO_ROOT / "ros2_ws" / "src" / "redrhex_rl_controller"
DEFAULT_MAX_MAIN_ACTION_SATURATION_FRACTION = 0.05
DEPLOYMENT_HARDWARE_CONFIG_PATH_V2 = (
    ROS_CONTROLLER_ROOT / "config" / "redrhex_policy_sensor_v2.yaml"
)
MAX_REAL_HARDWARE_TARGET_TIGHTENING_FRACTION_V2 = 0.0
HARDWARE_TARGET_TIGHTENING_LIMIT_SOURCE_V2 = (
    "exact sim/deployment raw-target parity"
)
MAIN_ACTION_SATURATION_SENSITIVITY = {
    "strict": 0.0,
    "base": DEFAULT_MAX_MAIN_ACTION_SATURATION_FRACTION,
    "relaxed": 0.10,
}
MAIN_ACTION_SATURATION_LIMIT_SOURCE = (
    "interim recorded-real no-saturation-anomaly safety gate; replace only "
    "with reviewed hardware characterization evidence"
)
for _path in (POLICY_IO_ROOT, ROS_CONTROLLER_ROOT, REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from redrhex_policy_io import (  # noqa: E402
    ContractError,
    ForwardResidualActionContractV2,
    SensorCalibrationProfileV2,
    SensorFrameBuilderV2,
    SensorHistoryBufferV2,
    StudentObservationContractV2,
)
from tools.sim2real.import_sensor_v2_rosbag import (  # noqa: E402
    MAX_PERIOD_ERROR_RATIO_V2,
    ValidatedSensorV2ImportReceipt,
    sha256_path_v2,
    validate_sensor_v2_import_receipt,
)
from redrhex_rl_controller.action_decoder_v2 import (  # noqa: E402
    ForwardResidualActionDecoderV2,
)


class PolicyRunnerV2(Protocol):
    observation_contract: StudentObservationContractV2
    action_contract: ForwardResidualActionContractV2
    metadata: Mapping[str, str]
    runtime_calibration_sha256: str
    training_calibration_sha256: str
    checkpoint_sha256: str

    def run(self, sensor_history: np.ndarray, command: np.ndarray) -> object: ...


def _runner_sha256(
    runner: PolicyRunnerV2 | None,
    metadata_key: str,
    *,
    attribute: str | None = None,
) -> str | None:
    if runner is None:
        return None
    value = getattr(runner, attribute, None) if attribute is not None else None
    if value is None:
        metadata = getattr(runner, "metadata", {})
        value = metadata.get(metadata_key) if isinstance(metadata, Mapping) else None
    if not isinstance(value, str) or len(value) != 64:
        return None
    if any(character not in "0123456789abcdef" for character in value):
        return None
    return value


def _runner_non_negative_int(
    runner: PolicyRunnerV2 | None,
    metadata_key: str,
) -> int | None:
    if runner is None:
        return None
    metadata = getattr(runner, "metadata", {})
    value = metadata.get(metadata_key) if isinstance(metadata, Mapping) else None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _array(trace: Mapping[str, Any], name: str, shape_tail: tuple[int, ...]) -> np.ndarray:
    if name not in trace:
        raise ContractError(f"trace is missing required array {name!r}")
    result = np.asarray(trace[name], dtype=np.float64)
    if result.ndim != len(shape_tail) + 1 or result.shape[1:] != shape_tail:
        raise ContractError(
            f"{name} must have shape (N, {', '.join(map(str, shape_tail))}), "
            f"got {result.shape}"
        )
    if not np.isfinite(result).all():
        raise ContractError(f"{name} contains NaN or Inf")
    return result


def _broadcast_rows(
    trace: Mapping[str, Any], name: str, rows: int, width: int
) -> np.ndarray:
    if name not in trace:
        raise ContractError(f"trace is missing required array {name!r}")
    values = np.asarray(trace[name], dtype=np.float64)
    if values.shape == (width,):
        values = np.broadcast_to(values, (rows, width)).copy()
    if values.shape != (rows, width) or not np.isfinite(values).all():
        raise ContractError(f"{name} must be finite with shape ({rows}, {width}) or ({width},)")
    return values


def _scalar_text(trace: Mapping[str, Any], name: str) -> str | None:
    if name not in trace:
        return None
    value = np.asarray(trace[name])
    if value.size != 1:
        raise ContractError(f"{name} must be a scalar string")
    return str(value.reshape(()).item())


def _validate_timing(
    timestamps_s: np.ndarray,
    *,
    sample_rate_hz: float,
    max_period_error_ratio: float,
) -> dict[str, float]:
    if timestamps_s.ndim != 1 or timestamps_s.size == 0:
        raise ContractError("timestamp_s must be a non-empty one-dimensional array")
    if not np.isfinite(timestamps_s).all():
        raise ContractError("timestamp_s contains NaN or Inf")
    periods = np.diff(timestamps_s)
    if periods.size and np.any(periods <= 0.0):
        raise ContractError("timestamp_s must be strictly increasing")
    expected = 1.0 / float(sample_rate_hz)
    if not 0.0 <= max_period_error_ratio < 1.0:
        raise ContractError("max_period_error_ratio must be in [0, 1)")
    if periods.size:
        period_error = np.abs(periods - expected) / expected
        if np.any(period_error > max_period_error_ratio):
            worst = float(np.max(period_error))
            raise ContractError(
                "trace sample cadence violates the observation contract: "
                f"maximum relative period error {worst:.6f} > {max_period_error_ratio:.6f}"
            )
        return {
            "mean_rate_hz": float(1.0 / np.mean(periods)),
            "min_period_s": float(np.min(periods)),
            "max_period_s": float(np.max(periods)),
            "max_period_error_ratio": float(np.max(period_error)),
        }
    return {
        "mean_rate_hz": float(sample_rate_hz),
        "min_period_s": expected,
        "max_period_s": expected,
        "max_period_error_ratio": 0.0,
    }


def _feature_statistics(
    frames: np.ndarray, contract: StudentObservationContractV2
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for spec in contract.FEATURE_LAYOUT:
        values = frames[:, spec.start : spec.stop].astype(np.float64)
        result[spec.name] = {
            "dimension": spec.dimension,
            "mean": np.mean(values, axis=0).tolist(),
            "std": np.std(values, axis=0).tolist(),
            "minimum": np.min(values, axis=0).tolist(),
            "maximum": np.max(values, axis=0).tolist(),
        }
    return result


def _domain_shift(
    frames: np.ndarray,
    reference_frames: np.ndarray,
    contract: StudentObservationContractV2,
) -> list[dict[str, Any]]:
    reference = np.asarray(reference_frames, dtype=np.float64)
    expected_tail = (contract.sensor_frame_dim,)
    if reference.ndim != 2 or reference.shape[1:] != expected_tail:
        raise ContractError(f"reference sensor_frames must have shape (N, {expected_tail[0]})")
    if reference.shape[0] == 0 or not np.isfinite(reference).all():
        raise ContractError("reference sensor_frames must be non-empty and finite")
    rows: list[dict[str, Any]] = []
    for spec in contract.FEATURE_LAYOUT:
        current = frames[:, spec.start : spec.stop].astype(np.float64)
        baseline = reference[:, spec.start : spec.stop]
        mean_delta = np.mean(current, axis=0) - np.mean(baseline, axis=0)
        pooled_scale = np.sqrt(
            0.5 * (np.var(current, axis=0) + np.var(baseline, axis=0)) + 1.0e-12
        )
        standardized = np.abs(mean_delta) / pooled_scale
        rows.append(
            {
                "feature": spec.name,
                "max_standardized_mean_shift": float(np.max(standardized)),
                "mean_standardized_mean_shift": float(np.mean(standardized)),
                "mean_delta": mean_delta.tolist(),
            }
        )
    return sorted(rows, key=lambda row: row["max_standardized_mean_shift"], reverse=True)


def _target_divergence_v2(
    before: np.ndarray,
    after: np.ndarray,
) -> dict[str, float | int]:
    """Return exact element-wise divergence metrics for one hardware stage."""

    if before.shape != after.shape or before.ndim != 2 or before.shape[1:] != (6,):
        raise ContractError("hardware target arrays must have matching shape (N, 6)")
    delta = np.abs(after - before)
    changed = delta != 0.0
    target_count = int(delta.size)
    changed_count = int(np.count_nonzero(changed))
    return {
        "target_count": target_count,
        "tightened_target_count": changed_count,
        "tightening_fraction": (
            float(changed_count / target_count) if target_count else 0.0
        ),
        "max_abs_delta_rad_s": float(np.max(delta)) if target_count else 0.0,
    }


def _load_deployment_hardware_config_v2(
    path: Path = DEPLOYMENT_HARDWARE_CONFIG_PATH_V2,
) -> tuple[dict[str, float], dict[str, str]]:
    """Read the exact ROS safety envelope and bind the source bytes."""

    try:
        import yaml
    except ImportError as exc:
        raise ContractError(
            "PyYAML is required to verify the deployment hardware target config"
        ) from exc
    resolved = path.expanduser().resolve()
    try:
        payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        safety = payload["redrhex_rl_controller_v2"]["ros__parameters"]["safety"]
    except (OSError, UnicodeError, TypeError, KeyError, yaml.YAMLError) as exc:
        raise ContractError(
            f"cannot read deployment hardware target config {resolved}: {exc}"
        ) from exc
    result: dict[str, float] = {}
    for name in (
        "action_clip",
        "main_drive_vel_limit_rad_s",
        "main_drive_slew_rate_rad_s2",
    ):
        value = safety.get(name) if isinstance(safety, Mapping) else None
        if isinstance(value, bool):
            raise ContractError(f"deployment safety.{name} must be positive and finite")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ContractError(
                f"deployment safety.{name} must be positive and finite"
            ) from exc
        if not math.isfinite(number) or number <= 0.0:
            raise ContractError(f"deployment safety.{name} must be positive and finite")
        result[name] = number
    return result, {
        "path": str(resolved),
        "sha256": sha256_path_v2(resolved),
    }


def replay_arrays(
    trace: Mapping[str, Any],
    *,
    contract: StudentObservationContractV2,
    action_contract: ForwardResidualActionContractV2 | None = None,
    calibration: SensorCalibrationProfileV2 | None = None,
    runner: PolicyRunnerV2 | None = None,
    trace_kind: str = "sim",
    max_period_error_ratio: float = 0.25,
    max_main_action_saturation_fraction: float = (
        DEFAULT_MAX_MAIN_ACTION_SATURATION_FRACTION
    ),
    deployment_hardware_config_path: Path = DEPLOYMENT_HARDWARE_CONFIG_PATH_V2,
    reference_frames: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build every frame/history and optionally execute the policy.

    ``trace_kind='real'`` additionally requires a hash-bound, hardware-ready
    calibration.  Both modes require already calibrated encoder radians; the
    function deliberately has no permissive raw-count fallback.
    """

    contract = contract.validate()
    if trace_kind not in {"sim", "real"}:
        raise ContractError("trace_kind must be 'sim' or 'real'")
    if not 0.0 <= max_main_action_saturation_fraction <= 1.0:
        raise ContractError("max_main_action_saturation_fraction must be in [0, 1]")
    if calibration is not None:
        calibration = calibration.validate(require_hardware_ready=trace_kind == "real")
        if calibration.observation_contract_sha256 != contract.sha256:
            raise ContractError("calibration references a different observation contract")
        if action_contract is not None and calibration.action_contract_sha256 != action_contract.sha256:
            raise ContractError("calibration references a different action contract")
    elif trace_kind == "real":
        raise ContractError("real traces require a hardware-ready calibration profile")

    runner_runtime_calibration_sha256 = _runner_sha256(
        runner,
        "calibration_sha256",
        attribute="runtime_calibration_sha256",
    )
    runtime_calibration_sha256 = (
        calibration.sha256
        if calibration is not None
        else runner_runtime_calibration_sha256
    )
    training_calibration_sha256 = _runner_sha256(
        runner,
        "training_calibration_sha256",
        attribute="training_calibration_sha256",
    )
    if (
        calibration is not None
        and runner_runtime_calibration_sha256 is not None
        and runner_runtime_calibration_sha256 != calibration.sha256
    ):
        raise ContractError(
            "policy runner runtime calibration does not match replay calibration"
        )

    if runner is not None:
        if runner.observation_contract.sha256 != contract.sha256:
            raise ContractError("policy runner observation contract does not match replay")
        if action_contract is not None and runner.action_contract.sha256 != action_contract.sha256:
            raise ContractError("policy runner action contract does not match replay")

    timestamps = np.asarray(trace.get("timestamp_s"), dtype=np.float64)
    timing = _validate_timing(
        timestamps,
        sample_rate_hz=contract.sample_rate_hz,
        max_period_error_ratio=max_period_error_ratio,
    )
    count = int(timestamps.size)
    minimum_source_samples = contract.history_length + 1
    if count < minimum_source_samples:
        raise ContractError(
            "trace must contain at least one encoder-velocity baseline plus "
            f"{contract.history_length} physical history frames "
            f"({minimum_source_samples} source samples total)"
        )
    gyro = _array(trace, "imu_gyro_rad_s", (3,))
    main_position = _array(trace, "main_position_rad", (6,))
    abad_position = _array(trace, "abad_position_rad", (6,))
    command = _broadcast_rows(trace, "command", count, contract.command_dim)
    for name, values in (
        ("imu_gyro_rad_s", gyro),
        ("main_position_rad", main_position),
        ("abad_position_rad", abad_position),
    ):
        if values.shape[0] != count:
            raise ContractError(f"{name} row count does not match timestamp_s")

    main_velocity = None
    if "main_velocity_rad_s" in trace:
        main_velocity = _array(trace, "main_velocity_rad_s", (6,))
        if main_velocity.shape[0] != count:
            raise ContractError("main_velocity_rad_s row count does not match timestamp_s")

    if contract.attitude_mode == "causal_gyro_accel":
        accel = _array(trace, "imu_linear_accel_m_s2", (3,))
        if accel.shape[0] != count:
            raise ContractError("imu_linear_accel_m_s2 row count does not match timestamp_s")
        orientation = covariance = None
        imu_frame_id = None
    else:
        orientation = _array(trace, "imu_orientation_xyzw", (4,))
        if orientation.shape[0] != count:
            raise ContractError("imu_orientation_xyzw row count does not match timestamp_s")
        covariance = _broadcast_rows(trace, "imu_orientation_covariance", count, 9)
        imu_frame_id = _scalar_text(trace, "imu_frame_id")
        if imu_frame_id is None:
            raise ContractError("validated quaternion traces require scalar imu_frame_id")
        accel = None

    neutral = (
        action_contract.abad_neutral_position_rad
        if action_contract is not None
        else (0.0,) * 6
    )
    builder = SensorFrameBuilderV2(contract, abad_neutral_position_rad=neutral)
    history = SensorHistoryBufferV2(contract)
    frames: list[np.ndarray] = []
    frame_timestamps: list[float] = []
    histories: list[np.ndarray] = []
    ready_timestamps: list[float] = []
    ready_commands: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    velocity_estimates: list[np.ndarray] = []
    latency_ms: list[float] = []
    raw_contract_targets: list[np.ndarray] = []
    action_clipped_contract_targets: list[np.ndarray] = []
    hardware_slew_targets: list[np.ndarray] = []
    hardware_targets: list[np.ndarray] = []
    hardware_action_clip_applied: list[bool] = []
    hardware_slew_applied: list[bool] = []
    hardware_velocity_limit_applied: list[bool] = []
    hardware_decoder = None
    deployment_hardware_config = None
    deployment_hardware_config_record = None
    if runner is not None:
        if action_contract is None:
            raise ContractError(
                "policy replay requires the bundle action contract for hardware-target verification"
            )
        deployment_hardware_config, deployment_hardware_config_record = (
            _load_deployment_hardware_config_v2(deployment_hardware_config_path)
        )
        try:
            hardware_decoder = ForwardResidualActionDecoderV2(
                action_contract,
                {
                    "action_clip": deployment_hardware_config["action_clip"],
                    "main_drive_vel_limit_rad_s": deployment_hardware_config[
                        "main_drive_vel_limit_rad_s"
                    ],
                    "main_drive_slew_rate_rad_s2": deployment_hardware_config[
                        "main_drive_slew_rate_rad_s2"
                    ],
                },
            )
        except ValueError as exc:
            raise ContractError(
                f"bundle action contract is incompatible with deployment hardware limits: {exc}"
            ) from exc
    for index in range(count):
        frame = builder.build(
            timestamp_s=float(timestamps[index]),
            imu_gyro_rad_s=gyro[index],
            main_position_rad=main_position[index],
            abad_position_rad=abad_position[index],
            imu_linear_accel_m_s2=None if accel is None else accel[index],
            imu_orientation_xyzw=None if orientation is None else orientation[index],
            imu_orientation_covariance=None if covariance is None else covariance[index],
            imu_frame_id=imu_frame_id,
            rest_gravity_evidence_valid=(
                trace_kind == "sim"
                or bool(calibration and calibration.rest_gravity_evidence.strip())
            ),
            main_velocity_rad_s=None if main_velocity is None else main_velocity[index],
            main_velocity_valid=main_velocity is not None,
        )
        # SensorFrameBuilderV2 deliberately returns zero derived velocities for
        # its first event because no causal difference exists yet.  Match the
        # simulator and ROS runtime ownership boundary: consume that event only
        # as the encoder baseline and never expose the fabricated zero-velocity
        # frame to the actor history or replay statistics.
        if index == 0:
            continue
        frames.append(frame)
        frame_timestamps.append(float(timestamps[index]))
        history.append(frame)
        if not history.ready:
            continue
        value = history.array()
        histories.append(value)
        ready_timestamps.append(float(timestamps[index]))
        ready_commands.append(command[index].astype(np.float32))
        if runner is not None:
            started = time.perf_counter_ns()
            output = runner.run(value, command[index].astype(np.float32))
            latency_ms.append((time.perf_counter_ns() - started) * 1.0e-6)
            action = np.asarray(getattr(output, "actions"), dtype=np.float32)
            velocity = np.asarray(
                getattr(output, "base_velocity_estimate"), dtype=np.float32
            )
            if action.shape != (12,) or velocity.shape != (3,):
                raise ContractError("policy runner returned invalid V2 output shapes")
            if not np.isfinite(action).all() or not np.isfinite(velocity).all():
                raise ContractError("policy runner returned NaN or Inf")
            actions.append(action)
            velocity_estimates.append(velocity)
            assert hardware_decoder is not None
            replay_dt_s = (
                1.0 / contract.sample_rate_hz
                if len(ready_timestamps) == 1
                else ready_timestamps[-1] - ready_timestamps[-2]
            )
            hardware_decoder.decode(
                action,
                main_position[index],
                command[index],
                replay_dt_s,
            )
            target_status = hardware_decoder.last_target_status
            if target_status is None:
                raise ContractError("deployment action decoder produced no target status")
            raw_contract_targets.append(
                target_status.raw_contract_target_main_drive_velocity
            )
            action_clipped_contract_targets.append(
                target_status.action_clipped_contract_target_main_drive_velocity
            )
            hardware_slew_targets.append(
                target_status.hardware_slew_target_main_drive_velocity
            )
            hardware_targets.append(
                target_status.hardware_target_main_drive_velocity
            )
            hardware_action_clip_applied.append(
                target_status.hardware_action_clip_applied
            )
            hardware_slew_applied.append(target_status.hardware_slew_applied)
            hardware_velocity_limit_applied.append(
                target_status.hardware_velocity_limit_applied
            )

    frame_array = np.asarray(frames, dtype=np.float32)
    history_array = np.asarray(histories, dtype=np.float32).reshape(
        -1, contract.history_length, contract.sensor_frame_dim
    )
    output_arrays: dict[str, np.ndarray] = {
        "timestamp_s": timestamps,
        "sensor_frame_timestamp_s": np.asarray(frame_timestamps, dtype=np.float64),
        "sensor_frames": frame_array,
        "history_timestamp_s": np.asarray(ready_timestamps, dtype=np.float64),
        "sensor_histories": history_array,
        "command": np.asarray(ready_commands, dtype=np.float32).reshape(-1, 3),
    }
    if runner is not None:
        output_arrays.update(
            actions=np.asarray(actions, dtype=np.float32).reshape(-1, 12),
            base_velocity_estimate=np.asarray(velocity_estimates, dtype=np.float32).reshape(-1, 3),
            policy_latency_ms=np.asarray(latency_ms, dtype=np.float64),
            raw_contract_target_main_drive_velocity_rad_s=np.asarray(
                raw_contract_targets, dtype=np.float64
            ).reshape(-1, 6),
            action_clipped_contract_target_main_drive_velocity_rad_s=np.asarray(
                action_clipped_contract_targets, dtype=np.float64
            ).reshape(-1, 6),
            hardware_slew_target_main_drive_velocity_rad_s=np.asarray(
                hardware_slew_targets, dtype=np.float64
            ).reshape(-1, 6),
            hardware_target_main_drive_velocity_rad_s=np.asarray(
                hardware_targets, dtype=np.float64
            ).reshape(-1, 6),
            hardware_action_clip_applied=np.asarray(
                hardware_action_clip_applied, dtype=np.bool_
            ),
            hardware_slew_applied=np.asarray(
                hardware_slew_applied, dtype=np.bool_
            ),
            hardware_velocity_limit_applied=np.asarray(
                hardware_velocity_limit_applied, dtype=np.bool_
            ),
        )

    summary: dict[str, Any] = {
        "schema": "redrhex.sensor-v2-replay.v2",
        "status": "passed",
        "trace_kind": trace_kind,
        "sample_count": count,
        "velocity_baseline_samples": 1,
        "sensor_frame_count": len(frames),
        "history_ready_count": len(histories),
        "warmup_source_samples": minimum_source_samples,
        "warmup_frames": contract.history_length,
        "contract_sha256": contract.sha256,
        "action_contract_sha256": None if action_contract is None else action_contract.sha256,
        # calibration_sha256 is retained as the compatibility alias for the
        # deployment/runtime calibration record.
        "calibration_sha256": runtime_calibration_sha256,
        "runtime_calibration_sha256": runtime_calibration_sha256,
        "training_calibration_sha256": training_calibration_sha256,
        "checkpoint_sha256": _runner_sha256(
            runner,
            "checkpoint_sha256",
            attribute="checkpoint_sha256",
        ),
        "architecture_sha256": _runner_sha256(runner, "architecture_sha256"),
        "config_sha256": _runner_sha256(runner, "config_sha256"),
        "canonical_config_sha256": _runner_sha256(
            runner, "canonical_config_sha256"
        ),
        "training_seed": _runner_non_negative_int(runner, "training_seed"),
        "timing": timing,
        "feature_statistics": _feature_statistics(frame_array, contract),
    }
    if runner is not None:
        action_array = output_arrays["actions"]
        latency_array = output_arrays["policy_latency_ms"]
        raw_target_array = output_arrays[
            "raw_contract_target_main_drive_velocity_rad_s"
        ]
        action_clipped_target_array = output_arrays[
            "action_clipped_contract_target_main_drive_velocity_rad_s"
        ]
        slew_target_array = output_arrays[
            "hardware_slew_target_main_drive_velocity_rad_s"
        ]
        hardware_target_array = output_arrays[
            "hardware_target_main_drive_velocity_rad_s"
        ]
        action_abs_max = float(np.max(np.abs(action_array))) if action_array.size else None
        main_action_saturation_fraction = (
            float(np.mean(np.abs(action_array[:, :6]) >= 0.999))
            if action_array.size
            else None
        )
        abad_action_abs_max = (
            float(np.max(np.abs(action_array[:, 6:]))) if action_array.size else None
        )
        saturation_gate_passed = bool(
            main_action_saturation_fraction is not None
            and main_action_saturation_fraction
            <= max_main_action_saturation_fraction
        )
        summary["policy"] = {
            "inference_count": int(action_array.shape[0]),
            "latency_ms_mean": float(np.mean(latency_array)) if latency_array.size else None,
            "latency_ms_p95": float(np.percentile(latency_array, 95)) if latency_array.size else None,
            "action_abs_max": action_abs_max,
            "main_action_saturation_fraction": main_action_saturation_fraction,
            "max_main_action_saturation_fraction": max_main_action_saturation_fraction,
            "main_action_saturation_limit_source": (
                MAIN_ACTION_SATURATION_LIMIT_SOURCE
            ),
            "main_action_saturation_sensitivity": dict(
                MAIN_ACTION_SATURATION_SENSITIVITY
            ),
            "main_action_saturation_gate_passed": saturation_gate_passed,
            "abad_action_abs_max": abad_action_abs_max,
        }
        action_clip_divergence = _target_divergence_v2(
            raw_target_array,
            action_clipped_target_array,
        )
        slew_divergence = _target_divergence_v2(
            action_clipped_target_array,
            slew_target_array,
        )
        velocity_limit_divergence = _target_divergence_v2(
            slew_target_array,
            hardware_target_array,
        )
        total_divergence = _target_divergence_v2(
            raw_target_array,
            hardware_target_array,
        )
        hardware_target_gate_passed = bool(
            total_divergence["tightening_fraction"]
            <= MAX_REAL_HARDWARE_TARGET_TIGHTENING_FRACTION_V2
            and total_divergence["max_abs_delta_rad_s"] == 0.0
        )
        summary["hardware_target_tightening"] = {
            "bundle_main_velocity_limit_rad_s": float(
                action_contract.main_velocity_limit_rad_s
            ),
            "deployment_action_clip": deployment_hardware_config["action_clip"],
            "deployment_main_velocity_limit_rad_s": deployment_hardware_config[
                "main_drive_vel_limit_rad_s"
            ],
            "deployment_main_slew_rate_rad_s2": deployment_hardware_config[
                "main_drive_slew_rate_rad_s2"
            ],
            "deployment_config": deployment_hardware_config_record,
            "raw_contract_target_abs_max_rad_s": (
                float(np.max(np.abs(raw_target_array)))
                if raw_target_array.size
                else None
            ),
            "hardware_target_abs_max_rad_s": (
                float(np.max(np.abs(hardware_target_array)))
                if hardware_target_array.size
                else None
            ),
            "action_clip": action_clip_divergence,
            "slew_rate": slew_divergence,
            "velocity_limit": velocity_limit_divergence,
            "total": total_divergence,
            "max_total_tightening_fraction": (
                MAX_REAL_HARDWARE_TARGET_TIGHTENING_FRACTION_V2
            ),
            "limit_source": HARDWARE_TARGET_TIGHTENING_LIMIT_SOURCE_V2,
            "required_for_real_replay": trace_kind == "real",
            "gate_passed": hardware_target_gate_passed,
        }
        failure_reasons: list[str] = []
        if not action_array.shape[0]:
            failure_reasons.append("policy produced no inference outputs")
        if action_abs_max is not None and action_abs_max > 1.0 + 1.0e-6:
            failure_reasons.append("policy action exceeded the normalized [-1, 1] contract")
        if abad_action_abs_max is not None and abad_action_abs_max > 1.0e-6:
            failure_reasons.append("strict-forward policy produced a non-zero ABAD residual")
        if not saturation_gate_passed:
            failure_reasons.append(
                "main-action saturation fraction exceeded the configured replay limit"
            )
        if trace_kind == "real" and not hardware_target_gate_passed:
            failure_reasons.append(
                "deployment hardware target differs from the raw bundle-contract target"
            )
        if failure_reasons:
            summary["status"] = "failed"
            summary["failure_reasons"] = failure_reasons
    if reference_frames is not None:
        summary["domain_shift_ranked"] = _domain_shift(
            frame_array, reference_frames, contract
        )
    return output_arrays, summary


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"JSON root must be an object: {path}")
    return value


def _load_bundle(
    onnx_path: Path,
    sidecar_path: Path | None,
    *,
    require_hardware_ready: bool,
) -> tuple[PolicyRunnerV2, StudentObservationContractV2, ForwardResidualActionContractV2, SensorCalibrationProfileV2]:
    from redrhex_rl_controller.policy_onnx_runner_v2 import SensorPolicyONNXRunnerV2

    resolved_sidecar = sidecar_path or onnx_path.with_suffix(onnx_path.suffix + ".json")
    payload = _load_json(resolved_sidecar)
    try:
        contract = StudentObservationContractV2.from_dict(payload["contract"])
        action_contract = ForwardResidualActionContractV2.from_dict(payload["action_contract"])
        calibration = SensorCalibrationProfileV2.from_dict(payload["calibration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(f"invalid policy sidecar records: {exc}") from exc
    calibration.validate(require_hardware_ready=require_hardware_ready)
    kwargs: dict[str, Any] = {
        "sidecar_path": str(resolved_sidecar),
        "expected_contract_sha256": contract.sha256,
        "expected_action_contract_sha256": action_contract.sha256,
        "expected_calibration_sha256": calibration.sha256,
    }
    # Older development runners do not expose this switch. Their metadata and
    # calibration are still checked above before construction.
    try:
        runner = SensorPolicyONNXRunnerV2(
            str(onnx_path), require_hardware_ready=require_hardware_ready, **kwargs
        )
    except TypeError as exc:
        if "require_hardware_ready" not in str(exc):
            raise
        runner = SensorPolicyONNXRunnerV2(str(onnx_path), **kwargs)
    return runner, contract, action_contract, calibration


def _validate_real_replay_source(
    args: argparse.Namespace,
) -> ValidatedSensorV2ImportReceipt | None:
    if args.trace_kind != "real":
        return None
    if args.onnx is None or args.contract is not None:
        raise ContractError("real Sensor-V2 replay requires a hash-bound ONNX bundle")
    if args.import_receipt is None or args.import_receipt_sha256 is None:
        raise ContractError(
            "real Sensor-V2 replay requires --import-receipt and "
            "--import-receipt-sha256"
        )
    if args.max_period_error_ratio != MAX_PERIOD_ERROR_RATIO_V2:
        raise ContractError(
            "real Sensor-V2 replay cannot relax the importer's fixed cadence bound"
        )
    return validate_sensor_v2_import_receipt(
        args.import_receipt,
        expected_receipt_sha256=args.import_receipt_sha256,
        expected_trace_path=args.trace,
    )


def _artifact_record(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    return {"path": str(resolved), "sha256": sha256_path_v2(resolved)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="Canonical synchronized sensor trace (.npz).")
    parser.add_argument("--onnx", type=Path, default=None, help="Hash-bound Sensor V2 ONNX bundle.")
    parser.add_argument("--sidecar", type=Path, default=None)
    parser.add_argument("--contract", type=Path, default=None, help="Contract JSON for preprocessing-only replay.")
    parser.add_argument("--action-contract", type=Path, default=None)
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--trace-kind", choices=("sim", "real"), default="sim")
    parser.add_argument(
        "--import-receipt",
        type=Path,
        default=None,
        help="Receipt emitted by import_sensor_v2_rosbag.py; required for real replay.",
    )
    parser.add_argument(
        "--import-receipt-sha256",
        default=None,
        help="Expected SHA-256 of --import-receipt; required for real replay.",
    )
    parser.add_argument("--reference", type=Path, default=None, help="Prior replay NPZ with sensor_frames.")
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-period-error-ratio", type=float, default=0.25)
    parser.add_argument(
        "--max-main-action-saturation-fraction",
        type=float,
        default=DEFAULT_MAX_MAIN_ACTION_SATURATION_FRACTION,
        help=(
            "Maximum fraction of |main action| >= 0.999; the conservative "
            "interim default is 0.05 and is recorded with sensitivity values."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    deployment_hardware_config_path = (
        DEPLOYMENT_HARDWARE_CONFIG_PATH_V2.expanduser().resolve()
    )
    if (args.onnx is None) == (args.contract is None):
        raise SystemExit("exactly one of --onnx or --contract is required")
    real_source = _validate_real_replay_source(args)
    runner = None
    if args.onnx is not None:
        onnx_path = args.onnx.expanduser().resolve()
        sidecar_path = (
            args.sidecar.expanduser().resolve()
            if args.sidecar is not None
            else onnx_path.with_suffix(onnx_path.suffix + ".json")
        )
        runner, contract, action_contract, calibration = _load_bundle(
            onnx_path,
            sidecar_path,
            require_hardware_ready=args.trace_kind == "real",
        )
    else:
        onnx_path = sidecar_path = None
        contract = StudentObservationContractV2.from_dict(_load_json(args.contract))
        action_contract = (
            None
            if args.action_contract is None
            else ForwardResidualActionContractV2.from_dict(_load_json(args.action_contract))
        )
        calibration = (
            None
            if args.calibration is None
            else SensorCalibrationProfileV2.from_dict(_load_json(args.calibration))
        )
    if real_source is not None:
        assert calibration is not None
        if (
            real_source.payload["observation_contract_sha256"] != contract.sha256
            or real_source.payload["attitude_mode"] != contract.attitude_mode
        ):
            raise ContractError(
                "real replay import receipt does not match the ONNX observation contract"
            )
        attested_calibration_sha256 = real_source.capture_attestation.get(
            "runtime_calibration_sha256"
        )
        if attested_calibration_sha256 != calibration.sha256:
            raise ContractError(
                "real replay capture attestation runtime calibration does not "
                "match the ONNX bundle runtime calibration"
            )
    with np.load(args.trace, allow_pickle=False) as payload:
        trace = {name: payload[name] for name in payload.files}
    if real_source is not None:
        if int(np.asarray(trace.get("timestamp_s", [])).size) != int(
            real_source.payload["sample_count"]
        ):
            raise ContractError(
                "real replay trace sample count disagrees with its import receipt"
            )
        trace_imu_frame_id = _scalar_text(trace, "imu_frame_id")
        if (
            trace_imu_frame_id != real_source.payload["imu_frame_id"]
            or trace_imu_frame_id != contract.imu_frame_id
        ):
            raise ContractError(
                "real replay IMU frame disagrees with receipt or observation contract"
            )
    reference_frames = None
    if args.reference is not None:
        with np.load(args.reference, allow_pickle=False) as payload:
            if "sensor_frames" not in payload:
                raise ContractError("reference replay is missing sensor_frames")
            reference_frames = payload["sensor_frames"]
    outputs, summary = replay_arrays(
        trace,
        contract=contract,
        action_contract=action_contract,
        calibration=calibration,
        runner=runner,
        trace_kind=args.trace_kind,
        max_period_error_ratio=args.max_period_error_ratio,
        max_main_action_saturation_fraction=(
            args.max_main_action_saturation_fraction
        ),
        deployment_hardware_config_path=deployment_hardware_config_path,
        reference_frames=reference_frames,
    )
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    output_npz = args.output_npz.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()
    if output_npz.suffix != ".npz":
        raise ContractError("Sensor-V2 replay output must use the .npz suffix")
    input_trace_path = args.trace.expanduser().resolve()
    if (
        output_npz == output_json
        or output_npz == input_trace_path
        or output_json == input_trace_path
    ):
        raise ContractError("replay outputs must not overwrite the input trace")
    if real_source is not None:
        assert onnx_path is not None and sidecar_path is not None
        protected = {
            real_source.receipt_path,
            real_source.source_bag_path,
            real_source.capture_attestation_path,
            onnx_path,
            sidecar_path,
            deployment_hardware_config_path,
        }
        if output_npz in protected or output_json in protected:
            raise ContractError("real replay outputs must not overwrite source artifacts")
        if real_source.source_bag_path.is_dir() and (
            output_npz.is_relative_to(real_source.source_bag_path)
            or output_json.is_relative_to(real_source.source_bag_path)
        ):
            raise ContractError("real replay outputs must stay outside the source rosbag")
    np.savez_compressed(args.output_npz, **outputs)
    if real_source is not None:
        assert onnx_path is not None and sidecar_path is not None
        summary["source_artifacts"] = {
            "source_bag": {
                "path": str(real_source.source_bag_path),
                "sha256": real_source.source_bag_sha256,
            },
            "import_receipt": {
                "path": str(real_source.receipt_path),
                "sha256": real_source.receipt_sha256,
            },
            "capture_attestation": {
                "path": str(real_source.capture_attestation_path),
                "sha256": real_source.capture_attestation_sha256,
            },
            "input_trace": {
                "path": str(real_source.trace_path),
                "sha256": real_source.trace_sha256,
            },
            "onnx": _artifact_record(onnx_path),
            "sidecar": _artifact_record(sidecar_path),
            "hardware_config": _artifact_record(
                deployment_hardware_config_path
            ),
            "output_npz": _artifact_record(output_npz),
        }
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0 if summary["status"] == "passed" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
