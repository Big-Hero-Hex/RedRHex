"""Strict ROS sensor ingestion for Sensor-Only Distillation V2."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np

from redrhex_policy_io.contracts import ContractError, StudentObservationContractV2
from redrhex_policy_io.freshness import ChannelFreshnessTrackerV2
from redrhex_policy_io.history import SensorHistoryBufferV2
from redrhex_policy_io.preprocessing import SensorFrameBuilderV2, transform_imu_vector


SENSOR_FRAME_DIM_V2 = 36
HISTORY_LENGTH_V2 = 60
COMMAND_DIM_V2 = 3
SAMPLE_RATE_HZ_V2 = 60.0
MAIN_JOINT_NAMES_V2 = (
    "Revolute_15",
    "Revolute_7",
    "Revolute_12",
    "Revolute_18",
    "Revolute_23",
    "Revolute_24",
)
ABAD_JOINT_NAMES_V2 = (
    "Revolute_14",
    "Revolute_6",
    "Revolute_11",
    "Revolute_17",
    "Revolute_22",
    "Revolute_21",
)


@dataclass(frozen=True)
class SensorPolicyInputV2:
    sensor_history: np.ndarray
    command: np.ndarray


@dataclass(frozen=True)
class ObservationStatusV2:
    ok: bool
    history_ready: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)


def _source_stamp_s(msg: object, explicit: float | None = None) -> float:
    if explicit is not None:
        value = float(explicit)
    else:
        stamp = getattr(getattr(msg, "header", None), "stamp", None)
        value = float(getattr(stamp, "sec", 0)) + 1.0e-9 * float(getattr(stamp, "nanosec", 0))
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("V2 sensor messages require a positive finite source timestamp")
    return value


def _rotation_from_rpy_deg(rpy_deg: Iterable[float]) -> np.ndarray:
    values = [float(value) for value in rpy_deg]
    if len(values) != 3 or not np.isfinite(values).all():
        raise ValueError("imu_mount_rpy_deg must contain three finite values")
    roll, pitch, yaw = (math.radians(value) for value in values)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _quaternion_wxyz_from_rpy_deg(rpy_deg: Iterable[float]) -> tuple[float, float, float, float]:
    values = [float(value) for value in rpy_deg]
    if len(values) != 3 or not np.isfinite(values).all():
        raise ValueError("imu_mount_rpy_deg must contain three finite values")
    roll, pitch, yaw = (0.5 * math.radians(value) for value in values)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _quat_inverse_rotate_xyzw(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion
    matrix = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return matrix.T @ vector


def _wrapped_delta(current: float, previous: float) -> float:
    return math.atan2(math.sin(current - previous), math.cos(current - previous))


class SensorObservationBuilderV2:
    """Build raw physical sensor frames and a fully valid oldest-first history.

    Normalization deliberately does not happen here; it is embedded in the V2
    ONNX graph.  Every actor feature is derived from measured IMU or encoder
    events.  Odometry, command-derived ABAD state, gait phase, previous action,
    zero-padding, and runtime attitude fallback have no API in this class.
    """

    def __init__(
        self,
        config: dict | None = None,
        *,
        contract: StudentObservationContractV2 | None = None,
    ) -> None:
        cfg = config or {}
        self.main_joint_names = tuple(cfg.get("main_drive_joint_names", MAIN_JOINT_NAMES_V2))
        self.abad_joint_names = tuple(cfg.get("abad_joint_names", ABAD_JOINT_NAMES_V2))
        self.all_joint_names = self.main_joint_names + self.abad_joint_names
        attitude_mode = str(cfg.get("attitude_mode", ""))
        imu_frame_id = str(cfg.get("imu_frame_id", ""))
        policy_body_frame_id = str(cfg.get("policy_body_frame_id", "redrhex_policy_body"))
        self.imu_mount_calibration_verified = bool(cfg.get("imu_mount_calibration_verified", False))
        self.rest_gravity_verified = bool(cfg.get("rest_gravity_verified", False))
        self.quaternion_norm_tolerance = float(cfg.get("quaternion_norm_tolerance", 0.02))
        self.accel_correction_gain = float(cfg.get("causal_accel_correction_gain", 0.02))
        self.sensor_timeout_s = float(cfg.get("sensor_timeout_s", 0.10))
        self.command_timeout_s = float(cfg.get("command_timeout_s", 0.25))
        self.min_channel_period_s = float(cfg.get("min_channel_period_s", 0.001))
        self.history_length = int(cfg.get("history_length", HISTORY_LENGTH_V2))
        self.sample_rate_hz = float(cfg.get("sample_rate_hz", SAMPLE_RATE_HZ_V2))
        self.max_history_gap_s = float(cfg.get("max_history_gap_s", 2.5 / self.sample_rate_hz))
        self.require_joint_validity = bool(cfg.get("require_joint_validity", True))
        mount_rpy_deg = cfg.get("imu_mount_rpy_deg", [0.0, 0.0, 0.0])
        self._mount_rotation = _rotation_from_rpy_deg(mount_rpy_deg)
        expected_rest = np.asarray(
            cfg.get("expected_rest_projected_gravity", [0.0, 0.0, -1.0]), dtype=np.float64
        )
        if expected_rest.shape != (3,) or not np.isfinite(expected_rest).all() or np.linalg.norm(expected_rest) < 1e-9:
            raise ValueError("expected_rest_projected_gravity must be a finite non-zero 3-vector")
        self.expected_rest_projected_gravity = expected_rest / np.linalg.norm(expected_rest)
        if contract is None:
            if attitude_mode == "validated_quaternion":
                attitude_parameters = (
                    ("max_orientation_variance", float(cfg.get("max_orientation_variance", 0.05))),
                    ("quaternion_norm_tolerance", self.quaternion_norm_tolerance),
                )
            else:
                attitude_parameters = (
                    ("accel_correction_gain", self.accel_correction_gain),
                    (
                        "accel_magnitude_tolerance_ratio",
                        float(cfg.get("accel_magnitude_tolerance_ratio", 0.25)),
                    ),
                    ("gravity_magnitude_m_s2", float(cfg.get("gravity_magnitude_m_s2", 9.80665))),
                )
            contract = StudentObservationContractV2(
                attitude_mode=attitude_mode,
                imu_frame_id=imu_frame_id,
                policy_body_frame_id=policy_body_frame_id,
                imu_to_body_wxyz=_quaternion_wxyz_from_rpy_deg(mount_rpy_deg),
                rest_projected_gravity=tuple(float(value) for value in self.expected_rest_projected_gravity),
                attitude_parameters=attitude_parameters,
            )
        self.contract = contract.validate()
        if attitude_mode and attitude_mode != self.contract.attitude_mode:
            raise ValueError("configured attitude mode disagrees with bundle observation contract")
        if imu_frame_id and imu_frame_id != self.contract.imu_frame_id:
            raise ValueError("configured IMU frame disagrees with bundle observation contract")
        self.attitude_mode = self.contract.attitude_mode
        self.imu_frame_id = self.contract.imu_frame_id
        self.policy_body_frame_id = self.contract.policy_body_frame_id
        self.expected_rest_projected_gravity = np.asarray(
            self.contract.rest_projected_gravity, dtype=np.float64
        )
        self._mount_rotation = np.column_stack(
            [
                transform_imu_vector(axis, self.contract.imu_to_body_wxyz)
                for axis in np.eye(3, dtype=np.float64)
            ]
        )
        expected_contract_hash = cfg.get("contract_sha256")
        if expected_contract_hash is not None and str(expected_contract_hash) != self.contract.sha256:
            raise ValueError("configured observation contract hash does not match V2 contract")
        self._validate_config()

        self._frame_builder = SensorFrameBuilderV2(
            self.contract,
            abad_neutral_position_rad=cfg.get("abad_neutral_position_rad", [0.0] * 6),
        )
        self._history = SensorHistoryBufferV2(self.contract)
        required_freshness_channels = (
            "imu",
            *(f"joint:{name}" for name in self.all_joint_names),
            *(f"validity:{name}" for name in self.all_joint_names),
        )
        self._freshness = ChannelFreshnessTrackerV2(
            required_freshness_channels,
            max_age_s=self.sensor_timeout_s,
        )
        self._last_history_sample_time: float | None = None
        self._last_history_generation: tuple[float | None, tuple[float | None, ...]] | None = None
        self._invalid_channels: dict[str, str] = {}
        self._channel_times: dict[str, float] = {}
        self._joint_times: dict[str, float] = {}
        self._joint_validity: dict[str, bool] = {}
        self._joint_validity_times: dict[str, float] = {}
        self._joint_position: dict[str, float] = {}
        self._joint_velocity: dict[str, float] = {}
        self._previous_joint_position: dict[str, float] = {}
        self._previous_joint_time: dict[str, float] = {}
        self._body_gyro: np.ndarray | None = None
        self._projected_gravity: np.ndarray | None = None
        self._imu_gyro_raw: np.ndarray | None = None
        self._imu_accel_raw: np.ndarray | None = None
        self._imu_orientation_xyzw: np.ndarray | None = None
        self._imu_orientation_covariance: np.ndarray | None = None
        self._causal_gravity = self.expected_rest_projected_gravity.copy()
        self._causal_imu_time: float | None = None
        self._command = np.zeros(COMMAND_DIM_V2, dtype=np.float32)
        self._command_time: float | None = None

    def _validate_config(self) -> None:
        if len(self.main_joint_names) != 6 or len(self.abad_joint_names) != 6:
            raise ValueError("V2 requires six main and six ABAD joint names")
        if len(set(self.all_joint_names)) != 12:
            raise ValueError("V2 main/ABAD joint names must be unique")
        if self.attitude_mode not in ("validated_quaternion", "causal_gyro_accel"):
            raise ValueError("attitude_mode must be validated_quaternion or causal_gyro_accel")
        if not self.imu_frame_id:
            raise ValueError("V2 requires an explicit IMU frame ID")
        if not self.imu_mount_calibration_verified:
            raise ValueError("V2 requires reviewed IMU mount calibration")
        if self.attitude_mode == "validated_quaternion" and not self.rest_gravity_verified:
            raise ValueError("validated_quaternion requires recorded rest-gravity evidence")
        if self.history_length != HISTORY_LENGTH_V2 or self.sample_rate_hz != SAMPLE_RATE_HZ_V2:
            raise ValueError("V2 history is fixed at 60 frames sampled at 60 Hz")
        if not 0.0 <= self.accel_correction_gain <= 1.0:
            raise ValueError("causal_accel_correction_gain must be in [0, 1]")
        for name, value in (
            ("sensor_timeout_s", self.sensor_timeout_s),
            ("command_timeout_s", self.command_timeout_s),
            ("min_channel_period_s", self.min_channel_period_s),
            ("max_history_gap_s", self.max_history_gap_s),
            ("quaternion_norm_tolerance", self.quaternion_norm_tolerance),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")

    @property
    def history_ready(self) -> bool:
        return self._history.ready

    @property
    def history_size(self) -> int:
        return self._history.count

    @property
    def command(self) -> np.ndarray:
        return self._command.copy()

    def reset_history(self, reason: str | None = None) -> None:
        self._history.reset()
        self._frame_builder.reset()
        self._last_history_sample_time = None
        if reason:
            self._invalid_channels["history"] = str(reason)

    def _accept_source_event(self, channel: str, source_time_s: float) -> bool:
        previous = self._channel_times.get(channel)
        if previous is not None:
            dt = source_time_s - previous
            if dt <= 0.0:
                kind = "repeated" if dt == 0.0 else "out-of-order"
                self._invalid_channels[channel] = f"{kind} source timestamp"
                self.reset_history(f"{channel}: {kind} source timestamp")
                return False
            if dt < self.min_channel_period_s:
                self._invalid_channels[channel] = "source rate exceeds configured policy boundary"
                self.reset_history(f"{channel}: source rate exceeds configured policy boundary")
                return False
        self._channel_times[channel] = source_time_s
        if channel in self._freshness.required_channels:
            try:
                self._freshness.update(channel, source_time_s, valid=True)
            except ContractError:
                self._freshness.invalidate(channel)
                self._invalid_channels[channel] = "shared freshness tracker rejected timestamp"
                self.reset_history(self._invalid_channels[channel])
                return False
        self._invalid_channels.pop(channel, None)
        self._invalid_channels.pop("history", None)
        return True

    def update_imu(self, msg: object, source_time_s: float | None = None) -> bool:
        stamp = _source_stamp_s(msg, source_time_s)
        frame_id = str(getattr(getattr(msg, "header", None), "frame_id", ""))
        if frame_id != self.imu_frame_id:
            self._invalid_channels["imu"] = f"IMU frame {frame_id!r} != {self.imu_frame_id!r}"
            self.reset_history(self._invalid_channels["imu"])
            return False
        if not self._accept_source_event("imu", stamp):
            return False

        angular = getattr(msg, "angular_velocity")
        gyro_imu = np.asarray([angular.x, angular.y, angular.z], dtype=np.float64)
        if not np.isfinite(gyro_imu).all():
            self._invalid_channels["imu"] = "non-finite body gyro"
            self.reset_history(self._invalid_channels["imu"])
            return False
        gyro_body = self._mount_rotation @ gyro_imu

        if self.attitude_mode == "validated_quaternion":
            covariance = np.asarray(getattr(msg, "orientation_covariance", []), dtype=np.float64)
            orientation = getattr(msg, "orientation")
            quaternion = np.asarray(
                [orientation.x, orientation.y, orientation.z, orientation.w], dtype=np.float64
            )
            norm = float(np.linalg.norm(quaternion))
            covariance_ok = covariance.shape == (9,) and covariance[0] >= 0.0 and np.isfinite(covariance).all()
            if (
                not covariance_ok
                or not np.isfinite(quaternion).all()
                or abs(norm - 1.0) > self.quaternion_norm_tolerance
            ):
                self._invalid_channels["imu"] = "invalid quaternion covariance or norm"
                self.reset_history(self._invalid_channels["imu"])
                return False
            quaternion /= norm
            projected = self._mount_rotation @ _quat_inverse_rotate_xyzw(
                quaternion, np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
            )
        else:
            acceleration = getattr(msg, "linear_acceleration")
            accel_imu = np.asarray([acceleration.x, acceleration.y, acceleration.z], dtype=np.float64)
            accel_body = self._mount_rotation @ accel_imu
            accel_norm = float(np.linalg.norm(accel_body))
            if not np.isfinite(accel_body).all() or accel_norm < 1.0e-6:
                self._invalid_channels["imu"] = "invalid acceleration for causal attitude"
                self.reset_history(self._invalid_channels["imu"])
                return False
            if self._causal_imu_time is not None:
                dt = stamp - self._causal_imu_time
                propagated = self._causal_gravity - np.cross(gyro_body, self._causal_gravity) * dt
                propagated /= max(float(np.linalg.norm(propagated)), 1.0e-9)
            else:
                propagated = self._causal_gravity.copy()
            accel_gravity = -accel_body / accel_norm
            projected = (1.0 - self.accel_correction_gain) * propagated + self.accel_correction_gain * accel_gravity
            projected /= max(float(np.linalg.norm(projected)), 1.0e-9)
            self._causal_gravity = projected.copy()
            self._causal_imu_time = stamp

        if not np.isfinite(projected).all():
            self._invalid_channels["imu"] = "non-finite projected gravity"
            self.reset_history(self._invalid_channels["imu"])
            return False
        self._body_gyro = gyro_body
        self._projected_gravity = projected
        self._imu_gyro_raw = gyro_imu.copy()
        acceleration = getattr(msg, "linear_acceleration", None)
        self._imu_accel_raw = None if acceleration is None else np.asarray(
            [acceleration.x, acceleration.y, acceleration.z], dtype=np.float64
        )
        orientation = getattr(msg, "orientation", None)
        self._imu_orientation_xyzw = None if orientation is None else np.asarray(
            [orientation.x, orientation.y, orientation.z, orientation.w], dtype=np.float64
        )
        covariance = getattr(msg, "orientation_covariance", None)
        self._imu_orientation_covariance = None if covariance is None else np.asarray(
            covariance, dtype=np.float64
        )
        self._invalid_channels.pop("imu", None)
        return True

    def update_joint_state(self, msg: object, source_time_s: float | None = None) -> bool:
        stamp = _source_stamp_s(msg, source_time_s)
        names = [str(value) for value in getattr(msg, "name", [])]
        positions = list(getattr(msg, "position", []))
        velocities = list(getattr(msg, "velocity", []))
        if len(names) != len(set(names)):
            self._invalid_channels["joints"] = "duplicate names in joint state"
            self.reset_history(self._invalid_channels["joints"])
            return False
        updated = False
        for index, name in enumerate(names):
            if name not in self.all_joint_names:
                continue
            if index >= len(positions) or not math.isfinite(float(positions[index])):
                self._invalid_channels[name] = "missing or non-finite measured position"
                self.reset_history(f"{name}: {self._invalid_channels[name]}")
                continue
            if not self._accept_source_event(f"joint:{name}", stamp):
                continue
            position = float(positions[index])
            previous_position = self._joint_position.get(name)
            previous_time = self._joint_times.get(name)
            if index < len(velocities) and math.isfinite(float(velocities[index])):
                velocity = float(velocities[index])
            elif previous_position is not None and previous_time is not None and stamp > previous_time:
                delta = (
                    _wrapped_delta(position, previous_position)
                    if name in self.main_joint_names
                    else position - previous_position
                )
                velocity = delta / (stamp - previous_time)
            else:
                velocity = 0.0
            self._previous_joint_position[name] = self._joint_position.get(name, position)
            self._previous_joint_time[name] = self._joint_times.get(name, stamp)
            self._joint_position[name] = position
            self._joint_velocity[name] = velocity
            self._joint_times[name] = stamp
            self._invalid_channels.pop(name, None)
            updated = True
        if updated:
            self._invalid_channels.pop("joints", None)
        return updated

    def update_joint_validity(
        self,
        validity: Mapping[str, bool],
        source_time_s: float,
    ) -> bool:
        stamp = float(source_time_s)
        if not math.isfinite(stamp) or stamp <= 0.0:
            raise ValueError("joint validity requires a positive finite source timestamp")
        accepted = True
        for name in self.all_joint_names:
            previous = self._joint_validity_times.get(name)
            if previous is not None and stamp <= previous:
                kind = "repeated" if stamp == previous else "out-of-order"
                self._invalid_channels[f"validity:{name}"] = f"{kind} validity timestamp"
                accepted = False
                continue
            valid = bool(validity.get(name, False))
            self._joint_validity[name] = valid
            self._joint_validity_times[name] = stamp
            channel = f"validity:{name}"
            try:
                self._freshness.update(channel, stamp, valid=valid)
            except ContractError:
                self._freshness.invalidate(channel)
                self._invalid_channels[channel] = "shared freshness tracker rejected timestamp"
                accepted = False
                continue
            if not valid:
                self._invalid_channels[f"validity:{name}"] = "joint feedback invalid or unverified"
                accepted = False
            else:
                self._invalid_channels.pop(f"validity:{name}", None)
        if not accepted:
            self.reset_history("one or more joint validity gates failed")
        return accepted

    def update_joint_validity_diagnostic(
        self,
        msg: object,
        source_time_s: float | None = None,
    ) -> bool:
        """Consume the bridge's versioned per-channel DiagnosticArray."""
        stamp = _source_stamp_s(msg, source_time_s)
        validity: dict[str, bool] = {}
        for status in getattr(msg, "status", []):
            fields = {
                str(getattr(value, "key", "")): str(getattr(value, "value", ""))
                for value in getattr(status, "values", [])
            }
            name = fields.get("joint_name")
            if name not in self.all_joint_names:
                continue
            validity[name] = fields.get("valid", "").strip().lower() == "true"
        return self.update_joint_validity(validity, stamp)

    def update_command(self, msg: object, source_time_s: float | None = None) -> None:
        stamp = _source_stamp_s(msg, source_time_s)
        linear = getattr(msg, "linear")
        angular = getattr(msg, "angular")
        command = np.asarray([linear.x, linear.y, angular.z], dtype=np.float32)
        if not np.isfinite(command).all():
            raise ValueError("command contains NaN or Inf")
        if self._command_time is not None and stamp <= self._command_time:
            raise ValueError("command source timestamp is repeated or out-of-order")
        self._command = command
        self._command_time = stamp

    def status(self, now_s: float) -> ObservationStatusV2:
        reasons = list(self._invalid_channels.values())
        # ROS stamps are nanosecond-quantized while the control clock is a
        # float; tolerate only that representation-level skew.
        shared_report = self._freshness.report(now_s + 1.0e-6)
        reasons.extend(f"missing channel {name}" for name in shared_report.missing)
        reasons.extend(f"invalid channel {name}" for name in shared_report.invalid)
        reasons.extend(f"stale channel {name}" for name in shared_report.stale)
        if self._body_gyro is None or self._projected_gravity is None:
            reasons.append("waiting for a valid IMU event")
        imu_time = self._channel_times.get("imu")
        timestamp_tolerance_s = 1.0e-6
        if imu_time is not None and (
            imu_time - now_s > timestamp_tolerance_s or now_s - imu_time > self.sensor_timeout_s
        ):
            reasons.append("IMU stale or future-dated")
        for name in self.all_joint_names:
            stamp = self._joint_times.get(name)
            if stamp is None:
                reasons.append(f"missing measured joint {name}")
            elif stamp - now_s > timestamp_tolerance_s or now_s - stamp > self.sensor_timeout_s:
                reasons.append(f"joint {name} stale or future-dated")
            if self.require_joint_validity and not self._joint_validity.get(name, False):
                reasons.append(f"joint {name} validity missing")
            validity_stamp = self._joint_validity_times.get(name)
            if self.require_joint_validity and (
                validity_stamp is None
                or validity_stamp - now_s > timestamp_tolerance_s
                or now_s - validity_stamp > self.sensor_timeout_s
            ):
                reasons.append(f"joint {name} validity stale or future-dated")
        if reasons and self.history_size:
            self.reset_history("sensor dropout invalidated V2 history")
        return ObservationStatusV2(
            ok=not reasons,
            history_ready=self.history_ready,
            reasons=tuple(dict.fromkeys(reasons)),
        )

    def build_sensor_frame(self, now_s: float) -> np.ndarray:
        status = self.status(now_s)
        if not status.ok:
            raise RuntimeError("cannot build V2 sensor frame: " + "; ".join(status.reasons))
        main_pos = np.asarray([self._joint_position[name] for name in self.main_joint_names])
        abad_pos = np.asarray([self._joint_position[name] for name in self.abad_joint_names])
        assert self._imu_gyro_raw is not None
        frame = self._frame_builder.build(
            timestamp_s=float(now_s),
            imu_gyro_rad_s=self._imu_gyro_raw,
            main_position_rad=main_pos,
            abad_position_rad=abad_pos,
            imu_linear_accel_m_s2=self._imu_accel_raw,
            imu_orientation_xyzw=self._imu_orientation_xyzw,
            imu_orientation_covariance=self._imu_orientation_covariance,
            imu_frame_id=self.imu_frame_id,
            rest_gravity_evidence_valid=self.rest_gravity_verified,
        )
        self._body_gyro = frame[0:3].astype(np.float64)
        self._projected_gravity = frame[3:6].astype(np.float64)
        if frame.shape != (SENSOR_FRAME_DIM_V2,) or not np.isfinite(frame).all():
            raise RuntimeError(f"invalid V2 sensor frame shape/content: {frame.shape}")
        return frame

    def append_sensor_frame(self, now_s: float) -> np.ndarray:
        frame = self.build_sensor_frame(now_s)
        generation = (
            self._channel_times.get("imu"),
            tuple(self._joint_times.get(name) for name in self.all_joint_names),
        )
        if generation == self._last_history_generation:
            self.reset_history("repeated sensor generation at a V2 policy tick")
            raise RuntimeError("V2 rejects repeated sensor generations")
        if (
            self._last_history_sample_time is not None
            and now_s - self._last_history_sample_time > self.max_history_gap_s
        ):
            self.reset_history("history sampling gap exceeded")
        self._history.append(frame)
        self._last_history_sample_time = float(now_s)
        self._last_history_generation = generation
        return frame

    def policy_inputs(self, now_s: float) -> SensorPolicyInputV2:
        status = self.status(now_s)
        if not status.ok:
            raise RuntimeError("V2 sensor inputs are not valid: " + "; ".join(status.reasons))
        if not self.history_ready:
            raise RuntimeError(f"V2 history not ready: {self.history_size}/{self.history_length}")
        command = self._command.copy()
        if (
            self._command_time is None
            or self._command_time - now_s > 1.0e-6
            or now_s - self._command_time > self.command_timeout_s
        ):
            command[:] = 0.0
        history = self._history.array(require_ready=True)
        if history.shape != (HISTORY_LENGTH_V2, SENSOR_FRAME_DIM_V2):
            raise RuntimeError(f"V2 history shape is invalid: {history.shape}")
        return SensorPolicyInputV2(sensor_history=history, command=command)

    def projected_gravity_body(self) -> np.ndarray:
        if self._projected_gravity is None:
            raise RuntimeError("projected gravity is unavailable")
        return self._projected_gravity.copy()

    def get_main_drive_positions(self) -> np.ndarray:
        return np.asarray([self._joint_position.get(name, 0.0) for name in self.main_joint_names])

    def get_abad_positions(self) -> np.ndarray:
        return np.asarray([self._joint_position.get(name, 0.0) for name in self.abad_joint_names])
