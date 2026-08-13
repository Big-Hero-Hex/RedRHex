"""Causal NumPy preprocessing shared by V2 simulation, replay, and deployment."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .contracts import ContractError, StudentObservationContractV2


def _vector(value: Sequence[float] | np.ndarray, length: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,):
        raise ContractError(f"{name} must have shape ({length},), got {result.shape}")
    if not np.isfinite(result).all():
        raise ContractError(f"{name} contains NaN or Inf")
    return result


def wrap_angle(angle: float | Sequence[float] | np.ndarray) -> float | np.ndarray:
    """Wrap radians to the half-open interval [-pi, pi)."""

    values = np.asarray(angle)
    result = (values + math.pi) % (2.0 * math.pi) - math.pi
    if values.ndim == 0:
        return float(result)
    return result


def wrapped_velocity(
    current_position_rad: Sequence[float] | np.ndarray,
    previous_position_rad: Sequence[float] | np.ndarray,
    dt_s: float,
) -> np.ndarray:
    """Causal finite difference for periodic encoder angles."""

    current = np.asarray(current_position_rad, dtype=np.float64)
    previous = np.asarray(previous_position_rad, dtype=np.float64)
    if current.shape != previous.shape or current.ndim == 0:
        raise ContractError("current and previous positions must have the same non-scalar shape")
    if not np.isfinite(current).all() or not np.isfinite(previous).all():
        raise ContractError("encoder positions contain NaN or Inf")
    dt = float(dt_s)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ContractError("dt_s must be positive and finite")
    return np.asarray(wrap_angle(current - previous), dtype=np.float64) / dt


def _quat_rotate_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = _vector(quaternion, 4, "quaternion_wxyz")
    v = np.asarray(vector, dtype=np.float64)
    if v.shape[-1:] != (3,) or not np.isfinite(v).all():
        raise ContractError("vector must be finite with final dimension 3")
    q_vec = q[1:]
    uv = np.cross(q_vec, v)
    uuv = np.cross(q_vec, uv)
    return v + 2.0 * (q[0] * uv + uuv)


def transform_imu_vector(
    vector_imu: Sequence[float] | np.ndarray,
    imu_to_body_wxyz: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Rotate an IMU-frame vector into the calibrated policy body frame."""

    quaternion = _vector(imu_to_body_wxyz, 4, "imu_to_body_wxyz")
    norm = float(np.linalg.norm(quaternion))
    if abs(norm - 1.0) > 1.0e-6:
        raise ContractError("imu_to_body_wxyz must be a unit quaternion")
    return _quat_rotate_wxyz(quaternion, np.asarray(vector_imu, dtype=np.float64))


def projected_gravity_from_validated_quaternion(
    orientation_xyzw: Sequence[float] | np.ndarray,
    *,
    covariance: Sequence[float] | np.ndarray,
    frame_id: str,
    expected_frame_id: str,
    imu_to_body_wxyz: Sequence[float] | np.ndarray = (1.0, 0.0, 0.0, 0.0),
    rest_gravity_evidence_valid: bool,
    quaternion_norm_tolerance: float = 0.01,
    max_orientation_variance: float = 0.05,
) -> np.ndarray:
    """Validate an IMU orientation event and project world gravity into body.

    ``orientation_xyzw`` rotates IMU-frame vectors into the world frame.  The
    mount quaternion rotates IMU-frame vectors into the policy body frame.
    Unknown covariance, wrong frames, missing rest evidence, and bad norms are
    hard errors; callers must never fall back to another attitude mode.
    """

    if frame_id != expected_frame_id or not frame_id:
        raise ContractError(
            f"IMU frame mismatch: expected {expected_frame_id!r}, got {frame_id!r}"
        )
    if not rest_gravity_evidence_valid:
        raise ContractError("validated_quaternion requires recorded rest-gravity evidence")
    quaternion_xyzw = _vector(orientation_xyzw, 4, "orientation_xyzw")
    quaternion_wxyz = quaternion_xyzw[[3, 0, 1, 2]]
    norm = float(np.linalg.norm(quaternion_wxyz))
    if abs(norm - 1.0) > float(quaternion_norm_tolerance):
        raise ContractError("IMU orientation quaternion norm is outside contract tolerance")
    quaternion_wxyz /= norm

    covariance_array = np.asarray(covariance, dtype=np.float64)
    if covariance_array.shape == (9,):
        covariance_array = covariance_array.reshape(3, 3)
    if covariance_array.shape != (3, 3) or not np.isfinite(covariance_array).all():
        raise ContractError("orientation covariance must be a finite 3x3 matrix")
    diagonal = np.diag(covariance_array)
    if np.any(diagonal < 0.0) or not np.any(diagonal > 0.0):
        raise ContractError("orientation covariance is unknown or invalid")
    if np.any(diagonal > float(max_orientation_variance)):
        raise ContractError("orientation covariance exceeds contract limit")

    inverse = quaternion_wxyz.copy()
    inverse[1:] *= -1.0
    gravity_imu = _quat_rotate_wxyz(inverse, np.array([0.0, 0.0, -1.0]))
    gravity_body = transform_imu_vector(gravity_imu, imu_to_body_wxyz)
    return gravity_body / np.linalg.norm(gravity_body)


class CausalGyroAccelAttitudeV2:
    """Causal complementary gravity estimator with no magnetometer or fallback.

    State is the unit gravity direction expressed in the policy body frame.
    Gyro propagation is always causal.  Acceleration correction is applied only
    inside the contract magnitude gate; a dynamic sample outside that gate does
    not switch estimator modes.
    """

    def __init__(
        self,
        *,
        correction_gain: float = 0.02,
        gravity_magnitude_m_s2: float = 9.80665,
        accel_magnitude_tolerance_ratio: float = 0.25,
        initial_projected_gravity: Sequence[float] = (0.0, 0.0, -1.0),
    ) -> None:
        self.correction_gain = float(correction_gain)
        self.gravity_magnitude_m_s2 = float(gravity_magnitude_m_s2)
        self.accel_magnitude_tolerance_ratio = float(accel_magnitude_tolerance_ratio)
        if not 0.0 <= self.correction_gain <= 1.0:
            raise ContractError("correction_gain must be in [0, 1]")
        if self.gravity_magnitude_m_s2 <= 0.0:
            raise ContractError("gravity_magnitude_m_s2 must be positive")
        if self.accel_magnitude_tolerance_ratio < 0.0:
            raise ContractError("accel_magnitude_tolerance_ratio must be non-negative")
        self._initial = self._unit(initial_projected_gravity, "initial_projected_gravity")
        self._gravity = self._initial.copy()

    @staticmethod
    def _unit(value: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
        vector = _vector(value, 3, name)
        norm = float(np.linalg.norm(vector))
        if norm <= 1.0e-12:
            raise ContractError(f"{name} must be non-zero")
        return vector / norm

    @property
    def projected_gravity(self) -> np.ndarray:
        return self._gravity.copy()

    def reset(self, projected_gravity: Sequence[float] | None = None) -> None:
        self._gravity = (
            self._initial.copy()
            if projected_gravity is None
            else self._unit(projected_gravity, "projected_gravity")
        )

    def update(
        self,
        body_gyro_rad_s: Sequence[float] | np.ndarray,
        body_linear_accel_m_s2: Sequence[float] | np.ndarray,
        dt_s: float,
    ) -> np.ndarray:
        gyro = _vector(body_gyro_rad_s, 3, "body_gyro_rad_s")
        accel = _vector(body_linear_accel_m_s2, 3, "body_linear_accel_m_s2")
        dt = float(dt_s)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ContractError("dt_s must be positive and finite")
        predicted = self._gravity - np.cross(gyro, self._gravity) * dt
        predicted = self._unit(predicted, "propagated gravity")

        accel_norm = float(np.linalg.norm(accel))
        if accel_norm <= 1.0e-12:
            raise ContractError("body_linear_accel_m_s2 must be non-zero")
        relative_error = abs(accel_norm - self.gravity_magnitude_m_s2) / self.gravity_magnitude_m_s2
        if relative_error <= self.accel_magnitude_tolerance_ratio:
            measured_gravity = -accel / accel_norm
            predicted = (1.0 - self.correction_gain) * predicted + self.correction_gain * measured_gravity
            predicted = self._unit(predicted, "corrected gravity")
        self._gravity = predicted
        return self.projected_gravity


def build_sensor_frame_numpy(
    body_gyro_rad_s: Sequence[float] | np.ndarray,
    projected_gravity: Sequence[float] | np.ndarray,
    main_position_rad: Sequence[float] | np.ndarray,
    main_velocity_rad_s: Sequence[float] | np.ndarray,
    abad_position_rad: Sequence[float] | np.ndarray,
    abad_velocity_rad_s: Sequence[float] | np.ndarray,
    *,
    abad_neutral_position_rad: Sequence[float] | np.ndarray = (0.0,) * 6,
) -> np.ndarray:
    """Compose the fixed raw 36-D frame after causal sensor preprocessing."""

    gyro = _vector(body_gyro_rad_s, 3, "body_gyro_rad_s")
    gravity = _vector(projected_gravity, 3, "projected_gravity")
    gravity_norm = float(np.linalg.norm(gravity))
    if abs(gravity_norm - 1.0) > 1.0e-4:
        raise ContractError("projected_gravity must be a unit vector")
    main_position = _vector(main_position_rad, 6, "main_position_rad")
    main_velocity = _vector(main_velocity_rad_s, 6, "main_velocity_rad_s")
    abad_position = _vector(abad_position_rad, 6, "abad_position_rad")
    abad_velocity = _vector(abad_velocity_rad_s, 6, "abad_velocity_rad_s")
    abad_neutral = _vector(abad_neutral_position_rad, 6, "abad_neutral_position_rad")
    return np.concatenate(
        (
            gyro,
            gravity,
            np.sin(main_position),
            np.cos(main_position),
            main_velocity,
            abad_position - abad_neutral,
            abad_velocity,
        )
    ).astype(np.float32)


class SensorFrameBuilderV2:
    """Turn timestamped raw IMU/encoder events into one strict V2 sensor frame."""

    def __init__(
        self,
        contract: StudentObservationContractV2,
        *,
        abad_neutral_position_rad: Sequence[float] = (0.0,) * 6,
    ) -> None:
        if not isinstance(contract, StudentObservationContractV2):
            raise ContractError("contract must be StudentObservationContractV2")
        self.contract = contract.validate()
        self.abad_neutral_position_rad = _vector(
            abad_neutral_position_rad, 6, "abad_neutral_position_rad"
        )
        parameters = contract.parameter_map
        self._attitude = None
        if contract.attitude_mode == "causal_gyro_accel":
            self._attitude = CausalGyroAccelAttitudeV2(
                correction_gain=parameters["accel_correction_gain"],
                gravity_magnitude_m_s2=parameters["gravity_magnitude_m_s2"],
                accel_magnitude_tolerance_ratio=parameters[
                    "accel_magnitude_tolerance_ratio"
                ],
                initial_projected_gravity=contract.rest_projected_gravity,
            )
        self.reset()

    def reset(self) -> None:
        self._previous_timestamp_s: float | None = None
        self._previous_main_position_rad: np.ndarray | None = None
        self._previous_abad_position_rad: np.ndarray | None = None
        if self._attitude is not None:
            self._attitude.reset(self.contract.rest_projected_gravity)

    def build(
        self,
        *,
        timestamp_s: float,
        imu_gyro_rad_s: Sequence[float] | np.ndarray,
        main_position_rad: Sequence[float] | np.ndarray,
        abad_position_rad: Sequence[float] | np.ndarray,
        imu_linear_accel_m_s2: Sequence[float] | np.ndarray | None = None,
        imu_orientation_xyzw: Sequence[float] | np.ndarray | None = None,
        imu_orientation_covariance: Sequence[float] | np.ndarray | None = None,
        imu_frame_id: str | None = None,
        rest_gravity_evidence_valid: bool = False,
        main_velocity_rad_s: Sequence[float] | np.ndarray | None = None,
        main_velocity_valid: bool = False,
    ) -> np.ndarray:
        timestamp = float(timestamp_s)
        if not math.isfinite(timestamp):
            raise ContractError("timestamp_s must be finite")
        if self._previous_timestamp_s is not None and timestamp <= self._previous_timestamp_s:
            raise ContractError("sensor timestamps must be strictly increasing")
        main_position = _vector(main_position_rad, 6, "main_position_rad")
        abad_position = _vector(abad_position_rad, 6, "abad_position_rad")
        gyro_body = transform_imu_vector(
            imu_gyro_rad_s, self.contract.imu_to_body_wxyz
        )

        if self._previous_timestamp_s is None:
            dt = 1.0 / self.contract.sample_rate_hz
            derived_main_velocity = np.zeros(6, dtype=np.float64)
            abad_velocity = np.zeros(6, dtype=np.float64)
        else:
            dt = timestamp - self._previous_timestamp_s
            assert self._previous_main_position_rad is not None
            assert self._previous_abad_position_rad is not None
            derived_main_velocity = wrapped_velocity(
                main_position, self._previous_main_position_rad, dt
            )
            abad_velocity = wrapped_velocity(
                abad_position, self._previous_abad_position_rad, dt
            )

        if main_velocity_rad_s is None:
            main_velocity = derived_main_velocity
        else:
            if not main_velocity_valid:
                raise ContractError("provided main velocity must be explicitly validated")
            main_velocity = _vector(main_velocity_rad_s, 6, "main_velocity_rad_s")

        if self.contract.attitude_mode == "validated_quaternion":
            if imu_orientation_xyzw is None or imu_orientation_covariance is None or imu_frame_id is None:
                raise ContractError("validated_quaternion mode requires orientation, covariance, and frame ID")
            parameters = self.contract.parameter_map
            projected_gravity = projected_gravity_from_validated_quaternion(
                imu_orientation_xyzw,
                covariance=imu_orientation_covariance,
                frame_id=imu_frame_id,
                expected_frame_id=self.contract.imu_frame_id,
                imu_to_body_wxyz=self.contract.imu_to_body_wxyz,
                rest_gravity_evidence_valid=rest_gravity_evidence_valid,
                quaternion_norm_tolerance=parameters["quaternion_norm_tolerance"],
                max_orientation_variance=parameters["max_orientation_variance"],
            )
        else:
            if imu_linear_accel_m_s2 is None:
                raise ContractError("causal_gyro_accel mode requires linear acceleration")
            assert self._attitude is not None
            accel_body = transform_imu_vector(
                imu_linear_accel_m_s2, self.contract.imu_to_body_wxyz
            )
            projected_gravity = self._attitude.update(gyro_body, accel_body, dt)

        frame = build_sensor_frame_numpy(
            gyro_body,
            projected_gravity,
            main_position,
            main_velocity,
            abad_position,
            abad_velocity,
            abad_neutral_position_rad=self.abad_neutral_position_rad,
        )
        self._previous_timestamp_s = timestamp
        self._previous_main_position_rad = main_position.copy()
        self._previous_abad_position_rad = abad_position.copy()
        return frame
