"""Immutable, deterministic contracts for the RedRHex sensor-only V2 route."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping, Sequence


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ATTITUDE_MODES = {"validated_quaternion", "causal_gyro_accel"}


class ContractError(ValueError):
    """Raised when V2 policy I/O data violates its declared contract."""


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON data identically across simulation, replay, export, and ROS."""

    if hasattr(value, "to_dict"):
        value = value.to_dict()
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ContractError(f"value is not canonical JSON data: {exc}") from exc
    return text.encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _finite_tuple(value: Sequence[float], length: int, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or len(value) != length:
        raise ContractError(f"{name} must contain exactly {length} values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ContractError(f"{name} contains NaN or Inf")
    return result


def _six_strings(value: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or len(value) != 6:
        raise ContractError(f"{name} must contain exactly 6 strings")
    result = tuple(str(item) for item in value)
    return result


def _six_optional_positive(
    value: Sequence[float | None], name: str
) -> tuple[float | None, ...]:
    if isinstance(value, (str, bytes)) or len(value) != 6:
        raise ContractError(f"{name} must contain exactly 6 values")
    result: list[float | None] = []
    for item in value:
        if item is None:
            result.append(None)
            continue
        number = float(item)
        if not math.isfinite(number) or number <= 0.0:
            raise ContractError(f"{name} values must be positive and finite when set")
        result.append(number)
    return tuple(result)


def _verify_hash(value: str, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ContractError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _strict_keys(data: Mapping[str, Any], required: set[str], optional: set[str] = set()) -> None:
    missing = required - set(data)
    unknown = set(data) - required - optional
    if missing:
        raise ContractError(f"missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ContractError(f"unknown fields: {', '.join(sorted(unknown))}")


def _check_embedded_hash(data: dict[str, Any], expected: str, field_name: str = "sha256") -> None:
    supplied = data.pop(field_name, None)
    if supplied is not None and supplied != expected:
        raise ContractError(f"{field_name} does not match canonical payload")


@dataclass(frozen=True)
class FeatureSpecV2:
    name: str
    start: int
    stop: int
    unit: str
    normalization: str
    source: str
    actor_allowed: bool = True
    privileged_only: bool = False
    causal_filter: str = "none"

    @property
    def dimension(self) -> int:
        return self.stop - self.start

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "slice": [self.start, self.stop],
            "dimension": self.dimension,
            "unit": self.unit,
            "normalization": self.normalization,
            "source": self.source,
            "actor_allowed": self.actor_allowed,
            "privileged_only": self.privileged_only,
            "causal_filter": self.causal_filter,
        }


_FEATURE_LAYOUT_V2 = (
    FeatureSpecV2(
        "body_gyro",
        0,
        3,
        "rad_s",
        "bundle_featurewise_affine",
        "calibrated_imu",
        causal_filter="contract_gyro_filter",
    ),
    FeatureSpecV2(
        "projected_gravity",
        3,
        6,
        "unit_vector",
        "bundle_featurewise_affine",
        "selected_attitude_mode",
        causal_filter="contract_attitude_filter",
    ),
    FeatureSpecV2(
        "main_position_sin",
        6,
        12,
        "unitless",
        "bundle_featurewise_affine",
        "six_measured_continuous_encoders",
    ),
    FeatureSpecV2(
        "main_position_cos",
        12,
        18,
        "unitless",
        "bundle_featurewise_affine",
        "six_measured_continuous_encoders",
    ),
    FeatureSpecV2(
        "main_velocity",
        18,
        24,
        "rad_s",
        "bundle_featurewise_affine",
        "validated_velocity_or_main_position",
        causal_filter="validated_or_wrapped_finite_difference",
    ),
    FeatureSpecV2(
        "abad_position",
        24,
        30,
        "rad",
        "bundle_featurewise_affine",
        "six_measured_calibrated_abad_encoders",
    ),
    FeatureSpecV2(
        "abad_velocity",
        30,
        36,
        "rad_s",
        "bundle_featurewise_affine",
        "six_measured_calibrated_abad_encoders",
        causal_filter="finite_difference",
    ),
)


@dataclass(frozen=True)
class StudentObservationContractV2:
    """Fixed 36-D physical sensor frame and 60-frame causal history contract."""

    attitude_mode: str
    imu_frame_id: str = "redrhex_imu"
    policy_body_frame_id: str = "base_link"
    imu_to_body_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    rest_projected_gravity: tuple[float, float, float] = (0.0, 0.0, -1.0)
    attitude_parameters: tuple[tuple[str, float], ...] = ()
    contract_id: str = field(default="redrhex.student-observation.v2", init=False)
    version: int = field(default=2, init=False)

    SENSOR_FRAME_DIM: ClassVar[int] = 36
    HISTORY_LENGTH: ClassVar[int] = 60
    SAMPLE_RATE_HZ: ClassVar[float] = 60.0
    COMMAND_DIM: ClassVar[int] = 3
    HISTORY_ORDER: ClassVar[str] = "oldest_to_newest"
    FEATURE_LAYOUT: ClassVar[tuple[FeatureSpecV2, ...]] = _FEATURE_LAYOUT_V2

    def __post_init__(self) -> None:
        if self.attitude_mode not in _ATTITUDE_MODES:
            raise ContractError(
                "attitude_mode must be validated_quaternion or causal_gyro_accel"
            )
        if not self.imu_frame_id or not self.policy_body_frame_id:
            raise ContractError("IMU and policy body frame IDs must be non-empty")
        mount = _finite_tuple(self.imu_to_body_wxyz, 4, "imu_to_body_wxyz")
        norm = math.sqrt(sum(value * value for value in mount))
        if abs(norm - 1.0) > 1.0e-6:
            raise ContractError("imu_to_body_wxyz must be a unit quaternion")
        gravity = _finite_tuple(self.rest_projected_gravity, 3, "rest_projected_gravity")
        gravity_norm = math.sqrt(sum(value * value for value in gravity))
        if abs(gravity_norm - 1.0) > 1.0e-6:
            raise ContractError("rest_projected_gravity must be a unit vector")
        object.__setattr__(self, "imu_to_body_wxyz", mount)
        object.__setattr__(self, "rest_projected_gravity", gravity)

        parameters = self.attitude_parameters
        if not parameters:
            if self.attitude_mode == "validated_quaternion":
                parameters = (
                    ("max_orientation_variance", 0.05),
                    ("quaternion_norm_tolerance", 0.01),
                )
            else:
                parameters = (
                    ("accel_correction_gain", 0.02),
                    ("accel_magnitude_tolerance_ratio", 0.25),
                    ("gravity_magnitude_m_s2", 9.80665),
                )
        clean_parameters: list[tuple[str, float]] = []
        seen: set[str] = set()
        for name, value in parameters:
            if not isinstance(name, str) or not name or name in seen:
                raise ContractError("attitude parameter names must be non-empty and unique")
            number = float(value)
            if not math.isfinite(number):
                raise ContractError(f"attitude parameter {name} must be finite")
            seen.add(name)
            clean_parameters.append((name, number))
        object.__setattr__(self, "attitude_parameters", tuple(sorted(clean_parameters)))

    @classmethod
    def validated_quaternion(cls, **kwargs: Any) -> "StudentObservationContractV2":
        return cls(attitude_mode="validated_quaternion", **kwargs)

    @classmethod
    def causal_gyro_accel(cls, **kwargs: Any) -> "StudentObservationContractV2":
        return cls(attitude_mode="causal_gyro_accel", **kwargs)

    @property
    def sensor_frame_dim(self) -> int:
        return self.SENSOR_FRAME_DIM

    @property
    def history_length(self) -> int:
        return self.HISTORY_LENGTH

    @property
    def sample_rate_hz(self) -> float:
        return self.SAMPLE_RATE_HZ

    @property
    def command_dim(self) -> int:
        return self.COMMAND_DIM

    @property
    def history_order(self) -> str:
        return self.HISTORY_ORDER

    @property
    def feature_slices(self) -> dict[str, slice]:
        return {spec.name: slice(spec.start, spec.stop) for spec in self.FEATURE_LAYOUT}

    @property
    def parameter_map(self) -> dict[str, float]:
        return dict(self.attitude_parameters)

    def to_dict(self, *, include_sha256: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "contract_id": self.contract_id,
            "version": self.version,
            "sensor_frame_dim": self.sensor_frame_dim,
            "history_length": self.history_length,
            "history_duration_s": self.history_length / self.sample_rate_hz,
            "history_order": self.history_order,
            "sample_rate_hz": self.sample_rate_hz,
            "command": {
                "dimension": self.command_dim,
                "ordering": ["vx", "vy", "wz"],
                "unit": ["m_s", "m_s", "rad_s"],
                "external_input": True,
                "repeated_in_history": False,
            },
            "attitude_mode": self.attitude_mode,
            "attitude_parameters": dict(self.attitude_parameters),
            "imu_frame_id": self.imu_frame_id,
            "policy_body_frame_id": self.policy_body_frame_id,
            "imu_to_body_wxyz": list(self.imu_to_body_wxyz),
            "rest_projected_gravity": list(self.rest_projected_gravity),
            "feature_layout": [spec.to_dict() for spec in self.FEATURE_LAYOUT],
            "forbidden_actor_inputs": [
                "true_base_velocity",
                "odometry",
                "gait_clock",
                "previous_action",
                "commanded_abad",
                "internal_controller_targets",
                "linear_acceleration",
            ],
        }
        if include_sha256:
            result["sha256"] = canonical_sha256(result)
        return result

    def to_json(self, *, include_sha256: bool = False) -> str:
        return canonical_json_bytes(self.to_dict(include_sha256=include_sha256)).decode("utf-8")

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def validate(self) -> "StudentObservationContractV2":
        return self.from_dict(self.to_dict(include_sha256=True))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StudentObservationContractV2":
        data = dict(payload)
        supplied_hash = data.pop("sha256", None)
        required = {
            "contract_id",
            "version",
            "sensor_frame_dim",
            "history_length",
            "history_duration_s",
            "history_order",
            "sample_rate_hz",
            "command",
            "attitude_mode",
            "attitude_parameters",
            "imu_frame_id",
            "policy_body_frame_id",
            "imu_to_body_wxyz",
            "rest_projected_gravity",
            "feature_layout",
            "forbidden_actor_inputs",
        }
        _strict_keys(data, required)
        if data["contract_id"] != "redrhex.student-observation.v2" or data["version"] != 2:
            raise ContractError("unsupported student observation contract")
        expected_fixed = {
            "sensor_frame_dim": cls.SENSOR_FRAME_DIM,
            "history_length": cls.HISTORY_LENGTH,
            "history_duration_s": 1.0,
            "history_order": cls.HISTORY_ORDER,
            "sample_rate_hz": cls.SAMPLE_RATE_HZ,
        }
        for name, expected in expected_fixed.items():
            if data[name] != expected:
                raise ContractError(f"{name} must be {expected!r}")
        parameters = data["attitude_parameters"]
        if not isinstance(parameters, Mapping):
            raise ContractError("attitude_parameters must be an object")
        contract = cls(
            attitude_mode=str(data["attitude_mode"]),
            imu_frame_id=str(data["imu_frame_id"]),
            policy_body_frame_id=str(data["policy_body_frame_id"]),
            imu_to_body_wxyz=tuple(data["imu_to_body_wxyz"]),
            rest_projected_gravity=tuple(data["rest_projected_gravity"]),
            attitude_parameters=tuple((str(key), float(value)) for key, value in parameters.items()),
        )
        if data != contract.to_dict():
            raise ContractError("student observation payload changes the fixed V2 layout")
        if supplied_hash is not None and supplied_hash != contract.sha256:
            raise ContractError("sha256 does not match canonical student observation payload")
        return contract


@dataclass(frozen=True)
class ForwardResidualActionContractV2:
    """Canonical forward CPG plus six learned main-drive residuals."""

    main_residual_scale_rad_s: float = 0.8
    forward_command_reference_m_s: float = 0.45
    forward_bias_scale: float = 1.0
    phase_lock_gain: float = 1.2
    phase_correction_limit_rad_s: float = 2.0
    residual_cap_ratio: float = 0.26
    action_clip: float = 1.0
    main_velocity_limit_rad_s: float = 3.0 * math.pi
    abad_neutral_position_rad: tuple[float, ...] = (0.0,) * 6
    main_output_sign: tuple[float, ...] = (1.0,) * 6
    abad_output_sign: tuple[float, ...] = (1.0,) * 6
    contract_id: str = field(default="redrhex.forward-residual-action.v2", init=False)
    version: int = field(default=2, init=False)

    ACTION_DIM: ClassVar[int] = 12
    POLICY_RATE_HZ: ClassVar[float] = 60.0
    NOMINAL_GAIT_FREQUENCY_HZ: ClassVar[float] = 1.0
    STANCE_PHASE_START_RAD: ClassVar[float] = -math.pi / 6.0
    STANCE_PHASE_END_RAD: ClassVar[float] = math.pi / 6.0
    STANCE_VELOCITY_RATIO: ClassVar[float] = 0.15
    SWING_VELOCITY_RATIO: ClassVar[float] = 1.5
    LEG_ORDER: ClassVar[tuple[str, ...]] = (
        "right_front",
        "right_middle",
        "right_rear",
        "left_front",
        "left_middle",
        "left_rear",
    )
    MAIN_JOINT_ORDER: ClassVar[tuple[str, ...]] = (
        "Revolute_15",
        "Revolute_7",
        "Revolute_12",
        "Revolute_18",
        "Revolute_23",
        "Revolute_24",
    )
    ABAD_JOINT_ORDER: ClassVar[tuple[str, ...]] = (
        "Revolute_14",
        "Revolute_6",
        "Revolute_11",
        "Revolute_17",
        "Revolute_22",
        "Revolute_21",
    )
    TRIPOD_A: ClassVar[tuple[int, ...]] = (0, 3, 5)
    TRIPOD_B: ClassVar[tuple[int, ...]] = (1, 2, 4)
    TRIPOD_PHASE_OFFSET_RAD: ClassVar[float] = math.pi
    LEG_DIRECTION_MULTIPLIER: ClassVar[tuple[float, ...]] = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        for name in (
            "main_residual_scale_rad_s",
            "forward_command_reference_m_s",
            "forward_bias_scale",
            "phase_lock_gain",
            "phase_correction_limit_rad_s",
            "residual_cap_ratio",
            "action_clip",
            "main_velocity_limit_rad_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ContractError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        neutral = _finite_tuple(self.abad_neutral_position_rad, 6, "abad_neutral_position_rad")
        object.__setattr__(self, "abad_neutral_position_rad", neutral)
        for name in ("main_output_sign", "abad_output_sign"):
            signs = _finite_tuple(getattr(self, name), 6, name)
            if any(value not in (-1.0, 1.0) for value in signs):
                raise ContractError(f"{name} must contain only -1 or +1")
            object.__setattr__(self, name, signs)

    @property
    def action_dim(self) -> int:
        return self.ACTION_DIM

    @property
    def policy_rate_hz(self) -> float:
        return self.POLICY_RATE_HZ

    def to_dict(self, *, include_sha256: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "contract_id": self.contract_id,
            "version": self.version,
            "action_dim": self.action_dim,
            "action_ordering": [
                *(f"main_residual_{leg}" for leg in self.LEG_ORDER),
                *(f"abad_forced_neutral_{leg}" for leg in self.LEG_ORDER),
            ],
            "learned_action_slice": [0, 6],
            "forced_neutral_action_slice": [6, 12],
            "leg_order": list(self.LEG_ORDER),
            "main_joint_order": list(self.MAIN_JOINT_ORDER),
            "abad_joint_order": list(self.ABAD_JOINT_ORDER),
            "tripod_a_indices": list(self.TRIPOD_A),
            "tripod_b_indices": list(self.TRIPOD_B),
            "tripod_phase_offset_rad": self.TRIPOD_PHASE_OFFSET_RAD,
            "leg_direction_multiplier": list(self.LEG_DIRECTION_MULTIPLIER),
            "main_output_sign": list(self.main_output_sign),
            "abad_output_sign": list(self.abad_output_sign),
            "policy_rate_hz": self.policy_rate_hz,
            "procedural_gait": {
                "frequency_hz": self.NOMINAL_GAIT_FREQUENCY_HZ,
                "stance_phase_start_rad": self.STANCE_PHASE_START_RAD,
                "stance_phase_end_rad": self.STANCE_PHASE_END_RAD,
                "stance_velocity_ratio": self.STANCE_VELOCITY_RATIO,
                "swing_velocity_ratio": self.SWING_VELOCITY_RATIO,
                "phase_lock_gain": self.phase_lock_gain,
                "phase_correction_limit_rad_s": self.phase_correction_limit_rad_s,
                "forward_command_reference_m_s": self.forward_command_reference_m_s,
                "forward_bias_scale": self.forward_bias_scale,
            },
            "main_residual_scale_rad_s": self.main_residual_scale_rad_s,
            "residual_cap_ratio": self.residual_cap_ratio,
            "action_clip": self.action_clip,
            "main_velocity_limit_rad_s": self.main_velocity_limit_rad_s,
            "abad_neutral_position_rad": list(self.abad_neutral_position_rad),
        }
        if include_sha256:
            result["sha256"] = canonical_sha256(result)
        return result

    def to_json(self, *, include_sha256: bool = False) -> str:
        return canonical_json_bytes(self.to_dict(include_sha256=include_sha256)).decode("utf-8")

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    @property
    def decoder_sha256(self) -> str:
        return self.sha256

    def validate(self) -> "ForwardResidualActionContractV2":
        return self.from_dict(self.to_dict(include_sha256=True))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ForwardResidualActionContractV2":
        data = dict(payload)
        supplied_hash = data.pop("sha256", None)
        defaults = cls()
        fixed = defaults.to_dict()
        required = set(fixed)
        _strict_keys(data, required)
        gait = data.get("procedural_gait")
        if not isinstance(gait, Mapping):
            raise ContractError("procedural_gait must be an object")
        contract = cls(
            main_residual_scale_rad_s=float(data["main_residual_scale_rad_s"]),
            forward_command_reference_m_s=float(gait["forward_command_reference_m_s"]),
            forward_bias_scale=float(gait["forward_bias_scale"]),
            phase_lock_gain=float(gait["phase_lock_gain"]),
            phase_correction_limit_rad_s=float(gait["phase_correction_limit_rad_s"]),
            residual_cap_ratio=float(data["residual_cap_ratio"]),
            action_clip=float(data["action_clip"]),
            main_velocity_limit_rad_s=float(data["main_velocity_limit_rad_s"]),
            abad_neutral_position_rad=tuple(data["abad_neutral_position_rad"]),
            main_output_sign=tuple(data["main_output_sign"]),
            abad_output_sign=tuple(data["abad_output_sign"]),
        )
        if data != contract.to_dict():
            raise ContractError("action payload changes fixed V2 decoder semantics")
        if supplied_hash is not None and supplied_hash != contract.sha256:
            raise ContractError("sha256 does not match canonical action payload")
        return contract


@dataclass(frozen=True)
class CalibrationRangeV2:
    minimum: float
    maximum: float
    enabled: bool = False
    evidence: str = ""

    def __post_init__(self) -> None:
        minimum = float(self.minimum)
        maximum = float(self.maximum)
        if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum > maximum:
            raise ContractError("calibration range bounds must be finite and ordered")
        if self.enabled and not self.evidence.strip():
            raise ContractError("an enabled calibration range requires evidence provenance")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum": self.minimum,
            "maximum": self.maximum,
            "enabled": self.enabled,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CalibrationRangeV2":
        data = dict(payload)
        _strict_keys(data, {"minimum", "maximum", "enabled", "evidence"})
        if not isinstance(data["enabled"], bool) or not isinstance(data["evidence"], str):
            raise ContractError("calibration range enabled/evidence types are invalid")
        return cls(float(data["minimum"]), float(data["maximum"]), data["enabled"], data["evidence"])


@dataclass(frozen=True)
class SensorCalibrationProfileV2:
    """Calibration and evidence profile that never upgrades provisional data silently."""

    profile_id: str
    observation_contract_sha256: str
    action_contract_sha256: str
    attitude_mode: str
    imu_frame_id: str
    imu_to_body_wxyz: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0)
    main_encoder_sign: tuple[float, ...] = (1.0,) * 6
    main_encoder_zero_rad: tuple[float, ...] = (0.0,) * 6
    main_counts_per_rad: tuple[float | None, ...] = (None,) * 6
    abad_encoder_sign: tuple[float, ...] = (1.0,) * 6
    abad_encoder_zero_rad: tuple[float, ...] = (0.0,) * 6
    abad_counts_per_rad: tuple[float | None, ...] = (None,) * 6
    main_encoder_evidence: tuple[str, ...] = ("",) * 6
    abad_encoder_evidence: tuple[str, ...] = ("",) * 6
    imu_mount_evidence: str = ""
    rest_gravity_evidence: str = ""
    uncertainty_ranges: tuple[tuple[str, CalibrationRangeV2], ...] = ()
    schema_version: int = field(default=2, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise ContractError("profile_id must be a non-empty string")
        _verify_hash(self.observation_contract_sha256, "observation_contract_sha256")
        _verify_hash(self.action_contract_sha256, "action_contract_sha256")
        if self.attitude_mode not in _ATTITUDE_MODES:
            raise ContractError("unsupported attitude_mode")
        if not self.imu_frame_id:
            raise ContractError("imu_frame_id must be non-empty")
        mount = _finite_tuple(self.imu_to_body_wxyz, 4, "imu_to_body_wxyz")
        if abs(math.sqrt(sum(v * v for v in mount)) - 1.0) > 1.0e-6:
            raise ContractError("imu_to_body_wxyz must be a unit quaternion")
        object.__setattr__(self, "imu_to_body_wxyz", mount)
        for name in ("main_encoder_sign", "abad_encoder_sign"):
            signs = _finite_tuple(getattr(self, name), 6, name)
            if any(value not in (-1.0, 1.0) for value in signs):
                raise ContractError(f"{name} must contain only -1 or +1")
            object.__setattr__(self, name, signs)
        for name in ("main_encoder_zero_rad", "abad_encoder_zero_rad"):
            object.__setattr__(self, name, _finite_tuple(getattr(self, name), 6, name))
        for name in ("main_counts_per_rad", "abad_counts_per_rad"):
            object.__setattr__(self, name, _six_optional_positive(getattr(self, name), name))
        for name in ("main_encoder_evidence", "abad_encoder_evidence"):
            object.__setattr__(self, name, _six_strings(getattr(self, name), name))
        clean_ranges: list[tuple[str, CalibrationRangeV2]] = []
        seen: set[str] = set()
        for name, range_spec in self.uncertainty_ranges:
            if not isinstance(name, str) or not name or name in seen:
                raise ContractError("uncertainty range names must be non-empty and unique")
            if not isinstance(range_spec, CalibrationRangeV2):
                raise ContractError("uncertainty range values must be CalibrationRangeV2")
            seen.add(name)
            clean_ranges.append((name, range_spec))
        object.__setattr__(self, "uncertainty_ranges", tuple(sorted(clean_ranges)))

    @classmethod
    def provisional(
        cls,
        observation_contract: StudentObservationContractV2,
        action_contract: ForwardResidualActionContractV2,
        *,
        profile_id: str = "unverified-v2",
    ) -> "SensorCalibrationProfileV2":
        return cls(
            profile_id=profile_id,
            observation_contract_sha256=observation_contract.sha256,
            action_contract_sha256=action_contract.sha256,
            attitude_mode=observation_contract.attitude_mode,
            imu_frame_id=observation_contract.imu_frame_id,
            imu_to_body_wxyz=observation_contract.imu_to_body_wxyz,
        )

    @property
    def readiness_blockers(self) -> tuple[str, ...]:
        blockers: list[str] = []
        if not self.imu_mount_evidence.strip():
            blockers.append("imu_mount_evidence")
        if not self.rest_gravity_evidence.strip():
            blockers.append("rest_gravity_evidence")
        for prefix, evidence in (
            ("main_encoder", self.main_encoder_evidence),
            ("abad_encoder", self.abad_encoder_evidence),
        ):
            blockers.extend(f"{prefix}_{index}_evidence" for index, item in enumerate(evidence) if not item.strip())
        blockers.extend(
            f"abad_encoder_{index}_counts_per_rad"
            for index, item in enumerate(self.abad_counts_per_rad)
            if item is None
        )
        return tuple(blockers)

    @property
    def hardware_ready(self) -> bool:
        return not self.readiness_blockers

    @property
    def active_ranges(self) -> dict[str, CalibrationRangeV2]:
        return {name: value for name, value in self.uncertainty_ranges if value.enabled}

    def to_dict(self, *, include_sha256: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "observation_contract_sha256": self.observation_contract_sha256,
            "action_contract_sha256": self.action_contract_sha256,
            "attitude_mode": self.attitude_mode,
            "imu_frame_id": self.imu_frame_id,
            "imu_to_body_wxyz": list(self.imu_to_body_wxyz),
            "main_encoder_sign": list(self.main_encoder_sign),
            "main_encoder_zero_rad": list(self.main_encoder_zero_rad),
            "main_counts_per_rad": list(self.main_counts_per_rad),
            "abad_encoder_sign": list(self.abad_encoder_sign),
            "abad_encoder_zero_rad": list(self.abad_encoder_zero_rad),
            "abad_counts_per_rad": list(self.abad_counts_per_rad),
            "main_encoder_evidence": list(self.main_encoder_evidence),
            "abad_encoder_evidence": list(self.abad_encoder_evidence),
            "imu_mount_evidence": self.imu_mount_evidence,
            "rest_gravity_evidence": self.rest_gravity_evidence,
            "uncertainty_ranges": {name: value.to_dict() for name, value in self.uncertainty_ranges},
            "hardware_ready": self.hardware_ready,
            "readiness_blockers": list(self.readiness_blockers),
        }
        if include_sha256:
            result["sha256"] = canonical_sha256(result)
        return result

    def to_json(self, *, include_sha256: bool = False) -> str:
        return canonical_json_bytes(self.to_dict(include_sha256=include_sha256)).decode("utf-8")

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def validate(self, *, require_hardware_ready: bool = False) -> "SensorCalibrationProfileV2":
        result = self.from_dict(self.to_dict(include_sha256=True))
        if require_hardware_ready and not result.hardware_ready:
            raise ContractError(
                "hardware calibration is incomplete: " + ", ".join(result.readiness_blockers)
            )
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SensorCalibrationProfileV2":
        data = dict(payload)
        supplied_hash = data.pop("sha256", None)
        required = {
            "schema_version",
            "profile_id",
            "observation_contract_sha256",
            "action_contract_sha256",
            "attitude_mode",
            "imu_frame_id",
            "imu_to_body_wxyz",
            "main_encoder_sign",
            "main_encoder_zero_rad",
            "main_counts_per_rad",
            "abad_encoder_sign",
            "abad_encoder_zero_rad",
            "abad_counts_per_rad",
            "main_encoder_evidence",
            "abad_encoder_evidence",
            "imu_mount_evidence",
            "rest_gravity_evidence",
            "uncertainty_ranges",
            "hardware_ready",
            "readiness_blockers",
        }
        _strict_keys(data, required)
        if data["schema_version"] != 2:
            raise ContractError("schema_version must be 2")
        ranges = data["uncertainty_ranges"]
        if not isinstance(ranges, Mapping):
            raise ContractError("uncertainty_ranges must be an object")
        profile = cls(
            profile_id=str(data["profile_id"]),
            observation_contract_sha256=str(data["observation_contract_sha256"]),
            action_contract_sha256=str(data["action_contract_sha256"]),
            attitude_mode=str(data["attitude_mode"]),
            imu_frame_id=str(data["imu_frame_id"]),
            imu_to_body_wxyz=tuple(data["imu_to_body_wxyz"]),
            main_encoder_sign=tuple(data["main_encoder_sign"]),
            main_encoder_zero_rad=tuple(data["main_encoder_zero_rad"]),
            main_counts_per_rad=tuple(data["main_counts_per_rad"]),
            abad_encoder_sign=tuple(data["abad_encoder_sign"]),
            abad_encoder_zero_rad=tuple(data["abad_encoder_zero_rad"]),
            abad_counts_per_rad=tuple(data["abad_counts_per_rad"]),
            main_encoder_evidence=tuple(data["main_encoder_evidence"]),
            abad_encoder_evidence=tuple(data["abad_encoder_evidence"]),
            imu_mount_evidence=str(data["imu_mount_evidence"]),
            rest_gravity_evidence=str(data["rest_gravity_evidence"]),
            uncertainty_ranges=tuple(
                (str(name), CalibrationRangeV2.from_dict(value)) for name, value in ranges.items()
            ),
        )
        if data != profile.to_dict():
            raise ContractError("calibration derived readiness fields disagree with profile")
        if supplied_hash is not None and supplied_hash != profile.sha256:
            raise ContractError("sha256 does not match canonical calibration profile")
        return profile
