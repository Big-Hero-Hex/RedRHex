"""Public API for the RedRHex sensor-only V2 policy I/O contract."""

from .action import DecodedForwardActionV2, decode_forward_residual_action_v2
from .contracts import (
    CalibrationRangeV2,
    ContractError,
    FeatureSpecV2,
    ForwardResidualActionContractV2,
    SensorCalibrationProfileV2,
    StudentObservationContractV2,
    canonical_json_bytes,
    canonical_sha256,
    validate_calibration_lineage_v2,
)
from .freshness import ChannelFreshnessReportV2, ChannelFreshnessTrackerV2
from .golden import generate_sensor_frame_golden_v2, load_sensor_frame_golden_v2
from .history import SensorHistoryBufferV2
from .preprocessing import (
    CausalGyroAccelAttitudeV2,
    SensorFrameBuilderV2,
    build_sensor_frame_numpy,
    projected_gravity_from_validated_quaternion,
    transform_imu_vector,
    wrap_angle,
    wrapped_velocity,
)


OBSERVATION_CONTRACT_ID_V2 = "redrhex.student-observation.v2"
ACTION_CONTRACT_ID_V2 = "redrhex.forward-residual-action.v2"
BUNDLE_CONTRACT_ID_V2 = "redrhex.sensor-policy-bundle.v2"


def build_sensor_frame_torch(*args, **kwargs):
    from .torch_utils import build_sensor_frame_torch as implementation

    return implementation(*args, **kwargs)


def decode_forward_residual_action_v2_torch(*args, **kwargs):
    from .torch_utils import decode_forward_residual_action_v2_torch as implementation

    return implementation(*args, **kwargs)


def projected_gravity_from_quaternion_torch(*args, **kwargs):
    from .torch_utils import projected_gravity_from_quaternion_torch as implementation

    return implementation(*args, **kwargs)


def transform_imu_vector_torch(*args, **kwargs):
    from .torch_utils import transform_imu_vector_torch as implementation

    return implementation(*args, **kwargs)


def wrap_angle_torch(*args, **kwargs):
    from .torch_utils import wrap_angle_torch as implementation

    return implementation(*args, **kwargs)


def wrapped_velocity_torch(*args, **kwargs):
    from .torch_utils import wrapped_velocity_torch as implementation

    return implementation(*args, **kwargs)


def __getattr__(name):
    if name == "DecodedForwardActionV2Torch":
        from .torch_utils import DecodedForwardActionV2Torch

        return DecodedForwardActionV2Torch
    if name == "BatchedCausalGyroAccelAttitudeV2":
        from .torch_utils import BatchedCausalGyroAccelAttitudeV2

        return BatchedCausalGyroAccelAttitudeV2
    raise AttributeError(name)


__all__ = [
    "ACTION_CONTRACT_ID_V2",
    "BUNDLE_CONTRACT_ID_V2",
    "OBSERVATION_CONTRACT_ID_V2",
    "BatchedCausalGyroAccelAttitudeV2",
    "CalibrationRangeV2",
    "CausalGyroAccelAttitudeV2",
    "ChannelFreshnessReportV2",
    "ChannelFreshnessTrackerV2",
    "ContractError",
    "DecodedForwardActionV2",
    "DecodedForwardActionV2Torch",
    "FeatureSpecV2",
    "ForwardResidualActionContractV2",
    "SensorCalibrationProfileV2",
    "SensorFrameBuilderV2",
    "SensorHistoryBufferV2",
    "StudentObservationContractV2",
    "build_sensor_frame_numpy",
    "build_sensor_frame_torch",
    "canonical_json_bytes",
    "canonical_sha256",
    "decode_forward_residual_action_v2",
    "decode_forward_residual_action_v2_torch",
    "generate_sensor_frame_golden_v2",
    "load_sensor_frame_golden_v2",
    "projected_gravity_from_quaternion_torch",
    "projected_gravity_from_validated_quaternion",
    "transform_imu_vector",
    "transform_imu_vector_torch",
    "validate_calibration_lineage_v2",
    "wrap_angle",
    "wrap_angle_torch",
    "wrapped_velocity",
    "wrapped_velocity_torch",
]
