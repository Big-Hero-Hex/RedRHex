"""Allowlisted ROS deployment routes with an explicit contract boundary."""

from __future__ import annotations

from dataclasses import dataclass
LEGACY_CONTRACT_ID = "redrhex.policy-observation.v1"
SENSOR_ONLY_CONTRACT_ID_V2 = "redrhex.student-observation.v2"


@dataclass(frozen=True)
class DeploymentRoute:
    contract_id: str
    observation_builder: type
    onnx_runner: type
    preflight_module: str


def resolve_deployment_route(contract_id: str) -> DeploymentRoute:
    """Resolve only canonical IDs; aliases and version coercion are forbidden."""
    if contract_id == LEGACY_CONTRACT_ID:
        from .observation_builder import ObservationBuilder
        from .policy_onnx_runner import PolicyONNXRunner

        return DeploymentRoute(
            contract_id=contract_id,
            observation_builder=ObservationBuilder,
            onnx_runner=PolicyONNXRunner,
            preflight_module="redrhex_rl_controller.preflight_check",
        )
    if contract_id == SENSOR_ONLY_CONTRACT_ID_V2:
        from .observation_builder_v2 import SensorObservationBuilderV2
        from .policy_onnx_runner_v2 import SensorPolicyONNXRunnerV2

        return DeploymentRoute(
            contract_id=contract_id,
            observation_builder=SensorObservationBuilderV2,
            onnx_runner=SensorPolicyONNXRunnerV2,
            preflight_module="redrhex_rl_controller.preflight_check_v2",
        )
    raise ValueError(
        f"unsupported RedRHex policy contract ID {contract_id!r}; "
        f"allowed IDs are {LEGACY_CONTRACT_ID!r} and {SENSOR_ONLY_CONTRACT_ID_V2!r}"
    )
