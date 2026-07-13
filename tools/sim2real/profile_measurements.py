from __future__ import annotations

import copy
import math
import re
from typing import Any, Mapping

from .contracts import CalibrationProfileV1, ContractError


_SHA256 = re.compile(r"[0-9a-f]{64}")
_ABAD_JOINT = re.compile(r"abad_[0-5]")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be an object")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{name} must be finite")
    return result


def _trace_sha(value: str | None, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ContractError(f"{name} trace SHA must be a lowercase SHA-256 digest")
    return value


def _abad_values(metrics: Mapping[str, Any]) -> tuple[str, float, float]:
    if metrics.get("schema_version") != 1 or metrics.get("metric_kind") != "abad_static_mapping":
        raise ContractError("ABAD metric contract is invalid")
    frame = metrics.get("frame")
    if not isinstance(frame, str) or not _ABAD_JOINT.fullmatch(frame):
        raise ContractError("ABAD metric frame must be a canonical abad_0..abad_5 joint")
    units = _mapping(metrics.get("units"), "ABAD metric units")
    if units.get("target_scale") != "1" or units.get("target_offset_rad") != "rad":
        raise ContractError("ABAD metric contract has incompatible units")
    aggregate = _mapping(metrics.get("aggregate"), "ABAD metric aggregate")
    scale = _number(aggregate.get("target_scale"), "ABAD target scale")
    offset = _number(aggregate.get("target_offset_rad"), "ABAD target offset")
    if scale <= 0.0:
        raise ContractError("ABAD target scale must be positive")
    return frame, scale, offset


def _friction_values(metrics: Mapping[str, Any]) -> tuple[float, float]:
    if metrics.get("schema_version") != 1 or metrics.get("metric_kind") != "ground_friction":
        raise ContractError("friction metric contract is invalid")
    units = _mapping(metrics.get("units"), "friction metric units")
    if units.get("coefficient") != "1":
        raise ContractError("friction metric coefficient must be dimensionless")
    static = _mapping(metrics.get("static"), "static friction metric")
    dynamic = _mapping(metrics.get("dynamic"), "dynamic friction metric")
    static_value = _number(static.get("coefficient_mean"), "static friction coefficient")
    dynamic_value = _number(dynamic.get("coefficient_mean"), "dynamic friction coefficient")
    if static_value < 0.0 or dynamic_value < 0.0:
        raise ContractError("friction coefficients must be non-negative")
    return static_value, dynamic_value


def apply_measurements_to_profile(
    baseline: CalibrationProfileV1,
    *,
    profile_id: str,
    abad_metrics: Mapping[str, Any] | None = None,
    abad_trace_sha256: str | None = None,
    friction_metrics: Mapping[str, Any] | None = None,
    friction_trace_sha256: str | None = None,
) -> CalibrationProfileV1:
    """Return a candidate profile populated only from versioned measurement results."""

    source = baseline.validate()
    if abad_metrics is None and friction_metrics is None:
        raise ContractError("at least one measurement result is required")
    if abad_metrics is None and abad_trace_sha256 is not None:
        raise ContractError("ABAD trace SHA was supplied without ABAD metrics")
    if friction_metrics is None and friction_trace_sha256 is not None:
        raise ContractError("friction trace SHA was supplied without friction metrics")

    payload = copy.deepcopy(source.to_dict())
    payload["profile_id"] = profile_id
    hardware = payload["hardware_mapping"]
    physics = payload["simulation_physics"]
    sources = payload["measurement_sources"]

    if abad_metrics is not None:
        trace_sha = _trace_sha(abad_trace_sha256, "ABAD")
        joint, scale, offset = _abad_values(_mapping(abad_metrics, "ABAD metrics"))
        hardware.setdefault("abad_target_scale", {})[joint] = scale
        hardware.setdefault("abad_target_offset_rad", {})[joint] = offset
        sources[f"abad_target:{joint}"] = trace_sha

    if friction_metrics is not None:
        trace_sha = _trace_sha(friction_trace_sha256, "friction")
        static, dynamic = _friction_values(
            _mapping(friction_metrics, "friction metrics")
        )
        ground = physics.setdefault("ground", {})
        ground["static_friction"] = static
        ground["dynamic_friction"] = dynamic
        sources["ground_friction"] = trace_sha

    return CalibrationProfileV1.from_dict(payload)
