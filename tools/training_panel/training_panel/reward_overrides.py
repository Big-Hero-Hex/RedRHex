from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any


class RewardOverrideError(ValueError):
    """Raised when reward override payloads are malformed."""


def _to_float(key: str, value: Any) -> float:
    if isinstance(value, bool):
        raise RewardOverrideError(f"Reward override {key!r} must be numeric, not boolean")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RewardOverrideError(f"Reward override {key!r} must be numeric") from exc
    if not math.isfinite(parsed):
        raise RewardOverrideError(f"Reward override {key!r} must be finite")
    return parsed


def normalize_reward_overrides(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for raw_key, raw_value in (overrides or {}).items():
        key = str(raw_key)
        if key == "v2_reward_scales":
            if not isinstance(raw_value, Mapping):
                raise RewardOverrideError("v2_reward_scales override must be an object")
            normalized[key] = {
                str(scale_key): _to_float(f"v2_reward_scales.{scale_key}", scale_value)
                for scale_key, scale_value in raw_value.items()
            }
            continue
        normalized[key] = _to_float(key, raw_value)
    return normalized


def _iter_overrides(overrides: Mapping[str, Any]) -> list[tuple[str, float]]:
    normalized = normalize_reward_overrides(overrides)
    flattened: list[tuple[str, float]] = []
    for key, value in normalized.items():
        if key == "v2_reward_scales":
            for nested_key, nested_value in value.items():
                flattened.append((f"v2_reward_scales.{nested_key}", nested_value))
        else:
            flattened.append((key, value))
    return flattened


def apply_reward_overrides(
    env_cfg: object,
    overrides: Mapping[str, Any] | None,
    *,
    require_all: bool = False,
) -> list[str]:
    applied: list[str] = []
    unknown: list[str] = []
    requested = _iter_overrides(overrides or {})
    for key, value in requested:
        if key.startswith("v2_reward_scales."):
            scale_name = key.split(".", 1)[1]
            current = getattr(env_cfg, "v2_reward_scales", None)
            if not isinstance(current, Mapping) or scale_name not in current:
                unknown.append(key)
                continue
            merged = dict(current)
            merged[scale_name] = value
            setattr(env_cfg, "v2_reward_scales", merged)
            applied.append(f"{key}={value}")
            continue
        if hasattr(env_cfg, key):
            setattr(env_cfg, key, value)
            applied.append(f"{key}={value}")
        else:
            unknown.append(key)
    if require_all and unknown:
        raise RewardOverrideError(
            "Unknown or unavailable reward override keys: " + ", ".join(sorted(unknown))
        )
    if require_all:
        mismatched: list[str] = []
        for key, requested_value in requested:
            if key.startswith("v2_reward_scales."):
                scale_name = key.split(".", 1)[1]
                current = getattr(env_cfg, "v2_reward_scales", None)
                resolved = current.get(scale_name) if isinstance(current, Mapping) else None
            else:
                resolved = getattr(env_cfg, key, None)
            if isinstance(resolved, bool):
                mismatched.append(key)
                continue
            try:
                resolved_value = float(resolved)
            except (TypeError, ValueError):
                mismatched.append(key)
                continue
            if not math.isfinite(resolved_value) or resolved_value != requested_value:
                mismatched.append(key)
        if mismatched:
            raise RewardOverrideError(
                "Reward overrides did not resolve exactly: " + ", ".join(sorted(mismatched))
            )
    return applied
