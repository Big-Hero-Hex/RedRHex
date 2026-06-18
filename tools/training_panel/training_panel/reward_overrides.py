from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class RewardOverrideError(ValueError):
    """Raised when reward override payloads are malformed."""


def _to_float(key: str, value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise RewardOverrideError(f"Reward override {key!r} must be numeric") from exc


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


def apply_reward_overrides(env_cfg: object, overrides: Mapping[str, Any] | None) -> list[str]:
    applied: list[str] = []
    for key, value in _iter_overrides(overrides or {}):
        if key.startswith("v2_reward_scales."):
            scale_name = key.split(".", 1)[1]
            current = getattr(env_cfg, "v2_reward_scales", None)
            if not isinstance(current, Mapping):
                continue
            merged = dict(current)
            merged[scale_name] = value
            setattr(env_cfg, "v2_reward_scales", merged)
            applied.append(f"{key}={value}")
            continue
        if hasattr(env_cfg, key):
            setattr(env_cfg, key, value)
            applied.append(f"{key}={value}")
    return applied
