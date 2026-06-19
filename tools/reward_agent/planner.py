from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RewardWeightSpec:
    name: str
    minimum: float
    maximum: float
    multipliers: tuple[float, ...] = (0.8, 1.2)
    group: str = "v2_reward_scales"


def _multiplier_token(multiplier: float) -> str:
    return str(multiplier).replace(".", "_").replace("-", "neg_")


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)


def _read_value(overrides: dict[str, Any], spec: RewardWeightSpec) -> float:
    if spec.group == "v2_reward_scales":
        return float(overrides.get("v2_reward_scales", {}).get(spec.name, 0.0))
    return float(overrides.get(spec.name, 0.0))


def _write_value(overrides: dict[str, Any], spec: RewardWeightSpec, value: float) -> None:
    if spec.group == "v2_reward_scales":
        overrides.setdefault("v2_reward_scales", {})
        overrides["v2_reward_scales"][spec.name] = value
        return
    overrides[spec.name] = value


def _change_key(spec: RewardWeightSpec) -> str:
    if spec.group == "v2_reward_scales":
        return f"v2_reward_scales.{spec.name}"
    return spec.name


def generate_weight_candidates(
    base_overrides: dict,
    specs: list[RewardWeightSpec],
    parent_candidate_id: str = "baseline",
) -> list[dict]:
    candidates: list[dict] = []
    sequence = 1
    for spec in specs:
        base_value = _read_value(base_overrides, spec)
        for multiplier in spec.multipliers:
            tuned_value = _clamp(base_value * float(multiplier), spec.minimum, spec.maximum)
            reward_overrides = deepcopy(base_overrides)
            _write_value(reward_overrides, spec, tuned_value)
            change_key = _change_key(spec)
            candidates.append(
                {
                    "id": f"cand-{sequence:03d}-{spec.name}-x{_multiplier_token(float(multiplier))}",
                    "parent_candidate_id": parent_candidate_id,
                    "reward_overrides": reward_overrides,
                    "changed": {
                        change_key: {
                            "from": base_value,
                            "to": tuned_value,
                            "multiplier": float(multiplier),
                        }
                    },
                    "hypothesis": f"Adjust {change_key} by x{float(multiplier):g} to test reward sensitivity.",
                    "risk_notes": [],
                }
            )
            sequence += 1
    return candidates
