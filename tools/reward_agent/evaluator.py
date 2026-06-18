from __future__ import annotations

from dataclasses import dataclass


REQUIRED_METRICS = (
    "command_tracking_score",
    "skill_pass_score",
    "stability_score",
    "energy_penalty",
    "fall_penalty",
)


@dataclass(frozen=True)
class EvaluationWeights:
    command_tracking_score: float = 1.0
    skill_pass_score: float = 1.0
    stability_score: float = 1.0
    energy_penalty: float = 1.0
    fall_penalty: float = 1.0
    regression_penalty: float = 1.0


def _metric(metrics: dict, key: str) -> float:
    try:
        return float(metrics.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def evaluate_metrics(
    metrics: dict,
    baseline: dict | None = None,
    weights: EvaluationWeights | None = None,
) -> dict:
    weights = weights or EvaluationWeights()
    missing = [key for key in REQUIRED_METRICS if key not in metrics]
    components = {
        "command_tracking_score": _metric(metrics, "command_tracking_score") * weights.command_tracking_score,
        "skill_pass_score": _metric(metrics, "skill_pass_score") * weights.skill_pass_score,
        "stability_score": _metric(metrics, "stability_score") * weights.stability_score,
        "energy_penalty": -_metric(metrics, "energy_penalty") * weights.energy_penalty,
        "fall_penalty": -_metric(metrics, "fall_penalty") * weights.fall_penalty,
    }
    base_score = sum(components.values())
    regression = 0.0
    if baseline and "overall_score" in baseline:
        baseline_score = _metric(baseline, "overall_score")
        if base_score < baseline_score:
            regression = (baseline_score - base_score) * weights.regression_penalty
    components["regression_penalty"] = -regression
    return {
        "complete": not missing,
        "missing_metrics": missing,
        "components": components,
        "overall_score": round(sum(components.values()), 6),
    }
