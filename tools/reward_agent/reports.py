from __future__ import annotations

from typing import Any


def _score(evaluation: dict[str, Any]) -> float:
    try:
        return float(evaluation.get("overall_score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def rank_evaluations(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        evaluations,
        key=lambda item: (bool(item.get("complete")), _score(item)),
        reverse=True,
    )


def build_comparison_report(
    session: dict[str, Any],
    trials: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
) -> dict[str, Any]:
    ranked = rank_evaluations(evaluations)
    best = ranked[0] if ranked else {}
    best_candidate_id = best.get("candidate_id")
    trial_by_candidate = {trial.get("candidate_id"): trial for trial in trials}
    best_trial = trial_by_candidate.get(best_candidate_id, {})
    objective = str((session.get("goal") or {}).get("objective") or "")
    return {
        "session_id": session.get("id"),
        "objective": objective,
        "best_candidate_id": best_candidate_id,
        "best_panel_run_id": best_trial.get("panel_run_id"),
        "ranked_candidates": ranked,
        "summary": f"Best candidate for {objective}: {best_candidate_id or 'none'}",
    }
