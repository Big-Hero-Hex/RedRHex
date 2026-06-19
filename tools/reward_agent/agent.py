from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any

from tools.training_panel.training_panel.commands import TrainingParams

from .experiment_store import ExperimentStore


def _trial_record(candidate: dict[str, Any], run: dict[str, Any], params: TrainingParams) -> dict[str, Any]:
    return {
        "id": f"trial_{candidate['id']}",
        "candidate_id": candidate["id"],
        "panel_run_id": run.get("id"),
        "status": run.get("status", "unknown"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "params": params.to_dict(),
    }


def _select_candidates(candidates: list[dict], limit: int | None = None) -> list[dict]:
    if limit is None:
        return list(candidates)
    return list(candidates[: max(0, int(limit))])


def candidate_training_params(
    base_params: dict,
    candidate: dict,
    session_id: str,
    max_iterations: int | None = None,
) -> TrainingParams:
    params_data = deepcopy(base_params)
    if max_iterations is not None:
        params_data["max_iterations"] = max_iterations
    params_data["reward_preset_id"] = candidate["id"]
    params_data["reward_overrides"] = candidate.get("reward_overrides", {})
    params_data["client_request_id"] = f"{session_id}:{candidate['id']}"
    return TrainingParams.from_dict(params_data)


def preview_candidate_trials(
    store: ExperimentStore,
    session_id: str,
    base_params: dict,
    candidates: list[dict],
    max_iterations: int | None = None,
    limit: int | None = None,
) -> list[dict]:
    trials = store.load_trials(session_id)
    new_trials = []
    for candidate in _select_candidates(candidates, limit):
        params = candidate_training_params(base_params, candidate, session_id, max_iterations)
        record = _trial_record(candidate, {"id": None, "status": "dry_run"}, params)
        new_trials.append(record)
        trials.append(record)
    store.save_trials(session_id, trials)
    return new_trials


def queue_candidate_trials(
    store: ExperimentStore,
    session_id: str,
    process_registry: object,
    base_params: dict,
    candidates: list[dict],
    max_iterations: int | None = None,
    limit: int | None = None,
) -> list[dict]:
    trials = store.load_trials(session_id)
    new_trials = []
    for candidate in _select_candidates(candidates, limit):
        params = candidate_training_params(base_params, candidate, session_id, max_iterations)
        run = process_registry.queue_training(params)
        record = _trial_record(candidate, run, params)
        new_trials.append(record)
        trials.append(record)
    store.save_trials(session_id, trials)
    return new_trials
