from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


SESSION_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _timestamp_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _safe_session_id(value: str) -> str:
    return SESSION_ID_RE.sub("_", value).strip("._") or f"session_{_timestamp_id()}"


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


@dataclass(frozen=True)
class RewardAgentPaths:
    root: Path

    @classmethod
    def from_repo_root(cls, repo_root: Path) -> "RewardAgentPaths":
        return cls(Path(repo_root) / "logs" / "reward_agent")

    @property
    def sessions_file(self) -> Path:
        return self.root / "sessions.json"

    def session_dir(self, session_id: str) -> Path:
        return self.root / "sessions" / _safe_session_id(session_id)


class ExperimentStore:
    def __init__(self, paths: RewardAgentPaths):
        self.paths = paths

    def list_sessions(self) -> list[dict[str, Any]]:
        data = _read_json(self.paths.sessions_file, {"sessions": []})
        sessions = data.get("sessions") if isinstance(data, dict) else []
        return list(sessions) if isinstance(sessions, list) else []

    def create_session(self, goal: dict[str, Any]) -> dict[str, Any]:
        now = datetime.now().isoformat(timespec="seconds")
        session = {
            "id": f"session_{_timestamp_id()}",
            "created_at": now,
            "updated_at": now,
            "status": "idle",
            "goal": dict(goal),
        }
        session_dir = self.paths.session_dir(session["id"])
        _write_json(session_dir / "goal.json", session)
        _write_json(self.paths.sessions_file, {"sessions": [*self.list_sessions(), session]})
        return session

    def append_conversation(self, session_id: str, entry: dict[str, Any]) -> None:
        path = self.paths.session_dir(session_id) / "conversation.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"created_at": datetime.now().isoformat(timespec="seconds"), **entry}
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, sort_keys=True) + "\n")

    def save_trials(self, session_id: str, trials: list[dict[str, Any]]) -> None:
        _write_json(self.paths.session_dir(session_id) / "trials.json", {"trials": trials})

    def load_trials(self, session_id: str) -> list[dict[str, Any]]:
        data = _read_json(self.paths.session_dir(session_id) / "trials.json", {"trials": []})
        trials = data.get("trials") if isinstance(data, dict) else []
        return list(trials) if isinstance(trials, list) else []

    def save_candidates(self, session_id: str, candidates: list[dict[str, Any]]) -> None:
        _write_json(self.paths.session_dir(session_id) / "candidates.json", {"candidates": candidates})

    def load_candidates(self, session_id: str) -> list[dict[str, Any]]:
        data = _read_json(self.paths.session_dir(session_id) / "candidates.json", {"candidates": []})
        candidates = data.get("candidates") if isinstance(data, dict) else []
        return list(candidates) if isinstance(candidates, list) else []

    def save_evaluations(self, session_id: str, evaluations: list[dict[str, Any]]) -> None:
        _write_json(self.paths.session_dir(session_id) / "evaluations.json", {"evaluations": evaluations})

    def load_evaluations(self, session_id: str) -> list[dict[str, Any]]:
        data = _read_json(self.paths.session_dir(session_id) / "evaluations.json", {"evaluations": []})
        evaluations = data.get("evaluations") if isinstance(data, dict) else []
        return list(evaluations) if isinstance(evaluations, list) else []

    def save_report(self, session_id: str, report_name: str, report: dict[str, Any]) -> Path:
        safe_name = _safe_session_id(report_name)
        path = self.paths.session_dir(session_id) / "reports" / f"{safe_name}.json"
        _write_json(path, report)
        return path
