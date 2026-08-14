from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from .autopilot import (
    CAMPAIGN_SCHEMA_VERSION,
    HOST_SLOT_STATES,
    MAX_CONNECTOR_POLLS,
    TERMINAL_STATES,
    AgentDecisionV1,
    AutopilotValidationError,
    CampaignSnapshotV1,
    EvaluationReportV1,
    GoalSpecV1,
    RewardCatalogEntryV1,
    SHA256_RE,
    evaluation_rank_key,
    next_permitted_actions,
    reward_lattice_values,
    reward_move_lattice,
    sha256_json,
    validate_candidate_decision,
    validate_transition,
)


STORE_SCHEMA_VERSION = 1
GPU_ACCOUNTING_INTERVAL_HOURS = 60.0 / 3600.0


class AutopilotStoreError(RuntimeError):
    code = "autopilot_store_error"


class CampaignNotFoundError(AutopilotStoreError):
    code = "campaign_not_found"


class AutopilotConflictError(AutopilotStoreError):
    code = "campaign_conflict"

    def __init__(self, message: str, *, current_revision: int | None = None, prior_result: Any = None):
        super().__init__(message)
        self.current_revision = current_revision
        self.prior_result = prior_result


class AutopilotBudgetError(AutopilotStoreError):
    code = "campaign_budget_exhausted"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _load(value: str | None, default: Any) -> Any:
    if value is None:
        return default
    return json.loads(value)


def _identifier(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _apply_cumulative_gpu_usage(
    metadata: Mapping[str, Any],
    budget: Mapping[str, Any],
    *,
    process_id: str,
    process_kind: str,
    cumulative_gpu_hours: float,
) -> tuple[dict[str, Any], dict[str, Any], float, bool, bool]:
    """Apply one process' cumulative runtime exactly once.

    The per-process high-water mark lives with the durable trial so a panel
    restart, a repeated controller tick, or a completion callback can only
    charge the unaccounted delta.
    """

    if process_kind not in {"training", "evaluation"}:
        raise ValueError("process_kind must be training or evaluation")
    if not isinstance(process_id, str) or not process_id.strip():
        raise ValueError("process_id must be non-empty")
    if (
        isinstance(cumulative_gpu_hours, bool)
        or not isinstance(cumulative_gpu_hours, (int, float))
        or not math.isfinite(float(cumulative_gpu_hours))
        or float(cumulative_gpu_hours) < 0.0
    ):
        raise ValueError("cumulative_gpu_hours must be a finite non-negative number")

    updated_metadata = dict(metadata)
    accounting = dict(updated_metadata.get("gpu_process_accounting") or {})
    prior_entry = accounting.get(process_id)
    if prior_entry is not None and not isinstance(prior_entry, Mapping):
        raise AutopilotStoreError("trial GPU accounting marker is malformed")
    prior = dict(prior_entry or {})
    if prior and prior.get("kind") != process_kind:
        raise AutopilotStoreError("process GPU accounting kind changed")
    previous_hours = float(prior.get("accounted_gpu_hours", 0.0))
    if not math.isfinite(previous_hours) or previous_hours < 0.0:
        raise AutopilotStoreError("trial GPU accounting marker is non-finite")

    # Completion timestamps have second precision while live accounting uses
    # the current clock. Never refund a fraction of a second at completion.
    target_hours = max(previous_hours, float(cumulative_gpu_hours))
    delta = target_hours - previous_hours
    entry = {
        "kind": process_kind,
        "accounted_gpu_hours": target_hours,
    }
    marker_changed = prior != entry
    accounting[process_id] = entry
    updated_metadata["gpu_process_accounting"] = accounting

    updated_budget = dict(budget)
    used = float(updated_budget.get("used_gpu_hours", 0.0)) + delta
    maximum = float(updated_budget.get("max_gpu_hours", 0.0))
    if not math.isfinite(used) or not math.isfinite(maximum) or maximum <= 0.0:
        raise AutopilotStoreError("campaign GPU-hour budget is invalid")
    updated_budget["used_gpu_hours"] = used
    updated_budget["remaining_gpu_hours"] = max(0.0, maximum - used)
    return updated_metadata, updated_budget, delta, used >= maximum, marker_changed


def _consume_connector_poll(budget: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    updated = dict(budget)
    polls = int(updated.get("connector_polls", 0))
    maximum = int(updated.get("max_connector_polls", MAX_CONNECTOR_POLLS))
    if polls < 0 or maximum <= 0 or polls > maximum:
        raise AutopilotStoreError("connector poll budget is invalid")
    if polls >= maximum:
        raise AutopilotBudgetError("connector poll budget is exhausted")
    updated["connector_polls"] = polls + 1
    return updated, polls + 1 >= maximum


class AutopilotStore:
    """Transactional campaign metadata, event, idempotency, and artifact store."""

    def __init__(self, database_file: Path, artifact_dir: Path, *, enabled: bool = False):
        self.database_file = Path(database_file)
        self.artifact_dir = Path(artifact_dir)
        self.enabled = bool(enabled)
        self._lock = threading.RLock()
        self.database_file.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_file, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()

    def _initialize(self) -> None:
        with self._transaction() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS autopilot_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS campaigns (
                    id TEXT PRIMARY KEY,
                    schema_version TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    goal_json TEXT NOT NULL,
                    reward_catalog_json TEXT NOT NULL,
                    leader_json TEXT NOT NULL,
                    budget_json TEXT NOT NULL,
                    active_process_json TEXT,
                    connector_json TEXT NOT NULL,
                    runtime_json TEXT NOT NULL,
                    resume_state TEXT,
                    terminal_reason TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS one_campaign_host_slot
                    ON campaigns ((1))
                    WHERE state IN (
                        'armed', 'control_training', 'control_evaluating', 'awaiting_advisor',
                        'candidate_training', 'candidate_evaluating', 'confirming', 'paused',
                        'waiting_for_chatgpt'
                    );
                CREATE TABLE IF NOT EXISTS campaign_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    campaign_id TEXT NOT NULL REFERENCES campaigns(id),
                    revision INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS campaign_events_lookup
                    ON campaign_events(campaign_id, sequence);
                CREATE TRIGGER IF NOT EXISTS campaign_events_no_update
                    BEFORE UPDATE ON campaign_events
                    BEGIN
                        SELECT RAISE(ABORT, 'campaign events are append-only');
                    END;
                CREATE TRIGGER IF NOT EXISTS campaign_events_no_delete
                    BEFORE DELETE ON campaign_events
                    BEGIN
                        SELECT RAISE(ABORT, 'campaign events are append-only');
                    END;
                CREATE TABLE IF NOT EXISTS campaign_decisions (
                    id TEXT PRIMARY KEY,
                    campaign_id TEXT NOT NULL REFERENCES campaigns(id),
                    revision INTEGER NOT NULL,
                    decision_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS campaign_trials (
                    id TEXT PRIMARY KEY,
                    campaign_id TEXT NOT NULL REFERENCES campaigns(id),
                    kind TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    reward_profile_json TEXT NOT NULL,
                    reward_profile_sha256 TEXT NOT NULL,
                    source_checkpoint_sha256 TEXT,
                    output_checkpoint_sha256 TEXT,
                    run_id TEXT,
                    evaluation_process_id TEXT,
                    evaluation_id TEXT,
                    status TEXT NOT NULL,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS campaign_trials_lookup
                    ON campaign_trials(campaign_id, created_at);
                CREATE TABLE IF NOT EXISTS campaign_evaluations (
                    id TEXT PRIMARY KEY,
                    campaign_id TEXT NOT NULL REFERENCES campaigns(id),
                    trial_id TEXT NOT NULL REFERENCES campaign_trials(id),
                    report_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS campaign_artifacts (
                    id TEXT PRIMARY KEY,
                    campaign_id TEXT NOT NULL REFERENCES campaigns(id),
                    kind TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(campaign_id, kind, sha256)
                );
                CREATE TABLE IF NOT EXISTS campaign_idempotency (
                    scope TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    request_sha256 TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    status_code INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(scope, idempotency_key)
                );
                CREATE TABLE IF NOT EXISTS legacy_reward_agent_sessions (
                    id TEXT PRIMARY KEY,
                    source_sha256 TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    imported_at TEXT NOT NULL
                );
                """
            )
            connection.execute(
                "INSERT OR IGNORE INTO autopilot_meta(key, value) VALUES('schema_version', ?)",
                (str(STORE_SCHEMA_VERSION),),
            )
            stored_version = connection.execute(
                "SELECT value FROM autopilot_meta WHERE key='schema_version'"
            ).fetchone()
            if stored_version is None or stored_version["value"] != str(STORE_SCHEMA_VERSION):
                raise AutopilotStoreError(
                    f"Unsupported autopilot store schema version: "
                    f"{None if stored_version is None else stored_version['value']}"
                )

    def capabilities(self) -> dict[str, Any]:
        from .autopilot import autopilot_capabilities

        result = autopilot_capabilities(enabled=self.enabled)
        connection = self._connect()
        try:
            result["legacy_reward_agent_import_count"] = int(
                connection.execute("SELECT COUNT(*) FROM legacy_reward_agent_sessions").fetchone()[0]
            )
        finally:
            connection.close()
        return result

    def import_legacy_reward_agent(self, sessions_file: Path) -> int:
        """Retain legacy JSON sessions as non-armable external references."""

        path = Path(sessions_file)
        if not path.is_file():
            return 0
        content = path.read_bytes()
        try:
            decoded = json.loads(content.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise AutopilotStoreError("legacy Reward Agent sessions JSON is corrupt") from exc
        sessions = decoded.get("sessions") if isinstance(decoded, dict) else None
        if not isinstance(sessions, list) or any(not isinstance(item, Mapping) for item in sessions):
            raise AutopilotStoreError("legacy Reward Agent sessions JSON has an invalid schema")
        source_sha = hashlib.sha256(content).hexdigest()
        imported = 0
        with self._transaction() as connection:
            for index, raw in enumerate(sessions):
                session = dict(raw)
                raw_id = session.get("id")
                legacy_id = str(raw_id) if isinstance(raw_id, str) and raw_id else f"legacy_{source_sha[:16]}_{index}"
                cursor = connection.execute(
                    """INSERT OR IGNORE INTO legacy_reward_agent_sessions
                       (id, source_sha256, payload_json, imported_at) VALUES (?, ?, ?, ?)""",
                    (legacy_id, source_sha, _json(session), _now()),
                )
                imported += int(cursor.rowcount > 0)
        return imported

    def list_legacy_reward_agent_sessions(self) -> list[dict[str, Any]]:
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT id, source_sha256, payload_json, imported_at FROM legacy_reward_agent_sessions ORDER BY imported_at, id"
            ).fetchall()
            return [
                {
                    "id": row["id"],
                    "source_sha256": row["source_sha256"],
                    "payload": _load(row["payload_json"], {}),
                    "imported_at": row["imported_at"],
                    "armable": False,
                }
                for row in rows
            ]
        finally:
            connection.close()

    @staticmethod
    def _request_sha(payload: Mapping[str, Any]) -> str:
        return sha256_json(payload)

    def _prior_idempotent(
        self,
        connection: sqlite3.Connection,
        *,
        scope: str,
        key: str,
        request: Mapping[str, Any],
    ) -> Any | None:
        if not isinstance(key, str) or not 8 <= len(key) <= 160:
            raise ValueError("Idempotency-Key must contain 8 to 160 characters")
        row = connection.execute(
            "SELECT request_sha256, response_json FROM campaign_idempotency WHERE scope=? AND idempotency_key=?",
            (scope, key),
        ).fetchone()
        if row is None:
            return None
        digest = self._request_sha(request)
        if row["request_sha256"] != digest:
            raise AutopilotConflictError("Idempotency-Key was already used for a different request")
        return json.loads(row["response_json"])

    def _remember_idempotent(
        self,
        connection: sqlite3.Connection,
        *,
        scope: str,
        key: str,
        request: Mapping[str, Any],
        response: Mapping[str, Any],
        status_code: int = 200,
    ) -> None:
        connection.execute(
            """INSERT INTO campaign_idempotency
               (scope, idempotency_key, request_sha256, response_json, status_code, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (scope, key, self._request_sha(request), _json(response), int(status_code), _now()),
        )

    @staticmethod
    def _campaign_row(connection: sqlite3.Connection, campaign_id: str) -> sqlite3.Row:
        row = connection.execute("SELECT * FROM campaigns WHERE id=?", (campaign_id,)).fetchone()
        if row is None:
            raise CampaignNotFoundError(f"Campaign not found: {campaign_id}")
        return row

    @staticmethod
    def _check_revision(row: sqlite3.Row, expected_revision: int) -> None:
        if isinstance(expected_revision, bool) or not isinstance(expected_revision, int):
            raise ValueError("expected revision must be an integer")
        if int(row["revision"]) != expected_revision:
            raise AutopilotConflictError(
                f"Stale campaign revision: expected {expected_revision}, current {row['revision']}",
                current_revision=int(row["revision"]),
            )

    @staticmethod
    def _event(
        connection: sqlite3.Connection,
        campaign_id: str,
        revision: int,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        connection.execute(
            """INSERT INTO campaign_events(campaign_id, revision, event_type, payload_json, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (campaign_id, revision, event_type, _json(dict(payload or {})), _now()),
        )

    def create_campaign(
        self,
        goal: GoalSpecV1,
        reward_catalog: Sequence[RewardCatalogEntryV1],
        *,
        idempotency_key: str,
        runtime: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        catalog = tuple(reward_catalog)
        if not catalog:
            raise ValueError("reward catalog may not be empty")
        self._validate_catalog(goal, catalog)
        request = {
            "goal": goal.to_dict(),
            "reward_catalog": [entry.to_dict() for entry in catalog],
            "runtime": dict(runtime or {}),
        }
        scope = "campaign:create"
        with self._transaction() as connection:
            prior = self._prior_idempotent(connection, scope=scope, key=idempotency_key, request=request)
            if prior is not None:
                return prior
            campaign_id = _identifier("campaign")
            timestamp = _now()
            reward_values = {entry.key: entry.start_value for entry in catalog}
            leader = {"trial_id": None, "evaluation_id": None, "reward_values": reward_values}
            budget = {
                "max_training_trials": goal.budget.max_training_trials,
                "max_gpu_hours": goal.budget.max_gpu_hours,
                "used_training_trials": 0,
                "used_gpu_hours": 0.0,
                "reserved_confirmation_trials": 4,
                "remaining_training_trials": goal.budget.max_training_trials,
                "remaining_gpu_hours": goal.budget.max_gpu_hours,
                "connector_polls": 0,
                "max_connector_polls": MAX_CONNECTOR_POLLS,
            }
            connector = {
                "last_heartbeat_at": None,
                "consecutive_missed_polls": 0,
                "prompt_version": None,
                "skill_version": None,
                "declared_model": None,
                "reasoning_effort": None,
                "metadata_schema": None,
            }
            connection.execute(
                """INSERT INTO campaigns
                   (id, schema_version, revision, state, goal_json, reward_catalog_json, leader_json,
                    budget_json, active_process_json, connector_json, runtime_json, resume_state,
                    terminal_reason, created_at, updated_at)
                   VALUES (?, ?, 0, 'draft', ?, ?, ?, ?, NULL, ?, ?, NULL, NULL, ?, ?)""",
                (
                    campaign_id,
                    CAMPAIGN_SCHEMA_VERSION,
                    _json(goal.to_dict()),
                    _json([entry.to_dict() for entry in catalog]),
                    _json(leader),
                    _json(budget),
                    _json(connector),
                    _json(dict(runtime or {})),
                    timestamp,
                    timestamp,
                ),
            )
            self._event(connection, campaign_id, 0, "campaign_created", {"state": "draft"})
            response = self._snapshot(connection, campaign_id)
            self._remember_idempotent(
                connection,
                scope=scope,
                key=idempotency_key,
                request=request,
                response=response,
                status_code=201,
            )
            return response

    @staticmethod
    def _validate_catalog(
        goal: GoalSpecV1,
        catalog: Sequence[RewardCatalogEntryV1],
    ) -> None:
        keys = [entry.key for entry in catalog]
        if len(set(keys)) != len(keys):
            raise ValueError("reward catalog keys must be unique")
        for entry in catalog:
            if goal.task not in entry.tasks or goal.stage not in entry.stages:
                raise ValueError(
                    f"reward catalog entry {entry.key} is incompatible with the campaign goal"
                )

    def list_campaigns(self, *, state: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 1000))
        connection = self._connect()
        try:
            if state:
                rows = connection.execute(
                    "SELECT id FROM campaigns WHERE state=? ORDER BY created_at DESC LIMIT ?",
                    (state, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT id FROM campaigns ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()
            return [self._snapshot(connection, str(row["id"])) for row in rows]
        finally:
            connection.close()

    def list_controller_campaigns(self) -> list[dict[str, Any]]:
        """Return every campaign that can require durable controller work."""

        excluded = ("draft", *sorted(TERMINAL_STATES))
        placeholders = ",".join("?" for _ in excluded)
        connection = self._connect()
        try:
            rows = connection.execute(
                f"""SELECT id FROM campaigns
                    WHERE state NOT IN ({placeholders})
                       OR active_process_json IS NOT NULL
                    ORDER BY rowid""",
                excluded,
            ).fetchall()
            return [self._snapshot(connection, str(row["id"])) for row in rows]
        finally:
            connection.close()

    def get_campaign(self, campaign_id: str) -> dict[str, Any]:
        connection = self._connect()
        try:
            return self._snapshot(connection, campaign_id)
        finally:
            connection.close()

    def get_runtime(self, campaign_id: str) -> dict[str, Any]:
        connection = self._connect()
        try:
            row = self._campaign_row(connection, campaign_id)
            return dict(_load(row["runtime_json"], {}))
        finally:
            connection.close()

    def _snapshot(self, connection: sqlite3.Connection, campaign_id: str) -> dict[str, Any]:
        row = self._campaign_row(connection, campaign_id)
        decisions = [
            _load(item["decision_json"], {})
            for item in connection.execute(
                "SELECT decision_json FROM campaign_decisions WHERE campaign_id=? ORDER BY rowid",
                (campaign_id,),
            ).fetchall()
        ]
        evaluations = [
            _load(item["report_json"], {})
            for item in connection.execute(
                "SELECT report_json FROM campaign_evaluations WHERE campaign_id=? ORDER BY rowid",
                (campaign_id,),
            ).fetchall()
        ]
        trials = [self._trial_dict(item) for item in connection.execute(
            "SELECT * FROM campaign_trials WHERE campaign_id=? ORDER BY rowid", (campaign_id,)
        ).fetchall()]
        snapshot = CampaignSnapshotV1(
            id=str(row["id"]),
            revision=int(row["revision"]),
            state=str(row["state"]),
            goal=GoalSpecV1.from_dict(_load(row["goal_json"], {})),
            reward_catalog=tuple(
                RewardCatalogEntryV1.from_dict(item) for item in _load(row["reward_catalog_json"], [])
            ),
            leader=dict(_load(row["leader_json"], {})),
            budget=dict(_load(row["budget_json"], {})),
            active_process=_load(row["active_process_json"], None),
            candidate_lineage=tuple(trials),
            decisions=tuple(decisions),
            evaluations=tuple(evaluations),
            connector=dict(_load(row["connector_json"], {})),
            resume_state=row["resume_state"],
            terminal_reason=row["terminal_reason"],
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        ).to_dict()
        return snapshot

    @staticmethod
    def _trial_dict(row: sqlite3.Row) -> dict[str, Any]:
        metadata = _load(row["metadata_json"], {})
        return {
            "id": row["id"],
            "kind": row["kind"],
            "seed": row["seed"],
            "reward_profile": _load(row["reward_profile_json"], {}),
            "reward_profile_sha256": row["reward_profile_sha256"],
            "source_checkpoint_sha256": row["source_checkpoint_sha256"],
            "output_checkpoint_sha256": row["output_checkpoint_sha256"],
            "run_id": row["run_id"],
            "evaluation_process_id": row["evaluation_process_id"],
            "evaluation_id": row["evaluation_id"],
            "status": row["status"],
            "retry_count": row["retry_count"],
            "evaluation_retry_count": int(metadata.get("evaluation_retry_count") or 0),
            "evaluation_retry_reason": metadata.get("evaluation_retry_reason"),
            "metadata": metadata,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def update_draft(
        self,
        campaign_id: str,
        goal: GoalSpecV1,
        reward_catalog: Sequence[RewardCatalogEntryV1],
        *,
        expected_revision: int,
        idempotency_key: str,
        runtime: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        catalog = tuple(reward_catalog)
        if not catalog:
            raise ValueError("reward catalog may not be empty")
        self._validate_catalog(goal, catalog)
        request = {
            "goal": goal.to_dict(),
            "reward_catalog": [entry.to_dict() for entry in catalog],
            "runtime": dict(runtime or {}),
            "expected_revision": expected_revision,
        }
        scope = f"campaign:{campaign_id}:update"
        with self._transaction() as connection:
            prior = self._prior_idempotent(connection, scope=scope, key=idempotency_key, request=request)
            if prior is not None:
                return prior
            row = self._campaign_row(connection, campaign_id)
            self._check_revision(row, expected_revision)
            if row["state"] != "draft":
                raise AutopilotConflictError("Only draft campaigns may be edited", current_revision=row["revision"])
            revision = int(row["revision"]) + 1
            leader = {
                "trial_id": None,
                "evaluation_id": None,
                "reward_values": {entry.key: entry.start_value for entry in catalog},
            }
            budget = {
                "max_training_trials": goal.budget.max_training_trials,
                "max_gpu_hours": goal.budget.max_gpu_hours,
                "used_training_trials": 0,
                "used_gpu_hours": 0.0,
                "reserved_confirmation_trials": 4,
                "remaining_training_trials": goal.budget.max_training_trials,
                "remaining_gpu_hours": goal.budget.max_gpu_hours,
                "connector_polls": 0,
                "max_connector_polls": MAX_CONNECTOR_POLLS,
            }
            connection.execute(
                """UPDATE campaigns SET revision=?, goal_json=?, reward_catalog_json=?, leader_json=?,
                   budget_json=?, runtime_json=?, updated_at=? WHERE id=?""",
                (
                    revision,
                    _json(goal.to_dict()),
                    _json([entry.to_dict() for entry in catalog]),
                    _json(leader),
                    _json(budget),
                    _json(dict(runtime or {})),
                    _now(),
                    campaign_id,
                ),
            )
            self._event(connection, campaign_id, revision, "draft_updated")
            response = self._snapshot(connection, campaign_id)
            self._remember_idempotent(connection, scope=scope, key=idempotency_key, request=request, response=response)
            return response

    def _transition(
        self,
        connection: sqlite3.Connection,
        campaign_id: str,
        target: str,
        *,
        expected_revision: int,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
        active_process: Mapping[str, Any] | None | object = ...,
        resume_state: str | None | object = ...,
        terminal_reason: str | None | object = ...,
        leader: Mapping[str, Any] | None = None,
        budget: Mapping[str, Any] | None = None,
        runtime: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        row = self._campaign_row(connection, campaign_id)
        self._check_revision(row, expected_revision)
        current = str(row["state"])
        validate_transition(current, target, resume_state=row["resume_state"])
        revision = int(row["revision"]) + 1
        assignments = ["revision=?", "state=?", "updated_at=?"]
        values: list[Any] = [revision, target, _now()]
        if active_process is not ...:
            assignments.append("active_process_json=?")
            values.append(None if active_process is None else _json(dict(active_process)))
        if resume_state is not ...:
            assignments.append("resume_state=?")
            values.append(resume_state)
        if terminal_reason is not ...:
            assignments.append("terminal_reason=?")
            values.append(terminal_reason)
        if leader is not None:
            assignments.append("leader_json=?")
            values.append(_json(dict(leader)))
        if budget is not None:
            assignments.append("budget_json=?")
            values.append(_json(dict(budget)))
        if runtime is not None:
            assignments.append("runtime_json=?")
            values.append(_json(dict(runtime)))
        values.append(campaign_id)
        try:
            connection.execute(f"UPDATE campaigns SET {', '.join(assignments)} WHERE id=?", values)
        except sqlite3.IntegrityError as exc:
            if target in HOST_SLOT_STATES:
                raise AutopilotConflictError(
                    "Another campaign already holds the host execution slot",
                    current_revision=int(row["revision"]),
                ) from exc
            raise
        self._event(connection, campaign_id, revision, event_type, {"from": current, "to": target, **dict(payload or {})})
        return self._snapshot(connection, campaign_id)

    def _idempotent_transition(
        self,
        campaign_id: str,
        target: str,
        *,
        expected_revision: int,
        idempotency_key: str,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
        **updates: Any,
    ) -> dict[str, Any]:
        request_updates: dict[str, Any] = {}
        for name, value in updates.items():
            if value is ...:
                continue
            request_updates[name] = dict(value) if isinstance(value, Mapping) else value
        request = {
            "target": target,
            "expected_revision": expected_revision,
            "payload": dict(payload or {}),
            "updates": request_updates,
        }
        scope = f"campaign:{campaign_id}:{event_type}"
        with self._transaction() as connection:
            prior = self._prior_idempotent(connection, scope=scope, key=idempotency_key, request=request)
            if prior is not None:
                return prior
            response = self._transition(
                connection,
                campaign_id,
                target,
                expected_revision=expected_revision,
                event_type=event_type,
                payload=payload,
                **updates,
            )
            self._remember_idempotent(connection, scope=scope, key=idempotency_key, request=request, response=response)
            return response

    def arm_campaign(self, campaign_id: str, *, expected_revision: int, idempotency_key: str) -> dict[str, Any]:
        if not self.enabled:
            raise AutopilotConflictError("Autopilot is disabled; drafts may be saved but not armed")
        return self._idempotent_transition(
            campaign_id,
            "armed",
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
            event_type="campaign_armed",
        )

    def pause_campaign(
        self,
        campaign_id: str,
        *,
        expected_revision: int,
        idempotency_key: str,
        reason: str = "",
        advisor_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._transaction() as connection:
            request = {
                "expected_revision": expected_revision,
                "reason": reason,
                "advisor_metadata": dict(advisor_metadata or {}),
            }
            scope = f"campaign:{campaign_id}:pause"
            prior = self._prior_idempotent(connection, scope=scope, key=idempotency_key, request=request)
            if prior is not None:
                return prior
            row = self._campaign_row(connection, campaign_id)
            self._check_revision(row, expected_revision)
            current = str(row["state"])
            if current == "paused":
                response = self._snapshot(connection, campaign_id)
            else:
                budget, poll_exhausted = self._apply_advisor_poll(
                    connection, row, advisor_metadata
                )
                if poll_exhausted:
                    response = self._transition(
                        connection,
                        campaign_id,
                        "budget_exhausted",
                        expected_revision=expected_revision,
                        event_type="connector_poll_budget_exhausted",
                        payload={"action": "campaign_terminated"},
                        resume_state=None,
                        terminal_reason="Connector poll budget is exhausted",
                        budget=budget,
                    )
                    self._remember_idempotent(
                        connection,
                        scope=scope,
                        key=idempotency_key,
                        request=request,
                        response=response,
                    )
                    return response
                response = self._transition(
                    connection,
                    campaign_id,
                    "paused",
                    expected_revision=expected_revision,
                    event_type="campaign_paused",
                    payload={"reason": reason},
                    resume_state=current,
                )
                if advisor_metadata:
                    self._event(
                        connection,
                        campaign_id,
                        int(response["revision"]),
                        "connector_heartbeat",
                        {"declared_model": advisor_metadata.get("declared_model")},
                    )
            self._remember_idempotent(connection, scope=scope, key=idempotency_key, request=request, response=response)
            return response

    def resume_campaign(
        self, campaign_id: str, *, expected_revision: int, idempotency_key: str
    ) -> dict[str, Any]:
        with self._transaction() as connection:
            request = {"expected_revision": expected_revision}
            scope = f"campaign:{campaign_id}:resume"
            prior = self._prior_idempotent(connection, scope=scope, key=idempotency_key, request=request)
            if prior is not None:
                return prior
            row = self._campaign_row(connection, campaign_id)
            self._check_revision(row, expected_revision)
            if row["state"] != "paused" or not row["resume_state"]:
                raise AutopilotConflictError("Campaign is not resumable", current_revision=row["revision"])
            runtime = dict(_load(row["runtime_json"], {}))
            if runtime.get("emergency_stop_intent"):
                raise AutopilotConflictError(
                    "Campaign has a pending emergency stop",
                    current_revision=row["revision"],
                )
            pending = runtime.get("pending_advisor_action")
            if isinstance(pending, Mapping) and pending.get("action") == "pause":
                runtime.pop("pending_advisor_action", None)
            response = self._transition(
                connection,
                campaign_id,
                str(row["resume_state"]),
                expected_revision=expected_revision,
                event_type="campaign_resumed",
                resume_state=None,
                runtime=runtime,
            )
            self._remember_idempotent(connection, scope=scope, key=idempotency_key, request=request, response=response)
            return response

    def stop_campaign(
        self,
        campaign_id: str,
        *,
        expected_revision: int,
        idempotency_key: str,
        reason: str = "operator request",
        after_current: bool = True,
        gpu_accounting: Mapping[str, Any] | None = None,
        advisor_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._transaction() as connection:
            request = {
                "expected_revision": expected_revision,
                "reason": reason,
                "after_current": bool(after_current),
                "gpu_accounting": dict(gpu_accounting or {}),
                "advisor_metadata": dict(advisor_metadata or {}),
            }
            scope = f"campaign:{campaign_id}:stop"
            prior = self._prior_idempotent(connection, scope=scope, key=idempotency_key, request=request)
            if prior is not None:
                return prior
            row = self._campaign_row(connection, campaign_id)
            self._check_revision(row, expected_revision)
            budget, poll_exhausted = self._apply_advisor_poll(
                connection, row, advisor_metadata
            )
            if poll_exhausted:
                response = self._transition(
                    connection,
                    campaign_id,
                    "budget_exhausted",
                    expected_revision=expected_revision,
                    event_type="connector_poll_budget_exhausted",
                    payload={"action": "campaign_terminated"},
                    resume_state=None,
                    terminal_reason="Connector poll budget is exhausted",
                    budget=budget,
                )
                self._remember_idempotent(
                    connection,
                    scope=scope,
                    key=idempotency_key,
                    request=request,
                    response=response,
                )
                return response
            active = _load(row["active_process_json"], None)
            stop_runtime = dict(_load(row["runtime_json"], {}))
            stop_runtime.pop("emergency_stop_intent", None)
            updated_budget: dict[str, Any] | None = None
            if gpu_accounting is not None:
                accounting = dict(gpu_accounting)
                required = {"trial_id", "process_id", "process_kind", "cumulative_gpu_hours"}
                if set(accounting) != required:
                    raise ValueError("stop GPU accounting has an unexpected schema")
                if not active or str(active.get("process_id") or "") != accounting["process_id"]:
                    raise AutopilotConflictError(
                        "stop GPU process is no longer active",
                        current_revision=int(row["revision"]),
                    )
                trial = connection.execute(
                    "SELECT * FROM campaign_trials WHERE campaign_id=? AND id=?",
                    (campaign_id, accounting["trial_id"]),
                ).fetchone()
                if trial is None:
                    raise CampaignNotFoundError(
                        f"Campaign trial not found: {accounting['trial_id']}"
                    )
                identity_column = (
                    "run_id" if accounting["process_kind"] == "training"
                    else "evaluation_process_id"
                )
                if str(trial[identity_column] or "") != accounting["process_id"]:
                    raise ValueError("stop GPU process identity does not match the trial")
                metadata, updated_budget, _, gpu_exhausted, _ = _apply_cumulative_gpu_usage(
                    _load(trial["metadata_json"], {}),
                    _load(row["budget_json"], {}),
                    process_id=str(accounting["process_id"]),
                    process_kind=str(accounting["process_kind"]),
                    cumulative_gpu_hours=accounting["cumulative_gpu_hours"],
                )
                connection.execute(
                    "UPDATE campaign_trials SET metadata_json=?, updated_at=? WHERE id=?",
                    (_json(metadata), _now(), accounting["trial_id"]),
                )
                if gpu_exhausted:
                    response = self._transition(
                        connection,
                        campaign_id,
                        "budget_exhausted",
                        expected_revision=expected_revision,
                        event_type="gpu_budget_exhausted",
                        payload={
                            "reason": "GPU-hour budget is exhausted",
                            "process_id": accounting["process_id"],
                        },
                        active_process=None,
                        terminal_reason="GPU-hour budget is exhausted",
                        budget=updated_budget,
                        runtime=stop_runtime,
                    )
                    self._remember_idempotent(
                        connection,
                        scope=scope,
                        key=idempotency_key,
                        request=request,
                        response=response,
                    )
                    return response
            if after_current and active:
                runtime = dict(_load(row["runtime_json"], {}))
                runtime["stop_after_current"] = True
                revision = int(row["revision"]) + 1
                if updated_budget is None:
                    connection.execute(
                        "UPDATE campaigns SET revision=?, runtime_json=?, updated_at=? WHERE id=?",
                        (revision, _json(runtime), _now(), campaign_id),
                    )
                else:
                    connection.execute(
                        """UPDATE campaigns SET revision=?, runtime_json=?, budget_json=?,
                           updated_at=? WHERE id=?""",
                        (revision, _json(runtime), _json(updated_budget), _now(), campaign_id),
                    )
                self._event(
                    connection,
                    campaign_id,
                    revision,
                    "stop_after_current_requested",
                    {"reason": reason},
                )
                response = self._snapshot(connection, campaign_id)
            else:
                response = self._transition(
                    connection,
                    campaign_id,
                    "stopped",
                    expected_revision=expected_revision,
                    event_type="campaign_stopped",
                    payload={"reason": reason},
                    active_process=None,
                    terminal_reason=reason,
                    budget=updated_budget,
                    runtime=stop_runtime,
                )
            self._remember_idempotent(connection, scope=scope, key=idempotency_key, request=request, response=response)
            if advisor_metadata:
                self._event(
                    connection,
                    campaign_id,
                    int(response["revision"]),
                    "connector_heartbeat",
                    {"declared_model": advisor_metadata.get("declared_model")},
                )
            return response

    @staticmethod
    def _apply_advisor_poll(
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        advisor_metadata: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], bool]:
        """Apply one advisor visit inside the surrounding lifecycle mutation."""

        budget = dict(_load(row["budget_json"], {}))
        if not advisor_metadata:
            return budget, False
        required = {
            "schema_version",
            "skill_version",
            "prompt_version",
            "declared_model",
            "reasoning_effort",
        }
        if set(advisor_metadata) != required:
            raise ValueError("advisor metadata has an unexpected schema")
        budget, exhausted = _consume_connector_poll(budget)
        connector = dict(_load(row["connector_json"], {}))
        connector.update(
            {
                "last_heartbeat_at": _now(),
                "consecutive_missed_polls": 0,
                "prompt_version": advisor_metadata["prompt_version"],
                "skill_version": advisor_metadata["skill_version"],
                "declared_model": advisor_metadata["declared_model"],
                "reasoning_effort": advisor_metadata["reasoning_effort"],
                "metadata_schema": advisor_metadata["schema_version"],
            }
        )
        connection.execute(
            "UPDATE campaigns SET connector_json=?, budget_json=? WHERE id=?",
            (_json(connector), _json(budget), row["id"]),
        )
        return budget, exhausted

    def begin_emergency_stop(
        self,
        campaign_id: str,
        *,
        expected_revision: int,
        idempotency_key: str,
        reason: str,
        gpu_accounting: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Persist exact stop intent before any process signal is sent."""

        accounting = dict(gpu_accounting)
        required = {"trial_id", "process_id", "process_kind", "cumulative_gpu_hours"}
        if set(accounting) != required:
            raise ValueError("emergency stop GPU accounting has an unexpected schema")
        request = {
            "expected_revision": expected_revision,
            "reason": reason,
            "gpu_accounting": accounting,
        }
        scope = f"campaign:{campaign_id}:emergency-stop-intent"
        with self._transaction() as connection:
            prior = self._prior_idempotent(
                connection, scope=scope, key=idempotency_key, request=request
            )
            if prior is not None:
                return prior
            row = self._campaign_row(connection, campaign_id)
            self._check_revision(row, expected_revision)
            if row["state"] in TERMINAL_STATES:
                raise AutopilotConflictError(
                    "Campaign is already terminal",
                    current_revision=int(row["revision"]),
                )
            active = _load(row["active_process_json"], None)
            if (
                not active
                or str(active.get("process_id") or "") != accounting["process_id"]
            ):
                raise AutopilotConflictError(
                    "emergency stop process is no longer active",
                    current_revision=int(row["revision"]),
                )
            runtime = dict(_load(row["runtime_json"], {}))
            runtime["emergency_stop_intent"] = {
                "idempotency_key": idempotency_key,
                "reason": reason,
                "gpu_accounting": accounting,
                "requested_at": _now(),
                "prior_state": row["state"],
            }
            revision = int(row["revision"]) + 1
            connection.execute(
                """UPDATE campaigns SET revision=?, state='paused', resume_state=NULL,
                   runtime_json=?, updated_at=? WHERE id=?""",
                (revision, _json(runtime), _now(), campaign_id),
            )
            self._event(
                connection,
                campaign_id,
                revision,
                "emergency_stop_intent_recorded",
                {"process_id": accounting["process_id"], "reason": reason},
            )
            response = self._snapshot(connection, campaign_id)
            self._remember_idempotent(
                connection,
                scope=scope,
                key=idempotency_key,
                request=request,
                response=response,
            )
            return response

    def begin_controller_failure_stop(
        self,
        campaign_id: str,
        *,
        expected_revision: int,
        idempotency_key: str,
        target: str,
        reason: str,
        gpu_accounting: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Persist a controller-failure stop intent before signaling a process."""

        if target not in {"blocked_safety", "failed"}:
            raise ValueError("controller failure stop target is invalid")
        accounting = dict(gpu_accounting)
        required = {"trial_id", "process_id", "process_kind", "cumulative_gpu_hours"}
        if set(accounting) != required:
            raise ValueError("controller failure GPU accounting has an unexpected schema")
        request = {
            "expected_revision": expected_revision,
            "target": target,
            "reason": reason,
            "gpu_accounting": accounting,
        }
        scope = f"campaign:{campaign_id}:controller-failure-stop-intent"
        with self._transaction() as connection:
            prior = self._prior_idempotent(
                connection, scope=scope, key=idempotency_key, request=request
            )
            if prior is not None:
                return prior
            row = self._campaign_row(connection, campaign_id)
            self._check_revision(row, expected_revision)
            if row["state"] in TERMINAL_STATES:
                raise AutopilotConflictError(
                    "Campaign is already terminal",
                    current_revision=int(row["revision"]),
                )
            active = _load(row["active_process_json"], None)
            if (
                not active
                or str(active.get("process_id") or "") != accounting["process_id"]
            ):
                raise AutopilotConflictError(
                    "controller failure process is no longer active",
                    current_revision=int(row["revision"]),
                )
            runtime = dict(_load(row["runtime_json"], {}))
            runtime["controller_failure_stop_intent"] = {
                "idempotency_key": idempotency_key,
                "target": target,
                "reason": reason,
                "gpu_accounting": accounting,
                "requested_at": _now(),
            }
            revision = int(row["revision"]) + 1
            connection.execute(
                "UPDATE campaigns SET revision=?, runtime_json=?, updated_at=? WHERE id=?",
                (revision, _json(runtime), _now(), campaign_id),
            )
            self._event(
                connection,
                campaign_id,
                revision,
                "controller_failure_stop_intent_recorded",
                {
                    "process_id": accounting["process_id"],
                    "target": target,
                    "reason": reason,
                },
            )
            response = self._snapshot(connection, campaign_id)
            self._remember_idempotent(
                connection,
                scope=scope,
                key=idempotency_key,
                request=request,
                response=response,
            )
            return response

    def finalize_controller_failure_stop(
        self,
        campaign_id: str,
        *,
        expected_revision: int,
        idempotency_key: str,
        gpu_accounting: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Charge the stopped process and terminalize its persisted failure intent."""

        accounting = dict(gpu_accounting)
        required = {"trial_id", "process_id", "process_kind", "cumulative_gpu_hours"}
        if set(accounting) != required:
            raise ValueError("controller failure GPU accounting has an unexpected schema")
        request = {
            "expected_revision": expected_revision,
            "gpu_accounting": accounting,
        }
        scope = f"campaign:{campaign_id}:controller-failure-stop-finalize"
        with self._transaction() as connection:
            prior = self._prior_idempotent(
                connection, scope=scope, key=idempotency_key, request=request
            )
            if prior is not None:
                return prior
            row = self._campaign_row(connection, campaign_id)
            self._check_revision(row, expected_revision)
            runtime = dict(_load(row["runtime_json"], {}))
            intent = runtime.get("controller_failure_stop_intent")
            if not isinstance(intent, Mapping):
                raise AutopilotConflictError(
                    "Campaign has no controller failure stop intent",
                    current_revision=int(row["revision"]),
                )
            intended_accounting = dict(intent.get("gpu_accounting") or {})
            for name in ("trial_id", "process_id", "process_kind"):
                if accounting.get(name) != intended_accounting.get(name):
                    raise AutopilotConflictError(
                        "controller failure stop identity changed",
                        current_revision=int(row["revision"]),
                    )
            active = _load(row["active_process_json"], None)
            if (
                not active
                or str(active.get("process_id") or "") != accounting["process_id"]
            ):
                raise AutopilotConflictError(
                    "controller failure process is no longer active",
                    current_revision=int(row["revision"]),
                )
            trial = connection.execute(
                "SELECT * FROM campaign_trials WHERE campaign_id=? AND id=?",
                (campaign_id, accounting["trial_id"]),
            ).fetchone()
            if trial is None:
                raise CampaignNotFoundError(
                    f"Campaign trial not found: {accounting['trial_id']}"
                )
            identity_column = (
                "run_id"
                if accounting["process_kind"] == "training"
                else "evaluation_process_id"
            )
            if str(trial[identity_column] or "") != accounting["process_id"]:
                raise ValueError("controller failure process identity does not match the trial")
            metadata, budget, _, gpu_exhausted, _ = _apply_cumulative_gpu_usage(
                _load(trial["metadata_json"], {}),
                _load(row["budget_json"], {}),
                process_id=str(accounting["process_id"]),
                process_kind=str(accounting["process_kind"]),
                cumulative_gpu_hours=accounting["cumulative_gpu_hours"],
            )
            connection.execute(
                "UPDATE campaign_trials SET metadata_json=?, updated_at=? WHERE id=?",
                (_json(metadata), _now(), accounting["trial_id"]),
            )
            runtime.pop("controller_failure_stop_intent", None)
            target = "budget_exhausted" if gpu_exhausted else str(intent["target"])
            reason = (
                "GPU-hour budget is exhausted while stopping failed campaign work"
                if gpu_exhausted
                else str(intent["reason"])
            )
            response = self._transition(
                connection,
                campaign_id,
                target,
                expected_revision=expected_revision,
                event_type="controller_failure_terminalized",
                payload={"reason": reason, "process_id": accounting["process_id"]},
                active_process=None,
                terminal_reason=reason,
                budget=budget,
                runtime=runtime,
            )
            self._remember_idempotent(
                connection,
                scope=scope,
                key=idempotency_key,
                request=request,
                response=response,
            )
            return response

    def transition_campaign(
        self,
        campaign_id: str,
        target: str,
        *,
        expected_revision: int,
        idempotency_key: str,
        reason: str | None = None,
        active_process: Mapping[str, Any] | None | object = ...,
        leader: Mapping[str, Any] | None = None,
        budget: Mapping[str, Any] | None = None,
        runtime: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._idempotent_transition(
            campaign_id,
            target,
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
            event_type="campaign_transition",
            payload={"reason": reason} if reason else {},
            active_process=active_process,
            terminal_reason=reason if target in {
                "simulation_goal_met", "failed", "blocked_safety", "budget_exhausted",
                "patch_handoff", "stopped",
            } else ...,
            leader=leader,
            budget=budget,
            runtime=runtime,
        )

    def record_decision(
        self,
        campaign_id: str,
        decision: AgentDecisionV1,
        *,
        expected_revision: int,
        idempotency_key: str,
        advisor_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        request = {
            "decision": decision.to_dict(),
            "expected_revision": expected_revision,
            "advisor_metadata": dict(advisor_metadata or {}),
        }
        scope = f"campaign:{campaign_id}:decision"
        with self._transaction() as connection:
            prior = self._prior_idempotent(connection, scope=scope, key=idempotency_key, request=request)
            if prior is not None:
                return prior
            row = self._campaign_row(connection, campaign_id)
            self._check_revision(row, expected_revision)
            if decision.campaign_id != campaign_id or decision.campaign_revision != expected_revision:
                raise AutopilotConflictError("Decision campaign/revision does not match the request")
            if row["state"] not in {"awaiting_advisor", "waiting_for_chatgpt"}:
                raise AutopilotConflictError("Campaign is not awaiting an advisor", current_revision=row["revision"])
            known_evidence = {
                str(item["id"])
                for item in connection.execute(
                    "SELECT id FROM campaign_evaluations WHERE campaign_id=?",
                    (campaign_id,),
                ).fetchall()
            }
            known_evidence.update(
                str(item["id"])
                for item in connection.execute(
                    "SELECT id FROM campaign_artifacts WHERE campaign_id=?",
                    (campaign_id,),
                ).fetchall()
            )
            missing_evidence = sorted(set(decision.evidence_ids) - known_evidence)
            if missing_evidence:
                raise ValueError(f"decision references unknown evidence: {missing_evidence[0]}")
            runtime = dict(_load(row["runtime_json"], {}))
            if runtime.get("pending_advisor_action") or runtime.get(
                "pending_candidate_profile"
            ):
                raise AutopilotConflictError(
                    "A previously accepted advisor action is still pending",
                    current_revision=int(row["revision"]),
                )
            pending_profile: dict[str, float] | None = None
            if decision.action == "propose_candidate":
                if runtime.get("pending_candidate_profile"):
                    raise AutopilotConflictError(
                        "A validated candidate is already pending launch",
                        current_revision=int(row["revision"]),
                    )
                goal = GoalSpecV1.from_dict(_load(row["goal_json"], {}))
                catalog = tuple(
                    RewardCatalogEntryV1.from_dict(item)
                    for item in _load(row["reward_catalog_json"], [])
                )
                leader = dict(_load(row["leader_json"], {}))
                pending_profile = validate_candidate_decision(
                    decision,
                    goal,
                    catalog,
                    dict(leader.get("reward_values") or {}),
                )
                prior_decisions = [
                    _load(item["decision_json"], {})
                    for item in connection.execute(
                        "SELECT decision_json FROM campaign_decisions WHERE campaign_id=? ORDER BY rowid",
                        (campaign_id,),
                    ).fetchall()
                ]
                remaining_moves = reward_move_lattice(
                    catalog,
                    dict(leader.get("reward_values") or {}),
                    prior_decisions,
                )["remaining"]
                if not any(
                    move["reward_key"] == decision.reward_key
                    and math.isclose(
                        float(move["proposed_value"]),
                        float(decision.proposed_value),
                        rel_tol=1e-12,
                        abs_tol=1e-15,
                    )
                    for move in remaining_moves
                ):
                    raise AutopilotValidationError(
                        "decision is not a remaining move in the finite approved reward lattice"
                    )
            revision = int(row["revision"]) + 1
            if pending_profile is not None:
                runtime["pending_candidate_profile"] = pending_profile
                runtime["pending_decision_revision"] = revision
            elif decision.action in {"pause", "request_patch_handoff"}:
                runtime["pending_advisor_action"] = {
                    "action": decision.action,
                    "decision_revision": revision,
                }
            decision_id = _identifier("decision")
            payload = {"id": decision_id, **decision.to_dict(), "created_at": _now()}
            connection.execute(
                "INSERT INTO campaign_decisions(id, campaign_id, revision, decision_json, created_at) VALUES(?, ?, ?, ?, ?)",
                (decision_id, campaign_id, revision, _json(payload), payload["created_at"]),
            )
            connector = dict(_load(row["connector_json"], {}))
            budget, connector_budget_exhausted = _consume_connector_poll(
                _load(row["budget_json"], {})
            )
            metadata = dict(advisor_metadata or {})
            connector.update(
                {
                    "last_heartbeat_at": _now(),
                    "consecutive_missed_polls": 0,
                    "prompt_version": metadata.get("prompt_version"),
                    "skill_version": metadata.get("skill_version"),
                    "declared_model": metadata.get("declared_model"),
                    "reasoning_effort": metadata.get("reasoning_effort"),
                    "metadata_schema": metadata.get("schema_version"),
                }
            )
            connection.execute(
                """UPDATE campaigns SET revision=?, state=?, resume_state=?, terminal_reason=?,
                   connector_json=?, budget_json=?, runtime_json=?, updated_at=? WHERE id=?""",
                (
                    revision,
                    "budget_exhausted" if connector_budget_exhausted else row["state"],
                    None if connector_budget_exhausted else row["resume_state"],
                    (
                        "Connector poll budget is exhausted"
                        if connector_budget_exhausted
                        else row["terminal_reason"]
                    ),
                    _json(connector),
                    _json(budget),
                    _json(runtime),
                    _now(),
                    campaign_id,
                ),
            )
            self._event(
                connection,
                campaign_id,
                revision,
                "advisor_decision_recorded",
                {
                    "decision_id": decision_id,
                    "action": decision.action,
                    "advisor_metadata": metadata,
                },
            )
            if connector_budget_exhausted:
                self._event(
                    connection,
                    campaign_id,
                    revision,
                    "connector_poll_budget_exhausted",
                    {"action": "campaign_terminated", "maximum": budget["max_connector_polls"]},
                )
            response = self._snapshot(connection, campaign_id)
            self._remember_idempotent(connection, scope=scope, key=idempotency_key, request=request, response=response, status_code=201)
            return response

    def record_patch_proposal(
        self,
        campaign_id: str,
        decision: AgentDecisionV1,
        content: bytes,
        *,
        media_type: str,
        artifact_metadata: Mapping[str, Any] | None = None,
        expected_revision: int,
        idempotency_key: str,
        advisor_metadata: Mapping[str, Any] | None = None,
        validate_before_commit: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        """Attach one immutable proposal to an existing patch handoff."""

        if not isinstance(content, bytes) or not content:
            raise ValueError("patch proposal content must be non-empty bytes")
        content_sha256 = hashlib.sha256(content).hexdigest()
        relative = Path(content_sha256[:2]) / content_sha256
        destination = self.artifact_dir / relative
        request = {
            "decision": decision.to_dict(),
            "content_sha256": content_sha256,
            "media_type": media_type,
            "artifact_metadata": dict(artifact_metadata or {}),
            "advisor_metadata": dict(advisor_metadata or {}),
            "expected_revision": expected_revision,
        }
        scope = f"campaign:{campaign_id}:patch-proposal"
        with self._transaction() as connection:
            prior = self._prior_idempotent(connection, scope=scope, key=idempotency_key, request=request)
            if prior is not None:
                return prior
            row = self._campaign_row(connection, campaign_id)
            self._check_revision(row, expected_revision)
            if row["state"] != "patch_handoff":
                raise AutopilotConflictError(
                    "Patch proposals are accepted only after deterministic patch handoff",
                    current_revision=int(row["revision"]),
                )
            if decision.campaign_id != campaign_id or decision.campaign_revision != expected_revision:
                raise AutopilotConflictError("Patch decision campaign/revision mismatch")
            if decision.action != "request_patch_handoff":
                raise ValueError("patch proposal decision action must be request_patch_handoff")
            runtime = dict(_load(row["runtime_json"], {}))
            if runtime.get("patch_proposal_artifact_id"):
                raise AutopilotConflictError(
                    "Campaign already has a patch proposal",
                    current_revision=int(row["revision"]),
                )
            budget, connector_budget_exhausted = _consume_connector_poll(
                _load(row["budget_json"], {})
            )
            if validate_before_commit is not None:
                validate_before_commit()
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if hashlib.sha256(destination.read_bytes()).hexdigest() != content_sha256:
                    raise AutopilotStoreError("content-addressed artifact collision")
            else:
                temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
                temporary.write_bytes(content)
                os.replace(temporary, destination)
            stored_artifact = connection.execute(
                """SELECT * FROM campaign_artifacts
                   WHERE campaign_id=? AND kind='patch_proposal' AND sha256=?""",
                (campaign_id, content_sha256),
            ).fetchone()
            if stored_artifact is None:
                artifact_id = _identifier("artifact")
                connection.execute(
                    """INSERT INTO campaign_artifacts
                       (id, campaign_id, kind, sha256, relative_path, media_type, size_bytes,
                        metadata_json, created_at) VALUES (?, ?, 'patch_proposal', ?, ?, ?, ?, ?, ?)""",
                    (
                        artifact_id,
                        campaign_id,
                        content_sha256,
                        str(relative),
                        str(media_type),
                        len(content),
                        _json(dict(artifact_metadata or {})),
                        _now(),
                    ),
                )
                stored_artifact = connection.execute(
                    "SELECT * FROM campaign_artifacts WHERE id=?", (artifact_id,)
                ).fetchone()
            artifact_id = str(stored_artifact["id"])
            decision_id = _identifier("decision")
            timestamp = _now()
            decision_payload = {"id": decision_id, **decision.to_dict(), "created_at": timestamp}
            connection.execute(
                "INSERT INTO campaign_decisions(id, campaign_id, revision, decision_json, created_at) VALUES(?, ?, ?, ?, ?)",
                (decision_id, campaign_id, int(row["revision"]) + 1, _json(decision_payload), timestamp),
            )
            runtime["patch_proposal_artifact_id"] = artifact_id
            connector = dict(_load(row["connector_json"], {}))
            metadata = dict(advisor_metadata or {})
            connector.update(
                {
                    "last_heartbeat_at": timestamp,
                    "consecutive_missed_polls": 0,
                    "prompt_version": metadata.get("prompt_version"),
                    "skill_version": metadata.get("skill_version"),
                    "declared_model": metadata.get("declared_model"),
                    "reasoning_effort": metadata.get("reasoning_effort"),
                    "metadata_schema": metadata.get("schema_version"),
                }
            )
            revision = int(row["revision"]) + 1
            connection.execute(
                """UPDATE campaigns SET revision=?, runtime_json=?, connector_json=?, budget_json=?,
                   updated_at=? WHERE id=?""",
                (revision, _json(runtime), _json(connector), _json(budget), timestamp, campaign_id),
            )
            self._event(
                connection,
                campaign_id,
                revision,
                "patch_proposal_recorded",
                {"decision_id": decision_id, "artifact_id": artifact_id, "applied": False},
            )
            if connector_budget_exhausted:
                self._event(
                    connection,
                    campaign_id,
                    revision,
                    "connector_poll_budget_exhausted",
                    {"action": "terminal_noop", "maximum": budget["max_connector_polls"]},
                )
            response = self._snapshot(connection, campaign_id)
            response["patch_proposal_artifact"] = self._artifact_dict(stored_artifact)
            self._remember_idempotent(
                connection,
                scope=scope,
                key=idempotency_key,
                request=request,
                response=response,
                status_code=201,
            )
            return response

    def reserve_trial(
        self,
        campaign_id: str,
        *,
        kind: str,
        seed: int,
        reward_profile: Mapping[str, Any],
        source_checkpoint_sha256: str | None,
        expected_revision: int,
        idempotency_key: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Reserve one training slot before a process may be launched.

        Screening candidates are not allowed to consume the four confirmation
        slots.  The reservation and budget decrement occur in the same WAL
        transaction, so a retry can never launch an unaccounted-for trial.
        """

        allowed_kinds = {
            "control",
            "candidate",
            "confirmation_control",
            "confirmation_candidate",
        }
        if kind not in allowed_kinds:
            raise ValueError(f"unsupported campaign trial kind: {kind}")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("trial seed must be a non-negative integer")
        profile: dict[str, float] = {}
        for key, value in reward_profile.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"reward profile value must be a finite number: {key}")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"reward profile value must be finite: {key}")
            profile[str(key)] = number
        profile_sha = sha256_json(profile)
        if source_checkpoint_sha256 is not None and (
            not isinstance(source_checkpoint_sha256, str)
            or SHA256_RE.fullmatch(source_checkpoint_sha256) is None
        ):
            raise ValueError("source checkpoint must be a lowercase SHA-256 digest")
        request = {
            "kind": kind,
            "seed": seed,
            "reward_profile": profile,
            "source_checkpoint_sha256": source_checkpoint_sha256,
            "metadata": dict(metadata or {}),
            "expected_revision": expected_revision,
        }
        scope = f"campaign:{campaign_id}:reserve-trial"
        with self._transaction() as connection:
            prior = self._prior_idempotent(connection, scope=scope, key=idempotency_key, request=request)
            if prior is not None:
                return prior
            row = self._campaign_row(connection, campaign_id)
            self._check_revision(row, expected_revision)
            if row["state"] not in {
                "armed", "awaiting_advisor", "confirming", "control_training", "candidate_training"
            }:
                raise AutopilotConflictError(
                    "Campaign state does not permit a trial reservation",
                    current_revision=int(row["revision"]),
                )
            goal = GoalSpecV1.from_dict(_load(row["goal_json"], {}))
            if goal.initialization_mode == "policy_only":
                if source_checkpoint_sha256 != goal.checkpoint_sha256:
                    raise ValueError("trial source checkpoint does not match the frozen campaign lineage")
            elif source_checkpoint_sha256 is not None:
                raise ValueError("fresh campaigns may not specify a source checkpoint")
            if kind in {"control", "candidate"} and seed != goal.training_seeds[0]:
                raise ValueError("screening control and candidate trials must use seed 42")
            if kind.startswith("confirmation_") and seed not in goal.training_seeds[1:]:
                raise ValueError("confirmation trials must use seed 43 or 44")
            catalog = tuple(
                RewardCatalogEntryV1.from_dict(item)
                for item in _load(row["reward_catalog_json"], [])
            )
            catalog_by_key = {entry.key: entry for entry in catalog}
            if set(profile) != set(catalog_by_key):
                unknown = sorted(set(profile) - set(catalog_by_key))
                missing = sorted(set(catalog_by_key) - set(profile))
                detail = unknown[0] if unknown else missing[0]
                raise ValueError(f"reward profile does not match the campaign catalog: {detail}")
            for key, value in profile.items():
                entry = catalog_by_key[key]
                if not entry.minimum <= value <= entry.maximum:
                    raise ValueError(f"reward profile value is outside approved bounds: {key}")
                if value == 0.0 or value * entry.start_value <= 0.0:
                    raise ValueError(f"reward profile value changes sign or reaches zero: {key}")
            start_profile = {entry.key: entry.start_value for entry in catalog}
            leader = dict(_load(row["leader_json"], {}))
            leader_profile = dict(leader.get("reward_values") or {})
            if kind in {"control", "confirmation_control"} and profile != start_profile:
                raise ValueError("control trials must use the immutable campaign-start reward profile")
            changed_from_leader = [key for key in profile if profile[key] != leader_profile.get(key)]
            if kind == "candidate" and len(changed_from_leader) != 1:
                raise ValueError("screening candidates must change exactly one reward from the leader")
            if kind == "confirmation_candidate" and profile != leader_profile:
                raise ValueError("confirmation candidates must reproduce the selected leader profile")
            budget = dict(_load(row["budget_json"], {}))
            used = int(budget.get("used_training_trials", 0))
            maximum = int(budget.get("max_training_trials", 0))
            reserved = int(budget.get("reserved_confirmation_trials", 0))
            if used >= maximum:
                raise AutopilotBudgetError("training trial budget is exhausted")
            if kind == "candidate" and maximum - used <= reserved:
                raise AutopilotBudgetError("remaining training slots are reserved for confirmation")
            if kind.startswith("confirmation_"):
                if reserved <= 0:
                    raise AutopilotBudgetError("confirmation reservation is exhausted")
                budget["reserved_confirmation_trials"] = reserved - 1
            duplicate = connection.execute(
                """SELECT id FROM campaign_trials
                   WHERE campaign_id=? AND kind=? AND seed=? AND reward_profile_sha256=?""",
                (campaign_id, kind, seed, profile_sha),
            ).fetchone()
            if duplicate is not None:
                raise AutopilotConflictError(
                    f"an equivalent trial is already reserved: {duplicate['id']}",
                    current_revision=int(row["revision"]),
                )
            budget["used_training_trials"] = used + 1
            budget["remaining_training_trials"] = maximum - used - 1
            trial_id = _identifier("trial")
            timestamp = _now()
            connection.execute(
                """INSERT INTO campaign_trials
                   (id, campaign_id, kind, seed, reward_profile_json, reward_profile_sha256,
                    source_checkpoint_sha256, output_checkpoint_sha256, run_id,
                    evaluation_process_id, evaluation_id, status, retry_count,
                    metadata_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, 'reserved', 0, ?, ?, ?)""",
                (
                    trial_id,
                    campaign_id,
                    kind,
                    seed,
                    _json(profile),
                    profile_sha,
                    source_checkpoint_sha256,
                    _json(dict(metadata or {})),
                    timestamp,
                    timestamp,
                ),
            )
            revision = int(row["revision"]) + 1
            connection.execute(
                "UPDATE campaigns SET revision=?, budget_json=?, updated_at=? WHERE id=?",
                (revision, _json(budget), timestamp, campaign_id),
            )
            self._event(
                connection,
                campaign_id,
                revision,
                "trial_reserved",
                {"trial_id": trial_id, "kind": kind, "seed": seed, "reward_profile_sha256": profile_sha},
            )
            response = {"trial": self._trial_dict(connection.execute(
                "SELECT * FROM campaign_trials WHERE id=?", (trial_id,)
            ).fetchone()), "campaign": self._snapshot(connection, campaign_id)}
            self._remember_idempotent(
                connection,
                scope=scope,
                key=idempotency_key,
                request=request,
                response=response,
                status_code=201,
            )
            return response

    def get_trial(self, campaign_id: str, trial_id: str) -> dict[str, Any]:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM campaign_trials WHERE campaign_id=? AND id=?",
                (campaign_id, trial_id),
            ).fetchone()
            if row is None:
                raise CampaignNotFoundError(f"Campaign trial not found: {trial_id}")
            return self._trial_dict(row)
        finally:
            connection.close()

    def update_trial(
        self,
        campaign_id: str,
        trial_id: str,
        *,
        expected_revision: int,
        idempotency_key: str,
        status: str,
        run_id: str | None | object = ...,
        evaluation_process_id: str | None | object = ...,
        output_checkpoint_sha256: str | None | object = ...,
        retry_count: int | None = None,
        metadata: Mapping[str, Any] | None = None,
        active_process: Mapping[str, Any] | None | object = ...,
        event_type: str = "trial_updated",
    ) -> dict[str, Any]:
        allowed_statuses = {
            "reserved", "queued", "training", "trained", "evaluating", "evaluated",
            "selected", "rejected", "failed", "cancelled",
        }
        if status not in allowed_statuses:
            raise ValueError(f"invalid trial status: {status}")
        request = {
            "trial_id": trial_id,
            "status": status,
            "run_id": None if run_id is ... else run_id,
            "evaluation_process_id": None if evaluation_process_id is ... else evaluation_process_id,
            "output_checkpoint_sha256": None if output_checkpoint_sha256 is ... else output_checkpoint_sha256,
            "retry_count": retry_count,
            "metadata": dict(metadata or {}),
            "active_process": None if active_process is ... else active_process,
            "expected_revision": expected_revision,
        }
        scope = f"campaign:{campaign_id}:trial:{trial_id}:{event_type}"
        with self._transaction() as connection:
            prior = self._prior_idempotent(connection, scope=scope, key=idempotency_key, request=request)
            if prior is not None:
                return prior
            campaign = self._campaign_row(connection, campaign_id)
            self._check_revision(campaign, expected_revision)
            row = connection.execute(
                "SELECT * FROM campaign_trials WHERE campaign_id=? AND id=?",
                (campaign_id, trial_id),
            ).fetchone()
            if row is None:
                raise CampaignNotFoundError(f"Campaign trial not found: {trial_id}")
            assignments = ["status=?", "updated_at=?"]
            values: list[Any] = [status, _now()]
            for column, value in (
                ("run_id", run_id),
                ("evaluation_process_id", evaluation_process_id),
                ("output_checkpoint_sha256", output_checkpoint_sha256),
            ):
                if value is not ...:
                    if column == "output_checkpoint_sha256" and value is not None and (
                        not isinstance(value, str) or SHA256_RE.fullmatch(value) is None
                    ):
                        raise ValueError("output checkpoint must be a lowercase SHA-256 digest")
                    assignments.append(f"{column}=?")
                    values.append(value)
            if retry_count is not None:
                if retry_count not in {0, 1}:
                    raise ValueError("retry_count must be zero or one")
                assignments.append("retry_count=?")
                values.append(retry_count)
            if metadata is not None:
                merged_metadata = dict(_load(row["metadata_json"], {}))
                merged_metadata.update(dict(metadata))
                assignments.append("metadata_json=?")
                values.append(_json(merged_metadata))
            values.append(trial_id)
            connection.execute(
                f"UPDATE campaign_trials SET {', '.join(assignments)} WHERE id=?",
                values,
            )
            revision = int(campaign["revision"]) + 1
            campaign_assignments = ["revision=?", "updated_at=?"]
            campaign_values: list[Any] = [revision, _now()]
            if active_process is not ...:
                campaign_assignments.append("active_process_json=?")
                campaign_values.append(None if active_process is None else _json(dict(active_process)))
            campaign_values.append(campaign_id)
            connection.execute(
                f"UPDATE campaigns SET {', '.join(campaign_assignments)} WHERE id=?",
                campaign_values,
            )
            self._event(
                connection,
                campaign_id,
                revision,
                event_type,
                {"trial_id": trial_id, "status": status},
            )
            response = {"trial": self._trial_dict(connection.execute(
                "SELECT * FROM campaign_trials WHERE id=?", (trial_id,)
            ).fetchone()), "campaign": self._snapshot(connection, campaign_id)}
            self._remember_idempotent(connection, scope=scope, key=idempotency_key, request=request, response=response)
            return response

    def account_process_gpu_usage(
        self,
        campaign_id: str,
        trial_id: str,
        *,
        process_id: str,
        process_kind: str,
        cumulative_gpu_hours: float,
        force: bool = False,
        expected_revision: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Durably charge only a process' newly observed active GPU time."""

        request = {
            "trial_id": trial_id,
            "process_id": process_id,
            "process_kind": process_kind,
            "cumulative_gpu_hours": float(cumulative_gpu_hours),
            "force": bool(force),
            "expected_revision": expected_revision,
        }
        scope = f"campaign:{campaign_id}:gpu-process:{process_id}"
        with self._transaction() as connection:
            prior = self._prior_idempotent(
                connection, scope=scope, key=idempotency_key, request=request
            )
            if prior is not None:
                return prior
            campaign = self._campaign_row(connection, campaign_id)
            self._check_revision(campaign, expected_revision)
            trial = connection.execute(
                "SELECT * FROM campaign_trials WHERE campaign_id=? AND id=?",
                (campaign_id, trial_id),
            ).fetchone()
            if trial is None:
                raise CampaignNotFoundError(f"Campaign trial not found: {trial_id}")
            identity_column = "run_id" if process_kind == "training" else "evaluation_process_id"
            if str(trial[identity_column] or "") != process_id:
                raise ValueError("GPU process identity does not match the campaign trial")

            metadata, budget, delta, exhausted, marker_changed = _apply_cumulative_gpu_usage(
                _load(trial["metadata_json"], {}),
                _load(campaign["budget_json"], {}),
                process_id=process_id,
                process_kind=process_kind,
                cumulative_gpu_hours=cumulative_gpu_hours,
            )
            persist = bool(force) or exhausted or delta >= GPU_ACCOUNTING_INTERVAL_HOURS
            if persist and (delta > 0.0 or marker_changed or exhausted):
                timestamp = _now()
                connection.execute(
                    "UPDATE campaign_trials SET metadata_json=?, updated_at=? WHERE id=?",
                    (_json(metadata), timestamp, trial_id),
                )
                revision = int(campaign["revision"]) + 1
                connection.execute(
                    """UPDATE campaigns SET revision=?, state=?, budget_json=?,
                       active_process_json=?, terminal_reason=?, updated_at=? WHERE id=?""",
                    (
                        revision,
                        "budget_exhausted" if exhausted else campaign["state"],
                        _json(budget),
                        campaign["active_process_json"],
                        "GPU-hour budget is exhausted" if exhausted else campaign["terminal_reason"],
                        timestamp,
                        campaign_id,
                    ),
                )
                self._event(
                    connection,
                    campaign_id,
                    revision,
                    "gpu_budget_exhausted" if exhausted else "gpu_usage_recorded",
                    {
                        "trial_id": trial_id,
                        "process_id": process_id,
                        "process_kind": process_kind,
                        "cumulative_gpu_hours": float(cumulative_gpu_hours),
                        "charged_gpu_hours": delta,
                    },
                )
            if not persist:
                delta = 0.0
            response = {
                "trial": self._trial_dict(connection.execute(
                    "SELECT * FROM campaign_trials WHERE id=?", (trial_id,)
                ).fetchone()),
                "campaign": self._snapshot(connection, campaign_id),
                "charged_gpu_hours": delta,
            }
            self._remember_idempotent(
                connection, scope=scope, key=idempotency_key, request=request, response=response
            )
            return response

    def clear_budget_exhausted_process(
        self,
        campaign_id: str,
        *,
        process_id: str,
        expected_revision: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Clear a bound process only after its budget stop is observed terminal."""

        request = {
            "process_id": process_id,
            "expected_revision": expected_revision,
        }
        scope = f"campaign:{campaign_id}:budget-process-stop:{process_id}"
        with self._transaction() as connection:
            prior = self._prior_idempotent(
                connection, scope=scope, key=idempotency_key, request=request
            )
            if prior is not None:
                return prior
            campaign = self._campaign_row(connection, campaign_id)
            self._check_revision(campaign, expected_revision)
            if campaign["state"] != "budget_exhausted":
                raise AutopilotConflictError(
                    "campaign is not GPU-budget exhausted",
                    current_revision=int(campaign["revision"]),
                )
            active = _load(campaign["active_process_json"], None)
            if active is None:
                response = self._snapshot(connection, campaign_id)
            else:
                if str(active.get("process_id") or "") != process_id:
                    raise AutopilotConflictError(
                        "budget stop process identity changed",
                        current_revision=int(campaign["revision"]),
                    )
                revision = int(campaign["revision"]) + 1
                timestamp = _now()
                connection.execute(
                    """UPDATE campaigns SET revision=?, active_process_json=NULL,
                       updated_at=? WHERE id=?""",
                    (revision, timestamp, campaign_id),
                )
                self._event(
                    connection,
                    campaign_id,
                    revision,
                    "gpu_budget_process_stopped",
                    {"process_id": process_id},
                )
                response = self._snapshot(connection, campaign_id)
            self._remember_idempotent(
                connection, scope=scope, key=idempotency_key, request=request, response=response
            )
            return response

    def record_gpu_usage(
        self,
        campaign_id: str,
        *,
        gpu_hours: float,
        expected_revision: int,
        idempotency_key: str,
        process_id: str,
    ) -> dict[str, Any]:
        if isinstance(gpu_hours, bool) or not isinstance(gpu_hours, (int, float)) or not 0.0 <= float(gpu_hours):
            raise ValueError("gpu_hours must be a finite non-negative number")
        if float(gpu_hours) != float(gpu_hours) or abs(float(gpu_hours)) == float("inf"):
            raise ValueError("gpu_hours must be finite")
        request = {"gpu_hours": float(gpu_hours), "process_id": process_id, "expected_revision": expected_revision}
        scope = f"campaign:{campaign_id}:gpu:{process_id}"
        with self._transaction() as connection:
            prior = self._prior_idempotent(connection, scope=scope, key=idempotency_key, request=request)
            if prior is not None:
                return prior
            row = self._campaign_row(connection, campaign_id)
            self._check_revision(row, expected_revision)
            budget = dict(_load(row["budget_json"], {}))
            used = float(budget.get("used_gpu_hours", 0.0)) + float(gpu_hours)
            maximum = float(budget.get("max_gpu_hours", 0.0))
            budget["used_gpu_hours"] = used
            budget["remaining_gpu_hours"] = max(0.0, maximum - used)
            revision = int(row["revision"]) + 1
            exhausted = used >= maximum
            state = "budget_exhausted" if exhausted else str(row["state"])
            terminal_reason = "GPU-hour budget is exhausted" if exhausted else row["terminal_reason"]
            connection.execute(
                """UPDATE campaigns SET revision=?, state=?, budget_json=?, active_process_json=?,
                   terminal_reason=?, updated_at=? WHERE id=?""",
                (
                    revision,
                    state,
                    _json(budget),
                    None if exhausted else row["active_process_json"],
                    terminal_reason,
                    _now(),
                    campaign_id,
                ),
            )
            self._event(
                connection,
                campaign_id,
                revision,
                "gpu_budget_exhausted" if exhausted else "gpu_usage_recorded",
                request,
            )
            response = self._snapshot(connection, campaign_id)
            self._remember_idempotent(connection, scope=scope, key=idempotency_key, request=request, response=response)
            return response

    def complete_training(
        self,
        campaign_id: str,
        trial_id: str,
        *,
        output_checkpoint_path: str,
        output_checkpoint_sha256: str,
        process_id: str,
        gpu_hours: float,
        expected_revision: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Atomically bind the trained checkpoint and charge its GPU time."""

        if not isinstance(output_checkpoint_sha256, str) or SHA256_RE.fullmatch(output_checkpoint_sha256) is None:
            raise ValueError("output checkpoint must be a lowercase SHA-256 digest")
        if (
            not isinstance(output_checkpoint_path, str)
            or not output_checkpoint_path
            or not Path(output_checkpoint_path).is_absolute()
        ):
            raise ValueError("output checkpoint path must be a non-empty absolute path")
        if (
            isinstance(gpu_hours, bool)
            or not isinstance(gpu_hours, (int, float))
            or not math.isfinite(float(gpu_hours))
            or float(gpu_hours) < 0.0
        ):
            raise ValueError("gpu_hours must be a finite non-negative number")
        request = {
            "trial_id": trial_id,
            "output_checkpoint_path": output_checkpoint_path,
            "output_checkpoint_sha256": output_checkpoint_sha256,
            "process_id": process_id,
            "gpu_hours": float(gpu_hours),
            "expected_revision": expected_revision,
        }
        scope = f"campaign:{campaign_id}:complete-training:{trial_id}"
        with self._transaction() as connection:
            prior = self._prior_idempotent(connection, scope=scope, key=idempotency_key, request=request)
            if prior is not None:
                return prior
            campaign = self._campaign_row(connection, campaign_id)
            self._check_revision(campaign, expected_revision)
            trial = connection.execute(
                "SELECT * FROM campaign_trials WHERE campaign_id=? AND id=?",
                (campaign_id, trial_id),
            ).fetchone()
            if trial is None:
                raise CampaignNotFoundError(f"Campaign trial not found: {trial_id}")
            if trial["run_id"] != process_id:
                raise ValueError("training process identity does not match the trial")
            if trial["status"] not in {"queued", "training"}:
                raise AutopilotConflictError("trial is not awaiting training completion")
            metadata, budget, charged_gpu_hours, exhausted, _ = _apply_cumulative_gpu_usage(
                _load(trial["metadata_json"], {}),
                _load(campaign["budget_json"], {}),
                process_id=process_id,
                process_kind="training",
                cumulative_gpu_hours=gpu_hours,
            )
            timestamp = _now()
            accounted_gpu_hours = float(
                metadata["gpu_process_accounting"][process_id]["accounted_gpu_hours"]
            )
            metadata.update(
                {
                    "training_process_id": process_id,
                    "output_checkpoint_path": output_checkpoint_path,
                    "training_gpu_hours": accounted_gpu_hours,
                    "training_gpu_accounted": True,
                }
            )
            connection.execute(
                """UPDATE campaign_trials SET status='trained', output_checkpoint_sha256=?,
                   metadata_json=?, updated_at=? WHERE id=?""",
                (
                    output_checkpoint_sha256,
                    _json(metadata),
                    timestamp,
                    trial_id,
                ),
            )
            revision = int(campaign["revision"]) + 1
            connection.execute(
                """UPDATE campaigns SET revision=?, state=?, budget_json=?, active_process_json=NULL,
                   terminal_reason=?, updated_at=? WHERE id=?""",
                (
                    revision,
                    "budget_exhausted" if exhausted else campaign["state"],
                    _json(budget),
                    "GPU-hour budget is exhausted" if exhausted else campaign["terminal_reason"],
                    timestamp,
                    campaign_id,
                ),
            )
            self._event(
                connection,
                campaign_id,
                revision,
                "gpu_budget_exhausted" if exhausted else "training_completed",
                {
                    "trial_id": trial_id,
                    "process_id": process_id,
                    "output_checkpoint_path": output_checkpoint_path,
                    "output_checkpoint_sha256": output_checkpoint_sha256,
                    "gpu_hours": accounted_gpu_hours,
                    "charged_gpu_hours": charged_gpu_hours,
                },
            )
            response = {
                "trial": self._trial_dict(connection.execute(
                    "SELECT * FROM campaign_trials WHERE id=?", (trial_id,)
                ).fetchone()),
                "campaign": self._snapshot(connection, campaign_id),
            }
            self._remember_idempotent(connection, scope=scope, key=idempotency_key, request=request, response=response)
            return response

    def record_evaluation(
        self,
        campaign_id: str,
        trial_id: str,
        report: EvaluationReportV1,
        *,
        expected_revision: int,
        idempotency_key: str,
        leader: Mapping[str, Any] | None = None,
        gpu_hours: float = 0.0,
        process_id: str | None = None,
    ) -> dict[str, Any]:
        if (
            isinstance(gpu_hours, bool)
            or not isinstance(gpu_hours, (int, float))
            or not math.isfinite(float(gpu_hours))
            or float(gpu_hours) < 0.0
        ):
            raise ValueError("gpu_hours must be a finite non-negative number")
        report_payload = report.to_dict()
        episode_count = len(report_payload["episode_metrics"])
        # Per-episode rows live in the immutable, hash-verified CSV artifact.
        # Keeping tens of thousands of rows in every campaign snapshot and
        # idempotency response would make a 24-trial campaign impractical.
        report_payload["episode_metrics"] = [
            {
                "schema_version": "redrhex.autopilot.episode-evidence-index.v1",
                "row_count": episode_count,
                "artifact_sha256": report.episode_artifact_sha256,
                "artifact_ids": list(report.artifact_ids),
            }
        ]
        request = {
            "trial_id": trial_id,
            "report": report_payload,
            "leader": None if leader is None else dict(leader),
            "gpu_hours": float(gpu_hours),
            "process_id": process_id,
            "expected_revision": expected_revision,
        }
        scope = f"campaign:{campaign_id}:evaluation:{trial_id}"
        with self._transaction() as connection:
            prior = self._prior_idempotent(connection, scope=scope, key=idempotency_key, request=request)
            if prior is not None:
                return prior
            campaign = self._campaign_row(connection, campaign_id)
            self._check_revision(campaign, expected_revision)
            trial = connection.execute(
                "SELECT * FROM campaign_trials WHERE campaign_id=? AND id=?",
                (campaign_id, trial_id),
            ).fetchone()
            if trial is None:
                raise CampaignNotFoundError(f"Campaign trial not found: {trial_id}")
            if report.trial_id != trial_id:
                raise ValueError("evaluation report trial identity mismatch")
            if report.reward_profile_sha256 != trial["reward_profile_sha256"]:
                raise ValueError("evaluation report reward profile identity mismatch")
            if not trial["output_checkpoint_sha256"]:
                raise ValueError("evaluation requires an exact recorded output checkpoint identity")
            if report.checkpoint_sha256 != trial["output_checkpoint_sha256"]:
                raise ValueError("evaluation report checkpoint identity mismatch")
            artifact_ids = tuple(report.artifact_ids)
            if len(artifact_ids) != 4 or len(set(artifact_ids)) != 4:
                raise ValueError("evaluation report must bind four unique immutable artifacts")
            placeholders = ",".join("?" for _ in artifact_ids)
            artifact_rows = connection.execute(
                f"""SELECT id, kind FROM campaign_artifacts
                    WHERE campaign_id=? AND id IN ({placeholders})""",
                (campaign_id, *artifact_ids),
            ).fetchall()
            if {str(row["id"]) for row in artifact_rows} != set(artifact_ids) or {
                str(row["kind"]) for row in artifact_rows
            } != {
                "evaluation_commands",
                "evaluation_episodes",
                "evaluation_summary",
                "evaluation_report",
            }:
                raise ValueError(
                    "evaluation report immutable artifact set is missing or mismatched"
                )
            if process_id is None:
                if float(gpu_hours) != 0.0:
                    raise ValueError("evaluation GPU usage requires an exact process identity")
                trial_metadata = dict(_load(trial["metadata_json"], {}))
                budget = dict(_load(campaign["budget_json"], {}))
                charged_gpu_hours = 0.0
                exhausted = False
            else:
                if str(trial["evaluation_process_id"] or "") != process_id:
                    raise ValueError("evaluation process identity does not match the trial")
                (
                    trial_metadata,
                    budget,
                    charged_gpu_hours,
                    exhausted,
                    _,
                ) = _apply_cumulative_gpu_usage(
                    _load(trial["metadata_json"], {}),
                    _load(campaign["budget_json"], {}),
                    process_id=process_id,
                    process_kind="evaluation",
                    cumulative_gpu_hours=gpu_hours,
                )
            evaluation_rank_key(report)
            if not report.hard_gates or "eligible" not in report.ranking:
                raise ValueError("evaluation report has not passed deterministic gate evaluation")
            timestamp = _now()
            connection.execute(
                """INSERT INTO campaign_evaluations(id, campaign_id, trial_id, report_json, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (report.id, campaign_id, trial_id, _json(report_payload), timestamp),
            )
            connection.execute(
                """UPDATE campaign_trials SET evaluation_id=?, status='evaluated',
                   metadata_json=?, updated_at=? WHERE id=?""",
                (report.id, _json(trial_metadata), timestamp, trial_id),
            )
            revision = int(campaign["revision"]) + 1
            assignments = [
                "revision=?", "active_process_json=NULL", "updated_at=?", "budget_json=?",
                "state=?", "terminal_reason=?",
            ]
            values: list[Any] = [
                revision,
                timestamp,
                _json(budget),
                "budget_exhausted" if exhausted else campaign["state"],
                "GPU-hour budget is exhausted" if exhausted else campaign["terminal_reason"],
            ]
            if leader is not None:
                assignments.append("leader_json=?")
                values.append(_json(dict(leader)))
            values.append(campaign_id)
            connection.execute(f"UPDATE campaigns SET {', '.join(assignments)} WHERE id=?", values)
            self._event(
                connection,
                campaign_id,
                revision,
                "gpu_budget_exhausted" if exhausted else "evaluation_recorded",
                {
                    "trial_id": trial_id,
                    "evaluation_id": report.id,
                    "eligible": bool(report.ranking.get("eligible")),
                    "process_id": process_id,
                    "gpu_hours": (
                        0.0
                        if process_id is None
                        else float(
                            trial_metadata["gpu_process_accounting"][process_id][
                                "accounted_gpu_hours"
                            ]
                        )
                    ),
                    "charged_gpu_hours": charged_gpu_hours,
                },
            )
            response = self._snapshot(connection, campaign_id)
            self._remember_idempotent(connection, scope=scope, key=idempotency_key, request=request, response=response)
            return response

    def record_connector_heartbeat(
        self,
        campaign_id: str,
        *,
        expected_revision: int,
        idempotency_key: str,
        prompt_version: str | None = None,
        skill_version: str | None = None,
        declared_model: str | None = None,
        reasoning_effort: str | None = None,
        metadata_schema: str | None = None,
    ) -> dict[str, Any]:
        request = {
            "expected_revision": expected_revision,
            "prompt_version": prompt_version,
            "skill_version": skill_version,
            "declared_model": declared_model,
            "reasoning_effort": reasoning_effort,
            "metadata_schema": metadata_schema,
        }
        scope = f"campaign:{campaign_id}:heartbeat"
        with self._transaction() as connection:
            prior = self._prior_idempotent(connection, scope=scope, key=idempotency_key, request=request)
            if prior is not None:
                return prior
            row = self._campaign_row(connection, campaign_id)
            self._check_revision(row, expected_revision)
            connector = dict(_load(row["connector_json"], {}))
            budget, connector_budget_exhausted = _consume_connector_poll(
                _load(row["budget_json"], {})
            )
            connector.update(
                {
                    "last_heartbeat_at": _now(),
                    "consecutive_missed_polls": 0,
                    "prompt_version": prompt_version,
                    "skill_version": skill_version,
                    "declared_model": declared_model,
                    "reasoning_effort": reasoning_effort,
                    "metadata_schema": metadata_schema,
                }
            )
            revision = int(row["revision"]) + 1
            connection.execute(
                """UPDATE campaigns SET revision=?, state=?, resume_state=?, terminal_reason=?,
                   connector_json=?, budget_json=?, updated_at=? WHERE id=?""",
                (
                    revision,
                    "budget_exhausted" if connector_budget_exhausted else row["state"],
                    None if connector_budget_exhausted else row["resume_state"],
                    (
                        "Connector poll budget is exhausted"
                        if connector_budget_exhausted
                        else row["terminal_reason"]
                    ),
                    _json(connector),
                    _json(budget),
                    _now(),
                    campaign_id,
                ),
            )
            self._event(connection, campaign_id, revision, "connector_heartbeat", {"declared_model": declared_model})
            if connector_budget_exhausted:
                self._event(
                    connection,
                    campaign_id,
                    revision,
                    "connector_poll_budget_exhausted",
                    {"action": "campaign_terminated", "maximum": budget["max_connector_polls"]},
                )
            response = self._snapshot(connection, campaign_id)
            self._remember_idempotent(connection, scope=scope, key=idempotency_key, request=request, response=response)
            return response

    def update_runtime(
        self,
        campaign_id: str,
        updates: Mapping[str, Any],
        *,
        expected_revision: int,
        idempotency_key: str,
        event_type: str = "campaign_runtime_updated",
    ) -> dict[str, Any]:
        request = {"updates": dict(updates), "expected_revision": expected_revision, "event_type": event_type}
        scope = f"campaign:{campaign_id}:runtime:{event_type}"
        with self._transaction() as connection:
            prior = self._prior_idempotent(connection, scope=scope, key=idempotency_key, request=request)
            if prior is not None:
                return prior
            row = self._campaign_row(connection, campaign_id)
            self._check_revision(row, expected_revision)
            runtime = dict(_load(row["runtime_json"], {}))
            runtime.update(dict(updates))
            revision = int(row["revision"]) + 1
            connection.execute(
                "UPDATE campaigns SET revision=?, runtime_json=?, updated_at=? WHERE id=?",
                (revision, _json(runtime), _now(), campaign_id),
            )
            self._event(connection, campaign_id, revision, event_type, dict(updates))
            response = self._snapshot(connection, campaign_id)
            self._remember_idempotent(connection, scope=scope, key=idempotency_key, request=request, response=response)
            return response

    def list_events(self, campaign_id: str, *, after: int = 0, limit: int = 500) -> list[dict[str, Any]]:
        connection = self._connect()
        try:
            self._campaign_row(connection, campaign_id)
            rows = connection.execute(
                """SELECT sequence, revision, event_type, payload_json, created_at
                   FROM campaign_events WHERE campaign_id=? AND sequence>?
                   ORDER BY sequence LIMIT ?""",
                (campaign_id, max(0, int(after)), max(1, min(int(limit), 1000))),
            ).fetchall()
            return [
                {
                    "id": f"event_{row['sequence']}",
                    "campaign_id": campaign_id,
                    "sequence": row["sequence"],
                    "revision": row["revision"],
                    "type": row["event_type"],
                    "payload": _load(row["payload_json"], {}),
                    "created_at": row["created_at"],
                }
                for row in rows
            ]
        finally:
            connection.close()

    def decision_context(self, campaign_id: str) -> dict[str, Any]:
        snapshot = self.get_campaign(campaign_id)
        leader = dict(snapshot["leader"])
        catalog = snapshot["reward_catalog"]
        current = dict(leader.get("reward_values") or {})
        catalog_entries = tuple(
            RewardCatalogEntryV1.from_dict(entry) for entry in catalog
        )
        lattice = reward_move_lattice(
            catalog_entries,
            current,
            snapshot["decisions"],
        )
        remaining_by_key: dict[str, list[float]] = {}
        for move in lattice["remaining"]:
            remaining_by_key.setdefault(str(move["reward_key"]), []).append(
                float(move["proposed_value"])
            )
        campaign_start = {
            entry.key: entry.start_value for entry in catalog_entries
        }
        baseline_profile = dict(campaign_start)
        for trial in snapshot["candidate_lineage"]:
            if trial.get("kind") == "control":
                baseline_profile = dict(trial.get("reward_profile") or campaign_start)
                break
        baseline_to_leader = {
            key: current[key] - baseline_profile[key]
            for key in campaign_start
        }
        allowable_moves = []
        for entry, contract in zip(catalog, catalog_entries):
            if not entry.get("enabled"):
                continue
            value = current.get(entry["key"], entry["start_value"])
            allowable_moves.append(
                {
                    "reward_key": entry["key"],
                    "current_value": value,
                    "campaign_start_value": entry["start_value"],
                    "minimum": entry["minimum"],
                    "maximum": entry["maximum"],
                    "sign": entry["sign"],
                    "lattice_values": list(reward_lattice_values(contract)),
                    "remaining_values": remaining_by_key.get(entry["key"], []),
                }
            )
        recent = snapshot["evaluations"][-6:]
        recent_decisions = snapshot["decisions"][-6:]
        return {
            "schema_version": "redrhex.autopilot.decision-context.v1",
            "campaign_id": campaign_id,
            "campaign_revision": snapshot["revision"],
            "state": snapshot["state"],
            "goal": {
                "description": snapshot["goal"]["description"],
                "task": snapshot["goal"]["task"],
                "stage": snapshot["goal"]["stage"],
                "gait": snapshot["goal"]["gait"],
                "directions": snapshot["goal"]["directions"],
                "command_envelope": snapshot["goal"]["command_envelope"],
                "skill_gates": snapshot["goal"]["skill_gates"],
            },
            "constraints": allowable_moves,
            "leader": leader,
            "campaign_start_reward_values": campaign_start,
            "baseline_to_leader_reward_deltas": baseline_to_leader,
            "remaining_allowable_moves": lattice["remaining"],
            "remaining_allowable_move_count": len(lattice["remaining"]),
            "attempted_moves": lattice["attempted"],
            "recent_decisions": recent_decisions,
            "recent_evaluations": recent,
            "remaining_budget": {
                "training_trials": snapshot["budget"].get("remaining_training_trials"),
                "gpu_hours": snapshot["budget"].get("remaining_gpu_hours"),
                "confirmation_trials": snapshot["budget"].get("reserved_confirmation_trials"),
                "connector_polls": max(
                    0,
                    int(snapshot["budget"].get("max_connector_polls", MAX_CONNECTOR_POLLS))
                    - int(snapshot["budget"].get("connector_polls", 0)),
                ),
            },
            "evidence_ids": [report.get("id") for report in recent if report.get("id")],
            "next_permitted_actions": snapshot["next_permitted_actions"],
            "terminal_reason": snapshot["terminal_reason"],
        }

    def store_artifact(
        self,
        campaign_id: str,
        *,
        kind: str,
        content: bytes,
        media_type: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(content, bytes) or not content:
            raise ValueError("artifact content must be non-empty bytes")
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("artifact kind must be a non-empty string")
        if not isinstance(media_type, str) or not media_type.strip():
            raise ValueError("artifact media type must be a non-empty string")
        # Validate ownership and JSON metadata before touching the filesystem,
        # so a rejected write cannot leave an unreferenced artifact behind.
        connection = self._connect()
        try:
            self._campaign_row(connection, campaign_id)
        finally:
            connection.close()
        metadata_json = _json(dict(metadata or {}))
        digest = hashlib.sha256(content).hexdigest()
        relative = Path(digest[:2]) / digest
        destination = self.artifact_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
                raise AutopilotStoreError("content-addressed artifact collision")
        else:
            temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
            temporary.write_bytes(content)
            os.replace(temporary, destination)
        with self._transaction() as connection:
            self._campaign_row(connection, campaign_id)
            existing = connection.execute(
                "SELECT * FROM campaign_artifacts WHERE campaign_id=? AND kind=? AND sha256=?",
                (campaign_id, kind, digest),
            ).fetchone()
            if existing is None:
                artifact_id = _identifier("artifact")
                connection.execute(
                    """INSERT INTO campaign_artifacts
                       (id, campaign_id, kind, sha256, relative_path, media_type, size_bytes,
                        metadata_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        artifact_id,
                        campaign_id,
                        kind,
                        digest,
                        str(relative),
                        str(media_type),
                        len(content),
                        metadata_json,
                        _now(),
                    ),
                )
                existing = connection.execute(
                    "SELECT * FROM campaign_artifacts WHERE id=?", (artifact_id,)
                ).fetchone()
            return self._artifact_dict(existing)

    @staticmethod
    def _artifact_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "campaign_id": row["campaign_id"],
            "kind": row["kind"],
            "sha256": row["sha256"],
            "media_type": row["media_type"],
            "size_bytes": row["size_bytes"],
            "metadata": _load(row["metadata_json"], {}),
            "created_at": row["created_at"],
        }

    def list_artifacts(self, campaign_id: str) -> list[dict[str, Any]]:
        connection = self._connect()
        try:
            self._campaign_row(connection, campaign_id)
            rows = connection.execute(
                "SELECT * FROM campaign_artifacts WHERE campaign_id=? ORDER BY created_at, id",
                (campaign_id,),
            ).fetchall()
            return [self._artifact_dict(row) for row in rows]
        finally:
            connection.close()

    def get_artifact(self, campaign_id: str, artifact_id: str) -> tuple[dict[str, Any], bytes]:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM campaign_artifacts WHERE campaign_id=? AND id=?",
                (campaign_id, artifact_id),
            ).fetchone()
            if row is None:
                raise CampaignNotFoundError(f"Campaign artifact not found: {artifact_id}")
            path = self.artifact_dir / str(row["relative_path"])
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != row["sha256"]:
                raise AutopilotStoreError("artifact hash verification failed")
            return self._artifact_dict(row), content
        finally:
            connection.close()

    def compare_trials(self, campaign_id: str, trial_ids: Sequence[str]) -> dict[str, Any]:
        snapshot = self.get_campaign(campaign_id)
        requested = list(dict.fromkeys(str(value) for value in trial_ids))
        if not requested:
            requested = [item["id"] for item in snapshot["candidate_lineage"][-4:]]
        if len(requested) > 12:
            raise ValueError("at most 12 trials may be compared")
        trials_by_id = {item["id"]: item for item in snapshot["candidate_lineage"]}
        evaluations_by_trial = {item["trial_id"]: item for item in snapshot["evaluations"]}
        missing = [trial_id for trial_id in requested if trial_id not in trials_by_id]
        if missing:
            raise CampaignNotFoundError(f"Campaign trial not found: {missing[0]}")
        rows = []
        for trial_id in requested:
            trial = trials_by_id[trial_id]
            report = evaluations_by_trial.get(trial_id)
            rows.append(
                {
                    "trial_id": trial_id,
                    "kind": trial["kind"],
                    "seed": trial["seed"],
                    "status": trial["status"],
                    "reward_profile_sha256": trial["reward_profile_sha256"],
                    "reward_profile": trial["reward_profile"],
                    "evaluation_id": None if report is None else report.get("id"),
                    "hard_gates": None if report is None else report.get("hard_gates"),
                    "ranking": None if report is None else report.get("ranking"),
                    "failure_reason": None if report is None else report.get("failure_reason"),
                }
            )
        return {
            "schema_version": "redrhex.autopilot.trial-comparison.v1",
            "campaign_id": campaign_id,
            "campaign_revision": snapshot["revision"],
            "trials": rows,
        }
