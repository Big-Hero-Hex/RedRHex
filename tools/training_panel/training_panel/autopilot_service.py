from __future__ import annotations

import csv
import functools
import hashlib
import io
import json
import math
import os
import re
import threading
import time
import uuid
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from .activity import ActivityStore
from .autopilot import (
    DEFAULT_SKILL_GATES,
    TERMINAL_STATES,
    AgentDecisionV1,
    AutopilotValidationError,
    CampaignBudgetV1,
    EvaluationReportV1,
    GoalSpecV1,
    RewardCatalogEntryV1,
    build_reward_catalog,
    compile_command_profile,
    compile_goal_spec,
    evaluate_report,
    evaluation_rank_key,
    reward_move_lattice,
    sha256_json,
    validate_candidate_decision,
)
from .autopilot_store import (
    GPU_ACCOUNTING_INTERVAL_HOURS,
    AutopilotBudgetError,
    AutopilotConflictError,
    AutopilotStore,
    AutopilotStoreError,
)
from .autopilot_identity import (
    AUTOPILOT_CODE_IDENTITY_PATHS,
    canonical_json_bytes,
    dependency_manifest_sha256,
    runtime_source_identities,
    source_code_identities,
    sha256_file as _sha256_file,
)
from .commands import DEFAULT_PANEL_SPRING_BACKEND, EvaluationParams, TrainingParams
from .config import PanelPaths
from .history import HistoryStore, tail_file
from .processes import GpuHostLeaseBusy, ProcessRegistry, ProcessStartError
from .rewards import reward_defaults


AUTOPILOT_ENABLED_ENV = "REDRHEX_AUTOPILOT_ENABLED"
AUTOPILOT_NUM_ENVS_ENV = "REDRHEX_AUTOPILOT_NUM_ENVS"
AUTOPILOT_POLL_SECONDS_ENV = "REDRHEX_AUTOPILOT_POLL_SECONDS"
AUTOPILOT_ADVISOR_WAIT_SECONDS_ENV = "REDRHEX_AUTOPILOT_ADVISOR_WAIT_SECONDS"


def _enabled_from_env() -> bool:
    return os.environ.get(AUTOPILOT_ENABLED_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _iso_to_epoch(value: Any) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise AutopilotValidationError(f"{name} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise AutopilotValidationError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise AutopilotValidationError(f"{name} must be finite")
    return parsed


def _csv_bool(value: Any, name: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "pass"}:
        return True
    if normalized in {"0", "false", "no", "fail"}:
        return False
    raise AutopilotValidationError(f"{name} must be an explicit boolean")


def _nested_reward_profile(flattened: Mapping[str, float]) -> dict[str, dict[str, float]]:
    nested: dict[str, float] = {}
    for key, value in flattened.items():
        prefix = "v2_reward_scales."
        if not key.startswith(prefix):
            raise AutopilotValidationError(f"campaign reward key is not a V2 shaping weight: {key}")
        nested[key.removeprefix(prefix)] = float(value)
    return {"v2_reward_scales": nested}


def _serialized_mutation(method):
    """Serialize controller and HTTP mutations across process side effects."""

    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._mutation_lock:
            return method(self, *args, **kwargs)

    return wrapped


class AutopilotService:
    """The panel-owned, durable mutation boundary for local V1 campaigns.

    There is deliberately no model client here.  An external advisor can only
    submit an AgentDecisionV1; this service validates, accounts, launches, and
    evaluates it deterministically.
    """

    def __init__(
        self,
        paths: PanelPaths,
        history: HistoryStore,
        processes: ProcessRegistry,
        activity: ActivityStore | None = None,
        *,
        enabled: bool | None = None,
        start_worker: bool = True,
        identity_python: str | Path | None = None,
    ):
        self.paths = paths
        self.history = history
        self.processes = processes
        self.activity = activity
        self._identity_python = Path(
            identity_python or (paths.conda_prefix / "bin" / "python")
        ).resolve()
        self.enabled = _enabled_from_env() if enabled is None else bool(enabled)
        self.store = AutopilotStore(
            paths.autopilot_db_file,
            paths.autopilot_artifact_dir,
            enabled=self.enabled,
        )
        try:
            imported = self.store.import_legacy_reward_agent(
                paths.repo_root / "logs" / "reward_agent" / "sessions.json"
            )
            if imported:
                self._audit(
                    "autopilot_legacy_sessions_imported",
                    "legacy-reward-agent",
                    {"count": imported, "armable": False},
                )
        except Exception as exc:
            self._audit(
                "autopilot_legacy_import_failed",
                "legacy-reward-agent",
                {"reason": str(exc)[:1000]},
            )
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._mutation_lock = threading.RLock()
        self._tick_lock = threading.Lock()
        self._worker: threading.Thread | None = None
        if self.enabled and start_worker:
            self._worker = threading.Thread(
                target=self._worker_loop,
                name="redrhex-autopilot-controller",
                daemon=True,
            )
            self._worker.start()

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        worker = self._worker
        if worker and worker.is_alive():
            worker.join(timeout=2.0)

    def _require_enabled(self) -> None:
        if not self.enabled:
            raise AutopilotConflictError(
                f"Autopilot is disabled; set {AUTOPILOT_ENABLED_ENV}=1 before starting the panel"
            )

    def _audit(self, event_type: str, campaign_id: str, payload: Mapping[str, Any] | None = None) -> None:
        if self.activity is None:
            return
        self.activity.record(
            event_type,
            summary=event_type.replace("_", " ").title(),
            subject_id=campaign_id,
            actor_name="Autopilot controller",
            payload=dict(payload or {}),
        )

    def _notify(self) -> None:
        self._wake.set()

    # -- public read surface -------------------------------------------------

    def capabilities(self) -> dict[str, Any]:
        capabilities = self.store.capabilities()
        reward_catalog: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for task, stages in capabilities["supported_stages"].items():
            defaults = reward_defaults(self.paths.repo_root, task=task)
            reward_catalog[task] = {}
            for stage in stages:
                stage_name = f"stage{stage}"
                available = [
                    key
                    for key in capabilities["default_reward_keys"][task][stage_name]
                    if key in defaults and defaults[key] != 0.0
                ]
                reward_catalog[task][stage_name] = (
                    [
                        entry.to_dict()
                        for entry in build_reward_catalog(
                            task,
                            stage,
                            defaults,
                            enabled_keys=available,
                        )
                    ]
                    if available
                    else []
                )
        capabilities["reward_catalog"] = reward_catalog
        capabilities["baselines"] = self._baseline_options()
        capabilities["default_target_gates"] = dict(DEFAULT_SKILL_GATES)
        capabilities["default_num_envs"] = self._default_num_envs()
        capabilities["advisor_mode"] = "external_only"
        capabilities["deployment_allowed"] = False
        return capabilities

    def list_campaigns(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self.store.list_campaigns(**kwargs)

    def get_campaign(self, campaign_id: str) -> dict[str, Any]:
        return self.store.get_campaign(campaign_id)

    def decision_context(self, campaign_id: str) -> dict[str, Any]:
        context = self.store.decision_context(campaign_id)
        if context["state"] == "patch_handoff":
            runtime = self.store.get_runtime(campaign_id)
            source_context = self._frozen_patch_context(campaign_id)
            context["patch_handoff"] = {
                **source_context,
                "proposal_already_submitted": bool(runtime.get("patch_proposal_artifact_id")),
                "source_application_allowed": False,
            }
            context["next_permitted_actions"] = (
                [] if runtime.get("patch_proposal_artifact_id") else ["submit_patch_proposal"]
            )
        return context

    def _reward_source_context(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        paths = (
            "source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py",
            "source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py",
        )
        terms = {
            str(entry["key"]).split(".", 1)[1]
            for entry in snapshot.get("reward_catalog", [])
            if str(entry.get("key") or "").startswith("v2_reward_scales.")
        }
        snippets: list[dict[str, Any]] = []
        remaining_chars = 50_000
        for relative in paths:
            path = self.paths.repo_root / relative
            lines = path.read_text(encoding="utf-8").splitlines()
            hits = [
                index
                for index, line in enumerate(lines)
                if any(f'"{term}"' in line or f"'{term}'" in line for term in terms)
            ]
            windows: list[tuple[int, int]] = []
            for hit in hits:
                start, end = max(0, hit - 4), min(len(lines), hit + 5)
                if windows and start <= windows[-1][1] + 1:
                    windows[-1] = (windows[-1][0], max(windows[-1][1], end))
                else:
                    windows.append((start, end))
            for start, end in windows:
                numbered = "\n".join(
                    f"{line_number + 1}: {lines[line_number]}"
                    for line_number in range(start, end)
                )
                encoded_length = len(numbered)
                if encoded_length > remaining_chars:
                    break
                snippets.append(
                    {
                        "source": relative,
                        "line_start": start + 1,
                        "line_end": end,
                        "text": numbered,
                    }
                )
                remaining_chars -= encoded_length
            if remaining_chars <= 0:
                break
        return {
            "schema_version": "redrhex.autopilot.patch-context.v1",
            "allowed_source_blobs": {
                relative: _sha256_file(self.paths.repo_root / relative) for relative in paths
            },
            "target_symbols": [
                "RedrhexEnv._compute_simplified_rewards",
                "RedrhexEnv._compute_energy_per_distance_reward",
                "RedrhexEnvCfg.v2_reward_scales",
                "RedrhexForwardFastEnvCfg.v2_reward_scales",
            ],
            "source_snippets": snippets,
            "snippet_truncated": remaining_chars <= 0,
        }

    def _frozen_patch_context(self, campaign_id: str) -> dict[str, Any]:
        artifacts = [
            item
            for item in self.store.list_artifacts(campaign_id)
            if item.get("kind") == "patch_context"
        ]
        if not artifacts:
            raise AutopilotValidationError(
                "patch handoff has no immutable source context"
            )
        _metadata, content = self.store.get_artifact(
            campaign_id, str(artifacts[0]["id"])
        )
        try:
            parsed = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AutopilotValidationError(
                "patch handoff source context is corrupt"
            ) from exc
        if (
            not isinstance(parsed, dict)
            or parsed.get("schema_version")
            != "redrhex.autopilot.patch-context.v1"
        ):
            raise AutopilotValidationError(
                "patch handoff source context has an invalid schema"
            )
        return parsed

    def list_events(self, campaign_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        return self.store.list_events(campaign_id, **kwargs)

    def list_artifacts(self, campaign_id: str) -> list[dict[str, Any]]:
        return self.store.list_artifacts(campaign_id)

    def get_artifact(self, campaign_id: str, artifact_id: str) -> tuple[dict[str, Any], bytes]:
        return self.store.get_artifact(campaign_id, artifact_id)

    def compare_trials(self, campaign_id: str, trial_ids: Sequence[str]) -> dict[str, Any]:
        return self.store.compare_trials(campaign_id, trial_ids)

    def patch_export(self, campaign_id: str) -> tuple[dict[str, Any], bytes]:
        artifact_id = self.store.get_runtime(campaign_id).get("patch_proposal_artifact_id")
        if not artifact_id:
            raise AutopilotConflictError("Campaign has no patch proposal artifact")
        return self.store.get_artifact(campaign_id, str(artifact_id))

    # -- draft compilation ---------------------------------------------------

    def _default_num_envs(self) -> int:
        raw = os.environ.get(AUTOPILOT_NUM_ENVS_ENV, "4096")
        try:
            return max(1, min(int(raw), 8192))
        except ValueError:
            return 4096

    def _baseline_options(self) -> list[dict[str, Any]]:
        options: list[dict[str, Any]] = []
        for run in self.history.list_runs():
            params = run.get("params") if isinstance(run.get("params"), dict) else {}
            task = str(params.get("task") or "")
            if task not in {"Template-Redrhex-ForwardFast-Direct-v0", "Template-Redrhex-Direct-v0"}:
                continue
            for iteration, path in self._recorded_checkpoint_choices(run):
                if not path.is_file():
                    continue
                options.append(
                    {
                        "run_id": str(run.get("id")),
                        "display_name": str(run.get("display_name") or run.get("id")),
                        "task": task,
                        "checkpoint_sha256": _sha256_file(path),
                        "checkpoint_iteration": iteration,
                        "spring_backend": str(
                            params.get("spring_backend") or DEFAULT_PANEL_SPRING_BACKEND
                        ),
                    }
                )
                if len(options) >= 100:
                    return options
        return options

    @staticmethod
    def _recorded_checkpoint_choices(run: Mapping[str, Any]) -> list[tuple[int, Path]]:
        """Return only checkpoint identities already materialized by history."""

        choices: dict[int, Path] = {}
        history = run.get("checkpoint_history")
        if isinstance(history, (list, tuple)):
            for item in history:
                if not isinstance(item, Mapping):
                    continue
                iteration = item.get("iteration")
                checkpoint = item.get("checkpoint")
                if (
                    isinstance(iteration, int)
                    and not isinstance(iteration, bool)
                    and iteration >= 0
                    and isinstance(checkpoint, str)
                    and checkpoint
                ):
                    choices[iteration] = Path(checkpoint).resolve()
        # Fake/legacy history may already contain one explicit checkpoint. It
        # is admissible only when its immutable model_<iteration>.pt identity
        # can be derived from that stored value; directories are never scanned.
        checkpoint = run.get("latest_checkpoint")
        if isinstance(checkpoint, str) and checkpoint:
            path = Path(checkpoint).resolve()
            match = re.fullmatch(r"model_(\d+)\.pt", path.name)
            if match:
                choices.setdefault(int(match.group(1)), path)
        return sorted(choices.items())

    def _resolve_baseline(self, payload: Mapping[str, Any]) -> tuple[dict[str, Any] | None, Path | None, str | None]:
        mode = payload.get("initialization_mode")
        if mode not in {"fresh", "policy_only"}:
            raise AutopilotValidationError("V1 permits only fresh or strict policy_only initialization")
        if mode == "fresh":
            if any(
                payload.get(name) is not None
                for name in (
                    "baseline_run_id",
                    "baseline_checkpoint_iteration",
                    "checkpoint_sha256",
                )
            ):
                raise AutopilotValidationError("fresh initialization may not include baseline checkpoint identity")
            return None, None, None
        run_id = payload.get("baseline_run_id")
        iteration = payload.get("baseline_checkpoint_iteration")
        expected = payload.get("checkpoint_sha256")
        if not isinstance(run_id, str) or not run_id:
            raise AutopilotValidationError("policy_only initialization requires a baseline run ID")
        if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
            raise AutopilotValidationError(
                "policy_only initialization requires an exact baseline checkpoint iteration"
            )
        if not isinstance(expected, str) or len(expected) != 64 or expected != expected.lower():
            raise AutopilotValidationError("policy_only initialization requires a lowercase checkpoint SHA-256")
        run = self.history.get_run(run_id)
        if not run:
            raise AutopilotValidationError("baseline run does not exist")
        choices = dict(self._recorded_checkpoint_choices(run))
        path = choices.get(iteration, Path(""))
        if not path.is_file():
            raise AutopilotValidationError(
                "baseline run does not contain the selected exact checkpoint iteration"
            )
        actual = _sha256_file(path)
        if actual != expected:
            raise AutopilotValidationError("baseline checkpoint SHA-256 no longer matches")
        params = run.get("params") if isinstance(run.get("params"), dict) else {}
        if str(params.get("task") or "") != str(payload.get("task") or ""):
            raise AutopilotValidationError("baseline task does not match the goal task")
        return run, path.resolve(), actual

    def _identity_bundle(
        self,
        task: str,
        baseline: Mapping[str, Any] | None,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        root = self.paths.repo_root
        try:
            source_identities, dependency_manifest = runtime_source_identities(
                root,
                simulator_root=self.paths.isaacsim_root,
                python_executable=self._identity_python,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise AutopilotValidationError(f"unable to resolve campaign identity: {exc}") from exc
        baseline_params = baseline.get("params") if baseline and isinstance(baseline.get("params"), dict) else {}
        spring_backend = str(baseline_params.get("spring_backend") or DEFAULT_PANEL_SPRING_BACKEND)
        if spring_backend != DEFAULT_PANEL_SPRING_BACKEND:
            raise AutopilotValidationError("V1 campaigns require the provisional native spring backend")
        physics_values = dict(baseline.get("physics_overrides") or {}) if baseline else {}
        terrain_values = dict(baseline.get("terrain_overrides") or {}) if baseline else {}
        if terrain_values:
            raise AutopilotValidationError(
                "V1 Autopilot requires the default terrain profile; deterministic "
                "evaluation does not yet apply baseline terrain overrides"
            )
        physics_identity = {
            "spring_backend": spring_backend,
            "physics_overrides": physics_values,
            "physics_preset_id": str((baseline or {}).get("physics_preset_id") or "baseline"),
        }
        spring_identity = {
            "backend": spring_backend,
            "calibration_status": "provisional_simulation_only",
            "physics": physics_identity,
        }
        identities = {
            "physics_profile_sha256": sha256_json(physics_identity),
            "spring_profile_sha256": sha256_json(spring_identity),
            "code_sha256": source_identities["code"],
            "config_sha256": source_identities["config"],
        }
        runtime = {
            "spring_backend": spring_backend,
            "physics_overrides": physics_values,
            "terrain_overrides": terrain_values,
            "terrain_profile_sha256": sha256_json(terrain_values),
            "physics_identity": physics_identity,
            "spring_identity": spring_identity,
            "task": task,
            "dependency_manifest": dependency_manifest,
            "dependency_sha256": source_identities["dependency"],
        }
        return identities, runtime

    @staticmethod
    def _code_identity_files() -> tuple[Path, ...]:
        return tuple(Path(relative) for relative in AUTOPILOT_CODE_IDENTITY_PATHS)

    def _assert_campaign_identity(
        self,
        snapshot: Mapping[str, Any],
        runtime: Mapping[str, Any],
        *,
        recheck_dependencies: bool = False,
    ) -> None:
        """Fail closed when a frozen campaign input changes after draft approval."""

        root = self.paths.repo_root
        goal = snapshot["goal"]
        frozen_dependency_manifest = runtime.get("dependency_manifest")
        if not isinstance(frozen_dependency_manifest, Mapping):
            raise AutopilotValidationError(
                "frozen campaign identity changed: dependency_manifest"
            )
        try:
            source_identities = source_code_identities(root)
            frozen_dependency_sha256 = dependency_manifest_sha256(
                dict(frozen_dependency_manifest)
            )
            current_dependency_sha256 = str(runtime.get("dependency_sha256") or "")
            if recheck_dependencies:
                current_source_identities, _current_dependency_manifest = (
                    runtime_source_identities(
                        root,
                        simulator_root=self.paths.isaacsim_root,
                        python_executable=self._identity_python,
                    )
                )
                source_identities = {
                    "code": current_source_identities["code"],
                    "config": current_source_identities["config"],
                }
                current_dependency_sha256 = current_source_identities["dependency"]
        except (OSError, RuntimeError, ValueError) as exc:
            raise AutopilotValidationError(
                f"frozen campaign identity cannot be resolved: {exc}"
            ) from exc
        current = {
            "code_sha256": source_identities["code"],
            "config_sha256": source_identities["config"],
            "dependency_sha256": current_dependency_sha256,
            "physics_profile_sha256": sha256_json(dict(runtime.get("physics_identity") or {})),
            "spring_profile_sha256": sha256_json(dict(runtime.get("spring_identity") or {})),
            "terrain_profile_sha256": sha256_json(dict(runtime.get("terrain_overrides") or {})),
        }
        expected = {
            "code_sha256": goal["code_sha256"],
            "config_sha256": goal["config_sha256"],
            "dependency_sha256": runtime.get("dependency_sha256"),
            "physics_profile_sha256": goal["physics_profile_sha256"],
            "spring_profile_sha256": goal["spring_profile_sha256"],
            "terrain_profile_sha256": runtime.get("terrain_profile_sha256"),
        }
        mismatched = sorted(key for key in expected if current[key] != expected[key])
        if frozen_dependency_sha256 != runtime.get("dependency_sha256"):
            mismatched.append("dependency_manifest")
        if runtime.get("task") != goal["task"]:
            mismatched.append("task")
        command_file = Path(str(runtime.get("command_profile_file") or ""))
        if (
            not command_file.is_file()
            or _sha256_file(command_file) != goal["command_profile_sha256"]
        ):
            mismatched.append("command_profile_sha256")
        checkpoint_path = runtime.get("checkpoint_path")
        if checkpoint_path:
            checkpoint = Path(str(checkpoint_path))
            if (
                not checkpoint.is_file()
                or _sha256_file(checkpoint) != goal.get("checkpoint_sha256")
            ):
                mismatched.append("checkpoint_sha256")
        if mismatched:
            raise AutopilotValidationError(
                "frozen campaign identity changed: " + ", ".join(sorted(set(mismatched)))
            )

    @staticmethod
    def _verify_client_identity(payload: Mapping[str, Any], name: str, actual: str) -> None:
        supplied = payload.get(name)
        if supplied not in (None, "") and str(supplied).lower() != actual:
            raise AutopilotValidationError(f"client-supplied {name} does not match the resolved panel identity")

    def _compile_draft(
        self,
        payload: Mapping[str, Any],
    ) -> tuple[GoalSpecV1, tuple[RewardCatalogEntryV1, ...], dict[str, Any]]:
        allowed_fields = {
            "schema_version", "description", "task", "stage", "evaluation_profile", "gait",
            "directions", "command_envelope", "skill_gates", "initialization_mode",
            "baseline_run_id", "baseline_checkpoint_iteration", "checkpoint_sha256", "physics_profile_sha256",
            "spring_profile_sha256", "code_sha256", "config_sha256", "command_profile_sha256",
            "training_seeds", "per_trial_iteration_cap", "budget", "tunable_reward_keys", "reward_bounds",
        }
        unknown_fields = sorted(set(payload) - allowed_fields)
        if unknown_fields:
            raise AutopilotValidationError(
                "goal draft has unknown fields: " + ", ".join(unknown_fields)
            )
        required_client_fields = {
            "schema_version", "description", "task", "stage", "evaluation_profile", "gait",
            "directions", "command_envelope", "skill_gates", "initialization_mode",
            "baseline_run_id", "baseline_checkpoint_iteration", "checkpoint_sha256", "training_seeds", "per_trial_iteration_cap", "budget",
            "tunable_reward_keys",
        }
        missing_fields = sorted(required_client_fields - set(payload))
        if missing_fields:
            raise AutopilotValidationError(
                "goal draft is missing fields: " + ", ".join(missing_fields)
            )
        if payload.get("schema_version") != "redrhex.autopilot.goal.v1":
            raise AutopilotValidationError("goal.schema_version must be redrhex.autopilot.goal.v1")
        requested_stage_value = payload.get("stage")
        if isinstance(requested_stage_value, bool) or not isinstance(requested_stage_value, int):
            raise AutopilotValidationError("goal.stage must be an integer")
        requested_stage = requested_stage_value
        if payload.get("evaluation_profile") != f"stage{requested_stage}":
            raise AutopilotValidationError("goal evaluation profile does not match its stage")
        if payload.get("training_seeds") != [42, 43, 44]:
            raise AutopilotValidationError("goal.training_seeds must be [42, 43, 44] in V1")
        if not isinstance(payload.get("budget"), Mapping):
            raise AutopilotValidationError("goal.budget must be an object")
        if not isinstance(payload.get("skill_gates"), Mapping):
            raise AutopilotValidationError("goal.skill_gates must be an object")
        if not isinstance(payload.get("directions"), (list, tuple)):
            raise AutopilotValidationError("goal.directions must be an array")
        if not isinstance(payload.get("command_envelope"), Mapping):
            raise AutopilotValidationError("goal.command_envelope must be an object")
        baseline, checkpoint_path, checkpoint_sha = self._resolve_baseline(payload)
        task = str(payload.get("task") or "")
        stage = requested_stage
        gait = str(payload.get("gait") or "")
        identities, runtime = self._identity_bundle(task, baseline)
        budget = CampaignBudgetV1.from_dict(payload["budget"])
        gates = payload["skill_gates"]
        goal = compile_goal_spec(
            description=str(payload.get("description") or ""),
            task=task,
            stage=stage,
            gait=gait,
            per_trial_iteration_cap=payload.get("per_trial_iteration_cap"),
            physics_profile_sha256=identities["physics_profile_sha256"],
            spring_profile_sha256=identities["spring_profile_sha256"],
            code_sha256=identities["code_sha256"],
            config_sha256=identities["config_sha256"],
            initialization_mode=str(payload["initialization_mode"]),
            baseline_run_id=None if baseline is None else str(baseline.get("id")),
            baseline_checkpoint_iteration=(
                None if baseline is None else int(payload["baseline_checkpoint_iteration"])
            ),
            checkpoint_sha256=checkpoint_sha,
            directions=payload["directions"],
            skill_gates=gates,
            budget=budget,
        )
        for name in (
            "physics_profile_sha256",
            "spring_profile_sha256",
            "code_sha256",
            "config_sha256",
            "command_profile_sha256",
        ):
            self._verify_client_identity(payload, name, str(getattr(goal, name)))
        supplied_envelope = payload["command_envelope"]
        if sha256_json(supplied_envelope) != sha256_json(goal.to_dict()["command_envelope"]):
            raise AutopilotValidationError("command envelope does not match the panel-compiled walk/run range")
        defaults = reward_defaults(self.paths.repo_root, task=task)
        enabled_keys = payload.get("tunable_reward_keys")
        if enabled_keys is not None and not isinstance(enabled_keys, (list, tuple)):
            raise AutopilotValidationError("tunable_reward_keys must be an array")
        narrowed_bounds = payload.get("reward_bounds")
        if narrowed_bounds is not None and not isinstance(narrowed_bounds, Mapping):
            raise AutopilotValidationError("reward_bounds must be an object")
        catalog = build_reward_catalog(
            task,
            stage,
            defaults,
            enabled_keys=enabled_keys,
            narrowed_bounds=narrowed_bounds,
        )
        runtime.update(
            {
                "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
                "baseline_checkpoint_iteration": goal.baseline_checkpoint_iteration,
                "checkpoint_sha256": checkpoint_sha,
                "num_envs": int(((baseline or {}).get("params") or {}).get("num_envs") or self._default_num_envs()),
                "device": str(((baseline or {}).get("params") or {}).get("device") or "cuda:0"),
                "original_reward_values": {entry.key: entry.start_value for entry in catalog},
                "non_improving_candidates": 0,
                "confirmation_queue": [],
                "stop_after_current": False,
            }
        )
        command_profile = compile_command_profile(task, stage, gait, goal.directions)
        command_content = json.dumps(
            command_profile,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        command_sha = hashlib.sha256(command_content).hexdigest()
        if command_sha != goal.command_profile_sha256:
            raise RuntimeError("compiled command profile hash is inconsistent")
        command_dir = self.paths.process_override_dir / "autopilot_command_profiles"
        command_dir.mkdir(parents=True, exist_ok=True)
        command_file = command_dir / f"{command_sha}.json"
        if command_file.exists():
            if hashlib.sha256(command_file.read_bytes()).hexdigest() != command_sha:
                raise AutopilotValidationError("immutable command profile artifact was modified")
        else:
            temporary = command_file.with_name(f".{command_file.name}.{uuid.uuid4().hex}.tmp")
            temporary.write_bytes(command_content)
            os.replace(temporary, command_file)
        runtime["command_profile_file"] = str(command_file)
        return goal, catalog, runtime

    def _store_identity_artifacts(
        self,
        campaign_id: str,
        runtime: Mapping[str, Any],
    ) -> None:
        dependency_manifest = runtime.get("dependency_manifest")
        if not isinstance(dependency_manifest, Mapping):
            raise AutopilotValidationError("campaign dependency manifest is unavailable")
        content = canonical_json_bytes(dict(dependency_manifest))
        digest = hashlib.sha256(content).hexdigest()
        if digest != runtime.get("dependency_sha256"):
            raise AutopilotValidationError("campaign dependency manifest hash is inconsistent")
        self.store.store_artifact(
            campaign_id,
            kind="dependency_manifest",
            content=content,
            media_type="application/vnd.redrhex.dependency-manifest+json",
            metadata={
                "schema_version": dependency_manifest.get("schema_version"),
                "sha256": digest,
            },
        )

    def _ensure_draft_artifacts(
        self,
        snapshot: Mapping[str, Any],
        runtime: Mapping[str, Any],
    ) -> None:
        """Reconcile crash-safe immutable draft artifacts before arming."""

        campaign_id = str(snapshot["id"])
        self._store_identity_artifacts(campaign_id, runtime)
        command_path = Path(str(runtime.get("command_profile_file") or ""))
        if not command_path.is_file():
            raise AutopilotValidationError(
                "frozen command profile is unavailable before arming"
            )
        command_content = command_path.read_bytes()
        command_sha = hashlib.sha256(command_content).hexdigest()
        if command_sha != snapshot["goal"].get("command_profile_sha256"):
            raise AutopilotValidationError(
                "frozen command profile artifact changed before arming"
            )
        self.store.store_artifact(
            campaign_id,
            kind="command_profile",
            content=command_content,
            media_type="application/vnd.redrhex.command-profile+json",
            metadata={"sha256": command_sha},
        )
        expected = {
            "dependency_manifest": str(runtime.get("dependency_sha256") or ""),
            "command_profile": command_sha,
        }
        actual = {
            (str(item.get("kind") or ""), str(item.get("sha256") or ""))
            for item in self.store.list_artifacts(campaign_id)
        }
        missing = [
            kind for kind, digest in expected.items() if (kind, digest) not in actual
        ]
        if missing:
            raise AutopilotValidationError(
                "draft immutable artifacts could not be reconciled: "
                + ", ".join(sorted(missing))
            )

    # -- public mutation surface --------------------------------------------

    @_serialized_mutation
    def create_campaign(self, payload: Mapping[str, Any], *, idempotency_key: str) -> dict[str, Any]:
        self._require_enabled()
        goal, catalog, runtime = self._compile_draft(payload)
        result = self.store.create_campaign(goal, catalog, idempotency_key=idempotency_key, runtime=runtime)
        self._store_identity_artifacts(result["id"], runtime)
        command_content = Path(runtime["command_profile_file"]).read_bytes()
        self.store.store_artifact(
            result["id"],
            kind="command_profile",
            content=command_content,
            media_type="application/vnd.redrhex.command-profile+json",
            metadata={"sha256": goal.command_profile_sha256},
        )
        self._audit("autopilot_campaign_created", result["id"])
        return result

    @_serialized_mutation
    def update_draft(
        self,
        campaign_id: str,
        payload: Mapping[str, Any],
        *,
        expected_revision: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        self._require_enabled()
        goal, catalog, runtime = self._compile_draft(payload)
        result = self.store.update_draft(
            campaign_id,
            goal,
            catalog,
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
            runtime=runtime,
        )
        self._store_identity_artifacts(campaign_id, runtime)
        self.store.store_artifact(
            campaign_id,
            kind="command_profile",
            content=Path(runtime["command_profile_file"]).read_bytes(),
            media_type="application/vnd.redrhex.command-profile+json",
            metadata={"sha256": goal.command_profile_sha256},
        )
        return result

    @_serialized_mutation
    def arm_campaign(self, campaign_id: str, *, expected_revision: int, idempotency_key: str) -> dict[str, Any]:
        self._require_enabled()
        current = self.store.get_campaign(campaign_id)
        self._assert_campaign_identity(
            current,
            self.store.get_runtime(campaign_id),
            recheck_dependencies=True,
        )
        self._ensure_draft_artifacts(
            current,
            self.store.get_runtime(campaign_id),
        )
        snapshot = self.store.arm_campaign(
            campaign_id, expected_revision=expected_revision, idempotency_key=idempotency_key
        )
        self._audit("autopilot_campaign_armed", campaign_id, {"budget": snapshot["budget"]})
        self._notify()
        return snapshot

    @_serialized_mutation
    def pause_campaign(
        self,
        campaign_id: str,
        *,
        expected_revision: int,
        idempotency_key: str,
        reason: str = "operator request",
        advisor_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._require_enabled()
        metadata = self._advisor_metadata(advisor_metadata)
        current = self.store.get_campaign(campaign_id)
        if current["state"] in TERMINAL_STATES:
            return current
        return self.store.pause_campaign(
            campaign_id,
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
            reason=reason,
            advisor_metadata=metadata,
        )

    @_serialized_mutation
    def resume_campaign(self, campaign_id: str, *, expected_revision: int, idempotency_key: str) -> dict[str, Any]:
        self._require_enabled()
        result = self.store.resume_campaign(
            campaign_id, expected_revision=expected_revision, idempotency_key=idempotency_key
        )
        self._notify()
        return result

    @_serialized_mutation
    def stop_campaign(
        self,
        campaign_id: str,
        *,
        expected_revision: int,
        idempotency_key: str,
        after_current: bool = True,
        reason: str = "operator request",
        advisor_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._require_enabled()
        metadata = self._advisor_metadata(advisor_metadata)
        snapshot = self.store.get_campaign(campaign_id)
        if snapshot["state"] in TERMINAL_STATES:
            return snapshot
        active = snapshot.get("active_process") or {}
        if after_current:
            return self.store.stop_campaign(
                campaign_id,
                expected_revision=expected_revision,
                idempotency_key=idempotency_key,
                reason=reason,
                after_current=True,
                advisor_metadata=metadata,
            )

        if metadata:
            raise AutopilotValidationError(
                "advisor-scoped stop may only request stop after current work"
            )

        runtime = self.store.get_runtime(campaign_id)
        existing_intent = runtime.get("emergency_stop_intent")
        if isinstance(existing_intent, Mapping):
            if existing_intent.get("idempotency_key") != idempotency_key:
                raise AutopilotConflictError(
                    "another emergency stop is already pending",
                    current_revision=int(snapshot["revision"]),
                )
            return self._finish_emergency_stop(snapshot, runtime)

        if int(snapshot["revision"]) != expected_revision:
            raise AutopilotConflictError(
                "stale campaign revision",
                current_revision=int(snapshot["revision"]),
            )
        if active.get("process_id"):
            process_id = str(active["process_id"])
            process_kind = str(active.get("kind") or "")
            trial = self._current_trial(snapshot)
            record = self._reconcile_campaign_process(process_id)
            if trial is None or record is None or process_kind not in {"training", "evaluation"}:
                raise AutopilotConflictError(
                    "campaign process could not be resolved for an exact emergency stop",
                    current_revision=int(snapshot["revision"]),
                )
            gpu_accounting = {
                "trial_id": str(trial["id"]),
                "process_id": process_id,
                "process_kind": process_kind,
                "cumulative_gpu_hours": self._gpu_hours(record),
            }
            intent_snapshot = self.store.begin_emergency_stop(
                campaign_id,
                expected_revision=expected_revision,
                idempotency_key=idempotency_key,
                reason=reason,
                gpu_accounting=gpu_accounting,
            )
            return self._finish_emergency_stop(
                intent_snapshot, self.store.get_runtime(campaign_id)
            )
        return self.store.stop_campaign(
            campaign_id,
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
            reason=reason,
            after_current=False,
        )

    def _finish_emergency_stop(
        self,
        snapshot: Mapping[str, Any],
        runtime: Mapping[str, Any],
    ) -> dict[str, Any]:
        intent = runtime.get("emergency_stop_intent")
        if not isinstance(intent, Mapping):
            raise AutopilotConflictError(
                "campaign has no durable emergency stop intent",
                current_revision=int(snapshot["revision"]),
            )
        accounting = dict(intent.get("gpu_accounting") or {})
        process_id = str(accounting.get("process_id") or "")
        record = self._reconcile_campaign_process(process_id)
        if record is None:
            raise AutopilotConflictError(
                "campaign process could not be resolved for emergency stop recovery",
                current_revision=int(snapshot["revision"]),
            )
        status = str(record.get("status") or "").lower()
        if status in {"queued", "running", "stopping", ""}:
            try:
                stopped = self.processes.stop(process_id)
            except Exception as exc:
                raise AutopilotConflictError(
                    "campaign process could not be stopped",
                    current_revision=int(snapshot["revision"]),
                ) from exc
            if not stopped:
                raise AutopilotConflictError(
                    "campaign process could not be stopped",
                    current_revision=int(snapshot["revision"]),
                )
            record = self._reconcile_campaign_process(process_id) or record
            if str(record.get("status") or "").lower() in {
                "queued",
                "running",
                "stopping",
                "",
            }:
                return dict(snapshot)
        accounting["cumulative_gpu_hours"] = max(
            float(accounting.get("cumulative_gpu_hours") or 0.0),
            self._gpu_hours(record),
        )
        return self.store.stop_campaign(
            str(snapshot["id"]),
            expected_revision=int(snapshot["revision"]),
            idempotency_key=f"{intent['idempotency_key']}:finalize",
            reason=str(intent.get("reason") or "operator emergency stop"),
            after_current=False,
            gpu_accounting=accounting,
        )

    @staticmethod
    def _advisor_metadata(value: Any) -> dict[str, str]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise AutopilotValidationError("advisor_metadata must be an object")
        required = {
            "schema_version", "skill_version", "prompt_version", "declared_model", "reasoning_effort"
        }
        if set(value) != required:
            raise AutopilotValidationError("advisor_metadata has an unexpected schema")
        if value["schema_version"] != "redrhex.autopilot.advisor-metadata.v1":
            raise AutopilotValidationError("unsupported advisor metadata schema")
        result: dict[str, str] = {}
        for key in required:
            item = value[key]
            if not isinstance(item, str) or not item.strip() or len(item) > 200:
                raise AutopilotValidationError(f"advisor_metadata.{key} is invalid")
            result[key] = item.strip()
        return result

    @_serialized_mutation
    def advisor_heartbeat(
        self,
        campaign_id: str,
        *,
        advisor_metadata: Mapping[str, Any],
        expected_revision: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        self._require_enabled()
        metadata = self._advisor_metadata(advisor_metadata)
        snapshot = self.store.get_campaign(campaign_id)
        heartbeat_states = {
            "armed", "control_training", "control_evaluating", "candidate_training",
            "candidate_evaluating", "confirming",
        }
        if snapshot["state"] not in heartbeat_states:
            raise AutopilotConflictError(
                "Connector heartbeat is allowed only while campaign work is active",
                current_revision=int(snapshot["revision"]),
            )
        return self.store.record_connector_heartbeat(
            campaign_id,
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
            prompt_version=metadata["prompt_version"],
            skill_version=metadata["skill_version"],
            declared_model=metadata["declared_model"],
            reasoning_effort=metadata["reasoning_effort"],
            metadata_schema=metadata["schema_version"],
        )

    @_serialized_mutation
    def submit_decision(
        self,
        campaign_id: str,
        payload: Mapping[str, Any],
        *,
        expected_revision: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        self._require_enabled()
        advisor_metadata = self._advisor_metadata(payload.get("advisor_metadata"))
        if isinstance(payload.get("decision"), Mapping):
            decision_payload = payload["decision"]
        else:
            decision_payload = {
                key: value for key, value in payload.items()
                if key not in {"advisor_metadata", "patch_proposal"}
            }
        decision = AgentDecisionV1.from_dict(decision_payload)
        snapshot = self.store.get_campaign(campaign_id)
        runtime = self.store.get_runtime(campaign_id)
        request_is_current = int(snapshot["revision"]) == expected_revision
        if request_is_current and runtime.get("pending_candidate_profile"):
            raise AutopilotConflictError(
                "A validated candidate is already pending launch",
                current_revision=int(snapshot["revision"]),
            )

        pending_profile: dict[str, float] | None = None
        if request_is_current:
            available_evidence = {
                str(item.get("id")) for item in snapshot["evaluations"] if item.get("id")
            }
            available_evidence.update(
                item["id"] for item in self.store.list_artifacts(campaign_id)
            )
            unknown_evidence = sorted(set(decision.evidence_ids) - available_evidence)
            if unknown_evidence:
                raise AutopilotValidationError(
                    "decision references unknown campaign evidence: "
                    + ", ".join(unknown_evidence)
                )
            if decision.action == "propose_candidate":
                remaining = int(snapshot["budget"].get("remaining_training_trials", 0))
                reserved = int(snapshot["budget"].get("reserved_confirmation_trials", 0))
                if remaining <= reserved:
                    raise AutopilotBudgetError(
                        "insufficient budget for a screen plus reserved confirmation"
                    )
                pending_profile = validate_candidate_decision(
                    decision,
                    GoalSpecV1.from_dict(snapshot["goal"]),
                    tuple(
                        RewardCatalogEntryV1.from_dict(item)
                        for item in snapshot["reward_catalog"]
                    ),
                    dict(snapshot["leader"].get("reward_values") or {}),
                )
                pending_sha = sha256_json(pending_profile)
                if any(
                    trial.get("kind") == "candidate"
                    and trial.get("reward_profile_sha256") == pending_sha
                    for trial in snapshot.get("candidate_lineage") or []
                ):
                    raise AutopilotConflictError(
                        "This exact reward profile has already been attempted in the campaign",
                        current_revision=int(snapshot["revision"]),
                    )

        recorded = self.store.record_decision(
            campaign_id,
            decision,
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
            advisor_metadata=advisor_metadata,
        )
        self._audit(
            "autopilot_advisor_decision",
            campaign_id,
            {
                "action": decision.action,
                "reward_key": decision.reward_key,
                "evidence_ids": list(decision.evidence_ids),
                "validator_result": "accepted",
            },
        )
        current = self.store.get_campaign(campaign_id)
        current_runtime = self.store.get_runtime(campaign_id)
        recorded_revision = decision.campaign_revision + 1
        if current["state"] in TERMINAL_STATES:
            return current
        if decision.action == "pause":
            if current["state"] == "paused" or current["state"] in TERMINAL_STATES:
                return current
            return self.store.pause_campaign(
                campaign_id,
                expected_revision=int(current["revision"]),
                idempotency_key=f"controller-pause-{campaign_id}-{recorded_revision}",
                reason="advisor requested pause",
            )
        if decision.action == "request_patch_handoff":
            if current["state"] == "patch_handoff":
                return current
            patch_proposal = payload.get("patch_proposal")
            if patch_proposal is not None:
                raise AutopilotValidationError(
                    "enter patch_handoff first, then submit the proposal against its exposed base hashes"
                )
            self._store_patch_context(campaign_id, current)
            return self.store.transition_campaign(
                campaign_id,
                "patch_handoff",
                expected_revision=int(current["revision"]),
                idempotency_key=f"controller-patch-{campaign_id}-{recorded_revision}",
                reason="advisor requested an allowlisted source patch handoff",
                active_process=None,
            )

        already_reserved = any(
            int((trial.get("metadata") or {}).get("decision_revision") or -1)
            == recorded_revision
            for trial in current.get("candidate_lineage") or []
        )
        if current_runtime.get("pending_candidate_profile") or already_reserved:
            return current
        if pending_profile is None:
            pending_profile = validate_candidate_decision(
                decision,
                GoalSpecV1.from_dict(current["goal"]),
                tuple(
                    RewardCatalogEntryV1.from_dict(item)
                    for item in current["reward_catalog"]
                ),
                dict(current["leader"].get("reward_values") or {}),
            )

        updated = self.store.update_runtime(
            campaign_id,
            {
                "pending_candidate_profile": pending_profile,
                "pending_decision_revision": recorded_revision,
            },
            expected_revision=int(current["revision"]),
            idempotency_key=f"controller-candidate-{campaign_id}-{recorded_revision}",
            event_type="candidate_validated",
        )
        self._notify()
        return updated

    @_serialized_mutation
    def submit_patch_proposal(
        self,
        campaign_id: str,
        decision: Mapping[str, Any] | AgentDecisionV1,
        patch_proposal: Mapping[str, Any],
        *,
        advisor_metadata: Mapping[str, Any] | None,
        expected_revision: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        self._require_enabled()
        parsed_decision = (
            decision if isinstance(decision, AgentDecisionV1) else AgentDecisionV1.from_dict(decision)
        )
        metadata = self._advisor_metadata(advisor_metadata)
        patch_context = self._frozen_patch_context(campaign_id)
        proposal, content = self._patch_proposal_content(
            patch_proposal, patch_context=patch_context
        )
        result = self.store.record_patch_proposal(
            campaign_id,
            parsed_decision,
            content,
            media_type="application/vnd.redrhex.patch-proposal+json",
            artifact_metadata={"applied": False, "source_mutated": False},
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
            advisor_metadata=metadata,
            validate_before_commit=lambda: self._assert_patch_base_hashes(
                proposal, patch_context
            ),
        )
        self._audit(
            "autopilot_patch_proposal_recorded",
            campaign_id,
            {
                "artifact_id": result["patch_proposal_artifact"]["id"],
                "proposal_sha256": hashlib.sha256(content).hexdigest(),
                "applied": False,
            },
        )
        return result

    @staticmethod
    def _validate_patch_proposal(
        value: Any, *, patch_context: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise AutopilotValidationError("patch_proposal must be an object")
        required = {
            "schema_version",
            "target_symbols",
            "base_blob_hashes",
            "unified_diff",
            "rationale",
            "test_plan",
            "rollback_notes",
        }
        if set(value) != required:
            missing = sorted(required - set(value))
            unknown = sorted(set(value) - required)
            raise AutopilotValidationError(
                f"patch_proposal fields mismatch; missing={missing}, unknown={unknown}"
            )
        if value["schema_version"] != "redrhex.autopilot.patch-proposal.v1":
            raise AutopilotValidationError("unsupported patch proposal schema")
        symbols = value["target_symbols"]
        if not isinstance(symbols, list) or not symbols or len(symbols) > 32:
            raise AutopilotValidationError("patch proposal target_symbols must contain 1-32 symbols")
        if any(not isinstance(symbol, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,127}", symbol) is None for symbol in symbols):
            raise AutopilotValidationError("patch proposal contains an invalid target symbol")
        if patch_context is not None:
            allowed_symbols = set(patch_context.get("target_symbols") or [])
            if not allowed_symbols or set(symbols) - allowed_symbols:
                raise AutopilotValidationError(
                    "patch proposal contains a symbol outside the frozen handoff context"
                )
        hashes = value["base_blob_hashes"]
        canonical = {
            "source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py",
            "source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py",
        }
        if not isinstance(hashes, Mapping) or not hashes or set(hashes) - canonical:
            raise AutopilotValidationError("patch proposal base blobs are outside the reward-source allowlist")
        if any(
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in hashes.values()
        ):
            raise AutopilotValidationError("patch proposal base blob hashes must be lowercase SHA-256")
        diff = value["unified_diff"]
        if not isinstance(diff, str) or not diff.strip() or len(diff.encode("utf-8")) > 200_000:
            raise AutopilotValidationError("patch proposal unified_diff must be non-empty and at most 200 KB")
        if "GIT binary patch" in diff or "Binary files " in diff:
            raise AutopilotValidationError("binary patches are forbidden")
        header_paths: set[str] = set()
        current_path: str | None = None
        hunks: list[tuple[str, int, int]] = []
        for line in diff.splitlines():
            if line.startswith("--- "):
                raw = line[4:].split("\t", 1)[0]
                if raw == "/dev/null":
                    raise AutopilotValidationError("patch proposals may not create or delete source files")
                normalized = raw.removeprefix("a/").removeprefix("b/")
                header_paths.add(normalized)
            elif line.startswith("+++ "):
                raw = line[4:].split("\t", 1)[0]
                if raw == "/dev/null":
                    raise AutopilotValidationError("patch proposals may not create or delete source files")
                current_path = raw.removeprefix("a/").removeprefix("b/")
                header_paths.add(current_path)
            elif line.startswith("@@ "):
                match = re.match(r"@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@", line)
                if match is None or current_path is None:
                    raise AutopilotValidationError("patch proposal has an invalid unified-diff hunk")
                hunks.append(
                    (
                        current_path,
                        int(match.group(1)),
                        int(match.group(2) or "1"),
                    )
                )
        if not header_paths or header_paths - canonical:
            raise AutopilotValidationError("patch proposal diff touches a non-allowlisted file")
        if set(hashes) != header_paths:
            raise AutopilotValidationError(
                "patch proposal must provide exactly one frozen base hash for every touched file"
            )
        if not hunks:
            raise AutopilotValidationError("patch proposal contains no unified-diff hunks")
        if patch_context is not None:
            frozen_hashes = dict(patch_context.get("allowed_source_blobs") or {})
            if any(frozen_hashes.get(path) != digest for path, digest in hashes.items()):
                raise AutopilotValidationError(
                    "patch proposal base hashes do not match the frozen handoff context"
                )
            allowed_ranges: dict[str, list[tuple[int, int]]] = {}
            for snippet in patch_context.get("source_snippets") or []:
                if not isinstance(snippet, Mapping):
                    continue
                source = str(snippet.get("source") or "")
                try:
                    start = int(snippet["line_start"])
                    end = int(snippet["line_end"])
                except (KeyError, TypeError, ValueError):
                    continue
                allowed_ranges.setdefault(source, []).append((start, end))
            for path, start, count in hunks:
                end = start + max(count, 1) - 1
                if not any(
                    start >= allowed_start and end <= allowed_end
                    for allowed_start, allowed_end in allowed_ranges.get(path, [])
                ):
                    raise AutopilotValidationError(
                        "patch proposal hunk is outside the frozen reward-source snippets"
                    )
        for name in ("rationale", "test_plan", "rollback_notes"):
            text = value[name]
            if not isinstance(text, str) or not text.strip() or len(text) > 8000:
                raise AutopilotValidationError(f"patch proposal {name} must be non-empty and at most 8000 characters")
        return {
            "schema_version": value["schema_version"],
            "target_symbols": list(symbols),
            "base_blob_hashes": dict(hashes),
            "unified_diff": diff,
            "rationale": value["rationale"].strip(),
            "test_plan": value["test_plan"].strip(),
            "rollback_notes": value["rollback_notes"].strip(),
        }

    def _patch_proposal_content(
        self,
        value: Any,
        *,
        patch_context: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], bytes]:
        proposal = self._validate_patch_proposal(
            value, patch_context=patch_context
        )
        content = (
            json.dumps(proposal, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        return proposal, content

    def _assert_patch_base_hashes(
        self,
        proposal: Mapping[str, Any],
        patch_context: Mapping[str, Any],
    ) -> None:
        # A new proposal must bind to the current immutable source. The store
        # invokes this only after checking for an exact idempotent retry.
        for relative, expected in proposal["base_blob_hashes"].items():
            if patch_context["allowed_source_blobs"].get(relative) != expected:
                raise AutopilotValidationError(
                    f"patch proposal does not bind to frozen source {relative}"
                )
            actual = _sha256_file(self.paths.repo_root / relative)
            if actual != expected:
                raise AutopilotValidationError(f"patch proposal base hash is stale for {relative}")

    def _store_patch_context(self, campaign_id: str, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        existing = [
            item
            for item in self.store.list_artifacts(campaign_id)
            if item.get("kind") == "patch_context"
        ]
        if existing:
            return existing[0]
        context = {
            **self._reward_source_context(snapshot),
            "campaign_id": campaign_id,
            "campaign_revision": snapshot["revision"],
            "goal": snapshot["goal"],
            "leader": snapshot["leader"],
            "applies_automatically": False,
        }
        return self.store.store_artifact(
            campaign_id,
            kind="patch_context",
            content=(json.dumps(context, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
            media_type="application/vnd.redrhex.patch-context+json",
            metadata={"source_mutated": False},
        )

    # -- durable controller --------------------------------------------------

    def _worker_loop(self) -> None:
        try:
            interval = max(1.0, min(float(os.environ.get(AUTOPILOT_POLL_SECONDS_ENV, "5")), 60.0))
        except ValueError:
            interval = 5.0
        while not self._stop.is_set():
            self._wake.clear()
            self.tick()
            self._wake.wait(interval)

    @_serialized_mutation
    def tick(self) -> None:
        """Advance every active local campaign by at most one durable action."""

        if not self.enabled or not self._tick_lock.acquire(blocking=False):
            return
        try:
            for snapshot in self.store.list_controller_campaigns():
                if snapshot["state"] == "budget_exhausted" and snapshot.get("active_process"):
                    try:
                        accounted, _ = self._account_bound_active_process(snapshot, force=True)
                        self._enforce_budget_process_stop(accounted)
                    except (AutopilotConflictError, AutopilotBudgetError):
                        pass
                    continue
                if snapshot["state"] in TERMINAL_STATES or snapshot["state"] == "draft":
                    continue
                try:
                    self._tick_campaign(snapshot)
                except (AutopilotConflictError, AutopilotBudgetError):
                    # Another request advanced the optimistic revision. The
                    # next tick re-reads authoritative state.
                    continue
                except Exception as exc:
                    self._fail_campaign(snapshot["id"], exc)
        finally:
            self._tick_lock.release()

    def _tick_campaign(self, snapshot: Mapping[str, Any]) -> None:
        campaign_id = str(snapshot["id"])
        state = str(snapshot["state"])
        runtime = self.store.get_runtime(campaign_id)
        if runtime.get("controller_failure_stop_intent"):
            self._finish_controller_failure_stop(snapshot, runtime)
            return
        if runtime.get("emergency_stop_intent"):
            self._finish_emergency_stop(snapshot, runtime)
            return
        active = snapshot.get("active_process")
        if active:
            snapshot, gpu_budget_exhausted = self._account_bound_active_process(snapshot)
            if gpu_budget_exhausted:
                return
            state = str(snapshot["state"])
            active = snapshot.get("active_process")
        if state != "paused":
            self._assert_campaign_identity(snapshot, runtime)
        if state == "paused":
            if active:
                self._poll_active(snapshot, runtime, launch_next=False)
            return
        if state == "armed":
            self._ensure_control(snapshot, runtime)
            return
        if state in {"control_training", "candidate_training"}:
            self._poll_training_state(snapshot, runtime)
            return
        if state in {"control_evaluating", "candidate_evaluating"}:
            self._poll_evaluation_state(snapshot, runtime)
            return
        if state == "awaiting_advisor":
            remaining = int(snapshot["budget"].get("remaining_training_trials", 0))
            reserved = int(snapshot["budget"].get("reserved_confirmation_trials", 0))
            if remaining <= reserved:
                self._enter_patch_handoff(
                    snapshot,
                    "insufficient training budget for one screen and all confirmation trials",
                )
                return
            if runtime.get("pending_candidate_profile"):
                self._ensure_candidate(snapshot, runtime)
                return
            if self._recover_recorded_advisor_action(snapshot):
                return
            try:
                wait_seconds = max(
                    60.0,
                    float(os.environ.get(AUTOPILOT_ADVISOR_WAIT_SECONDS_ENV, "1800")),
                )
            except ValueError:
                wait_seconds = 1800.0
            updated_at = _iso_to_epoch(snapshot.get("updated_at"))
            if updated_at is not None and time.time() - updated_at >= wait_seconds:
                self.store.transition_campaign(
                    campaign_id,
                    "waiting_for_chatgpt",
                    expected_revision=int(snapshot["revision"]),
                    idempotency_key=f"controller-waiting-chatgpt-{campaign_id}-{snapshot['revision']}",
                    reason="no advisor decision arrived within the scheduled-task grace window",
                    active_process=None,
                )
            return
        if state == "waiting_for_chatgpt":
            if runtime.get("pending_candidate_profile"):
                self.store.transition_campaign(
                    campaign_id,
                    "awaiting_advisor",
                    expected_revision=int(snapshot["revision"]),
                    idempotency_key=(
                        f"controller-chatgpt-returned-{campaign_id}-{snapshot['revision']}"
                    ),
                    active_process=None,
                )
                self._notify()
                return
            if self._recover_recorded_advisor_action(snapshot):
                return
        if state == "confirming":
            self._ensure_confirmation(snapshot, runtime)

    def _poll_active(self, snapshot: Mapping[str, Any], runtime: Mapping[str, Any], *, launch_next: bool) -> None:
        active = snapshot.get("active_process") or {}
        if active.get("kind") == "training":
            self._poll_training_state(snapshot, runtime, launch_next=launch_next)
        elif active.get("kind") == "evaluation":
            self._poll_evaluation_state(snapshot, runtime, launch_next=launch_next)

    @staticmethod
    def _current_trial(snapshot: Mapping[str, Any]) -> dict[str, Any] | None:
        active = snapshot.get("active_process") or {}
        trial_id = active.get("trial_id")
        trials = list(snapshot.get("candidate_lineage") or [])
        if trial_id:
            return next((trial for trial in trials if trial.get("id") == trial_id), None)
        for trial in reversed(trials):
            if trial.get("status") in {"reserved", "queued", "training", "trained", "evaluating"}:
                return trial
        return None

    @staticmethod
    def _evaluated_trial_pending_transition(
        snapshot: Mapping[str, Any],
        runtime: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        evaluated = [
            trial for trial in snapshot.get("candidate_lineage") or []
            if trial.get("status") == "evaluated" and trial.get("evaluation_id")
        ]
        if snapshot.get("state") == "control_evaluating":
            return next(
                (
                    trial for trial in reversed(evaluated)
                    if trial.get("kind") == "control" and int(trial.get("seed") or -1) == 42
                ),
                None,
            )
        if snapshot.get("state") == "candidate_evaluating":
            if runtime.get("confirmation_winner_trial_id"):
                confirmation = next(
                    (
                        trial for trial in reversed(evaluated)
                        if str(trial.get("kind") or "").startswith("confirmation_")
                    ),
                    None,
                )
                if confirmation is not None:
                    return confirmation
            return next(
                (
                    trial for trial in reversed(evaluated)
                    if trial.get("kind") == "candidate"
                ),
                None,
            )
        pending = runtime.get("pending_candidate_profile")
        if isinstance(pending, Mapping):
            pending_sha = sha256_json(dict(pending))
            return next(
                (
                    trial for trial in reversed(evaluated)
                    if trial.get("kind") == "candidate"
                    and trial.get("reward_profile_sha256") == pending_sha
                ),
                None,
            )
        return next(
            (
                trial for trial in reversed(evaluated)
                if str(trial.get("kind") or "").startswith("confirmation_")
            ),
            None,
        )

    @staticmethod
    def _failed_evaluation_pending_terminal(
        snapshot: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Return the current failed evaluation before considering older evidence."""

        if snapshot.get("state") not in {
            "control_evaluating",
            "candidate_evaluating",
        }:
            return None
        return next(
            (
                trial
                for trial in reversed(snapshot.get("candidate_lineage") or [])
                if trial.get("status") == "failed"
                and trial.get("evaluation_process_id")
            ),
            None,
        )

    def _finish_failed_evaluation(
        self,
        snapshot: Mapping[str, Any],
        trial: Mapping[str, Any],
    ) -> None:
        """Fail closed after a durable evaluation-failure trial update."""

        if trial.get("status") != "failed":
            raise AutopilotValidationError(
                "evaluation failure recovery requires a failed trial"
            )
        if snapshot.get("state") == "paused":
            return
        if snapshot.get("state") not in {
            "control_evaluating",
            "candidate_evaluating",
        }:
            raise AutopilotValidationError(
                "failed evaluation is outside an evaluation lifecycle state"
            )
        metadata = dict(trial.get("metadata") or {})
        reason = metadata.get("evaluation_terminal_reason")
        if not isinstance(reason, str) or not reason.strip():
            reason = (
                "evaluation evidence was missing, malformed, or the evaluator failed"
            )
        self.store.transition_campaign(
            str(snapshot["id"]),
            "blocked_safety",
            expected_revision=int(snapshot["revision"]),
            idempotency_key=f"controller-evaluation-terminal-{trial['id']}",
            reason=reason.strip()[:1900],
            active_process=None,
        )

    def _recover_recorded_advisor_action(self, snapshot: Mapping[str, Any]) -> bool:
        """Finish an advisor action interrupted after its decision commit."""

        decisions = list(snapshot.get("decisions") or [])
        if not decisions:
            return False
        decision = AgentDecisionV1.from_dict(
            {
                key: value for key, value in decisions[-1].items()
                if key not in {"id", "created_at"}
            }
        )
        recorded_revision = decision.campaign_revision + 1
        if decision.action == "propose_candidate":
            already_reserved = any(
                int((trial.get("metadata") or {}).get("decision_revision") or -1)
                == recorded_revision
                for trial in snapshot.get("candidate_lineage") or []
            )
            if already_reserved:
                return False
            profile = validate_candidate_decision(
                decision,
                GoalSpecV1.from_dict(snapshot["goal"]),
                tuple(
                    RewardCatalogEntryV1.from_dict(item)
                    for item in snapshot["reward_catalog"]
                ),
                dict(snapshot["leader"].get("reward_values") or {}),
            )
            self.store.update_runtime(
                str(snapshot["id"]),
                {
                    "pending_candidate_profile": profile,
                    "pending_decision_revision": recorded_revision,
                },
                expected_revision=int(snapshot["revision"]),
                idempotency_key=(
                    f"controller-recover-candidate-{decision.campaign_id}-{recorded_revision}"
                ),
                event_type="candidate_validated",
            )
            self._notify()
            return True
        if decision.action == "pause":
            self.store.pause_campaign(
                str(snapshot["id"]),
                expected_revision=int(snapshot["revision"]),
                idempotency_key=(
                    f"controller-recover-pause-{decision.campaign_id}-{recorded_revision}"
                ),
                reason="advisor requested pause",
            )
            return True
        if decision.action == "request_patch_handoff":
            self._store_patch_context(str(snapshot["id"]), snapshot)
            self.store.transition_campaign(
                str(snapshot["id"]),
                "patch_handoff",
                expected_revision=int(snapshot["revision"]),
                idempotency_key=(
                    f"controller-recover-patch-{decision.campaign_id}-{recorded_revision}"
                ),
                reason="advisor requested an allowlisted source patch handoff",
                active_process=None,
            )
            return True
        return False

    @staticmethod
    def _trial_identity_metadata(
        snapshot: Mapping[str, Any],
        runtime: Mapping[str, Any],
        extra: Mapping[str, Any],
    ) -> dict[str, Any]:
        goal = snapshot["goal"]
        return {
            "task": goal["task"],
            "stage": goal["stage"],
            "evaluation_profile": goal["evaluation_profile"],
            "code_sha256": goal["code_sha256"],
            "config_sha256": goal["config_sha256"],
            "dependency_sha256": runtime["dependency_sha256"],
            "physics_profile_sha256": goal["physics_profile_sha256"],
            "spring_profile_sha256": goal["spring_profile_sha256"],
            "terrain_profile_sha256": runtime["terrain_profile_sha256"],
            "command_profile_sha256": goal["command_profile_sha256"],
            **dict(extra),
        }

    def _ensure_control(self, snapshot: Mapping[str, Any], runtime: Mapping[str, Any]) -> None:
        existing = next(
            (
                trial for trial in snapshot["candidate_lineage"]
                if trial["kind"] == "control" and int(trial["seed"]) == 42
            ),
            None,
        )
        if existing is None:
            result = self.store.reserve_trial(
                snapshot["id"],
                kind="control",
                seed=42,
                reward_profile=dict(snapshot["leader"]["reward_values"]),
                source_checkpoint_sha256=snapshot["goal"].get("checkpoint_sha256"),
                expected_revision=int(snapshot["revision"]),
                idempotency_key=f"controller-control-{snapshot['id']}-42",
                metadata=self._trial_identity_metadata(
                    snapshot,
                    runtime,
                    {"phase": "screen", "unchanged_control": True},
                ),
            )
            snapshot = result["campaign"]
            existing = result["trial"]
        if snapshot["state"] == "armed":
            snapshot = self.store.transition_campaign(
                snapshot["id"],
                "control_training",
                expected_revision=int(snapshot["revision"]),
                idempotency_key=f"controller-control-training-{snapshot['id']}-{existing['id']}",
            )
        self._launch_training_if_needed(snapshot, existing, runtime)

    def _ensure_candidate(self, snapshot: Mapping[str, Any], runtime: Mapping[str, Any]) -> None:
        profile = dict(runtime["pending_candidate_profile"])
        profile_sha = sha256_json(profile)
        existing = next(
            (
                trial for trial in snapshot["candidate_lineage"]
                if trial["kind"] == "candidate"
                and trial["seed"] == 42
                and trial["reward_profile_sha256"] == profile_sha
                and int((trial.get("metadata") or {}).get("decision_revision") or -1)
                == int(runtime.get("pending_decision_revision") or -2)
                and trial.get("status") in {"reserved", "queued", "training", "trained", "evaluating"}
            ),
            None,
        )
        if existing is None:
            result = self.store.reserve_trial(
                snapshot["id"],
                kind="candidate",
                seed=42,
                reward_profile=profile,
                source_checkpoint_sha256=snapshot["goal"].get("checkpoint_sha256"),
                expected_revision=int(snapshot["revision"]),
                idempotency_key=f"controller-screen-{snapshot['id']}-{profile_sha[:16]}",
                metadata=self._trial_identity_metadata(
                    snapshot,
                    runtime,
                    {"phase": "screen", "decision_revision": runtime.get("pending_decision_revision")},
                ),
            )
            snapshot, existing = result["campaign"], result["trial"]
        snapshot = self.store.transition_campaign(
            snapshot["id"],
            "candidate_training",
            expected_revision=int(snapshot["revision"]),
            idempotency_key=f"controller-candidate-training-{snapshot['id']}-{existing['id']}",
        )
        self._launch_training_if_needed(snapshot, existing, runtime)

    def _find_training_run(
        self,
        campaign_id: str,
        trial_id: str,
        *,
        exclude_run_id: str | None = None,
    ) -> dict[str, Any] | None:
        for run in self.history.list_runs():
            params = run.get("params") if isinstance(run.get("params"), dict) else {}
            if str(run.get("campaign_trial_id") or params.get("campaign_trial_id") or "") == trial_id and str(
                run.get("campaign_id") or params.get("campaign_id") or ""
            ) == campaign_id and run.get("source") == "training_panel" and str(
                run.get("id") or ""
            ) != str(exclude_run_id or ""):
                return run
        return None

    def _reconcile_campaign_process(self, process_id: str | None) -> dict[str, Any] | None:
        if not process_id:
            return None
        reconcile = getattr(self.processes, "reconcile_campaign_process", None)
        if callable(reconcile):
            reconcile(str(process_id))
        return self.history.get_run(str(process_id))

    def _validate_training_binding(
        self,
        snapshot: Mapping[str, Any],
        trial: Mapping[str, Any],
        runtime: Mapping[str, Any],
        run: Mapping[str, Any],
    ) -> dict[str, Any]:
        params = run.get("params") if isinstance(run.get("params"), dict) else {}
        expected_reward = _nested_reward_profile(trial["reward_profile"])
        expected = {
            "task": snapshot["goal"]["task"],
            "seed": trial["seed"],
            "max_iterations": snapshot["goal"]["per_trial_iteration_cap"],
            "curriculum_stage": snapshot["goal"]["stage"],
            "evaluation_profile": snapshot["goal"]["evaluation_profile"],
            "initialization_mode": "policy_only" if runtime.get("checkpoint_path") else "fresh",
            "strict_checkpoint_loading": True,
            "checkpoint_sha256": snapshot["goal"].get("checkpoint_sha256"),
            "reward_overrides": expected_reward,
            "terrain_overrides": dict(runtime.get("terrain_overrides") or {}),
            "physics_overrides": dict(runtime.get("physics_overrides") or {}),
            "campaign_id": snapshot["id"],
            "campaign_trial_id": trial["id"],
        }
        for key, value in expected.items():
            if params.get(key) != value:
                raise AutopilotValidationError(f"resolved training input mismatch: {key}")
        artifact_hashes: dict[str, Any] = {}
        for kind in ("reward", "terrain"):
            path = Path(str(run.get(f"{kind}_profile_file") or ""))
            recorded_sha = run.get(f"{kind}_profile_sha256")
            if not path.is_file() or not recorded_sha or _sha256_file(path) != recorded_sha:
                raise AutopilotValidationError(f"immutable {kind} profile binding is invalid")
            artifact_hashes[f"{kind}_profile_artifact_sha256"] = recorded_sha
        physics_file = run.get("physics_profile_file")
        if physics_file:
            path = Path(str(physics_file))
            if not path.is_file():
                raise AutopilotValidationError("immutable physics profile binding is invalid")
            artifact_hashes["physics_profile_artifact_sha256"] = _sha256_file(path)
        else:
            artifact_hashes["physics_profile_artifact_sha256"] = None
        return artifact_hashes

    def _launch_training_if_needed(
        self,
        snapshot: Mapping[str, Any],
        trial: Mapping[str, Any],
        runtime: Mapping[str, Any],
    ) -> None:
        self._assert_campaign_identity(
            snapshot, runtime, recheck_dependencies=True
        )
        campaign_id = str(snapshot["id"])
        trial_metadata = dict(trial.get("metadata") or {})
        retry_source_run_id = trial_metadata.get("retry_source_run_id")
        recovered = self._find_training_run(
            campaign_id,
            str(trial["id"]),
            exclude_run_id=(str(retry_source_run_id) if retry_source_run_id else None),
        )
        if recovered is None:
            goal = snapshot["goal"]
            checkpoint = runtime.get("checkpoint_path")
            params = TrainingParams(
                training_route="standard",
                task=str(goal["task"]),
                num_envs=int(runtime.get("num_envs") or self._default_num_envs()),
                max_iterations=int(goal["per_trial_iteration_cap"]),
                device=str(runtime.get("device") or "cuda:0"),
                spring_backend=str(runtime.get("spring_backend") or DEFAULT_PANEL_SPRING_BACKEND),
                headless=True,
                seed=int(trial["seed"]),
                resume=bool(checkpoint),
                checkpoint=str(checkpoint) if checkpoint else None,
                checkpoint_sha256=goal.get("checkpoint_sha256"),
                initialization_mode="policy_only" if checkpoint else "fresh",
                strict_checkpoint_loading=True,
                curriculum_stage=int(goal["stage"]),
                evaluation_profile=str(goal["evaluation_profile"]),
                reward_overrides=_nested_reward_profile(trial["reward_profile"]),
                terrain_overrides=dict(runtime.get("terrain_overrides") or {}),
                physics_overrides=dict(runtime.get("physics_overrides") or {}),
                campaign_id=campaign_id,
                campaign_trial_id=str(trial["id"]),
                requester_id="autopilot-controller",
                requester_label="Autopilot controller",
                display_name=f"Autopilot {campaign_id[-8:]} {trial['kind']} seed {trial['seed']}",
                folder=f"autopilot/{campaign_id}",
                client_request_id=f"autopilot:{campaign_id}:{trial['id']}:retry{trial.get('retry_count', 0)}",
            )
            recovered = self.processes.queue_training(params)
        binding_hashes = self._validate_training_binding(snapshot, trial, runtime, recovered)
        status = str(recovered.get("status") or "queued")
        result = self.store.update_trial(
            campaign_id,
            str(trial["id"]),
            expected_revision=int(snapshot["revision"]),
            idempotency_key=f"controller-bind-training-{trial['id']}-{recovered['id']}",
            status="training" if status == "running" else "queued",
            run_id=str(recovered["id"]),
            active_process={
                "kind": "training",
                "process_id": str(recovered["id"]),
                "trial_id": str(trial["id"]),
                "status": status,
                "started_at": recovered.get("started_at") or recovered.get("queued_at"),
            },
            metadata=binding_hashes,
            event_type="training_bound",
        )
        self._audit("autopilot_training_started", campaign_id, {"trial_id": trial["id"], "run_id": recovered["id"]})
        return None

    @staticmethod
    def _gpu_hours(record: Mapping[str, Any]) -> float:
        started = _iso_to_epoch(record.get("started_at") or record.get("created_at"))
        status = str(record.get("status") or "").lower()
        if status == "queued":
            return 0.0
        if status in {"running", "stopping", ""}:
            finished = time.time()
        else:
            finished = _iso_to_epoch(record.get("completed_at") or record.get("updated_at"))
        if started is None or finished is None:
            return 0.0
        return max(0.0, finished - started) / 3600.0

    def _account_gpu_attempt(
        self,
        snapshot: Mapping[str, Any],
        trial: Mapping[str, Any],
        record: Mapping[str, Any],
        *,
        process_kind: str,
        force: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any], bool]:
        """Charge one durable process high-water mark and enforce the cap."""

        process_id = str(record.get("id") or "")
        if not process_id:
            raise AutopilotValidationError("campaign GPU process has no durable identity")
        active = snapshot.get("active_process") or {}
        active_process_id = str(active.get("process_id") or "")
        if active_process_id and active_process_id != process_id:
            raise AutopilotValidationError("campaign active process identity changed")
        if not active_process_id:
            rebound = self.store.update_trial(
                str(snapshot["id"]),
                str(trial["id"]),
                expected_revision=int(snapshot["revision"]),
                idempotency_key=(
                    f"controller-rebind-active-{process_kind}-{process_id}-{snapshot['revision']}"
                ),
                status=str(trial["status"]),
                active_process={
                    "kind": process_kind,
                    "process_id": process_id,
                    "trial_id": str(trial["id"]),
                    "status": str(record.get("status") or "running"),
                    "started_at": record.get("started_at") or record.get("created_at"),
                },
                event_type=f"{process_kind}_process_rebound",
            )
            snapshot = rebound["campaign"]
            trial = rebound["trial"]
        cumulative_gpu_hours = self._gpu_hours(record)
        status = str(record.get("status") or "").lower()
        markers = dict((trial.get("metadata") or {}).get("gpu_process_accounting") or {})
        prior_marker = dict(markers.get(process_id) or {})
        accounted_gpu_hours = float(prior_marker.get("accounted_gpu_hours", 0.0))
        unaccounted_gpu_hours = max(0.0, cumulative_gpu_hours - accounted_gpu_hours)
        remaining_gpu_hours = float(snapshot["budget"].get("remaining_gpu_hours", 0.0))
        process_is_terminal = force or status not in {"running", "stopping", "queued", ""}
        if (
            not process_is_terminal
            and unaccounted_gpu_hours < GPU_ACCOUNTING_INTERVAL_HOURS
            and unaccounted_gpu_hours < remaining_gpu_hours
        ):
            return dict(snapshot), dict(trial), False
        accounting = self.store.account_process_gpu_usage(
            str(snapshot["id"]),
            str(trial["id"]),
            process_id=process_id,
            process_kind=process_kind,
            cumulative_gpu_hours=cumulative_gpu_hours,
            force=process_is_terminal,
            expected_revision=int(snapshot["revision"]),
            idempotency_key=(
                f"controller-gpu-account-{process_id}-{snapshot['revision']}-"
                f"{sha256_json({'gpu_hours': cumulative_gpu_hours})[:16]}"
            ),
        )
        campaign = accounting["campaign"]
        exhausted = campaign["state"] == "budget_exhausted"
        if exhausted:
            campaign = self._enforce_budget_process_stop(campaign, record)
        return campaign, accounting["trial"], exhausted

    def _enforce_budget_process_stop(
        self,
        snapshot: Mapping[str, Any],
        record: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Retry the exact durable budget stop until the process is terminal."""

        active = snapshot.get("active_process") or {}
        process_id = str(active.get("process_id") or "")
        if not process_id:
            return dict(snapshot)
        current = record or self._reconcile_campaign_process(process_id)
        status = str((current or {}).get("status") or "").lower()
        if status in {"running", "stopping", "queued", ""}:
            try:
                self.processes.stop(process_id)
            except Exception:
                # Keep active_process durable. A later controller tick (or a
                # restarted panel) retries this same campaign-owned stop.
                return dict(snapshot)
            current = self._reconcile_campaign_process(process_id) or current
            status = str((current or {}).get("status") or "").lower()
        if status in {"running", "stopping", "queued", ""}:
            return dict(snapshot)
        return self.store.clear_budget_exhausted_process(
            str(snapshot["id"]),
            process_id=process_id,
            expected_revision=int(snapshot["revision"]),
            idempotency_key=(
                f"controller-gpu-budget-stopped-{process_id}-{snapshot['revision']}"
            ),
        )

    def _account_bound_active_process(
        self,
        snapshot: Mapping[str, Any],
        *,
        force: bool = False,
    ) -> tuple[dict[str, Any], bool]:
        active = snapshot.get("active_process") or {}
        process_id = str(active.get("process_id") or "")
        process_kind = str(active.get("kind") or "")
        if not process_id or process_kind not in {"training", "evaluation"}:
            return dict(snapshot), False
        trial = self._current_trial(snapshot)
        if trial is None or str(active.get("trial_id") or "") != str(trial.get("id") or ""):
            raise AutopilotValidationError("active campaign process has no matching durable trial")
        record = self._reconcile_campaign_process(process_id)
        if record is None:
            return dict(snapshot), False
        campaign, _, exhausted = self._account_gpu_attempt(
            snapshot,
            trial,
            record,
            process_kind=process_kind,
            force=force,
        )
        return campaign, exhausted

    @staticmethod
    def _known_infrastructure_failure(record: Mapping[str, Any]) -> bool:
        failure_class = str(
            record.get("failure_class")
            or record.get("failure_kind")
            or record.get("failure_category")
            or ""
        ).strip().lower()
        if failure_class in {"configuration", "divergence", "safety", "evidence"}:
            return False
        if any(
            record.get(key)
            for key in (
                "configuration_failure",
                "divergence_detected",
                "safety_failure",
                "evidence_failure",
            )
        ):
            return False
        process_log_tail = ""
        process_log = record.get("process_log")
        if process_log:
            process_log_tail = tail_file(Path(str(process_log)), max_chars=50000)
        text = " ".join(
            str(record.get(key) or "")
            for key in ("failure_reason", "error", "process_log_tail")
        ) + " " + process_log_tail
        text = text.lower()
        markers = (
            "failed to create cuda context",
            "cuda driver version is insufficient",
            "connection reset by peer",
            "temporarily unavailable",
            "nucleus server",
            "asset server unavailable",
        )
        return any(marker in text for marker in markers)

    def _poll_training_state(
        self,
        snapshot: Mapping[str, Any],
        runtime: Mapping[str, Any],
        *,
        launch_next: bool = True,
    ) -> None:
        campaign_id = str(snapshot["id"])
        trial = self._current_trial(snapshot)
        if trial is None:
            raise RuntimeError("training state has no durable trial")
        if trial["status"] == "reserved":
            if launch_next:
                self._launch_training_if_needed(snapshot, trial, runtime)
            return
        if trial["status"] == "trained":
            if self._finish_stop_after_current(snapshot, runtime, trial, phase="training"):
                return
            if launch_next:
                self._start_trial_evaluation(snapshot, trial, runtime)
            return
        run_id = trial.get("run_id")
        retry_source_run_id = (trial.get("metadata") or {}).get("retry_source_run_id")
        run = self._reconcile_campaign_process(str(run_id)) if run_id else self._find_training_run(
            campaign_id,
            str(trial["id"]),
            exclude_run_id=(str(retry_source_run_id) if retry_source_run_id else None),
        )
        if run and not run_id:
            run = self._reconcile_campaign_process(str(run.get("id") or "")) or run
        if not run:
            # A crash may happen after queueing and before binding. Search one
            # more time on the next tick rather than launching a duplicate now.
            if not snapshot.get("active_process") and launch_next:
                self._launch_training_if_needed(snapshot, trial, runtime)
            return
        active_process_id = str((snapshot.get("active_process") or {}).get("process_id") or "")
        if active_process_id != str(run.get("id") or ""):
            snapshot, trial, gpu_budget_exhausted = self._account_gpu_attempt(
                snapshot,
                trial,
                run,
                process_kind="training",
            )
            if gpu_budget_exhausted:
                return
        status = str(run.get("status") or "").lower()
        if status in {"queued", "running", "stopping", ""}:
            return
        if status != "completed":
            if int(trial.get("retry_count") or 0) == 0 and self._known_infrastructure_failure(run):
                retry = self.store.update_trial(
                    campaign_id,
                    str(trial["id"]),
                    expected_revision=int(snapshot["revision"]),
                    idempotency_key=f"controller-infra-retry-{trial['id']}-{run.get('id')}",
                    status="reserved",
                    run_id=None,
                    retry_count=1,
                    active_process=None,
                    metadata={
                        "retry_reason": "classified infrastructure failure",
                        "retry_source_run_id": str(run.get("id") or ""),
                    },
                    event_type="infrastructure_retry_reserved",
                )
                if launch_next:
                    self._launch_training_if_needed(retry["campaign"], retry["trial"], runtime)
                return
            result = self.store.update_trial(
                campaign_id,
                str(trial["id"]),
                expected_revision=int(snapshot["revision"]),
                idempotency_key=f"controller-training-failed-{trial['id']}-{run.get('id')}",
                status="failed",
                active_process=None,
                metadata={"failure_reason": str(run.get("failure_reason") or f"training status {status}")},
                event_type="training_failed",
            )
            if snapshot["state"] == "paused":
                return
            self.store.transition_campaign(
                campaign_id,
                "failed",
                expected_revision=int(result["campaign"]["revision"]),
                idempotency_key=f"controller-training-terminal-{trial['id']}",
                reason="campaign training failed",
                active_process=None,
            )
            return

        # The process monitor must nominate the exact cap-bound output once.
        # Never consume History's display-only, directory-scanned "latest"
        # pointer for campaign evidence.
        checkpoint = run.get("output_checkpoint_path")
        recorded_checkpoint_sha = run.get("output_checkpoint_sha256")
        expected_iteration = int(snapshot["goal"]["per_trial_iteration_cap"]) - 1
        checkpoint_path = Path(str(checkpoint or ""))
        if not checkpoint_path.is_file():
            raise AutopilotValidationError(
                "completed training did not record an exact model checkpoint"
            )
        checkpoint_path = checkpoint_path.resolve()
        log_dir = Path(str(run.get("log_dir") or "")).resolve()
        if (
            checkpoint_path.parent != log_dir
            or checkpoint_path.name != f"model_{expected_iteration}.pt"
            or run.get("output_checkpoint_iteration") != expected_iteration
            or not isinstance(recorded_checkpoint_sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", recorded_checkpoint_sha) is None
            or _sha256_file(checkpoint_path) != recorded_checkpoint_sha
        ):
            raise AutopilotValidationError(
                "completed training checkpoint does not match its immutable completion receipt"
            )
        completed = self.store.complete_training(
            campaign_id,
            str(trial["id"]),
            output_checkpoint_path=str(checkpoint_path),
            output_checkpoint_sha256=recorded_checkpoint_sha,
            process_id=str(run["id"]),
            gpu_hours=self._gpu_hours(run),
            expected_revision=int(snapshot["revision"]),
            idempotency_key=f"controller-training-completed-{trial['id']}-{run['id']}",
        )
        campaign = completed["campaign"]
        if campaign["state"] in TERMINAL_STATES:
            return
        current_runtime = self.store.get_runtime(campaign_id)
        if current_runtime.get("stop_after_current"):
            self.store.stop_campaign(
                campaign_id,
                expected_revision=int(campaign["revision"]),
                idempotency_key=f"controller-stop-after-training-{trial['id']}",
                reason="stop-after-current completed",
                after_current=False,
            )
            return
        if snapshot["state"] == "paused" or not launch_next:
            return
        self._start_trial_evaluation(campaign, completed["trial"], current_runtime)

    def _find_evaluation_run(
        self,
        campaign_id: str,
        trial_id: str,
        *,
        exclude_process_id: str | None = None,
    ) -> dict[str, Any] | None:
        for run in self.history.list_runs():
            params = run.get("params") if isinstance(run.get("params"), dict) else {}
            if (
                run.get("source") == "autopilot_evaluation"
                and str(run.get("id") or "") != str(exclude_process_id or "")
                and str(run.get("campaign_id") or params.get("campaign_id") or "") == campaign_id
                and str(run.get("campaign_trial_id") or params.get("campaign_trial_id") or "") == trial_id
            ):
                return run
        return None

    @staticmethod
    def _evaluation_launch_is_gpu_contention(error: Exception) -> bool:
        if isinstance(error, GpuHostLeaseBusy):
            return True
        if not isinstance(error, ProcessStartError):
            return False
        payload = error.payload if isinstance(error.payload, Mapping) else {}
        return payload.get("code") == "gpu_host_lease_busy" or "running_processes" in payload

    def _start_trial_evaluation(
        self,
        snapshot: Mapping[str, Any],
        trial: Mapping[str, Any],
        runtime: Mapping[str, Any],
    ) -> None:
        self._assert_campaign_identity(
            snapshot, runtime, recheck_dependencies=True
        )
        campaign_id = str(snapshot["id"])
        state = str(snapshot["state"])
        target = "control_evaluating" if trial["kind"] == "control" else "candidate_evaluating"
        if state != target:
            snapshot = self.store.transition_campaign(
                campaign_id,
                target,
                expected_revision=int(snapshot["revision"]),
                idempotency_key=f"controller-evaluating-state-{trial['id']}",
            )
        training_run = self.history.get_run(str(trial.get("run_id") or ""))
        if not training_run:
            raise RuntimeError("trained trial run record is unavailable")
        checkpoint = dict(trial.get("metadata") or {}).get("output_checkpoint_path")
        checkpoint_path = Path(str(checkpoint or ""))
        if not checkpoint_path.is_file() or _sha256_file(checkpoint_path) != trial["output_checkpoint_sha256"]:
            raise AutopilotValidationError(
                "frozen trained checkpoint changed before evaluation"
            )
        physics_file = training_run.get("physics_profile_file")
        frozen_physics_sha = dict(trial.get("metadata") or {}).get(
            "physics_profile_artifact_sha256"
        )
        if frozen_physics_sha is not None and (
            not physics_file
            or not Path(str(physics_file)).is_file()
            or _sha256_file(Path(str(physics_file))) != frozen_physics_sha
        ):
            raise AutopilotValidationError(
                "frozen training physics profile changed before evaluation"
            )
        params = EvaluationParams(
            source_run_id=str(training_run["id"]),
            checkpoint=str(checkpoint_path.resolve()),
            checkpoint_sha256=str(trial["output_checkpoint_sha256"]),
            task=str(snapshot["goal"]["task"]),
            agent_entry_point="rsl_rl_cfg_entry_point",
            seed=int(trial["seed"]),
            evaluation_profile=str(snapshot["goal"]["evaluation_profile"]),
            curriculum_stage=int(snapshot["goal"]["stage"]),
            command_profile_file=str(runtime["command_profile_file"]),
            command_profile_sha256=str(snapshot["goal"]["command_profile_sha256"]),
            code_sha256=str(snapshot["goal"]["code_sha256"]),
            config_sha256=str(snapshot["goal"]["config_sha256"]),
            dependency_sha256=str(runtime["dependency_sha256"]),
            reward_profile_sha256=str(trial["reward_profile_sha256"]),
            physics_identity_sha256=str(snapshot["goal"]["physics_profile_sha256"]),
            spring_identity_sha256=str(snapshot["goal"]["spring_profile_sha256"]),
            terrain_profile_sha256=str(runtime["terrain_profile_sha256"]),
            num_envs=int(runtime.get("num_envs") or self._default_num_envs()),
            device=str(runtime.get("device") or "cuda:0"),
            spring_backend=str(runtime.get("spring_backend") or DEFAULT_PANEL_SPRING_BACKEND),
            physics_profile_file=physics_file,
            campaign_id=campaign_id,
            campaign_trial_id=str(trial["id"]),
        )
        evaluation_input_sha256 = sha256_json(vars(params))
        trial_metadata = dict(trial.get("metadata") or {})
        prior_input_sha256 = trial_metadata.get("evaluation_input_sha256")
        if prior_input_sha256 and prior_input_sha256 != evaluation_input_sha256:
            raise AutopilotValidationError("evaluation retry inputs changed after the first attempt")
        excluded_process_id = trial_metadata.get("evaluation_retry_source_process_id")
        evaluation = self._find_evaluation_run(
            campaign_id,
            str(trial["id"]),
            exclude_process_id=(str(excluded_process_id) if excluded_process_id else None),
        )
        if evaluation is None:
            try:
                evaluation = self.processes.start_evaluation(params)
            except (GpuHostLeaseBusy, ProcessStartError) as exc:
                if not self._evaluation_launch_is_gpu_contention(exc):
                    raise
                if trial_metadata.get("evaluation_wait_reason") != "gpu_host_contention":
                    self.store.update_trial(
                        campaign_id,
                        str(trial["id"]),
                        expected_revision=int(snapshot["revision"]),
                        idempotency_key=f"controller-evaluation-gpu-wait-{trial['id']}",
                        status="trained",
                        evaluation_process_id=None,
                        active_process=None,
                        metadata={
                            "evaluation_input_sha256": evaluation_input_sha256,
                            "evaluation_wait_reason": "gpu_host_contention",
                        },
                        event_type="evaluation_waiting_for_gpu",
                    )
                return
        result = self.store.update_trial(
            campaign_id,
            str(trial["id"]),
            expected_revision=int(snapshot["revision"]),
            idempotency_key=f"controller-bind-evaluation-{trial['id']}-{evaluation['id']}",
            status="evaluating",
            evaluation_process_id=str(evaluation["id"]),
            active_process={
                "kind": "evaluation",
                "process_id": str(evaluation["id"]),
                "trial_id": str(trial["id"]),
                "status": str(evaluation.get("status") or "running"),
                "started_at": evaluation.get("started_at") or evaluation.get("created_at"),
            },
            metadata={
                "evaluation_input_sha256": evaluation_input_sha256,
                "evaluation_wait_reason": None,
            },
            event_type="evaluation_bound",
        )
        self._audit(
            "autopilot_evaluation_started",
            campaign_id,
            {"trial_id": trial["id"], "evaluation_process_id": evaluation["id"]},
        )

    def _poll_evaluation_state(
        self,
        snapshot: Mapping[str, Any],
        runtime: Mapping[str, Any],
        *,
        launch_next: bool = True,
    ) -> None:
        campaign_id = str(snapshot["id"])
        trial = self._current_trial(snapshot)
        if trial is None:
            failed_trial = self._failed_evaluation_pending_terminal(snapshot)
            if failed_trial is not None:
                self._finish_failed_evaluation(snapshot, failed_trial)
                return
            trial = self._evaluated_trial_pending_transition(snapshot, runtime)
        if trial is None:
            raise RuntimeError("evaluation state has no durable trial")
        if trial["status"] == "failed":
            self._finish_failed_evaluation(snapshot, trial)
            return
        if trial["status"] == "evaluated":
            if self._finish_stop_after_current(snapshot, runtime, trial, phase="evaluation"):
                return
            stored = next(
                (
                    item for item in snapshot.get("evaluations") or []
                    if item.get("trial_id") == trial["id"]
                ),
                None,
            )
            if stored is None:
                raise RuntimeError("evaluated trial has no durable evaluation report")
            self._verify_evaluation_artifacts(campaign_id, stored)
            self._after_evaluation(
                snapshot,
                trial,
                EvaluationReportV1.from_dict(stored),
                runtime,
            )
            return
        if trial["status"] == "trained":
            if launch_next:
                self._start_trial_evaluation(snapshot, trial, runtime)
            return
        evaluation = self._reconcile_campaign_process(
            str(trial.get("evaluation_process_id") or "")
        )
        if evaluation is None:
            excluded_process_id = (trial.get("metadata") or {}).get(
                "evaluation_retry_source_process_id"
            )
            evaluation = self._find_evaluation_run(
                campaign_id,
                str(trial["id"]),
                exclude_process_id=(
                    str(excluded_process_id) if excluded_process_id else None
                ),
            )
            if evaluation:
                evaluation = self._reconcile_campaign_process(
                    str(evaluation.get("id") or "")
                ) or evaluation
        if not evaluation:
            if launch_next:
                self._start_trial_evaluation(snapshot, trial, runtime)
            return
        active_process_id = str((snapshot.get("active_process") or {}).get("process_id") or "")
        if active_process_id != str(evaluation.get("id") or ""):
            snapshot, trial, gpu_budget_exhausted = self._account_gpu_attempt(
                snapshot,
                trial,
                evaluation,
                process_kind="evaluation",
            )
            if gpu_budget_exhausted:
                return
        status = str(evaluation.get("status") or "").lower()
        if status in {"running", "queued", "stopping", ""}:
            return
        if status != "completed":
            retry_count = int(trial.get("evaluation_retry_count") or 0)
            if retry_count == 0 and self._known_infrastructure_failure(evaluation):
                reason = str(
                    evaluation.get("failure_reason")
                    or evaluation.get("error")
                    or f"evaluation status {status}"
                )
                self.store.update_trial(
                    campaign_id,
                    str(trial["id"]),
                    expected_revision=int(snapshot["revision"]),
                    idempotency_key=(
                        f"controller-evaluation-infra-retry-{trial['id']}-{evaluation.get('id')}"
                    ),
                    status="trained",
                    evaluation_process_id=None,
                    active_process=None,
                    metadata={
                        "evaluation_retry_count": 1,
                        "evaluation_retry_reason": reason,
                        "evaluation_retry_source_process_id": str(evaluation.get("id") or ""),
                    },
                    event_type="evaluation_infrastructure_retry_reserved",
                )
                return
            terminal_reason = (
                "evaluation evidence was missing, malformed, or the evaluator failed"
            )
            result = self.store.update_trial(
                campaign_id,
                str(trial["id"]),
                expected_revision=int(snapshot["revision"]),
                idempotency_key=f"controller-evaluation-failed-{trial['id']}-{evaluation.get('id')}",
                status="failed",
                active_process=None,
                metadata={
                    "failure_reason": str(
                        evaluation.get("failure_reason")
                        or f"evaluation status {status}"
                    ),
                    "evaluation_terminal_reason": terminal_reason,
                },
                event_type="evaluation_failed",
            )
            if snapshot["state"] != "paused":
                self._finish_failed_evaluation(
                    result["campaign"], result["trial"]
                )
            return
        try:
            report = self._parse_evaluation_report(snapshot, trial, evaluation, runtime)
            original = dict(runtime["original_reward_values"])
            evaluated = evaluate_report(
                report,
                GoalSpecV1.from_dict(snapshot["goal"]),
                expected_trial_checkpoint_sha256=str(trial["output_checkpoint_sha256"]),
                original_reward_values=original,
                candidate_reward_values=trial["reward_profile"],
            )
        except (AutopilotValidationError, OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            self._block_invalid_evidence(snapshot, trial, evaluation, exc)
            return
        report_artifact = self.store.store_artifact(
            campaign_id,
            kind="evaluation_report",
            content=canonical_json_bytes(
                self._compact_evaluation_report_payload(evaluated)
            ),
            media_type="application/vnd.redrhex.autopilot-evaluation-report+json",
            metadata={
                "trial_id": trial["id"],
                "evaluation_id": evaluated.id,
            },
        )
        evaluated = replace(
            evaluated,
            artifact_ids=(*evaluated.artifact_ids, str(report_artifact["id"])),
        )
        leader = self._leader_after_evaluation(snapshot, trial, evaluated)
        recorded = self.store.record_evaluation(
            campaign_id,
            str(trial["id"]),
            evaluated,
            expected_revision=int(snapshot["revision"]),
            idempotency_key=f"controller-evaluation-recorded-{trial['id']}-{evaluated.id}",
            leader=leader,
            gpu_hours=self._gpu_hours(evaluation),
            process_id=str(evaluation.get("id") or ""),
        )
        campaign = recorded
        if campaign["state"] in TERMINAL_STATES:
            return
        current_runtime = self.store.get_runtime(campaign_id)
        if current_runtime.get("stop_after_current"):
            self.store.stop_campaign(
                campaign_id,
                expected_revision=int(campaign["revision"]),
                idempotency_key=f"controller-stop-after-evaluation-{trial['id']}",
                reason="stop-after-current completed",
                after_current=False,
            )
            return
        if snapshot["state"] == "paused" or not launch_next:
            return
        self._after_evaluation(campaign, trial, evaluated, current_runtime)

    def _finish_stop_after_current(
        self,
        snapshot: Mapping[str, Any],
        runtime: Mapping[str, Any],
        trial: Mapping[str, Any],
        *,
        phase: str,
    ) -> bool:
        """Honor a durable boundary stop before any recovered follow-up launch."""

        if not runtime.get("stop_after_current"):
            return False
        self.store.stop_campaign(
            str(snapshot["id"]),
            expected_revision=int(snapshot["revision"]),
            idempotency_key=f"controller-stop-after-{phase}-{trial['id']}",
            reason="stop-after-current completed",
            after_current=False,
        )
        return True

    def _block_invalid_evidence(
        self,
        snapshot: Mapping[str, Any],
        trial: Mapping[str, Any],
        evaluation: Mapping[str, Any],
        error: Exception,
    ) -> None:
        campaign_id = str(snapshot["id"])
        reason = f"invalid evaluation evidence: {type(error).__name__}: {error}"[:1900]
        failed = self.store.update_trial(
            campaign_id,
            str(trial["id"]),
            expected_revision=int(snapshot["revision"]),
            idempotency_key=(
                f"controller-evidence-invalid-{trial['id']}-{evaluation.get('id')}"
            ),
            status="failed",
            active_process=None,
            metadata={
                "failure_reason": reason,
                "evaluation_terminal_reason": reason,
            },
            event_type="evaluation_evidence_rejected",
        )
        if snapshot["state"] == "paused":
            return
        self._finish_failed_evaluation(failed["campaign"], failed["trial"])

    @staticmethod
    def _read_csv_bytes(content: bytes, name: str) -> list[dict[str, str]]:
        try:
            text = content.decode("utf-8")
            reader = csv.DictReader(io.StringIO(text, newline=""))
            fieldnames = reader.fieldnames
            if (
                not fieldnames
                or any(not isinstance(field, str) or not field.strip() for field in fieldnames)
                or len(set(fieldnames)) != len(fieldnames)
            ):
                raise AutopilotValidationError(
                    f"evaluation CSV has blank or duplicate headers: {name}"
                )
            rows = list(reader)
        except (csv.Error, UnicodeError) as exc:
            raise AutopilotValidationError(f"malformed evaluation CSV: {name}") from exc
        if not rows:
            raise AutopilotValidationError(f"evaluation CSV is empty: {name}")
        if any(None in row or None in row.values() for row in rows):
            raise AutopilotValidationError(f"evaluation CSV has malformed columns: {name}")
        return rows

    @staticmethod
    def _compact_evaluation_report_payload(
        report: EvaluationReportV1,
    ) -> dict[str, Any]:
        payload = report.to_dict()
        payload["episode_metrics"] = [
            {
                "schema_version": "redrhex.autopilot.episode-evidence-index.v1",
                "row_count": len(report.episode_metrics),
                "artifact_sha256": report.episode_artifact_sha256,
                "artifact_ids": list(report.artifact_ids),
            }
        ]
        return payload

    @staticmethod
    def _summary_map(rows: Sequence[Mapping[str, str]]) -> dict[str, str]:
        result: dict[str, str] = {}
        for index, row in enumerate(rows):
            if set(row) != {"metric", "value"}:
                raise AutopilotValidationError(
                    f"evaluation summary row {index} has an unexpected schema"
                )
            key = str(row.get("metric") or "")
            if not key or key in result:
                raise AutopilotValidationError("evaluation summary has a blank or duplicate metric")
            result[key] = str(row.get("value") or "")
        return result

    def _verify_evaluation_artifacts(
        self,
        campaign_id: str,
        report: Mapping[str, Any],
    ) -> None:
        """Re-open all content-addressed evidence before a delayed decision."""

        try:
            artifacts: dict[str, tuple[dict[str, Any], bytes]] = {}
            for artifact_id in report.get("artifact_ids") or []:
                metadata, content = self.store.get_artifact(
                    campaign_id, str(artifact_id)
                )
                artifacts[str(metadata.get("kind") or "")] = (metadata, content)
            required = {
                "evaluation_commands",
                "evaluation_episodes",
                "evaluation_summary",
                "evaluation_report",
            }
            if set(artifacts) != required:
                raise AutopilotValidationError(
                    "evaluation report does not reference the complete immutable evidence set"
                )
            command_meta, _command_content = artifacts["evaluation_commands"]
            episode_meta, _episode_content = artifacts["evaluation_episodes"]
            _summary_meta, summary_content = artifacts["evaluation_summary"]
            _report_meta, report_content = artifacts["evaluation_report"]
            summary = self._summary_map(
                self._read_csv_bytes(summary_content, "immutable evaluation summary")
            )
            if (
                summary.get("artifact.command_csv_sha256")
                != command_meta.get("sha256")
                or summary.get("artifact.episode_csv_sha256")
                != episode_meta.get("sha256")
                or report.get("episode_artifact_sha256")
                != episode_meta.get("sha256")
            ):
                raise AutopilotValidationError(
                    "evaluation report artifact identity mismatch"
                )
            evidence_ids = [
                str(metadata["id"])
                for kind, (metadata, _content) in artifacts.items()
                if kind != "evaluation_report"
            ]
            expected_report = dict(report)
            expected_report["artifact_ids"] = evidence_ids
            episode_index = expected_report.get("episode_metrics")
            if (
                isinstance(episode_index, list)
                and len(episode_index) == 1
                and isinstance(episode_index[0], Mapping)
                and episode_index[0].get("schema_version")
                == "redrhex.autopilot.episode-evidence-index.v1"
            ):
                expected_report["episode_metrics"] = [
                    {**dict(episode_index[0]), "artifact_ids": evidence_ids}
                ]
            if canonical_json_bytes(expected_report) != report_content:
                raise AutopilotValidationError(
                    "immutable evaluation report does not match the durable report"
                )
        except AutopilotValidationError:
            raise
        except (AutopilotStoreError, OSError, UnicodeError, csv.Error, ValueError) as exc:
            raise AutopilotValidationError(
                "evaluation report evidence artifact is missing or corrupt"
            ) from exc

    @staticmethod
    def _episode_row(row: Mapping[str, str], index: int) -> dict[str, Any]:
        required = {
            "command", "skill", "environment_index", "episode_index", "complete", "sample_count",
            "fall_count", "mae_vx", "mae_vy", "mae_wz", "success_ratio",
            "energy_mech_power_total_mean", "energy_effort_mean",
        }
        if set(row) != required:
            raise AutopilotValidationError(f"episode evidence row {index} has an unexpected schema")
        command = str(row["command"]).strip()
        skill = str(row["skill"]).strip()
        if not command or skill not in {"forward", "lateral", "diagonal", "yaw"}:
            raise AutopilotValidationError(f"episode evidence row {index} has invalid identity")
        integers: dict[str, int] = {}
        for key in ("environment_index", "episode_index", "sample_count", "fall_count"):
            try:
                parsed = int(str(row[key]))
            except ValueError as exc:
                raise AutopilotValidationError(f"episode evidence row {index}.{key} must be an integer") from exc
            if parsed < 0:
                raise AutopilotValidationError(f"episode evidence row {index}.{key} must be non-negative")
            integers[key] = parsed
        if integers["sample_count"] <= 0:
            raise AutopilotValidationError("episode evidence sample_count must be positive")
        parsed = {
            "command": command,
            "skill": skill,
            **integers,
            "complete": _csv_bool(row["complete"], f"episode[{index}].complete"),
            "mae_vx": _finite_float(row["mae_vx"], f"episode[{index}].mae_vx"),
            "mae_vy": _finite_float(row["mae_vy"], f"episode[{index}].mae_vy"),
            "mae_wz": _finite_float(row["mae_wz"], f"episode[{index}].mae_wz"),
            "success_ratio": _finite_float(row["success_ratio"], f"episode[{index}].success_ratio"),
            "energy_mech_power_total_mean": _finite_float(
                row["energy_mech_power_total_mean"],
                f"episode[{index}].energy_mech_power_total_mean",
            ),
            "energy_effort_mean": _finite_float(
                row["energy_effort_mean"],
                f"episode[{index}].energy_effort_mean",
            ),
        }
        for key in (
            "mae_vx", "mae_vy", "mae_wz", "energy_mech_power_total_mean", "energy_effort_mean"
        ):
            if parsed[key] < 0.0:
                raise AutopilotValidationError(
                    f"episode evidence row {index}.{key} must be non-negative"
                )
        if not 0.0 <= parsed["success_ratio"] <= 1.0:
            raise AutopilotValidationError(
                f"episode evidence row {index}.success_ratio must be between 0 and 1"
            )
        if parsed["fall_count"] not in {0, 1}:
            raise AutopilotValidationError(
                f"episode evidence row {index}.fall_count must be 0 or 1"
            )
        if not parsed["complete"] and parsed["fall_count"]:
            raise AutopilotValidationError(
                f"episode evidence row {index} cannot record a fall on an incomplete episode"
            )
        return parsed

    @staticmethod
    def _command_row(
        row: Mapping[str, str], index: int, *, evaluation_duration_s: float
    ) -> dict[str, Any]:
        required = {
            "command", "skill", "cmd_vx", "cmd_vy", "cmd_wz", "mae_vx", "mae_vy", "mae_wz",
            "actual_forward_speed_mean", "actual_lateral_leak_mean", "actual_yaw_leak_mean",
            "success_duration_s", "success_ratio", "success_vy_duration_s", "success_wz_duration_s",
            "diag_sign_match_ratio", "yaw_tilt_ok_ratio", "fall_rate", "energy_mech_power_main_mean",
            "energy_mech_power_total_mean", "energy_cost_of_transport_proxy", "energy_spring_energy_mean",
            "energy_spring_release_power_mean", "energy_spring_store_power_mean",
            "energy_spring_recovery_ratio", "energy_motion_speed_mean", "energy_progress_speed_mean",
            "energy_cost_mean", "energy_progress_distance_mean", "energy_per_distance",
            "energy_power_per_motion", "tracking_quality", "stability_quality", "score", "accept_pass",
        }
        if set(row) != required:
            raise AutopilotValidationError(f"command evidence row {index} has an unexpected schema")
        skill = str(row["skill"]).strip()
        if skill not in {"forward", "lateral", "diagonal", "yaw"}:
            raise AutopilotValidationError(f"command evidence row {index} has an invalid skill")
        numeric = {
            key: _finite_float(value, f"command[{index}].{key}")
            for key, value in row.items()
            if key not in {"command", "skill", "accept_pass"}
        }
        if numeric["success_duration_s"] < 0.0:
            raise AutopilotValidationError(
                f"command evidence row {index}.success_duration_s must be non-negative"
            )
        if not math.isclose(
            numeric["success_duration_s"],
            numeric["success_ratio"] * evaluation_duration_s,
            rel_tol=1.0e-5,
            abs_tol=1.0e-6,
        ):
            raise AutopilotValidationError(
                f"command evidence row {index} success duration/ratio mismatch"
            )
        for key in ("success_vy_duration_s", "success_wz_duration_s"):
            if not 0.0 <= numeric[key] <= evaluation_duration_s + 1.0e-6:
                raise AutopilotValidationError(
                    f"command evidence row {index}.{key} is outside the frozen duration"
                )
        for key in ("success_ratio", "diag_sign_match_ratio", "yaw_tilt_ok_ratio", "fall_rate"):
            if not 0.0 <= numeric[key] <= 1.0:
                raise AutopilotValidationError(
                    f"command evidence row {index}.{key} must be between 0 and 1"
                )
        if not math.isclose(
            numeric["energy_power_per_motion"],
            numeric["energy_per_distance"],
            rel_tol=1.0e-5,
            abs_tol=1.0e-6,
        ):
            raise AutopilotValidationError(
                f"command evidence row {index} energy alias mismatch"
            )
        if skill == "yaw" and not math.isclose(
            numeric["energy_per_distance"],
            numeric["energy_mech_power_total_mean"],
            rel_tol=1.0e-5,
            abs_tol=1.0e-6,
        ):
            raise AutopilotValidationError(
                f"command evidence row {index} pure-yaw energy must equal absolute mechanical power"
            )
        cmd_vx, cmd_vy, cmd_wz = numeric["cmd_vx"], numeric["cmd_vy"], numeric["cmd_wz"]
        if skill == "forward":
            expected_tracking = max(0.0, 1.0 - numeric["mae_vx"] / max(1e-6, abs(cmd_vx)))
        elif skill == "lateral":
            expected_tracking = max(0.0, 1.0 - numeric["mae_vy"] / max(1e-6, abs(cmd_vy)))
        elif skill == "diagonal":
            quality_x = max(0.0, 1.0 - numeric["mae_vx"] / max(1e-6, abs(cmd_vx)))
            quality_y = max(0.0, 1.0 - numeric["mae_vy"] / max(1e-6, abs(cmd_vy)))
            expected_tracking = 0.5 * (quality_x + quality_y)
        else:
            expected_tracking = max(0.0, 1.0 - numeric["mae_wz"] / max(1e-6, abs(cmd_wz)))
        expected_tracking = min(1.0, expected_tracking)
        expected_stability = min(1.0, max(0.0, 1.0 - numeric["fall_rate"] / 0.20))
        expected_accept = numeric["success_duration_s"] >= 2.0 and numeric["fall_rate"] <= 0.20
        if skill == "diagonal":
            expected_accept = expected_accept and numeric["diag_sign_match_ratio"] >= 0.70
        if skill == "yaw":
            expected_accept = expected_accept and numeric["yaw_tilt_ok_ratio"] >= 0.70
        reported_accept = _csv_bool(row["accept_pass"], f"command[{index}].accept_pass")
        for name, reported, expected in (
            ("tracking_quality", numeric["tracking_quality"], expected_tracking),
            ("stability_quality", numeric["stability_quality"], expected_stability),
        ):
            if not math.isclose(reported, expected, rel_tol=1.0e-5, abs_tol=1.0e-6):
                raise AutopilotValidationError(
                    f"command evidence row {index} derived {name} mismatch"
                )
        if reported_accept != expected_accept:
            raise AutopilotValidationError(
                f"command evidence row {index} derived accept_pass mismatch"
            )
        # Tracking MAE is independently reconciled to per-episode evidence and
        # therefore provides a tamper-resistant direction/magnitude measure.
        # Diagonal evaluation retains its stricter explicit two-axis sign gate.
        direction_ratio = (
            min(numeric["diag_sign_match_ratio"], expected_tracking)
            if skill == "diagonal"
            else expected_tracking
        )
        if skill == "forward":
            linear_leak = numeric["mae_vy"]
        elif skill == "lateral":
            linear_leak = numeric["mae_vx"]
        elif skill == "yaw":
            linear_leak = max(numeric["mae_vx"], numeric["mae_vy"])
        else:
            linear_leak = 0.0
        yaw_leak = 0.0 if skill == "yaw" else numeric["mae_wz"]
        return {
            "name": str(row["command"]).strip(),
            "skill": skill,
            "command": {"vx": numeric["cmd_vx"], "vy": numeric["cmd_vy"], "wz": numeric["cmd_wz"]},
            "accept_pass": expected_accept,
            "tracking_quality": expected_tracking,
            "stability_quality": expected_stability,
            "fall_rate": numeric["fall_rate"],
            "energy_per_distance": numeric["energy_per_distance"],
            "direction_sign_ratio": direction_ratio,
            "linear_leak": linear_leak,
            "yaw_leak": yaw_leak,
        }

    @staticmethod
    def _command_episode_aggregates(
        row: Mapping[str, str], index: int
    ) -> dict[str, float]:
        aggregates = {
            key: _finite_float(row[key], f"command[{index}].{key}")
            for key in (
                "mae_vx",
                "mae_vy",
                "mae_wz",
                "success_ratio",
                "energy_mech_power_total_mean",
                "energy_per_distance",
                "fall_rate",
            )
        }
        for key in (
            "mae_vx",
            "mae_vy",
            "mae_wz",
            "energy_mech_power_total_mean",
            "energy_per_distance",
        ):
            if aggregates[key] < 0.0:
                raise AutopilotValidationError(
                    f"command evidence row {index}.{key} must be non-negative"
                )
        for key in ("success_ratio", "fall_rate"):
            if not 0.0 <= aggregates[key] <= 1.0:
                raise AutopilotValidationError(
                    f"command evidence row {index}.{key} must be between 0 and 1"
                )
        return aggregates

    @staticmethod
    def _validate_episode_evidence(
        episodes: Sequence[Mapping[str, Any]],
        command_aggregates: Mapping[str, Mapping[str, float]],
        *,
        num_envs: int,
        sweep_steps: int,
    ) -> None:
        if num_envs < 1:
            raise AutopilotValidationError("evaluation environment count must be positive")
        expected_environments = set(range(num_envs))
        grouped: dict[str, dict[int, list[Mapping[str, Any]]]] = {
            name: {} for name in command_aggregates
        }
        identities: set[tuple[str, int, int]] = set()
        for episode in episodes:
            name = str(episode["command"])
            environment_index = int(episode["environment_index"])
            episode_index = int(episode["episode_index"])
            identity = (name, environment_index, episode_index)
            if identity in identities:
                raise AutopilotValidationError(
                    "episode evidence contains a duplicate command/environment/episode identity"
                )
            identities.add(identity)
            if environment_index not in expected_environments:
                raise AutopilotValidationError(
                    f"episode evidence environment index is outside the frozen range for {name}"
                )
            grouped.setdefault(name, {}).setdefault(environment_index, []).append(episode)

        # Per-environment episode indices are emitted in order by the evaluator.
        # A trailing partial row is valid when a sweep ends between resets; no
        # earlier episode in that stream may be incomplete.
        for name, environments in grouped.items():
            if set(environments) != expected_environments:
                raise AutopilotValidationError(
                    f"episode evidence does not cover every frozen environment for {name}"
                )
            for environment_index, rows in environments.items():
                ordered = sorted(rows, key=lambda item: int(item["episode_index"]))
                indices = [int(item["episode_index"]) for item in ordered]
                if indices != list(range(len(indices))):
                    raise AutopilotValidationError(
                        f"episode evidence indices are not contiguous for {name} environment {environment_index}"
                    )
                incomplete = [
                    int(item["episode_index"])
                    for item in ordered
                    if not bool(item["complete"])
                ]
                if incomplete and incomplete != [indices[-1]]:
                    raise AutopilotValidationError(
                        f"only the final episode may be incomplete for {name} environment {environment_index}"
                    )
                environment_samples = sum(int(item["sample_count"]) for item in ordered)
                if environment_samples != sweep_steps:
                    raise AutopilotValidationError(
                        f"episode evidence environment sample horizon mismatch for {name} "
                        f"environment {environment_index}"
                    )

        # CSV values are decimal renderings of GPU reductions.  A 1e-5 relative
        # / 1e-6 absolute tolerance covers float32 reduction-order noise while
        # remaining far tighter than any locomotion gate or displayed metric.
        def require_close(name: str, metric: str, actual: float, expected: float) -> None:
            if not math.isclose(actual, expected, rel_tol=1.0e-5, abs_tol=1.0e-6):
                raise AutopilotValidationError(
                    f"episode evidence aggregate mismatch for {name}.{metric}"
                )

        for name, aggregates in command_aggregates.items():
            rows = [
                episode
                for environment_rows in grouped[name].values()
                for episode in environment_rows
            ]
            sample_count = sum(int(item["sample_count"]) for item in rows)
            if sample_count != sweep_steps * num_envs:
                raise AutopilotValidationError(
                    f"episode evidence sample horizon mismatch for {name}"
                )
            for metric in (
                "mae_vx",
                "mae_vy",
                "mae_wz",
                "success_ratio",
                "energy_mech_power_total_mean",
                "energy_per_distance",
            ):
                episode_metric = "energy_effort_mean" if metric == "energy_per_distance" else metric
                weighted = sum(
                    float(item[episode_metric]) * int(item["sample_count"])
                    for item in rows
                ) / float(sample_count)
                require_close(name, metric, weighted, float(aggregates[metric]))
            complete_rows = [item for item in rows if bool(item["complete"])]
            episode_fall_rate = (
                sum(int(item["fall_count"]) for item in complete_rows)
                / float(len(complete_rows))
                if complete_rows
                else 0.0
            )
            require_close(name, "fall_rate", episode_fall_rate, float(aggregates["fall_rate"]))

    def _parse_evaluation_report(
        self,
        snapshot: Mapping[str, Any],
        trial: Mapping[str, Any],
        evaluation: Mapping[str, Any],
        runtime: Mapping[str, Any],
    ) -> EvaluationReportV1:
        self._assert_campaign_identity(snapshot, runtime)
        campaign_id = str(snapshot["id"])
        paths = {
            "command_csv": Path(str(evaluation.get("command_csv") or "")),
            "episode_csv": Path(str(evaluation.get("episode_csv") or "")),
            "summary_csv": Path(str(evaluation.get("summary_csv") or "")),
        }
        for path in paths.values():
            if not path.is_file():
                raise AutopilotValidationError("evaluation evidence artifact is missing")
        command_bytes = paths["command_csv"].read_bytes()
        episode_bytes = paths["episode_csv"].read_bytes()
        summary_bytes = paths["summary_csv"].read_bytes()
        command_sha = hashlib.sha256(command_bytes).hexdigest()
        episode_sha = hashlib.sha256(episode_bytes).hexdigest()
        summary_sha = hashlib.sha256(summary_bytes).hexdigest()
        for field, actual_sha in (
            ("command_csv_sha256", command_sha),
            ("episode_csv_sha256", episode_sha),
            ("summary_csv_sha256", summary_sha),
        ):
            recorded_sha = evaluation.get(field)
            if (
                not isinstance(recorded_sha, str)
                or re.fullmatch(r"[0-9a-f]{64}", recorded_sha) is None
                or recorded_sha != actual_sha
            ):
                raise AutopilotValidationError(
                    f"evaluation artifact changed after process completion: {field}"
                )
        command_rows = self._read_csv_bytes(command_bytes, paths["command_csv"].name)
        episode_rows = self._read_csv_bytes(episode_bytes, paths["episode_csv"].name)
        summary = self._summary_map(
            self._read_csv_bytes(summary_bytes, paths["summary_csv"].name)
        )
        required_summary = {
            "evaluation.seed", "evaluation.num_envs", "evaluation.sweep_steps",
            "evaluation.step_dt", "evaluation.duration_s", "eval.profile",
            "checkpoint.path", "checkpoint.sha256",
            "command.profile_sha256", "evaluation.agent_entry_point",
            "checkpoint.strict_load", "energy.strict_evidence", "spring.backend", "spring.calibration_status",
            "spring.checkpoint_calibration_status", "spring.profile_id", "spring.profile_sha256",
            "artifact.command_csv_sha256", "artifact.episode_csv_sha256", "evidence.episode_row_count",
            "identity.code_sha256", "identity.config_sha256", "identity.dependency_sha256",
            "identity.reward_profile_sha256", "identity.physics.sha256",
            "identity.spring.sha256", "identity.terrain.sha256",
        }
        missing = sorted(required_summary - set(summary))
        if missing:
            raise AutopilotValidationError(f"evaluation summary is missing {missing[0]}")
        if summary["artifact.command_csv_sha256"] != command_sha:
            raise AutopilotValidationError("command CSV hash does not match its signed summary")
        if summary["artifact.episode_csv_sha256"] != episode_sha:
            raise AutopilotValidationError("episode CSV hash does not match its signed summary")
        if int(summary["evidence.episode_row_count"]) != len(episode_rows):
            raise AutopilotValidationError("episode evidence row count mismatch")
        if int(summary["evaluation.seed"]) != int(trial["seed"]):
            raise AutopilotValidationError("evaluation seed identity mismatch")
        evaluation_params = (
            evaluation.get("params")
            if isinstance(evaluation.get("params"), Mapping)
            else {}
        )
        expected_num_envs = int(runtime.get("num_envs") or self._default_num_envs())
        expected_sweep_steps = int(evaluation_params.get("sweep_steps") or 0)
        expected_step_dt = _finite_float(
            evaluation_params.get("expected_step_dt"),
            "evaluation.expected_step_dt",
        )
        try:
            summary_num_envs = int(summary["evaluation.num_envs"])
            summary_sweep_steps = int(summary["evaluation.sweep_steps"])
        except ValueError as exc:
            raise AutopilotValidationError(
                "evaluation horizon identity must be integral"
            ) from exc
        step_dt = _finite_float(summary["evaluation.step_dt"], "evaluation.step_dt")
        evaluation_duration_s = _finite_float(
            summary["evaluation.duration_s"], "evaluation.duration_s"
        )
        if (
            summary_num_envs != expected_num_envs
            or summary_sweep_steps != expected_sweep_steps
            or summary_sweep_steps < 1
            or step_dt <= 0.0
            or not math.isclose(
                step_dt,
                expected_step_dt,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or evaluation_duration_s <= 0.0
            or not math.isclose(
                evaluation_duration_s,
                summary_sweep_steps * step_dt,
                rel_tol=1.0e-9,
                abs_tol=1.0e-9,
            )
        ):
            raise AutopilotValidationError(
                "evaluation horizon identity mismatch"
            )
        if summary["eval.profile"] != snapshot["goal"]["evaluation_profile"]:
            raise AutopilotValidationError("evaluation profile identity mismatch")
        if summary["command.profile_sha256"] != snapshot["goal"]["command_profile_sha256"]:
            raise AutopilotValidationError("evaluation command profile identity mismatch")
        expected_identities = {
            "identity.code_sha256": snapshot["goal"]["code_sha256"],
            "identity.config_sha256": snapshot["goal"]["config_sha256"],
            "identity.dependency_sha256": runtime["dependency_sha256"],
            "identity.reward_profile_sha256": trial["reward_profile_sha256"],
            "identity.physics.sha256": snapshot["goal"]["physics_profile_sha256"],
            "identity.spring.sha256": snapshot["goal"]["spring_profile_sha256"],
            "identity.terrain.sha256": runtime["terrain_profile_sha256"],
        }
        mismatched_identity = next(
            (name for name, expected in expected_identities.items() if summary[name] != expected),
            None,
        )
        if mismatched_identity is not None:
            raise AutopilotValidationError(
                f"evaluation runtime identity mismatch: {mismatched_identity}"
            )
        if summary["checkpoint.sha256"] != trial["output_checkpoint_sha256"]:
            raise AutopilotValidationError("evaluation selected a different checkpoint")
        if summary["evaluation.agent_entry_point"] != "rsl_rl_cfg_entry_point":
            raise AutopilotValidationError(
                "evaluation selected a different agent entry point"
            )
        if not _csv_bool(summary["checkpoint.strict_load"], "checkpoint.strict_load"):
            raise AutopilotValidationError("evaluation did not use strict checkpoint loading")
        if not _csv_bool(summary["energy.strict_evidence"], "energy.strict_evidence"):
            raise AutopilotValidationError("evaluation did not require strict energy evidence")
        checkpoint_path = Path(summary["checkpoint.path"])
        expected_checkpoint = dict(trial.get("metadata") or {}).get(
            "output_checkpoint_path"
        )
        if not expected_checkpoint or checkpoint_path.resolve() != Path(str(expected_checkpoint)).resolve():
            raise AutopilotValidationError("evaluation checkpoint path identity mismatch")
        if (
            not checkpoint_path.is_file()
            or _sha256_file(checkpoint_path) != trial["output_checkpoint_sha256"]
        ):
            raise AutopilotValidationError("evaluation checkpoint content identity mismatch")
        if summary["spring.backend"] != runtime.get("spring_backend"):
            raise AutopilotValidationError("evaluation spring backend identity mismatch")
        calibration_status = summary["spring.calibration_status"].strip().lower()
        checkpoint_calibration_status = summary[
            "spring.checkpoint_calibration_status"
        ].strip().lower()
        if calibration_status not in {"calibrated", "uncalibrated"}:
            raise AutopilotValidationError("evaluation spring calibration status is invalid")
        if checkpoint_calibration_status != calibration_status:
            raise AutopilotValidationError(
                "evaluation checkpoint spring calibration identity mismatch"
            )
        physics_values = dict(runtime.get("physics_overrides") or {})
        evaluator_profile_sha = summary["spring.profile_sha256"]
        if physics_values:
            training_run = self.history.get_run(str(trial.get("run_id") or "")) or {}
            physics_file = training_run.get("physics_profile_file")
            if not physics_file or not Path(str(physics_file)).is_file():
                raise AutopilotValidationError("evaluation physics profile artifact is missing")
            frozen_physics_sha = dict(trial.get("metadata") or {}).get(
                "physics_profile_artifact_sha256"
            )
            if (
                not frozen_physics_sha
                or _sha256_file(Path(str(physics_file))) != frozen_physics_sha
            ):
                raise AutopilotValidationError(
                    "evaluation physics profile no longer matches the frozen training artifact"
                )
            resolved_profile = json.loads(Path(str(physics_file)).read_text(encoding="utf-8"))
            resolved_profile_sha = sha256_json(resolved_profile)
            if evaluator_profile_sha != resolved_profile_sha:
                raise AutopilotValidationError("evaluation physics profile identity mismatch")
            if summary["spring.profile_id"] != str(resolved_profile.get("profile_id") or ""):
                raise AutopilotValidationError("evaluation physics profile ID mismatch")
        elif evaluator_profile_sha.strip().lower() not in {"", "none", "null"}:
            raise AutopilotValidationError("evaluation unexpectedly selected a physics profile")
        elif summary["spring.profile_id"].strip().lower() not in {"", "none", "null"}:
            raise AutopilotValidationError(
                "evaluation unexpectedly selected a physics profile ID"
            )

        command_profile_file = Path(str(runtime.get("command_profile_file") or ""))
        if not command_profile_file.is_file():
            raise AutopilotValidationError("immutable command profile artifact is missing")
        if _sha256_file(command_profile_file) != snapshot["goal"]["command_profile_sha256"]:
            raise AutopilotValidationError("immutable command profile artifact changed")
        command_profile = json.loads(command_profile_file.read_text(encoding="utf-8"))
        expected_commands = command_profile.get("commands")
        if not isinstance(expected_commands, list) or not expected_commands:
            raise AutopilotValidationError("immutable command profile contains no commands")
        parsed_commands = tuple(
            self._command_row(
                row, index, evaluation_duration_s=evaluation_duration_s
            )
            for index, row in enumerate(command_rows)
        )
        command_aggregates = {
            parsed_commands[index]["name"]: self._command_episode_aggregates(row, index)
            for index, row in enumerate(command_rows)
        }
        expected_by_name = {
            str(item.get("name") or ""): item
            for item in expected_commands
            if isinstance(item, Mapping)
        }
        actual_by_name = {item["name"]: item for item in parsed_commands}
        if (
            len(expected_by_name) != len(expected_commands)
            or len(actual_by_name) != len(parsed_commands)
            or set(actual_by_name) != set(expected_by_name)
        ):
            raise AutopilotValidationError(
                "command evidence does not match the exact command profile"
            )
        for name, actual in actual_by_name.items():
            expected = expected_by_name[name]
            if actual["skill"] != expected.get("skill") or any(
                actual["command"][axis] != float(expected.get(axis))
                for axis in ("vx", "vy", "wz")
            ):
                raise AutopilotValidationError(
                    f"command evidence identity mismatch for {name}"
                )
        parsed_episodes = tuple(
            self._episode_row(row, index) for index, row in enumerate(episode_rows)
        )
        episode_names = {item["command"] for item in parsed_episodes}
        if episode_names != set(expected_by_name):
            raise AutopilotValidationError(
                "episode evidence does not cover the exact command profile"
            )
        if any(
            item["skill"] != expected_by_name[item["command"]].get("skill")
            for item in parsed_episodes
        ):
            raise AutopilotValidationError("episode evidence skill identity mismatch")
        self._validate_episode_evidence(
            parsed_episodes,
            command_aggregates,
            num_envs=expected_num_envs,
            sweep_steps=summary_sweep_steps,
        )

        artifacts = [
            self.store.store_artifact(
                campaign_id,
                kind=kind,
                content=content,
                media_type="text/csv",
                metadata={"trial_id": trial["id"], "evaluation_process_id": evaluation["id"]},
            )
            for kind, content in (
                ("evaluation_commands", command_bytes),
                ("evaluation_episodes", episode_bytes),
                ("evaluation_summary", summary_bytes),
            )
        ]
        return EvaluationReportV1(
            id=f"evaluation_report_{uuid.uuid4().hex}",
            trial_id=str(trial["id"]),
            checkpoint_sha256=str(trial["output_checkpoint_sha256"]),
            config_sha256=str(snapshot["goal"]["config_sha256"]),
            reward_profile_sha256=str(trial["reward_profile_sha256"]),
            physics_profile_sha256=str(snapshot["goal"]["physics_profile_sha256"]),
            spring_profile_sha256=str(snapshot["goal"]["spring_profile_sha256"]),
            command_profile_sha256=str(snapshot["goal"]["command_profile_sha256"]),
            seed=int(trial["seed"]),
            evaluation_profile=str(snapshot["goal"]["evaluation_profile"]),
            strict_checkpoint_load=True,
            episode_artifact_sha256=episode_sha,
            command_metrics=parsed_commands,
            episode_metrics=parsed_episodes,
            artifact_ids=tuple(item["id"] for item in artifacts),
            failure_reason=None,
        )

    @staticmethod
    def _safety_gates_pass(
        report: EvaluationReportV1 | Mapping[str, Any],
    ) -> bool:
        # Identity/evidence and non-compensable safety gates are the survivor
        # filter. Command-duration and per-skill pass ratios are goal-progress
        # gates: a survivor may lead while still missing them, which permits
        # bounded one-weight improvements to accumulate. Tracking itself is a
        # hard survivor gate and can never be offset by a soft score.
        required = {
            "config_identity",
            "physics_identity",
            "spring_identity",
            "command_identity",
            "evaluation_profile",
            "seed_protocol",
            "checkpoint_identity",
            "strict_checkpoint_load",
            "episode_evidence",
            "evaluation_complete",
            "fall_rate",
            "tracking_quality",
            "stability_quality",
            "direction_sign_ratio",
            "linear_leak",
            "yaw_leak",
            "energy_per_distance",
        }
        gates = (
            report.hard_gates
            if isinstance(report, EvaluationReportV1)
            else dict(report.get("hard_gates") or {})
        )
        return required <= set(gates) and all(
            gates[key] for key in required
        )

    def _leader_after_evaluation(
        self,
        snapshot: Mapping[str, Any],
        trial: Mapping[str, Any],
        report: EvaluationReportV1,
    ) -> dict[str, Any] | None:
        current = dict(snapshot["leader"])
        if trial["kind"] in {"confirmation_control", "confirmation_candidate"}:
            return None
        current_report = next(
            (item for item in snapshot["evaluations"] if item.get("id") == current.get("evaluation_id")),
            None,
        )
        better = current_report is None
        if current_report is not None:
            # Compare ranks only within the survivor set. An unsafe control is
            # useful paired evidence, but it must never outrank a candidate
            # that satisfies every non-compensable identity/safety gate.
            better = not self._safety_gates_pass(current_report) or (
                tuple(report.ranking["sort_key"])
                > tuple(current_report["ranking"]["sort_key"])
            )
        if trial["kind"] == "control" or (self._safety_gates_pass(report) and better):
            return {
                "trial_id": trial["id"],
                "evaluation_id": report.id,
                "reward_values": dict(trial["reward_profile"]),
                "rank_key": list(evaluation_rank_key(report)[:-1]),
            }
        return None

    def _after_evaluation(
        self,
        snapshot: Mapping[str, Any],
        trial: Mapping[str, Any],
        report: EvaluationReportV1,
        runtime: Mapping[str, Any],
    ) -> None:
        campaign_id = str(snapshot["id"])
        eligible = bool(report.ranking.get("eligible"))
        if trial["kind"] == "control":
            if eligible:
                self.store.transition_campaign(
                    campaign_id,
                    "simulation_goal_met",
                    expected_revision=int(snapshot["revision"]),
                    idempotency_key=f"controller-control-passed-{trial['id']}",
                    reason="the unchanged seed-42 control already satisfies every simulation goal gate",
                    active_process=None,
                )
            else:
                awaiting = self.store.transition_campaign(
                    campaign_id,
                    "awaiting_advisor",
                    expected_revision=int(snapshot["revision"]),
                    idempotency_key=f"controller-await-advisor-{trial['id']}",
                    active_process=None,
                )
                remaining = int(awaiting["budget"].get("remaining_training_trials", 0))
                reserved = int(awaiting["budget"].get("reserved_confirmation_trials", 0))
                if remaining <= reserved:
                    self._enter_patch_handoff(
                        awaiting,
                        "insufficient training budget for one screen and all confirmation trials",
                    )
            return
        if trial["kind"] == "candidate":
            if eligible:
                remaining = int(snapshot["budget"].get("remaining_training_trials", 0))
                reserved = int(snapshot["budget"].get("reserved_confirmation_trials", 0))
                if remaining < 4 or reserved < 4:
                    self._enter_patch_handoff(
                        snapshot,
                        "insufficient reserved budget to run all four confirmation trials",
                    )
                    return
                queue = [
                    {"kind": "confirmation_control", "seed": 43},
                    {"kind": "confirmation_candidate", "seed": 43},
                    {"kind": "confirmation_control", "seed": 44},
                    {"kind": "confirmation_candidate", "seed": 44},
                ]
                if runtime.get("confirmation_winner_trial_id") == trial["id"]:
                    updated = snapshot
                else:
                    updated = self.store.update_runtime(
                        campaign_id,
                        {
                            "confirmation_queue": queue,
                            "confirmation_winner_trial_id": trial["id"],
                            "pending_candidate_profile": None,
                        },
                        expected_revision=int(snapshot["revision"]),
                        idempotency_key=f"controller-confirmation-queue-{trial['id']}",
                        event_type="confirmation_started",
                    )
                self.store.transition_campaign(
                    campaign_id,
                    "confirming",
                    expected_revision=int(updated["revision"]),
                    idempotency_key=f"controller-confirming-{trial['id']}",
                    active_process=None,
                )
                self._notify()
                return
            leader_selected = snapshot["leader"].get("trial_id") == trial["id"]
            if runtime.get("last_screened_trial_id") == trial["id"]:
                count = int(runtime.get("non_improving_candidates", 0))
                updated = snapshot
            else:
                count = (
                    0
                    if leader_selected
                    else int(runtime.get("non_improving_candidates", 0)) + 1
                )
                updated = self.store.update_runtime(
                    campaign_id,
                    {
                        "pending_candidate_profile": None,
                        "non_improving_candidates": count,
                        "last_screened_trial_id": trial["id"],
                    },
                    expected_revision=int(snapshot["revision"]),
                    idempotency_key=f"controller-screen-outcome-{trial['id']}",
                    event_type="candidate_screened",
                )
            remaining = int(updated["budget"].get("remaining_training_trials", 0))
            reserved = int(updated["budget"].get("reserved_confirmation_trials", 0))
            if remaining <= reserved:
                self._enter_patch_handoff(
                    updated,
                    "no screening budget remains beyond the reserved confirmation trials",
                )
            elif count >= 4 or not self._has_eligible_move(updated):
                self._enter_patch_handoff(updated, "four candidates did not improve, or no bounded move remains")
            else:
                self.store.transition_campaign(
                    campaign_id,
                    "awaiting_advisor",
                    expected_revision=int(updated["revision"]),
                    idempotency_key=f"controller-await-next-advisor-{trial['id']}",
                    active_process=None,
                )
            return
        # Confirmation evaluation returns to the confirmation scheduler.
        resumed = self.store.transition_campaign(
            campaign_id,
            "confirming",
            expected_revision=int(snapshot["revision"]),
            idempotency_key=f"controller-confirmation-recorded-{trial['id']}",
            active_process=None,
        )
        self._notify()

    @staticmethod
    def _has_eligible_move(snapshot: Mapping[str, Any]) -> bool:
        catalog = tuple(
            RewardCatalogEntryV1.from_dict(entry)
            for entry in snapshot["reward_catalog"]
        )
        return bool(
            reward_move_lattice(
                catalog,
                dict(snapshot["leader"].get("reward_values") or {}),
                snapshot.get("decisions", []),
            )["remaining"]
        )

    def _ensure_confirmation(self, snapshot: Mapping[str, Any], runtime: Mapping[str, Any]) -> None:
        campaign_id = str(snapshot["id"])
        queue = list(runtime.get("confirmation_queue") or [])
        existing = list(snapshot["candidate_lineage"])
        for item in queue:
            match = next(
                (
                    trial for trial in existing
                    if trial["kind"] == item["kind"] and int(trial["seed"]) == int(item["seed"])
                ),
                None,
            )
            if match and match["status"] == "evaluated":
                continue
            if match is None:
                reward_profile = (
                    dict(runtime["original_reward_values"])
                    if item["kind"] == "confirmation_control"
                    else dict(snapshot["leader"]["reward_values"])
                )
                result = self.store.reserve_trial(
                    campaign_id,
                    kind=str(item["kind"]),
                    seed=int(item["seed"]),
                    reward_profile=reward_profile,
                    source_checkpoint_sha256=snapshot["goal"].get("checkpoint_sha256"),
                    expected_revision=int(snapshot["revision"]),
                    idempotency_key=f"controller-confirm-{campaign_id}-{item['kind']}-{item['seed']}",
                    metadata=self._trial_identity_metadata(
                        snapshot,
                        runtime,
                        {"phase": "confirmation", "winner_trial_id": runtime.get("confirmation_winner_trial_id")},
                    ),
                )
                snapshot, match = result["campaign"], result["trial"]
            training_state = self.store.transition_campaign(
                campaign_id,
                "candidate_training",
                expected_revision=int(snapshot["revision"]),
                idempotency_key=f"controller-confirm-training-{match['id']}",
            )
            self._launch_training_if_needed(training_state, match, runtime)
            return
        self._finish_confirmation(snapshot)

    def _finish_confirmation(self, snapshot: Mapping[str, Any]) -> None:
        runtime = self.store.get_runtime(str(snapshot["id"]))
        winner_id = runtime.get("confirmation_winner_trial_id")
        reports = {item["trial_id"]: item for item in snapshot["evaluations"]}
        controls: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []
        for trial in snapshot["candidate_lineage"]:
            report = reports.get(trial["id"])
            if report is None:
                continue
            if trial["kind"] in {"control", "confirmation_control"}:
                controls.append(report)
            elif trial["id"] == winner_id or trial["kind"] == "confirmation_candidate":
                candidates.append(report)
        valid = len(controls) == 3 and len(candidates) == 3
        if valid:
            for report in (*controls, *candidates):
                self._verify_evaluation_artifacts(str(snapshot["id"]), report)
        candidate_passes = sum(bool(report.get("ranking", {}).get("eligible")) for report in candidates)
        energy_safe = valid and all(report.get("hard_gates", {}).get("energy_per_distance") is True for report in candidates)
        tracking_improved = valid and median(
            float(report["ranking"]["mean_tracking_quality"]) for report in candidates
        ) > median(float(report["ranking"]["mean_tracking_quality"]) for report in controls)
        if valid and candidate_passes >= 2 and energy_safe and tracking_improved:
            self.store.transition_campaign(
                str(snapshot["id"]),
                "simulation_goal_met",
                expected_revision=int(snapshot["revision"]),
                idempotency_key=f"controller-simulation-goal-{snapshot['id']}",
                reason="at least two of three candidate replicas passed and median tracking improved over paired controls",
                active_process=None,
            )
        else:
            reason = (
                "confirmation failed closed: three valid paired replicas, two candidate passes, "
                "tracking improvement, and energy compliance were not all established"
            )
            self._enter_patch_handoff(snapshot, reason)

    def _enter_patch_handoff(self, snapshot: Mapping[str, Any], reason: str) -> None:
        self._store_patch_context(str(snapshot["id"]), snapshot)
        self.store.transition_campaign(
            str(snapshot["id"]),
            "patch_handoff",
            expected_revision=int(snapshot["revision"]),
            idempotency_key=f"controller-patch-handoff-{snapshot['id']}-{snapshot['revision']}",
            reason=reason,
            active_process=None,
        )

    def _fail_campaign(self, campaign_id: str, error: Exception) -> None:
        try:
            snapshot = self.store.get_campaign(campaign_id)
            if snapshot["state"] in TERMINAL_STATES:
                return
            reason = f"controller failure: {type(error).__name__}: {error}"
            target = "blocked_safety" if isinstance(error, AutopilotValidationError) else "failed"
            runtime = self.store.get_runtime(campaign_id)
            if runtime.get("controller_failure_stop_intent"):
                self._finish_controller_failure_stop(snapshot, runtime)
                return
            active = snapshot.get("active_process") or {}
            process_id = str(active.get("process_id") or "")
            process_kind = str(active.get("kind") or "")
            trial = self._current_trial(snapshot)
            if process_id and process_kind in {"training", "evaluation"} and trial:
                record = self._reconcile_campaign_process(process_id)
                accounting = {
                    "trial_id": str(trial["id"]),
                    "process_id": process_id,
                    "process_kind": process_kind,
                    "cumulative_gpu_hours": (
                        self._gpu_hours(record) if record is not None else 0.0
                    ),
                }
                intent = self.store.begin_controller_failure_stop(
                    campaign_id,
                    expected_revision=int(snapshot["revision"]),
                    idempotency_key=(
                        f"controller-failure-stop-{campaign_id}-{snapshot['revision']}"
                    ),
                    target=target,
                    reason=reason[:1900],
                    gpu_accounting=accounting,
                )
                self._finish_controller_failure_stop(
                    intent, self.store.get_runtime(campaign_id)
                )
            else:
                self.store.transition_campaign(
                    campaign_id,
                    target,
                    expected_revision=int(snapshot["revision"]),
                    idempotency_key=f"controller-failure-{campaign_id}-{snapshot['revision']}",
                    reason=reason[:1900],
                    active_process=None,
                )
            self._audit("autopilot_campaign_failed", campaign_id, {"reason": reason[:1000]})
        except Exception:
            # The original exception and any store conflict are durable in the
            # process log; never kill the panel's controller thread.
            return

    def _finish_controller_failure_stop(
        self,
        snapshot: Mapping[str, Any],
        runtime: Mapping[str, Any],
    ) -> dict[str, Any]:
        intent = runtime.get("controller_failure_stop_intent")
        if not isinstance(intent, Mapping):
            raise AutopilotConflictError(
                "campaign has no durable controller failure stop intent",
                current_revision=int(snapshot["revision"]),
            )
        accounting = dict(intent.get("gpu_accounting") or {})
        process_id = str(accounting.get("process_id") or "")
        record = self._reconcile_campaign_process(process_id)
        if record is None:
            raise AutopilotConflictError(
                "campaign process could not be resolved for failure-stop recovery",
                current_revision=int(snapshot["revision"]),
            )
        status = str(record.get("status") or "").lower()
        if status in {"queued", "running", "stopping", ""}:
            try:
                stopped = self.processes.stop(process_id)
            except Exception as exc:
                raise AutopilotConflictError(
                    "failed campaign process could not be stopped",
                    current_revision=int(snapshot["revision"]),
                ) from exc
            if not stopped:
                raise AutopilotConflictError(
                    "failed campaign process could not be stopped",
                    current_revision=int(snapshot["revision"]),
                )
            record = self._reconcile_campaign_process(process_id) or record
            if str(record.get("status") or "").lower() in {
                "queued",
                "running",
                "stopping",
                "",
            }:
                return dict(snapshot)
        accounting["cumulative_gpu_hours"] = max(
            float(accounting.get("cumulative_gpu_hours") or 0.0),
            self._gpu_hours(record),
        )
        result = self.store.finalize_controller_failure_stop(
            str(snapshot["id"]),
            expected_revision=int(snapshot["revision"]),
            idempotency_key=f"{intent['idempotency_key']}:finalize",
            gpu_accounting=accounting,
        )
        self._audit(
            "autopilot_campaign_failed",
            str(snapshot["id"]),
            {"reason": str(intent.get("reason") or "controller failure")[:1000]},
        )
        return result
