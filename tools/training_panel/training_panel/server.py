from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import secrets
import shlex
import shutil
import subprocess
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from tools.training_panel import __version__

from .activity import ActivityStore
from .autopilot import GOAL_SCHEMA_VERSION, AutopilotValidationError, autopilot_capabilities
from .autopilot_service import AutopilotService
from .autopilot_store import (
    AutopilotBudgetError,
    AutopilotConflictError,
    AutopilotStoreError,
    CampaignNotFoundError,
)
from .commands import (
    DEFAULT_TRAINING_TASK,
    DEFAULT_VIDEO_PRESET,
    VIDEO_PRESETS,
    TrainingParams,
    VideoParams,
    resolve_spring_backend,
)
from .config import PanelPaths
from .deploy import deploy_defaults, latest_deploy_report, list_deploy_reports
from .google_drive import (
    GoogleDriveBusyError,
    GoogleDriveExporter,
    GoogleDrivePathError,
    GoogleDriveUnavailableError,
)
from .history import HistoryStore, checkpoint_inventory
from .physics import PhysicsPresetStore, physics_catalog
from .presets import PresetStore
from .robot_geometry import robot_geometry
from .processes import CudaPreflightError, ProcessRegistry, ProcessStartError
from .remote_config import RemoteStateStore
from .remote_manager import RemoteWorkerManager
from .rewards import reward_defaults, reward_file_index
from .terrain import TerrainPresetStore, terrain_defaults, terrain_file_index
from .tweaks import build_tweak_payload, newest_finished_tweak_run


STATIC_DIR = Path(__file__).resolve().parents[1] / "static"

# Browsers refuse to execute an ES module served under a non-JavaScript MIME type, and
# the system mimetypes database does not reliably cover .mjs. Pin the types we serve.
_STATIC_CONTENT_TYPES = {
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
}
_PRESET_FILE = Path(__file__).resolve().parents[1] / "reward_presets.json"
_TERRAIN_PRESET_FILE = Path(__file__).resolve().parents[1] / "terrain_presets.json"
_AUTOPILOT_ERROR_SCHEMA_VERSION = "redrhex.autopilot.error.v1"
_AUTOPILOT_ADVISOR_METADATA_SCHEMA_VERSION = "redrhex.autopilot.advisor-metadata.v1"
_AUTOPILOT_SKILL_VERSION = "redrhex-autopilot/0.1.0"
_AUTOPILOT_PROMPT_VERSION = "scheduled-advisor.v1"
_AUTOPILOT_MAX_REQUEST_BYTES = 1024 * 1024
_AUTOPILOT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_AUTOPILOT_IDEMPOTENCY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")
_AUTOPILOT_IF_MATCH_RE = re.compile(r'^"(0|[1-9][0-9]*)"$')
_AUTOPILOT_DRAFT_FIELDS = {
    "schema_version",
    "description",
    "task",
    "stage",
    "evaluation_profile",
    "gait",
    "directions",
    "command_envelope",
    "skill_gates",
    "baseline_run_id",
    "baseline_checkpoint_iteration",
    "checkpoint_sha256",
    "initialization_mode",
    "training_seeds",
    "per_trial_iteration_cap",
    "budget",
    "tunable_reward_keys",
    "reward_bounds",
    "physics_profile_sha256",
    "spring_profile_sha256",
    "code_sha256",
    "config_sha256",
    "command_profile_sha256",
}


def route_id(path: str) -> str:
    return unquote(path.split("/")[3])


def route_id2(path: str) -> str:
    """Extract the second path segment ID (index 4), e.g. /api/presets/{id}/delete."""
    return unquote(path.split("/")[4])


def _is_within(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    resolved_root = root.resolve()
    return resolved == resolved_root or resolved_root in resolved.parents


class RunVideoError(ValueError):
    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.status = status


class AutopilotRouteError(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
        status: int = 400,
        details: object | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.status = status
        self.details = details


def _sync_active_terrain_override_file(paths: PanelPaths, values: dict) -> None:
    """Keep train.py's global terrain override file aligned with the active preset."""
    override_file = paths.terrain_override_file
    if values:
        override_file.parent.mkdir(parents=True, exist_ok=True)
        override_file.write_text(json.dumps(values, indent=2), encoding="utf-8")
    elif override_file.exists():
        override_file.unlink()


class PanelState:
    def __init__(self, paths: PanelPaths):
        self.paths = paths
        self.history = HistoryStore(paths)
        self.google_drive = GoogleDriveExporter(paths, self.history)
        self.processes = ProcessRegistry(paths, self.history, cuda_preflight=True)
        self.presets = PresetStore(_PRESET_FILE)
        self.terrain_presets = TerrainPresetStore(_TERRAIN_PRESET_FILE)
        self.physics_presets = PhysicsPresetStore(paths.physics_preset_file)
        self.activity = ActivityStore(paths)
        self.remote_state = RemoteStateStore(paths.remote_state_file)
        self.remote_worker = RemoteWorkerManager(paths, self.remote_state)
        # Reconciliation is a startup mutation. GET endpoints remain observational and
        # cannot advance queues merely because a browser or connector polls them.
        self.processes.reconcile_stale_history()
        self.processes.start_next_queued_training()
        # This is the only Autopilot service instance in the panel process. Start its
        # worker only after the process registry has reconciled persisted work.
        self.autopilot = None
        try:
            self.autopilot = AutopilotService(
                paths,
                self.history,
                self.processes,
                self.activity,
            )
        except Exception:
            # Keep the ordinary panel usable when campaign storage cannot start;
            # Autopilot capabilities then advertise unavailable and writes fail closed.
            traceback.print_exc()
        self.remote_worker.autostart_if_enabled()


# Reading TensorBoard event files is expensive, and the panel polls. Cache per
# (log_dir, tag) and invalidate on the event files' size+mtime signature.
_SCALAR_CACHE: dict[tuple[str, str], tuple[tuple, list]] = {}
_SCALAR_CACHE_LOCK = threading.Lock()


def _event_signature(log_dir: Path) -> tuple:
    entries = []
    for path in sorted(log_dir.glob("events.out.tfevents.*")):
        try:
            stat = path.stat()
        except OSError:
            continue
        entries.append((path.name, stat.st_size, int(stat.st_mtime)))
    return tuple(entries)


def _scalar_cache_get(log_dir: Path, tag: str, checker) -> list:
    key = (str(log_dir), tag)
    signature = _event_signature(log_dir)
    with _SCALAR_CACHE_LOCK:
        cached = _SCALAR_CACHE.get(key)
        if cached and cached[0] == signature:
            return cached[1]
    scalars = checker.read_scalars(log_dir, tag)
    with _SCALAR_CACHE_LOCK:
        if len(_SCALAR_CACHE) > 200:
            _SCALAR_CACHE.clear()
        _SCALAR_CACHE[key] = (signature, scalars)
    return scalars


class PanelHandler(BaseHTTPRequestHandler):
    state: PanelState

    def do_GET(self) -> None:
        return self._handle_request(self._do_GET)

    def _do_GET(self) -> None:
        parsed = urlparse(self.path)
        if self._is_autopilot_path(parsed.path):
            return self._handle_autopilot_get(parsed)
        if parsed.path == "/":
            return self._send_static("index.html")
        if parsed.path.startswith("/static/"):
            return self._send_static(parsed.path.removeprefix("/static/"))
        if parsed.path == "/api/system":
            return self._json(
                {
                    "repo_root": str(self.state.paths.repo_root),
                    "rsl_rl_log_root": str(self.state.paths.rsl_rl_log_root),
                    "default_task": DEFAULT_TRAINING_TASK,
                    "version": __version__,
                    "cuda_health": self.state.processes.cuda_health(),
                    "google_drive_export": self.state.google_drive.status(),
                    "local_url_hint": "http://127.0.0.1:8080",
                    "lan_hint": "Run with --host 0.0.0.0 and open http://<machine-ip>:8080",
                    "ssh_tunnel_hint": "ssh -L 8080:127.0.0.1:8080 user@host",
                }
            )
        if parsed.path == "/api/activity":
            query = parse_qs(parsed.query)
            try:
                limit = int(query.get("limit", ["80"])[0])
            except ValueError:
                limit = 80
            limit = max(1, min(limit, 200))
            include_remote = query.get("remote", ["1"])[0] not in {"0", "false", "False"}
            window = query.get("window", ["7d"])[0]
            if window not in {"today", "7d", "30d"}:
                window = "7d"
            member = query.get("member", [""])[0]
            category = query.get("category", [""])[0]
            if category not in {"", "training", "artifact", "preset", "metadata", "admin", "system"}:
                category = ""
            return self._json(
                self.state.activity.snapshot(
                    limit=limit,
                    include_remote=include_remote,
                    window=window,
                    member=member,
                    category=category,
                )
            )
        if parsed.path == "/api/remote/status":
            processes = self.state.processes.list_processes()
            active_isaac = self.state.processes.running_isaac_processes()
            return self._json(
                {
                    **self.state.remote_worker.status(),
                    "active_process_count": len([p for p in processes if p.get("returncode") is None]),
                    "active_isaac_process_count": len(active_isaac),
                    "active_isaac_processes": active_isaac,
                }
            )
        if parsed.path == "/api/training/defaults":
            return self._json(TrainingParams().to_dict())
        if parsed.path == "/api/deploy/defaults":
            return self._json(deploy_defaults(self.state.paths))
        if parsed.path == "/api/video/presets":
            return self._json({"presets": [params.to_dict() for params in VIDEO_PRESETS.values()]})
        if parsed.path == "/api/runs":
            return self._json(self._runs_payload())
        if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/mujoco/video"):
            run_id = route_id(parsed.path)
            return self._send_run_mujoco_video(run_id)
        if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/deploy"):
            run_id = route_id(parsed.path)
            run = self.state.history.get_run(run_id)
            if not run:
                return self._json({"error": "Run not found"}, status=404)
            return self._json({"run_id": run_id, "reports": list_deploy_reports(run), "latest": latest_deploy_report(run)})
        if parsed.path.startswith("/api/deploy/") and parsed.path.endswith("/debug"):
            pipeline_id = route_id(parsed.path)
            debug = self.state.processes.get_process_debug(pipeline_id)
            if not debug:
                return self._json({"error": "Deploy process not found"}, status=404)
            return self._json(debug)
        if parsed.path == "/api/tweaks/last-run":
            self.state.processes.reconcile_stale_history()
            reward_presets = self.state.presets.list_presets()
            run = newest_finished_tweak_run(self.state.history.list_runs(), reward_presets)
            if not run:
                return self._json({"error": "No finished run with usable tweak data found"}, status=404)
            try:
                return self._json(
                    build_tweak_payload(
                        run,
                        reward_presets=reward_presets,
                        terrain_presets=self.state.terrain_presets.list_presets(),
                    )
                )
            except ValueError as exc:
                return self._json({"error": str(exc)}, status=400)
        if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/tweak"):
            run_id = route_id(parsed.path)
            run = self.state.history.get_run(run_id)
            if not run:
                return self._json({"error": "Run not found"}, status=404)
            try:
                return self._json(
                    build_tweak_payload(
                        run,
                        reward_presets=self.state.presets.list_presets(),
                        terrain_presets=self.state.terrain_presets.list_presets(),
                    )
                )
            except ValueError as exc:
                return self._json({"error": str(exc)}, status=400)
        if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/notes"):
            run_id = route_id(parsed.path)
            return self._json({"run_id": run_id, "notes": self.state.history.get_note(run_id)})
        if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/debug"):
            run_id = route_id(parsed.path)
            debug = self.state.history.get_debug(run_id)
            if not debug:
                return self._json({"error": "Run not found"}, status=404)
            return self._json(debug)
        if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/reward-config"):
            run_id = route_id(parsed.path)
            query = parse_qs(parsed.query)
            compare_to = query.get("compare", ["default"])[0]
            config = self.state.history.get_reward_config_for_run(run_id, compare_to=compare_to)
            if config is None:
                return self._json({"error": "No reward config found for run"}, status=404)
            return self._json(config)
        if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/scalars"):
            run_id = route_id(parsed.path)
            return self._send_run_scalars(run_id, parse_qs(parsed.query))
        if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/terrain-config"):
            run_id = route_id(parsed.path)
            config = self.state.history.get_terrain_config_for_run(run_id)
            if config is None:
                return self._json({"error": "No terrain config found for run"}, status=404)
            return self._json(config)
        if parsed.path == "/api/folders":
            return self._json({"folders": self.state.history.get_folders()})
        if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/video"):
            run_id = route_id(parsed.path)
            return self._send_run_video(run_id, parse_qs(parsed.query))
        if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/tensorboard-summary.png"):
            run_id = route_id(parsed.path)
            return self._send_run_tensorboard_summary(run_id)
        if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/delete-preview"):
            run_id = route_id(parsed.path)
            preview = self.state.history.delete_preview(run_id)
            if not preview:
                return self._json({"error": "Run not found"}, status=404)
            return self._json(preview)
        if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/compact-preview"):
            run_id = route_id(parsed.path)
            try:
                return self._json(self.state.history.compact_preview(run_id))
            except ValueError as exc:
                return self._json({"error": str(exc)}, status=404 if str(exc) == "Run not found" else 400)
        if parsed.path == "/api/tweakables":
            return self._json(reward_file_index(self.state.paths.repo_root))
        if parsed.path == "/api/rewards/defaults":
            return self._json(reward_defaults(self.state.paths.repo_root))
        if parsed.path == "/api/terrain":
            return self._json(terrain_file_index(self.state.paths.repo_root))
        if parsed.path == "/api/terrain/defaults":
            return self._json(terrain_defaults(self.state.paths.repo_root))
        if parsed.path == "/api/terrain/presets":
            return self._json({
                "presets": self.state.terrain_presets.list_presets(),
                "active_preset_id": self.state.terrain_presets.get_active_preset_id(),
            })
        if parsed.path == "/api/physics":
            return self._json(physics_catalog())
        if parsed.path == "/api/physics/robot-geometry":
            return self._json(robot_geometry(self.state.paths))
        if parsed.path == "/api/physics/presets":
            return self._json({
                "presets": self.state.physics_presets.list_presets(),
                "active_preset_id": self.state.physics_presets.get_active_preset_id(),
            })
        if parsed.path.startswith("/api/physics/presets/") and not parsed.path.endswith("/update") and not parsed.path.endswith("/delete"):
            preset_id = route_id2(parsed.path)
            preset = self.state.physics_presets.get_preset(preset_id)
            if not preset:
                return self._json({"error": "Physics preset not found"}, status=404)
            return self._json(preset)
        if parsed.path.startswith("/api/terrain/presets/") and not parsed.path.endswith("/update") and not parsed.path.endswith("/delete"):
            preset_id = route_id2(parsed.path)
            preset = self.state.terrain_presets.get_preset(preset_id)
            if not preset:
                return self._json({"error": "Terrain preset not found"}, status=404)
            return self._json(preset)
        if parsed.path == "/api/presets":
            return self._json({
                "presets": self.state.presets.list_presets(),
                "active_preset_id": self.state.presets.get_active_preset_id(),
            })
        if parsed.path.startswith("/api/presets/") and not parsed.path.endswith("/update") and not parsed.path.endswith("/delete"):
            preset_id = route_id(parsed.path)
            preset = self.state.presets.get_preset(preset_id)
            if not preset:
                return self._json({"error": "Preset not found"}, status=404)
            return self._json(preset)
        if parsed.path == "/api/processes":
            return self._json({"processes": self.state.processes.list_processes()})
        if parsed.path.startswith("/api/processes/") and parsed.path.endswith("/debug"):
            process_id = route_id(parsed.path)
            debug = self.state.processes.get_process_debug(process_id)
            if not debug:
                return self._json({"error": "Process not found"}, status=404)
            return self._json(debug)
        if parsed.path == "/api/convergence/settings":
            from .convergence import PRESETS, load_convergence_config
            cfg = load_convergence_config(self.state.paths.convergence_config_file)
            from dataclasses import asdict
            return self._json({"config": asdict(cfg), "presets": PRESETS})
        self._not_found()

    def do_POST(self) -> None:
        return self._handle_request(self._do_POST)

    def _do_POST(self) -> None:
        parsed = urlparse(self.path)
        if self._is_autopilot_path(parsed.path):
            return self._handle_autopilot_write(parsed, method="POST")
        try:
            payload = self._payload()
            if parsed.path == "/api/training/start":
                params = TrainingParams.from_dict(payload)
                run = self.state.processes.queue_training(params)
                queued = str(run.get("status") or "").lower() == "queued"
                self._record_activity(
                    "training_queue" if queued else "training_start",
                    summary=f"{'Queued' if queued else 'Started'} training {run.get('id')}",
                    subject_id=str(run.get("id") or ""),
                    payload={
                        "run_id": run.get("id"),
                        "status": run.get("status"),
                        "reward_preset_id": params.reward_preset_id,
                        "terrain_preset_id": params.terrain_preset_id,
                        "physics_preset_id": params.physics_preset_id,
                        "params": params.to_dict(),
                    },
                )
                return self._json(run, status=201)
            if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/cancel-queue"):
                run_id = route_id(parsed.path)
                cancelled = self.state.processes.cancel_queued_training(run_id)
                if cancelled:
                    self._record_activity("training_queue_cancel", summary=f"Cancelled queued training {run_id}", subject_id=run_id)
                return self._json({"cancelled": cancelled, "run_id": run_id})
            if parsed.path == "/api/remote/settings":
                state = self.state.remote_worker.save_settings(payload)
                return self._json({"saved": True, "remote_state": state, "status": self.state.remote_worker.status()})
            if parsed.path == "/api/google-drive/settings":
                try:
                    status = self.state.google_drive.save_destination(
                        str(payload.get("destination") or payload.get("destination_folder") or "")
                    )
                except GoogleDriveBusyError as exc:
                    return self._json({"error": str(exc)}, status=409)
                except GoogleDriveUnavailableError as exc:
                    return self._json({"error": str(exc)}, status=503)
                self._record_activity(
                    "google_drive_settings_update",
                    summary="Updated the private Google Drive export destination",
                    payload={
                        "destination_mode": status.get("destination_mode"),
                        "destination_display": status.get("destination_display"),
                    },
                )
                return self._json({"saved": True, "google_drive_export": status})
            if parsed.path == "/api/google-drive/reconnect":
                try:
                    reconnect, started = self.state.google_drive.start_reconnect()
                except GoogleDriveBusyError as exc:
                    return self._json({"error": str(exc)}, status=409)
                except GoogleDriveUnavailableError as exc:
                    return self._json({"error": str(exc)}, status=503)
                if started:
                    self._record_activity(
                        "google_drive_reconnect_start",
                        summary="Started Google Drive account reconnection on the training PC",
                    )
                return self._json(
                    {
                        "started": started,
                        "reconnect": reconnect,
                        "google_drive_export": self.state.google_drive.status(),
                    },
                    status=202 if reconnect.get("status") == "authorizing" else 200,
                )
            if parsed.path == "/api/remote/worker/start":
                return self._json(self.state.remote_worker.start(str(payload.get("mode") or "")))
            if parsed.path == "/api/remote/worker/stop":
                return self._json(self.state.remote_worker.stop())
            if parsed.path == "/api/remote/worker/restart":
                return self._json(self.state.remote_worker.restart(str(payload.get("mode") or "")))
            if parsed.path == "/api/presets":
                name = str(payload.get("name") or "").strip()
                description = str(payload.get("description") or "").strip()
                values = {str(k): float(v) for k, v in (payload.get("values") or {}).items()}
                if not name:
                    return self._json({"error": "name is required"}, status=400)
                preset = self.state.presets.create_preset(name, description, values)
                self._record_activity(
                    "reward_preset_create",
                    summary=f"Created reward profile {preset.get('name')}",
                    subject_id=str(preset.get("id") or ""),
                    payload={"reward_preset_id": preset.get("id")},
                )
                return self._json(preset, status=201)
            if parsed.path.startswith("/api/presets/") and parsed.path.endswith("/update"):
                preset_id = route_id(parsed.path)
                updates: dict = {}
                if "name" in payload:
                    updates["name"] = str(payload["name"])
                if "description" in payload:
                    updates["description"] = str(payload["description"])
                if "values" in payload:
                    updates["values"] = {str(k): float(v) for k, v in payload["values"].items()}
                try:
                    preset = self.state.presets.update_preset(preset_id, **updates)
                except (KeyError, ValueError) as exc:
                    return self._json({"error": str(exc)}, status=400)
                self._record_activity(
                    "reward_preset_edit",
                    summary=f"Edited reward profile {preset.get('name')}",
                    subject_id=preset_id,
                    payload={"reward_preset_id": preset_id},
                )
                return self._json(preset)
            if parsed.path.startswith("/api/presets/") and parsed.path.endswith("/delete"):
                preset_id = route_id(parsed.path)
                try:
                    deleted = self.state.presets.delete_preset(preset_id)
                except ValueError as exc:
                    return self._json({"error": str(exc)}, status=400)
                if deleted:
                    self._record_activity(
                        "reward_preset_delete",
                        summary=f"Deleted reward profile {preset_id}",
                        subject_id=preset_id,
                        payload={"reward_preset_id": preset_id},
                    )
                return self._json({"deleted": deleted})
            if parsed.path == "/api/presets/activate":
                preset_id = str(payload.get("preset_id") or "")
                try:
                    self.state.presets.set_active_preset(preset_id)
                except KeyError as exc:
                    return self._json({"error": str(exc)}, status=404)
                self._record_activity(
                    "reward_preset_activate",
                    summary=f"Activated reward profile {preset_id}",
                    subject_id=preset_id,
                    payload={"reward_preset_id": preset_id},
                )
                return self._json({"active_preset_id": preset_id})
            if parsed.path == "/api/terrain/presets":
                name = str(payload.get("name") or "").strip()
                description = str(payload.get("description") or "").strip()
                values = dict(payload.get("values") or {})
                if not name:
                    return self._json({"error": "name is required"}, status=400)
                preset = self.state.terrain_presets.create_preset(name, description, values)
                self._record_activity(
                    "terrain_preset_create",
                    summary=f"Created terrain profile {preset.get('name')}",
                    subject_id=str(preset.get("id") or ""),
                    payload={"terrain_preset_id": preset.get("id")},
                )
                return self._json(preset, status=201)
            if parsed.path.startswith("/api/terrain/presets/") and parsed.path.endswith("/update"):
                preset_id = route_id2(parsed.path)
                updates: dict = {}
                if "name" in payload:
                    updates["name"] = str(payload["name"])
                if "description" in payload:
                    updates["description"] = str(payload["description"])
                if "values" in payload:
                    updates["values"] = dict(payload["values"])
                try:
                    preset = self.state.terrain_presets.update_preset(preset_id, **updates)
                except (KeyError, ValueError) as exc:
                    return self._json({"error": str(exc)}, status=400)
                self._record_activity(
                    "terrain_preset_edit",
                    summary=f"Edited terrain profile {preset.get('name')}",
                    subject_id=preset_id,
                    payload={"terrain_preset_id": preset_id},
                )
                if self.state.terrain_presets.get_active_preset_id() == preset_id:
                    _sync_active_terrain_override_file(self.state.paths, dict(preset.get("values") or {}))
                return self._json(preset)
            if parsed.path.startswith("/api/terrain/presets/") and parsed.path.endswith("/delete"):
                preset_id = route_id2(parsed.path)
                try:
                    deleted = self.state.terrain_presets.delete_preset(preset_id)
                except ValueError as exc:
                    return self._json({"error": str(exc)}, status=400)
                if deleted:
                    self._record_activity(
                        "terrain_preset_delete",
                        summary=f"Deleted terrain profile {preset_id}",
                        subject_id=preset_id,
                        payload={"terrain_preset_id": preset_id},
                    )
                    active = self.state.terrain_presets.get_preset(self.state.terrain_presets.get_active_preset_id()) or {}
                    _sync_active_terrain_override_file(self.state.paths, dict(active.get("values") or {}))
                return self._json({"deleted": deleted})
            if parsed.path == "/api/terrain/presets/activate":
                preset_id = str(payload.get("preset_id") or "")
                try:
                    self.state.terrain_presets.set_active_preset(preset_id)
                except KeyError as exc:
                    return self._json({"error": str(exc)}, status=404)
                self._record_activity(
                    "terrain_preset_activate",
                    summary=f"Activated terrain profile {preset_id}",
                    subject_id=preset_id,
                    payload={"terrain_preset_id": preset_id},
                )
                preset = self.state.terrain_presets.get_preset(preset_id) or {}
                _sync_active_terrain_override_file(self.state.paths, dict(preset.get("values") or {}))
                return self._json({"active_preset_id": preset_id})
            if parsed.path == "/api/physics/presets":
                try:
                    preset = self.state.physics_presets.create_preset(
                        str(payload.get("name") or ""),
                        str(payload.get("description") or ""),
                        dict(payload.get("values") or {}),
                    )
                except ValueError as exc:
                    return self._json({"error": str(exc)}, status=400)
                self._record_activity(
                    "physics_preset_create",
                    summary=f"Created physics profile {preset.get('name')}",
                    subject_id=str(preset.get("id") or ""),
                    payload={"physics_preset_id": preset.get("id")},
                )
                return self._json(preset, status=201)
            if parsed.path.startswith("/api/physics/presets/") and parsed.path.endswith("/update"):
                preset_id = route_id2(parsed.path)
                updates = {
                    key: payload[key]
                    for key in ("name", "description", "values")
                    if key in payload
                }
                try:
                    preset = self.state.physics_presets.update_preset(preset_id, **updates)
                except (KeyError, ValueError) as exc:
                    return self._json({"error": str(exc)}, status=400)
                self._record_activity(
                    "physics_preset_edit",
                    summary=f"Edited physics profile {preset.get('name')}",
                    subject_id=preset_id,
                    payload={"physics_preset_id": preset_id},
                )
                return self._json(preset)
            if parsed.path.startswith("/api/physics/presets/") and parsed.path.endswith("/delete"):
                preset_id = route_id2(parsed.path)
                try:
                    deleted = self.state.physics_presets.delete_preset(preset_id)
                except ValueError as exc:
                    return self._json({"error": str(exc)}, status=400)
                if deleted:
                    self._record_activity(
                        "physics_preset_delete",
                        summary=f"Deleted physics profile {preset_id}",
                        subject_id=preset_id,
                        payload={"physics_preset_id": preset_id},
                    )
                return self._json({"deleted": deleted})
            if parsed.path == "/api/physics/presets/activate":
                preset_id = str(payload.get("preset_id") or "")
                try:
                    self.state.physics_presets.set_active_preset(preset_id)
                except KeyError as exc:
                    return self._json({"error": str(exc)}, status=404)
                self._record_activity(
                    "physics_preset_activate",
                    summary=f"Activated physics profile {preset_id}",
                    subject_id=preset_id,
                    payload={"physics_preset_id": preset_id},
                )
                return self._json({"active_preset_id": preset_id})
            if parsed.path == "/api/training/stop":
                run_id = str(payload.get("run_id") or "")
                stopped = self.state.processes.stop(run_id)
                if stopped:
                    self._record_activity("process_stop", summary=f"Stopped process {run_id}", subject_id=run_id)
                return self._json({"stopped": stopped})
            if parsed.path == "/api/folders":
                folder = self.state.history.create_folder(str(payload.get("name") or ""))
                self._record_activity("folder_create", summary=f"Created folder {folder}", subject_id=folder)
                return self._json({"folder": folder, "folders": self.state.history.get_folders()}, status=201)
            if parsed.path == "/api/folders/delete":
                result = self.state.history.delete_folder(str(payload.get("folder") or payload.get("name") or ""))
                self._record_activity(
                    "folder_delete",
                    summary=f"Removed folder {result.get('folder')}",
                    subject_id=str(result.get("folder") or ""),
                    payload=result,
                )
                return self._json({**result, "folders": self.state.history.get_folders()})
            if parsed.path == "/api/folders/rename":
                result = self.state.history.rename_folder(
                    str(payload.get("old_name") or payload.get("folder") or ""),
                    str(payload.get("new_name") or payload.get("name") or ""),
                )
                self._record_activity(
                    "folder_rename",
                    summary=f"Renamed folder {result.get('old_folder')} to {result.get('new_folder')}",
                    subject_id=str(result.get("new_folder") or ""),
                    payload=result,
                )
                return self._json({**result, "folders": self.state.history.get_folders()})
            if parsed.path == "/api/folders/assign":
                return self._json(self._assign_folders(payload))
            if parsed.path == "/api/runs/delete-preview":
                run_ids = payload.get("run_ids") or []
                if not isinstance(run_ids, list):
                    return self._json({"error": "run_ids must be a list"}, status=400)
                return self._json(
                    self.state.history.bulk_delete_preview(
                        [str(run_id) for run_id in run_ids],
                        delete_logs=bool(payload.get("delete_logs", True)),
                    )
                )
            if parsed.path == "/api/runs/delete":
                run_ids = payload.get("run_ids") or []
                if not isinstance(run_ids, list):
                    return self._json({"error": "run_ids must be a list"}, status=400)
                running_by_run = self._running_by_run_or_log_dir([str(run_id) for run_id in run_ids])
                if running_by_run:
                    return self._json(
                        {"error": "Stop running processes for selected runs before deleting them", "processes": running_by_run},
                        status=409,
                    )
                result = self.state.history.bulk_delete_runs(
                    [str(run_id) for run_id in run_ids],
                    delete_logs=bool(payload.get("delete_logs", True)),
                    confirm=bool(payload.get("confirm")),
                )
                remote_deleted = self._sync_remote_deleted_runs(result.get("run_ids") or [])
                if remote_deleted is not None:
                    result["remote_delete_requests"] = remote_deleted
                self._record_activity(
                    "bulk_run_delete",
                    summary=f"Deleted {result.get('deleted_count', 0)} selected runs",
                    payload=result,
                )
                return self._json(result)
            if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/stop"):
                run_id = route_id(parsed.path)
                stopped = self.state.processes.stop_all_for_run(run_id)
                if stopped:
                    self._record_activity(
                        "run_process_stop",
                        summary=f"Stopped processes for {run_id}",
                        subject_id=run_id,
                        payload={"stopped_ids": stopped},
                    )
                return self._json({"stopped": bool(stopped), "stopped_ids": stopped})
            if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/folder"):
                run_id = route_id(parsed.path)
                data = self._assign_folders({"run_ids": [run_id], "folder": payload.get("folder")})
                return self._json({"folder": data["folder"], "run_id": run_id, "folders": data["folders"]})
            if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/export-video-to-drive"):
                run_id = route_id(parsed.path)
                try:
                    _run, video, iteration = self._resolve_run_video(
                        run_id,
                        payload.get("checkpoint_iteration"),
                    )
                    export, started, deduplicated = self.state.google_drive.start_export(
                        run_id,
                        video,
                        checkpoint_iteration=iteration,
                    )
                except RunVideoError as exc:
                    return self._json({"error": str(exc)}, status=exc.status)
                except GoogleDriveUnavailableError as exc:
                    return self._json({"error": str(exc)}, status=503)
                except GoogleDriveBusyError as exc:
                    return self._json({"error": str(exc)}, status=409)
                except GoogleDrivePathError as exc:
                    status = 403 if "outside" in str(exc).lower() else 404
                    return self._json({"error": str(exc)}, status=status)
                result = {
                    "run_id": run_id,
                    "checkpoint_iteration": iteration,
                    "export": export,
                    "started": started,
                    "deduplicated": deduplicated,
                }
                if started:
                    self._record_activity(
                        "video_drive_export_start",
                        summary=f"Started Google Drive video export for {run_id}",
                        subject_id=run_id,
                        payload=result,
                    )
                status = 202 if started or export.get("status") in {"queued", "uploading"} else 200
                return self._json(result, status=status)
            if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/record-video"):
                run_id = route_id(parsed.path)
                run = self.state.history.get_run(run_id)
                if not run or not run.get("latest_checkpoint"):
                    return self._json({"error": "No checkpoint found for run"}, status=404)
                checkpoint = str(run["latest_checkpoint"])
                requested_iteration = payload.get("checkpoint_iteration")
                if requested_iteration not in (None, ""):
                    try:
                        iteration = int(requested_iteration)
                    except (TypeError, ValueError) as exc:
                        raise ValueError("checkpoint_iteration must be an integer") from exc
                    if iteration < 0:
                        raise ValueError("checkpoint_iteration must be zero or greater")
                    log_dir = Path(str(run.get("log_dir") or ""))
                    if not run.get("log_dir") or not log_dir.is_dir():
                        return self._json({"error": "No log directory found for run"}, status=404)
                    selected = next(
                        (
                            path
                            for candidate_iteration, path in checkpoint_inventory(log_dir)
                            if candidate_iteration == iteration
                        ),
                        None,
                    )
                    if selected is None:
                        return self._json({"error": f"Checkpoint iteration {iteration} was not found for run"}, status=404)
                    checkpoint = str(selected)
                active_media = self.state.processes.running_isaac_processes()
                if active_media:
                    return self._json(
                        {"error": "Stop the active Isaac process before starting another Isaac action.", "processes": active_media},
                        status=409,
                    )
                result = self.state.processes.start_video_recording(
                    run_id=run_id,
                    checkpoint=checkpoint,
                    device=str(payload.get("device") or "cuda:0"),
                    video_params=VideoParams.from_preset(DEFAULT_VIDEO_PRESET),
                )
                self._record_activity(
                    "video_record_start",
                    summary=f"Started video recording for {run_id} at iteration {result.get('checkpoint_iteration')}",
                    subject_id=run_id,
                    payload=result,
                )
                return self._json(result, status=201)
            if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/export-onnx"):
                run_id = route_id(parsed.path)
                run = self.state.history.get_run(run_id)
                if not run or not run.get("latest_checkpoint"):
                    return self._json({"error": "No checkpoint found for run"}, status=404)
                active_media = self.state.processes.running_isaac_processes()
                if active_media:
                    return self._json(
                        {"error": "Stop the active Isaac process before starting another Isaac action.", "processes": active_media},
                        status=409,
                    )
                result = self.state.processes.start_onnx_export(
                    run_id=run_id,
                    checkpoint=str(run["latest_checkpoint"]),
                    device=str(payload.get("device") or "cuda:0"),
                )
                self._record_activity("onnx_export_start", summary=f"Started ONNX export for {run_id}", subject_id=run_id, payload=result)
                return self._json(result, status=201)
            if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/deploy/start"):
                run_id = route_id(parsed.path)
                run = self.state.history.get_run(run_id)
                if not run:
                    return self._json({"error": "Run not found"}, status=404)
                export_first = bool(payload.get("export_first", False))
                if export_first and not run.get("latest_checkpoint"):
                    return self._json({"error": "No checkpoint found for run"}, status=404)
                if not export_first and not run.get("onnx_path"):
                    return self._json({"error": "No exported policy.onnx found for run"}, status=404)
                active_media = self.state.processes.running_isaac_processes()
                if active_media:
                    return self._json(
                        {"error": "Stop the active Isaac/deploy process before starting deploy readiness.", "processes": active_media},
                        status=409,
                    )
                result = self.state.processes.start_deploy_validation(
                    run_id=run_id,
                    export_first=export_first,
                    device=str(payload.get("device") or "cuda:0"),
                    include_ros_mock=bool(payload.get("include_ros_mock", False)),
                    include_mujoco=bool(payload.get("include_mujoco", True)),
                    use_cuda=bool(payload.get("use_cuda", False)),
                    use_tensorrt=bool(payload.get("use_tensorrt", False)),
                    mujoco_model_path=str(payload.get("mujoco_model_path") or "") or None,
                    mujoco_only=bool(payload.get("mujoco_only", False)),
                )
                self._record_activity(
                    "deploy_readiness_start",
                    summary=f"Started deploy readiness for {run_id}",
                    subject_id=run_id,
                    payload={**result, "export_first": export_first},
                )
                return self._json(result, status=201)
            if parsed.path.startswith("/api/runs/") and (
                parsed.path.endswith("/mujoco/viewer/start") or parsed.path.endswith("/mujoco/video/start")
            ):
                run_id = route_id(parsed.path)
                run = self.state.history.get_run(run_id)
                if not run:
                    return self._json({"error": "Run not found"}, status=404)
                if not run.get("onnx_path"):
                    return self._json({"error": "No exported policy.onnx found for run"}, status=404)
                defaults = deploy_defaults(self.state.paths)
                mode = "viewer" if parsed.path.endswith("/mujoco/viewer/start") else "record"
                if mode == "viewer" and not defaults.get("mujoco_viewer_available"):
                    return self._json({"error": "MuJoCo viewer is not available on this display.", "defaults": defaults}, status=409)
                if mode == "record" and not defaults.get("mujoco_encoder_available"):
                    return self._json({"error": "MuJoCo MP4 encoder dependencies are missing.", "defaults": defaults}, status=409)
                result = self.state.processes.start_mujoco_playback(
                    run_id=run_id,
                    mode=mode,
                    scenario=str(payload.get("scenario") or "stand_zero"),
                    steps=int(payload.get("steps") or defaults.get("mujoco_playback_defaults", {}).get("steps") or 1250),
                    width=int(payload.get("width") or defaults.get("mujoco_playback_defaults", {}).get("width") or 1280),
                    height=int(payload.get("height") or defaults.get("mujoco_playback_defaults", {}).get("height") or 720),
                    fps=int(payload.get("fps") or defaults.get("mujoco_playback_defaults", {}).get("fps") or 30),
                    mujoco_model_path=str(payload.get("mujoco_model_path") or "") or None,
                )
                self._record_activity(
                    "mujoco_playback_start",
                    summary=f"Started MuJoCo {mode} for {run_id}",
                    subject_id=run_id,
                    payload={**result, "mode": mode},
                )
                return self._json(result, status=201)
            if parsed.path.startswith("/api/deploy/") and parsed.path.endswith("/stop"):
                pipeline_id = route_id(parsed.path)
                stopped = self.state.processes.stop(pipeline_id)
                if stopped:
                    self._record_activity("deploy_readiness_stop", summary=f"Stopped deploy readiness {pipeline_id}", subject_id=pipeline_id)
                return self._json({"stopped": stopped, "pipeline_id": pipeline_id})
            if parsed.path.startswith("/api/mujoco/") and parsed.path.endswith("/stop"):
                process_id = route_id(parsed.path)
                stopped = self.state.processes.stop(process_id)
                if stopped:
                    self._record_activity("mujoco_playback_stop", summary=f"Stopped MuJoCo playback {process_id}", subject_id=process_id)
                return self._json({"stopped": stopped, "process_id": process_id})
            if parsed.path == "/api/open-location":
                return self._json(self._open_location(str(payload.get("path") or "")))
            if parsed.path == "/api/tensorboard/start":
                host = str(payload.get("host") or "127.0.0.1")
                port = int(payload["port"]) if payload.get("port") else None
                return self._json(self.state.processes.start_tensorboard(host=host, port=port))
            if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/rename"):
                run_id = route_id(parsed.path)
                record = self.state.history.rename_run(run_id, str(payload.get("display_name") or ""))
                self._record_activity(
                    "run_rename",
                    summary=f"Renamed run {run_id}",
                    subject_id=run_id,
                    payload={"display_name": record.get("display_name")},
                )
                return self._json({"saved": True, "run": record})
            if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/tensorboard"):
                run_id = route_id(parsed.path)
                run = self.state.history.get_run(run_id)
                if not run or not run.get("log_dir"):
                    return self._json({"error": "No log directory found for run"}, status=404)
                host = str(payload.get("host") or "127.0.0.1")
                port = int(payload["port"]) if payload.get("port") else None
                result = self.state.processes.start_tensorboard(
                    host=host,
                    port=port,
                    logdir=Path(str(run["log_dir"])),
                    source_run_id=run_id,
                )
                self._record_activity("tensorboard_start", summary=f"Started TensorBoard for {run_id}", subject_id=run_id, payload=result)
                return self._json(result)
            if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/delete"):
                run_id = route_id(parsed.path)
                running = self._running_for_run_or_log_dir(run_id)
                if running:
                    return self._json(
                        {"error": "Stop running processes for this run before deleting it", "processes": running},
                        status=409,
                    )
                result = self.state.history.delete_run(
                    run_id,
                    confirmation=str(payload.get("confirmation") or ""),
                    delete_logs=bool(payload.get("delete_logs", True)),
                    confirm=bool(payload.get("confirm")),
                )
                remote_deleted = self._sync_remote_deleted_runs([run_id])
                if remote_deleted is not None:
                    result["remote_delete_requests"] = remote_deleted
                self._record_activity(
                    "run_delete",
                    summary=f"Deleted run {run_id}",
                    subject_id=run_id,
                    payload=result,
                )
                return self._json(result)
            if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/compact"):
                run_id = route_id(parsed.path)
                running = self.state.processes.running_for_run(run_id)
                if running:
                    return self._json(
                        {"error": "Stop running processes for this run before compacting it", "processes": running},
                        status=409,
                    )
                result = self.state.history.compact_run(
                    run_id,
                    confirmation=str(payload.get("confirmation") or ""),
                )
                self._record_activity(
                    "run_compact",
                    summary=f"Compacted run {run_id}",
                    subject_id=run_id,
                    payload=result,
                )
                return self._json(result)
            if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/notes"):
                run_id = route_id(parsed.path)
                self.state.history.set_note(run_id, str(payload.get("notes") or ""))
                return self._json({"saved": True, "run_id": run_id})
            if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/play"):
                run_id = route_id(parsed.path)
                run = self.state.history.get_run(run_id)
                if not run or not run.get("latest_checkpoint"):
                    return self._json({"error": "No checkpoint found for run"}, status=404)
                active_media = self.state.processes.running_isaac_processes()
                if active_media:
                    return self._json(
                        {"error": "Stop the active Isaac process before starting another Isaac action.", "processes": active_media},
                        status=409,
                    )
                result = self.state.processes.start_play(
                    run_id=run_id,
                    checkpoint=str(run["latest_checkpoint"]),
                    device=str(payload.get("device") or "cuda:0"),
                )
                self._record_activity("play_start", summary=f"Started play for {run_id}", subject_id=run_id, payload=result)
                return self._json(result, status=201)
            if parsed.path == "/api/convergence/settings":
                from dataclasses import asdict
                from .convergence import PRESETS, apply_settings
                cfg = apply_settings(payload, self.state.paths.convergence_config_file)
                return self._json({"saved": True, "config": asdict(cfg), "presets": PRESETS})
        except ProcessStartError as exc:
            return self._json(exc.payload, status=500)
        except CudaPreflightError as exc:
            return self._json(exc.payload, status=409)
        except ValueError as exc:
            return self._json({"error": str(exc)}, status=400)
        self._not_found()

    def do_PATCH(self) -> None:
        return self._handle_request(self._do_PATCH)

    def _do_PATCH(self) -> None:
        parsed = urlparse(self.path)
        if self._is_autopilot_path(parsed.path):
            return self._handle_autopilot_write(parsed, method="PATCH")
        self._not_found()

    @staticmethod
    def _is_autopilot_path(path: str) -> bool:
        return path == "/api/autopilot" or path.startswith("/api/autopilot/")

    def _handle_autopilot_get(self, parsed) -> None:
        try:
            self._require_autopilot_auth()
            segments = self._autopilot_segments(parsed.path)
            if segments == ["capabilities"]:
                self._reject_autopilot_query(parsed.query)
                service = getattr(self.state, "autopilot", None)
                if service is None:
                    payload = autopilot_capabilities(enabled=False)
                    payload["service_state"] = "unavailable"
                    return self._json(payload)
                method = self._autopilot_method(service, "capabilities")
                return self._json(self._autopilot_value(method()))

            service = self._require_autopilot_service()
            if segments == ["campaigns"]:
                query = self._autopilot_query(parsed.query, allowed={"state", "limit"})
                state = query.get("state", [""])[0].strip() or None
                if state is not None and (len(state) > 64 or not re.fullmatch(r"[a-z_]+", state)):
                    raise AutopilotRouteError("invalid_query", "state is invalid")
                try:
                    limit = int(query.get("limit", ["100"])[0])
                except (TypeError, ValueError) as exc:
                    raise AutopilotRouteError("invalid_query", "limit must be an integer") from exc
                if not 1 <= limit <= 100:
                    raise AutopilotRouteError("invalid_query", "limit must be within 1..100")
                campaigns = self._autopilot_method(service, "list_campaigns")(
                    state=state,
                    limit=limit,
                )
                if isinstance(campaigns, dict) and "campaigns" in campaigns:
                    return self._json(self._autopilot_value(campaigns))
                return self._json({"campaigns": self._autopilot_value(campaigns or [])})

            if len(segments) < 2 or segments[0] != "campaigns":
                raise AutopilotRouteError("not_found", "Autopilot endpoint not found", 404)
            campaign_id = self._autopilot_identifier(segments[1], "campaign_id")
            if len(segments) == 2:
                self._reject_autopilot_query(parsed.query)
                campaign = self._autopilot_method(service, "get_campaign")(campaign_id)
                if campaign is None:
                    raise AutopilotRouteError("campaign_not_found", "Campaign not found", 404)
                return self._json(self._wrap_autopilot("campaign", campaign))
            if len(segments) == 3 and segments[2] == "decision-context":
                self._reject_autopilot_query(parsed.query)
                context = self._autopilot_method(
                    service,
                    "get_decision_context",
                    "decision_context",
                )(campaign_id)
                if context is None:
                    raise AutopilotRouteError("campaign_not_found", "Campaign not found", 404)
                return self._json(self._autopilot_value(context))
            if len(segments) == 3 and segments[2] == "events":
                query = self._autopilot_query(parsed.query, allowed={"after", "limit"})
                try:
                    after = int(query.get("after", ["0"])[0])
                    limit = int(query.get("limit", ["500"])[0])
                except (TypeError, ValueError) as exc:
                    raise AutopilotRouteError(
                        "invalid_query", "after and limit must be integers"
                    ) from exc
                if after < 0:
                    raise AutopilotRouteError(
                        "invalid_query", "after must be non-negative"
                    )
                if not 1 <= limit <= 500:
                    raise AutopilotRouteError(
                        "invalid_query", "limit must be within 1..500"
                    )
                events = self._autopilot_method(service, "list_events")(
                    campaign_id,
                    after=after,
                    limit=limit,
                )
                if isinstance(events, dict) and "events" in events:
                    return self._json(self._autopilot_value(events))
                return self._json({"events": self._autopilot_value(events or [])})
            if len(segments) == 3 and segments[2] == "artifacts":
                self._reject_autopilot_query(parsed.query)
                artifacts = self._autopilot_method(service, "list_artifacts")(campaign_id)
                if isinstance(artifacts, dict) and "artifacts" in artifacts:
                    return self._json(self._autopilot_value(artifacts))
                return self._json({"artifacts": self._autopilot_value(artifacts or [])})
            if len(segments) == 4 and segments[2] == "artifacts":
                self._reject_autopilot_query(parsed.query)
                artifact_id = self._autopilot_identifier(segments[3], "artifact_id")
                artifact = self._autopilot_method(
                    service,
                    "get_artifact",
                )(campaign_id, artifact_id)
                if artifact is None:
                    raise AutopilotRouteError("artifact_not_found", "Artifact not found", 404)
                metadata, _content = self._autopilot_artifact_parts(artifact)
                normalized = self._autopilot_value(metadata)
                if not isinstance(normalized, dict):
                    raise AutopilotRouteError(
                        "autopilot_service_contract_error",
                        "Autopilot artifact metadata is invalid",
                        503,
                    )
                normalized = dict(normalized)
                normalized["download_url"] = (
                    f"/api/autopilot/campaigns/{campaign_id}/artifacts/{artifact_id}/download"
                )
                return self._json({"artifact": normalized})
            if len(segments) == 5 and segments[2] == "artifacts" and segments[4] == "download":
                self._reject_autopilot_query(parsed.query)
                artifact_id = self._autopilot_identifier(segments[3], "artifact_id")
                artifact = self._autopilot_method(service, "get_artifact")(
                    campaign_id,
                    artifact_id,
                )
                metadata, content = self._autopilot_artifact_parts(artifact)
                return self._send_autopilot_artifact(metadata, content)
            if len(segments) == 3 and segments[2] == "compare":
                query = self._autopilot_query(parsed.query, allowed={"trial_ids"})
                raw_ids = []
                for value in query.get("trial_ids", []):
                    raw_ids.extend(value.split(","))
                trial_ids = [
                    self._autopilot_identifier(value.strip(), "trial_id")
                    for value in raw_ids
                    if value.strip()
                ]
                if not 2 <= len(trial_ids) <= 12 or len(set(trial_ids)) != len(trial_ids):
                    raise AutopilotRouteError(
                        "invalid_query",
                        "trial_ids must contain 2..12 unique identifiers",
                    )
                comparison = self._autopilot_method(service, "compare_trials")(
                    campaign_id,
                    trial_ids,
                )
                return self._json(self._autopilot_value(comparison))
            if len(segments) == 3 and segments[2] == "patch-export":
                self._reject_autopilot_query(parsed.query)
                export = self._autopilot_method(service, "export_patch", "patch_export")(
                    campaign_id
                )
                if export is None:
                    raise AutopilotRouteError("patch_export_not_found", "Patch handoff not found", 404)
                return self._send_autopilot_patch_export(campaign_id, export)
            raise AutopilotRouteError("not_found", "Autopilot endpoint not found", 404)
        except Exception as exc:
            return self._handle_autopilot_exception(exc)

    def _handle_autopilot_write(self, parsed, *, method: str) -> None:
        try:
            self._require_autopilot_auth()
            payload = self._autopilot_payload()
            idempotency_key, expected_revision = self._autopilot_mutation_metadata(payload)
            segments = self._autopilot_segments(parsed.path)
            self._reject_autopilot_query(parsed.query)
            service = self._require_autopilot_service()
            request_payload = dict(payload)
            request_payload.pop("expected_revision", None)

            if method == "POST" and segments == ["campaigns"]:
                if expected_revision != 0:
                    raise AutopilotRouteError(
                        "invalid_expected_revision",
                        "new campaigns require expected_revision=0",
                    )
                self._validate_autopilot_draft_wrapper(request_payload)
                result = self._autopilot_method(service, "create_campaign")(
                    request_payload,
                    idempotency_key=idempotency_key,
                )
                return self._json(self._wrap_autopilot("campaign", result), status=201)

            if len(segments) < 2 or segments[0] != "campaigns":
                raise AutopilotRouteError("not_found", "Autopilot endpoint not found", 404)
            campaign_id = self._autopilot_identifier(segments[1], "campaign_id")
            if method == "PATCH" and len(segments) == 2:
                self._validate_autopilot_draft_wrapper(request_payload)
                result = self._autopilot_method(service, "update_draft")(
                    campaign_id,
                    request_payload,
                    idempotency_key=idempotency_key,
                    expected_revision=expected_revision,
                )
                return self._json(self._wrap_autopilot("campaign", result))
            if method != "POST" or len(segments) != 3:
                raise AutopilotRouteError("not_found", "Autopilot endpoint not found", 404)
            action = segments[2]
            if action == "heartbeat":
                self._require_autopilot_keys(
                    request_payload,
                    allowed={"advisor_metadata"},
                )
                metadata = self._autopilot_advisor_metadata(
                    request_payload.get("advisor_metadata")
                )
                result = self._autopilot_method(service, "advisor_heartbeat")(
                    campaign_id,
                    advisor_metadata=metadata,
                    idempotency_key=idempotency_key,
                    expected_revision=expected_revision,
                )
                return self._json(self._wrap_autopilot("campaign", result))
            if action == "arm":
                self._require_autopilot_keys(request_payload, allowed=set())
                result = self._autopilot_method(service, "arm_campaign")(
                    campaign_id,
                    idempotency_key=idempotency_key,
                    expected_revision=expected_revision,
                )
                return self._json(self._wrap_autopilot("campaign", result))
            if action == "pause":
                self._require_autopilot_keys(
                    request_payload,
                    allowed={"reason", "advisor_metadata"},
                )
                kwargs = {}
                if "reason" in request_payload:
                    kwargs["reason"] = self._autopilot_reason(request_payload["reason"])
                if "advisor_metadata" in request_payload:
                    kwargs["advisor_metadata"] = self._autopilot_advisor_metadata(
                        request_payload["advisor_metadata"]
                    )
                result = self._autopilot_method(service, "pause_campaign")(
                    campaign_id,
                    idempotency_key=idempotency_key,
                    expected_revision=expected_revision,
                    **kwargs,
                )
                return self._json(self._wrap_autopilot("campaign", result))
            if action == "resume":
                self._require_autopilot_keys(request_payload, allowed=set())
                result = self._autopilot_method(service, "resume_campaign")(
                    campaign_id,
                    idempotency_key=idempotency_key,
                    expected_revision=expected_revision,
                )
                return self._json(self._wrap_autopilot("campaign", result))
            if action == "stop":
                self._require_autopilot_keys(
                    request_payload,
                    allowed={"mode", "reason", "advisor_metadata"},
                )
                mode = request_payload.get("mode", "after_current")
                if mode not in {"after_current", "emergency"}:
                    raise AutopilotRouteError(
                        "invalid_stop_mode",
                        "stop mode must be after_current or emergency",
                    )
                kwargs = {"after_current": mode == "after_current"}
                if "reason" in request_payload:
                    kwargs["reason"] = self._autopilot_reason(request_payload["reason"])
                if "advisor_metadata" in request_payload:
                    kwargs["advisor_metadata"] = self._autopilot_advisor_metadata(
                        request_payload["advisor_metadata"]
                    )
                result = self._autopilot_method(service, "stop_campaign")(
                    campaign_id,
                    idempotency_key=idempotency_key,
                    expected_revision=expected_revision,
                    **kwargs,
                )
                return self._json(self._wrap_autopilot("campaign", result))
            if action == "decisions":
                metadata = self._autopilot_advisor_metadata(
                    request_payload.get("advisor_metadata")
                )
                if "patch_proposal" in request_payload:
                    if set(request_payload) != {
                        "decision",
                        "patch_proposal",
                        "advisor_metadata",
                    }:
                        raise AutopilotRouteError(
                            "invalid_patch_wrapper",
                            "patch proposal requests require only decision and patch_proposal",
                        )
                    result = self._autopilot_method(service, "submit_patch_proposal")(
                        campaign_id,
                        request_payload["decision"],
                        request_payload["patch_proposal"],
                        advisor_metadata=metadata,
                        idempotency_key=idempotency_key,
                        expected_revision=expected_revision,
                    )
                else:
                    if set(request_payload) != {"decision", "advisor_metadata"}:
                        raise AutopilotRouteError(
                            "invalid_decision_wrapper",
                            "advisor decisions require only decision and advisor_metadata",
                        )
                    result = self._autopilot_method(service, "submit_decision")(
                        campaign_id,
                        {
                            "decision": request_payload["decision"],
                            "advisor_metadata": metadata,
                        },
                        idempotency_key=idempotency_key,
                        expected_revision=expected_revision,
                    )
                return self._json(self._wrap_autopilot("campaign", result), status=201)
            raise AutopilotRouteError("not_found", "Autopilot endpoint not found", 404)
        except Exception as exc:
            return self._handle_autopilot_exception(exc)

    def _require_autopilot_auth(self) -> None:
        expected = os.environ.get("REDRHEX_AUTOPILOT_BEARER_TOKEN", "").strip()
        if not expected:
            return
        values = self._header_values("Authorization")
        supplied = values[0] if len(values) == 1 else ""
        if not secrets.compare_digest(supplied, f"Bearer {expected}"):
            raise AutopilotRouteError(
                "unauthorized",
                "Autopilot bearer authentication failed",
                401,
            )

    def _require_autopilot_service(self):
        service = getattr(self.state, "autopilot", None)
        if service is None:
            raise AutopilotRouteError(
                "autopilot_unavailable",
                "Autopilot service is disabled or unavailable",
                503,
            )
        return service

    @staticmethod
    def _require_autopilot_keys(payload: dict, *, allowed: set[str]) -> None:
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise AutopilotRouteError(
                "invalid_payload",
                f"unknown request field(s): {', '.join(unknown)}",
            )

    @staticmethod
    def _autopilot_reason(value: object) -> str:
        if not isinstance(value, str) or not value.strip() or len(value) > 1000:
            raise AutopilotRouteError(
                "invalid_reason",
                "reason must be a non-empty string of at most 1000 characters",
            )
        return value.strip()

    @staticmethod
    def _autopilot_advisor_metadata(value: object) -> dict[str, str]:
        if not isinstance(value, dict):
            raise AutopilotRouteError(
                "invalid_advisor_metadata",
                "advisor_metadata must be an object",
            )
        required = {
            "schema_version",
            "skill_version",
            "prompt_version",
            "declared_model",
            "reasoning_effort",
        }
        if set(value) != required:
            raise AutopilotRouteError(
                "invalid_advisor_metadata",
                "advisor_metadata fields do not match the V1 contract",
            )
        if value.get("schema_version") != _AUTOPILOT_ADVISOR_METADATA_SCHEMA_VERSION:
            raise AutopilotRouteError(
                "unsupported_schema_version",
                "advisor_metadata schema_version is unsupported",
            )
        if value.get("skill_version") != _AUTOPILOT_SKILL_VERSION:
            raise AutopilotRouteError(
                "invalid_advisor_metadata",
                "advisor_metadata skill_version is unsupported",
            )
        if value.get("prompt_version") != _AUTOPILOT_PROMPT_VERSION:
            raise AutopilotRouteError(
                "invalid_advisor_metadata",
                "advisor_metadata prompt_version is unsupported",
            )
        model = value.get("declared_model")
        if (
            not isinstance(model, str)
            or not model
            or len(model) > 128
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]*", model) is None
        ):
            raise AutopilotRouteError(
                "invalid_advisor_metadata",
                "advisor_metadata declared_model is invalid",
            )
        if value.get("reasoning_effort") != "medium":
            raise AutopilotRouteError(
                "invalid_advisor_metadata",
                "advisor_metadata reasoning_effort must be medium",
            )
        return dict(value)

    @staticmethod
    def _validate_autopilot_draft_wrapper(payload: dict) -> None:
        PanelHandler._require_autopilot_keys(payload, allowed=_AUTOPILOT_DRAFT_FIELDS)
        if payload.get("schema_version") != GOAL_SCHEMA_VERSION:
            raise AutopilotRouteError(
                "unsupported_schema_version",
                f"draft schema_version must be {GOAL_SCHEMA_VERSION}",
            )

    @staticmethod
    def _autopilot_method(service, *names: str):
        for name in names:
            method = getattr(service, name, None)
            if callable(method):
                return method
        raise AutopilotRouteError(
            "autopilot_service_contract_error",
            f"Autopilot service does not implement {names[0]}",
            503,
        )

    @staticmethod
    def _autopilot_segments(path: str) -> list[str]:
        prefix = "/api/autopilot"
        suffix = path[len(prefix):]
        if not suffix.startswith("/"):
            raise AutopilotRouteError("not_found", "Autopilot endpoint not found", 404)
        raw_segments = suffix[1:].split("/")
        if not raw_segments or any(not segment for segment in raw_segments):
            raise AutopilotRouteError("not_found", "Autopilot endpoint not found", 404)
        segments = [unquote(segment) for segment in raw_segments]
        if any("/" in segment or "\\" in segment for segment in segments):
            raise AutopilotRouteError("invalid_identifier", "Encoded path separators are forbidden")
        return segments

    @staticmethod
    def _autopilot_identifier(value: str, name: str) -> str:
        if not isinstance(value, str) or _AUTOPILOT_ID_RE.fullmatch(value) is None:
            raise AutopilotRouteError("invalid_identifier", f"{name} is invalid")
        return value

    @staticmethod
    def _autopilot_query(query: str, *, allowed: set[str]) -> dict[str, list[str]]:
        parsed = parse_qs(query, keep_blank_values=True)
        unknown = sorted(set(parsed) - allowed)
        if unknown:
            raise AutopilotRouteError(
                "invalid_query",
                f"unknown query parameter(s): {', '.join(unknown)}",
            )
        return parsed

    @classmethod
    def _reject_autopilot_query(cls, query: str) -> None:
        cls._autopilot_query(query, allowed=set())

    def _autopilot_payload(self) -> dict:
        content_types = self._header_values("Content-Type")
        if len(content_types) != 1 or content_types[0].split(";", 1)[0].strip().lower() != "application/json":
            raise AutopilotRouteError(
                "invalid_content_type",
                "Content-Type must be application/json",
                415,
            )
        lengths = self._header_values("Content-Length")
        if len(lengths) != 1:
            raise AutopilotRouteError(
                "content_length_required",
                "one Content-Length header is required",
                411,
            )
        try:
            length = int(lengths[0])
        except (TypeError, ValueError) as exc:
            raise AutopilotRouteError(
                "invalid_content_length",
                "Content-Length must be an integer",
            ) from exc
        if length < 0 or length > _AUTOPILOT_MAX_REQUEST_BYTES:
            raise AutopilotRouteError(
                "request_too_large" if length > _AUTOPILOT_MAX_REQUEST_BYTES else "invalid_content_length",
                "Autopilot request body is too large" if length > _AUTOPILOT_MAX_REQUEST_BYTES else "Content-Length is invalid",
                413 if length > _AUTOPILOT_MAX_REQUEST_BYTES else 400,
            )
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise AutopilotRouteError("incomplete_request", "Request body is incomplete")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AutopilotRouteError("invalid_json", "Request body must be valid UTF-8 JSON") from exc
        if not isinstance(payload, dict):
            raise AutopilotRouteError("invalid_payload", "Request body must be a JSON object")
        return payload

    def _autopilot_mutation_metadata(self, payload: dict) -> tuple[str, int]:
        keys = self._header_values("Idempotency-Key")
        if len(keys) != 1 or _AUTOPILOT_IDEMPOTENCY_RE.fullmatch(keys[0]) is None:
            raise AutopilotRouteError(
                "invalid_idempotency_key",
                "Idempotency-Key must be a safe 8..128 character identifier",
            )
        matches = self._header_values("If-Match")
        match = _AUTOPILOT_IF_MATCH_RE.fullmatch(matches[0]) if len(matches) == 1 else None
        if match is None:
            raise AutopilotRouteError(
                "invalid_if_match",
                'If-Match must contain one quoted non-negative revision, for example "3"',
            )
        header_revision = int(match.group(1))
        body_revision = payload.get("expected_revision")
        if isinstance(body_revision, bool) or not isinstance(body_revision, int) or body_revision < 0:
            raise AutopilotRouteError(
                "invalid_expected_revision",
                "expected_revision must be a non-negative integer",
            )
        secondary = self._header_values("X-Expected-Revision")
        if len(secondary) > 1:
            raise AutopilotRouteError(
                "invalid_expected_revision_header",
                "X-Expected-Revision may appear at most once",
            )
        if secondary:
            try:
                secondary_revision = int(secondary[0])
            except ValueError as exc:
                raise AutopilotRouteError(
                    "invalid_expected_revision_header",
                    "X-Expected-Revision must be a non-negative integer",
                ) from exc
            if secondary_revision < 0 or secondary_revision != header_revision:
                raise AutopilotRouteError(
                    "revision_header_mismatch",
                    "revision headers disagree",
                )
        if body_revision != header_revision:
            raise AutopilotRouteError(
                "revision_header_mismatch",
                "expected_revision does not match If-Match",
            )
        return keys[0], body_revision

    def _header_values(self, name: str) -> list[str]:
        getter = getattr(self.headers, "get_all", None)
        if callable(getter):
            return [str(value) for value in (getter(name) or [])]
        value = self.headers.get(name)
        return [] if value is None else [str(value)]

    @staticmethod
    def _autopilot_value(value):
        serializer = getattr(value, "to_dict", None)
        if callable(serializer):
            return serializer()
        if isinstance(value, tuple):
            return [PanelHandler._autopilot_value(item) for item in value]
        if isinstance(value, list):
            return [PanelHandler._autopilot_value(item) for item in value]
        return value

    @staticmethod
    def _autopilot_artifact_parts(value) -> tuple[object, bytes]:
        if (
            not isinstance(value, tuple)
            or len(value) != 2
            or not isinstance(value[1], bytes)
        ):
            raise AutopilotRouteError(
                "autopilot_service_contract_error",
                "Autopilot artifact service returned an invalid record",
                503,
            )
        return value[0], value[1]

    @classmethod
    def _wrap_autopilot(cls, key: str, value) -> dict:
        normalized = cls._autopilot_value(value)
        if isinstance(normalized, dict) and key in normalized:
            return normalized
        return {key: normalized}

    def _send_autopilot_patch_export(self, campaign_id: str, export) -> None:
        metadata, body = self._autopilot_artifact_parts(export)
        normalized = self._autopilot_value(metadata)
        media_type = (
            str(normalized.get("media_type") or "application/json")
            if isinstance(normalized, dict)
            else "application/json"
        )
        filename = f"{campaign_id}-patch-handoff.json"
        self.send_response(200)
        self.send_header("Content-Type", media_type)
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_autopilot_artifact(self, metadata, body: bytes) -> None:
        normalized = self._autopilot_value(metadata)
        if not isinstance(normalized, dict):
            raise AutopilotRouteError(
                "autopilot_service_contract_error",
                "Autopilot artifact metadata is invalid",
                503,
            )
        media_type = str(normalized.get("media_type") or "application/octet-stream")
        artifact_id = self._autopilot_identifier(
            str(normalized.get("id") or "artifact"),
            "artifact_id",
        )
        self.send_response(200)
        self.send_header("Content-Type", media_type)
        self.send_header("Content-Disposition", f'attachment; filename="{artifact_id}"')
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _handle_autopilot_exception(self, exc: Exception) -> None:
        if isinstance(exc, AutopilotRouteError):
            return self._autopilot_error(exc.code, str(exc), exc.status, exc.details)
        if isinstance(exc, AutopilotValidationError):
            return self._autopilot_error("validation_error", str(exc), 400)
        if isinstance(exc, CampaignNotFoundError):
            return self._autopilot_error(exc.code, str(exc), 404)
        if isinstance(exc, AutopilotConflictError):
            details = None
            if exc.current_revision is not None:
                details = {"current_revision": exc.current_revision}
            return self._autopilot_error(exc.code, str(exc), 409, details)
        if isinstance(exc, AutopilotBudgetError):
            return self._autopilot_error(exc.code, str(exc), 409)
        if isinstance(exc, AutopilotStoreError):
            traceback.print_exc()
            return self._autopilot_error(
                exc.code,
                "Autopilot storage request failed safely",
                500,
            )
        if isinstance(exc, KeyError):
            return self._autopilot_error("not_found", "Autopilot record not found", 404)
        if isinstance(exc, ValueError):
            return self._autopilot_error("validation_error", str(exc), 400)
        status = getattr(exc, "status", None)
        code = getattr(exc, "code", None)
        if isinstance(status, int) and 400 <= status <= 599 and isinstance(code, str):
            return self._autopilot_error(
                code,
                str(exc) or "Autopilot request failed",
                status,
                getattr(exc, "details", None),
            )
        traceback.print_exc()
        return self._autopilot_error(
            "internal_error",
            "Autopilot request failed safely",
            500,
        )

    def _autopilot_error(
        self,
        code: str,
        message: str,
        status: int,
        details: object | None = None,
    ) -> None:
        payload = {
            "schema_version": _AUTOPILOT_ERROR_SCHEMA_VERSION,
            "error": message,
            "code": code,
            "message": message,
        }
        if details is not None:
            payload["details"] = details
        self._json(payload, status=status)

    def _handle_request(self, handler) -> None:
        try:
            return handler()
        except (BrokenPipeError, ConnectionResetError):
            return None
        except Exception as exc:
            traceback.print_exc()
            try:
                return self._json({"error": f"Internal server error: {exc}"}, status=500)
            except (BrokenPipeError, ConnectionResetError):
                return None

    def log_message(self, fmt: str, *args) -> None:
        print(f"[training-panel] {self.address_string()} - {fmt % args}")

    def _payload(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _record_activity(
        self,
        event_type: str,
        *,
        summary: str = "",
        subject_id: str = "",
        payload: dict | None = None,
    ) -> None:
        try:
            self.state.activity.record(
                event_type,
                summary=summary,
                subject_id=subject_id,
                payload=payload or {},
            )
        except Exception as exc:
            print(f"[training-panel] activity log skipped: {exc}")

    def _running_by_run(self, run_ids: list[str]) -> dict[str, list[dict]]:
        running_by_run = {}
        for run_id in run_ids:
            running = self.state.processes.running_for_run(str(run_id))
            if running:
                running_by_run[str(run_id)] = running
        return running_by_run

    def _running_for_run_or_log_dir(self, run_id: str) -> list[dict]:
        running = list(self.state.processes.running_for_run(str(run_id)))
        seen = {str(process.get("run_id")) for process in running}
        preview = self.state.history.delete_preview(str(run_id))
        for item in (preview or {}).get("paths") or []:
            if item.get("kind") != "rsl_rl_log_dir":
                continue
            for process in self.state.processes.running_for_log_dir(item.get("path") or ""):
                process_id = str(process.get("run_id"))
                if process_id not in seen:
                    running.append(process)
                    seen.add(process_id)
        return running

    def _running_by_run_or_log_dir(self, run_ids: list[str]) -> dict[str, list[dict]]:
        running_by_run = {}
        for run_id in run_ids:
            running = self._running_for_run_or_log_dir(str(run_id))
            if running:
                running_by_run[str(run_id)] = running
        return running_by_run

    def _sync_remote_deleted_runs(self, run_ids: list[str] | None = None) -> int | None:
        try:
            from .remote_worker import RemoteWorker
            from .supabase_client import SupabaseClient

            config = self.state.remote_worker.config()
            if not config.configured:
                return None
            client = SupabaseClient(config, timeout=5.0)
            worker = RemoteWorker(
                config,
                self.state.paths,
                client,
                state_store=self.state.remote_state,
            )
            tombstones = None
            if run_ids is not None:
                tombstones = self.state.history.deleted_run_tombstones(run_ids=run_ids)
            return worker.sync_deleted_runs(tombstones=tombstones)
        except Exception as exc:
            print(f"[training-panel] remote delete sync skipped: {exc}")
            return None

    def _assign_folders(self, payload: dict) -> dict:
        raw_run_ids = payload.get("run_ids")
        if not isinstance(raw_run_ids, list):
            raise ValueError("run_ids must be a list")
        run_ids = [str(run_id).strip() for run_id in raw_run_ids if str(run_id or "").strip()]
        if not run_ids:
            raise ValueError("at least one run_id is required")
        raw_folder = payload.get("folder")
        folder = str(raw_folder).strip() if raw_folder is not None else None
        if folder == "":
            folder = None
        runs = self.state.history.assign_runs_to_folder(run_ids, folder)
        return {
            "folder": folder,
            "run_ids": [run.get("id") for run in runs],
            "runs": runs,
            "folders": self.state.history.get_folders(),
        }

    def _open_location(self, requested_path: str) -> dict:
        if not requested_path:
            raise ValueError("path is required")
        path = Path(requested_path).expanduser()
        if not path.exists():
            raise ValueError("path does not exist")
        resolved = path.resolve()
        allowed_roots = (self.state.paths.rsl_rl_log_root, self.state.paths.panel_log_root)
        if not any(_is_within(resolved, root) for root in allowed_roots):
            raise ValueError("Refusing to open a path outside repo-owned log roots")
        opener = shutil.which("xdg-open") or shutil.which("gio")
        command = f"xdg-open {shlex.quote(str(resolved))}"
        opened = False
        error = ""
        if opener:
            argv = [opener, str(resolved)] if Path(opener).name == "xdg-open" else [opener, "open", str(resolved)]
            command = " ".join(shlex.quote(part) for part in argv)
            try:
                subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                opened = True
            except OSError as exc:
                error = str(exc)
        return {
            "path": str(resolved),
            "opened": opened,
            "opener": Path(opener).name if opener else None,
            "command": command,
            "error": error,
        }

    def _send_run_scalars(self, run_id: str, query: dict) -> None:
        from .convergence import MAX_SCALAR_POINTS, ConvergenceChecker, downsample, requested_tags

        run = self.state.history.get_run(run_id)
        if not run:
            return self._json({"error": "Run not found"}, status=404)
        log_dir = run.get("log_dir")
        if not log_dir or not Path(log_dir).is_dir():
            return self._json({"tags": {}, "log_dir": log_dir or "", "reason": "no log directory"})

        try:
            points_limit = int(query.get("points", ["200"])[0])
        except ValueError:
            points_limit = 200
        points_limit = max(2, min(points_limit, MAX_SCALAR_POINTS))

        checker = ConvergenceChecker()
        series: dict[str, list[list[float]]] = {}
        for tag in requested_tags(query.get("tags", [""])[0]):
            scalars = _scalar_cache_get(Path(log_dir), tag, checker)
            if scalars:
                series[tag] = [[step, value] for step, value in downsample(scalars, points_limit)]
        return self._json({"tags": series, "log_dir": str(log_dir)})

    def _resolve_run_video(
        self,
        run_id: str,
        requested_iteration: object = None,
    ) -> tuple[dict, Path, int | None]:
        run = self.state.history.get_run(run_id)
        if not run:
            raise RunVideoError("Run not found", 404)
        video_path = run.get("latest_video") if run else None
        iteration = None
        if requested_iteration not in (None, ""):
            try:
                iteration = int(requested_iteration)
            except (TypeError, ValueError):
                raise RunVideoError("checkpoint_iteration must be an integer", 400) from None
            if iteration < 0:
                raise RunVideoError("checkpoint_iteration must be zero or greater", 400)
            checkpoint = next(
                (item for item in (run.get("checkpoint_history") or []) if item.get("iteration") == iteration),
                None,
            )
            if checkpoint is None:
                raise RunVideoError(f"Checkpoint iteration {iteration} was not found for run", 404)
            video_path = checkpoint.get("video")
            if not video_path:
                raise RunVideoError(f"No recorded video found for checkpoint iteration {iteration}", 404)
        video = Path(str(video_path)) if video_path else None
        if not video or not video.exists() or not video.is_file():
            raise RunVideoError("No recorded video found for run", 404)
        log_dir = Path(str(run.get("log_dir"))) if run and run.get("log_dir") else None
        if not log_dir or not log_dir.exists() or not log_dir.is_dir():
            raise RunVideoError("No log directory found for run", 404)
        resolved_video = video.resolve()
        resolved_log_dir = log_dir.resolve()
        resolved_root = (self.state.paths.repo_root / "logs" / "rsl_rl").resolve()
        if not _is_within(resolved_log_dir, resolved_root):
            raise RunVideoError("Run log directory is outside the RSL-RL log root", 403)
        if not _is_within(resolved_video, resolved_log_dir):
            raise RunVideoError("Video path is outside the selected run log directory", 403)
        return run, resolved_video, iteration

    def _send_run_video(self, run_id: str, query: dict | None = None) -> None:
        requested_iteration = (query or {}).get("checkpoint_iteration", [""])[0]
        try:
            _run, resolved_video, _iteration = self._resolve_run_video(run_id, requested_iteration)
        except RunVideoError as exc:
            return self._json({"error": str(exc)}, status=exc.status)
        self._send_file_response(resolved_video, "video/mp4")

    def _send_run_mujoco_video(self, run_id: str) -> None:
        run = self.state.history.get_run(run_id)
        video = Path(str(run.get("latest_mujoco_video"))) if run and run.get("latest_mujoco_video") else None
        if not video or not video.exists() or not video.is_file():
            return self._json({"error": "No recorded MuJoCo video found for run"}, status=404)
        log_dir = Path(str(run.get("log_dir") or ""))
        if not log_dir.is_dir():
            return self._json({"error": "No log directory found for run"}, status=404)
        resolved_video = video.resolve()
        resolved_root = log_dir.resolve()
        if resolved_video != resolved_root and resolved_root not in resolved_video.parents:
            return self._json({"error": "MuJoCo video path is outside the run log directory"}, status=403)
        self._send_file_response(resolved_video, "video/mp4")

    def _send_run_tensorboard_summary(self, run_id: str) -> None:
        run = self.state.history.get_run(run_id)
        if not run:
            return self._json({"error": "Run not found"}, status=404)
        log_dir = Path(str(run.get("log_dir") or ""))
        if not log_dir.is_dir():
            return self._json({"error": "No log directory found for run"}, status=404)

        try:
            from .tensorboard_summary import ensure_tensorboard_summary, tensorboard_summary_path

            title = str(run.get("display_name") or run.get("id") or log_dir.name)
            summary = ensure_tensorboard_summary(log_dir, title=title)
            summary_path = Path(str(summary or run.get("tensorboard_summary_path") or tensorboard_summary_path(log_dir)))
            if summary:
                self.state.history.update_run(
                    run_id,
                    tensorboard_summary_path=str(summary),
                    tensorboard_summary_status="completed",
                    tensorboard_summary_error=None,
                )
        except Exception as exc:
            return self._json({"error": f"TensorBoard summary generation failed: {exc}"}, status=500)

        if not summary_path.exists() or not summary_path.is_file():
            return self._json({"error": "No TensorBoard summary image found for run"}, status=404)
        resolved_summary = summary_path.resolve()
        resolved_root = (self.state.paths.repo_root / "logs" / "rsl_rl").resolve()
        if not _is_within(log_dir.resolve(), resolved_root) or not _is_within(resolved_summary, log_dir):
            return self._json({"error": "TensorBoard summary path is outside the selected run log directory"}, status=403)
        self._send_file_response(resolved_summary, "image/png")

    def _send_file_response(self, path: Path, content_type: str) -> None:
        file_size = path.stat().st_size
        range_header = self.headers.get("Range")
        start = 0
        end = file_size - 1
        status = 200
        if range_header:
            match = range_header.strip().removeprefix("bytes=").split("-", 1)
            try:
                if match[0]:
                    start = int(match[0])
                if len(match) > 1 and match[1]:
                    end = int(match[1])
            except ValueError:
                start = file_size
            end = min(end, file_size - 1)
            if start < 0 or start >= file_size or end < start:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.end_headers()
                return
            status = 206
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()
        with path.open("rb") as file:
            file.seek(start)
            remaining = length
            while remaining > 0:
                chunk = file.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _send_static(self, relative: str) -> None:
        path = (STATIC_DIR / relative).resolve()
        if not path.is_file() or STATIC_DIR not in path.parents:
            return self._not_found()
        body = path.read_bytes()
        content_type = _STATIC_CONTENT_TYPES.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _not_found(self) -> None:
        self._json({"error": "not found"}, status=404)

    def _runs_payload(self) -> dict:
        runs = self.state.history.list_runs()
        for run in runs:
            run["effective_spring_backend"] = resolve_spring_backend(run)
            latest = latest_deploy_report(run)
            if latest:
                run["deploy_latest_report"] = {
                    key: value
                    for key, value in latest.items()
                    if key in {"path", "pipeline_id", "created_at", "completed_at", "overall_status", "readiness_level", "stage_counts"}
                }
        return {"runs": runs, "folders": self.state.history.folders_for_runs(runs)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RedRHex local training panel.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host. Use 0.0.0.0 for LAN access.")
    parser.add_argument("--port", type=int, default=8080, help="Bind port.")
    args = parser.parse_args()

    autopilot_enabled = os.environ.get("REDRHEX_AUTOPILOT_ENABLED", "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    loopback_hosts = {"127.0.0.1", "localhost", "::1"}
    autopilot_token = os.environ.get("REDRHEX_AUTOPILOT_BEARER_TOKEN", "").strip()
    if autopilot_enabled and args.host not in loopback_hosts and len(autopilot_token) < 16:
        parser.error(
            "non-loopback Autopilot requires REDRHEX_AUTOPILOT_BEARER_TOKEN with at least 16 characters"
        )

    paths = PanelPaths.from_env()
    paths.ensure_dirs()
    PanelHandler.state = PanelState(paths)
    server = ThreadingHTTPServer((args.host, args.port), PanelHandler)
    print(f"RedRHex training panel: http://{args.host}:{args.port}")
    print("For SSH tunnel: ssh -L 8080:127.0.0.1:8080 user@host")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping RedRHex training panel.")
    finally:
        if PanelHandler.state.autopilot is not None:
            PanelHandler.state.autopilot.close()
        server.server_close()
