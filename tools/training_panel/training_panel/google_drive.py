from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import quote

from .config import PanelPaths
from .history import HistoryStore


RCLONE_REMOTE = "redrhex-drive"
DESTINATION_FOLDER = "RedRHex Videos"
EXPORT_FIELD = "google_drive_video_exports"
ACTIVE_STATUSES = {"queued", "uploading"}
ERROR_LIMIT = 2000


class GoogleDriveUnavailableError(RuntimeError):
    pass


class GoogleDrivePathError(ValueError):
    pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _is_within(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    resolved_root = root.resolve()
    return resolved == resolved_root or resolved_root in resolved.parents


def _safe_segment(value: str, fallback: str, limit: int = 120) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._-")
    return (cleaned or fallback)[:limit]


def _safe_video_name(path: Path) -> str:
    return f"{_safe_segment(path.stem, 'video', limit=110)}.mp4"


def _bounded_error(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(
        r'''(?i)((?:access|refresh)?_?token|client_secret)(["']?\s*[=:]\s*["']?)([^,\s}"']+)''',
        r"\1\2<redacted>",
        text,
    )
    text = re.sub(r"(?i)(bearer\s+)([^,\s}\]]+)", r"\1<redacted>", text)
    return text[-ERROR_LIMIT:]


class GoogleDriveExporter:
    def __init__(
        self,
        paths: PanelPaths,
        history: HistoryStore,
        *,
        rclone_path: str | None = None,
        run_command: Callable | None = None,
        start_background: Callable[[Callable[[], None]], None] | None = None,
    ):
        self.paths = paths
        self.history = history
        self.rclone_path = rclone_path or shutil.which("rclone")
        self._run_command = run_command or subprocess.run
        self._start_background = start_background or self._start_thread
        self._lock = threading.RLock()
        self._active: set[tuple[str, str]] = set()
        self.reconcile_interrupted()

    @staticmethod
    def _start_thread(target: Callable[[], None]) -> None:
        threading.Thread(target=target, name="google-drive-export", daemon=True).start()

    def _run(self, args: list[str], *, timeout: float | None = None) -> subprocess.CompletedProcess:
        kwargs = {"capture_output": True, "text": True, "check": False, "shell": False}
        if timeout is not None:
            kwargs["timeout"] = timeout
        return self._run_command(args, **kwargs)

    def status(self) -> dict:
        remote = f"{RCLONE_REMOTE}:"
        base = {
            "available": bool(self.rclone_path),
            "configured": False,
            "remote": remote,
            "folder": DESTINATION_FOLDER,
        }
        if not self.rclone_path:
            return {
                **base,
                "remediation": (
                    f"Install rclone, then run `rclone config` and create the "
                    f"`{RCLONE_REMOTE}` Google Drive remote."
                ),
            }
        try:
            result = self._run([self.rclone_path, "listremotes"], timeout=5.0)
        except (OSError, subprocess.TimeoutExpired):
            return {
                **base,
                "remediation": (
                    "Run `rclone listremotes` on the training PC and repair the local "
                    "rclone configuration."
                ),
            }
        remotes = {line.strip() for line in str(result.stdout or "").splitlines() if line.strip()}
        configured = result.returncode == 0 and remote in remotes
        return {
            **base,
            "configured": configured,
            "remediation": (
                ""
                if configured
                else f"Run `rclone config` on the training PC and create the `{RCLONE_REMOTE}` Google Drive remote."
            ),
        }

    def _validated_source(self, run: dict, source_path: str | Path) -> tuple[Path, str, int, int]:
        log_dir_value = str(run.get("log_dir") or "")
        log_dir = Path(log_dir_value) if log_dir_value else None
        if not log_dir or not log_dir.is_dir():
            raise GoogleDrivePathError("No log directory found for run")
        resolved_log_dir = log_dir.resolve()
        if not _is_within(resolved_log_dir, self.paths.rsl_rl_log_root):
            raise GoogleDrivePathError("Run log directory is outside the RSL-RL log root")
        source = Path(source_path).resolve()
        if source.suffix.lower() != ".mp4" or not source.is_file():
            raise GoogleDrivePathError("No recorded MP4 found for run")
        if not _is_within(source, resolved_log_dir):
            raise GoogleDrivePathError("Video path is outside the selected run log directory")
        stat = source.stat()
        return source, source.relative_to(resolved_log_dir).as_posix(), stat.st_size, stat.st_mtime_ns

    def _remote_path(self, run_id: str, source: Path) -> str:
        run_folder = _safe_segment(run_id, "run")
        return f"{RCLONE_REMOTE}:{DESTINATION_FOLDER}/{run_folder}/{_safe_video_name(source)}"

    @staticmethod
    def _exports(run: dict | None) -> dict:
        value = (run or {}).get(EXPORT_FIELD)
        return dict(value) if isinstance(value, dict) else {}

    def _persist(self, run_id: str, key: str, record: dict) -> dict:
        run = self.history.get_run(run_id) or {"id": run_id}
        exports = self._exports(run)
        exports[key] = dict(record)
        self.history.patch_run_metadata(run_id, **{EXPORT_FIELD: exports})
        return exports[key]

    def _update(self, run_id: str, key: str, **updates) -> dict:
        with self._lock:
            run = self.history.get_run(run_id) or {"id": run_id}
            exports = self._exports(run)
            record = dict(exports.get(key) or {})
            record.update(updates)
            exports[key] = record
            self.history.patch_run_metadata(run_id, **{EXPORT_FIELD: exports})
            return record

    def start_export(
        self,
        run_id: str,
        source_path: str | Path,
        *,
        checkpoint_iteration: int | None = None,
    ) -> tuple[dict, bool, bool]:
        readiness = self.status()
        if not readiness["configured"]:
            raise GoogleDriveUnavailableError(str(readiness["remediation"]))
        canonical_run_id = self.history.canonical_run_id(run_id)
        run = self.history.get_run(canonical_run_id)
        if not run:
            raise GoogleDrivePathError("Run not found")
        source, key, source_size, source_mtime_ns = self._validated_source(run, source_path)
        active_key = (canonical_run_id, key)

        with self._lock:
            latest_run = self.history.get_run(canonical_run_id) or run
            existing = self._exports(latest_run).get(key)
            fingerprint_matches = bool(
                existing
                and existing.get("source_size") == source_size
                and existing.get("source_mtime_ns") == source_mtime_ns
            )
            if fingerprint_matches and existing.get("status") == "completed":
                return dict(existing), False, True
            if fingerprint_matches and existing.get("status") in ACTIVE_STATUSES and active_key in self._active:
                return dict(existing), False, False

            record = {
                "status": "uploading",
                "source_path": key,
                "source_size": source_size,
                "source_mtime_ns": source_mtime_ns,
                "checkpoint_iteration": checkpoint_iteration,
                "remote_path": self._remote_path(canonical_run_id, source),
                "file_id": "",
                "web_view_url": "",
                "started_at": _now_iso(),
                "finished_at": "",
                "error": "",
            }
            self._persist(canonical_run_id, key, record)
            self._active.add(active_key)

        def perform() -> None:
            self._perform_export(canonical_run_id, key, source, record["remote_path"], source_size)

        try:
            self._start_background(perform)
        except Exception as exc:
            with self._lock:
                self._active.discard(active_key)
            self._update(
                canonical_run_id,
                key,
                status="failed",
                finished_at=_now_iso(),
                error=_bounded_error(f"Could not start Drive export: {exc}"),
            )
            raise
        return dict(record), True, False

    def _perform_export(self, run_id: str, key: str, source: Path, remote_path: str, source_size: int) -> None:
        active_key = (run_id, key)
        try:
            copy_result = self._run(
                [self.rclone_path, "copyto", str(source), remote_path, "--log-level", "ERROR"]
            )
            if copy_result.returncode != 0:
                detail = copy_result.stderr or copy_result.stdout or "rclone copyto failed"
                raise RuntimeError(detail)
            stat_result = self._run(
                [self.rclone_path, "lsjson", "--stat", remote_path, "--log-level", "ERROR"]
            )
            if stat_result.returncode != 0:
                detail = stat_result.stderr or stat_result.stdout or "rclone lsjson failed"
                raise RuntimeError(detail)
            metadata = json.loads(stat_result.stdout or "{}")
            file_id = str(metadata.get("ID") or "").strip()
            remote_size = metadata.get("Size")
            if not file_id:
                raise RuntimeError("Google Drive did not return a file ID after upload")
            if remote_size is not None and int(remote_size) != source_size:
                raise RuntimeError("Google Drive file size did not match the source video")
            self._update(
                run_id,
                key,
                status="completed",
                file_id=file_id,
                web_view_url=f"https://drive.google.com/file/d/{quote(file_id, safe='')}/view",
                finished_at=_now_iso(),
                error="",
            )
        except Exception as exc:
            self._update(
                run_id,
                key,
                status="failed",
                finished_at=_now_iso(),
                error=_bounded_error(str(exc) or exc.__class__.__name__),
            )
        finally:
            with self._lock:
                self._active.discard(active_key)

    def reconcile_interrupted(self) -> int:
        repaired = 0
        for run in self.history.list_runs():
            exports = self._exports(run)
            changed = False
            for key, value in list(exports.items()):
                if not isinstance(value, dict) or value.get("status") not in ACTIVE_STATUSES:
                    continue
                exports[key] = {
                    **value,
                    "status": "interrupted",
                    "finished_at": _now_iso(),
                    "error": "Panel restarted before Drive export completion was confirmed. Retry the export.",
                }
                repaired += 1
                changed = True
            if changed:
                self.history.patch_run_metadata(str(run.get("id") or ""), **{EXPORT_FIELD: exports})
        return repaired
