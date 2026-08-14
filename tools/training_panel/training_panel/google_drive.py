from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, quote, urlparse

from .config import PanelPaths
from .history import HistoryStore


RCLONE_REMOTE = "redrhex-drive"
DESTINATION_FOLDER = "RedRHex Videos"
EXPORT_FIELD = "google_drive_video_exports"
ACTIVE_STATUSES = {"queued", "uploading"}
ERROR_LIMIT = 2000
RECONNECT_TIMEOUT_SECONDS = 10 * 60


class GoogleDriveUnavailableError(RuntimeError):
    pass


class GoogleDriveBusyError(RuntimeError):
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


def normalize_destination_folder(value: str) -> str:
    folder = str(value or "").strip()
    if not folder:
        raise ValueError("Drive destination folder is required")
    if len(folder) > 240:
        raise ValueError("Drive destination folder must be 240 characters or fewer")
    if folder.startswith("/") or folder.endswith("/"):
        raise ValueError("Drive destination folder must be relative and cannot start or end with /")
    if "\\" in folder or ":" in folder or any(ord(char) < 32 for char in folder):
        raise ValueError("Drive destination folder contains unsupported characters")
    parts = [part.strip() for part in folder.split("/")]
    if any(not part or part in {".", ".."} for part in parts):
        raise ValueError("Drive destination folder contains an invalid path segment")
    return "/".join(parts)


def parse_drive_destination(value: str) -> dict:
    destination = str(value or "").strip()
    if not destination:
        raise ValueError("Paste a Google Drive folder link or enter a My Drive folder path")
    if len(destination) > 2048:
        raise ValueError("Google Drive destination is too long")
    if "://" not in destination:
        return {
            "destination_mode": "my_drive_path",
            "destination_folder": normalize_destination_folder(destination),
            "root_folder_id": "",
            "resource_key": "",
            "folder_url": "",
        }

    parsed = urlparse(destination)
    if parsed.scheme != "https" or parsed.hostname != "drive.google.com":
        raise ValueError("Paste a Google Drive folder link from drive.google.com")
    match = re.search(r"/folders/([A-Za-z0-9_-]+)(?:/|$)", parsed.path)
    if not match or len(match.group(1)) < 10:
        raise ValueError("That is not a Google Drive folder link")
    folder_id = match.group(1)
    resource_key = str((parse_qs(parsed.query).get("resourcekey") or [""])[0]).strip()
    if resource_key and not re.fullmatch(r"[A-Za-z0-9_-]{1,240}", resource_key):
        raise ValueError("The Google Drive folder link has an invalid resource key")
    return {
        "destination_mode": "folder_link",
        "destination_folder": "",
        "root_folder_id": folder_id,
        "resource_key": resource_key,
        "folder_url": f"https://drive.google.com/drive/folders/{quote(folder_id, safe='')}",
    }


def _bounded_error(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"(?i)(--drive-resource-key\s+)(\S+)", r"\1<redacted>", text)
    text = re.sub(
        r'''(?i)((?:access|refresh)?_?token|client_secret|resource_?key)(["']?\s*[=:]\s*["']?)([^,\s}"']+)''',
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
        popen_factory: Callable | None = None,
        start_background: Callable[[Callable[[], None]], None] | None = None,
        settings_file: Path | None = None,
    ):
        self.paths = paths
        self.history = history
        self.rclone_path = rclone_path or shutil.which("rclone")
        self._run_command = run_command or subprocess.run
        self._popen_factory = popen_factory or subprocess.Popen
        self._start_background = start_background or self._start_thread
        self.settings_file = settings_file or (paths.panel_log_root / "google_drive_settings.json")
        self._lock = threading.RLock()
        self._active: set[tuple[str, str]] = set()
        self._settings_changing = False
        self._settings = self._load_settings()
        self._reconnect = {
            "status": "idle",
            "started_at": "",
            "finished_at": "",
            "error": "",
        }
        self.reconcile_interrupted()

    @staticmethod
    def _start_thread(target: Callable[[], None]) -> None:
        threading.Thread(target=target, name="google-drive-export", daemon=True).start()

    def _run(self, args: list[str], *, timeout: float | None = None) -> subprocess.CompletedProcess:
        kwargs = {"capture_output": True, "text": True, "check": False, "shell": False}
        if timeout is not None:
            kwargs["timeout"] = timeout
        return self._run_command(args, **kwargs)

    def _load_settings(self) -> dict:
        try:
            payload = json.loads(self.settings_file.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            payload = {}
        try:
            revision = max(1, int(payload.get("destination_revision") or 1))
        except (TypeError, ValueError):
            revision = 1
        if payload.get("destination_mode") == "folder_link":
            folder_id = str(payload.get("root_folder_id") or "").strip()
            if re.fullmatch(r"[A-Za-z0-9_-]{10,}", folder_id):
                resource_key = str(payload.get("resource_key") or "").strip()
                if not resource_key or re.fullmatch(r"[A-Za-z0-9_-]{1,240}", resource_key):
                    return {
                        "destination_mode": "folder_link",
                        "destination_folder": "",
                        "root_folder_id": folder_id,
                        "resource_key": resource_key,
                        "folder_url": f"https://drive.google.com/drive/folders/{quote(folder_id, safe='')}",
                        "destination_revision": revision,
                    }
        try:
            destination = parse_drive_destination(payload.get("destination_folder") or DESTINATION_FOLDER)
        except ValueError:
            destination = parse_drive_destination(DESTINATION_FOLDER)
        return {**destination, "destination_revision": revision}

    def _write_settings(self, settings: dict) -> None:
        self.settings_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.settings_file.with_suffix(f"{self.settings_file.suffix}.tmp")
        temporary.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, self.settings_file)

    def _bump_destination_revision(self, *, destination: dict | None = None) -> dict:
        next_settings = dict(self._settings)
        if destination is not None:
            next_settings.update(destination)
        next_settings["destination_revision"] = int(next_settings.get("destination_revision") or 1) + 1
        self._write_settings(next_settings)
        self._settings = next_settings
        return dict(next_settings)

    def reconnect_state(self) -> dict:
        with self._lock:
            return dict(self._reconnect)

    @staticmethod
    def _drive_root_args(settings: dict) -> list[str]:
        if settings.get("destination_mode") != "folder_link":
            return []
        args = ["--drive-root-folder-id", str(settings["root_folder_id"])]
        if settings.get("resource_key"):
            args.extend(["--drive-resource-key", str(settings["resource_key"])])
        return args

    def status(self) -> dict:
        remote = f"{RCLONE_REMOTE}:"
        with self._lock:
            settings = dict(self._settings)
            reconnect = dict(self._reconnect)
        base = {
            "available": bool(self.rclone_path),
            "configured": False,
            "remote": remote,
            "folder": settings["destination_folder"],
            "destination_mode": settings["destination_mode"],
            "destination_display": (
                "Linked Google Drive folder"
                if settings["destination_mode"] == "folder_link"
                else settings["destination_folder"]
            ),
            "folder_url": settings["folder_url"],
            "destination_revision": settings["destination_revision"],
            "reconnect": reconnect,
            "reconnect_command": f"rclone config reconnect {remote}",
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

    def save_destination(self, destination_value: str) -> dict:
        destination = parse_drive_destination(destination_value)
        readiness = self.status()
        if not readiness["configured"]:
            raise GoogleDriveUnavailableError(str(readiness["remediation"]))
        with self._lock:
            if self._active or self._settings_changing:
                raise GoogleDriveBusyError("Wait for the active Drive operation to finish before changing its destination")
            if (
                destination["destination_mode"] == "folder_link"
                and destination["root_folder_id"] == self._settings.get("root_folder_id")
                and not destination["resource_key"]
            ):
                destination["resource_key"] = str(self._settings.get("resource_key") or "")
            self._settings_changing = True
        try:
            if destination["destination_mode"] == "folder_link":
                result = self._run(
                    [
                        self.rclone_path,
                        "lsjson",
                        "--stat",
                        f"{RCLONE_REMOTE}:",
                        *self._drive_root_args(destination),
                        "--log-level",
                        "ERROR",
                    ],
                    timeout=30.0,
                )
                try:
                    metadata = json.loads(result.stdout or "{}")
                except json.JSONDecodeError:
                    metadata = {}
                valid_destination = result.returncode == 0 and metadata.get("IsDir") is True
            else:
                result = self._run(
                    [
                        self.rclone_path,
                        "mkdir",
                        f"{RCLONE_REMOTE}:{destination['destination_folder']}",
                        "--log-level",
                        "ERROR",
                    ],
                    timeout=30.0,
                )
                valid_destination = result.returncode == 0
            if not valid_destination:
                raise GoogleDriveUnavailableError(
                    "The connected Google account cannot access that Drive folder. Check the link or change accounts."
                )
            with self._lock:
                current_destination = {
                    key: self._settings.get(key)
                    for key in (
                        "destination_mode",
                        "destination_folder",
                        "root_folder_id",
                        "resource_key",
                        "folder_url",
                    )
                }
                if destination != current_destination:
                    self._bump_destination_revision(destination=destination)
                return self.status()
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GoogleDriveUnavailableError("Google Drive did not confirm that destination in time") from exc
        finally:
            with self._lock:
                self._settings_changing = False

    def start_reconnect(self) -> tuple[dict, bool]:
        readiness = self.status()
        if not readiness["configured"]:
            raise GoogleDriveUnavailableError(str(readiness["remediation"]))
        with self._lock:
            if self._reconnect["status"] == "authorizing":
                return dict(self._reconnect), False
            if self._active or self._settings_changing:
                raise GoogleDriveBusyError("Wait for the active Drive operation to finish before changing accounts")
            self._settings_changing = True
            self._reconnect = {
                "status": "authorizing",
                "started_at": _now_iso(),
                "finished_at": "",
                "error": "",
            }

        def perform() -> None:
            process = None
            try:
                process = self._popen_factory(
                    [self.rclone_path, "config", "reconnect", f"{RCLONE_REMOTE}:", "--auto-confirm"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    shell=False,
                )
                process.communicate(timeout=RECONNECT_TIMEOUT_SECONDS)
                if process.returncode != 0:
                    raise RuntimeError("Google authorization was not completed")
                with self._lock:
                    self._bump_destination_revision()
                    self._reconnect = {
                        "status": "completed",
                        "started_at": self._reconnect["started_at"],
                        "finished_at": _now_iso(),
                        "error": "",
                    }
            except subprocess.TimeoutExpired:
                if process is not None:
                    process.kill()
                    process.communicate()
                with self._lock:
                    self._reconnect = {
                        "status": "failed",
                        "started_at": self._reconnect["started_at"],
                        "finished_at": _now_iso(),
                        "error": "Google authorization timed out. Retry from Settings.",
                    }
            except Exception:
                with self._lock:
                    self._reconnect = {
                        "status": "failed",
                        "started_at": self._reconnect["started_at"],
                        "finished_at": _now_iso(),
                        "error": "Google authorization did not complete. Retry from Settings or use the reconnect command.",
                    }
            finally:
                with self._lock:
                    self._settings_changing = False

        try:
            self._start_background(perform)
        except Exception:
            with self._lock:
                self._settings_changing = False
                self._reconnect = {
                    "status": "failed",
                    "started_at": self._reconnect["started_at"],
                    "finished_at": _now_iso(),
                    "error": "Could not start Google authorization from the training PC.",
                }
            raise
        return self.reconnect_state(), True

    def _validated_source(self, run: dict, source_path: str | Path) -> tuple[Path, str, int, int]:
        log_dir_value = str(run.get("log_dir") or "")
        log_dir = Path(log_dir_value) if log_dir_value else None
        if not log_dir or not log_dir.is_dir():
            raise GoogleDrivePathError("No log directory found for run")
        resolved_log_dir = log_dir.resolve()
        if not _is_within(resolved_log_dir, self.paths.rsl_rl_log_root.parent):
            raise GoogleDrivePathError("Run log directory is outside the RSL-RL log root")
        source = Path(source_path).resolve()
        if source.suffix.lower() != ".mp4" or not source.is_file():
            raise GoogleDrivePathError("No recorded MP4 found for run")
        if not _is_within(source, resolved_log_dir):
            raise GoogleDrivePathError("Video path is outside the selected run log directory")
        stat = source.stat()
        return source, source.relative_to(resolved_log_dir).as_posix(), stat.st_size, stat.st_mtime_ns

    def _remote_path(self, run_id: str, source: Path, settings: dict) -> str:
        run_folder = _safe_segment(run_id, "run")
        prefix = str(settings.get("destination_folder") or "").strip("/")
        relative_path = "/".join(part for part in (prefix, run_folder, _safe_video_name(source)) if part)
        return f"{RCLONE_REMOTE}:{relative_path}"

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
            if self._settings_changing:
                raise GoogleDriveBusyError("Google Drive settings are changing. Retry after authorization finishes")
            settings = dict(self._settings)
            latest_run = self.history.get_run(canonical_run_id) or run
            existing = self._exports(latest_run).get(key)
            fingerprint_matches = bool(
                existing
                and existing.get("source_size") == source_size
                and existing.get("source_mtime_ns") == source_mtime_ns
                and existing.get("destination_revision") == settings["destination_revision"]
                and existing.get("destination_folder") == settings["destination_folder"]
                and existing.get("destination_mode", "my_drive_path") == settings["destination_mode"]
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
                "destination_mode": settings["destination_mode"],
                "destination_folder": settings["destination_folder"],
                "destination_display": (
                    "Linked Google Drive folder"
                    if settings["destination_mode"] == "folder_link"
                    else settings["destination_folder"]
                ),
                "destination_folder_url": settings["folder_url"],
                "destination_revision": settings["destination_revision"],
                "remote_path": self._remote_path(canonical_run_id, source, settings),
                "file_id": "",
                "web_view_url": "",
                "started_at": _now_iso(),
                "finished_at": "",
                "error": "",
            }
            self._persist(canonical_run_id, key, record)
            self._active.add(active_key)

        def perform() -> None:
            self._perform_export(
                canonical_run_id,
                key,
                source,
                record["remote_path"],
                source_size,
                self._drive_root_args(settings),
            )

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

    def _perform_export(
        self,
        run_id: str,
        key: str,
        source: Path,
        remote_path: str,
        source_size: int,
        drive_root_args: list[str],
    ) -> None:
        active_key = (run_id, key)
        try:
            copy_result = self._run(
                [
                    self.rclone_path,
                    "copyto",
                    str(source),
                    remote_path,
                    *drive_root_args,
                    "--log-level",
                    "ERROR",
                ]
            )
            if copy_result.returncode != 0:
                detail = copy_result.stderr or copy_result.stdout or "rclone copyto failed"
                raise RuntimeError(detail)
            stat_result = self._run(
                [
                    self.rclone_path,
                    "lsjson",
                    "--stat",
                    remote_path,
                    *drive_root_args,
                    "--log-level",
                    "ERROR",
                ]
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
