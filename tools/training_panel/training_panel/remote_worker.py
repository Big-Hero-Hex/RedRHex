from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from uuid import UUID

from .activity import ActivityStore, category_for_job_type, outcome_for_status, score_activity_event
from .commands import DEFAULT_VIDEO_PRESET, TrainingParams, VideoParams
from .config import PanelPaths
from .convergence import ConvergenceChecker, DEFAULT_SCALAR_TAGS, downsample, load_convergence_config
from .deploy import deploy_defaults, latest_deploy_report
from .google_drive import GoogleDriveExporter
from .history import HistoryStore, checkpoint_inventory
from .mujoco_rollout import default_scenarios
from .notifications import notification_events_for_run
from .physics import normalize_physics_values, physics_catalog
from .processes import ProcessRegistry
from .remote_config import (
    REMOTE_PROTOCOL_VERSION,
    RemoteConfig,
    RemoteStateStore,
    heartbeat_payload,
    role_allows,
)
from .supabase_client import SupabaseClient
from .tensorboard_summary import ensure_tensorboard_summary, tensorboard_summary_path

VIDEO_BUCKET = "redrhex-videos"
TENSORBOARD_SUMMARY_CONTENT_TYPE = "image/png"
GPU_JOB_TYPES = {"start_training", "record_video", "export_onnx", "export_validate_deploy"}
FINISHED_RUN_STATUSES = {"completed", "failed", "interrupted"}
MEDIA_SYNC_INTERVAL_SECONDS = 3.0
MEDIA_SYNC_RECENT_SECONDS = 20 * 60
MEDIA_SYNC_STATUSES = {"recording", "completed", "failed", "missing_checkpoint"}
MAX_VIDEO_ITERATIONS_PER_RUN = 8
MAX_REMOTE_SCALAR_POINTS = 400
NOTIFICATION_SETTING_KEYS = {
    "training_converged": "notify_training_converged",
    "training_completed": "notify_training_completed",
    "training_failed": "notify_training_failed",
    "video_ready": "notify_video_ready",
}
HANDLED_NOTIFICATION_STATUSES = {"sent", "no_channels", "disabled", "skipped"}


def _storage_safe(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return safe or "run"


def _checkpoint_iteration(path: str) -> int | None:
    match = re.search(r"model_(\d+)(?:\.pt|[^0-9]|$)", str(path or ""))
    return int(match.group(1)) if match else None


def _folder_key(name: str) -> str:
    return re.sub(r"\s+", " ", str(name or "").strip()).lower()


def run_artifacts(run: dict) -> list[dict]:
    artifacts = []
    seen: set[tuple[str, str]] = set()

    def add_artifact(kind: str, path: str | None, checkpoint_iteration: int | None = None) -> None:
        if not path:
            return
        local_path = str(path)
        key = (kind, local_path)
        if key in seen:
            return
        seen.add(key)
        artifact = {
            "kind": kind,
            "path": local_path,
            "local_path": local_path,
            "run_id": run.get("id"),
            "bytes": None,
            "checkpoint_iteration": checkpoint_iteration,
        }
        try:
            file_path = Path(local_path)
            if file_path.is_file():
                artifact["bytes"] = file_path.stat().st_size
        except OSError:
            pass
        artifacts.append(artifact)

    latest_checkpoint_iteration = _checkpoint_iteration(str(run.get("latest_checkpoint") or ""))
    add_artifact("checkpoint", run.get("latest_checkpoint"), latest_checkpoint_iteration)
    add_artifact("video", run.get("latest_video"), _checkpoint_iteration(str(run.get("latest_video") or "")) or latest_checkpoint_iteration)
    add_artifact("onnx", run.get("onnx_path"))
    add_artifact("tensorboard_summary", run.get("tensorboard_summary_path"))
    add_artifact("mujoco_video", run.get("latest_mujoco_video"))
    log_dir = Path(str(run.get("log_dir") or ""))
    if log_dir.is_dir():
        summary = tensorboard_summary_path(log_dir)
        if summary.is_file():
            add_artifact("tensorboard_summary", str(summary))
        for iteration, checkpoint in checkpoint_inventory(log_dir):
            add_artifact("checkpoint", str(checkpoint), iteration)
        video_dir = log_dir / "videos" / "play"
        if video_dir.is_dir():
            videos = [video for video in video_dir.glob("*.mp4") if video.is_file()]
            videos.sort(key=lambda item: item.stat().st_mtime, reverse=True)
            by_iteration: dict[int | None, Path] = {}
            for video in videos:
                by_iteration.setdefault(_checkpoint_iteration(video.name), video)
            numeric_iterations = {iteration for iteration in by_iteration if iteration is not None}
            if len(numeric_iterations) > MAX_VIDEO_ITERATIONS_PER_RUN:
                latest = Path(str(run.get("latest_video") or "")) if run.get("latest_video") else None
                if latest and latest.is_file():
                    add_artifact("video", str(latest), _checkpoint_iteration(latest.name) or latest_checkpoint_iteration)
                elif videos:
                    video = videos[0]
                    add_artifact("video", str(video), _checkpoint_iteration(video.name))
            else:
                for iteration, video in sorted(
                    by_iteration.items(),
                    key=lambda item: (-1 if item[0] is None else -int(item[0]), str(item[1])),
                ):
                    add_artifact("video", str(video), iteration)
    return artifacts


def remote_capabilities(paths: PanelPaths, *, drive_status: dict | None = None) -> dict:
    """Return the host-safe, versioned capability contract consumed by Child."""

    deploy = deploy_defaults(paths)
    convergence = asdict(load_convergence_config(paths.convergence_config_file))
    scenarios = [scenario.name for scenario in default_scenarios()]
    return {
        "machine_id": "",
        "protocol_version": REMOTE_PROTOCOL_VERSION,
        "feature_flags": {
            "physics_presets": True,
            "checkpoint_references": True,
            "bounded_metrics": True,
            "remote_deploy": True,
            "mujoco_recording": True,
            "drive_export": True,
            "activity_projection": True,
            "metadata_rpc": True,
            "queued_job_cancellation": True,
        },
        "training_routes": [
            "standard",
            "sensor_v2_full",
            "sensor_v2_ungated_debug",
            "sensor_v2_teacher",
            "sensor_v2_distillation",
            "sensor_v2_ppo",
        ],
        "physics": physics_catalog(),
        "deploy": {
            "target": deploy.get("target"),
            "target_runtime": deploy.get("target_runtime"),
            "report_version": deploy.get("report_version"),
            "include_ros_mock_default": bool(deploy.get("include_ros_mock_default")),
            "include_mujoco_default": bool(deploy.get("include_mujoco_default")),
            "onnx_installed": bool(deploy.get("onnx_installed")),
            "onnx_version": deploy.get("onnx_version"),
            "onnxruntime_installed": bool(deploy.get("onnxruntime_installed")),
            "onnxruntime_version": deploy.get("onnxruntime_version"),
            "mujoco_installed": bool(deploy.get("mujoco_installed")),
            "mujoco_version": deploy.get("mujoco_version"),
            "mujoco_renderer_available": bool(deploy.get("mujoco_renderer_available")),
            "mujoco_encoder_available": bool(deploy.get("mujoco_encoder_available")),
            "mujoco_calibrated": bool(deploy.get("mujoco_calibrated")),
            "scenarios": scenarios,
            "playback_defaults": deploy.get("mujoco_playback_defaults") or {},
        },
        "detection": convergence,
        "integration_readiness": {
            "google_drive": drive_status or {},
            "ros_mock": True,
            "physical_robot_deployment": False,
            "terminal": False,
            "host_paths": False,
            "gui_viewer": False,
        },
    }


def _bounded_progress(value: object) -> dict:
    if not isinstance(value, dict):
        return {}
    allowed = {
        "iteration", "max_iterations", "percent", "elapsed_seconds", "eta_seconds",
        "fps", "updated_at", "mean_reward", "mean_episode_length",
    }
    result = {key: value[key] for key in allowed if key in value}
    if result.get("percent") is not None:
        try:
            result["percent"] = min(100.0, max(0.0, float(result["percent"])))
        except (TypeError, ValueError):
            result.pop("percent", None)
    return result


def _safe_deploy_projection(run: dict) -> dict:
    state = {
        "status": run.get("deploy_status"),
        "overall_status": run.get("deploy_overall_status"),
        "readiness_level": run.get("deploy_readiness_level"),
        "completed_at": run.get("deploy_completed_at"),
        "error": run.get("deploy_error"),
    }
    try:
        latest = latest_deploy_report(run)
    except Exception:
        latest = None
    report = latest.get("report") if isinstance(latest, dict) else None
    if isinstance(report, dict):
        stages = []
        for stage in report.get("stages") or []:
            if not isinstance(stage, dict):
                continue
            stages.append({
                key: stage.get(key)
                for key in ("name", "label", "title", "status", "summary", "duration_ms", "error")
                if stage.get(key) not in (None, "")
            })
        state["report"] = {
            "version": report.get("version"),
            "created_at": report.get("created_at"),
            "completed_at": report.get("completed_at"),
            "overall_status": report.get("overall_status"),
            "readiness_level": report.get("readiness_level"),
            "stages": stages,
            "operator_checklist": list(report.get("operator_checklist") or [])[:20],
            "assumptions": list(report.get("assumptions") or [])[:20],
        }
    return {key: value for key, value in state.items() if value not in (None, "", [])}


def _safe_drive_projection(run: dict) -> list[dict]:
    values = run.get("google_drive_video_exports")
    if not isinstance(values, list):
        return []
    allowed = {
        "id", "status", "checkpoint_iteration", "created_at", "updated_at",
        "completed_at", "bytes", "web_view_url", "error",
    }
    return [
        {key: item.get(key) for key in allowed if item.get(key) not in (None, "")}
        for item in values[-20:]
        if isinstance(item, dict)
    ]


def _parse_time(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        normalized = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
        return dt.timestamp()
    except ValueError:
        return 0.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _uuid_or_none(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return str(UUID(str(value)))
    except ValueError:
        return None


def _job_payload(job: dict) -> dict:
    payload = job.get("payload")
    return payload if isinstance(payload, dict) else {}


def _job_result(job: dict) -> dict:
    result = job.get("result")
    return result if isinstance(result, dict) else {}


def _requester_id_for_job(job: dict) -> str | None:
    payload = _job_payload(job)
    return _uuid_or_none(str(job.get("actor_id") or "")) or _uuid_or_none(str(payload.get("requester_id") or ""))


def _requester_label_for_job(job: dict, requester_id: str | None = None) -> str | None:
    payload = _job_payload(job)
    label = (
        payload.get("requester_label")
        or job.get("actor_name")
        or job.get("actor_email")
        or job.get("actor_role")
        or requester_id
    )
    return str(label) if label else None


def _run_id_for_job(job: dict) -> str:
    payload = _job_payload(job)
    result = _job_result(job)
    result_payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
    return str(
        payload.get("run_id")
        or result.get("local_run_id")
        or result.get("process_id")
        or result_payload.get("id")
        or result_payload.get("run_id")
        or result_payload.get("source_run_id")
        or ""
    )


@dataclass
class RemoteJobResult:
    local_run_id: str | None = None
    process_id: str | None = None
    payload: dict | None = None

    def to_dict(self) -> dict:
        return {
            "local_run_id": self.local_run_id,
            "process_id": self.process_id,
            "payload": self.payload or {},
        }


class RemoteJobExecutor:
    def __init__(self, paths: PanelPaths):
        self.paths = paths
        self.history = HistoryStore(paths)
        self.processes = ProcessRegistry(paths, self.history, cuda_preflight=True)
        self.google_drive = GoogleDriveExporter(paths, self.history)

    def gpu_locked(self) -> bool:
        return bool(self.processes.running_isaac_processes())

    def execute(self, job: dict) -> RemoteJobResult:
        job_type = str(job.get("type") or job.get("job_type") or "")
        actor_role = str(job.get("actor_role") or job.get("role") or "viewer")
        payload = job.get("payload") or {}
        if not role_allows(actor_role, job_type):
            raise PermissionError(f"Role '{actor_role}' cannot run remote job '{job_type}'")
        if job_type in GPU_JOB_TYPES and self.gpu_locked():
            raise RuntimeError("Another Isaac/GPU action is already running")

        if job_type == "start_training":
            training_payload = dict(payload)
            if training_payload.get("checkpoint"):
                raise ValueError("Remote training accepts checkpoint_ref, not a host checkpoint path")
            checkpoint_ref = training_payload.pop("checkpoint_ref", None)
            if checkpoint_ref:
                training_payload["checkpoint"] = str(self._checkpoint_for_reference(checkpoint_ref))
            params = TrainingParams.from_dict(training_payload)
            requester_id = _requester_id_for_job(job)
            if requester_id:
                params.requester_id = requester_id
                params.requester_label = _requester_label_for_job(job, requester_id) or "Remote member"
            result = self.processes.start_training(params)
            return RemoteJobResult(local_run_id=result.get("id"), process_id=result.get("id"), payload=result)

        if job_type == "stop_process":
            process_id = str(payload.get("process_id") or payload.get("run_id") or "")
            if not process_id:
                raise ValueError("process_id is required")
            stopped_ids = []
            if self.processes.stop(process_id):
                stopped_ids.append(process_id)
            else:
                stopped_ids = self.processes.stop_all_for_run(process_id)
            return RemoteJobResult(process_id=process_id, payload={"stopped": bool(stopped_ids), "stopped_ids": stopped_ids})

        if job_type in {"record_video", "export_onnx"}:
            run = self._run_with_checkpoint(str(payload.get("run_id") or ""))
            checkpoint = self._checkpoint_for_payload(run, payload)
            if job_type == "record_video":
                result = self.processes.start_video_recording(
                    run_id=str(run["id"]),
                    checkpoint=str(checkpoint),
                    device=str(payload.get("device") or "cuda:0"),
                    video_params=VideoParams.from_preset(DEFAULT_VIDEO_PRESET),
                )
            else:
                result = self.processes.start_onnx_export(
                    run_id=str(run["id"]),
                    checkpoint=str(checkpoint),
                    device=str(payload.get("device") or "cuda:0"),
                )
            result["checkpoint"] = str(checkpoint)
            result["checkpoint_iteration"] = _checkpoint_iteration(str(checkpoint))
            return RemoteJobResult(local_run_id=str(run["id"]), process_id=result.get("id"), payload=result)

        if job_type == "export_video_drive":
            run = self.history.get_run(str(payload.get("run_id") or "")) or {}
            if not run:
                raise ValueError("Run was not found")
            video, iteration = self._video_for_payload(run, payload)
            record, started, reused = self.google_drive.start_export(
                str(run["id"]), video, checkpoint_iteration=iteration
            )
            return RemoteJobResult(
                local_run_id=str(run["id"]),
                payload={**record, "started": bool(started), "reused": bool(reused)},
            )

        if job_type in {"validate_deploy", "export_validate_deploy", "mujoco_smoke"}:
            run_id = str(payload.get("run_id") or "")
            run = self.history.get_run(run_id) or {}
            if not run:
                raise ValueError("Run was not found")
            include_ros_mock = bool(payload.get("include_ros_mock", False))
            if job_type == "validate_deploy" and not run.get("onnx_path"):
                raise ValueError("Export ONNX before validating the existing model")
            result = self.processes.start_deploy_validation(
                run_id,
                export_first=job_type == "export_validate_deploy",
                device="cuda:0",
                include_ros_mock=include_ros_mock,
                include_mujoco=True,
                use_cuda=False,
                use_tensorrt=False,
                mujoco_only=job_type == "mujoco_smoke",
            )
            return RemoteJobResult(local_run_id=run_id, process_id=result.get("id"), payload=result)

        if job_type == "record_mujoco_video":
            run_id = str(payload.get("run_id") or "")
            if not self.history.get_run(run_id):
                raise ValueError("Run was not found")
            scenario = str(payload.get("scenario") or "stand_zero")
            allowed = {item.name for item in default_scenarios()}
            if scenario not in allowed:
                raise ValueError(f"scenario must be one of: {', '.join(sorted(allowed))}")
            result = self.processes.start_mujoco_playback(
                run_id,
                mode="record",
                scenario=scenario,
            )
            return RemoteJobResult(local_run_id=run_id, process_id=result.get("id"), payload=result)

        if job_type == "tensorboard":
            run_id = str(payload.get("run_id") or "")
            run = self.history.get_run(run_id) or {}
            log_dir = Path(str(run.get("log_dir") or self.paths.rsl_rl_log_root))
            result = self.processes.start_tensorboard(
                host=str(payload.get("host") or "0.0.0.0"),
                port=int(payload["port"]) if payload.get("port") else None,
                logdir=log_dir,
                source_run_id=run_id or None,
            )
            return RemoteJobResult(local_run_id=run_id or None, process_id=result.get("id"), payload=result)

        if job_type == "compact_run":
            run_id = str(payload.get("run_id") or "")
            if self.processes.running_for_run(run_id):
                raise RuntimeError("Stop running processes for this run before compacting it")
            result = self.history.compact_run(run_id, confirmation=str(payload.get("confirmation") or run_id))
            return RemoteJobResult(local_run_id=run_id, payload=result)

        if job_type == "delete_run":
            run_id = str(payload.get("run_id") or "")
            if self.processes.running_for_run(run_id):
                raise RuntimeError("Stop running processes for this run before deleting it")
            result = self.history.delete_run(
                run_id,
                confirmation=str(payload.get("confirmation") or run_id),
                delete_logs=bool(payload.get("delete_logs", True)),
            )
            return RemoteJobResult(local_run_id=run_id, payload=result)

        raise ValueError(f"Unsupported remote job type: {job_type}")

    def _run_with_checkpoint(self, run_id: str) -> dict:
        run = self.history.get_run(run_id)
        if not run or not run.get("latest_checkpoint"):
            raise ValueError("No checkpoint found for run")
        return run

    def _checkpoint_for_payload(self, run: dict, payload: dict) -> Path:
        if payload.get("checkpoint"):
            raise ValueError("Remote actions accept checkpoint_ref, not a host checkpoint path")
        checkpoint_ref = payload.get("checkpoint_ref")
        if checkpoint_ref:
            expected_run_id = str(run.get("id") or "")
            referenced_run_id = str(checkpoint_ref.get("run_id") or "") if isinstance(checkpoint_ref, dict) else ""
            if referenced_run_id != expected_run_id:
                raise ValueError("checkpoint_ref run_id must match the selected run")
            return self._checkpoint_for_reference(checkpoint_ref)
        requested = ""
        log_dir_value = str(run.get("log_dir") or "")
        log_dir = Path(log_dir_value) if log_dir_value else None
        if requested:
            checkpoint = Path(requested).expanduser()
            if not checkpoint.is_absolute() and log_dir:
                checkpoint = log_dir / checkpoint
            if not checkpoint.is_file():
                raise ValueError(f"Checkpoint does not exist: {checkpoint}")
            if log_dir and log_dir.is_dir():
                try:
                    resolved = checkpoint.resolve()
                    resolved_log = log_dir.resolve()
                    if resolved != resolved_log and resolved_log not in resolved.parents:
                        raise ValueError("Checkpoint must be inside the selected run log directory")
                except OSError as exc:
                    raise ValueError(f"Checkpoint could not be resolved: {checkpoint}") from exc
            return checkpoint
        iteration = payload.get("checkpoint_iteration")
        if iteration is not None and log_dir and log_dir.is_dir():
            try:
                target_iteration = int(iteration)
            except (TypeError, ValueError):
                raise ValueError("checkpoint_iteration must be an integer") from None
            for checkpoint_iteration, checkpoint in checkpoint_inventory(log_dir):
                if checkpoint_iteration == target_iteration:
                    return checkpoint
            raise ValueError(f"No checkpoint found for iteration {target_iteration}")
        return Path(str(run["latest_checkpoint"]))

    def _checkpoint_for_reference(self, checkpoint_ref: dict) -> Path:
        if not isinstance(checkpoint_ref, dict):
            raise ValueError("checkpoint_ref must contain run_id and checkpoint_iteration")
        run_id = str(checkpoint_ref.get("run_id") or "").strip()
        if not run_id:
            raise ValueError("checkpoint_ref.run_id is required")
        run = self._run_with_checkpoint(run_id)
        reference_payload = {"checkpoint_iteration": checkpoint_ref.get("checkpoint_iteration")}
        return self._checkpoint_for_payload(run, reference_payload)

    def _video_for_payload(self, run: dict, payload: dict) -> tuple[Path, int | None]:
        iteration = payload.get("checkpoint_iteration")
        if iteration is not None:
            try:
                iteration = int(iteration)
            except (TypeError, ValueError):
                raise ValueError("checkpoint_iteration must be an integer") from None
        candidates = [
            artifact for artifact in run_artifacts(run)
            if artifact.get("kind") == "video" and (iteration is None or artifact.get("checkpoint_iteration") == iteration)
        ]
        if not candidates:
            raise ValueError("No recorded MP4 found for the selected checkpoint")
        video = Path(str(candidates[0].get("local_path") or ""))
        if not video.is_file():
            raise ValueError("No recorded MP4 found for the selected checkpoint")
        return video, candidates[0].get("checkpoint_iteration")

    def sync_runs_payload(self, run_ids: set[str] | None = None) -> list[dict]:
        runs = self.history.list_runs()
        scalar_reader = ConvergenceChecker()
        if run_ids is not None:
            normalized_ids = {str(run_id) for run_id in run_ids if str(run_id)}
            runs = [run for run in runs if str(run.get("id") or "") in normalized_ids]
        result = []
        for run in runs:
            run_id = str(run.get("id") or "")
            log_dir = Path(str(run.get("log_dir") or ""))
            status = str(run.get("status") or "").lower()
            if run_id and log_dir.is_dir() and status in FINISHED_RUN_STATUSES:
                try:
                    title = str(run.get("display_name") or run_id or log_dir.name)
                    summary = ensure_tensorboard_summary(log_dir, title=title)
                    if summary:
                        run = {
                            **run,
                            "tensorboard_summary_path": str(summary),
                            "tensorboard_summary_status": "completed",
                            "tensorboard_summary_error": None,
                        }
                        self.history.update_run(
                            run_id,
                            tensorboard_summary_path=str(summary),
                            tensorboard_summary_status="completed",
                            tensorboard_summary_error=None,
                        )
                except Exception as exc:
                    run = {
                        **run,
                        "tensorboard_summary_status": "failed",
                        "tensorboard_summary_error": str(exc),
                    }
                    self.history.update_run(
                        run_id,
                        tensorboard_summary_status="failed",
                        tensorboard_summary_error=str(exc),
                    )
            notes = self.history.get_note(run_id)
            note_exists = self.history.note_exists(run_id)
            record: dict = {
                "id": run.get("id"),
                "status": run.get("status"),
                "created_at": run.get("created_at"),
                "updated_at": run.get("updated_at"),
                "log_dir": run.get("log_dir"),
                "params": run.get("params") or {},
                "latest_checkpoint": run.get("latest_checkpoint"),
                "latest_video": run.get("latest_video"),
                "onnx_path": run.get("onnx_path"),
                "created_by": run.get("created_by") or (run.get("params") or {}).get("requester_id"),
                # Live training state — enables the child to show convergence + recording progress.
                "convergence_detected": bool(run.get("convergence_detected")),
                "convergence_iteration": run.get("convergence_iteration"),
                "convergence_improvement_pct": run.get("convergence_improvement_pct"),
                "video_status": run.get("video_status"),
                "returncode": run.get("returncode"),
                "compacted_at": run.get("compacted_at"),
                "compacted_deleted_count": run.get("compacted_deleted_count"),
                "compacted_bytes_freed": run.get("compacted_bytes_freed"),
                "artifacts": run_artifacts(run),
                "progress": _bounded_progress(run.get("progress")),
                "git_provenance": run.get("git") if isinstance(run.get("git"), dict) else {},
                "effective_spring_backend": run.get("effective_spring_backend") or (run.get("params") or {}).get("spring_backend") or "native",
                "divergence_detected": bool(run.get("divergence_detected")),
                "divergence_iteration": run.get("divergence_iteration"),
                "divergence_kind": run.get("divergence_kind"),
                "divergence_reason": run.get("divergence_reason"),
                "deploy_state": _safe_deploy_projection(run),
                "mujoco_state": {
                    key: run.get(key)
                    for key in (
                        "mujoco_playback_status", "mujoco_returncode", "mujoco_error",
                        "mujoco_completed_at",
                    )
                    if run.get(key) not in (None, "")
                },
                "google_drive_video_exports": _safe_drive_projection(run),
            }
            metrics: dict[str, list[list[float | int]]] = {}
            if log_dir.is_dir():
                for tag in DEFAULT_SCALAR_TAGS:
                    points = downsample(scalar_reader.read_scalars(log_dir, tag), MAX_REMOTE_SCALAR_POINTS)
                    if points:
                        metrics[tag] = [[int(step), float(value)] for step, value in points]
            record["metrics"] = metrics
            # Include metadata only when the mother has an explicit local opinion.
            # Empty values are important: they clear stale child/Supabase metadata
            # after a mother-side rename, uncategorize, or note clear. Purely
            # discovered runs without local metadata still omit these fields.
            if "display_name" in run:
                record["display_name"] = run.get("display_name") or None
            if "folder" in run:
                record["folder"] = run.get("folder") or None
            if note_exists:
                record["notes"] = notes
            result.append(record)
        return result


class RemoteWorker:
    def __init__(
        self,
        config: RemoteConfig,
        paths: PanelPaths,
        client: SupabaseClient,
        executor: RemoteJobExecutor | None = None,
        state_store: RemoteStateStore | None = None,
    ):
        self.config = config
        self.paths = paths
        self.client = client
        self.executor = executor or RemoteJobExecutor(paths)
        self.history = getattr(self.executor, "history", HistoryStore(paths))
        self.activity = ActivityStore(paths)
        self.state_store = state_store or RemoteStateStore(paths.remote_state_file)
        self.active_job_id: str | None = None
        self.last_sync_at = 0.0
        self.last_sync_error = ""
        self.last_sync_completed_at = ""
        self.last_sync_duration_ms = 0
        self.last_sync_summary: dict = {}
        self.last_media_sync_at = 0.0
        self._last_synced_run_fingerprints: dict[str, str] = {}

    def send_heartbeat(self, queue_depth: int = 0, gpu_locked: bool | None = None) -> dict:
        accept_jobs = self.state_store.effective_accept_jobs(self.config)
        payload = heartbeat_payload(
            self.config,
            self.paths,
            active_job_id=self.active_job_id,
            queue_depth=queue_depth,
            gpu_locked=self.executor.gpu_locked() if gpu_locked is None else gpu_locked,
            accept_jobs=accept_jobs,
        )
        if self.last_sync_completed_at or self.last_sync_error or self.last_sync_summary:
            payload.update(
                {
                    "last_sync_at": self.last_sync_completed_at or None,
                    "last_sync_duration_ms": int(self.last_sync_duration_ms or 0),
                    "last_sync_error": self.last_sync_error or "",
                    "last_sync_summary": self.last_sync_summary or {},
                }
            )
        self.client.heartbeat(payload)
        return payload

    def sync_capabilities(self) -> bool:
        try:
            drive_status = self.executor.google_drive.status() if hasattr(self.executor, "google_drive") else {}
            payload = remote_capabilities(self.paths, drive_status=drive_status)
            payload["machine_id"] = self.config.machine_id
            payload["updated_at"] = _now_iso()
            self.client.upsert("machine_capabilities", payload)
            return True
        except Exception:
            # Additive rollout: an older schema must keep heartbeat and inspection alive.
            return False

    def sync_local_activity(self) -> int:
        records = []
        known_run_ids = {str(run.get("id") or "") for run in self.history.list_runs()}
        category_map = {
            "artifact": "media",
            "preset": "organization",
            "metadata": "organization",
            "admin": "organization",
        }
        safe_metadata_keys = {
            "job_type", "type", "status", "run_id", "reward_preset_id",
            "terrain_preset_id", "physics_preset_id", "folder", "source_run_id",
        }
        for event in self.activity.local_events(limit=250):
            source_id = str(event.get("id") or "").strip()
            event_type = re.sub(r"[^a-z0-9_.-]+", "_", str(event.get("event_type") or "activity").lower())[:120]
            if not source_id or not event_type:
                continue
            raw_metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else event.get("payload")
            raw_metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
            metadata = {
                key: raw_metadata[key]
                for key in safe_metadata_keys
                if key in raw_metadata and isinstance(raw_metadata[key], (str, int, float, bool, type(None)))
            }
            run_id = str(event.get("subject_id") or metadata.get("run_id") or "").strip()
            category = category_map.get(str(event.get("category") or "system"), str(event.get("category") or "system"))
            records.append({
                "machine_id": self.config.machine_id,
                "actor_id": None,
                "actor_name": str(event.get("actor_name") or "Mother operator")[:160],
                "actor_role": "admin",
                "event_type": event_type,
                "category": category[:80],
                "outcome": outcome_for_status(str(event.get("outcome") or "info"))[:40],
                "run_id": run_id if run_id in known_run_ids else None,
                "job_id": None,
                "points": int(event.get("points") or 0),
                "metadata": metadata,
                "source_type": "mother_activity",
                "source_id": source_id[:200],
                "created_at": event.get("created_at") or _now_iso(),
            })
        if not records:
            return 0
        try:
            self.client.upsert(
                "team_activity_events",
                records,
                query={"on_conflict": "machine_id,source_type,source_id"},
            )
        except Exception:
            return 0
        return len(records)

    def pull_remote_run_metadata(self) -> int:
        try:
            rows = self.client.select(
                "runs",
                query={
                    "machine_id": f"eq.{self.config.machine_id}",
                    "select": "id,display_name,folder,notes,log_dir,updated_at",
                },
            )
        except Exception:
            return 0
        if not isinstance(rows, list):
            return 0
        updated = 0
        for remote in rows:
            run_id = str(remote.get("id") or "")
            if not run_id:
                continue
            canonical_id = self.history.canonical_run_id(run_id, {"log_dir": remote.get("log_dir")})
            if canonical_id != run_id:
                continue
            local = self.history.get_run(run_id)
            remote_log_dir_text = str(remote.get("log_dir") or "").strip()
            remote_log_dir = Path(remote_log_dir_text) if remote_log_dir_text else None
            if not local and not (remote_log_dir and remote_log_dir.is_dir()):
                continue
            local = local or {}
            if _parse_time(str(remote.get("updated_at") or "")) <= _parse_time(str(local.get("updated_at") or "")):
                continue
            metadata = {
                key: remote.get(key)
                for key in ("display_name", "folder")
                if key in remote
            }
            if metadata:
                self.history.patch_run_metadata(run_id, **metadata)
            if "notes" in remote and remote.get("notes") is not None:
                self.history.set_note(run_id, str(remote.get("notes") or ""))
            updated += 1
        return updated

    def pull_remote_folders(self) -> int:
        try:
            rows = self.client.select(
                "team_folders",
                query={
                    "machine_id": f"eq.{self.config.machine_id}",
                    "select": "name,updated_at",
                },
            )
        except Exception:
            return 0
        if not isinstance(rows, list):
            return 0
        deleted_by_key = {
            str(tombstone.get("folder_key") or _folder_key(str(tombstone.get("name") or ""))): _parse_time(
                str(tombstone.get("deleted_at") or "")
            )
            for tombstone in self.history.deleted_folder_tombstones()
        }
        created = 0
        for row in rows:
            name = str(row.get("name") or "").strip()
            if not name:
                continue
            key = _folder_key(name)
            remote_time = _parse_time(str(row.get("updated_at") or row.get("created_at") or ""))
            deleted_time = deleted_by_key.get(key, 0.0)
            if deleted_time and (not remote_time or remote_time <= deleted_time):
                continue
            try:
                self.history.create_folder(name)
                created += 1
            except Exception:
                continue
        return created

    def push_local_folders(self) -> int:
        records = []
        for folder in self.history.get_folders():
            name = str(folder or "").strip()
            key = _folder_key(name)
            if not name or not key:
                continue
            records.append(
                {
                    "machine_id": self.config.machine_id,
                    "name": name,
                    "folder_key": key,
                }
            )
        if not records:
            return 0
        try:
            self.client.upsert("team_folders", records, query={"on_conflict": "machine_id,folder_key"})
        except Exception:
            return 0
        return len(records)

    def sync_deleted_folders(self) -> int:
        deleted = 0
        for tombstone in self.history.deleted_folder_tombstones():
            key = str(tombstone.get("folder_key") or _folder_key(str(tombstone.get("name") or ""))).strip()
            if not key:
                continue
            try:
                self.client.delete(
                    "team_folders",
                    query={"machine_id": f"eq.{self.config.machine_id}", "folder_key": f"eq.{key}"},
                )
                deleted += 1
            except Exception:
                continue
        return deleted

    def sync_folders(self, *, sync_deletions: bool = True) -> dict:
        return {
            "folders_pulled": self.pull_remote_folders(),
            "folders_pushed": self.push_local_folders(),
            "folders_deleted": self.sync_deleted_folders() if sync_deletions else 0,
        }

    @staticmethod
    def _run_fingerprint(run_record: dict, artifacts: list[dict]) -> str:
        payload = {
            "run": run_record,
            "artifacts": sorted(
                artifacts,
                key=lambda item: (
                    str(item.get("run_id") or ""),
                    str(item.get("kind") or ""),
                    str(item.get("local_path") or item.get("path") or ""),
                    str(item.get("checkpoint_iteration") or ""),
                    str(item.get("storage_path") or ""),
                    str(item.get("public_url") or ""),
                ),
            ),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)

    def _executor_sync_payloads(self, run_ids: set[str] | None = None) -> list[dict]:
        if run_ids is None:
            return list(self.executor.sync_runs_payload())
        try:
            return list(self.executor.sync_runs_payload(run_ids=run_ids))
        except TypeError:
            normalized_ids = {str(run_id) for run_id in run_ids if str(run_id)}
            return [
                run
                for run in self.executor.sync_runs_payload()
                if str(run.get("id") or "") in normalized_ids
            ]

    def sync_runs(
        self,
        *,
        force: bool = False,
        run_ids: set[str] | None = None,
        pull_metadata: bool = True,
        sync_artifacts: bool = True,
        sync_deletions: bool = True,
        sync_aliases: bool = True,
    ) -> dict:
        started = time.time()
        capabilities_synced = self.sync_capabilities()
        activity_synced = self.sync_local_activity()
        if pull_metadata:
            self.pull_remote_run_metadata()
            folder_summary = self.sync_folders(sync_deletions=sync_deletions)
        else:
            folder_summary = {"folders_pulled": 0, "folders_pushed": 0, "folders_deleted": 0}
        normalized_run_ids = {str(run_id) for run_id in run_ids or set() if str(run_id)}
        targeted = bool(normalized_run_ids)
        tombstones = self.history.deleted_run_tombstones(run_ids=normalized_run_ids) if targeted else None
        deletion_summary = (
            self.sync_deleted_runs(tombstones=tombstones)
            if sync_deletions
            else {"tombstones": 0, "deleted_remote_rows": 0, "deleted_remote_artifact_rows": 0}
        )
        run_payloads = self._executor_sync_payloads(normalized_run_ids if targeted else None)
        alias_summary = (
            {"deleted_remote_alias_rows": 0, "deleted_remote_alias_artifact_rows": 0}
            if targeted or not sync_aliases
            else self.sync_remote_alias_runs(run_payloads)
        )
        runs = []
        pending_artifacts: list[tuple[str, dict]] = []
        artifacts = []
        artifact_errors = []
        artifact_failed_run_ids: set[str] = set()
        artifact_skipped_run_ids: set[str] = set()
        candidate_fingerprints: dict[str, str] = {}
        artifacts_skipped = 0
        artifact_repair_runs: set[str] = set()
        self._sync_videos_uploaded = 0
        self._sync_storage_reused = 0
        unchanged = 0
        for run in run_payloads:
            run_record = dict(run)
            artifact_records = run_record.pop("artifacts", []) or []
            now = _now_iso()
            run_record["created_at"] = run_record.get("created_at") or now
            run_record["updated_at"] = run_record.get("updated_at") or run_record["created_at"]
            run_record["params"] = run_record.get("params") or {}
            remote_run = {**run_record, "machine_id": self.config.machine_id}
            fingerprint = self._run_fingerprint(remote_run, artifact_records)
            run_id = str(remote_run.get("id") or "")
            if not force and run_id and self._last_synced_run_fingerprints.get(run_id) == fingerprint:
                unchanged += 1
                if targeted and sync_artifacts and artifact_records:
                    artifact_repair_runs.add(run_id)
                    for artifact in artifact_records:
                        pending_artifacts.append((run_id, artifact))
                continue
            runs.append(remote_run)
            if sync_artifacts:
                for artifact in artifact_records:
                    pending_artifacts.append((run_id, artifact))
            elif artifact_records:
                artifacts_skipped += len(artifact_records)
                if run_id:
                    artifact_skipped_run_ids.add(run_id)
            if run_id:
                candidate_fingerprints[run_id] = fingerprint
        if runs:
            self._upsert_grouped("runs", runs)
        for run_id, artifact in pending_artifacts:
            try:
                record = self._remote_artifact_record(artifact)
                if record:
                    artifacts.append(record)
            except Exception as exc:
                if run_id:
                    artifact_failed_run_ids.add(run_id)
                artifact_errors.append(str(exc))
        if artifacts:
            try:
                self._upsert_grouped("artifacts", artifacts, query={"on_conflict": "run_id,kind,local_path"})
            except Exception as exc:
                message = str(exc)
                if "checkpoint_iteration" in message or "schema cache" in message:
                    try:
                        fallback_artifacts = [
                            {key: value for key, value in artifact.items() if key != "checkpoint_iteration"}
                            for artifact in artifacts
                        ]
                        self._upsert_grouped(
                            "artifacts",
                            fallback_artifacts,
                            query={"on_conflict": "run_id,kind,local_path"},
                        )
                        artifact_errors.append(
                            "Supabase schema is missing artifacts.checkpoint_iteration; "
                            "apply tools/training_panel/supabase/schema.sql and reload PostgREST."
                        )
                    except Exception as fallback_exc:
                        artifact_errors.append(str(fallback_exc))
                        artifact_failed_run_ids.update(str(artifact.get("run_id") or "") for artifact in artifacts)
                else:
                    artifact_errors.append(message)
                    artifact_failed_run_ids.update(str(artifact.get("run_id") or "") for artifact in artifacts)
        if sync_artifacts:
            for run in run_payloads:
                self._prune_stale_checkpoint_artifacts(run)
        notification_errors = self._dispatch_notification_events(runs, artifacts)
        if notification_errors:
            artifact_errors.extend(notification_errors)
        if artifact_errors:
            self.last_sync_error = "; ".join(artifact_errors[-3:])
        else:
            self.last_sync_error = ""
        for run_id, fingerprint in candidate_fingerprints.items():
            if run_id and run_id not in artifact_failed_run_ids and run_id not in artifact_skipped_run_ids:
                self._last_synced_run_fingerprints[run_id] = fingerprint
        duration_ms = int((time.time() - started) * 1000)
        self.last_sync_completed_at = _now_iso()
        self.last_sync_duration_ms = duration_ms
        self.last_sync_summary = {
            "runs_changed": len(runs),
            "runs_unchanged": unchanged,
            "artifacts": len(artifacts),
            "artifacts_skipped": artifacts_skipped,
            "artifact_repair_runs": len(artifact_repair_runs),
            "videos_uploaded": int(getattr(self, "_sync_videos_uploaded", 0)),
            "storage_reused": int(getattr(self, "_sync_storage_reused", 0)),
            "schema": REMOTE_PROTOCOL_VERSION,
            "capabilities_synced": capabilities_synced,
            "activity_synced": activity_synced,
            **deletion_summary,
            **alias_summary,
            **folder_summary,
            "targeted_run_ids": sorted(normalized_run_ids),
            "duration_ms": duration_ms,
        }
        return self.last_sync_summary

    def _prune_stale_checkpoint_artifacts(self, run: dict) -> int:
        run_id = str(run.get("id") or "").strip()
        if not run_id:
            return 0
        current = {
            str(artifact.get("local_path") or artifact.get("path") or "").strip()
            for artifact in (run.get("artifacts") or [])
            if str(artifact.get("kind") or "") == "checkpoint"
        }
        current.discard("")
        if not current:
            return 0
        try:
            rows = self.client.select(
                "artifacts",
                query={
                    "machine_id": f"eq.{self.config.machine_id}",
                    "run_id": f"eq.{run_id}",
                    "kind": "eq.checkpoint",
                    "select": "id,local_path",
                    "limit": "1000",
                },
            )
        except Exception:
            return 0
        deleted = 0
        for row in rows if isinstance(rows, list) else []:
            local_path = str(row.get("local_path") or "").strip()
            row_id = str(row.get("id") or "").strip()
            if not row_id or not local_path or local_path in current:
                continue
            try:
                self.client.delete("artifacts", query={"id": f"eq.{row_id}"})
                deleted += 1
            except Exception:
                continue
        return deleted

    def _upsert_grouped(self, table: str, records: list[dict], **kwargs) -> None:
        groups: dict[tuple[str, ...], list[dict]] = {}
        for record in records:
            key = tuple(sorted(str(item) for item in record.keys()))
            groups.setdefault(key, []).append(record)
        for group in groups.values():
            self.client.upsert(table, group, **kwargs)

    def sync_remote_alias_runs(self, local_runs: list[dict]) -> dict:
        """Remove stale remote aliases that point at a canonical local log_dir."""
        canonical_by_log_dir: dict[str, str] = {}
        for run in local_runs:
            run_id = str(run.get("id") or "").strip()
            log_dir = str(run.get("log_dir") or "").strip()
            if run_id and log_dir:
                canonical_by_log_dir.setdefault(log_dir, run_id)
        if not canonical_by_log_dir:
            return {"deleted_remote_alias_rows": 0, "deleted_remote_alias_artifact_rows": 0}

        deleted_runs = 0
        deleted_artifacts = 0
        for remote in self._remote_runs():
            remote_id = str(remote.get("id") or "").strip()
            log_dir = str(remote.get("log_dir") or "").strip()
            canonical_id = canonical_by_log_dir.get(log_dir)
            if not remote_id or not canonical_id or remote_id == canonical_id:
                continue
            query = {"machine_id": f"eq.{self.config.machine_id}", "run_id": f"eq.{remote_id}"}
            self.client.delete("artifacts", query=query)
            deleted_artifacts += 1
            self.client.delete("runs", query={"machine_id": f"eq.{self.config.machine_id}", "id": f"eq.{remote_id}"})
            deleted_runs += 1
        return {
            "deleted_remote_alias_rows": deleted_runs,
            "deleted_remote_alias_artifact_rows": deleted_artifacts,
        }

    def _dispatch_notification_events(self, runs: list[dict], artifacts: list[dict]) -> list[str]:
        errors: list[str] = []
        artifacts_by_run: dict[str, list[dict]] = {}
        for artifact in artifacts:
            run_id = str(artifact.get("run_id") or "")
            if run_id:
                artifacts_by_run.setdefault(run_id, []).append(artifact)
        for run in runs:
            run_id = str(run.get("id") or "")
            events = notification_events_for_run(
                run,
                machine_id=self.config.machine_id,
                artifacts=artifacts_by_run.get(run_id, []),
                remote_url=self.config.cloudflare_tunnel_host,
                require_video_storage=True,
            )
            for event in events:
                try:
                    if self._notification_handled_exists(str(event.get("event_key") or "")):
                        continue
                    self.client.function_request("notify", event)
                except Exception as exc:
                    errors.append(f"notify {event.get('event_type')} {run_id}: {exc}")
        return errors

    def sync_deleted_runs(self, tombstones: list[dict] | None = None) -> dict:
        deleted_count = 0
        artifact_delete_count = 0
        tombstone_records = []
        tombstone_source = self.history.deleted_run_tombstones() if tombstones is None else tombstones
        for tombstone in tombstone_source:
            record = self._remote_deletion_record(tombstone)
            if record:
                tombstone_records.append(record)
            for query in self._deleted_artifact_queries(tombstone):
                self.client.delete("artifacts", query=query)
                artifact_delete_count += 1
            for query in self._deleted_run_queries(tombstone):
                self.client.delete("runs", query=query)
                deleted_count += 1
        if tombstone_records:
            self.client.upsert("run_deletions", tombstone_records, query={"on_conflict": "machine_id,id"})
        return {
            "tombstones": len(tombstone_records),
            "deleted_remote_rows": deleted_count,
            "deleted_remote_artifact_rows": artifact_delete_count,
        }

    def _remote_deletion_record(self, tombstone: dict) -> dict | None:
        run_id = str(tombstone.get("id") or tombstone.get("log_dir_name") or "").strip()
        if not run_id:
            return None
        log_dir = str(tombstone.get("log_dir") or "").strip()
        log_dir_name = str(tombstone.get("log_dir_name") or "").strip()
        if not log_dir_name and log_dir:
            log_dir_name = Path(log_dir).name
        metadata = {
            key: value
            for key, value in tombstone.items()
            if key not in {"id", "log_dir", "log_dir_name", "deleted_at", "deleted_by"}
        }
        return {
            "machine_id": self.config.machine_id,
            "id": run_id,
            "log_dir": log_dir or None,
            "log_dir_name": log_dir_name or None,
            "deleted_by": _uuid_or_none(str(tombstone.get("deleted_by") or "")),
            "deleted_at": tombstone.get("deleted_at") or _now_iso(),
            "metadata": metadata,
        }

    def _deleted_artifact_queries(self, tombstone: dict) -> list[dict]:
        queries = []
        seen: set[str] = set()

        def add(run_id: str) -> None:
            cleaned = str(run_id or "").strip()
            if not cleaned or cleaned in seen:
                return
            seen.add(cleaned)
            queries.append({"machine_id": f"eq.{self.config.machine_id}", "run_id": f"eq.{cleaned}"})

        add(str(tombstone.get("id") or ""))
        log_dir_name = str(tombstone.get("log_dir_name") or "").strip()
        if log_dir_name:
            add(log_dir_name)
        return queries

    def _deleted_run_queries(self, tombstone: dict) -> list[dict]:
        queries = []
        seen: set[tuple[tuple[str, str], ...]] = set()

        def add(query: dict) -> None:
            normalized = tuple(sorted((str(key), str(value)) for key, value in query.items()))
            if normalized in seen:
                return
            seen.add(normalized)
            queries.append(query)

        run_id = str(tombstone.get("id") or "").strip()
        log_dir = str(tombstone.get("log_dir") or "").strip()
        log_dir_name = str(tombstone.get("log_dir_name") or "").strip()
        base = {"machine_id": f"eq.{self.config.machine_id}"}
        if run_id:
            add({**base, "id": f"eq.{run_id}"})
        if log_dir_name and log_dir_name != run_id:
            add({**base, "id": f"eq.{log_dir_name}"})
        if log_dir:
            add({**base, "log_dir": f"eq.{log_dir}"})
        return queries

    def sync_if_due(self, force: bool = False) -> str:
        now = time.time()
        if not force and now - self.last_sync_at < self.config.sync_interval_seconds:
            return self.last_sync_error
        try:
            summary = self.sync_runs(
                force=force,
                sync_artifacts=force,
                sync_deletions=force,
                sync_aliases=force,
            )
            self.last_sync_summary = summary
        except Exception as exc:
            self.last_sync_error = str(exc)
            self.last_sync_completed_at = _now_iso()
            self.last_sync_summary = {**(self.last_sync_summary or {}), "error": self.last_sync_error}
        finally:
            self.last_sync_at = time.time()
        return self.last_sync_error

    def _recent_media_run_ids(self, now: float | None = None) -> set[str]:
        now = time.time() if now is None else now
        cutoff = now - MEDIA_SYNC_RECENT_SECONDS
        run_ids: set[str] = set()
        for run in self.history.list_runs():
            run_id = str(run.get("id") or "").strip()
            if not run_id:
                continue
            video_status = str(run.get("video_status") or "").strip().lower()
            if video_status == "recording":
                run_ids.add(run_id)
                continue
            if (
                video_status not in MEDIA_SYNC_STATUSES
                and not run.get("latest_video")
                and not run.get("tensorboard_summary_path")
            ):
                continue
            timestamp = _parse_time(str(run.get("updated_at") or run.get("created_at") or ""))
            if timestamp and timestamp >= cutoff:
                run_ids.add(run_id)
        return run_ids

    def sync_recent_media_runs_if_due(self) -> str:
        now = time.time()
        if now - self.last_media_sync_at < MEDIA_SYNC_INTERVAL_SECONDS:
            return self.last_sync_error
        self.last_media_sync_at = now
        run_ids = self._recent_media_run_ids(now)
        if not run_ids:
            return self.last_sync_error
        try:
            summary = self.sync_runs(force=False, run_ids=run_ids, pull_metadata=False)
            self.last_sync_summary = summary
            if summary.get("runs_changed") or summary.get("artifacts"):
                self.last_sync_at = time.time()
        except Exception as exc:
            self.last_sync_error = str(exc)
            self.last_sync_completed_at = _now_iso()
            self.last_sync_summary = {**(self.last_sync_summary or {}), "error": self.last_sync_error}
        return self.last_sync_error

    @staticmethod
    def _job_run_ids_for_sync(job: dict, result: dict) -> set[str]:
        job_type = str(job.get("type") or job.get("job_type") or "")
        payload = _job_payload(job)
        result_payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
        ids = {
            str(payload.get("run_id") or ""),
            str(result.get("local_run_id") or ""),
            str(result_payload.get("run_id") or ""),
            str(result_payload.get("source_run_id") or ""),
        }
        if job_type == "start_training":
            ids.add(str(result.get("process_id") or ""))
            ids.add(str(result_payload.get("id") or ""))
        return {run_id for run_id in ids if run_id}

    def sync_after_job(self, job: dict, result: dict) -> str:
        if str(job.get("type") or job.get("job_type") or "") == "send_missed_notifications":
            return ""
        run_ids = self._job_run_ids_for_sync(job, result)
        if not run_ids:
            return self.sync_if_due(force=True)
        try:
            summary = self.sync_runs(force=True, run_ids=run_ids, pull_metadata=False)
            self.last_sync_summary = summary
            self.last_sync_at = time.time()
        except Exception as exc:
            self.last_sync_error = str(exc)
            self.last_sync_completed_at = _now_iso()
            self.last_sync_summary = {**(self.last_sync_summary or {}), "error": self.last_sync_error}
        return self.last_sync_error

    def _run_record_for_upsert(self, run: dict, requester_id: str | None = None, requester_label: str | None = None) -> dict:
        now = _now_iso()
        params = run.get("params") if isinstance(run.get("params"), dict) else {}
        params = dict(params)
        if requester_id:
            params["requester_id"] = requester_id
        if requester_label:
            params["requester_label"] = requester_label
        created_by = requester_id or run.get("created_by") or params.get("requester_id")
        record = {
            "id": run.get("id"),
            "machine_id": self.config.machine_id,
            "status": run.get("status") or "running",
            "created_at": run.get("created_at") or now,
            "updated_at": run.get("updated_at") or now,
            "log_dir": run.get("log_dir"),
            "params": params,
            "latest_checkpoint": run.get("latest_checkpoint"),
            "latest_video": run.get("latest_video"),
            "onnx_path": run.get("onnx_path"),
            "created_by": created_by,
            "convergence_detected": bool(run.get("convergence_detected")),
            "convergence_iteration": run.get("convergence_iteration"),
            "convergence_improvement_pct": run.get("convergence_improvement_pct"),
            "video_status": run.get("video_status"),
            "returncode": run.get("returncode"),
            "compacted_at": run.get("compacted_at"),
            "compacted_deleted_count": run.get("compacted_deleted_count"),
            "compacted_bytes_freed": run.get("compacted_bytes_freed"),
        }
        display_name = run.get("display_name") if "display_name" in run else params.get("display_name")
        folder = run.get("folder") if "folder" in run else params.get("folder")
        if "display_name" in run or "display_name" in params:
            record["display_name"] = display_name or None
        if "folder" in run or "folder" in params:
            record["folder"] = folder or None
        if "notes" in run:
            record["notes"] = run.get("notes") or ""
        return record

    def _sync_launched_training_run(self, job: dict, result: dict) -> str:
        if str(job.get("type") or job.get("job_type") or "") != "start_training":
            return ""
        result_payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
        run_id = str(result.get("local_run_id") or result.get("process_id") or result_payload.get("id") or "")
        if not run_id:
            return ""
        requester_id = _requester_id_for_job(job) or _uuid_or_none(str(result_payload.get("created_by") or ""))
        requester_label = _requester_label_for_job(job, requester_id) or result_payload.get("requester_label")
        run_record = {
            **result_payload,
            "id": run_id,
            "status": result_payload.get("status") or "running",
        }
        try:
            self.client.upsert("runs", self._run_record_for_upsert(run_record, requester_id, requester_label))
        except Exception as exc:
            return str(exc)
        return ""

    def _mark_job_running(self, job_id: str) -> None:
        try:
            if hasattr(self.client, "mark_job_running"):
                self.client.mark_job_running(job_id)
            else:
                self.client.update("jobs", {"status": "running"}, {"id": f"eq.{job_id}"})
        except Exception:
            # Queue visibility is helpful, but a status PATCH must not block launch.
            return

    def _execute_job(self, job: dict) -> RemoteJobResult:
        job_type = str(job.get("type") or job.get("job_type") or "")
        profile = self._profile_for_job(job)
        actor_role = str(profile.get("role") or "viewer").strip().lower()
        if not profile.get("id"):
            raise PermissionError("Remote job actor does not have an authoritative profile")
        if not role_allows(actor_role, job_type):
            raise PermissionError(f"Role '{actor_role}' cannot run remote job '{job_type}'")
        job["actor_role"] = actor_role
        job["actor_name"] = str(profile.get("display_name") or profile.get("email") or profile.get("id"))
        if job_type == "send_missed_notifications":
            return RemoteJobResult(payload=self._send_missed_notifications(job))
        return self.executor.execute(job)

    def _notification_settings_for_requester(self, requester_id: str) -> dict:
        try:
            rows = self.client.select(
                "notification_settings",
                query={
                    "user_id": f"eq.{requester_id}",
                    "machine_id": f"eq.{self.config.machine_id}",
                    "select": "*",
                    "limit": "1",
                },
            )
        except Exception:
            return {}
        return rows[0] if isinstance(rows, list) and rows and isinstance(rows[0], dict) else {}

    def _notification_enabled(self, event: dict, settings: dict) -> bool:
        if settings and settings.get("discord_enabled") is not True:
            return False
        setting_key = NOTIFICATION_SETTING_KEYS.get(str(event.get("event_type") or ""))
        return not setting_key or settings.get(setting_key, True) is not False

    def _sent_event_exists(self, event_key: str) -> bool:
        if not event_key:
            return False
        try:
            rows = self.client.select(
                "run_events",
                query={
                    "event_key": f"eq.{event_key}",
                    "notification_status": "eq.sent",
                    "select": "event_key,notification_status",
                    "limit": "1",
                },
            )
        except Exception:
            return False
        return isinstance(rows, list) and bool(rows)

    def _notification_handled_exists(self, event_key: str) -> bool:
        if not event_key:
            return False
        try:
            rows = self.client.select(
                "run_events",
                query={
                    "event_key": f"eq.{event_key}",
                    "notification_status": f"in.({','.join(sorted(HANDLED_NOTIFICATION_STATUSES))})",
                    "select": "event_key,notification_status",
                    "limit": "1",
                },
            )
        except Exception:
            return False
        return isinstance(rows, list) and bool(rows)

    def _remote_start_jobs(self) -> list[dict]:
        try:
            rows = self.client.select(
                "jobs",
                query={
                    "type": "eq.start_training",
                    "select": "id,machine_id,type,status,payload,result,actor_id,actor_role,created_at,updated_at",
                    "order": "created_at.desc",
                    "limit": "200",
                },
            )
        except Exception:
            return []
        return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []

    def _remote_runs(self) -> list[dict]:
        try:
            rows = self.client.select(
                "runs",
                query={
                    "machine_id": f"eq.{self.config.machine_id}",
                    "select": "*",
                    "order": "created_at.desc",
                    "limit": "200",
                },
            )
        except Exception:
            return []
        return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []

    def _missed_run_candidates(self, requester_id: str, requester_label: str | None = None) -> list[tuple[dict, list[dict]]]:
        jobs_by_run: dict[str, dict] = {}
        for job in self._remote_start_jobs():
            if job.get("machine_id") not in (None, "", self.config.machine_id):
                continue
            if _requester_id_for_job(job) != requester_id:
                continue
            run_id = _run_id_for_job(job)
            if run_id:
                jobs_by_run[run_id] = job

        combined: dict[str, dict] = {}
        artifacts_by_run: dict[str, list[dict]] = {}
        for run in self._remote_runs():
            run_id = str(run.get("id") or "")
            if run_id:
                combined[run_id] = dict(run)
        for local in self.executor.sync_runs_payload():
            run_id = str(local.get("id") or "")
            if not run_id:
                continue
            artifacts_by_run[run_id] = list(local.get("artifacts") or [])
            existing = combined.get(run_id, {})
            existing_params = existing.get("params") if isinstance(existing.get("params"), dict) else {}
            local_params = local.get("params") if isinstance(local.get("params"), dict) else {}
            merged = {**existing, **local}
            merged["params"] = {**existing_params, **local_params}
            if not merged.get("created_by"):
                merged["created_by"] = existing.get("created_by") or local.get("created_by") or merged["params"].get("requester_id")
            combined[run_id] = merged

        candidates: list[tuple[dict, list[dict]]] = []
        for run_id, run in combined.items():
            status = str(run.get("status") or "").lower()
            if status not in FINISHED_RUN_STATUSES:
                continue
            params = run.get("params") if isinstance(run.get("params"), dict) else {}
            owned = str(run.get("created_by") or params.get("requester_id") or "") == requester_id
            if not owned and run_id not in jobs_by_run:
                continue
            if not owned:
                params = {**params, "requester_id": requester_id}
                if requester_label:
                    params["requester_label"] = requester_label
                run = {**run, "created_by": requester_id, "params": params}
                try:
                    self.history.patch_run_metadata(
                        run_id,
                        created_by=requester_id,
                        requester_label=requester_label,
                        params=params,
                    )
                    self.client.upsert("runs", self._run_record_for_upsert(run, requester_id, requester_label))
                except Exception:
                    pass
            candidates.append((run, artifacts_by_run.get(run_id, [])))

        return sorted(
            candidates,
            key=lambda item: str(item[0].get("updated_at") or item[0].get("created_at") or ""),
            reverse=True,
        )

    def _send_missed_notifications(self, job: dict) -> dict:
        payload = _job_payload(job)
        requester_id = _requester_id_for_job(job)
        if not requester_id:
            raise ValueError("requester_id is required to send missed notifications")
        requester_label = _requester_label_for_job(job, requester_id)
        scope = str(payload.get("scope") or payload.get("mode") or "latest").lower()
        if scope in {"future", "future_only", "none"}:
            return {"scope": scope, "checked": 0, "sent": 0, "skipped": 0, "message": "Future notifications only."}
        if scope not in {"latest", "all"}:
            raise ValueError("scope must be latest or all")

        settings = self._notification_settings_for_requester(requester_id)
        candidates = self._missed_run_candidates(requester_id, requester_label)
        selected = candidates[:1] if scope == "latest" else candidates
        summary = {
            "scope": scope,
            "requester_id": requester_id,
            "checked": len(candidates),
            "matched": len(selected),
            "sent": 0,
            "skipped_sent": 0,
            "skipped_disabled": 0,
            "errors": [],
        }
        for run, artifacts in selected:
            events = notification_events_for_run(
                run,
                machine_id=self.config.machine_id,
                artifacts=artifacts,
                remote_url=self.config.cloudflare_tunnel_host,
                require_video_storage=True,
            )
            for event in events:
                if not self._notification_enabled(event, settings):
                    summary["skipped_disabled"] += 1
                    continue
                event_key = str(event.get("event_key") or "")
                if self._notification_handled_exists(event_key):
                    summary["skipped_sent"] += 1
                    continue
                try:
                    self.client.function_request("notify", event)
                    summary["sent"] += 1
                except Exception as exc:
                    summary["errors"].append(f"{event.get('event_type')} {run.get('id')}: {exc}")
        return summary

    def _remote_artifact_record(self, artifact: dict) -> dict | None:
        run_id = str(artifact.get("run_id") or "")
        local_path = str(artifact.get("local_path") or artifact.get("path") or "")
        kind = str(artifact.get("kind") or "")
        if not run_id or not local_path or not kind:
            return None
        record = {
            "run_id": run_id,
            "machine_id": self.config.machine_id,
            "kind": kind,
            "local_path": local_path,
            "storage_path": None,
            "public_url": None,
            "bytes": None,
            "checkpoint_iteration": artifact.get("checkpoint_iteration"),
        }
        if record["checkpoint_iteration"] is None:
            record["checkpoint_iteration"] = _checkpoint_iteration(local_path)
        if record["checkpoint_iteration"] is not None:
            record["checkpoint_iteration"] = int(record["checkpoint_iteration"])
        if artifact.get("bytes") is not None:
            record["bytes"] = int(artifact["bytes"])
        elif Path(local_path).is_file():
            record["bytes"] = Path(local_path).stat().st_size
        if artifact.get("public_url"):
            record["public_url"] = str(artifact["public_url"])
        if kind in {"video", "mujoco_video"}:
            storage_path = self._ensure_video_storage_path(run_id, local_path, kind=kind)
            if storage_path:
                record["storage_path"] = storage_path
        elif kind == "tensorboard_summary":
            storage_path = self._ensure_tensorboard_summary_storage_path(run_id, local_path)
            if storage_path:
                record["storage_path"] = storage_path
            if not record["public_url"] and self.config.cloudflare_tunnel_host:
                record["public_url"] = (
                    f"{self.config.cloudflare_tunnel_host.rstrip('/')}"
                    f"/api/runs/{quote(run_id, safe='')}/tensorboard-summary.png"
                )
        return record

    def _ensure_video_storage_path(self, run_id: str, local_path: str, *, kind: str = "video") -> str | None:
        path = Path(local_path)
        if not path.is_file():
            return None
        existing = self._existing_artifact(run_id, kind, local_path)
        if existing and existing.get("storage_path"):
            self._sync_storage_reused = getattr(self, "_sync_storage_reused", 0) + 1
            return str(existing["storage_path"])
        folder = "mujoco" if kind == "mujoco_video" else "videos"
        storage_path = f"runs/{_storage_safe(run_id)}/{folder}/{_storage_safe(path.name)}"
        self.client.upload_storage_object(VIDEO_BUCKET, storage_path, path, content_type="video/mp4")
        self._sync_videos_uploaded = getattr(self, "_sync_videos_uploaded", 0) + 1
        return storage_path

    def _ensure_tensorboard_summary_storage_path(self, run_id: str, local_path: str) -> str | None:
        path = Path(local_path)
        if not path.is_file():
            return None
        existing = self._existing_artifact(run_id, "tensorboard_summary", local_path)
        if existing and existing.get("storage_path"):
            self._sync_storage_reused = getattr(self, "_sync_storage_reused", 0) + 1
            return str(existing["storage_path"])
        storage_path = f"runs/{_storage_safe(run_id)}/tensorboard/{_storage_safe(path.name)}"
        self.client.upload_storage_object(
            VIDEO_BUCKET,
            storage_path,
            path,
            content_type=TENSORBOARD_SUMMARY_CONTENT_TYPE,
        )
        return storage_path

    def _existing_artifact(self, run_id: str, kind: str, local_path: str) -> dict | None:
        try:
            rows = self.client.select(
                "artifacts",
                query={
                    "run_id": f"eq.{run_id}",
                    "kind": f"eq.{kind}",
                    "local_path": f"eq.{local_path}",
                    "select": "run_id,kind,local_path,storage_path",
                    "limit": "1",
                },
            )
        except Exception:
            return None
        if isinstance(rows, list) and rows:
            return rows[0]
        return None

    def _profile_for_job(self, job: dict) -> dict:
        actor_id = _uuid_or_none(str(job.get("actor_id") or ""))
        if not actor_id:
            return {}
        try:
            rows = self.client.select(
                "profiles",
                query={
                    "id": f"eq.{actor_id}",
                    "select": "id,email,display_name,role",
                    "limit": "1",
                },
            )
        except Exception:
            return {}
        if isinstance(rows, list) and rows:
            row = rows[0]
            return row if isinstance(row, dict) else {}
        return {}

    def _record_job_activity(
        self,
        job: dict,
        outcome: str,
        *,
        result: dict | None = None,
        error: str = "",
    ) -> None:
        try:
            job_type = str(job.get("type") or job.get("job_type") or "job")
            payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
            result = result or {}
            result_payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
            profile = self._profile_for_job(job)
            actor_id = _uuid_or_none(str(job.get("actor_id") or ""))
            actor_name = str(
                profile.get("display_name")
                or profile.get("email")
                or job.get("actor_name")
                or actor_id
                or job.get("actor_role")
                or "Remote member"
            )
            actor_role = str(profile.get("role") or job.get("actor_role") or "viewer")
            category = category_for_job_type(job_type)
            normalized_outcome = outcome_for_status(outcome)
            run_id = str(
                payload.get("run_id")
                or result.get("local_run_id")
                or result_payload.get("run_id")
                or result_payload.get("source_run_id")
                or ""
            )
            metadata = {
                "job_type": job_type,
                "status": normalized_outcome,
                "run_id": run_id,
                "process_id": result.get("process_id") or result_payload.get("id") or "",
                "reward_preset_id": payload.get("reward_preset_id") or "",
                "terrain_preset_id": payload.get("terrain_preset_id") or "",
                "physics_preset_id": payload.get("physics_preset_id") or "",
                "error": error,
            }
            record = {
                "machine_id": self.config.machine_id,
                "actor_id": actor_id,
                "actor_name": actor_name,
                "actor_role": actor_role if actor_role in {"viewer", "operator", "admin"} else "viewer",
                "event_type": f"remote_job_{normalized_outcome}",
                "category": category,
                "outcome": normalized_outcome,
                "run_id": run_id or None,
                "job_id": _uuid_or_none(str(job.get("id") or "")),
                "points": score_activity_event(job_type, outcome=normalized_outcome, category=category, job_type=job_type),
                "metadata": metadata,
                "source_type": "worker_job",
                "source_id": f"{job.get('id') or 'unknown'}:{normalized_outcome}",
                "created_at": _now_iso(),
            }
            self.client.insert("team_activity_events", record, prefer="return=minimal")
        except Exception:
            # Analytics must never block training, video, export, or stop jobs.
            return

    def poll_once(self) -> dict:
        accept_jobs = self.state_store.effective_accept_jobs(self.config)
        gpu_locked = self.executor.gpu_locked()
        heartbeat = self.send_heartbeat(gpu_locked=gpu_locked)
        heartbeat_sync_at = heartbeat.get("last_sync_at")
        if not accept_jobs:
            self.sync_if_due()
            self.sync_recent_media_runs_if_due()
            if self.last_sync_completed_at and self.last_sync_completed_at != heartbeat_sync_at:
                heartbeat = self.send_heartbeat(gpu_locked=gpu_locked)
            return {"status": "disabled", "heartbeat": heartbeat}
        job = self.client.claim_next_job(self.config.machine_id, gpu_locked=gpu_locked)
        if not job:
            self.sync_if_due()
            self.sync_recent_media_runs_if_due()
            if self.last_sync_completed_at and self.last_sync_completed_at != heartbeat_sync_at:
                heartbeat = self.send_heartbeat(gpu_locked=gpu_locked)
            return {"status": "idle", "heartbeat": heartbeat}

        job_id = str(job.get("id"))
        self.active_job_id = job_id
        self._mark_job_running(job_id)
        job["status"] = "running"
        self._record_job_activity(job, "queued")
        self._record_job_activity(job, "claimed")
        self._record_job_activity(job, "running")
        try:
            result = self._execute_job(job).to_dict()
            launch_sync_error = self._sync_launched_training_run(job, result)
            sync_error = self.sync_after_job(job, result)
            sync_errors = [error for error in (launch_sync_error, sync_error) if error]
            if sync_errors:
                result.setdefault("payload", {})["sync_error"] = "; ".join(sync_errors)
            self.client.complete_job(job_id, result)
            self._record_job_activity(job, "completed", result=result)
            return {"status": "completed", "job_id": job_id, "result": result}
        except Exception as exc:
            sync_error = self.sync_if_due(force=True)
            result = {"job": job}
            if sync_error:
                result["sync_error"] = sync_error
            self.client.fail_job(job_id, str(exc), result)
            self._record_job_activity(job, "failed", result=result, error=str(exc))
            return {"status": "failed", "job_id": job_id, "error": str(exc)}
        finally:
            self.active_job_id = None
            self.send_heartbeat()

    def run_forever(self) -> None:
        while True:
            try:
                self.poll_once()
            except Exception as exc:
                print(f"[remote-worker] poll error: {exc}")
            time.sleep(self.config.poll_interval_seconds)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the RedRHex remote training worker.")
    parser.add_argument("--once", action="store_true", help="Poll once and exit.")
    args = parser.parse_args(argv)

    paths = PanelPaths.from_env()
    paths.ensure_dirs()
    config = RemoteConfig.from_env()
    client = SupabaseClient(config)
    worker = RemoteWorker(config, paths, client)
    if args.once:
        print(worker.poll_once())
    else:
        print(f"RedRHex remote worker online for machine {config.machine_id}")
        worker.run_forever()
