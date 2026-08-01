from __future__ import annotations

import json
import os
import re
import signal
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .commands import (
    DEFAULT_TASK,
    DEFAULT_VIDEO_PRESET,
    DEFAULT_FOLLOW_CAMERA_EYE,
    DEFAULT_FOLLOW_CAMERA_LOOKAT,
    TrainingParams,
    VideoParams,
    display_isaaclab_command,
    export_onnx_argv,
    play_argv,
    resolve_spring_backend,
    shell_for_command,
    shell_for_isaaclab,
    tensorboard_argv,
    training_argv,
)
from .config import PanelPaths, timestamp_id
from .deploy import latest_deploy_report
from .history import HistoryStore, latest_checkpoint, latest_onnx, latest_video, tail_file
from .terrain import read_terrain_values_from_yaml

EXTERNAL_PLAY_ID_PREFIX = "external_play_"
EXTERNAL_VIDEO_ID_PREFIX = "external_video_"
EXTERNAL_ONNX_ID_PREFIX = "external_onnx_"
EXTERNAL_TRAINING_ID_PREFIX = "external_training_"
EXTERNAL_TENSORBOARD_ID_PREFIX = "external_tensorboard_"
EXTERNAL_GPU_ID_PREFIX = "external_gpu_"
EXTERNAL_ID_PREFIXES = (
    EXTERNAL_PLAY_ID_PREFIX,
    EXTERNAL_VIDEO_ID_PREFIX,
    EXTERNAL_ONNX_ID_PREFIX,
    EXTERNAL_TRAINING_ID_PREFIX,
    EXTERNAL_TENSORBOARD_ID_PREFIX,
    EXTERNAL_GPU_ID_PREFIX,
)
GPU_PROCESS_KINDS = {"training", "play", "video", "onnx", "deploy", "gpu"}
DEFAULT_ISAAC_SETTLE_SECONDS = 5.0


def _isaac_settle_seconds_from_env() -> float:
    value = os.environ.get("TRAINING_PANEL_ISAAC_SETTLE_SECONDS")
    if value is None or value == "":
        return DEFAULT_ISAAC_SETTLE_SECONDS
    try:
        return max(0.0, float(value))
    except ValueError:
        return DEFAULT_ISAAC_SETTLE_SECONDS

@dataclass
class ProcessInfo:
    kind: str
    pid: int
    run_id: str
    log_file: str
    started_at: str
    command: str
    source_run_id: str | None = None
    tmux_session: str | None = None
    attach_command: str | None = None
    exit_file: str | None = None


@dataclass
class SpawnedProcess:
    proc: subprocess.Popen
    tmux_session: str | None = None
    attach_command: str | None = None
    exit_file: str | None = None


class ProcessStartError(RuntimeError):
    def __init__(self, message: str, payload: dict):
        super().__init__(message)
        self.payload = {"error": message, **payload}


class CudaPreflightError(RuntimeError):
    def __init__(self, health: dict):
        message = str(health.get("error") or "CUDA driver preflight failed")
        super().__init__(message)
        self.health = health
        self.payload = {"error": message, "cuda_health": health}


class ProcessRegistry:
    def __init__(
        self,
        paths: PanelPaths,
        history: HistoryStore,
        isaac_settle_seconds: float | None = None,
        *,
        cuda_preflight: bool = False,
    ):
        self.paths = paths
        self.history = history
        self.cuda_preflight = bool(cuda_preflight)
        self._lock = threading.Lock()
        self._queue_lock = threading.Lock()
        self._processes: dict[str, subprocess.Popen] = {}
        self._infos: dict[str, ProcessInfo] = {}
        self._tensorboards_by_logdir: dict[str, str] = {}
        self.isaac_settle_seconds = (
            _isaac_settle_seconds_from_env()
            if isaac_settle_seconds is None
            else max(0.0, float(isaac_settle_seconds))
        )
        self._last_isaac_exit_at = 0.0
        self._queued_training_timer: threading.Timer | None = None

    def list_processes(self) -> list[dict]:
        with self._lock:
            infos = []
            for run_id, info in self._infos.items():
                proc = self._processes.get(run_id)
                infos.append(self._process_info_snapshot(info, proc))
        known_groups = {self._process_group_for_pid(info["pid"]) for info in infos}
        known_groups.update(info.get("process_group") for info in infos if info.get("process_group"))
        known_log_files = {str(info.get("log_file") or "") for info in infos if info.get("log_file")}
        external = [
            process
            for process in self._external_processes()
            if process.get("process_group") not in known_groups
            and not self._matches_known_process_log(process, known_log_files)
        ]
        return infos + external

    def _process_info_snapshot(self, info: ProcessInfo, proc: subprocess.Popen | None) -> dict:
        returncode = self._returncode(info, proc) if proc else None
        process_group = self._process_group_for_info(info) if returncode is None else None
        gpu_pids = self._gpu_pids_for_group(process_group)
        snapshot = {
            **info.__dict__,
            "returncode": returncode,
        }
        if process_group:
            snapshot["process_group"] = process_group
        if gpu_pids:
            snapshot["gpu_pids"] = gpu_pids
            snapshot["gpu_pid"] = gpu_pids[0]
        return snapshot

    def queue_training(self, params: TrainingParams) -> dict:
        self._assert_cuda_ready(params.device)
        with self._queue_lock:
            settle_delay = self._isaac_settle_delay()
            if self._queued_training_runs() or self.running_isaac_processes() or settle_delay > 0:
                run = self._create_queued_training_run(params)
                if settle_delay > 0:
                    self._schedule_queued_training_start_locked(settle_delay)
                return run
            return self._start_training_run(params)

    def start_training(self, params: TrainingParams) -> dict:
        return self._start_training_run(params)

    def _create_queued_training_run(self, params: TrainingParams) -> dict:
        self.paths.ensure_dirs()
        run_id = f"panel_{timestamp_id()}"
        queued_at = datetime.now().isoformat(timespec="seconds")
        script_argv = training_argv(params)
        log_file = self.paths.process_log_dir / f"{run_id}.log"
        record = {
            "id": run_id,
            "source": "training_panel",
            "status": "queued",
            "created_at": queued_at,
            "updated_at": queued_at,
            "queued_at": queued_at,
            "params": params.to_dict(),
            "command": display_isaaclab_command(self.paths, script_argv),
            "process_log": str(log_file),
            "log_dir": None,
            "reward_preset_id": params.reward_preset_id,
            "reward_overrides": params.reward_overrides,
            "terrain_preset_id": params.terrain_preset_id,
            "terrain_overrides": params.terrain_overrides,
            "created_by": params.requester_id,
            "requester_label": params.requester_label,
            "display_name": params.display_name,
            "folder": params.folder,
            "client_request_id": params.client_request_id,
        }
        self.history.add_run(record)
        return record

    def _start_training_run(
        self,
        params: TrainingParams,
        *,
        run_id: str | None = None,
        existing_record: dict | None = None,
    ) -> dict:
        self._assert_cuda_ready(params.device)
        self.paths.ensure_dirs()
        run_id = run_id or f"panel_{timestamp_id()}"
        started_at_epoch = time.time()
        started_at = datetime.now().isoformat(timespec="seconds")
        script_argv = training_argv(params)
        shell = shell_for_isaaclab(self.paths, script_argv)
        log_file = self.paths.process_log_dir / f"{run_id}.log"
        # Write panel overrides immediately before spawning so train.py reads the right queued run settings.
        self._write_reward_override(params.reward_overrides)
        self._write_terrain_override(params.terrain_overrides)
        record = {
            "id": run_id,
            "source": "training_panel",
            "status": "running",
            "created_at": (existing_record or {}).get("created_at", started_at),
            "updated_at": started_at,
            "started_at": started_at,
            "queued_at": (existing_record or {}).get("queued_at"),
            "params": params.to_dict(),
            "command": display_isaaclab_command(self.paths, script_argv),
            "process_log": str(log_file),
            "log_dir": None,
            "reward_preset_id": params.reward_preset_id,
            "reward_overrides": params.reward_overrides,
            "terrain_preset_id": params.terrain_preset_id,
            "terrain_overrides": params.terrain_overrides,
            "created_by": params.requester_id,
            "requester_label": params.requester_label,
            "display_name": params.display_name,
            "folder": params.folder,
            "client_request_id": params.client_request_id,
        }
        if existing_record:
            self.history.update_run(run_id, **record)
        else:
            self.history.add_run(record)
        spawned = self._spawn_shell(run_id, shell, log_file)
        proc = spawned.proc
        self.history.update_run(run_id, pid=proc.pid)
        self.history.update_run(
            run_id,
            tmux_session=spawned.tmux_session,
            attach_command=spawned.attach_command,
            exit_file=spawned.exit_file,
        )
        self._register(run_id, "training", spawned, log_file, started_at, shell)
        thread = threading.Thread(
            target=self._monitor_training,
            args=(run_id, proc, started_at_epoch),
            daemon=True,
        )
        thread.start()
        return {
            **record,
            "pid": proc.pid,
            "tmux_session": spawned.tmux_session,
            "attach_command": spawned.attach_command,
        }

    def _queued_training_runs(self) -> list[dict]:
        queued = [
            run
            for run in self.history.list_runs()
            if run.get("source") == "training_panel" and str(run.get("status") or "").lower() == "queued"
        ]
        return sorted(queued, key=lambda run: str(run.get("queued_at") or run.get("created_at") or ""))

    def start_next_queued_training(self) -> dict | None:
        with self._queue_lock:
            if self.running_isaac_processes():
                return None
            queued = self._queued_training_runs()
            if not queued:
                return None
            settle_delay = self._isaac_settle_delay()
            if settle_delay > 0:
                self._schedule_queued_training_start_locked(settle_delay)
                return None
            run = queued[0]
            try:
                params = TrainingParams.from_dict(run.get("params") or {})
                return self._start_training_run(params, run_id=str(run["id"]), existing_record=run)
            except CudaPreflightError as exc:
                self.history.update_run(
                    str(run.get("id") or ""),
                    status="failed",
                    returncode=None,
                    queue_error=str(exc),
                    error=str(exc),
                    cuda_health=exc.health,
                )
                return None
            except Exception as exc:
                self.history.update_run(
                    str(run.get("id") or ""),
                    status="failed",
                    returncode=1,
                    queue_error=str(exc),
                )
                return None

    def _mark_isaac_process_finished(self) -> None:
        self._last_isaac_exit_at = time.time()

    def _isaac_settle_delay(self) -> float:
        if self.isaac_settle_seconds <= 0 or self._last_isaac_exit_at <= 0:
            return 0.0
        return max(0.0, self.isaac_settle_seconds - (time.time() - self._last_isaac_exit_at))

    def _schedule_queued_training_start_locked(self, delay_seconds: float) -> None:
        if delay_seconds <= 0:
            return
        if self._queued_training_timer and self._queued_training_timer.is_alive():
            return
        timer = threading.Timer(delay_seconds, self.start_next_queued_training)
        timer.daemon = True
        self._queued_training_timer = timer
        timer.start()

    def cancel_queued_training(self, run_id: str) -> bool:
        run = self.history.get_run(run_id)
        if not run or str(run.get("status") or "").lower() != "queued":
            return False
        self.history.update_run(run_id, status="cancelled", cancelled_at=datetime.now().isoformat(timespec="seconds"))
        return True

    def _write_reward_override(self, overrides: dict) -> None:
        import json as _json
        override_file = self.paths.reward_override_file
        if overrides:
            override_file.parent.mkdir(parents=True, exist_ok=True)
            override_file.write_text(_json.dumps(overrides, indent=2), encoding="utf-8")
        elif override_file.exists():
            override_file.unlink()

    def _write_terrain_override(self, overrides: dict) -> None:
        import json as _json
        override_file = self.paths.terrain_override_file
        if overrides:
            override_file.parent.mkdir(parents=True, exist_ok=True)
            override_file.write_text(_json.dumps(overrides, indent=2), encoding="utf-8")
        elif override_file.exists():
            override_file.unlink()

    def _terrain_overrides_for_run(self, run_id: str, checkpoint: str | Path | None = None) -> tuple[dict, str]:
        run = self.history.get_run(run_id) or {}
        metadata_overrides = run.get("terrain_overrides") or (run.get("params") or {}).get("terrain_overrides") or {}
        if metadata_overrides:
            return dict(metadata_overrides), "run metadata"

        log_dir = Path(str(run["log_dir"])) if run.get("log_dir") else None
        if log_dir is None and checkpoint:
            checkpoint_path = Path(str(checkpoint))
            if checkpoint_path.name.startswith("model_"):
                log_dir = checkpoint_path.parent

        if log_dir:
            env_yaml = log_dir / "params" / "env.yaml"
            if env_yaml.exists():
                values = read_terrain_values_from_yaml(env_yaml)
                if values:
                    return values, str(env_yaml)

        return {}, "none"

    def _write_process_terrain_override(self, process_id: str, run_id: str, checkpoint: str | Path | None = None) -> str | None:
        overrides, source = self._terrain_overrides_for_run(run_id, checkpoint)
        if not overrides:
            return None
        self.paths.process_override_dir.mkdir(parents=True, exist_ok=True)
        path = self.paths.process_override_dir / f"{process_id}_terrain.json"
        path.write_text(json.dumps({"source": source, "overrides": overrides}, indent=2), encoding="utf-8")
        return str(path)

    def stop(self, run_id: str) -> bool:
        if self.cancel_queued_training(run_id):
            return True
        with self._lock:
            proc = self._processes.get(run_id)
            info = self._infos.get(run_id)
        if not proc and run_id.startswith(EXTERNAL_ID_PREFIXES):
            return self._stop_external_group(run_id)
        if not proc or proc.poll() is not None:
            return False
        process_group = os.getpgid(proc.pid)
        if info and info.tmux_session:
            if not self._send_tmux_interrupt(info.tmux_session):
                os.killpg(process_group, signal.SIGINT)
        else:
            os.killpg(process_group, signal.SIGINT)
        if info and info.kind == "mujoco" and info.source_run_id:
            self.history.update_run(info.source_run_id, mujoco_playback_status="stopping")
        self.history.update_run(run_id, status="stopping")
        threading.Thread(
            target=self._force_stop_after_grace,
            args=(proc, process_group, info.tmux_session if info else None),
            daemon=True,
        ).start()
        return True

    def start_tensorboard(
        self,
        host: str = "127.0.0.1",
        port: int | None = None,
        logdir: Path | None = None,
        source_run_id: str | None = None,
    ) -> dict:
        target_logdir = logdir or (self.paths.repo_root / "logs" / "rsl_rl")
        target_key = str(target_logdir.resolve() if target_logdir.exists() else target_logdir)
        with self._lock:
            existing_id = self._tensorboards_by_logdir.get(target_key)
            existing = self._processes.get(existing_id or "")
            if existing_id and existing and existing.poll() is None:
                existing_info = self._infos[existing_id]
                existing_port = int(existing_info.run_id.rsplit("_", 1)[-1])
                return self._tensorboard_response(
                    host,
                    existing_port,
                    existing.pid,
                    existing_id,
                    existing_info.log_file,
                    existing_info.command,
                    already_running=True,
                    attach_command=existing_info.attach_command,
                    tmux_session=existing_info.tmux_session,
                )
        selected_port = port or self._find_free_port(6006)
        run_id = f"tensorboard_{selected_port}"
        argv = tensorboard_argv(target_logdir, host, selected_port)
        shell = shell_for_command(self.paths, argv)
        log_file = self.paths.process_log_dir / f"{run_id}.log"
        spawned = self._spawn_shell(run_id, shell, log_file)
        proc = spawned.proc
        self._register(
            run_id,
            "tensorboard",
            spawned,
            log_file,
            datetime.now().isoformat(timespec="seconds"),
            shell,
            source_run_id=source_run_id,
        )
        with self._lock:
            self._tensorboards_by_logdir[target_key] = run_id
        self._raise_if_immediate_exit(proc, run_id, "TensorBoard", wait_seconds=2.0)
        return self._tensorboard_response(
            host,
            selected_port,
            proc.pid,
            run_id,
            str(log_file),
            shell,
            already_running=False,
            attach_command=spawned.attach_command,
            tmux_session=spawned.tmux_session,
        )

    def start_play(self, run_id: str, checkpoint: str, device: str = "cuda:0") -> dict:
        self.paths.ensure_dirs()
        play_id = f"play_{timestamp_id()}"
        run = self.history.get_run(run_id)
        spring_backend = resolve_spring_backend(run, checkpoint)
        task = str(((run or {}).get("params") or {}).get("task") or DEFAULT_TASK)
        terrain_override_file = self._write_process_terrain_override(play_id, run_id, checkpoint)
        argv = play_argv(
            checkpoint=checkpoint,
            task=task,
            device=device,
            spring_backend=spring_backend,
            terrain_override_file=terrain_override_file,
            camera_follow_robot=True,
            camera_eye=DEFAULT_FOLLOW_CAMERA_EYE,
            camera_lookat=DEFAULT_FOLLOW_CAMERA_LOOKAT,
        )
        shell = shell_for_isaaclab(self.paths, argv)
        log_file = self.paths.process_log_dir / f"{play_id}.log"
        spawned = self._spawn_shell(play_id, shell, log_file)
        proc = spawned.proc
        self._register(
            play_id,
            "play",
            spawned,
            log_file,
            datetime.now().isoformat(timespec="seconds"),
            shell,
            source_run_id=run_id,
        )
        self._raise_if_immediate_exit(proc, play_id, "Play", wait_seconds=1.0)
        return {
            "id": play_id,
            "source_run_id": run_id,
            "pid": proc.pid,
            "process_log": str(log_file),
            "command": shell,
            "tmux_session": spawned.tmux_session,
            "attach_command": spawned.attach_command,
            "exit_file": spawned.exit_file,
        }

    def start_video_recording(
        self,
        run_id: str,
        checkpoint: str,
        device: str = "cuda:0",
        video_params: VideoParams | None = None,
    ) -> dict:
        self.paths.ensure_dirs()
        params = video_params or VideoParams.from_preset(DEFAULT_VIDEO_PRESET)
        params.validate()
        video_id = f"video_{timestamp_id()}"
        run = self.history.get_run(run_id)
        spring_backend = resolve_spring_backend(run, checkpoint)
        task = str(((run or {}).get("params") or {}).get("task") or DEFAULT_TASK)
        terrain_override_file = self._write_process_terrain_override(video_id, run_id, checkpoint)
        argv = play_argv(
            checkpoint=checkpoint,
            task=task,
            device=device,
            spring_backend=spring_backend,
            num_envs=1,
            headless=True,
            video=True,
            video_length=params.length,
            video_width=params.width,
            video_height=params.height,
            video_fps=params.fps,
            rendering_mode=params.rendering_mode,
            terrain_override_file=terrain_override_file,
            camera_follow_robot=True,
            camera_eye=DEFAULT_FOLLOW_CAMERA_EYE,
            camera_lookat=DEFAULT_FOLLOW_CAMERA_LOOKAT,
        )
        shell = shell_for_isaaclab(self.paths, argv)
        log_file = self.paths.process_log_dir / f"{video_id}.log"
        spawned = self._spawn_shell(video_id, shell, log_file)
        proc = spawned.proc
        self._register(
            video_id,
            "video",
            spawned,
            log_file,
            datetime.now().isoformat(timespec="seconds"),
            shell,
            source_run_id=run_id,
        )
        self.history.update_run(
            run_id,
            video_status="recording",
            video_process_id=video_id,
            video_pid=proc.pid,
            video_process_log=str(log_file),
            video_command=shell,
            video_process_attach_command=spawned.attach_command,
            video_tmux_session=spawned.tmux_session,
            video_exit_file=spawned.exit_file,
            video_preset=params.preset,
            video_params=params.to_dict(),
            video_length=params.length,
            video_checkpoint=str(checkpoint),
            video_checkpoint_iteration=self._checkpoint_iteration(checkpoint),
        )
        thread = threading.Thread(
            target=self._monitor_video,
            args=(run_id, video_id, proc),
            daemon=True,
        )
        thread.start()
        return {
            "id": video_id,
            "source_run_id": run_id,
            "pid": proc.pid,
            "process_log": str(log_file),
            "command": shell,
            "tmux_session": spawned.tmux_session,
            "attach_command": spawned.attach_command,
            "exit_file": spawned.exit_file,
            "video_params": params.to_dict(),
            "checkpoint": str(checkpoint),
            "checkpoint_iteration": self._checkpoint_iteration(checkpoint),
        }

    def start_onnx_export(self, run_id: str, checkpoint: str, device: str = "cuda:0") -> dict:
        onnx_id = f"onnx_{timestamp_id()}"
        run = self.history.get_run(run_id)
        spring_backend = resolve_spring_backend(run, checkpoint)
        task = str(((run or {}).get("params") or {}).get("task") or DEFAULT_TASK)
        argv = export_onnx_argv(checkpoint=checkpoint, task=task, device=device, spring_backend=spring_backend)
        shell = shell_for_isaaclab(self.paths, argv)
        log_file = self.paths.process_log_dir / f"{onnx_id}.log"
        spawned = self._spawn_shell(onnx_id, shell, log_file)
        proc = spawned.proc
        self._register(
            onnx_id,
            "onnx",
            spawned,
            log_file,
            datetime.now().isoformat(timespec="seconds"),
            shell,
            source_run_id=run_id,
        )
        self.history.update_run(
            run_id,
            onnx_status="exporting",
            onnx_process_id=onnx_id,
            onnx_pid=proc.pid,
            onnx_process_log=str(log_file),
            onnx_command=shell,
            onnx_process_attach_command=spawned.attach_command,
            onnx_tmux_session=spawned.tmux_session,
            onnx_exit_file=spawned.exit_file,
            onnx_error=None,
        )
        thread = threading.Thread(
            target=self._monitor_onnx,
            args=(run_id, onnx_id, proc),
            daemon=True,
        )
        thread.start()
        return {
            "id": onnx_id,
            "source_run_id": run_id,
            "pid": proc.pid,
            "process_log": str(log_file),
            "command": shell,
            "tmux_session": spawned.tmux_session,
            "attach_command": spawned.attach_command,
            "exit_file": spawned.exit_file,
        }

    def start_deploy_validation(
        self,
        run_id: str,
        *,
        export_first: bool = False,
        device: str = "cuda:0",
        include_ros_mock: bool = False,
        include_mujoco: bool = True,
        use_cuda: bool = False,
        use_tensorrt: bool = False,
        mujoco_model_path: str | None = None,
        mujoco_only: bool = False,
    ) -> dict:
        deploy_id = f"deploy_{timestamp_id()}"
        argv = [
            sys.executable,
            "-m",
            "tools.training_panel.deploy_pipeline",
            "--run-id",
            run_id,
            "--pipeline-id",
            deploy_id,
            "--device",
            device,
        ]
        if export_first:
            argv.append("--export-first")
        if include_ros_mock:
            argv.append("--include-ros-mock")
        if not include_mujoco:
            argv.append("--no-mujoco")
        if use_cuda:
            argv.append("--use-cuda")
        if use_tensorrt:
            argv.append("--use-tensorrt")
        if mujoco_only:
            argv.append("--mujoco-only")
        if mujoco_model_path:
            argv.extend(["--mujoco-model-path", str(mujoco_model_path)])
        shell = "\n".join(
            [
                f"cd {shlex.quote(str(self.paths.repo_root))}",
                "exec " + " ".join(shlex.quote(arg) for arg in argv),
            ]
        )
        log_file = self.paths.process_log_dir / f"{deploy_id}.log"
        spawned = self._spawn_shell(deploy_id, shell, log_file)
        proc = spawned.proc
        self._register(
            deploy_id,
            "deploy",
            spawned,
            log_file,
            datetime.now().isoformat(timespec="seconds"),
            shell,
            source_run_id=run_id,
        )
        self.history.update_run(
            run_id,
            deploy_status="running",
            deploy_process_id=deploy_id,
            deploy_pid=proc.pid,
            deploy_process_log=str(log_file),
            deploy_command=shell,
            deploy_process_attach_command=spawned.attach_command,
            deploy_tmux_session=spawned.tmux_session,
            deploy_exit_file=spawned.exit_file,
            deploy_error=None,
            deploy_options={
                "export_first": bool(export_first),
                "device": device,
                "include_ros_mock": bool(include_ros_mock),
                "include_mujoco": bool(include_mujoco),
                "use_cuda": bool(use_cuda),
                "use_tensorrt": bool(use_tensorrt),
                "mujoco_model_path": str(mujoco_model_path or ""),
                "mujoco_only": bool(mujoco_only),
            },
        )
        thread = threading.Thread(
            target=self._monitor_deploy,
            args=(run_id, deploy_id, proc),
            daemon=True,
        )
        thread.start()
        return {
            "id": deploy_id,
            "source_run_id": run_id,
            "pid": proc.pid,
            "process_log": str(log_file),
            "command": shell,
            "tmux_session": spawned.tmux_session,
            "attach_command": spawned.attach_command,
            "exit_file": spawned.exit_file,
        }

    def start_mujoco_playback(
        self,
        run_id: str,
        *,
        mode: str,
        scenario: str = "stand_zero",
        steps: int = 1250,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        mujoco_model_path: str | None = None,
    ) -> dict:
        if self.running_for_run(run_id, "mujoco"):
            raise ProcessStartError(
                "A MuJoCo playback process is already running for this run.",
                {"processes": self.running_for_run(run_id, "mujoco")},
            )
        mode = str(mode or "").lower()
        if mode not in {"viewer", "record"}:
            raise ValueError("MuJoCo playback mode must be viewer or record")
        process_id = f"mujoco_{mode}_{timestamp_id()}"
        argv = [
            sys.executable,
            "-m",
            "tools.training_panel.mujoco_playback",
            "--run-id",
            run_id,
            "--process-id",
            process_id,
            "--mode",
            mode,
            "--scenario",
            scenario,
            "--steps",
            str(max(1, int(steps))),
            "--width",
            str(max(320, int(width))),
            "--height",
            str(max(240, int(height))),
            "--fps",
            str(max(1, int(fps))),
        ]
        if mujoco_model_path:
            argv.extend(["--mujoco-model-path", str(mujoco_model_path)])
        shell = "\n".join(
            [
                f"cd {shlex.quote(str(self.paths.repo_root))}",
                "exec " + " ".join(shlex.quote(arg) for arg in argv),
            ]
        )
        log_file = self.paths.process_log_dir / f"{process_id}.log"
        spawned = self._spawn_shell(process_id, shell, log_file)
        proc = spawned.proc
        self._register(
            process_id,
            "mujoco",
            spawned,
            log_file,
            datetime.now().isoformat(timespec="seconds"),
            shell,
            source_run_id=run_id,
        )
        self.history.update_run(
            run_id,
            mujoco_playback_status="running",
            mujoco_playback_mode=mode,
            mujoco_playback_scenario=scenario,
            mujoco_process_id=process_id,
            mujoco_pid=proc.pid,
            mujoco_process_log=str(log_file),
            mujoco_command=shell,
            mujoco_process_attach_command=spawned.attach_command,
            mujoco_tmux_session=spawned.tmux_session,
            mujoco_exit_file=spawned.exit_file,
            mujoco_error=None,
        )
        thread = threading.Thread(
            target=self._monitor_mujoco,
            args=(run_id, process_id, proc),
            daemon=True,
        )
        thread.start()
        return {
            "id": process_id,
            "source_run_id": run_id,
            "pid": proc.pid,
            "process_log": str(log_file),
            "command": shell,
            "tmux_session": spawned.tmux_session,
            "attach_command": spawned.attach_command,
            "exit_file": spawned.exit_file,
        }

    def get_process_debug(self, process_id: str) -> dict | None:
        with self._lock:
            info = self._infos.get(process_id)
            proc = self._processes.get(process_id)
        if not info:
            external = next(
                (
                    process
                    for process in self._external_processes()
                    if process.get("run_id") == process_id
                ),
                None,
            )
            if not external:
                return None
            return {
                **external,
                "log_tail": tail_file(Path(external["log_file"])) if external.get("log_file") else "",
            }
        return {
            **info.__dict__,
            "returncode": self._returncode(info, proc) if proc else None,
            "log_tail": tail_file(Path(info.log_file)),
        }

    def running_for_run(self, source_run_id: str, kind: str | None = None) -> list[dict]:
        return [
            process
            for process in self.list_processes()
            if process.get("returncode") is None
            and (process.get("run_id") == source_run_id or process.get("source_run_id") == source_run_id)
            and (kind is None or process.get("kind") == kind)
        ]

    def running_for_log_dir(self, log_dir: str | Path) -> list[dict]:
        target = Path(log_dir).resolve()
        try:
            target_mtime = target.stat().st_mtime
        except OSError:
            target_mtime = 0.0
        running = []
        for process in self.list_processes():
            if process.get("returncode") is not None:
                continue
            source_run_id = process.get("source_run_id") or process.get("run_id")
            run = self.history.get_run(str(source_run_id)) if source_run_id else None
            run_log_dir = Path(str(run.get("log_dir"))).resolve() if run and run.get("log_dir") else None
            if run_log_dir == target:
                running.append(process)
                continue
            if process.get("kind") == "training" and self._training_process_may_own_log_dir(process, target, target_mtime):
                running.append(process)
        return running

    def _training_process_may_own_log_dir(self, process: dict, log_dir: Path, log_dir_mtime: float) -> bool:
        if not log_dir_mtime:
            return False
        run_id = str(process.get("source_run_id") or process.get("run_id") or "")
        run = self.history.get_run(run_id) if run_id else None
        started_at = self._process_started_at_epoch(process, run)
        if not started_at or log_dir_mtime < started_at:
            return False
        process_log = Path(str(process.get("log_file") or ""))
        if process_log.exists():
            text = self._head_file(process_log, max_chars=80000) + "\n" + tail_file(process_log, max_chars=200000)
            if str(log_dir) in text or log_dir.name in text:
                return True
            exact_names = re.findall(r"Exact experiment name requested from command line:\s*(\S+)", text)
            if any(log_dir.name == name or log_dir.name.startswith(f"{name}_") for name in exact_names):
                return True
        return bool(run and run.get("source") == "training_panel" and not run.get("log_dir"))

    def _process_started_at_epoch(self, process: dict, run: dict | None = None) -> float:
        for value in (process.get("started_at"), (run or {}).get("created_at")):
            if not value:
                continue
            try:
                return datetime.fromisoformat(str(value)).timestamp()
            except ValueError:
                continue
        return 0.0

    def running_media_processes(self) -> list[dict]:
        return self.running_isaac_processes()

    def running_isaac_processes(self) -> list[dict]:
        return [
            process
            for process in self.list_processes()
            if process.get("returncode") is None and process.get("kind") in GPU_PROCESS_KINDS
        ]

    def cuda_health(self) -> dict:
        return self._cuda_health()

    def _assert_cuda_ready(self, device: str) -> None:
        if not self.cuda_preflight or not str(device or "").startswith("cuda"):
            return
        health = self.cuda_health()
        if not health.get("ok"):
            raise CudaPreflightError(health)

    @staticmethod
    def _cuda_health() -> dict:
        kernel_version = ProcessRegistry._loaded_nvidia_kernel_version()
        userspace_version = ProcessRegistry._nvidia_userspace_version()
        reboot_required = Path("/var/run/reboot-required").exists()
        remediation = (
            "Reboot this machine so the installed NVIDIA userspace driver and loaded kernel module match. "
            "If it still fails after reboot, reinstall the NVIDIA 580 driver packages so libcuda/NVML and the kernel module are the same version."
        )
        base = {
            "ok": True,
            "kernel_driver_version": kernel_version,
            "userspace_driver_version": userspace_version,
            "reboot_required": reboot_required,
            "remediation": remediation,
        }
        if kernel_version and userspace_version and kernel_version != userspace_version:
            return {
                **base,
                "ok": False,
                "reason": "driver_library_mismatch",
                "error": (
                    f"CUDA driver preflight failed: loaded NVIDIA kernel module is {kernel_version}, "
                    f"but libcuda/NVML userspace is {userspace_version}."
                ),
            }
        try:
            result = subprocess.run(
                ["nvidia-smi", "-L"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                timeout=8,
                check=False,
            )
        except FileNotFoundError:
            return {
                **base,
                "ok": False,
                "reason": "nvidia_smi_missing",
                "error": "CUDA driver preflight failed: nvidia-smi is not installed or not on PATH.",
            }
        except (OSError, subprocess.SubprocessError) as exc:
            return {
                **base,
                "ok": False,
                "reason": "nvidia_smi_error",
                "error": f"CUDA driver preflight failed while running nvidia-smi: {exc}",
            }
        output = f"{result.stdout}\n{result.stderr}".strip()
        if result.returncode != 0:
            return {
                **base,
                "ok": False,
                "reason": "nvidia_smi_failed",
                "error": f"CUDA driver preflight failed: nvidia-smi exited {result.returncode}.",
                "nvidia_smi_output": output,
            }
        return {**base, "nvidia_smi_output": output}

    @staticmethod
    def _loaded_nvidia_kernel_version() -> str | None:
        try:
            text = Path("/proc/driver/nvidia/version").read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        match = re.search(r"\b(\d+\.\d+\.\d+)\b", text)
        return match.group(1) if match else None

    @staticmethod
    def _nvidia_userspace_version() -> str | None:
        for name in ("libcuda.so.1", "libnvidia-ml.so.1"):
            path = Path("/usr/lib/x86_64-linux-gnu") / name
            try:
                resolved = path.resolve(strict=True)
            except OSError:
                continue
            match = re.search(r"\b(\d+\.\d+\.\d+)\b", resolved.name)
            if match:
                return match.group(1)
        return None

    def stop_all_for_run(self, source_run_id: str) -> list[str]:
        stopped = []
        for process in self.running_for_run(source_run_id):
            process_id = process.get("run_id", "")
            if process_id and self.stop(process_id):
                stopped.append(process_id)
        return stopped

    def reconcile_stale_history(self) -> None:
        processes = self.list_processes()
        self._repair_active_training_history(processes)
        self._repair_panel_log_history()
        known_process_runs = set()
        for process in processes:
            if process.get("kind") != "training":
                continue
            if process.get("run_id"):
                known_process_runs.add(process["run_id"])
            if process.get("source_run_id"):
                known_process_runs.add(process["source_run_id"])
        for run in self.history.list_runs():
            if run.get("source") != "training_panel":
                continue
            if run.get("status") not in ("running", "stopping"):
                continue
            if run.get("log_dir"):
                self.history.update_run(run["id"], log_dir=run["log_dir"])
            else:
                log_dir = self._log_dir_from_process_log(run["id"])
                if log_dir:
                    self.history.update_run(run["id"], log_dir=log_dir)
                    run = {**run, "log_dir": log_dir}
            exit_code = self._exit_code_from_history(run)
            if exit_code is not None:
                log_dir = run.get("log_dir") or self._completed_log_for_run(
                    run,
                    allow_time_fallback=exit_code == 0,
                )
                if log_dir:
                    status = "completed" if exit_code == 0 else "failed"
                    self.history.link_run_to_log(run["id"], log_dir, status=status, returncode=exit_code)
                    continue
            if run.get("id") not in known_process_runs:
                self.history.update_run(run["id"], status="interrupted")
        self.start_next_queued_training()

    def _repair_active_training_history(self, processes: list[dict]) -> None:
        """Keep history records useful even if metadata sync left a live run as a stub."""
        for process in processes:
            if process.get("kind") != "training" or process.get("returncode") is not None:
                continue
            run_id = str(process.get("source_run_id") or process.get("run_id") or "")
            if not run_id or run_id.startswith(EXTERNAL_ID_PREFIXES):
                continue
            current = self.history.get_run(run_id) or {}
            status = "stopping" if str(current.get("status") or "").lower() == "stopping" else "running"
            updates = {
                "source": "training_panel",
                "status": status,
                "pid": process.get("pid"),
                "process_group": process.get("process_group"),
                "gpu_pid": process.get("gpu_pid"),
                "gpu_pids": process.get("gpu_pids"),
                "process_log": process.get("log_file"),
                "command": process.get("command"),
                "started_at": process.get("started_at"),
                "created_at": current.get("created_at") or process.get("started_at"),
                "tmux_session": process.get("tmux_session"),
                "attach_command": process.get("attach_command"),
                "exit_file": process.get("exit_file"),
            }
            self.history.patch_run_metadata(
                run_id,
                **{key: value for key, value in updates.items() if value not in (None, "", [], {})},
            )

    def _repair_panel_log_history(self) -> None:
        """Recover panel records that were reduced to metadata-only stubs."""
        for record in self.history._load_records():
            run_id = str(record.get("id") or "")
            if not run_id.startswith("panel_") or record.get("source") != "training_panel":
                continue
            process_log = Path(str(record.get("process_log") or self.paths.process_log_dir / f"{run_id}.log"))
            if not process_log.is_file():
                continue
            updates: dict[str, object] = {}
            if not record.get("process_log"):
                updates["process_log"] = str(process_log)
            exit_file = Path(str(record.get("exit_file") or self.paths.process_log_dir / f"{run_id}.exit"))
            exit_code = self._exit_code_from_path(exit_file)
            if exit_code is not None and not record.get("exit_file"):
                updates["exit_file"] = str(exit_file)
            created_at = self._panel_created_at_from_run_id(run_id)
            if created_at and record.get("created_at") != created_at:
                updates["created_at"] = created_at
            if created_at and not record.get("started_at"):
                updates["started_at"] = created_at
            completed_at = self._completed_at_from_exit_file(exit_file) if exit_code is not None else None
            if completed_at and record.get("completed_at") != completed_at:
                updates["completed_at"] = completed_at
            if not record.get("command"):
                command = self._command_from_process_log(process_log)
                if command:
                    updates["command"] = command
            if not record.get("log_dir"):
                log_dir = self._log_dir_from_process_log_path(process_log)
                if log_dir:
                    updates["log_dir"] = log_dir
            if exit_code is not None and str(record.get("status") or "").lower() not in {"running", "stopping"}:
                desired_status = "completed" if exit_code == 0 else "failed"
                if record.get("status") != desired_status:
                    updates["status"] = desired_status
                if record.get("returncode") != exit_code:
                    updates["returncode"] = exit_code
            if updates:
                self.history.patch_run_metadata(run_id, **updates)

    @staticmethod
    def _panel_created_at_from_run_id(run_id: str) -> str | None:
        match = re.match(r"^panel_(\d{8})_(\d{6})", str(run_id or ""))
        if not match:
            return None
        try:
            return datetime.strptime(f"{match.group(1)}_{match.group(2)}", "%Y%m%d_%H%M%S").isoformat(
                timespec="seconds"
            )
        except ValueError:
            return None

    @staticmethod
    def _completed_at_from_exit_file(exit_file: Path) -> str | None:
        try:
            return datetime.fromtimestamp(exit_file.stat().st_mtime).isoformat(timespec="seconds")
        except OSError:
            return None

    def _exit_code_from_history(self, run: dict) -> int | None:
        exit_file = run.get("exit_file")
        if not exit_file:
            return None
        return self._exit_code_from_path(Path(str(exit_file)))

    @staticmethod
    def _exit_code_from_path(path: Path) -> int | None:
        try:
            return int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return None

    def _completed_log_for_run(self, run: dict, *, allow_time_fallback: bool = True) -> str | None:
        process_log = Path(str(run.get("process_log") or ""))
        if process_log.exists():
            text = self._head_file(process_log, max_chars=120000) + "\n" + tail_file(process_log, max_chars=120000)
            log_dir, has_exact_name = self._log_dir_from_process_log_text(text)
            if log_dir:
                return log_dir
            if has_exact_name:
                return None
        if not allow_time_fallback:
            return None
        created_at = run.get("created_at")
        if created_at:
            try:
                return self.history.find_latest_log_after(datetime.fromisoformat(str(created_at)).timestamp())
            except ValueError:
                return None
        return None

    @staticmethod
    def _head_file(path: Path, max_chars: int = 120000) -> str:
        if not path.exists() or not path.is_file():
            return ""
        with path.open("rb") as file:
            return file.read(max_chars).decode("utf-8", errors="replace")

    def _external_processes(self) -> list[dict]:
        processes = [
            *self._external_training_processes(),
            *self._external_onnx_processes(),
            *self._external_video_processes(),
            *self._external_play_processes(),
            *self._external_gpu_python_processes(),
            *self._external_tensorboard_processes(),
        ]
        for process in processes:
            if process.get("kind") not in GPU_PROCESS_KINDS:
                continue
            process_group = process.get("process_group")
            if not isinstance(process_group, int):
                continue
            gpu_pids = self._gpu_pids_for_group(process_group)
            if gpu_pids:
                process["gpu_pids"] = gpu_pids
                process["gpu_pid"] = gpu_pids[0]
        return processes

    @staticmethod
    def _matches_known_process_log(process: dict, known_log_files: set[str]) -> bool:
        if not known_log_files:
            return False
        log_file = str(process.get("log_file") or "")
        command = str(process.get("command") or "")
        return any(path and (path == log_file or path in command) for path in known_log_files)

    def _external_training_processes(self) -> list[dict]:
        try:
            output = subprocess.check_output(
                ["ps", "-eo", "pid=,pgid=,stat=,args="],
                text=True,
                errors="replace",
            )
        except (OSError, subprocess.SubprocessError):
            return []
        by_group = {}
        for line in output.splitlines():
            parts = line.strip().split(maxsplit=3)
            if len(parts) < 4:
                continue
            pid_text, pgid_text, stat, command = parts
            if not self._is_repo_training_process(command, pid_text):
                continue
            try:
                pid = int(pid_text)
                process_group = int(pgid_text)
            except ValueError:
                continue
            source_run_id = self._source_run_id_from_training_process(pid, process_group, command)
            existing = by_group.get(process_group)
            if existing and existing["pid"] == process_group:
                continue
            log_file = self._matching_training_log(pid, process_group, command)
            tmux_session = self._tmux_session_for_process_group(process_group)
            by_group[process_group] = {
                "kind": "training",
                "pid": pid,
                "process_group": process_group,
                "run_id": f"{EXTERNAL_TRAINING_ID_PREFIX}{process_group}",
                "source_run_id": source_run_id,
                "log_file": str(log_file) if log_file else "",
                "started_at": "",
                "command": command,
                "returncode": None,
                "external": True,
                "stat": stat,
                "tmux_session": tmux_session,
                "attach_command": f"tmux attach -t {shlex.quote(tmux_session)}" if tmux_session else None,
            }
        return list(by_group.values())

    def _external_gpu_python_processes(self) -> list[dict]:
        pids = self._gpu_device_pids()
        if not pids:
            return []
        try:
            output = subprocess.check_output(
                ["ps", "-o", "pid=,pgid=,stat=,args=", "-p", ",".join(str(pid) for pid in sorted(pids))],
                text=True,
                errors="replace",
            )
        except (OSError, subprocess.SubprocessError):
            return []
        by_group = {}
        for line in output.splitlines():
            parts = line.strip().split(maxsplit=3)
            if len(parts) < 4:
                continue
            pid_text, pgid_text, stat, command = parts
            try:
                pid = int(pid_text)
                process_group = int(pgid_text)
            except ValueError:
                continue
            if not self._is_external_gpu_python_process(command, pid_text):
                continue
            existing = by_group.get(process_group)
            if existing and existing["pid"] == process_group:
                continue
            tmux_session = self._tmux_session_for_process_group(process_group)
            by_group[process_group] = {
                "kind": "gpu",
                "pid": pid,
                "process_group": process_group,
                "run_id": f"{EXTERNAL_GPU_ID_PREFIX}{process_group}",
                "source_run_id": None,
                "log_file": "",
                "started_at": "",
                "command": command,
                "returncode": None,
                "external": True,
                "stat": stat,
                "tmux_session": tmux_session,
                "attach_command": f"tmux attach -t {shlex.quote(tmux_session)}" if tmux_session else None,
            }
        return list(by_group.values())

    def _external_play_processes(self) -> list[dict]:
        try:
            output = subprocess.check_output(
                ["ps", "-eo", "pid=,pgid=,stat=,args="],
                text=True,
                errors="replace",
            )
        except (OSError, subprocess.SubprocessError):
            return []
        by_group = {}
        for line in output.splitlines():
            parts = line.strip().split(maxsplit=3)
            if len(parts) < 4:
                continue
            pid_text, pgid_text, stat, command = parts
            if not self._is_repo_play_process(command, pid_text):
                continue
            try:
                pid = int(pid_text)
                process_group = int(pgid_text)
            except ValueError:
                continue
            source_run_id = self._source_run_id_from_command(command)
            if not source_run_id:
                continue
            existing = by_group.get(process_group)
            if existing and existing["pid"] == process_group:
                continue
            log_file = self._matching_process_log(command)
            by_group[process_group] = {
                "kind": "play",
                "pid": pid,
                "process_group": process_group,
                "run_id": f"{EXTERNAL_PLAY_ID_PREFIX}{process_group}",
                "source_run_id": source_run_id,
                "log_file": str(log_file) if log_file else "",
                "started_at": "",
                "command": command,
                "returncode": None,
                "external": True,
                "stat": stat,
            }
        return list(by_group.values())

    def _external_video_processes(self) -> list[dict]:
        try:
            output = subprocess.check_output(
                ["ps", "-eo", "pid=,pgid=,stat=,args="],
                text=True,
                errors="replace",
            )
        except (OSError, subprocess.SubprocessError):
            return []
        by_group = {}
        for line in output.splitlines():
            parts = line.strip().split(maxsplit=3)
            if len(parts) < 4:
                continue
            pid_text, pgid_text, stat, command = parts
            if not self._is_repo_video_process(command, pid_text):
                continue
            try:
                pid = int(pid_text)
                process_group = int(pgid_text)
            except ValueError:
                continue
            source_run_id = self._source_run_id_from_command(command)
            if not source_run_id:
                continue
            existing = by_group.get(process_group)
            if existing and existing["pid"] == process_group:
                continue
            log_file = self._matching_process_log(command)
            by_group[process_group] = {
                "kind": "video",
                "pid": pid,
                "process_group": process_group,
                "run_id": f"{EXTERNAL_VIDEO_ID_PREFIX}{process_group}",
                "source_run_id": source_run_id,
                "log_file": str(log_file) if log_file else "",
                "started_at": "",
                "command": command,
                "returncode": None,
                "external": True,
                "stat": stat,
            }
        return list(by_group.values())

    def _external_onnx_processes(self) -> list[dict]:
        try:
            output = subprocess.check_output(
                ["ps", "-eo", "pid=,pgid=,stat=,args="],
                text=True,
                errors="replace",
            )
        except (OSError, subprocess.SubprocessError):
            return []
        by_group = {}
        for line in output.splitlines():
            parts = line.strip().split(maxsplit=3)
            if len(parts) < 4:
                continue
            pid_text, pgid_text, stat, command = parts
            if not self._is_repo_onnx_process(command, pid_text):
                continue
            try:
                pid = int(pid_text)
                process_group = int(pgid_text)
            except ValueError:
                continue
            source_run_id = self._source_run_id_from_command(command)
            if not source_run_id:
                continue
            log_file = self._matching_onnx_log(command)
            by_group[process_group] = {
                "kind": "onnx",
                "pid": pid,
                "process_group": process_group,
                "run_id": f"{EXTERNAL_ONNX_ID_PREFIX}{process_group}",
                "source_run_id": source_run_id,
                "log_file": str(log_file) if log_file else "",
                "started_at": "",
                "command": command,
                "returncode": None,
                "external": True,
                "stat": stat,
            }
        return list(by_group.values())

    def _external_tensorboard_processes(self) -> list[dict]:
        try:
            output = subprocess.check_output(
                ["ps", "-eo", "pid=,pgid=,stat=,args="],
                text=True,
                errors="replace",
            )
        except (OSError, subprocess.SubprocessError):
            return []
        by_group = {}
        for line in output.splitlines():
            parts = line.strip().split(maxsplit=3)
            if len(parts) < 4:
                continue
            pid_text, pgid_text, stat, command = parts
            if "tensorboard_data_server" in command:
                continue
            if self._is_tmux_server_command(command):
                continue
            if not self._is_repo_tensorboard_process(command):
                continue
            try:
                pid = int(pid_text)
                process_group = int(pgid_text)
            except ValueError:
                continue
            source_run_id = self._source_run_id_from_tensorboard_command(command)
            if not source_run_id:
                continue
            log_file = self._matching_tensorboard_log(command)
            by_group[process_group] = {
                "kind": "tensorboard",
                "pid": pid,
                "process_group": process_group,
                "run_id": f"{EXTERNAL_TENSORBOARD_ID_PREFIX}{process_group}",
                "source_run_id": source_run_id,
                "log_file": str(log_file) if log_file else "",
                "started_at": "",
                "command": command,
                "returncode": None,
                "external": True,
                "stat": stat,
            }
        return list(by_group.values())

    def _source_run_id_from_command(self, command: str) -> str | None:
        match = re.search(r"logs/rsl_rl/redrhex_wheg/([^/\s]+)/model_\d+\.pt", command)
        return match.group(1) if match else None

    def _source_run_id_from_tensorboard_command(self, command: str) -> str | None:
        match = re.search(r"logs/rsl_rl/redrhex_wheg/([^/\s]+)(?:\s|$)", command)
        return match.group(1) if match else None

    def _source_run_id_from_training_process(self, pid: int, process_group: int, command: str) -> str | None:
        run_id = self._source_run_id_from_training_process_log(command)
        if run_id:
            return run_id
        for run in self.history.list_runs():
            if run.get("source") != "training_panel":
                continue
            if run.get("pid") in (pid, process_group):
                return run.get("id")
        matches = []
        for run in self.history.list_runs():
            if run.get("source") != "training_panel":
                continue
            recorded_command = run.get("command") or ""
            if recorded_command and self._training_commands_match(recorded_command, command):
                matches.append(run.get("id"))
        return matches[0] if len(matches) == 1 else None

    def _matching_process_log(self, command: str) -> Path | None:
        checkpoint_match = re.search(r"--checkpoint\s+(\S+)", command)
        checkpoint = checkpoint_match.group(1) if checkpoint_match else ""
        candidates = []
        for path in self.paths.process_log_dir.glob("play_*.log"):
            text = tail_file(path, max_chars=20000)
            if checkpoint and checkpoint in text:
                candidates.append(path)
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def _matching_onnx_log(self, command: str) -> Path | None:
        checkpoint_match = re.search(r"--checkpoint\s+(\S+)", command)
        checkpoint = checkpoint_match.group(1) if checkpoint_match else ""
        candidates = []
        for path in self.paths.process_log_dir.glob("onnx_*.log"):
            text = tail_file(path, max_chars=20000)
            if checkpoint and checkpoint in text:
                candidates.append(path)
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def _matching_tensorboard_log(self, command: str) -> Path | None:
        logdir_match = re.search(r"--logdir\s+(\S+)", command)
        logdir = logdir_match.group(1) if logdir_match else ""
        candidates = []
        for path in self.paths.process_log_dir.glob("tensorboard_*.log"):
            text = tail_file(path, max_chars=20000)
            if logdir and logdir in text:
                candidates.append(path)
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def _matching_training_log(self, pid: int, process_group: int, command: str) -> Path | None:
        process_log = self._training_process_log_from_command(command)
        if process_log:
            return process_log
        for run in self.history.list_runs():
            if run.get("source") != "training_panel":
                continue
            if run.get("pid") in (pid, process_group):
                process_log = run.get("process_log")
                if process_log:
                    return Path(process_log)
        matches = []
        for run in self.history.list_runs():
            if run.get("source") != "training_panel":
                continue
            if self._training_commands_match(run.get("command") or "", command):
                process_log = run.get("process_log")
                if process_log:
                    matches.append(Path(process_log))
        return matches[0] if len(matches) == 1 else None

    def _source_run_id_from_training_process_log(self, command: str) -> str | None:
        process_log = self._training_process_log_from_command(command)
        if not process_log:
            return None
        log_name = process_log.name
        for run in self.history.list_runs():
            if run.get("source") != "training_panel":
                continue
            run_log = run.get("process_log")
            if run_log and Path(str(run_log)).name == log_name:
                return run.get("id")
        stem = process_log.stem
        run = self.history.get_run(stem)
        if run and run.get("source") == "training_panel":
            return stem
        return None

    def _training_process_log_from_command(self, command: str) -> Path | None:
        pattern = r"(/[^\s'\"<>]*?/logs/training_panel/process_logs/panel_[A-Za-z0-9_]+\.log)"
        matches = re.findall(pattern, command)
        if not matches:
            return None
        return Path(matches[-1])

    def _command_from_process_log(self, process_log: Path) -> str | None:
        text = self._head_file(process_log, max_chars=120000)
        match = re.search(r"\$ bash -lc <<'PANEL_COMMAND'\n(.*?)\nPANEL_COMMAND", text, re.DOTALL)
        if not match:
            return None
        command = match.group(1).strip()
        return command or None

    def _log_dir_from_process_log(self, run_id: str) -> str | None:
        run = self.history.get_run(run_id) or {}
        process_log = Path(str(run.get("process_log") or ""))
        if not process_log.exists():
            return None
        return self._log_dir_from_process_log_path(process_log)

    def _log_dir_from_process_log_path(self, process_log: Path) -> str | None:
        text = self._head_file(process_log, max_chars=120000) + "\n" + tail_file(process_log, max_chars=300000)
        log_dir, _ = self._log_dir_from_process_log_text(text)
        return log_dir

    def _log_dir_from_process_log_text(self, text: str) -> tuple[str | None, bool]:
        experiment_roots = list(
            re.finditer(r"\[INFO\][ \t]+Logging experiment in directory:[ \t]*([^\r\n]+)", text)
        )
        exact_names = list(re.finditer(r"Exact experiment name requested from command line:\s*(\S+)", text))
        rsl_rl_root = (self.paths.repo_root / "logs" / "rsl_rl").resolve()
        for name_match in reversed(exact_names):
            name = name_match.group(1)
            root_matches = [root_match for root_match in experiment_roots if root_match.start() < name_match.start()]
            if root_matches:
                root_path = Path(root_matches[-1].group(1).strip()).resolve()
                if not root_path.is_relative_to(rsl_rl_root):
                    return None, True
                candidates = [path for path in root_path.glob(f"{name}*") if path.is_dir()]
                if candidates:
                    return str(max(candidates, key=lambda path: path.stat().st_mtime)), True
                return None, True
            candidates = [path for path in self.paths.rsl_rl_log_root.glob(f"{name}*") if path.is_dir()]
            if candidates:
                return str(max(candidates, key=lambda path: path.stat().st_mtime)), True
        root = re.escape(str(self.paths.rsl_rl_log_root))
        matches = re.findall(rf"({root}/[^\s'\"<>]+)", text)
        log_dirs = []
        for match in matches:
            path = Path(match)
            if path.name.startswith("events.out.tfevents") or path.name.startswith("model_"):
                path = path.parent
            while path.parent != self.paths.rsl_rl_log_root and path != self.paths.rsl_rl_log_root:
                path = path.parent
            if path.parent == self.paths.rsl_rl_log_root:
                log_dirs.append(path)
        if not log_dirs:
            return None, bool(exact_names)
        return str(log_dirs[-1]), bool(exact_names)

    def _is_repo_training_process(self, command: str, pid_text: str) -> bool:
        if self._is_tmux_server_command(command):
            return False
        if "scripts/rsl_rl/train.py" not in command:
            return False
        if "isaaclab.sh -p" not in command and "python" not in command:
            return False
        return self._command_or_cwd_matches_repo(command, pid_text)

    def _is_repo_play_process(self, command: str, pid_text: str) -> bool:
        if self._is_tmux_server_command(command):
            return False
        if "--export_policy_only" in command or "--video" in command:
            return False
        if "scripts/rsl_rl/play.py" not in command:
            return False
        if "isaaclab.sh -p" not in command and "python" not in command:
            return False
        return self._command_or_cwd_matches_repo(command, pid_text)

    def _is_repo_video_process(self, command: str, pid_text: str) -> bool:
        if self._is_tmux_server_command(command):
            return False
        if "scripts/rsl_rl/play.py" not in command or "--video" not in command:
            return False
        if "isaaclab.sh -p" not in command and "python" not in command:
            return False
        return self._command_or_cwd_matches_repo(command, pid_text)

    def _is_repo_onnx_process(self, command: str, pid_text: str) -> bool:
        if self._is_tmux_server_command(command):
            return False
        if "scripts/rsl_rl/play.py" not in command or "--export_policy_only" not in command:
            return False
        if "isaaclab.sh -p" not in command and "python" not in command:
            return False
        return self._command_or_cwd_matches_repo(command, pid_text)

    def _is_external_gpu_python_process(self, command: str, pid_text: str) -> bool:
        command_lower = command.lower()
        if self._is_tmux_server_command(command):
            return False
        if "python" not in command_lower and "isaac" not in command_lower:
            return False
        if "tools.training_panel" in command or "tensorboard" in command_lower:
            return False
        if not (
            self._command_or_cwd_matches_path(command, pid_text, self.paths.repo_root)
            or self._command_or_cwd_matches_path(command, pid_text, self.paths.isaaclab_root)
        ):
            return False
        if (
            self._is_repo_training_process(command, pid_text)
            or self._is_repo_play_process(command, pid_text)
            or self._is_repo_video_process(command, pid_text)
            or self._is_repo_onnx_process(command, pid_text)
        ):
            return False
        return True

    @staticmethod
    def _gpu_device_pids() -> set[int]:
        devices = [path for path in ("/dev/nvidia0", "/dev/nvidia-uvm", "/dev/nvidiactl") if Path(path).exists()]
        if not devices or not shutil.which("fuser"):
            return set()
        try:
            result = subprocess.run(
                ["fuser", *devices],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                check=False,
            )
        except OSError:
            return set()
        return {int(match) for match in re.findall(r"\b\d+\b", f"{result.stdout}\n{result.stderr}") if int(match) > 1}

    def _is_repo_tensorboard_process(self, command: str) -> bool:
        return "tensorboard" in command and "--logdir" in command and str(self.paths.repo_root) in command

    @staticmethod
    def _is_tmux_server_command(command: str) -> bool:
        return "tmux new-session" in command or "tmux: server" in command or "/tmux new-session" in command

    def _command_or_cwd_matches_repo(self, command: str, pid_text: str) -> bool:
        return self._command_or_cwd_matches_path(command, pid_text, self.paths.repo_root)

    @staticmethod
    def _command_or_cwd_matches_path(command: str, pid_text: str, root: Path) -> bool:
        if str(root) in command:
            return True
        try:
            return Path(f"/proc/{int(pid_text)}/cwd").resolve() == root.resolve()
        except (OSError, RuntimeError, ValueError):
            return False

    @staticmethod
    def _training_commands_match(recorded_command: str, process_command: str) -> bool:
        if not recorded_command or "scripts/rsl_rl/train.py" not in process_command:
            return False
        keys = ("--task", "--num_envs", "--max_iterations", "--device", "--seed", "--checkpoint", "--spring-backend")
        matched = 0
        for key in keys:
            recorded = ProcessRegistry._arg_value(recorded_command, key)
            observed = ProcessRegistry._arg_value(process_command, key)
            if not recorded or not observed:
                continue
            if recorded != observed:
                return False
            if recorded == observed:
                matched += 1
        return matched >= 3

    @staticmethod
    def _arg_value(command: str, key: str) -> str | None:
        match = re.search(rf"{re.escape(key)}(?:=|\s+)([^\s]+)", command)
        return match.group(1) if match else None

    @staticmethod
    def _process_group_for_pid(pid: int) -> int | None:
        try:
            return os.getpgid(pid)
        except ProcessLookupError:
            return None

    def _process_group_for_info(self, info: ProcessInfo) -> int | None:
        if info.tmux_session:
            tmux_group = self._tmux_process_group(info.tmux_session)
            if tmux_group:
                return tmux_group
        return self._process_group_for_pid(info.pid)

    @staticmethod
    def _tmux_process_group(session: str) -> int | None:
        tmux = shutil.which("tmux")
        if not tmux:
            return None
        try:
            output = subprocess.check_output(
                [tmux, "list-panes", "-t", session, "-F", "#{pane_pid}"],
                stderr=subprocess.DEVNULL,
                text=True,
                errors="replace",
            )
        except (OSError, subprocess.SubprocessError):
            return None
        for line in output.splitlines():
            try:
                return os.getpgid(int(line.strip()))
            except (ValueError, ProcessLookupError):
                continue
        return None

    def _gpu_pids_for_group(self, process_group: int | None) -> list[int]:
        if not process_group:
            return []
        gpu_pids = []
        for pid in self._gpu_device_pids():
            try:
                if os.getpgid(pid) == process_group:
                    gpu_pids.append(pid)
            except ProcessLookupError:
                continue
        return sorted(gpu_pids)

    def _stop_external_group(self, run_id: str) -> bool:
        process_id = run_id
        for prefix in EXTERNAL_ID_PREFIXES:
            process_id = process_id.removeprefix(prefix)
        try:
            process_group = int(process_id)
        except ValueError:
            return False
        process = next((item for item in self._external_processes() if item.get("run_id") == run_id), {})
        tmux_session = str(process.get("tmux_session") or "")
        try:
            if not tmux_session or not self._send_tmux_interrupt(tmux_session):
                os.killpg(process_group, signal.SIGINT)
        except ProcessLookupError:
            return False
        threading.Thread(
            target=self._force_kill_group_after_grace,
            args=(process_group, tmux_session or None),
            daemon=True,
        ).start()
        return True

    @staticmethod
    def _tmux_session_for_process_group(process_group: int) -> str | None:
        tmux = shutil.which("tmux")
        if not tmux:
            return None
        try:
            output = subprocess.check_output(
                [tmux, "list-panes", "-a", "-F", "#{session_name} #{pane_pid}"],
                stderr=subprocess.DEVNULL,
                text=True,
                errors="replace",
            )
        except (OSError, subprocess.SubprocessError):
            return None
        for line in output.splitlines():
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            session, pane_pid_text = parts
            try:
                pane_pid = int(pane_pid_text)
            except ValueError:
                continue
            if pane_pid == process_group:
                return session
            try:
                if os.getpgid(pane_pid) == process_group:
                    return session
            except ProcessLookupError:
                continue
        return None

    def _spawn_shell(self, run_id: str, shell: str, log_file: Path) -> SpawnedProcess:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        tmux = shutil.which("tmux")
        if tmux:
            return self._spawn_tmux(run_id, shell, log_file, tmux)

        log_handle = log_file.open("w", encoding="utf-8")
        try:
            log_handle.write("$ bash -lc <<'PANEL_COMMAND'\n")
            log_handle.write(shell)
            log_handle.write("\nPANEL_COMMAND\n\n")
            log_handle.flush()
            proc = subprocess.Popen(
                ["bash", "-lc", shell],
                cwd=self.paths.repo_root,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log_handle.close()
        return SpawnedProcess(proc=proc)

    def _spawn_tmux(self, run_id: str, shell: str, log_file: Path, tmux: str) -> SpawnedProcess:
        session = self._safe_tmux_session(run_id)
        done_signal = f"done_{session}"
        exit_file = self.paths.process_log_dir / f"{run_id}.exit"
        attach_command = f"tmux attach -t {shlex.quote(session)}"
        log_file.write_text("", encoding="utf-8")
        inner = "\n".join(
            [
                "set +e",
                f"exec > >(tee -a {shlex.quote(str(log_file))}) 2>&1",
                "echo \"$ bash -lc <<'PANEL_COMMAND'\"",
                "cat <<'PANEL_COMMAND'",
                shell,
                "PANEL_COMMAND",
                "echo",
                f"bash -lc {shlex.quote(shell)}",
                "status=$?",
                f"printf '%s' \"$status\" > {shlex.quote(str(exit_file))}",
                f"{shlex.quote(tmux)} wait-for -S {shlex.quote(done_signal)}",
                "exit \"$status\"",
            ]
        )
        outer = "\n".join(
            [
                f"{shlex.quote(tmux)} new-session -d -s {shlex.quote(session)} -- bash -lc {shlex.quote(inner)}"
                " || exit $?",
                f"{shlex.quote(tmux)} wait-for {shlex.quote(done_signal)}",
                "status=0",
                f"if [ -f {shlex.quote(str(exit_file))} ]; then status=$(cat {shlex.quote(str(exit_file))}); fi",
                'exit "$status"',
            ]
        )
        proc = subprocess.Popen(
            ["bash", "-lc", outer],
            cwd=self.paths.repo_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return SpawnedProcess(
            proc=proc,
            tmux_session=session,
            attach_command=attach_command,
            exit_file=str(exit_file),
        )

    def _register(
        self,
        run_id: str,
        kind: str,
        spawned: SpawnedProcess,
        log_file: Path,
        started_at: str,
        command: str,
        source_run_id: str | None = None,
    ) -> None:
        info = ProcessInfo(
            kind=kind,
            pid=spawned.proc.pid,
            run_id=run_id,
            log_file=str(log_file),
            started_at=started_at,
            command=command,
            source_run_id=source_run_id,
            tmux_session=spawned.tmux_session,
            attach_command=spawned.attach_command,
            exit_file=spawned.exit_file,
        )
        with self._lock:
            self._processes[run_id] = spawned.proc
            self._infos[run_id] = info

    @staticmethod
    def _safe_tmux_session(run_id: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id).strip("._-")
        return f"redrhex_{safe or timestamp_id()}"[:80]

    @staticmethod
    def _send_tmux_interrupt(session: str) -> bool:
        try:
            result = subprocess.run(
                ["tmux", "send-keys", "-t", session, "C-c"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except OSError:
            return False
        return result.returncode == 0

    @staticmethod
    def _kill_tmux_session(session: str) -> None:
        try:
            subprocess.run(
                ["tmux", "kill-session", "-t", session],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except OSError:
            return
        try:
            subprocess.run(
                ["tmux", "wait-for", "-S", f"done_{session}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except OSError:
            return

    @staticmethod
    def _returncode(info: ProcessInfo, proc: subprocess.Popen) -> int | None:
        returncode = proc.poll()
        if returncode is None:
            return None
        if info.exit_file:
            exit_path = Path(info.exit_file)
            if exit_path.exists():
                try:
                    return int(exit_path.read_text(encoding="utf-8").strip())
                except ValueError:
                    return returncode
        return returncode

    def _raise_if_immediate_exit(
        self,
        proc: subprocess.Popen,
        process_id: str,
        label: str,
        wait_seconds: float,
    ) -> None:
        deadline = time.time() + wait_seconds
        returncode = proc.poll()
        while returncode is None and time.time() < deadline:
            time.sleep(0.1)
            returncode = proc.poll()
        if returncode is None:
            return
        debug = self.get_process_debug(process_id) or {"returncode": returncode}
        raise ProcessStartError(f"{label} exited while starting. Open Console or check the process log.", debug)

    def _monitor_training(self, run_id: str, proc: subprocess.Popen, started_at_epoch: float) -> None:
        import time
        from .convergence import ConvergenceChecker, load_convergence_config

        checker = ConvergenceChecker()
        convergence_detected = False
        convergence_notified_at: float | None = None
        log_dir: str | None = None
        log_poll_interval = 2.0
        convergence_poll_interval = 60.0
        next_convergence_check = 0.0

        # Poll while training is running, checking for convergence each cycle.
        while proc.poll() is None:
            if not log_dir:
                log_dir = self._log_dir_from_process_log(run_id)
                if log_dir:
                    self.history.update_run(run_id, log_dir=log_dir)
            now = time.time()
            should_check_convergence = False
            if now >= next_convergence_check:
                should_check_convergence = True
                next_convergence_check = now + convergence_poll_interval
                if not log_dir:
                    log_dir = self._log_dir_from_process_log(run_id) or self.history.find_latest_log_after(started_at_epoch)
                    if log_dir:
                        self.history.update_run(run_id, log_dir=log_dir)
            if log_dir and not convergence_detected and should_check_convergence:
                try:
                    cfg = load_convergence_config(self.paths.convergence_config_file)
                    if cfg.enabled:
                        cooled = (
                            convergence_notified_at is None
                            or (time.time() - convergence_notified_at) > cfg.cooldown_minutes * 60
                        )
                        if cooled:
                            result = checker.check(Path(log_dir), cfg)
                            if result.detected:
                                convergence_detected = True
                                convergence_notified_at = time.time()
                                self.history.update_run(
                                    run_id,
                                    convergence_detected=True,
                                    convergence_iteration=result.iteration,
                                    convergence_improvement_pct=round(result.improvement_pct, 2),
                                )
                                if cfg.auto_record_video:
                                    self.history.update_run(run_id, queue_video_on_completion=True)
                except Exception:
                    pass  # never let convergence logic crash the monitor thread
            time.sleep(log_poll_interval)

        returncode = proc.wait()
        self._mark_isaac_process_finished()
        status = "completed" if returncode == 0 else "failed"
        if not log_dir:
            log_dir = self._log_dir_from_process_log(run_id)
            if not log_dir and returncode == 0:
                log_dir = self.history.find_latest_log_after(started_at_epoch)
        completed_at = datetime.now().isoformat(timespec="seconds")
        self.history.update_run(run_id, status=status, returncode=returncode, log_dir=log_dir, completed_at=completed_at)
        self._refresh_tensorboard_summary(run_id, log_dir)

        # Record video when: training succeeded normally, OR convergence was detected and
        # auto_record_video was requested (even if training was stopped early).
        run = self.history.get_run(run_id) or {}
        force_video = bool(run.get("queue_video_on_completion")) and convergence_detected
        video_started = False
        if (returncode == 0 or force_video) and log_dir:
            checkpoint = latest_checkpoint(Path(log_dir))
            if not checkpoint:
                self.history.update_run(run_id, video_status="missing_checkpoint")
                self.start_next_queued_training()
                return
            try:
                params = run.get("params") or {}
                self.start_video_recording(
                    run_id=run_id,
                    checkpoint=checkpoint,
                    device=str(params.get("device") or "cuda:0"),
                    video_params=VideoParams.from_preset(DEFAULT_VIDEO_PRESET),
                )
                video_started = True
            except Exception as exc:
                self.history.update_run(run_id, video_status="failed", video_error=str(exc))
        elif not log_dir:
            pass  # no log dir found — nothing to record
        if not video_started:
            self.start_next_queued_training()

    def _refresh_tensorboard_summary(self, run_id: str, log_dir: str | None) -> None:
        if not log_dir:
            return
        try:
            from .tensorboard_summary import ensure_tensorboard_summary

            run = self.history.get_run(run_id) or {}
            title = str(run.get("display_name") or run.get("id") or Path(log_dir).name)
            summary = ensure_tensorboard_summary(Path(log_dir), title=title)
            if summary:
                self.history.update_run(
                    run_id,
                    tensorboard_summary_path=str(summary),
                    tensorboard_summary_status="completed",
                    tensorboard_summary_error=None,
                )
        except Exception as exc:
            self.history.update_run(
                run_id,
                tensorboard_summary_status="failed",
                tensorboard_summary_error=str(exc),
            )

    def _monitor_video(self, source_run_id: str, video_id: str, proc: subprocess.Popen) -> None:
        returncode = proc.wait()
        self._mark_isaac_process_finished()
        run = self.history.get_run(source_run_id) or {}
        log_dir = Path(run["log_dir"]) if run.get("log_dir") else None
        video = latest_video(log_dir) if log_dir and log_dir.exists() else None
        if video:
            video = self._tag_video_with_checkpoint(Path(video), video_id, run)
        self.history.update_run(
            source_run_id,
            video_status="completed" if returncode == 0 and video else "failed",
            video_returncode=returncode,
            video_process_id=video_id,
            latest_video=video,
            has_video=bool(video),
            video_error=None if returncode == 0 and video else "Video process finished but no MP4 was produced.",
        )
        self.start_next_queued_training()

    @staticmethod
    def _checkpoint_iteration(checkpoint: str | Path) -> int | None:
        match = re.search(r"model_(\d+)\.pt$", str(checkpoint or ""))
        return int(match.group(1)) if match else None

    def _tag_video_with_checkpoint(self, video: Path, video_id: str, run: dict) -> str:
        iteration = run.get("video_checkpoint_iteration") or self._checkpoint_iteration(str(run.get("video_checkpoint") or ""))
        if not iteration or not video.is_file():
            return str(video)
        if f"model_{iteration}" in video.name:
            return str(video)
        safe_video_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", video_id).strip("._-") or "video"
        target = video.with_name(f"model_{iteration}_{safe_video_id}{video.suffix}")
        counter = 2
        while target.exists() and target != video:
            target = video.with_name(f"model_{iteration}_{safe_video_id}_{counter}{video.suffix}")
            counter += 1
        try:
            video.rename(target)
            return str(target)
        except OSError:
            return str(video)

    def _monitor_onnx(self, source_run_id: str, onnx_id: str, proc: subprocess.Popen) -> None:
        returncode = proc.wait()
        self._mark_isaac_process_finished()
        run = self.history.get_run(source_run_id) or {}
        log_dir = Path(run["log_dir"]) if run.get("log_dir") else None
        onnx_path = latest_onnx(log_dir) if log_dir and log_dir.exists() else None
        self.history.update_run(
            source_run_id,
            onnx_status="completed" if returncode == 0 and onnx_path else "failed",
            onnx_returncode=returncode,
            onnx_process_id=onnx_id,
            onnx_path=onnx_path,
            has_onnx=bool(onnx_path),
            onnx_error=None if returncode == 0 and onnx_path else "ONNX export finished but policy.onnx was not produced.",
        )
        self.start_next_queued_training()

    def _monitor_deploy(self, source_run_id: str, deploy_id: str, proc: subprocess.Popen) -> None:
        returncode = proc.wait()
        self._mark_isaac_process_finished()
        run = self.history.get_run(source_run_id) or {}
        latest = latest_deploy_report(run) if run else None
        report = latest.get("report") if latest else None
        self.history.update_run(
            source_run_id,
            deploy_status="completed" if latest else "failed",
            deploy_returncode=returncode,
            deploy_process_id=deploy_id,
            deploy_report_path=latest.get("path") if latest else None,
            deploy_overall_status=report.get("overall_status") if isinstance(report, dict) else None,
            deploy_readiness_level=report.get("readiness_level") if isinstance(report, dict) else None,
            deploy_completed_at=datetime.now().isoformat(timespec="seconds"),
            deploy_error=None if latest else "Deploy validation finished but no readiness report was produced.",
        )
        self.start_next_queued_training()

    def _monitor_mujoco(self, source_run_id: str, process_id: str, proc: subprocess.Popen) -> None:
        returncode = proc.wait()
        run = self.history.get_run(source_run_id) or {}
        log_dir = Path(str(run.get("log_dir") or ""))
        report_path = log_dir / "deploy" / f"mujoco_playback_{process_id}" / "mujoco_playback_report.json"
        report = None
        if report_path.is_file():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                report = None
        video = ""
        if isinstance(report, dict):
            video = str(report.get("video_path") or report.get("artifacts", {}).get("video") or "")
        if video and not Path(video).is_file():
            video = ""
        if returncode == 130 or returncode < 0:
            status = "stopped"
        elif returncode == 0 and isinstance(report, dict) and report.get("status") == "completed":
            status = "completed"
        else:
            status = "failed"
        error = None
        if status == "failed":
            error = "MuJoCo playback finished but no successful playback report was produced."
            if isinstance(report, dict):
                error = str(report.get("summary") or error)
        self.history.update_run(
            source_run_id,
            mujoco_playback_status=status,
            mujoco_returncode=returncode,
            mujoco_process_id=process_id,
            mujoco_video_report=str(report_path) if report_path.is_file() else None,
            latest_mujoco_video=video or None,
            mujoco_error=error,
            mujoco_completed_at=datetime.now().isoformat(timespec="seconds"),
        )

    @staticmethod
    def _force_stop_after_grace(proc: subprocess.Popen, process_group: int, tmux_session: str | None = None) -> None:
        try:
            proc.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            pass
        if tmux_session:
            ProcessRegistry._kill_tmux_session(tmux_session)
            try:
                proc.wait(timeout=3)
                return
            except subprocess.TimeoutExpired:
                pass
        for sig in (signal.SIGTERM, signal.SIGKILL):
            if proc.poll() is not None:
                return
            try:
                os.killpg(process_group, sig)
            except ProcessLookupError:
                return
            try:
                proc.wait(timeout=3)
                return
            except subprocess.TimeoutExpired:
                continue

    @staticmethod
    def _force_kill_group_after_grace(process_group: int, tmux_session: str | None = None) -> None:
        time.sleep(5)
        if tmux_session:
            ProcessRegistry._kill_tmux_session(tmux_session)
            time.sleep(2)
            try:
                os.killpg(process_group, 0)
            except ProcessLookupError:
                return
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(process_group, 0)
            except ProcessLookupError:
                return
            try:
                os.killpg(process_group, sig)
            except ProcessLookupError:
                return
            time.sleep(3)

    @staticmethod
    def _find_free_port(start: int) -> int:
        for port in range(start, start + 100):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    sock.bind(("127.0.0.1", port))
                except OSError:
                    continue
                return port
        raise RuntimeError("No free port found for TensorBoard")

    @staticmethod
    def _tensorboard_response(
        host: str,
        port: int,
        pid: int,
        process_id: str,
        process_log: str,
        command: str,
        already_running: bool,
        attach_command: str | None = None,
        tmux_session: str | None = None,
    ) -> dict:
        display_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
        return {
            "pid": pid,
            "id": process_id,
            "already_running": already_running,
            "url": f"http://{display_host}:{port}",
            "host": host,
            "port": port,
            "process_log": process_log,
            "command": command,
            "attach_command": attach_command,
            "tmux_session": tmux_session,
        }
