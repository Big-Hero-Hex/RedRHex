from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .commands import DEFAULT_TASK
from .config import PanelPaths
from .history import HistoryStore, latest_checkpoint


EXACT_EXPERIMENT_RE = re.compile(r"Exact experiment name requested from command line:\s*(\S+)")


class SmokeValidationError(RuntimeError):
    pass


@dataclass
class SmokeConfig:
    task: str = DEFAULT_TASK
    experiment_name: str = "redrhex_wheg"
    run_name: str = ""
    num_envs: int = 4
    max_iterations: int = 1
    device: str = "cuda:0"
    headless: bool = True
    seed: int | None = 1
    include_export: bool = False
    include_video: bool = False
    video_length: int = 120
    video_width: int = 640
    video_height: int = 360
    video_fps: int = 30
    timeout_seconds: int = 1800
    panel_url: str = ""
    run_dir: str = ""


@dataclass
class CommandResult:
    argv: list[str]
    returncode: int
    output: str


def default_run_name(now: datetime | None = None) -> str:
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"disposable_smoke_{stamp}"


def _paths_from_env() -> PanelPaths:
    return PanelPaths.from_env()


def _command_env(paths: PanelPaths) -> dict[str, str]:
    env = os.environ.copy()
    env["REDRHEX_ROOT"] = str(paths.repo_root)
    env["ISAACLAB_ROOT"] = str(paths.isaaclab_root)
    env["ISAACSIM_ROOT"] = str(paths.isaacsim_root)
    return env


def build_train_command(paths: PanelPaths, config: SmokeConfig, extra_args: list[str] | None = None) -> list[str]:
    argv = [
        str(paths.isaaclab_launcher),
        "-p",
        "scripts/rsl_rl/train.py",
        "--task",
        config.task,
        "--num_envs",
        str(config.num_envs),
        "--max_iterations",
        str(config.max_iterations),
        "--device",
        config.device,
        "--experiment_name",
        config.experiment_name,
        "--run_name",
        config.run_name,
    ]
    if config.headless:
        argv.append("--headless")
    if config.seed is not None:
        argv.extend(["--seed", str(config.seed)])
    argv.extend(extra_args or [])
    return argv


def build_export_command(paths: PanelPaths, config: SmokeConfig, checkpoint: str) -> list[str]:
    return [
        str(paths.isaaclab_launcher),
        "-p",
        "scripts/rsl_rl/play.py",
        "--task",
        config.task,
        "--num_envs",
        "1",
        "--device",
        config.device,
        "--headless",
        "--export_policy_only",
        "--checkpoint",
        checkpoint,
    ]


def build_video_command(paths: PanelPaths, config: SmokeConfig, checkpoint: str) -> list[str]:
    return [
        str(paths.isaaclab_launcher),
        "-p",
        "scripts/rsl_rl/play.py",
        "--task",
        config.task,
        "--num_envs",
        "1",
        "--device",
        config.device,
        "--headless",
        "--video",
        "--video_length",
        str(config.video_length),
        "--video_width",
        str(config.video_width),
        "--video_height",
        str(config.video_height),
        "--video_fps",
        str(config.video_fps),
        "--disable_keyboard_control",
        "--checkpoint",
        checkpoint,
    ]


def command_display(argv: list[str]) -> str:
    return shlex.join(argv)


def run_command(argv: list[str], *, cwd: Path, env: dict[str, str], timeout_seconds: int) -> CommandResult:
    completed = subprocess.run(
        argv,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_seconds,
        check=False,
    )
    return CommandResult(argv=argv, returncode=completed.returncode, output=completed.stdout or "")


def exact_experiment_name(output: str) -> str | None:
    matches = EXACT_EXPERIMENT_RE.findall(output or "")
    return matches[-1] if matches else None


def _candidate_run_dirs(root: Path, config: SmokeConfig, output: str, start_time: float) -> list[Path]:
    candidates: list[Path] = []
    if config.run_dir:
        candidates.append(Path(config.run_dir).expanduser().resolve())
    exact = exact_experiment_name(output)
    if exact:
        candidates.append(root / f"{exact}_{config.run_name}" if config.run_name else root / exact)
    if config.run_name and root.exists():
        for path in root.iterdir():
            if path.is_dir() and path.name.endswith(f"_{config.run_name}"):
                candidates.append(path)
    if root.exists():
        for path in root.iterdir():
            if path.is_dir() and path.stat().st_mtime >= start_time - 5:
                candidates.append(path)

    seen: set[str] = set()
    unique = []
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def resolve_run_dir(paths: PanelPaths, config: SmokeConfig, output: str, start_time: float) -> Path:
    root = (
        paths.rsl_rl_log_root
        if config.experiment_name == paths.rsl_rl_log_root.name
        else paths.repo_root / "logs" / "rsl_rl" / config.experiment_name
    )
    candidates = _candidate_run_dirs(root, config, output, start_time)
    existing = [path for path in candidates if path.is_dir()]
    if existing:
        return max(existing, key=lambda path: path.stat().st_mtime)
    detail = ", ".join(str(path) for path in candidates) or str(root)
    raise SmokeValidationError(f"Could not find smoke run directory. Checked: {detail}")


def _event_files(run_dir: Path) -> list[Path]:
    return sorted(path for path in run_dir.glob("events.out.tfevents.*") if path.is_file())


def _panel_runs(panel_url: str) -> list[dict[str, Any]]:
    if not panel_url:
        return []
    url = f"{panel_url.rstrip('/')}/api/runs"
    try:
        with urllib.request.urlopen(url, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise SmokeValidationError(f"could not read panel runs from {url}: {exc}") from exc
    runs = payload.get("runs") if isinstance(payload, dict) else None
    return runs if isinstance(runs, list) else []


def _local_history_runs(paths: PanelPaths) -> list[dict[str, Any]]:
    return HistoryStore(paths).list_runs()


def _matching_history_run(runs: list[dict[str, Any]], run_dir: Path) -> dict[str, Any] | None:
    run_dir_text = str(run_dir)
    for run in runs:
        if str(run.get("log_dir") or "") == run_dir_text or run.get("id") == run_dir.name:
            return run
    return None


def validate_smoke_run(paths: PanelPaths, config: SmokeConfig, output: str = "", start_time: float = 0.0) -> dict:
    run_dir = resolve_run_dir(paths, config, output, start_time)
    checkpoint = latest_checkpoint(run_dir)
    events = _event_files(run_dir)
    params_dir = run_dir / "params"
    failures = []
    if not checkpoint:
        failures.append(f"missing model_*.pt checkpoint in {run_dir}")
    if not events:
        failures.append(f"missing TensorBoard event file in {run_dir}")
    if not (params_dir / "env.yaml").is_file():
        failures.append(f"missing params/env.yaml in {run_dir}")
    if not (params_dir / "agent.yaml").is_file():
        failures.append(f"missing params/agent.yaml in {run_dir}")

    local_run = _matching_history_run(_local_history_runs(paths), run_dir)
    if not local_run:
        failures.append("local panel history backend does not discover the run")
    elif local_run.get("latest_checkpoint") != checkpoint:
        failures.append("local panel history latest checkpoint does not match filesystem")

    remote_run = None
    if config.panel_url:
        remote_run = _matching_history_run(_panel_runs(config.panel_url), run_dir)
        if not remote_run:
            failures.append(f"panel API at {config.panel_url} does not list the smoke run")

    if failures:
        raise SmokeValidationError("; ".join(failures))

    result = {
        "ok": True,
        "run_dir": str(run_dir),
        "run_id": run_dir.name,
        "checkpoint": checkpoint,
        "event_files": [str(path) for path in events],
        "local_history_run_id": local_run.get("id") if local_run else None,
        "panel_history_run_id": remote_run.get("id") if remote_run else None,
    }
    return result


def validate_export(run_dir: Path) -> dict:
    exported = run_dir / "exported"
    jit = exported / "policy.pt"
    onnx = exported / "policy.onnx"
    missing = [str(path) for path in (jit, onnx) if not path.is_file()]
    if missing:
        raise SmokeValidationError(f"missing exported policy artifact(s): {', '.join(missing)}")
    return {"policy_pt": str(jit), "policy_onnx": str(onnx)}


def validate_video(run_dir: Path) -> dict:
    videos = sorted((run_dir / "videos" / "play").glob("*.mp4"))
    if not videos:
        raise SmokeValidationError(f"missing recorded MP4 under {run_dir / 'videos' / 'play'}")
    newest = max(videos, key=lambda path: path.stat().st_mtime)
    return {"video": str(newest)}


def summary_path(paths: PanelPaths, run_name: str) -> Path:
    return paths.panel_log_root / f"smoke_pipeline_{run_name}.json"


def write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run and validate a disposable RedRHex training smoke pipeline.")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--experiment-name", default="redrhex_wheg")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument("--include-export", action="store_true", help="Run play.py --export_policy_only and verify policy.pt/policy.onnx.")
    parser.add_argument("--include-video", action="store_true", help="Record a short headless MP4 and verify it exists.")
    parser.add_argument("--video-length", type=int, default=120)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=360)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--panel-url", default="", help="Optional running local panel URL to verify /api/runs.")
    parser.add_argument("--run-dir", default="", help="Existing run directory for --validate-only.")
    parser.add_argument("--validate-only", action="store_true", help="Skip training and validate an existing run.")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without launching Isaac.")
    parser.add_argument("--summary-file", default="", help="Write validation summary JSON to this path.")
    parser.add_argument(
        "extra_train_args",
        nargs=argparse.REMAINDER,
        help="Extra args passed to train.py after '--', for example -- env.stage=1.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> SmokeConfig:
    run_name = args.run_name.strip() or default_run_name()
    return SmokeConfig(
        task=args.task,
        experiment_name=args.experiment_name,
        run_name=run_name,
        num_envs=args.num_envs,
        max_iterations=args.max_iterations,
        device=args.device,
        headless=not args.no_headless,
        seed=args.seed,
        include_export=args.include_export,
        include_video=args.include_video,
        video_length=args.video_length,
        video_width=args.video_width,
        video_height=args.video_height,
        video_fps=args.video_fps,
        timeout_seconds=args.timeout_seconds,
        panel_url=args.panel_url,
        run_dir=args.run_dir,
    )


def run_smoke_pipeline(args: argparse.Namespace) -> dict:
    paths = _paths_from_env()
    config = config_from_args(args)
    extra = list(args.extra_train_args or [])
    if extra and extra[0] == "--":
        extra = extra[1:]
    train_command = build_train_command(paths, config, extra)
    payload: dict[str, Any] = {
        "config": asdict(config),
        "repo_root": str(paths.repo_root),
        "train_command": train_command,
        "steps": [],
    }

    if args.dry_run:
        payload["dry_run"] = True
        payload["commands"] = {"train": command_display(train_command)}
        print(json.dumps(payload, indent=2))
        return payload

    if not args.validate_only and not paths.isaaclab_launcher.is_file():
        raise SmokeValidationError(f"Isaac Lab launcher not found: {paths.isaaclab_launcher}")

    output = ""
    start_time = time.time()
    if not args.validate_only:
        print(f"[smoke] launching training: {command_display(train_command)}")
        train = run_command(
            train_command,
            cwd=paths.repo_root,
            env=_command_env(paths),
            timeout_seconds=config.timeout_seconds,
        )
        output = train.output
        sys.stdout.write(output)
        payload["steps"].append({"name": "train", "returncode": train.returncode})
        if train.returncode != 0:
            raise SmokeValidationError(f"training command failed with exit code {train.returncode}")

    validation = validate_smoke_run(paths, config, output, start_time)
    payload["validation"] = validation
    run_dir = Path(validation["run_dir"])
    checkpoint = str(validation["checkpoint"])

    if config.include_export:
        export_command = build_export_command(paths, config, checkpoint)
        payload["export_command"] = export_command
        print(f"[smoke] launching export: {command_display(export_command)}")
        export = run_command(
            export_command,
            cwd=paths.repo_root,
            env=_command_env(paths),
            timeout_seconds=config.timeout_seconds,
        )
        sys.stdout.write(export.output)
        payload["steps"].append({"name": "export", "returncode": export.returncode})
        if export.returncode != 0:
            raise SmokeValidationError(f"export command failed with exit code {export.returncode}")
        payload["export_validation"] = validate_export(run_dir)

    if config.include_video:
        video_command = build_video_command(paths, config, checkpoint)
        payload["video_command"] = video_command
        print(f"[smoke] launching video: {command_display(video_command)}")
        video = run_command(
            video_command,
            cwd=paths.repo_root,
            env=_command_env(paths),
            timeout_seconds=config.timeout_seconds,
        )
        sys.stdout.write(video.output)
        payload["steps"].append({"name": "video", "returncode": video.returncode})
        if video.returncode != 0:
            raise SmokeValidationError(f"video command failed with exit code {video.returncode}")
        payload["video_validation"] = validate_video(run_dir)

    output_path = Path(args.summary_file).expanduser() if args.summary_file else summary_path(paths, config.run_name)
    write_summary(output_path, payload)
    payload["summary_file"] = str(output_path)
    print(f"[smoke] validation complete: {json.dumps(payload['validation'], indent=2)}")
    print(f"[smoke] summary: {output_path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        run_smoke_pipeline(args)
    except subprocess.TimeoutExpired as exc:
        print(f"[smoke] timeout after {exc.timeout}s: {exc.cmd}", file=sys.stderr)
        return 124
    except SmokeValidationError as exc:
        print(f"[smoke] failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
