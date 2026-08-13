#!/usr/bin/env python3
"""Run the approved Sensor V2 F1 -> F2 -> F3 training lineage sequentially."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SENSOR_TASK = "Template-Redrhex-ForwardSensorV2-Direct-v0"
EXPERIMENTS = {
    "teacher": "redrhex_forward_v2_teacher",
    "distillation": "redrhex_forward_v2_distillation",
    "ppo": "redrhex_forward_v2_ppo",
}
AGENTS = {
    "teacher": "rsl_rl_teacher_v2_cfg_entry_point",
    "distillation": "rsl_rl_distillation_v2_cfg_entry_point",
    "ppo": "rsl_rl_ppo_v2_cfg_entry_point",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--teacher_iterations", type=int, default=1500)
    parser.add_argument("--distillation_iterations", type=int, default=800)
    parser.add_argument("--ppo_iterations", type=int, default=1500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--pipeline_id", default="")
    parser.add_argument("--physics-profile", default=None)
    return parser.parse_args()


def _safe_pipeline_id(value: str) -> str:
    raw = value.strip() or datetime.now().strftime("sensor_v2_%Y%m%d_%H%M%S")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_.-")
    if not safe:
        raise ValueError("pipeline_id has no usable characters")
    return safe[:96]


def _latest_checkpoint(run_dir: Path) -> Path:
    candidates: list[tuple[int, Path]] = []
    for path in run_dir.glob("model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", path.name)
        if match and path.is_file():
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise RuntimeError(f"stage completed without a model_*.pt checkpoint: {run_dir}")
    return max(candidates, key=lambda item: item[0])[1].resolve()


def _find_run_dir(experiment: str, run_name: str) -> Path:
    root = REPO_ROOT / "logs" / "rsl_rl" / experiment
    candidates = [path for path in root.glob(f"*_{run_name}") if path.is_dir()]
    if not candidates:
        raise RuntimeError(f"could not find completed {experiment} run named {run_name}")
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def _run_stage(
    args: argparse.Namespace,
    *,
    stage: str,
    iterations: int,
    run_name: str,
    bootstrap_flag: str | None = None,
    bootstrap_checkpoint: Path | None = None,
) -> tuple[Path, Path]:
    launcher = Path(os.environ.get("ISAACLAB_ROOT", "/home/lab_user1/isaac_lab_ws/IsaacLab")) / "isaaclab.sh"
    command = [
        str(launcher),
        "-p",
        "scripts/rsl_rl/train.py",
        "--task",
        SENSOR_TASK,
        "--agent",
        AGENTS[stage],
        "--num_envs",
        str(args.num_envs),
        "--max_iterations",
        str(iterations),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--run_name",
        run_name,
    ]
    if args.headless:
        command.append("--headless")
    if args.physics_profile:
        command.extend(["--physics-profile", args.physics_profile])
    if bootstrap_flag is not None:
        if bootstrap_checkpoint is None:
            raise ValueError(f"{stage} requires a bootstrap checkpoint")
        command.extend([bootstrap_flag, str(bootstrap_checkpoint)])

    print(f"[SENSOR_V2_PIPELINE] Starting {stage}: {' '.join(command)}", flush=True)
    child = subprocess.Popen(command, cwd=REPO_ROOT, env={**os.environ, "TERM": "xterm"})

    def forward_signal(signum: int, _frame: object) -> None:
        if child.poll() is None:
            child.send_signal(signum)

    previous_handlers = {
        signum: signal.signal(signum, forward_signal)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        returncode = child.wait()
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    if returncode != 0:
        raise RuntimeError(f"Sensor V2 {stage} stage exited with code {returncode}")
    run_dir = _find_run_dir(EXPERIMENTS[stage], run_name)
    checkpoint = _latest_checkpoint(run_dir)
    print(f"[SENSOR_V2_PIPELINE] Completed {stage}: {checkpoint}", flush=True)
    return run_dir, checkpoint


def main() -> int:
    args = _arguments()
    if args.num_envs < 1:
        raise ValueError("num_envs must be positive")
    for name in ("teacher_iterations", "distillation_iterations", "ppo_iterations"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    pipeline_id = _safe_pipeline_id(args.pipeline_id)

    teacher_dir, teacher_checkpoint = _run_stage(
        args,
        stage="teacher",
        iterations=args.teacher_iterations,
        run_name=f"{pipeline_id}_f1_teacher",
    )
    distillation_dir, distilled_checkpoint = _run_stage(
        args,
        stage="distillation",
        iterations=args.distillation_iterations,
        run_name=f"{pipeline_id}_f2_distillation",
        bootstrap_flag="--teacher_checkpoint",
        bootstrap_checkpoint=teacher_checkpoint,
    )
    ppo_dir, ppo_checkpoint = _run_stage(
        args,
        stage="ppo",
        iterations=args.ppo_iterations,
        run_name=f"{pipeline_id}_f3_ppo",
        bootstrap_flag="--student_checkpoint",
        bootstrap_checkpoint=distilled_checkpoint,
    )
    result = {
        "pipeline_id": pipeline_id,
        "status": "completed",
        "teacher_log_dir": str(teacher_dir),
        "teacher_checkpoint": str(teacher_checkpoint),
        "distillation_log_dir": str(distillation_dir),
        "distilled_checkpoint": str(distilled_checkpoint),
        "ppo_log_dir": str(ppo_dir),
        "ppo_checkpoint": str(ppo_checkpoint),
    }
    result_path = ppo_dir / "sensor_v2_pipeline.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"SENSOR_V2_PIPELINE_RESULT: {json.dumps(result, sort_keys=True)}", flush=True)
    print(f"SENSOR_V2_FINAL_LOG_DIR: {ppo_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
