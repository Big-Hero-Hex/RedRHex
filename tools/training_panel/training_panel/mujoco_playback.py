from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .config import PanelPaths
from .deploy import FALLBACK_CONTRACT, build_policy_manifest, default_mujoco_model_path
from .history import HistoryStore
from .mujoco_rollout import (
    DEFAULT_PLAYBACK_FPS,
    DEFAULT_PLAYBACK_HEIGHT,
    DEFAULT_PLAYBACK_STEPS,
    DEFAULT_PLAYBACK_WIDTH,
    MujocoPlaybackConfig,
    load_calibration_config,
    run_mujoco_playback,
    scenario_by_name,
)


def _onnx_input_dim(policy_path: Path, default: int) -> int:
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
        shape = list(session.get_inputs()[0].shape)
    except Exception:
        return default
    for value in reversed(shape):
        if isinstance(value, int) and value > 0:
            return int(value)
    return default


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RedRHex MuJoCo policy playback.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--process-id", required=True)
    parser.add_argument("--mode", choices=["viewer", "record"], required=True)
    parser.add_argument("--scenario", default="stand_zero")
    parser.add_argument("--steps", type=int, default=DEFAULT_PLAYBACK_STEPS)
    parser.add_argument("--width", type=int, default=DEFAULT_PLAYBACK_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_PLAYBACK_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_PLAYBACK_FPS)
    parser.add_argument("--mujoco-model-path", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = PanelPaths.from_env()
    history = HistoryStore(paths)
    run = history.get_run(args.run_id)
    if not run:
        print(json.dumps({"error": f"Run not found: {args.run_id}"}, indent=2), file=sys.stderr)
        return 2
    try:
        scenario_by_name(args.scenario, steps=args.steps)
        manifest = build_policy_manifest(paths, run)
        policy_path = Path(manifest.policy_onnx_path)
        if not policy_path.is_file():
            print(json.dumps({"error": f"No exported policy.onnx found: {policy_path}"}, indent=2), file=sys.stderr)
            return 2
        model_path = Path(args.mujoco_model_path).expanduser() if args.mujoco_model_path else default_mujoco_model_path(paths)
        if not model_path.is_file():
            print(json.dumps({"error": f"MuJoCo model not found: {model_path}"}, indent=2), file=sys.stderr)
            return 2
        artifact_dir = Path(manifest.log_dir) / "deploy" / f"mujoco_playback_{args.process_id}"
        config = load_calibration_config(repo_root=paths.repo_root, model_path=model_path, contract=FALLBACK_CONTRACT)
        obs_dim = _onnx_input_dim(policy_path, manifest.expected_obs_dim)
        report = run_mujoco_playback(
            repo_root=paths.repo_root,
            model_path=model_path,
            policy_path=policy_path,
            artifact_dir=artifact_dir,
            config=config,
            obs_dim=obs_dim,
            action_dim=manifest.expected_action_dim,
            playback=MujocoPlaybackConfig(
                mode=args.mode,
                scenario=args.scenario,
                steps=args.steps,
                width=args.width,
                height=args.height,
                fps=args.fps,
            ),
        )
    except KeyboardInterrupt:
        print(json.dumps({"status": "stopped", "process_id": args.process_id}, indent=2), file=sys.stderr)
        return 130
    except Exception as exc:
        print(json.dumps({"error": str(exc), "process_id": args.process_id}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(report.to_dict(), indent=2))
    sys.stdout.flush()
    sys.stderr.flush()
    if args.mode == "viewer" and report.status == "completed":
        os._exit(0)
    return 0 if report.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
