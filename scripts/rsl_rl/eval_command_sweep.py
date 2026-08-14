# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained RSL-RL policy using locomotion acceptance metrics."""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

_REDRHEX_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REDRHEX_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REDRHEX_REPO_ROOT))

from tools.training_panel.training_panel.autopilot_identity import (  # noqa: E402
    build_dependency_manifest,
    dependency_manifest_sha256,
    sha256_file as _sha256_file,
    source_code_identities,
)

_EVALUATOR_DEPENDENCY_MANIFEST: dict[str, object] | None = None

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Evaluate RSL-RL policy with command sweep.")
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--spring-backend",
    choices=("explicit", "native"),
    default="explicit",
    help="Passive torsion-spring implementation used by the environment.",
)
parser.add_argument(
    "--physics-profile",
    type=str,
    default=None,
    help="Explicit CalibrationProfileV1 JSON applied during command-sweep evaluation.",
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--sweep_steps", type=int, default=600, help="Evaluation steps per command.")
parser.add_argument(
    "--expected-step-dt",
    type=float,
    default=None,
    help="Exact policy control timestep required by a strict Autopilot evaluation.",
)
parser.add_argument("--warmup_steps", type=int, default=120, help="Warm-up steps per command.")
parser.add_argument("--csv", type=str, default=None, help="Optional command-table CSV path.")
parser.add_argument("--checkpoint-sha256", type=str, default=None, help="Expected exact checkpoint SHA-256.")
parser.add_argument(
    "--command-profile",
    type=str,
    default=None,
    help="Exact immutable redrhex.autopilot.command-profile.v1 JSON.",
)
parser.add_argument(
    "--command-profile-sha256",
    type=str,
    default=None,
    help="Expected SHA-256 of --command-profile.",
)
for _identity_name in (
    "code",
    "config",
    "dependency",
    "reward-profile",
    "physics",
    "spring",
    "terrain",
):
    parser.add_argument(
        f"--identity-{_identity_name}-sha256",
        type=str,
        default=None,
        help=f"Frozen Autopilot {_identity_name} identity SHA-256.",
    )
parser.add_argument(
    "--strict-checkpoint-loading",
    action="store_true",
    default=False,
    help="Require an exact model_*.pt path and forbid fallback or partial loading.",
)
parser.add_argument(
    "--strict-energy-evidence",
    action="store_true",
    default=False,
    help="Fail evaluation when simulator torque evidence is unavailable; forbids proxy/zero energy evidence.",
)
parser.add_argument(
    "--curriculum-stage",
    type=int,
    choices=(1, 2, 3, 4, 5),
    default=None,
    help="Typed curriculum stage; disables path-based stage inference.",
)
parser.add_argument(
    "--eval_profile",
    type=str,
    default="stage5",
    choices=["stage1", "stage2", "stage3", "stage4", "stage5", "full"],
    help="Command-sweep profile that matches the 5-stage curriculum.",
)
parser.add_argument("--command_scale", type=float, default=1.0, help="Scale factor applied to all sweep commands.")
parser.add_argument("--accept_duration_s", type=float, default=2.0, help="Minimum success duration for lateral/yaw tests.")
parser.add_argument("--accept_vx_abs", type=float, default=0.15, help="Forward speed threshold for acceptance.")
parser.add_argument("--accept_vy_abs", type=float, default=0.15, help="Lateral speed threshold for acceptance.")
parser.add_argument("--accept_wz_abs", type=float, default=0.40, help="Yaw rate threshold for acceptance.")
parser.add_argument("--accept_lin_ratio", type=float, default=0.55, help="Required |v| / |v_cmd| ratio for linear commands.")
parser.add_argument("--accept_wz_ratio", type=float, default=0.55, help="Required |wz| / |wz_cmd| ratio for yaw commands.")
parser.add_argument("--accept_yaw_tilt_bound", type=float, default=0.60, help="Max |roll|/|pitch| bound during yaw acceptance.")
parser.add_argument("--accept_yaw_tilt_ratio", type=float, default=0.70, help="Required fraction of yaw samples within tilt bound.")
parser.add_argument("--accept_forward_lateral_leak", type=float, default=0.12, help="Max |vy| allowed in forward acceptance.")
parser.add_argument("--accept_forward_yaw_leak", type=float, default=0.30, help="Max |wz| allowed in forward acceptance.")
parser.add_argument("--accept_lateral_forward_leak", type=float, default=0.12, help="Max |vx| allowed in lateral acceptance.")
parser.add_argument("--accept_lateral_yaw_leak", type=float, default=0.30, help="Max |wz| allowed in lateral acceptance.")
parser.add_argument("--accept_diag_sign_ratio", type=float, default=0.70, help="Required sign-match ratio for diagonal commands.")
parser.add_argument(
    "--accept_diag_component_ratio",
    type=float,
    default=0.50,
    help="Required per-axis speed ratio (|v|/|v_cmd|) for diagonal acceptance.",
)
parser.add_argument("--accept_diag_yaw_leak", type=float, default=0.35, help="Max |wz| allowed in diagonal acceptance.")
parser.add_argument("--accept_yaw_lin_leak", type=float, default=0.18, help="Max linear speed allowed in yaw acceptance.")
parser.add_argument("--accept_min_base_height", type=float, default=0.12, help="Min base height during yaw acceptance.")
parser.add_argument("--accept_max_fall_rate", type=float, default=0.20, help="Max fall-rate allowed per command.")
parser.add_argument("--accept_skill_pass_ratio", type=float, default=0.60, help="Skill-level pass ratio threshold.")
parser.add_argument("--accept_overall_pass_ratio", type=float, default=0.70, help="Overall command pass-ratio threshold.")
parser.add_argument(
    "--disable_auto_stage_from_checkpoint",
    action="store_true",
    default=False,
    help="Do not auto-set env.stage from checkpoint run-name suffix like *_stage4.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.strict_checkpoint_loading:
    if not args_cli.checkpoint or not args_cli.checkpoint_sha256:
        parser.error("--strict-checkpoint-loading requires --checkpoint and --checkpoint-sha256")
    _checkpoint_file = Path(args_cli.checkpoint)
    if not _checkpoint_file.is_file() or re.fullmatch(r"model_(\d+)\.pt", _checkpoint_file.name) is None:
        parser.error("strict checkpoint loading requires an existing model_*.pt file")
    _checkpoint_digest = hashlib.sha256(_checkpoint_file.read_bytes()).hexdigest()
    if _checkpoint_digest != args_cli.checkpoint_sha256.lower():
        parser.error(
            "checkpoint SHA-256 mismatch: "
            f"expected {args_cli.checkpoint_sha256.lower()}, got {_checkpoint_digest}"
        )
    _simulator_root = os.environ.get("ISAACSIM_ROOT")
    if not _simulator_root:
        parser.error("strict Autopilot evaluation requires ISAACSIM_ROOT")
    try:
        _EVALUATOR_DEPENDENCY_MANIFEST = build_dependency_manifest(
            _REDRHEX_REPO_ROOT,
            simulator_root=Path(_simulator_root),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(f"unable to attest Autopilot dependencies: {exc}")

if bool(args_cli.command_profile) != bool(args_cli.command_profile_sha256):
    parser.error("--command-profile and --command-profile-sha256 must be supplied together")
_command_profile_payload = None
if args_cli.command_profile:
    _command_profile_file = Path(args_cli.command_profile)
    if not _command_profile_file.is_file():
        parser.error("--command-profile must be an existing JSON file")
    _command_profile_digest = hashlib.sha256(_command_profile_file.read_bytes()).hexdigest()
    if _command_profile_digest != args_cli.command_profile_sha256.lower():
        parser.error(
            "command profile SHA-256 mismatch: "
            f"expected {args_cli.command_profile_sha256.lower()}, got {_command_profile_digest}"
        )
    try:
        _command_profile_payload = json.loads(_command_profile_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        parser.error(f"invalid command profile JSON: {exc}")

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import RedRhex.tasks  # noqa: F401
from scripts.rsl_rl.runner_factory import create_runner, get_exportable_actor, runner_protocol


def _load_runner_checkpoint_with_policy_fallback(runner, resume_path: str, device: str) -> None:
    """Load a checkpoint, falling back to actor-compatible weights for old critic shapes."""
    try:
        runner.load(resume_path)
        return
    except RuntimeError as exc:
        original_error = exc

    policy_module = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic
    checkpoint = torch.load(resume_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise original_error

    state_dict = None
    for key in ("model_state_dict", "policy_state_dict", "actor_critic_state_dict", "student_state_dict"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            state_dict = candidate
            break
    if state_dict is None:
        state_dict = checkpoint if all(hasattr(v, "shape") for v in checkpoint.values()) else None
    if state_dict is None:
        raise original_error

    current_state = policy_module.state_dict()
    compatible_state = {}
    for key, value in state_dict.items():
        if key in current_state and hasattr(value, "shape") and current_state[key].shape == value.shape:
            compatible_state[key] = value.to(device=current_state[key].device, dtype=current_state[key].dtype)
    if not compatible_state:
        raise original_error

    merged_state = dict(current_state)
    merged_state.update(compatible_state)
    policy_module.load_state_dict(merged_state, strict=True)
    skipped = len(state_dict) - len(compatible_state)
    print(
        "[WARN] Full checkpoint load failed, likely due to critic/privileged-observation shape changes. "
        f"Loaded {len(compatible_state)} actor-compatible tensors and skipped {skipped} tensors for evaluation."
    )


def _model_step_from_name(path: Path) -> int:
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else -1


def _pick_latest_model_checkpoint(run_dir: Path) -> Path | None:
    if not run_dir.exists() or not run_dir.is_dir():
        return None
    candidates = [p for p in run_dir.glob("model_*.pt") if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=_model_step_from_name)


def _pick_latest_model_checkpoint_recursive(root_dir: Path) -> Path | None:
    if not root_dir.exists() or not root_dir.is_dir():
        return None
    candidates = [p for p in root_dir.rglob("model_*.pt") if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _resolve_rsl_rl_checkpoint_path(
    raw_checkpoint: str,
    fallback_root: str | None = None,
    *,
    strict: bool = False,
) -> str:
    """Resolve user checkpoint arg to a valid rsl_rl training checkpoint (model_*.pt)."""
    resolved = Path(retrieve_file_path(raw_checkpoint))
    fallback_root_path = Path(fallback_root) if fallback_root is not None else None

    if strict:
        if not resolved.is_file() or re.fullmatch(r"model_(\d+)\.pt", resolved.name) is None:
            raise ValueError(f"Strict evaluation requires an exact model_*.pt file: {resolved}")
        return str(resolved)

    if resolved.is_dir():
        latest = _pick_latest_model_checkpoint(resolved)
        if latest is None:
            raise FileNotFoundError(
                f"No model_*.pt found under directory: {resolved}. "
                "Please pass a training checkpoint like .../model_11999.pt."
            )
        print(f"[WARN] --checkpoint points to a directory. Auto-select latest checkpoint: {latest}")
        return str(latest)

    is_training_ckpt = re.fullmatch(r"model_(\d+)\.pt", resolved.name) is not None
    if is_training_ckpt:
        return str(resolved)

    # Common user mistake: passing TensorBoard event file.
    if resolved.name.startswith("events.out.tfevents") or resolved.suffix != ".pt":
        latest = _pick_latest_model_checkpoint(resolved.parent)
        if latest is None:
            latest = _pick_latest_model_checkpoint_recursive(fallback_root_path) if fallback_root_path else None
        if latest is not None:
            print(
                "[WARN] --checkpoint is not a training checkpoint file. "
                f"Auto-fallback to latest model checkpoint: {latest}"
            )
            return str(latest)
        raise ValueError(
            f"Invalid checkpoint: {resolved}. Expected model_*.pt, but got '{resolved.name}'. "
            "No sibling model_*.pt found."
        )

    # .pt but not model_*.pt (e.g. exported policy.pt) is usually not loadable by runner.load()
    latest = _pick_latest_model_checkpoint(resolved.parent)
    if latest is None:
        latest = _pick_latest_model_checkpoint_recursive(fallback_root_path) if fallback_root_path else None
    if latest is not None:
        print(
            "[WARN] --checkpoint points to a non-training .pt file. "
            f"Auto-fallback to latest model checkpoint: {latest}"
        )
        return str(latest)
    raise ValueError(
        f"Invalid checkpoint: {resolved}. Expected training checkpoint model_*.pt, got '{resolved.name}'."
    )


def _infer_stage_from_checkpoint_path(path: str) -> int | None:
    lowered = path.lower()
    match = re.search(r"(?:^|[_/-])stage([1-5])(?:$|[_/-])", lowered)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def circular_distance(a: torch.Tensor, b: float | torch.Tensor) -> torch.Tensor:
    return torch.abs(torch.atan2(torch.sin(a - b), torch.cos(a - b)))


def in_stance_phase(unwrapped_env, phase: torch.Tensor) -> torch.Tensor:
    if hasattr(unwrapped_env, "_in_stance_phase"):
        return unwrapped_env._in_stance_phase(phase)
    stance_start = float(unwrapped_env.stance_phase_start)
    stance_end = float(unwrapped_env.stance_phase_end)
    if stance_start < 0:
        stance_start += 2.0 * math.pi
        return (phase >= stance_start) | (phase < stance_end)
    return (phase >= stance_start) & (phase < stance_end)


def summarize_contact_hist(contact_hist: torch.Tensor) -> str:
    total = int(contact_hist.sum().item())
    if total == 0:
        return "n/a"
    parts = []
    for k in range(contact_hist.shape[0]):
        pct = 100.0 * float(contact_hist[k].item()) / float(total)
        parts.append(f"{k}:{pct:.1f}%")
    return " ".join(parts)


def _runtime_source_identities() -> dict[str, str]:
    return {
        **source_code_identities(_REDRHEX_REPO_ROOT),
        "dependency": (
            dependency_manifest_sha256(_EVALUATOR_DEPENDENCY_MANIFEST)
            if _EVALUATOR_DEPENDENCY_MANIFEST is not None
            else ""
        ),
    }


def classify_command_skill(cmd: Sequence[float], eps: float = 1e-5) -> str:
    vx, vy, wz = float(cmd[0]), float(cmd[1]), float(cmd[2])
    if abs(wz) > eps and abs(vx) <= eps and abs(vy) <= eps:
        return "yaw"
    if abs(vx) > eps and abs(vy) > eps and abs(wz) <= eps:
        return "diagonal"
    if abs(vy) > eps and abs(vx) <= eps and abs(wz) <= eps:
        return "lateral"
    if abs(vx) > eps and abs(vy) <= eps and abs(wz) <= eps:
        return "forward"
    return "other"


def _linspace(lo: float, hi: float, count: int) -> list[float]:
    lo = float(lo)
    hi = float(hi)
    if count <= 1 or abs(hi - lo) < 1e-6:
        return [0.5 * (lo + hi)]
    return [lo + (hi - lo) * (float(i) / float(count - 1)) for i in range(count)]


def _to_triples(commands: Iterable[Sequence[float]]) -> list[tuple[float, float, float]]:
    triples: list[tuple[float, float, float]] = []
    for cmd in commands:
        if len(cmd) >= 3:
            triples.append((float(cmd[0]), float(cmd[1]), float(cmd[2])))
        elif len(cmd) == 2:
            triples.append((float(cmd[0]), float(cmd[1]), 0.0))
    return triples


def _name_command(cmd: tuple[float, float, float], skill: str, name_counts: dict[str, int]) -> str:
    vx, vy, wz = cmd
    if skill == "forward":
        base = "forward"
    elif skill == "lateral":
        base = "left" if vy >= 0.0 else "right"
    elif skill == "diagonal":
        base = "diag_left" if vy >= 0.0 else "diag_right"
    elif skill == "yaw":
        base = "yaw_ccw" if wz >= 0.0 else "yaw_cw"
    else:
        base = "cmd"
    name_counts[base] += 1
    if name_counts[base] == 1:
        return base
    return f"{base}_{name_counts[base]}"


def _generate_named_commands(commands: Iterable[Sequence[float]]) -> list[tuple[str, tuple[float, float, float], str]]:
    named: list[tuple[str, tuple[float, float, float], str]] = []
    seen: set[tuple[float, float, float]] = set()
    name_counts: defaultdict[str, int] = defaultdict(int)
    for cmd in _to_triples(commands):
        key = (round(cmd[0], 4), round(cmd[1], 4), round(cmd[2], 4))
        if key in seen:
            continue
        seen.add(key)
        skill = classify_command_skill(cmd)
        name = _name_command(cmd, skill, name_counts)
        named.append((name, cmd, skill))
    return named


def command_set_from_profile(
    payload: object,
    *,
    task: str,
    stage: int | None,
    evaluation_profile: str,
) -> list[tuple[str, tuple[float, float, float], str]]:
    """Validate an exact Autopilot command profile and return named commands."""

    if not isinstance(payload, dict):
        raise ValueError("command profile must be a JSON object")
    required = {
        "schema_version", "task", "stage", "evaluation_profile", "gait", "directions",
        "command_envelope", "commands",
    }
    if set(payload) != required:
        raise ValueError("command profile has an unexpected schema")
    if payload["schema_version"] != "redrhex.autopilot.command-profile.v1":
        raise ValueError("unsupported command profile schema")
    if payload["task"] != task or payload["stage"] != stage or payload["evaluation_profile"] != evaluation_profile:
        raise ValueError("command profile task/stage/evaluation identity mismatch")
    envelope = payload["command_envelope"]
    if not isinstance(envelope, dict) or set(envelope) != {"vx", "vy", "wz"}:
        raise ValueError("command profile envelope is invalid")
    parsed_envelope: dict[str, list[tuple[float, float]]] = {}
    for axis in ("vx", "vy", "wz"):
        raw_intervals = envelope[axis]
        if not isinstance(raw_intervals, list) or not raw_intervals:
            raise ValueError(f"command profile {axis} envelope is empty")
        parsed_envelope[axis] = []
        for interval in raw_intervals:
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError(f"command profile {axis} interval is invalid")
            if isinstance(interval[0], bool) or isinstance(interval[1], bool):
                raise ValueError(f"command profile {axis} interval must be numeric")
            low, high = float(interval[0]), float(interval[1])
            if not math.isfinite(low) or not math.isfinite(high) or low > high:
                raise ValueError(f"command profile {axis} interval is invalid")
            parsed_envelope[axis].append((low, high))
    raw_commands = payload["commands"]
    if not isinstance(raw_commands, list) or not raw_commands or len(raw_commands) > 64:
        raise ValueError("command profile must contain 1-64 commands")
    commands: list[tuple[str, tuple[float, float, float], str]] = []
    names: set[str] = set()
    for raw in raw_commands:
        if not isinstance(raw, dict) or set(raw) != {"name", "skill", "vx", "vy", "wz"}:
            raise ValueError("command profile command has an unexpected schema")
        name = raw["name"]
        skill = raw["skill"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("command profile command names must be non-empty and unique")
        if skill not in {"forward", "lateral", "diagonal", "yaw"}:
            raise ValueError("command profile command skill is invalid")
        values = []
        for axis in ("vx", "vy", "wz"):
            value = raw[axis]
            if isinstance(value, bool):
                raise ValueError("command profile commands must be numeric")
            number = float(value)
            if not math.isfinite(number) or not any(low <= number <= high for low, high in parsed_envelope[axis]):
                raise ValueError(f"command profile command lies outside the {axis} envelope")
            values.append(number)
        command = (values[0], values[1], values[2])
        if classify_command_skill(command) != skill:
            raise ValueError("command profile skill does not match its numeric command")
        names.add(name)
        commands.append((name, command, skill))
    return commands


def build_command_set(env_cfg, profile: str, command_scale: float) -> list[tuple[str, tuple[float, float, float], str]]:
    profile = profile.lower()
    scale = float(command_scale)

    def _scale(commands: Iterable[Sequence[float]]) -> list[tuple[float, float, float]]:
        out: list[tuple[float, float, float]] = []
        for vx, vy, wz in _to_triples(commands):
            out.append((vx * scale, vy * scale, wz * scale))
        return out

    def _stage1() -> list[tuple[float, float, float]]:
        if getattr(env_cfg, "stage1_use_discrete_directions", False):
            dirs = getattr(env_cfg, "stage1_discrete_directions", [[0.4, 0.0, 0.0]])
            return _scale(dirs)
        vx_min, vx_max = getattr(env_cfg, "stage1_forward_vx_range", [0.20, 0.45])
        vx_samples = [v for v in _linspace(vx_min, vx_max, 3) if v > 1e-4]
        return _scale([(vx, 0.0, 0.0) for vx in vx_samples])

    def _stage2() -> list[tuple[float, float, float]]:
        if getattr(env_cfg, "stage2_use_discrete_directions", False):
            dirs = getattr(env_cfg, "stage2_discrete_directions", [[0.0, 0.3, 0.0], [0.0, -0.3, 0.0]])
            return _scale(dirs)
        vy_min, vy_max = getattr(env_cfg, "stage2_lateral_vy_abs_range", [0.20, 0.40])
        vy_samples = [max(1e-4, abs(v)) for v in _linspace(vy_min, vy_max, 2)]
        cmds: list[tuple[float, float, float]] = []
        for vy in vy_samples:
            cmds.append((0.0, vy, 0.0))
            cmds.append((0.0, -vy, 0.0))
        return _scale(cmds)

    def _stage3() -> list[tuple[float, float, float]]:
        if getattr(env_cfg, "stage3_use_discrete_directions", True):
            dirs = getattr(env_cfg, "stage3_discrete_directions", [[0.3, 0.2, 0.0], [0.3, -0.2, 0.0]])
            return _scale(dirs)
        vx_min, vx_max = getattr(env_cfg, "stage3_diag_vx_range", [0.22, 0.40])
        vy_min, vy_max = getattr(env_cfg, "stage3_diag_vy_abs_range", [0.18, 0.30])
        vx_samples = _linspace(vx_min, vx_max, 2)
        vy_samples = _linspace(vy_min, vy_max, 2)
        cmds = []
        for vx, vy in zip(vx_samples, vy_samples):
            cmds.append((vx, abs(vy), 0.0))
            cmds.append((vx, -abs(vy), 0.0))
        return _scale(cmds)

    def _stage4() -> list[tuple[float, float, float]]:
        if getattr(env_cfg, "stage4_use_discrete_directions", True):
            dirs = getattr(env_cfg, "stage4_discrete_directions", [[0.0, 0.0, 0.65], [0.0, 0.0, -0.65]])
            return _scale(dirs)
        wz_min, wz_max = getattr(env_cfg, "stage4_yaw_wz_abs_range", [0.35, 0.75])
        wz_samples = [max(1e-4, abs(w)) for w in _linspace(wz_min, wz_max, 2)]
        cmds = []
        for wz in wz_samples:
            cmds.append((0.0, 0.0, wz))
            cmds.append((0.0, 0.0, -wz))
        return _scale(cmds)

    def _stage5() -> list[tuple[float, float, float]]:
        dirs = getattr(env_cfg, "stage5_discrete_directions", None)
        if dirs is None or len(dirs) == 0:
            dirs = [
                [0.40, 0.00, 0.00],
                [0.00, 0.30, 0.00],
                [0.00, -0.30, 0.00],
                [0.30, 0.20, 0.00],
                [0.30, -0.20, 0.00],
                [0.00, 0.00, 0.70],
                [0.00, 0.00, -0.70],
            ]
        return _scale(dirs)

    if profile == "stage1":
        commands = _stage1()
    elif profile == "stage2":
        commands = _stage2()
    elif profile == "stage3":
        commands = _stage3()
    elif profile == "stage4":
        commands = _stage4()
    elif profile == "full":
        commands = _stage1() + _stage2() + _stage3() + _stage4() + _stage5()
    else:
        commands = _stage5()

    return _generate_named_commands(commands)


def collect_energy_metrics(
    unwrapped_env,
    actual_vx: torch.Tensor,
    actual_vy: torch.Tensor,
    actual_wz: torch.Tensor,
    cmd_vx: float,
    cmd_vy: float,
) -> dict[str, torch.Tensor]:
    """Collect torque-backed energy metrics.

    Translational commands use mechanical energy per forward-progress distance.
    Pure-yaw commands use absolute total mechanical power so their energy gate
    cannot collapse to a vacuous zero when commanded translation is absent.
    """
    num_envs = unwrapped_env.num_envs
    device = unwrapped_env.device
    zeros = torch.zeros(num_envs, device=device)

    lin_xy = torch.stack((actual_vx, actual_vy), dim=1)
    motion_speed = torch.linalg.norm(lin_xy, dim=1)
    cmd_lin = torch.stack(
        (
            torch.full_like(actual_vx, float(cmd_vx)),
            torch.full_like(actual_vy, float(cmd_vy)),
        ),
        dim=1,
    )
    cmd_lin_speed = torch.linalg.norm(cmd_lin, dim=1)
    safe_cmd_lin_speed = torch.clamp(cmd_lin_speed, min=1e-6)
    cmd_dir = cmd_lin / safe_cmd_lin_speed.unsqueeze(1)
    min_cmd_motion = float(
        getattr(
            unwrapped_env,
            "_energy_command_threshold",
            getattr(unwrapped_env.cfg, "energy_command_threshold", 0.05),
        )
    )
    distance_eps = max(
        float(getattr(unwrapped_env, "_energy_distance_eps", getattr(unwrapped_env.cfg, "energy_distance_eps", 1e-4))),
        1e-8,
    )
    energy_max = max(
        float(getattr(unwrapped_env, "_energy_per_distance_max", getattr(unwrapped_env.cfg, "energy_per_distance_max", 500.0))),
        distance_eps,
    )
    progress_speed = torch.sum(lin_xy * cmd_dir, dim=1)
    progress_speed = torch.where(
        cmd_lin_speed > min_cmd_motion,
        torch.clamp(progress_speed, min=0.0),
        torch.zeros_like(progress_speed),
    )

    main_power = zeros.clone()
    abad_power = zeros.clone()
    total_power = zeros.clone()

    if hasattr(unwrapped_env, "_main_drive_indices"):
        main_omega = unwrapped_env.joint_vel[:, unwrapped_env._main_drive_indices]
        abad_omega = (
            unwrapped_env.joint_vel[:, unwrapped_env._abad_indices]
            if hasattr(unwrapped_env, "_abad_indices")
            else None
        )
        main_torque = None
        abad_torque = None

        if hasattr(unwrapped_env, "_get_active_joint_torques") and abad_omega is not None:
            try:
                main_torque, abad_torque = unwrapped_env._get_active_joint_torques(main_omega, abad_omega)
            except Exception:
                main_torque, abad_torque = None, None

        if main_torque is None and hasattr(unwrapped_env.robot.data, "applied_torque"):
            main_torque = unwrapped_env.robot.data.applied_torque[:, unwrapped_env._main_drive_indices]
            if abad_omega is not None and hasattr(unwrapped_env, "_abad_indices"):
                abad_torque = unwrapped_env.robot.data.applied_torque[:, unwrapped_env._abad_indices]

        if main_torque is not None:
            main_power = torch.sum(torch.abs(main_torque * main_omega), dim=1)
            if abad_torque is not None and abad_omega is not None:
                abad_power = torch.sum(torch.abs(abad_torque * abad_omega), dim=1)
            total_power = main_power + abad_power
        elif args_cli.strict_energy_evidence:
            raise RuntimeError(
                "Strict energy evidence requires simulator-applied or environment-resolved joint torques."
            )
        elif hasattr(unwrapped_env, "_target_drive_vel"):
            # Last-resort proxy used only when torque is unavailable.
            main_power = torch.sum(torch.abs(unwrapped_env._target_drive_vel * main_omega), dim=1)
            total_power = main_power
    elif args_cli.strict_energy_evidence:
        raise RuntimeError(
            "Strict energy evidence requires resolved main-drive joint indices and torques."
        )

    spring_energy = zeros.clone()
    spring_release = zeros.clone()
    spring_store = zeros.clone()
    spring_recovery_ratio = zeros.clone()
    damper_dissipation = zeros.clone()

    if hasattr(unwrapped_env, "_damper_indices") and hasattr(unwrapped_env, "_damper_initial_pos"):
        wrapped_damp_pos = unwrapped_env.joint_pos[:, unwrapped_env._damper_indices]
        damp_pos = getattr(unwrapped_env, "_spring_unwrapped_pos", wrapped_damp_pos)
        damp_vel = unwrapped_env.joint_vel[:, unwrapped_env._damper_indices]
        spring_rest = getattr(
            unwrapped_env, "_spring_rest_pos", unwrapped_env._damper_initial_pos
        )
        damp_defl = damp_pos - spring_rest
        spring_k = torch.as_tensor(getattr(unwrapped_env, "_spring_k", 200.0), device=damp_defl.device, dtype=damp_defl.dtype)
        spring_d = torch.as_tensor(getattr(unwrapped_env, "_spring_d", 0.0), device=damp_vel.device, dtype=damp_vel.dtype)
        spring_energy = torch.sum(0.5 * spring_k * torch.square(damp_defl), dim=1)
        spring_power = spring_k * damp_defl * damp_vel
        if hasattr(unwrapped_env, "_current_leg_in_stance") and unwrapped_env._current_leg_in_stance.shape == damp_defl.shape:
            contact_mask = unwrapped_env._current_leg_in_stance.float()
        else:
            contact_mask = torch.ones_like(damp_defl)
        spring_release = torch.sum(torch.clamp(-spring_power, min=0.0) * contact_mask, dim=1)
        spring_store = torch.sum(torch.clamp(spring_power, min=0.0) * contact_mask, dim=1)
        spring_recovery_ratio = spring_release / (spring_release + spring_store + 1e-6)
        damper_dissipation = torch.sum(spring_d * torch.square(damp_vel), dim=1)

    robot_mass = float(getattr(unwrapped_env, "_robot_mass", getattr(unwrapped_env.cfg, "robot_mass_kg", 14.0)))
    dt = float(getattr(unwrapped_env, "step_dt", getattr(getattr(unwrapped_env.cfg, "sim", None), "dt", 1.0 / 60.0)))
    energy_cost = total_power * dt
    progress_distance = progress_speed * dt
    has_translation_cmd = cmd_lin_speed > min_cmd_motion
    energy_per_distance = energy_cost / torch.clamp(progress_distance, min=distance_eps)
    energy_per_distance = torch.where(
        has_translation_cmd & (progress_distance <= distance_eps),
        torch.full_like(energy_per_distance, energy_max),
        energy_per_distance,
    )
    # The V1 gate is an absolute energy-effort ceiling. Pure yaw has no linear
    # progress distance, so use total mechanical power rather than reporting
    # zero and allowing unbounded yaw effort to pass.
    energy_per_distance = torch.where(has_translation_cmd, energy_per_distance, total_power)
    energy_per_distance = torch.clamp(
        torch.nan_to_num(energy_per_distance, nan=energy_max, posinf=energy_max),
        max=energy_max,
    )
    cot_proxy = total_power / (robot_mass * 9.81 * (motion_speed + 0.1))

    return {
        "motion_speed": motion_speed,
        "progress_speed": progress_speed,
        "energy_cost": energy_cost,
        "progress_distance": progress_distance,
        "mech_power_main": main_power,
        "mech_power_abad": abad_power,
        "mech_power_total": total_power,
        "cot_proxy": cot_proxy,
        "energy_per_distance": energy_per_distance,
        "spring_energy": spring_energy,
        "spring_release": spring_release,
        "spring_store": spring_store,
        "spring_recovery_ratio": spring_recovery_ratio,
        "damper_dissipation": damper_dissipation,
    }


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    runtime_identities = _runtime_source_identities()
    if args_cli.strict_checkpoint_loading:
        missing_identities = [
            name
            for name in (
                "code", "config", "dependency", "reward_profile", "physics", "spring", "terrain"
            )
            if getattr(args_cli, f"identity_{name}_sha256") is None
        ]
        if missing_identities:
            raise RuntimeError(
                "Strict Autopilot evaluation requires every frozen identity: "
                + ", ".join(missing_identities)
            )
    for name, actual in runtime_identities.items():
        expected = getattr(args_cli, f"identity_{name}_sha256")
        if expected is not None and expected != actual:
            raise RuntimeError(
                f"Frozen Autopilot {name} identity mismatch: expected {expected}, got {actual}"
            )
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.spring_backend = args_cli.spring_backend

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    if args_cli.checkpoint:
        resume_path = _resolve_rsl_rl_checkpoint_path(
            args_cli.checkpoint,
            fallback_root=log_root_path,
            strict=args_cli.strict_checkpoint_loading,
        )
    else:
        # Some configs may resolve to tensorboard event files by default (e.g. checkpt1/events...).
        # Always sanitize to a real training checkpoint (model_*.pt).
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        resume_path = _resolve_rsl_rl_checkpoint_path(resume_path, fallback_root=log_root_path)

    if args_cli.curriculum_stage is not None:
        if not hasattr(env_cfg, "stage"):
            raise ValueError("--curriculum-stage is not supported by the selected task")
        env_cfg.stage = int(args_cli.curriculum_stage)
    elif not args_cli.disable_auto_stage_from_checkpoint and hasattr(env_cfg, "stage"):
        inferred_stage = _infer_stage_from_checkpoint_path(resume_path)
        if inferred_stage is not None:
            prev_stage = int(getattr(env_cfg, "stage"))
            env_cfg.stage = inferred_stage
            if prev_stage != inferred_stage:
                print(
                    f"[INFO] Auto-set env.stage from {prev_stage} to {inferred_stage} "
                    f"based on checkpoint path: {resume_path}"
                )

    env_cfg.log_dir = os.path.dirname(resume_path)
    physics_profile = None
    spring_profile_id = None
    spring_profile_sha256 = None
    if args_cli.physics_profile is not None:
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from tools.sim2real.physics_profile import apply_profile_to_config, load_optional_profile
        from tools.sim2real.traces import sha256_json

        physics_profile = load_optional_profile(args_cli.physics_profile)
        apply_profile_to_config(env_cfg, physics_profile)
        spring_profile_id = physics_profile.profile_id
        spring_profile_sha256 = sha256_json(physics_profile.to_dict())
        print(
            f"[INFO] Explicit physics profile applied to config: {spring_profile_id} "
            f"({spring_profile_sha256})"
        )

    from tools.sim2real.checkpoint_spring import validate_checkpoint_spring_evaluation

    checkpoint_spring_calibration_status = validate_checkpoint_spring_evaluation(
        env_cfg.log_dir,
        selected_backend=args_cli.spring_backend,
        selected_profile_id=spring_profile_id,
        selected_profile_sha256=spring_profile_sha256,
    )
    spring_calibration_status = (
        "calibrated" if bool(getattr(env_cfg, "spring_calibrated", False)) else "uncalibrated"
    )
    if checkpoint_spring_calibration_status == "calibrated" and spring_calibration_status != "calibrated":
        raise RuntimeError("calibrated checkpoint profile did not produce a calibrated spring configuration")
    print(
        f"[INFO] Torsion spring backend={args_cli.spring_backend}, "
        f"calibration_status={spring_calibration_status}, "
        f"checkpoint_calibration_status={checkpoint_spring_calibration_status}, "
        f"profile_id={spring_profile_id}, profile_sha256={spring_profile_sha256}"
    )
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    if physics_profile is not None:
        from tools.sim2real.isaac_profile import apply_profile_to_runtime_env

        apply_profile_to_runtime_env(env, physics_profile)
        print(f"[INFO] Explicit physics profile applied at runtime: {spring_profile_id}")

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    protocol = runner_protocol(agent_cfg.class_name)
    runner = create_runner(
        agent_cfg.class_name,
        env,
        agent_cfg.to_dict(),
        log_dir=None,
        device=agent_cfg.device,
    )

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if protocol.strict_checkpoint:
        runner.load(resume_path, load_optimizer=False)
    elif args_cli.strict_checkpoint_loading:
        try:
            runner.load(resume_path, load_optimizer=False)
        except TypeError:
            runner.load(resume_path)
    else:
        _load_runner_checkpoint_with_policy_fallback(runner, resume_path, env.unwrapped.device)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    policy_nn = get_exportable_actor(runner, protocol)

    unwrapped_env = env.unwrapped
    if hasattr(unwrapped_env, "external_control"):
        unwrapped_env.external_control = True

    command_set = (
        command_set_from_profile(
            _command_profile_payload,
            task=args_cli.task,
            stage=args_cli.curriculum_stage,
            evaluation_profile=args_cli.eval_profile,
        )
        if _command_profile_payload is not None
        else build_command_set(env_cfg, args_cli.eval_profile, args_cli.command_scale)
    )
    if len(command_set) == 0:
        raise RuntimeError(f"No commands generated for eval profile: {args_cli.eval_profile}")
    print(f"[INFO] Eval profile: {args_cli.eval_profile}, command_scale={args_cli.command_scale:.2f}")
    print("[INFO] Command set:")
    for name, cmd, skill in command_set:
        print(f"  - {name:<14} skill={skill:<8} cmd=({cmd[0]:+.2f}, {cmd[1]:+.2f}, {cmd[2]:+.2f})")

    results = []
    episode_results = []
    num_envs = unwrapped_env.num_envs
    device = unwrapped_env.device
    total_steps = args_cli.warmup_steps + args_cli.sweep_steps
    step_dt = float(getattr(unwrapped_env, "step_dt", unwrapped_env.cfg.sim.dt * unwrapped_env.cfg.decimation))
    if args_cli.expected_step_dt is not None and (
        not math.isfinite(args_cli.expected_step_dt)
        or not math.isclose(
            step_dt,
            float(args_cli.expected_step_dt),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise RuntimeError(
            "Evaluation policy timestep does not match --expected-step-dt: "
            f"runtime={step_dt!r}, expected={args_cli.expected_step_dt!r}"
        )
    eval_duration_s = float(args_cli.sweep_steps) * step_dt

    # Global acceptance accumulators
    sample_count = 0
    err_vx_sum = 0.0
    err_vy_sum = 0.0
    err_wz_sum = 0.0

    base_h_sum = 0.0
    base_h_sq_sum = 0.0
    pitch_sq_sum = 0.0
    roll_sq_sum = 0.0
    stability_count = 0

    fall_events = 0
    episode_ends = 0

    contact_hist = torch.zeros(7, dtype=torch.long)
    transition_samples = 0
    transition_contact_ge4 = 0

    action_rate_sum = 0.0
    effort_proxy_sum = 0.0
    energy_count = 0
    effort_proxy_name = "mean(|target_vel * omega|)"

    # ★ Energy-aware KPIs
    spring_energy_sum = 0.0
    spring_release_sum = 0.0
    spring_store_sum = 0.0
    mech_power_main_sum = 0.0
    mech_power_total_sum = 0.0
    cot_proxy_sum = 0.0
    damper_dissipation_sum = 0.0
    motion_speed_sum = 0.0
    progress_speed_sum = 0.0
    energy_cost_sum = 0.0
    progress_distance_sum = 0.0
    energy_per_distance_sum = 0.0
    energy_kpi_count = 0

    forward_phase_diff_sum = 0.0
    forward_phase_err_to_pi_sum = 0.0
    forward_stance_frac_sum = 0.0
    forward_stance_frac_err_sum = 0.0
    forward_stance_speed_sum = 0.0
    forward_swing_speed_sum = 0.0
    forward_count = 0

    skill_total: defaultdict[str, int] = defaultdict(int)
    skill_pass: defaultdict[str, int] = defaultdict(int)
    skill_energy_steps: defaultdict[str, int] = defaultdict(int)
    skill_mech_power_total_sum: defaultdict[str, float] = defaultdict(float)
    skill_cot_sum: defaultdict[str, float] = defaultdict(float)
    skill_spring_recovery_ratio_sum: defaultdict[str, float] = defaultdict(float)
    skill_motion_speed_sum: defaultdict[str, float] = defaultdict(float)
    skill_progress_speed_sum: defaultdict[str, float] = defaultdict(float)
    skill_energy_cost_sum: defaultdict[str, float] = defaultdict(float)
    skill_progress_distance_sum: defaultdict[str, float] = defaultdict(float)
    skill_energy_per_distance_sum: defaultdict[str, float] = defaultdict(float)
    score_sum = 0.0

    for name, cmd, skill in command_set:
        env.reset()
        obs = env.get_observations()

        cmd_tensor = torch.tensor(cmd, device=device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1)
        cmd_err_vx = 0.0
        cmd_err_vy = 0.0
        cmd_err_wz = 0.0
        cmd_forward_speed_sum = 0.0
        cmd_lateral_leak_sum = 0.0
        cmd_yaw_leak_sum = 0.0
        cmd_samples = 0
        cmd_success_steps = 0
        cmd_success_vy_steps = 0
        cmd_success_wz_steps = 0
        cmd_diag_sign_match = 0
        cmd_diag_sign_total = 0
        cmd_yaw_tilt_ok_steps = 0
        cmd_fall_events = 0
        cmd_episode_ends = 0
        cmd_mech_power_main_sum = 0.0
        cmd_mech_power_total_sum = 0.0
        cmd_cot_sum = 0.0
        cmd_spring_energy_sum = 0.0
        cmd_spring_release_sum = 0.0
        cmd_spring_store_sum = 0.0
        cmd_spring_recovery_ratio_sum = 0.0
        cmd_motion_speed_sum = 0.0
        cmd_progress_speed_sum = 0.0
        cmd_energy_cost_sum = 0.0
        cmd_progress_distance_sum = 0.0
        cmd_energy_per_distance_sum = 0.0
        cmd_energy_steps = 0

        episode_number = torch.zeros(num_envs, dtype=torch.long, device=device)
        episode_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        episode_err_vx = torch.zeros(num_envs, dtype=torch.float64, device=device)
        episode_err_vy = torch.zeros(num_envs, dtype=torch.float64, device=device)
        episode_err_wz = torch.zeros(num_envs, dtype=torch.float64, device=device)
        episode_success = torch.zeros(num_envs, dtype=torch.long, device=device)
        episode_falls = torch.zeros(num_envs, dtype=torch.long, device=device)
        episode_energy = torch.zeros(num_envs, dtype=torch.float64, device=device)
        episode_energy_effort = torch.zeros(num_envs, dtype=torch.float64, device=device)

        def flush_episode_evidence(indices: torch.Tensor, *, complete: bool) -> None:
            for env_index in indices.detach().cpu().tolist():
                steps = int(episode_steps[env_index].item())
                if steps <= 0:
                    continue
                episode_results.append(
                    {
                        "command": name,
                        "skill": skill,
                        "environment_index": int(env_index),
                        "episode_index": int(episode_number[env_index].item()),
                        "complete": bool(complete),
                        "sample_count": steps,
                        "fall_count": int(episode_falls[env_index].item()),
                        "mae_vx": float(episode_err_vx[env_index].item()) / steps,
                        "mae_vy": float(episode_err_vy[env_index].item()) / steps,
                        "mae_wz": float(episode_err_wz[env_index].item()) / steps,
                        "success_ratio": float(episode_success[env_index].item()) / steps,
                        "energy_mech_power_total_mean": float(episode_energy[env_index].item()) / steps,
                        "energy_effort_mean": float(episode_energy_effort[env_index].item()) / steps,
                    }
                )
            if indices.numel() == 0:
                return
            episode_number[indices] += 1
            for accumulator in (
                episode_steps,
                episode_err_vx,
                episode_err_vy,
                episode_err_wz,
                episode_success,
                episode_falls,
                episode_energy,
                episode_energy_effort,
            ):
                accumulator[indices] = 0

        last_actions = None

        for step in range(total_steps):
            if hasattr(unwrapped_env, "commands"):
                unwrapped_env.commands[:] = cmd_tensor

            # Use no_grad instead of inference_mode: inference tensors can break subsequent env.reset() writes.
            with torch.no_grad():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                policy_nn.reset(dones)

            if step < args_cli.warmup_steps:
                last_actions = actions.clone()
                continue

            actual_vx = unwrapped_env.base_lin_vel[:, 0]
            actual_vy = unwrapped_env.base_lin_vel[:, 1]
            actual_wz = unwrapped_env.base_ang_vel[:, 2]
            actual_lin_speed = torch.sqrt(actual_vx**2 + actual_vy**2)

            dvx = torch.abs(actual_vx - cmd[0])
            dvy = torch.abs(actual_vy - cmd[1])
            dwz = torch.abs(actual_wz - cmd[2])

            cmd_err_vx += dvx.sum().item()
            cmd_err_vy += dvy.sum().item()
            cmd_err_wz += dwz.sum().item()
            if skill == "forward":
                forward_sign = 1.0 if float(cmd[0]) > 0.0 else -1.0
                cmd_forward_speed_sum += (actual_vx * forward_sign).sum().item()
                cmd_lateral_leak_sum += torch.abs(actual_vy).sum().item()
                cmd_yaw_leak_sum += torch.abs(actual_wz).sum().item()
            cmd_samples += num_envs
            episode_steps += 1
            episode_err_vx += dvx.to(dtype=torch.float64)
            episode_err_vy += dvy.to(dtype=torch.float64)
            episode_err_wz += dwz.to(dtype=torch.float64)

            err_vx_sum += dvx.sum().item()
            err_vy_sum += dvy.sum().item()
            err_wz_sum += dwz.sum().item()
            sample_count += num_envs

            # Stability
            base_h = unwrapped_env.robot.data.root_pos_w[:, 2]
            gravity_body = unwrapped_env.projected_gravity
            roll = torch.atan2(gravity_body[:, 1], -gravity_body[:, 2])
            pitch = torch.atan2(-gravity_body[:, 0], torch.sqrt(gravity_body[:, 1] ** 2 + gravity_body[:, 2] ** 2))

            base_h_sum += base_h.sum().item()
            base_h_sq_sum += (base_h**2).sum().item()
            pitch_sq_sum += (pitch**2).sum().item()
            roll_sq_sum += (roll**2).sum().item()
            stability_count += num_envs

            terminated = torch.zeros(num_envs, dtype=torch.bool, device=device)
            time_outs = torch.zeros(num_envs, dtype=torch.bool, device=device)
            if hasattr(unwrapped_env, "reset_terminated") and hasattr(unwrapped_env, "reset_time_outs"):
                terminated = unwrapped_env.reset_terminated
                time_outs = unwrapped_env.reset_time_outs
                fall_events += int(torch.count_nonzero(terminated).item())
                episode_ends += int(torch.count_nonzero(terminated | time_outs).item())
                cmd_fall_events += int(torch.count_nonzero(terminated).item())
                cmd_episode_ends += int(torch.count_nonzero(terminated | time_outs).item())
                episode_falls += terminated.to(dtype=torch.long)

            # Contact statistics
            if hasattr(unwrapped_env, "_contact_count"):
                contact_count = torch.clamp(torch.round(unwrapped_env._contact_count), min=0, max=6).to(torch.long)
            elif hasattr(unwrapped_env, "_current_leg_in_stance"):
                contact_count = unwrapped_env._current_leg_in_stance.float().sum(dim=1).to(torch.long)
            else:
                contact_count = torch.zeros(num_envs, dtype=torch.long, device=device)
            contact_hist += torch.bincount(contact_count.cpu(), minlength=7)

            # Forward gait metrics
            if skill == "forward" and hasattr(unwrapped_env, "_main_drive_indices"):
                main_pos = unwrapped_env.joint_pos[:, unwrapped_env._main_drive_indices]
                main_vel = unwrapped_env.joint_vel[:, unwrapped_env._main_drive_indices]
                direction_multiplier = unwrapped_env._direction_multiplier
                leg_phase = torch.remainder(main_pos * direction_multiplier, 2.0 * math.pi)
                leg_in_stance = in_stance_phase(unwrapped_env, leg_phase)

                phase_a = leg_phase[:, unwrapped_env._tripod_a_indices]
                phase_b = leg_phase[:, unwrapped_env._tripod_b_indices]
                mean_phase_a = torch.atan2(torch.sin(phase_a).mean(dim=1), torch.cos(phase_a).mean(dim=1))
                mean_phase_b = torch.atan2(torch.sin(phase_b).mean(dim=1), torch.cos(phase_b).mean(dim=1))

                phase_diff = circular_distance(mean_phase_a, mean_phase_b)
                forward_phase_diff_sum += phase_diff.mean().item()
                forward_phase_err_to_pi_sum += torch.abs(phase_diff - math.pi).mean().item()

                stance_fraction = leg_in_stance.float().mean(dim=1)
                forward_stance_frac_sum += stance_fraction.mean().item()
                forward_stance_frac_err_sum += torch.abs(stance_fraction - 0.65).mean().item()

                signed_speed = torch.abs(main_vel * direction_multiplier)
                stance_mask = leg_in_stance.float()
                swing_mask = (~leg_in_stance).float()
                stance_speed = (signed_speed * stance_mask).sum(dim=1) / stance_mask.sum(dim=1).clamp(min=1.0)
                swing_speed = (signed_speed * swing_mask).sum(dim=1) / swing_mask.sum(dim=1).clamp(min=1.0)

                forward_stance_speed_sum += stance_speed.mean().item()
                forward_swing_speed_sum += swing_speed.mean().item()
                forward_count += 1

                start_phase = unwrapped_env.stance_phase_start
                if start_phase < 0:
                    start_phase += 2.0 * math.pi
                end_phase = unwrapped_env.stance_phase_end
                transition_window = float(getattr(unwrapped_env.cfg, "forward_transition_window", 0.35))

                dist_a = torch.minimum(circular_distance(mean_phase_a, start_phase), circular_distance(mean_phase_a, end_phase))
                dist_b = torch.minimum(circular_distance(mean_phase_b, start_phase), circular_distance(mean_phase_b, end_phase))
                transition_mask = torch.minimum(dist_a, dist_b) < transition_window

                transition_samples += int(torch.count_nonzero(transition_mask).item())
                transition_contact_ge4 += int(torch.count_nonzero((contact_count >= 4) & transition_mask).item())

            # Energy / smoothness
            if last_actions is not None:
                action_rate = torch.linalg.vector_norm(actions - last_actions, dim=1)
                action_rate_sum += action_rate.mean().item()
            last_actions = actions.clone()

            # Per-command acceptance counters
            not_fallen = ~terminated
            abs_cmd_vx = abs(float(cmd[0]))
            abs_cmd_vy = abs(float(cmd[1]))
            abs_cmd_wz = abs(float(cmd[2]))
            success_mask = not_fallen.clone()

            if skill == "forward":
                vx_req = max(args_cli.accept_vx_abs, args_cli.accept_lin_ratio * abs_cmd_vx)
                success_mask = (
                    (actual_vx * float(cmd[0]) > 0.0)
                    & (torch.abs(actual_vx) >= vx_req)
                    & (torch.abs(actual_vy) <= args_cli.accept_forward_lateral_leak)
                    & (torch.abs(actual_wz) <= args_cli.accept_forward_yaw_leak)
                    & not_fallen
                )
            elif skill == "lateral":
                vy_req = max(args_cli.accept_vy_abs, args_cli.accept_lin_ratio * abs_cmd_vy)
                success_mask = (
                    (actual_vy * float(cmd[1]) > 0.0)
                    & (torch.abs(actual_vy) >= vy_req)
                    & (torch.abs(actual_vx) <= args_cli.accept_lateral_forward_leak)
                    & (torch.abs(actual_wz) <= args_cli.accept_lateral_yaw_leak)
                    & not_fallen
                )
                cmd_success_vy_steps += int(torch.count_nonzero(success_mask).item())
            elif skill == "diagonal":
                vx_req = max(0.08, args_cli.accept_diag_component_ratio * abs_cmd_vx)
                vy_req = max(0.08, args_cli.accept_diag_component_ratio * abs_cmd_vy)
                diag_sign_ok = (actual_vx * float(cmd[0]) > 0.0) & (actual_vy * float(cmd[1]) > 0.0)
                cmd_diag_sign_match += int(torch.count_nonzero(diag_sign_ok).item())
                cmd_diag_sign_total += int(diag_sign_ok.numel())
                success_mask = (
                    diag_sign_ok
                    & (torch.abs(actual_vx) >= vx_req)
                    & (torch.abs(actual_vy) >= vy_req)
                    & (torch.abs(actual_wz) <= args_cli.accept_diag_yaw_leak)
                    & not_fallen
                )
            elif skill == "yaw":
                wz_req = max(args_cli.accept_wz_abs, args_cli.accept_wz_ratio * abs_cmd_wz)
                tilt_ok = (torch.abs(roll) <= args_cli.accept_yaw_tilt_bound) & (
                    torch.abs(pitch) <= args_cli.accept_yaw_tilt_bound
                )
                cmd_yaw_tilt_ok_steps += int(torch.count_nonzero(tilt_ok & not_fallen).item())
                success_mask = (
                    (actual_wz * float(cmd[2]) > 0.0)
                    & (torch.abs(actual_wz) >= wz_req)
                    & tilt_ok
                    & (actual_lin_speed <= args_cli.accept_yaw_lin_leak)
                    & (base_h >= args_cli.accept_min_base_height)
                    & not_fallen
                )
                cmd_success_wz_steps += int(torch.count_nonzero(success_mask).item())

            cmd_success_steps += int(torch.count_nonzero(success_mask).item())
            episode_success += success_mask.to(dtype=torch.long)

            energy_metrics = collect_energy_metrics(
                unwrapped_env,
                actual_vx,
                actual_vy,
                actual_wz,
                float(cmd[0]),
                float(cmd[1]),
            )
            if torch.count_nonzero(energy_metrics["mech_power_total"]).item() > 0:
                effort_proxy_name = "mean(total_mech_power)"
                effort_proxy_sum += energy_metrics["mech_power_total"].mean().item()
            elif hasattr(unwrapped_env, "_target_drive_vel"):
                effort_proxy_name = "mean(|target_vel * omega|)"
                omegas = unwrapped_env.joint_vel[:, unwrapped_env._main_drive_indices]
                effort_proxy = torch.mean(torch.abs(unwrapped_env._target_drive_vel * omegas), dim=1)
                effort_proxy_sum += effort_proxy.mean().item()
            else:
                effort_proxy_sum += 0.0
            energy_count += 1

            spring_energy_sum += energy_metrics["spring_energy"].mean().item()
            spring_release_sum += energy_metrics["spring_release"].mean().item()
            spring_store_sum += energy_metrics["spring_store"].mean().item()
            damper_dissipation_sum += energy_metrics["damper_dissipation"].mean().item()
            mech_power_main_sum += energy_metrics["mech_power_main"].mean().item()
            mech_power_total_sum += energy_metrics["mech_power_total"].mean().item()
            cot_proxy_sum += energy_metrics["cot_proxy"].mean().item()
            motion_speed_sum += energy_metrics["motion_speed"].mean().item()
            progress_speed_sum += energy_metrics["progress_speed"].mean().item()
            energy_cost_sum += energy_metrics["energy_cost"].mean().item()
            progress_distance_sum += energy_metrics["progress_distance"].mean().item()
            energy_per_distance_sum += energy_metrics["energy_per_distance"].mean().item()

            cmd_mech_power_main_sum += energy_metrics["mech_power_main"].mean().item()
            cmd_mech_power_total_sum += energy_metrics["mech_power_total"].mean().item()
            cmd_cot_sum += energy_metrics["cot_proxy"].mean().item()
            cmd_spring_energy_sum += energy_metrics["spring_energy"].mean().item()
            cmd_spring_release_sum += energy_metrics["spring_release"].mean().item()
            cmd_spring_store_sum += energy_metrics["spring_store"].mean().item()
            cmd_spring_recovery_ratio_sum += energy_metrics["spring_recovery_ratio"].mean().item()
            cmd_motion_speed_sum += energy_metrics["motion_speed"].mean().item()
            cmd_progress_speed_sum += energy_metrics["progress_speed"].mean().item()
            cmd_energy_cost_sum += energy_metrics["energy_cost"].mean().item()
            cmd_progress_distance_sum += energy_metrics["progress_distance"].mean().item()
            cmd_energy_per_distance_sum += energy_metrics["energy_per_distance"].mean().item()
            cmd_energy_steps += 1
            episode_energy += energy_metrics["mech_power_total"].to(dtype=torch.float64)
            episode_energy_effort += energy_metrics["energy_per_distance"].to(dtype=torch.float64)

            skill_mech_power_total_sum[skill] += energy_metrics["mech_power_total"].mean().item()
            skill_cot_sum[skill] += energy_metrics["cot_proxy"].mean().item()
            skill_spring_recovery_ratio_sum[skill] += energy_metrics["spring_recovery_ratio"].mean().item()
            skill_motion_speed_sum[skill] += energy_metrics["motion_speed"].mean().item()
            skill_progress_speed_sum[skill] += energy_metrics["progress_speed"].mean().item()
            skill_energy_cost_sum[skill] += energy_metrics["energy_cost"].mean().item()
            skill_progress_distance_sum[skill] += energy_metrics["progress_distance"].mean().item()
            skill_energy_per_distance_sum[skill] += energy_metrics["energy_per_distance"].mean().item()
            skill_energy_steps[skill] += 1
            energy_kpi_count += 1

            done_indices = torch.nonzero(terminated | time_outs, as_tuple=False).flatten()
            flush_episode_evidence(done_indices, complete=True)

        trailing_indices = torch.nonzero(episode_steps > 0, as_tuple=False).flatten()
        flush_episode_evidence(trailing_indices, complete=False)

        denom = float(max(1, cmd_samples))
        result = {
            "command": name,
            "cmd_vx": cmd[0],
            "cmd_vy": cmd[1],
            "cmd_wz": cmd[2],
            "mae_vx": cmd_err_vx / denom,
            "mae_vy": cmd_err_vy / denom,
            "mae_wz": cmd_err_wz / denom,
        }
        result["actual_forward_speed_mean"] = cmd_forward_speed_sum / denom
        result["actual_lateral_leak_mean"] = cmd_lateral_leak_sum / denom
        result["actual_yaw_leak_mean"] = cmd_yaw_leak_sum / denom
        result["skill"] = skill
        result["success_duration_s"] = float(cmd_success_steps) * step_dt / float(max(1, num_envs))
        result["success_ratio"] = result["success_duration_s"] / float(max(1e-6, eval_duration_s))
        result["success_vy_duration_s"] = float(cmd_success_vy_steps) * step_dt / float(max(1, num_envs))
        result["success_wz_duration_s"] = float(cmd_success_wz_steps) * step_dt / float(max(1, num_envs))
        result["diag_sign_match_ratio"] = float(cmd_diag_sign_match) / float(max(1, cmd_diag_sign_total))
        result["yaw_tilt_ok_ratio"] = float(cmd_yaw_tilt_ok_steps) / float(max(1, args_cli.sweep_steps * num_envs))
        result["fall_rate"] = float(cmd_fall_events) / float(max(1, cmd_episode_ends))
        cmd_energy_denom = float(max(1, cmd_energy_steps))
        result["energy_mech_power_main_mean"] = cmd_mech_power_main_sum / cmd_energy_denom
        result["energy_mech_power_total_mean"] = cmd_mech_power_total_sum / cmd_energy_denom
        result["energy_cost_of_transport_proxy"] = cmd_cot_sum / cmd_energy_denom
        result["energy_spring_energy_mean"] = cmd_spring_energy_sum / cmd_energy_denom
        result["energy_spring_release_power_mean"] = cmd_spring_release_sum / cmd_energy_denom
        result["energy_spring_store_power_mean"] = cmd_spring_store_sum / cmd_energy_denom
        result["energy_spring_recovery_ratio"] = cmd_spring_recovery_ratio_sum / cmd_energy_denom
        result["energy_motion_speed_mean"] = cmd_motion_speed_sum / cmd_energy_denom
        result["energy_progress_speed_mean"] = cmd_progress_speed_sum / cmd_energy_denom
        result["energy_cost_mean"] = cmd_energy_cost_sum / cmd_energy_denom
        result["energy_progress_distance_mean"] = cmd_progress_distance_sum / cmd_energy_denom
        result["energy_per_distance"] = cmd_energy_per_distance_sum / cmd_energy_denom
        # 保留舊欄位名稱，避免外部 CSV/plot 腳本中斷；其語意已改為每單位有效位移能耗。
        result["energy_power_per_motion"] = result["energy_per_distance"]

        # Tracking quality (0~1), normalized by command magnitude
        if skill == "forward":
            tracking_quality = max(0.0, 1.0 - result["mae_vx"] / max(1e-6, abs(float(cmd[0]))))
        elif skill == "lateral":
            tracking_quality = max(0.0, 1.0 - result["mae_vy"] / max(1e-6, abs(float(cmd[1]))))
        elif skill == "diagonal":
            qx = max(0.0, 1.0 - result["mae_vx"] / max(1e-6, abs(float(cmd[0]))))
            qy = max(0.0, 1.0 - result["mae_vy"] / max(1e-6, abs(float(cmd[1]))))
            tracking_quality = 0.5 * (qx + qy)
        elif skill == "yaw":
            tracking_quality = max(0.0, 1.0 - result["mae_wz"] / max(1e-6, abs(float(cmd[2]))))
        else:
            tracking_quality = 0.0
        result["tracking_quality"] = min(1.0, max(0.0, tracking_quality))

        stability_quality = 1.0 - result["fall_rate"] / max(1e-6, args_cli.accept_max_fall_rate)
        result["stability_quality"] = min(1.0, max(0.0, stability_quality))

        if skill == "yaw":
            score = 100.0 * (
                0.50 * result["success_ratio"]
                + 0.25 * result["tracking_quality"]
                + 0.15 * result["stability_quality"]
                + 0.10 * result["yaw_tilt_ok_ratio"]
            )
        elif skill == "diagonal":
            score = 100.0 * (
                0.50 * result["success_ratio"]
                + 0.25 * result["tracking_quality"]
                + 0.15 * result["stability_quality"]
                + 0.10 * result["diag_sign_match_ratio"]
            )
        else:
            score = 100.0 * (
                0.55 * result["success_ratio"]
                + 0.30 * result["tracking_quality"]
                + 0.15 * result["stability_quality"]
            )
        result["score"] = score

        accept_pass = (
            (result["success_duration_s"] >= args_cli.accept_duration_s)
            and (result["fall_rate"] <= args_cli.accept_max_fall_rate)
        )
        if skill == "diagonal":
            accept_pass = accept_pass and (result["diag_sign_match_ratio"] >= args_cli.accept_diag_sign_ratio)
        if skill == "yaw":
            accept_pass = accept_pass and (result["yaw_tilt_ok_ratio"] >= args_cli.accept_yaw_tilt_ratio)
        result["accept_pass"] = accept_pass

        skill_total[skill] += 1
        if result["accept_pass"]:
            skill_pass[skill] += 1
        score_sum += result["score"]
        results.append(result)

    safe_samples = max(1, sample_count)
    mean_abs_vx = err_vx_sum / safe_samples
    mean_abs_vy = err_vy_sum / safe_samples
    mean_abs_wz = err_wz_sum / safe_samples

    safe_stability = max(1, stability_count)
    base_h_mean = base_h_sum / safe_stability
    base_h_var = max(0.0, base_h_sq_sum / safe_stability - base_h_mean * base_h_mean)
    base_h_std = math.sqrt(base_h_var)
    pitch_rms = math.sqrt(pitch_sq_sum / safe_stability)
    roll_rms = math.sqrt(roll_sq_sum / safe_stability)

    fall_rate = float(fall_events) / float(max(1, episode_ends))

    safe_forward = max(1, forward_count)
    phase_diff_mean = forward_phase_diff_sum / safe_forward
    phase_err_to_pi = forward_phase_err_to_pi_sum / safe_forward
    stance_fraction_mean = forward_stance_frac_sum / safe_forward
    stance_fraction_err = forward_stance_frac_err_sum / safe_forward
    stance_speed_mean = forward_stance_speed_sum / safe_forward
    swing_speed_mean = forward_swing_speed_sum / safe_forward
    swing_to_stance_ratio = swing_speed_mean / max(stance_speed_mean, 1e-6)

    transition_ratio_ge4 = float(transition_contact_ge4) / float(max(1, transition_samples))

    action_rate_mean = action_rate_sum / float(max(1, energy_count))
    effort_proxy_mean = effort_proxy_sum / float(max(1, energy_count))
    mech_power_main_mean = mech_power_main_sum / float(max(1, energy_kpi_count))
    mech_power_total_mean = mech_power_total_sum / float(max(1, energy_kpi_count))
    cot_proxy_mean = cot_proxy_sum / float(max(1, energy_kpi_count))
    motion_speed_mean = motion_speed_sum / float(max(1, energy_kpi_count))
    progress_speed_mean = progress_speed_sum / float(max(1, energy_kpi_count))
    energy_cost_mean = energy_cost_sum / float(max(1, energy_kpi_count))
    progress_distance_mean = progress_distance_sum / float(max(1, energy_kpi_count))
    energy_per_distance_mean = energy_per_distance_sum / float(max(1, energy_kpi_count))
    command_pass_ratio = float(sum(1 for row in results if row["accept_pass"])) / float(max(1, len(results)))
    max_command_fall_rate = max(float(row["fall_rate"]) for row in results)
    overall_score_mean = score_sum / float(max(1, len(results)))
    skill_pass_ratio = {
        skill: float(skill_pass[skill]) / float(max(1, skill_total[skill])) for skill in sorted(skill_total.keys())
    }
    skill_energy_summary = {
        skill: {
            "mech_power_total_mean": skill_mech_power_total_sum[skill] / float(max(1, skill_energy_steps[skill])),
            "cost_of_transport_proxy": skill_cot_sum[skill] / float(max(1, skill_energy_steps[skill])),
            "spring_recovery_ratio": skill_spring_recovery_ratio_sum[skill] / float(max(1, skill_energy_steps[skill])),
            "motion_speed_mean": skill_motion_speed_sum[skill] / float(max(1, skill_energy_steps[skill])),
            "progress_speed_mean": skill_progress_speed_sum[skill] / float(max(1, skill_energy_steps[skill])),
            "energy_cost_mean": skill_energy_cost_sum[skill] / float(max(1, skill_energy_steps[skill])),
            "progress_distance_mean": skill_progress_distance_sum[skill] / float(max(1, skill_energy_steps[skill])),
            "energy_per_distance": skill_energy_per_distance_sum[skill] / float(max(1, skill_energy_steps[skill])),
        }
        for skill in sorted(skill_total.keys())
    }
    min_skill_pass_ratio = min(skill_pass_ratio.values()) if len(skill_pass_ratio) > 0 else 0.0
    overall_accept_pass = (
        (command_pass_ratio >= args_cli.accept_overall_pass_ratio)
        and (min_skill_pass_ratio >= args_cli.accept_skill_pass_ratio)
        and (max_command_fall_rate <= args_cli.accept_max_fall_rate)
    )

    print("\n=== Command Tracking (MAE) ===")
    print(
        f"{'command':<14} {'skill':<9} {'cmd(vx,vy,wz)':<24} "
        f"{'|vx-vx*|':>10} {'|vy-vy*|':>10} {'|wz-wz*|':>10} {'score':>8}"
    )
    for row in results:
        cmd_str = f"({row['cmd_vx']:.2f},{row['cmd_vy']:.2f},{row['cmd_wz']:.2f})"
        print(
            f"{row['command']:<14} {row['skill']:<9} {cmd_str:<24} "
            f"{row['mae_vx']:>10.4f} {row['mae_vy']:>10.4f} {row['mae_wz']:>10.4f} {row['score']:>8.2f}"
        )

    print("\n=== Skill Acceptance (Command-level) ===")
    for row in results:
        extra = ""
        if row["skill"] == "lateral":
            extra = (
                f"success_s={row['success_duration_s']:.2f}, vy_success_s={row['success_vy_duration_s']:.2f}, "
                f"fall_rate={row['fall_rate']:.3f}"
            )
        elif row["skill"] == "yaw":
            extra = (
                f"success_s={row['success_duration_s']:.2f}, wz_success_s={row['success_wz_duration_s']:.2f}, "
                f"tilt_ok_ratio={row['yaw_tilt_ok_ratio']:.3f}, fall_rate={row['fall_rate']:.3f}"
            )
        elif row["skill"] == "diagonal":
            extra = (
                f"success_s={row['success_duration_s']:.2f}, diag_sign_match={row['diag_sign_match_ratio']:.3f}, "
                f"fall_rate={row['fall_rate']:.3f}"
            )
        elif row["skill"] == "forward":
            extra = f"success_s={row['success_duration_s']:.2f}, fall_rate={row['fall_rate']:.3f}"
        status = "PASS" if row["accept_pass"] else "FAIL"
        print(f"{row['command']:<14} {status:<4} {extra}")

    print("\n=== Energy By Command ===")
    print(
        f"{'command':<14} {'skill':<9} {'P_total(W)':>11} {'E_cost':>11} "
        f"{'dist':>11} {'E/dist':>11}"
    )
    for row in results:
        print(
            f"{row['command']:<14} {row['skill']:<9} "
            f"{row['energy_mech_power_total_mean']:>11.4f} "
            f"{row['energy_cost_mean']:>11.6f} "
            f"{row['energy_progress_distance_mean']:>11.6f} "
            f"{row['energy_per_distance']:>11.4f} "
        )

    print("\n=== Skill-level Pass Ratio ===")
    for skill, ratio in skill_pass_ratio.items():
        status = "PASS" if ratio >= args_cli.accept_skill_pass_ratio else "FAIL"
        print(
            f"{skill:<9} {status:<4} pass_ratio={ratio:.3f} "
            f"(threshold={args_cli.accept_skill_pass_ratio:.2f}, {skill_pass[skill]}/{skill_total[skill]})"
        )

    print("\n=== Skill-level Energy Summary ===")
    for skill, metrics in skill_energy_summary.items():
        print(
            f"{skill:<9} P_total={metrics['mech_power_total_mean']:.4f} W, "
            f"E_cost={metrics['energy_cost_mean']:.6f}, "
            f"dist={metrics['progress_distance_mean']:.6f}, "
            f"E/dist={metrics['energy_per_distance']:.4f}, "
            f"spring_rec={metrics['spring_recovery_ratio']:.4f}, "
            f"progress={metrics['progress_speed_mean']:.4f}, "
            f"motion_eq={metrics['motion_speed_mean']:.4f}"
        )

    overall_status = "PASS" if overall_accept_pass else "FAIL"
    print(
        f"\n=== Overall Acceptance ===\n"
        f"profile={args_cli.eval_profile} status={overall_status} "
        f"command_pass_ratio={command_pass_ratio:.3f} (threshold={args_cli.accept_overall_pass_ratio:.2f}) "
        f"min_skill_pass_ratio={min_skill_pass_ratio:.3f} (threshold={args_cli.accept_skill_pass_ratio:.2f}) "
        f"max_command_fall_rate={max_command_fall_rate:.3f} (threshold={args_cli.accept_max_fall_rate:.2f}) "
        f"score_mean={overall_score_mean:.2f}"
    )

    print("\n=== Acceptance Metrics Summary ===")
    print(f"tracking.mean|vx-vx_cmd|: {mean_abs_vx:.6f}")
    print(f"tracking.mean|vy-vy_cmd|: {mean_abs_vy:.6f}")
    print(f"tracking.mean|wz-wz_cmd|: {mean_abs_wz:.6f}")
    print(f"forward.phase_diff_mean(rad): {phase_diff_mean:.6f}")
    print(f"forward.phase_diff_abs_to_pi(rad): {phase_err_to_pi:.6f}")
    print(f"forward.stance_fraction_mean: {stance_fraction_mean:.6f}")
    print(f"forward.stance_fraction_abs_err_to_0.65: {stance_fraction_err:.6f}")
    print(f"forward.stance_speed_mean(rad/s): {stance_speed_mean:.6f}")
    print(f"forward.swing_speed_mean(rad/s): {swing_speed_mean:.6f}")
    print(f"forward.swing_to_stance_speed_ratio: {swing_to_stance_ratio:.6f}")
    print(f"stability.fall_rate: {fall_rate:.6f}")
    print(f"stability.base_height_mean(m): {base_h_mean:.6f}")
    print(f"stability.base_height_std(m): {base_h_std:.6f}")
    print(f"stability.base_height_var(m^2): {base_h_var:.6f}")
    print(f"stability.pitch_rms(rad): {pitch_rms:.6f}")
    print(f"stability.roll_rms(rad): {roll_rms:.6f}")
    print(f"contact.histogram: {summarize_contact_hist(contact_hist)}")
    print(f"contact.transition_ratio_ge4: {transition_ratio_ge4:.6f}")
    print(f"energy.action_rate_mean: {action_rate_mean:.6f}")
    print(f"energy.effort_proxy_mean [{effort_proxy_name}]: {effort_proxy_mean:.6f}")
    # ★ Energy-aware KPIs
    _ekc = float(max(1, energy_kpi_count))
    print(f"energy.spring_energy_mean(J): {spring_energy_sum / _ekc:.6f}")
    print(f"energy.spring_release_power_mean(W): {spring_release_sum / _ekc:.6f}")
    print(f"energy.spring_store_power_mean(W): {spring_store_sum / _ekc:.6f}")
    print(f"energy.damper_dissipation_mean(W): {damper_dissipation_sum / _ekc:.6f}")
    print(f"energy.mech_power_main_mean(W): {mech_power_main_mean:.6f}")
    print(f"energy.mech_power_total_mean(W): {mech_power_total_mean:.6f}")
    print(f"energy.motion_speed_equiv_mean: {motion_speed_mean:.6f}")
    print(f"energy.progress_speed_mean: {progress_speed_mean:.6f}")
    print(f"energy.mean_energy_cost: {energy_cost_mean:.6f}")
    print(f"energy.mean_progress_distance: {progress_distance_mean:.6f}")
    print(f"energy.mean_energy_per_distance: {energy_per_distance_mean:.6f}")
    print(f"energy.per_distance: {energy_per_distance_mean:.6f}")
    print(f"energy.cost_of_transport_proxy: {cot_proxy_mean:.6f}")
    if spring_release_sum + spring_store_sum > 0:
        print(f"energy.spring_recovery_ratio: {spring_release_sum / (spring_release_sum + spring_store_sum):.6f}")
    else:
        print(f"energy.spring_recovery_ratio: N/A")
    print(f"acceptance.command_pass_ratio: {command_pass_ratio:.6f}")
    print(f"acceptance.min_skill_pass_ratio: {min_skill_pass_ratio:.6f}")
    print(f"acceptance.max_command_fall_rate: {max_command_fall_rate:.6f}")
    print(f"acceptance.overall_score_mean: {overall_score_mean:.6f}")

    if args_cli.csv:
        csv_path = os.path.abspath(args_cli.csv)
        csv_dir = os.path.dirname(csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "command",
                    "skill",
                    "cmd_vx",
                    "cmd_vy",
                    "cmd_wz",
                    "mae_vx",
                    "mae_vy",
                    "mae_wz",
                    "actual_forward_speed_mean",
                    "actual_lateral_leak_mean",
                    "actual_yaw_leak_mean",
                    "success_duration_s",
                    "success_ratio",
                    "success_vy_duration_s",
                    "success_wz_duration_s",
                    "diag_sign_match_ratio",
                    "yaw_tilt_ok_ratio",
                    "fall_rate",
                    "energy_mech_power_main_mean",
                    "energy_mech_power_total_mean",
                    "energy_cost_of_transport_proxy",
                    "energy_spring_energy_mean",
                    "energy_spring_release_power_mean",
                    "energy_spring_store_power_mean",
                    "energy_spring_recovery_ratio",
                    "energy_motion_speed_mean",
                    "energy_progress_speed_mean",
                    "energy_cost_mean",
                    "energy_progress_distance_mean",
                    "energy_per_distance",
                    "energy_power_per_motion",
                    "tracking_quality",
                    "stability_quality",
                    "score",
                    "accept_pass",
                ],
            )
            writer.writeheader()
            writer.writerows(results)

        episode_path = os.path.splitext(csv_path)[0] + "_episodes.csv"
        with open(episode_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "command",
                    "skill",
                    "environment_index",
                    "episode_index",
                    "complete",
                    "sample_count",
                    "fall_count",
                    "mae_vx",
                    "mae_vy",
                    "mae_wz",
                    "success_ratio",
                    "energy_mech_power_total_mean",
                    "energy_effort_mean",
                ],
            )
            writer.writeheader()
            writer.writerows(episode_results)

        summary_path = os.path.splitext(csv_path)[0] + "_summary.csv"
        command_csv_sha256 = _sha256_file(csv_path)
        episode_csv_sha256 = _sha256_file(episode_path)
        summary_rows = [
            {"metric": "evaluation.seed", "value": int(agent_cfg.seed)},
            {"metric": "evaluation.num_envs", "value": int(num_envs)},
            {"metric": "evaluation.sweep_steps", "value": int(args_cli.sweep_steps)},
            {"metric": "evaluation.step_dt", "value": float(step_dt)},
            {"metric": "evaluation.duration_s", "value": float(eval_duration_s)},
            {"metric": "eval.profile", "value": args_cli.eval_profile},
            {"metric": "evaluation.agent_entry_point", "value": args_cli.agent},
            {
                "metric": "command.profile_sha256",
                "value": args_cli.command_profile_sha256,
            },
            {"metric": "checkpoint.path", "value": os.path.abspath(resume_path)},
            {"metric": "checkpoint.sha256", "value": _sha256_file(resume_path)},
            {"metric": "checkpoint.strict_load", "value": bool(args_cli.strict_checkpoint_loading)},
            {"metric": "energy.strict_evidence", "value": bool(args_cli.strict_energy_evidence)},
            {"metric": "identity.code_sha256", "value": runtime_identities["code"]},
            {"metric": "identity.config_sha256", "value": runtime_identities["config"]},
            {"metric": "identity.dependency_sha256", "value": runtime_identities["dependency"]},
            {"metric": "identity.reward_profile_sha256", "value": args_cli.identity_reward_profile_sha256},
            {"metric": "identity.physics.sha256", "value": args_cli.identity_physics_sha256},
            {"metric": "identity.spring.sha256", "value": args_cli.identity_spring_sha256},
            {"metric": "identity.terrain.sha256", "value": args_cli.identity_terrain_sha256},
            {"metric": "spring.backend", "value": args_cli.spring_backend},
            {"metric": "spring.calibration_status", "value": spring_calibration_status},
            {
                "metric": "spring.checkpoint_calibration_status",
                "value": checkpoint_spring_calibration_status,
            },
            {"metric": "spring.profile_id", "value": spring_profile_id},
            {"metric": "spring.profile_sha256", "value": spring_profile_sha256},
            {
                "metric": "artifact.command_csv_sha256",
                "value": command_csv_sha256,
            },
            {
                "metric": "artifact.episode_csv_sha256",
                "value": episode_csv_sha256,
            },
            {"metric": "evidence.episode_row_count", "value": len(episode_results)},
            {"metric": "tracking.mean_abs_vx", "value": mean_abs_vx},
            {"metric": "tracking.mean_abs_vy", "value": mean_abs_vy},
            {"metric": "tracking.mean_abs_wz", "value": mean_abs_wz},
            {"metric": "forward.phase_diff_mean", "value": phase_diff_mean},
            {"metric": "forward.phase_diff_abs_to_pi", "value": phase_err_to_pi},
            {"metric": "forward.stance_fraction_mean", "value": stance_fraction_mean},
            {"metric": "forward.stance_fraction_abs_err_to_0.65", "value": stance_fraction_err},
            {"metric": "forward.stance_speed_mean", "value": stance_speed_mean},
            {"metric": "forward.swing_speed_mean", "value": swing_speed_mean},
            {"metric": "forward.swing_to_stance_speed_ratio", "value": swing_to_stance_ratio},
            {"metric": "stability.fall_rate", "value": fall_rate},
            {"metric": "stability.base_height_mean", "value": base_h_mean},
            {"metric": "stability.base_height_std", "value": base_h_std},
            {"metric": "stability.base_height_var", "value": base_h_var},
            {"metric": "stability.pitch_rms", "value": pitch_rms},
            {"metric": "stability.roll_rms", "value": roll_rms},
            {"metric": "contact.transition_ratio_ge4", "value": transition_ratio_ge4},
            {"metric": "energy.action_rate_mean", "value": action_rate_mean},
            {"metric": "energy.effort_proxy_mean", "value": effort_proxy_mean},
            {"metric": "energy.effort_proxy_name", "value": effort_proxy_name},
            {"metric": "energy.spring_energy_mean", "value": spring_energy_sum / _ekc},
            {"metric": "energy.spring_release_power_mean", "value": spring_release_sum / _ekc},
            {"metric": "energy.spring_store_power_mean", "value": spring_store_sum / _ekc},
            {"metric": "energy.damper_dissipation_mean", "value": damper_dissipation_sum / _ekc},
            {"metric": "energy.mech_power_main_mean", "value": mech_power_main_mean},
            {"metric": "energy.mech_power_total_mean", "value": mech_power_total_mean},
            {"metric": "energy.motion_speed_equiv_mean", "value": motion_speed_mean},
            {"metric": "energy.progress_speed_mean", "value": progress_speed_mean},
            {"metric": "energy.mean_energy_cost", "value": energy_cost_mean},
            {"metric": "energy.mean_progress_distance", "value": progress_distance_mean},
            {"metric": "energy.mean_energy_per_distance", "value": energy_per_distance_mean},
            {"metric": "energy.per_distance", "value": energy_per_distance_mean},
            {"metric": "energy.cost_of_transport_proxy", "value": cot_proxy_mean},
            {"metric": "energy.spring_recovery_ratio", "value": spring_release_sum / max(spring_release_sum + spring_store_sum, 1e-6)},
            {"metric": "contact.histogram", "value": summarize_contact_hist(contact_hist)},
            {"metric": "acceptance.command_pass_ratio", "value": command_pass_ratio},
            {"metric": "acceptance.min_skill_pass_ratio", "value": min_skill_pass_ratio},
            {"metric": "acceptance.max_command_fall_rate", "value": max_command_fall_rate},
            {"metric": "acceptance.overall_score_mean", "value": overall_score_mean},
            {"metric": "acceptance.overall_status", "value": "PASS" if overall_accept_pass else "FAIL"},
        ]
        for skill, ratio in skill_pass_ratio.items():
            summary_rows.append({"metric": f"acceptance.skill_pass_ratio.{skill}", "value": ratio})
        for skill, metrics in skill_energy_summary.items():
            summary_rows.append({"metric": f"energy.skill.{skill}.mech_power_total_mean", "value": metrics["mech_power_total_mean"]})
            summary_rows.append({"metric": f"energy.skill.{skill}.cost_of_transport_proxy", "value": metrics["cost_of_transport_proxy"]})
            summary_rows.append({"metric": f"energy.skill.{skill}.spring_recovery_ratio", "value": metrics["spring_recovery_ratio"]})
            summary_rows.append({"metric": f"energy.skill.{skill}.motion_speed_mean", "value": metrics["motion_speed_mean"]})
            summary_rows.append({"metric": f"energy.skill.{skill}.progress_speed_mean", "value": metrics["progress_speed_mean"]})
            summary_rows.append({"metric": f"energy.skill.{skill}.energy_cost_mean", "value": metrics["energy_cost_mean"]})
            summary_rows.append({"metric": f"energy.skill.{skill}.progress_distance_mean", "value": metrics["progress_distance_mean"]})
            summary_rows.append({"metric": f"energy.skill.{skill}.energy_per_distance", "value": metrics["energy_per_distance"]})

        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["metric", "value"])
            writer.writeheader()
            writer.writerows(summary_rows)

        print(f"[INFO] Wrote command table: {csv_path}")
        print(f"[INFO] Wrote episode evidence: {episode_path}")
        print(f"[INFO] Wrote summary table: {summary_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
