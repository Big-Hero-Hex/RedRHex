# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""
#import isaaclab_tasks  # :)
#import ext_furuta_pendulum  # :)

import argparse
import hashlib
import json
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from tools.sim2real.repo_binding import (  # noqa: E402
    assert_redrhex_module_source,
    bind_redrhex_source,
)


bind_redrhex_source(_REPO_ROOT)

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--physics-profile",
    type=str,
    default=None,
    help="Explicit CalibrationProfileV1 JSON override; defaults never load a candidate profile.",
)
parser.add_argument(
    "--sensor-dr-profile",
    type=str,
    default=None,
    help="Sensor V2 only: evidence-bound F4 training_curriculum profile JSON.",
)
parser.add_argument(
    "--sensor-dr-profile-sha256",
    type=str,
    default=None,
    help="Required exact SHA-256 for --sensor-dr-profile.",
)
parser.add_argument(
    "--spring-backend",
    choices=("explicit", "native"),
    default="native",
    help=(
        "Passive torsion-spring implementation used by the environment. Native is "
        "the provisional safe default; Explicit policy training is quarantined and "
        "remains available through the sim2real spring-release characterization workflow."
    ),
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--store_code_state",
    action="store_true",
    default=False,
    help="Store git code-state snapshots in the run folder (disabled by default to avoid unicode git-diff errors).",
)
parser.add_argument(
    "--panel_overrides",
    action="store_true",
    default=False,
    help=(
        "Apply reward/terrain override files written by the training panel "
        "(tools/training_panel/active_*_override.json). The panel passes this flag itself; "
        "manual runs ignore stale override files unless it is given explicitly."
    ),
)
parser.add_argument(
    "--reward-profile",
    type=str,
    default=None,
    help="Immutable per-run reward override JSON. Prefer this over --panel_overrides.",
)
parser.add_argument("--reward-profile-sha256", type=str, default=None, help="Expected reward profile SHA-256.")
parser.add_argument(
    "--terrain-profile",
    type=str,
    default=None,
    help="Immutable per-run terrain override JSON. Prefer this over --panel_overrides.",
)
parser.add_argument("--terrain-profile-sha256", type=str, default=None, help="Expected terrain profile SHA-256.")
parser.add_argument("--checkpoint-sha256", type=str, default=None, help="Expected exact source checkpoint SHA-256.")
parser.add_argument(
    "--strict-checkpoint-loading",
    action="store_true",
    default=False,
    help="Forbid legacy shape-compatible policy fallback.",
)
parser.add_argument(
    "--curriculum-stage",
    type=int,
    choices=(1, 2, 3, 4, 5),
    default=None,
    help="Typed Direct curriculum stage selected by the panel.",
)
parser.add_argument(
    "--resume_policy_only",
    action="store_true",
    default=False,
    help="Resume from checkpoint weights only (skip optimizer state and reset learning iteration).",
)
parser.add_argument(
    "--reset_action_std",
    type=float,
    default=None,
    help="Optional action std reset value after loading checkpoint (useful with --resume_policy_only).",
)
parser.add_argument(
    "--teacher_checkpoint",
    type=str,
    default=None,
    help="V2 only: strict teacher_v2 checkpoint used to start a new F2 distillation run.",
)
parser.add_argument(
    "--student_checkpoint",
    type=str,
    default=None,
    help="V2 only: strict student_distilled_v2 checkpoint used to bootstrap a new F3 PPO run.",
)
parser.add_argument(
    "--ppo_checkpoint",
    type=str,
    default=None,
    help="V2 only: strict student_ppo_v2 checkpoint used to bootstrap F4 robustness PPO.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()


_V2_AGENT_CLASS_BY_ENTRY_POINT = {
    "rsl_rl_distillation_v2_cfg_entry_point": "SensorDistillationRunnerV2",
    "rsl_rl_distillation_v2_no_aux_cfg_entry_point": "SensorDistillationRunnerV2",
    "rsl_rl_distillation_v2_velocity_cfg_entry_point": "SensorDistillationRunnerV2",
    "rsl_rl_distillation_v2_velocity_dynamics_cfg_entry_point": "SensorDistillationRunnerV2",
    "rsl_rl_ppo_v2_cfg_entry_point": "SensorOnPolicyRunnerV2",
    "rsl_rl_robust_ppo_v2_cfg_entry_point": "SensorRobustnessRunnerV2",
}
_V2_BOOTSTRAP_RUNNER_CLASS = {
    "teacher_checkpoint": "SensorDistillationRunnerV2",
    "student_checkpoint": "SensorOnPolicyRunnerV2",
    "ppo_checkpoint": "SensorRobustnessRunnerV2",
}


def _is_sensor_v2_task(task_name: str | None) -> bool:
    return bool(task_name and task_name.split(":")[-1] == "Template-Redrhex-ForwardSensorV2-Direct-v0")


def _validate_spring_backend_cli() -> None:
    if args_cli.spring_backend == "explicit":
        parser.error(
            "Explicit torsion-spring policy training is quarantined: the current "
            "uncalibrated 200 N·m/rad model is numerically unstable at 120 Hz. "
            "Use --spring-backend native for provisional training or use "
            "`python -m tools.sim2real run-sim --scenario spring-release "
            "--spring-backend explicit ...` for deterministic characterization."
        )


def _validate_checkpoint_cli() -> None:
    bootstrap = [
        name
        for name, value in (
            ("teacher_checkpoint", args_cli.teacher_checkpoint),
            ("student_checkpoint", args_cli.student_checkpoint),
            ("ppo_checkpoint", args_cli.ppo_checkpoint),
        )
        if value is not None
    ]
    if len(bootstrap) > 1:
        parser.error("V2 bootstrap checkpoint flags are mutually exclusive")
    if bootstrap:
        checkpoint_name = bootstrap[0]
        flag = f"--{checkpoint_name}"
        if args_cli.resume:
            parser.error(f"{flag} cannot be combined with --resume")
        if args_cli.checkpoint is not None:
            parser.error(f"{flag} cannot be combined with --checkpoint")
        if not _is_sensor_v2_task(args_cli.task):
            parser.error(f"{flag} is valid only for the ForwardSensorV2 task")
        expected_class = _V2_BOOTSTRAP_RUNNER_CLASS[checkpoint_name]
        selected_class = _V2_AGENT_CLASS_BY_ENTRY_POINT.get(args_cli.agent)
        if selected_class != expected_class:
            parser.error(
                f"{flag} requires a Sensor V2 agent entry point whose runner class is "
                f"{expected_class}; --agent {args_cli.agent!r} resolves to "
                f"{selected_class or 'no allowlisted V2 runner class'}"
            )
    if _is_sensor_v2_task(args_cli.task) and args_cli.resume and args_cli.checkpoint is None:
        parser.error("V2 resume requires --resume --checkpoint PATH for exact same-kind loading")
    if _is_sensor_v2_task(args_cli.task) and args_cli.resume_policy_only:
        parser.error("V2 forbids --resume_policy_only and shape-compatible partial loading")
    if args_cli.resume_policy_only and not args_cli.resume:
        parser.error("--resume_policy_only requires --resume")
    if args_cli.strict_checkpoint_loading and args_cli.checkpoint is None:
        parser.error("--strict-checkpoint-loading requires --checkpoint")


def _validate_bootstrap_runner_class(class_name: str) -> None:
    """Recheck the Hydra-resolved class before constructing an environment."""

    for checkpoint_name, expected_class in _V2_BOOTSTRAP_RUNNER_CLASS.items():
        if getattr(args_cli, checkpoint_name) is None:
            continue
        if class_name != expected_class:
            raise ValueError(
                f"--{checkpoint_name} requires runner class {expected_class}, got {class_name!r}"
            )
        return


def _validate_sensor_dr_cli() -> None:
    supplied = args_cli.sensor_dr_profile is not None
    pinned = args_cli.sensor_dr_profile_sha256 is not None
    if supplied != pinned:
        parser.error(
            "--sensor-dr-profile and --sensor-dr-profile-sha256 must be supplied together"
        )
    if supplied and not _is_sensor_v2_task(args_cli.task):
        parser.error("--sensor-dr-profile is valid only for the ForwardSensorV2 task")
    robust_agent = "rsl_rl_robust_ppo_v2_cfg_entry_point"
    if supplied and args_cli.agent != robust_agent:
        parser.error(f"--sensor-dr-profile requires --agent {robust_agent}")
    if supplied and args_cli.ppo_checkpoint is None:
        parser.error("F4 --sensor-dr-profile requires --ppo_checkpoint")
    if args_cli.ppo_checkpoint is not None and not supplied:
        parser.error("--ppo_checkpoint requires an evidence-bound --sensor-dr-profile")


def _validate_sha256_bound_file(label: str, raw_path: str | None, expected: str | None) -> None:
    if expected is None:
        return
    if raw_path is None:
        parser.error(f"--{label}-sha256 requires --{label}")
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected.lower()):
        parser.error(f"--{label}-sha256 must be a 64-character hexadecimal digest")
    path = Path(raw_path)
    if not path.is_file():
        parser.error(f"--{label} must name an existing regular file when a digest is supplied")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected.lower():
        parser.error(f"--{label} SHA-256 mismatch: expected {expected.lower()}, got {actual}")


_validate_spring_backend_cli()
_validate_checkpoint_cli()
_validate_sensor_dr_cli()
_validate_sha256_bound_file("reward-profile", args_cli.reward_profile, args_cli.reward_profile_sha256)
_validate_sha256_bound_file("terrain-profile", args_cli.terrain_profile, args_cli.terrain_profile_sha256)
_validate_sha256_bound_file("checkpoint", args_cli.checkpoint, args_cli.checkpoint_sha256)
_validate_sha256_bound_file(
    "sensor-dr-profile", args_cli.sensor_dr_profile, args_cli.sensor_dr_profile_sha256
)

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
#import ext_furuta_pendulum.tasks.registration  # noqa: F401

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)
if _is_sensor_v2_task(args_cli.task) and not (
    version.parse("3.1.2") <= version.parse(installed_version) < version.parse("3.2")
):
    raise RuntimeError(
        "Sensor Distillation V2 is version-gated to rsl-rl-lib >=3.1.2,<3.2; "
        f"installed version is {installed_version}. Legacy tasks retain the existing >=3.0.1 gate."
    )

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import RedRhex.tasks as _redrhex_tasks  # noqa: F401
from scripts.rsl_rl.runner_factory import create_runner, runner_protocol


assert_redrhex_module_source(_redrhex_tasks, _REPO_ROOT)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _load_runner_checkpoint_with_policy_fallback(
    runner,
    resume_path: str,
    device: str,
    *,
    load_optimizer: bool = True,
    allow_partial_policy: bool = False,
) -> None:
    """Load a checkpoint; optionally fall back to actor-compatible weights only."""
    try:
        runner.load(resume_path, load_optimizer=load_optimizer)
        return
    except TypeError:
        if not load_optimizer:
            raise
        try:
            runner.load(resume_path)
            return
        except RuntimeError as exc:
            original_error = exc
    except RuntimeError as exc:
        original_error = exc

    if not allow_partial_policy:
        raise original_error

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
        "[WARN] Full policy-only checkpoint load failed, likely due to critic/privileged-observation shape changes. "
        f"Loaded {len(compatible_state)} actor-compatible tensors and skipped {skipped} tensors."
    )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    _validate_bootstrap_runner_class(agent_cfg.class_name)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.curriculum_stage is not None:
        if not hasattr(env_cfg, "stage"):
            raise ValueError("--curriculum-stage is not supported by the selected task")
        env_cfg.stage = int(args_cli.curriculum_stage)
        print(f"[INFO] Training curriculum stage fixed to {env_cfg.stage}")
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        print(
            "[WARN] IO descriptors are only supported for manager based RL environments. "
            "No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # New panel launches use immutable hash-bound per-run snapshots. The global files
    # remain a compatibility path for older manual tooling only.
    _override_file = Path(__file__).parents[2] / "tools" / "training_panel" / "active_reward_override.json"
    _terrain_override_file = Path(__file__).parents[2] / "tools" / "training_panel" / "active_terrain_override.json"
    _reward_input = Path(args_cli.reward_profile) if args_cli.reward_profile else None
    _terrain_input = Path(args_cli.terrain_profile) if args_cli.terrain_profile else None
    if args_cli.panel_overrides:
        _reward_input = _reward_input or (_override_file if _override_file.exists() else None)
        _terrain_input = _terrain_input or (_terrain_override_file if _terrain_override_file.exists() else None)

    if _reward_input is not None:
        from tools.training_panel.training_panel.reward_overrides import apply_reward_overrides

        _overrides = json.loads(_reward_input.read_text(encoding="utf-8"))
        if not isinstance(_overrides, dict):
            raise ValueError(f"Reward profile must contain a JSON object: {_reward_input}")
        _applied = apply_reward_overrides(
            env_cfg,
            _overrides,
            require_all=bool(args_cli.reward_profile),
        )
        if _applied:
            print(f"[INFO] Training panel reward overrides applied: {', '.join(_applied)}")

    if _terrain_input is not None:
        from tools.training_panel.training_panel.terrain import apply_terrain_overrides

        _terrain_overrides = json.loads(_terrain_input.read_text(encoding="utf-8"))
        if not isinstance(_terrain_overrides, dict):
            raise ValueError(f"Terrain profile must contain a JSON object: {_terrain_input}")
        _applied = apply_terrain_overrides(
            env_cfg,
            _terrain_overrides,
            require_exact=bool(args_cli.terrain_profile),
        )
        if args_cli.terrain_profile and len(_applied) != len(_terrain_overrides):
            raise ValueError(
                "Terrain profile did not resolve exactly: "
                f"requested={sorted(_terrain_overrides)}, applied={_applied}"
            )
        if _applied:
            print(f"[INFO] Training panel terrain overrides applied: {', '.join(_applied)}")

    if not args_cli.panel_overrides and _reward_input is None and _terrain_input is None:
        for _stale in (_override_file, _terrain_override_file):
            if _stale.exists():
                print(
                    f"[WARN] Ignoring training panel override file {_stale} "
                    "(pass --panel_overrides to apply it)."
                )

    env_cfg.spring_backend = args_cli.spring_backend
    protocol = runner_protocol(agent_cfg.class_name)
    bootstrap_path = (
        args_cli.teacher_checkpoint
        or args_cli.student_checkpoint
        or args_cli.ppo_checkpoint
    )
    resume_requested = agent_cfg.resume or bootstrap_path is not None or (
        not protocol.v2 and agent_cfg.algorithm.class_name == "Distillation"
    )
    resume_path = None
    if resume_requested:
        if bootstrap_path is not None:
            resume_path = retrieve_file_path(bootstrap_path)
        elif protocol.v2 and args_cli.checkpoint is not None:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        elif agent_cfg.load_checkpoint and (
            os.path.isabs(agent_cfg.load_checkpoint) or os.path.exists(agent_cfg.load_checkpoint)
        ):
            resume_path = retrieve_file_path(agent_cfg.load_checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    physics_profile = None
    physics_profile_config_application = None
    physics_profile_runtime_application = None
    spring_profile_id = None
    spring_profile_sha256 = None
    if args_cli.physics_profile is not None:
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from tools.sim2real.physics_profile import (
            apply_profile_to_config,
            load_optional_profile,
            write_training_profile_snapshot,
        )
        from tools.sim2real.traces import sha256_json

        physics_profile = load_optional_profile(args_cli.physics_profile)
        physics_profile_config_application = apply_profile_to_config(
            env_cfg, physics_profile
        )
        spring_profile_id = physics_profile.profile_id
        spring_profile_sha256 = sha256_json(physics_profile.to_dict())
        print(
            f"[INFO] Explicit physics profile applied to config: {spring_profile_id} "
            f"({spring_profile_sha256})"
        )

    sensor_dr_profile = None
    sensor_dr_profile_sha256 = None
    if args_cli.sensor_dr_profile is not None:
        from tools.sim2real.sensor_dr_profile_v2 import (
            PHYSICAL_PROFILE_PARAMETERS_V2,
            apply_sensor_dr_profile_v2,
            load_sensor_dr_profile_v2,
        )

        sensor_dr_profile, sensor_dr_profile_sha256 = load_sensor_dr_profile_v2(
            args_cli.sensor_dr_profile,
            expected_sha256=args_cli.sensor_dr_profile_sha256,
            expected_purpose="training_curriculum",
        )
        physical_overlap = set(sensor_dr_profile.parameters) & PHYSICAL_PROFILE_PARAMETERS_V2
        if physics_profile is not None and physical_overlap:
            raise ValueError(
                "physics and Sensor V2 profiles both define actuator/timing fields: "
                f"{sorted(physical_overlap)}"
            )
        apply_sensor_dr_profile_v2(
            env_cfg, sensor_dr_profile, sensor_dr_profile_sha256
        )
        print(
            f"[INFO] Sensor V2 training DR profile applied: {sensor_dr_profile.profile_id} "
            f"({sensor_dr_profile_sha256})"
        )

    spring_calibration_status = (
        "calibrated" if bool(getattr(env_cfg, "spring_calibrated", False)) else "uncalibrated"
    )
    resume_checkpoint_calibration_status = None
    if resume_requested:
        from tools.sim2real.checkpoint_spring import validate_checkpoint_spring_evaluation

        resume_checkpoint_calibration_status = validate_checkpoint_spring_evaluation(
            os.path.dirname(resume_path),
            selected_backend=args_cli.spring_backend,
            selected_profile_id=spring_profile_id,
            selected_profile_sha256=spring_profile_sha256,
        )
        if resume_checkpoint_calibration_status == "calibrated" and spring_calibration_status != "calibrated":
            raise RuntimeError("calibrated checkpoint profile did not produce a calibrated spring configuration")
    print(
        f"[INFO] Torsion spring backend={args_cli.spring_backend}, "
        f"calibration_status={spring_calibration_status}, "
        f"resume_checkpoint_calibration_status={resume_checkpoint_calibration_status}, "
        f"profile_id={spring_profile_id}, profile_sha256={spring_profile_sha256}"
    )

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.physics_profile is not None:
        from tools.sim2real.isaac_profile import apply_profile_to_runtime_env

        physics_profile_runtime_application = apply_profile_to_runtime_env(
            env, physics_profile
        )
        print(f"[INFO] Explicit physics profile applied at runtime: {physics_profile.profile_id}")

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = create_runner(
        agent_cfg.class_name,
        env,
        agent_cfg.to_dict(),
        log_dir=log_dir,
        device=agent_cfg.device,
    )
    # write git state to logs (opt-in; some repos contain non-UTF8 filenames that can crash logging)
    if args_cli.store_code_state:
        runner.add_git_repo_to_log(__file__)
    else:
        runner.git_status_repos = []
    # load the checkpoint
    if resume_requested:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        if protocol.v2:
            if args_cli.teacher_checkpoint:
                runner.load_teacher_v2(resume_path)
            elif args_cli.student_checkpoint:
                runner.bootstrap_student_v2(resume_path)
            elif args_cli.ppo_checkpoint:
                runner.bootstrap_robustness_v2(resume_path)
            else:
                runner.load(resume_path, load_optimizer=True)
        elif args_cli.resume_policy_only:
            _load_runner_checkpoint_with_policy_fallback(
                runner,
                resume_path,
                env.unwrapped.device,
                load_optimizer=False,
                allow_partial_policy=not args_cli.strict_checkpoint_loading,
            )
            runner.current_learning_iteration = 0
            print("[INFO]: Resume mode = policy-only (optimizer state skipped, iteration reset to 0).")
            if args_cli.reset_action_std is not None:
                target_std = max(float(args_cli.reset_action_std), 1e-4)
                policy_module = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic
                with torch.no_grad():
                    if hasattr(policy_module, "std"):
                        policy_module.std.fill_(target_std)
                        print(f"[INFO]: Reset action std (std) to {target_std:.4f}")
                    elif hasattr(policy_module, "log_std"):
                        log_target_std = torch.log(
                            torch.tensor(target_std, device=policy_module.log_std.device, dtype=policy_module.log_std.dtype)
                        )
                        policy_module.log_std.fill_(log_target_std)
                        print(f"[INFO]: Reset action std (log_std) to log({target_std:.4f})")
        else:
            # load previously trained model + optimizer state
            _load_runner_checkpoint_with_policy_fallback(
                runner,
                resume_path,
                env.unwrapped.device,
                load_optimizer=True,
                allow_partial_policy=False,
            )

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_yaml(
        os.path.join(log_dir, "params", "torsion_spring.yaml"),
        {
            "spring_backend": args_cli.spring_backend,
            "calibration_status": spring_calibration_status,
            "profile_id": spring_profile_id,
            "profile_sha256": spring_profile_sha256,
            "joint_aliases": [f"damper_{index}" for index in range(6)],
            "joint_names": list(env_cfg.damper_joint_names),
            "stiffness_nm_per_rad": list(env_cfg.spring_stiffness_nm_per_rad),
            "damping_nm_s_per_rad": list(env_cfg.spring_damping_nm_s_per_rad),
            "neutral_angle_rad": [
                float(env_cfg.robot_cfg.init_state.joint_pos.get(joint_name, 0.0))
                for joint_name in env_cfg.damper_joint_names
            ],
            "resume_checkpoint_calibration_status": resume_checkpoint_calibration_status,
        },
    )
    if physics_profile is not None:
        write_training_profile_snapshot(
            log_dir,
            physics_profile,
            config_application=physics_profile_config_application,
            runtime_application=physics_profile_runtime_application,
        )
    if sensor_dr_profile is not None:
        sensor_dr_snapshot = {
            **sensor_dr_profile.to_dict(),
            "profile_sha256": sensor_dr_profile_sha256,
        }
        dump_yaml(
            os.path.join(log_dir, "params", "sensor_domain_randomization_v2.yaml"),
            sensor_dr_snapshot,
        )

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
