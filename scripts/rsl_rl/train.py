# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""
#import isaaclab_tasks  # :)
#import ext_furuta_pendulum  # :)

import argparse
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
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

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

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

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
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
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

    # Reward/terrain override files written by the training panel or Reward Agent Lab.
    # Only applied when --panel_overrides is passed (the panel adds it when spawning
    # train.py); otherwise a stale override file must not silently reshape manual runs.
    _override_file = Path(__file__).parents[2] / "tools" / "training_panel" / "active_reward_override.json"
    _terrain_override_file = Path(__file__).parents[2] / "tools" / "training_panel" / "active_terrain_override.json"
    if args_cli.panel_overrides:
        if _override_file.exists():
            import json as _json
            from tools.training_panel.training_panel.reward_overrides import apply_reward_overrides

            _overrides = _json.loads(_override_file.read_text(encoding="utf-8"))
            _applied = apply_reward_overrides(env_cfg, _overrides)
            if _applied:
                print(f"[INFO] Training panel reward overrides applied: {', '.join(_applied)}")

        if _terrain_override_file.exists():
            import json as _json
            from tools.training_panel.training_panel.terrain import apply_terrain_overrides

            _terrain_overrides = _json.loads(_terrain_override_file.read_text(encoding="utf-8"))
            _applied = apply_terrain_overrides(env_cfg, _terrain_overrides)
            if _applied:
                print(f"[INFO] Training panel terrain overrides applied: {', '.join(_applied)}")
    else:
        for _stale in (_override_file, _terrain_override_file):
            if _stale.exists():
                print(
                    f"[WARN] Ignoring training panel override file {_stale} "
                    "(pass --panel_overrides to apply it)."
                )

    physics_profile = None
    physics_profile_config_application = None
    physics_profile_runtime_application = None
    if args_cli.physics_profile is not None:
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from tools.sim2real.physics_profile import (
            apply_profile_to_config,
            load_optional_profile,
            write_training_profile_snapshot,
        )

        physics_profile = load_optional_profile(args_cli.physics_profile)
        physics_profile_config_application = apply_profile_to_config(
            env_cfg, physics_profile
        )
        print(f"[INFO] Explicit physics profile applied to config: {physics_profile.profile_id}")

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

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        if agent_cfg.load_checkpoint and (
            os.path.isabs(agent_cfg.load_checkpoint) or os.path.exists(agent_cfg.load_checkpoint)
        ):
            resume_path = retrieve_file_path(agent_cfg.load_checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

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

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs (opt-in; some repos contain non-UTF8 filenames that can crash logging)
    if args_cli.store_code_state:
        runner.add_git_repo_to_log(__file__)
    else:
        runner.git_status_repos = []
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        if args_cli.resume_policy_only:
            _load_runner_checkpoint_with_policy_fallback(
                runner,
                resume_path,
                env.unwrapped.device,
                load_optimizer=False,
                allow_partial_policy=True,
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
    if physics_profile is not None:
        write_training_profile_snapshot(
            log_dir,
            physics_profile,
            config_application=physics_profile_config_application,
            runtime_application=physics_profile_runtime_application,
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
