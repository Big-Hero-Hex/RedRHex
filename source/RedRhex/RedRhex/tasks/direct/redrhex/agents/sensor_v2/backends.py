"""Executable Isaac/RSL backends for the Sensor Distillation V2 runners."""

from __future__ import annotations

import copy
import importlib.metadata
import math
import os
import platform
import time
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
from rsl_rl.modules import ActorCritic
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from redrhex_policy_io import SensorCalibrationProfileV2, StudentObservationContractV2

from .bundle_contract import causal_attitude_parameters_v2, environment_rest_projected_gravity_v2
from ...sensor_v2_action import forward_residual_action_contract_v2_from_config
from .checkpoint import (
    CheckpointIntentV2,
    CheckpointKindV2,
    CheckpointManifestV2,
    architecture_hash_v2,
    canonical_hash_v2,
    canonical_training_config_hash_v2,
    file_sha256_v2,
    load_checkpoint_v2,
    save_checkpoint_v2,
)
from .distillation import SensorDistillationLossWeightsV2, SensorDistillationV2
from .export import BundleRecordsV2
from .models import (
    ACTION_DIM_V2,
    SENSOR_HISTORY_LENGTH_V2,
    SensorStudentCoreV2,
    SensorStudentTeacherV2,
    pad_main_actions_v2,
    strict_forward_actions_v2,
)
from .ppo import SensorActorCriticV2, SensorPPOBatchV2, SensorPPOV2
from .schedules import LinearWeightScheduleV2, RolloutMixtureScheduleV2
from .storage import (
    SensorDistillationBatchV2,
    SensorDistillationStorageV2,
    causal_gae_step_v2,
    causal_transition_mask_v2,
    clone_observations_v2,
    has_trainable_history_v2,
)


_TEACHER_CAPABILITIES = frozenset(
    {"strict_checkpoint_v2", "teacher_rollout_v2", "versioned_provenance_v2"}
)
_DISTILLATION_CAPABILITIES = frozenset(
    {
        "strict_checkpoint_v2",
        "two_input_sensor_actor_v2",
        "three_action_streams_v2",
        "next_frame_terminal_mask_v2",
    }
)
_PPO_CAPABILITIES = frozenset(
    {
        "strict_checkpoint_v2",
        "two_input_sensor_actor_v2",
        "asymmetric_critic_v2",
        "distilled_actor_exact_bootstrap_v2",
    }
)
_ROBUST_PPO_CAPABILITIES = _PPO_CAPABILITIES | frozenset(
    {"robustness_bootstrap_v2"}
)

_TENSORBOARD_ALIASES_V2 = {
    "Distill/action_loss_main": ("loss/main_drive_huber", "loss/teacher_bc_main"),
    "Distill/action_loss_abad": ("loss/forward_abad_huber", "loss/teacher_bc_abad"),
    "Distill/action_mae_main": ("mae/main_drive",),
    "Distill/action_mae_abad": ("mae/forward_abad",),
    "Distill/teacher_student_disagreement": (
        "rollout/teacher_student_disagreement",
    ),
    "Estimator/base_vel_rmse_x": ("rmse/base_velocity_x",),
    "Estimator/base_vel_rmse_y": ("rmse/base_velocity_y",),
    "Estimator/base_vel_rmse_z": ("rmse/base_velocity_z",),
}

def _mount_quaternion_wxyz(rpy: Any) -> tuple[float, float, float, float]:
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _package_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    for distribution in ("rsl-rl-lib", "isaaclab"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _environment_provenance_v2(env: Any) -> dict[str, str]:
    cfg = env.cfg
    values = {
        "sensor_dr_profile_id": getattr(cfg, "sensor_dr_profile_id", None),
        "sensor_dr_profile_sha256": getattr(cfg, "sensor_dr_profile_sha256", None),
        "sensor_dr_profile_purpose": getattr(cfg, "sensor_dr_profile_purpose", None),
        "sensor_dr_physical_stage_scale": getattr(
            cfg, "sensor_dr_physical_stage_scale", None
        ),
        "sensor_history_action_gate": getattr(
            cfg, "sensor_history_action_gate", None
        ),
    }
    return {name: str(value) for name, value in values.items() if value is not None}


def _build_bundle_records(env: Any) -> BundleRecordsV2:
    cfg = env.cfg
    attitude_mode = str(cfg.sensor_attitude_mode)
    if attitude_mode == "validated_quaternion":
        attitude_parameters = (
            ("max_orientation_variance", float(cfg.sensor_quaternion_covariance_max)),
            ("quaternion_norm_tolerance", 1.0e-3),
        )
    else:
        attitude_parameters = causal_attitude_parameters_v2(
            correction_gain=float(cfg.sensor_gravity_correction_gain),
            accel_norm_gate_m_s2=cfg.sensor_accel_norm_gate_mps2,
            gravity_vector_m_s2=getattr(cfg.sim, "gravity", (0.0, 0.0, -9.81)),
        )
    observation = StudentObservationContractV2(
        attitude_mode=attitude_mode,
        imu_frame_id=str(cfg.sensor_imu_frame_id),
        policy_body_frame_id="base_link",
        imu_to_body_wxyz=_mount_quaternion_wxyz(cfg.sensor_imu_mount_rpy_rad),
        rest_projected_gravity=environment_rest_projected_gravity_v2(env),
        attitude_parameters=attitude_parameters,
    )

    action = forward_residual_action_contract_v2_from_config(cfg)
    calibration = SensorCalibrationProfileV2.provisional(
        observation,
        action,
        profile_id=str(cfg.calibration_profile_id),
    )
    contract_record = observation.to_dict()
    versions = _package_versions()
    versions.update(_environment_provenance_v2(env))
    return BundleRecordsV2(
        contract=contract_record,
        action_contract=action.to_dict(),
        calibration=calibration.to_dict(),
        training_calibration=calibration.to_dict(),
        feature_layout={"features": contract_record["feature_layout"]},
        versions=versions,
    )


def _manifest(
    *,
    kind: CheckpointKindV2,
    stage: str,
    model: torch.nn.Module,
    config: dict[str, Any],
    records: BundleRecordsV2,
    iteration: int,
    source_checkpoint_hash: str | None = None,
    source_checkpoint_kind: str | None = None,
) -> CheckpointManifestV2:
    assert records.contract is not None
    assert records.action_contract is not None
    assert records.calibration is not None
    return CheckpointManifestV2(
        kind=kind,
        stage=stage,
        observation_contract_id=str(records.contract["contract_id"]),
        contract_hash=canonical_hash_v2(records.contract),
        action_contract_id=str(records.action_contract["contract_id"]),
        action_contract_hash=canonical_hash_v2(records.action_contract),
        calibration_hash=canonical_hash_v2(records.calibration),
        architecture_hash=architecture_hash_v2(model),
        config_hash=canonical_hash_v2(config),
        canonical_config_hash=canonical_training_config_hash_v2(config),
        training_seed=config["seed"],
        action_order=tuple(records.action_contract["action_ordering"]),
        iteration=int(iteration),
        source_checkpoint_hash=source_checkpoint_hash,
        source_checkpoint_kind=source_checkpoint_kind,
        package_versions=dict(records.versions),
    )


def _records_with_manifest(records: BundleRecordsV2, manifest: CheckpointManifestV2) -> BundleRecordsV2:
    return replace(records, checkpoint=manifest.to_dict())


def _restore_records(payload: dict[str, Any], fallback: BundleRecordsV2) -> BundleRecordsV2:
    raw = payload.get("extra_state", {}).get("bundle_records")
    if not isinstance(raw, dict):
        return fallback
    return BundleRecordsV2(**raw)


def _assert_shared_contract(actual: CheckpointManifestV2, expected: CheckpointManifestV2) -> None:
    fields = (
        "observation_contract_id",
        "contract_hash",
        "action_contract_id",
        "action_contract_hash",
        "calibration_hash",
        "action_order",
    )
    mismatches = [name for name in fields if getattr(actual, name) != getattr(expected, name)]
    if mismatches:
        raise ValueError(f"V2 source checkpoint changes shared contracts: {', '.join(mismatches)}")


def _assert_inference_identity(
    actual: CheckpointManifestV2,
    expected: CheckpointManifestV2,
    *,
    model: torch.nn.Module,
    allowed_stages: set[str],
) -> None:
    """Validate immutable inference identity while ignoring training-only config."""

    _assert_shared_contract(actual, expected)
    if actual.kind != expected.kind:
        raise ValueError(
            f"V2 inference checkpoint kind {actual.kind!r} does not match {expected.kind!r}"
        )
    if actual.stage not in allowed_stages:
        raise ValueError(
            f"V2 inference checkpoint stage {actual.stage!r} is not in {sorted(allowed_stages)}"
        )
    if actual.architecture_hash != architecture_hash_v2(model):
        raise ValueError("V2 inference checkpoint changes the policy architecture")


def _teacher_policy(obs: TensorDict, policy_cfg: dict[str, Any], device: str) -> ActorCritic:
    hidden = policy_cfg.get("teacher_hidden_dims", policy_cfg.get("actor_hidden_dims", [256, 128, 128]))
    normalization = bool(
        policy_cfg.get("teacher_obs_normalization", policy_cfg.get("actor_obs_normalization", True))
    )
    return StrictForwardTeacherActorCriticV2(
        obs,
        {
            "policy": ["teacher_physical_v2"],
            "critic": ["critic_privileged_v2"],
        },
        12,
        actor_obs_normalization=normalization,
        critic_obs_normalization=normalization,
        actor_hidden_dims=hidden,
        critic_hidden_dims=hidden,
        activation=str(policy_cfg.get("activation", "elu")),
        init_noise_std=float(policy_cfg.get("init_noise_std", 0.55) or 0.55),
    ).to(device)


class StrictForwardTeacherActorCriticV2(ActorCritic):
    """Stock-RSL-compatible teacher with only six stochastic main actions.

    The environment and hardware contract expose a 12-D public action for
    compatibility, but ABAD is fixed to zero throughout forward F0--F5.  This
    class keeps PPO log-probability, entropy, and KL strictly six-dimensional,
    while padding the public action and distribution statistics to the 12-D
    storage shape expected by RSL-RL.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        **kwargs: Any,
    ) -> None:
        if int(num_actions) != ACTION_DIM_V2:
            raise ValueError(
                f"Teacher V2 requires a {ACTION_DIM_V2}-D public action contract"
            )
        if bool(kwargs.get("state_dependent_std", False)):
            raise ValueError("Teacher V2 does not support state-dependent action noise")
        super().__init__(
            obs,
            obs_groups,
            ACTION_DIM_V2 // 2,
            **kwargs,
        )

    def act(self, obs: TensorDict, **kwargs: Any) -> torch.Tensor:
        return pad_main_actions_v2(super().act(obs, **kwargs))

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        return pad_main_actions_v2(super().act_inference(obs))

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.shape[-1] != ACTION_DIM_V2:
            raise ValueError(
                f"Teacher V2 actions must end in dimension {ACTION_DIM_V2}"
            )
        return self.distribution.log_prob(actions[..., : ACTION_DIM_V2 // 2]).sum(
            dim=-1
        )

    @property
    def action_mean(self) -> torch.Tensor:
        return pad_main_actions_v2(self.distribution.mean)

    @property
    def action_std(self) -> torch.Tensor:
        main = self.distribution.stddev
        # RSL-RL computes KL over its public 12-D storage tensors.  Constant
        # unit dummy statistics make each deterministic ABAD contribution
        # exactly zero without introducing division-by-zero or NaN.
        return torch.cat((main, torch.ones_like(main)), dim=-1)


class _TeacherInferenceAdapter(torch.nn.Module):
    """Expose a stock TensorDict teacher as a one-tensor inference module."""

    def __init__(self, policy: ActorCritic) -> None:
        super().__init__()
        self.policy = policy

    def act_inference(self, privileged_observation: torch.Tensor) -> torch.Tensor:
        normalized = self.policy.actor_obs_normalizer(privileged_observation)
        return pad_main_actions_v2(self.policy.actor(normalized))


class VersionedTeacherBackendV2(OnPolicyRunner):
    sensor_v2_capabilities = _TEACHER_CAPABILITIES

    def __init__(self, env: Any, config: dict[str, Any], *, log_dir: str | None, device: str) -> None:
        if not hasattr(env.cfg, "sensor_history_action_gate"):
            raise ValueError("F1 Teacher V2 requires the Sensor V2 environment contract")
        # Teacher A observes privileged instantaneous state, not sensor history.
        # Leaving the student warmup gate enabled would make PPO assign the
        # sampled-action log probability to a zero residual actually executed
        # for the first 60 samples of every episode.
        env.cfg.sensor_history_action_gate = False
        self._manifest_config = copy.deepcopy(config)
        self._base_records = _build_bundle_records(env)
        self.checkpoint_manifest: CheckpointManifestV2 | None = None
        self.bundle_records = self._base_records
        self._teacher_learn_active = False
        self._teacher_learn_has_iterations = False
        super().__init__(env, copy.deepcopy(config), log_dir=log_dir, device=device)

    def _construct_algorithm(self, obs: TensorDict):
        """Construct stock PPO around the strict six-stochastic-action policy."""

        from rsl_rl.algorithms import PPO

        policy_class = self.policy_cfg.pop("class_name")
        if policy_class != "StrictForwardTeacherActorCriticV2":
            raise ValueError(
                "F1 Teacher V2 requires StrictForwardTeacherActorCriticV2"
            )
        actor_critic = StrictForwardTeacherActorCriticV2(
            obs,
            self.cfg["obs_groups"],
            self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)
        algorithm_class = self.alg_cfg.pop("class_name")
        if algorithm_class != "PPO":
            raise ValueError("F1 Teacher V2 supports only the stock PPO objective")
        algorithm = PPO(
            actor_critic,
            device=self.device,
            **self.alg_cfg,
            multi_gpu_cfg=self.multi_gpu_cfg,
        )
        algorithm.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )
        return algorithm

    @property
    def _stage(self) -> str:
        return (
            "teacher_b_research_only"
            if self._manifest_config.get("lineage", "teacher_a") != "teacher_a"
            else "teacher_a"
        )

    def _expected_manifest(self, iteration: int) -> CheckpointManifestV2:
        return _manifest(
            kind=CheckpointKindV2.TEACHER,
            stage=self._stage,
            model=self.alg.policy,
            config=self._manifest_config,
            records=self._base_records,
            iteration=iteration,
        )

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Adapt RSL-RL's completed-index cursor to the V2 next-update cursor."""

        count = int(num_learning_iterations)
        if count < 0:
            raise ValueError("num_learning_iterations must be non-negative")
        start = self.current_learning_iteration
        self._teacher_learn_active = True
        self._teacher_learn_has_iterations = count > 0
        try:
            super().learn(count, init_at_random_ep_len=init_at_random_ep_len)
        except BaseException:
            raise
        else:
            self.current_learning_iteration = start + count
        finally:
            self._teacher_learn_active = False
            self._teacher_learn_has_iterations = False

    def save(self, path: str, infos: dict | None = None) -> None:
        checkpoint_iteration = self.current_learning_iteration + int(
            self._teacher_learn_active and self._teacher_learn_has_iterations
        )
        manifest = self._expected_manifest(checkpoint_iteration)
        self.checkpoint_manifest = manifest
        self.bundle_records = _records_with_manifest(self._base_records, manifest)
        save_checkpoint_v2(
            path,
            manifest=manifest,
            model=self.alg.policy,
            optimizer=self.alg.optimizer,
            update=checkpoint_iteration,
            extra_state={
                "infos": infos,
                "bundle_records": asdict(self.bundle_records),
                "tot_timesteps": self.tot_timesteps,
                "tot_time": self.tot_time,
            },
        )

    def load(
        self,
        path: str,
        load_optimizer: bool = True,
        map_location: str | None = None,
    ) -> dict[str, Any] | None:
        expected = self._expected_manifest(0)
        manifest, payload = load_checkpoint_v2(
            path,
            model=self.alg.policy,
            intent=CheckpointIntentV2.TEACHER_RESUME,
            expected_manifest=expected,
            optimizer=self.alg.optimizer if load_optimizer else None,
            map_location=map_location or self.device,
        )
        self.current_learning_iteration = manifest.iteration
        self.checkpoint_manifest = manifest
        self.bundle_records = _restore_records(payload, _records_with_manifest(self._base_records, manifest))
        extra = payload.get("extra_state", {})
        self.tot_timesteps = int(extra.get("tot_timesteps", 0))
        self.tot_time = float(extra.get("tot_time", 0.0))
        return extra.get("infos")

    def load_inference_v2(self, path: str, map_location: str | None = None) -> None:
        expected = self._expected_manifest(0)
        manifest, payload = load_checkpoint_v2(
            path,
            model=self.alg.policy,
            intent=CheckpointIntentV2.INFERENCE,
            map_location=map_location or self.device,
        )
        _assert_inference_identity(
            manifest,
            expected,
            model=self.alg.policy,
            allowed_stages={self._stage},
        )
        self.current_learning_iteration = manifest.iteration
        self.checkpoint_manifest = manifest
        self.bundle_records = _restore_records(
            payload, _records_with_manifest(self._base_records, manifest)
        )

    def get_exportable_actor(self) -> torch.nn.Module:
        return self.alg.policy


class _CustomRunnerV2:
    sensor_v2_capabilities: frozenset[str] = frozenset()
    checkpoint_kind: CheckpointKindV2
    stage: str

    def __init__(self, env: Any, config: dict[str, Any], *, log_dir: str | None, device: str) -> None:
        self.env = env
        self.cfg = copy.deepcopy(config)
        environment_provenance = _environment_provenance_v2(env)
        if environment_provenance:
            # The checkpoint config hash must bind the F4 profile even though
            # its numerical ranges belong to the environment, not the actor.
            self.cfg["environment_provenance_v2"] = environment_provenance
        self.device = device
        self.log_dir = log_dir
        self.num_steps_per_env = int(config["num_steps_per_env"])
        self.save_interval = int(config["save_interval"])
        self.total_updates = int(config["max_iterations"])
        self.current_learning_iteration = 0
        self.git_status_repos: list[str] = []
        self.writer = None
        self.checkpoint_manifest: CheckpointManifestV2 | None = None
        self._base_records = _build_bundle_records(env)
        self.bundle_records = self._base_records
        self.source_checkpoint_hash: str | None = None
        self.source_checkpoint_kind: str | None = None

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        self.git_status_repos.append(repo_file_path)

    def _prepare_writer(self) -> None:
        if self.log_dir is not None and self.writer is None:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

    @staticmethod
    def _history_ready_mask(observations: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return the per-environment actor-history validity mask."""

        try:
            ready = observations["history_ready_v2"]
        except KeyError as exc:
            raise KeyError("Sensor V2 runner requires history_ready_v2") from exc
        if ready.ndim == 2 and ready.shape[-1] == 1:
            ready = ready[:, 0]
        if ready.ndim != 1:
            raise ValueError(
                "history_ready_v2 must have shape [num_envs] or [num_envs,1]; "
                f"got {tuple(ready.shape)}"
            )
        return ready > 0.5

    def _warm_sensor_history(self, observations: Any) -> Any:
        """Let every environment independently reach a full causal history once."""

        obs = observations
        maximum_steps = SENSOR_HISTORY_LENGTH_V2 * 10
        zero_actions = torch.zeros(
            self.env.num_envs,
            ACTION_DIM_V2,
            device=self.env.device,
        )
        for _ in range(maximum_steps + 1):
            ready = self._history_ready_mask(obs)
            if has_trainable_history_v2(ready):
                return obs
            with torch.no_grad():
                obs, _, _, _ = self.env.step(zero_actions)
                obs = obs.to(self.device)
        raise RuntimeError(
            "Sensor V2 could not accumulate one complete physical history before "
            f"learning after {maximum_steps} steps"
        )

    def _write_metrics(
        self,
        metrics: dict[str, float],
        *,
        iteration: int,
        reward_mean: float,
        elapsed: float,
    ) -> None:
        if self.writer is not None:
            for name, value in metrics.items():
                self.writer.add_scalar(name, value, iteration)
            for alias, sources in _TENSORBOARD_ALIASES_V2.items():
                source = next((name for name in sources if name in metrics), None)
                if source is not None:
                    self.writer.add_scalar(alias, metrics[source], iteration)
            self.writer.add_scalar("Train/mean_reward", reward_mean, iteration)
            self.writer.add_scalar("Perf/iteration_time", elapsed, iteration)
        compact = " ".join(
            f"{name}={value:.5f}" for name, value in metrics.items() if name.startswith("loss/")
        )
        print(
            f"Learning iteration {iteration}/{self.total_updates} "
            f"reward={reward_mean:.4f} time={elapsed:.3f}s {compact}",
            flush=True,
        )

    @staticmethod
    def _collect_sensor_dr_extras(
        extras: Mapping[str, Any], values: dict[str, list[float]]
    ) -> None:
        log_values = extras.get("log")
        if not isinstance(log_values, Mapping):
            return
        for name, raw in log_values.items():
            if not str(name).startswith("SensorDR/"):
                continue
            tensor = torch.as_tensor(raw).detach().float()
            if tensor.numel() == 0 or not bool(torch.isfinite(tensor).all()):
                continue
            values[str(name)].append(float(tensor.mean().item()))

    def _expected_manifest(self, model: torch.nn.Module, iteration: int) -> CheckpointManifestV2:
        return _manifest(
            kind=self.checkpoint_kind,
            stage=self.stage,
            model=model,
            config=self.cfg,
            records=self._base_records,
            iteration=iteration,
            source_checkpoint_hash=self.source_checkpoint_hash,
            source_checkpoint_kind=self.source_checkpoint_kind,
        )

    def _save_model(
        self,
        path: str,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        extra_state: dict[str, Any],
    ) -> None:
        manifest = self._expected_manifest(model, self.current_learning_iteration)
        self.checkpoint_manifest = manifest
        self.bundle_records = _records_with_manifest(self._base_records, manifest)
        save_checkpoint_v2(
            path,
            manifest=manifest,
            model=model,
            optimizer=optimizer,
            update=self.current_learning_iteration,
            extra_state={
                **extra_state,
                "bundle_records": asdict(self.bundle_records),
            },
        )

    def _finish(self) -> None:
        if self.writer is not None:
            self.writer.flush()


class SensorDistillationBackendV2(_CustomRunnerV2):
    sensor_v2_capabilities = _DISTILLATION_CAPABILITIES
    checkpoint_kind = CheckpointKindV2.DISTILLED
    stage = "distillation_f2"

    def __init__(self, env: Any, config: dict[str, Any], *, log_dir: str | None, device: str) -> None:
        super().__init__(env, config, log_dir=log_dir, device=device)
        obs = env.get_observations().to(device)
        policy_cfg = copy.deepcopy(config["policy"])
        algorithm_cfg = copy.deepcopy(config["algorithm"])
        policy_cfg.pop("class_name", None)
        algorithm_cfg.pop("class_name", None)
        self.teacher_policy = _teacher_policy(obs, policy_cfg, device)
        self.student = SensorStudentCoreV2().to(device)
        self.model = SensorStudentTeacherV2(
            _TeacherInferenceAdapter(self.teacher_policy),
            self.student,
        ).to(device)
        weights = SensorDistillationLossWeightsV2(
            main_drive=float(algorithm_cfg["main_drive_loss_weight"]),
            forward_abad=float(algorithm_cfg["forward_abad_loss_weight"]),
            base_velocity=float(algorithm_cfg["velocity_loss_weight"]),
            next_frame=float(algorithm_cfg["dynamics_loss_weight"]),
            latent_regularization=float(algorithm_cfg["latent_regularization_weight"]),
            contact=float(algorithm_cfg["contact_loss_weight"]),
        )
        self.algorithm = SensorDistillationV2(
            self.student,
            learning_rate=float(algorithm_cfg["learning_rate"]),
            max_grad_norm=float(algorithm_cfg["max_grad_norm"]),
            weights=weights,
        )
        self.num_learning_epochs = int(algorithm_cfg["num_learning_epochs"])
        self.num_mini_batches = int(algorithm_cfg.get("num_mini_batches", 4))
        if self.num_mini_batches <= 0:
            raise ValueError("num_mini_batches must be positive")
        self.max_mini_batch_size = int(algorithm_cfg.get("max_mini_batch_size", 2048))
        if self.max_mini_batch_size <= 0:
            raise ValueError("max_mini_batch_size must be positive")
        self.schedule = RolloutMixtureScheduleV2(
            total_updates=self.total_updates,
            anneal_fraction=float(algorithm_cfg["rollout_anneal_fraction"]),
            initial_noise_std=float(algorithm_cfg["rollout_initial_noise_std"]),
        )
        self.teacher_loaded = False

    def load_teacher_v2(self, path: str) -> None:
        expected = _manifest(
            kind=CheckpointKindV2.TEACHER,
            stage="teacher_a",
            model=self.teacher_policy,
            config=self.cfg,
            records=self._base_records,
            iteration=0,
        )
        manifest, _ = load_checkpoint_v2(
            path,
            model=self.teacher_policy,
            intent=CheckpointIntentV2.DISTILLATION_BOOTSTRAP,
            map_location=self.device,
        )
        _assert_shared_contract(manifest, expected)
        if manifest.stage != "teacher_a":
            raise ValueError("production distillation accepts only Teacher A checkpoints")
        self.source_checkpoint_hash = file_sha256_v2(path)
        self.source_checkpoint_kind = manifest.kind
        self.teacher_loaded = True

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if not self.teacher_loaded:
            raise ValueError("F2 requires a strict Teacher A checkpoint loaded with --teacher_checkpoint")
        self._prepare_writer()
        if init_at_random_ep_len:
            print(
                "[INFO] Sensor V2 preserves the coherent reset CPG phase; "
                "random episode-age initialization is disabled."
            )
        obs = self._warm_sensor_history(self.env.get_observations().to(self.device))
        start = self.current_learning_iteration
        stop = start + int(num_learning_iterations)
        for iteration in range(start, stop):
            tick = time.time()
            storage = SensorDistillationStorageV2(
                self.num_steps_per_env * self.env.num_envs,
                device=self.device,
            )
            rewards_seen = []
            sensor_dr_metrics: dict[str, list[float]] = defaultdict(list)
            beta, noise_std = self.schedule.value(iteration)
            for _ in range(self.num_steps_per_env):
                rollout_obs = clone_observations_v2(
                    obs,
                    (
                        "sensor_history_v2",
                        "command_v2",
                        "teacher_physical_v2",
                        "aux_base_vel_target",
                        "history_ready_v2",
                    ),
                )
                history = rollout_obs["sensor_history_v2"]
                command = rollout_obs["command_v2"]
                privileged = rollout_obs["teacher_physical_v2"]
                ready = self._history_ready_mask(rollout_obs)
                if bool(ready.any()):
                    self.student.update_normalization(history[ready], command[ready])
                with torch.no_grad():
                    actions = self.model.rollout(
                        history,
                        command,
                        privileged,
                        beta=beta,
                        noise_std=noise_std,
                    )
                    next_obs, rewards, dones, extras = self.env.step(
                        actions.executed.to(self.env.device)
                    )
                    self._collect_sensor_dr_extras(extras, sensor_dr_metrics)
                    next_obs = next_obs.to(self.device)
                    next_ready = self._history_ready_mask(next_obs)
                    transition_valid = causal_transition_mask_v2(
                        dones.to(self.device),
                        next_ready,
                    )
                    if bool(ready.any()):
                        storage.add(
                            SensorDistillationBatchV2(
                                sensor_history=history[ready],
                                command=command[ready],
                                teacher_actions=actions.teacher[ready],
                                student_actions=actions.student[ready],
                                executed_actions=actions.executed[ready],
                                base_velocity_target=rollout_obs["aux_base_vel_target"][ready],
                                next_sensor_frame_target=next_obs["sensor_frame_v2"][ready],
                                terminal=~transition_valid[ready],
                            )
                        )
                rewards_seen.append(rewards.to(self.device))
                obs = next_obs
            metric_lists: dict[str, list[float]] = defaultdict(list)
            if storage.size == 0:
                raise RuntimeError(
                    "Sensor V2 F2 rollout contained no complete physical histories"
                )
            for _ in range(self.num_learning_epochs):
                for mini_batch in storage.mini_batches(
                    self.num_mini_batches,
                    max_batch_size=self.max_mini_batch_size,
                ):
                    for name, value in self.algorithm.update(mini_batch).items():
                        metric_lists[name].append(value)
            metrics = {name: sum(values) / len(values) for name, values in metric_lists.items()}
            metrics.update(
                {
                    name: sum(values) / len(values)
                    for name, values in sensor_dr_metrics.items()
                    if values
                }
            )
            metrics["rollout/history_ready_fraction"] = storage.size / float(
                self.num_steps_per_env * self.env.num_envs
            )
            metrics["rollout/teacher_coefficient"] = beta
            metrics["rollout/noise_std"] = noise_std
            reward_mean = float(torch.cat([value.reshape(-1) for value in rewards_seen]).mean().item())
            self.current_learning_iteration = iteration + 1
            self._write_metrics(
                metrics,
                iteration=self.current_learning_iteration,
                reward_mean=reward_mean,
                elapsed=time.time() - tick,
            )
            if (
                self.log_dir is not None
                and self.current_learning_iteration % self.save_interval == 0
            ):
                self.save(
                    os.path.join(
                        self.log_dir,
                        f"model_{self.current_learning_iteration}.pt",
                    )
                )
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
        self._finish()

    def save(self, path: str, infos: dict | None = None) -> None:
        del infos
        self._save_model(
            path,
            model=self.student,
            optimizer=self.algorithm.optimizer,
            extra_state={"teacher_state_dict": self.teacher_policy.state_dict()},
        )

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> None:
        expected = self._expected_manifest(self.student, 0)
        manifest, payload = load_checkpoint_v2(
            path,
            model=self.student,
            intent=CheckpointIntentV2.DISTILLATION_RESUME,
            expected_manifest=expected,
            optimizer=self.algorithm.optimizer if load_optimizer else None,
            map_location=map_location or self.device,
        )
        teacher_state = payload.get("extra_state", {}).get("teacher_state_dict")
        if not isinstance(teacher_state, dict):
            raise ValueError("distillation checkpoint is missing its frozen Teacher A state")
        self.teacher_policy.load_state_dict(teacher_state, strict=True)
        self.current_learning_iteration = manifest.iteration
        self.source_checkpoint_hash = manifest.source_checkpoint_hash
        self.source_checkpoint_kind = manifest.source_checkpoint_kind
        self.teacher_loaded = True
        self.checkpoint_manifest = manifest
        self.bundle_records = _restore_records(payload, _records_with_manifest(self._base_records, manifest))

    def load_inference_v2(self, path: str, map_location: str | None = None) -> None:
        expected = self._expected_manifest(self.student, 0)
        manifest, payload = load_checkpoint_v2(
            path,
            model=self.student,
            intent=CheckpointIntentV2.INFERENCE,
            map_location=map_location or self.device,
        )
        _assert_inference_identity(
            manifest,
            expected,
            model=self.student,
            allowed_stages={"distillation_f2"},
        )
        self.current_learning_iteration = manifest.iteration
        self.source_checkpoint_hash = manifest.source_checkpoint_hash
        self.source_checkpoint_kind = manifest.source_checkpoint_kind
        self.teacher_loaded = True
        self.checkpoint_manifest = manifest
        self.bundle_records = _restore_records(
            payload, _records_with_manifest(self._base_records, manifest)
        )

    def get_inference_policy(self, device: str | None = None):
        if device is not None:
            self.student.to(device)
        self.student.eval()
        return lambda obs: self.student.act(obs["sensor_history_v2"], obs["command_v2"])

    def get_exportable_actor(self) -> SensorStudentCoreV2:
        return self.student


class SensorOnPolicyBackendV2(_CustomRunnerV2):
    sensor_v2_capabilities = _PPO_CAPABILITIES
    checkpoint_kind = CheckpointKindV2.PPO
    stage = "ppo_f3"

    def __init__(self, env: Any, config: dict[str, Any], *, log_dir: str | None, device: str) -> None:
        super().__init__(env, config, log_dir=log_dir, device=device)
        obs = env.get_observations().to(device)
        policy_cfg = copy.deepcopy(config["policy"])
        algorithm_cfg = copy.deepcopy(config["algorithm"])
        policy_cfg.pop("class_name", None)
        algorithm_cfg.pop("class_name", None)
        if str(algorithm_cfg.get("schedule", "fixed")) != "fixed":
            raise ValueError("SensorPPOV2 supports only a fixed learning rate; adaptive KL is not implemented")
        self.policy = SensorActorCriticV2(
            critic_observation_dim=int(obs["critic_privileged_v2"].shape[-1]),
            critic_hidden_dims=tuple(policy_cfg["critic_hidden_dims"]),
            init_noise_std=float(policy_cfg["init_noise_std"]),
        ).to(device)
        weights = SensorDistillationLossWeightsV2(
            main_drive=1.0,
            forward_abad=0.0,
            base_velocity=float(algorithm_cfg["velocity_loss_weight"]),
            next_frame=float(algorithm_cfg["dynamics_loss_weight"]),
            latent_regularization=float(algorithm_cfg["latent_regularization_weight"]),
            contact=float(algorithm_cfg["contact_loss_weight"]),
        )
        self.algorithm = SensorPPOV2(
            self.policy,
            learning_rate=float(algorithm_cfg["learning_rate"]),
            clip_param=float(algorithm_cfg["clip_param"]),
            value_loss_coefficient=float(algorithm_cfg["value_loss_coef"]),
            entropy_coefficient=float(algorithm_cfg["entropy_coef"]),
            max_grad_norm=float(algorithm_cfg["max_grad_norm"]),
            auxiliary_weights=weights,
        )
        self.gamma = float(algorithm_cfg["gamma"])
        self.lam = float(algorithm_cfg["lam"])
        self.num_learning_epochs = int(algorithm_cfg["num_learning_epochs"])
        self.num_mini_batches = int(algorithm_cfg["num_mini_batches"])
        self.bc_schedule = LinearWeightScheduleV2(
            initial_weight=float(algorithm_cfg["teacher_bc_initial_weight"]),
            total_updates=self.total_updates,
            anneal_fraction=float(algorithm_cfg["teacher_bc_anneal_fraction"]),
        )
        self.teacher_policy = _teacher_policy(obs, {"teacher_hidden_dims": [256, 128, 128]}, device)
        self.teacher = _TeacherInferenceAdapter(self.teacher_policy).to(device)
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        self.student_loaded = False

    def bootstrap_student_v2(self, path: str) -> None:
        distilled = SensorStudentCoreV2().to(self.device)
        manifest, payload = load_checkpoint_v2(
            path,
            model=distilled,
            intent=CheckpointIntentV2.PPO_BOOTSTRAP,
            map_location=self.device,
        )
        expected = _manifest(
            kind=CheckpointKindV2.DISTILLED,
            stage="distillation_f2",
            model=distilled,
            config=self.cfg,
            records=self._base_records,
            iteration=0,
        )
        _assert_shared_contract(manifest, expected)
        teacher_state = payload.get("extra_state", {}).get("teacher_state_dict")
        if not isinstance(teacher_state, dict):
            raise ValueError("distilled checkpoint is missing the Teacher A state required by F3 BC")
        self.policy.bootstrap_distilled_actor(distilled)
        self.teacher_policy.load_state_dict(teacher_state, strict=True)
        self.source_checkpoint_hash = file_sha256_v2(path)
        self.source_checkpoint_kind = manifest.kind
        self.student_loaded = True

    @staticmethod
    def _flatten_observations(storage: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {name: torch.stack(values).flatten(0, 1) for name, values in storage.items()}

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if not self.student_loaded:
            raise ValueError(
                f"{self.stage} requires its strict bootstrap checkpoint before learning"
            )
        self._prepare_writer()
        if init_at_random_ep_len:
            print(
                "[INFO] Sensor V2 preserves the coherent reset CPG phase; "
                "random episode-age initialization is disabled."
            )
        obs = self._warm_sensor_history(self.env.get_observations().to(self.device))
        start = self.current_learning_iteration
        stop = start + int(num_learning_iterations)
        observation_names = (
            "sensor_history_v2",
            "command_v2",
            "critic_privileged_v2",
        )
        for iteration in range(start, stop):
            tick = time.time()
            observations: dict[str, list[torch.Tensor]] = {name: [] for name in observation_names}
            actions_seen: list[torch.Tensor] = []
            log_probabilities: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            rewards_seen: list[torch.Tensor] = []
            dones_seen: list[torch.Tensor] = []
            teacher_actions: list[torch.Tensor] = []
            velocity_targets: list[torch.Tensor] = []
            next_frame_targets: list[torch.Tensor] = []
            history_ready_seen: list[torch.Tensor] = []
            next_history_ready_seen: list[torch.Tensor] = []
            sensor_dr_metrics: dict[str, list[float]] = defaultdict(list)
            for _ in range(self.num_steps_per_env):
                rollout_obs = clone_observations_v2(
                    obs,
                    (
                        *observation_names,
                        "teacher_physical_v2",
                        "aux_base_vel_target",
                        "history_ready_v2",
                    ),
                )
                ready = self._history_ready_mask(rollout_obs)
                with torch.no_grad():
                    actions = self.policy.act(rollout_obs)
                    log_probability = self.policy.get_actions_log_prob(actions)
                    value = self.policy.evaluate(rollout_obs)
                    teacher_action = strict_forward_actions_v2(
                        self.teacher.act_inference(rollout_obs["teacher_physical_v2"])
                    )
                    next_obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    next_obs = next_obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    next_ready = self._history_ready_mask(next_obs)
                    time_outs = extras.get("time_outs")
                    if time_outs is not None:
                        rewards = rewards + self.gamma * value.squeeze(-1) * time_outs.to(self.device).float()
                    self._collect_sensor_dr_extras(extras, sensor_dr_metrics)
                for name in observation_names:
                    observations[name].append(rollout_obs[name])
                actions_seen.append(actions)
                log_probabilities.append(log_probability)
                values.append(value)
                rewards_seen.append(rewards)
                dones_seen.append(dones)
                teacher_actions.append(teacher_action)
                velocity_targets.append(rollout_obs["aux_base_vel_target"])
                next_frame_targets.append(next_obs["sensor_frame_v2"].detach().clone())
                history_ready_seen.append(ready)
                next_history_ready_seen.append(next_ready)
                obs = next_obs

            with torch.no_grad():
                next_value = self.policy.evaluate(obs)
            reward_tensor = torch.stack(rewards_seen).unsqueeze(-1)
            done_tensor = torch.stack(dones_seen).unsqueeze(-1).float()
            history_ready_tensor = torch.stack(history_ready_seen).unsqueeze(-1)
            next_history_ready_tensor = torch.stack(next_history_ready_seen).unsqueeze(-1)
            transition_valid_tensor = causal_transition_mask_v2(
                done_tensor,
                next_history_ready_tensor,
            )
            value_tensor = torch.stack(values)
            returns = torch.zeros_like(value_tensor)
            advantage = torch.zeros_like(next_value)
            for step in reversed(range(self.num_steps_per_env)):
                following_value = next_value if step == self.num_steps_per_env - 1 else value_tensor[step + 1]
                returns[step], advantage = causal_gae_step_v2(
                    reward=reward_tensor[step],
                    value=value_tensor[step],
                    following_value=following_value,
                    following_advantage=advantage,
                    current_ready=history_ready_tensor[step],
                    transition_valid=transition_valid_tensor[step],
                    gamma=self.gamma,
                    lam=self.lam,
                )
            advantages = returns - value_tensor

            ready_flat = history_ready_tensor.flatten().bool()
            ready_count = int(ready_flat.sum().item())
            if ready_count == 0:
                raise RuntimeError(
                    "Sensor V2 PPO rollout contained no complete physical histories"
                )
            flat_obs = {
                name: value[ready_flat]
                for name, value in self._flatten_observations(observations).items()
            }
            flat = {
                "actions": torch.stack(actions_seen).flatten(0, 1)[ready_flat],
                "log_probabilities": torch.stack(log_probabilities).flatten(0, 1)[ready_flat],
                "values": value_tensor.flatten(0, 1)[ready_flat],
                "advantages": advantages.flatten(0, 1)[ready_flat],
                "returns": returns.flatten(0, 1)[ready_flat],
                "teacher_actions": torch.stack(teacher_actions).flatten(0, 1)[ready_flat],
                "velocity_targets": torch.stack(velocity_targets).flatten(0, 1)[ready_flat],
                "next_frame_targets": torch.stack(next_frame_targets).flatten(0, 1)[ready_flat],
                "terminal": (~transition_valid_tensor).flatten(0, 1)[ready_flat],
            }
            flat["advantages"] = (
                flat["advantages"] - flat["advantages"].mean()
            ) / (flat["advantages"].std(unbiased=False) + 1.0e-8)
            batch_size = flat["actions"].shape[0]
            mini_batch_count = min(self.num_mini_batches, batch_size)
            mini_batch_size = batch_size // mini_batch_count
            metric_lists: dict[str, list[float]] = defaultdict(list)
            bc_coefficient = self.bc_schedule.value(iteration)
            for _ in range(self.num_learning_epochs):
                indices = torch.randperm(batch_size, device=self.device)
                for mini_batch in range(mini_batch_count):
                    begin = mini_batch * mini_batch_size
                    end = batch_size if mini_batch == mini_batch_count - 1 else begin + mini_batch_size
                    index = indices[begin:end]
                    batch = SensorPPOBatchV2(
                        observations={name: value[index] for name, value in flat_obs.items()},
                        actions=flat["actions"][index],
                        old_action_log_probability=flat["log_probabilities"][index],
                        old_values=flat["values"][index],
                        advantages=flat["advantages"][index],
                        returns=flat["returns"][index],
                        teacher_actions=flat["teacher_actions"][index],
                        base_velocity_target=flat["velocity_targets"][index],
                        next_sensor_frame_target=flat["next_frame_targets"][index],
                        terminal=flat["terminal"][index],
                    )
                    for name, value in self.algorithm.update(
                        batch,
                        teacher_bc_coefficient=bc_coefficient,
                    ).items():
                        metric_lists[name].append(value)
            # Keep normalization frozen from action sampling through all PPO
            # epochs.  Updating it during collection would make the stored old
            # log-probability and the recomputed log-probability use different
            # input transforms, corrupting the importance ratio.  The complete
            # ready-only rollout becomes the normalization state for the next
            # iteration after the current update is finished.
            self.policy.update_normalization(flat_obs)
            metrics = {name: sum(values) / len(values) for name, values in metric_lists.items()}
            metrics.update(
                {
                    name: sum(values) / len(values)
                    for name, values in sensor_dr_metrics.items()
                    if values
                }
            )
            metrics["rollout/history_ready_fraction"] = ready_count / float(
                self.num_steps_per_env * self.env.num_envs
            )
            metrics["policy/main_action_noise_std"] = float(
                self.policy.main_action_std.detach().mean().item()
            )
            metrics["policy/initial_main_action_noise_std"] = self.policy.initial_noise_std
            reward_mean = float(reward_tensor.mean().item())
            self.current_learning_iteration = iteration + 1
            self._write_metrics(
                metrics,
                iteration=self.current_learning_iteration,
                reward_mean=reward_mean,
                elapsed=time.time() - tick,
            )
            if (
                self.log_dir is not None
                and self.current_learning_iteration % self.save_interval == 0
            ):
                self.save(
                    os.path.join(
                        self.log_dir,
                        f"model_{self.current_learning_iteration}.pt",
                    )
                )
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
        self._finish()

    def save(self, path: str, infos: dict | None = None) -> None:
        del infos
        self._save_model(
            path,
            model=self.policy,
            optimizer=self.algorithm.optimizer,
            extra_state={"teacher_state_dict": self.teacher_policy.state_dict()},
        )

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> None:
        expected = self._expected_manifest(self.policy, 0)
        manifest, payload = load_checkpoint_v2(
            path,
            model=self.policy,
            intent=CheckpointIntentV2.PPO_RESUME,
            expected_manifest=expected,
            optimizer=self.algorithm.optimizer if load_optimizer else None,
            map_location=map_location or self.device,
        )
        teacher_state = payload.get("extra_state", {}).get("teacher_state_dict")
        if not isinstance(teacher_state, dict):
            raise ValueError("PPO checkpoint is missing the Teacher A state required by F3 BC")
        self.teacher_policy.load_state_dict(teacher_state, strict=True)
        self.current_learning_iteration = manifest.iteration
        self.source_checkpoint_hash = manifest.source_checkpoint_hash
        self.source_checkpoint_kind = manifest.source_checkpoint_kind
        self.student_loaded = True
        self.checkpoint_manifest = manifest
        self.bundle_records = _restore_records(payload, _records_with_manifest(self._base_records, manifest))

    def load_inference_v2(self, path: str, map_location: str | None = None) -> None:
        expected = self._expected_manifest(self.policy, 0)
        manifest, payload = load_checkpoint_v2(
            path,
            model=self.policy,
            intent=CheckpointIntentV2.INFERENCE,
            map_location=map_location or self.device,
        )
        allowed_stages = {"ppo_f3"} if self.stage == "ppo_f3" else {"ppo_f3", "ppo_f4"}
        _assert_inference_identity(
            manifest,
            expected,
            model=self.policy,
            allowed_stages=allowed_stages,
        )
        self.current_learning_iteration = manifest.iteration
        self.source_checkpoint_hash = manifest.source_checkpoint_hash
        self.source_checkpoint_kind = manifest.source_checkpoint_kind
        self.student_loaded = True
        self.checkpoint_manifest = manifest
        self.bundle_records = _restore_records(
            payload, _records_with_manifest(self._base_records, manifest)
        )

    def get_inference_policy(self, device: str | None = None):
        if device is not None:
            self.policy.to(device)
        self.policy.eval()
        return self.policy.act_inference

    def get_exportable_actor(self) -> SensorStudentCoreV2:
        return self.policy.actor


class SensorRobustnessBackendV2(SensorOnPolicyBackendV2):
    """F4 continuation with a fresh optimizer and evidence-profiled environment."""

    sensor_v2_capabilities = _ROBUST_PPO_CAPABILITIES
    stage = "ppo_f4"

    def bootstrap_robustness_v2(self, path: str) -> None:
        manifest, payload = load_checkpoint_v2(
            path,
            model=self.policy,
            intent=CheckpointIntentV2.ROBUSTNESS_BOOTSTRAP,
            map_location=self.device,
        )
        expected = self._expected_manifest(self.policy, 0)
        _assert_shared_contract(manifest, expected)
        if manifest.architecture_hash != architecture_hash_v2(self.policy):
            raise ValueError("F4 source checkpoint changes the Sensor V2 policy architecture")
        if manifest.stage not in {"ppo_f3", "ppo_f4"}:
            raise ValueError(
                "F4 accepts only an F3 PPO checkpoint or an earlier F4 curriculum checkpoint"
            )
        teacher_state = payload.get("extra_state", {}).get("teacher_state_dict")
        if not isinstance(teacher_state, dict):
            raise ValueError("F4 source checkpoint is missing the frozen Teacher A state")
        self.teacher_policy.load_state_dict(teacher_state, strict=True)
        self.source_checkpoint_hash = file_sha256_v2(path)
        self.source_checkpoint_kind = manifest.kind
        self.current_learning_iteration = 0
        self.student_loaded = True


def backend_factories_v2() -> dict[str, Any]:
    """Return the exact backend allowlist consumed by the outer runner factory."""

    return {
        "VersionedTeacherRunnerV2": VersionedTeacherBackendV2,
        "SensorDistillationRunnerV2": SensorDistillationBackendV2,
        "SensorOnPolicyRunnerV2": SensorOnPolicyBackendV2,
        "SensorRobustnessRunnerV2": SensorRobustnessBackendV2,
    }
