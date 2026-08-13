"""Executable Isaac/RSL backends for the Sensor Distillation V2 runners."""

from __future__ import annotations

import copy
import importlib.metadata
import math
import os
import platform
import time
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
from rsl_rl.modules import ActorCritic
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from redrhex_policy_io import (
    ForwardResidualActionContractV2,
    SensorCalibrationProfileV2,
    StudentObservationContractV2,
)

from .checkpoint import (
    CheckpointIntentV2,
    CheckpointKindV2,
    CheckpointManifestV2,
    architecture_hash_v2,
    canonical_hash_v2,
    file_sha256_v2,
    load_checkpoint_v2,
    save_checkpoint_v2,
)
from .distillation import SensorDistillationLossWeightsV2, SensorDistillationV2
from .export import BundleRecordsV2
from .models import SensorStudentCoreV2, SensorStudentTeacherV2
from .ppo import SensorActorCriticV2, SensorPPOBatchV2, SensorPPOV2
from .schedules import LinearWeightScheduleV2, RolloutMixtureScheduleV2
from .storage import SensorDistillationBatchV2, SensorDistillationStorageV2


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


def _first(value: Any, fallback: float) -> float:
    if isinstance(value, (list, tuple)) and value:
        return float(value[0])
    return float(fallback)


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


def _build_bundle_records(env: Any) -> BundleRecordsV2:
    cfg = env.cfg
    attitude_mode = str(cfg.sensor_attitude_mode)
    if attitude_mode == "validated_quaternion":
        attitude_parameters = (
            ("max_orientation_variance", float(cfg.sensor_quaternion_covariance_max)),
            ("quaternion_norm_tolerance", 1.0e-3),
        )
    else:
        low, high = (float(value) for value in cfg.sensor_accel_norm_gate_mps2)
        attitude_parameters = (
            ("accel_correction_gain", float(cfg.sensor_gravity_correction_gain)),
            ("accel_gate_high_m_s2", high),
            ("accel_gate_low_m_s2", low),
            ("gravity_magnitude_m_s2", 9.81),
        )
    observation = StudentObservationContractV2(
        attitude_mode=attitude_mode,
        imu_frame_id=str(cfg.sensor_imu_frame_id),
        policy_body_frame_id="base_link",
        imu_to_body_wxyz=_mount_quaternion_wxyz(cfg.sensor_imu_mount_rpy_rad),
        attitude_parameters=attitude_parameters,
    )

    drive_scale = _first(
        getattr(cfg, "stage_drive_vel_scale", None),
        getattr(cfg, "main_drive_vel_scale", 8.0),
    )
    residual_ratio = _first(
        getattr(cfg, "stage_forward_policy_drive_residual_scale", None),
        getattr(cfg, "main_drive_residual_scale", 0.1),
    )
    swing_velocity = 2.0 * math.pi * float(getattr(cfg, "swing_velocity_ratio", 1.5))
    action = ForwardResidualActionContractV2(
        main_residual_scale_rad_s=drive_scale * residual_ratio,
        forward_command_reference_m_s=float(getattr(cfg, "drive_bias_vx_ref", 0.45)),
        forward_bias_scale=_first(
            getattr(cfg, "stage_forward_bias_scale", None),
            getattr(cfg, "forward_drive_action_scale", 1.0),
        ),
        phase_lock_gain=float(getattr(cfg, "forward_phase_lock_gain", 1.2)),
        phase_correction_limit_rad_s=2.0,
        residual_cap_ratio=_first(
            getattr(cfg, "stage_forward_residual_cap_ratio", None),
            getattr(cfg, "forward_residual_cap_ratio", 0.26),
        ),
        main_velocity_limit_rad_s=max(swing_velocity * 1.5, drive_scale * 1.5),
    )
    calibration = SensorCalibrationProfileV2.provisional(
        observation,
        action,
        profile_id=str(cfg.calibration_profile_id),
    )
    contract_record = observation.to_dict()
    return BundleRecordsV2(
        contract=contract_record,
        action_contract=action.to_dict(),
        calibration=calibration.to_dict(),
        feature_layout={"features": contract_record["feature_layout"]},
        versions=_package_versions(),
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


def _teacher_policy(obs: TensorDict, policy_cfg: dict[str, Any], device: str) -> ActorCritic:
    hidden = policy_cfg.get("teacher_hidden_dims", policy_cfg.get("actor_hidden_dims", [256, 128, 128]))
    normalization = bool(
        policy_cfg.get("teacher_obs_normalization", policy_cfg.get("actor_obs_normalization", True))
    )
    return ActorCritic(
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


class _TeacherInferenceAdapter(torch.nn.Module):
    """Expose a stock TensorDict teacher as a one-tensor inference module."""

    def __init__(self, policy: ActorCritic) -> None:
        super().__init__()
        self.policy = policy

    def act_inference(self, privileged_observation: torch.Tensor) -> torch.Tensor:
        normalized = self.policy.actor_obs_normalizer(privileged_observation)
        return self.policy.actor(normalized)


class VersionedTeacherBackendV2(OnPolicyRunner):
    sensor_v2_capabilities = _TEACHER_CAPABILITIES

    def __init__(self, env: Any, config: dict[str, Any], *, log_dir: str | None, device: str) -> None:
        self._manifest_config = copy.deepcopy(config)
        self._base_records = _build_bundle_records(env)
        self.checkpoint_manifest: CheckpointManifestV2 | None = None
        self.bundle_records = self._base_records
        super().__init__(env, copy.deepcopy(config), log_dir=log_dir, device=device)

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

    def save(self, path: str, infos: dict | None = None) -> None:
        manifest = self._expected_manifest(self.current_learning_iteration)
        self.checkpoint_manifest = manifest
        self.bundle_records = _records_with_manifest(self._base_records, manifest)
        save_checkpoint_v2(
            path,
            manifest=manifest,
            model=self.alg.policy,
            optimizer=self.alg.optimizer,
            update=self.current_learning_iteration,
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

    def get_exportable_actor(self) -> torch.nn.Module:
        return self.alg.policy


class _CustomRunnerV2:
    sensor_v2_capabilities: frozenset[str] = frozenset()
    checkpoint_kind: CheckpointKindV2
    stage: str

    def __init__(self, env: Any, config: dict[str, Any], *, log_dir: str | None, device: str) -> None:
        self.env = env
        self.cfg = copy.deepcopy(config)
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

    def _randomize_episode_lengths(self) -> None:
        self.env.episode_length_buf = torch.randint_like(
            self.env.episode_length_buf,
            high=int(self.env.max_episode_length),
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
            self._randomize_episode_lengths()
        obs = self.env.get_observations().to(self.device)
        start = self.current_learning_iteration
        stop = start + int(num_learning_iterations)
        for iteration in range(start, stop):
            tick = time.time()
            storage = SensorDistillationStorageV2(
                self.num_steps_per_env * self.env.num_envs,
                device=self.device,
            )
            rewards_seen = []
            beta, noise_std = self.schedule.value(iteration)
            for _ in range(self.num_steps_per_env):
                history = obs["sensor_history_v2"]
                command = obs["command_v2"]
                privileged = obs["teacher_physical_v2"]
                self.student.update_normalization(history, command)
                with torch.no_grad():
                    actions = self.model.rollout(
                        history,
                        command,
                        privileged,
                        beta=beta,
                        noise_std=noise_std,
                    )
                    next_obs, rewards, dones, _ = self.env.step(actions.executed.to(self.env.device))
                    next_obs = next_obs.to(self.device)
                    storage.add(
                        SensorDistillationBatchV2(
                            sensor_history=history,
                            command=command,
                            teacher_actions=actions.teacher,
                            student_actions=actions.student,
                            executed_actions=actions.executed,
                            base_velocity_target=obs["aux_base_vel_target"],
                            next_sensor_frame_target=next_obs["sensor_frame_v2"],
                            terminal=dones.to(self.device),
                        )
                    )
                rewards_seen.append(rewards.to(self.device))
                obs = next_obs
            metric_lists: dict[str, list[float]] = defaultdict(list)
            for _ in range(self.num_learning_epochs):
                for name, value in self.algorithm.update(storage.batch()).items():
                    metric_lists[name].append(value)
            metrics = {name: sum(values) / len(values) for name, values in metric_lists.items()}
            metrics["rollout/teacher_coefficient"] = beta
            metrics["rollout/noise_std"] = noise_std
            reward_mean = float(torch.cat([value.reshape(-1) for value in rewards_seen]).mean().item())
            self.current_learning_iteration = iteration
            self._write_metrics(metrics, iteration=iteration, reward_mean=reward_mean, elapsed=time.time() - tick)
            if self.log_dir is not None and iteration % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{iteration}.pt"))
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
            raise ValueError("F3 requires a distilled checkpoint loaded with --student_checkpoint")
        self._prepare_writer()
        if init_at_random_ep_len:
            self._randomize_episode_lengths()
        obs = self.env.get_observations().to(self.device)
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
            for _ in range(self.num_steps_per_env):
                self.policy.update_normalization(obs)
                with torch.no_grad():
                    actions = self.policy.act(obs)
                    log_probability = self.policy.get_actions_log_prob(actions)
                    value = self.policy.evaluate(obs)
                    teacher_action = self.teacher.act_inference(obs["teacher_physical_v2"])
                    next_obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    next_obs = next_obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    time_outs = extras.get("time_outs")
                    if time_outs is not None:
                        rewards = rewards + self.gamma * value.squeeze(-1) * time_outs.to(self.device).float()
                for name in observation_names:
                    observations[name].append(obs[name])
                actions_seen.append(actions)
                log_probabilities.append(log_probability)
                values.append(value)
                rewards_seen.append(rewards)
                dones_seen.append(dones)
                teacher_actions.append(teacher_action)
                velocity_targets.append(obs["aux_base_vel_target"])
                next_frame_targets.append(next_obs["sensor_frame_v2"])
                obs = next_obs

            with torch.no_grad():
                next_value = self.policy.evaluate(obs)
            reward_tensor = torch.stack(rewards_seen).unsqueeze(-1)
            done_tensor = torch.stack(dones_seen).unsqueeze(-1).float()
            value_tensor = torch.stack(values)
            returns = torch.zeros_like(value_tensor)
            advantage = torch.zeros_like(next_value)
            for step in reversed(range(self.num_steps_per_env)):
                following_value = next_value if step == self.num_steps_per_env - 1 else value_tensor[step + 1]
                not_terminal = 1.0 - done_tensor[step]
                delta = reward_tensor[step] + self.gamma * not_terminal * following_value - value_tensor[step]
                advantage = delta + self.gamma * self.lam * not_terminal * advantage
                returns[step] = advantage + value_tensor[step]
            advantages = returns - value_tensor
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)

            flat_obs = self._flatten_observations(observations)
            flat = {
                "actions": torch.stack(actions_seen).flatten(0, 1),
                "log_probabilities": torch.stack(log_probabilities).flatten(0, 1),
                "values": value_tensor.flatten(0, 1),
                "advantages": advantages.flatten(0, 1),
                "returns": returns.flatten(0, 1),
                "teacher_actions": torch.stack(teacher_actions).flatten(0, 1),
                "velocity_targets": torch.stack(velocity_targets).flatten(0, 1),
                "next_frame_targets": torch.stack(next_frame_targets).flatten(0, 1),
                "terminal": torch.stack(dones_seen).flatten(0, 1),
            }
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
            metrics = {name: sum(values) / len(values) for name, values in metric_lists.items()}
            reward_mean = float(reward_tensor.mean().item())
            self.current_learning_iteration = iteration
            self._write_metrics(metrics, iteration=iteration, reward_mean=reward_mean, elapsed=time.time() - tick)
            if self.log_dir is not None and iteration % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{iteration}.pt"))
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

    def get_inference_policy(self, device: str | None = None):
        if device is not None:
            self.policy.to(device)
        self.policy.eval()
        return self.policy.act_inference

    def get_exportable_actor(self) -> SensorStudentCoreV2:
        return self.policy.actor


def backend_factories_v2() -> dict[str, Any]:
    """Return the exact backend allowlist consumed by the outer runner factory."""

    return {
        "VersionedTeacherRunnerV2": VersionedTeacherBackendV2,
        "SensorDistillationRunnerV2": SensorDistillationBackendV2,
        "SensorOnPolicyRunnerV2": SensorOnPolicyBackendV2,
    }
