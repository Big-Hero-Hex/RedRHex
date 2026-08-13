"""Isaac Lab configuration entry points for Sensor-Only Distillation V2."""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class SensorStudentPolicyCfgV2(RslRlPpoActorCriticCfg):
    class_name = "SensorActorCriticV2"
    init_noise_std = 0.55
    actor_obs_normalization = False
    critic_obs_normalization = False
    actor_hidden_dims = [128, 128]
    critic_hidden_dims = [256, 128, 128]
    activation = "elu"
    sensor_frame_dim = 36
    sensor_history_length = 60
    command_dim = 3
    latent_dim = 64


@configclass
class SensorStudentTeacherCfgV2(RslRlDistillationStudentTeacherCfg):
    class_name = "SensorStudentTeacherV2"
    init_noise_std = 0.0
    student_obs_normalization = False
    teacher_obs_normalization = True
    student_hidden_dims = [128, 128]
    teacher_hidden_dims = [256, 128, 128]
    activation = "elu"
    sensor_frame_dim = 36
    sensor_history_length = 60
    command_dim = 3
    latent_dim = 64


@configclass
class SensorDistillationAlgorithmCfgV2(RslRlDistillationAlgorithmCfg):
    class_name = "SensorDistillationV2"
    num_learning_epochs = 2
    learning_rate = 1.0e-3
    gradient_length = 1
    max_grad_norm = 1.0
    loss_type = "huber"
    main_drive_loss_weight = 1.0
    forward_abad_loss_weight = 0.0
    velocity_loss_weight = 0.5
    dynamics_loss_weight = 0.1
    latent_regularization_weight = 1.0e-4
    contact_loss_weight = 0.0
    rollout_anneal_fraction = 0.70
    rollout_initial_teacher_coefficient = 1.0
    rollout_final_teacher_coefficient = 0.0
    rollout_initial_noise_std = 0.05
    rollout_final_noise_std = 0.0


@configclass
class SensorPpoAlgorithmCfgV2(RslRlPpoAlgorithmCfg):
    class_name = "SensorPPOV2"
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.003
    num_learning_epochs = 6
    num_mini_batches = 4
    learning_rate = 3.0e-4
    schedule = "adaptive"
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.0
    teacher_bc_initial_weight = 0.2
    teacher_bc_anneal_fraction = 0.60
    velocity_loss_weight = 0.5
    dynamics_loss_weight = 0.1
    latent_regularization_weight = 1.0e-4
    contact_loss_weight = 0.0


def _teacher_policy() -> RslRlPpoActorCriticCfg:
    return RslRlPpoActorCriticCfg(
        init_noise_std=0.55,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )


def _teacher_algorithm() -> RslRlPpoAlgorithmCfg:
    return RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=6,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class ForwardSensorV2TeacherRunnerCfg(RslRlOnPolicyRunnerCfg):
    """F1 production lineage: physically meaningful 65-D Teacher A."""

    class_name = "VersionedTeacherRunnerV2"
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "redrhex_forward_v2_teacher"
    run_name = "forward_sensor_v2_teacher_a"
    logger = "tensorboard"
    clip_actions = 1.0
    obs_groups = {
        "policy": ["teacher_physical_v2"],
        "critic": ["critic_privileged_v2"],
    }
    policy = _teacher_policy()
    algorithm = _teacher_algorithm()
    checkpoint_kind = "teacher_v2"
    lineage = "teacher_a"


@configclass
class ForwardSensorV2TeacherBRunnerCfg(ForwardSensorV2TeacherRunnerCfg):
    """Research-only internal-target ablation, excluded from production lineage."""

    run_name = "forward_sensor_v2_teacher_b_ablation"
    obs_groups = {
        "policy": ["teacher_internal_target_ablation_v2"],
        "critic": ["critic_privileged_v2"],
    }
    lineage = "teacher_b_research_only"
    production_lineage_allowed = False


@configclass
class ForwardSensorV2DistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """F2 sensor-only TCN distillation configuration."""

    class_name = "SensorDistillationRunnerV2"
    num_steps_per_env = 24
    max_iterations = 800
    save_interval = 50
    experiment_name = "redrhex_forward_v2_distillation"
    run_name = "forward_sensor_v2_distilled"
    logger = "tensorboard"
    clip_actions = 1.0
    obs_groups = {
        "policy": ["sensor_history_v2", "command_v2"],
        "teacher": ["teacher_physical_v2"],
    }
    policy = SensorStudentTeacherCfgV2()
    algorithm = SensorDistillationAlgorithmCfgV2()
    checkpoint_kind = "student_distilled_v2"


@configclass
class ForwardSensorV2PpoRunnerCfg(RslRlOnPolicyRunnerCfg):
    """F3 asymmetric PPO configuration with sensor actor and privileged critic."""

    class_name = "SensorOnPolicyRunnerV2"
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "redrhex_forward_v2_ppo"
    run_name = "forward_sensor_v2_ppo"
    logger = "tensorboard"
    clip_actions = 1.0
    obs_groups = {
        "policy": ["sensor_history_v2", "command_v2"],
        "critic": ["critic_privileged_v2"],
    }
    policy = SensorStudentPolicyCfgV2()
    algorithm = SensorPpoAlgorithmCfgV2()
    checkpoint_kind = "student_ppo_v2"
