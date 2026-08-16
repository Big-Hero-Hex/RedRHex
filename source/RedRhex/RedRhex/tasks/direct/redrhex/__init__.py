# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Template-Redrhex-Direct-v0",
    entry_point=f"{__name__}.redrhex_env:RedrhexEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.redrhex_env_cfg:RedrhexEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        "rsl_rl_teacher_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerPrivilegedTeacherCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg:RedrhexDistillationRunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Template-Redrhex-ForwardFast-Direct-v0",
    entry_point=f"{__name__}.redrhex_env:RedrhexEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.redrhex_env_cfg:RedrhexForwardFastEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerForwardFastCfg",
        "rsl_rl_teacher_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerForwardFastPrivilegedTeacherCfg"
        ),
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:RedrhexForwardFastDistillationRunnerCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)


gym.register(
    id="Template-Redrhex-ForwardSensorV2-Direct-v0",
    entry_point=f"{__name__}.redrhex_sensor_v2_env:RedrhexForwardSensorV2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.redrhex_sensor_v2_env_cfg:RedrhexForwardSensorV2EnvCfg"
        ),
        # The default V2 route is the versioned privileged Teacher A.  F2/F3
        # are selected explicitly with --agent and cannot alter a V1 runner.
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.sensor_v2.config:ForwardSensorV2TeacherRunnerCfg"
        ),
        "rsl_rl_teacher_v2_cfg_entry_point": (
            f"{agents.__name__}.sensor_v2.config:ForwardSensorV2TeacherRunnerCfg"
        ),
        "rsl_rl_teacher_b_v2_cfg_entry_point": (
            f"{agents.__name__}.sensor_v2.config:ForwardSensorV2TeacherBRunnerCfg"
        ),
        "rsl_rl_distillation_v2_cfg_entry_point": (
            f"{agents.__name__}.sensor_v2.config:ForwardSensorV2DistillationRunnerCfg"
        ),
        "rsl_rl_distillation_v2_no_aux_cfg_entry_point": (
            f"{agents.__name__}.sensor_v2.config:ForwardSensorV2DistillationNoAuxRunnerCfg"
        ),
        "rsl_rl_distillation_v2_velocity_cfg_entry_point": (
            f"{agents.__name__}.sensor_v2.config:ForwardSensorV2DistillationVelocityRunnerCfg"
        ),
        "rsl_rl_distillation_v2_velocity_dynamics_cfg_entry_point": (
            f"{agents.__name__}.sensor_v2.config:ForwardSensorV2DistillationVelocityDynamicsRunnerCfg"
        ),
        "rsl_rl_ppo_v2_cfg_entry_point": (
            f"{agents.__name__}.sensor_v2.config:ForwardSensorV2PpoRunnerCfg"
        ),
        "rsl_rl_robust_ppo_v2_cfg_entry_point": (
            f"{agents.__name__}.sensor_v2.config:ForwardSensorV2RobustPpoRunnerCfg"
        ),
    },
)
