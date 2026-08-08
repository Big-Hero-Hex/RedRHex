---
id: environment-reward-development
title: Change the Environment or Rewards
lang: en
audience: developer
type: how-to
status: active
owner: core
last_reviewed: 2026-08-07
---

<a id="classify"></a>
## Classify the change

Decide whether the change affects simulator physics, observations, actions, rewards, curriculum, randomization, termination, training configuration, or only internal structure. Record an ADR for a cross-cutting decision, an approved design for a material feature, and a temporary plan for multi-step work.

<a id="trace"></a>
## Trace the contract

Start in `redrhex_env_cfg.py`, follow the consuming code in `redrhex_env.py`, then inspect PPO/distillation configuration, training/play/evaluation scripts, panel parameter construction, deployment parity, and ROS observation/action code. Search for mirrored dimensions, rates, joint order, scales, command limits, and artifact names.

<a id="test-first"></a>
## Add evidence first

Add the smallest test that fails for the intended behavior. Pure helpers and contract facts should have CPU tests; Isaac behavior needs bounded simulator validation. Observation or action changes require shape/order tests and Torch/ONNX/ROS parity. Reward changes require component-level diagnostics plus an evaluation or ablation protocol.

<a id="implement"></a>
## Implement one semantic axis

Avoid mixing a reward change with physics, timing, logging, or broad refactoring. Preserve old interfaces unless the design explicitly changes them. Do not make panel override files affect manual training without `--panel_overrides`, and do not load a calibration candidate without `--physics-profile`.

<a id="verify"></a>
## Verify and document

Run the relevant CPU suites, `validate_reform_stack.py` for Isaac-stack changes, a short PPO smoke for training changes, and deployment parity for contract changes. Update both locale files for affected operator/developer journeys, add a component release entry for shipped behavior, and declare the documentation impact in the PR.
