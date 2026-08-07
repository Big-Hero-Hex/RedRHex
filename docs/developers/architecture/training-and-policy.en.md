---
id: training-policy-architecture
title: Training and Policy Architecture
lang: en
audience: developer
type: explanation
status: active
owner: training
last_reviewed: 2026-08-07
---

<a id="tasks"></a>
## Tasks and agents

`Template-Redrhex-Direct-v0` registers the complete environment. `Template-Redrhex-ForwardFast-Direct-v0` derives a deterministic, forward-only stage-1 profile. Each exposes default PPO, privileged-teacher PPO, and distillation entry points; SKRL entries remain registered for compatibility.

<a id="observations"></a>
## Observation paths

The deployable policy consumes one 56-value frame plus four prior frames, giving 280 values. The current 56-value order is base linear velocity, base angular velocity, projected gravity, main-drive sine/cosine and scaled velocity, ABAD position and velocity, the 3-value command, gait phase sine/cosine, and 12 previous actions.

Default PPO maps `policy + history` to both actor and critic. A privileged teacher uses the `teacher` group. Distillation maps `policy + history` to the student and `teacher` to the teacher network. Do not expose privileged training-only signals to deployment.

<a id="curriculum"></a>
## Curriculum and action path

The five stages isolate forward, lateral, diagonal, yaw, and mixed commands. Stage-specific command distributions, residual scales, warmup, reward multipliers, and safety thresholds alter behavior. The stage pipeline defaults to full resume. `play.py` infers a stage from a checkpoint run-name suffix unless explicitly disabled.

Policy output has 12 actions: six main-drive and six ABAD controls. Damper joints are passive model elements and receive no deployed command. The environment computes one control intent at the training control rate and applies simulator-specific targets through Isaac.

<a id="randomization"></a>
## Randomization and symmetry

The complete environment includes terrain, actuator, per-leg fault, delay, noise, and push controls. Recovery baselines may disable them; robustness claims must record the resolved configuration. Left-right data augmentation is currently disabled because the tripod grouping is not mirror-symmetric while gait phase is unchanged.

<a id="artifacts"></a>
## Artifact contract

Training writes timestamped runs under `logs/rsl_rl/<experiment>/`, with `model_*.pt`, events, and resolved parameters. Playback exports JIT and ONNX beside the checkpoint. Panel and deployment tooling depend on this layout; changes require compatibility tests and a release entry.
