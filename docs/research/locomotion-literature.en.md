---
id: locomotion-literature
title: Locomotion Research Foundations
lang: en
audience: developer
type: explanation
status: active
owner: training
last_reviewed: 2026-08-07
---

<a id="purpose"></a>
## Purpose

This curated summary records the research ideas that motivated RedRHex interfaces. It does not claim that a cited method has been reproduced or that quadruped results transfer directly to a six-legged wheg robot.

<a id="terrain-adaptation"></a>
## Terrain and adaptation

Robust perceptive locomotion and RMA motivate terrain curriculum, history, privileged information during training, and a deployable observation boundary. Relevant sources include [Learning robust perceptive locomotion](https://arxiv.org/abs/2201.08117) and [Rapid Motor Adaptation](https://arxiv.org/abs/2107.04034).

<a id="teacher-student"></a>
## Teacher and student training

DreamWaQ and concurrent teacher-student work motivate separating privileged teacher observations from deployable student inputs and validating the distillation checkpoint path. See [DreamWaQ](https://arxiv.org/abs/2301.10602) and [Concurrent Teacher-Student](https://arxiv.org/abs/2403.04359).

<a id="terrain-skill"></a>
## Terrain and skill progression

ANYmal parkour and Extreme Parkour motivate staged or curriculum-driven skill acquisition, but RedRHex uses its own five-stage command design and wheg dynamics. See [ANYmal Parkour](https://www.science.org/doi/10.1126/scirobotics.adi7566) and [Extreme Parkour](https://arxiv.org/abs/2309.14341).

<a id="symmetry-actuation"></a>
## Symmetry and actuation

Morphological symmetry research supports augmentation only when the robot mapping and phase semantics are actually equivariant. RedRHex currently disables its inconsistent mirror transform. Actuator-network work motivates measuring and modeling actuation rather than assuming ideal targets. See [Morphological Symmetry](https://hybrid-robotics.berkeley.edu/publications/IROS2024_Symmetry_RL_LeggedLoco.pdf) and [Learning agile and dynamic motor skills](https://www.science.org/doi/10.1126/scirobotics.aau5872).

<a id="evaluation"></a>
## Evaluation boundary

MPC-versus-RL and CPG-RL work motivate controlled baselines and gait priors, not unsupported superiority claims. RedRHex must compare identical command envelopes, success/fall criteria, energy proxies, and hardware conditions before claiming a method advantage.
