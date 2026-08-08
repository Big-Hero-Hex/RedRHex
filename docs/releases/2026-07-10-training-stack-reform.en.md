---
id: milestone-training-stack-reform
title: 2026-07-10 Training Stack Reform Milestone
lang: en
audience: shared
type: release
status: published
owner: project
last_reviewed: 2026-08-07
---

<a id="scope"></a>
## Scope

This dated project milestone records the implemented training and deployment hardening available at commit `5cdc824` and later on the documentation source line. It is not a global semantic version.

<a id="training"></a>
## Training and environment

- Manual `train.py` applies panel override files only with `--panel_overrides`; panel-launched runs supply it explicitly.
- Action intent is computed once per control step rather than accumulating time twice across physics substeps.
- Observation-noise slices, initialization configuration, height termination, global step counting, and duplicate legacy reward handling were corrected.
- Physically inconsistent left-right augmentation is disabled.
- Full, ForwardFast, privileged-teacher, and distillation paths remain registered.

<a id="deployment"></a>
## Deployment and operations

- The mirrored deployment contract uses the training rate of 60 Hz, ABAD normalization `0.60`, and a 60-degree stage limit.
- Contract parity tests protect mirrored constants.
- IMU mount rotation and rest projected-gravity checks gate policy enable, pending site-specific hardware verification.
- Training Panel history writes use locking and atomic replacement.

<a id="remaining"></a>
## Remaining limitations

Contact sensing, base linear-velocity estimation, ambiguous diagonal reward magnitude, observation-side state mutation, physics mass/actuator assumptions, convergence-window semantics, and broader modularization were not resolved by this milestone. They remain documented priorities rather than implied fixes.
