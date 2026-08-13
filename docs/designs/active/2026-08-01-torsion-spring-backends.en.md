---
id: torsion-spring-backends-design
title: Passive Torsion-Spring Backends
lang: en
audience: developer
type: design
status: approved
owner: sim2real
last_reviewed: 2026-08-13
---

<a id="problem"></a>
## Problem

The six passive leg joints need a calibrated spring model that behaves consistently in training, playback, command sweeps, characterization, Training Panel actions, and deployment evidence. A provisional simulator result must not be mistaken for a selected or deployable physical model.

<a id="contract"></a>
## Stable contract

The canonical aliases are `damper_0` through `damper_5`. Each joint follows `tau = -k(q-q0)-c*qdot`, using unwrapped continuous-joint displacement. The joints remain passive and outside the policy action vector, so the policy contract stays at 12 actions and 56 observations.

Training, playback, evaluation, sim-to-real execution, and Panel processes accept exactly `explicit` or `native`. The chosen backend, ordered effective parameters, calibration status, profile identity/hash, deflection, torque estimate, potential energy, power, and passivity diagnostics are recorded with each run.

<a id="backends"></a>
## Backends

- `explicit` zeros PhysX spring gains and applies restoring effort every physics substep.
- `native` writes stiffness and damping to fixed-target PhysX implicit drives at the configured neutral angles.

Neither backend adds a spring-law clip, artificial velocity brake, or policy-controlled spring action. The native applied-torque channel is an implicit-PD estimate rather than force-sensor evidence. Repository defaults (`200 N*m/rad`, zero damping, provisional neutral angles, `explicit`) are an uncalibrated starting point, not a production choice.

<a id="evidence"></a>
## Evidence and selection

Physical calibration uses representative `damper_0`, immutable calibration and holdout episodes, mechanical-owner approval, and fail-closed quality gates. A calibrated fit propagates neutral-constrained stiffness to all aliases while keeping damping zero until separately measured. Backend selection requires both implementations to pass matched 120/240 Hz release characterization before retraining.

<a id="panel"></a>
## Training Panel propagation

Panel run creation validates and stores `spring_backend`; train, Play, automatic video, export, deployment validation, history, and remote synchronization reuse the recorded value. ForwardFast automatic video alone supplies `--initial_command forward`; interactive Play and full Direct recordings retain their existing command behavior.

<a id="non-goals"></a>
## Non-goals

This design does not select a backend without calibrated evidence, identify damping from static data, alter the policy tensor contract, redesign rewards, authorize hardware deployment, or treat the V11 smoke checkpoint as production evidence.
