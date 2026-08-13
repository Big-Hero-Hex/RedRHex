---
id: torsion-spring-v11-checkpoint
title: Torsion-Spring V11 Provisional Checkpoint
lang: en
audience: developer
type: experiment-summary
status: published
owner: sim2real
last_reviewed: 2026-08-13
---

<a id="question"></a>
## Question

Did the provisional linear spring implementation provide enough simulator evidence to select `explicit` or `native` before physical calibration?

<a id="method"></a>
## Method

The checkpoint used source commit `185a4fd5627ae6e8c0d33caad6ab38cea3b09e0a`, seed 0, and uncalibrated defaults of `200 N*m/rad`, zero damping, and provisional neutral angles. Four `spring-release` traces compared explicit/native at 120 Hz and 240 Hz under runtime bundle hash `dba80874b37fb0895ac6b90d353eed375b2f90a35edc88b06cdbb89162f69ec7`. One-iteration ForwardFast seed-42 smoke runs checked environment creation, metadata, checkpoint shape, and playback.

<a id="results"></a>
## Results

Selection status was `blocked_uncalibrated`; `physics_passed` was false and `selected_backend` was null.

- Explicit ran away at both timesteps, with maximum amplitude ratios about 2,391.54 at 120 Hz and 1,767.15 at 240 Hz. It failed energy, fixture, unwrap-ambiguity, runaway, and cross-timestep gates; rebound peak difference was about 85.53%.
- Native remained finite, completed rebounds, and passed fixture checks, but failed energy/work balance and timestep agreement. Residual fractions were 1.00 at 120 Hz and about 2.78 at 240 Hz against the 0.02 limit; rebound peak difference was about 61.76%.
- Both smoke runs produced 12-action/56-observation checkpoints stamped `uncalibrated`; deployment validation rejected them. Native playback rendered 120 frames successfully.

<a id="limitations"></a>
## Limitations

No physical calibration or holdout existed, so the simulator comparison cannot establish real-spring fidelity. The applied-torque comparison uses an implicit-PD estimate rather than a measured PhysX joint torque. One-iteration smoke runs verify integration only, not locomotion quality.

<a id="decision"></a>
## Decision impact

V11 is retained as an implementation checkpoint, not backend-selection or deployment evidence. The next required step is approved physical `damper_0` / `Revolute_5` calibration and holdout, followed by a repeated four-run characterization with an authenticated profile.

<a id="provenance"></a>
## Provenance

The original detailed report and commands remain reachable from `archive/source/torsion-spring-2026-08-13` at `docs/torsion_spring_workflow.md`. Raw outputs and logs remain uncommitted runtime artifacts.
