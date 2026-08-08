---
id: operator-physics-calibration
title: Calibrate Sim-to-Real Physics
lang: en
audience: operator
type: how-to
status: active
owner: sim2real
last_reviewed: 2026-08-07
---

<a id="safety"></a>
## Safety and evidence boundary

Calibration uses bounded scenarios and immutable evidence; it does not authorize policy deployment. Keep the robot lifted or otherwise constrained for initial probes, prepare physical E-stop and power cutoff, and stop if the low-level heartbeat, publisher graph, timing, mapping, or stationarity checks fail.

<a id="order"></a>
## Required order

1. Build and verify the ROS command contract.
2. Record direct physical measurements and their reference pose.
3. Run one bounded hardware probe without competing publishers.
4. Import the raw episode into the managed evidence store.
5. Replay the matching initial state and scenario in Isaac.
6. Compare held-out metrics and generate bounded candidates.
7. Build a profile only from authenticated direct measurements and evidence.
8. Promote a profile only after all audit and holdout gates pass.

<a id="tool"></a>
## Calibration tool

Use `python -m tools.sim2real --help` to list the current subcommands and required provenance fields. Scenario specifications live under `tools/sim2real/scenario_specs/`. Candidate profiles are never loaded by training or playback unless supplied explicitly with `--physics-profile`.

<a id="promotion"></a>
## Promotion rules

Do not promote evidence with missing hashes, unresolved joint mapping, aliased ROS publishers, incomplete initial state, failed geometry or mass/CoM audit, nonstationary holdout data, or missing held-out metrics. A replay match is evidence for the measured scenario, not proof that all locomotion physics are correct.

<a id="developer-context"></a>
## Developer context

See [sim-to-real architecture](../../developers/architecture/sim-to-real.en.md) for evidence roles, profile application, and implementation boundaries.
