---
id: sensor-distillation-v2-architecture
title: Sensor-Only Distillation V2 Architecture
lang: en
audience: developer
type: explanation
status: active
owner: training
last_reviewed: 2026-08-14
---

<a id="purpose"></a>
## Purpose

Sensor Distillation V2 is an additive forward-only research architecture. It exists to test whether one second of measurable temporal feedback can replace simulator velocity and controller-state inputs without destabilizing V1. The task ID is `Template-Redrhex-ForwardSensorV2-Direct-v0`; selecting any legacy task returns to the legacy contract.

<a id="data-flow"></a>
## Data flow

Timestamped IMU and measured main/ABAD encoder events enter `redrhex_policy_io`. Its causal preprocessor emits a strict 36-D frame. A 60-sample oldest-to-newest buffer and the current three-value command feed the TCN actor. The six learned main outputs are residuals around the versioned forward CPG; all six ABAD outputs are neutral. True base velocity and physical randomization state are training targets or Teacher A/critic inputs, never actor inputs.

The same contracts and hashes travel through simulation, real-trace replay, checkpoint manifests, the custom two-input ONNX exporter, the V2 ROS builder, and deployment preflight. A mismatch stops loading rather than selecting semantics from tensor dimensions.

<a id="training-stages"></a>
## Training stages

- F0 proves the procedural controller with zero policy residuals.
- F1 trains the 65-D physically privileged Teacher A. Teacher B, which adds controller targets, is an isolated ablation.
- F2 distills Teacher A into the causal TCN with rollout mixing and velocity/next-frame auxiliaries.
- F3 fine-tunes the identical actor with an asymmetric physical critic, annealed teacher BC, and persistent auxiliaries.
- F4 introduces evidence-backed sensor/actuator randomization and raw-event replay.
- F5 compares three-seed policy lineages and ablations using the existing command-sweep acceptance core.

No later stage may compensate for a failed earlier mapping, parity, or provenance gate.

The executable V2 backends keep the existing RSL-RL PPO implementation for F1 while replacing its checkpoint writer with the strict V2 format. F2 owns the three action streams and terminal-masked next-frame targets. F3 owns the asymmetric rollout, GAE/minibatches, exact distilled-actor bootstrap, and retained Teacher A state. The CLI can launch each stage or a fail-closed sequential F1 → F2 → F3 pipeline; it never infers transitions from tensor shape. The recovered browser route remains on the stacked Panel proposal branch pending separate review.

<a id="boundaries"></a>
## Boundaries

Contact supervision is absent because current simulator contact state is a phase proxy. The production IMU attitude mode and ABAD calibration are unverified. Learned ABAD, direct targets, lateral/yaw expansion, hardware motor enable, and physical promotion remain outside this architecture until new evidence and design approval exist. The stacked Panel proposal is limited to launch, strict checkpoint handoff, monitoring, and final-artifact routing; it cannot promote a model or relax an evidence gate.

<a id="verification-status"></a>
## Verification status

Dependency-light contract, training, replay, and ROS tests can establish implementation correctness. One-update Isaac gates have completed for F1, F2, F3, and the sequential pipeline. Deterministic forward acceptance, full-run quality, three-seed results, recorded-sensor replay, and hardware preflight remain separate pending evidence gates. Consult the [active plan](../../plans/active/2026-08-13-student-distillation-v2.en.md) for current completion state and the [approved design](../../designs/active/2026-08-13-student-distillation-v2.en.md) for exact interfaces.
