---
id: student-distillation-v2-plan
title: Sensor-Only Student Distillation V2 Implementation Plan
lang: en
audience: developer
type: plan
status: active
owner: training
last_reviewed: 2026-08-14
---

<a id="objective"></a>
## Objective

Deliver the approved sensor-only forward route as additive CLI functionality, prove each evidence gate in order, and keep every V1 artifact and rollback path intact.

<a id="context"></a>
## Context

The [approved design](../../designs/active/2026-08-13-student-distillation-v2.en.md) fixes the 36-D frame, 60-frame history, separate command, residual-CPG action decoder, TCN, loss schedules, strict artifacts, two-input ONNX graph, and ROS safety boundary. The [code-path audit](../../research/2026-08-13-student-distillation-v2-audit.en.md) blocks contact supervision and hardware promotion. The core CLI route is reconstructed here; the recovered browser route is isolated on the stacked Panel physics/calibration proposal branch for separate review. A checked item means its code or local test exists; it does not imply a later full-run, multi-seed, recorded-sensor, or physical gate passed.

<a id="phased-checklist"></a>
## Phased checklist

<a id="phase-compatibility"></a>
### Compatibility and contracts

- [x] Publish bilingual audit, approved design, active plan, migration/rollback, and documentation-impact boundaries.
- [ ] Freeze V1 with observation, registration, checkpoint-loader, ROS, and preprocessing regression tests.
- [ ] Land `redrhex_policy_io`, canonical hashes, calibration V2, NumPy/Torch parity, and golden fixtures.

<a id="phase-f0"></a>
### F0 deterministic baseline

- [ ] Register the isolated V2 task, 36-D causal event pipeline, observation groups, and residual action contract.
- [ ] Validate zero residual, neutral ABAD, leg/tripod/sign/phase/timing mapping, and sim/ROS decoder trace parity.
- [ ] Select nonzero regularizers only through half/base/double sensitivity and pass the existing forward command sweep.

<a id="phase-f1"></a>
### F1 privileged teacher

- [x] Add Teacher A 65-D physical privilege and an explicitly isolated Teacher B target-state ablation.
- [x] Produce versioned Teacher A checkpoints in `redrhex_forward_v2_teacher`.
- [ ] Pass the existing forward protocol for three independent Teacher A seeds before distillation.

<a id="phase-f2"></a>
### F2 sensor-only distillation

- [x] Land the TCN, normalizer, auxiliary heads, custom storage, rollout mixture, split losses, metrics, and strict checkpoint transition.
- [ ] Export and validate the named two-input ONNX graph with embedded metadata and matching sidecar.
- [ ] Pass CPU and Isaac update, finite-gradient, save/resume, ONNX Runtime parity, and three-seed screening gates.

<a id="phase-f3"></a>
### F3 asymmetric PPO

- [x] Strictly copy the distilled actor/normalizer, verify equality, and create a fresh physical critic and optimizer.
- [x] Add annealed Teacher A BC and persistent velocity/dynamics losses without privileged actor input.
- [ ] Pass PPO update/save/resume and three-seed forward gates.

<a id="phase-panel"></a>
### Training Panel route

- [ ] Review and merge the recovered explicit F1 Teacher, F2 Distillation, and F3 Student PPO browser routes from the stacked Panel proposal.
- [ ] Review and merge its fail-closed full F1 → F2 → F3 browser pipeline and final-F3 history routing.
- [ ] Verify that it preserves the standard Panel route and excludes mutable V1 reward/terrain overrides from Sensor V2 launches.
- [ ] Complete the production-length run and multi-seed quality gates; Panel completion alone is not promotion evidence.

<a id="phase-f4"></a>
### F4 calibration and replay

- [ ] Import raw ABAD encoder and `cmd_vel` channels and bind a `SensorCalibrationProfileV2` to each artifact.
- [ ] Model only evidence-backed sensor/actuator ranges and log every sampled range and provenance.
- [ ] Run raw-event observation/ONNX replay on held-out recorded traces with no missing contract, NaN, invalid action, or unexplained saturation.

<a id="phase-deployment"></a>
### V2 ROS and bridge path

- [ ] Publish twelve calibrated measured joints with per-channel time, validity, freshness, and causal velocity.
- [ ] Add contract-selected V2 YAML, builder, named-I/O runner, preflight, 60-frame warm-up, and protective dropout reset.
- [ ] Pass synthetic ROS and recorded offline inference parity at `rtol=atol=1e-4`; automated validation must not enable motors.

<a id="phase-f5"></a>
### F5 evaluation and promotion

- [ ] Compare Teacher A, legacy, V2 distilled, V2 PPO, and required auxiliary/PPO ablations under identical seeds, commands, and domains.
- [ ] Publish raw per-seed data, mean/std, teacher gaps, provenance, parity, and blocked contact status.
- [ ] Promote only if all three PPO seeds pass, match Teacher A's accepted commands, and pass leak, parity, replay, and provenance gates.

<a id="verification"></a>
## Verification

Run dependency-light unit suites first, then documentation validation and diff checks. The executable F1, F2, F3, and full sequential pipeline have each passed a one-environment, one-update Isaac smoke on 2026-08-13. That proves launch, finite update, strict checkpoint handoff, and final-artifact routing only. Production-length training, three-seed command sweeps, recorded-sensor replay, and ROS offline inference still require their named evidence. Hardware preflight and physical tests remain manual, disabled by default, and blocked until the reviewed IMU mode and all twelve encoder calibrations exist.

<a id="completion-summary"></a>
## Completion summary

Implementation is active. V1 preservation and configuration-only rollback are mandatory throughout. Completion requires F5 evidence and maintained bilingual component documentation; code presence alone cannot close the plan.
