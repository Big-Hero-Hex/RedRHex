---
id: student-distillation-v2-plan
title: Sensor-Only Student Distillation V2 Implementation Plan
lang: en
audience: developer
type: plan
status: active
owner: training
last_reviewed: 2026-08-17
---

<a id="objective"></a>
## Objective

Deliver the approved sensor-only forward route as additive, explicitly selected CLI and ROS functionality, prove each evidence gate in order, and keep every V1 artifact and rollback path intact.

<a id="context"></a>
## Context

The [approved design](../../designs/active/2026-08-13-student-distillation-v2.en.md) fixes the 36-D frame, 60-frame history, separate command, residual procedural action decoder, TCN, loss schedules, strict artifacts, two-input ONNX graph, and ROS safety boundary. The updated [code-path audit](../../research/2026-08-13-student-distillation-v2-audit.en.md) confirms that the additive training, replay, export, and ROS routes now exist, while contact supervision and hardware promotion remain blocked.

A checked item means the implementation and its scoped proof exist. It does not imply production-length or multi-seed results, recorded-sensor replay, calibrated hardware, or a physical gate passed. The hash-bound schema-v2 structural and seed-42 Isaac F0 results are `PASS`; F1-F5 and hardware evidence remain `NOT_RUN`.

<a id="phased-checklist"></a>
## Phased checklist

<a id="phase-compatibility"></a>
### Compatibility and contracts

- [x] Publish and maintain the bilingual audit, approved design, active plan, rollback boundary, and documentation-impact declaration.
- [x] Preserve explicit V1 task/runner, observation, checkpoint, ROS configuration, launch, and preprocessing paths without V2 auto-detection.
- [x] Land the installable `redrhex_policy_io` package, canonical observation/action/calibration hashes, NumPy/Torch seams, strict history, and dependency-light fixtures.

<a id="phase-f0"></a>
### F0 deterministic baseline

- [x] Register the isolated V2 task, 36-D causal event pipeline, observation groups, and forward residual action contract.
- [x] Replace the stale schema-v1 π-reset interpretation with the supported all-`-π/4` effective reset; keep the π tripod offset in the restored 65/35 time-warped CPG reference.
- [x] Produce immutable schema-v2 evidence in which structural, simulator, and every per-command row pass. The current report SHA-256 is `2e108004c75e74e2e5df08d29ed8aac28b67f7cf8e5cc410135cd36975a70132`.

<a id="phase-f1"></a>
### F1 privileged teacher

- [x] Implement Teacher A's 65-D physical privilege and an explicitly isolated 77-D Teacher B target-state ablation.
- [x] Implement versioned Teacher A checkpoint manifests and the `redrhex_forward_v2_teacher` experiment route.
- [ ] Produce current-revision Teacher A checkpoints and pass the required forward protocol for three independent seeds before distillation promotion.

<a id="phase-f2"></a>
### F2 sensor-only distillation

- [x] Land the causal TCN, in-model normalizers, auxiliary heads, custom storage, rollout mixture, split losses, metrics, and strict teacher-to-student transition.
- [x] Implement the named fixed-shape two-input ONNX exporter, embedded metadata, hash-bound sidecar, and fail-closed Torch/ONNX Runtime parity gate.
- [ ] Produce a current F2 screening artifact and pass CPU/Isaac update, finite-gradient, save/resume, ONNX Runtime parity, and three-seed screening gates; promotion remains reserved for the exact F4 artifact.

<a id="phase-f3"></a>
### F3 asymmetric PPO

- [x] Strictly copy the distilled actor/normalizer, verify equality, and create a fresh physical critic and optimizer.
- [x] Add annealed Teacher A behavior cloning and persistent velocity/dynamics losses without privileged actor input.
- [ ] Pass current-revision PPO update/save/resume, command sweep, and three-seed forward gates.

<a id="phase-panel"></a>
### Training Panel route

- [x] Add explicit F1 Teacher, F2 Distillation, F3 Student PPO, evidence-gated full F0-F5, and labeled non-promotable `sensor_v2_ungated_debug` browser routes; reject new `sensor_v2_f1_f3` launches while preserving its historical runs as read-only, derived-noneligible recovery records.
- [x] Add strict source-checkpoint requirements, full-pipeline result/provenance validation, and final-F4 log/history routing in `tools/training_panel/training_panel/processes.py` while preserving the standard route.
- [x] Keep Sensor V2 launches explicit and prevent mutable standard V1 reward/terrain overrides from being applied implicitly.
- [ ] Complete current production-length and multi-seed quality gates; Panel process completion alone is not promotion evidence.

<a id="phase-f4"></a>
### F4 calibration, robustness, and replay

- [x] Add `tools/sim2real/sensor_dr_profile_v2.py` with exact profile SHA-256 binding, relative evidence-artifact resolution and hash verification, and separate `training_curriculum` and `held_out_evaluation` purposes.
- [x] Add `SensorRobustnessRunnerV2` / `rsl_rl_robust_ppo_v2_cfg_entry_point` with an explicit `--ppo_checkpoint` F3-to-F4 boundary, compatible contract checks, and a fresh optimizer.
- [x] Add a contract-bound four-topic rosbag importer plus synchronized observation/ONNX replay for calibrated IMU, main/ABAD encoder, and command channels; require an independently supplied hash-bound capture attestation and hardware-ready calibration for real traces, read and SHA-bind the canonical controller YAML without override, and recompute raw, action-clipped, slew-limited, and final targets through the stateful ROS decoder.
- [ ] Produce reviewed non-neutral training and held-out profiles from measured evidence, run F4 robustness training, and pass recorded replay with no missing contract, NaN, invalid action, or unexplained saturation and with element-wise total target-divergence fraction `0` and maximum delta `0`.

<a id="phase-deployment"></a>
### V2 ROS and bridge path

- [x] Add the V2 bridge overlay and measured twelve-joint validity/freshness diagnostics without changing the V1 bridge configuration.
- [x] Add the dedicated V2 node, YAML, launch, builder, named-I/O runner, preflight, source-time validation, complete-generation source-skew and 60 Hz per-channel cadence gates, 60-real-frame warm-up, and protective history/baseline reset.
- [x] Keep policy/motor startup disabled and require `hardware_gate.allow_motor_enable`, bundle calibration hardware readiness, and exact configured-to-bundle action-envelope equality for motor authorization. The simulator/bundle/PhysX ceiling is `15.0` rad/s; the checked-in YAML has an unevidenced `9.0` rad/s limit and `120.0` rad/s² slew rate. The velocity mismatch statically blocks authorization, while any runtime action clip, slew, or velocity tightening that changes a target latches authorization off and enters protective stop.
- [ ] Pass synthetic ROS inference parity at `rtol=atol=1e-4` and recorded stateful action-target parity at exact zero divergence; there is no replay override, and automated validation must not enable motors.

<a id="phase-f5"></a>
### F5 evaluation and promotion

- [x] Add `scripts/rsl_rl/train_sensor_v2_full_pipeline.py` as the fail-closed three-seed F0-F5 route, requiring immutable Isaac F0 evidence and rejecting F4/F5 profile-hash, `profile_id`, or evidence-artifact-hash overlap.
- [ ] Compare Teacher A, legacy, V2 distilled, V2 PPO, V2 robustness, and required auxiliary/PPO ablations under identical seeds, commands, and held-out domains.
- [ ] Publish raw per-seed data, mean/std, teacher gaps, provenance, parity, recorded replay, and blocked contact status.
- [ ] Promote only an exact `ppo_f4` checkpoint whose embedded ONNX metadata, sidecar metadata, embedded checkpoint manifest, and sidecar checkpoint manifest agree on that stage, and whose sensor-replay ONNX and sidecar SHA-256 values are byte-identical to the canonical `torch_onnx_parity` verified sources, after all required seeds pass, match Teacher A's accepted commands, and pass F0, leak, parity, replay, provenance, calibration, and safety gates; neither `ppo_f3` nor a distinct rehashed replay graph can substitute.

<a id="verification"></a>
## Verification

Run dependency-light contract, action-decoder, runner, exporter, replay, ROS wiring/packaging, and documentation suites first. The 2026-08-17 schema-v2 structural F0 report passes the same-phase reset, 65/35 time warp, 0.9 Hz gait, 60 Hz timing, exact `15.0` rad/s simulator/bundle/PhysX binding, and shared-decoder checks. Its seed-42, eight-environment native-spring rollout also passes all `0.22`, `0.35`, and `0.42` m/s rows with zero falls and contiguous-success ratio `1.0`, using exact command-scaled full-cycle velocity windows of 121/76/67 samples while tilt, height, and episode boundaries remain pointwise under unchanged evaluator thresholds. This F0 run did not start F1. The full promotion route still requires at least three unique seeds, exact F0/profile hashes, an independently evidenced F5 domain, exact `ppo_f4` provenance, and a canonical-YAML-bound real replay with zero action-target divergence. No F1/F2/F3/F4/F5 result, promoted ONNX bundle, recorded real replay, ROS offline parity report, hardware preflight PASS, or physical test was produced.

The checked-in ROS configuration and bridge overlay stay disabled by default. Its unevidenced YAML contains a `9.0` rad/s velocity limit and `120.0` rad/s² slew rate; the velocity limit does not match the `15.0` rad/s bundle envelope, so static preflight blocks motor authorization, and any runtime tightening that changes a target triggers latch-off and protective stop. Hardware testing remains separately authorized and blocked until the selected IMU mode, all twelve encoder calibrations, exact hashes, hardware-ready bundle, exact action-envelope and offline-replay parity, and safety preflight have reviewed evidence.

<a id="documentation-impact"></a>
## Documentation impact

This change updates the existing active bilingual plan and paired research audit in place. It adds no document path, navigation entry, redirect, or migration stub. The approved design remains unchanged because its fixed 60 Hz timestamp/history contract and fail-closed safety boundary already require the new runtime enforcement; implemented-route and evidence status are synchronized here. English and Traditional Chinese anchors, metadata, checkboxes, and semantics remain paired.

<a id="completion-summary"></a>
## Completion summary

Implementation remains active. The executable V2 route and structural-plus-Isaac F0 gate now pass, but empirical F1-F5, replay, calibration, and hardware gates remain open. V1 preservation and configuration-only rollback are mandatory throughout. Only immutable evidence for the remaining gates, including independent F5 evidence, plus maintained bilingual documentation can close this plan; F0 or code presence alone cannot.
