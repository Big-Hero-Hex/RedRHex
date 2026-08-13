---
id: torsion-spring-rollout-plan
title: Torsion-Spring Calibration and Policy Rollout
lang: en
audience: developer
type: plan
status: active
owner: sim2real
last_reviewed: 2026-08-13
---

<a id="objective"></a>
## Objective

Turn the implemented explicit/native spring system into an evidence-backed selected backend and accepted ForwardFast and Direct policies without changing the 12-action/56-observation contract.

<a id="context"></a>
## Context

Implementation, profile binding, Panel propagation, fail-closed deployment checks, characterization, and acceptance validators exist. V11 is uncalibrated and selected no backend. Physical spring calibration and holdout are the external blocking inputs.

<a id="phased-checklist"></a>
## Phased checklist

<a id="physical"></a>
### Physical evidence

- [ ] Obtain mechanical-owner fixture approval with exactly one safe envelope.
- [ ] Record three signed loading/unloading repeats for `torsion-spring` and a distinct `torsion-spring-holdout` episode.
- [ ] Import immutable episodes and require every linear-model quality gate to pass; otherwise stop and specify a nonlinear model.

<a id="physics"></a>
### Physics selection

- [ ] Build an authenticated profile that applies the accepted representative stiffness to all six aliases and keeps damping zero.
- [ ] Run explicit/native `spring-release` characterization at 120 Hz and 240 Hz under identical provenance.
- [ ] Select a backend only when both implementations pass the deterministic gate; preserve a blocked report otherwise.

<a id="policy"></a>
### Policy acceptance

- [ ] Train ForwardFast seeds 42–44, evaluate the fixed command sweep, and require at least two passing seeds.
- [ ] Only after ForwardFast passes, train and evaluate full Direct seeds 42–44 and require at least two passing seeds.
- [ ] Keep the historical high-gain-hold comparison observational because its physics metadata differs.

<a id="integration"></a>
### Integration

- [ ] Run the full sim-to-real and Training Panel suites plus real Panel video/export checks with recorded backend reuse.
- [ ] Update canonical operator/developer documentation and publish evidence-backed release and experiment records.
- [ ] Resolve this plan and the approved design only after calibrated evidence, selected backend, accepted policies, and reviewed integration all exist.

<a id="verification"></a>
## Verification

Required evidence includes immutable source hashes, profile revalidation, four matched characterization artifacts, backend selection output, six ForwardFast command/summary files, six Direct files, deployment rejection of uncalibrated checkpoints, Panel command/history assertions, and readable recorded video.

<a id="completion-summary"></a>
## Completion summary

Pending physical calibration and holdout; no production backend or deployable torsion policy is currently selected.
