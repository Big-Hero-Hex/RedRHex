---
id: project-roadmap
title: Current Project Priorities
lang: en
audience: shared
type: roadmap
status: active
owner: project
last_reviewed: 2026-08-13
---

<a id="validation"></a>
## Evidence and locomotion validation

- Close the correctness gates for reward-preset resolution, command bias, exploration-scale diagnostics, evaluation method identity, energy provenance, and training/deployment observation parity.
- Establish hardware ground truth for IMU frames, base velocity, contact behavior, mass/CoM/inertia, joint stops, friction/backlash, passive springs, actuation, and electrical energy; bind it to held-out calibration evidence.
- Freeze the task, command envelope, metrics, resolved configuration, code/dependency revisions, checkpoints, hardware revision, and an immutable held-out suite before baseline comparison.
- Use one-seed screening only to reject bad candidates. Use at least three independent seeds for exploration and preferably five for confirmatory results; keep per-episode rows and report intervals rather than treating environment-time samples as independent.
- Validate energy changes with matched commanded and achieved speed, tracking, success, falls, recovery, temperature, peak current, and measured electrical cost of transport. Compare passive-spring conditions with randomized paired hardware trials where mechanically possible.
- Release a result only with its protocol, configurations, calibration evidence, checkpoints, per-episode data, failures, and representative video. Complete a dedicated literature and prior-art review before asserting novelty.
- Compare RL and MPC only after common tasks, command envelopes, metrics, seeds, and hardware conditions are defined.

The evidence gates and their interpretation are recorded in the [2026-08-13 research-readiness audit](../research/2026-08-13-research-readiness-audit.en.md).

<a id="core"></a>
## Core maintainability

- Decide whether to approve and execute the proposed core-first, simulation-first soft reboot.
- If approved, validate legacy gravity, frames, mass, contacts, timing, and determinism before capturing any golden baseline.
- Separate stable contracts and pure Torch behavior from the Isaac adapter without changing external tasks or artifacts during extraction.

<a id="operations"></a>
## Operations and deployment

- Resolve or explicitly accept the panel authentication boundary before wider LAN exposure.
- Add a deployed base-linear-velocity estimator or an evidence-backed training alternative.
- Complete hardware bring-up evidence and keep ROS contract parity at 60 Hz.
- Implement and verify the approved Windows remote launcher from its active design and plan.

<a id="reward-agent"></a>
## Reward Agent

- Keep bounded candidate planning, dry-run inspection, explicit launch, trial persistence, and metric-based reports.
- Add proposal UI and deeper Training Panel integration only after a separate design; do not silently edit reward source or let an LLM declare success without metrics.

<a id="documentation"></a>
## Documentation

- Review operator/reference/roadmap documents every 90 days and developer architecture every 180 days.
- Publish experiment summaries only when evidence changes a baseline, recommendation, decision, or result.
- Begin Git history and branch reorganization only after the documentation-system v1 checkpoint tag.
