---
id: project-roadmap
title: Current Project Priorities
lang: en
audience: shared
type: roadmap
status: active
owner: project
last_reviewed: 2026-08-07
---

<a id="validation"></a>
## Evidence and locomotion validation

- Run fixed-protocol long training and command-sweep comparisons for the full and ForwardFast tasks.
- Validate energy changes with tracking, success, fall, cost-of-transport proxy, and power-per-motion criteria by skill.
- Establish hardware ground truth for IMU frames, base velocity, contact behavior, mass/CoM, actuation, and electrical energy before stronger sim-to-real claims.
- Compare RL and MPC only after common tasks, command envelopes, metrics, seeds, and hardware conditions are defined.

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
