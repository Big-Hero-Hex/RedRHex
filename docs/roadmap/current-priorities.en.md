---
id: project-roadmap
title: Current Project Priorities
lang: en
audience: shared
type: roadmap
status: active
owner: project
last_reviewed: 2026-08-15
---

<a id="integration"></a>
## Integration and release discipline

- Reconstruct the Autopilot and Student V2 recovery snapshots as separate, reviewable changes from current `origin/main`; do not merge a recovery commit directly or discard the mainline desktop-remote, UI, and code-CI fixes that landed after their bases.
- Return the repository root to an up-to-date, clean `main` and place each recovery reconstruction in its own `.worktrees/` checkout before integration work continues.
- Keep `main` as the only shipped baseline. Treat Training Panel 3.8 Autopilot and the Student V2 follow-up as branch-local previews until their reconstructed pull requests pass documentation, service, browser, and target-environment gates.
- Split the 27,000-line Autopilot snapshot into reviewable contracts, controller, connector, UI, and documentation concerns where dependencies allow. Preserve exact cross-layer identities and run the complete combined suite before each integration step.
- Extend lightweight code CI to cover the Autopilot MCP adapter, Reward Agent, and ROS contract suites. Add a browser job or require recorded local Playwright evidence whenever Mother or Child UI behavior changes.
- Remove the tracked `.vscode/browse.vc.db-shm` and `.vscode/browse.vc.db-wal` generated database files in a focused hygiene change, then keep them ignored.

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
- Run the remaining Windows PowerShell 5.1+ and supported-macOS host smokes for installation, interactive SSH, fixed forwards, browser capability state, timeout behavior, tunnel shutdown, and existing-tunnel reuse. Source-level launcher tests do not close these host gates.
- Run the Child 3.7 Supabase staging smoke with real role/RLS rejection, old/new worker compatibility, queue and cancellation, media and Drive flows, deployment jobs, activity attribution, and admin deletion before accepting production remote jobs.

<a id="student-v2"></a>
## Student Distillation V2

- Reconstruct the preserved Student V2 follow-up from current `main`, retain its simulator stop-point evidence, and review it independently from the Panel physics proposal.
- Freeze V1 compatibility and the shared `redrhex_policy_io` contracts with canonical hashes, golden fixtures, zero-residual/action mapping, causal observation, and simulator-to-ROS trace parity.
- Close the F1/F2/F3 evidence gates: three Teacher A seeds, named two-input ONNX export and runtime parity, finite update/save/resume checks, strict actor/normalizer transfer, and three-seed forward screening.
- Bind every promoted artifact to measured sensor calibration and held-out raw-event replay; finish the V2 ROS builder, warm-up, validity/dropout behavior, and offline inference parity without enabling motors.
- Promote only after the same command protocol compares Teacher A, legacy, distilled, PPO, and required ablations across three passing PPO seeds with complete provenance and teacher-gap evidence.

<a id="reward-agent"></a>
## Reward Agent

- Retain bounded candidate planning, dry-run inspection, explicit legacy launch, trial persistence, and metric-based reports as a manual workflow outside armed Autopilot campaigns.
- Keep legacy session import non-armable and preserve its source JSON; do not reinterpret legacy scores as deterministic campaign evidence.

<a id="autopilot-rollout"></a>
## Autopilot rollout

- Keep `REDRHEX_AUTOPILOT_ENABLED` off by default until fake-advisor loops, restart recovery, idempotency, and single-GPU host serialization pass on the target training host.
- Provision ChatGPT Scheduled and OpenAI Secure MCP Tunnel externally, with runtime credentials outside the repository; verify shadow proposals and tunnel-loss waiting before any unattended launch.
- Progress from a four-trial ForwardFast pilot through restart recovery to an opt-in 24-trial qualification campaign. Enable Direct stages 2–5 only after those gates pass.
- Keep Sensor V2, remote-child campaign control, automatic source application, policy export/deployment, hardware promotion, and cross-campaign learning out of V1. A later compatibility design is required before changing the `3.7.0-remote-parity` protocol.

<a id="documentation"></a>
## Documentation

- Connect the PM agent read-only to GitHub `main` and project repository changes into the PM Control Center through a deduplicated Update Log intake. Keep Drive copies of maintained Markdown non-authoritative and replace recurring snapshot links with canonical GitHub links.
- Review operator/reference/roadmap documents every 90 days and developer architecture every 180 days.
- Publish experiment summaries only when evidence changes a baseline, recommendation, decision, or result.
- Keep the branch-preservation manifest and local recovery bundles exact until reconstructed work is reviewed into `main` or deliberately archived; remove redundant branches and worktrees only after reachability and clean-state checks pass.
- Resolve active designs and plans into durable architecture, release, audit, or roadmap records after their real host, staging, simulation, or hardware gates close.
