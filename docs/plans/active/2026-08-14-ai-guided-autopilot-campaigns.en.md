---
id: plan-ai-guided-autopilot-campaigns
title: AI-Guided Autopilot Campaign Implementation
lang: en
audience: developer
type: plan
status: active
owner: project
last_reviewed: 2026-08-14
---

<a id="objective"></a>
## Objective

Deliver the approved off-by-default campaign controller, deterministic evaluator integration, bounded ChatGPT connector, panel workspace, and recovery evidence needed to run standard-PPO reward experiments safely while a human is absent.

<a id="context"></a>
## Context

The approved input is the [Autopilot campaign design](../../designs/active/2026-08-14-ai-guided-autopilot-campaigns.en.md). The panel remains the only authority for execution and success. V1 covers ForwardFast stage 1 and Direct stages 1–5, keeps the existing remote-child protocol unchanged, and excludes Sensor V2, deployment, hardware, automatic source edits, and automatic budget expansion.

<a id="phased-checklist"></a>
## Phased checklist

<a id="phase-foundation"></a>
### Safety and reproducibility foundation

- [x] Discover nested `v2_reward_scales` in saved configuration, history diffs, and tweak reconstruction.
- [x] Reject boolean and non-finite reward values and provide exact resolved-key validation.
- [x] Add typed stage, initialization, strict checkpoint, profile path, profile hash, and campaign ownership fields to standard training.
- [x] Replace panel-launched global reward and terrain mutation with immutable per-run SHA-256 snapshots.
- [x] Bind policy-only initialization to a recorded run, exact checkpoint iteration, and SHA-256 without scanning a run directory for a latest checkpoint; reject non-default terrain overrides in V1.
- [ ] Complete cross-process host lease, heartbeat, and startup reconciliation coverage for every campaign lifecycle state. The lease and active training/evaluation recovery paths are covered; the all-state crash matrix remains a rollout gate.

<a id="phase-controller"></a>
### Durable deterministic controller

- [x] Ship SQLite WAL campaign contracts, event log, idempotency results, one-armed-campaign invariant, budgets, trials, reports, and artifact records.
- [x] Serialize control/candidate training and exact-checkpoint command evaluation through the panel-owned process service.
- [x] Parse command and episode CSVs fail closed, apply hard gates before ranking, and reserve confirmation capacity.
- [x] Restrict advisor proposals to the finite 80/90/100/110/120 campaign-start lattice and expose complete/remaining values, attempted moves, leader deltas, evidence IDs, and remaining budgets in decision context.
- [x] Store the evaluated report as a fourth immutable artifact, revalidate its bindings during delayed recovery and confirmation, and durably stop campaign-owned work before terminalizing a controller failure.
- [x] Prove control, screen, leader, early-confirmation, three-seed success, non-improvement, budget, pause, stop, and patch-handoff transitions with dependency-light fake-process tests.

<a id="phase-advisor-and-ui"></a>
### Advisor boundary and panel workspace

- [x] Expose versioned `/api/autopilot` read and idempotent/revision-checked write routes.
- [x] Ship a loopback MCP adapter with five read tools, five bounded decision/lifecycle write tools, a no-op active-campaign heartbeat, safety annotations, stdio development transport, and Streamable HTTP tunnel transport.
- [x] Ship the advisor skill and durable 15-minute same-chat task prompt; record heartbeat, declared model, prompt/skill version, proposal, validation, and process identifiers. External task creation remains a rollout step.
- [x] Ship the Autopilot workspace for goal preview, explicit arming, lifecycle, budget, heartbeat, trial/evaluation comparison, campaign-owned stop controls, and patch export.

<a id="phase-rollout"></a>
### Staged rollout

- [ ] Run a deterministic fake-advisor end-to-end loop and crash/restart matrix.
- [ ] Run ChatGPT shadow proposals without launches.
- [ ] Run a four-trial ForwardFast pilot and restart/recovery pilot.
- [ ] Run a full 24-trial ForwardFast qualification campaign.
- [ ] Enable Direct stages 2–5 only after the earlier gates pass.

<a id="verification"></a>
## Verification

Run unit/property tests for contracts, validation, state transitions, ranking, hard gates, budgets, hashes, and corrupt input; fake-process lifecycle/restart/idempotency/concurrency tests; MCP schema/security tests; browser accessibility/mobile/recovery tests; the existing Reward Agent, Training Panel, and sim2real suites; and `python -m tools.documentation validate --all`. Run a one-iteration Isaac smoke before a short shadow campaign. Keep the expensive three-seed campaign explicit and opt-in.

<a id="completion-summary"></a>
## Completion summary

The off-by-default repository implementation, deterministic fake-process/restart coverage, finite reward lattice, immutable evaluation-report evidence, durable controller-failure stop recovery, Streamable HTTP boundary, and browser workspace are present. The plan remains active because the external ChatGPT Scheduled task and Secure MCP Tunnel are not provisioned, and shadow, real Isaac, four-trial, restart-host, 24-trial, and Direct rollout gates remain incomplete. No result from this plan may be called hardware-ready.
