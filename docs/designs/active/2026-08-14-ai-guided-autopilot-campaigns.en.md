---
id: design-ai-guided-autopilot-campaigns
title: AI-Guided Autopilot Campaigns
lang: en
audience: developer
type: design
status: approved
owner: project
last_reviewed: 2026-08-14
---

<a id="problem"></a>
## Problem

The Training Panel can launch and observe PPO runs, the Reward Agent can persist bounded candidates, and the command sweep can measure locomotion. They do not yet form a restart-safe experiment loop. Human absence therefore leaves the GPU idle, while granting an AI direct control of shell commands, reward source, success decisions, or deployment would violate the project safety boundary.

<a id="goals-and-non-goals"></a>
## Goals and non-goals

- Goal: let a human arm one numeric locomotion goal, bounded reward catalog, trial limit, iteration limit, and active-GPU-hour limit.
- Goal: persist campaigns, decisions, trials, evaluations, artifacts, revisions, idempotency results, and recovery events in SQLite with WAL transactions.
- Goal: let ChatGPT read compact evidence and propose one allowlisted reward change while deterministic panel code owns validation, execution, ranking, budgets, and success.
- Goal: support standard PPO for ForwardFast stage 1 and Direct curriculum stages 1–5, with exact checkpoint, seed, task, command, physics, spring, configuration, and code identities.
- Non-goal: Sensor V2, source-tree mutation, policy export or deployment, hardware promotion, remote-child campaign control, or a panel-side model/API key.

<a id="proposal-and-interfaces"></a>
## Proposal and interfaces

The durable flow is `draft → armed → control training/evaluation → awaiting advisor → candidate training/evaluation → confirmation → simulation goal met`, with explicit paused, waiting, patch-handoff, budget, safety, stopped, and failed outcomes. Only one campaign may be armed on a host.

`GoalSpecV1` stores the numeric `{vx, vy, wz}` envelope, task and stage, evaluation profile, directions, safety gates, exact initialization checkpoint identity, seeds, iteration cap, and campaign budget. A policy-only baseline is the indivisible tuple `baseline_run_id + baseline_checkpoint_iteration + checkpoint_sha256`; the panel resolves only checkpoint identities already recorded in History and never scans a run directory to select a “latest” file. Display labels such as “walk” and “run” compile to the lower and upper halves of the existing task/stage range, but the resulting numbers are always shown before arming. `RewardCatalogEntryV1` permits only nonzero shaping weights compatible with the selected task and stage. Its sign is fixed and its absolute bounds cannot exceed 80–120% of the campaign-start value. Advisor moves are restricted to the finite campaign-start lattice at 80%, 90%, 100%, 110%, and 120%, clipped and deduplicated after any human narrowing; the current and already-attempted points are omitted. Termination, fall and health gates, physics, targets, sigmas, caps, command ranges, and terrain are immutable. V1 rejects a baseline with any non-default terrain override because deterministic evaluation does not apply that override.

`AgentDecisionV1` can propose one remaining lattice point for one catalog key, pause, or request a review-only patch handoff. Its decision context includes each key's campaign-start/current values, hard bounds, complete and remaining lattice values, baseline-to-leader deltas, attempted moves, recent decisions/evaluations, evidence IDs, and remaining trial/GPU/confirmation/poll budgets. `EvaluationReportV1` binds command and episode evidence to the exact checkpoint, seed, evaluator profile, configuration, and artifact hashes. The command-sweep horizon identity also binds the recorded evaluator `num_envs`, `sweep_steps=600`, the V1 control timestep `step_dt = 1/60` second, and `duration_s = sweep_steps × step_dt`; every command/environment must contribute the complete 600-sample horizon, and each command success duration must equal its episode-reconciled success ratio multiplied by `duration_s`. In addition to immutable command, episode, and summary artifacts, the evaluated report itself is stored as a fourth SHA-256-addressed artifact and is reopened before delayed recovery or final confirmation. Invalid, missing, truncated, tampered, non-finite, partial-load, fallback-selected, artifact-divergent, or identity-mismatched evidence fails closed. `CampaignSnapshotV1` is the compact read model.

The panel exposes schema-versioned `/api/autopilot` endpoints for capabilities, campaign drafts and lifecycle, decision context, bounded decisions, events, comparisons, artifacts, and patch export. Every write requires an idempotency key and expected revision. A narrow loopback MCP server exposes five read tools, five bounded decision/lifecycle write tools, and one no-op active-campaign heartbeat; it never proxies the administrative panel API. Streamable HTTP is intended for OpenAI Secure MCP Tunnel, while local stdio remains available for installed development tooling.

Each panel launch receives immutable SHA-256-bound reward, terrain, physics, command, and checkpoint inputs. Installed dependency files are content-hashed; editable dependencies bind their scoped Git tree, dirty diff, and untracked content without exposing their absolute origin. Campaign completion records only the exact `model_{iteration_cap-1}.pt` path and digest, including monitorless restart reconciliation, and never derives an output from History's display-only latest-checkpoint scan. Training uses fresh initialization or a strict policy-only fork with optimizer and iteration reset. Candidate policies never chain: the control and every candidate start from the same frozen initialization. The evaluator is a serialized first-class GPU job and emits command plus environment-episode evidence. Training reward and TensorBoard data remain diagnostics only.

The 24-trial funnel reserves one seed-42 control, up to nineteen seed-42 screens, and four seed-43/44 control/winner confirmations. Confirmation starts as soon as a screen passes. Simulation success requires all three candidate reports to be valid, at least two to pass every goal gate, median tracking to improve over paired controls, and energy to remain under the absolute cap.

<a id="failure-modes"></a>
## Failure modes

- Panel or host restart: reconcile durable state and exact process/artifact identities; do not relaunch a completed or live idempotent action.
- Controller validation or internal failure while campaign work is active: persist the exact campaign-owned stop intent before signaling the process, retain the active identity if stopping cannot yet be confirmed, and finish GPU accounting plus terminalization on a later tick or restart. A controller exception must not silently leave GPU work running or terminate the panel controller thread.
- ChatGPT, schedule, or tunnel unavailable: finish active local work, persist it, then enter `waiting_for_chatgpt`; no hidden model fallback exists.
- Invalid evidence, identity mismatch, fall/health failure, or non-finite value: reject the trial and block or pause according to the deterministic state table.
- Infrastructure launch failure: retry at most once with identical inputs; configuration, divergence, safety, and evidence failures do not retry automatically.
- Four valid non-improving screens, no eligible move, or insufficient confirmation budget: enter patch handoff without widening bounds.
- Patch proposal: store a hash-addressed unified-diff artifact only; never apply it. Any accepted source edit begins a new linked campaign because code identity changed.

<a id="acceptance"></a>
## Acceptance

- [ ] Two advisor-to-training-to-evaluation iterations complete without a watching human.
- [ ] Restart recovery produces no duplicate GPU work or duplicate campaign event.
- [ ] Every budget and safety guardrail terminates or pauses automatically and deterministically.
- [ ] Success depends only on valid command-sweep evidence, never on ChatGPT opinion or training reward.
- [ ] The connector cannot mutate source, deployment, hardware, unrelated processes, budgets, or arming state.
- [ ] A one-iteration Isaac smoke and short ForwardFast shadow campaign pass; the three-seed qualification campaign remains opt-in.

<a id="resolution"></a>
## Resolution

Approved for an off-by-default staged rollout. The active implementation and pilot gates are tracked in the [Autopilot campaign implementation plan](../../plans/active/2026-08-14-ai-guided-autopilot-campaigns.en.md), and the authority boundary is permanent in [ADR-0003](../../decisions/adr-0003-ai-advisor-deterministic-authority.en.md).
