---
id: adr-0003-ai-advisor-deterministic-authority
title: "ADR-0003: AI Advisor and Deterministic Authority Boundary"
lang: en
audience: developer
type: decision
status: accepted
owner: project
last_reviewed: 2026-08-14
---

<a id="context"></a>
## Context

Iterative reward experiments benefit from an AI that can form hypotheses from recent evidence, but robot training is expensive, stateful, and safety-sensitive. Natural-language judgment is not a reproducible acceptance test, and exposing the administrative panel, shell, source tree, deployment, or hardware to an unattended connector would turn advisory uncertainty into uncontrolled actuation.

<a id="decision"></a>
## Decision

ChatGPT is an advisor only. It may read compact structured campaign evidence and submit exactly one allowlisted reward-weight proposal, request a pause, request stop-after-current, or store a review-only patch proposal. It cannot arm or resume a campaign, widen bounds or budgets, alter immutable goal/physics/safety fields, choose arbitrary launch arguments, declare success, edit source, export/deploy a policy, operate hardware, or stop unrelated work.

Deterministic panel code is authoritative for schema and identity validation, idempotency, revision ordering, GPU serialization, trial allocation, hard safety gates, ranking, confirmation, budgets, state transitions, and the `simulation_goal_met` declaration. Missing, non-finite, mismatched, partial-load, fallback-selected, or corrupt evidence fails closed. Training reward and model commentary are diagnostics, not success evidence.

The connector is a narrow loopback MCP service with separate read/write tools. ChatGPT Scheduled may revisit the same chat, but absence or failure only leaves a campaign waiting after active local work finishes; no panel-side model fallback is permitted.

<a id="alternatives"></a>
## Alternatives

- Let an AI directly operate the full panel or shell: rejected because tool scope would include unrelated processes, files, deployment, and unsafe parameter changes.
- Let an AI rank evidence and declare success: rejected because results would not be deterministic, reproducible, or independently auditable.
- Run a hidden panel-side model when ChatGPT is absent: rejected because it introduces a second authority, secret storage, and behavior that the user cannot observe in the scheduled chat.
- Keep all iteration manual: retained as a fallback, but rejected as the only workflow because it leaves safe pre-approved experiment capacity idle.

<a id="consequences"></a>
## Consequences

Campaigns can advance unattended only inside a human-armed goal, reward catalog, and budget. The panel must maintain durable revisioned state and sufficient structured evidence for an advisor to act without raw logs or secrets. ChatGPT quality can affect which valid hypothesis is tested next, but cannot weaken constraints or convert a failing experiment into success. Hardware readiness and deployment remain separate human-governed processes.

<a id="supersession"></a>
## Supersession

Any future design that grants a model broader mutation or success authority must supersede this ADR explicitly and provide a new authentication, safety, evaluation, and human-approval argument.
