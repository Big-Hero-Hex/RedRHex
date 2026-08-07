---
id: adr-five-stage-curriculum
title: ADR 0001 Five-Stage Locomotion Curriculum
lang: en
audience: developer
type: decision
status: accepted
owner: training
last_reviewed: 2026-08-07
---

<a id="context"></a>
## Context

One mixed command distribution made forward, lateral, diagonal, and yaw failures difficult to isolate. Later skills could also damage an already useful forward gait, while reward totals hid whether a policy actually followed the requested direction.

<a id="decision"></a>
## Decision

The full RedRHex task supports a five-stage curriculum: forward, lateral, diagonal, yaw, then mixed integration. Each stage owns a command distribution, behavior scaling, warmup and safety thresholds, and skill-specific reward multipliers. The supported pipeline hands off by full checkpoint resume by default.

ForwardFast remains a separate forward-only task for bounded iteration. It is not stage 5 and must not be presented as a complete locomotion policy.

<a id="consequences"></a>
## Consequences

Training can diagnose one skill at a time and apply health gates between stages. Run names and checkpoint paths carry stage meaning, so playback may infer `env.stage`. Operators must preserve the run tag during resume and evaluate each skill separately. The additional state and configuration increase contract surface and require tests when stage behavior changes.

<a id="alternatives"></a>
## Alternatives considered

A single mixed task was retained as the final integration stage but rejected as the only learning path. Four stages were rejected because diagonal locomotion needs its own transition before mixed integration. Policy-only handoff remains optional but is not the default because it loses optimizer and iteration continuity.

<a id="evidence"></a>
## Evidence and limits

The staged pipeline, stage-aware environment, playback inference, and health gates are implemented. Smoke evidence proves execution paths, not final tracking performance or superiority to another controller. Long-run comparisons remain roadmap work.
