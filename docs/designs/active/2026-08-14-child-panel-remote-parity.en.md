---
id: child-panel-remote-parity
title: Child Panel 3.7 Remote Parity
lang: en
audience: developer
type: design
status: approved
owner: panel
last_reviewed: 2026-08-14
---

<a id="problem"></a>
## Problem

RedRHex To Go has authentication, roles, remote queueing, shared Reward and Terrain presets, folders, and History, but its navigation, feedback, training routes, Physics, deployment evidence, detection visibility, and team activity trail lag behind Mother 3.6.4. Operators need the same safe mental model on a phone without exposing Mother's administrative surface.

<a id="experience"></a>
## Approved experience

Keep To Go static, buildless, team-scoped, and phone-first. Desktop uses a persistent Mother-style sidebar. Phone and tablet use Dashboard, Train, History, and More; More owns Rewards, Terrain, Physics, Deploy, Detection, Activity, and Connection. URL state preserves the view, folder, run, search, status, and sort. Shared design tokens, focus states, notices, toasts, freshness, and responsive cards must avoid horizontal overflow at 390 px, 768 px, and desktop widths.

Train exposes Standard, F1, F2, F3, and full-pipeline routes, always uses Native springs, omits irrelevant fields, and represents a checkpoint only as run ID plus iteration. Reward, Terrain, and sparse Physics presets are team-shared and protect built-ins. History defaults to all runs and owns folders, filters, keyboard selection, drag/drop, bulk move and admin-only bulk deletion, comparison, progress, bounded curves, provenance, checkpoint evolution, and remote-safe run actions.

<a id="boundary"></a>
## Security boundary

Viewer is inspection-only. Operator may edit metadata, presets, and folders and run non-destructive jobs. Admin also deletes. The worker resolves the authoritative profile role; the browser's role field is never trusted. Terminal access, raw logs, worker administration, arbitrary host paths, GUI viewers, convergence edits, and physical robot actuation remain Mother-only.

Remote Deploy accepts repository-owned models and enumerated MuJoCo scenarios only. It may validate existing ONNX, export and validate, run a MuJoCo smoke, record a MuJoCo MP4, and optionally include the ROS mock. It cannot open a viewer or actuate hardware. Detection settings are read-only.

<a id="protocol"></a>
## Protocol and compatibility

The contract version is `3.7.0-remote-parity`. An additive Supabase migration adds machine capabilities, Physics presets, request idempotency, bounded run projections, activity source identity, constrained metadata and cancellation RPCs, and authoritative job attribution. Existing 3.4.10 rows remain valid. Older schema or worker state keeps sign-in and inspection available but disables every mutation with exact migration and restart guidance.

<a id="rollout"></a>
## Rollout and rollback

Pause acceptance, apply the additive migration, update Mother, restart the worker, confirm its capability row and heartbeat, publish Child assets, then resume acceptance. Any different ordering leaves Child read-only. Rollback pauses acceptance and restores previous Child and worker assets; additive database objects remain because older clients ignore them.

<a id="acceptance"></a>
## Acceptance boundary

Local completion requires Node protocol tests, Python worker/schema tests, dedicated Child Playwright coverage, the complete Mother UI and Training Panel suites, documentation validation, and diff checks. Staging completion additionally requires Supabase login, RLS spoof rejection, old-worker fallback, new-worker activation, queue/stop/media/Drive/Deploy flows, audit attribution, and admin deletion. Hardware actuation is explicitly outside this design.
