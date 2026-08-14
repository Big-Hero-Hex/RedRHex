---
id: child-panel-remote-parity-plan
title: Child Panel 3.7 Remote Parity Implementation Plan
lang: en
audience: developer
type: plan
status: active
owner: panel
last_reviewed: 2026-08-14
---

<a id="objective"></a>
## Objective

Deliver the [approved remote-parity design](../../designs/active/2026-08-14-child-panel-remote-parity.en.md) as one `3.7.0-remote-parity` release while preserving Mother 3.6.4 Drive export as a separately recorded prerequisite baseline and preserving every Mother-only boundary.

<a id="implementation"></a>
## Implementation checklist

- [x] Add the versioned protocol, additive schema/security migration, authoritative roles, request idempotency, constrained metadata and cancellation RPCs, Physics presets, capability data, and bounded run projections.
- [x] Add worker-side checkpoint resolution, GPU classification, Drive export, Deploy validation, MuJoCo smoke/recording, scalar downsampling, run evidence, and redacted idempotent local activity projection.
- [x] Upgrade the buildless Child shell, routing, drafts, responsive navigation, Train routes, Physics, all-runs History, safe actions, Deploy, Detection, Activity, compatibility fallback, and role gates.
- [x] Add Node, Python, schema, and mocked Child Playwright coverage at 390 px, 768 px, and desktop widths.
- [x] Update bilingual maintained architecture, operator, remote-operation, troubleshooting, compatibility, release, design, plan, indexes, and routers.
- [ ] Run the staging Supabase smoke with deployed credentials and record the outcome before accepting production remote jobs.

<a id="staging"></a>
## Staging checklist

Pause acceptance. Apply `tools/training_panel/supabase/migrations/20260814_370_remote_parity.sql`, update Mother, restart the worker, and confirm the `3.7.0-remote-parity` heartbeat and capability row. Verify viewer/operator/admin RLS behavior, actor-role spoof rejection, old-worker read-only fallback, queue/stop, video/ONNX/Drive, Deploy/MuJoCo, activity attribution, queued cancellation, and single/bulk admin deletion. Publish Child assets only after those checks pass, then re-enable acceptance.

<a id="verification"></a>
## Verification

The repository gate is Node remote tests, Training Panel Python tests, the dedicated Child Playwright suite, the complete Mother Playwright suite, documentation validation and unit tests, and `git diff --check`. Staging is intentionally still open because this workspace has no Supabase deployment credentials; code completion does not claim that external evidence.

<a id="completion-summary"></a>
## Completion summary

Implementation and local verification are complete when all local gates pass. The plan remains active until the staging checklist is executed. After staging evidence is recorded, migrate any remaining durable detail into maintained docs and the release record, remove this temporary plan, and update the plan index according to lifecycle governance.
