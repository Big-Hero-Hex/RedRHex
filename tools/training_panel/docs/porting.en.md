---
id: training-panel-porting
title: Extend or Port the Training Panel
lang: en
audience: developer
type: how-to
status: active
owner: panel
last_reviewed: 2026-08-07
---

<a id="preserve"></a>
## Preserve boundaries

Keep mother authoritative for local process and file operations; keep child static and team-scoped; keep the worker as the execution bridge. Reuse `TrainingParams`, `ProcessRegistry`, `HistoryStore`, role/job definitions, and artifact discovery rather than inventing a second launcher or store.

<a id="add-local"></a>
## Add a local capability

Define the data/command contract first, add backend tests, expose one API, then add the local UI. Any Isaac action must participate in the GPU lock, queue, settle window, process console, stop handling, activity, and history reconciliation. Validate paths remain within repository-owned artifact roots before allowing open, compact, or delete operations.

<a id="add-remote"></a>
## Add a remote capability

Add the job type, role permission, Supabase schema/policy if needed, worker execution, sync/artifact representation, and child UI together. Default to denied or paused. Never put machine credentials in static assets. Make requests idempotent and retain requester attribution.

<a id="version"></a>
## Version and release

Update all version surfaces belonging to the released contract together. The local Mother/UI and remote Child/worker protocol are independently deployed surfaces; do not advance the remote version for a local-only change. Add a canonical bilingual release document, do not create version gaps after the fact, and reapply and test the Supabase schema when its contract changes.

<a id="verify"></a>
## Verify

Run the relevant `tools/training_panel/tests`, remote web Node tests, and local UI tests. Check the mother workflow, child role boundary, queue/GPU lock, history convergence, artifact preservation, remote sync, and unchanged Pages root. Update operator documentation for any visible workflow change.
