---
id: training-panel-remote-operation
title: Operate RedRHex To Go
lang: en
audience: operator
type: how-to
status: active
owner: panel
last_reviewed: 2026-08-14
---

<a id="architecture"></a>
## Remote boundary

GitHub Pages hosts the static Child UI. Supabase stores team identity, roles, jobs, runs, artifacts, events, capabilities, shared presets, and machine heartbeat. A worker on the training PC polls and executes accepted jobs. Cloudflare Tunnel may expose TensorBoard; requester-scoped Discord notifications are dispatched by the Supabase function.

Child is team-scoped and remote-safe. It does not expose a terminal, raw process logs, worker administration, arbitrary host paths, GUI viewers, convergence edits, or physical robot deployment. Use Mother over a trusted SSH tunnel when those local administrative capabilities are required.

<a id="secrets"></a>
## Configure secrets

Create `~/.redrhex_remote.env` on the training PC with `REDRHEX_SUPABASE_URL`, `REDRHEX_SUPABASE_ANON_KEY`, `REDRHEX_SUPABASE_MACHINE_TOKEN`, `REDRHEX_MACHINE_ID`, `REDRHEX_REMOTE_ACCEPT_JOBS`, and optional tunnel settings. Set mode `600`. Only the public Supabase URL and anonymous/publishable key may appear in the child; machine/service-role tokens remain private.

<a id="start-worker"></a>
## Start and verify the worker

Use Control Center to start, stop, restart, choose tmux or child-process mode, enable auto-start, and accept or pause jobs. Manual verification:

```bash
source ~/.redrhex_remote.env
python -m tools.training_panel.remote_worker --once
```

Confirm the expected machine ID, `online: true`, a fresh heartbeat, and the intended `accept_jobs` state before continuous operation.

For 3.7.0, pause acceptance and apply `supabase/migrations/20260814_370_remote_parity.sql` before restarting Mother and the worker. Confirm that both the heartbeat and `machine_capabilities.protocol_version` report `3.7.0-remote-parity`, the intended machine is selected, and schema warnings are absent. Restarting Mother alone leaves an already-running worker unchanged.

If either version is older, sign-in and inspection remain available but every mutation is disabled. Follow the Child banner exactly: apply the migration, update Mother, restart the worker, then refresh. Do not submit jobs optimistically through another client while compatibility is unresolved.

<a id="roles"></a>
## Roles and actions

Viewers inspect all team-safe data. Operators can edit run metadata, presets, and folders and launch every non-destructive remote action: route-aware training, direct stop, resume/tweak, TensorBoard, video, ONNX, private Drive export, compaction, deployment validation, export-and-validate, MuJoCo smoke/MP4, queued cancellation, and notification delivery. Admins also delete individual or selected runs. Bulk deletion requires typing `DELETE`; each worker delete still requires the exact run ID and reports success or failure per run.

The database stamps job identity and authoritative profile role. The worker resolves that role again and ignores browser-provided `actor_role`. Preset/folder triggers, metadata RPCs, and worker events create authenticated audit entries.

<a id="sync"></a>
## Sync expectations

Version 3.7.0 synchronizes machine-scoped run/job/artifact state, explicit metadata clearing, folder state and tombstones, Reward/Terrain/Physics presets, bounded progress and scalar curves, Git and spring provenance, divergence, deployment/MuJoCo evidence, completed private Drive links, redacted idempotent activity, queue/lock state, capabilities, and worker health. It never synchronizes TensorBoard event files, raw logs, credentials, or unrestricted paths.

Training, video, ONNX, and export-and-validate share the Isaac GPU lock. Stop is prioritized. Drive export, existing-ONNX validation, and MuJoCo-only work do not take that lock. Check action-local status and History before retrying; request IDs prevent duplicate submission.

<a id="rollout"></a>
## Roll out or roll back

Pause acceptance, apply the additive migration, deploy/update Mother, restart the worker, verify the capability row, heartbeat, roles, and schema, publish Child assets, then resume acceptance. If the order differs, leave Child read-only. To roll back, pause acceptance and restore the previous Child and worker assets. Leave the additive tables, columns, policies, RPCs, and triggers in place because older clients ignore them.
