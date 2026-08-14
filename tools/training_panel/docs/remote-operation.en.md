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

GitHub Pages hosts the static child UI. Supabase stores team identity, roles, jobs, runs, artifacts, events, and machine heartbeat. A worker on the training PC polls and executes accepted jobs. Cloudflare Tunnel may expose live panel/TensorBoard services; requester-scoped Discord notifications are dispatched by the Supabase function.

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

After a 3.6.1 spring-safety update, pause job acceptance and explicitly restart the worker. Restarting Mother alone leaves an already-running worker unchanged. Confirm the refreshed worker heartbeat before accepting jobs; the 3.4.10 Child does not send `spring_backend`, so only the updated worker supplies the Native training default and Explicit quarantine.

<a id="roles"></a>
## Roles and actions

Viewers inspect data. Operators can launch training and non-destructive operational actions, including stop, video, export, TensorBoard, compaction, and missed-notification delivery. Admins also delete. The mother panel retains terminal, local file, worker administration, and full debugging capabilities.

<a id="sync"></a>
## Sync expectations

Version 3.4.10 synchronizes machine-scoped run/job/artifact state, explicit metadata clearing, folder state and tombstones, active presets, queue/lock status, and worker health. Reapply the checked-in Supabase schema when release notes require it. If mother and child disagree, pause jobs and diagnose sync before editing both sides.
