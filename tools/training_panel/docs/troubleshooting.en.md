---
id: training-panel-troubleshooting
title: Troubleshoot the Training Panel
lang: en
audience: operator
type: troubleshooting
status: active
owner: panel
last_reviewed: 2026-08-14
---

<a id="startup"></a>
## Panel does not start

Confirm Python can import `tools.training_panel`, port 8080 is free, and `REDRHEX_ROOT` resolves to this repository. Start on `127.0.0.1` and inspect the terminal traceback before changing host or port.

<a id="cuda"></a>
## CUDA preflight fails

The panel compares loaded NVIDIA kernel and userspace driver versions before GPU work. Reboot after a driver update; if versions still differ, repair the NVIDIA installation before launching Isaac. Reducing environments does not fix a driver mismatch.

<a id="queue"></a>
## Run remains queued

Inspect Process Console and the GPU lock. Local training, playback, video, and export share the lock. For remote work, training, video, ONNX, and export-and-validate share it; Stop remains prioritized. Drive export, existing-ONNX validation, and MuJoCo-only work should not wait for the Isaac lock. Cancel a queued job only as its requester or an admin. Resubmit only after checking History; the same `client_request_id` is rejected as a duplicate.

<a id="history"></a>
## History or folders disagree

Pause remote acceptance. Confirm Mother history, worker heartbeat/protocol, selected machine, capability row, schema, and tombstones. Do not edit Mother and Child metadata concurrently. Machine-scoped queries and explicit clearing prevent stale names, notes, or folders returning; 3.7 metadata writes must use the constrained RPC.

<a id="remote"></a>
## Worker is offline or disabled

Run `source ~/.redrhex_remote.env` and `python -m tools.training_panel.remote_worker --once`. Verify required variables, token scope, system time, machine ID, heartbeat freshness, and `accept_jobs`. Keep the environment file mode at `600`.

<a id="compatibility"></a>
## Child is read-only after sign-in

Read-only fallback is expected when either the selected machine heartbeat or its capability row is older than `3.7.0-remote-parity`, or when the additive schema is missing. Pause acceptance. Apply `supabase/migrations/20260814_370_remote_parity.sql`, update Mother, restart the worker, and verify that both protocol fields match before refreshing Child. Do not bypass the banner by inserting jobs manually.

If the migration is already applied, confirm the worker token can upsert its own `machine_capabilities` row, the browser selected the intended machine, realtime is subscribed, and system clocks are correct. Keep acceptance paused until the mismatch is explained.

<a id="artifacts"></a>
## Video, export, or readiness fails

In Mother, open the selected process output and verify the run has a real `model_*.pt`. In Child, reselect the run and checkpoint iteration; browser-supplied paths are intentionally rejected. Stop other Isaac jobs for video/ONNX/export-and-validate. For readiness, inspect per-stage results and the reported panel Python, `onnx`, `onnxruntime`, `torch`, and optional MuJoCo dependencies. Remote scenarios must be selected from the published capability list. A missing optional stage produces review; a required failure produces blocked.
