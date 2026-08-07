---
id: training-panel-troubleshooting
title: Troubleshoot the Training Panel
lang: en
audience: operator
type: troubleshooting
status: active
owner: panel
last_reviewed: 2026-08-07
---

<a id="startup"></a>
## Panel does not start

Confirm Python can import `tools.training_panel`, port 8080 is free, and `REDRHEX_ROOT` resolves to this repository. Start on `127.0.0.1` and inspect the terminal traceback before changing host or port.

<a id="cuda"></a>
## CUDA preflight fails

The panel compares loaded NVIDIA kernel and userspace driver versions before GPU work. Reboot after a driver update; if versions still differ, repair the NVIDIA installation before launching Isaac. Reducing environments does not fix a driver mismatch.

<a id="queue"></a>
## Run remains queued

Inspect Process Console and the GPU lock. Training, playback, video, and export share the lock. Stop or wait for the active Isaac job and its settle window. Cancel and resubmit only after confirming no orphan tmux or Isaac process remains.

<a id="history"></a>
## History or folders disagree

Pause remote acceptance. Confirm mother history, worker heartbeat/version, machine ID, schema, and tombstones. Do not edit mother and child metadata concurrently. Version 3.4.10 requires machine-scoped queries and explicit clearing to prevent stale names, notes, or folders returning.

<a id="remote"></a>
## Worker is offline or disabled

Run `source ~/.redrhex_remote.env` and `python -m tools.training_panel.remote_worker --once`. Verify required variables, token scope, system time, machine ID, heartbeat freshness, and `accept_jobs`. Keep the environment file mode at `600`.

<a id="artifacts"></a>
## Video, export, or readiness fails

Open the selected process output and verify the run has a real `model_*.pt`. Stop other Isaac jobs for video/export. For readiness, check the reported panel Python and `onnx`, `onnxruntime`, `torch`, and optional MuJoCo dependencies. A missing optional stage produces review; a required failure produces blocked.
