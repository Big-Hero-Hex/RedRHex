---
id: operator-training-panel
title: Operate the Training Panel
lang: en
audience: operator
type: how-to
status: active
owner: panel
last_reviewed: 2026-08-07
---

<a id="start"></a>
## Start the local panel

```bash
python -m tools.training_panel --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080`. For another trusted machine, keep the panel bound to localhost and use `ssh -L 8080:127.0.0.1:8080 user@host`. Binding to `0.0.0.0` exposes an unauthenticated administrative interface to the LAN and should be limited to a trusted network.

<a id="launch-and-history"></a>
## Launch and manage runs

Use Train to select task, environments, iterations, reward and terrain presets, and resume options. The panel passes `--panel_overrides` to bind each launched run to its saved override files. History discovers checkpoints, events, exports, videos, notes, folders, and deployment reports.

Only one Isaac GPU job should run at a time. The queue inserts a settle window between jobs. Stop a selected process from Process Console and wait for termination before launching another.

<a id="remote"></a>
## Use remote team mode

The remote worker requires Supabase URL, anonymous key, machine token, machine ID, and an explicit accept-jobs setting. Store them in `~/.redrhex_remote.env` with mode `600`; never place the service-role or machine token in GitHub Pages or committed files.

Start and supervise the worker from Control Center. Leave remote job acceptance disabled until configuration and ownership are verified. Role boundaries are viewer, operator, and admin; destructive actions remain admin-only.

<a id="safety"></a>
## Operational boundaries

The panel launches existing scripts; it does not make a training result safe for hardware. Export, deploy readiness, ROS preflight, physical E-stop preparation, and staged motor enable remain separate gates. Compact or delete only after preserving the selected checkpoint and evidence paths.

<a id="component-docs"></a>
## Component documentation

See the [Training Panel component portal](../../../tools/training_panel/docs/index.en.md) for architecture, remote contracts, deployment, troubleshooting, and release details.
