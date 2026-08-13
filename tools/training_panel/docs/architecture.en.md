---
id: training-panel-architecture
title: Training Panel Architecture and Contracts
lang: en
audience: developer
type: explanation
status: active
owner: panel
last_reviewed: 2026-08-13
---

<a id="components"></a>
## Components

The local mother is a Python `ThreadingHTTPServer` with static assets and APIs backed by configuration, process registry, history, activity, presets, convergence, deployment, and remote-worker management. GPU actions run as subprocesses or detached tmux sessions. The child is a static ES-module app. Supabase holds remote coordination state; the worker is the only component that executes jobs on the training PC.

<a id="process-contract"></a>
## Process and artifact contract

`TrainingParams` builds the established `train.py` interface and passes `--panel_overrides`. The process registry serializes Isaac/GPU work, records commands and logs, reconciles RSL-RL artifacts, and associates panel request IDs with discovered run directories. History writes are guarded by an `RLock` and atomic replacement.

<a id="remote-contract"></a>
## Remote contract

Remote roles and job types are defined centrally in `remote_config.py`. A heartbeat reports panel version, machine ID, paths, active job, queue depth, GPU lock, acceptance state, tunnel, and time. The worker claims authorized jobs, executes through the same process registry, and syncs runs, artifacts, metadata, folders, tombstones, and notifications.

<a id="spring-contract"></a>
## Spring-backend contract

Run creation accepts only `explicit` or `native` and records the selection in parameters and history. Training, Play, automatic video, export, deployment validation, and remote synchronization reuse the stored backend and fail closed on missing or invalid spring metadata. ForwardFast automatic recording alone adds `--initial_command forward`; interactive Play and full Direct recordings do not.

<a id="security"></a>
## Security boundary

Mother has no built-in authentication and can launch or delete local work; localhost plus SSH is the default boundary. The child may expose only publishable configuration. Service-role/machine tokens stay on the worker host. Role checks are defense in depth, not a replacement for Supabase policy and secret handling.

<a id="version"></a>
## Version contract

The local Mother package and UI are release `3.6.0-panel-ux`. The independently deployed remote Child assets, Child release metadata, heartbeat schema, and worker synchronization contract remain at `3.4.10-sync-health`. A local UI release does not silently change the remote protocol. Update every surface belonging to the contract being released, retain compatibility evidence, and add a bilingual release entry.

V3.5 adds progress parsing, TensorBoard summaries, divergence monitoring, Git provenance, and recorded random seeds. V3.6 adds URL-backed navigation, action-local error reporting, first-load skeletons, keyboard-focus tooltips, run-card action menus, and backend-freshness state. Tests cover command construction, history, progress/convergence/provenance, queue/process behavior, remote roles/sync, notifications, contract parity, deployment, and UI assets.

<a id="pages"></a>
## Pages artifact

The checked-in remote web source remains the GitHub Pages root. Documentation-site staging must place the bilingual docs under `/docs/` without changing remote asset paths or behavior.
