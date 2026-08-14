---
id: training-panel-architecture
title: Training Panel Architecture and Contracts
lang: en
audience: developer
type: explanation
status: active
owner: panel
last_reviewed: 2026-08-14
---

<a id="components"></a>
## Components

The local mother is a Python `ThreadingHTTPServer` with static assets and APIs backed by configuration, process registry, history, activity, presets, convergence, deployment, and remote-worker management. GPU actions run as subprocesses or detached tmux sessions. The child is a static ES-module app. Supabase holds remote coordination state; the worker is the only component that executes jobs on the training PC.

<a id="process-contract"></a>
## Process and artifact contract

`TrainingParams` builds the established `train.py` interface and passes `--panel_overrides`. The process registry serializes Isaac/GPU work, records commands and logs, reconciles RSL-RL artifacts, and associates panel request IDs with discovered run directories. History writes are guarded by an `RLock` and atomic replacement.

<a id="google-drive-contract"></a>
## Google Drive export contract

`google_drive.py` owns the host-side rclone readiness check and background video export. The fixed `redrhex-drive:` remote writes to `RedRHex Videos/<sanitized-run-id>/<sanitized-video-name>`. The server resolves the latest or requested checkpoint video and rejects missing files, non-MP4 sources, run directories outside the RSL-RL root, and videos outside the selected run directory before rclone starts.

Each run persists `google_drive_video_exports`, keyed by the run-relative video path. An entry records source size and nanosecond mtime, checkpoint iteration, lifecycle state, destination, Drive file ID, private view URL, timestamps, and a bounded redacted error. Matching completed entries are reused, concurrent clicks for one source coalesce, changed or failed sources start a new attempt, and startup converts stale uploading entries to interrupted. Export does not acquire the GPU lock. Rclone `copyto` receives an argument vector rather than a shell command; `lsjson --stat` supplies the Drive ID. The exporter never calls `rclone link` or changes sharing permissions.

<a id="physics-profile-contract"></a>
## Physics profile contract

`physics.py` owns the browser-facing schema and the local sparse preset store. Its 113 fields are the independently adjustable `CalibrationProfileV1` values consumed by the current Isaac integration. API and command payloads accept only schema keys and finite bounded numbers. Cross-field validation is delegated to `CalibrationProfileV1`; coupled ground friction and passive-spring requirements are completed while materializing the candidate.

The six torsion springs retain stable `damper_0` through `damper_5` profile aliases. Their uncalibrated damping default is zero. Compatibility fields for uniform spring stiffness and damping map to the effective backend-aware spring parameters; large effort and velocity values keep actuator-path limits nonbinding and do not clip or brake the spring law.

The process registry writes each non-empty candidate to `logs/training_panel/process_overrides/<process>_physics.json` and forwards it with `--physics-profile`. `train.py` applies the profile through the sim2real integration and snapshots the exact applied contract under the run's `params/`. History stores preset identity, sparse values, and candidate path. Evaluation and export prefer the immutable run snapshot and reconstruct only when no snapshot exists. Empty Baseline passes no profile and removes a stale process candidate.

<a id="remote-contract"></a>
## Remote contract

Remote roles and job types are defined centrally in `remote_config.py`. Protocol `3.7.0-remote-parity` adds a host-safe `machine_capabilities` row containing feature flags, the route catalog, the Physics field schema, enumerated deployment scenarios, read-only detection settings, and integration readiness. A heartbeat and sync summary carry the same protocol version. Child enters inspection-only mode unless the selected machine and capability row both match it.

The worker claims authorized jobs, executes through the same process registry, and syncs runs, artifacts, metadata, folders, tombstones, notifications, bounded scalar series, progress, provenance, spring identity, divergence, deployment evidence, MuJoCo artifacts, Drive state, and an allowlisted Mother activity projection. Browser checkpoint input is `{run_id, checkpoint_iteration}`; the worker resolves and containment-checks the path. Remote deployment accepts repository-owned models and enumerated scenarios, never paths or shell fragments.

`start_training`, video, ONNX, and export-and-validate are serialized by the Isaac GPU lock. Stop remains prioritized. Drive export, existing-ONNX validation, and MuJoCo-only work do not take that lock. `client_request_id` is unique for the machine and actor so retries do not enqueue duplicate work.

<a id="spring-contract"></a>
## Spring-backend contract

Run metadata accepts only `explicit` or `native`. New policy requests default to Native, and the process boundary rejects Explicit training before history creation or process spawn because the current 120 Hz uncalibrated model is numerically unstable. Explicit remains available to the sim-to-real characterization path.

Play, automatic video, export, deployment validation, and remote synchronization reuse the stored backend and fail closed on invalid spring metadata. Stamped uncalibrated checkpoints reject backend mismatch; metadata-free legacy runs retain the historical Explicit fallback instead of being silently relabeled. Play and recording explicitly add `--initial_command forward`; export does not add a motion command.

<a id="security"></a>
## Security boundary

Mother has no built-in authentication and can launch or delete local work; localhost plus SSH is the default boundary. The child may expose only publishable configuration. Service-role/machine tokens and Google Drive credentials stay on the worker host. Role checks are defense in depth, not a replacement for Supabase policy and secret handling.

Viewer is read-only. Operator can mutate shared metadata, presets, and folders and submit non-destructive jobs. Admin additionally deletes. Supabase constrains run metadata to an RPC and queued cancellation to the requester or an admin; triggers/RPCs and worker events audit authenticated actors. The worker resolves the authoritative profile role and ignores any browser-supplied `actor_role`. Terminal, raw logs, worker controls, arbitrary host files, convergence edits, GUI viewers, and robot actuation are never projected to Child.

<a id="version"></a>
## Version contract

The Mother package and UI, remote Child assets, heartbeat, capability row, cache keys, sync summary, schema label, and worker synchronization contract are release `3.7.0-remote-parity`. The migration is additive: existing 3.4.10 rows and read paths stay valid. An older schema or worker keeps authentication and inspection available, but the Child disables mutations until both compatibility signals match. Update every surface belonging to the contract being released, retain compatibility evidence, and add a bilingual release entry.

V3.5 adds progress parsing, TensorBoard summaries, divergence monitoring, Git provenance, and recorded random seeds. V3.6 adds URL-backed navigation, action-local error reporting, first-load skeletons, keyboard-focus tooltips, run-card action menus, and backend-freshness state. V3.6.1 quarantines Explicit policy training, makes Native the provisional new-run default, and preserves backend identity for stamped checkpoints. V3.6.2 makes the Train form route-aware, restores reliable hidden-state rendering, and omits irrelevant stage fields from browser requests. V3.6.3 gives run comparison its own panel, confirms destructive History actions before showing them as in progress, persists run-list filters, and adds keyboard navigation to the run list. V3.6.4 adds private, checkpoint-aware Google Drive video export through a host-configured rclone remote. V3.7 adopts those Mother-grade contracts in the remote-safe Child and adds the capability/security protocol described above. Tests cover command construction, history, progress/convergence/provenance, queue/process behavior, remote roles/sync, notifications, contract parity, deployment, Drive export, and both UI surfaces.

<a id="pages"></a>
## Pages artifact

The checked-in remote web source remains the GitHub Pages root. Documentation-site staging must place the bilingual docs under `/docs/` without changing remote asset paths or behavior.
