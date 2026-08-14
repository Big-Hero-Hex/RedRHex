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

Autopilot adds one long-lived panel-owned `AutopilotService`, a SQLite store, deterministic controller, exact-checkpoint evaluator process, and local Goals workspace. HTTP handlers call this service; reads never reconcile or mutate state. The service deliberately has no model client or model API key. A separate `redrhex-autopilot` plugin translates a narrow MCP tool set into revisioned panel requests.

<a id="process-contract"></a>
## Process and artifact contract

`TrainingParams` builds the established `train.py` interface and passes `--panel_overrides`. The process registry serializes Isaac/GPU work, records commands and logs, reconciles RSL-RL artifacts, and associates panel request IDs with discovered run directories. A nonblocking `fcntl` lease on `logs/training_panel/gpu_process.lock` serializes GPU launches across registry instances and processes; the child inherits the descriptor so the kernel releases the slot after process exit even if Mother restarts. History writes are guarded by an `RLock` and atomic replacement.

Campaign training adds typed stage/evaluation selection, immutable reward and terrain snapshot paths and hashes, exact source-checkpoint identity, strict policy-only loading, and campaign/trial ownership. Every control and candidate starts from the same frozen checkpoint with optimizer reset, or from fresh initialization; candidate checkpoints never become the next candidate's policy source. The command sweep is a first-class serialized process and evaluates the exact trained output checkpoint against an immutable command profile.

<a id="autopilot-contract"></a>
## Autopilot campaign contract

`autopilot.py` owns the V1 schemas, allowlists, command compilation, hard gates, ranking, states, and transition table. `autopilot_store.py` persists revisioned campaign snapshots in SQLite WAL mode, enforces the one-host-slot invariant, and appends immutable events, trials, decisions, evaluations, artifacts, and idempotency results. `autopilot_service.py` compiles human drafts against resolved repository state, owns worker/recovery ticks, and is the only code allowed to advance a campaign or declare `simulation_goal_met`.

The controller allocates one seed-42 control, at most nineteen adaptive seed-42 screens, and four reserved seed-43/44 control/winner confirmations inside the 24-trial ceiling. Hard identity and safety gates run before deterministic ranking. Training reward and TensorBoard curves remain diagnostics. A successful campaign requires valid evidence for all paired replicas, at least two of three candidate passes, improved median tracking over controls, and energy below the cap.

SQLite lives at `logs/training_panel/autopilot.sqlite3`; content-addressed evidence lives below `logs/training_panel/autopilot_artifacts/`. Existing Reward Agent JSON is retained and imported only as non-armable provenance. See the [Autopilot API reference](autopilot-api.en.md) for schemas, routes, mutation headers, MCP tools, and the patch-handoff contract.

<a id="google-drive-contract"></a>
## Google Drive export contract

`google_drive.py` owns the host-side rclone readiness check, Mother-wide destination settings, account-reconnect lifecycle, and background video export. The remote name remains fixed at `redrhex-drive:`. `POST /api/google-drive/settings` accepts either a validated relative My Drive path or an HTTPS `drive.google.com/drive/folders/<id>` URL. Path mode verifies or creates the private folder with argument-vector `rclone mkdir`; link mode parses the folder ID and optional legacy resource key, then verifies directory access with `rclone lsjson --stat` plus `--drive-root-folder-id` and, when required, `--drive-resource-key`. The selected destination is stored in mode `600` at `logs/training_panel/google_drive_settings.json`; OAuth output and resource keys are excluded from API state, activity, and run history.

`POST /api/google-drive/reconnect` coalesces concurrent requests and runs `rclone config reconnect redrhex-drive:` in a bounded background process that opens authorization on the training PC. Folder or successful account changes increment a local destination revision. Export commands apply the same root-folder options used for link validation, so the effective destination is either `<configured-path>/<sanitized-run-id>/<sanitized-video-name>` or `<linked-folder>/<sanitized-run-id>/<sanitized-video-name>`.

The server resolves the latest or requested checkpoint video and rejects missing files, non-MP4 sources, run directories outside the repository RSL-RL root, and videos outside the selected run directory before rclone starts. This accepts task-specific roots such as `redrhex_wheg`, `redrhex_forward_fast`, and Sensor V2 directories without weakening per-run containment.

Each run persists `google_drive_video_exports`, keyed by the run-relative video path. An entry records source size and nanosecond mtime, checkpoint iteration, lifecycle state, a secret-free destination mode/display/URL and revision, Drive file ID, private view URL, timestamps, and a bounded redacted error. A completed entry is reused only when both the source fingerprint and destination identity match. Concurrent clicks for one source coalesce, changed or failed sources start a new attempt, settings changes are rejected while an upload is active, and startup converts stale uploading entries to interrupted. Export does not acquire the GPU lock. Rclone `copyto` receives an argument vector rather than a shell command; `lsjson --stat` supplies the Drive ID. The exporter never calls `rclone link` or changes sharing permissions.

<a id="physics-profile-contract"></a>
## Physics profile contract

`physics.py` owns the browser-facing schema and the local sparse preset store. Its 113 fields are the independently adjustable `CalibrationProfileV1` values consumed by the current Isaac integration. API and command payloads accept only schema keys and finite bounded numbers. Cross-field validation is delegated to `CalibrationProfileV1`; coupled ground friction and passive-spring requirements are completed while materializing the candidate.

The six torsion springs retain stable `damper_0` through `damper_5` profile aliases. Their uncalibrated damping default is zero. Compatibility fields for uniform spring stiffness and damping map to the effective backend-aware spring parameters; large effort and velocity values keep actuator-path limits nonbinding and do not clip or brake the spring law.

The process registry writes each non-empty candidate to `logs/training_panel/process_overrides/<process>_physics.json` and forwards it with `--physics-profile`. `train.py` applies the profile through the sim2real integration and snapshots the exact applied contract under the run's `params/`. History stores preset identity, sparse values, and candidate path. Evaluation and export prefer the immutable run snapshot and reconstruct only when no snapshot exists. Empty Baseline passes no profile and removes a stale process candidate.

<a id="physics-robot-preview"></a>
## Physics robot preview

`robot_geometry.py` parses the deploy pipeline's canonical URDF and serves a layout at `GET /api/physics/robot-geometry`. The layout is keyed by canonical leg index and carries, per leg, the body mount point and the origin, axis, canonical id, URDF name, and resting angle of its ABAD, main-drive, and torsion-spring joints. Joint ordering is taken from the shared contract rather than re-derived, and the torsion-spring resting angle is the tunable field's inherited schema default so an unmodified preset shows the real spawn pose. The layout is cached against the URDF's size and modification time. A missing or unparseable URDF returns a flat six-leg fallback layout marked `source: "fallback"` instead of failing the page.

The layout also reports a `label_audit`. Leg positions derived from the URDF disagree with `_LEG_LABELS` on the right side: indices 0, 1, and 2 are named right front, middle, and rear but are mounted at the right middle, rear, and front positions. Canonical indices, actuator grouping, and tripod membership are unaffected, so this is a naming defect rather than a training one. The panel therefore draws URDF positions and states the mismatch rather than implying agreement.

`robot_view.js` renders that layout. It is the panel's only ES module; `index.html` resolves the bare `three` specifier through an import map to a vendored, pinned three.js build under `static/vendor/`, keeping the panel free of a bundler and of network requests. `app.js` remains a classic script and drives the module through `window.RedRHexRobotView`. Geometry is procedural, so no mesh assets are served. Each joint family is drawn as the mechanism it is rather than as an interchangeable marker: the ABAD is a capped hinge pin on its abduction axis with the arc it swings through, the main drive is a motor barrel on the lateral axis with a rotor key and sweep arc for its continuous velocity-controlled rotation, and the torsion spring is a bare-metal helical coil, uncoloured because it is passive. Joint origins and axes come straight from the URDF, and the structure between them is drawn too -- the standoff to each hip, the outriggers that carry the splayed middle legs out past the shell, and the link between the main drive and the torsion spring -- so the parts read as a connected mechanism. The shell is sized from the data rather than guessed: its width stops inboard of the nearest leg mount so hip hardware stands clear of the flank, and its top reaches the hip line derived from the mount and ABAD offsets. The URDF frame is already Y-up, matching three.js, and is rendered unchanged; the simulator's spawn rotation maps the asset into Isaac's Z-up world and is deliberately not reapplied. Fields authored in the simulator frame, such as the center-of-mass offset, are converted per value.

The preview shows preset values, never robot state; the panel has no joint telemetry. Quantities with no honest spatial depiction, including linear and angular damping and aggregate command delay, are reported as text beside the model. Selecting a leg reuses the existing Physics search filter rather than introducing a second filtering mechanism. The preview is sticky: it pins to the top of the window while the field list scrolls and collapses to a compact bar, dropping the readout strip and shortening the label warning. Stuck state is measured on a frame-throttled scroll listener from a zero-height anchor at the preview's resting position. An IntersectionObserver is unsuitable because an instant scroll can carry a sentinel from below the fold to above it without ever intersecting, so the ratio never crosses a threshold and the observer stays silent. Measuring the preview's own rect is unsuitable because collapsing changes its height, letting the collapse invalidate the condition that produced it and latch. The anchor cannot be moved by the collapse, so it stays correct; the computed position is also checked, since sticking is disabled on short viewports. Physics rows carry a matching `scroll-margin-top` so focus and `scrollIntoView` never park a field under the pinned bar, and sticking is disabled on short viewports. The render loop is mounted lazily, and stops whenever the Physics view or the document is hidden. When WebGL is unavailable or the context is lost, the viewer downgrades to a top-down SVG schematic that keeps leg selection and the center-of-mass marker; `?robot3d=off` forces that path for testing.

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

Autopilot is separately disabled by default through `REDRHEX_AUTOPILOT_ENABLED`. Keep its panel and MCP endpoints on loopback. The narrow adapter cannot arm, resume, widen constraints, access a generic shell or file API, apply patches, deploy, or actuate hardware. OpenAI Secure MCP Tunnel and ChatGPT Scheduled are external operator-owned services: this repository supplies the local adapter and recurring prompt, but does not provision their identity, API key, connection, or schedule.

Viewer is read-only. Operator can mutate shared metadata, presets, and folders and submit non-destructive jobs. Admin additionally deletes. Supabase constrains run metadata to an RPC and queued cancellation to the requester or an admin; triggers/RPCs and worker events audit authenticated actors. The worker resolves the authoritative profile role and ignores any browser-supplied `actor_role`. Terminal, raw logs, worker controls, arbitrary host files, convergence edits, GUI viewers, and robot actuation are never projected to Child.

<a id="version"></a>
## Version contract

Mother release `3.8.0-autopilot-preview` adds the local off-by-default campaign surface without changing the remote protocol. Remote Child assets, capability rows, cache keys, schema label, and worker synchronization remain `3.7.0-remote-parity`. The remote migration is additive: existing 3.4.10 rows and read paths stay valid. An older schema or worker keeps authentication and inspection available, but the Child disables mutations until both remote compatibility signals match. Update every surface belonging to the contract being released, retain compatibility evidence, and add a bilingual release entry.

V3.5 adds progress parsing, TensorBoard summaries, divergence monitoring, Git provenance, and recorded random seeds. V3.6 adds URL-backed navigation, action-local error reporting, first-load skeletons, keyboard-focus tooltips, run-card action menus, and backend-freshness state. V3.6.1 quarantines Explicit policy training, makes Native the provisional new-run default, and preserves backend identity for stamped checkpoints. V3.6.2 makes the Train form route-aware, restores reliable hidden-state rendering, and omits irrelevant stage fields from browser requests. V3.6.3 gives run comparison its own panel, confirms destructive History actions before showing them as in progress, persists run-list filters, and adds keyboard navigation to the run list. V3.6.4 adds private, checkpoint-aware Google Drive video export through a host-configured rclone remote. V3.7 adopts those Mother-grade contracts in the remote-safe Child and adds the capability/security protocol described above. V3.8 adds the local Autopilot preview, deterministic campaign/evaluation authority, and narrow external-advisor boundary. Tests cover command construction, history, progress/convergence/provenance, queue/process behavior, remote roles/sync, notifications, contract parity, deployment, Drive export, campaigns, MCP scope, and both UI surfaces.

<a id="pages"></a>
## Pages artifact

The checked-in remote web source remains the GitHub Pages root. Documentation-site staging must place the bilingual docs under `/docs/` without changing remote asset paths or behavior.
