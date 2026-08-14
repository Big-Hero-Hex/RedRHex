---
id: operator-training-panel
title: Operate the Training Panel
lang: en
audience: operator
type: how-to
status: active
owner: panel
last_reviewed: 2026-08-14
---

<a id="start"></a>
## Start the local panel

```bash
python -m tools.training_panel --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080`. For another trusted machine, keep the panel bound to localhost and use `ssh -L 8080:127.0.0.1:8080 user@host`. Binding to `0.0.0.0` exposes an unauthenticated administrative interface to the LAN and should be limited to a trusted network.

<a id="launch-and-history"></a>
## Launch and manage runs

Use Train to select task, environments, iterations, reward, terrain, and physics presets, spring backend, and resume options. New policy runs use `native` as a provisional default. `explicit` training is visible but disabled because the current uncalibrated `200 N*m/rad` model is numerically unstable at 120 Hz; investigate the backend through the [torsion-spring calibration and characterization workflow](../calibration/torsion-spring-calibration.en.md#backend). This is an operational quarantine, not production selection of Native.

The panel passes `--panel_overrides` and, for a non-empty physics candidate, `--physics-profile` to bind each launched run to its saved inputs. History discovers checkpoints, events, exports, videos, notes, folders, and deployment reports. Running cards expose iteration progress, throughput, and ETA; details include reward and episode-length curves plus launch-time Git provenance. Existing checkpoint playback and export reuse the recorded backend, and a stamped checkpoint cannot be silently evaluated with the other backend.

A blank seed makes the panel select and record one. Keep divergence handling on notification-only until its behavior is verified against the task's reward scale; automatic stop must be explicitly selected. The current view/run survives refresh through the URL, while the topbar freshness indicator shows whether the backend is still responding.

Only one Isaac GPU job should run at a time. The queue inserts a settle window between jobs. Stop a selected process from Process Console and wait for termination before launching another.

<a id="physics-presets"></a>
## Use physics presets

Physics exposes 113 schema-validated simulation quantities covering mass and center of mass, contact, actuator limits and constants, all joint-friction terms, passive springs, command delay, and ABAD calibration. Torsion-spring damping defaults to zero; the stable `damper_0` through `damper_5` aliases remain for profile compatibility. Large spring actuator effort and velocity limits are nonbinding and are not spring-law clipping or a velocity brake.

Schema validation does not prove a physical value. Baseline inherits repository and USD defaults. Duplicate it, set only measured or intentional overrides, save the preset, and confirm its name in Train. Search and **Show changed only** keep large profiles reviewable.

Each non-empty candidate becomes a run-scoped `CalibrationProfileV1`. Play, video, and ONNX export reuse that run's saved profile. Physics presets affect simulation experiments; they do not authorize hardware operation.

<a id="google-drive-export"></a>
## Export recorded video to Google Drive

Install `rclone` on the training PC and run `rclone config`. Create a Google Drive remote named exactly `redrhex-drive`; for a personal My Drive, prefer the `drive.file` scope so the remote is limited to files it creates. Verify the host-side connection before restarting Mother:

```bash
rclone listremotes
rclone lsd redrhex-drive:
```

In History, select a run and choose the latest recording or an older checkpoint video, then click **Export to Drive**. The background upload targets `RedRHex Videos/<run-id>/<video-filename>`. An unchanged completed export is reused, failed or interrupted work can be retried, and **Open in Drive** opens the connected account's private file without changing sharing permissions. Rclone credentials remain on the training PC and are never returned by the Panel API.

<a id="remote"></a>
## Use remote team mode

The remote worker requires Supabase URL, anonymous key, machine token, machine ID, and an explicit accept-jobs setting. Store them in `~/.redrhex_remote.env` with mode `600`; never place the service-role or machine token in GitHub Pages or committed files.

Start and supervise the worker from Control Center. Leave remote job acceptance disabled until configuration and ownership are verified. Role boundaries are viewer, operator, and admin; destructive actions remain admin-only.

After installing the 3.6.1 spring-safety update, pause acceptance and restart the remote worker as well as Mother. Mother startup does not replace an already-running worker. Confirm that the heartbeat comes from the updated worker before accepting jobs; the checked-in 3.4.10 Child omits `spring_backend`, so the updated worker must supply the safe Native default.

<a id="safety"></a>
## Operational boundaries

The panel launches existing scripts; it does not make a training result safe for hardware. Export, deploy readiness, ROS preflight, physical E-stop preparation, and staged motor enable remain separate gates. Compact or delete only after preserving the selected checkpoint and evidence paths.

<a id="component-docs"></a>
## Component documentation

See the [Training Panel component portal](../../../tools/training_panel/docs/index.en.md) for architecture, remote contracts, deployment, troubleshooting, and release details.
