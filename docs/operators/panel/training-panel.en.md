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

<a id="autopilot-campaigns"></a>
## Run an Autopilot campaign

Autopilot is an off-by-default, simulation-only preview for bounded standard-PPO reward experiments. Start Mother with `REDRHEX_AUTOPILOT_ENABLED=1` on loopback, open **Autopilot**, and draft the task, stage, walk/run label, directions, exact numeric command envelope, initialization, tunable weights, explicit iteration cap, and campaign budget. Review the generated numbers and immutable identities before the human-only **Arm** action. Only one campaign may be armed; the maximum is 24 training trials and 72 active GPU-hours.

After arming, the panel owns training, exact-checkpoint command evaluation, safety gates, ranking, confirmation seeds, recovery, and the `simulation_goal_met` decision. ChatGPT can only advise through the narrow connector; it cannot arm, resume, widen budgets, declare success, deploy, or operate hardware. The repository includes the adapter and recurring prompt but does not provision ChatGPT Scheduled, an OpenAI Secure MCP Tunnel, or credentials. Without those external services, active work finishes and the campaign waits safely.

Use pause-after-current, resume, stop-after-current, or campaign-only emergency stop according to the displayed state. A patch handoff is a download-only review artifact and is never applied automatically. For the complete draft, connector, and recovery workflow, see the [component operator guide](../../../tools/training_panel/docs/operator-guide.en.md#autopilot). `simulation_goal_met` is not policy export, deployment readiness, or hardware authorization.

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

Open **Settings → Google Drive** to manage two explicit choices: **Google account** and **Choose a Drive folder**. To use an existing folder, copy its Google Drive folder URL, paste it into **Google Drive folder**, and click **Use this folder**. The panel extracts the folder ID locally and verifies that the connected account can access it; it never changes sharing. You may instead type a relative My Drive path such as `Robotics/Panel Exports`, which rclone verifies or creates privately. With the restricted `drive.file` scope, a manually created folder link may not be visible; in that case use a path created through Settings or reconfigure the rclone scope deliberately.

To switch owners for new uploads, wait for active Drive work to finish and click **Change Google account**, then choose the account in the browser opened on the training PC. Existing files and private links stay in their original account. Account and folder changes are versioned so the next export does not incorrectly reuse a record from the previous destination.

In History, select a run and choose the latest recording or an older checkpoint video, then click **Export to Drive**. The background upload targets `<configured-folder>/<run-id>/<video-filename>`. An unchanged completed export for the same destination is reused, failed or interrupted work can be retried, and **Open in Drive** opens the connected account's private file without changing sharing permissions. Rclone credentials remain on the training PC and are never returned by the Panel API. **Copy terminal command** provides the host-side reconnect command when browser authorization cannot be started from Settings.

<a id="remote"></a>
## Use remote team mode

The remote worker requires Supabase URL, anonymous key, machine token, machine ID, and an explicit accept-jobs setting. Store them in `~/.redrhex_remote.env` with mode `600`; never place the service-role or machine token in GitHub Pages or committed files.

Start and supervise the worker from **Settings → Remote operations**. Leave remote job acceptance disabled until configuration and ownership are verified. Apply the additive 3.7 migration, update Mother, restart the worker, and confirm that its heartbeat and capability row both report `3.7.0-remote-parity` before publishing Child assets or accepting jobs. An older schema or worker leaves Child signed in but read-only. Connection URLs and integration health are under **Connections**; launch mode, auto-start, logs, and raw status are under **Advanced**.

Child keeps Dashboard, Train, History, and More phone-first while adding Mother-grade routes, shared Reward/Terrain/Physics presets, folders, comparison, bounded curves, provenance, private Drive links, deployment evidence, read-only detection, activity attribution, and Connection health. Checkpoints are selected by run and iteration; the worker resolves the host path. Viewer is inspection-only, operator may edit shared metadata and run non-destructive jobs, and admin additionally deletes. Bulk deletion requires `DELETE` and reports each run separately.

Terminal access, raw logs, worker administration, arbitrary host paths, GUI viewers, convergence edits, and physical deployment remain Mother-only. Remote Deploy accepts repository-owned inputs and enumerated MuJoCo scenarios for validation and recording; it cannot actuate hardware.

<a id="safety"></a>
## Operational boundaries

The panel launches existing scripts; it does not make a training result safe for hardware. Export, deploy readiness, ROS preflight, physical E-stop preparation, and staged motor enable remain separate gates. Compact or delete only after preserving the selected checkpoint and evidence paths.

<a id="component-docs"></a>
## Component documentation

See the [Training Panel component portal](../../../tools/training_panel/docs/index.en.md) for architecture, remote contracts, deployment, troubleshooting, and release details.
