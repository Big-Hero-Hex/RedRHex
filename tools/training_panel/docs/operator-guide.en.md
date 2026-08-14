---
id: training-panel-operator-guide
title: Training Panel 3.6.1 Operator Guide
lang: en
audience: operator
type: how-to
status: active
owner: panel
last_reviewed: 2026-08-14
---

<a id="start"></a>
## Start Mother

```bash
python -m tools.training_panel --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080`. Prefer an SSH tunnel for another machine. The local panel is an unauthenticated administrative surface; bind `0.0.0.0` only on a trusted LAN.

With VS Code Remote SSH, start the panel in the remote terminal, then forward remote ports `8080` and `6006` from the Ports view and open their forwarded local URLs. Keep the panel bound to `127.0.0.1`; the SSH tunnel provides access without exposing the administrative surface to the LAN.

<a id="train"></a>
## Train and queue

In Train, choose task, environments, iterations, device, reward preset, terrain preset, spring backend, and resume mode. Native is the provisional default for new policy runs. Explicit is visible but disabled for policy training because its current uncalibrated `200 N*m/rad` model is numerically unstable at 120 Hz; investigate it with the [torsion-spring calibration and characterization workflow](../../../docs/operators/calibration/torsion-spring-calibration.en.md#backend). This quarantine is not production selection of Native.

Isaac/GPU actions—training, playback, video, and ONNX export—are serialized. A training request becomes queued while another GPU action is active; History can cancel it. A settle window separates completed Isaac jobs.

Standard Panel-launched training uses run-scoped override snapshots and passes `--panel_overrides` plus an optional `--physics-profile`. Built-in reward, terrain, and physics presets are read-only; duplicate before editing. Sensor V2 routes use their fixed versioned reward/terrain contract described below, but may use the selected physics profile.

For straight-gait-only work, the panel now defaults to `Template-Redrhex-ForwardFast-Direct-v0`, and `Straight Forward Focus` is active by default. Confirm both before launch. First use 4 environments and 1 iteration as a stack smoke test; after it passes, use the intended environment count and up to 1,500 iterations for the bounded ForwardFast profile.

<a id="sensor-v2-distillation"></a>
### Train the Sensor V2 teacher and student

In **Training Route**, select **Sensor V2 — Full Teacher → Student**. The production defaults are 64 environments, 1,500 F1 Teacher updates, 800 F2 distillation updates, and 1,500 F3 student PPO updates. Give the run a name, keep Headless enabled on a remote machine, and click **Start Training** once.

The pipeline runs F1, F2, and F3 sequentially. It passes the completed `teacher_v2` checkpoint to F2 with `--teacher_checkpoint`, then passes the completed `student_distilled_v2` checkpoint to F3 with `--student_checkpoint`. A failed stage stops the pipeline. Sensor V2 uses its fixed forward reward contract and does not apply the Panel reward or terrain override files. The same spring backend and, if selected, the same run-scoped physics profile are forwarded to all three stages.

Use **Sensor V2 — F1 Teacher only**, **F2 Distillation only**, or **F3 Student PPO only** for a single stage. F2 requires a Teacher checkpoint and F3 requires a distilled checkpoint: select the source run in History, click **Resume to Train** to populate the checkpoint field, then choose the intended route. The strict loader rejects an incorrect checkpoint kind.

History attaches the completed full pipeline to its final F3 PPO directory. Open **Process Console** to see the active stage and exact checkpoint handoff; open **TensorBoard** to inspect the final F3 metrics. The F1 and F2 stage directories remain under `logs/rsl_rl/redrhex_forward_v2_teacher` and `logs/rsl_rl/redrhex_forward_v2_distillation`. A completed run is one seed of training, not the three-seed, recorded-sensor, or hardware promotion evidence.

<a id="physics-presets"></a>
### Tune physical quantities

Open **Physics** and select **Baseline** to inherit every repository and USD physical default. To make a candidate, duplicate Baseline or create a preset. Use Search to find a body, joint, limit, unit, or description; use **Show changed only** to audit the sparse candidate. A blank field means inherit. **Reset** clears one override; it does not write a default into the preset.

The editor exposes 113 independently adjustable simulation quantities: rigid-body damping; mass scale, added root mass, and root center-of-mass offsets; contact friction and restitution; aggregate command delay; stiffness, damping, effort limit, velocity limit, armature, and friction for each actuator group; static, dynamic, and viscous friction for all 18 joints; six passive springs; and six ABAD target scales and offsets. Ground static and dynamic friction are a coupled contract. Invalid values are rejected before launch.

Torsion-spring damping defaults to zero. `damper_0` through `damper_5` remain stable profile aliases. The large torsion-spring actuator effort and velocity limits are nonbinding; they are not spring torque clipping or an artificial velocity brake. Do not use arbitrary armature or uniform mass scaling as an instability fix without reviewed physical evidence.

Save the preset to keep it for later. The selected draft is included in the next run even before saving, and the Train page shows its name. Training snapshots the exact `CalibrationProfileV1` into the run. Play, Record Video, and Export ONNX reuse that snapshot rather than the currently selected preset. These settings change simulation behavior only; hardware calibration, E-stop preparation, and motor-enable authorization remain separate gates.

<a id="history"></a>
## Use History

History combines panel requests and discovered RSL-RL runs. Select a run to inspect configuration, checkpoint, spring backend/calibration status, reward/terrain differences, saved physics metadata, notes, folder, event state, video, export, and readiness evidence. Play, recording, export, and deployment checks reuse the recorded backend and reject incompatible spring metadata. An Explicit checkpoint remains available for inspection and provenance-preserving playback, but Resume to Train is blocked; backend characterization uses the deterministic workflow rather than a learned checkpoint. Available actions include TensorBoard, Play, Record Video, Export ONNX, Resume to Train, Compare, Compact Run, and Process Console.

Running cards show iteration progress, throughput, and ETA parsed from the process log. Run details draw mean reward and episode-length curves from local TensorBoard scalars. The run record also captures the Git commit, branch, and dirty state used at launch. A blank seed makes the panel choose and record a seed, so panel-launched runs remain reproducible.
Play and Record Video reuse the selected run's saved task and start with the forward command, equivalent to `W`. Export ONNX reuses the saved task without adding a motion command. The Process Console command is the source of truth for both checks.

The Windows and macOS launchers open the panel with an explicit desktop-remote marker. In that mode, **TensorBoard** starts or reuses one all-runs server on the forwarded `6006` port. Headless training is enforced. **Play**, **Open MuJoCo Viewer**, and host file-manager buttons are disabled because those windows would open on the training PC, not in the remote browser. Use recorded Isaac videos, recorded MuJoCo MP4s, browser console output, and copy-path controls instead.

Compaction keeps the highest top-level `model_*.pt` and preserves events, parameters, videos, exports, notes, and deployment reports. Deletion requires the exact run ID and is rejected while a related process is active.

<a id="convergence"></a>
## Monitor convergence

The Convergence view configures divergence detection for non-finite scalar values and sustained reward collapse. Notifications are enabled by the configured channel. Automatic stopping is opt-in: keep the action on `notify` until the detector has been checked against the task's reward scale, then select `stop` only if that behavior is intended.

<a id="navigation"></a>
## Navigate and diagnose the UI

The current view and selected run are stored in the URL, so refresh and shared links preserve context. The top bar reports backend freshness; a stale indicator means operators should stop issuing new actions until connectivity is understood. Initial loading uses skeletons instead of reporting empty data, and action failures from Rewards, Terrain, Convergence, Activity, and Control Center appear in their current view.

<a id="console"></a>
## Use Process Console

Launch Command is the command requested by the panel; Output is the captured process stream. When tmux exists, jobs run in detached sessions and the console exposes an attach command. Stop Process sends an interrupt and escalates only when Isaac does not close.

<a id="artifacts"></a>
## Export and record

Export ONNX produces `exported/policy.pt` and `exported/policy.onnx` from the selected training checkpoint. The default high-quality video preset is 1920×1080, 1,200 steps, and 30 FPS. Standard runs use their saved terrain override when available; Sensor V2 uses its fixed reward/terrain contract and saved runner. Both routes reuse the run's saved physics profile. Video also passes `--initial_command forward`.

<a id="next"></a>
## Next steps

- [Remote operation](remote-operation.en.md)
- [Deployment readiness](deploy-readiness.en.md)
- [Troubleshooting](troubleshooting.en.md)
