---
id: training-panel-operator-guide
title: Training Panel 3.4.10 Operator Guide
lang: en
audience: operator
type: how-to
status: active
owner: panel
last_reviewed: 2026-08-13
---

<a id="start"></a>
## Start Mother

```bash
python -m tools.training_panel --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080`. Prefer an SSH tunnel for another machine. The local panel is an unauthenticated administrative surface; bind `0.0.0.0` only on a trusted LAN.

<a id="train"></a>
## Train and queue

In Train, choose task, environments, iterations, device, reward preset, terrain preset, spring backend, and resume mode. Isaac/GPU actions—training, playback, video, and ONNX export—are serialized. A training request becomes queued while another GPU action is active; History can cancel it. A settle window separates completed Isaac jobs.

Panel-launched training uses a run-scoped override snapshot and passes `--panel_overrides`. Built-in reward and terrain presets are read-only; duplicate before editing.

<a id="history"></a>
## Use History

History combines panel requests and discovered RSL-RL runs. Select a run to inspect configuration, checkpoint, spring backend/calibration status, reward/terrain differences, notes, folder, event state, video, export, and readiness evidence. Play, recording, export, and deployment checks reuse the recorded backend and reject incompatible spring metadata. Available actions include TensorBoard, Play, Record Video, Export ONNX, Resume to Train, Compare, Compact Run, and Process Console.

Compaction keeps the highest top-level `model_*.pt` and preserves events, parameters, videos, exports, notes, and deployment reports. Deletion requires the exact run ID and is rejected while a related process is active.

<a id="console"></a>
## Use Process Console

Launch Command is the command requested by the panel; Output is the captured process stream. When tmux exists, jobs run in detached sessions and the console exposes an attach command. Stop Process sends an interrupt and escalates only when Isaac does not close.

<a id="artifacts"></a>
## Export and record

Export ONNX produces `exported/policy.pt` and `exported/policy.onnx` from the selected training checkpoint. The default high-quality video preset is 1920×1080, 1,200 steps, and 30 FPS. Both actions use the run's saved terrain override when available.

<a id="next"></a>
## Next steps

- [Remote operation](remote-operation.en.md)
- [Deployment readiness](deploy-readiness.en.md)
- [Troubleshooting](troubleshooting.en.md)
