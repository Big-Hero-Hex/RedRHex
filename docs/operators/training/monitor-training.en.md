---
id: operator-monitor-training
title: Monitor Training
lang: en
audience: operator
type: how-to
status: active
owner: training
last_reviewed: 2026-08-07
---

<a id="tensorboard"></a>
## Start TensorBoard

```bash
tensorboard --logdir logs/rsl_rl --host 127.0.0.1 --port 6006
```

Open `http://127.0.0.1:6006`. The Training Panel can also start a run-scoped TensorBoard instance and select another free port if 6006 is occupied.

<a id="signals"></a>
## Read the main signals

Check reward terms together with `Mean episode length`, `Episode_Termination/terminated`, command tracking errors, and task-specific diagnostics. For energy experiments also inspect mechanical power, cost-of-transport proxy, spring recovery, and motion speed; lower power caused only by slower or failed motion is not an improvement.

<a id="artifacts"></a>
## Locate artifacts

RSL-RL runs live under `logs/rsl_rl/<experiment>/<timestamp>_<run-name>/`. A run normally contains `model_*.pt`, TensorBoard events, and `params/`. Playback creates `videos/play/`; export creates `exported/policy.pt` and `exported/policy.onnx` beside the checkpoint.

Panel process logs, notes, activity, and history live under `logs/training_panel/`. They are runtime state and must not be committed.

<a id="stop-conditions"></a>
## Stop conditions

Stop and diagnose when termination spikes, episode length collapses, values become non-finite, GPU memory is exhausted, Isaac reports a fatal error, or the robot achieves reward without the commanded motion. Preserve the run ID and relevant metrics before deleting or compacting artifacts.
