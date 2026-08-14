---
id: operator-play-export-video
title: Play, Export, and Record a Policy
lang: en
audience: operator
type: how-to
status: active
owner: training
last_reviewed: 2026-08-14
---

<a id="checkpoint"></a>
## Select a checkpoint

Use a training checkpoint named `model_*.pt`, not a TensorBoard event or exported `policy.pt`. Keep its run directory because configuration and automatic stage inference use the checkpoint path.

<a id="play"></a>
## Play the policy

```bash
"$ISAACLAB_ROOT/isaaclab.sh" -p scripts/rsl_rl/play.py \
  --task Template-Redrhex-Direct-v0 \
  --load_run RUN_DIRECTORY \
  --checkpoint model_ITERATION.pt \
  --initial_command forward \
  --num_envs 1
```

Keyboard control starts with the forward command, equivalent to pressing `W`; `--initial_command forward` makes that intent explicit in saved commands. Use `--initial_command stop` for a stationary start, or `--disable_keyboard_control` to retain sampled environment commands. A checkpoint path containing `_stage1` through `_stage5` sets `env.stage` automatically unless `--disable_auto_stage_from_checkpoint` is supplied.

<a id="export"></a>
## Export JIT and ONNX

```bash
"$ISAACLAB_ROOT/isaaclab.sh" -p scripts/rsl_rl/play.py \
  --task Template-Redrhex-Direct-v0 \
  --load_run RUN_DIRECTORY \
  --checkpoint model_ITERATION.pt \
  --export_policy_only \
  --headless
```

The command writes `exported/policy.pt` and `exported/policy.onnx`. Run [deployment readiness](../deployment/deployment-readiness.en.md) before copying ONNX to hardware.

<a id="video"></a>
## Record a video

```bash
"$ISAACLAB_ROOT/isaaclab.sh" -p scripts/rsl_rl/play.py \
  --task Template-Redrhex-Direct-v0 \
  --load_run RUN_DIRECTORY \
  --checkpoint model_ITERATION.pt \
  --initial_command forward \
  --video \
  --video_length 1200 \
  --headless
```

Add `--camera_follow_robot` when the fixed camera loses the robot. The result is written under the run's `videos/play/` directory.

Training Panel Play and Record Video reuse the selected run's saved task and explicitly start with the forward command. ONNX export also reuses the saved task, but does not need an initial motion command. Confirm the Process Console command contains both the expected `--task` and `--initial_command forward` before diagnosing a stationary clip as a failed policy.

<a id="evaluate"></a>
## Evaluate behavior

Use `scripts/rsl_rl/eval_command_sweep.py` for repeatable command profiles and CSV output. Compare tracking, success, falls, and energy metrics by skill rather than judging one playback clip.
