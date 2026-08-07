---
id: operator-play-export-video
title: Play, Export, and Record a Policy
lang: en
audience: operator
type: how-to
status: active
owner: training
last_reviewed: 2026-08-07
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
  --num_envs 1
```

Keyboard control starts with `stop`. Use `--disable_keyboard_control` to retain sampled commands. A checkpoint path containing `_stage1` through `_stage5` sets `env.stage` automatically unless `--disable_auto_stage_from_checkpoint` is supplied.

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
  --video \
  --video_length 1200 \
  --headless
```

Add `--camera_follow_robot` when the fixed camera loses the robot. The result is written under the run's `videos/play/` directory.

<a id="evaluate"></a>
## Evaluate behavior

Use `scripts/rsl_rl/eval_command_sweep.py` for repeatable command profiles and CSV output. Compare tracking, success, falls, and energy metrics by skill rather than judging one playback clip.
