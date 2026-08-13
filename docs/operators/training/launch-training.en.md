---
id: operator-launch-training
title: Launch a Training Run
lang: en
audience: operator
type: how-to
status: active
owner: training
last_reviewed: 2026-08-14
---

<a id="choose-task"></a>
## Choose a task

- `Template-Redrhex-Direct-v0` is the full locomotion task and supports the five-stage curriculum.
- `Template-Redrhex-ForwardFast-Direct-v0` is a bounded forward-only profile for faster iteration.

Use ordinary PPO unless you intentionally select a teacher or distillation agent configuration.

For the bounded straight-gait experiment, use the exact task ID `Template-Redrhex-ForwardFast-Direct-v0`; it is now the Training Panel default. The source profile and the panel's active `Straight Forward Focus` preset use the same forward-tracking weights. They target the active simplified reward dictionary; the older flat `rew_scale_*` fields do not tune the simplified reward path.

<a id="smoke-run"></a>
## Run a smoke training

Set `ISAACLAB_ROOT`, then run a small headless job:

```bash
"$ISAACLAB_ROOT/isaaclab.sh" -p scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --num_envs 4 \
  --max_iterations 1 \
  --headless
```

Success means the process reaches an update and writes a run directory under `logs/rsl_rl/redrhex_wheg/`. A smoke result proves the stack executes; it does not prove locomotion quality.

<a id="full-run"></a>
## Start a longer run

Increase environment count and iterations only after the smoke run passes. Values depend on available GPU memory and the experiment protocol:

```bash
"$ISAACLAB_ROOT/isaaclab.sh" -p scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --num_envs 4096 \
  --max_iterations 8000 \
  --headless \
  --run_name baseline
```

Manual runs ignore panel-generated reward and terrain override files by default. Add `--panel_overrides` only when the run is intentionally tied to those files.

For a ForwardFast run, start with a one-iteration smoke test, then use the profile's 1,500-iteration training horizon after the smoke test passes:

```bash
"$ISAACLAB_ROOT/isaaclab.sh" -p scripts/rsl_rl/train.py \
  --task Template-Redrhex-ForwardFast-Direct-v0 \
  --num_envs 4096 \
  --max_iterations 1500 \
  --headless \
  --run_name forward_spring_baseline
```

<a id="resume"></a>
## Resume a run

Full resume restores policy, optimizer, and iteration state:

```bash
"$ISAACLAB_ROOT/isaaclab.sh" -p scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --resume \
  --load_run RUN_DIRECTORY \
  --checkpoint model_ITERATION.pt \
  --headless
```

Use `--resume_policy_only` only for an intentional policy-weight handoff. It resets optimizer continuity and should be recorded in the experiment notes.

<a id="monitor"></a>
## Monitor the run

Continue with [monitoring training](monitor-training.en.md). Stop a run with `Ctrl+C` and wait for Isaac Sim to exit before starting another GPU job.
