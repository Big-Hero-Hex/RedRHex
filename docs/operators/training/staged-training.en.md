---
id: operator-staged-training
title: Run the Five-Stage Curriculum
lang: en
audience: operator
type: how-to
status: active
owner: training
last_reviewed: 2026-08-07
---

<a id="stage-purpose"></a>
## Stage purpose

The full task separates forward, lateral, diagonal, yaw, and mixed locomotion into stages 1–5. The pipeline keeps one run tag across stages and defaults to full checkpoint resume so policy, optimizer, and iteration state remain continuous.

<a id="launch"></a>
## Launch the pipeline

```bash
bash scripts/rsl_rl/train_stage_pipeline.sh \
  --run_tag experiment-name \
  --num_envs 4096
```

Defaults are 8,000, 8,000, 9,000, 10,000, and 12,000 iterations. Override them with `--s1` through `--s5`. Use `--precheck_gui 1` for a short visual check before an overnight headless run.

<a id="health-gate"></a>
## Understand the health gate

The enabled-by-default stability gate reads episode length and termination metrics after each stage. Missing metrics warn unless strict mode is enabled; clearly unhealthy metrics stop the pipeline. The pipeline log is under `logs/rsl_rl/pipeline/`.

<a id="restart"></a>
## Restart from a later stage

Reuse the original run tag and choose `--start_stage 2` through `5`. The previous stage checkpoint must still be discoverable under `logs/rsl_rl/redrhex_wheg/`.

Do not change the run tag during a resume. Use `--resume_policy_only 1` only for an intentional policy-only handoff; model numbering and optimizer state no longer represent one continuous curriculum.

<a id="acceptance"></a>
## Accept a stage

Do not accept a stage from reward total alone. Review episode length, terminations, command tracking, the skill-specific metrics, playback, and the next-stage health gate. A high return with collapse, low motion, or command avoidance is not success.
