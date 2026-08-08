---
id: reward-agent-weight-tuning
title: Tune Reward Weights with Reward Agent
lang: en
audience: developer
type: how-to
status: active
owner: reward-agent
last_reviewed: 2026-08-07
---

<a id="baseline"></a>
## Establish a baseline

Choose one task, checkpoint policy, command profile, seed set, environment count, iteration budget, and evaluation metric contract. The required score inputs are command tracking, skill pass, stability, energy penalty, and fall penalty. Do not tune until the baseline is complete and reproducible.

<a id="session"></a>
## Create a session and candidates

```bash
python -m tools.reward_agent create-session --objective "improve forward tracking without worse falls"
python -m tools.reward_agent propose-candidates \
  --session-id SESSION_ID \
  --base-overrides-json '{"v2_reward_scales":{"velocity_tracking":4.0}}' \
  --scale velocity_tracking:3.5:4.5
```

The default multipliers are 0.8 and 1.2, then clamped to the supplied bounds. Generated IDs and change records are deterministic for the declared order.

<a id="preview"></a>
## Preview trials

Pass a complete `TrainingParams` JSON object and inspect the saved dry-run records:

```bash
python -m tools.reward_agent queue-trials \
  --session-id SESSION_ID \
  --base-params-json '{"task":"Template-Redrhex-Direct-v0","num_envs":4,"max_iterations":1,"device":"cuda:0"}' \
  --limit 1 \
  --dry-run
```

Confirm task, run budget, device, candidate ID, reward overrides, and client request ID before launch.

<a id="launch"></a>
## Launch explicitly

Repeat the inspected command with `--launch`. The adapter queues through the Training Panel registry so run history and override snapshots use the established operational path. Never run a large batch before one candidate completes the smoke and evaluation path.

<a id="evaluate"></a>
## Evaluate and report

Collect the same required metrics for baseline and candidates. Incomplete evaluations rank below complete ones. A higher score is only a selection aid; inspect component metrics, regressions, run configuration, artifacts, and behavior. Commit a bilingual experiment summary only when the result changes the accepted baseline or recommendation.
