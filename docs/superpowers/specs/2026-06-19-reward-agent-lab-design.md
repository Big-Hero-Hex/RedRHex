# Reward Agent Lab Design

## Context

RedRHex already has a Training Panel that handles manual RSL-RL training operations: launching runs, queueing jobs, tracking history, opening TensorBoard, recording videos, managing reward and terrain presets, running deploy checks, and supporting remote team workflows.

The new feature should automate reward iteration for reinforcement learning without making the existing panel hard to understand. The approved direction is a dedicated Reward Agent Lab inside this repository, with only a thin integration into the current panel.

## Goals

- Let the user describe a reward-iteration goal in natural language.
- Let the agent generate bounded reward-weight candidates and launch controlled trials.
- Evaluate runs with deterministic metrics, not subjective LLM judgment.
- Allow the agent to propose new reward function code, but only behind explicit user approval.
- Keep all trials linked to existing panel run history, TensorBoard, video, and deploy workflows.
- Keep the existing Training Panel focused on manual operations.

## Non-Goals

- Do not replace the Training Panel.
- Do not let the agent silently edit reward source code.
- Do not let the LLM decide success without metric-based scoring.
- Do not change physics, observations, action space, reset logic, or robot configuration as part of low-risk automatic proposals.
- Do not duplicate training launch, history, or TensorBoard parsing code already owned by the panel.

## Architecture

Create a dedicated package:

```text
tools/reward_agent/
  __init__.py
  __main__.py
  agent.py
  evaluator.py
  planner.py
  code_proposals.py
  experiment_store.py
  server.py
  static/
```

The Reward Agent Lab owns:

- reward iteration goals
- candidate reward presets
- experiment queues
- run scoring and comparison reports
- approval-gated reward code proposals
- agent conversation and decision history

The Reward Agent Lab reuses existing Training Panel modules for:

- training launch and queue behavior
- run history discovery and updates
- TensorBoard scalar parsing
- convergence checks
- reward and terrain preset formats
- video, playback, and deploy actions when needed

The existing panel gets only a thin integration:

- an `Open Reward Agent Lab` button or link
- a small status card showing `idle`, `running trials`, `waiting for approval`, or `error`
- an emergency pause/stop scheduling control

## Existing-Code Fit

The active RedRHex simplified reward implementation reads `env_cfg.v2_reward_scales` in `source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py`.

The first foundation task must add proper override support for individual `v2_reward_scales` keys. The current panel writes reward overrides before launch, but `scripts/rsl_rl/train.py` only applies direct `env_cfg` attributes. The Reward Agent Lab should support nested reward-scale overrides without requiring source edits for ordinary weight tuning.

## Data Model

Store reward-agent state under:

```text
logs/reward_agent/
  sessions.json
  sessions/<session_id>/
    goal.json
    trials.json
    evaluations.json
    conversation.jsonl
    proposals/
    reports/
```

Core records:

- `AgentSession`: user goal, constraints, target task, terrain preset, baseline run, budget, guardrails, and current status.
- `RewardCandidate`: generated reward weights, parent candidate, hypothesis, bounds used, risk notes, and creation reason.
- `Trial`: linked panel run ID, launch parameters, seed, max iterations, candidate ID, status, and failure details.
- `EvaluationReport`: TensorBoard metrics, command-sweep metrics, safety regressions, aggregate score, and confidence/completeness flags.
- `CodeProposal`: proposed patch diff, explanation, new config keys, approval state, expected improvements, guardrails, risk level, and rollback notes.

Every trial must keep the real panel run ID so existing History, TensorBoard, video, and deploy workflows remain usable.

## Workflow

1. User creates an agent session with a goal such as improving diagonal movement, reducing energy cost, or improving yaw stability.
2. Agent selects a baseline run or asks the user to choose one.
3. Agent generates small bounded reward-weight candidates.
4. Agent queues short training trials through the existing panel backend.
5. Evaluator scores completed trials.
6. Agent promotes promising candidates and discards poor candidates.
7. If weight tuning stalls, or metrics indicate a missing reward signal, the agent drafts a new reward code proposal.
8. User approves or rejects the code proposal.
9. Approved proposals are applied as controlled experiments and compared against the previous best candidate.
10. Agent writes a final comparison report.

## Evaluator

The evaluator is deterministic and owns the definition of whether a run improved.

Inputs:

- TensorBoard scalars such as `Train/mean_reward`, episode length, reward components, and diagnostics.
- `scripts/rsl_rl/eval_command_sweep.py` output for forward, lateral, diagonal, and yaw command performance.
- Safety metrics such as fall rate, base height, tilt, action NaN count, and energy per distance.
- Regression comparisons against both the baseline and the current best candidate.

Scoring is multi-objective:

```text
overall_score =
  command_tracking_score
  + skill_pass_score
  + stability_score
  - energy_penalty
  - fall_penalty
  - regression_penalty
```

The user goal adjusts score weights. For example, a diagonal-improvement goal increases diagonal metrics, while forward and yaw behavior remain guardrails.

Missing metrics produce an incomplete evaluation instead of a confident pass/fail.

## Agent Behavior

The LLM role is:

- interpret the user goal
- generate reward-tuning hypotheses
- propose bounded candidate changes
- explain tradeoffs
- draft reward code proposals
- summarize experiment results

Deterministic code is responsible for:

- launching runs
- reading metrics
- computing scores
- enforcing candidate bounds
- detecting guardrail regressions
- blocking unsafe automatic approval
- persisting state and recovering after interruption

The agent starts with bounded weight edits. It runs short trials first and only promotes promising candidates to longer runs. If all candidates regress, the agent stops and asks for user guidance.

## Code Proposal Safety

New reward code goes through an explicit proposal lifecycle:

```text
Drafted -> Awaiting Approval -> Approved -> Applied -> Trial Running -> Accepted / Rejected / Rolled Back
```

A `CodeProposal` must include:

- target files, usually `redrhex_env.py` and/or `redrhex_env_cfg.py`
- proposed reward name
- hypothesis
- exact patch diff
- new config keys and default values
- expected metrics to improve
- guardrail metrics that must not regress
- rollback instructions
- estimated risk: `low`, `medium`, or `high`

The first implementation only allows low-risk reward code proposals in this constrained pattern:

- add config keys under `v2_reward_scales`
- add one clearly named reward tensor inside `_compute_simplified_rewards`
- add matching `episode_sums["rew_*"]`
- avoid physics, observations, action space, reset logic, robot config, and termination changes

Higher-risk proposals may still be drafted, but they require manual handling outside automatic apply/train flow.

## UI

Run the dedicated lab locally:

```bash
python -m tools.reward_agent --host 127.0.0.1 --port 8090
```

Main views:

- `Goal`: define optimization goal, baseline run, task, terrain, budget, and guardrails.
- `Experiments`: view candidates, linked panel run IDs, trial state, scores, and notes.
- `Compare`: compare best candidate against baseline using command sweep, TensorBoard trends, safety metrics, energy, and videos.
- `Proposals`: review reward code patches, hypotheses, risk, approval state, and apply/reject controls.
- `Conversation`: persistent agent discussion log for steering the optimization.

Panel integration is intentionally small:

- a Reward Agent card in the existing panel dashboard or control area
- current status
- lab URL button
- pause scheduling control

The existing panel remains the manual operations center.

## Milestones

### Milestone 1: Foundation

- Add `v2_reward_scales` override support to training launch.
- Add reward-agent storage under `logs/reward_agent/`.
- Add an evaluator that scores existing completed runs.
- Do not add autonomous training yet.

### Milestone 2: Weight-Tuning Agent

- Generate bounded reward-weight candidates.
- Queue short training trials through the existing panel backend.
- Score trials and rank candidates.
- Produce comparison reports.

### Milestone 3: Code Proposal Agent

- Generate approval-gated reward code patches.
- Show patch diff and hypothesis in the lab UI.
- Apply only after approval.
- Run A/B trials against the previous best candidate.

### Milestone 4: Panel Integration

- Add the small existing-panel status card.
- Add the `Open Reward Agent Lab` button.
- Add pause/stop scheduling controls.
- Keep detailed interaction inside the dedicated lab.

## Testing

Test coverage should include:

- nested `v2_reward_scales` override parsing and application
- evaluator scoring from fixture TensorBoard and command-sweep data
- candidate generation bounds
- guardrail regression logic
- agent state persistence and resume
- code proposal lifecycle transitions
- failed, queued, running, and completed trial handling
- dry-run training launch without Isaac
- panel status-card API behavior

## Failure Behavior

- If a run fails, mark the trial failed and continue only if enough candidates remain.
- If all candidates regress, stop and ask the user for guidance.
- If metrics are missing, produce an incomplete evaluation.
- If a code patch conflicts, mark it as `needs-rebase` and do not apply it.
- If the GPU is busy, use the existing panel queue behavior.
- If the Reward Agent Lab crashes, recover from `logs/reward_agent/` state on restart.

## Approval Status

The user approved this design direction section by section:

- dedicated `tools/reward_agent` package with thin panel integration
- separate reward-agent state model
- deterministic evaluator with LLM explanation
- approval-gated reward code proposals
- dedicated local Reward Agent Lab UI
- four implementation milestones with focused testing
