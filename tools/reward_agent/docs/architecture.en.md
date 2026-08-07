---
id: reward-agent-architecture
title: Reward Agent Architecture
lang: en
audience: developer
type: explanation
status: active
owner: reward-agent
last_reviewed: 2026-08-07
---

<a id="boundary"></a>
## Boundary

Reward Agent is a headless experiment-orchestration foundation. It creates sessions, generates one-weight-at-a-time bounded candidates, converts candidates to Training Panel parameters, previews or queues trials, evaluates supplied metrics, and builds comparison reports. It does not edit reward source code, scrape TensorBoard automatically, or let a language model decide success.

<a id="modules"></a>
## Modules

- `planner.py` clamps candidate changes to declared bounds and records the changed field and hypothesis.
- `experiment_store.py` persists sessions, candidates, trials, evaluations, and reports under `logs/reward_agent/`.
- `agent.py` converts candidate overrides into `TrainingParams` and records dry-run or queued trials.
- `launcher.py` constructs the existing Training Panel process registry for explicit launch mode.
- `evaluator.py` scores complete metric sets and penalizes regression against a supplied baseline.
- `reports.py` ranks complete evaluations and links the best candidate to its panel run.

<a id="safety"></a>
## Safety properties

Actual execution requires `queue-trials --launch`; `--dry-run` is separately available and required by the operator workflow. Candidate generation changes one declared weight at a time, preserves other overrides, and applies minimum/maximum bounds. A report prefers complete evaluations but does not prove the metrics are scientifically valid.

<a id="integration"></a>
## Integration status

The CLI and Training Panel registry adapter are implemented. A proposal UI, automated evaluation ingestion, source-code proposals, and deeper panel integration are roadmap work and require a new design. Runtime session data is ignored and is not canonical documentation or published experiment evidence.

<a id="tests"></a>
## Tests

Run `python -m unittest discover -s tools/reward_agent/tests -p 'test_*.py'`. Tests cover store round trips, bounded candidates, metric completeness and regression, report ranking, dry-run trial persistence, explicit launch through a fake registry, and launcher construction.
