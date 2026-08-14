---
id: reward-agent-architecture
title: Reward Agent Architecture
lang: en
audience: developer
type: explanation
status: active
owner: reward-agent
last_reviewed: 2026-08-14
---

<a id="boundary"></a>
## Boundary

Reward Agent is the retained headless, manually driven experiment-orchestration foundation. It creates sessions, generates one-weight-at-a-time bounded candidates, converts candidates to Training Panel parameters, previews or queues trials, evaluates supplied metrics, and builds comparison reports. It does not edit reward source code, scrape TensorBoard automatically, advise an Autopilot campaign, or let a language model decide success.

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

The standalone CLI and Training Panel registry adapter remain implemented for the explicit legacy workflow. The Training Panel 3.8 Autopilot preview is a separate, panel-owned lifecycle with its own UI, SQLite state, exact-checkpoint evaluator, deterministic acceptance, and narrow external-advisor connector. Reward Agent runtime JSON at `logs/reward_agent/sessions.json` is imported into the Autopilot store as non-armable historical provenance and is not deleted or promoted into a campaign.

Do not use `queue-trials --launch` as an alternative controller for an armed campaign. It does not submit `AgentDecisionV1`, consume the campaign budget, or establish campaign success. New unattended work starts in the panel's Autopilot workspace; the legacy CLI remains useful for manual dry-run planning and comparison outside that lifecycle. Source patch handoff also belongs to Autopilot and only stores a review artifact—it does not give either component permission to edit the source tree.

Runtime session data remains ignored and is not canonical documentation or published experiment evidence. See the [Training Panel Autopilot API reference](../../training_panel/docs/autopilot-api.en.md) for the new authority and compatibility boundary.

<a id="tests"></a>
## Tests

Run `python -m unittest discover -s tools/reward_agent/tests -p 'test_*.py'`. Tests cover store round trips, bounded candidates, metric completeness and regression, report ranking, dry-run trial persistence, explicit launch through a fake registry, and launcher construction. Autopilot contracts, legacy import, controller recovery, and connector scope are covered by the Training Panel and plugin suites rather than this legacy suite.
