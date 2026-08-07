---
id: developer-validation
title: Test and Validate RedRHex
lang: en
audience: developer
type: how-to
status: active
owner: core
last_reviewed: 2026-08-07
---

<a id="tiers"></a>
## Validation tiers

Use the cheapest sufficient tier first: pure Python unit tests, component integration tests, documentation validation, bounded Isaac smoke, short PPO/teacher/distillation smoke, command-sweep evaluation, deployment readiness, ROS mock/preflight, then constrained hardware evidence. A lower tier cannot replace a required higher-tier result.

<a id="cpu"></a>
## CPU and component suites

```bash
python -m unittest discover -s tools/documentation/tests -p 'test_*.py'
python -m unittest discover -s tools/reward_agent/tests -p 'test_*.py'
pytest -q tools/sim2real/tests tools/training_panel/tests
```

Some Training Panel UI tests have separate browser/runtime requirements; run them when changing UI behavior.

<a id="isaac"></a>
## Isaac validation

Run `scripts/rsl_rl/validate_reform_stack.py` for observation groups, terrain, faults, PPO, teacher, and distillation wiring. Start with few environments and steps. A passing random rollout is not a learning result; add the runner smoke when training code changed.

<a id="deployment"></a>
## Deployment validation

Export from the selected training checkpoint, run panel readiness, verify 56/280 observations, 12 actions, 60 Hz, joint order, limits, safety faults, and Torch/ONNX parity, then run ROS preflight and mock mode. Hardware tests require the operator safety sequence and reviewed evidence.

<a id="docs"></a>
## Documentation validation

```bash
python -m tools.documentation validate --all
python -m tools.documentation inventory --format json
```

Before commit, use `validate --staged`; in a PR, include exact `Docs impact` and `Docs reason` fields. Semantic freshness remains a review responsibility.
