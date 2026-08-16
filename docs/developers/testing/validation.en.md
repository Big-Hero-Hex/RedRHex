---
id: developer-validation
title: Test and Validate RedRHex
lang: en
audience: developer
type: how-to
status: active
owner: core
last_reviewed: 2026-08-15
---

<a id="tiers"></a>
## Validation tiers

Use the cheapest sufficient tier first: pure Python unit tests, component integration tests, documentation validation, bounded Isaac smoke, short PPO/teacher/distillation smoke, command-sweep evaluation, deployment readiness, ROS mock/preflight, then constrained hardware evidence. A lower tier cannot replace a required higher-tier result.

<a id="cpu"></a>
## CPU and component suites

Run these dependency-light contracts from the repository root as the minimum merge floor. Mainline code CI currently covers the Training Panel services, CPU sim-to-real subset, browser-independent JavaScript, and desktop-launcher source tests; until CI coverage expands, run the Reward Agent, Autopilot MCP, and ROS contracts locally as well. The Autopilot MCP HTTP tests require permission to bind a loopback socket, and process tests may use `tmux`.

```bash
python -m unittest discover -s tools/documentation/tests -p 'test_*.py'
python -m unittest discover -s tools/reward_agent/tests -p 'test_*.py'
python -m unittest discover -s plugins/redrhex-autopilot/tests -p 'test_*.py'
python -m pytest -q tools/training_panel/tests
python -m pytest -q tools/sim2real/tests \
  --ignore=tools/sim2real/tests/test_abad_target_mapping.py \
  --ignore=tools/sim2real/tests/test_physics_profile.py \
  --ignore=tools/sim2real/tests/test_target_delay.py \
  --ignore=tools/sim2real/tests/test_torsion_spring_model.py
PYTHONPATH="$PWD:$PWD/source/redrhex_policy_io:$PWD/ros2_ws/src/redrhex_rl_controller:$PWD/ros2_ws/src/redrhex_lowlevel_bridge" \
  python -m pytest -q ros2_ws/src/redrhex_lowlevel_bridge/test ros2_ws/src/redrhex_rl_controller/test
node --check tools/training_panel/static/app.js
node --check tools/training_panel/remote_web/remote_app.js
node --test tools/training_panel/remote_web/*.test.mjs
```

The four excluded sim-to-real modules require their heavier project runtime and remain required when their contracts or consumers change. A passing lightweight suite does not replace those targeted tests.

<a id="ui-and-launchers"></a>
## Browser and desktop launchers

Run the complete browser suite whenever Mother or Child markup, styles, navigation, roles, or actions change. Install the repository's Playwright browser/runtime prerequisites first.

```bash
python -m pytest -q tools/training_panel/ui_tests
bash tools/macos/tests/test_redrhex_remote.sh
pwsh -NoProfile -File tools/windows/tests/test_redrhex_remote.ps1
```

The PowerShell source test requires `pwsh`; also run the active launcher plans' interactive smoke checklists on supported Windows and macOS hosts. Source and mocked browser tests do not prove first-launch security, SSH authentication, tunnel lifetime, or target-workstation behavior.

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

For publication changes, install `docs/requirements-site.txt`, stage the canonical sources to an empty temporary directory, and run the strict bilingual MkDocs build through `mkdocs.yml`. Before commit, use `validate --staged`; in a PR, include exact `Docs impact` and `Docs reason` fields. Semantic freshness remains a review responsibility.
