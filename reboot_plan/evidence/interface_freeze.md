# P0 Interface Freeze Evidence

Status: **drafted; guard implementation pending design review**

Behavioral source: `fix/review-2026-07@5cdc824`

Target branch: `reboot/core-sim-first`

## Frozen source trees

These consumers remain usable but read-only throughout the core reboot. Their Git tree
IDs provide a cheap exact guard against accidental edits.

| Surface | Frozen path | Source tree ID |
|---|---|---|
| Working/training panel, including remote worker/web/Supabase | `tools/training_panel/**` | `569814ac11b831b615e3a15e82f2d650853bdc42` |
| Reward agent | `tools/reward_agent/**` | `7556e5f9e0e53f83674760c9dad32a06f407ff50` |
| ROS2 interfaces and implementation | `ros2_ws/**` | `e61c5212346e97023fb37ac958620ba5128ff81f` |

Allowed interaction is limited to read-only inspection and regression tests. The
pre-existing ignored `tools/reward_agent/tests/test_evaluator.py` is evidence of the
ignore defect, but strict freeze means it is not added under the reward-agent tree.
P0 recreates equivalent black-box coverage under root `tests/frozen_consumers/`, importing
the frozen runtime only. Any change under a frozen path requires a separate, explicit
scope decision.

## Public behavior held stable

### Gym registration

Both existing IDs and their exact mappings remain valid:

| Task ID | Env cfg | PPO | Teacher | Distillation | SKRL |
|---|---|---|---|---|---|
| `Template-Redrhex-Direct-v0` | `RedrhexEnvCfg` | `PPORunnerCfg` | `PPORunnerPrivilegedTeacherCfg` | `RedrhexDistillationRunnerCfg` | `skrl_ppo_cfg.yaml` |
| `Template-Redrhex-ForwardFast-Direct-v0` | `RedrhexForwardFastEnvCfg` | `PPORunnerForwardFastCfg` | `PPORunnerForwardFastPrivilegedTeacherCfg` | `RedrhexForwardFastDistillationRunnerCfg` | `skrl_ppo_cfg.yaml` |

Registration blob at the source snapshot:
`844e36a3ec448e0df8b319a2d8ff86cebbf65d4c`.

No replacement task ID is introduced during the reboot.

### Training/play boundary

The existing `scripts/rsl_rl/train.py`, `play.py`, and evaluation entry points remain
the process boundary used by the panel and by manual commands. Preserve:

- task, environment count, iteration, seed, device, headless, resume, checkpoint, and
  video arguments;
- explicit `--panel_overrides` behavior and terrain replay inputs;
- checkpoint lookup/resume semantics and run/log/artifact layout;
- policy observation/action dimensions and ONNX input/output behavior.

Source-snapshot boundary blobs:

| File | Git blob |
|---|---|
| task registration | `844e36a3ec448e0df8b319a2d8ff86cebbf65d4c` |
| `scripts/rsl_rl/train.py` | `7586fa916d2e3832959fa36bd7594336edf963cc` |
| `scripts/rsl_rl/play.py` | `fca4b21b7938f70dcff9698393516da4236657a8` |
| `scripts/rsl_rl/eval_command_sweep.py` | `e8f72213bf539d21cdaf6f6bf06e0f411b6a09ad` |
| `scripts/rsl_rl/cli_args.py` | `c176f774515ceb49d4284cde723ebc5ef8be97ed` |

These hashes are provenance, not a ban on adapter-compatible internal changes. P0 must
capture executable `--help`/dry-run/default/artifact snapshots so later behavior can be
regression-tested rather than judged only by file equality.

Current defaults are distinct and both are frozen: the local panel/smoke path defaults
to 4 environments and 1 iteration; Remote Web defaults to 4 environments and 8
iterations. A future named smoke/reference command must not silently redefine either.

### ROS boundary

`ros2_ws/**` remains byte-for-byte frozen. `redrhex_contract` may model stable facts
such as joint ordering, shapes, units, rates, and scales, but it only compares against
ROS during this reboot. Generating or rewriting ROS files is post-reboot work.

## Baseline regression evidence

Commands run from the Desktop worktree on 2026-07-13:

```text
python -m pytest -q tools/reward_agent/tests tools/training_panel/tests
=> 282 passed, 41 subtests passed

node --test tools/training_panel/remote_web/*.test.mjs
=> 2 passed

python -m pytest -q tools/training_panel/ui_tests
=> 7 passed

python -m pytest --collect-only -q
=> 289 collected, 2 collection errors
```

The two collection errors are the root diagnostic scripts importing `isaaclab` at
module import time. This is a P0 defect, not a regression caused by the reboot branch.

The source checkout also contains an ignored
`tools/reward_agent/tests/test_evaluator.py` with two passing tests. Because the blanket
`tests/` ignore hides it, the isolated checkout has 282 rather than 284 maintained Python
tests. P0 preserves that ignored file as local defect evidence and recreates its two
black-box invariants outside the frozen tree. The expected inventory then becomes 291
Python tests plus 41 subtests: 282 frozen panel/reward tests, 2 external black-box reward
tests, and 7 panel UI tests. The two Node tests are a separate required tier.

## P0 exit conditions

- Root CPU test discovery is explicit, passes, and never imports Isaac diagnostics.
- Root `tests/` and intended nested test files are no longer hidden by `.gitignore`.
- A frozen-path guard proves the three tree IDs above remain unchanged and detects
  committed, staged, unstaged, and untracked files. Test bytecode/cache output is
  disabled or redirected outside frozen paths.
- Toolchain provenance records the repository/runtime root policy, external dirty-state
  policy, Isaac Lab SHA, Isaac Sim build, and Python/package versions.
- Every P1/P2 run manifest separately records device, seed/RNG state, exact command,
  resolved configuration, asset hashes, and toolchain-manifest ID.
- Existing smoke/reference interfaces are captured with non-GPU snapshots. P1 diagnostic
  arguments, refusal rules, manifest schema, and tracked-vs-local artifact locations are
  specified in P0; their executable dry-run is G0 after the CLI exists.
- The design checkpoint is approved before implementation commits begin.
