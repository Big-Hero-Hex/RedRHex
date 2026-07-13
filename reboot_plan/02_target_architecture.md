# 02 — Current and Target Architecture

## Dependency law

```text
redrhex_contract -> Python standard library only
redrhex_core     -> Torch + redrhex_contract
RedRhex adapter  -> Isaac Lab + redrhex_core + redrhex_contract
frozen consumers -> existing scripts/Gym/checkpoint interfaces only
```

Reverse imports are prohibited. In particular, contract/core code must never import
Isaac, Gym, ROS, the panel, remote system, or reward agent.

## Current feature graph

```text
┌─────────────────────────────────────────────────────────────────────┐
│ Working consumers                                                   │
│                                                                     │
│ training panel ─┬─ local queue/history/video/deploy                  │
│                 └─ remote worker/web/Supabase                       │
│ reward agent ───── candidate/trial orchestration                    │
└─────────────────────────────┬───────────────────────────────────────┘
                              │ CLI args, global override files,
                              │ checkpoints/log/artifact conventions
                              v
                    train / play / evaluation scripts
                              │
                              v
                     Gym task registration (`RedRhex`)
                              │
                              v
┌─────────────────────────────────────────────────────────────────────┐
│ `redrhex_env.py` + `redrhex_env_cfg.py`                             │
│                                                                     │
│ simulator I/O ─ observations ─ rewards ─ gait/FSM/CPG              │
│ commands/resets ─ randomization ─ buffers/logging ─ frame math      │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              v
                       Isaac Lab / PhysX / USD

ROS2 deploy stack ── independently copied joint/rate/scale/frame facts ── policy
```

Main coupling consequences:

- importing `RedRhex` eagerly registers Isaac tasks, so CPU-only logic has no clean
  import boundary;
- math and state transitions reach directly into environment buffers;
- duplicated interface facts can drift;
- simulator diagnostics are mixed with default CPU test discovery;
- a legacy simulator can be preserved exactly even when its physics/frame model is wrong.

## Target feature graph

```text
┌─────────────────────────────────────────────────────────────────────┐
│ FROZEN, still usable                                                │
│ panel + remote system | reward agent | ROS2                         │
└───────────────┬─────────────────────────────┬───────────────────────┘
                │ existing CLI/Gym/artifacts  │ read-only parity
                v                             │
       ┌─────────────────────┐                │
       │ RedRhex             │                │
       │ Isaac adapter       │                │
       │                     │                │
       │ scene/reset/step    │                │
       │ tensor snapshots    │                │
       │ simulator writes    │                │
       └──────┬────────┬─────┘                │
              │        │                      │
              v        └──────────────────┐   │
┌──────────────────────┐                  v   v
│ redrhex_core         │──────────>┌──────────────────────┐
│ Torch only           │           │ redrhex_contract     │
│ observations/rewards │           │ stdlib-only facts    │
│ gait/commands/DR     │           │ order/shape/unit/rate│
│ kinematics/buffers   │           │ scale/slice/version  │
└──────────────────────┘           └──────────────────────┘
```

## Target repository layout

```text
source/
├── redrhex_contract/
│   ├── pyproject.toml
│   └── src/redrhex_contract/
│       ├── __init__.py
│       ├── model.py              # immutable facts and validation
│       └── layout.py             # named joint/obs/action ordering and slices
├── redrhex_core/
│   ├── pyproject.toml
│   └── src/redrhex_core/
│       ├── kinematics.py
│       ├── observations.py
│       ├── rewards.py
│       ├── actuation.py
│       ├── terminations.py
│       ├── gait.py
│       ├── commands.py
│       ├── domain_rand.py
│       └── buffers.py
└── RedRhex/                       # existing extension/package path retained
    └── RedRhex/tasks/direct/redrhex/
        ├── redrhex_env.py         # Isaac adapter/orchestrator
        ├── redrhex_env_cfg.py     # compatibility configuration during reboot
        └── agents/
tests/
├── contract/
├── core/
├── adapter/
├── sim/validation/
└── fixtures/reboot/<baseline-id>/
```

Names inside each package are provisional until the P3 implementation plan, but package
ownership and dependency direction are fixed by this design.

## Ownership by layer

| Concern | Owner | Notes |
|---|---|---|
| Joint/action/observation ordering, dimensions, units, rates, scales, slices, version | `redrhex_contract` | Stable interface facts only; no curriculum/experiment policy. |
| Quaternion/frame math, observation assembly, reward/termination terms, action decoding/targets, gait/command state, DR sampling, buffers | `redrhex_core` | Explicit tensor/state inputs and outputs; no environment object access. |
| Scene/articulation handles, reset/step hooks, applying DR, sensor reads, actuator writes | `RedRhex` | Isaac-specific adapter. |
| Legacy cfg aliases and task registration compatibility | `RedRhex` | Kept until a separately approved post-reboot cleanup. |
| Panel/remote/reward orchestration | Frozen existing tools | Exercise existing scripts only. |
| ROS messages/topics/safety/deployment | Frozen `ros2_ws` | Read-only comparisons during P4/P7. |

## Core API shape

Core calls use explicit records rather than passing the env:

```text
snapshot(sim state, commands, timers, RNG)
  -> core function(snapshot, config, contract)
  -> result(tensors, component map, next pure state, diagnostics)
```

Required properties:

- batched first dimension and stable dtype/device preservation;
- no global mutable state or hidden RNG;
- no simulator writes;
- shapes validated at adapter boundaries;
- reward components returned separately from the total;
- termination causes returned separately from the combined done/reset masks;
- action decoding computes one intent per control step; the adapter owns measured
  Isaac-specific target flushing across physics substeps;
- time expressed once in seconds/control-step `dt`, never inferred from call count;
- command/gait/reset transitions observable in golden fixtures.

## Frozen compatibility boundary

The following remain unchanged through P7:

- `Template-Redrhex-Direct-v0` and `Template-Redrhex-ForwardFast-Direct-v0`;
- their registered runner/distillation/SKRL entry-point keys;
- `scripts/rsl_rl/train.py`, `play.py`, and evaluation command behavior as seen by users
  and the panel;
- checkpoint loading/resume and policy I/O shape;
- panel-visible run naming, logs, video/ONNX/deploy artifact discovery;
- ROS source tree and external message/topic contract.

P4 may prove that a frozen ROS fact differs from the validated simulator. It records a
finding; it does not edit ROS during this reboot.

## Deliberately deferred architecture

Generated ROS contracts, layered experiment configuration, new Gym IDs, panel per-run
IPC, reward-agent store unification, PPO/script deduplication, and asset relocation may
all be useful later. They are excluded because they add a second behavior-changing axis
while the core is being validated and extracted.
