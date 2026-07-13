# Core-First, Simulation-First Soft Reboot Design

## Context

RedRHex has valuable trained behavior, a working operations panel, remote workflows, a
reward agent, and a safety-oriented ROS2 deployment stack. The research core is hard to
debug because simulation orchestration, reward/observation math, gait state, command
mutation, randomization, and logging are coupled inside a large Isaac-dependent
environment.

The reboot begins from `fix/review-2026-07@5cdc824` on the dedicated branch
`reboot/core-sim-first`. The planning baseline is commit `a795c06`. No production code
has been changed on the reboot branch at this design checkpoint.

The approved direction is a soft reboot: preserve the working behavior and external
interfaces, validate the legacy simulator before trusting it as an oracle, then extract
small dependency-controlled packages behind the existing Isaac Lab task.

## Decision summary

1. Freeze the working/training panel, remote system, reward agent, and all ROS2 source
   and interfaces during the core reboot.
2. Validate gravity, frames, masses, contacts, timing, and determinism before creating
   any golden rollout or reference-training baseline.
3. Create sibling installable packages `redrhex_contract` and `redrhex_core`.
4. Retain `RedRhex` as the Isaac Lab adapter and keep its current Gym IDs, script entry
   points, checkpoint behavior, and artifact layout.
5. Extract behavior one slice at a time. Do not combine extraction with physics,
   reward-intent, timing, logging-semantic, config-redesign, panel, remote, or ROS work.
6. Reintroduce frozen consumers only through regression checks during the reboot. Any
   later changes to them require separate approval after the core passes acceptance.

## Goals

- Make contract facts importable without Isaac, ROS, Gym, or UI dependencies.
- Make reward, observation, gait, command, randomization, and kinematic math testable
  with Torch on CPU.
- Make simulator behavior measurable before attempting to fix the reported “weird”
  gravity feeling.
- Preserve the existing task IDs and operational workflow while replacing internals.
- Produce a documented evidence chain for every gate and every accepted behavior change.

## Non-goals

- No rewrite of `tools/training_panel/**`, its remote worker/web/Supabase system, or
  `tools/reward_agent/**`.
- No edits to `ros2_ws/**`, generated ROS contract, topics, messages, or nodes.
- No new Gym task ID, CLI redesign, experiment-store migration, or artifact-layout change.
- No broad config redesign, asset relocation, repository cleanup, reward redesign,
  contact-sensor feature, hardware estimator, or research feature during extraction.
- No physics fix based only on visual intuition. A failed diagnostic must localize the
  cause before a change is proposed.

## Current structure and coupling

```text
working panel / remote system / reward agent
                  |
                  | builds CLI args and override files
                  v
       train.py / play.py / evaluation scripts
                  |
                  v
        RedRhex Isaac extension and Gym registry
                  |
                  v
  redrhex_env.py + redrhex_env_cfg.py monolith
     | observations, rewards, gait, commands,
     | randomization, buffers, simulator handles
     v
         Isaac Lab / PhysX / robot USD

ROS2 deploy stack ------------------------------+
  independently mirrors joint/rate/scale/obs facts
  and therefore can drift from the simulator      |
                                                  v
                                    policy/checkpoint contract
```

The important failure mode is not merely file size. Pure math cannot be imported
without simulator registration, cross-boundary facts are copied, observation calls can
mutate state, and the default test collector mistakes simulator diagnostics for CPU
tests.

## Target structure and dependency graph

```text
frozen panel / remote / reward agent
              |
       existing CLI/task/checkpoint interface
              v
        RedRhex (Isaac adapter)
          |                 |
          v                 v
 redrhex_core --------> redrhex_contract
 (Torch only)          (stdlib data; no Isaac/ROS)
          |
          v
  typed tensors/results returned to the adapter

frozen ros2_ws ---- read-only compatibility tests ----> redrhex_contract
```

Proposed layout:

```text
source/
├── redrhex_contract/
│   ├── pyproject.toml
│   └── src/redrhex_contract/
├── redrhex_core/
│   ├── pyproject.toml
│   └── src/redrhex_core/
└── RedRhex/                       # existing Isaac extension, retained as adapter
    ├── setup.py
    └── RedRhex/tasks/direct/redrhex/
tests/
├── contract/
├── core/
├── adapter/
├── sim/validation/
└── fixtures/reboot/<baseline-id>/ # small committed parity fixture
```

Dependency law:

- `redrhex_contract`: Python standard library only. It owns stable ordering, dimensions,
  units, rates, scales, named slices, and a contract version. It does not own curriculum
  stages, experiment policy, ROS types, or simulator objects.
- `redrhex_core`: may depend on Torch and `redrhex_contract`; it must not import Isaac,
  Gym, ROS, the panel, remote code, or reward-agent code.
- `RedRhex`: may depend on both sibling packages and Isaac Lab. It owns scene handles,
  configuration compatibility, reset/step hooks, and conversion between simulator
  tensors and core inputs/results.
- Frozen consumers continue to call the existing scripts and task IDs. They do not
  import the new core packages directly during the reboot.

## Runtime data flow

At each control step, the adapter reads simulator state once and builds an explicit
input record. Pure core functions consume that record plus contract/config values and
return explicit results. The adapter alone writes actuator targets and simulator state.

```text
PhysX state
   -> RedRHex adapter snapshot
   -> redrhex_core state/gait/observation/reward/termination/actuation functions
   -> typed observation, reward/termination components, next pure state, actuator intent
   -> RedRHex adapter validates shapes/ranges
   -> one intent computation per control step; adapter flushes cached targets at the
      Isaac-specific substep cadence measured by P1
```

No core function may reach back into an environment object. Random generators/state,
time step, command state, reset masks, and any history needed for deterministic replay
must be explicit inputs or outputs.

## Migration gates

### P0 — Foundation and interface freeze

- Repair root test discovery: remove the blanket root-test ignore, track intended tests,
  register markers, and ensure CPU collection excludes simulator-launch scripts.
- Preserve the already-existing ignored reward evaluator as defect evidence and recreate
  its two invariants as black-box root tests; do not modify the frozen reward-agent tree.
- Record simulator/toolchain provenance. The current Isaac Lab checkout is at
  `v2.3.2` / `37ddf626…` but has a local Flatdict change; use a clean dedicated checkout
  or pin and verify that patch explicitly before collecting evidence.
- Record the source commit and exact Git tree IDs of every frozen consumer; the guard
  also rejects staged, unstaged, and untracked files under those paths and redirects
  bytecode/test caches elsewhere.
- Define environment-variable-based command and artifact schemas without changing local
  panel defaults (4 environments / 1 iteration) or Remote Web defaults (4 / 8). The P1
  diagnostic CLI is specified here but implemented/non-GPU-dry-run at G0.
- Capture executable snapshots/regressions for both Gym registrations and entry-point
  mappings, CLI parser/default behavior, policy I/O dimensions, checkpoint loading, and
  panel-visible artifact discovery.

P0 order is safe tests → toolchain provenance → executable interface snapshots/frozen
guard → command/artifact schemas. P0 blocks P1 until those foundations are trustworthy.

### P1 — Validate the legacy simulation and gravity/frame model

Run diagnostics by the prerequisite graph in `reboot_plan/09_sim_validation.md`; a
failure blocks descendants but not independent branches:

1. G0 implementation/non-GPU dry-run of the diagnostic orchestrator plus provenance and
   resolved configuration, with one environment and DR/noise/pushes disabled.
2. G1 composed-stage units/up-axis/PhysicsScene/effective gravity audit.
3. G2 isolated canonical-body free fall, independent of the robot asset.
4. G3 asset root, semantic forward/left/up axes, spawn transform, and reset quaternion.
5. G4 live articulation mass/density provenance, center-of-mass, and inertia audit.
6. G5 robot whole-COM free fall plus isolated linear/angular damping probes.
7. G6 physical contact/drop/settle behavior without policy or gait proxies.
8. G7 full-vector projected-gravity and linear/angular velocity frame tests.
9. G8 action decoding and actuator response/limit tests.
10. G9 task-specific reward components, termination causes, and command-domain tests.
11. G10 expected step-index contract, actual timing/state freshness/substep write cadence,
    and same-seed characterization followed by fresh-process holdout repeats.
12. C1 frozen ROS comparison reported separately as `MATCH`, `MISMATCH`, or blocked on
    IMU ground truth; it does not force a simulator or ROS edit.

The configured world vector appears to be `(0, 0, -9.81)`, so it is not yet evidence of
a bug. A higher-risk hypothesis is the configured +90° X spawn quaternion. Under the
current convention, neutral projected gravity is expected near `(0, -1, 0)`, while code
may interpret root Y/Z components as semantic lateral/up axes. This is a hypothesis to
measure, not permission to rotate the asset or change rewards.

Mandatory blockers include wrong gravity magnitude/direction, inconsistent SI scale,
unexplained semantic axes, invalid or physically ungrounded mass/inertia/actuators,
airborne acceleration inconsistent with the canonical probe, unstable contacts,
projected-gravity/reward/command-axis errors, or unexplained timing/nondeterminism. Every
G0–G10 check must pass; missing ground truth stays `BLOCKED`. C1 is compatibility evidence
and may remain a recorded mismatch because ROS is frozen.

### P2 — Capture the validated legacy oracle

Only after P1 passes:

- tag the validated behavioral commit (`v0-validated-pre-reboot`);
- inventory/version the legacy raw-capture seams, then capture full local rollouts with
  simulator state, actions, observations, reward and termination components/totals,
  dones, gait/command state, resets, configuration, seed/RNG state, current boundary
  facts, and manifest hashes; later derived fixtures hash this immutable raw source;
- commit a small reduced fixture for CPU CI and keep full tensors/checkpoints/TensorBoard
  outputs under ignored local artifact paths;
- run a named, immutable reference-training protocol with fixed seeds and manifests.

### P3 — Scaffold packages and test harness

Create both sibling packages with minimal imports, dependency-prohibition tests, and
adapter import/smoke checks. Fix the current `setup.py` limitation (`packages=["RedRhex"]`)
without making a CPU import eagerly register Isaac tasks.

### P4 — Extract `redrhex_contract`

Move stable facts only. Initially compare against the existing cfg and frozen ROS files
read-only. Do not generate or modify ROS source. The adapter remains the compatibility
mapping for legacy cfg names and task registration.

### P5 — Extract `redrhex_core`

Use test-first, behavior-preserving slices. Suggested order is kinematics, observation
layout/assembly, reward decomposition, terminations, action decoding/intent, gait state,
commands, randomization, and buffers.
Each slice must pass hand-computable unit tests and the P2 fixture before the adapter
switches to it. Intentional semantic changes are deferred, not bundled into extraction.

### P6 — Thin adapter cutover

Switch one seam at a time while retaining both existing Gym registrations, script
arguments, checkpoint loading, and panel-visible run/artifact behavior. Delete legacy
code only in the same commit that proves the replacement path.

### P7 — Acceptance

Run CPU suites, frozen-boundary regressions, full local golden replay, simulator smoke,
P1 invariant diagnostics, and the fixed-seed reference protocol. Compare learning and
evaluation metrics with predeclared tolerance/statistical rules. Produce an acceptance
report and tag the accepted reboot state.

## Test and artifact policy

| Evidence | Tracked? | Location |
|---|---|---|
| Source/config/manifests/results summaries | Yes | `reboot_plan/evidence/**` |
| Small deterministic parity tensors | Yes | `tests/fixtures/reboot/<baseline-id>/` |
| Full rollout tensors/checkpoints/reference runs | No | `baselines/reboot/<baseline-id>/` |
| Raw diagnostic logs/CSV/plots/screenshots | No by default | `artifacts/reboot/<run-id>/` |

Every evidence manifest records the source commit, dirty-state policy, asset hash,
resolved config hash, Isaac Lab/Sim versions, Python/packages, GPU/device, seed, command,
and artifact hashes. A baseline is immutable; changed physics or thresholds create a
new baseline ID.

## Failure handling and rollback

- Stop a diagnostic chain at the first failed prerequisite and label the likely layer:
  world/stage, robot asset/forces, contacts, frames, then reward/contract.
- Do not widen a tolerance after seeing a failure without an approved ADR and rerun.
- Do not use a zero-action policy as a gravity probe; implicit actuators and environment
  control can still apply forces.
- Every extraction commit stays independently revertible. The legacy call path remains
  available until the new path passes its gate.
- A frozen-path diff fails the reboot gate. Urgent consumer fixes occur on a separate
  explicitly scoped commit/branch and are then deliberately rebased into the source
  snapshot if accepted.

## Documentation and progress

- Live gate state: `reboot_plan/STATUS.md`
- Frozen boundary evidence: `reboot_plan/evidence/interface_freeze.md`
- Simulation results: `reboot_plan/evidence/sim_validation/RESULTS.md`
- Baseline manifests: `reboot_plan/evidence/baselines/<baseline-id>.md`
- Acceptance reports: `reboot_plan/evidence/acceptance/<run-id>.md`

The next action after this design checkpoint is user review. After approval, write the
task-level implementation plan for P0; do not begin simulator or core code changes from
this design document alone.
