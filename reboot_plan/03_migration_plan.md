# 03 — Migration Plan: P0–P7

The legacy task stays runnable throughout. Only one implementation slice may be in
flight. A task is complete only when its gate, evidence, and commit/ADR are recorded.

## P0 — Foundation and interface freeze

Purpose: make the evidence system trustworthy before asking the simulator a physics
question.

| Status | Task | Gate | Evidence | Commit/ADR |
|---|---|---|---|---|
| [x] | Create `reboot/core-sim-first`, preserve the original plan, and attach an isolated Desktop worktree | Historical creation facts: branch/worktree registered from planning commit; original checkout returned to `fix/review-2026-07` | `git worktree list`, `STATUS.md` | `a795c06` |
| [ ] | Record and seal frozen consumer trees and current interface inventory | Source trees plus executable task/CLI/default/artifact boundaries captured; ignored reward-test invariants recreated outside frozen paths | `evidence/interface_freeze.md` | — |
| [ ] | Establish safe root CPU test discovery | Intended tests are tracked; default collection excludes simulator-launch diagnostics; suite passes | `evidence/foundation/RESULTS.md` | — |
| [ ] | Add a frozen-path guard | Committed, staged, unstaged, or untracked changes under panel/remote/reward/ROS fail; caches write elsewhere | deliberate-change failure tests | — |
| [ ] | Lock simulator provenance | Toolchain checker verifies source SHA, clean/pinned external state, Isaac Sim build, Python/packages, and runtime roots | `evidence/provenance/TOOLCHAIN.md` | — |
| [ ] | Define canonical command and artifact schemas | Existing smoke/reference interfaces have path-independent snapshots; future P1 diagnostic arguments, refusal rules, manifest schema, and tracked-vs-local locations are specified without pretending the CLI exists | `evidence/foundation/RESULTS.md` | — |

First implementation slice after design approval:

1. remove the blanket `tests/` ignore while keeping runtime outputs ignored;
2. preserve the ignored reward evaluator as defect evidence and recreate its two
   black-box invariants under `tests/frozen_consumers/` without touching the frozen tree;
3. add root pytest configuration with explicit CPU test paths and registered `fast`,
   `golden`, and `isaac` markers;
4. prevent `test_joint_velocity*.py` from entering default CPU collection without
   changing their diagnostic behavior yet;
5. verify collection and the complete CPU suite.

P0 implementation order is strict: safe test discovery → clean/pinned toolchain
provenance → executable task/CLI/default/artifact snapshots plus frozen guard → command
and artifact schemas. Do not combine these with physics/config/asset changes or
lint/pre-commit modernization.

**P0 exit:** CPU/UI/Node tests are complete and trustworthy, frozen paths have exact
guards, simulator/toolchain provenance is reproducible, existing smoke/reference
interfaces are snapshotted, and the future diagnostic command/manifest/artifact schema
is fixed.

## P1 — Validate legacy simulation, gravity, and frames

Purpose: decide whether the current simulator is fit to become an oracle.

| Status | Diagnostic | Gate | Evidence | Commit/ADR |
|---|---|---|---|---|
| [ ] | G0 implement and non-GPU dry-run the diagnostic orchestrator, then write the resolved run manifest with DR/noise/pushes disabled | CLI arguments/refusal/output schema valid; no hidden inputs/overrides | dry-run + manifest/config | — |
| [ ] | G1 stage/world audit: up axis, units, PhysicsScene, raw/converted gravity vector | One active scene; documented SI scale/Z-up; effective `(0,0,-9.81)` within tolerance | stage/asset JSON | — |
| [ ] | G2 two isolated canonical bodies in free fall | vertical acceleration within 0.5%; horizontal near zero; mass-independent | raw/fitted CSV+JSON | — |
| [ ] | G3 asset semantic forward/left/up and spawn/root transforms | dimensions/axes/reset pose match sourced intent | axis report + screenshot | — |
| [ ] | G4 live articulation mass/density provenance/COM/inertia audit | sourced totals/tensors and scale-aware inertia plausible; unknown facts remain BLOCKED | mass/inertia CSV | — |
| [ ] | G5 robot whole-COM free fall plus isolated linear/angular damping matrix | zero-damping acceleration and sourced damping responses correct | traces/plots | — |
| [ ] | G6a ideal-constrained contact/settle, then G6b configured-actuator rest-hold after G8 | both isolate and pass penetration/support/settle bounds; actuator weakness cannot masquerade as contact | contact/force traces + overlay | — |
| [ ] | G7 full-vector projected-gravity and velocity/angular frame probes | analytic poses and settled rest match documented policy frame | frame CSV/JSON | — |
| [ ] | G8 action decoding and actuator response/limits | both action families match order/scales/limits and sourced response bands | actuator traces | — |
| [ ] | G9 task-specific rewards/terminations/command domains | both frozen tasks match numeric component/cause expectations | component table | — |
| [ ] | G10 expected step-index contract, timing/write cadence, determinism characterization + fresh holdouts | rates/timers/indexing correct; hashed envelope frozen before three holdouts and all hold | timing/repeatability report | — |
| [ ] | C1 frozen ROS compatibility finding | `MATCH`, `MISMATCH`, or `BLOCKED_ON_IMU_GROUND_TRUTH` recorded read-only | compatibility report | — |

Follow the prerequisite graph in 09: stop failed descendants, but continue independent
branches. Fix a confirmed blocker one at a time, record an ADR, then rerun the affected
check and invalidated descendants. Missing physical facts are BLOCKED, not PASS.

**P1 exit:** every G0–G10 check is `PASS`; any `FAIL` or `BLOCKED` prevents baseline
tagging/capture. C1 must be recorded but may be `MISMATCH` or blocked on IMU ground truth
without forcing changes to the validated simulator or frozen ROS. No baseline tag exists
before this exit.

## P2 — Capture the validated legacy baseline

| Status | Task | Gate | Evidence | Commit/ADR |
|---|---|---|---|---|
| [ ] | Tag the exact validated behavioral state `v0-validated-pre-reboot` | Tag resolves to the commit used by all captures | tag + manifest | — |
| [ ] | Capture full local seam-level rollout | Re-run matches declared deterministic envelope | full manifest under `baselines/reboot/<id>/` | — |
| [ ] | Create reduced committed fixture | CPU replay loads and validates hashes | `tests/fixtures/reboot/<id>/` | — |
| [ ] | Archive compatible checkpoints/evaluation | Play/eval command succeeds on the tagged state | baseline evidence | — |
| [ ] | Run fixed-seed reference-training protocol | All seeds complete; metrics/export rules predeclared | TB/metrics manifests | — |

Before capture, inventory the current legacy seams and version a raw-capture schema using
facts available in the legacy env. The raw rollout includes simulator state, actions,
observations, per-term and total rewards, termination causes/dones, gait/command state,
resets, config, seeds/RNG state, current boundary facts, and source/asset hashes. P4/P5
may add immutable *derived* fixtures computed from that raw capture, each hashing its P2
source; they do not silently recapture or overwrite P2. Full tensors/checkpoints remain
local/ignored; reduced fixtures and manifests are tracked.

**P2 exit:** the legacy oracle is immutable, reproducible, and available at both CI and
full-local scales.

## P3 — Scaffold sibling packages and tests

| Status | Task | Gate | Evidence | Commit/ADR |
|---|---|---|---|---|
| [ ] | Add installable `redrhex_contract` skeleton | Imports without Torch/Isaac/ROS/Gym/tools | dependency test | — |
| [ ] | Add installable `redrhex_core` skeleton | Imports with Torch on CPU; no forbidden imports | dependency test | — |
| [ ] | Make packaging discover all intended packages | Editable install/import checks pass in base and Isaac environments | install log | — |
| [ ] | Add contract/core/adapter/sim test tiers and commands | CPU tiers do not initialize Isaac; Isaac tier is explicit | collection logs | — |
| [ ] | Verify legacy task unchanged | Existing IDs register and tracked smoke command runs | adapter smoke | — |

**P3 exit:** package boundaries are real and verified before behavior moves into them.

## P4 — Extract stable contract facts

Move one fact group at a time: joint/action ordering, observation layout/slices,
dimensions, units, control/simulation rates, scales/limits, then versioning.

For every group:

1. write a failing hand-computable contract test;
2. implement the minimal stdlib representation and validation;
3. compare it read-only with legacy cfg and frozen ROS facts;
4. adapt `RedRhex` without changing public behavior;
5. run CPU contract/golden, frozen-boundary, and Isaac smoke gates.

Differences become findings. ROS remains frozen; curriculum and experiment policy stay
outside the contract.

**P4 exit:** stable facts have one core-reboot owner, all legacy/frozen comparisons are
explicit, and no frozen consumer file changed.

## P5 — Extract pure-Torch behavior

Recommended dependency order:

1. kinematics/quaternion/frame helpers;
2. observation layout, assembly, normalization, and noise;
3. reward components and total decomposition;
4. termination causes and reset-mask composition;
5. action decoding/gating and actuator-intent calculation;
6. gait/CPG/FSM state;
7. command sampling/state transitions;
8. domain-randomization sampling;
9. buffers/history bookkeeping.

Per slice: failing unit test, pure implementation, reduced-fixture parity, adapter
cutover, full local parity, Isaac smoke, review, then delete only the replaced legacy
code. Logging normalization, command-timing shifts, reward-intent changes, and physics
changes are not extraction work and remain deferred.

**P5 exit:** planned math/state behavior is CPU-testable, explicit, and golden-equivalent.

## P6 — Thin Isaac adapter cutover

- Finish converting the environment to snapshots/core calls/single simulator writes.
- Keep cfg/task/agent/checkpoint compatibility in the adapter.
- Verify both existing task IDs, train/play/eval, policy export, panel command dry-runs,
  artifact discovery, and read-only ROS parity.
- Remove dead legacy paths only with direct replacement evidence.

**P6 exit:** `RedRHex` is an Isaac adapter rather than the owner of core math, and every
frozen consumer still works through its existing boundary.

## P7 — Acceptance and handoff

- Run all CPU tiers, frozen consumer regressions, full golden replay, both task smokes,
  P1 simulator invariants, export/deploy read-only checks, and fixed-seed reference
  training/evaluation.
- Compare against predeclared P2 rules; do not tune thresholds after observing results.
- Write `evidence/acceptance/<run-id>.md`, record remaining risks, and tag the accepted
  core reboot.

**P7 exit:** reproducible acceptance evidence exists and no required work remains.

## Post-reboot decisions

Only after P7: decide separately whether to unfreeze panel/remote/reward/ROS work and
whether to pursue config redesign, asset moves, contact sensors, physics/reward changes,
hardware estimator work, a new task ID, or other research features.

## Standing rules

1. Update `STATUS.md` and evidence links in the same commit as a completed gate.
2. Do not check a task without its command, artifact, and commit/ADR.
3. Do not use a baseline created before P1.
4. Do not mix structural extraction with an intended behavior change.
5. Do not modify frozen paths without an explicit scope change.
6. Keep every implementation slice independently revertible.
