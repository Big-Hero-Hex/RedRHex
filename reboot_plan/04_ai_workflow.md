# 04 — Agent-Assisted Development Workflow

This workflow applies only to reboot-owned documentation, tests, diagnostics,
`redrhex_contract`, `redrhex_core`, and the `RedRHex` adapter. Panel, remote, reward-agent,
and ROS paths are frozen regardless of what an automated tool proposes.

## Three loops

### Loop A — Fast CPU edit loop

For contract/core work:

1. state one invariant and write the cheapest failing test;
2. make the smallest implementation change;
3. run the focused test, then the complete CPU tier;
4. inspect the diff for forbidden imports, hidden state, and scope creep;
5. commit one reversible slice with its evidence link.

### Loop B — Simulator diagnostic loop

For P1 and later adapter checks:

1. verify the runtime/provenance manifest;
2. run one diagnostic prerequisite only;
3. archive raw measurements and a fitted summary;
4. stop on failure and classify the layer before proposing a fix;
5. after an approved fix, rerun the failed check and invalidated downstream checks.

`validate-sim-gravity` always precedes `capture-golden` and `train-ref`.

### Loop C — Extraction loop

For each P4/P5/P6 seam:

1. identify exact legacy inputs, outputs, mutation, and ordering;
2. add unit and reduced-fixture tests;
3. implement a pure function/state object;
4. cut over only that adapter seam;
5. run CPU, frozen-boundary, full-local parity, and Isaac smoke gates;
6. review, document, and delete only the replaced legacy code.

## Guardrails

- A frozen-tree guard must fail on changes under `tools/training_panel/**`,
  `tools/reward_agent/**`, or `ros2_ws/**`.
- Core packages must fail dependency tests if they import Isaac, Gym, ROS, or tools.
- Never weaken a threshold, skip a failed test, or update a golden fixture simply to
  make a change pass. Threshold/baseline changes require an ADR and explicit approval.
- Never run hardware commands from the core reboot workflow.
- No destructive Git cleanup; preserve unrelated user work and use the dedicated
  Desktop worktree.
- Simulator evidence is invalid when runtime provenance is missing or unexpectedly dirty.

## Session start checklist

1. Read `reboot_plan/STATUS.md` and the current gate evidence.
2. Confirm branch/worktree and a scoped clean status.
3. Confirm no frozen-path diff against `fix/review-2026-07@5cdc824`.
4. Run the cheapest gate relevant to the proposed edit.
5. State assumptions, success criteria, and rollback before implementation.

## Session finish checklist

1. Run focused and required aggregate verification fresh.
2. Run `git diff --check` and inspect every changed path.
3. Confirm frozen tree IDs plus staged, unstaged, and untracked-path checks are clean;
   disable/redirect test bytecode and caches.
4. Link command output/artifacts in the appropriate evidence document.
5. Update `STATUS.md` only if a gate actually advanced.
6. Commit one purpose; do not mix diagnostics, generated evidence, and physics fixes.

## Human decisions

Explicit human approval is required for:

- correcting a diagnostic threshold after evidence shows the criterion itself was
  invalid (ADR + fresh rerun required); blocked/failed facts still block P2;
- changing gravity, frames, masses, inertia, collisions, damping, actuators, rewards,
  command timing, observation meaning, or reset semantics;
- replacing a baseline or approving an intentional golden difference;
- changing a frozen consumer or any hardware-facing behavior;
- accepting P7 when learning/evaluation results fall outside predeclared rules.

## Instruction hierarchy

Repository guidance should remain agent-neutral:

```text
root instructions             # universal safety, commands, frozen paths
source/redrhex_contract/       # stdlib-only dependency and contract rules
source/redrhex_core/           # Torch-only purity and explicit-state rules
source/RedRhex/                # Isaac adapter and compatibility rules
tests/sim/                     # GPU/runtime evidence rules
```

Progress lives in `STATUS.md`, not in prose claims or unchecked narrative milestones.
