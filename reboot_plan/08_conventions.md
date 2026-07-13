# 08 — Reboot Conventions

## Branch and worktree

- Integration branch: `reboot/core-sim-first`.
- Behavioral source: `fix/review-2026-07@5cdc824`.
- Isolated worktree: `/home/lab_user1/Desktop/RedRHex-core-sim-first`.
- Do not use the reboot branch for panel, remote, reward-agent, ROS, or unrelated
  research work.

## Commit scope

Use one purpose per commit, for example:

```text
test: establish safe root test discovery
chore: lock simulator provenance
test(sim): add canonical gravity probe
feat(contract): define observation layout
refactor(core): extract projected-gravity math
refactor(adapter): adopt core observation assembly
docs: record P1 gravity evidence
```

Do not combine a diagnostic with its physics fix, a package extraction with a semantic
change, or generated/raw evidence with source code unless the artifact is a deliberately
tracked compact fixture.

## Gate records

Live state belongs in `STATUS.md`. Every completed task records:

- **Gate:** exact pass/fail criterion;
- **Command:** exact command and environment;
- **Evidence:** tracked summary plus raw artifact/hash location;
- **Commit/ADR:** immutable source/decision reference.

Narrative progress without these fields does not advance a gate.

## Baselines and tags

- `v0-validated-pre-reboot` is created only after P1 passes.
- Baseline IDs and fixtures are immutable. A changed validated state, schema, physics
  model, or accepted tolerance creates a new ID.
- A baseline update is never the first response to a parity failure.
- The accepted reboot receives a separate final tag after P7.

## Code ownership

- Stable interface facts: `source/redrhex_contract/**`.
- Pure Torch behavior: `source/redrhex_core/**`.
- Isaac integration and legacy compatibility: `source/RedRhex/**`.
- Consumer source remains frozen under the paths recorded in
  `evidence/interface_freeze.md`.

Prefer explicit typed records, pure functions, immutable contract data, and visible
state/RNG/time flow. Avoid new abstractions until a real extracted seam needs them.

## Documentation

- Architecture decisions: `docs/adr/` or the existing design checkpoint.
- Gate state: `reboot_plan/STATUS.md`.
- Interface evidence: `reboot_plan/evidence/interface_freeze.md`.
- Simulation evidence: `reboot_plan/evidence/sim_validation/`.
- Baseline/acceptance evidence: their corresponding `evidence/` folders.

Paths in commands must come from repository/environment variables; do not introduce
machine-specific `/home/<user>` source paths. The documented Desktop worktree path is
operational context, not a runtime dependency.
