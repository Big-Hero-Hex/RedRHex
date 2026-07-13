# RedRHex Core-First Soft Reboot

This folder is the operating record for rebuilding the RedRHex contract,
simulation, and training core while preserving the working research system around it.

The reboot branch is `reboot/core-sim-first`, rooted at the behavioral source snapshot
`fix/review-2026-07@5cdc824`. Its first commit, `a795c06`, preserves the original reboot
draft. The active worktree is `/home/lab_user1/Desktop/RedRHex-core-sim-first`.

## Hard boundary

During the core reboot, do not modify:

- `tools/training_panel/**`, including the local panel, remote worker/web, and Supabase;
- `tools/reward_agent/**`;
- `ros2_ws/**`, including topics, messages, nodes, configuration, and deploy contract.

These systems remain usable and are exercised by read-only regression tests. They are
pulled back into the active development path only after the core passes acceptance.

## Authoritative reading order

| Document | Purpose |
|---|---|
| [STATUS.md](STATUS.md) | Current gate, evidence, and what is blocked |
| [core-first design](../docs/superpowers/specs/2026-07-13-core-sim-first-soft-reboot-design.md) | Approved architecture, boundaries, data flow, and failure policy |
| [00_overview.md](00_overview.md) | Scope, principles, and definition of done |
| [01_current_state_audit.md](01_current_state_audit.md) | Source snapshot and keep/freeze/extract/defer inventory |
| [02_target_architecture.md](02_target_architecture.md) | Current and target structure graphs and dependency law |
| [03_migration_plan.md](03_migration_plan.md) | P0–P7 gates and task order |
| [05_testing_and_ci.md](05_testing_and_ci.md) | Test tiers, artifacts, and verification gates |
| [09_sim_validation.md](09_sim_validation.md) | Mandatory pre-baseline gravity/frame diagnostic ladder |
| [evidence/](evidence/) | Interface, simulation, baseline, and acceptance evidence |

`04_ai_workflow.md`, `06_experiment_management.md`, `07_roadmap.md`, and
`08_conventions.md` provide workflow/roadmap policy. If any old wording conflicts with
the design, `STATUS.md`, or P0–P7 order, the core-first design is authoritative.

## The short version

1. Freeze external interfaces and make test discovery/provenance trustworthy.
2. Validate the legacy simulator: world gravity, units, frames, masses/inertias,
   airborne motion, contacts, projected gravity, timing, and reward axes.
3. Only after that gate passes, tag and capture the golden/reference oracle.
4. Scaffold sibling `redrhex_contract` and `redrhex_core` packages.
5. Extract stable facts and pure-Torch behavior slice by slice behind `RedRhex`, which
   remains the Isaac Lab adapter and retains both existing Gym IDs.
6. Run acceptance, then separately decide when to unfreeze panel/remote/reward/ROS work.

## Current status

The branch is at P0. Maintained consumer suites pass, but repository-wide test discovery
and simulator provenance are not yet trustworthy. The baseline is therefore
**not frozen**, and no claim has been made that the current gravity behavior is correct
or incorrect. See [STATUS.md](STATUS.md) for exact evidence.
