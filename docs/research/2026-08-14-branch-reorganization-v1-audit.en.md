---
id: branch-reorganization-v1-audit
title: Branch Reorganization V1 Completion Audit
lang: en
audience: developer
type: audit
status: published
owner: project
last_reviewed: 2026-08-14
---

<a id="scope"></a>
## Scope

This audit closes Branch Reorganization V1 through the protected-`main` cutover, feature recovery, remote-ref contraction, merged-worktree cleanup, and final inspection checkout on 2026-08-14. It excludes physical spring calibration, workstation smoke for the remote launchers, Student production evidence, the draft Panel physics/calibration proposal, and the separate reboot and sim2real worktrees.

<a id="method"></a>
## Method

GitHub API responses were inspected for effective administrator permission and stored `main` protection. PR state and checks were read from GitHub. Every remote deletion candidate was compared with the exact SHA in the [preservation manifest](../governance/branch-preservation-manifest.en.md) immediately before deletion. Local worktrees were removed only after `git status --porcelain` was clean; local branches were deleted only when `git branch --merged origin/main` proved reachability. Recovery bundles were verified with `git bundle verify` and SHA-256.

<a id="findings"></a>
## Findings

- PR #7 merged at protected `main` commit `5a6037b434ab6ba2ca54ba1474f5ff556a790115`. Protection requires strict `validate`, pull-request workflow, and resolved conversations; administrator enforcement is enabled, and force pushes and branch deletion are disabled.
- Torsion PR #3, Training Panel V3.6 PR #4, Windows launcher PR #5, macOS launcher PR #10, and Student V2 core PR #11 are reachable from `main`. Their documented hardware, workstation, and production-evidence limitations remain in force.
- Twenty-one legacy remote source refs, the merged macOS and Student topic refs, and the merged PR #7 source ref were deleted after exact-tip comparison. The remote now contains only `main`, draft PR #12, `proposal/core-sim-first`, and 16 exact `archive/*` refs.
- Thirteen clean merged or archived secondary worktrees and nineteen Git-proven merged local branches were removed. The remaining three worktrees are the root inspection checkout, the explicit reboot proposal, and the explicit sim2real recovery worktree.
- Draft PR #12 is green at `40b841b00f3749e36bc26de9207f42e3c5ca3d31` and remains outside `main`. It is a simulation-only Panel physics/calibration and Student-browser proposal, not calibrated hardware behavior.
- Four root recovery generations remain local-only. The latest focused path repair is commit `1a436d26a167ce8431a916a4fef78534259bde11`; its complete-history bundle has SHA-256 `6b8abcb8ed4d7e6d9c4d03c0559e50046e2cab9da969dccdd901e0415fd515b8`. Its focused test passes, but the inherited F0/F1 recovery line is not a merge candidate.

<a id="actions"></a>
## Actions

- [x] Enable and verify protected `main`.
- [x] Merge the branch-governance checkpoint.
- [x] Delete exact-SHA-verified obsolete remote source refs.
- [x] Remove clean merged/archived secondary worktrees and Git-proven merged local branches.
- [x] Move the root checkout to the green Panel physics/calibration WIP based on final `main`.
- [ ] Review the local-only Student F0/F1 recovery as a separate project before any publication or merge.
- [ ] Keep PR #12 draft until its simulation and physical evidence gates are explicitly reviewed.

<a id="evidence"></a>
## Evidence

Durable controls and recovery evidence are recorded in [ADR-0002](../decisions/adr-0002-single-protected-main.en.md), [branch management](../governance/branch-management.en.md), and the [preservation manifest](../governance/branch-preservation-manifest.en.md). GitHub retains the merged [governance PR #7](https://github.com/Big-Hero-Hex/RedRHex/pull/7) and the green draft [Panel WIP PR #12](https://github.com/Big-Hero-Hex/RedRHex/pull/12).

<a id="follow-up"></a>
## Follow-up

Branch Reorganization V1 is closed. Review this result only if `main` protection changes, removal of an `archive/*` ref is proposed, a retained recovery line is promoted, or PR #12 reaches a new evidence-backed lifecycle state.
