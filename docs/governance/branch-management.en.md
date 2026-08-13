---
id: branch-management
title: Branch and Worktree Management
lang: en
audience: developer
type: reference
status: active
owner: project
last_reviewed: 2026-08-14
---

<a id="branch-roles"></a>
## Branch roles

- `main` is the only permanent integration branch and must remain releasable and protected.
- `feature/*`, `fix/*`, `docs/*`, and `chore/*` are short-lived reviewed change branches.
- `proposal/*` preserves an unapproved long-lived design or evidence line and must not be described as current implementation.
- `archive/*` preserves historical or recovery tips. Archive branches are read-only recovery references, not merge targets or development bases.

<a id="change-flow"></a>
## Change flow

Start new work from current `origin/main`, keep one coherent concern per branch, and open a draft pull request early. A stacked branch may target another feature branch only when it has a real dependency; after the dependency merges, retarget or rebase the dependent pull request onto `main`. Delete merged or superseded change branches after their tip is reachable from `main` or a recorded archive ref.

A `*-wip` branch may remain temporarily when recovered code is useful for inspection but its evidence gates are incomplete. It must use a draft pull request, state the missing gates, and avoid claims that the branch is shipped or hardware-ready.

Every pull request follows the repository documentation-impact declaration. Direct pushes, force pushes, and deletion of `main` are prohibited after the documented 2026-08-13 cutover.

<a id="worktrees"></a>
## Worktrees

The repository root checks out clean `main`. Additional active or proposal branches use `.worktrees/<branch-slug>`. Before removing a worktree, its status must be clean or its exact tracked, untracked, ignored-but-required, and stashed changes must be committed to a recovery ref. Generated caches may be excluded from a clean feature branch only when the recovery record identifies them explicitly.

<a id="archive-contract"></a>
## Archive contract

The branch archive manifest records each old ref, exact tip SHA, classification, retained ref, final disposition, and verification result. Significant unmatched work receives an `archive/*` branch; tips already reachable or patch-equivalent to a retained ref may be deleted after the mapping is recorded. Archive contraction occurs only after remote refs, local bundles, reconstructed branches, and validation evidence agree.

<a id="recovery"></a>
## Recovery

To recover pre-cutover history, create a new branch from the recorded archive ref or SHA; never move `main` during diagnosis. Reverting the cutover itself requires an explicit reviewed decision and the archived old-main SHA.
