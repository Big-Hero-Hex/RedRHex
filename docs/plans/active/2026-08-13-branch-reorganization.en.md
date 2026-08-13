---
id: branch-reorganization-plan
title: Branch Reorganization V1
lang: en
audience: developer
type: plan
status: active
owner: project
last_reviewed: 2026-08-13
---

<a id="objective"></a>
## Objective

Establish one protected `main`, retain every significant historical and unfinished change, reconstruct the intact torsion, Training Panel, and Windows launcher work on explicit branches, and remove redundant branch and worktree clutter only after preservation proof passes.

<a id="context"></a>
## Context

The reviewed integration tip is `fix/review-2026-07` after Documentation System v1. The authorized cutover force-replaces the older remote `main`, but only after reversible recovery refs exist. Torsion and Training Panel work are loss-intolerant: their original commit objects, merge relationship, working-tree changes, and stash hunks must remain reachable. Generated `.pyc` mutations are recovery-only; the ignored PowerShell launcher test is source and must be force-tracked.

<a id="phased-checklist"></a>
## Phased checklist

<a id="preserve"></a>
### Preserve

- [x] Inventory local and remote refs, worktrees, dirty files, and stash parents.
- [x] Commit exact dirty and stash snapshots to recovery refs and create a verified all-ref Git bundle.
- [ ] Publish source, recovery, legacy, proposal, and old-main archive refs before moving `main`.

<a id="reconstruct"></a>
### Reconstruct

- [ ] Merge the original torsion source tip onto the reviewed main and migrate unique temporary documentation into canonical bilingual files.
- [ ] Stack the original Training Panel V3.6 tip on torsion, then port both recovery snapshots hunk by hunk without overwriting later fixes.
- [ ] Rebuild the Windows launcher independently from main, including its PowerShell test and canonical bilingual operator/developer documentation.
- [ ] Record every source commit, dirty path, stash hunk, duplicate, generated exception, and destination in the preservation manifest.

<a id="cutover"></a>
### Cut over

- [ ] Verify archive refs and reconstructed branches from a fresh clone or bundle restore.
- [ ] Force-update remote `main` from the reviewed integration SHA, then immediately enable protected PR-only rules.
- [ ] Publish draft feature PRs with torsion first, Training Panel stacked on torsion, and Windows parallel from main.

<a id="contract"></a>
### Contract

- [ ] Delete or rename obsolete refs only after their recorded reachability or patch-equivalence check passes.
- [ ] Return the root checkout to clean `main` and keep retained feature/proposal worktrees under `.worktrees/`.
- [ ] Replace this temporary plan with a bilingual published audit and milestone record.

<a id="verification"></a>
## Verification

Verification includes exact SHA/ref comparison, `git fsck`, bundle verification, reachability and patch-equivalence reports, tree and `range-diff` checks for reconstructed features, per-hunk recovery disposition, documentation validation, affected Python and Node test suites, PowerShell tests on Windows, fresh-clone branch visibility, pull-request CI, and GitHub branch-protection inspection.

<a id="completion-summary"></a>
## Completion summary

The plan is complete only when `main` is the protected reviewed line, all retained work is remotely recoverable, feature branches and worktrees have one documented role, redundant refs are contracted, and the final audit reports zero unclassified tips or recovery hunks.
