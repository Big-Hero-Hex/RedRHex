---
id: branch-reorganization-plan
title: Branch Reorganization V1
lang: en
audience: developer
type: plan
status: active
owner: project
last_reviewed: 2026-08-14
---

<a id="objective"></a>
## Objective

Establish one protected `main`, retain every significant historical and unfinished change, reconstruct the intact torsion, Training Panel, remote-launcher, and Student V2 work on explicit branches, isolate calibration WIP, and remove redundant branch and worktree clutter only after preservation proof passes.

<a id="context"></a>
## Context

The reviewed integration tip is `fix/review-2026-07` after Documentation System v1. The authorized cutover force-replaces the older remote `main`, but only after reversible recovery refs exist. Torsion and Training Panel work are loss-intolerant: their original commit objects, merge relationship, working-tree changes, and stash hunks must remain reachable. Generated `.pyc` mutations are recovery-only; the ignored PowerShell launcher test is source and must be force-tracked.

<a id="phased-checklist"></a>
## Phased checklist

<a id="preserve"></a>
### Preserve

- [x] Inventory local and remote refs, worktrees, dirty files, and stash parents.
- [x] Commit exact dirty and stash snapshots to recovery refs and create a verified all-ref Git bundle.
- [x] Publish source, recovery, legacy, proposal, and old-main archive refs before moving `main`.

<a id="reconstruct"></a>
### Reconstruct

- [x] Merge the original torsion source tip onto the reviewed main and migrate unique temporary documentation into canonical bilingual files.
- [x] Stack the original Training Panel V3.6 tip on torsion, then port both recovery snapshots hunk by hunk without overwriting later fixes.
- [x] Rebuild the Windows launcher independently from main, including its PowerShell test and canonical bilingual operator/developer documentation.
- [x] Rebuild and merge the macOS launcher as an implementation candidate with workstation smoke still pending.
- [x] Reconstruct and merge the Student Distillation V2 core; isolate its co-developed Panel browser and physics/calibration work in a separate draft proposal.
- [x] Preserve all three root recovery generations and the generated research report in verified local-only commits and bundles.
- [x] Record every source commit, dirty path, stash hunk, duplicate, generated exception, and destination in the preservation manifest.

<a id="cutover"></a>
### Cut over

- [x] Verify archive refs and reconstructed branches from a fresh clone or bundle restore.
- [x] Force-update remote `main` from the reviewed integration SHA.
- [x] Enable and verify protected PR-only rules for `main`.
- [x] Publish, validate, and merge the torsion, Training Panel, Windows, macOS, and Student V2 PRs; retain Panel physics/calibration as draft PR #12.

<a id="contract"></a>
### Contract

- [ ] Delete or rename obsolete refs only after their recorded reachability or patch-equivalence check passes.
- [ ] Return the root checkout to clean `main` and keep retained feature/proposal worktrees under `.worktrees/`.
- [ ] Replace this temporary plan with a bilingual published audit and milestone record.

<a id="verification"></a>
## Verification

Verification includes exact SHA/ref comparison, `git fsck`, bundle verification, reachability and patch-equivalence reports, tree and `range-diff` checks for reconstructed features, per-hunk recovery disposition, documentation validation, affected Python and Node test suites, PowerShell tests on Windows, fresh-clone branch visibility, pull-request CI, and GitHub branch-protection inspection.

<a id="open-gates"></a>
## Open gates

- GitHub verified protected `main` with strict `validate`, required pull-request workflow, resolved conversations, administrator enforcement, and force-push/deletion disabled.
- Torsion and Panel V3.6 are merged as status-honest implementation candidates; physical spring calibration and production retraining remain separate pending evidence.
- Windows and macOS launchers are merged as implementation candidates; their named workstation smoke checklists remain pending.
- Student V2 core is merged without production-length, multi-seed, recorded-sensor, or hardware claims. Panel physics/calibration remains draft PR #12.
- Remote and local obsolete-ref contraction remains pending until this governance pull request merges and the recorded tips pass an immediate pre-deletion comparison. All dirty root material, including the later F0/F1 revision, now has an exact local recovery commit and verified bundle.

<a id="completion-summary"></a>
## Completion summary

The plan is complete only when `main` is the protected reviewed line, all retained work is remotely recoverable, feature branches and worktrees have one documented role, redundant refs are contracted, and the final audit reports zero unclassified tips or recovery hunks.
