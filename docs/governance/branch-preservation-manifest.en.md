---
id: branch-preservation-manifest
title: Branch Preservation Manifest
lang: en
audience: developer
type: reference
status: active
owner: project
last_reviewed: 2026-08-13
---

<a id="purpose"></a>
## Purpose

The machine-readable `docs/governance/branch-preservation-manifest.csv` records the exact disposition of pre-cutover refs, dirty paths, stash hunks, and the recovery bundle. It is the expand-and-verify record for Branch Reorganization V1; it does not authorize deletion by itself.

<a id="fields"></a>
## Fields

- `record_type` distinguishes remote refs, local refs, recovery objects, source paths, and stash hunks.
- `source` is the original ref or path-qualified recovery source.
- `tip_or_object` is the full Git object ID or the bundle SHA-256.
- `classification` explains why the record is retained, migrated, redundant, or excluded.
- `retained_or_destination` identifies the durable recovery ref or reconstructed destination.
- `verification` records the preservation proof.
- `disposition` records the completed or gated contraction action.

<a id="current-result"></a>
## Current result

The 2026-08-13 remote audit found zero unclassified branch tips. Significant unmatched histories have exact `archive/legacy/*` refs. Original torsion, Panel V3.6, Panel/torsion integration, and sim2real tips have exact `archive/source/*` refs. All dirty snapshots and the original three-parent stash object have exact `archive/recovery/*` refs. A fresh public mirror clone passed `git fsck --no-dangling` and exposed current main, all three reconstructed feature branches, the proposal branch, and the sampled cutover/source/recovery refs.

`feature/direction-tracking` is not an ancestor by topology, but `git cherry origin/main origin/feature/direction-tracking` reports its only non-merge patch with `-`, proving patch equivalence to retained main history. It therefore does not require another archive ref.

<a id="pending-contraction"></a>
## Pending contraction

No obsolete remote ref may be deleted until `main` protection is verified with repository-owner authority. The torsion and stacked Training Panel feature PRs remain draft until their physical acceptance gates pass. Windows launcher PR #5 was merged as `8d17dbf7976a88d2978a0e9b41245d80bec40fd7` after the owner explicitly accepted the unexecuted Windows-only smoke risk; its design remains approved, its plan remains active, and no release evidence is claimed. The post-snapshot `docs/reports/` path is no longer unclassified: its exact generated bundle is preserved in local recovery commit `02ebb53b32ff385fc0e8c36ef75e88ba8d944f70`, and its durable findings were migrated in documentation commit `2dff78e2410350defcf1603b9ca67f09bec030d3` (PR #8). The owner selected local-only retention on 2026-08-13; the raw PDF, HTML, previews, scripts, and SQL must not be pushed to the public remote. Local worktree/ref contraction remains gated by `main` protection.

<a id="recovery"></a>
## Recovery

Recover historical work from the recorded `archive/*` ref, not by moving `main`. The verified branch-reorganization bundle is `.worktrees/branch-reorg-recovery-2026-08-13.bundle` with SHA-256 `fbb10c25f87c87fa34c7360c2baf73508c6d31d3c38ffef5d3a3c50250bf86d2`. The raw-report bundle is `.worktrees/research-roadmap-report-2026-08-13.bundle` with SHA-256 `4c8eec2f76357d1cc6b0fcea929efd2712acc437f79e3bb13b208a2e4f6585db`; `git bundle verify` confirms complete history through the corrected recovery commit. Keep the branch-reorganization bundle through the final published audit. Keep the raw-report bundle as the durable local-only archive and exclude it from remote publication and worktree cleanup.
