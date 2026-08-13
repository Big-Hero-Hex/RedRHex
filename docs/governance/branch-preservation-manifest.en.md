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

No obsolete remote ref may be deleted until `main` protection is verified with repository-owner authority. Feature PRs remain draft until their physical torsion or Windows acceptance gates pass. Local worktree/ref contraction remains pending because the root contains the post-snapshot untracked path `docs/reports/`; that path is user-owned and has not been modified or classified by this migration.

<a id="recovery"></a>
## Recovery

Recover historical work from the recorded `archive/*` ref, not by moving `main`. The verified local bundle is `.worktrees/branch-reorg-recovery-2026-08-13.bundle` with SHA-256 `fbb10c25f87c87fa34c7360c2baf73508c6d31d3c38ffef5d3a3c50250bf86d2`. Keep it through the final published audit.
