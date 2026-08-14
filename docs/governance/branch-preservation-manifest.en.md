---
id: branch-preservation-manifest
title: Branch Preservation Manifest
lang: en
audience: developer
type: reference
status: active
owner: project
last_reviewed: 2026-08-14
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

The 2026-08-13 remote audit found zero unclassified branch tips. Significant unmatched histories have exact `archive/legacy/*` refs. Original torsion, Panel V3.6, Panel/torsion integration, and sim2real tips have exact `archive/source/*` refs. All audited dirty snapshots and the original three-parent stash object have exact `archive/recovery/*` refs. A fresh public mirror clone passed `git fsck --no-dangling` and exposed current main, reconstructed features, the proposal branch, and sampled cutover/source/recovery refs.

Torsion PR #3 merged as `6e5fa75cfff727bdd3ab74a1fcbba541f1de2281`; Panel V3.6 PR #4 merged as `39b983abfe70d1d0a82ff727407a3eeacc0b92ba`; macOS launcher PR #10 merged as `55e6611008e4ec7c0508cd4be1d1e639de42655b`; and Student V2 PR #11 merged as `b66d3760ffeffd1c0f95c1b9bb018ec797bb0357`. The recovered Panel physics and Student browser work is isolated in draft PR #12 at `4df5cc5698be7605db0319c5157838d72c4e32cd`; it is a simulation-only proposal and remains outside `main`.

`feature/direction-tracking` is not an ancestor by topology, but `git cherry origin/main origin/feature/direction-tracking` reports its only non-merge patch with `-`, proving patch equivalence to retained main history. It therefore does not require another archive ref.

<a id="pending-contraction"></a>
## Pending contraction

GitHub verified protected `main` on 2026-08-14 with strict `validate`, pull-request review, resolved-conversation, administrator-enforcement, no-force-push, and no-deletion settings. Obsolete remote refs still remain until this governance pull request merges and their tips are compared with this manifest immediately before deletion. Windows launcher PR #5 remains an implementation candidate without a Windows workstation smoke. The macOS design and plan remain draft/active because Finder, Terminal, and workstation smoke are still pending. Student V2 is merged as an additive research route without production-length, multi-seed, recorded-sensor, or hardware evidence. Panel physics/calibration PR #12 remains draft and must not be treated as calibrated hardware behavior.

The original root recovery is commit `8a6d5d2d28d940fab62d20bfd6141e54dd7eb9c4` with local bundle SHA-256 `93915814e54c115282274337f4d65fc0ee1e37f808b294db835461064a8baaa2`. A later partial Student F0/screening line was preserved as local commit `13f0549ac3d3de23a56a09355ac21781993a1d19` and bundle SHA-256 `606a778293e9eafa9a13c9ab2371651be6d9c9e994b0fca3203322651869b438`. A subsequent 46-file F0/F1 pipeline revision is preserved separately at local commit `a95991621a55c62fb8b67660471285c2c3aebd75` with complete-history bundle SHA-256 `66b6c7579ae2e6277d3229a487c29479cfb254afdc31a01dacd6007453e6e61d`; staged documentation validation fails on two inherited generated-roadmap files, so this revision is preservation-only and is not a merge candidate. The post-snapshot `docs/reports/` path remains in local recovery commit `02ebb53b32ff385fc0e8c36ef75e88ba8d944f70`, while its durable findings were migrated in PR #8. These local-only bundles and generated research artifacts must not be pushed to the public remote.

<a id="recovery"></a>
## Recovery

Recover historical work from the recorded `archive/*` ref or local recovery commit, not by moving `main`. The verified branch-reorganization bundle is `.worktrees/branch-reorg-recovery-2026-08-13.bundle` with SHA-256 `fbb10c25f87c87fa34c7360c2baf73508c6d31d3c38ffef5d3a3c50250bf86d2`. The root bundles are `.worktrees/root-wip-pre-main-sync-2026-08-13.bundle`, `.worktrees/root-post-snapshot-wip-2026-08-14.bundle`, and `.worktrees/root-f0-f1-wip-2026-08-14.bundle` with the hashes above. The raw-report bundle is `.worktrees/research-roadmap-report-2026-08-13.bundle` with SHA-256 `4c8eec2f76357d1cc6b0fcea929efd2712acc437f79e3bb13b208a2e4f6585db`. Keep these local bundles through the final published audit; exclude them from remote publication and automated worktree cleanup.
