---
id: branch-preservation-manifest
title: 分支保存清單
lang: zh-TW
audience: developer
type: reference
status: active
owner: project
last_reviewed: 2026-08-14
---

<a id="purpose"></a>
## 用途

Machine-readable `docs/governance/branch-preservation-manifest.csv` 記錄切換前 ref、dirty path、stash hunk 與 recovery bundle 的確切 disposition。它是 Branch Reorganization V1 的 expand-and-verify 紀錄；此清單本身不授權刪除。

<a id="fields"></a>
## 欄位

- `record_type` 區分 remote ref、local ref、recovery object、source path 與 stash hunk。
- `source` 是原始 ref 或包含 recovery source 的 path。
- `tip_or_object` 是完整 Git object ID 或 bundle SHA-256。
- `classification` 說明該紀錄為何保留、遷移、重複或排除。
- `retained_or_destination` 指出 durable recovery ref 或 reconstructed destination。
- `verification` 記錄 preservation proof。
- `disposition` 記錄已完成或仍受 gate 限制的 contraction action。

<a id="current-result"></a>
## 目前結果

2026-08-13 remote audit 找到零個未分類 branch tip。重要且 unmatched 的歷史都有 exact `archive/legacy/*` ref。原始 torsion、Panel V3.6、Panel/torsion integration 與 sim2real tip 都有 exact `archive/source/*` ref。所有已稽核 dirty snapshot 與原始 three-parent stash object 都有 exact `archive/recovery/*` ref。Fresh public mirror clone 通過 `git fsck --no-dangling`，並可看到目前 main、reconstructed features、proposal branch，以及抽查的 cutover/source/recovery ref。

Torsion PR #3 已以 `6e5fa75cfff727bdd3ab74a1fcbba541f1de2281` merge；Panel V3.6 PR #4 已以 `39b983abfe70d1d0a82ff727407a3eeacc0b92ba` merge；macOS launcher PR #10 已以 `55e6611008e4ec7c0508cd4be1d1e639de42655b` merge；Student V2 PR #11 已以 `b66d3760ffeffd1c0f95c1b9bb018ec797bb0357` merge。復原出的 Panel physics 與 Student browser 工作已隔離在 draft PR #12，tip 為 `4df5cc5698be7605db0319c5157838d72c4e32cd`；它是 simulation-only proposal，仍在 `main` 之外。

`feature/direction-tracking` 在 topology 上不是 ancestor，但 `git cherry origin/main origin/feature/direction-tracking` 對唯一 non-merge patch 回報 `-`，證明它與保留的 main history patch-equivalent，因此不需要另一個 archive ref。

<a id="pending-contraction"></a>
## 待完成 contraction

使用 repository-owner authority 驗證 `main` protection 前，不得刪除 obsolete remote ref。Windows launcher PR #5 仍是尚未完成 Windows workstation smoke 的 implementation candidate。macOS design 與 plan 維持 draft/active，因 Finder、Terminal 與 workstation smoke 仍 pending。Student V2 已作為 additive research route merge，但沒有 production-length、multi-seed、recorded-sensor 或 hardware evidence。Panel physics/calibration PR #12 維持 draft，不得視為 calibrated hardware behavior。

原始 root recovery 是 commit `8a6d5d2d28d940fab62d20bfd6141e54dd7eb9c4`，local bundle SHA-256 為 `93915814e54c115282274337f4d65fc0ee1e37f808b294db835461064a8baaa2`。較晚的 partial Student F0/screening line 已另外保存在 local commit `13f0549ac3d3de23a56a09355ac21781993a1d19`，bundle SHA-256 為 `606a778293e9eafa9a13c9ab2371651be6d9c9e994b0fca3203322651869b438`；它只供 preservation，不是 merge candidate。Snapshot 後新增的 `docs/reports/` path 仍保存在 local recovery commit `02ebb53b32ff385fc0e8c36ef75e88ba8d944f70`，durable findings 則已透過 PR #8 遷移。這些 local-only bundle 與 generated research artifacts 不得推送到 public remote。

<a id="recovery"></a>
## 復原

從清單記錄的 `archive/*` ref 或 local recovery commit 復原歷史工作，不可移動 `main`。已驗證的 branch-reorganization bundle 是 `.worktrees/branch-reorg-recovery-2026-08-13.bundle`，SHA-256 為 `fbb10c25f87c87fa34c7360c2baf73508c6d31d3c38ffef5d3a3c50250bf86d2`。Root bundles 是 `.worktrees/root-wip-pre-main-sync-2026-08-13.bundle` 與 `.worktrees/root-post-snapshot-wip-2026-08-14.bundle`，hash 如上。Raw-report bundle 是 `.worktrees/research-roadmap-report-2026-08-13.bundle`，SHA-256 為 `4c8eec2f76357d1cc6b0fcea929efd2712acc437f79e3bb13b208a2e4f6585db`。這些 local bundles 保留至 final published audit，並排除於 remote publication 與 automated worktree cleanup。
