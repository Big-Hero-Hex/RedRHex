---
id: branch-preservation-manifest
title: 分支保存清單
lang: zh-TW
audience: developer
type: reference
status: active
owner: project
last_reviewed: 2026-08-13
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

2026-08-13 remote audit 找到零個未分類 branch tip。重要且 unmatched 的歷史都有 exact `archive/legacy/*` ref。原始 torsion、Panel V3.6、Panel/torsion integration 與 sim2real tip 都有 exact `archive/source/*` ref。所有 dirty snapshot 與原始 three-parent stash object 都有 exact `archive/recovery/*` ref。Fresh public mirror clone 通過 `git fsck --no-dangling`，並可看到目前 main、三個 reconstructed feature branch、proposal branch，以及抽查的 cutover/source/recovery ref。

`feature/direction-tracking` 在 topology 上不是 ancestor，但 `git cherry origin/main origin/feature/direction-tracking` 對唯一 non-merge patch 回報 `-`，證明它與保留的 main history patch-equivalent，因此不需要另一個 archive ref。

<a id="pending-contraction"></a>
## 待完成 contraction

使用 repository-owner authority 驗證 `main` protection 前，不得刪除 obsolete remote ref。Feature PR 在 physical torsion 或 Windows acceptance gate 通過前維持 draft。Snapshot 後新增的 `docs/reports/` path 已完成分類：exact generated bundle 保存在 local recovery commit `02ebb53b32ff385fc0e8c36ef75e88ba8d944f70`，durable finding 已透過 documentation commit `2dff78e2410350defcf1603b9ca67f09bec030d3`（PR #8）遷移。Owner 已於 2026-08-13 選擇 local-only retention；raw PDF、HTML、preview、script 與 SQL 不得推送到 public remote。Local worktree/ref contraction 仍受 `main` protection gate 限制。

<a id="recovery"></a>
## 復原

從清單記錄的 `archive/*` ref 復原歷史工作，不可移動 `main`。已驗證的 branch-reorganization bundle 是 `.worktrees/branch-reorg-recovery-2026-08-13.bundle`，SHA-256 為 `fbb10c25f87c87fa34c7360c2baf73508c6d31d3c38ffef5d3a3c50250bf86d2`。Raw-report bundle 是 `.worktrees/research-roadmap-report-2026-08-13.bundle`，SHA-256 為 `4c8eec2f76357d1cc6b0fcea929efd2712acc437f79e3bb13b208a2e4f6585db`；`git bundle verify` 確認它包含至修正後 recovery commit 的 complete history。Branch-reorganization bundle 保留至 final published audit。Raw-report bundle 作為 durable local-only archive 保留，並排除於 remote publication 與 worktree cleanup。
