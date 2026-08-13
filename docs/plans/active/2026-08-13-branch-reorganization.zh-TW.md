---
id: branch-reorganization-plan
title: 分支重整 V1
lang: zh-TW
audience: developer
type: plan
status: active
owner: project
last_reviewed: 2026-08-13
---

<a id="objective"></a>
## 目標

建立唯一受保護的 `main`，保留所有重要歷史與未完成變更，在角色明確的分支上完整重建 torsion、Training Panel 與 Windows launcher 工作，並且只有在保存證明通過後才移除多餘的分支與 worktree。

<a id="context"></a>
## 背景

已審查的整合 tip 是完成 Documentation System v1 後的 `fix/review-2026-07`。已授權的切換會強制取代較舊的遠端 `main`，但必須先建立可逆的 recovery refs。Torsion 與 Training Panel 工作不容遺失：原始 commit objects、merge 關係、working-tree 變更與 stash hunks 都必須可到達。Generated `.pyc` 變更只保留在 recovery；被 ignore 的 PowerShell launcher test 屬於原始碼，必須 force-track。

<a id="phased-checklist"></a>
## 分階段清單

<a id="preserve"></a>
### 保存

- [x] 盤點本機與遠端 refs、worktrees、dirty files 與 stash parents。
- [x] 將精確 dirty 與 stash snapshots 提交到 recovery refs，並建立已驗證的全 ref Git bundle。
- [x] 在移動 `main` 前發布 source、recovery、legacy、proposal 與 old-main archive refs。

<a id="reconstruct"></a>
### 重建

- [x] 將原始 torsion source tip merge 到已審查的 main，並把暫時文件的獨特內容遷移到 canonical bilingual files。
- [x] 將原始 Training Panel V3.6 tip 疊在 torsion 上，再逐 hunk 移植兩個 recovery snapshots，不得覆寫較新的修正。
- [x] 從 main 獨立重建 Windows launcher，包括 PowerShell test 與 canonical bilingual operator/developer documentation。
- [x] 在 preservation manifest 記錄每個 source commit、dirty path、stash hunk、duplicate、generated exception 與 destination。

<a id="cutover"></a>
### 切換

- [x] 從 fresh clone 或 bundle restore 驗證 archive refs 與重建分支。
- [ ] 由已審查的整合 SHA force-update 遠端 `main`，隨即啟用受保護的 PR-only 規則。
- [x] 發布 draft feature PR：torsion 優先、Training Panel 疊在 torsion 上、Windows 則平行從 main 開始。

<a id="contract"></a>
### 收斂

- [ ] 只有在記錄的 reachability 或 patch-equivalence 檢查通過後，才刪除或重新命名舊 refs。
- [ ] 將 root checkout 恢復為乾淨的 `main`，並把保留的 feature/proposal worktrees 放在 `.worktrees/`。
- [ ] 以 bilingual published audit 與 milestone record 取代本暫時 plan。

<a id="verification"></a>
## 驗證

驗證包含精確 SHA/ref 比對、`git fsck`、bundle verification、reachability 與 patch-equivalence reports、重建功能的 tree 與 `range-diff` 檢查、逐 hunk recovery disposition、documentation validation、受影響的 Python 與 Node 測試、Windows 上的 PowerShell tests、fresh-clone branch visibility、pull-request CI，以及 GitHub branch-protection inspection。

<a id="open-gates"></a>
## 未完成 gate

- 遠端 `main` 現已指向 reviewed line，但仍需 repository-owner authentication 才能啟用並檢查 branch protection；因此 combined cutover checkbox 維持未完成。
- Torsion 與 stacked Panel PR 在 physical spring calibration、backend selection 與必要 retraining evidence 完成前維持 draft。
- Windows launcher PR 在 Windows 通過 PowerShell 5.1 與 Tailscale/SSH smoke checklist 前維持 draft。
- Protection 啟用前，remote/local obsolete-ref contraction 維持 pending。Root worktree contraction 也會等待使用者決定 snapshot 後新增的 untracked `docs/reports/` path 如何處理。

<a id="completion-summary"></a>
## 完成摘要

只有在 `main` 成為受保護的已審查整合線、所有保留工作都可由遠端復原、feature branches 與 worktrees 各自只有一個明確角色、多餘 refs 已收斂，而且最終 audit 報告零個未分類 tip 或 recovery hunk 時，本計畫才算完成。
