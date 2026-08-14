---
id: branch-reorganization-v1-audit
title: 分支重整 V1 完成稽核
lang: zh-TW
audience: developer
type: audit
status: published
owner: project
last_reviewed: 2026-08-14
---

<a id="scope"></a>
## 範圍

本稽核結束截至 2026-08-14 的 Branch Reorganization V1，涵蓋受保護 `main` 切換、feature recovery、remote-ref contraction、merged-worktree cleanup 與最終 inspection checkout。範圍不包含 physical spring calibration、remote launcher workstation smoke、Student production evidence、draft Panel physics/calibration proposal，以及分開的 reboot 與 sim2real worktrees。

<a id="method"></a>
## 方法

透過 GitHub API 檢查有效 administrator permission 與已儲存的 `main` protection，並從 GitHub 讀取 PR 狀態及 checks。刪除每個 remote candidate 前，立即與[保存清單](../governance/branch-preservation-manifest.zh-TW.md)中的 exact SHA 比對。只有在 `git status --porcelain` 顯示 clean 後才移除 local worktree；只有 `git branch --merged origin/main` 證明可達時才刪除 local branch。Recovery bundles 以 `git bundle verify` 與 SHA-256 驗證。

<a id="findings"></a>
## 發現

- PR #7 已 merge 到受保護的 `main` commit `5a6037b434ab6ba2ca54ba1474f5ff556a790115`。Protection 要求 strict `validate`、pull-request workflow 與 resolved conversations；administrator enforcement 已啟用，force push 與 branch deletion 已停用。
- Torsion PR #3、Training Panel V3.6 PR #4、Windows launcher PR #5、macOS launcher PR #10 與 Student V2 core PR #11 均可從 `main` 到達；其文件中的 hardware、workstation 與 production-evidence 限制仍然有效。
- 經 exact-tip 比對後，已刪除 21 個 legacy remote source refs、已 merge 的 macOS 與 Student topic refs，以及已 merge 的 PR #7 source ref。Remote 現在只保留 `main`、draft PR #12、`proposal/core-sim-first` 與 16 個 exact `archive/*` refs。
- 已移除 13 個 clean merged/archived secondary worktrees 與 19 個由 Git 證明 merged 的 local branches。剩餘三個 worktrees 是 root inspection checkout、明確的 reboot proposal 與明確的 sim2real recovery worktree。
- Draft PR #12 在 `40b841b00f3749e36bc26de9207f42e3c5ca3d31` 為 green，且仍在 `main` 之外。它是 simulation-only Panel physics/calibration 與 Student-browser proposal，不是 calibrated hardware behavior。
- 四代 root recovery 維持 local-only。最新的 focused path repair 是 commit `1a436d26a167ce8431a916a4fef78534259bde11`；其 complete-history bundle SHA-256 為 `6b8abcb8ed4d7e6d9c4d03c0559e50046e2cab9da969dccdd901e0415fd515b8`。Focused test 已通過，但 inherited F0/F1 recovery line 不是 merge candidate。

<a id="actions"></a>
## 行動

- [x] 啟用並驗證受保護的 `main`。
- [x] Merge branch-governance checkpoint。
- [x] 刪除已通過 exact-SHA 驗證的 obsolete remote source refs。
- [x] 移除 clean merged/archived secondary worktrees 與 Git 證明 merged 的 local branches。
- [x] 將 root checkout 移到以 final `main` 為基礎的 green Panel physics/calibration WIP。
- [ ] 任何 publication 或 merge 前，將 local-only Student F0/F1 recovery 作為分開專案審查。
- [ ] PR #12 在 simulation 與 physical evidence gates 被明確審查前維持 draft。

<a id="evidence"></a>
## 證據

Durable controls 與 recovery evidence 記錄於 [ADR-0002](../decisions/adr-0002-single-protected-main.zh-TW.md)、[branch management](../governance/branch-management.zh-TW.md)與[保存清單](../governance/branch-preservation-manifest.zh-TW.md)。GitHub 保留已 merge 的 [governance PR #7](https://github.com/Big-Hero-Hex/RedRHex/pull/7)與 green draft [Panel WIP PR #12](https://github.com/Big-Hero-Hex/RedRHex/pull/12)。

<a id="follow-up"></a>
## 後續追蹤

Branch Reorganization V1 已結束。只有在 `main` protection 變更、有人提議移除 `archive/*` ref、retained recovery line 被 promotion，或 PR #12 進入新的 evidence-backed lifecycle state 時才重新審查本結果。
