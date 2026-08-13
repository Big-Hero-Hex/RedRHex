---
id: branch-management
title: 分支與 Worktree 管理
lang: zh-TW
audience: developer
type: reference
status: active
owner: project
last_reviewed: 2026-08-14
---

<a id="branch-roles"></a>
## 分支角色

- `main` 是唯一永久整合分支，必須保持可發布並受到保護。
- `feature/*`、`fix/*`、`docs/*` 與 `chore/*` 是短期且需審查的變更分支。
- `proposal/*` 保存尚未核准的長期設計或證據線，不得被描述為目前實作。
- `archive/*` 保存歷史或 recovery tips。Archive 分支是唯讀復原參照，不是 merge target 或開發基底。

<a id="change-flow"></a>
## 變更流程

新工作從目前的 `origin/main` 開始，每個分支只處理一個完整且一致的關注點，並儘早開啟 draft pull request。只有存在真正依賴時，stacked branch 才可把另一個 feature branch 當作目標；依賴合併後，必須將後續 pull request retarget 或 rebase 到 `main`。已合併或被取代的變更分支，只有在其 tip 已可從 `main` 或有紀錄的 archive ref 到達後才可刪除。

當 recovered code 適合檢視，但 evidence gates 尚未完成時，可暫時保留 `*-wip` branch。它必須使用 draft pull request、列出未完成 gates，且不得宣稱 branch 已 shipped 或 hardware-ready。

每個 pull request 都必須遵循 repository documentation-impact declaration。完成有紀錄的 2026-08-13 切換後，禁止直接 push、force push 或刪除 `main`。

<a id="worktrees"></a>
## Worktree

repository root checkout 保持在乾淨的 `main`。其他進行中或 proposal 分支使用 `.worktrees/<branch-slug>`。移除 worktree 前，其狀態必須乾淨；否則必須把精確的 tracked、untracked、被 ignore 但必要，以及 stashed 變更提交到 recovery ref。只有在 recovery 紀錄明確辨識後，generated cache 才可不進入乾淨的 feature branch。

<a id="archive-contract"></a>
## Archive 契約

branch archive manifest 記錄每個舊 ref、精確 tip SHA、分類、保留 ref、最終 disposition 與驗證結果。重要且未匹配的工作必須取得 `archive/*` 分支；已可從保留 ref 到達或 patch-equivalent 的 tip，在記錄 mapping 後可刪除。只有遠端 refs、本機 bundle、重建分支與驗證證據一致後，才執行 archive contraction。

<a id="recovery"></a>
## 復原

要復原切換前的歷史，應從有紀錄的 archive ref 或 SHA 建立新分支；診斷期間不得移動 `main`。若要回復切換本身，必須有明確且已審查的決策，並使用 archive 中記錄的舊 main SHA。
