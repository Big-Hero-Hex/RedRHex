---
id: adr-0002-single-protected-main
title: 單一受保護的 Main 分支
lang: zh-TW
audience: developer
type: decision
status: accepted
owner: project
last_reviewed: 2026-08-13
---

<a id="context"></a>
## 背景

RedRHex 累積了兩條彼此競爭的整合線、長期存在的實驗分支、只存在本機的功能堆疊、混合多種工作的未提交 worktree，以及與 GitHub 分歧的本機 `main`。Documentation System v1 已審查並合併到 `fix/review-2026-07`，但遠端 `main` 仍指向較舊的歷史線。若繼續保留兩條整合線，貢獻者與自動化流程就沒有唯一可信的起點。

<a id="decision"></a>
## 決策

`main` 是唯一永久整合分支。作為一次性的遷移例外，只有在舊 `main`、每個重要且未匹配的 tip、所有未提交 worktree 與保留的 stash 都具備可復原的 archive ref，並通過分支保存稽核後，才可將遠端 `main` 強制更新到已審查的 `fix/review-2026-07` tip。

切換後，`main` 必須受保護：變更透過 pull request 進入、必要驗證必須通過、review 對話必須解決、禁止 force push 與刪除；進行中的開發使用短期 `feature/*`、`fix/*`、`docs/*` 或 `chore/*` 分支。尚未核准且需長期保留的架構提案放在 `proposal/*`；歷史 tip 放在 `archive/*`，不得作為開發基底。

完整的 torsion 工作先從新的 `main` 重建。Training Panel V3.6 疊在 torsion 分支上，使既有 merge 與整合行為仍可追溯。Windows 遠端啟動器則獨立從 `main` 重建。在 tree、commit 與 hunk 層級的保存檢查全部通過前，recovery 與 source 分支都必須保留。

<a id="alternatives"></a>
## 替代方案

- 不強制更新而以 ancestry merge 串接歷史，雖能保留舊 `main` 圖形，卻會把無關的舊 UI/shim commits 留在 canonical line，並模糊已審查的切換點。
- 永久保留 `fix/review-2026-07` 作為整合分支，會延續雙 trunk 的歧義。
- 永久增加 `develop` 分支，在目前沒有 release train 需求時只會增加協調成本。
- 將 torsion 與 Panel 歷史 squash 或 replay，雖可簡化圖形，卻會削弱「既有功能工作零遺失」的證明。

<a id="consequences"></a>
## 後果

既有 clone 在切換後必須 fetch，並明確重新對齊本機 `main`；archive refs 仍可用於復原。這次 force update 有文件紀錄且可回復，但之後一律禁止對 `main` force push。功能 PR 可暫時堆疊；已合併或被取代的分支，只有在 archive manifest 記錄其 tip SHA 與 disposition 後才可移除。

<a id="supersession"></a>
## 取代條件

只有當專案出現單一受保護 trunk 無法滿足的具體 release-management 需求時，後續 ADR 才可引入另一種永久整合模型。
