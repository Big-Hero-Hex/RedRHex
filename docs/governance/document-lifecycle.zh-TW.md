---
id: document-lifecycle
title: 文件生命週期
lang: zh-TW
audience: developer
type: reference
status: active
owner: project
last_reviewed: 2026-08-01
---

<a id="lifecycles"></a>
## 生命週期

- 維護中的知識依 `draft -> active -> deprecated -> removed` 流轉。
- 決策狀態為 `accepted` 或 `superseded`。已接受與已取代的 ADR 都保留於儲存庫；後續 ADR 應取代舊 ADR，不得重寫舊紀錄。
- 設計從 `proposed` 進入 `approved`，再以 `implemented`、`rejected` 或 `superseded` 結案。
- 計畫從 `draft` 進入 `active`，期間可為 `blocked`，最後以 `completed` 或 `cancelled` 結案。
- 路線圖狀態為 `active`，且只保留目前優先事項。
- 版本、實驗摘要及稽核是狀態為 `published` 的紀錄。

<a id="resolution"></a>
## 結案與保留

現行設計應依情況摘要至持續維護的架構、ADR、版本或路線圖項目，結案後再從 `designs/active/` 移除。已完成或已取消的計畫應摘要至永久文件，再從 `plans/active/` 移除。已接受與已取代的 ADR 均保留。

已完成的路線圖工作移至版本或里程碑紀錄。實驗摘要不可修改；更正須以有日期的附錄追加，不可默默重寫。原始執行日誌與筆記忽略不納入。只有當證據改變基準、建議、決策或結果時才發布實驗摘要。

<a id="stale-warnings"></a>
## 過時警告

- 操作人員、參考及路線圖文件超過 90 天未審查即警告。
- 開發人員架構文件超過 180 天未審查即警告。
- 決策與版本文件不使用依時間判定的警告；只在受到矛盾或已被取代時警告。

過時警告用來促使審查，不會自動改變文件的真實性狀態。

<a id="removal-preconditions"></a>
## 移除前提

刪除永久或歷史來源文件前：

1. 建立完整的替代語言配對。
2. 驗證替代文件的連結及相同明確錨點。
3. 在遷移紀錄中記錄各標題的處置方式及替代路徑。
4. 通過文件驗證器。
5. 刪除來源，並在遷移紀錄中記下移除 commit。

上述條件達成後，由 Git 歷史負責封存；現行檔案樹不得為了保存歷史而保留重複原始檔。
