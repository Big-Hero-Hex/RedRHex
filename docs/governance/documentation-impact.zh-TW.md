---
id: documentation-impact
title: 文件影響規則
lang: zh-TW
audience: developer
type: reference
status: active
owner: project
last_reviewed: 2026-08-01
---

<a id="change-triggers"></a>
## 變更觸發條件

| 變更觸發條件 | 必要目標類型 |
| --- | --- |
| CLI、工作流程、環境、路徑、設定、硬體或安全變更 | 操作人員文件或共用參考資料 |
| API、結構描述、架構、相依性、測試或擴充方式變更 | 開發人員文件或共用參考資料 |
| 已發布行為或相容性變更 | 受影響元件的版本文件 |
| 跨領域決策 | ADR（`decision`） |
| 已核准功能 | `design` |
| 多步驟實作 | `plan` |
| 不影響讀者的內部重構 | 具體的無影響宣告 |
| 改變基準、建議、決策或結果的證據 | `experiment-summary` |

變更若跨越多列，必須更新每個受影響讀者或類型；選擇一個目標不會免除其他目標。

<a id="pull-request-declaration"></a>
## Pull request 宣告

每個 pull request 必須完整宣告下列欄位：

```text
Docs impact: none | operator | developer | shared | release | experiment
Docs reason: <required explanation>
```

內部重構仍必須提供具體的 `Docs impact: none` 理由，說明為何持續維護的讀者旅程、參考資料、版本、決策、設計、計畫及證據摘要均不受影響。

<a id="review-responsibility"></a>
## 審查責任

自動化會驗證宣告是否存在及格式是否正確。語意正確性仍由審查者負責；自動化不會從變更的原始碼路徑推論文件影響。

<a id="stable-tool-interface"></a>
## 穩定工具介面

階段 3 將實作下列穩定介面。這些命令是未來工具的契約，在目前檢查點尚不可用：

```text
python -m tools.documentation validate --all
python -m tools.documentation validate --staged
python -m tools.documentation validate --changed-from REF
python -m tools.documentation inventory --format json
python -m tools.documentation stage-site --output DIR
```

成功時命令結束碼為 `0`，驗證失敗時為 `1`。Pre-commit 將使用 `validate --staged`，CI 將使用 `validate --all`。
