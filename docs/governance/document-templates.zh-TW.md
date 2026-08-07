---
id: document-templates
title: 文件範本
lang: zh-TW
audience: developer
type: reference
status: active
owner: project
last_reviewed: 2026-08-07
---

<a id="purpose"></a>
## 目的

這些範本提供各種維護文件類型所需的起始結構。請依文件用途與語言複製相符範本；不得把範例視為專案事實。

<a id="template-selection"></a>
## 選擇範本

- 入口或區段首頁使用 [index 範本](https://github.com/JasonLiaoJCS/RedRHex/blob/main/docs/governance/templates/index.zh-TW.md.template)。
- `tutorial`、`how-to`、`reference`、`explanation`、`safety` 或 `troubleshooting` 使用 [knowledge 範本](https://github.com/JasonLiaoJCS/RedRHex/blob/main/docs/governance/templates/knowledge.zh-TW.md.template)。作者必須選擇一個確切的 `type`，以及治理規則允許該類型使用的一個狀態。
- ADR 使用 [decision 範本](https://github.com/JasonLiaoJCS/RedRHex/blob/main/docs/governance/templates/decision.zh-TW.md.template)。
- 提案中或已核准的功能或重大變更使用 [design 範本](https://github.com/JasonLiaoJCS/RedRHex/blob/main/docs/governance/templates/design.zh-TW.md.template)。
- 多步驟實作使用 [plan 範本](https://github.com/JasonLiaoJCS/RedRHex/blob/main/docs/governance/templates/plan.zh-TW.md.template)。
- 目前優先事項與未解決未來工作使用 [roadmap 範本](https://github.com/JasonLiaoJCS/RedRHex/blob/main/docs/governance/templates/roadmap.zh-TW.md.template)。
- 已發布行為或有日期的專案里程碑使用 [release 範本](https://github.com/JasonLiaoJCS/RedRHex/blob/main/docs/governance/templates/release.zh-TW.md.template)。
- 會改變基準、建議、決策或結果的證據使用 [experiment-summary 範本](https://github.com/JasonLiaoJCS/RedRHex/blob/main/docs/governance/templates/experiment-summary.zh-TW.md.template)。
- 可長期保存的審查或合規發現使用 [audit 範本](https://github.com/JasonLiaoJCS/RedRHex/blob/main/docs/governance/templates/audit.zh-TW.md.template)。

<a id="authoring-contract"></a>
## 撰寫契約

發布前須取代每個角括號佔位符。在同一變更中建立兩種語言檔案，維持相同的非在地化中繼資料與明確錨點 ID，並翻譯為意義對等的內容。請從[中繼資料結構描述](metadata-schema.zh-TW.md)選擇確切的中繼資料值與生命週期狀態。階段 3 提供驗證器後，提交前執行 `python -m tools.documentation validate --staged`。

<a id="template-assets"></a>
## 範本資產

範本刻意使用 `.md.template` 結尾；它們是撰寫資產，而不是網站頁面。未取代所有佔位符前，絕不可將範本重新命名為正式輸出。請使用網站語言切換器開啟此目錄的英文版本。
