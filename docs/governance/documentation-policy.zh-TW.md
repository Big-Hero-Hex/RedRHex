---
id: documentation-policy
title: 文件政策
lang: zh-TW
audience: developer
type: reference
status: active
owner: project
last_reviewed: 2026-08-01
---

<a id="audiences"></a>
## 讀者路徑

RedRHex 維護兩條明確分開的雙語入口：

- **操作人員文件**提供給設定、操作、校正、部署或排除機器人問題的人員。
- **開發人員文件**提供給理解、測試、擴充或維護軟體的人員。

共用參考資料可以同時服務兩條路徑，但各入口都必須維持清楚的讀者旅程。

<a id="bilingual-scope"></a>
## 雙語範圍與例外

英文與繁體中文是同等正式來源。雙語要求涵蓋現行的操作人員與開發人員知識、共用參考資料、治理、設計、計畫、ADR、路線圖、版本發布及已發布證據。

自動化、產生檔、技能、原始執行筆記及已刪除的歷史資料不要求翻譯。根目錄與元件 `README.md` 是簡短的單檔雙語路由，也是唯一面向人員的雙語例外。

<a id="placement"></a>
## 混合放置方式

中央文件負責入口、治理、路線圖、決策及跨領域架構或參考資料。元件專屬細節文件則與其說明的元件並置。中央導覽與納入版本控制的網站清單同時收錄中央及並置文件，讓讀者使用一套相互連結的系統。

<a id="document-types"></a>
## 文件類型與用途

- `index` 用作入口或區段首頁。
- `tutorial` 透過引導式學習流程進行教學。
- `how-to` 提供特定任務的執行步驟。
- `reference` 記錄精確事實、介面、命令、結構描述或契約。
- `explanation` 說明概念、理由或架構。
- `safety` 記錄安全關鍵限制與程序。
- `troubleshooting` 診斷症狀及復原操作。
- `decision` 是記錄長期跨領域決策的 ADR。
- `design` 在實作前規定已核准功能或重大變更。
- `plan` 安排多步驟實作順序。
- `roadmap` 記錄目前優先事項與尚未解決的未來工作。
- `release` 記錄已發布的元件行為或有日期的專案里程碑。
- `experiment-summary` 發布會改變基準、建議、決策或結果的證據。
- `audit` 保留可長期保存的審查或合規發現。

<a id="retention-and-review"></a>
## 保留與審查

Git 是檔案庫；現行文件樹不作為雜物間。可長期保存的內容應遷移至持續維護的雙語配對文件；重複的歷史來源文件只能在替代內容可追溯且驗證成功後刪除。安全文件採一般審查流程，不需要另外設置人工安全關卡。

<a id="related-governance"></a>
## 相關治理規則

詳細契約請使用[中繼資料結構描述](metadata-schema.zh-TW.md)、[命名慣例](naming-conventions.zh-TW.md)、[文件生命週期](document-lifecycle.zh-TW.md)、[翻譯指南](translation-guide.zh-TW.md)、[文件範本](document-templates.zh-TW.md)、[遷移清單](migration-manifest.zh-TW.md)、[文件影響規則](documentation-impact.zh-TW.md)及 [README 路由慣例](readme-router-convention.zh-TW.md)。
