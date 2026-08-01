---
id: documentation-system-design
title: RedRHex 雙語文件系統設計
lang: zh-TW
audience: developer
type: design
status: approved
owner: project
last_reviewed: 2026-08-01
---

<a id="summary"></a>
## 摘要

RedRHex 將維護兩條明確分開的雙語入口：**操作人員文件**提供給設定、操作、校正、部署或排除機器人問題的人員；**開發人員文件**提供給理解、測試、擴充或維護軟體的人員。英文與繁體中文具有同等的正式地位，任何一種語言都不是附屬或可有可無的翻譯。

Markdown 是唯一真實來源。文件採混合配置：入口網站、政策、路線圖及架構集中管理，元件細節文件則與元件並置；納入版本控制的網站清單會把集中與並置文件一併收錄至發布網站。

<a id="language-and-scope"></a>
## 語言模式與範圍

正式維護的文件一律成對使用 `.en.md` 與 `.zh-TW.md` 後綴。根目錄與各元件的 `README.md` 是導向兩種語言入口的簡短雙語路由頁，也是唯一刻意保留在單一檔案內的雙語例外。

雙語要求涵蓋現行的操作人員與開發人員知識、共用參考資料、治理、設計、計畫、ADR、路線圖、版本發布及已發布證據。自動化、產生檔、技能、原始執行筆記及已刪除的歷史資料不要求翻譯。

<a id="information-architecture"></a>
## 資訊架構

目標結構如下：

```text
README.md
docs/
├── index.en.md
├── index.zh-TW.md
├── operators/{index.*,getting-started.*,training/,panel/,deployment/,calibration/,troubleshooting/}
├── developers/{index.*,architecture/,development/,testing/,subsystems/}
├── reference/
├── decisions/
├── designs/active/
├── plans/active/
├── roadmap/
├── releases/
├── research/
└── governance/
```

中央入口負責導覽、政策、路線圖與跨領域架構。元件儲存庫或目錄負責元件專屬的細節資料。中央入口與網站清單會納入這些並置文件，讓讀者在單一可導覽的系統中使用文件，同時不讓文件脫離其所說明的程式碼。

<a id="naming"></a>
## 命名規則

- 一般文件使用 `lowercase-kebab-case.<locale>.md`。
- 區段首頁使用 `index.<locale>.md`。
- 具有時間性的文件使用 `YYYY-MM-DD-slug.<locale>.md`。
- ADR 使用 `adr-0001-slug.<locale>.md`。
- 只有在時間順序屬於文件本質時才加日期。

<a id="metadata"></a>
## 中繼資料

每份維護中的文件都必須具備包含以下欄位的 YAML frontmatter：

```yaml
id: stable-identity
title: 在地化標題
lang: zh-TW
audience: developer
type: explanation
status: active
owner: project
last_reviewed: 2026-08-01
```

允許值如下：

- `lang`：`en`、`zh-TW`
- `audience`：`operator`、`developer`、`shared`
- `owner`：`project`、`core`、`training`、`panel`、`deployment`、`sim2real`、`reward-agent`
- `type`：`index`、`tutorial`、`how-to`、`reference`、`explanation`、`safety`、`troubleshooting`、`decision`、`design`、`plan`、`roadmap`、`release`、`experiment-summary`、`audit`
- 知識狀態：`draft`、`active`、`deprecated`
- 決策狀態：`accepted`、`superseded`
- 設計狀態：`proposed`、`approved`、`implemented`、`rejected`、`superseded`
- 計畫狀態：`draft`、`active`、`blocked`、`completed`、`cancelled`
- 路線圖狀態：`active`
- 版本、實驗與稽核狀態：`published`

位置、類型與狀態必須相符。某一生命週期的狀態不得用於另一生命週期的文件。

<a id="pair-contract"></a>
## 雙語配對契約

英文與繁體中文配對檔的中繼資料除 `title` 與 `lang` 外必須完全相同。任何改變意義的編輯都必須同時更新兩個檔案。相對應的標題前必須放置穩定、明確且相同的 HTML 錨點 ID。翻譯應保留意義、證據、限制、警告、命令、路徑與連結，而非逐字複製句型。

<a id="lifecycle"></a>
## 生命週期與保留政策

- 維護中的知識依 `draft -> active -> deprecated -> removed` 流轉。
- 現行設計在落實到維護中的架構、ADR、版本發布或路線圖項目後，從 `designs/active/` 移除。
- 已完成或已取消的計畫在摘要寫入永久文件後，從 `plans/active/` 移除。
- ADR 保留於儲存庫；以 supersede 取代重寫。
- 路線圖只保留目前優先事項；完成的工作移至版本或里程碑紀錄。
- 實驗摘要不可修改；更正須以有日期的附錄追加，不可默默重寫。
- 原始執行日誌與筆記忽略不納入；只有當證據改變基準、建議、決策或結果時才發布摘要。

Git 是已移除工作文件的檔案庫；現行檔案樹不作為雜物間。

<a id="staleness"></a>
## 過時政策

- 操作人員、參考及路線圖文件超過 90 天未審查即警告。
- 開發人員架構文件超過 180 天未審查即警告。
- 決策與版本文件只在受到矛盾或已被取代時警告。

警告用來促使審查，不會自動改變文件的真實性狀態。

<a id="documentation-impact"></a>
## 文件影響規則

變更依下列規則觸發文件工作：

- CLI、工作流程、環境、路徑、設定、硬體或安全變更，更新操作人員或參考文件。
- API、結構描述、架構、相依性、測試或擴充方式變更，更新開發人員或參考文件。
- 已發布的行為或相容性變更，更新受影響元件的版本文件。
- 跨領域決策需要 ADR。
- 已核准功能需要設計文件。
- 多步驟實作需要計畫。
- 內部重構必須記錄 `Docs impact: none` 的理由。

正式治理文件保存這些長期文件影響規則。根目錄 `AGENTS.md` 連結至該治理規則，而不重複其內容。

<a id="tooling"></a>
## 驗證與清冊工具

文件 CLI 如下：

```text
python -m tools.documentation validate --all
python -m tools.documentation validate --staged
python -m tools.documentation validate --changed-from REF
python -m tools.documentation inventory --format json
python -m tools.documentation stage-site --output DIR
```

成功時命令結束碼為 `0`，驗證失敗時為 `1`。驗證範圍包括檔名、必要 frontmatter 與列舉值、ID 唯一性、生命週期與位置、雙語配對與中繼資料一致性、連結與錨點，以及變更配對的一致性。

<a id="pull-requests"></a>
## Pull request 宣告與 CI

每個 pull request 必須宣告：

```text
Docs impact: none | operator | developer | shared | release | experiment
Docs reason: <required explanation>
```

CI 強制要求宣告存在且格式正確，但不從原始碼路徑推論語意影響。Pre-commit 驗證使用 `--staged`；CI 使用 `--all` 驗證。

<a id="repository-skills"></a>
## 儲存庫技能

`.agents/skills` 下設置兩個儲存庫技能：`writing-redrhex-docs` 與 `reviewing-redrhex-docs`。它們分別引導作者與審查者，但應引用正式治理文件，不得把政策複製到技能中。

<a id="cross-agent-adapters"></a>
## 跨代理程式轉接層

治理 Markdown 是文件政策的中立真實來源。根目錄 `AGENTS.md` 是 Codex 使用的廣泛開放代理程式轉接層；根目錄 `CLAUDE.md` 則為 Claude Code 匯入該檔案，不重複政策。

正式技能實作維持於 `.agents/skills`。精簡的 `.claude/skills` 包裝檔提供 Claude Code 探索能力，並將其導向正式技能檔，不複製工作流程政策。文件驗證器、pre-commit hook 與 CI 是跨代理程式保證：無論使用何種撰寫工具，都會拒絕不符合規則的輸出。

<a id="site"></a>
## 文件網站

Markdown 保持為正式來源，產生的 HTML 予以忽略。網站使用鎖定版本的 MkDocs、Material for MkDocs 與 `mkdocs-static-i18n` 相依套件。納入版本控制的清單會暫存中央與並置文件。Panel UI 維持為 GitHub Pages 根目錄，文件則發布於 `/docs/`。

兩種語言都會建置。每頁提供對等頁面的語言切換，搜尋索引同時涵蓋兩種語言，嚴格連結檢查則避免發布損壞的內部導覽。

<a id="release-model"></a>
## 發布模式

Panel 保留 SemVer 與其元件 changelog。中央版本索引連結各元件版本，而全專案里程碑使用日期，不套用全專案 SemVer。不得為 panel 虛構 `3.4.4` 至 `3.4.9` 版本；有證據的變更統一彙整於 `V3.4.10`。

<a id="historical-migration"></a>
## 歷史資料遷移

遷移工作應整理可長期保存的內容，而不是整份複製舊檔。遷移清單記錄每個舊標題、處置方式、取代它的一份或多份文件，以及移除該內容的 commit。只有在替代文件與可追溯性都通過驗證後才刪除重複的原始檔。歷史封存由 Git 提供。

<a id="safety-and-cleanup"></a>
## 安全與後續清理

安全文件採一般審查流程，不另設人工安全關卡。Git 歷史與分支清理是日後另外審查的專案，不在本次文件重整範圍內。
