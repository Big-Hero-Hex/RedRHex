---
id: documentation-system-v1-release
title: 文件系統 V1 里程碑
lang: zh-TW
audience: shared
type: release
status: published
owner: project
last_reviewed: 2026-08-07
---

<a id="scope"></a>
## 範圍

此日期式專案 milestone 建立第一版受維護的 RedRHex 文件系統。它不是 RedRHex 全域 semantic version，也不改變獨立版本化的 Training Panel release。

<a id="published-system"></a>
## 已發布系統

- 根目錄 README 導向彼此分離的操作人員與開發人員旅程。
- 受維護且面向人員的知識以英文及繁體中文配對檔案作為同等正式來源；簡短雙語 README routers 是明文規定的例外。
- 中央 portals 涵蓋操作流程、開發架構與擴充、參考資料、決策、進行中設計與計畫、路線圖、發行、研究證據及治理。
- Training Panel、Reward Agent 與 ROS 2 部署的詳細文件留在所屬程式碼旁，並納入中央 portal 與生成網站。
- Metadata、命名、生命週期、翻譯、範本、documentation-impact declaration、過時門檻與遷移可追溯性都有可執行的 repository contracts。

<a id="migration"></a>
## 遷移

舊命令指南、訓練／播放指南、報告、sim-to-real 資料、Reward Agent 工作、Windows launcher 工作、Training Panel manuals/changelogs 與 ROS monolith 的持久內容已精選為正式文件。在記錄 heading-level dispositions 後，移除重複 originals，以及生成 PDF、重複 LaTeX appendix、未使用 package changelog template 與 editor workspace artifact。Git 仍是歷史 archive。

Reboot branch 只貢獻一份 `proposed` design；此 milestone 不會將提案分類為目前實作。Windows launcher design 與 plan 是 approved/active records，但此文件分支不宣稱已完成其實作。

<a id="tooling-and-agents"></a>
## 工具與 agents

`tools.documentation` 介面驗證名稱、metadata、IDs、生命週期位置、配對、anchors、links 與 changed-pair parity，並可產生 inventory 及 stage site sources。Pre-commit 與 CI 執行結構契約及 pull-request documentation-impact declaration。

Repository-local 撰寫與審查 skills 為 Codex 提供單一正式流程。精簡 Claude adapters 指向相同 skills，避免把文件政策複製到互相競爭的 instruction files。

<a id="publication"></a>
## 發布

固定版本的 MkDocs、Material 與 static-i18n dependencies 會建立 suffix-based English 與 Traditional Chinese routes。語言切換會保留對等頁面，search index 同時包含兩種語言。既有 remote Training Panel 保持在 Pages artifact root；文件網站發布於 `/docs/`。生成 HTML 不會納入追蹤。

<a id="compatibility-and-boundaries"></a>
## 相容性與邊界

此 milestone 變更文件、驗證工具、agent instructions 與 publication workflows。它不重寫 Git history、不 retarget branches、不匯入未提交的 Windows launcher code、不改變 training/runtime behavior，也不宣稱已執行 ROS hardware procedures。只有在 `docs-reorg-v1` checkpoint 後，才會以獨立審查專案開始 Git history 與 branch reorganization。
