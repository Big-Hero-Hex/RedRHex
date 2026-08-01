---
id: migration-manifest
title: 遷移清單
lang: zh-TW
audience: developer
type: reference
status: active
owner: project
last_reviewed: 2026-08-01
---

<a id="purpose"></a>
## 目的

[遷移清單](migration-manifest.csv)是來源文件遷移所使用、可由機器讀取且不依賴語言的可追溯帳冊。原始來源標題須逐字保留，不翻譯。

<a id="provenance-and-coverage"></a>
## 來源與涵蓋範圍

清冊只從 annotated tag `docs-reorg-source-2026-08-01`、commit `5de992d7afac77e566b6deebb99dc813eb87b612` 產生。每份來源都有一筆文件根紀錄，且每個 Markdown ATX 標題、reStructuredText 底線標題，以及 LaTeX `part`、`chapter`、`section`、`subsection` 或 `subsubsection` 都各有一筆紀錄。重複標題文字以 `source_path` 與 `source_line` 區分。

<a id="row-contract"></a>
## 紀錄契約

CSV 欄位為 `source_path`、`source_line`、`heading_level`、`source_heading`、`disposition`、`replacement_ids` 及 `removal_commit`。紀錄依核准的來源路徑順序排列，再依來源行號數值排列。文件根紀錄使用行號 `0`、層級 `0` 與標題 `<document>`。結構標記須移除，標題文字則逐字保留。

<a id="dispositions-and-removal"></a>
## 處置方式與移除

允許的處置方式只有 `pending`、`migrated`、`obsolete`、`duplicate` 及 `git-history-only`。移除來源前，其每筆紀錄都必須不再是 `pending`。`migrated` 紀錄必須列出一個以上的正式替代文件 ID；只有刪除來源後才填寫來源移除 commit。

<a id="field-format"></a>
## 欄位格式

`replacement_ids` 以分號分隔正式文件 ID。`removal_commit` 是完整 Git commit hash；來源仍保留時留白。帳冊是 RFC 4180 相容的 UTF-8 CSV，使用 `\n` 行尾。
