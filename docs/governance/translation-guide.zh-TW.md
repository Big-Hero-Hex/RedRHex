---
id: translation-guide
title: 文件翻譯指南
lang: zh-TW
audience: developer
type: reference
status: active
owner: project
last_reviewed: 2026-08-01
---

<a id="canonical-equality"></a>
## 正式來源同等地位

英文與繁體中文是同等正式來源。任何已提交檔案都不是由另一語言產生，也不從屬於另一語言。翻譯要求意義對等，而非逐句字面對等；只要保留相同事實、意圖、限制與讀者結果，應優先使用自然的表達方式。

<a id="change-together"></a>
## 同步變更

改變意義的編輯必須在同一次變更中更新兩個語言檔。只涉及單一語言的錯字或文法修正可以只更新該語言，但 commit 或 PR 理由必須明確宣告這是僅限在地化內容的修正。

<a id="pair-contract"></a>
## 配對契約

非在地化 frontmatter 必須完全一致；只有 `title` 與 `lang` 不同。對應標題必須使用順序相同的明確 HTML 錨點 ID。程式碼、命令名稱、路徑、識別符、版本、連接埠、單位、連結、證據、警告及安全限制不得在翻譯中偏移。

<a id="pair-review"></a>
## 配對審查清單

- 確認每項意義變更都同時修改兩個語言檔。
- 比較所有非在地化 frontmatter 值。
- 依序比較明確錨點 ID。
- 檢查命令、路徑、識別符、版本、連接埠、單位、連結、證據、警告及安全限制。
- 分別閱讀各語言，確認意義對等且讀者旅程完整。
