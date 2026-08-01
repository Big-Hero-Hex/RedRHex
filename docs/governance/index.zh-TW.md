---
id: documentation-governance
title: 文件治理
lang: zh-TW
audience: developer
type: index
status: active
owner: project
last_reviewed: 2026-08-01
---

<a id="purpose"></a>
## 目的

這些治理文件是人員與代理程式使用的 RedRHex 文件規則唯一真實來源。英文與繁體中文均為同等正式來源。

<a id="documentation-portals"></a>
## 文件入口

- [文件首頁](../index.zh-TW.md)
- [操作人員文件](../operators/index.zh-TW.md)
- [開發人員文件](../developers/index.zh-TW.md)

<a id="governance-documents"></a>
## 治理文件

- [文件政策](documentation-policy.zh-TW.md)定義範圍、讀者、放置方式及文件類型。
- [中繼資料結構描述](metadata-schema.zh-TW.md)定義必要 frontmatter 與允許值。
- [命名慣例](naming-conventions.zh-TW.md)定義正式檔名。
- [文件生命週期](document-lifecycle.zh-TW.md)定義狀態流轉、過時警告、保留及移除規則。
- [翻譯指南](translation-guide.zh-TW.md)定義雙語一致性與配對審查。
- [文件影響](documentation-impact.zh-TW.md)把儲存庫變更對應至文件工作與 PR 宣告。
- [README 路由慣例](readme-router-convention.zh-TW.md)定義簡短雙語 README 例外。

<a id="consumers-and-enforcement"></a>
## 使用者與強制執行

根目錄 `AGENTS.md` 與 `CLAUDE.md`、儲存庫技能、hooks 及 CI 使用或引用這套治理規則，但不重新定義規則。代理程式指示改善撰寫行為；驗證器、pre-commit hook 與 CI 則提供不依賴代理程式的強制執行。
