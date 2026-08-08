---
id: naming-conventions
title: 文件命名慣例
lang: zh-TW
audience: developer
type: reference
status: active
owner: project
last_reviewed: 2026-08-01
---

<a id="canonical-patterns"></a>
## 正式命名模式

- 一般文件：`lowercase-kebab-case.<locale>.md`
- 區段首頁：`index.<locale>.md`
- 時序文件：`YYYY-MM-DD-slug.<locale>.md`
- 架構決策紀錄：`adr-0001-slug.<locale>.md`

請以 `en` 或拼寫與大小寫完全一致的 `zh-TW` 取代 `<locale>`。Slug 使用小寫 kebab case：小寫單字間以單一連字號分隔，不含空白或底線。

<a id="date-use"></a>
## 日期使用時機

只有當時間順序屬於文件本質時，檔名才包含日期，例如有日期的計畫、設計、里程碑或時限稽核。穩定知識即使審查日期改變，仍使用一般檔名。

<a id="readme-exception"></a>
## README 例外

根目錄與元件 `README.md` 保留慣用的大寫檔名且不使用語言後綴，因為它們是簡短的單檔雙語路由。它們不是保存持續維護詳細知識的正式容器。

<a id="colocated-documents"></a>
## 並置文件

與元件並置的文件遵守與中央文件相同的檔名模式、語言配對、中繼資料及明確錨點規則。並置不構成命名或雙語例外。
