---
id: metadata-schema
title: 文件中繼資料結構描述
lang: zh-TW
audience: developer
type: reference
status: active
owner: project
last_reviewed: 2026-08-01
---

<a id="required-fields"></a>
## 必要欄位

每份正式且持續維護的文件都必須具備且完整使用下列 frontmatter 欄位：

- `id`：穩定識別資訊；在整個儲存庫內唯一，並由語言配對共用。
- `title`：在地化、供人員閱讀的標題。
- `lang`：文件語言。
- `audience`：主要讀者路徑。
- `type`：文件類型與用途。
- `status`：所選文件類型允許的生命週期狀態。
- `owner`：負責技術正確性與審查的團隊；擁有權不代表只能由該團隊撰寫。
- `last_reviewed`：最近一次實質審查日期，使用 ISO `YYYY-MM-DD` 格式。

配對中繼資料除 `title` 與 `lang` 外必須相同。英文檔使用 `lang: en`；繁體中文檔使用拼寫與大小寫完全一致的 `lang: zh-TW`。

<a id="allowed-values"></a>
## 允許值

- `lang`：`en`、`zh-TW`
- `audience`：`operator`、`developer`、`shared`
- `owner`：`project`、`core`、`training`、`panel`、`deployment`、`sim2real`、`reward-agent`
- `type`：`index`、`tutorial`、`how-to`、`reference`、`explanation`、`safety`、`troubleshooting`、`decision`、`design`、`plan`、`roadmap`、`release`、`experiment-summary`、`audit`

<a id="status-by-type"></a>
## 各類型允許的狀態

- 知識類型（`index`、`tutorial`、`how-to`、`reference`、`explanation`、`safety`、`troubleshooting`）：`draft`、`active`、`deprecated`
- `decision`：`accepted`、`superseded`
- `design`：`proposed`、`approved`、`implemented`、`rejected`、`superseded`
- `plan`：`draft`、`active`、`blocked`、`completed`、`cancelled`
- `roadmap`：`active`
- `release`、`experiment-summary` 與 `audit`：`published`

位置、類型與狀態必須相符。某一生命週期的狀態不得用於另一生命週期的文件。

<a id="paired-example"></a>
## 配對範例

有效的英文檔：

```yaml
---
id: locomotion-architecture
title: Locomotion Architecture
lang: en
audience: developer
type: explanation
status: active
owner: core
last_reviewed: 2026-08-01
---
```

有效的繁體中文配對檔：

```yaml
---
id: locomotion-architecture
title: 運動架構
lang: zh-TW
audience: developer
type: explanation
status: active
owner: core
last_reviewed: 2026-08-01
---
```

<a id="validation-boundary"></a>
## 驗證邊界

正式 frontmatter 驗證適用於持續維護且面向人員的文件，包括中央與並置配對。範本與機器產生檔不屬於正式 frontmatter 驗證範圍；其結構可由特定用途工具檢查。
