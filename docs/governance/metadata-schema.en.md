---
id: metadata-schema
title: Documentation Metadata Schema
lang: en
audience: developer
type: reference
status: active
owner: project
last_reviewed: 2026-08-01
---

<a id="required-fields"></a>
## Required fields

Every canonical maintained document has exactly these required frontmatter fields:

- `id`: stable identity, unique across the entire repository and shared by a locale pair.
- `title`: localized human-readable title.
- `lang`: document locale.
- `audience`: primary reader path.
- `type`: document family and purpose.
- `status`: lifecycle state allowed for the selected type.
- `owner`: team responsible for technical correctness and review; ownership does not imply sole authorship.
- `last_reviewed`: most recent substantive review date in ISO `YYYY-MM-DD` form.

Pair metadata must be identical except for `title` and `lang`. The English file uses `lang: en`; the Traditional Chinese file uses the exact spelling and case `lang: zh-TW`.

<a id="allowed-values"></a>
## Allowed values

- `lang`: `en`, `zh-TW`
- `audience`: `operator`, `developer`, `shared`
- `owner`: `project`, `core`, `training`, `panel`, `deployment`, `sim2real`, `reward-agent`
- `type`: `index`, `tutorial`, `how-to`, `reference`, `explanation`, `safety`, `troubleshooting`, `decision`, `design`, `plan`, `roadmap`, `release`, `experiment-summary`, `audit`

<a id="status-by-type"></a>
## Status by type

- Knowledge types (`index`, `tutorial`, `how-to`, `reference`, `explanation`, `safety`, `troubleshooting`): `draft`, `active`, `deprecated`
- `decision`: `accepted`, `superseded`
- `design`: `proposed`, `approved`, `implemented`, `rejected`, `superseded`
- `plan`: `draft`, `active`, `blocked`, `completed`, `cancelled`
- `roadmap`: `active`
- `release`, `experiment-summary`, and `audit`: `published`

Location, type, and status must agree. A status from one lifecycle is invalid for a document in another lifecycle.

<a id="paired-example"></a>
## Paired example

Valid English file:

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

Valid Traditional Chinese pair:

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
## Validation boundary

Canonical frontmatter validation applies to maintained human-facing documentation, including central and colocated pairs. Templates and machine-generated files are outside canonical frontmatter validation. Their structure may be checked by purpose-specific tooling instead.
