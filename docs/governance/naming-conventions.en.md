---
id: naming-conventions
title: Documentation Naming Conventions
lang: en
audience: developer
type: reference
status: active
owner: project
last_reviewed: 2026-08-01
---

<a id="canonical-patterns"></a>
## Canonical patterns

- Normal documents: `lowercase-kebab-case.<locale>.md`
- Section landing pages: `index.<locale>.md`
- Chronological documents: `YYYY-MM-DD-slug.<locale>.md`
- Architecture Decision Records: `adr-0001-slug.<locale>.md`

Replace `<locale>` with `en` or the exact spelling and case `zh-TW`. Slugs use lowercase kebab case: lowercase words separated by single hyphens, with no spaces or underscores.

<a id="date-use"></a>
## Date use

Dates belong in filenames only when chronology is intrinsic to the document, such as a dated plan, design, milestone, or time-bound audit. Stable knowledge uses a normal filename even when its review date changes.

<a id="readme-exception"></a>
## README exception

Root and component `README.md` files keep that conventional uppercase filename and have no locale suffix because they are concise single-file bilingual routers. They are not canonical containers for detailed maintained knowledge.

<a id="colocated-documents"></a>
## Colocated documents

Documents colocated beside components follow the same filename patterns, locale pairing, metadata, and explicit-anchor rules as central documentation. Colocation does not create a naming or bilingual exception.
