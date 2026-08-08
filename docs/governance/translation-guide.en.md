---
id: translation-guide
title: Documentation Translation Guide
lang: en
audience: developer
type: reference
status: active
owner: project
last_reviewed: 2026-08-01
---

<a id="canonical-equality"></a>
## Canonical equality

English and Traditional Chinese are equal canonical sources. Neither committed file is generated from or subordinate to the other. Translation requires meaning equivalence, not literal sentence equivalence; natural wording is preferred when it preserves the same facts, intent, constraints, and reader outcome.

<a id="change-together"></a>
## Change together

Meaning-changing edits update both locale files in the same change. A localized-only typo or grammar correction may update one locale only, but the commit or PR reason must explicitly declare it as a localized-only correction.

<a id="pair-contract"></a>
## Pair contract

Nonlocalized frontmatter must match exactly; only `title` and `lang` differ. Corresponding headings must have the same sequence of explicit HTML anchor IDs. Code, command names, paths, identifiers, versions, ports, units, links, evidence, warnings, and safety constraints must not drift in translation.

<a id="pair-review"></a>
## Pair-review checklist

- Confirm both locale files changed for every meaning change.
- Compare all nonlocalized frontmatter values.
- Compare explicit anchor IDs in order.
- Check commands, paths, identifiers, versions, ports, units, links, evidence, warnings, and safety constraints.
- Read each locale independently for equivalent meaning and a complete reader journey.
