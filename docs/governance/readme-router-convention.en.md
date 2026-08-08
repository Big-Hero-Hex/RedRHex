---
id: readme-router-convention
title: README Router Convention
lang: en
audience: developer
type: reference
status: active
owner: project
last_reviewed: 2026-08-01
---

<a id="exception"></a>
## Bilingual router exception

Root and component `README.md` files are concise, single-file bilingual routers. They are the only human-facing exception to paired `.en.md` and `.zh-TW.md` files. A router should stay roughly within one screen and help a new reader choose the next canonical page.

<a id="root-router"></a>
## Root router

The root `README.md` must:

- identify RedRHex in English and Traditional Chinese;
- offer English and Traditional Chinese links for both operator and developer entry paths;
- avoid volatile commands, versions, configuration values, and maintained procedural detail.

The established English destinations are the [documentation home](../index.en.md), [Operator Documentation](../operators/index.en.md), and [Developer Documentation](../developers/index.en.md); the root router also links their Traditional Chinese equivalents.

<a id="component-router"></a>
## Component router

A component `README.md` must identify the component's purpose and owner in both languages, then link to its colocated operator, developer, reference, and release documentation when those families exist.

<a id="knowledge-boundary"></a>
## Knowledge boundary

Detailed maintained knowledge belongs in paired canonical files, not in a README. The router may contain only enough stable context to identify the destination and must not become a second policy, setup guide, architecture explanation, or changelog.

<a id="review-checklist"></a>
## Review checklist

- Both languages identify the same project or component and owner.
- Operator and developer destinations are easy to distinguish.
- Locale links reach equivalent canonical pages and relative links resolve.
- No volatile commands, versions, configuration, or detailed maintained knowledge appears.
- The router remains roughly one screen.
