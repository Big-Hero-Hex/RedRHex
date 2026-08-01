---
id: documentation-governance
title: Documentation Governance
lang: en
audience: developer
type: index
status: active
owner: project
last_reviewed: 2026-08-01
---

<a id="purpose"></a>
## Purpose

These governance documents are the single source of truth for RedRHex documentation rules for humans and agents. English and Traditional Chinese are equal canonical sources.

<a id="governance-documents"></a>
## Governance documents

- [Documentation policy](documentation-policy.en.md) defines scope, audiences, placement, and document families.
- [Metadata schema](metadata-schema.en.md) defines required frontmatter and allowed values.
- [Naming conventions](naming-conventions.en.md) defines canonical filenames.
- [Document lifecycle](document-lifecycle.en.md) defines status transitions, staleness, retention, and removal.
- [Translation guide](translation-guide.en.md) defines bilingual parity and pair review.
- [Documentation impact](documentation-impact.en.md) maps repository changes to documentation work and PR declarations.
- [README router convention](readme-router-convention.en.md) defines the concise bilingual README exception.

<a id="consumers-and-enforcement"></a>
## Consumers and enforcement

Root `AGENTS.md` and `CLAUDE.md`, repository skills, hooks, and CI consume or reference this governance. They do not redefine it. Agent instructions improve authoring behavior; the validator, pre-commit hook, and CI provide agent-independent enforcement.
