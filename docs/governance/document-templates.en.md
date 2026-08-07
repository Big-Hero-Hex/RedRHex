---
id: document-templates
title: Documentation Templates
lang: en
audience: developer
type: reference
status: active
owner: project
last_reviewed: 2026-08-07
---

<a id="purpose"></a>
## Purpose

These templates provide the required starting structure for each maintained document family. Copy the template that matches the document's purpose and locale; do not treat the examples as project facts.

<a id="template-selection"></a>
## Choose a template

- Use the [index template](https://github.com/JasonLiaoJCS/RedRHex/blob/main/docs/governance/templates/index.en.md.template) for a portal or section landing page.
- Use the [knowledge template](https://github.com/JasonLiaoJCS/RedRHex/blob/main/docs/governance/templates/knowledge.en.md.template) for `tutorial`, `how-to`, `reference`, `explanation`, `safety`, or `troubleshooting`. Select exactly one `type` and one status allowed for that type by governance.
- Use the [decision template](https://github.com/JasonLiaoJCS/RedRHex/blob/main/docs/governance/templates/decision.en.md.template) for an ADR.
- Use the [design template](https://github.com/JasonLiaoJCS/RedRHex/blob/main/docs/governance/templates/design.en.md.template) for a proposed or approved feature or material change.
- Use the [plan template](https://github.com/JasonLiaoJCS/RedRHex/blob/main/docs/governance/templates/plan.en.md.template) for a multi-step implementation.
- Use the [roadmap template](https://github.com/JasonLiaoJCS/RedRHex/blob/main/docs/governance/templates/roadmap.en.md.template) for current priorities and unresolved future work.
- Use the [release template](https://github.com/JasonLiaoJCS/RedRHex/blob/main/docs/governance/templates/release.en.md.template) for shipped behavior or a dated project milestone.
- Use the [experiment-summary template](https://github.com/JasonLiaoJCS/RedRHex/blob/main/docs/governance/templates/experiment-summary.en.md.template) for evidence that changes a baseline, recommendation, decision, or result.
- Use the [audit template](https://github.com/JasonLiaoJCS/RedRHex/blob/main/docs/governance/templates/audit.en.md.template) for a durable review or compliance finding.

<a id="authoring-contract"></a>
## Authoring contract

Replace every angle-bracket placeholder before publishing. Create both locale files in the same change, preserve identical nonlocalized metadata and explicit anchor IDs, and translate for equivalent meaning. Select exact metadata values and lifecycle status from the [metadata schema](metadata-schema.en.md). Once Phase 3 provides it, run `python -m tools.documentation validate --staged` before committing.

<a id="template-assets"></a>
## Template assets

Templates intentionally end in `.md.template`; they are authoring assets, not site pages. Never rename one into canonical output without replacing every placeholder. Use the site language switcher to open this catalog in Traditional Chinese.
