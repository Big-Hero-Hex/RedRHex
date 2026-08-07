---
id: documentation-system-v1-release
title: Documentation System V1 Milestone
lang: en
audience: shared
type: release
status: published
owner: project
last_reviewed: 2026-08-07
---

<a id="scope"></a>
## Scope

This dated project milestone establishes the first maintained RedRHex documentation system. It is not a global RedRHex semantic version and does not change the independently versioned Training Panel release.

<a id="published-system"></a>
## Published system

- The root README routes to distinct operator and developer journeys.
- Maintained human-facing knowledge is canonical in paired English and Traditional Chinese files, with short bilingual README routers as the documented exception.
- Central portals cover operator workflows, developer architecture and extension, references, decisions, active designs and plans, roadmap, releases, research evidence, and governance.
- Detailed Training Panel, Reward Agent, and ROS 2 deployment documentation remains beside the owning code and is included in the central portal and generated site.
- Metadata, naming, lifecycle, translation, templates, documentation-impact declarations, stale thresholds, and migration traceability have enforceable repository contracts.

<a id="migration"></a>
## Migration

Durable content from the legacy command guide, training/play guide, reports, sim-to-real material, Reward Agent work, Windows launcher work, Training Panel manuals/changelogs, and ROS monolith was curated into canonical documentation. Redundant originals and a generated PDF, repeated LaTeX appendix, unused package changelog template, and editor workspace artifact were removed after heading-level dispositions were recorded. Git remains the archive.

The reboot branch contributed only a `proposed` design; this milestone does not classify that proposal as current implementation. The Windows launcher design and plan are approved/active records, but its implementation is not claimed on this documentation branch.

<a id="tooling-and-agents"></a>
## Tooling and agents

The `tools.documentation` interface validates names, metadata, IDs, lifecycle locations, pairs, anchors, links, and changed-pair parity; it also produces inventory and stages site sources. Pre-commit and CI enforce structural contracts and the pull-request documentation-impact declaration.

Repository-local authoring and review skills provide one canonical workflow for Codex. Thin Claude adapters point to the same skills so documentation policy is not copied into competing instruction files.

<a id="publication"></a>
## Publication

Pinned MkDocs, Material, and static-i18n dependencies build suffix-based English and Traditional Chinese routes. Language switching preserves the equivalent page, and the search index includes both locales. The existing remote Training Panel remains at the Pages artifact root; the documentation site is published under `/docs/`. Generated HTML is not tracked.

<a id="compatibility-and-boundaries"></a>
## Compatibility and boundaries

This milestone changes documentation, validation tooling, agent instructions, and publication workflows. It does not rewrite Git history, retarget branches, import uncommitted Windows launcher code, alter training/runtime behavior, or claim that ROS hardware procedures were executed. Git history and branch reorganization begins only as a separately reviewed project after the `docs-reorg-v1` checkpoint.
