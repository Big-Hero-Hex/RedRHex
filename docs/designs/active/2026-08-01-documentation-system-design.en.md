---
id: documentation-system-design
title: RedRHex Bilingual Documentation System Design
lang: en
audience: developer
type: design
status: approved
owner: project
last_reviewed: 2026-08-01
---

<a id="summary"></a>
## Summary

RedRHex will maintain two distinct bilingual entry paths: **Operator Documentation** for people who set up, run, calibrate, deploy, or troubleshoot the robot, and **Developer Documentation** for people who understand, test, extend, or maintain the software. English and Traditional Chinese are equally canonical; neither locale is a derivative or optional translation.

Markdown is the source of truth. Placement is hybrid: the system combines central portals, policy, roadmaps, and architecture with detailed documentation colocated beside components. A checked-in site manifest includes both central and colocated material in the published site.

<a id="language-and-scope"></a>
## Language model and scope

Canonical maintained documents are pairs named with `.en.md` and `.zh-TW.md` suffixes. Root and component `README.md` files are short bilingual routers to the two locale entry points; they are the only intentional single-file bilingual exception.

The bilingual requirement covers current operator and developer knowledge, shared references, governance, designs, plans, ADRs, roadmaps, releases, and published evidence. Automation, generated files, skills, raw run notes, and deleted historical material do not require translation.

<a id="information-architecture"></a>
## Information architecture

The target structure is:

```text
README.md
docs/
├── index.en.md
├── index.zh-TW.md
├── operators/{index.*,getting-started.*,training/,panel/,deployment/,calibration/,troubleshooting/}
├── developers/{index.*,architecture/,development/,testing/,subsystems/}
├── reference/
├── decisions/
├── designs/active/
├── plans/active/
├── roadmap/
├── releases/
├── research/
└── governance/
```

Central portals own navigation, policy, roadmap, and cross-cutting architecture. Component repositories or directories own detailed, component-specific material. The central portal and site manifest include those colocated documents so readers have one navigable system without separating documentation from the code it explains.

<a id="naming"></a>
## Naming

- General documents use `lowercase-kebab-case.<locale>.md`.
- Section landing pages use `index.<locale>.md`.
- Time-bound documents use `YYYY-MM-DD-slug.<locale>.md`.
- ADRs use `adr-0001-slug.<locale>.md`.
- Dates appear only when chronology is intrinsic to the document.

<a id="metadata"></a>
## Metadata

Every maintained document has YAML frontmatter with these required fields:

```yaml
id: stable-identity
title: Localized title
lang: en
audience: developer
type: explanation
status: active
owner: project
last_reviewed: 2026-08-01
```

Allowed values are:

- `lang`: `en`, `zh-TW`
- `audience`: `operator`, `developer`, `shared`
- `owner`: `project`, `core`, `training`, `panel`, `deployment`, `sim2real`, `reward-agent`
- `type`: `index`, `tutorial`, `how-to`, `reference`, `explanation`, `safety`, `troubleshooting`, `decision`, `design`, `plan`, `roadmap`, `release`, `experiment-summary`, `audit`
- knowledge status: `draft`, `active`, `deprecated`
- decision status: `accepted`, `superseded`
- design status: `proposed`, `approved`, `implemented`, `rejected`, `superseded`
- plan status: `draft`, `active`, `blocked`, `completed`, `cancelled`
- roadmap status: `active`
- release, experiment, and audit status: `published`

Location, type, and status must agree. A status from one lifecycle is invalid for a document in another lifecycle.

<a id="pair-contract"></a>
## Bilingual pair contract

The English and Traditional Chinese files in a pair have identical metadata except for `title` and `lang`. Meaning-changing edits update both files together. Corresponding headings are preceded by stable, explicit, matching HTML anchor IDs. Translations preserve meaning, evidence, constraints, warnings, commands, paths, and links rather than copying sentence structure literally.

<a id="lifecycle"></a>
## Lifecycle and retention

- Maintained knowledge moves `draft -> active -> deprecated -> removed`.
- An active design is resolved into maintained architecture, an ADR, a release, or a roadmap item and is then removed from `designs/active/`.
- A completed or cancelled plan is summarized into durable documentation and then removed from `plans/active/`.
- ADRs remain in the repository and are superseded rather than rewritten.
- Roadmaps contain current priorities; completed work moves to releases or milestone records.
- Experiment summaries are immutable. Corrections are dated addenda rather than silent rewrites.
- Raw run logs and notes are ignored. A summary is published only when evidence changes a baseline, recommendation, decision, or result.

Git is the archive for removed working documents; the live tree is not an attic.

<a id="staleness"></a>
## Staleness policy

- Operator, reference, and roadmap documents warn after 90 days without review.
- Developer architecture documents warn after 180 days without review.
- Decisions and releases warn only when contradicted or superseded.

Warnings prompt review; they do not automatically change a document's truth status.

<a id="documentation-impact"></a>
## Documentation impact rules

Changes trigger documentation work as follows:

- CLI, workflow, environment, path, setup, hardware, or safety changes update operator or reference documentation.
- API, schema, architecture, dependency, testing, or extension changes update developer or reference documentation.
- Shipped behavior or compatibility changes update the affected component release.
- Cross-cutting decisions require an ADR.
- Approved features require a design.
- Multi-step implementations require a plan.
- Internal refactors require a recorded `Docs impact: none` reason.

The canonical governance documents carry these durable impact rules. Root `AGENTS.md` links to that governance rather than duplicating it.

<a id="tooling"></a>
## Validation and inventory tool

The documentation CLI is:

```text
python -m tools.documentation validate --all
python -m tools.documentation validate --staged
python -m tools.documentation validate --changed-from REF
python -m tools.documentation inventory --format json
python -m tools.documentation stage-site --output DIR
```

Commands exit `0` on success and validation exits `1` on failure. Validation covers file names, required frontmatter and enums, unique IDs, lifecycle and location, bilingual pairs and metadata parity, links and anchors, and changed-pair parity.

<a id="pull-requests"></a>
## Pull-request declaration and CI

Each pull request declares:

```text
Docs impact: none | operator | developer | shared | release | experiment
Docs reason: <required explanation>
```

CI enforces the declaration's presence and structure. It does not infer semantic impact from source paths. Pre-commit validation uses `--staged`; CI validates `--all`.

<a id="repository-skills"></a>
## Repository skills

Two repository skills live under `.agents/skills`: `writing-redrhex-docs` and `reviewing-redrhex-docs`. They guide authors and reviewers but reference the canonical governance documents instead of copying policy into the skills.

<a id="cross-agent-adapters"></a>
## Cross-agent adapters

Governance Markdown is the neutral source of truth for documentation policy. Root `AGENTS.md` is the broad, open agent adapter used by Codex; root `CLAUDE.md` imports it for Claude Code rather than duplicating policy.

Canonical skill implementations remain under `.agents/skills`. Thin `.claude/skills` wrappers provide Claude Code discovery and direct it to the canonical skill files without copying workflow policy. The documentation validator, pre-commit hook, and CI are the cross-agent guarantee: they reject nonconforming output independently of the authoring tool.

<a id="site"></a>
## Documentation site

Markdown remains canonical and generated HTML is ignored. The site uses pinned MkDocs, Material for MkDocs, and `mkdocs-static-i18n` dependencies. A checked-in manifest stages central and colocated documentation. The panel UI remains the GitHub Pages root, while documentation is published under `/docs/`.

Both locales are built. Each page provides an equivalent-page language switch, search indexes both languages, and strict link checking prevents publishing broken internal navigation.

<a id="release-model"></a>
## Release model

The panel keeps SemVer and its component changelog. A central release index links component releases, while whole-project milestones are dated rather than assigned a project-wide SemVer. Versions `3.4.4` through `3.4.9` must not be invented for the panel; evidenced changes are consolidated in `V3.4.10`.

<a id="historical-migration"></a>
## Historical migration

Migration curates durable content instead of copying old files wholesale. A migration manifest records every old heading, its disposition, replacement document or documents, and the commit that removes it. Redundant originals are deleted only after replacements and traceability validate. Git provides the historical archive.

<a id="safety-and-cleanup"></a>
## Safety and later cleanup

Safety documentation follows normal review and has no separate human safety gate. Git history and branch cleanup are a later, separately reviewed project and are outside this documentation reorganization.
