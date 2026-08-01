---
id: documentation-system-reorganization
title: RedRHex Documentation System Reorganization Plan
lang: en
audience: developer
type: plan
status: active
owner: project
last_reviewed: 2026-08-02
---

<a id="objective"></a>
## Objective

Implement the approved bilingual documentation system without losing durable knowledge, breaking operator or developer journeys, or disturbing the existing panel application. Execute the phases in order and do not remove a source document until its paired replacements and heading-level traceability validate.

<a id="checkpoint-context"></a>
## Checkpoint context

The source is commit `5de992d`. Baseline test results recorded at that commit are:

- Reward Agent: `14` passed.
- Training panel: `208` passed.
- Sim-to-real: `528` passed.

Preserve these results as checkpoint evidence; later failures must be explained or resolved before completion.

<a id="phase-1-isolation"></a>
## Phase 1 — Isolation and written checkpoint

- [x] Create a dedicated worktree and branch from `5de992d`; preserve all unrelated worktrees.
- [x] Add the annotated source tag `docs-reorg-source-2026-08-01` at the source commit.
- [x] Persist the approved English and Traditional Chinese design pair and this implementation-plan pair.
- [x] Manually compare each pair's nonlocalized frontmatter, explicit anchor IDs, constraints, commands, and acceptance criteria.
- [x] Hold a written-spec review checkpoint and record approval before migration or tooling implementation begins.

**Phase acceptance:** the branch is isolated at the correct source, unrelated worktrees are unchanged, the annotated source tag resolves to `5de992d`, all four canonical checkpoint documents validate as equivalent pairs, and written approval is recorded.

<a id="phase-2-governance"></a>
## Phase 2 — Governance and target tree

- [x] Create bilingual operator and developer portals and the approved central directory tree.
- [x] Define the frontmatter schema, enums, lifecycle/location rules, naming rules, staleness policy, and documentation-impact policy in governance documents.
- [x] Add templates for every maintained document family and a translation guide covering meaning parity and stable explicit anchors.
- [x] Define the short bilingual root/component `README.md` router convention.
- [x] Add root `AGENTS.md` as a thin operational adapter that points agents to canonical governance.
- [x] Add root `CLAUDE.md` with an `@AGENTS.md` import so Claude Code receives the same repository requirements.
- [x] Create a heading-level migration manifest with columns for old path, old heading, disposition, replacement document or documents, and removal commit.
- [x] Add every heading from every source document to the manifest before migrating or deleting that source.

**Phase acceptance:** portals and governance are paired and linked; schema examples pass the documented rules; router and translation conventions are unambiguous; root agent adapters point to canonical governance without policy duplication; and every in-scope old heading has a manifest row before content migration starts.

<a id="phase-3-validator"></a>
## Phase 3 — Validator, hooks, and CI through TDD

- [x] Write failing unit tests before each validator behavior is implemented.
- [x] Implement these interfaces exactly:

```text
python -m tools.documentation validate --all
python -m tools.documentation validate --staged
python -m tools.documentation validate --changed-from REF
python -m tools.documentation inventory --format json
python -m tools.documentation stage-site --output DIR
```

- [x] Return exit `0` on success and exit `1` on validation failure.
- [x] Validate names, required frontmatter and enums, unique IDs, lifecycle/location compatibility, bilingual pair presence and metadata parity, links and anchors, and changed-pair parity.
- [x] Add inventory reports suitable for migration, ownership, staleness, and CI evidence.
- [x] Run `validate --staged` in pre-commit and `validate --all` in CI.
- [x] Require the structured PR fields `Docs impact: none | operator | developer | shared | release | experiment` and `Docs reason: <required explanation>`.
- [x] Enforce declaration presence and shape without inferring semantic impact from source paths.

- [x] **Phase acceptance:** tests demonstrate failure for bad names, missing or invalid frontmatter, duplicate IDs, invalid lifecycle/location combinations, missing pairs, pair metadata drift, changed-only-one-locale edits, broken links, missing anchors, and mismatched anchors; all positive fixtures pass; CLI exit codes match the contract; inventory output is machine-readable; and pre-commit/CI gates are fast and balanced rather than duplicating costly unrelated test suites.

<a id="phase-4-central-migration"></a>
## Phase 4 — Central documentation migration

- [ ] Split `docs/COMMANDS.md` and `docs/redrhex_train_play_guide.md` into bilingual operator setup, training, evaluation, export-video, and troubleshooting guides plus shared command and version references.
- [ ] Verify every migrated command, version, and path against the code before publishing it.
- [ ] Extract durable architecture, decisions, validated results, citations, and unresolved work from the midterm, meeting, training, ForwardFast, improvement, energy, and July review reports.
- [ ] For energy material, retain the verified energy model, reward rationale, limitations, and validation; drop snapshots and the duplicated LaTeX appendix.
- [ ] Split sim-to-real content into operator calibration guidance and developer evidence architecture.
- [ ] Update every manifest row with its disposition and paired replacement links.
- [ ] Remove an original only after both replacements, links, anchors, commands, and traceability validate; record the removal commit in the manifest.

**Phase acceptance:** complete bilingual operator journeys cover setup through training, evaluation, video export, calibration, deployment, and troubleshooting; complete bilingual developer journeys cover architecture, development, testing, subsystems, decisions, and evidence; commands, versions, and paths match code; every removed heading is represented in the migration manifest; and no redundant original remains without an explicit disposition.

<a id="phase-5-project-components"></a>
## Phase 5 — Project and component migration

- [ ] Move the implemented Reward Agent foundation and weight-tuning knowledge into maintained documentation; place unfinished proposal UI and panel integration work on the roadmap.
- [ ] Migrate the Windows remote launcher design and plan into bilingual active pairs.
- [ ] Preserve the reboot branch and classify its unique content as `proposed`, not `active`.
- [ ] Produce paired panel documentation, a paired manual, and a consolidated evidenced `3.4.10` release; do not invent versions `3.4.4` through `3.4.9`.
- [ ] Split the monolithic ROS README into colocated bilingual architecture, policy contract, bring-up, deployment, and troubleshooting documents, leaving a short bilingual router `README.md`.
- [ ] Remove the unused package changelog stub after confirming it has no durable content.
- [ ] Keep ignored runtime and generated artifacts out of Git throughout migration.

**Phase acceptance:** component portals link every colocated pair; implemented and unfinished work are separated correctly; the reboot proposal remains recoverable; the panel release is evidence-backed and version-correct; the ROS router reaches each paired guide; the unused stub is accounted for in the manifest; and Git contains no generated or runtime artifacts.

<a id="phase-6-skills"></a>
## Phase 6 — Repository skills, tested sequentially

- [ ] Implement `writing-redrhex-docs` and `reviewing-redrhex-docs` one at a time; each references governance instead of copying it.
- [ ] Add thin `.claude/skills` wrappers for both skills; each directs Claude Code to the canonical `.agents/skills` implementation without copying workflow policy.
- [ ] For each skill, record a RED baseline without the skill, then a GREEN run with the skill.
- [ ] Inspect failures and loopholes, refine the skill, and rerun the same scenario.
- [ ] Repeat at least five micro-tests per skill with a no-guidance control.
- [ ] Cover operator, developer, release, ADR, design, plan, experiment, and no-impact cases across the scenario suite.
- [ ] Test that every Claude wrapper path resolves to its canonical skill file and that Claude follows the canonical workflow.

**Phase acceptance:** both skills show reproducible improvement over their no-guidance controls; each has at least five recorded micro-test repetitions; every required scenario class is exercised; Claude wrapper paths resolve and use the canonical implementations; loophole fixes are documented; and outputs comply with governance without policy duplication.

<a id="phase-7-site"></a>
## Phase 7 — Bilingual site

- [ ] Pin MkDocs, Material for MkDocs, and `mkdocs-static-i18n` dependencies.
- [ ] Configure suffix-based localization for `.en.md` and `.zh-TW.md`.
- [ ] Check in a staging manifest that includes both central and colocated documentation.
- [ ] Stage documentation deterministically with `stage-site --output DIR`; keep generated HTML ignored.
- [ ] Preserve the panel UI at the remote Pages root and publish documentation beneath `/docs/`.
- [ ] Test strict builds, internal and cross-locale links, equivalent-page language switching, and bilingual search.

**Phase acceptance:** both locale sites build strictly; each equivalent page switches correctly; English and Traditional Chinese content are searchable; central and colocated pages are present; broken links fail the build; generated HTML is absent from Git; and the existing panel application at the Pages root is unchanged.

<a id="phase-8-final-checkpoint"></a>
## Phase 8 — Final checkpoint

- [ ] Run full inventory, link, site, migration, and staleness reports and retain their outputs as review evidence.
- [ ] Run repository tests affected by documentation tooling and compare relevant results with the source checkpoint.
- [ ] Confirm all migration-manifest rows have final dispositions and all removed headings have removal commits.
- [ ] Confirm the working tree contains no generated site output, runtime artifacts, or unaccounted source-document deletions.
- [ ] Obtain final review, then create the annotated tag `docs-reorg-v1` on the accepted commit.
- [ ] Treat Git history and branch cleanup as a separate, later-reviewed project; do not perform it as part of this plan.

**Phase acceptance:** all reports are clean or have explicitly accepted findings, tests pass, bilingual journeys are complete, manifest traceability is total, the panel app remains unchanged, the final annotated tag identifies the reviewed result, and cleanup has not been mixed into this change.

<a id="overall-acceptance"></a>
## Overall acceptance criteria

- [ ] Validator failure fixtures cover naming, metadata/enums, ID uniqueness, lifecycle/location, pairs/parity, changed-pair parity, links, and anchors, with exit `1`; valid inputs exit `0`.
- [ ] Every heading removed from an old document has a migration-manifest row with disposition, paired replacement, and removal commit.
- [ ] Every published command, version, and path has been verified against code.
- [ ] Operator setup, training, evaluation, export-video, calibration, deployment, and troubleshooting form complete journeys in both languages.
- [ ] Developer architecture, development, testing, subsystem, decision, and evidence documentation form complete journeys in both languages.
- [ ] Skill tests cover operator, developer, release, ADR, design, plan, experiment, and no-impact scenarios, including RED/GREEN evidence, loophole refinement, at least five repetitions, and no-guidance controls.
- [ ] Both site locales build, link, switch equivalents, and search successfully while the panel application remains unchanged at the root.
- [ ] Pre-commit and CI gates enforce the documentation contract without excessive duplication or inferred semantic impact.
- [ ] `AGENTS.md` points to canonical documentation governance.
- [ ] `CLAUDE.md` imports `AGENTS.md`.
- [ ] After Phase 6, Claude wrapper paths resolve to the canonical `.agents/skills` files.
- [ ] Agent-independent validation rejects nonconforming output regardless of authoring tool.
- [ ] Generated HTML, staged site output, caches, logs, videos, and other runtime artifacts are absent from Git.
