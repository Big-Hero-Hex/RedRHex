---
id: migration-manifest
title: Migration Manifest
lang: en
audience: developer
type: reference
status: active
owner: project
last_reviewed: 2026-08-07
---

<a id="purpose"></a>
## Purpose

The [migration manifest](https://github.com/JasonLiaoJCS/RedRHex/blob/main/docs/governance/migration-manifest.csv) is the machine-readable, language-neutral traceability ledger for source-document migration. Original source headings remain verbatim and are not translated.

<a id="provenance-and-coverage"></a>
## Provenance and coverage

The inventory is derived only from annotated tag `docs-reorg-source-2026-08-01`, commit `5de992d7afac77e566b6deebb99dc813eb87b612`. Each source has one document-root row and one row for every Markdown ATX heading, reStructuredText underlined heading, and LaTeX `part`, `chapter`, `section`, `subsection`, or `subsubsection`. Duplicate heading text is disambiguated by `source_path` and `source_line`.

<a id="row-contract"></a>
## Row contract

The CSV columns are `source_path`, `source_line`, `heading_level`, `source_heading`, `disposition`, `replacement_ids`, and `removal_commit`. Rows are ordered by the approved source-path order and then numeric source line. A document-root row uses line `0`, level `0`, and heading `<document>`. Heading markers are stripped while heading text remains verbatim.

<a id="dispositions-and-removal"></a>
## Dispositions and removal

Allowed dispositions are exactly `pending`, `migrated`, `obsolete`, `duplicate`, and `git-history-only`. Before removing a source, every row for it must leave `pending`. A `migrated` row must name one or more canonical replacement IDs, and the source removal commit is filled only after deletion.

<a id="field-format"></a>
## Field format

`replacement_ids` contains semicolon-separated canonical document IDs. `removal_commit` is a full Git commit hash, or blank while the source remains. The ledger is RFC 4180-compatible UTF-8 CSV with `\n` line endings.
