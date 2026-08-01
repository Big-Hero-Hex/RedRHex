---
id: document-lifecycle
title: Documentation Lifecycle
lang: en
audience: developer
type: reference
status: active
owner: project
last_reviewed: 2026-08-01
---

<a id="lifecycles"></a>
## Lifecycles

- Maintained knowledge moves `draft -> active -> deprecated -> removed`.
- Decisions are `accepted` or `superseded`. Accepted and superseded ADRs remain in the repository; a later ADR supersedes an earlier one rather than rewriting it.
- Designs move from `proposed` to `approved`, then resolve as `implemented`, `rejected`, or `superseded`.
- Plans move from `draft` to `active`, may be `blocked`, and resolve as `completed` or `cancelled`.
- Roadmaps are `active` and contain only current priorities.
- Releases, experiment summaries, and audits are `published` records.

<a id="resolution"></a>
## Resolution and retention

An active design is summarized into maintained architecture, an ADR, a release, or a roadmap item as appropriate, then removed from `designs/active/` when resolved. A completed or cancelled plan is summarized into durable documentation, then removed from `plans/active/`. Accepted and superseded ADRs remain.

Completed roadmap work moves to a release or milestone record. Experiment summaries are immutable; corrections use dated addenda rather than silent rewrites. Raw run logs and notes are ignored. Publish an experiment summary only when evidence changes a baseline, recommendation, decision, or result.

<a id="stale-warnings"></a>
## Stale warnings

- Operator, reference, and roadmap documents warn after 90 days without review.
- Developer architecture documents warn after 180 days without review.
- Decisions and releases are exempt from age-based warnings; they warn only when contradicted or superseded.

A stale warning prompts review and does not automatically change the document's truth status.

<a id="removal-preconditions"></a>
## Removal preconditions

Before deleting a durable or historical source document:

1. Create the complete replacement locale pair.
2. Validate replacement links and matching explicit anchors.
3. Record heading-level disposition and replacement paths in the migration record.
4. Pass the documentation validator.
5. Delete the source and record the removal commit in the migration record.

Git history is the archive after these conditions are met; the live tree must not retain redundant originals as historical storage.
