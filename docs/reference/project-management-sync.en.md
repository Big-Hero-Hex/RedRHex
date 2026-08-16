---
id: project-management-sync
title: Project Management Synchronization
lang: en
audience: shared
type: reference
status: active
owner: project
last_reviewed: 2026-08-15
---

<a id="outcome"></a>
## Outcome

This contract keeps repository work visible to the RedRHex project-management agent without creating a second editable copy of maintained documentation. The agent reads [GitHub](https://github.com/Big-Hero-Hex/RedRHex) directly, projects coordination changes into the [PM Control Center](https://docs.google.com/spreadsheets/d/1DC4RlA1jVbcLEFDr0Jom1bysjc8cBxX8w_8d7uv0zjs/edit), and links to evidence in the [shared Drive](https://drive.google.com/drive/folders/1REUOdoJO_CHnxuBSGBHZwVIgA1KFW7gJ).

<a id="prerequisites-and-context"></a>
## Prerequisites and context

- The PM agent needs read access to the GitHub repository and write access to the PM Control Center.
- A scheduled prompt or webhook receiver must be able to invoke the PM agent. Repository or Drive credentials must not be committed to GitHub or written into project documents.
- The repository default branch is `main`. An immutable commit SHA identifies the exact code and documentation revision under discussion.
- The PM agent's approval limits remain unchanged: it may reconcile routine records and propose work, but it may not authorize source changes, deployment, budgets, publication claims, autonomous campaign arming, or hardware activity without explicit human approval.

<a id="instructions-or-explanation"></a>
## Instructions or explanation

Use one authoritative home for each kind of fact:

| Project fact | Authoritative home | Other-system treatment |
| --- | --- | --- |
| Code, configuration, tests, maintained documentation, plans, designs, decisions, roadmaps, and releases | GitHub `main` at an immutable SHA | The Control Center stores status and permalinks, not copied canonical text. |
| Accepted tasks, human DRIs, dates, status, decisions, risks, meetings, and paper readiness | PM Control Center | Repository and Drive documents link to the relevant record when needed. |
| Meeting records, hardware measurements, CAD/manufacturing material, experiment packages, videos, reports, and paper assets | Shared Drive | The Control Center stores evidence links and readiness state. |
| Run identity, metrics, checkpoints, logs, and evaluation evidence | Training Panel or experiment database | The Control Center stores immutable evidence links and the code SHA. |

Drive copies of maintained repository Markdown are convenience snapshots only. A snapshot must show its GitHub source URL, branch or tag, commit SHA, and refresh time. It must not be used to decide current scope or close a task when its SHA differs from `main`. Prefer replacing recurring snapshot links with GitHub links instead of manually refreshing copies.

For each repository intake:

1. Read the current `refs/heads/main` SHA. Treat unmerged branches, recovery snapshots, and pull requests as candidate work only; never report them as shipped or current.
2. Compare the SHA with the last successfully ingested SHA. An unchanged SHA is a no-op.
3. Read the intervening commits and changed paths. Changes under `docs/roadmap/`, `docs/plans/active/`, `docs/designs/active/`, `docs/decisions/`, and `docs/releases/` always require PM review. Source, test, workflow, and configuration changes require review when they alter deliverables, gates, risks, evidence, compatibility, or dates.
4. Append one deduplicated `Update Log` record for the new SHA or contiguous commit range. Record the SHA, changed canonical paths, observed facts, interpretation or task proposals, blockers, evidence permalinks, confidence, and `Pending` review state. Do not infer completion from code presence alone.
5. Run PM reconciliation. Update `Tasks`, `Experiments`, `Decisions`, `Risks`, `Roadmap`, or `Meetings` only where the evidence supports it. Every actionable task needs one human DRI, an exact date, acceptance criteria, dependencies, and an evidence destination.
6. Persist the new intake cursor only after the Control Center write succeeds and a read-back confirms the update. Retries deduplicate on repository identity, branch, and commit SHA.

Scheduled polling is the recommended initial notification path because it needs no inbound endpoint: check GitHub at the start of every PM sync and on the chosen active-project cadence. Use a GitHub webhook later only when the PM agent has an authenticated endpoint, retry handling, deduplication, and secret rotation. The webhook is a wake-up signal; GitHub remains the source read by the agent.

Documentation intake follows the repository [documentation governance](../governance/index.en.md). Meaning-changing edits must update both locale files, and the PM agent should use canonical paths and anchor links rather than copying prose into Drive.

<a id="verification"></a>
## Verification

The integration is operating only when all of these checks pass:

- The Control Center records the repository URL, default branch, and last observed and last successfully ingested `main` SHAs.
- One new `main` SHA produces exactly one pending Update Log record with GitHub permalinks; polling again produces no duplicate.
- A failed Control Center write leaves the intake cursor unchanged, and a retry produces the missing record.
- A branch-only documentation or source change is described as proposed or in progress, never shipped.
- A plan change merged to `main` becomes a reviewable PM update, and accepted work becomes person-specific task packets with owner, date, acceptance criteria, dependencies, and evidence.
- A stale Drive snapshot is identified by its SHA or missing provenance and is not treated as current.

As of 2026-08-15, Control Center tasks `T-003` and `T-020` track repository identity and intake automation. Update `UPD-20260815-REPO-001` is the first pending repository audit record.

<a id="troubleshooting-and-limits"></a>
## Troubleshooting and limits

- If GitHub cannot be read, report the repository sync as Yellow or Red and do not advance the cursor.
- If Drive or the Control Center cannot be written, preserve the pending change packet and retry; do not claim reconciliation succeeded.
- If two systems disagree, use the authoritative-home table above and record the mismatch as a correction or blocker.
- Repository polling detects committed GitHub state. It cannot discover an unpushed local working tree, so local work must remain explicitly labeled as uncommitted until it is pushed or merged.
- This contract defines intake and reconciliation. It does not by itself provision a GitHub connector, scheduled prompt, webhook endpoint, or credentials; `T-020` closes that implementation gap.

