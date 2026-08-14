---
id: adr-0002-single-protected-main
title: Single Protected Main Branch
lang: en
audience: developer
type: decision
status: accepted
owner: project
last_reviewed: 2026-08-13
---

<a id="context"></a>
## Context

RedRHex accumulated two competing integration lines, long-lived experiment branches, local-only feature stacks, mixed dirty worktrees, and a local `main` that diverged from GitHub. Documentation System v1 was reviewed and merged into `fix/review-2026-07`, while the remote `main` still pointed to an older line. Continuing with both lines would leave contributors and automation without one authoritative starting point.

<a id="decision"></a>
## Decision

`main` is the only permanent integration branch. As a one-time migration exception, its remote ref may be force-updated to the reviewed `fix/review-2026-07` tip only after the old `main`, every significant unmatched tip, all dirty worktrees, and the preserved stash have recoverable archive refs and pass the branch-preservation audit.

After the cutover, `main` is protected: changes arrive through pull requests, required validation must pass, review conversations must be resolved, force pushes and deletion are disabled, and active development uses short-lived `feature/*`, `fix/*`, `docs/*`, or `chore/*` branches. Long-lived unapproved architecture remains under `proposal/*`; retained historical tips live under `archive/*` and are not development bases.

The intact torsion work is reconstructed first from the new `main`. Training Panel V3.6 is stacked on the torsion branch so its prior merge and integration behavior remain reachable. The Windows remote launcher is reconstructed independently from `main`. Recovery and source branches remain until tree, commit, and hunk-level preservation checks pass.

<a id="alternatives"></a>
## Alternatives

- A non-forced ancestry merge would preserve the old `main` graph but keep unrelated historical UI/shim commits on the canonical line and obscure the reviewed cutover.
- Keeping `fix/review-2026-07` as a permanent integration branch would preserve the two-trunk ambiguity.
- A permanent `develop` branch would add coordination overhead without a current release-train need.
- Squashing or replaying the torsion and Panel histories would produce a simpler graph but weaken proof that no existing feature work was dropped.

<a id="consequences"></a>
## Consequences

Existing clones must fetch and explicitly realign local `main` after the cutover; archived refs remain available for recovery. The one-time force update is documented and reversible, but future force pushes to `main` are forbidden. Feature PRs may be stacked temporarily, while merged or superseded branches are removed after their tip SHA and disposition are recorded in the archive manifest.

<a id="supersession"></a>
## Supersession

A later ADR may introduce another permanent integration model only when the project has a concrete release-management requirement that a single protected trunk cannot satisfy.
