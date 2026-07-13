# 07 — Roadmap, Milestones, and Risks

Dates are not gates. Evidence is. The one-GPU machine and unknown physical facts make
P1 duration uncertain, so the roadmap is dependency-ordered rather than calendar-driven.

## Core-reboot milestones

| Milestone | Phases | Exit evidence |
|---|---|---|
| M0 — Trustworthy foundation | P0 | safe complete CPU tests, frozen interface guard, reproducible simulator provenance, artifact/command contract |
| M1 — Validated legacy simulator | P1 | mandatory gravity/frame/model checks pass; confirmed blockers fixed and rerun |
| M2 — Frozen validated oracle | P2 | validated tag, reduced/full golden data, checkpoint/eval manifests, fixed-seed reference |
| M3 — Real package boundaries | P3–P4 | sibling packages install independently; stable facts extracted; frozen ROS comparison only |
| M4 — Clean testable core | P5 | pure-Torch slices pass unit and golden gates; no forbidden dependencies |
| M5 — Thin compatible adapter | P6 | both existing tasks and operational boundaries pass adapter/smoke/export regressions |
| M6 — Accepted reboot | P7 | fresh all-tier evidence, reference comparison, acceptance report, final tag |

Panel/remote/reward-agent/ROS changes, hardware work, MPC experiments, and new research
features are not milestones on this critical path.

## Immediate sequence

```text
design checkpoint review
  -> P0 test-discovery commit
  -> P0 simulator-provenance commit
  -> P0 executable-interface snapshots/frozen-guard commit
  -> P0 command/artifact-schema commit
  -> P1 diagnostic orchestrator implementation + G0 dry-run
  -> P1 G0–G10/C1 execution
  -> only then P2 tag/golden/reference capture
```

## Risk register

| Risk | Impact | Mitigation / early signal |
|---|---|---|
| P1 reveals frame/physics errors, invalidating historical behavior as oracle | High | This is why validation precedes baseline. Fix one localized blocker, ADR it, rerun diagnostics, then create P2 from the corrected legacy state. |
| Dirty/patched external Isaac checkout makes results irreproducible | High | Clean dedicated checkout preferred; otherwise pin exact SHA and patch hash in every manifest. P0 blocks evidence until resolved. |
| Hidden tests produce false green | High | Remove blanket ignore, explicit CPU test paths, collection completeness probes, frozen regression suite. |
| Extraction accidentally changes semantics | High | Do not bundle logging/timing/reward/physics changes; unit + reduced/full golden + adapter smoke per seam. |
| GPU nondeterminism weakens exact replay | Medium | Measure envelope only after P1, replay pure seams on recorded inputs, freeze tolerance policy with ADR control. |
| Frozen consumers drift or become accidentally edited | High | Tree-ID plus staged/unstaged/untracked guards, caches redirected outside frozen paths, and existing regressions on each cutover. |
| Core imports pull Isaac through eager package registration | Medium | Separate sibling packages, independent install/import tests, forbidden-import checks. |
| Baseline storage remains contradictory or too large for CI | Medium | Small tracked fixture; full local ignored artifacts; tracked hashes/manifests. |
| Reboot stalls with two half-architectures | High | One slice in flight, revertible commits, delete legacy only with replacement evidence, scope ends at P7. |
| Visual “gravity feels weird” drives speculative changes | High | Canonical free fall → robot free fall → contacts → frames → rewards causal chain; no fix before localization. |

## Human decision points

1. Approve this detailed design before the P0 implementation plan and code changes.
2. Decide how to resolve/pin the dirty external Isaac Lab checkout.
3. Provide or explicitly mark unavailable the physical facts required by G3/G4.
4. Approve any P1 physics/frame fix or evidence-backed threshold correction; both require
   an ADR and fresh affected rerun, and neither waives a blocked/failed result.
5. Approve the immutable P2 baseline protocol before the first reference run.
6. Approve any intentional golden difference discovered during P4–P6.
7. Accept or reject the P7 reference comparison.
8. Decide post-P7 which frozen systems or research features to resume.
