# PR / merge checklist

<!-- Paste into the PR description (or run through it mentally when self-merging).
     This checklist is the stand-in for GPU CI — do not skip the preflight line. -->

- [ ] One issue/step per commit; messages name the plan step or finding
- [ ] `make test` green (CI confirms) — and no test was weakened/skipped to get there
- [ ] `make preflight` run locally on the final commit (GPU tiers) — paste the tail
- [ ] Touched contract/layout? → `make contract` ran; generated file committed together
- [ ] Behavior-changing? → ADR linked + baseline updated + validation run id
- [ ] New failure mode discovered? → regression test added in the cheapest tier
- [ ] CLAUDE.md / plan checkboxes updated if this makes them stale
- [ ] /code-review findings addressed (or explicitly waived with a reason)
