# 07 — Roadmap, Milestones, Risks

Assumes AI-heavy execution on this machine (one RTX 5080), human available for review
and judgment ~daily. Calendar estimates are deliberately padded ~30% — solo-project
weeks are never clean.

## 1. Milestones

| Milestone | Content (phases from 03) | Target | Exit evidence |
|---|---|---|---|
| **M0 — Frozen baseline** | Phase 0 + **V0.5 early sim screen (09)** | Week of Jul 13 (1–2 days) | tag `v0-pre-reboot`; golden dump; L0+L1+rates screened; reference run archived |
| **M1 — Scaffolded** | Phase 1 + V0 (ladder scripts, sim facts sheet) | +3 days | fresh AI session runs smoke+tests unaided; CI green; ladder scripts runnable |
| **M2 — Clean core** | Phase 2 ∥ **V1 (full sim validation run, 09)** | +2–3 weeks (→ ~mid-Aug) | env < 800 lines; parity suite green; 3-seed gate run matches baseline curve; sim-validation RESULTS.md complete + findings triaged |
| **M3 — Drift-proof deploy** | Phase 3 | +1 week (→ ~late Aug) | corrupt-a-constant drill passes; ONNX equivalence in preflight |
| **M4 — Research restart** | Phase 4 items 4.1–4.3 first (4.3 = V2, driven by sim-validation triage) | Sep | contact-sensor A/B report; lin-vel decision; physics-fidelity ADRs; affected ladder levels re-run green |
| **M5 — Hardware-ready** | Phase 5 | Sep–Oct (hardware-availability-bound) | IMU values captured; HIL dry run clean; panel IPC/auth done |
| **M6 — MPC campaign** | Phase 4.7 + experiments | Oct–Nov | benchmark suite regenerable from a tag; headline numbers 3–5 seeds |

Training experiments (Loop B) resume permanently at M2; M4–M6 overlap freely. The
strict serialization is only M0 → M1 → M2.

## 2. Effort split (why this is fast despite the list)

- Phases 0–3 are ~90% agent-executable with the gates from 05; human time ≈ review +
  ~5 decisions (diag-sign intent, logging change, timing change, baseline promotions,
  scope cuts).
- The expensive-looking part (Phase 2) is 6 repetitions of one mechanical recipe; each
  step is a bounded overnight-able unit (Loop C).
- After M2, every future feature/experiment is cheaper than it is today — the payback
  starts ~3 weeks in.

## 3. Risk register

| # | Risk | L×I | Mitigation / early signal |
|---|---|---|---|
| 1 | Golden parity impossible due to GPU nondeterminism → gates get mushy | M×H | Phase 0.2 measures this FIRST; tiers 1–3 replay recorded tensors (sim never enters); tolerance policy in 05 §3 |
| 2 | Hidden coupling in the monolith (order-dependent state) breaks an extraction | H×M | strangler order chosen easy→hard; one step in flight; revert is one commit; step 2.5 (the known coupling) pre-declared behavior-changing |
| 3 | Migration stalls, research pressure kills it half-done (worst outcome: two half-structures) | M×H | standing rule 4 in 03 (scope-cut at 1-week stall); M2 is the only must-finish milestone; each step leaves the repo consistent |
| 4 | Agent silently degrades behavior in a way parity can't see (e.g. logging-path math) | M×M | 3-seed gate runs at phase exits; nightly bounded smoke train (05 §4) |
| 5 | Single GPU contention: verification blocks training or vice versa | H×L | 04 §4 queue discipline; CPU tiers cover 90% of iterations |
| 6 | Isaac Lab/Sim upgrade forced mid-migration (driver, CUDA) | L×H | pinned versions (02 §6); upgrade only post-M3 with parity suite as net |
| 7 | Hardware access slips → M5 drags | M×M | M5 isolated; only 5.4/5.5 need the robot; everything else proceeds |
| 8 | Panel freeze frustrates ongoing training needs | M×L | freeze covers *structure*, not usage; urgent panel fixes allowed as `fix:` commits |
| 9 | Baseline checkpoint incompatible after cfg modularization | M×M | compat shim (2.9) until Phase 5; checkpoint metadata versioning (3.5) |
| 10 | Docs/plan rot (this folder becomes fiction) | M×M | phase checklist ticked per commit; plan changes are commits to this folder, reviewed like code |

## 4. Explicit decision points for the human (calendar these)

1. **Before 2.6**: diag-sign double-count — intended or bug? (changes reward magnitude)
2. **At 2.2 / 2.5**: accept the two declared behavior changes (logging normalization;
   command-timing shift) after seeing 3-seed curves.
3. **Post-M2**: promote `ref_run_v1` as the new baseline.
4. **M4 ordering**: contact sensors vs lin-vel vs physics pass — pick by hardware
   timeline.
5. **M5**: panel exposure scope (localhost-only vs LAN+token).
6. **At V0.5 (before the reference run)**: escape-hatch call — if the early sim screen
   finds something catastrophic (mass/units-level wrong), fix it *before* investing in
   the baseline training run (09 §0).
7. **At V1 triage**: priority order of ❌ sim-validation findings for Phase 4.3, by
   expected sim2real impact.

## 5. What to do the moment this plan is approved

```
Day 1 morning : Phase 0.1 (merge + tag) — 30 min
Day 1         : Phase 0.2 golden dump script + capture (agent), 0.5 hygiene (agent)
Day 1 night   : Phase 0.3 reference run queued (3 seeds, overnight)
Day 2         : Phase 1 scaffolding (agent, Loop C-able), human reviews
Day 3–4       : Finish M1, run the fresh-session test, start extraction 2.1
```
