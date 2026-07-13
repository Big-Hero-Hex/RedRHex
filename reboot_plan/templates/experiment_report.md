# <run id> — <overlay name>

<!-- Copy to experiments/reports/<date>_<slug>_s<seed>.md. Agent drafts, human sets verdict. -->

- **Date:** YYYY-MM-DD → YYYY-MM-DD
- **Hypothesis:** <the one sentence from LOG.md>
- **Overlay:** `cfg/experiments/exp_....py` (diff vs base: <2-3 bullet summary>)
- **Commit:** `<hash>` · **Contract:** v<N> · **Seeds:** 42[,43,44] · **Iterations:** N
- **Baseline compared against:** `ref_run_v<N>`
- **Logs:** `logs/rsl_rl/redrhex_wheg/<dirs>`

## Result
<Curves: reward / tracking / termination vs baseline band — PNG committed next to this
file. 3 standard eval metrics from the fixed command sweep, table: run vs baseline.>

| metric | baseline | this run | Δ |
|---|---|---|---|
| tracking error | | | |
| energy proxy (CoT) | | | |
| termination rate | | | |

## Verdict (human)
✅ adopt / ❌ reject / 🔁 iterate — <one sentence why>

## Notes & follow-ups
- <anomalies, instability, anything the curves hide>
- <next overlay if 🔁; ADR link if this changes base cfg>
