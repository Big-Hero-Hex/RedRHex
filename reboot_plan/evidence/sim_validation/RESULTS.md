# P1 Simulation and Gravity Results

Status: **NOT RUN**

Baseline eligibility: **BLOCKED**

Source snapshot under review: `fix/review-2026-07@5cdc824`

No GPU simulation diagnostic was executed during the documentation checkpoint. The
configured gravity/frame observations in the design are code-audit hypotheses only.

| Check | Status | Run ID | Evidence | Notes |
|---|---|---|---|---|
| G0 provenance/resolved config | NOT RUN | — | — | P0 must pass first. |
| G1 composed stage/world gravity | NOT RUN | — | — | — |
| G2 canonical-body free fall | NOT RUN | — | — | — |
| G3 asset/root/semantic axes | NOT RUN | — | — | — |
| G4 mass/COM/inertia | NOT RUN | — | — | Physical ground truth may be required. |
| G5 robot whole-COM free fall/damping | NOT RUN | — | — | — |
| G6 contact/drop/settle | NOT RUN | — | — | — |
| G7 projected gravity/policy frame | NOT RUN | — | — | — |
| G8 action decoding/actuator response | NOT RUN | — | — | — |
| G9 task-specific rewards/terminations | NOT RUN | — | — | — |
| G10 timing/freshness/determinism | NOT RUN | — | — | Defines P2 tolerance. |
| C1 frozen ROS compatibility | NOT RUN | — | — | Read-only; report match/mismatch/IMU block. |

Each completed row must link a tracked summary and raw artifact hashes. Do not replace
`NOT RUN` with `PASS` based on static inspection alone.
