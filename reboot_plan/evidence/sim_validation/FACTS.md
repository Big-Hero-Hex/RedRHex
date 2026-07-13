# P1 Physical and Frame Facts

Status: **INCOMPLETE**

This is the instantiated P1 input sheet. Missing facts do not become simulator truth and
block the affected required G-check. C1 alone is a non-mutating frozen-compatibility
finding rather than a simulator-correctness gate.

## Semantic frames

| Fact | Value | Source/status |
|---|---|---|
| World handedness/up axis/units | ___ | G1 NOT RUN |
| Robot semantic forward/left/up | ___ | CAD/photo BLOCKED |
| Quaternion direction/order/composition | Isaac `wxyz`; remaining convention ___ | source audit, runtime proof NOT RUN |
| Root-to-policy fixed transform | ___ | BLOCKED |
| Intended spawn transform | cfg quaternion near +90° X; semantic intent ___ | runtime/source proof NOT RUN |
| Neutral settled chassis attitude | ___ | CAD/measurement BLOCKED |

## Physical facts

| Fact | Value | Source/status |
|---|---|---|
| Total deployed mass | ___ kg | measured BLOCKED |
| Per-link/module masses | ___ | CAD/measured BLOCKED |
| Body dimensions/CoM/inertia references | ___ | CAD BLOCKED |
| Main-drive torque/speed/response | ___ | datasheet/measured BLOCKED |
| ABAD limits/response | ___ | datasheet/measured BLOCKED |
| Ground/contact material facts | ___ | measured/datasheet BLOCKED |

## Timing/freshness contract to fill before G10

| Quantity | Expected source step/state | Expected mutation/write point | Source |
|---|---|---|---|
| Action intent | ___ | once per control step ___ | intended adapter contract BLOCKED |
| Physics state snapshot | ___ | ___ | Isaac lifecycle source BLOCKED |
| Observation | ___ | ___ | intended policy contract BLOCKED |
| Reward components/total | ___ | ___ | intended task contract BLOCKED |
| Termination/done | ___ | ___ | intended task contract BLOCKED |
| Reset state | ___ | ___ | intended task contract BLOCKED |

## Determinism characterization inputs to fill before G10

| Item | Predeclared value |
|---|---|
| Seed, rollout length, fixed action/command input | ___ |
| Float channel normalization scales | ___ |
| Per-channel numeric floors `F_m` | ___ |
| Exact integer/boolean/event fields | ___ |
| Characterization/holdout process launch command | ___ |

Use `templates/sim_facts.md` as the collection guide. Link sources by document/version,
measurement date/method, or CAD revision; do not replace blanks with assumptions silently.
