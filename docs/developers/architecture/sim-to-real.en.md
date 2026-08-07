---
id: sim-to-real-architecture
title: Sim-to-Real Calibration Architecture
lang: en
audience: developer
type: explanation
status: active
owner: sim2real
last_reviewed: 2026-08-07
---

<a id="roles"></a>
## Evidence roles

The calibration system distinguishes raw hardware traces, managed immutable episodes, reviewed replay fixtures, simulated traces, comparison results, direct-measurement profiles, audit artifacts, and held-out promotion evidence. Each transition records hashes and provenance so a candidate cannot silently substitute another input.

<a id="flow"></a>
## Data flow

`tools.sim2real` imports a real trace with scenario, units, frames, time bases, dataset ID, and episode ID. Replay eligibility additionally requires an operator-reviewed fixed-base fixture. `run-sim` applies the same scenario and optional explicit profile. `compare` produces metric differences. `sweep` generates bounded candidates and executes only when the required real trace and audit evidence are supplied.

<a id="profile"></a>
## Profile boundary

A `CalibrationProfileV1` is versioned data, not an implicit global setting. Training and playback default to no candidate profile and accept one only through `--physics-profile`. Profile construction and promotion are separate: a syntactically valid profile is not promoted until hash-bound audit and held-out evidence pass.

<a id="failure-model"></a>
## Failure model

The workflow fails closed on incomplete provenance, duplicate or non-finite JSON, unresolved hardware mapping, publisher ambiguity, timing conflicts, unauthenticated artifacts, failed physics audits, incomplete held-out metrics, and nonstationary evidence. Output paths are not overwritten silently.

<a id="limits"></a>
## Interpretation limits

A scenario comparison localizes differences for the measured state and command envelope. It does not validate unmeasured contacts, terrain, thermal effects, structural compliance, estimator behavior, or long-horizon locomotion. Expand scenarios only with new reviewed evidence and explicit acceptance criteria.

<a id="operator"></a>
## Operator procedure

See [physics calibration](../../operators/calibration/physics-calibration.en.md) for the safety-ordered workflow.
