---
id: panel-physics-calibration
title: Panel Physics and Calibration Workspace
lang: en
audience: developer
type: design
status: proposed
owner: panel
last_reviewed: 2026-08-14
---

<a id="problem"></a>
## Problem

The recovery snapshot combines a browser editor for simulation physics, Sensor V2 launch routes, and ForwardFast defaults in the same Training Panel surfaces. The code is useful for inspection, but the physical values are candidates rather than measured calibration evidence and must not be presented as hardware-ready defaults.

<a id="boundary"></a>
## Proposed boundary

Keep this work on `feature/panel-physics-calibration-wip`, stacked on the Student Distillation V2 core. The branch may expose schema-validated simulation fields, sparse presets, run-scoped `CalibrationProfileV1` snapshots, and strict Student V2 checkpoint handoffs. It must preserve the released Panel 3.6 navigation, security, provenance, progress, spring-backend, and rollback behavior.

Baseline inherits repository and USD defaults. A non-empty candidate is explicit, run-scoped, and simulation-only. The UI must not label a numerical value as measured, calibrated, safe, or suitable for hardware without the corresponding reviewed evidence record.

<a id="integration"></a>
## Integration contract

Play, recording, export, and deployment checks reuse the selected run's task, agent route, physics snapshot, and spring backend. Sensor V2 browser routes remain additive and fail closed on checkpoint kind. The standard V1 route remains available. Process and artifact paths remain confined to repository-owned roots.

<a id="merge-gates"></a>
## Merge gates

- Preserve all current Panel, sim-to-real, documentation, and UI regression suites.
- Review all 113 schema fields against the active `CalibrationProfileV1` consumer and prove sparse round-trip behavior.
- Verify that Baseline sends no candidate profile and that every non-empty profile is snapshotted and reused.
- Complete a local browser smoke for standard training and each Sensor V2 route without starting a production or hardware run.
- Keep physical calibration, motor enable, and hardware promotion blocked until their existing evidence gates pass.

<a id="rollback"></a>
## Rollback

Do not merge this branch to remove it from normal operation. Within the branch, select Baseline and the standard training route to avoid candidate physics and Student V2 routing. Existing V1 artifacts and Panel 3.6 behavior remain the compatibility baseline.
