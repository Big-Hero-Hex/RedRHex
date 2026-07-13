# RedRHex Sim Facts Sheet — physical ground truth

<!-- Copy to docs/sim_facts.md (Phase V0). This file is the authority the simulation is
     validated AGAINST (reboot_plan/09). Every value needs a source; "assumed" values
     propagate a ⚠️ into every validation check that uses them. Update only with a new
     measurement/datasheet — never to "make the sim pass". -->

Legend for **source**: `measured` (we measured it, date + method) · `datasheet` (link/PDF)
· `CAD` (model name/version) · `assumed` (guess — flag downstream).

## 1. Mass & geometry

| Fact | Value | Source | Notes |
|---|---|---|---|
| Total mass, as deployed (with battery) | ___ kg | measured ____ | weigh on lab scale |
| Chassis mass | ___ kg | | |
| Leg module mass (× 6) | ___ kg | | |
| Battery mass | ___ kg | | |
| CoM location (chassis frame) | (_, _, _) m | CAD/measured | balance test optional |
| Body bounding box L×W×H | ___ m | CAD | L0.6 units check |
| Wheg radius / leg geometry | ___ m | CAD | drives L4.1 speed expectation |
| Structural material + density | ___ (UPE? ___ kg/m³) | BOM | sim currently uses density=2500 — reconcile |

## 2. Actuators

### Main drive (× 6)
| Fact | Value | Source |
|---|---|---|
| Motor model | | |
| Gear ratio | | |
| Stall torque (post-gearbox) | ___ N·m | datasheet |
| No-load speed (post-gearbox) | ___ rad/s | datasheet |
| Rated voltage / current | ___ V / ___ A | datasheet |
| Battery voltage under load | ___ V | measured |

### ABAD (× 6)
| Fact | Value | Source |
|---|---|---|
| Actuator model | | |
| Max torque | ___ N·m | |
| Max speed | ___ rad/s | |
| Mechanical range | ± ___ ° | |
| Hardware-enforced limit (if any) | ± ___ ° | |

### (Other joints, if actuated — toe/foot: fill or mark N/A)

## 3. Contact & environment

| Fact | Value | Source | Notes |
|---|---|---|---|
| Leg-tip material | | | |
| Friction μ vs lab floor | ___ | measured (incline slip, date) | L4.3 |
| Friction μ vs deployment terrain(s) | ___ | | DR range should bracket this |
| Leg tip compliance/restitution | | assumed? | affects L4.5 |

## 4. Sensors

| Fact | Value | Source | Notes |
|---|---|---|---|
| IMU model | | | |
| IMU mounting orientation (RPY vs chassis frame) | (_, _, _)° | CAD + photo | → `imu_mount_rpy_deg` |
| IMU axes convention | | datasheet | |
| Rest projected gravity (sim, obs[6:9]) | (_, _, _) | L1.3 capture, commit ___ | → `expected_rest_projected_gravity` |
| Rest projected gravity (hardware) | (_, _, _) | measured (Phase 5.4) | must match sim value |
| Encoder resolution / joint sensing | | | |

## 5. Rates & control (must equal contract.py — listed for hardware cross-check)

| Fact | Value | Source |
|---|---|---|
| Policy rate | 60 Hz | contract.py v___ |
| Low-level bridge rate | ___ Hz | |
| IMU publish rate | ___ Hz | |
| Serial/UDP latency (measured round trip) | ___ ms | measured |

## 6. Known sim↔real residuals (running list, updated from L6.3)

| # | Residual | Size | Mitigation | Status |
|---|---|---|---|---|
| 1 | e.g. no contact sensing in sim (phase proxy) | | contact reporter (Phase 4.1) | open |
