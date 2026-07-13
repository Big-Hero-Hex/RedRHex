# RedRHex P1 Physical Facts Sheet

Every value needs a source: measured (date/method), datasheet (document/version), CAD
(model/version), or assumed. An assumed or missing required value produces `BLOCKED` and
keeps P2 blocked until sourced. Human acknowledgement records the open risk; it does not
turn the check into PASS. Never edit a fact to make the sim pass.

## Mass, center of mass, and geometry

| Fact | Value | Source | Used by |
|---|---|---|---|
| Total deployed mass with battery | ___ kg | measured ___ | G4/G6 |
| Chassis mass | ___ kg | ___ | G4 |
| One leg/module mass (×6) | ___ kg | ___ | G4 |
| Battery mass | ___ kg | ___ | G4 |
| Whole-robot CoM in semantic chassis frame | (___, ___, ___) m | ___ | G4/G5 |
| Body bounding box L×W×H | ___ m | CAD ___ | G3 |
| WHEG radius/geometry | ___ m | CAD ___ | later dynamics |
| Structural materials/densities | ___ | BOM/CAD ___ | G4 |

## Semantic frames and asset transform

| Fact | Value | Source | Used by |
|---|---|---|---|
| Chassis semantic forward axis | ___ | CAD/photo ___ | G3/G7 |
| Chassis semantic left axis | ___ | CAD/photo ___ | G3/G7 |
| Chassis semantic up axis | ___ | CAD/photo ___ | G3/G7 |
| Intended spawn pose in world | position ___; quaternion convention/value ___ | design source ___ | G3 |
| Policy frame relative to root frame | transform ___ | source ___ | G7/G8 |
| IMU frame relative to semantic chassis | RPY/quaternion ___ | CAD/photo ___ | G8/later hardware |

## Main-drive actuator (×6)

| Fact | Value | Source |
|---|---|---|
| Motor/gearbox model and ratio | ___ | ___ |
| Stall/rated torque post-gearbox | ___ N·m | datasheet ___ |
| No-load/rated speed post-gearbox | ___ rad/s | datasheet ___ |
| Rated voltage/current | ___ V / ___ A | datasheet ___ |
| Battery voltage under load | ___ V | measured ___ |
| Passive damping/coast-down behavior | ___ | measured ___ |

## ABAD actuator (×6)

| Fact | Value | Source |
|---|---|---|
| Model | ___ | ___ |
| Torque/speed limits | ___ N·m / ___ rad/s | ___ |
| Mechanical range | ___ rad or degrees | ___ |
| Hardware-enforced limits | ___ | ___ |

## Contact and environment

| Fact | Value | Source |
|---|---|---|
| Leg-tip material/geometry | ___ | CAD/BOM ___ |
| Friction vs lab floor | ___ | measured incline-slip ___ |
| Restitution/compliance evidence | ___ | measured/datasheet/assumed ___ |
| Lab ground normal/up convention | ___ | measured/setup ___ |

## Rates and deploy facts (read-only comparison)

| Fact | Value | Source |
|---|---|---|
| Physics step rate | ___ Hz | current resolved cfg at commit ___ |
| Control/policy rate | ___ Hz | current resolved cfg at commit ___ |
| Low-level/IMU rates | ___ Hz | frozen ROS config commit `5cdc824` |
| Sim rest projected gravity | (___, ___, ___) | G7 run ___ |
| Hardware rest projected gravity | (___, ___, ___) | later measured run ___ |

P1 records sim/deploy mismatches but does not edit frozen ROS configuration.

## Known residuals and unknowns

| ID | Fact/residual | Evidence/status | Consequence |
|---|---|---|---|
| ___ | ___ | PASS / FAIL / BLOCKED | affected G-checks ___ |
