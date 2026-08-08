---
id: reward-energy-model
title: Reward and Energy Model
lang: en
audience: developer
type: explanation
status: active
owner: training
last_reviewed: 2026-08-07
---

<a id="current-reward"></a>
## Current reward boundary

The active environment uses the simplified command-aware reward path. It combines forward progress, linear and angular tracking, mode specialization, suppression of unwanted axes, gait and height shaping, movement/alive behavior, stall/fall penalties, and one energy term. Values in `RedrhexEnvCfg.v2_reward_scales` are the source of truth.

The earlier four-term proposal (`power_efficiency`, `spring_recovery`, `spring_utilization`, and `torque_penalty`) is not the current active reward interface. Spring and power quantities remain useful diagnostics and experiment hypotheses.

<a id="mechanical-model"></a>
## Mechanical model

For active joints, estimated instantaneous mechanical power is based on `abs(torque * angular_velocity)`. Main-drive torque is bounded from the velocity-control error; ABAD torque is bounded from position and velocity error. Passive damper energy uses the configured linear torsion model with stiffness `200 N·m/rad` and damping `20 N·m·s/rad`.

These are simulator/controller estimates, not electrical battery power. Current, voltage, gearing, friction, hysteresis, driver loss, and sensor error are outside the estimate unless measured separately.

<a id="active-energy-term"></a>
## Active energy term

`energy_per_distance` penalizes accumulated estimated mechanical energy divided by positive motion in the commanded direction, with an epsilon and a maximum clamp. The default weight is `0.001`; ForwardFast uses `0.0005`. It remains secondary to tracking and gait terms.

For yaw, lateral, and diagonal work, compare the command-aware motion definition rather than raw forward distance. A lower proxy produced by slower, stalled, or falling behavior is not an efficiency gain.

<a id="validation"></a>
## Validation protocol

Compare a baseline and bounded candidates on identical seeds and command profiles. Require tracking and success to remain acceptable, fall rate not to worsen, and both cost-of-transport proxy and power per motion to improve. Report results by forward, lateral, diagonal, and yaw skill. Publish an experiment summary only when evidence changes a recommendation or baseline.

<a id="limitations"></a>
## Limitations

The current model uses torque and contact proxies and has not established true electrical energy savings on hardware. Treat cost of transport as a comparison proxy until robot mass, velocity, current, voltage, timing, and sensor calibration are bound to reviewed evidence.
