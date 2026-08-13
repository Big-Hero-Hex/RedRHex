---
id: research-readiness-audit-2026-08-13
title: 2026-08-13 Research Readiness Audit
lang: en
audience: developer
type: audit
status: published
owner: project
last_reviewed: 2026-08-13
---

<a id="scope"></a>
## Scope

This audit asks what RedRHex still needs before its locomotion, robustness, sim-to-real, or energy results can support a research claim. It covers the repository's implemented training and deployment contracts, the generated 2026-08-13 research-roadmap report, and the associated stage-1 training record. It does not establish hardware performance, electrical-energy savings, or novelty.

<a id="method"></a>
## Method

The generated report, source notes, materialized SQL snapshots, resolved run configuration, TensorBoard event, checkpoints, current code, tests, and canonical documentation were cross-checked. Statements below distinguish direct observations, diagnostic interpretations, and proposed research work. The TensorBoard event was read again after the report snapshot to detect late-arriving data.

<a id="executive-finding"></a>
## Executive finding

RedRHex has a substantial engineering scaffold: a hybrid controller and residual-policy path, PPO and teacher/student entry points, explicit sim-to-real calibration boundaries, ONNX/ROS deployment checks, a Training Panel, and automated tests. The missing research asset is a closed evidence loop:

1. measure the physical robot;
2. calibrate and hold out simulator evidence;
3. freeze a reproducible baseline and evaluation protocol;
4. run multi-seed ablations; and
5. repeat randomized hardware trials with measured outcomes.

Adding more reward terms or advanced learning methods before closing this loop would increase experimental ambiguity.

<a id="evidence-status"></a>
## Evidence status

| Area | Current evidence supports | Current evidence does not support |
| --- | --- | --- |
| Training stack | Environment, PPO, privileged-teacher, and distillation paths are implemented and have smoke evidence. | Converged policy quality, cross-seed stability, or superiority over another controller. |
| Robustness | Terrain, randomization, latency, noise, push, and per-leg fault controls exist. | Robustness claims when the reviewed run disabled those controls. |
| Sim-to-real | Profiles, trace provenance, replay, comparison, audit, and held-out promotion boundaries exist. | Fidelity outside measured states, commands, contacts, thermal conditions, and hardware mappings. |
| Energy | Mechanical-energy and spring diagnostics exist. | Measured battery-energy savings or electrical cost-of-transport improvement. |
| Deployment | Observation/action dimensions, ONNX export, ROS preflight, and safety contracts exist. | Full equivalence when estimators, sensor validity, latency, filtering, or hardware feedback differ from training. |

<a id="training-run-observation"></a>
## Training-run observation

The reviewed run `2026-08-13_11-05-09_wheg_locomotion_reform_v1` used one seed, stage 1, and plane terrain. Its resolved configuration disabled domain randomization, mass/friction randomization, actuator faults, observation latency and noise, pushes, and terrain curriculum; `spring_calibrated` was false.

The generated report froze its chart at iteration 9,545. In that snapshot, `Policy/mean_noise_std` rose from `0.600279` to `10.229902`; the RSL-RL configuration clips actions at `1.0`. A later direct read found 10,000 scalar records through iteration 9,999, a final value of `10.695706`, `model_9999.pt`, and exported Torch and ONNX policies. This corrects the report's uncertainty about run completion but does not establish policy quality.

At iteration 9,545 the materialized snapshot recorded mean commanded forward velocity `0.339982 m/s` and mean actual forward velocity `0.550391 m/s`. That is a signal to test command bias on a held-out sweep, not proof that reward shaping caused overspeed. The saved `velocity_error` channel has its own aggregation semantics and must not be treated as the arithmetic difference of those two means.

The increasing exploration scale is likewise diagnostic, not a failure verdict. Interpret it only with action-saturation histograms, KL divergence, entropy, log-standard-deviation behavior, and held-out evaluation.

<a id="evidence-gates"></a>
## Evidence gates

| Gate | Present risk or unknown | Closure evidence |
| --- | --- | --- |
| Physical-model truth | Mass, center of mass, link inertia, joint stops, friction, backlash, spring, and damping are not all bound to accepted measurements. | Versioned measurements plus held-out sim-to-real error thresholds for the intended operating envelope. |
| Contact truth | Simulator contact force and phase proxies are not yet validated against real contact labels. | Synchronized FSR, current, or foot-contact labels with timing, precision, and recall evidence. |
| Reward-preset contract | The panel supports run-scoped overrides, but the report flags possible drift between editable presets and the active reward schema. | Contract tests proving each preset changes the resolved active reward fields and saved run configuration as intended. |
| Command objective | The run shows forward command bias, but causality is untested. | A bounded A/B shaping ablation and held-out command sweeps reporting bias and RMSE across the command envelope. |
| Exploration scale | Policy standard deviation rose while actions were clipped; saturation and policy-update diagnostics were absent from the report. | Saturation, KL, entropy, and log-standard-deviation telemetry with explicit stop criteria and stable held-out results. |
| Train/deploy observations | The deployed stack can substitute estimators or default values for signals available in simulation. | Torch-to-ONNX-to-ROS replay parity plus estimator, latency, filter, dropout/mask, and hardware sensor-contract evidence. |
| Evaluation method identity | The report flags a risk that evaluation compatibility behavior may change the effective controller. | A fail-closed evaluation configuration, or an explicit controller-plus-policy method with compatibility on/off ablation. |
| Energy provenance | Mechanical power, spring terms, electrical measurements, and proxies can be confused. | Per-channel provenance labels and measured electrical cost of transport, `integral(VI dt) / (mgd)`, at matched command and achieved speed. |

<a id="minimum-evidence-contract"></a>
## Minimum evidence contract

- Freeze tasks, command envelopes, metrics, resolved configurations, checkpoints, code revision, dependencies, and hardware revision before comparison.
- Use a cheap funnel: unit/contract checks, bounded simulator smoke, one-seed screening, at least three independent seeds for exploration, and preferably five independent seeds for confirmatory results.
- Save per-episode rows. Treat training seed and hardware trial as experimental units; do not count correlated environment-time samples as independent results.
- Report tracking error, success, falls, recovery, distance, temperature, peak current, contact accuracy, latency, and sim-to-real error together with energy. Report failures, backtracking, and zero-distance episodes rather than dropping them.
- For hardware comparisons, randomize trial order and compare controller-only, residual-policy, and relevant direct-policy baselines under the same command and condition. Use paired spring enabled, locked, bypassed, or swapped trials where mechanically possible.
- Use hierarchical resampling across seeds and episodes for continuous outcomes and interval estimates for success proportions. Publish the protocol, resolved configuration, checkpoints, calibration evidence, per-episode data, and representative video with any result.

<a id="research-direction"></a>
## Research direction

The strongest near-term hypothesis is a causal study of passive-compliance energy effects at matched locomotion performance, combined with contact-aware residual control under sensor or leg faults. Its primary energy endpoint must be measured electrical energy, not a simulator torque proxy. This is a proposed direction, not an accepted novelty claim.

Contact belief, history-based state estimation, sensor dropout with validity masks, concurrent teacher/student training, symmetry-aware policies, residual dynamics, cross-simulator prediction, off-policy RL, and world models remain candidate methods. Advance them only when they close a named evidence gate or provide a preregistered comparison; defer them when they merely add method complexity.

<a id="actions"></a>
## Actions

- [ ] Close the correctness gates for reward configuration, command bias, exploration telemetry, observation parity, evaluation identity, and energy labels before another publication-scale run.
- [ ] Establish physical-model, contact, and electrical measurement truth with accepted calibration evidence.
- [ ] Freeze a baseline and held-out suite, then run the multi-seed and hardware protocol.
- [ ] Select a paper claim only after the evidence identifies a defensible effect; complete a dedicated literature and prior-art review before asserting novelty.

The ordered work is maintained in the [current project roadmap](../roadmap/current-priorities.en.md).

<a id="evidence"></a>
## Evidence

- [Training and policy architecture](../developers/architecture/training-and-policy.en.md)
- [Sim-to-real calibration architecture](../developers/architecture/sim-to-real.en.md)
- [Reward and energy model](../developers/architecture/reward-and-energy.en.md)
- [Policy and deployment contract](../reference/policy-contract.en.md)
- [Developer validation tiers](../developers/testing/validation.en.md)
- [2026-07-09 project audit](2026-07-09-project-audit.en.md)

<a id="limitations"></a>
## Limitations

The audit had no direct access to a weighed robot, link-level inertial measurements, calibrated motor bench, synchronized contact labels, voltage/current/temperature traces, or repeated hardware trajectories. It evaluated one deterministic stage-1 seed, not a multi-condition result. The report's literature scan was not systematic, so all proposed paper directions require a fresh primary-source and prior-art review.

<a id="provenance"></a>
## Provenance

The raw generated bundle is preserved outside canonical documentation in local recovery commit `02ebb53cf9da8db47952d3cf264801f44f27d82c`. The PDF SHA-256 is `91ac09d053d3859fd1dadc9b0c73d31e3d0afdd8febb2aa9ba6b93c8420b6dca`. Its chart and SQL snapshot stop at iteration 9,545; the completion correction above comes from a later direct read of the same TensorBoard event and run directory. Generated HTML, previews, scripts, SQL snapshots, and the PDF are intentionally not part of canonical `main`.

<a id="follow-up"></a>
## Follow-up

Review this audit when a gate receives durable closure evidence or when a proposed research claim is selected. New experimental results belong in a new bilingual experiment summary; corrections to this published audit require a dated addendum.
