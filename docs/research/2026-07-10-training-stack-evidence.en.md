---
id: training-stack-evidence-2026-07
title: 2026-07 Training Stack Evidence Summary
lang: en
audience: developer
type: experiment-summary
status: published
owner: training
last_reviewed: 2026-08-07
---

<a id="question"></a>
## Question

Did the reworked RedRHex training stack execute the intended environment, PPO, privileged-teacher, and distillation paths before long-run performance experiments?

<a id="method"></a>
## Method

Historical smoke runs used `validate_reform_stack.py` with small environment counts and short rollouts. Separate modes exercised random environment stepping, one PPO update, and a teacher checkpoint followed by one distillation update. The source report recorded temporary JSON outputs and checkpoint existence.

<a id="results"></a>
## Results

The environment smoke reported generated terrain, a 224-value history group, a 47-value critic group, a 327-value teacher group, and active fault injection in the configured sample. The PPO smoke completed an update. The teacher/distillation smoke produced both teacher and student checkpoints and completed the distillation update.

<a id="interpretation"></a>
## Interpretation

These results establish wiring and executable-path evidence: the observation groups, PPO runner, privileged teacher, and distillation runner could operate together in the tested environment. They do not establish final tracking quality, energy improvement, robustness, hardware transfer, or superiority to MPC.

<a id="provenance"></a>
## Provenance and correction policy

The values were migrated from `docs/2026_Midterm.md` at the documentation source checkpoint. Raw `/tmp` artifacts were not committed and are not independently re-executable evidence today. Corrections to this immutable summary require a dated addendum; new results require a new experiment summary.
