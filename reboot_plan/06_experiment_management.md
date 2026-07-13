# 06 — Baseline and Experiment Evidence

Most experiment-management redesign is deferred until after P7. During the core reboot,
this document governs only the P2 legacy reference and P7 acceptance comparison.

## Baseline lifecycle

```text
P1 simulator validated
  -> P2 baseline ID created and immutable
  -> extraction compares against P2
  -> P7 acceptance comparison
  -> optional accepted reboot baseline promoted after human review
```

No checkpoint, rollout, or historical run becomes the reboot oracle merely because it
already exists. It must be linked to the validated P1 source/config/provenance.

## Baseline ID and manifest

Use a descriptive immutable ID, for example:

```text
legacy-validated-2026-07-<short-sha>
reboot-accepted-2026-08-<short-sha>
```

Each manifest records task, source/tag, asset/config hashes, runtime provenance, command,
environment count, iterations/steps, seeds, checkpoint hashes, metrics schema, golden
schema, comparison rules, and all artifact hashes.

## Reference protocol rules

- Define commands and thresholds before launching the first seed.
- Use the same task, seed set, environment count, iteration budget, evaluation command,
  and metric export at P2 and P7.
- Keep raw TensorBoard, checkpoints, videos, and full tensors local/ignored; track the
  manifest and compact metrics needed to review the decision.
- Report every seed, failed run, interruption, and retry. Do not select only the best.
- Treat learning curves as noisy evidence. Predeclare aggregation/bands rather than
  demanding stepwise numerical equality from training.

## Golden versus training evidence

- Golden rollout replay answers: “Did this structural seam preserve recorded behavior?”
- Simulator diagnostics answer: “Was the legacy model internally consistent enough to
  serve as that oracle?”
- Reference training answers: “Did the accepted whole system preserve learning behavior
  within expected variation?”

None substitutes for the others.

## Deferred until after P7

- new experiment-overlay architecture;
- panel history/override migration;
- reward-agent experiment-store integration;
- automatic ablation orchestration;
- MPC campaign and research KPI redesign.

Those may reuse the accepted contract/core packages, but they are not acceptance
criteria for the core reboot.
