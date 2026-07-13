# RedRHex Core-Reboot Agent Brief — Design Template

Use this only after adapting it to the repository's actual instruction-file convention.
The live gate is `reboot_plan/STATUS.md`; never infer progress from this template.

## Objective

Validate the legacy simulator, then extract sibling `redrhex_contract` and
`redrhex_core` packages behind the existing `RedRhex` Isaac adapter. Preserve current
Gym, CLI, checkpoint, and artifact behavior.

## Frozen source

Do not modify:

- `tools/training_panel/**`, including remote worker/web/Supabase;
- `tools/reward_agent/**`;
- `ros2_ws/**`.

Read-only inspection and regression tests are allowed. A frozen-path guard is mandatory.

## Dependency rules

- `redrhex_contract`: standard library only; stable order/shape/unit/rate/scale/slice/version facts.
- `redrhex_core`: Torch + contract only; explicit state/RNG/time; no simulator writes.
- `RedRhex`: Isaac adapter and legacy compatibility; may depend on both sibling packages.

No core package may import Isaac, Gym, ROS, panel, remote, or reward-agent code.

## Gate order

```text
P0 safe tests/provenance/interfaces
-> P1 validate gravity/frames/model
-> P2 validated tag/golden/reference
-> P3 package scaffolding
-> P4 contract extraction
-> P5 core extraction
-> P6 adapter cutover
-> P7 acceptance
```

Never capture/update a baseline to resolve a failure before P1. Never combine extraction
with physics, reward, command-timing, or logging-semantic changes.

## Working rules

1. Read `STATUS.md`, current evidence, and scoped diff before work.
2. State one invariant and write the cheapest failing test first.
3. Make one reversible slice; preserve the legacy path until replacement evidence passes.
4. Run focused then aggregate required gates; verify runtime provenance before Isaac work.
5. Stop at unexplained parity/physics failures. Do not widen thresholds or skip tests.
6. Update evidence and status only with commands, artifacts, hashes, and commit/ADR.
7. Hardware actions and changes to frozen consumers always require separate approval.

The P0 implementation plan supplies canonical commands; do not invent hard-coded machine
paths or silently change the existing panel smoke defaults.
