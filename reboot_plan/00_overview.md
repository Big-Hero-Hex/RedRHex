# 00 — Overview: Core-First Soft Reboot

## Why reboot now

RedRHex has working research and operations around a core that is difficult to reason
about in isolation. The Isaac environment currently mixes simulator I/O with observation
and reward math, gait/command state, randomization, buffers, and compatibility behavior.
The default repository test collector is also unsafe: it imports two root Isaac scripts
as tests, while `.gitignore` hides intended tests.

A hard rewrite would discard valuable behavior and make comparison impossible. This
soft reboot retains the existing task and operational interfaces, validates the legacy
simulator, and then replaces internal seams incrementally with evidence at every gate.

## Scope

### In scope

- A reproducible P0 foundation: safe test discovery, runtime provenance, frozen-path
  guards, and explicit artifact policy.
- A mandatory pre-baseline simulation/gravity diagnostic gate.
- A sibling, dependency-light `redrhex_contract` package for stable interface facts.
- A sibling, Torch-only `redrhex_core` package for testable behavior/math.
- The existing `RedRhex` package retained and reduced to the Isaac Lab adapter.
- CPU unit/contract/golden tests, Isaac adapter/simulation tests, and acceptance evidence.

### Frozen during the reboot

- Working/training panel and its remote system: `tools/training_panel/**`.
- Reward agent: `tools/reward_agent/**`.
- ROS2 implementation and interfaces: `ros2_ws/**`.
- Existing Gym IDs, train/play/evaluation entry points, checkpoint compatibility,
  observation/action dimensions, and run/log/artifact behavior.

Frozen means read-only plus regression testing, not removed or disabled.

### Deferred until after core acceptance

- Panel auth/IPC/performance work, remote-system changes, reward-agent integration, and
  ROS generation or deploy changes.
- Physics/reward research changes, contact sensors, estimator work, config redesign,
  asset relocation, broad repository cleanup, and new task IDs.
- Any Isaac Lab/Sim upgrade.

## Principles

1. **Validate before preserving.** A golden dump of wrong gravity or frame semantics
   would only preserve a bug. P1 must pass before P2 captures a baseline.
2. **One-way dependencies.** Core code cannot import Isaac, Gym, ROS, panel, remote, or
   reward-agent code. The adapter may depend on core and contract, never the reverse.
3. **Behavior-preserving extraction.** Structural extraction does not include intended
   reward, timing, logging, or physics changes. Those need separate post-reboot decisions.
4. **Explicit state and evidence.** Time, RNG state, command/gait state, reset masks,
   config, and provenance are recorded at the seams needed for replay.
5. **Small, reversible cuts.** One seam at a time; legacy code is deleted only when its
   replacement has direct unit, golden, adapter, and smoke evidence.
6. **Humans judge physical truth.** Automation measures and localizes; it does not turn
   an unknown or blocked physical fact into a pass.

## Definition of done

- [ ] P0–P7 are complete with linked command, artifact, and commit/ADR evidence.
- [ ] `redrhex_contract` imports with the Python standard library only and owns stable
      ordering, dimensions, units, rates, scales, and versioning.
- [ ] `redrhex_core` imports and runs on CPU with Torch but without Isaac/ROS/Gym/UI.
- [ ] `RedRhex` remains the Isaac adapter; both current Gym IDs and current CLI/checkpoint
      behavior pass regression tests.
- [ ] The mandatory gravity/frame/model gate passes before the baseline tag exists.
- [ ] Small CI fixtures and full local artifacts have a consistent, documented policy.
- [ ] Every extracted slice passes hand-computable unit tests and golden replay.
- [ ] Frozen consumer tree guards and their existing regressions pass at acceptance.
- [ ] Simulator smoke and invariant diagnostics pass on the pinned environment.
- [ ] The rebooted fixed-seed reference protocol is within its predeclared comparison
      rules, or an explicitly approved ADR explains an intentional difference.
- [ ] An acceptance report and final reboot tag make the result reproducible.

## Completion boundary

The core reboot ends at P7. Unfreezing the panel, remote system, reward agent, or ROS is
a new decision after P7, even if they already work unchanged against the accepted core.
