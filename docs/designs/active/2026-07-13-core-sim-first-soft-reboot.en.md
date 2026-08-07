---
id: core-sim-first-reboot-design
title: Core-First Simulation-First Soft Reboot
lang: en
audience: developer
type: design
status: proposed
owner: core
last_reviewed: 2026-08-07
---

<a id="provenance"></a>
## Provenance and status

This proposal imports unique durable architecture from branch `reboot/core-sim-first` at `f40d3c2`. It is not active implementation. The source branch remains preserved for its detailed plans and evidence drafts; this pair records only the proposal needed for current project decisions.

<a id="problem"></a>
## Problem

The working Isaac environment combines simulator I/O, observation and reward math, gait/command state, randomization, buffers, and logging. Pure behavior is difficult to import without Isaac, while contract facts are mirrored into ROS and can drift. A hard rewrite would discard working training, panel, reward-agent, and deployment behavior before trustworthy comparison exists.

<a id="proposal"></a>
## Proposed architecture

Retain `RedRhex` as the Isaac adapter and preserve both Gym IDs, script interfaces, checkpoints, and artifact layout. Introduce sibling packages:

- `redrhex_contract`: standard-library-only ordering, dimensions, units, rates, scales, slices, and a contract version.
- `redrhex_core`: Torch-only observation, reward, termination, action, gait, command, randomization, and buffer logic with explicit inputs and outputs.

The adapter alone reads simulator state and writes actuator targets. Panel, remote, Reward Agent, and ROS stay frozen during extraction and are exercised through regression tests.

<a id="gates"></a>
## Proposed gates

1. Establish safe test discovery, toolchain provenance, frozen-interface guards, and artifact contracts.
2. Validate gravity, units, frames, mass/inertia, contacts, timing, action response, rewards, commands, and determinism.
3. Only then capture a validated legacy oracle and reference-training baseline.
4. Scaffold packages, extract contract facts, extract pure behavior one seam at a time, and thin the adapter.
5. Accept only after CPU, golden, simulator, frozen-boundary, and fixed-seed comparison evidence.

<a id="decision-points"></a>
## Required decisions

Approval is still required for the reboot itself, the pinned Isaac checkout, missing physical facts, any physics/frame correction, the immutable baseline protocol, intentional golden differences, and final acceptance. A configured gravity vector or visual impression alone is not evidence of a bug.

<a id="non-goals"></a>
## Non-goals

The proposal excludes panel/remote/reward-agent/ROS redesign, new task IDs, artifact migration, reward research, hardware estimator work, asset relocation, broad cleanup, and Isaac upgrades during extraction. Those remain separate post-acceptance projects.
