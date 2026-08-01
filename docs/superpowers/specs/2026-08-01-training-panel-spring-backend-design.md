# Training Panel Spring-Backend Propagation Design

## Goal

Allow a training-panel run to select `explicit` or `native` torsion-spring physics and guarantee that later panel playback, automatic video recording, and policy export reuse the same backend. Launch a clearly labeled provisional native ForwardFast run for 200 iterations after the integration is verified.

## Scope

This change is limited to `tools/training_panel` parameter handling, command construction, run metadata, the local panel form, and panel tests. It does not change spring equations, Isaac environment configuration defaults, rewards, task registration, checkpoint dimensions, calibration gates, promotion rules, or deployment rules.

## Data Flow

1. The training form exposes a required spring-backend selector with `explicit` and `native` choices. Existing behavior remains the default: `explicit`.
2. `TrainingParams` parses, validates, serializes, and stores that value in the panel run record.
3. `training_argv()` passes `--spring-backend <value>` to `scripts/rsl_rl/train.py`.
4. Panel playback, video, and export resolve the backend from the originating run metadata. For older or discovered runs, they fall back to `params/torsion_spring.yaml`; if no valid metadata exists, they retain the historical `explicit` default.
5. `play_argv()` and `export_onnx_argv()` pass the resolved backend to `scripts/rsl_rl/play.py`, preventing a native checkpoint from being replayed under explicit physics.

## UI and Compatibility

The selector sits beside the existing task/device fields. Training defaults, tweak-to-form restoration, run summaries, and run-detail metadata expose the selected backend. API clients that omit the new field continue to receive `explicit`, so existing panel and remote requests remain compatible.

Invalid backend values fail before a process is queued. A stored invalid value never reaches Isaac; playback/export resolution falls back only when metadata is absent, not when a new request is malformed.

## Verification

Tests will be written first to prove:

- training parameters accept only `explicit` or `native`;
- training commands include the selected backend;
- playback/video/export commands include the originating run backend;
- old run records preserve explicit behavior;
- the form submits and restores the selector without changing unrelated fields.

After focused and full panel tests pass, start a worktree-local panel instance and submit:

- task: `Template-Redrhex-ForwardFast-Direct-v0`;
- backend: `native`;
- seed: `42`;
- environments: `4096`;
- iterations: `200`;
- label: `torsion_native_v11_provisional_200`;
- headless training with the panel's automatic high-quality playback video.

The run remains stamped `uncalibrated` and cannot be promoted or deployed. Merge/push consideration happens only after training completes, the panel records its video successfully, and that video is inspected.
