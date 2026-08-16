---
id: operator-training-troubleshooting
title: Troubleshoot Training and Playback
lang: en
audience: operator
type: troubleshooting
status: active
owner: training
last_reviewed: 2026-08-16
---

<a id="imports"></a>
## Import or task errors

If `isaaclab`, `pxr`, `redrhex_policy_io`, or the RedRHex task cannot be imported, run `isaaclab.sh -p scripts/install_redrhex.py` and then verify with `scripts/list_envs.py`. The installer uses that same interpreter and installs the repository-local shared distribution before the extension. Do not mix a normal shell Python with the Isaac Lab interpreter.

<a id="assets"></a>
## Missing or tiny USD assets

Run `git lfs install` and `git lfs pull`. A roughly 130-byte USD file is normally an unresolved LFS pointer, not a valid robot asset.

<a id="memory"></a>
## CUDA out of memory

Reduce `--num_envs`, stop other Isaac or TensorBoard processes using GPU memory, and rerun the smoke job. Do not increase environment count again until memory use is stable.

<a id="checkpoint"></a>
## Checkpoint not found or rejected

Select a `model_*.pt` training checkpoint. When using a relative filename, also provide the matching `--load_run`. Exported `policy.pt` and TensorBoard event files are not runner checkpoints.

<a id="behavior"></a>
## Exploding, collapsing, or motionless behavior

First reproduce with a small fixed task and the original checkpoint configuration. Check automatic stage inference, terrain/reward overrides, termination metrics, base height/tilt, command tracking, and whether playback begins at `stop`. Disable stale assumptions rather than immediately changing reward weights.

<a id="panel"></a>
## Panel or remote worker problems

Confirm port 8080 is free, inspect Process Console, and verify the configured repository and Isaac paths. For remote mode, run the worker once with its private environment file, check machine heartbeat and `accept_jobs`, and keep secrets out of browser-visible configuration.

<a id="deployment"></a>
## Deployment problems

Treat any blocked readiness report, ONNX shape mismatch, non-finite inference, contract mismatch, multiple motor publishers, missing heartbeat, or E-stop fault as a stop condition. Continue with the [ROS troubleshooting guide](../../../ros2_ws/src/redrhex_rl_controller/docs/troubleshooting.en.md).
