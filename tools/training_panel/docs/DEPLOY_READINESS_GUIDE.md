# RedRHex Deploy Readiness Guide

This guide defines the v1 policy deployment gate used by the local Training Panel's **Deploy** tab. It is designed for Jetson ROS2 deployment through `redrhex_rl_controller` and ONNX Runtime.

## Readiness Levels

- `ready`: required export, ONNX, runtime, contract, parity, and safety checks passed.
- `review`: no blocking failure, but at least one stage warned or was skipped.
- `blocked`: at least one required stage failed. Do not proceed to robot bring-up.

MuJoCo and ROS mock checks are useful evidence, but v1 keeps them advisory or optional when local dependencies are missing.

## Runtime Environment

Deploy readiness validation runs in the panel Python environment, normally `/home/lab_user1/miniconda3/bin/python`. Install deploy-only validation dependencies there: `onnx`, `onnxruntime`, `mujoco`, and `torch`.

Isaac-dependent actions still use the Isaac launcher environment. Training, play, Isaac video recording, and standalone ONNX export use `isaaclab.sh -p`; `Export ONNX + Validate` first launches export through that Isaac path, then returns to the panel Python for ONNX Runtime, parity, safety, and MuJoCo validation.

The Deploy defaults API and deploy process log print the exact validation Python executable and dependency status. When a stage is skipped for a missing module, check that panel runtime first, not `env_isaaclab_bin`.

## Stage Meaning

- **Export Integrity** verifies `model_*.pt`, `exported/policy.pt`, `exported/policy.onnx`, `params/env.yaml`, and `params/agent.yaml`, then records file sizes and SHA-256 hashes.
- **Static ONNX** runs ONNX checker and shape inference when the `onnx` package is installed.
- **ONNX Runtime** loads the model with explicit providers, runs finite deterministic inference, and records latency percentiles.
- **Torch/ONNX Parity** compares TorchScript and ONNX outputs on zero, random, boundary, and optional golden observations.
- **Observation/Action Contract** checks RedRHex observation dimensions, history length, action dimension, command ranges, joint order, and deployment YAML safety defaults.
- **Safety Fault Injection** verifies that synthetic E-stop, timeout, tilt, NaN, action limit, command range, and loop deadline failures are rejected.
- **ROS Mock/Fake Sensor** optionally launches fake sensors and the mock bridge with motor output disabled.
- **MuJoCo Readiness** resolves the RedRHex URDF mesh paths, applies MuJoCo compiler repair options, generates a rollout MJCF, injects the 12 policy-controlled motor actuators, then runs deterministic ONNX policy scenarios. This remains advisory until the model config is explicitly marked calibrated.

## Panel Workflow

1. Open the local panel and choose **Deploy**.
2. Select a completed run.
3. Use **Validate Existing ONNX** if `exported/policy.onnx` already exists.
4. Use **Export ONNX + Validate** when the run has a checkpoint but no current ONNX export.
5. Use **Run MuJoCo Smoke** to run only the MuJoCo rollout stage against the selected run's ONNX.
6. Use **Open MuJoCo Viewer** for a native desktop playback window on the training PC.
7. Use **Record MuJoCo MP4** to save deterministic MuJoCo playback as a browser-playable run artifact.
8. Keep `ROS mock/fake sensor launch` off unless ROS2 is sourced and `ros2_ws` has been built.
9. Inspect the readiness badge, stage list, JSON report, and deploy console.

Reports are written under:

```text
<run_log_dir>/deploy/readiness_<pipeline_id>.json
<run_log_dir>/deploy/readiness_<pipeline_id>.md
```

MuJoCo smoke artifacts are written under:

```text
<run_log_dir>/deploy/mujoco_<pipeline_id>/redrhex_resolved.urdf
<run_log_dir>/deploy/mujoco_<pipeline_id>/redrhex_generated.mjcf.xml
<run_log_dir>/deploy/mujoco_<pipeline_id>/redrhex_rollout_model.xml
<run_log_dir>/deploy/mujoco_<pipeline_id>/mujoco_calibration.json
<run_log_dir>/deploy/mujoco_<pipeline_id>/mujoco_rollout_metrics.json
<run_log_dir>/deploy/mujoco_<pipeline_id>/mujoco_trace_<scenario>.json
```

MuJoCo viewer/recording artifacts are written under:

```text
<run_log_dir>/deploy/mujoco_playback_<process_id>/redrhex_resolved.urdf
<run_log_dir>/deploy/mujoco_playback_<process_id>/redrhex_generated.mjcf.xml
<run_log_dir>/deploy/mujoco_playback_<process_id>/redrhex_rollout_model.xml
<run_log_dir>/deploy/mujoco_playback_<process_id>/mujoco_playback_report.json
<run_log_dir>/deploy/mujoco_playback_<process_id>/mujoco_playback_trace_<scenario>.json
<run_log_dir>/deploy/mujoco_playback_<process_id>/mujoco_<scenario>.mp4
```

Generated MuJoCo artifacts are evidence for the selected run only. Do not commit them unless a future calibrated model is intentionally promoted into source control.

## MuJoCo Calibration

The default MuJoCo rollout config uses the RedRHex deploy contract: 56 or 280 observations, 12 actions, a 60 Hz policy loop, 6 main-drive motors, and 6 ABAD motors. The default config sets `calibrated=false`, so clean rollouts produce a `warn` stage result rather than a hard `pass`.

The built-in smoke scenarios are:

- `stand_zero`: zero command.
- `forward_mid`: moderate forward command.
- `yaw_mid`: moderate yaw command.
- `boundary_command`: command limits.

Each scenario records compile status, completed steps, fall/divergence flags, NaN/Inf detection, base height and roll/pitch ranges, joint-limit violations, actuator saturation, action min/max, latency percentiles, and a final state summary. Once the model is calibrated against real robot or trusted simulator behavior, set the calibration config to `calibrated=true` so threshold failures can become blocking evidence.

Viewer and MP4 playback use the same scenarios and policy/controller path. The policy is evaluated at 60 Hz and its action is held across MuJoCo substeps according to the model timestep.

## Jetson Handoff

Copy only the reviewed artifacts:

```text
exported/policy.onnx
ros2_ws/src/redrhex_rl_controller/config/redrhex_policy.yaml
deploy/readiness_<pipeline_id>.json
deploy/readiness_<pipeline_id>.md
```

On Jetson, run preflight before any hardware enable:

```bash
source /opt/ros/humble/setup.bash
source ~/RedRhex/RedRhex/ros2_ws/install/setup.bash
ros2 run redrhex_rl_controller preflight_check \
  --onnx /home/jetson/redrhex_models/policy.onnx \
  --config ~/RedRhex/RedRhex/ros2_ws/src/redrhex_rl_controller/config/redrhex_policy.yaml
```

Then run mock mode with fake sensors and disabled motor output before connecting motor power.

## Hardware Bring-Up Gate

Do not enable policy or motors until all of these are true:

- Physical E-stop and power cutoff are ready.
- Low-level bridge heartbeat is understood.
- Single ABAD motion test passes at low power.
- Single main-drive motion test passes with the robot lifted.
- `enable_policy_on_start=false` and `enable_motor_output_on_start=false`.
- No readiness report is `blocked`.

## Rollback

If a policy fails validation or behaves unexpectedly:

1. Assert software E-stop.
2. Disable motor output.
3. Restore the last `ready` ONNX and YAML bundle.
4. Re-run deploy readiness and Jetson preflight.
5. Record the failing report path in the run notes.

## Troubleshooting

- Missing `onnx`: install the `onnx` package in the panel Python to enable checker and shape inference.
- Missing `onnxruntime`: install `onnxruntime` in the panel Python locally or `onnxruntime-gpu` on Jetson. MuJoCo rollouts require ONNX Runtime because they execute the exported policy in the simulation loop.
- Dependency appears installed but stage still skips: open `/api/deploy/defaults` or the deploy process log and confirm `deploy_runtime_python` is the panel Python. Readiness validation does not use `env_isaaclab_bin`.
- Torch/ONNX mismatch: re-export from the same checkpoint and verify normalizer export.
- Contract warning for base linear velocity: bench checks can use zero velocity, but real locomotion needs an estimator.
- ROS mock skipped: build `ros2_ws` with `colcon build` and source `install/setup.bash`.
- MuJoCo skipped: install `mujoco`, install `onnxruntime`, and confirm the Deploy tab model path points at `test_7_description/test_7_description/urdf/test_7.urdf` or another valid model.
- MuJoCo warned: inspect the per-scenario metrics JSON and traces. A warning is expected while `calibrated=false`; metric failures under a calibrated config should block hardware bring-up.
- MuJoCo viewer unavailable: confirm the panel process has a desktop display such as `DISPLAY=:10.0` and that `mujoco.viewer` plus `glfw` import in the panel Python.
- MuJoCo MP4 unavailable: install `imageio` and `imageio-ffmpeg` in the panel Python environment.
