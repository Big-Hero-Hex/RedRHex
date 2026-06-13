# RedRHex Deploy Readiness Guide

This guide defines the v1 policy deployment gate used by the local Training Panel's **Deploy** tab. It is designed for Jetson ROS2 deployment through `redrhex_rl_controller` and ONNX Runtime.

## Readiness Levels

- `ready`: required export, ONNX, runtime, contract, parity, and safety checks passed.
- `review`: no blocking failure, but at least one stage warned or was skipped.
- `blocked`: at least one required stage failed. Do not proceed to robot bring-up.

MuJoCo and ROS mock checks are useful evidence, but v1 keeps them advisory or optional when local dependencies are missing.

## Stage Meaning

- **Export Integrity** verifies `model_*.pt`, `exported/policy.pt`, `exported/policy.onnx`, `params/env.yaml`, and `params/agent.yaml`, then records file sizes and SHA-256 hashes.
- **Static ONNX** runs ONNX checker and shape inference when the `onnx` package is installed.
- **ONNX Runtime** loads the model with explicit providers, runs finite deterministic inference, and records latency percentiles.
- **Torch/ONNX Parity** compares TorchScript and ONNX outputs on zero, random, boundary, and optional golden observations.
- **Observation/Action Contract** checks RedRHex observation dimensions, history length, action dimension, command ranges, joint order, and deployment YAML safety defaults.
- **Safety Fault Injection** verifies that synthetic E-stop, timeout, tilt, NaN, action limit, command range, and loop deadline failures are rejected.
- **ROS Mock/Fake Sensor** optionally launches fake sensors and the mock bridge with motor output disabled.
- **MuJoCo Readiness** optionally compiles/steps a MuJoCo or URDF model with the exported policy. This remains advisory until the model is calibrated.

## Panel Workflow

1. Open the local panel and choose **Deploy**.
2. Select a completed run.
3. Use **Validate Existing ONNX** if `exported/policy.onnx` already exists.
4. Use **Export ONNX + Validate** when the run has a checkpoint but no current ONNX export.
5. Keep `ROS mock/fake sensor launch` off unless ROS2 is sourced and `ros2_ws` has been built.
6. Inspect the readiness badge, stage list, JSON report, and deploy console.

Reports are written under:

```text
<run_log_dir>/deploy/readiness_<pipeline_id>.json
<run_log_dir>/deploy/readiness_<pipeline_id>.md
```

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

- Missing `onnx`: install the `onnx` package to enable checker and shape inference.
- Missing `onnxruntime`: install `onnxruntime` locally or `onnxruntime-gpu` on Jetson.
- Torch/ONNX mismatch: re-export from the same checkpoint and verify normalizer export.
- Contract warning for base linear velocity: bench checks can use zero velocity, but real locomotion needs an estimator.
- ROS mock skipped: build `ros2_ws` with `colcon build` and source `install/setup.bash`.
- MuJoCo skipped or warned: add a calibrated MJCF/URDF model and update the Deploy tab model path.
