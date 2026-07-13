# RedRHex Sim-to-Real Physics Calibration

This workflow calibrates one observable subsystem at a time. It does not change the production training physics unless an operator explicitly passes a reviewed `CalibrationProfileV1` with `--physics-profile`.

## Current blocking audit result

The production USD currently resolves to a runtime mass of approximately `1.7985 kg`, while comments elsewhere refer to `14 kg`. Do not assume either value is correct. Weigh the assembled robot and complete the mass/CoM audit before fitting actuators or contact. The measured correction belongs in a candidate profile; it must not be copied into the default training configuration without held-out validation and review.

## Non-negotiable hardware prerequisites

Before any enabled probe:

1. Suspend and mechanically secure the robot so the selected leg cannot strike a person, cable, fixture, or another leg.
2. Verify a physical E-stop by observing power removal, not only a ROS boolean.
3. Configure conservative hardware current limits.
4. Verify the sbRIO watchdog disables outputs when command traffic stops.
5. Stop every other `/motor/command` and `/redrhex/motor_commands` publisher.
6. Confirm raw `/motor/state` is live and fresh.
7. Keep ABAD output disabled for the main-drive probe.

Software checks are additional safeguards; they do not replace these controls.

## Build the ROS command contract

The message definition changed, so rebuild and source the workspace before running the bridge, controller, or probe:

```bash
cd ros2_ws
colcon build --packages-select redrhex_msgs redrhex_lowlevel_bridge redrhex_rl_controller
source install/setup.bash
```

The current development environment does not contain ROS 2 or `colcon`; this build must be completed in the robot's sourced ROS environment.

## Calibration order and gates

Run the stages in this order. Do not use a later stage to compensate for a failed earlier stage.

1. **Contract and geometry audit:** mass, link/joint order, axes, encoder count/zero/sign, mechanical limits, units, frames, and IMU mounting.
2. **Command mapping and timing:** PWM mapping, observed state rate, velocity filtering, and aggregate command-to-motion delay.
3. **Main-drive response:** both directions, all legs, at least three repetitions, with velocity derived offline from raw encoder position.
4. **Known load:** short, manually supervised force/lever-arm measurements before fitting effort saturation.
5. **Mass, CoM, ABAD scale, and passive springs:** apply measured quantities directly.
6. **Contact:** measure static/dynamic friction physically, apply the coefficients directly, then validate foot contact and static settling in Isaac.
7. **Held-out validation:** replay conditions not used for fitting and report timing, actuator, rigid-body, spring, and contact results separately.

Unit, frame, sign, mass, or contact-sensor failures block later fitting. `run-sim` deliberately refuses the `friction` scenario because passive settling is not a controlled pull test.

## Scenario and probe workflow

List the reviewed scenarios:

```bash
python -m tools.sim2real list
```

Preview the bounded probe before starting ROS output:

```bash
ros2 run redrhex_rl_controller sim2real_probe --main-index 0 --dry-run
```

The probe has a fixed 60 Hz, three-repeat, low-energy sequence. Its only selectable physical output is main-drive index `0..5`; amplitude, rate, durations, waveform, and safety checks are not CLI parameters. Actual output requires both `--enable` and `--confirm-risk`. Every tick requires the probe to be the sole `/redrhex/motor_commands` publisher, and a scheduler delay that reaches one 60 Hz period aborts instead of replaying stale commands in a burst.

Only after every prerequisite above has been physically verified, start the reviewed sequence with:

```bash
ros2 run redrhex_rl_controller sim2real_probe \
  --main-index 0 \
  --enable \
  --confirm-risk
```

Each leg has a bound reviewed scenario, `suspended-main-0-step-coast` through `suspended-main-5-step-coast`. The preview and JSON event stream report that exact scenario ID, version, and SHA-256 hash. Scenarios 0–4 are calibration conditions; main 5 is the default unused-leg holdout. Do not move the holdout into a sweep after looking at its result.

## Record raw evidence

Record the raw BioRoLa topics. `/motor/command` and `/motor/state` are mandatory for main-drive response tests; IMU and power are optional independent streams:

```bash
ros2 bag record -o datasets/raw/main-leg0-run1 \
  /motor/command \
  /motor/state \
  /redrhex/sim2real_probe/events \
  /imu/data \
  /power/state
```

Do not use `/motor_feedback.header.stamp` for latency. The importer uses each rosbag record's `SequentialReader` receive timestamp and preserves independent command, motor-state, IMU, and power time vectors. Never edit or overwrite the raw bag.

## Import an immutable real episode

Use a profile containing the encoder/PWM calibration when one exists. Imports without it are marked as provisional:

```bash
python -m tools.sim2real import-real datasets/raw/main-leg0-run1 \
  --scenario suspended-main-0-step-coast \
  --output . \
  --dataset-id main-drive-bench-v1 \
  --episode-id leg0-run1 \
  --units-json '{"command":"rad/s","position":"rad"}' \
  --frames-json '{"command":"main_0","position":"main_0"}' \
  --profile profiles/candidate-v1.json
```

For another leg, select its matching built-in scenario and frames. The importer checks the raw BioRoLa enable bits and rejects bags that enable a different main drive, enable multiple main drives, or never enable the scenario joint.

Within a reviewed hardware profile, `pwm_scale.main_N` is the inverse command conversion in `(rad/s)/raw-PWM`; for example, a bridge setting of `120 PWM/(rad/s)` starts at `1/120`. It is a recorded command mapping, not an actuator-physics fit. Imports using repository fallbacks are marked provisional; do not label a provisional normalized-PWM command as `rad/s` or compare it to an Isaac velocity target.

The resulting layout is:

```text
datasets/sim2real/<dataset-id>/
  manifest.json
  raw/<immutable-rosbag>/
  episodes/<episode-id>/
    trace.npz
    metadata.json
```

`trace.npz` contains numeric arrays only. Metadata records units, frames, joint order, clock semantics, scenario/profile versions, Git and asset/config hashes, calibration constants, and raw-data hashes.

## Run the matching Isaac scenario

Command scenarios always run their full declared duration. Do not pass `--steps` for them:

```bash
export PYTHONPATH="$PWD/source/RedRhex:$PWD${PYTHONPATH:+:$PYTHONPATH}"
$ISAACLAB_ROOT/isaaclab.sh -p -m tools.sim2real run-sim \
  --scenario suspended-main-0-step-coast \
  --mode fixed-base \
  --physics-profile profiles/candidate-v1.json \
  --output outputs/sim2real/main-0-step-coast-candidate-v1 \
  --headless \
  --device cuda:0
```

The audit scenario alone permits an explicit finite duration, for example:

```bash
$ISAACLAB_ROOT/isaaclab.sh -p -m tools.sim2real run-sim \
  --scenario audit \
  --mode contact \
  --steps 240 \
  --require-contact \
  --output outputs/sim2real/audit-contact \
  --headless \
  --device cuda:0
```

Only the six terminal-foot sensor can satisfy contact validation. Body/chassis contacts are logged separately.

## Compare and generate bounded candidates

Compare one real episode with one matching simulator trace:

```bash
python -m tools.sim2real compare \
  datasets/sim2real/main-drive-bench-v1/episodes/leg0-run1 \
  outputs/sim2real/main-0-step-coast-candidate-v1 \
  --scenario suspended-main-0-step-coast \
  --output outputs/sim2real/main-0-step-coast-comparison.json
```

Generate a one-factor sensitivity set first:

```bash
python -m tools.sim2real sweep profiles/candidate-v1.json \
  --scenario suspended-main-0-step-coast \
  --mode one-factor \
  --space-json '{"simulation_physics.main_drive.damping":[0.8,1.0,1.2]}' \
  --output outputs/sim2real/main-drive-damping-sweep
```

Use a bounded two-parameter coarse grid only after sensitivity work shows correlation. Candidate cache keys bind the scenario and full profile. Run each candidate in a fresh Isaac process. Do not combine subsystem errors into a global RMSE, and do not introduce an optimizer until repeatability and identifiability are demonstrated.

## Profile validation and explicit use

Validate JSON fields and physical ranges:

```bash
python -m tools.sim2real validate-profile profiles/candidate-v1.json
```

An explicit candidate may be used for training or playback:

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task <task-id> \
  --physics-profile profiles/candidate-v1.json

$ISAACLAB_ROOT/isaaclab.sh -p scripts/rsl_rl/play.py \
  --task <task-id> \
  --physics-profile profiles/candidate-v1.json
```

Omitting `--physics-profile` preserves existing defaults and does not import the calibration package.

## Promotion gate

A candidate remains experimental until all of the following are reviewed:

- every mandatory real condition has at least three repetitions;
- each fitted subsystem has an unused leg, direction, level, or load condition;
- every held-out simulated metric lies within `max(instrument uncertainty, 2 × real-run standard deviation)` of the real mean;
- actuator, timing, rigid-body, spring, and contact results are reported independently;
- the audit has no unresolved unit, frame, sign, mass, collision, or contact failure;
- no unrelated parameter was used to conceal a subsystem-model mismatch;
- a reviewed configuration change explicitly promotes the profile.

If the implicit actuator cannot enter the held-out response envelope, stop tuning mass or friction and open a separate DC/PWM or learned-actuator follow-up.
