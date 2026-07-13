# RedRHex Sim-to-Real Physics Calibration

This workflow calibrates one observable subsystem at a time. It does not change the production training physics unless an operator explicitly passes a reviewed `CalibrationProfileV1` with `--physics-profile`.

## Current blocking audit result

The production USD currently resolves to a runtime mass of approximately `1.7985 kg`, while comments elsewhere refer to `14 kg`. Do not assume either value is correct. Weigh the assembled robot and complete the mass/CoM audit before fitting actuators or contact. The measured total mass and planar CoM belong in a candidate profile; they must not be copied into the default training configuration without held-out validation and review.

The same live audit currently reports all 18 production joints as continuous. If any ABAD, main-drive, or damper joint has a real mechanical stop, record that limited range in the physical audit and correct the USD range before fitting: the mechanical-range gate intentionally fails when a physically limited joint is modeled as continuous.

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

Record the raw BioRoLa topics. `/motor/command` and `/motor/state` are mandatory for main-drive response tests. IMU is also mandatory for replay-state verification; power remains an optional independent stream:

```bash
ros2 bag record -o datasets/raw/main-leg0-run1 \
  /motor/command \
  /motor/state \
  /redrhex/sim2real_probe/events \
  /imu/data \
  /power/state
```

Do not use `/motor_feedback.header.stamp` for latency. The importer uses each rosbag record's `SequentialReader` receive timestamp and preserves independent event-command, raw-motor-command, motor-state, IMU, and power time vectors. The event topic is mandatory for the six bound step/coast probes: import requires the matching scenario ID/schema/hash, fixed 60 Hz segment order, three repetitions, ABAD disabled, one `complete`, and no `abort`. Never edit or overwrite the raw bag.

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
  --profile profiles/candidate-v1.json \
  --replay-fixture fixtures/suspended-level-v1.json
```

`--replay-fixture` is optional for metric-only imports and mandatory if the episode will be replayed. It is a reviewed JSON object containing `schema_version`, `fixture_id`, `scene_mode`, `fixture_frame`, the simulator `root_orientation_wxyz`, and the raw sensor `expected_imu_orientation_xyzw`. The importer requires at least three encoder and IMU samples in the first 0.4 s neutral window, derives all six initial velocities, and rejects joint motion, IMU angular motion, or more than 5 degrees of fixture-orientation error. Record `/imu/data` when replay is intended.

For another leg, select its matching built-in scenario and frames. The importer checks the raw BioRoLa enable bits and rejects bags that enable a different main drive, enable multiple main drives, or never enable the scenario joint. The authenticated segment events form the requested `command` timeline; `motor_command_pwm_raw` remains unchanged on its independent clock for mapping work. This preserves the initial disabled neutral even though the bridge intentionally suppresses repeated disabled raw packets.

Within a reviewed hardware profile, `pwm_scale.main_N` is the inverse command conversion in `(rad/s)/raw-PWM`; for example, a bridge setting of `120 PWM/(rad/s)` starts at `1/120`. `pwm_cap.main_N` is the resulting canonical velocity cap in `rad/s` (so a raw cap of `500` with that scale is `500/120 = 4.1667 rad/s`), not a normalized PWM fraction. `joint_direction.main_N` maps the raw bridge direction to the canonical simulator direction. Count/revolution, zero, encoder sign, command direction, PWM scale, and PWM cap must all be present before the mapping is considered complete. Imports using any fallback are marked provisional and are rejected by physics comparison.

The resulting layout is:

```text
datasets/sim2real/<dataset-id>/
  manifest.json
  raw/<immutable-rosbag>/
  episodes/<episode-id>/
    trace.npz
    metadata.json
```

`trace.npz` contains numeric arrays only. Metadata records units, frames, joint order, clock semantics, scenario/profile versions, Git and asset/config hashes, calibration constants, and raw-data hashes. Loading a managed dataset rechecks the detached metadata hash, trace hash, copied raw artifact hash, and dataset linkage; editing any of them fails closed.

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

`runtime_audit.json` uses audit schema version 2. It records the ordered
`main_0..5`, `abad_0..5`, and `damper_0..5` mapping to resolved articulation
joints, each USD axis and effective range, collision prims, per-link mass,
inertia and CoM, and the aggregate planar CoM in the robot body frame. Its
canonical JSON hash is embedded in the audit trace metadata; replacing either
file without the other is rejected.

Before fitting, create a physical audit JSON with schema version 2. It must
contain the corresponding 18 ordered joint records, six raw main-encoder
observations (start/end counts, observed count/revolution and zero, physical
angle change, and uncertainties), at least three scale and planar-CoM
measurements, the base plus all six terminal collision bodies, and two or more
known IMU resting orientations. Wrap it with the simulator artifacts in an
`audit_artifact` binding:

```json
{
  "runtime_trace": {
    "path": "audit-contact",
    "trace_sha256": "<trace SHA-256>",
    "metadata_sha256": "<metadata SHA-256>"
  },
  "runtime_audit": {
    "path": "audit-contact/runtime_audit.json",
    "sha256": "<file SHA-256>"
  },
  "physical_measurements": {
    "path": "physical-audit-v2.json",
    "sha256": "<file SHA-256>"
  }
}
```

Paths are relative to the binding file's directory. The gate derives separate
checks for units, frames, joint order, axes, encoder scale/zero/sign,
mechanical range, total mass, per-link inertia/CoM validity, measured planar
CoM, collision geometry, IMU mounting, and contact response. These checks are
computed from numeric evidence; the file cannot supply its own pass booleans.

Replay a verified real episode's independent command timeline with the same scenario and profile:

```bash
$ISAACLAB_ROOT/isaaclab.sh -p -m tools.sim2real run-sim \
  --scenario suspended-main-0-step-coast \
  --mode fixed-base \
  --replay-trace datasets/sim2real/main-drive-bench-v1/episodes/leg0-run1 \
  --physics-profile profiles/candidate-v1.json \
  --output outputs/sim2real/main-0-replay-candidate-v1 \
  --headless
```

Replay accepts only a managed dataset episode with a hash-bound manifest, metadata file, raw source, complete six-encoder mapping, and six canonical initial main-joint positions. The importer declares zero initial joint velocities plus the scenario's fixed/free-root fixture and root-state source. The runner verifies those declarations and hashes, applies all six observable joint states before frame zero, and records the effective initial state in the simulator results. `sensor_timing.aggregate_command_delay_s` is quantized to the 120 Hz physics clock and applied to requested versus applied targets in characterization, training, and playback. Other timing/filter fields are measurement metadata only and are rejected by simulation profile application instead of being silently treated as active physics.

## Compare and generate bounded candidates

Compare one real episode with one matching simulator trace:

```bash
python -m tools.sim2real compare \
  datasets/sim2real/main-drive-bench-v1/episodes/leg0-run1 \
  outputs/sim2real/main-0-step-coast-candidate-v1 \
  --scenario suspended-main-0-step-coast \
  --output outputs/sim2real/main-0-step-coast-comparison.json
```

Execute a one-factor sensitivity set first. Each uncached candidate starts in a fresh deterministic Isaac process; interrupted runs resume from independently verified artifacts:

```bash
python -m tools.sim2real sweep profiles/candidate-v1.json \
  --scenario suspended-main-0-step-coast \
  --real-trace datasets/sim2real/main-drive-bench-v1/episodes/leg0-run1 \
  --audit-evidence outputs/sim2real/audit-artifact.json \
  --mode one-factor \
  --space-json '{"simulation_physics.main_drive.damping":[0.8,1.0,1.2]}' \
  --seed 0 \
  --headless \
  --output outputs/sim2real/main-drive-damping-sweep
```

The command uses `$ISAACLAB_ROOT/isaaclab.sh`; alternatively pass `--isaaclab-root`. Executed sweeps require a verified real episode and a hash-bound audit artifact whose every derived check passes, and persist separate real, simulator, and delta metrics for every candidate. Add `--generate-only` to create immutable candidate/scenario/provenance snapshots without launching Isaac; generation does not claim that fitting is safe. Use a bounded two-parameter coarse grid only after sensitivity work shows correlation. Cache keys bind the audit artifact and derived report hashes, real trace, scenario, profile, seed, mode, device, Git revision, production asset/config, and runner hashes. Do not combine subsystem errors into a global RMSE, and do not introduce an optimizer until repeatability and identifiability are demonstrated.

## Build a profile from managed direct measurements

Mass/CoM, spring stiffness, and main-drive effort saturation use the same immutable managed-dataset path as the actuator traces. They are applied as measured quantities rather than included in a simulator parameter sweep:

- `mass-com` requires three repeats and at least three non-collinear planar support positions. It writes an absolute `target_total_mass_kg` and `reference_planar_com_xy_m`, bound to the reviewed neutral reference pose. At runtime Isaac applies and verifies the achieved whole-robot mass and planar CoM; the audit fails if the target, achieved value, or reference pose does not agree.
- `spring` computes the damper's torsional stiffness from measured force, lever arm, and deflection. It changes only `passive_spring.<joint>.stiffness`; spring damping and rest angle remain separate, currently unidentifiable parameters.
- `manual-load` computes main-drive effort saturation from short, manually supervised, hardware-current-limited measurements. Every sample must set numeric `saturation_confirmed=1`, and every repeat must contain both positive and negative directions. This is not permission to automate a stalled-motor sweep. The result updates `main_drive.effort_limit`; any later effort-limit sweep must supply the managed known-load trace, and every candidate must remain within its measured repeat envelope.

Use a distinct calibration and holdout episode for each direct measurement. The holdout should change a support geometry, spring load range, or known-load condition that was excluded from fitting.

ABAD and friction are direct managed measurements too. `abad-static` identifies only the static measured relation

`actual_angle = target_scale * requested_angle + target_offset_rad`; it does not claim a dynamic gain. Record at least three distinct settled poses in each of the three repeats. The `repeat_index` and `settled` numeric annotations use the position clock; only samples with `settled=1` enter the fit. Results include the aggregate fit plus per-repeat scale, offset, residual, mean, standard deviation, and count.

`friction` is a manual scenario and cannot run in Isaac. Enter exactly one `breakaway_force` and positive `static_normal_load` for every `static_repeat_index`. Dynamic samples use their own `dynamic_time_s` clock and include `dynamic_pull_force`, `dynamic_normal_load`, `dynamic_speed`, and `dynamic_repeat_index`. Each repeat must contain at least two nonzero, slow, constant-speed samples. The report keeps static and dynamic coefficients separate and includes dimensionless units, the foot/ground frame, and repeat variation.

Apply verified measurement results without manually transcribing field names:

```python
from pathlib import Path
import json

from tools.sim2real.contracts import load_profile
from tools.sim2real.profile_measurements import apply_measurements_to_profile

candidate = apply_measurements_to_profile(
    load_profile("profiles/candidate-v1.json"),
    profile_id="candidate-v2",
    trace_paths=[
        "datasets/sim2real/rigid-body/episodes/mass-com-calibration",
        "datasets/sim2real/springs/episodes/damper-0-calibration",
        "datasets/sim2real/main-drive-load/episodes/main-0-calibration",
        "datasets/sim2real/abad/episodes/abad-0",
        "datasets/sim2real/contact/episodes/friction",
    ],
)
Path("profiles/candidate-v2.json").write_text(
    json.dumps(candidate.to_dict(), indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
```

The helper accepts only hash-verified real episodes linked by a managed dataset, enforces each reviewed scenario's units, frames, and repeat count, and recomputes the metrics itself. It preserves unrelated profile values and records dataset/episode identity plus actual trace, metadata, and scenario hashes in `measurement_sources`; caller-supplied metrics and hashes are not accepted. It maps mass/CoM to the absolute mass target, spring slope to passive stiffness, known-load torque to the main-drive effort limit, ABAD results to `hardware_mapping.abad_target_scale` and `abad_target_offset_rad`, and friction to `simulation_physics.ground`.

Characterization, training, and playback apply the ABAD relation and then clamp the final target to the configured physical joint range. Measured foot/ground friction uses explicit max-combine materials on both sides, with runtime robot collision coefficients overwritten to the measured pair values, so the effective coefficient is the measurement rather than its square. With no measured ABAD fields, scale `1` and offset `0` preserve existing behavior.

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
- every executable simulator holdout metric, or direct profile value for a manual holdout, lies within `max(instrument uncertainty, 2 × real-run standard deviation)` of the held-out real mean;
- actuator, timing, rigid-body, spring, and contact results are reported independently;
- the audit has no unresolved unit, frame, joint order/axis/range, encoder
  scale/zero/sign, mass/CoM/inertia, collision, IMU, or contact failure;
- no unrelated parameter was used to conceal a subsystem-model mismatch;
- a reviewed configuration change explicitly promotes the profile.

If the implicit actuator cannot enter the held-out response envelope, stop tuning mass or friction and open a separate DC/PWM or learned-actuator follow-up.

Encode those checks in a version-1 evidence JSON and evaluate it against the exact candidate hash:

```bash
python -m tools.sim2real validate-promotion \
  profiles/candidate-v1.json \
  outputs/sim2real/candidate-v1-validation-evidence.json \
  --output outputs/sim2real/candidate-v1-promotion-report.json
```

The evidence binds each real episode and each applicable simulator trace by SHA-256, declares calibration versus holdout conditions and what was held out, records instrument uncertainty, and reports whether bounded actuator candidates entered the real envelope. Executable main-drive step/coast, ABAD, and static-contact holdouts require simulator artifacts. Manual mass/CoM, passive-spring, and known-load holdouts must not claim an Isaac artifact; the gate instead compares the exact candidate profile target with the managed held-out measurements. Their mandatory metrics are total mass plus planar CoM, spring stiffness, and positive plus negative saturation torque, respectively.

Executable sweeps are accepted only after the hash-bound pre-fit audit passes every derived geometry, mapping, mass/CoM, IMU, collision, and contact check. This prevents a candidate sweep from becoming evidence for a profile built on a known-bad asset or frame convention. The promotion command returns nonzero when any audit, repetition, holdout, metric, measurement-source, identifiability, or actuator-model check fails. A passing report says only `eligible_for_review`; it never edits training defaults or promotes a profile automatically.
