---
id: explicit-torsion-spring-instability-2026-08-14
title: Explicit Torsion-Spring Numerical Instability
lang: en
audience: developer
type: experiment-summary
status: published
owner: sim2real
last_reviewed: 2026-08-14
---

<a id="question"></a>
## Question

Why does the robot shake and repeatedly reset with the `explicit` torsion-spring backend, and what operation is justified before physical calibration?

<a id="baseline"></a>
## Baseline

The deterministic baseline is fixed-base, zero-gravity `spring-release` with no policy or action input, seed `0`, physics at 120 Hz, and the uncalibrated `200 N*m/rad`, zero-damping spring. It used source commit `e86da0055d0b7db6da6af0017e33c8882f4b1413`, runtime bundle `5010e8362e522ebb45d2131338d62253cf9f6acbc76242376548ebed1b33a707`, and trace `505e62f936200542e9deda1f177e73a2dbe6e1161d8d4c3061915d3a246eb193`.

<a id="method"></a>
## Method

The run isolated `Revolute_5`, locked the other joints, and released the spring from `+0.1 rad`. The review checked the requested torque, applied-effort path, PhysX and actuator gains, the first four physics samples, runtime mass properties, and the semi-implicit stability condition. Two diagnostic-only variants reduced all six stiffness values to `20 N*m/rad` or uniformly scaled total robot mass from `1.7985 kg` to `14 kg`; both were evaluated with the same formal release gates.

<a id="results"></a>
## Results

The restoring law and sign are correct: `+0.1 rad` requests `-20 N*m`, static torque RMSE is zero, and the effort is written once per physics substep with explicit PhysX gains at zero. The first four deflections at `0`, `8.33`, `16.67`, and `25 ms` are approximately `+0.100`, `-2.357`, `+23.417`, and `-233.263 rad`; spring energy grows from `1.0 J` to about `5.44 MJ`.

The first step implies effective joint inertia of about `0.00056538 kg*m^2`. With `k=200 N*m/rad`, the undamped natural frequency is about `594.8 rad/s`, so `dt*omega` is about `4.956` at 120 Hz and `2.478` at 240 Hz. Both exceed the semi-implicit stability boundary of `2`. The runtime articulation mass is `1.7985 kg`, not the approximately `14 kg` value assumed by configuration diagnostics.

The baseline maximum amplitude ratio is about `2391.54`, with energy/work residual fraction about `5.872e15`. Lowering stiffness to `20 N*m/rad` reduces the ratio to about `1.4566` but leaves residual fraction about `15.68`. Uniform 14 kg mass scaling gives ratio about `2.3602` and residual fraction about `67.34`. All three fail runaway, energy-creation, and energy/work gates; neither diagnostic variant is a validated fix.

<a id="decision-impact"></a>
## Decision impact

`explicit` policy training is quarantined. New environment and Training Panel policy runs use `native` as a provisional operational default, while stamped checkpoint backend identity remains immutable. `explicit` remains available for deterministic spring characterization and investigation. This does not select `native` for production: physical calibration, matched backend characterization, and policy acceptance are still required.

<a id="limitations"></a>
## Limitations

No physical stiffness, damping, per-link mass, inertia, or release-response measurement exists. The diagnostic mass scale is not a per-link calibration, and the reduced stiffness is not a measured spring. Native remained finite in the earlier V11 experiment but failed strict energy/work and timestep-agreement gates. Raw traces are local ignored artifacts rather than committed evidence packages.

<a id="artifacts-and-addenda"></a>
## Artifacts and addenda

The baseline runtime artifact is `outputs/sim2real/explicit-shake-baseline-120-seed0`; diagnostic variants are `explicit-shake-k20-120-seed0` and `explicit-shake-mass14-120-seed0`. The preceding evidence record is the [torsion-spring V11 provisional checkpoint](2026-08-01-torsion-spring-v11-checkpoint.en.md).
