---
id: runtime-layout
title: Runtime Paths and Artifact Layout
lang: en
audience: shared
type: reference
status: active
owner: project
last_reviewed: 2026-08-14
---

<a id="configuration"></a>
## Runtime configuration

The Training Panel reads `REDRHEX_ROOT`, `ISAACLAB_ROOT`, `ISAACSIM_ROOT`, `CONDA_SH`, and `REDRHEX_CONDA_ENV`. Defaults are machine-specific conveniences and must not be treated as portable project requirements. Scripts should receive explicit paths or run from the repository root. The local campaign service is separately enabled by `REDRHEX_AUTOPILOT_ENABLED`; its optional panel bearer token and MCP/tunnel credentials are process environment secrets and must not enter repository files or artifacts.

<a id="training-artifacts"></a>
## Training artifacts

```text
logs/rsl_rl/<experiment>/<timestamp>_<run-name>/
├── model_*.pt
├── events.out.tfevents.*
├── params/
├── exported/policy.pt
├── exported/policy.onnx
├── videos/play/
└── deploy/
```

Only `model_*.pt` is a training runner checkpoint. Exported models are deployment artifacts.

<a id="panel-state"></a>
## Panel state

`logs/training_panel/` contains process logs, per-run override snapshots, notes, history, activity, remote state, convergence configuration, and the cross-process `gpu_process.lock`. Autopilot adds `autopilot.sqlite3`, SHA-256-addressed files below `autopilot_artifacts/`, and command/episode evidence below `evaluations/`. These are runtime records, not tracked documentation. Active override files under `tools/training_panel/` are transient IPC. Manual `train.py` ignores them unless `--panel_overrides` is present.

<a id="evidence"></a>
## Calibration and experiment evidence

Raw run logs, full traces, videos, campaign databases/artifacts, and local calibration artifacts remain ignored. Commit only small canonical configuration, manifests, reviewed summaries, or explicitly approved fixtures. Reward Agent sessions live under `logs/reward_agent/` relative to the selected repository root; Autopilot imports them as non-armable references without deleting them.

<a id="repository-rule"></a>
## Repository rule

Generated HTML, staged site files, caches, runtime logs, raw experiment artifacts, secrets, and worktree metadata must not be tracked. Git stores durable source and reviewed summaries, not operational state.
