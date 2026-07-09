# Training Panel Porting Guide for AI Agents

This guide explains how to port the RedRHex Training Panel to another Isaac Lab / RSL-RL project without breaking the connection between the panel, `train.py`, `play.py`, history, videos, TensorBoard, and the child web UI.

The panel is not only a static web app. It is a coordinator around three contracts:

1. The panel builds commands for `scripts/rsl_rl/train.py` and `scripts/rsl_rl/play.py`.
2. The scripts write logs, checkpoints, exported policies, and videos in predictable places.
3. The panel watches process logs and run folders to update history and remote sync state.

If you move the panel to another repo, port these contracts first.

## Files to Port

Copy or adapt these panel files:

- `tools/training_panel/training_panel/`
- `tools/training_panel/static/`
- `tools/training_panel/remote_web/`
- `tools/training_panel/supabase/schema.sql`
- `tools/training_panel/reward_presets.json`
- `tools/training_panel/terrain_presets.json`
- `tools/training_panel/docs/`

Also adapt the target project's RSL-RL scripts:

- `scripts/rsl_rl/train.py`
- `scripts/rsl_rl/play.py`

The panel can live under `tools/training_panel`, but `train.py` and `play.py` must understand the panel-specific arguments and file layout described below.

## Panel Configuration Contract

Update these values for the target project:

- `tools/training_panel/training_panel/commands.py`
  - `DEFAULT_TASK`
  - video defaults if needed
- `tools/training_panel/training_panel/config.py`
  - `PanelPaths.rsl_rl_log_root`
  - default `ISAACLAB_ROOT`
  - default `ISAACSIM_ROOT`
  - default `REDRHEX_CONDA_ENV`
- Project launch environment:
  - `PROJECT_ROOT` equivalent, currently `REDRHEX_ROOT`
  - `ISAACLAB_ROOT`
  - `ISAACSIM_ROOT`
  - `CONDA_SH`
  - project conda env name

The command wrapper uses:

```bash
<IsaacLab>/isaaclab.sh -p scripts/rsl_rl/train.py ...
<IsaacLab>/isaaclab.sh -p scripts/rsl_rl/play.py ...
```

If the target repo renames these scripts, update `training_argv()` and `play_argv()`.

## `train.py` Required Contract

The panel's training command currently passes:

```bash
scripts/rsl_rl/train.py \
  --task <task> \
  --num_envs <n> \
  --max_iterations <n> \
  --device <cpu-or-cuda> \
  [--headless] \
  [--seed <seed>] \
  [--resume --checkpoint <checkpoint>]
```

The target `train.py` must support those arguments through Isaac Lab / RSL-RL CLI parsing.

### Required Log Lines

Print both lines during startup:

```python
print(f"[INFO] Logging experiment in directory: {log_root_path}")
print(f"Exact experiment name requested from command line: {log_dir_name}")
```

Do not change the second string. The panel reads process logs using this exact prefix to map a panel run ID to the real RSL-RL log folder.

### Required Log Layout

Use this layout:

```text
logs/
  rsl_rl/
    <experiment_name>/
      <timestamp>[_run_name]/
        model_*.pt
        events.out.tfevents.*
        params/
          env.yaml
          agent.yaml
        videos/
          train/
```

The panel expects:

- checkpoints as top-level `model_*.pt`
- TensorBoard event files inside the run folder
- environment and agent YAML under `params/`
- optional train videos under `videos/train/`

Set:

```python
env_cfg.log_dir = log_dir
```

before constructing the environment so the task can write run-local diagnostics.

### Panel Overrides

Training presets are passed through small JSON files written by the panel just before launch.

Reward override file:

```text
tools/training_panel/active_reward_override.json
```

Minimal behavior:

```python
if override_file.exists():
    overrides = json.loads(override_file.read_text(encoding="utf-8"))
    for key, value in overrides.items():
        if hasattr(env_cfg, key):
            setattr(env_cfg, key, float(value))
```

Terrain override file:

```text
tools/training_panel/active_terrain_override.json
```

Minimal behavior:

```python
from tools.training_panel.training_panel.terrain import apply_terrain_overrides

if terrain_override_file.exists():
    overrides = json.loads(terrain_override_file.read_text(encoding="utf-8"))
    apply_terrain_overrides(env_cfg, overrides)
```

If the target project has no reward or terrain preset UI, keep the hooks harmless: ignore missing files and ignore unknown keys.

### Resume Behavior

Support absolute checkpoint paths:

```python
if agent_cfg.load_checkpoint and (
    os.path.isabs(agent_cfg.load_checkpoint) or os.path.exists(agent_cfg.load_checkpoint)
):
    resume_path = retrieve_file_path(agent_cfg.load_checkpoint)
else:
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
```

The panel passes absolute checkpoint paths when resuming from a selected history run.

### Git State Safety

If the target repo has unusual filenames or large diffs, make git-state logging opt-in. This avoids crashes during training startup:

```python
if args_cli.store_code_state:
    runner.add_git_repo_to_log(__file__)
else:
    runner.git_status_repos = []
```

## `play.py` Required Contract

The panel uses `play.py` for three jobs:

- live play from a checkpoint
- automatic video recording
- policy export to JIT / ONNX

The panel currently passes:

```bash
scripts/rsl_rl/play.py \
  --task <task> \
  --num_envs 1 \
  --device <device> \
  [--headless] \
  [--video --video_length <steps> --video_width <px> --video_height <px> --video_fps <fps> --rendering_mode <mode>] \
  [--terrain_override_file <json>] \
  [--camera_follow_robot --camera_eye X Y Z --camera_lookat X Y Z] \
  [--export_policy_only] \
  --checkpoint <checkpoint>
```

The target `play.py` must parse every argument above or the panel's Play, Video, or Export buttons can fail. If the target script does not use `--rendering_mode`, still accept it and ignore it.

### Video Recording

When `--video` is set:

1. Enable cameras before creating `AppLauncher`.
2. Apply width and height to the launcher args if provided.
3. Create the environment with `render_mode="rgb_array"`.
4. Wrap with `gym.wrappers.RecordVideo`.

Required output folder:

```text
<run_log_dir>/videos/play/
```

Use a trigger that records immediately:

```python
video_kwargs = {
    "video_folder": os.path.join(log_dir, "videos", "play"),
    "step_trigger": lambda step: step == 0,
    "video_length": args_cli.video_length,
    "fps": args_cli.video_fps,
    "disable_logger": True,
}
env = gym.wrappers.RecordVideo(env, **video_kwargs)
```

The child web UI and Supabase artifacts should point only to videos from this run's own `log_dir`.

### Checkpoint Resolution

Support:

- absolute checkpoint file path
- run directory containing `model_*.pt`
- `--load_run <run> --checkpoint model_XXXX.pt`

Always sanitize to a real model checkpoint. Do not allow TensorBoard event files or arbitrary `.pt` files to become the selected checkpoint.

Recommended behavior:

- if `--checkpoint` is a directory, pick the newest `model_*.pt`
- if `--checkpoint` is a file named `model_*.pt`, use it
- otherwise fall back to the latest sibling `model_*.pt`

Then set:

```python
log_dir = os.path.dirname(resume_path)
env_cfg.log_dir = log_dir
```

### Terrain Replay

For faithful playback and video, accept:

```bash
--terrain_override_file <json>
```

and apply it with:

```python
from tools.training_panel.training_panel.terrain import apply_terrain_overrides
```

Ignore missing files gracefully.

### Policy Export

After loading the checkpoint, export both files:

```text
<run_log_dir>/exported/policy.pt
<run_log_dir>/exported/policy.onnx
```

If `--export_policy_only` is set, exit after export.

The panel uses these paths to enable ONNX/export actions in history.

## History and Media Contract

A run is considered useful when the panel can find these inside its `log_dir`:

- checkpoint: `model_*.pt`
- TensorBoard data: `events.out.tfevents.*`
- video: `videos/play/*.mp4` or `videos/train/*.mp4`
- exported policy: `exported/policy.onnx`

Do not attach media by filename alone. Always resolve media through the owning run's `log_dir`; otherwise child history can show another run's video.

## Child Web and Remote Worker Contract

For remote queueing, the child inserts Supabase `jobs` rows with payload fields:

```json
{
  "task": "...",
  "num_envs": 4,
  "max_iterations": 1000,
  "device": "cuda:0",
  "headless": true,
  "display_name": "optional run name",
  "folder": "optional history folder",
  "client_request_id": "browser-generated-id"
}
```

The mother/worker must preserve these into `TrainingParams`, the local history record, and the remote `runs.params`.

The worker should:

1. heartbeat
2. claim queued jobs if jobs are accepted
3. run routine history/media sync only when idle
4. force-sync after starting/completing jobs or generating media

This keeps child queue feedback fast without making history sync expensive.

## Robot-Specific Parts You Can Replace

These RedRHex pieces are not generic panel requirements:

- keyboard command mapping in `play.py`
- RedRHex gait command names
- RedRHex stage inference from checkpoint names
- reward field names such as `rew_scale_*`
- terrain preset contents
- task name `Template-Redrhex-Direct-v0`
- experiment name `redrhex_wheg`

Replace them with target-project equivalents, but keep the command, log, checkpoint, and media contracts.

## Porting Sequence

1. Copy `tools/training_panel`.
2. Update `DEFAULT_TASK` and `PanelPaths.rsl_rl_log_root`.
3. Adapt `train.py` to the required training contract.
4. Adapt `play.py` to the required playback/video/export contract.
5. Register the target Isaac Lab task and agent config.
6. Run the local mother panel and start one disposable short training run.
7. Confirm the history record links to the real RSL-RL log folder.
8. Generate one video from the run and confirm it appears only on that run.
9. If using child web, apply `supabase/schema.sql`, deploy the static child site, and run the remote worker.

## Validation Commands

Run these from the repository root:

```bash
python -m py_compile tools/training_panel/training_panel/*.py scripts/rsl_rl/train.py scripts/rsl_rl/play.py
PYTHONPATH=. pytest -q tools/training_panel/tests
for f in tools/training_panel/remote_web/*.js; do node --check "$f"; done
```

For command-contract checks:

```bash
PYTHONPATH=. pytest -q tools/training_panel/tests/test_commands.py tools/training_panel/tests/test_processes.py
```

For a disposable smoke test:

```bash
<IsaacLab>/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task <target-task> \
  --num_envs 1 \
  --max_iterations 1 \
  --device cuda:0 \
  --headless
```

Then check:

- process log contains `Exact experiment name requested from command line:`
- a run folder was created under `logs/rsl_rl/<experiment_name>/`
- `model_*.pt` exists
- TensorBoard event file exists
- the panel history card shows the correct run

## Common Failure Symptoms

- History says `no training log`: `train.py` did not print the exact experiment-name line, or `PanelPaths.rsl_rl_log_root` points to the wrong experiment folder.
- Run finishes but has no checkpoint: checkpoint naming is not `model_*.pt`, or the panel is linked to the wrong log directory.
- Video button fails immediately: `play.py` does not parse one of the panel arguments, often `--rendering_mode`.
- Child shows another run's video: artifact/media sync is matching by filename or latest video globally instead of by the selected run's `log_dir`.
- Child queue stays gray too long: worker is syncing history before claiming queued jobs, or Realtime/polling is not refreshing jobs quickly.
- TensorBoard works only on mother: the child needs exported static snapshots or uploaded artifacts; local TensorBoard URLs are not portable across devices.

