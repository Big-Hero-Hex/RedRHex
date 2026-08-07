# ForwardFast Panel Recording Command Design

## Goal

Make Training Panel videos for `Template-Redrhex-ForwardFast-Direct-v0` record an in-distribution forward gait instead of holding the robot at the playback default of zero commanded velocity.

## Scope

This is a panel-only recording change. ForwardFast video recording will pass `--initial_command forward` to `scripts/rsl_rl/play.py`. Interactive Play, policy export, training, and recordings for the full Direct task keep their existing behavior. Spring physics, checkpoints, rewards, terrain replay, camera settings, action/observation contracts, and calibration status do not change.

## Alternatives Considered

1. **Pass a deterministic forward command only for ForwardFast recordings (selected).** This reproduces the direct preview, stays inside the task's `vx=0.22-0.42 m/s` training range, and changes only the broken path.
2. Disable keyboard control for every recording. This would restore environment command sampling, but recordings would be nondeterministic and full Direct videos could sample unrelated skills.
3. Force forward for every panel playback and recording. This would silently change interactive controls and would misrepresent multi-skill Direct checkpoints.

## Data Flow

`ProcessRegistry.start_video_recording()` already resolves the originating task before building the `play.py` arguments. It will request `initial_command="forward"` only when that exact task is ForwardFast. `play_argv()` will append `--initial_command forward` when the optional argument is supplied and omit the flag otherwise, preserving every existing caller.

## Verification

Tests must prove that a ForwardFast video command contains `--initial_command forward`, a Direct video command does not, and interactive ForwardFast Play does not acquire the recording-only flag. After the panel suite passes, restart the worktree panel on port 8080, record the existing native `model_199.pt`, and confirm its process log reports `vx=0.42, vy=0, wz=0`.
