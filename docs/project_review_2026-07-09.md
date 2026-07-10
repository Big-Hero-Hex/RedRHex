# RedRHex Full Project Review — state file (2026-07-09)

Task: find all issues (bugs, efficiency, sim setup errors, structure, upgradability) across the repo.
Process: review area by area; after each area, append findings below and tick the checkbox.
If context was compacted: continue from the first unchecked area. Final deliverable = summary of ALL findings in chat, one by one, grouped by area/severity.

## Checklist
- [ ] 1. Core RL env: source/RedRhex/RedRhex/tasks/direct/redrhex/ (redrhex_env.py, redrhex_env_cfg.py, symmetry, agents cfg)
- [x] 2. Training scripts: DONE
- [ ] 3. Panel backend: tools/training_panel/training_panel/ (server.py, processes.py, deploy.py, remote_worker.py, history.py, convergence.py, mujoco_rollout.py; skim rest)
- [ ] 4. Panel frontend: tools/training_panel/static/ + remote_web/ (skim: structure, obvious bugs)
- [ ] 5. ROS2 deploy: ros2_ws/src/ (safety_filter, observation_builder, action_decoder, policy_onnx_runner, rl_controller_node, serial/udp bridges)
- [ ] 6. Reward agent tool: tools/reward_agent/
- [ ] 7. Repo hygiene/structure: root stray files, .gitignore, duplication (training_panel/*.py top-level vs package), docs

## Findings (append per area)

### Area 1.5: Symmetry + PPO cfg — DONE
20. [RL][MED] Left-right symmetry augmentation (redrhex_symmetry.py, enabled in ALL PPO cfgs via use_data_augmentation=True) is physically inconsistent: tripod grouping A={0,3,5}/B={1,2,4} is NOT mirror-symmetric (mirror swaps 0↔3,1↔4,2↔5 → A maps to {0,2,3} ≠ A or B). gait_phase obs (cols 42:44) is left untransformed, but legs 2/5 swap tripods → mirrored samples pair leg obs with wrong CPG phase offsets. Augmented data teaches wrong phase relationships for legs 2 & 5. Fix: mirror + shift gait_phase by π only works if tripod assignment were symmetric; either make tripods mirror-symmetric or disable augmentation.
21. [STRUCT][LOW] 4 PPO cfg classes are ~95% copy-paste; only dims/obs_groups/names differ.

### Area 2: Training scripts — DONE
22. [RISK][HIGH] train.py silently applies `tools/training_panel/active_reward_override.json` + `active_terrain_override.json` if present. `active_terrain_override.json` EXISTS now (generator terrain, stairs 40%...) → every `train.py` run silently trains on rough-terrain overrides even when launched manually without the panel. No CLI flag to disable; only a print notice. Stale-override footgun.
23. [STRUCT][MED] `_load_runner_checkpoint_with_policy_fallback`, `_resolve_rsl_rl_checkpoint_path`, `_infer_stage_from_checkpoint_path` duplicated across train.py/play.py/eval_command_sweep.py (3 copies, already drifting: train version has load_optimizer param, play doesn't).
24. [NOTE][LOW] play.py keyboard 's' = backward (vx<0) but training never samples vx<0 (docs: RHex can't walk backward) → out-of-distribution command in play; harmless but confusing to users.
25. [NOTE][LOW] play.py auto-infers stage from checkpoint path name (`*stage4*`) — clever but brittle; a run dir named e.g. "stage4_test" of a stage-5 model silently changes gating behavior. Has disable flag, OK.

### Area 3: Panel backend — DONE (server.py full; processes.py core; history/convergence/deploy outline)
26. [BUG][HIGH] HistoryStore (history.py) has NO locking and non-atomic writes (`history_file.write_text(json.dumps(...))`). Server is ThreadingHTTPServer + monitor threads + queue timers all do load→modify→save on the same JSON → lost updates under concurrency; crash mid-write corrupts entire run history. Needs a lock + tmp-file/os.replace atomic write.
27. [DESIGN][MED] Global override files (`active_reward_override.json`, `active_terrain_override.json`) as IPC between panel and train.py: `_start_training_run` rewrites them just before spawn; active-preset activation writes the same terrain file; a run with no terrain_overrides DELETES the active preset's file. Races between queued runs/preset activation/manual runs; ties into finding #22.
28. [SEC][LOW] Panel has zero auth; docs suggest `--host 0.0.0.0` for LAN — anyone on LAN can start GPU jobs, delete runs, execute xdg-open. Fine for trusted lab, worth a note in docs/simple token.
29. [NOTE][LOW] convergence.py: EventAccumulator `size_guidance={"scalars": 2000}` downsamples across the whole run, so `window_iterations=200` is really "last 200 of ≤2000 reservoir samples", not last 200 iterations; plateau metric is max-min range → one noise spike defeats detection.
30. [EFF][LOW] `/api/runs` GET runs `reconcile_stale_history()` (scans processes/logs) on every poll.

### Area 5: ROS2 deploy stack — DONE (contract, obs builder, action decoder core, controller node rate, deploy validate_contract)
31. [BUG][CRITICAL] Control-rate mismatch: redrhex_contract.py `SIM_DT = 1/250` → CONTROL_DT=1/125 → POLICY_HZ=125; config yaml comment confirms "repo-derived 125 Hz". Training env: sim.dt=1/120, decimation=2 → 60 Hz. Deployed policy runs at ~2× the trained rate (last_actions dynamics, warmup steps, slew rates, action-rate all wrong). deploy.py validate_contract does NOT check this.
32. [BUG][HIGH?verify] Frame convention: sim root init rot = 90° about X (USD Y-up) → policy's projected_gravity at rest ≈ (0,-1,0), base_ang_vel/base_lin_vel in that rotated body frame. ROS observation_builder feeds raw IMU quat → projected gravity at rest = (0,0,-1). Unless IMU is deliberately mounted/configured in the rotated USD frame, deployed obs are 90°-rotated vs training → policy sees "fallen" robot. Needs a hardware-frame cross-check test; deploy readiness can't catch it (synthetic golden obs).
33. [BUG][MED] Contract constant drift (hand-mirrored, no automated parity test): ABAD_POS_SCALE=0.61096 vs cfg 0.60; STAGE_ABAD_POS_LIMIT=0.62 vs cfg math.radians(60)=1.047 (deployed ABAD range ~60% tighter than sim); stage-5 constants must be re-checked whenever env_cfg changes. validate_contract only checks dims/joint names/slices.
34. [RISK][MED] base_lin_vel_source default "zero": policy trained WITH true base_lin_vel obs gets zeros on hardware (deploy.py emits a warning only). Distribution shift; consider training with lin-vel obs dropout or an estimator.
35. [GOOD] action_decoder replicates env gating with per-robot sign/offset config and slew limits; safety_filter + preflight + state machine exist; deploy readiness pipeline is a strong idea — main gap is it doesn't validate the drifting constants (#31/#33).

### Area 1: Core RL env — DONE (env.py + env_cfg.py read fully)

**BUGS (high confidence):**
1. [BUG][HIGH] `_apply_action` runs once per physics substep (decimation=2 → 2×/control step in DirectRLEnv), but it accumulates time-based state with `dt_sim = sim.dt * decimation`: `_lateral_state_time += dt_sim`, `_lateral_gait_phase += 2π·f·dt_sim`, `_lateral_timeout_cooldown -= 1`. All lateral FSM timers/CPG run at 2× intended speed. (redrhex_env.py:2176-2196, 2255-2257). Also whole gating logic recomputed twice per step = wasted CPU.
2. [BUG][HIGH] Obs-noise index mismatch in `_apply_observation_domain_randomization` (env.py:1178-1189): layout is sin(9:15),cos(15:21),main_vel(21:27),abad_pos(27:33),abad_vel(33:39) but code treats 9:27 as "pos" and 27:39 as "vel" → main_drive_vel gets 0.01 pos-noise; abad_pos gets 1.5 rad vel-noise (huge). Latent (DR noise off by default) but breaks robustness runs.
3. [BUG][MED] Full-reward mode: `rew_lateral_direction` computed AND added to total twice (G6.5.6 duplicated block, env.py:3654-3661 and 3674-3688). Legacy path only (simplified is default).
4. [BUG][MED] `_setup_buffers` env.py:619: `self.cfg.robot.init_state.pos[2]` — attribute is `robot_cfg`, so always throws → silently falls back to target_base_height (0.10) instead of intended init height.
5. [BUG][LOW] Simplified rewards: diag_sign bonus/penalty double-counted — included inside `diagonal_reward` (scaled by mode_specialization=2.5×diag_mult) AND added again as separate `rew_diag_sign` (env.py:2579-2599).
6. [BUG][LOW] `_get_dones` hardcodes `too_high = base_height > 2.0`; cfg.max_base_height=0.8 unused.
7. [BUG][LOW] `external_control=True` freezes `_global_step_count` (early return in `_update_commands` before increment) → curriculum auto-progress + push interval logic frozen in play/eval. Probably OK but undocumented coupling.

**SIM SETUP:**
8. [SIM][HIGH] ContactSensor configured in cfg (`contact_sensor`, `body_names`, `leg_names`) but never instantiated — env comment: USD lacks contact-reporter API. ALL contact-based rewards/termination use joint-phase proxies ("in stance window" ≠ actually touching ground) and height/tilt proxies for body contact. Big sim-fidelity gap for gait rewards + deploy readiness.
9. [SIM][MED] Mass set via `density=2500` on every body in the USD (mass_props applies globally), with comments claiming UPE ≈940 kg/m³; actual masses depend on USD volumes — fragile & unverifiable; explicit per-link masses would be robust.
10. [SIM][MED] main_drive ImplicitActuator damping=1.0 with effort_limit 15: torque = 1·(vel error) → needs 15 rad/s error for max torque = very soft velocity tracking; cfg comment describes damping=50 example ("力矩 = 50 × ..."), misleading.
11. [SIM][LOW] Control-layer DR proxies scale *velocity targets* by mass/friction (`_compute_main_drive_targets`) — crude physics proxy, worth documenting limits.
12. [SIM][LOW] `linear_damping=0.05`/`angular_damping=0.10` on all rigid bodies acts as fake drag on base too — affects sim-to-real.

**STRUCTURE / EFFICIENCY:**
13. [EFF][MED] `episode_sums` = ~140 tensors updated one-by-one every step (140 kernel launches) + 140 means at every reset; deprecated reward terms still computed then zeroed (rew_abad_smooth, sync_jitter, lateral_low_freq, etc.). Whole full-mode path (~1000 lines) is dead by default (use_simplified_rewards=True).
14. [EFF][LOW] Per-step tensor allocations: gravity_vec in `_update_state`, `torch.tensor(directions)` per resample, many `torch.zeros` temporaries; getattr(self.cfg,...)+float() everywhere in hot path.
15. [STRUCT][MED] State mutation inside `_get_observations` (`_update_state` advances gait_phase, resamples commands, applies pushes) — fragile if obs computed more than once; reward computed with 1-step-stale kinematics (rewards read joint state updated at *previous* step's `_get_observations`).
16. [STRUCT][MED] 4459-line env file mixing: legacy 50-reward path, simplified path, LAT FSM, DR, visualization. Cfg has ~250 scalars incl. deprecated + alias fields (`randomize_mass = dr_randomize_mass` etc. — class-level copies that silently desync if Hydra overrides dr_*). Stage lists (length-5) have no validation.
17. [STRUCT][LOW] Stale comments: init pos comment says 0.15m but pos z=0.3; episode_length comment says 30s but =60; USD path comment refers to /home/jasonliao.
18. [STRUCT][LOW] obs `abad_vel` unnormalized while other channels normalized; add_noise path also noises commands.
19. [LOG][LOW] Episode reward logging divides by max_episode_length_s regardless of actual episode length → early-terminating episodes under-report per-second reward.

---

## Fix status (2026-07-10, branch fix/review-2026-07)

FIXED (one commit each; see git log on this branch):
- #4/#22/#27 train.py override gating: `--panel_overrides` flag now required; panel passes it automatically.
- #3 (#1 in env list) `_apply_action` per-substep time accumulation: targets computed once per control step.
- #6 (#2 in env list) DR obs-noise slice indices corrected.
- #13 (#4) `cfg.robot` -> `cfg.robot_cfg` typo.
- #17 (#6/#7) `too_high` uses cfg.max_base_height; `_global_step_count` no longer freezes under external_control.
- #11 (#3) legacy duplicated lateral-direction reward removed.
- #5 (#20) mirror data augmentation disabled in all 4 PPO cfgs (tripod grouping not mirror-symmetric).
- #1/#7 (#31/#33) deploy contract: SIM_DT 1/250 -> 1/120 (60 Hz), ABAD_POS_SCALE 0.60, STAGE_ABAD_POS_LIMIT radians(60); all 125 Hz fallbacks/docs updated; NEW tools/training_panel/tests/test_contract_parity.py (AST-based) prevents future drift.
- #2 (#32) IMU frame: observation.imu_mount_rpy_deg + expected_rest_projected_gravity rest-attitude gate on /redrhex/enable_policy; still requires recording sim obs[6:9] at rest and hardware verification.
- #8 (#26) HistoryStore: RLock + atomic tmp+os.replace writes.
- Hygiene: MUJOCO_LOG.TXT gitignored; stale comments fixed; world-gravity tensor cached.

DEFERRED (need owner decision / hardware / larger refactor):
- #9 contact sensors (USD needs contact-reporter API authoring).
- #10 base_lin_vel=zero at deploy (needs estimator or lin-vel-dropout training).
- #12 diag sign double-count in simplified rewards (ambiguous intent; changes reward magnitudes).
- #14 reward-on-stale-state / state mutation in _get_observations (refactor, subtle training change).
- #15 density-based mass, #16 actuator damping softness, body damping (physics tuning; retrain to validate).
- #18 convergence reservoir/window semantics.
- #19 episode_sums perf refactor; legacy reward path removal; cfg modularization (#22-24).
- #25 stale .patch/test files at repo root; #26 panel auth.
