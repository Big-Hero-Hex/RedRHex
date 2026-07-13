# 04 — AI-Heavy Development Workflow

Goal: the AI agent (Claude Code on this machine) does most implementation, testing,
review, and documentation; the human does research judgment and hardware safety. The
enabling insight: **AI throughput = how fast the agent can verify its own work.** Every
element below either speeds up verification or encodes judgment the agent must not make
alone.

## 1. The context layer (what the agent knows at session start)

### CLAUDE.md hierarchy (templates in `templates/`)
```
/CLAUDE.md                       # root: machine facts, make targets, hard rules,
                                 #   pointers to area files (keep < ~150 lines)
/source/RedRhex/CLAUDE.md        # env architecture, obs layout, extraction status,
                                 #   "which module owns what", parity-test workflow
/tools/training_panel/CLAUDE.md  # panel run/test commands, frozen-during-migration note
/ros2_ws/CLAUDE.md               # generated-contract warning, build/test commands,
                                 #   hardware = human-gated
```
Rules for these files: facts and commands, not prose; update in the same commit that
makes them stale; never duplicate what code/tests already express.

### Machine facts the root CLAUDE.md must pin (from docs/COMMANDS.md)
- Repo `/home/lab_user1/Py/RedRHex`; Isaac Lab `/home/lab_user1/isaac_lab_ws/IsaacLab`
  (2.3.2); Isaac Sim 5.1.0-rc.19; conda env `env_isaaclab_bin` (NOT `base`/`env_isaaclab`);
  GPU RTX 5080 16 GB (one GPU — training and smoke tests contend, see §4).
- Task ids; `make` targets as the only sanctioned entry points; `export TERM=xterm`.

### Persistent memory
Claude Code's project memory already tracks review state. Keep using it for:
durable user preferences, cross-session project facts not derivable from the repo, and
pointers into `docs/` — never for things the repo itself records.

## 2. The verification layer (what lets the agent self-check)

The Makefile is the agent's API to the project. Canonical ladder, cheapest first:

| Target | Time | Needs | Agent uses it for |
|---|---|---|---|
| `make lint` | s | CPU | every edit |
| `make test-fast` | < 30 s | CPU | every core/ or cfg change |
| `make test-contract` | s | CPU | anything touching contract/layout/deploy |
| `make test-golden` | < 1 min | CPU (saved tensors) | every extraction step |
| `make smoke` | ~3–5 min | GPU + Isaac | env construction / integration changes |
| `make preflight` | ~10–15 min | GPU | before merge to main; before ending an unattended session |
| `make train-ref` | hours | GPU | behavior-changing merges (human decides when) |

Design consequences already baked into 02/03: pure-torch core modules exist *so that*
the first four tiers need no simulator. This is the single highest-leverage decision in
the whole reboot for AI velocity.

## 3. The standard loops

### Loop A — structural change (migration steps, refactors)
```
1. PLAN MODE: agent explores, writes step plan w/ parity strategy → human approves
2. Implement (one extraction step; one issue per commit)
3. make lint test-fast test-golden; make smoke if env touched
4. /code-review on the diff; agent fixes findings
5. Commit + short note in the phase checklist (03) — the review-file-as-state pattern
   from July, which survives context loss and session restarts
```

### Loop B — experiment (reward/physics/curriculum change)
```
1. Hypothesis in one sentence → experiments/LOG.md (see 06)
2. New overlay in cfg/experiments/ (never edit base)
3. make smoke → launch via panel or make train EXP=<name> SEED=…
4. While training runs, agent does Loop-A work (GPU-idle multiplexing, §4)
5. Agent pulls TB scalars, writes experiments/reports/<run>.md from template:
   curves vs baseline, verdict, next action
6. Human reads report, makes the shaping/physics judgment call
```

### Loop C — unattended session (overnight / long-running)
```
1. Human leaves a scoped task list (e.g. "extraction steps 2.3 + 2.4")
2. Agent works Loop A per item; stops at any parity failure it cannot explain
   (NEVER "fixes" a parity test by loosening tolerance — hard rule)
3. Ends with: make preflight + a session report (what/why/verified/open questions)
4. Human reviews the report + diffs next morning; nothing merged unreviewed
```

## 4. One-GPU discipline

Training runs and Isaac smoke tests fight over the RTX 5080 (16 GB).
- The panel's queue is the arbiter for training runs; the agent checks for a live run
  (panel API / `nvidia-smi`) before `make smoke`/`preflight` and otherwise runs
  CPU tiers only, queueing GPU verification.
- Reference runs (3-seed) go overnight; agent schedules them last in a session.
- `make smoke` uses few envs (e.g. 128) + few iters (5) to fit beside nothing — it
  must never run beside a big training job; fail fast with a clear message instead.

## 5. Guardrails (what the agent must NOT do alone)

Encode these as hooks/settings where possible, as CLAUDE.md hard rules otherwise:

1. **Generated files**: never edit `ros2_ws/.../redrhex_contract.py` by hand — edit
   `contract.py` + `make contract`. (Hook: block Edit/Write on the generated path.)
2. **Parity tolerances**: never widen `atol`/`rtol` or skip a golden test to make it
   pass. A parity failure is a finding, not an obstacle.
3. **Hardware**: anything that publishes to real actuators (`enable_policy`, bridge
   launch on hardware) is human-present only.
4. **Baselines**: `baselines/` is append-only; replacing a golden dump requires an ADR.
5. **Reward intent**: the agent may implement and *report* on reward changes; it doesn't
   decide shaping direction. (It may propose, with evidence — that's the reward_agent
   tool's whole job.)
6. **Deletions**: dead-code deletion only inside a planned migration step or with the
   review file naming the item.
7. **Long trainings**: > 30 GPU-minutes requires either the human's ask or a queued
   overnight slot the human pre-approved.

## 6. Division of labor (be explicit; it's a research project)

| AI owns | Human owns |
|---|---|
| Extractions, refactors, dedup | Extraction *order* changes, scope cuts |
| Test writing (unit/parity/contract) | Accepting a behavior change (ADR sign-off) |
| Experiment execution + report drafts | Hypothesis choice, verdict on shaping/physics |
| Literature-to-code proposals | Research direction, what goes in the thesis |
| Panel/tooling code | Exposing the panel beyond localhost |
| Analysis scripts, plots, docs | Hardware sessions, safety checklists |

## 7. Claude Code specifics worth setting up (Phase 1.5)

- **Project skills** (`.claude/skills/`): `train` (launch via panel/CLI with overlay),
  `analyze-run` (TB scalars → report draft), `parity` (run + interpret golden suite).
  Skills make Loop B a one-liner for future sessions.
- **Permissions** (`.claude/settings.json`): pre-allow `make *`, `pytest *`,
  `nvidia-smi`, panel API curls — the agent shouldn't stall on prompts for sanctioned
  verification. Run `/fewer-permission-prompts` after a week of real usage.
- **Hooks**: block-edit hook for generated files (§5.1); optional pre-Stop hook that
  reminds "run make lint test-fast before finishing".
- **Subagents**: use Explore for broad code searches during extractions; a second
  opinion review (`/code-review high`) before each phase-gate merge.
- **Plan mode** for every multi-file step: cheap insurance against half-refactors.

## 8. Failure modes to watch (AI-specific, RL-specific)

| Failure mode | Mitigation |
|---|---|
| Agent "fixes" a failing parity test instead of the code | Guardrail §5.2 + human reviews every tolerance diff |
| Plausible-but-wrong physics/reward math (silent RL killer: code runs, learning quietly degrades) | Unit tests with hand-computed values; 3-seed gate runs on behavior changes; never merge math on green-lint alone |
| Context rot in long sessions → forgotten constraints | Review-file-as-state pattern; CLAUDE.md hard rules restated per area; scoped session tasks |
| Reward hacking of the *workflow* (agent optimizes for green checks, e.g. trivial tests) | Human spot-reads tests in review; /code-review explicitly checks test quality |
| Stale CLAUDE.md misleading future sessions | Update-in-same-commit rule (§1); Phase-1.5 exit test rerun occasionally |
| GPU contention breaks a training run mid-experiment | §4 queue discipline; smoke refuses to start beside a training run |
