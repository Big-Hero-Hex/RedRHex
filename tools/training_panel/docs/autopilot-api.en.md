---
id: training-panel-autopilot-api
title: Training Panel Autopilot API Reference
lang: en
audience: developer
type: reference
status: active
owner: panel
last_reviewed: 2026-08-14
---

<a id="boundary"></a>
## Boundary

Autopilot V1 is an off-by-default, local campaign API for standard PPO on `Template-Redrhex-ForwardFast-Direct-v0` stage 1 and `Template-Redrhex-Direct-v0` stages 1–5. The panel service is the sole campaign mutation authority. An external advisor may submit a bounded decision, but deterministic panel code validates identities, allocates trials, launches and evaluates work, enforces budgets, ranks evidence, and declares `simulation_goal_met`.

Sensor V2, remote-child campaign control, arbitrary launch arguments, non-default terrain, source application, export, deployment, hardware actuation, and a panel-side model client are outside V1. A policy-only baseline may retain a frozen physics profile, but a baseline with any terrain override is rejected because V1 deterministic evaluation uses the default terrain. The existing `3.7.0-remote-parity` Child/worker protocol is unchanged.

<a id="enablement-and-storage"></a>
## Enablement and storage

Set `REDRHEX_AUTOPILOT_ENABLED=1` before starting Mother to enable campaign mutations. Without it, `GET /api/autopilot/capabilities` reports `enabled: false` and campaign writes fail closed. Keep Mother on loopback. If Autopilot is deliberately bound elsewhere, startup requires `REDRHEX_AUTOPILOT_BEARER_TOKEN` with at least 16 characters; this does not turn the complete Mother administrative surface into a public API.

Campaign metadata is stored in `logs/training_panel/autopilot.sqlite3` with SQLite WAL transactions. SHA-256-addressed immutable artifacts are stored below `logs/training_panel/autopilot_artifacts/`, and evaluator outputs are stored below `logs/training_panel/evaluations/`. Existing `logs/reward_agent/sessions.json` entries are imported as non-armable legacy references without deleting the source file.

<a id="schemas"></a>
## Versioned schemas

- `redrhex.autopilot.goal.v1` binds the task, stage, gait, selected directions, exact numeric command envelope, gates, initialization identity, immutable configuration identities, seeds, iteration cap, and campaign budget. Policy-only initialization requires all three of `baseline_run_id`, `baseline_checkpoint_iteration`, and `checkpoint_sha256`; fresh initialization requires all three to be null. The selected checkpoint must be an exact identity already recorded by History. The resolver does not scan a run directory or select a filesystem “latest” checkpoint.
- `redrhex.autopilot.reward-catalog.v1` exposes only compatible nonzero shaping weights. A mutable weight keeps its sign and remains within 80–120% of its campaign-start value. Its finite V1 lattice is the campaign-start value multiplied by 80%, 90%, 100%, 110%, and 120%. Optional draft `reward_bounds` supplies an absolute `[minimum, maximum]` for a selected key; it must contain the start value and stay inside the generated hard range. Narrowing clips and deduplicates the lattice. A draft may instead disable a key, and no connector write can widen any bound.
- `redrhex.autopilot.decision.v1` permits one action: `propose_candidate`, `pause`, or `request_patch_handoff`. A candidate changes exactly one catalog key from the current leader and its value must be one of the context's remaining lattice moves; the current point and previously attempted key/value points are unavailable.
- `redrhex.autopilot.evaluation.v1` binds per-command and per-episode evidence to the exact trained checkpoint and configuration/profile hashes. Its horizon identity requires the recorded evaluator `num_envs`, exactly 600 sweep steps, the V1 control timestep `step_dt = 1/60` second, and `duration_s = 600 × step_dt`. Episode evidence must cover every command/environment for all 600 samples, while each command's success duration must reconcile to its episode-aggregated success ratio and the frozen duration. The command CSV, episode CSV, summary CSV, and compact evaluated report are four distinct SHA-256-addressed artifacts. The report artifact is reopened and compared with durable state before delayed recovery and final confirmation. Missing, truncated, tampered, partial, fallback-selected, mismatched, malformed, artifact-divergent, or non-finite evidence is invalid.
- `redrhex.autopilot.campaign.v1` is the revisioned snapshot containing lifecycle, goal, catalog, leader, budget, process, lineage, decisions, evaluations, connector state, next actions, and terminal reason.

The maximum V1 budget is 24 training trials and 72 active GPU-hours, with at most 300 advisor polls. The reviewed draft must contain an explicit per-trial iteration cap. The fixed seed allocation is control and screening at 42, then paired control/winner confirmation at 43 and 44.

<a id="lifecycle"></a>
## Lifecycle and acceptance

The main path is `draft → armed → control_training → control_evaluating → awaiting_advisor → candidate_training → candidate_evaluating → confirming → simulation_goal_met`. Other durable states are `paused`, `waiting_for_chatgpt`, `patch_handoff`, `budget_exhausted`, `blocked_safety`, `stopped`, and `failed`. Only one campaign can hold the host campaign slot; drafts and terminal campaigns remain readable.

Training reward and TensorBoard trends are diagnostic only. Evaluation first applies identity, finite-data, strict-load, fall/health, tracking, sign/direction, leakage, stability, and absolute-energy gates. Only surviving reports receive deterministic ranking. Confirmation requires valid evidence for all three candidate and paired-control seeds, at least two candidate replicas passing every goal gate, improved median tracking over controls, and energy within its cap.

If validation or an internal controller operation fails while campaign training or evaluation is active, the service first commits a campaign-owned failure-stop intent. It keeps the exact active process identity until stop is confirmed, charges cumulative GPU usage, and then enters `blocked_safety` for validation failures or `failed` for internal failures. If signaling fails or Mother restarts between these steps, a later controller tick resumes the same intent instead of abandoning or duplicating the work.

<a id="rest-api"></a>
## Panel REST API

Read operations are side-effect free:

| Method and path | Result |
| --- | --- |
| `GET /api/autopilot/capabilities` | Feature state, supported tasks/stages, exact command profiles, default gates, budgets, and recorded baseline choices identified by run, checkpoint iteration, and SHA-256. |
| `GET /api/autopilot/campaigns` | Campaign list; optional `state` and `limit` filters. |
| `GET /api/autopilot/campaigns/{id}` | One `CampaignSnapshotV1`. |
| `GET /api/autopilot/campaigns/{id}/decision-context` | Compact bounded advisor context with per-key campaign-start/current values, hard bounds, complete/remaining lattice values, baseline-to-leader deltas, attempted moves, recent evidence, evidence IDs, remaining trial/GPU/confirmation/poll budgets, and permitted actions; raw logs, runtime paths, secrets, and video are omitted. Patch handoff adds only allowlisted source names/snippets and blob hashes. |
| `GET /api/autopilot/campaigns/{id}/events` | Append-only campaign events. |
| `GET /api/autopilot/campaigns/{id}/artifacts` | Artifact metadata. |
| `GET /api/autopilot/campaigns/{id}/artifacts/{artifact_id}` | One artifact descriptor and contained download link. |
| `GET /api/autopilot/campaigns/{id}/artifacts/{artifact_id}/download` | Immutable artifact bytes. |
| `GET /api/autopilot/campaigns/{id}/compare?trial_ids=a,b` | Deterministic comparison for 2–12 campaign trials. |
| `GET /api/autopilot/campaigns/{id}/patch-export` | The accepted review-only patch proposal; unavailable until one exists. |

Mutations are `POST /api/autopilot/campaigns`, `PATCH /api/autopilot/campaigns/{id}`, and `POST` to `/{id}/arm`, `/pause`, `/resume`, `/stop`, `/heartbeat`, or `/decisions`. Draft update, arm, resume, and emergency stop are administrative panel operations and are not exposed by the advisor connector.

<a id="mutation-contract"></a>
## Mutation contract

Every mutation requires `Content-Type: application/json`, one `Idempotency-Key` containing a safe 8–128 character identifier, `If-Match: "<revision>"`, and the same non-negative integer in body field `expected_revision`. Creating a draft uses revision `0`. Retrying the same intent reuses the idempotency key; reusing it with different input is rejected. A stale or reordered write returns a typed `redrhex.autopilot.error.v1` conflict and does not launch duplicate work. A finite candidate value that lies inside the hard numeric range but is not a remaining lattice point is also rejected.

The optional `REDRHEX_AUTOPILOT_BEARER_TOKEN` protects only `/api/autopilot` and is read from the panel process environment. Never place it in a request body, campaign artifact, repository file, or advisor context.

<a id="mcp-connector"></a>
## Narrow MCP connector

`plugins/redrhex-autopilot/` supplies a local skill and an adapter that accepts only a loopback panel URL. Its read tools list/get campaigns, obtain decision context, compare trials, and return artifact links. Its bounded writes create a draft, record an active-campaign heartbeat, propose one candidate, pause, request stop-after-current, or submit one review-only patch proposal. It cannot arm or resume, widen constraints or budgets, invoke arbitrary processes, access general files or shell commands, apply a patch, export or deploy a policy, actuate hardware, or affect unrelated work.

The installed plugin uses stdio. For a tunnel client, start the same adapter as stateless Streamable HTTP on loopback:

```bash
python3 plugins/redrhex-autopilot/scripts/redrhex_autopilot_mcp.py \
  --http --host 127.0.0.1 --port 8787
```

Point the tunnel client at `http://127.0.0.1:8787/mcp`. Configure a separate `REDRHEX_AUTOPILOT_MCP_TOKEN` and exact allowed origins when needed; do not expose the adapter on `0.0.0.0`. The supplied recurring prompt is designed for a 15-minute same-chat task using `gpt-5.6-terra` with medium reasoning; the declared model is audit metadata and does not select a model. The repository does not provision an OpenAI Secure MCP Tunnel, Platform tunnel ID/permissions, runtime API key, or ChatGPT Scheduled task. Those are external operator-owned services and credentials and must stay outside panel state and logs.

Follow the official [Scheduled tasks](https://learn.chatgpt.com/docs/automations), [Secure MCP Tunnel](https://developers.openai.com/api/docs/guides/secure-mcp-tunnels), [GPT-5.6 Terra](https://developers.openai.com/api/docs/models/gpt-5.6-terra), and [plugin authentication](https://developers.openai.com/plugins/build/auth) documentation for external setup. The tunnel uses outbound HTTPS and requires its own runtime API key. A future public or multi-user endpoint requires a separate reviewed authentication design; an authenticated MCP server is expected to implement OAuth 2.1.

<a id="patch-handoff"></a>
## Patch handoff

After deterministic exhaustion, `patch_handoff` exposes only allowlisted reward-source snippets, target symbols, and exact source blob hashes. One `redrhex.autopilot.patch-proposal.v1` unified-diff draft may be stored as an immutable runtime artifact. The panel never applies it. A human must review and hand it to Codex or an engineer; an accepted source edit changes code identity and therefore starts a new linked campaign.
