---
name: redrhex-autopilot
description: Inspect and advance a bounded RedRHex Training Panel Autopilot campaign through its narrow MCP tools. Use when the user asks ChatGPT or Codex to create a locomotion-goal draft, monitor an armed campaign, compare training trials, propose the next reward-weight candidate, pause or stop a campaign, prepare a source-patch handoff, or configure the recurring 15-minute same-chat advisor workflow.
---

# RedRHex Autopilot

Keep the Training Panel authoritative. Use evidence to propose one bounded action; never infer permission to arm, resume, widen, deploy, edit source, or operate hardware.

## One-decision workflow

1. Call `redrhex_list_campaigns` when no campaign ID is supplied. Work on only one explicitly selected campaign.
2. Call `redrhex_get_campaign` and classify its state. Stop without a write for a terminal, paused, or draft campaign. Continue to the no-op heartbeat step for active non-advisor work; to decision context for `awaiting_advisor` or `waiting_for_chatgpt`; or to the patch path for a `patch_handoff` campaign whose context still permits one `submit_patch_proposal`. Never arm or resume it.
3. For a nonterminal no-op visit while campaign work is armed, training, evaluating, or confirming, call `redrhex_advisor_heartbeat` once and make no decision. Its returned revision is authoritative; re-read only if needed for the one-sentence report. Do not heartbeat drafts, paused campaigns, `waiting_for_chatgpt`, terminal campaigns, or a visit that will submit a decision/proposal—the metadata on that write records the visit already.
4. When the campaign is `awaiting_advisor` or `waiting_for_chatgpt`, or is in patch handoff with a proposal still permitted, call `redrhex_get_decision_context`. Treat its allowed moves, immutable constraints, hard-gate evidence, revision, and remaining budget as authoritative.
5. If evidence is incomplete or invalid, make no reward proposal. Pause only when the context identifies a safety/configuration problem; otherwise record a heartbeat only if the current state permits it and none has yet been recorded.
6. Compare trials with `redrhex_compare_trials` when the context identifies two or more comparable trial IDs. Do not compare mismatched checkpoints, profiles, backends, seeds, or invalid reports.
7. Choose exactly one decision action:
   - Call `redrhex_propose_candidate` for one allowlisted reward key and finite value already permitted by the context's `constraints` or `remaining_allowable_moves`.
   - Call `redrhex_pause_campaign` when deterministic evidence reports a safety concern or human review is required.
   - Call `redrhex_request_stop_after_current` only when the user requested a graceful stop or the context exposes a terminal budget/guardrail reason.
   - Call `redrhex_submit_patch_proposal` only after the campaign enters `patch_handoff`, the decision context permits `submit_patch_proposal`, supplies the allowlisted source blobs and exact hashes, and the user authorized a draft. This records one review artifact and leaves the campaign in patch handoff; it does not edit the repository.
   - Make no decision write when training/evaluation is active, no valid move exists, or the campaign is terminal, except for the one patch-handoff proposal above.
8. Re-read the campaign after a decision write. Report the returned state and next permitted action without declaring success yourself.

Use a fresh UUID as `idempotency_key` for a new intent. Reuse the same key when retrying that exact write. Supply the revision returned by the latest read as `expected_revision`; on a conflict, re-read and reconsider instead of replaying blindly. The adapter records fixed skill/prompt versions, medium reasoning, and the model declared by `REDRHEX_AUTOPILOT_ADVISOR_MODEL` (default `gpt-5.6-terra`) with advisor writes; never accept those audit fields from tool arguments.

## Candidate policy

- Base every hypothesis on deterministic evaluation deltas, not training reward alone.
- Change one weight from the current leader. Keep its sign and server-provided 80–120% campaign bounds.
- Prefer the smallest allowed change that tests a clear hypothesis.
- Do not alter command ranges, termination, fall/health gates, physics, spring settings, tracking targets, checkpoints, seeds, iteration caps, budgets, or arbitrary Hydra arguments.
- Never call a campaign successful. Only report `simulation_goal_met` after the panel enters that state.
- Never claim hardware readiness, export/deploy a policy, apply a patch, or control unrelated processes.

## Recurring same-chat task

When the user requests unattended advising, create a recurring task in this same chat with a 15-minute cadence. Select `gpt-5.6-terra` with medium reasoning when the scheduler exposes model controls. Use this instruction:

> Use `$redrhex-autopilot` to inspect campaign `<campaign-id>` and take at most one permitted decision. Reuse no idempotency key except for an exact retry. For a nonterminal no-op visit, record exactly one advisor heartbeat; never heartbeat a visit that submits a decision. Make no write for a terminal, paused, or draft campaign, except for one explicitly permitted patch proposal in `patch_handoff`. Report the current state in one sentence. Pause this recurring task after terminal handling is complete.

Do not create more than one recurring task for a campaign or exceed the campaign's 300-poll limit. Each invocation must take at most one write action. Treat connector or panel unavailability as a no-op; do not substitute shell access or direct filesystem inspection. After five consecutive unavailable polls, tell the user that the campaign is waiting for ChatGPT, but do not stop active local work.

## Stop rules

- Stop polling and pause the recurring task for `simulation_goal_met`, `budget_exhausted`, `blocked_safety`, `stopped`, or `failed`. For `patch_handoff`, first submit at most one authorized proposal when `proposal_already_submitted` is false and the context permits it; then pause the task. If no patch draft was authorized, pause without writing.
- Make no decision mutation for `draft`, `paused`, `armed`, training/evaluating states, or any state whose `next_permitted_actions` excludes the proposed tool. A single metadata-only heartbeat is permitted only in the active no-op states listed above.
- Pause the campaign if context identity hashes disagree, evidence contains a non-finite value, or the server marks a safety gate invalid.
- Never widen reward bounds after four valid non-improving candidates, no eligible move, or insufficient confirmation budget. Let the deterministic panel enter patch handoff; do not fabricate another candidate to force progress.
- On stale revision, validation rejection, or duplicate conflict, re-read once. Do not evade the server decision with a different payload or key.

## Connection boundary

The MCP adapter accepts only a loopback panel URL from `REDRHEX_AUTOPILOT_PANEL_URL` (default `http://127.0.0.1:8080`). If the panel requires its optional bearer secret, set `REDRHEX_AUTOPILOT_BEARER_TOKEN` in the MCP process environment. Never request, display, or place that secret in tool arguments.

Set `REDRHEX_AUTOPILOT_ADVISOR_MODEL` only when the scheduled advisor actually uses a model other than the default `gpt-5.6-terra`. This value is audit metadata, not a way to select or invoke a model; the scheduler remains responsible for its real model setting.

Installed plugins use the stdio adapter declared in `.mcp.json`. For Secure MCP Tunnel, start the same adapter as a loopback-only stateless Streamable HTTP server:

```bash
python3 plugins/redrhex-autopilot/scripts/redrhex_autopilot_mcp.py \
  --http --host 127.0.0.1 --port 8787
```

Point the tunnel at `http://127.0.0.1:8787/mcp`. Set a separate `REDRHEX_AUTOPILOT_MCP_TOKEN` of at least 16 non-whitespace characters and configure the tunnel to send it as a bearer token. Use `REDRHEX_AUTOPILOT_MCP_ALLOWED_ORIGINS` only when the tunnel forwards an `Origin` header that is not a local default; provide comma-separated exact origins, never wildcards. Do not bind the adapter to `0.0.0.0` or expose the Training Panel itself.
