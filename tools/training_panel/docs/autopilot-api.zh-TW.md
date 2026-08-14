---
id: training-panel-autopilot-api
title: Training Panel Autopilot API 參考
lang: zh-TW
audience: developer
type: reference
status: active
owner: panel
last_reviewed: 2026-08-14
---

<a id="boundary"></a>
## 邊界

Autopilot V1 是預設關閉的本機 campaign API，支援 `Template-Redrhex-ForwardFast-Direct-v0` stage 1 與 `Template-Redrhex-Direct-v0` stage 1–5 的 standard PPO。Panel service 是唯一的 campaign mutation authority。外部 advisor 可以提交有界 decision，但 identity validation、trial allocation、工作啟動與評估、budget enforcement、evidence ranking，以及 `simulation_goal_met` 宣告都由 deterministic panel code 負責。

Sensor V2、remote-child campaign control、任意 launch argument、non-default terrain、source application、export、deployment、hardware actuation 與 panel-side model client 都不在 V1 範圍內。Policy-only baseline 可以保留 frozen physics profile，但任何含 terrain override 的 baseline 都會被拒絕，因為 V1 deterministic evaluation 使用 default terrain。既有 `3.7.0-remote-parity` Child/worker protocol 不變。

<a id="enablement-and-storage"></a>
## 啟用與儲存

啟動 Mother 前設定 `REDRHEX_AUTOPILOT_ENABLED=1`，才能啟用 campaign mutation。未設定時，`GET /api/autopilot/capabilities` 會回報 `enabled: false`，campaign write 則 fail closed。Mother 應維持 loopback binding。若刻意讓 Autopilot 使用非 loopback binding，startup 會要求至少 16 字元的 `REDRHEX_AUTOPILOT_BEARER_TOKEN`；這不會把完整 Mother administrative surface 變成 public API。

Campaign metadata 以 SQLite WAL transaction 保存於 `logs/training_panel/autopilot.sqlite3`。SHA-256 addressing 的 immutable artifact 位於 `logs/training_panel/autopilot_artifacts/`，evaluator output 位於 `logs/training_panel/evaluations/`。既有 `logs/reward_agent/sessions.json` entry 會匯入成不可 arm 的 legacy reference，且不會刪除來源檔案。

<a id="schemas"></a>
## Versioned schema

- `redrhex.autopilot.goal.v1` 綁定 task、stage、gait、所選 direction、確切 numeric command envelope、gate、initialization identity、immutable configuration identity、seed、iteration cap 與 campaign budget。Policy-only initialization 必須同時提供 `baseline_run_id`、`baseline_checkpoint_iteration` 與 `checkpoint_sha256`；fresh initialization 則要求三者全為 null。所選 checkpoint 必須是已由 History 記錄的 exact identity。Resolver 不會掃描 run directory，也不會選取 filesystem 的「latest」checkpoint。
- `redrhex.autopilot.reward-catalog.v1` 只暴露相容且非零的 shaping weight。可調 weight 必須維持 sign，且範圍不得超出 campaign-start value 的 80–120%。V1 finite lattice 是 campaign-start value 的 80%、90%、100%、110% 與 120%。Optional draft `reward_bounds` 可為所選 key 提供 absolute `[minimum, maximum]`；它必須包含 start value，且保持在產生的 hard range 內。縮窄範圍會裁切 lattice 並移除重複值。Draft 也可以停用 key，任何 connector write 都不能擴大任何 bound。
- `redrhex.autopilot.decision.v1` 只允許 `propose_candidate`、`pause` 或 `request_patch_handoff`。Candidate 與目前 leader 相比只能改動一個 catalog key，且 value 必須是 context 中尚未使用的 lattice move；目前 point 與先前已嘗試的 key/value point 均不可使用。
- `redrhex.autopilot.evaluation.v1` 把 per-command 與 per-episode evidence 綁定到確切 trained checkpoint 及 configuration/profile hash。其 horizon identity 要求 evaluator 記錄的 `num_envs`、恰好 600 個 sweep step、V1 control timestep `step_dt = 1/60` 秒，以及 `duration_s = 600 × step_dt`。Episode evidence 必須為每個 command/environment 涵蓋全部 600 個 sample；每個 command 的 success duration 也必須與 episode aggregate 的 success ratio 及 frozen duration 一致。Command CSV、episode CSV、summary CSV 與 compact evaluated report 是四個獨立且以 SHA-256 addressing 的 artifact。延後復原與最終 confirmation 前會重新開啟 report artifact，並與 durable state 比對。缺漏、截斷、遭竄改、partial、fallback-selected、mismatched、malformed、artifact-divergent 或 non-finite evidence 都無效。
- `redrhex.autopilot.campaign.v1` 是 revisioned snapshot，包含 lifecycle、goal、catalog、leader、budget、process、lineage、decision、evaluation、connector state、next action 與 terminal reason。

V1 budget 上限為 24 個 training trial 與 72 active GPU-hour，advisor poll 最多 300 次。經確認的 draft 必須包含明確的 per-trial iteration cap。固定 seed allocation 為 control/screening 使用 42，paired control/winner confirmation 使用 43 與 44。

<a id="lifecycle"></a>
## Lifecycle 與 acceptance

主要路徑為 `draft → armed → control_training → control_evaluating → awaiting_advisor → candidate_training → candidate_evaluating → confirming → simulation_goal_met`。其他 durable state 為 `paused`、`waiting_for_chatgpt`、`patch_handoff`、`budget_exhausted`、`blocked_safety`、`stopped` 與 `failed`。同一時間只有一個 campaign 可以持有 host campaign slot；draft 與 terminal campaign 仍可讀取。

Training reward 與 TensorBoard trend 僅供診斷。Evaluation 會先套用 identity、finite-data、strict-load、fall/health、tracking、sign/direction、leakage、stability 與 absolute-energy gate。只有通過的 report 才會進入 deterministic ranking。Confirmation 要求三個 candidate seed 及其 paired control 都有 valid evidence，至少兩個 candidate replica 通過所有 goal gate，candidate median tracking 優於 control，且 energy 維持在 cap 內。

若 campaign training 或 evaluation 執行中發生 validation 或 internal controller failure，service 會先 commit 只屬於該 campaign 的 failure-stop intent。它會保留 exact active process identity，直到確認停止、計入 cumulative GPU usage，然後 validation failure 進入 `blocked_safety`，internal failure 則進入 `failed`。若 signal 失敗，或 Mother 在這些步驟間 restart，後續 controller tick 會繼續同一 intent，不會遺棄或重複工作。

<a id="rest-api"></a>
## Panel REST API

Read operation 不產生 side effect：

| Method 與 path | 結果 |
| --- | --- |
| `GET /api/autopilot/capabilities` | Feature state、支援的 task/stage、確切 command profile、default gate、budget，以及由 run、checkpoint iteration 與 SHA-256 識別的已記錄 baseline choice。 |
| `GET /api/autopilot/campaigns` | Campaign list；可使用 `state` 與 `limit` filter。 |
| `GET /api/autopilot/campaigns/{id}` | 一個 `CampaignSnapshotV1`。 |
| `GET /api/autopilot/campaigns/{id}/decision-context` | Compact bounded advisor context，包含每個 key 的 campaign-start/current value、hard bound、完整/剩餘 lattice value、baseline-to-leader delta、attempted move、recent evidence、evidence ID、剩餘 trial/GPU/confirmation/poll budget 與 permitted action；不含 raw log、runtime path、secret 或 video。Patch handoff 只另加 allowlisted source name/snippet 與 blob hash。 |
| `GET /api/autopilot/campaigns/{id}/events` | Append-only campaign event。 |
| `GET /api/autopilot/campaigns/{id}/artifacts` | Artifact metadata。 |
| `GET /api/autopilot/campaigns/{id}/artifacts/{artifact_id}` | 一個 artifact descriptor 與受 containment 保護的 download link。 |
| `GET /api/autopilot/campaigns/{id}/artifacts/{artifact_id}/download` | Immutable artifact byte。 |
| `GET /api/autopilot/campaigns/{id}/compare?trial_ids=a,b` | 2–12 個 campaign trial 的 deterministic comparison。 |
| `GET /api/autopilot/campaigns/{id}/patch-export` | 已接受的 review-only patch proposal；尚未建立時不可用。 |

Mutation 包含 `POST /api/autopilot/campaigns`、`PATCH /api/autopilot/campaigns/{id}`，以及對 `/{id}/arm`、`/pause`、`/resume`、`/stop`、`/heartbeat` 或 `/decisions` 的 `POST`。Draft update、arm、resume 與 emergency stop 都是 administrative panel operation，不會暴露給 advisor connector。

<a id="mutation-contract"></a>
## Mutation contract

每個 mutation 都需要 `Content-Type: application/json`、一個含 8–128 個安全字元的 `Idempotency-Key`、`If-Match: "<revision>"`，以及 body 中數值相同的 non-negative integer `expected_revision`。建立 draft 時 revision 為 `0`。重試同一 intent 時應重用 idempotency key；同一 key 搭配不同 input 會被拒絕。Stale 或 reordered write 會回傳 typed `redrhex.autopilot.error.v1` conflict，且不會啟動重複工作。Finite candidate value 即使位於 hard numeric range 內，只要不是 remaining lattice point，也會被拒絕。

可選的 `REDRHEX_AUTOPILOT_BEARER_TOKEN` 只保護 `/api/autopilot`，並從 panel process environment 讀取。絕不可把 token 放入 request body、campaign artifact、repository file 或 advisor context。

<a id="mcp-connector"></a>
## Narrow MCP connector

`plugins/redrhex-autopilot/` 提供 local skill，以及只接受 loopback panel URL 的 adapter。Read tool 可以 list/get campaign、取得 decision context、比較 trial 與取得 artifact link。有界 write 可以建立 draft、記錄 active-campaign heartbeat、提出一個 candidate、pause、要求 stop-after-current，或提交一個 review-only patch proposal。它不能 arm 或 resume、擴大 constraint 或 budget、呼叫任意 process、存取通用 file 或 shell command、套用 patch、export/deploy policy、操作 hardware，或影響無關工作。

Installed plugin 使用 stdio。Tunnel client 可用同一 adapter 在 loopback 啟動 stateless Streamable HTTP：

```bash
python3 plugins/redrhex-autopilot/scripts/redrhex_autopilot_mcp.py \
  --http --host 127.0.0.1 --port 8787
```

Tunnel client 應連到 `http://127.0.0.1:8787/mcp`。需要時另行設定 `REDRHEX_AUTOPILOT_MCP_TOKEN` 與確切 allowed origin；不得在 `0.0.0.0` 暴露 adapter。隨附 recurring prompt 是為每 15 分鐘一次、使用 `gpt-5.6-terra` 與 medium reasoning 的 same-chat task 所設計；declared model 只是 audit metadata，不會選擇 model。Repository 不會 provision OpenAI Secure MCP Tunnel、Platform tunnel ID/permission、runtime API key 或 ChatGPT Scheduled task。這些都是 operator 管理的外部 service 與 credential，必須留在 panel state 與 log 之外。

外部設定請遵循官方 [Scheduled tasks](https://learn.chatgpt.com/docs/automations)、[Secure MCP Tunnel](https://developers.openai.com/api/docs/guides/secure-mcp-tunnels)、[GPT-5.6 Terra](https://developers.openai.com/api/docs/models/gpt-5.6-terra) 與 [plugin authentication](https://developers.openai.com/plugins/build/auth)文件。Tunnel 使用 outbound HTTPS，並需要自己的 runtime API key。未來若提供 public 或 multi-user endpoint，必須另做經 review 的 authentication design；authenticated MCP server 預期實作 OAuth 2.1。

<a id="patch-handoff"></a>
## Patch handoff

Deterministic exhaustion 後，`patch_handoff` 只暴露 allowlisted reward-source snippet、target symbol 與確切 source blob hash。系統可把一份 `redrhex.autopilot.patch-proposal.v1` unified-diff draft 保存成 immutable runtime artifact。Panel 絕不套用它。Human 必須先 review，再交給 Codex 或 engineer；任何接受的 source edit 都會改變 code identity，因此必須建立新的 linked campaign。
