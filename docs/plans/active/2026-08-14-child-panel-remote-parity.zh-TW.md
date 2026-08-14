---
id: child-panel-remote-parity-plan
title: Child Panel 3.7 遠端同等升級實作計畫
lang: zh-TW
audience: developer
type: plan
status: active
owner: panel
last_reviewed: 2026-08-14
---

<a id="objective"></a>
## 目標

將[核准的 remote-parity 設計](../../designs/active/2026-08-14-child-panel-remote-parity.zh-TW.md)交付為單一 `3.7.0-remote-parity` release，同時保留 Mother 3.6.4 Drive export 作為獨立記錄的 prerequisite baseline，並保留所有 Mother-only boundary。

<a id="implementation"></a>
## 實作 checklist

- [x] 新增 versioned protocol、additive schema/security migration、authoritative role、request idempotency、受限 metadata/cancellation RPC、Physics preset、capability data 與 bounded run projection。
- [x] 新增 worker-side checkpoint resolution、GPU classification、Drive export、Deploy validation、MuJoCo smoke/recording、scalar downsampling、run evidence 與經 redaction 且 idempotent 的 local activity projection。
- [x] 升級 buildless Child shell、routing、draft、responsive navigation、Train route、Physics、all-runs History、safe action、Deploy、Detection、Activity、compatibility fallback 與 role gate。
- [x] 新增 Node、Python、schema 與 mocked Child Playwright coverage，涵蓋 390 px、768 px 與 desktop width。
- [x] 更新雙語 maintained architecture、operator、remote-operation、troubleshooting、compatibility、release、design、plan、index 與 router。
- [ ] 使用 deployed credential 執行 staging Supabase smoke，並在接受 production remote job 前記錄結果。

<a id="staging"></a>
## Staging checklist

Pause acceptance。套用 `tools/training_panel/supabase/migrations/20260814_370_remote_parity.sql`、更新 Mother、restart worker，確認 `3.7.0-remote-parity` heartbeat 與 capability row。驗證 viewer/operator/admin RLS 行為、actor-role spoof rejection、old-worker read-only fallback、queue/stop、video/ONNX/Drive、Deploy/MuJoCo、activity attribution、queued cancellation 與 single/bulk admin deletion。上述檢查通過後才發布 Child asset，再恢復 acceptance。

<a id="verification"></a>
## 驗證

Repository gate 包含 Node remote test、Training Panel Python test、專用 Child Playwright suite、完整 Mother Playwright suite、documentation validation/unit test 與 `git diff --check`。此 workspace 沒有 Supabase deployment credential，因此 staging 刻意維持未完成；code completion 不會假稱已有該外部 evidence。

<a id="completion-summary"></a>
## 完成摘要

所有 local gate 通過後，implementation 與 local verification 即完成。本計畫在 staging checklist 實際執行前保持 active。記錄 staging evidence 後，將剩餘 durable detail 移入 maintained doc 與 release record，依 lifecycle governance 移除此 temporary plan，並更新 plan index。
