---
id: adr-five-stage-curriculum
title: ADR 0001 五階段移動 Curriculum
lang: zh-TW
audience: developer
type: decision
status: accepted
owner: training
last_reviewed: 2026-08-07
---

<a id="context"></a>
## 背景

單一 mixed command distribution 讓 forward、lateral、diagonal 與 yaw 失敗難以隔離。後續技能也可能破壞已可用的 forward gait，而總 reward 無法清楚呈現 policy 是否真的遵循要求方向。

<a id="decision"></a>
## 決策

完整 RedRHex task 支援五階段 curriculum：forward、lateral、diagonal、yaw，最後 mixed integration。每階段擁有自己的 command distribution、behavior scaling、warmup/safety threshold 與 skill-specific reward multiplier。受支援的 pipeline 預設使用完整 checkpoint resume 交接。

ForwardFast 保持為獨立 forward-only task，用於受限快速迭代。它不是 stage 5，也不可描述成完整 locomotion policy。

<a id="consequences"></a>
## 後果

訓練可一次診斷一項技能，並在階段間套用 health gate。Run name 與 checkpoint path 帶有 stage 意義，因此 playback 可能推斷 `env.stage`。Operator resume 時必須保留 run tag，並分別評估每項技能。額外狀態與設定會增加 contract surface；stage 行為改變時需要測試。

<a id="alternatives"></a>
## 曾考慮方案

單一 mixed task 保留為最終整合階段，但不再作為唯一學習路徑。四階段方案被拒絕，因 diagonal locomotion 在 mixed integration 前需要獨立轉換。Policy-only handoff 仍可選用，但因不保留 optimizer 與 iteration 連續性，所以不是預設。

<a id="evidence"></a>
## 證據與限制

Staged pipeline、stage-aware environment、playback inference 與 health gate 均已實作。Smoke 證據只證明執行路徑，不證明最終 tracking performance 或優於其他 controller。長時間比較仍屬 roadmap 工作。
