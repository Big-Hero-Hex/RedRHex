---
id: panel-physics-calibration
title: Panel Physics 與校正工作區
lang: zh-TW
audience: developer
type: design
status: proposed
owner: panel
last_reviewed: 2026-08-14
---

<a id="problem"></a>
## 問題

Recovery snapshot 在相同 Training Panel 介面中混合了 simulation physics browser editor、Sensor V2 launch routes 與 ForwardFast defaults。這些程式適合檢視，但 physical values 是 candidates，不是 measured calibration evidence，不得被描述成 hardware-ready defaults。

<a id="boundary"></a>
## 提案邊界

此工作保留在 `feature/panel-physics-calibration-wip`，並堆疊於 Student Distillation V2 core。Branch 可提供 schema-validated simulation fields、sparse presets、run-scoped `CalibrationProfileV1` snapshots 與 strict Student V2 checkpoint handoffs，但必須保留已發行 Panel 3.6 的 navigation、security、provenance、progress、spring-backend 與 rollback behavior。

Baseline 繼承 repository 與 USD defaults。Non-empty candidate 必須為 explicit、run-scoped 且 simulation-only。若沒有對應的 reviewed evidence record，UI 不得把數值標示為 measured、calibrated、safe 或適用於 hardware。

<a id="integration"></a>
## 整合契約

Play、recording、export 與 deployment checks 會重用所選 run 的 task、agent route、physics snapshot 與 spring backend。Sensor V2 browser routes 維持 additive，並在 checkpoint kind 不符時 fail closed。Standard V1 route 保持可用。Process 與 artifact paths 仍限制在 repository-owned roots。

<a id="merge-gates"></a>
## 合併 gates

- 保留所有目前 Panel、sim-to-real、documentation 與 UI regression suites。
- 依 active `CalibrationProfileV1` consumer 審查全部 113 個 schema fields，並證明 sparse round-trip behavior。
- 驗證 Baseline 不傳送 candidate profile，且每個 non-empty profile 都會 snapshot 並重用。
- 對 standard training 與每個 Sensor V2 route 完成本機 browser smoke，且不啟動 production 或 hardware run。
- 在既有 evidence gates 通過前，physical calibration、motor enable 與 hardware promotion 維持 blocked。

<a id="rollback"></a>
## Rollback

不合併此 branch，即可讓正常操作不包含此功能。在 branch 內選擇 Baseline 與 standard training route，可避免 candidate physics 與 Student V2 routing。現有 V1 artifacts 與 Panel 3.6 behavior 保持 compatibility baseline。
