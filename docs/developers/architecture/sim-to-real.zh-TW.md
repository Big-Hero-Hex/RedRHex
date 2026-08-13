---
id: sim-to-real-architecture
title: Sim-to-Real 校準架構
lang: zh-TW
audience: developer
type: explanation
status: active
owner: sim2real
last_reviewed: 2026-08-13
---

<a id="roles"></a>
## 證據角色

校準系統區分 raw hardware trace、受管理的 immutable episode、已審查 replay fixture、simulated trace、comparison result、direct-measurement profile、audit artifact 與 held-out promotion evidence。每次轉換都記錄 hash 與 provenance，避免 candidate 靜默替換輸入。

<a id="flow"></a>
## 資料流程

`tools.sim2real` 匯入 real trace 時需要 scenario、unit、frame、time base、dataset ID 與 episode ID。要具備 replay 資格，還需 operator-reviewed fixed-base fixture。`run-sim` 使用相同 scenario 與可選的明確 profile。`compare` 產生 metric difference。`sweep` 產生有界 candidate，且只有提供必要 real trace 與 audit evidence 才會執行。

<a id="profile"></a>
## Profile 邊界

`CalibrationProfileV1` 是有版本資料，不是隱含全域設定。訓練與 playback 預設不使用 candidate profile，只有 `--physics-profile` 會明確載入。Profile 建構與 promotion 分開；語法有效不代表已 promotion，仍需通過 hash-bound audit 與 held-out evidence。

<a id="failure-model"></a>
## 失敗模型

若 provenance 不完整、JSON 有重複 key 或非有限值、hardware mapping 未解決、publisher 混淆、timing 衝突、artifact 未驗證、physics audit 失敗、held-out metric 不完整，或 evidence 非穩態，workflow 會 fail closed。輸出路徑也不會被靜默覆寫。

<a id="torsion-springs"></a>
## Torsion-spring 邊界

Passive-spring profile 使用 canonical aliases `damper_0` 到 `damper_5`。代表性 calibration 與另一份 holdout evidence 來自 `damper_0`；通過的 neutral-constrained stiffness 傳播到所有 aliases，而 damping 在另行識別前維持零。`explicit` 與 `native` 是相同 contract 的實作，不是可互換的 evidence：selected backend、profile ID/hash 與 calibration status 會綁定 training、playback、evaluation、Panel history 與 deployment checks。

<a id="limits"></a>
## 解讀限制

Scenario comparison 只能定位已量測 state 與 command envelope 的差異。它不能驗證未量測 contact、terrain、thermal effect、structural compliance、estimator 行為或長時 locomotion。擴充 scenario 時必須提供新審查證據與明確驗收條件。

<a id="operator"></a>
## 操作程序

依安全順序執行的流程請見[物理校準](../../operators/calibration/physics-calibration.zh-TW.md)。
