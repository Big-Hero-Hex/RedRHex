---
id: operator-physics-calibration
title: 校準 Sim-to-Real 物理參數
lang: zh-TW
audience: operator
type: how-to
status: active
owner: sim2real
last_reviewed: 2026-08-07
---

<a id="safety"></a>
## 安全與證據邊界

校準使用受限場景與不可變證據；它不會授權 policy 部署。初次 probe 時讓機器人架空或以其他方式限制，準備實體急停與斷電方式；若 low-level heartbeat、publisher graph、timing、mapping 或 stationarity 檢查失敗，立即停止。

<a id="order"></a>
## 必要順序

1. 建立並驗證 ROS command contract。
2. 記錄直接物理量測與其 reference pose。
3. 在沒有競爭 publisher 的情況下執行一次受限硬體 probe。
4. 將 raw episode 匯入受管理的 evidence store。
5. 在 Isaac 重播相同初始狀態與場景。
6. 比較 held-out 指標並產生有界 candidate。
7. 只用已驗證的直接量測與證據建立 profile。
8. 所有 audit 與 holdout gate 通過後才可 promote profile。

<a id="tool"></a>
## 校準工具

執行 `python -m tools.sim2real --help` 查看目前 subcommand 與必要 provenance 欄位。場景規格位於 `tools/sim2real/scenario_specs/`。除非明確傳入 `--physics-profile`，訓練與 playback 絕不載入 candidate profile。

<a id="promotion"></a>
## Promotion 規則

若證據缺少 hash、joint mapping 未解決、ROS publisher 混淆、初始狀態不完整、geometry 或 mass/CoM audit 失敗、holdout 資料非穩態，或 held-out 指標不完整，均不可 promote。Replay 相符只代表該量測場景的證據，不能證明所有 locomotion 物理都正確。

<a id="developer-context"></a>
## 開發背景

證據角色、profile 套用方式與實作邊界請見 [sim-to-real 架構](../../developers/architecture/sim-to-real.zh-TW.md)。
