---
id: ros2-deployment-troubleshooting
title: ROS 2 部署疑難排解
lang: zh-TW
audience: shared
type: troubleshooting
status: active
owner: deployment
last_reviewed: 2026-08-07
---

<a id="troubleshoot-ros2-deployment"></a>
# ROS 2 部署疑難排解

診斷期間保持馬達輸出停用。若動作不符合預期，先觸發實體急停並移除馬達電源。

<a id="policy-enable-is-rejected"></a>
## Policy enable 被拒絕

檢查 `/redrhex/state_machine_state` 與 `/redrhex/diagnostics`。在到達 `POLICY_READY` 前、E-stop 或 protective fault 後、必要訊號逾時，或設定的靜止姿態閘門失敗時，policy enable 預期會被拒絕。

不可為了消除 fault 而放寬 threshold。修正 frame、訊號、時序或 state transition，並重做 dry-run 證據。

<a id="motor-output-remains-disabled"></a>
## 馬達輸出保持停用

即使推論正在執行，這也可能是正確行為。確認操作員刻意要求了 `/redrhex/enable_motors`、目前 state 允許、低階 heartbeat 要求已滿足，且沒有 safety fault 清除 latch。Policy enable 與 motor enable 彼此獨立。

<a id="observation-or-inference-dimension-fails"></a>
## 觀測或推論維度失敗

核對 ONNX graph：單幀輸入 56 或五幀輸入 280，輸出 12。確認部署 config 與 policy 來自相同環境契約。不可用填補、裁切或重排 tensor 的方式強迫不相容 artifact 載入。

<a id="robot-appears-tilted-while-level"></a>
## 機器人水平時看似傾斜

IMU frame 可能與訓練 body frame 不同。量測並設定 `imu_mount_rpy_deg`，再將靜止 projected gravity 與訓練證據比較。不可用放寬 roll/pitch safety limits 補償。

<a id="joint-motion-has-the-wrong-direction-or-zero"></a>
## Joint 方向或零點錯誤

停止所有多關節測試。機器人架空並限流，重新執行單關節 preview 與 command 測試。只修改 `main_drive_sign`、`abad_sign` 或對應 zero-offset array 中已被證明的項目，再重跑 readiness 與單關節測試。

<a id="heartbeat-or-sensor-is-stale"></a>
## Heartbeat 或 sensor 逾時

檢查 timestamps、network route、workspace sourcing、topic names 與 publisher 數量。預設 heartbeat timeout 為 0.10 秒、sensor timeout 0.10 秒、motor feedback timeout 0.25 秒、command timeout 0.25 秒。修正 producer 或 transport；硬體操作時不可繞過必要 heartbeat。

<a id="more-than-one-motor-command-publisher-exists"></a>
## 出現多個 motor-command publishers

保持 relay 關閉。停止 calibration、standing、tripod 或先前 RL process 等競爭控制器。重新檢查 graph，直到只剩一條經授權的 command path。

<a id="recovery-checklist"></a>
## 復原檢查表

修正原因後，回到停用輸出，重新 assert 並刻意 clear 軟體 E-stop，重做 mock 或 preview 階段，再逐階段取回硬體控制權。Protective stop 後絕不可直接從失敗階段繼續。
