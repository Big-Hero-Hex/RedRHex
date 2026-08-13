---
id: policy-contract-reference
title: Policy 與部署 Contract
lang: zh-TW
audience: shared
type: reference
status: active
owner: deployment
last_reviewed: 2026-08-14
---

<a id="dimensions"></a>
## 維度與頻率

以下是 legacy V1 contract，legacy tasks 與 ROS YAML 的 default 行為維持不變：

- 單幀 observation：56 個值
- Policy history：5 幀、280 個值
- Action：12 個值
- Simulation step：`1/120 s`
- Decimation：`2`
- Policy/control rate：`60 Hz`

<a id="observation-order"></a>
## Observation 順序

| Slice | 值 |
| --- | ---: |
| base linear velocity | 0–2 |
| base angular velocity | 3–5 |
| projected gravity | 6–8 |
| main-drive position sine/cosine | 9–20 |
| scaled main-drive velocity | 21–26 |
| scaled ABAD position | 27–32 |
| ABAD velocity | 33–38 |
| velocity command | 39–41 |
| gait phase sine/cosine | 42–43 |
| previous actions | 44–55 |

<a id="sensor-v2-contract"></a>
## Sensor-only V2 contract

V2 只能由 `Template-Redrhex-ForwardSensorV2-Direct-v0` 加 V2 runner/ROS route 選取。其 sensor frame 為 36 個 physical-feedback values：body gyro `0:3`、projected gravity `3:6`、六個 main sine values `6:12`、六個 main cosine values `12:18`、六個 main velocities `18:24`、六個 measured neutral-relative ABAD positions `24:30`，以及六個 measured ABAD velocities `30:36`。

Actor 收到 oldest-to-newest 的 `sensor_history [1,60,36]`，以及獨立 current `command [1,3]`。它絕不接收 base velocity、gait clock、previous action、odometry、commanded ABAD feedback、controller target 或 privilege。固定 ONNX outputs 為 `actions [1,12]` 與 `base_velocity_estimate [1,3]`。Contact output 不可用。

<a id="actions"></a>
## Action 與 joint

Action 依序控制六個 main-drive joint 與六個 ABAD joint。六個 damper joint 是被動元件，真機不接受 command。確切 joint name 與 stage-5 scaling constant 定義於 `redrhex_contract.py`，並對 deployment YAML 與 training configuration 執行檢查。

V2 forward F0–F5 的前六個 outputs 是 versioned procedural CPG 周圍的 learned residuals，後六個 outputs 強制 neutral。Decoder semantics 與 observation/action 兩個 SHA-256 都儲存在每個 V2 checkpoint 與 deployment bundle。

<a id="commands"></a>
## Command envelope

鏡像的部署 limit 為 `vx` `0.0` 至 `0.56 m/s`、`vy` `-0.60` 至 `0.60 m/s`，以及 yaw rate `-0.70` 至 `0.70 rad/s`。特定 training stage 可能只取樣更窄範圍。

<a id="change-rule"></a>
## 變更規則

任何 dimension、order、normalization、rate、stage scaling、joint mapping、command envelope 或 frame 變更，都要審查 training、export、panel readiness、ONNX、ROS 與 hardware preflight。Parity test 與雙語操作/開發文件必須在同一變更更新。

V2 另外要求 contract、action-decoder 與 calibration hash 完全相符。改變明確 IMU attitude mode 必須重新 training 與 export bundle。在一個 attitude mode 與十二個 encoder calibrations 都有經審查的 recorded evidence 支持前，hardware V2 維持 blocked。
