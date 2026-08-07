---
id: policy-contract-reference
title: Policy 與部署 Contract
lang: zh-TW
audience: shared
type: reference
status: active
owner: deployment
last_reviewed: 2026-08-07
---

<a id="dimensions"></a>
## 維度與頻率

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

<a id="actions"></a>
## Action 與 joint

Action 依序控制六個 main-drive joint 與六個 ABAD joint。六個 damper joint 是被動元件，真機不接受 command。確切 joint name 與 stage-5 scaling constant 定義於 `redrhex_contract.py`，並對 deployment YAML 與 training configuration 執行檢查。

<a id="commands"></a>
## Command envelope

鏡像的部署 limit 為 `vx` `0.0` 至 `0.56 m/s`、`vy` `-0.60` 至 `0.60 m/s`，以及 yaw rate `-0.70` 至 `0.70 rad/s`。特定 training stage 可能只取樣更窄範圍。

<a id="change-rule"></a>
## 變更規則

任何 dimension、order、normalization、rate、stage scaling、joint mapping、command envelope 或 frame 變更，都要審查 training、export、panel readiness、ONNX、ROS 與 hardware preflight。Parity test 與雙語操作/開發文件必須在同一變更更新。
