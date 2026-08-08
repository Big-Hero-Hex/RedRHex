---
id: system-architecture
title: RedRHex 系統架構
lang: zh-TW
audience: developer
type: explanation
status: active
owner: core
last_reviewed: 2026-08-07
---

<a id="layers"></a>
## 系統分層

RedRHex 目前連接四個受維護系統：

```text
Training Panel / Reward Agent
            -> train、play、evaluation scripts
            -> RedRHex Isaac Lab task 與 RSL-RL
            -> checkpoints、events、exports、reports
            -> ROS2 ONNX controller 與 low-level bridge
```

面板與 reward agent 編排現有 script 介面。`RedRhex` 擴充套件擁有模擬與訓練行為。ROS2 workspace 使用匯出的 ONNX 並鏡像部署 contract。Sim-to-real 工具建立已驗證證據，以及僅在明確指定時使用的 physics profile。

<a id="source-ownership"></a>
## 原始碼責任

- `source/RedRhex/RedRhex/tasks/direct/redrhex/` 負責 Isaac task、環境設定、行為與 agent entry point。
- `scripts/rsl_rl/` 負責訓練、播放、分階段訓練、驗證與 command sweep entry point。
- `tools/training_panel/` 負責本機與遠端操作、產物及 deployment readiness。
- `tools/reward_agent/` 負責有界 reward candidate session 與 trial 編排。
- `tools/sim2real/` 負責 characterization 證據、比較、profile 驗證與 promotion gate。
- `ros2_ws/src/` 負責部署 message、policy control、安全與硬體 bridge。

<a id="stable-interfaces"></a>
## 穩定介面

目前公開邊界包括兩個 Gym task ID、command-line entry point、RSL-RL checkpoint layout、面板 run/artifact discovery、56/280 observation 與 12-action policy contract、60 Hz control rate，以及 ROS message/topic。跨越任一邊界的變更都需要操作或開發文件與相容性審查。

<a id="known-coupling"></a>
## 已知耦合

Isaac 環境目前仍同時包含 simulator I/O、reward/observation math、gait/command state、randomization、buffer 與 logging。部分 contract fact 仍鏡像到 ROS，透過 parity test 保護，而非由單一套件產生。所提議的 core-first reboot 只描述可能的抽離方式，並非目前已啟用架構。

<a id="next"></a>
## 相關文件

- [訓練與 policy 架構](training-and-policy.zh-TW.md)
- [Reward 與能量模型](reward-and-energy.zh-TW.md)
- [Sim-to-real 架構](sim-to-real.zh-TW.md)
- [子系統責任](../subsystems/ownership.zh-TW.md)
