---
id: training-policy-architecture
title: 訓練與 Policy 架構
lang: zh-TW
audience: developer
type: explanation
status: active
owner: training
last_reviewed: 2026-08-07
---

<a id="tasks"></a>
## Task 與 agent

`Template-Redrhex-Direct-v0` 註冊完整環境。`Template-Redrhex-ForwardFast-Direct-v0` 衍生出 deterministic、forward-only 的 stage-1 profile。兩者都提供預設 PPO、privileged-teacher PPO 與 distillation entry point；SKRL entry 仍為相容性保留。

<a id="observations"></a>
## Observation 路徑

可部署 policy 使用一個 56-value frame 加上前四個 frame，共 280 個值。目前 56-value 順序為 base linear velocity、base angular velocity、projected gravity、main-drive sine/cosine 與 scaled velocity、ABAD position/velocity、3-value command、gait phase sine/cosine，以及 12 個 previous action。

預設 PPO 把 `policy + history` 同時提供給 actor 與 critic。Privileged teacher 使用 `teacher` group。Distillation 把 `policy + history` 提供給 student，並把 `teacher` 提供給 teacher network。不可讓訓練專用 privileged signal 進入部署。

<a id="curriculum"></a>
## Curriculum 與 action 路徑

五個階段分別隔離 forward、lateral、diagonal、yaw 與 mixed command。各階段的 command distribution、residual scale、warmup、reward multiplier 與 safety threshold 都會改變行為。Stage pipeline 預設完整 resume。除非明確關閉，`play.py` 會從 checkpoint run-name suffix 推斷 stage。

Policy 輸出 12 個 action：六個 main-drive 與六個 ABAD control。Damper joint 是被動模型元件，部署時不接受 command。環境在訓練 control rate 計算一次 control intent，再透過 Isaac 套用 simulator-specific target。

<a id="randomization"></a>
## Randomization 與 symmetry

完整環境包含 terrain、actuator、per-leg fault、delay、noise 與 push control。Recovery baseline 可能關閉它們；任何 robustness 宣稱都必須記錄 resolved configuration。目前 left-right data augmentation 已停用，因 tripod grouping 並非鏡像對稱，而 gait phase 不會一起轉換。

<a id="artifacts"></a>
## Artifact contract

訓練把 timestamped run 寫入 `logs/rsl_rl/<experiment>/`，包含 `model_*.pt`、event 與 resolved parameter。Playback 在 checkpoint 旁匯出 JIT 與 ONNX。面板及部署工具依賴此 layout；變更時需相容性測試與 release entry。
