---
id: research-readiness-audit-2026-08-13
title: 2026-08-13 研究就緒度稽核
lang: zh-TW
audience: developer
type: audit
status: published
owner: project
last_reviewed: 2026-08-13
---

<a id="scope"></a>
## 範圍

本稽核檢視 RedRHex 的 locomotion、robustness、sim-to-real 或 energy 結果在支持研究主張前仍缺少什麼。範圍包含 repository 已實作的 training 與 deployment contract、2026-08-13 產生的研究路線報告，以及相關的 stage-1 training record。本稽核不證明硬體效能、電能節省或研究新穎性。

<a id="method"></a>
## 方法

交叉核對 generated report、source notes、materialized SQL snapshot、resolved run configuration、TensorBoard event、checkpoint、目前程式碼、測試與 canonical documentation。下文區分直接觀察、診斷性解讀及 proposed research work。報告 snapshot 後再次讀取 TensorBoard event，以辨識較晚寫入的資料。

<a id="executive-finding"></a>
## 核心結論

RedRHex 已具備相當完整的工程骨架：hybrid controller 與 residual-policy path、PPO 與 teacher/student entry point、明確的 sim-to-real calibration boundary、ONNX/ROS deployment check、Training Panel 與自動化測試。欠缺的研究資產是完整證據閉環：

1. 量測實體機器人；
2. 校準 simulator 並保留 held-out evidence；
3. 凍結可重現 baseline 與 evaluation protocol；
4. 執行 multi-seed ablation；
5. 以實測 outcome 重複進行 randomized hardware trial。

在關閉此閉環前加入更多 reward term 或進階 learning method，會增加實驗歧義。

<a id="evidence-status"></a>
## 證據狀態

| 領域 | 目前證據能支持 | 目前證據不能支持 |
| --- | --- | --- |
| Training stack | Environment、PPO、privileged-teacher 與 distillation path 已實作並有 smoke evidence。 | 收斂後 policy quality、cross-seed stability，或優於其他 controller。 |
| Robustness | Terrain、randomization、latency、noise、push 與 per-leg fault control 已存在。 | Reviewed run 關閉上述控制時所提出的 robustness claim。 |
| Sim-to-real | Profile、trace provenance、replay、comparison、audit 與 held-out promotion boundary 已存在。 | 超出已量測 state、command、contact、thermal condition 與 hardware mapping 的 fidelity。 |
| Energy | Mechanical-energy 與 spring diagnostic 已存在。 | 實測 battery-energy 節省或 electrical cost-of-transport 改善。 |
| Deployment | Observation/action dimension、ONNX export、ROS preflight 與 safety contract 已存在。 | Estimator、sensor validity、latency、filter 或 hardware feedback 與 training 不同時的完整等價性。 |

<a id="training-run-observation"></a>
## Training run 觀察

Reviewed run `2026-08-13_11-05-09_wheg_locomotion_reform_v1` 使用單一 seed、stage 1 與 plane terrain。其 resolved configuration 關閉 domain randomization、mass/friction randomization、actuator fault、observation latency 與 noise、push 和 terrain curriculum；`spring_calibrated` 為 false。

Generated report 的圖表 snapshot 截止於 iteration 9,545；其中 `Policy/mean_noise_std` 從 `0.600279` 升至 `10.229902`，RSL-RL 設定將 action clip 於 `1.0`。後續直接讀取時，找到至 iteration 9,999 的 10,000 筆 scalar record、最終值 `10.695706`、`model_9999.pt`，以及 exported Torch 與 ONNX policy。這修正了報告對 run 是否完成的不確定性，但不證明 policy quality。

Iteration 9,545 的 materialized snapshot 記錄 mean commanded forward velocity 為 `0.339982 m/s`，mean actual forward velocity 為 `0.550391 m/s`。這是應在 held-out sweep 檢查 command bias 的訊號，不是 reward shaping 導致 overspeed 的證明。儲存的 `velocity_error` channel 有其自身 aggregation semantics，不可視為上述兩個 mean 的算術差。

探索尺度增加同樣是診斷訊號，不是失敗定論。只有搭配 action-saturation histogram、KL divergence、entropy、log-standard-deviation behavior 與 held-out evaluation 才能解讀。

<a id="evidence-gates"></a>
## 證據關卡

| 關卡 | 目前風險或未知 | 結案證據 |
| --- | --- | --- |
| Physical-model truth | Mass、center of mass、link inertia、joint stop、friction、backlash、spring 與 damping 尚未全數綁定 accepted measurement。 | Versioned measurement，以及預定 operating envelope 的 held-out sim-to-real error threshold。 |
| Contact truth | Simulator contact force 與 phase proxy 尚未用真實 contact label 驗證。 | 同步 FSR、current 或 foot-contact label，加上 timing、precision 與 recall 證據。 |
| Reward-preset contract | Panel 支援 run-scoped override，但報告指出 editable preset 與 active reward schema 可能 drift。 | Contract test 證明每個 preset 依預期改變 resolved active reward field 與 saved run configuration。 |
| Command objective | Run 顯示 forward command bias，但因果未驗證。 | 有界的 A/B shaping ablation，以及回報整個 command envelope bias 與 RMSE 的 held-out command sweep。 |
| Exploration scale | Action 被 clip 時 policy standard deviation 上升；報告缺少 saturation 與 policy-update diagnostic。 | Saturation、KL、entropy 與 log-standard-deviation telemetry，加上明確 stop criteria 與穩定 held-out result。 |
| Train/deploy observation | Deployed stack 可能用 estimator 或 default value 替代 simulation 可用訊號。 | Torch-to-ONNX-to-ROS replay parity，以及 estimator、latency、filter、dropout/mask 與 hardware sensor-contract 證據。 |
| Evaluation method identity | 報告指出 evaluation compatibility behavior 可能改變 effective controller 的風險。 | Fail-closed evaluation configuration，或明確定義 controller-plus-policy method 並執行 compatibility on/off ablation。 |
| Energy provenance | Mechanical power、spring term、electrical measurement 與 proxy 可能混淆。 | Per-channel provenance label，以及在 matched command 與 achieved speed 下量測 electrical cost of transport：`integral(VI dt) / (mgd)`。 |

<a id="minimum-evidence-contract"></a>
## 最低證據契約

- 比較前凍結 task、command envelope、metric、resolved configuration、checkpoint、code revision、dependency 與 hardware revision。
- 使用低成本 funnel：unit/contract check、bounded simulator smoke、one-seed screening、至少三個 independent seed 進行探索，confirmatory result 最好使用五個 independent seed。
- 儲存 per-episode row。將 training seed 與 hardware trial 視為 experimental unit；不可把高度相關的 environment-time sample 當成獨立結果。
- 除 energy 外，同時回報 tracking error、success、fall、recovery、distance、temperature、peak current、contact accuracy、latency 與 sim-to-real error。保留 failure、backtracking 與 zero-distance episode，不可刪除。
- Hardware comparison 應 randomized trial order，並在相同 command 與 condition 比較 controller-only、residual-policy 與相關 direct-policy baseline。機構允許時，使用 spring enabled、locked、bypassed 或 swapped 的 paired trial。
- Continuous outcome 使用跨 seed 與 episode 的 hierarchical resampling；success proportion 使用 interval estimate。任何結果均應發布 protocol、resolved configuration、checkpoint、calibration evidence、per-episode data 與 representative video。

<a id="research-direction"></a>
## 研究方向

最有力的近期 hypothesis，是在 matched locomotion performance 下研究 passive-compliance energy effect 的因果關係，並結合 sensor 或 leg fault 下的 contact-aware residual control。主要 energy endpoint 必須是實測 electrical energy，而非 simulator torque proxy。這是 proposed direction，不是已接受的新穎性主張。

Contact belief、history-based state estimation、帶 validity mask 的 sensor dropout、concurrent teacher/student training、symmetry-aware policy、residual dynamics、cross-simulator prediction、off-policy RL 與 world model 仍是 candidate method。只有在能關閉具名 evidence gate 或提供 preregistered comparison 時才推進；若只增加 method complexity 則延後。

<a id="actions"></a>
## 行動

- [ ] 下一次 publication-scale run 前，關閉 reward configuration、command bias、exploration telemetry、observation parity、evaluation identity 與 energy label 的 correctness gate。
- [ ] 以 accepted calibration evidence 建立 physical-model、contact 與 electrical measurement truth。
- [ ] 凍結 baseline 與 held-out suite，再執行 multi-seed 與 hardware protocol。
- [ ] Evidence 能辨識可辯護 effect 後才選定 paper claim；宣稱 novelty 前完成專門的 literature 與 prior-art review。

有順序的工作維護於[目前專案路線圖](../roadmap/current-priorities.zh-TW.md)。

<a id="evidence"></a>
## 證據

- [訓練與 Policy 架構](../developers/architecture/training-and-policy.zh-TW.md)
- [Sim-to-real 校準架構](../developers/architecture/sim-to-real.zh-TW.md)
- [Reward 與能量模型](../developers/architecture/reward-and-energy.zh-TW.md)
- [Policy 與部署 Contract](../reference/policy-contract.zh-TW.md)
- [開發驗證層級](../developers/testing/validation.zh-TW.md)
- [2026-07-09 專案稽核](2026-07-09-project-audit.zh-TW.md)

<a id="limitations"></a>
## 限制

本稽核無法直接取得秤重後的機器人、link-level inertial measurement、calibrated motor bench、同步 contact label、voltage/current/temperature trace 或重複 hardware trajectory。它評估單一 deterministic stage-1 seed，而非 multi-condition result。報告的 literature scan 並非 systematic review，因此所有 proposed paper direction 都需要重新進行 primary-source 與 prior-art review。

<a id="provenance"></a>
## 來源

Raw generated bundle 保存在 canonical documentation 之外的 local recovery commit `02ebb53cf9da8db47952d3cf264801f44f27d82c`。PDF SHA-256 為 `91ac09d053d3859fd1dadc9b0c73d31e3d0afdd8febb2aa9ba6b93c8420b6dca`。其圖表與 SQL snapshot 截止於 iteration 9,545；上文的完成狀態修正來自稍後直接讀取相同 TensorBoard event 與 run directory。Generated HTML、preview、script、SQL snapshot 與 PDF 刻意不納入 canonical `main`。

<a id="follow-up"></a>
## 後續追蹤

Evidence gate 取得 durable closure evidence，或選定 proposed research claim 時，重新 review 本稽核。新的實驗結果應建立新的雙語 experiment summary；對此 published audit 的修正須使用 dated addendum。
