---
id: student-distillation-v2-plan
title: Sensor-Only Student Distillation V2 實作計畫
lang: zh-TW
audience: developer
type: plan
status: active
owner: training
last_reviewed: 2026-08-17
---

<a id="objective"></a>
## 目標

以 additive、明確選取的 CLI 與 ROS functionality 交付已核准的 sensor-only forward route，依序證明每個 evidence gate，並完整保留所有 V1 artifact 與 rollback path。

<a id="context"></a>
## 背景

[已核准設計](../../designs/active/2026-08-13-student-distillation-v2.zh-TW.md)固定 36-D frame、60-frame history、separate command、residual procedural action decoder、TCN、loss schedules、strict artifacts、two-input ONNX graph 與 ROS safety boundary。更新後的[程式路徑稽核](../../research/2026-08-13-student-distillation-v2-audit.zh-TW.md)確認 additive training、replay、export 與 ROS routes 已存在，而 contact supervision 與 hardware promotion 仍為 blocked。

勾選項目表示 implementation 與其 scoped proof 已存在，不代表已有 production-length 或 multi-seed 結果、recorded-sensor replay、calibrated hardware 或 physical gate 通過。目前以 hash 綁定的 schema-v2 structural 與 seed 42 Isaac F0 結果都為 `PASS`；F1-F5 與 hardware evidence 仍為 `NOT_RUN`。

<a id="phased-checklist"></a>
## 分階段清單

<a id="phase-compatibility"></a>
### 相容性與 contracts

- [x] 發布並維護 bilingual audit、approved design、active plan、rollback boundary 與 documentation-impact declaration。
- [x] 保留明確的 V1 task/runner、observation、checkpoint、ROS configuration、launch 與 preprocessing paths，不使用 V2 auto-detection。
- [x] 完成可安裝的 `redrhex_policy_io` package、canonical observation/action/calibration hashes、NumPy/Torch seams、strict history 與 dependency-light fixtures。

<a id="phase-f0"></a>
### F0 deterministic baseline

- [x] 註冊 isolated V2 task、36-D causal event pipeline、observation groups 與 forward residual action contract。
- [x] 以受支援、全部 effective reset 為 `-π/4` 的 semantics 取代過時的 schema-v1 π-reset 解讀；π tripod offset 保留在已還原的 65/35 time-warped CPG reference。
- [x] 產生 structural、simulator 與每一個 per-command row 都通過的 immutable schema-v2 evidence。目前 report SHA-256 為 `2e108004c75e74e2e5df08d29ed8aac28b67f7cf8e5cc410135cd36975a70132`。

<a id="phase-f1"></a>
### F1 privileged teacher

- [x] 實作 Teacher A 的 65-D physical privilege 與明確隔離的 77-D Teacher B target-state ablation。
- [x] 實作 versioned Teacher A checkpoint manifests 與 `redrhex_forward_v2_teacher` experiment route。
- [ ] 產生 current-revision Teacher A checkpoints，並在開始 distillation promotion 前通過三個獨立 seeds 的必要 forward protocol。

<a id="phase-f2"></a>
### F2 sensor-only distillation

- [x] 完成 causal TCN、in-model normalizers、auxiliary heads、custom storage、rollout mixture、split losses、metrics 與 strict teacher-to-student transition。
- [x] 實作 named fixed-shape two-input ONNX exporter、embedded metadata、hash-bound sidecar 與 fail-closed Torch/ONNX Runtime parity gate。
- [ ] 產生目前的 F2 screening artifact，並通過 CPU/Isaac update、finite-gradient、save/resume、ONNX Runtime parity 與 three-seed screening gates；promotion 仍保留給 exact F4 artifact。

<a id="phase-f3"></a>
### F3 asymmetric PPO

- [x] Strictly copy distilled actor/normalizer、驗證 equality，並建立全新的 physical critic 與 optimizer。
- [x] 加入 annealed Teacher A behavior cloning 與持續的 velocity/dynamics losses，且 actor 沒有 privileged input。
- [ ] 通過 current-revision PPO update/save/resume、command sweep 與 three-seed forward gates。

<a id="phase-panel"></a>
### Training Panel route

- [x] 新增明確的 F1 Teacher、F2 Distillation、F3 Student PPO、evidence-gated 完整 F0-F5，以及標示為不可 promotion 的 `sensor_v2_ungated_debug` browser routes；拒絕新的 `sensor_v2_f1_f3` launches，同時將其 historical runs 保留為 read-only、衍生 noneligible markers 的 recovery records。
- [x] 在 `tools/training_panel/training_panel/processes.py` 新增 strict source-checkpoint requirements、full-pipeline result/provenance validation 與 final-F4 log/history routing，同時保留 standard route。
- [x] Sensor V2 launches 維持明確選取，並防止 mutable standard V1 reward/terrain overrides 被隱含套用。
- [ ] 完成目前的 production-length 與 multi-seed quality gates；Panel process 完成不等於 promotion evidence。

<a id="phase-f4"></a>
### F4 calibration、robustness 與 replay

- [x] 新增 `tools/sim2real/sensor_dr_profile_v2.py`，提供 exact profile SHA-256 binding、relative evidence-artifact resolution 與 hash verification，以及分離的 `training_curriculum` 與 `held_out_evaluation` purposes。
- [x] 新增 `SensorRobustnessRunnerV2`／`rsl_rl_robust_ppo_v2_cfg_entry_point`，以明確的 `--ppo_checkpoint` 建立 F3-to-F4 boundary，並包含 compatible contract checks 與 fresh optimizer。
- [x] 新增 contract-bound four-topic rosbag importer 與 synchronized observation/ONNX replay，包含完成校正的 IMU、main/ABAD encoder 與 command channels；real traces 必須具有另行提供的 hash-bound capture attestation 與 hardware-ready calibration，且系統不提供 override，會讀取 canonical controller YAML、綁定其 SHA，並透過 stateful ROS decoder 重新計算 raw、action-clipped、slew-limited 與 final targets。
- [ ] 從 measured evidence 產生經審查的 non-neutral training 與 held-out profiles、執行 F4 robustness training，並通過 recorded replay，不得出現 missing contract、NaN、invalid action 或 unexplained saturation，且 element-wise total target-divergence fraction 必須為 `0`、maximum delta 也必須為 `0`。

<a id="phase-deployment"></a>
### V2 ROS 與 bridge path

- [x] 新增 V2 bridge overlay 與 measured twelve-joint validity/freshness diagnostics，且不改變 V1 bridge configuration。
- [x] 新增 dedicated V2 node、YAML、launch、builder、named-I/O runner、preflight、source-time validation、complete-generation source-skew 與 60 Hz per-channel cadence gates、60-real-frame warm-up，以及 protective history/baseline reset。
- [x] Policy/motor startup 維持 disabled，且 motor authorization 同時要求 `hardware_gate.allow_motor_enable`、bundle calibration hardware readiness，以及 configured-to-bundle action envelope 完全相等。Simulator/bundle/PhysX ceiling 是 `15.0` rad/s；checked-in YAML 包含沒有 evidence 的 `9.0` rad/s limit 與 `120.0` rad/s² slew rate。Velocity mismatch 會靜態阻擋 authorization，runtime 中任何會改變 target 的 action clip、slew 或 velocity tightening 則會將 authorization latch off 並進入 protective stop。
- [ ] 以 `rtol=atol=1e-4` 通過 synthetic ROS inference parity，並以 exact zero divergence 通過 recorded stateful action-target parity；replay 沒有 override，且 automated validation 不得 enable motors。

<a id="phase-f5"></a>
### F5 evaluation 與 promotion

- [x] 新增 `scripts/rsl_rl/train_sensor_v2_full_pipeline.py` 作為 fail-closed three-seed F0-F5 route，要求 immutable Isaac F0 evidence，並拒絕 F4/F5 的 profile hash、`profile_id` 或 evidence-artifact hash 重疊。
- [ ] 在相同 seeds、commands 與 held-out domains 比較 Teacher A、legacy、V2 distilled、V2 PPO、V2 robustness 與必要 auxiliary/PPO ablations。
- [ ] 發布 raw per-seed data、mean/std、teacher gaps、provenance、parity、recorded replay 與 blocked contact status。
- [ ] 只有在 embedded ONNX metadata、sidecar metadata、embedded checkpoint manifest 與 sidecar checkpoint manifest 都同意 stage，且 sensor-replay ONNX 與 sidecar SHA-256 和 canonical `torch_onnx_parity` 已驗證來源達到 byte-identical 時，才可 promotion exact `ppo_f4` checkpoint；此外所有必要 seeds 都必須通過、匹配 Teacher A accepted commands，並通過 F0、leak、parity、replay、provenance、calibration 與 safety gates，且 `ppo_f3` 或 distinct rehashed replay graph 都不得替代。

<a id="verification"></a>
## 驗證

先執行 dependency-light contract、action-decoder、runner、exporter、replay、ROS wiring/packaging 與 documentation suites。2026-08-17 schema-v2 structural F0 report 已通過 same-phase reset、65/35 time warp、0.9 Hz gait、60 Hz timing、exact `15.0` rad/s simulator/bundle/PhysX binding 與 shared-decoder checks。Seed 42、八個 environments 的 native-spring rollout 也通過 `0.22`、`0.35` 與 `0.42` m/s 全部 rows，falls 為零、contiguous-success ratio 為 `1.0`；velocity 使用 121/76/67 samples 的精確 command-scaled 完整 cycle windows，tilt、height 與 episode boundaries 在未變更的 evaluator thresholds 下保持 pointwise。此 F0 run 並未啟動 F1。Full promotion route 仍要求至少三個 unique seeds、exact F0/profile hashes、具有獨立 evidence 的 F5 domain、exact `ppo_f4` provenance，以及 action-target divergence 為零且綁定 canonical YAML 的 real replay。本次沒有產生 F1/F2/F3/F4/F5 結果、promoted ONNX bundle、recorded real replay、ROS offline parity report、hardware preflight PASS 或 physical test。

Checked-in ROS configuration 與 bridge overlay 維持 disabled by default。其沒有 evidence 的 YAML 包含 `9.0` rad/s velocity limit 與 `120.0` rad/s² slew rate；velocity limit 與 `15.0` rad/s bundle envelope 不一致，因此 static preflight 會阻擋 motor authorization，而任何會改變 target 的 runtime tightening 都會觸發 latch-off 與 protective stop。Hardware testing 必須另行授權，而且在 selected IMU mode、全部十二個 encoder calibrations、exact hashes、hardware-ready bundle、exact action-envelope 與 offline-replay parity，以及 safety preflight 都有 reviewed evidence 前維持 blocked。

<a id="documentation-impact"></a>
## 文件影響

本次變更原地更新既有 active bilingual plan 與配對的 research audit，沒有新增 document path、navigation entry、redirect 或 migration stub。Approved design 的 fixed 60 Hz timestamp/history contract 與 fail-closed safety boundary 已要求新的 runtime enforcement，因此設計維持不變；implemented-route 與 evidence status 在此同步。英文與繁體中文的 anchors、metadata、checkboxes 與 semantics 保持成對一致。

<a id="completion-summary"></a>
## 完成摘要

Implementation 仍為 active。可執行 V2 route 與 structural-plus-Isaac F0 gate 現在已通過，但 empirical F1-F5、replay、calibration 和 hardware gates 仍未關閉。整個過程都必須保留 V1 與 configuration-only rollback。只有其餘 gates 都具有 immutable evidence，包含獨立 F5 evidence，並維護 bilingual documentation，才能關閉本計畫；F0 或 code presence 本身都不足以完成。
