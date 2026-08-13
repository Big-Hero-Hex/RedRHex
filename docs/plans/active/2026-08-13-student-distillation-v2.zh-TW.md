---
id: student-distillation-v2-plan
title: Sensor-Only Student Distillation V2 實作計畫
lang: zh-TW
audience: developer
type: plan
status: active
owner: training
last_reviewed: 2026-08-14
---

<a id="objective"></a>
## 目標

以 additive CLI functionality 交付已核准的 sensor-only forward route，依序證明每個 evidence gate，並完整保留所有 V1 artifact 與 rollback path。

<a id="context"></a>
## 背景

[已核准設計](../../designs/active/2026-08-13-student-distillation-v2.zh-TW.md)固定 36-D frame、60-frame history、separate command、residual-CPG action decoder、TCN、loss schedules、strict artifacts、two-input ONNX graph 與 ROS safety boundary。[程式路徑稽核](../../research/2026-08-13-student-distillation-v2-audit.zh-TW.md)阻擋 contact supervision 與 hardware promotion。Core CLI route 在此重建；復原出的 browser route 被隔離到堆疊的 Panel physics/calibration proposal branch，以便分開審查。勾選項目只表示程式或 local test 已存在，不代表後續 full-run、multi-seed、recorded-sensor 或 physical gate 已通過。

<a id="phased-checklist"></a>
## 分階段清單

<a id="phase-compatibility"></a>
### 相容性與 contracts

- [x] 發布 bilingual audit、approved design、active plan、migration/rollback 與 documentation-impact boundaries。
- [ ] 以 observation、registration、checkpoint-loader、ROS 與 preprocessing regression tests 凍結 V1。
- [ ] 完成 `redrhex_policy_io`、canonical hashes、calibration V2、NumPy/Torch parity 與 golden fixtures。

<a id="phase-f0"></a>
### F0 deterministic baseline

- [ ] 註冊 isolated V2 task、36-D causal event pipeline、observation groups 與 residual action contract。
- [ ] 驗證 zero residual、neutral ABAD、leg/tripod/sign/phase/timing mapping 與 sim/ROS decoder trace parity。
- [ ] 只透過 half/base/double sensitivity 選取 nonzero regularizers，並通過既有 forward command sweep。

<a id="phase-f1"></a>
### F1 privileged teacher

- [x] 新增 Teacher A 65-D physical privilege 與明確隔離的 Teacher B target-state ablation。
- [x] 在 `redrhex_forward_v2_teacher` 產生 versioned Teacher A checkpoints。
- [ ] 三個獨立 Teacher A seeds 都通過既有 forward protocol 後才開始 distillation。

<a id="phase-f2"></a>
### F2 sensor-only distillation

- [x] 完成 TCN、normalizer、auxiliary heads、custom storage、rollout mixture、split losses、metrics 與 strict checkpoint transition。
- [ ] Export 並驗證 named two-input ONNX graph、embedded metadata 與相符 sidecar。
- [ ] 通過 CPU 與 Isaac update、finite-gradient、save/resume、ONNX Runtime parity 與 three-seed screening gates。

<a id="phase-f3"></a>
### F3 asymmetric PPO

- [x] Strictly copy distilled actor/normalizer、驗證 equality，並建立全新 physical critic 與 optimizer。
- [x] 加入 annealed Teacher A BC 與持續 velocity/dynamics losses，且 actor 沒有 privileged input。
- [ ] 通過 PPO update/save/resume 與 three-seed forward gates。

<a id="phase-panel"></a>
### Training Panel route

- [ ] 審查並合併堆疊 Panel proposal 中復原出的明確 F1 Teacher、F2 Distillation 與 F3 Student PPO browser routes。
- [ ] 審查並合併其 fail-closed 完整 F1 → F2 → F3 browser pipeline 與 final-F3 history routing。
- [ ] 驗證其保留 standard Panel route，且 Sensor V2 launch 不使用 mutable V1 reward/terrain overrides。
- [ ] 完成 production-length run 與 multi-seed quality gates；Panel 顯示完成不等於 promotion evidence。

<a id="phase-f4"></a>
### F4 calibration 與 replay

- [ ] Import raw ABAD encoder 與 `cmd_vel` channels，並將 `SensorCalibrationProfileV2` 綁定每個 artifact。
- [ ] 只 model 有 evidence 的 sensor/actuator ranges，並記錄每個 sampled range 與 provenance。
- [ ] 在 held-out recorded traces 執行 raw-event observation/ONNX replay，沒有 missing contract、NaN、invalid action 或 unexplained saturation。

<a id="phase-deployment"></a>
### V2 ROS 與 bridge path

- [ ] 發布十二個 calibrated measured joints，包含 per-channel time、validity、freshness 與 causal velocity。
- [ ] 新增 contract-selected V2 YAML、builder、named-I/O runner、preflight、60-frame warm-up 與 protective dropout reset。
- [ ] 以 `rtol=atol=1e-4` 通過 synthetic ROS 與 recorded offline inference parity；automated validation 不得 enable motors。

<a id="phase-f5"></a>
### F5 evaluation 與 promotion

- [ ] 在相同 seeds、commands 與 domains 比較 Teacher A、legacy、V2 distilled、V2 PPO 與 required auxiliary/PPO ablations。
- [ ] 發布 raw per-seed data、mean/std、teacher gaps、provenance、parity 與 blocked contact status。
- [ ] 只有三個 PPO seeds 全數通過、匹配 Teacher A accepted commands，並通過 leak、parity、replay 與 provenance gates 才 promotion。

<a id="verification"></a>
## 驗證

先執行 dependency-light unit suites，再跑 documentation validation 與 diff checks。可執行 F1、F2、F3 與完整 sequential pipeline 已在 2026-08-13 分別通過 one-environment、one-update Isaac smoke；這只證明 launch、finite update、strict checkpoint handoff 與 final-artifact routing。Production-length training、three-seed command sweeps、recorded-sensor replay 與 ROS offline inference 仍需要各自 named evidence。Hardware preflight 與 physical tests 維持 manual 與 disabled by default，直到 reviewed IMU mode 與十二個 encoder calibrations 完成前都為 blocked。

<a id="completion-summary"></a>
## 完成摘要

Implementation 目前 active。整個過程都必須保留 V1 與 configuration-only rollback。完成條件是 F5 evidence 加上 maintained bilingual component documentation；只有 code presence 不足以關閉 plan。
