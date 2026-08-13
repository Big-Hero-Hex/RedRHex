---
id: student-distillation-v2-audit
title: Sensor-Only Student Distillation V2 實際程式路徑稽核
lang: zh-TW
audience: developer
type: audit
status: published
owner: training
last_reviewed: 2026-08-14
---

<a id="scope"></a>
## 範圍

本稽核檢視限制新增 sensor-only teacher–student 路線的 RedRHex observation、action、RSL-RL、ONNX、ROS 2、encoder bridge 與 sim-to-real 路徑。它只建立相容性事實，不宣稱已有 trained-policy、recorded-hardware 或實體機器人成果。

<a id="method"></a>
## 方法

檢視沿可執行路徑，從 Gym registration 追到 environment observation construction、runner selection、checkpoint loading、export、ROS inference、sensor ingestion 與 calibration/replay tooling。Repository 將行為委派給 upstream 時，也檢查已安裝的 RSL-RL 3.1.2 source。只有目前程式、設定或明確的缺項檢查能支持的內容才列為 findings。

<a id="findings"></a>
## 發現

- V1 是 56-D current frame 加四個 prior frames（280-D actor input）。內容包含 simulator base linear velocity、procedural gait clock 與 previous action；privileged group 也包含 internal drive 與 ABAD targets。這些 semantics 不能安全地改作 hardware sensor contract。
- 既有 distillation 路線確實存在，但只是 flat-MLP behavior-cloning path。Upstream RSL-RL 執行 student actions，並只最佳化單一 whole-action MSE/Huber loss；沒有 causal TCN、rollout mixture、next-frame target、auxiliary loss 或 PPO teacher-BC hook。
- Forward environment 已將六個 main outputs 解讀為 procedural gait 上的 residual，且會抑制 forward ABAD outputs。此 decoder 可在本次試驗中版本化；它的 phase 必須留在內部，不能成為 actor input。
- Simulator contact sensor 目前停用。現有「contact」state 由 encoder phase 推導，因此 contact supervision 與 export 必須維持停用。
- ROS V1 只接受單一 56- 或 280-D ONNX input、以零補 base velocity、可使用 commanded ABAD state、對不完整 history 補零，且 policy execution 開始後才填 history。V2 禁止這些行為，但為了 V1 rollback 必須原樣保留。
- Low-level bridge 已收到六個 raw ABAD encoders，卻尚未發布校正後的 ABAD joint feedback。Counts-per-radian 與部分 encoder zeros 仍是暫定值。
- Repository 沒有 production IMU publisher，也沒有 recorded evidence 能證明 quaternion covariance、frame identity、mount calibration 與 rest-gravity behavior。因此 hardware V2 目前沒有核准的 attitude mode。
- 既有 Torch/ONNX parity 從已組裝好的 observation 才開始，而 real-trace import 缺少 ABAD 與 `cmd_vel`。因此需要 shared causal preprocessor 與 raw-event replay gate。
- 初始稽核發現 Training Panel configuration 與 history discovery 綁定 V1 experiment roots，也沒有 V2 runner selector。Browser implementation 已復原，但被隔離在堆疊的 Panel physics/calibration proposal branch，等待分開審查；此 core branch 不宣稱該缺口已合併。

<a id="actions"></a>
## 行動

- [x] 保留 V1 registrations、loaders、preprocessing、ROS YAML 與 panel routes。
- [x] 核准新增 `Template-Redrhex-ForwardSensorV2-Direct-v0` research route，observation 與 action contract 各自獨立 hash。
- [ ] 審查並合併復原出的 kind-checked Panel stage transitions、sequential checkpoint handoff 與 final-F3 history routing，同時保留 standard training。
- [ ] 通過 deterministic forward baseline，以及各三個 seeds 的 Teacher A、distilled-student 與 PPO gates。
- [ ] 以經審查的 recorded evidence 證明十二個 encoder calibrations 與一個明確 IMU attitude mode。
- [ ] 任何 deployment promotion 前，完成 raw-event replay、shared-preprocessor parity、ONNX Runtime parity 與 synthetic ROS safety tests。

<a id="evidence"></a>
## 證據

主要程式證據位於 [legacy environment](../../source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py)、[environment configuration](../../source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py)、[V2 training backends](../../source/RedRhex/RedRhex/tasks/direct/redrhex/agents/sensor_v2/backends.py)、[training entry point](../../scripts/rsl_rl/train.py)、[full pipeline](../../scripts/rsl_rl/train_sensor_v2_pipeline.py)、[ROS observation builder](../../ros2_ws/src/redrhex_rl_controller/redrhex_rl_controller/observation_builder.py)、[ONNX runner](../../ros2_ws/src/redrhex_rl_controller/redrhex_rl_controller/policy_onnx_runner.py)、[Rinbo bridge](../../ros2_ws/src/redrhex_lowlevel_bridge/redrhex_lowlevel_bridge/rinbo_ros_backend.py) 與 [real-trace importer](../../tools/sim2real/import_real.py)。復原出的 Panel paths 是堆疊 proposal 的證據，不屬於此 core branch。

<a id="follow-up"></a>
## 後續

F5 promotion evidence 完成，或任何 observation feature、attitude mode、action decoder、calibration profile、runner role 或 checkpoint transition 改變時，必須重新稽核。目前已有 one-update F1/F2/F3/pipeline Isaac smoke；production-length、multi-seed、recorded-sensor 與 physical tests 仍明確為 pending，不可從 smoke 或 unit tests 推定通過。
