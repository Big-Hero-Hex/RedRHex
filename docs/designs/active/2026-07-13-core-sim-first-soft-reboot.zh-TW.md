---
id: core-sim-first-reboot-design
title: Core-First、Simulation-First Soft Reboot
lang: zh-TW
audience: developer
type: design
status: proposed
owner: core
last_reviewed: 2026-08-07
---

<a id="provenance"></a>
## Provenance 與狀態

此 proposal 從 `reboot/core-sim-first` branch 的 `f40d3c2` 匯入獨特 durable architecture。它不是 active implementation。來源 branch 保留其詳細 plan 與 evidence draft；本 pair 只記錄目前專案決策所需提案。

<a id="problem"></a>
## 問題

目前 Isaac environment 同時包含 simulator I/O、observation/reward math、gait/command state、randomization、buffer 與 logging。Pure behavior 很難在沒有 Isaac 的情況下 import，而 contract fact 鏡像到 ROS 後可能 drift。Hard rewrite 會在建立可信比較前丟棄可用的訓練、面板、reward-agent 與 deployment 行為。

<a id="proposal"></a>
## 提議架構

保留 `RedRhex` 作為 Isaac adapter，維持兩個 Gym ID、script interface、checkpoint 與 artifact layout。新增 sibling package：

- `redrhex_contract`：只依賴 standard library，負責 ordering、dimension、unit、rate、scale、slice 與 contract version。
- `redrhex_core`：只依賴 Torch，使用明確 input/output 實作 observation、reward、termination、action、gait、command、randomization 與 buffer logic。

只有 adapter 讀取 simulator state 並寫入 actuator target。抽離期間 panel、remote、Reward Agent 與 ROS 維持 frozen，並透過 regression test 驗證。

<a id="gates"></a>
## 提議 gate

1. 建立安全 test discovery、toolchain provenance、frozen-interface guard 與 artifact contract。
2. 驗證 gravity、unit、frame、mass/inertia、contact、timing、action response、reward、command 與 determinism。
3. 只有之後才能捕捉 validated legacy oracle 與 reference-training baseline。
4. 建立 package、抽離 contract fact、一次抽離一個 pure behavior seam，再縮薄 adapter。
5. 只有 CPU、golden、simulator、frozen-boundary 與 fixed-seed comparison evidence 通過才可 accept。

<a id="decision-points"></a>
## 必要決策

Reboot 本身、pinned Isaac checkout、缺少的 physical fact、任何 physics/frame correction、immutable baseline protocol、intentional golden difference 與最終 acceptance 都仍需核准。設定中的 gravity vector 或視覺感受本身不是 bug 證據。

<a id="non-goals"></a>
## 非目標

提案排除抽離期間的 panel/remote/reward-agent/ROS redesign、新 task ID、artifact migration、reward research、hardware estimator、asset relocation、廣泛 cleanup 與 Isaac upgrade。這些都是 acceptance 後的獨立專案。
