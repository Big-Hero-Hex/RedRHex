---
id: locomotion-literature
title: 移動研究基礎
lang: zh-TW
audience: developer
type: explanation
status: active
owner: training
last_reviewed: 2026-08-07
---

<a id="purpose"></a>
## 目的

此精選摘要記錄促成 RedRHex 介面設計的研究概念。它不宣稱已重現引用方法，也不宣稱四足結果可直接轉移到六足 wheg 機器人。

<a id="terrain-adaptation"></a>
## Terrain 與 adaptation

Robust perceptive locomotion 與 RMA 促成 terrain curriculum、history、訓練期 privileged information，以及可部署 observation boundary。相關來源包括 [Learning robust perceptive locomotion](https://arxiv.org/abs/2201.08117) 與 [Rapid Motor Adaptation](https://arxiv.org/abs/2107.04034)。

<a id="teacher-student"></a>
## Teacher 與 student 訓練

DreamWaQ 與 concurrent teacher-student 工作促成 privileged teacher observation 與 deployable student input 分離，並驗證 distillation checkpoint 路徑。請參考 [DreamWaQ](https://arxiv.org/abs/2301.10602) 與 [Concurrent Teacher-Student](https://arxiv.org/abs/2403.04359)。

<a id="terrain-skill"></a>
## Terrain 與技能進程

ANYmal parkour 與 Extreme Parkour 促成 staged 或 curriculum-driven skill acquisition；但 RedRHex 使用自身五階段 command 設計與 wheg dynamics。請參考 [ANYmal Parkour](https://www.science.org/doi/10.1126/scirobotics.adi7566) 與 [Extreme Parkour](https://arxiv.org/abs/2309.14341)。

<a id="symmetry-actuation"></a>
## Symmetry 與 actuation

Morphological symmetry 研究支持 augmentation，但前提是 robot mapping 與 phase semantics 真正 equivariant。RedRHex 目前停用不一致的 mirror transform。Actuator-network 工作則促成量測與建模 actuation，而非假設理想 target。請參考 [Morphological Symmetry](https://hybrid-robotics.berkeley.edu/publications/IROS2024_Symmetry_RL_LeggedLoco.pdf) 與 [Learning agile and dynamic motor skills](https://www.science.org/doi/10.1126/scirobotics.aau5872)。

<a id="evaluation"></a>
## 評估邊界

MPC-versus-RL 與 CPG-RL 工作促成受控 baseline 與 gait prior，不代表可以做無證據的優越性宣稱。RedRHex 必須比較相同 command envelope、success/fall criterion、energy proxy 與 hardware condition，才能宣稱方法差異。
