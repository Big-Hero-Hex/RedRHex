---
id: subsystem-ownership
title: 子系統責任與文件
lang: zh-TW
audience: developer
type: reference
status: active
owner: project
last_reviewed: 2026-08-07
---

<a id="ownership"></a>
## 責任對照

| 子系統 | Owner | 詳細規範位置 |
| --- | --- | --- |
| Isaac task 與核心行為 | `core` | `docs/developers/architecture/` 與原始碼旁文件 |
| 訓練 script 與 evaluation | `training` | 操作訓練/評估與開發架構文件 |
| Training Panel 與 remote UI | `panel` | `tools/training_panel/docs/` |
| Reward Agent | `reward-agent` | `tools/reward_agent/docs/` |
| Sim-to-real 校準 | `sim2real` | 操作校準、開發架構與 `tools/sim2real/` |
| ROS2 policy 部署 | `deployment` | `ros2_ws/src/redrhex_rl_controller/docs/` |
| 文件系統 | `project` | `docs/governance/` |

<a id="change-routing"></a>
## 變更路由

元件專用行為放在程式旁維護，並從中央入口連結。跨專案 contract、decision、roadmap、release 與 governance 仍放中央。元件 README 是雙語 router，不得重複受維護程序。

<a id="release-routing"></a>
## Release 路由

Training Panel 獨立版本化，並擁有自己的 component release note。整個 repository 使用日期 milestone，不虛構全域 SemVer。ROS、reward-agent 或 sim-to-real 若未來建立獨立版本線，才使用 component release record；目前則列入日期式專案 milestone。

<a id="review-routing"></a>
## Review 路由

Metadata `owner` 表示技術審查責任。若變更跨越 contract 邊界，仍不得省略雙語 parity、documentation-impact 規則，或其他受影響子系統的 review。
