---
id: training-stack-evidence-2026-07
title: 2026-07 訓練 Stack 證據摘要
lang: zh-TW
audience: developer
type: experiment-summary
status: published
owner: training
last_reviewed: 2026-08-07
---

<a id="question"></a>
## 問題

在長時間 performance 實驗前，重構後的 RedRHex training stack 是否能執行預期 environment、PPO、privileged-teacher 與 distillation 路徑？

<a id="method"></a>
## 方法

歷史 smoke run 使用 `validate_reform_stack.py`，搭配少量 environment 與短 rollout。不同模式分別執行 random environment stepping、一次 PPO update，以及 teacher checkpoint 後的一次 distillation update。來源報告記錄暫存 JSON 輸出與 checkpoint 是否存在。

<a id="results"></a>
## 結果

Environment smoke 回報 generated terrain、224-value history group、47-value critic group、327-value teacher group，以及設定樣本中啟用的 fault injection。PPO smoke 完成一次 update。Teacher/distillation smoke 產生 teacher 與 student checkpoint，並完成 distillation update。

<a id="interpretation"></a>
## 解讀

這些結果建立 wiring 與 executable-path 證據：observation group、PPO runner、privileged teacher 與 distillation runner 可在測試環境一起運作。它們不證明最終 tracking quality、energy improvement、robustness、hardware transfer 或優於 MPC。

<a id="provenance"></a>
## Provenance 與修正政策

數值由 documentation source checkpoint 的 `docs/2026_Midterm.md` 遷移。Raw `/tmp` artifact 未提交，因此目前不是可獨立重跑的證據。修正此 immutable summary 需要日期式 addendum；新結果必須建立新 experiment summary。
