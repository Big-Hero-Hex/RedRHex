---
id: operator-getting-started
title: 開始使用 RedRHex
lang: zh-TW
audience: operator
type: tutorial
status: active
owner: training
last_reviewed: 2026-08-16
---

<a id="prerequisites"></a>
## 先決條件

請使用具備可用 NVIDIA 驅動程式、Isaac Lab、Git LFS，以及 Python 3.10 以上版本的 Ubuntu。擴充套件宣告相容 Isaac Sim 4.5、5.0 與 5.1；目前的 Isaac Lab checkout 必須與其 Python 環境一致。

先在目前 shell 設定 launcher 路徑：

```bash
export ISAACLAB_ROOT=/path/to/IsaacLab
```

<a id="install"></a>
## 安裝專案擴充套件

在儲存庫根目錄執行：

```bash
git lfs install
git lfs pull
"$ISAACLAB_ROOT/isaaclab.sh" -p scripts/install_redrhex.py
```

Installer 會使用目前的 Isaac Python，先安裝儲存庫內的 policy-I/O distribution，再安裝 RedRHex 擴充套件；`redrhex-policy-io` 不會從 package index 取得。可加上 `--dry-run` 檢查兩條依序執行的 pip 指令。USD 資產由 Git LFS 管理。若 `RedRhex.usd` 只是很小的指標檔，表示 `git lfs pull` 尚未完成。

<a id="verify"></a>
## 驗證環境

```bash
"$ISAACLAB_ROOT/isaaclab.sh" -p scripts/list_envs.py
```

確認清單包含 `Template-Redrhex-Direct-v0` 與 `Template-Redrhex-ForwardFast-Direct-v0`，再執行一次性 smoke pipeline：

```bash
python -m tools.training_panel.smoke_pipeline
```

它會啟動一次 iteration，並檢查 checkpoint、TensorBoard event、儲存參數與面板歷史紀錄。若只想先查看產生的 Isaac 指令，請加上 `--dry-run`。

<a id="next"></a>
## 下一步

- [啟動訓練](training/launch-training.zh-TW.md)
- [使用分階段訓練](training/staged-training.zh-TW.md)
- [操作 Training Panel](panel/training-panel.zh-TW.md)
- [排除設定或訓練問題](troubleshooting/training-troubleshooting.zh-TW.md)
