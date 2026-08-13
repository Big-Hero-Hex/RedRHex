---
id: windows-remote-launcher-operator-guide
title: Install and Use RedRHex Remote on Windows
lang: en
audience: operator
type: how-to
status: draft
owner: panel
last_reviewed: 2026-08-13
---

<a id="prerequisites"></a>
## Prerequisites and status

Use Windows PowerShell 5.1 or newer with Windows OpenSSH Client and Tailscale connected. This launcher is a release candidate until the active Windows smoke checklist is completed. It stores no password, private key, or Tailscale credential.

<a id="install"></a>
## Install for the current user

Open Windows PowerShell and run:

```powershell
$download = Join-Path $env:TEMP "redrhex_remote.ps1"
scp lab_user1@100.90.246.97:/home/lab_user1/Py/RedRHex/tools/windows/redrhex_remote.ps1 $download
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $download -Install
```

Enter the Ubuntu password only in the `scp` or SSH prompt. Installation requires no administrator access. It copies the script to `%LOCALAPPDATA%\RedRHex Remote\` and creates `RedRHex Remote.lnk` on the current user's desktop.

<a id="use"></a>
## Connect

1. Confirm that Tailscale is connected.
2. Double-click `RedRHex Remote`.
3. If prompted in the visible `RedRHex SSH Tunnel` window, verify the host and enter the Ubuntu password.
4. Keep that window open while using the forwarded services.

The launcher reuses a responsive tunnel. Otherwise it connects to `lab_user1@100.90.246.97`, forwards local ports `8080` and `6006`, and starts the workstation panel in the `env_isaaclab_bin` environment. It opens the Training Panel at `http://localhost:8080`. It opens TensorBoard at `http://localhost:6006` only when TensorBoard is already responding.

<a id="stop"></a>
## Stop and remove

Close the `RedRHex SSH Tunnel` window to stop both forwards. To remove the per-user installation:

```powershell
Remove-Item (Join-Path ([Environment]::GetFolderPath("Desktop")) "RedRHex Remote.lnk") -Force
Remove-Item (Join-Path $env:LOCALAPPDATA "RedRHex Remote") -Recurse -Force
```

These commands remove only the shortcut and installed launcher copy.

<a id="troubleshoot"></a>
## Troubleshoot

- Missing `ssh.exe`: install OpenSSH Client from Windows Optional Features.
- Authentication or reachability failure: confirm Tailscale, the workstation address, host-key prompt, and Ubuntu password in the visible tunnel window.
- Local bind failure: stop the process already using port `8080` or `6006`, then retry.
- Panel readiness timeout: inspect the SSH window and workstation `tmux` session `redrhex_panel`; without tmux, inspect `logs/training_panel/remote_panel.log`.
- TensorBoard does not open: start it from the Training Panel, then open `http://localhost:6006`.
