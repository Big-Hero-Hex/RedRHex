# RedRHex Remote for Windows

This launcher starts the Training Panel remotely on the RedRHex workstation, opens SSH forwards to it, and then opens the Training Panel and TensorBoard in the default Windows browser.

## Install

Open Windows PowerShell and run:

~~~powershell
$download = Join-Path $env:TEMP "redrhex_remote.ps1"
scp lab_user1@100.90.246.97:/home/lab_user1/Py/RedRHex/tools/windows/redrhex_remote.ps1 $download
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $download -Install
~~~

Enter the Ubuntu `lab_user1` password when `scp` requests it. The installer creates a `RedRHex Remote` shortcut on the current user's desktop and does not require administrator access.

## Use

1. Make sure Tailscale is connected on Windows.
2. Double-click `RedRHex Remote`.
3. If SSH asks, enter the Ubuntu `lab_user1` password.
4. Keep the `RedRHex SSH Tunnel` window open while using the panel or TensorBoard.

The launcher starts the remote panel with the repository's documented `env_isaaclab_bin` Conda environment. It uses a detached `tmux` session when available, or a background log at `logs/training_panel/remote_panel.log` otherwise. If the panel is already running, it reuses it.

TensorBoard is opened automatically only when it is already running. Otherwise, start it from the Training Panel when needed.

The launcher opens:

- Training Panel: <http://localhost:8080>
- TensorBoard: <http://localhost:6006>

If the tunnel is already running, the shortcut reuses it and only opens the browser pages.

## Remove

Run in Windows PowerShell:

~~~powershell
Remove-Item (Join-Path ([Environment]::GetFolderPath("Desktop")) "RedRHex Remote.lnk") -Force
Remove-Item (Join-Path $env:LOCALAPPDATA "RedRHex Remote") -Recurse -Force
~~~

Removal deletes only the per-user shortcut and installed launcher copy.
