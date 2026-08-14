---
id: windows-remote-launcher-architecture
title: Windows Remote Launcher Architecture and Verification
lang: en
audience: developer
type: explanation
status: draft
owner: panel
last_reviewed: 2026-08-14
---

<a id="boundary"></a>
## Component boundary

`redrhex_remote.ps1` is one PowerShell 5.1-compatible, self-installing launcher. It owns deterministic configuration, installation paths, shortcut creation, endpoint probes, SSH arguments, the visible tunnel window, readiness polling, browser launch, and the panel's Windows remote-mode marker. It does not change Tailscale, the SSH server, Isaac Lab, or training artifacts.

<a id="connection"></a>
## Connection contract

The launcher forwards local `8080` and `6006` to workstation loopback through `lab_user1@100.90.246.97`. The remote command is UTF-8 encoded as Base64 before transport, decoded by the workstation shell, and starts the panel in `/home/lab_user1/Py/RedRHex` with Conda environment `env_isaaclab_bin`.

The remote process reuses a responding panel, otherwise starts `redrhex_panel` in detached tmux or falls back to `nohup` and `logs/training_panel/remote_panel.log`. It remains attached through SSH so closing the visible Windows terminal closes the forwards. No credential is embedded or persisted.

The launcher opens `/?remote_client=windows`. This is a capability marker, not an authorization boundary. The panel uses it to force headless training, route **TensorBoard** to one all-runs process on the fixed `6006` forward, and disable actions whose file-manager or live viewer windows can only open on the workstation. Recorded media, console output, copy controls, and other HTTP/API actions remain available.

<a id="installation"></a>
## Installation contract

`-Install` copies the current script to `%LOCALAPPDATA%\RedRHex Remote\redrhex_remote.ps1` and creates the current user's desktop shortcut through `WScript.Shell`. The shortcut invokes `powershell.exe` with `-NoProfile` and the installed file. No administrator-owned location is modified.

<a id="verification"></a>
## Verification contract

`tests/test_redrhex_remote.ps1` checks the remote-aware panel URL, SSH option/forward ordering, one quote-safe remote command argument, deterministic install paths, and the visible tunnel command. Training Panel browser tests verify both desktop markers, fixed-port TensorBoard routing, and disabled host-only controls. Run the launcher test on Windows PowerShell 5.1 or newer:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools\windows\tests\test_redrhex_remote.ps1
```

Static source preservation is verified in Git, but release still requires the [active Windows smoke checklist](../../../docs/plans/active/2026-08-01-windows-remote-launcher.en.md#windows-smoke). The design stays `approved` and these component documents stay `draft` until that evidence exists.
