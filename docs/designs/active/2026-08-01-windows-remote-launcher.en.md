---
id: windows-remote-launcher-design
title: Windows Remote Launcher Design
lang: en
audience: developer
type: design
status: approved
owner: panel
last_reviewed: 2026-08-14
---

<a id="goal"></a>
## Goal

Provide a Windows desktop shortcut that connects a laptop to the RedRHex workstation through Tailscale, forwards the local Training Panel and TensorBoard ports over SSH, waits for panel readiness, and opens a remote-aware panel that can start TensorBoard in the default browser.

<a id="scope"></a>
## Scope

One self-installing PowerShell 5.1+ script owns configuration, per-user installation, shortcut creation, SSH command construction, readiness polling, and browser launch. A narrow Training Panel UI contract recognizes the launcher marker, routes TensorBoard through the fixed forward, and disables host-only GUI actions. The launcher does not modify Isaac Lab, Tailscale, SSH server, or training code. The workstation remains the execution host and artifact source.

<a id="connection"></a>
## Connection contract

- SSH target: `lab_user1@100.90.246.97`
- Panel: local `8080` to workstation `127.0.0.1:8080`
- TensorBoard: local `6006` to workstation `127.0.0.1:6006`
- URLs: `http://localhost:8080/?remote_client=windows` and `http://localhost:6006`
- Install directory: `%LOCALAPPDATA%\RedRHex Remote\`
- Shortcut: current user's desktop, `RedRHex Remote.lnk`

The SSH terminal remains visible for host-key, password, bind, and connectivity errors. Closing it closes the tunnel. The launcher stores no password, private key, Tailscale credential, or other secret.

<a id="flow"></a>
## Launch flow

1. Require Windows OpenSSH `ssh.exe`.
2. Reuse an already-responsive panel tunnel instead of starting a duplicate.
3. Otherwise open a visible SSH process with both forwards, `ExitOnForwardFailure`, and keepalives.
4. Poll the panel for at most 45 seconds.
5. On success, open the marked panel URL. The History **TensorBoard** action starts or reuses one all-runs server on `6006`; host-only file-manager and live-viewer controls stay disabled.
6. On failure, explain Tailscale, authentication, SSH, and port-conflict causes.

<a id="verification"></a>
## Verification

Dependency-free PowerShell tests cover the marked panel URL, deterministic arguments, install paths, and tunnel command construction. Browser tests cover fixed-forward TensorBoard and disabled host-only controls. Windows smoke verification covers install/uninstall ownership, shortcut launch, authentication, both forwards, UI capability state, timeout behavior, closing the tunnel, and reusing an existing tunnel.

<a id="status"></a>
## Status boundary

This design remains approved. The preserved implementation and dependency-free test source are now committed on the feature branch, but the PowerShell 5.1 and end-to-end Windows smoke evidence is still pending. Do not describe the launcher as shipped until that evidence is committed and a release record is published.
