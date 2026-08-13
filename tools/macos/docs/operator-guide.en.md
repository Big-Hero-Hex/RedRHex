---
id: macos-remote-launcher-operator-guide
title: Install and Use RedRHex Remote on macOS
lang: en
audience: operator
type: how-to
status: draft
owner: panel
last_reviewed: 2026-08-14
---

<a id="prerequisites"></a>
## Prerequisites and status

Use macOS Terminal with Tailscale connected. The launcher uses the macOS-provided `ssh`, `curl`, `base64`, and `open` commands. It is a release candidate until the active macOS smoke checklist is completed, and it stores no password, private key, or Tailscale credential.

<a id="install"></a>
## Install for the current user

Open Terminal and run:

```sh
download="$(mktemp -t redrhex_remote)"
scp lab_user1@100.90.246.97:/home/lab_user1/Py/RedRHex/tools/macos/redrhex_remote.command "$download"
sh "$download" --install
rm -f "$download"
```

Enter the Ubuntu password only in the `scp` or SSH prompt. Installation requires no administrator access. It creates an executable `RedRHex Remote.command` on the current user's desktop.

<a id="use"></a>
## Connect

1. Confirm that Tailscale is connected.
2. Double-click `RedRHex Remote.command` on the desktop.
3. If Terminal displays a host-key or password prompt, verify the host and enter the Ubuntu password.
4. Keep that Terminal window open while using the forwarded services.

The launcher reuses a responsive tunnel. Otherwise it connects to `lab_user1@100.90.246.97`, forwards local ports `8080` and `6006`, and starts the workstation panel in the `env_isaaclab_bin` environment. It opens the Training Panel at `http://localhost:8080`. It opens TensorBoard at `http://localhost:6006` only when TensorBoard is already responding.

<a id="stop"></a>
## Stop and remove

Close the launcher Terminal window or press Control-C to stop both forwards. To remove the per-user installation:

```sh
rm "$HOME/Desktop/RedRHex Remote.command"
```

This command removes only the installed launcher copy.

<a id="troubleshoot"></a>
## Troubleshoot

- macOS refuses to open the file: Control-click `RedRHex Remote.command`, choose **Open**, and confirm the first launch.
- Authentication or reachability failure: confirm Tailscale, the workstation address, host-key prompt, and Ubuntu password in Terminal.
- Local bind failure: stop the process already using port `8080` or `6006`, then retry.
- Panel readiness timeout: inspect Terminal and workstation tmux session `redrhex_panel`; without tmux, inspect `logs/training_panel/remote_panel.log`.
- TensorBoard does not open: start it from the Training Panel, then open `http://localhost:6006`.
