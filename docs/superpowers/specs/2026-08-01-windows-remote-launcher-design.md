# RedRHex Windows Remote Launcher Design

## Goal

Provide a dependable Windows desktop shortcut that connects a laptop to the RedRHex workstation over Tailscale, forwards the Training Panel and TensorBoard ports, and opens both services in the laptop's default browser.

## Scope

The launcher is a Windows convenience tool. It does not modify the Training Panel, Isaac Lab, Tailscale, SSH server settings, or training code. The workstation remains the execution host and source of run artifacts.

## Deliverable

The repository will contain one self-installing PowerShell script. Running it once with an install option will copy the script to a stable per-user Windows location and create a `RedRHex Remote` shortcut on the current user's desktop. The shortcut will run the installed script without requiring the repository to exist locally.

## Connection Configuration

The launcher will use these fixed values:

- SSH host: `100.90.246.97`
- SSH user: `lab_user1`
- Panel forwarding: local `8080` to workstation `127.0.0.1:8080`
- TensorBoard forwarding: local `6006` to workstation `127.0.0.1:6006`
- Panel URL: `http://localhost:8080`
- TensorBoard URL: `http://localhost:6006`

The SSH process will run in a visible terminal so password prompts, host-key confirmation, and connection errors remain understandable. Installing an SSH key later will remove the recurring password prompt without changing the shortcut.

## Launch Flow

1. Check that Windows OpenSSH (`ssh.exe`) is available.
2. Check whether the local panel URL already responds. If it does, reuse the existing tunnel and open the two browser pages without starting a duplicate SSH process.
3. If no tunnel is active, start a visible SSH terminal with both local forwards and keepalive options.
4. Poll the panel URL for up to 45 seconds.
5. Once the panel responds, open the panel and TensorBoard URLs in the default browser.
6. Leave the SSH terminal visible. Closing that terminal closes the tunnel.

## Error Handling

- If `ssh.exe` is unavailable, show a clear instruction to install the Windows OpenSSH Client.
- If the workstation cannot be reached within 45 seconds, explain that Tailscale may be disconnected, the Ubuntu password may not have been entered, or SSH may have failed.
- If local port `8080` or `6006` is occupied by an unrelated process, SSH will show the bind error in the visible terminal and the launcher will not claim success.
- The launcher will not store passwords, private keys, Tailscale credentials, or other secrets.

## Installation and Removal

Installation will be initiated from Windows PowerShell after copying the single script from the workstation with `scp`. The installer mode will create only:

- A per-user launcher script under `%LOCALAPPDATA%\RedRHex Remote\`
- A `RedRHex Remote` shortcut on the current user's desktop

Removal will be documented as deleting those two user-owned items. It will not require administrator access.

## Verification

Verification on Windows will confirm:

1. The install option creates the script and desktop shortcut.
2. Double-clicking the shortcut presents the SSH authentication terminal when no tunnel exists.
3. After authentication, `http://localhost:8080` loads the Training Panel.
4. `http://localhost:6006` loads TensorBoard when TensorBoard is running on the workstation.
5. Closing the SSH terminal makes the forwarded URLs unavailable.
6. Running the shortcut while a tunnel already exists opens the URLs without creating a second SSH process.

