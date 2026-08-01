# RedRHex Windows Remote Launcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-installing Windows PowerShell launcher that opens secure SSH forwards for the RedRHex Training Panel and TensorBoard, waits for readiness, and opens both pages.

**Architecture:** One PowerShell file owns configuration, installation, tunnel startup, readiness polling, and browser launch. A dependency-free PowerShell test script verifies deterministic configuration and command-building functions; Windows smoke verification covers shortcut creation and the interactive SSH boundary.

**Tech Stack:** Windows PowerShell 5.1+, Windows OpenSSH Client, Tailscale, WScript.Shell shortcut COM API

## Global Constraints

- SSH target is exactly `lab_user1@100.90.246.97`.
- Forward local `8080` to workstation `127.0.0.1:8080`.
- Forward local `6006` to workstation `127.0.0.1:6006`.
- Store the installed launcher under `%LOCALAPPDATA%\RedRHex Remote\`.
- Create `RedRHex Remote.lnk` on the current user's Windows desktop.
- Do not store passwords, private keys, Tailscale credentials, or other secrets.
- Do not require administrator access or modify workstation services.
- Keep the SSH terminal visible and treat closing it as closing the tunnel.

---

### Task 1: Self-installing Windows launcher

**Files:**
- Create: `tools/windows/redrhex_remote.ps1`
- Create: `tools/windows/tests/test_redrhex_remote.ps1`
- Create: `tools/windows/README.md`

**Interfaces:**
- Consumes: Windows `ssh.exe`, `powershell.exe`, Tailscale connectivity to `100.90.246.97`, workstation HTTP endpoints on ports `8080` and `6006`
- Produces: `Get-RedRHexSshArguments` returning `string[]`; `Get-RedRHexInstallPaths` returning a `PSCustomObject`; `New-RedRHexTunnelCommand` returning `string`; `Install-RedRHexLauncher`; `Start-RedRHexRemote`; desktop shortcut `RedRHex Remote.lnk`

- [ ] **Step 1: Write the failing dependency-free PowerShell tests**

Create `tools/windows/tests/test_redrhex_remote.ps1`:

~~~powershell
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$launcherPath = Join-Path (Split-Path -Parent $PSScriptRoot) "redrhex_remote.ps1"
if (-not (Test-Path -LiteralPath $launcherPath)) {
    throw "Launcher not found: $launcherPath"
}

. $launcherPath

$script:FailureCount = 0

function Assert-Equal {
    param(
        [Parameter(Mandatory = $true)]$Actual,
        [Parameter(Mandatory = $true)]$Expected,
        [Parameter(Mandatory = $true)][string]$Name
    )

    if ($Actual -ne $Expected) {
        Write-Host "FAIL: $Name`n  expected: $Expected`n  actual:   $Actual" -ForegroundColor Red
        $script:FailureCount += 1
        return
    }
    Write-Host "PASS: $Name" -ForegroundColor Green
}

function Assert-True {
    param(
        [Parameter(Mandatory = $true)][bool]$Condition,
        [Parameter(Mandatory = $true)][string]$Name
    )

    if (-not $Condition) {
        Write-Host "FAIL: $Name" -ForegroundColor Red
        $script:FailureCount += 1
        return
    }
    Write-Host "PASS: $Name" -ForegroundColor Green
}

$arguments = Get-RedRHexSshArguments
$expectedArguments = @(
    "-N",
    "-o", "ExitOnForwardFailure=yes",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=3",
    "-L", "8080:127.0.0.1:8080",
    "-L", "6006:127.0.0.1:6006",
    "lab_user1@100.90.246.97"
)

Assert-Equal ($arguments -join "|") ($expectedArguments -join "|") "SSH arguments"

$paths = Get-RedRHexInstallPaths -LocalAppData "C:\Users\Test\AppData\Local" -Desktop "C:\Users\Test\Desktop"
Assert-Equal $paths.InstallDirectory "C:\Users\Test\AppData\Local\RedRHex Remote" "install directory"
Assert-Equal $paths.InstalledScript "C:\Users\Test\AppData\Local\RedRHex Remote\redrhex_remote.ps1" "installed script"
Assert-Equal $paths.Shortcut "C:\Users\Test\Desktop\RedRHex Remote.lnk" "desktop shortcut"

$tunnelCommand = New-RedRHexTunnelCommand -SshPath "C:\Windows\System32\OpenSSH\ssh.exe" -Arguments $arguments
Assert-True ($tunnelCommand.Contains("RedRHex SSH Tunnel")) "tunnel title"
Assert-True ($tunnelCommand.Contains("lab_user1@100.90.246.97")) "tunnel target"
Assert-True ($tunnelCommand.Contains("Press Enter to close")) "interactive error pause"

if ($script:FailureCount -gt 0) {
    throw "$script:FailureCount test(s) failed."
}

Write-Host "All RedRHex Remote tests passed." -ForegroundColor Green
~~~

- [ ] **Step 2: Run the tests to verify the launcher is missing**

Run on Windows from the repository root:

~~~powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\windows\tests\test_redrhex_remote.ps1
~~~

Expected: FAIL with `Launcher not found` because `tools/windows/redrhex_remote.ps1` does not exist.

- [ ] **Step 3: Implement the self-installing launcher**

Create `tools/windows/redrhex_remote.ps1`:

~~~powershell
[CmdletBinding()]
param(
    [switch]$Install
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:RedRHexConfig = [pscustomobject]@{
    SshHost          = "100.90.246.97"
    SshUser          = "lab_user1"
    PanelPort        = 8080
    TensorBoardPort  = 6006
    PanelUrl         = "http://localhost:8080"
    TensorBoardUrl   = "http://localhost:6006"
    ReadinessTimeout = 45
}

function Get-RedRHexSshArguments {
    [OutputType([string[]])]
    param(
        [pscustomobject]$Config = $script:RedRHexConfig
    )

    return [string[]]@(
        "-N",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-L", "$($Config.PanelPort):127.0.0.1:$($Config.PanelPort)",
        "-L", "$($Config.TensorBoardPort):127.0.0.1:$($Config.TensorBoardPort)",
        "$($Config.SshUser)@$($Config.SshHost)"
    )
}

function Get-RedRHexInstallPaths {
    [OutputType([pscustomobject])]
    param(
        [string]$LocalAppData = [Environment]::GetFolderPath("LocalApplicationData"),
        [string]$Desktop = [Environment]::GetFolderPath("Desktop")
    )

    if ([string]::IsNullOrWhiteSpace($LocalAppData)) {
        throw "Windows did not provide a Local AppData directory."
    }
    if ([string]::IsNullOrWhiteSpace($Desktop)) {
        throw "Windows did not provide a Desktop directory."
    }

    $installDirectory = Join-Path $LocalAppData "RedRHex Remote"
    return [pscustomobject]@{
        InstallDirectory = $installDirectory
        InstalledScript  = Join-Path $installDirectory "redrhex_remote.ps1"
        Shortcut         = Join-Path $Desktop "RedRHex Remote.lnk"
    }
}

function Test-RedRHexEndpoint {
    [OutputType([bool])]
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [int]$TimeoutSeconds = 2
    )

    try {
        $null = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec $TimeoutSeconds
        return $true
    }
    catch {
        return $false
    }
}

function Open-RedRHexPages {
    param(
        [pscustomobject]$Config = $script:RedRHexConfig
    )

    Start-Process $Config.PanelUrl
    Start-Process $Config.TensorBoardUrl
}

function New-RedRHexTunnelCommand {
    [OutputType([string])]
    param(
        [Parameter(Mandatory = $true)][string]$SshPath,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    $escapedPath = $SshPath.Replace("'", "''")
    $escapedArguments = $Arguments | ForEach-Object {
        "'" + $_.Replace("'", "''") + "'"
    }

    return @"
`$Host.UI.RawUI.WindowTitle = 'RedRHex SSH Tunnel'
& '$escapedPath' $($escapedArguments -join " ")
`$tunnelExitCode = `$LASTEXITCODE
if (`$tunnelExitCode -ne 0) {
    Write-Host ''
    Write-Host "SSH failed with exit code `$tunnelExitCode." -ForegroundColor Red
    Write-Host 'Check that Tailscale is connected and that the Ubuntu password is correct.'
    Read-Host 'Press Enter to close'
}
exit `$tunnelExitCode
"@
}

function Start-RedRHexTunnelWindow {
    [OutputType([System.Diagnostics.Process])]
    param(
        [Parameter(Mandatory = $true)][string]$SshPath,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    $powershellCommand = Get-Command powershell.exe -ErrorAction SilentlyContinue
    if ($null -eq $powershellCommand) {
        throw "Windows PowerShell (powershell.exe) is unavailable."
    }

    $command = New-RedRHexTunnelCommand -SshPath $SshPath -Arguments $Arguments
    $encodedCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($command))

    return Start-Process `
        -FilePath $powershellCommand.Source `
        -ArgumentList @("-NoLogo", "-NoProfile", "-EncodedCommand", $encodedCommand) `
        -PassThru
}

function Install-RedRHexLauncher {
    param(
        [string]$SourcePath = $PSCommandPath
    )

    if ([string]::IsNullOrWhiteSpace($SourcePath) -or -not (Test-Path -LiteralPath $SourcePath)) {
        throw "Cannot locate the launcher script being installed."
    }

    $paths = Get-RedRHexInstallPaths
    New-Item -ItemType Directory -Path $paths.InstallDirectory -Force | Out-Null

    $sourceFullPath = [IO.Path]::GetFullPath($SourcePath)
    $destinationFullPath = [IO.Path]::GetFullPath($paths.InstalledScript)
    if ($sourceFullPath -ne $destinationFullPath) {
        Copy-Item -LiteralPath $sourceFullPath -Destination $destinationFullPath -Force
    }

    $powershellCommand = Get-Command powershell.exe -ErrorAction SilentlyContinue
    if ($null -eq $powershellCommand) {
        throw "Windows PowerShell (powershell.exe) is unavailable."
    }

    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($paths.Shortcut)
    $shortcut.TargetPath = $powershellCommand.Source
    $shortcut.Arguments = "-NoLogo -NoProfile -ExecutionPolicy Bypass -File `"$($paths.InstalledScript)`""
    $shortcut.WorkingDirectory = $paths.InstallDirectory
    $shortcut.Description = "Connect to the RedRHex Training Panel and TensorBoard"
    $shortcut.IconLocation = "$env:SystemRoot\System32\imageres.dll,101"
    $shortcut.Save()

    Write-Host "Installed RedRHex Remote." -ForegroundColor Green
    Write-Host "Shortcut: $($paths.Shortcut)"
}

function Start-RedRHexRemote {
    param(
        [pscustomobject]$Config = $script:RedRHexConfig
    )

    if (Test-RedRHexEndpoint -Url $Config.PanelUrl) {
        Write-Host "An existing RedRHex tunnel is ready; opening the browser pages."
        Open-RedRHexPages -Config $Config
        return
    }

    $sshCommand = Get-Command ssh.exe -ErrorAction SilentlyContinue
    if ($null -eq $sshCommand) {
        throw "Windows OpenSSH Client is not installed. Install 'OpenSSH Client' in Windows Optional Features."
    }

    $arguments = Get-RedRHexSshArguments -Config $Config
    $tunnelProcess = Start-RedRHexTunnelWindow -SshPath $sshCommand.Source -Arguments $arguments

    Write-Host "The RedRHex SSH window is open."
    Write-Host "Enter the Ubuntu password there if prompted."
    Write-Host "Waiting for the Training Panel..."

    $deadline = (Get-Date).AddSeconds($Config.ReadinessTimeout)
    while ((Get-Date) -lt $deadline) {
        if ($tunnelProcess.HasExited) {
            throw "The SSH tunnel closed before the Training Panel became ready."
        }
        if (Test-RedRHexEndpoint -Url $Config.PanelUrl) {
            Open-RedRHexPages -Config $Config
            Write-Host "RedRHex Remote is ready." -ForegroundColor Green
            return
        }
        Start-Sleep -Seconds 1
    }

    throw "The Training Panel did not respond within $($Config.ReadinessTimeout) seconds. Check Tailscale, the Ubuntu password prompt, and the SSH window for errors."
}

if ($MyInvocation.InvocationName -ne ".") {
    try {
        if ($Install) {
            Install-RedRHexLauncher
        }
        else {
            Start-RedRHexRemote
        }
    }
    catch {
        Write-Host ""
        Write-Host "RedRHex Remote failed: $($_.Exception.Message)" -ForegroundColor Red
        Read-Host "Press Enter to close"
        exit 1
    }
}
~~~

- [ ] **Step 4: Run the dependency-free tests**

Run on Windows from the repository root:

~~~powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\windows\tests\test_redrhex_remote.ps1
~~~

Expected: every assertion prints `PASS` and the final line is `All RedRHex Remote tests passed.`

- [ ] **Step 5: Document installation, use, and removal**

Create `tools/windows/README.md`:

~~~~markdown
# RedRHex Remote for Windows

This launcher opens SSH forwards to the RedRHex workstation and then opens the Training Panel and TensorBoard in the default Windows browser.

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
~~~~

- [ ] **Step 6: Run repository-side static checks**

Run on the Ubuntu workstation:

~~~bash
git diff --check
rg -n "password\s*=|private.?key|auth.?key|tailscale.?key" tools/windows
~~~

Expected: `git diff --check` exits successfully. The secret scan returns no credential assignments; documentation mentions of passwords are descriptive only.

- [ ] **Step 7: Perform the Windows installation smoke test**

Run on the Windows laptop:

~~~powershell
$download = Join-Path $env:TEMP "redrhex_remote.ps1"
scp lab_user1@100.90.246.97:/home/lab_user1/Py/RedRHex/tools/windows/redrhex_remote.ps1 $download
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $download -Install
Test-Path (Join-Path ([Environment]::GetFolderPath("Desktop")) "RedRHex Remote.lnk")
Test-Path (Join-Path $env:LOCALAPPDATA "RedRHex Remote\redrhex_remote.ps1")
~~~

Expected: the installer prints `Installed RedRHex Remote.` and both `Test-Path` commands return `True`.

Double-click `RedRHex Remote`, authenticate in the visible SSH window, and confirm:

- <http://localhost:8080> displays the RedRHex Training Panel.
- <http://localhost:6006> displays TensorBoard when it is running on the workstation.
- Closing the tunnel window makes the forwarded pages unavailable.
- Double-clicking the shortcut while a tunnel is already active opens the pages without creating another tunnel.

- [ ] **Step 8: Commit the launcher**

~~~bash
git add tools/windows/redrhex_remote.ps1 tools/windows/tests/test_redrhex_remote.ps1 tools/windows/README.md
git commit -m "feat: add Windows remote launcher"
~~~
