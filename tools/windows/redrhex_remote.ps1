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
    PanelUrl         = "http://localhost:8080/?remote_client=windows"
    TensorBoardUrl   = "http://localhost:6006"
    PanelRoot        = "/home/lab_user1/Py/RedRHex"
    CondaInit        = "/home/lab_user1/miniconda3/etc/profile.d/conda.sh"
    CondaEnvironment  = "env_isaaclab_bin"
    PanelSession      = "redrhex_panel"
    ReadinessTimeout = 45
}

function Get-RedRHexRemoteSessionCommand {
    [OutputType([string])]
    param(
        [pscustomobject]$Config = $script:RedRHexConfig
    )

    $command = @'
set -eu

panel_url="http://127.0.0.1:__PANEL_PORT__"
panel_root="__PANEL_ROOT__"
conda_init="__CONDA_INIT__"
conda_environment="__CONDA_ENVIRONMENT__"
tmux_session="__PANEL_SESSION__"
panel_log="$panel_root/logs/training_panel/remote_panel.log"
panel_command="source $conda_init && conda activate $conda_environment && cd $panel_root && exec python -m tools.training_panel --host 127.0.0.1 --port __PANEL_PORT__"

if curl -fsS --max-time 2 "$panel_url" >/dev/null 2>&1; then
    echo "Training Panel is already running."
elif command -v tmux >/dev/null 2>&1; then
    if tmux has-session -t "$tmux_session" 2>/dev/null; then
        echo "Training Panel session already exists: $tmux_session"
    else
        tmux new-session -d -s "$tmux_session" -- bash -lc "$panel_command"
        echo "Started Training Panel in tmux session: $tmux_session"
    fi
else
    mkdir -p "$(dirname "$panel_log")"
    nohup bash -lc "$panel_command" >"$panel_log" 2>&1 </dev/null &
    echo "Started Training Panel with nohup. Log: $panel_log"
fi

attempts=__READINESS_TIMEOUT__
while [ "$attempts" -gt 0 ]; do
    if curl -fsS --max-time 2 "$panel_url" >/dev/null 2>&1; then
        echo "Training Panel is ready at $panel_url"
        while :; do sleep 3600; done
    fi
    sleep 1
    attempts=$((attempts - 1))
done

echo "Training Panel did not become ready."
if command -v tmux >/dev/null 2>&1 && tmux has-session -t "$tmux_session" 2>/dev/null; then
    tmux capture-pane -p -t "$tmux_session" -S -40 || true
elif [ -f "$panel_log" ]; then
    tail -40 "$panel_log" || true
fi
exit 1
'@

    $command = $command.Replace("__PANEL_PORT__", [string]$Config.PanelPort)
    $command = $command.Replace("__PANEL_ROOT__", $Config.PanelRoot)
    $command = $command.Replace("__CONDA_INIT__", $Config.CondaInit)
    $command = $command.Replace("__CONDA_ENVIRONMENT__", $Config.CondaEnvironment)
    $command = $command.Replace("__PANEL_SESSION__", $Config.PanelSession)
    $command = $command.Replace("__READINESS_TIMEOUT__", [string]$Config.ReadinessTimeout)
    return $command
}

function Get-RedRHexSshArguments {
    [OutputType([string[]])]
    param(
        [pscustomobject]$Config = $script:RedRHexConfig
    )

    $remoteScript = Get-RedRHexRemoteSessionCommand -Config $Config
    $remoteScriptBase64 = [Convert]::ToBase64String(
        [Text.Encoding]::UTF8.GetBytes($remoteScript)
    )
    $remoteCommand = "printf %s $remoteScriptBase64 | base64 -d | bash"

    return [string[]]@(
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-L", "$($Config.PanelPort):127.0.0.1:$($Config.PanelPort)",
        "-L", "$($Config.TensorBoardPort):127.0.0.1:$($Config.TensorBoardPort)",
        "$($Config.SshUser)@$($Config.SshHost)",
        $remoteCommand
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

    $trimCharacters = [char[]]@("\", "/")
    $installDirectory = $LocalAppData.TrimEnd($trimCharacters) + "\RedRHex Remote"
    return [pscustomobject]@{
        InstallDirectory = $installDirectory
        InstalledScript  = $installDirectory + "\redrhex_remote.ps1"
        Shortcut         = $Desktop.TrimEnd($trimCharacters) + "\RedRHex Remote.lnk"
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
    if (Test-RedRHexEndpoint -Url $Config.TensorBoardUrl) {
        Start-Process $Config.TensorBoardUrl
    }
    else {
        Write-Host "TensorBoard is not running; start it from the Training Panel when needed."
    }
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
