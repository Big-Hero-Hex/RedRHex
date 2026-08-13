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
$expectedPrefix = @(
    "-o", "ExitOnForwardFailure=yes",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=3",
    "-L", "8080:127.0.0.1:8080",
    "-L", "6006:127.0.0.1:6006",
    "lab_user1@100.90.246.97"
)

Assert-Equal ($arguments[0..($expectedPrefix.Count - 1)] -join "|") ($expectedPrefix -join "|") "SSH arguments"
Assert-True ($arguments.Count -eq ($expectedPrefix.Count + 1)) "SSH remote command argument"
Assert-True ($arguments[-1] -match '^printf %s [A-Za-z0-9+/=]+ \| base64 -d \| bash$') "SSH remote command is quote-safe"

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
