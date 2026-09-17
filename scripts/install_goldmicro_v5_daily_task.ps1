param(
    [int]$EveryHours = 4,
    [string]$TaskName = "GOLDmicro V5 Prospective"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ($EveryHours -lt 1 -or $EveryHours -gt 24) {
    throw "EveryHours must be between 1 and 24"
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Runner = Join-Path $RepoRoot "scripts\run_goldmicro_v5_daily.ps1"
if (-not (Test-Path $Runner)) { throw "Runner not found: $Runner" }

$PowerShell = (Get-Command powershell.exe).Source
$Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$Runner`""
$Action = New-ScheduledTaskAction -Execute $PowerShell -Argument $Arguments -WorkingDirectory $RepoRoot

# Build repetition through New-ScheduledTaskTrigger parameters rather than mutating
# Trigger.Repetition.* afterwards. Some Windows/PowerShell combinations expose the
# returned CIM object's Repetition members as unavailable/read-only, which caused
# installer failures on the target workstation.
$Start = (Get-Date).AddMinutes(5)
$Interval = New-TimeSpan -Hours $EveryHours
$Duration = New-TimeSpan -Days 3650
$Trigger = New-ScheduledTaskTrigger `
    -Once `
    -At $Start `
    -RepetitionInterval $Interval `
    -RepetitionDuration $Duration

$Identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$Principal = New-ScheduledTaskPrincipal -UserId $Identity -LogonType Interactive -RunLevel Limited
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 30)

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Principal $Principal -Settings $Settings -Description "Read-only GOLDmicro V5 prospective M15 collection and blind readiness check. No trading mutations." -Force | Out-Null

Write-Host "=== GOLDmicro V5 Scheduled Task Installed ==="
Write-Host "Task        : $TaskName"
Write-Host "User        : $Identity"
Write-Host "First run   : $Start"
Write-Host "Repeat      : every $EveryHours hour(s)"
Write-Host "Runner      : $Runner"
Write-Host "Mode        : Interactive user session; no order placement"
Write-Host ""
Write-Host "To test now:"
Write-Host "  Start-ScheduledTask -TaskName `"$TaskName`""
Write-Host "To inspect:"
Write-Host "  Get-ScheduledTask -TaskName `"$TaskName`" | Get-ScheduledTaskInfo"
