param(
    [string]$Manifest = ".\models\reports\event_edge_v4_20260916_132710\smc_setup_v5_prospective_manifest.json",
    [string]$Snapshot = ".\models\reports\event_edge_v4_20260916_132710\market_snapshot_m15_v5_prospective.parquet"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

$LogDir = Join-Path $RepoRoot "runtime\goldmicro_v5"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogPath = Join-Path $LogDir "prospective_$Stamp.log"

function Write-RunLog([string]$Text) {
    $line = "$(Get-Date -Format s) $Text"
    Write-Host $line
    Add-Content -Path $LogPath -Value $line
}

function Invoke-LoggedNative {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $stdoutPath = [System.IO.Path]::GetTempFileName()
    $stderrPath = [System.IO.Path]::GetTempFileName()
    try {
        $process = Start-Process `
            -FilePath $FilePath `
            -ArgumentList $ArgumentList `
            -WorkingDirectory $RepoRoot `
            -NoNewWindow `
            -Wait `
            -PassThru `
            -RedirectStandardOutput $stdoutPath `
            -RedirectStandardError $stderrPath

        if (Test-Path $stdoutPath) {
            Get-Content $stdoutPath | Tee-Object -FilePath $LogPath -Append | Write-Host
        }
        if (Test-Path $stderrPath) {
            Get-Content $stderrPath | Tee-Object -FilePath $LogPath -Append | Write-Host
        }

        if ($process.ExitCode -ne 0) {
            throw "$Label failed with exit code $($process.ExitCode)"
        }
    }
    finally {
        Remove-Item -Force -ErrorAction SilentlyContinue $stdoutPath, $stderrPath
    }
}

function Get-BlockValue($array, [int]$index) {
    if ($null -eq $array) { return 0 }
    if ($array.Count -le $index) { return 0 }
    return [int]$array[$index]
}

function Get-ConfiguredEnvironmentVariable([string]$Name) {
    $processValue = [Environment]::GetEnvironmentVariable($Name, "Process")
    if (-not [string]::IsNullOrWhiteSpace($processValue)) { return $processValue }

    $userValue = [Environment]::GetEnvironmentVariable($Name, "User")
    if (-not [string]::IsNullOrWhiteSpace($userValue)) { return $userValue }

    return $null
}

try {
    Write-RunLog "GOLDmicro V5 automated prospective run START"

    # Anti-overfit / protocol guard: fail closed if the frozen V5 generator changed.
    $ExpectedGeneratorBlob = "19ea155c9319515de906dfbdb9b5e67b5e7864de"
    $ActualGeneratorBlob = (git hash-object ".\src\goldmicro_smc_setup_v5.py").Trim()
    if ($ActualGeneratorBlob -ne $ExpectedGeneratorBlob) {
        throw "Frozen V5 generator mismatch. Expected $ExpectedGeneratorBlob, got $ActualGeneratorBlob. Refusing automated collection."
    }
    Write-RunLog "Frozen generator verified: $ActualGeneratorBlob"

    $Python = Join-Path $env:LOCALAPPDATA "xaubot-ai\goldmicro-research-py311\Scripts\python.exe"
    if (-not (Test-Path $Python)) {
        throw "Research Python not found: $Python"
    }
    if (-not (Test-Path $Manifest)) {
        throw "Prospective manifest not found: $Manifest"
    }

    Write-RunLog "Running read-only MT5 collector"
    Invoke-LoggedNative `
        -FilePath $Python `
        -ArgumentList @(".\scripts\collect_goldmicro_smc_setup_v5_m15.py", "--manifest", $Manifest) `
        -Label "Collector"

    Write-RunLog "Running blind readiness checker"
    Invoke-LoggedNative `
        -FilePath $Python `
        -ArgumentList @(".\scripts\check_goldmicro_smc_setup_v5_readiness.py", "--manifest", $Manifest, "--snapshot", $Snapshot) `
        -Label "Readiness checker"

    $Report = ".\models\reports\event_edge_v4_20260916_132710\smc_setup_v5_prospective_readiness.json"
    if (-not (Test-Path $Report)) { throw "Readiness report missing after run: $Report" }
    $ReportJson = Get-Content -Raw -Path $Report | ConvertFrom-Json
    $GitHead = (git rev-parse --short=12 HEAD).Trim()
    $RunTime = (Get-Date).ToString("o")

    # Always keep a compact local history independent from the large parquet snapshot.
    $HistoryCsv = Join-Path $LogDir "prospective_run_history.csv"
    $historyRow = [pscustomobject]@{
        run_time = $RunTime
        git_head = $GitHead
        status = [string]$ReportJson.status
        fresh_rows = [int]$ReportJson.fresh_rows
        fresh_setup_events = [int]$ReportJson.fresh_setup_events
        matured_setup_events = [int]$ReportJson.matured_setup_events
        block_1 = Get-BlockValue $ReportJson.block_event_counts 0
        block_2 = Get-BlockValue $ReportJson.block_event_counts 1
        block_3 = Get-BlockValue $ReportJson.block_event_counts 2
        block_4 = Get-BlockValue $ReportJson.block_event_counts 3
        block_5 = Get-BlockValue $ReportJson.block_event_counts 4
        economic_outcomes = "HIDDEN / NOT EVALUATED"
        promotion = "DISABLED"
    }
    if (Test-Path $HistoryCsv) {
        $historyRow | Export-Csv -Path $HistoryCsv -NoTypeInformation -Append -Encoding UTF8
    }
    else {
        $historyRow | Export-Csv -Path $HistoryCsv -NoTypeInformation -Encoding UTF8
    }

    $LatestStatus = Join-Path $LogDir "latest_status.json"
    $historyRow | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 -Path $LatestStatus
    Write-RunLog "Updated local run history and latest status"

    # Optional archive: user-scope fallback lets Task Scheduler see newly configured
    # values immediately without requiring a Windows sign-out/sign-in.
    $ArchiveDir = Get-ConfiguredEnvironmentVariable "GOLDMICRO_ARCHIVE_DIR"
    if ($ArchiveDir) {
        New-Item -ItemType Directory -Force -Path $ArchiveDir | Out-Null
        Copy-Item -Force $Report (Join-Path $ArchiveDir "smc_setup_v5_prospective_readiness.json")
        if (Test-Path $Snapshot) {
            Copy-Item -Force $Snapshot (Join-Path $ArchiveDir "market_snapshot_m15_v5_prospective.parquet")
        }
        $Meta = [System.IO.Path]::ChangeExtension($Snapshot, ".json")
        if (Test-Path $Meta) {
            Copy-Item -Force $Meta (Join-Path $ArchiveDir "market_snapshot_m15_v5_prospective.json")
        }
        Copy-Item -Force $HistoryCsv (Join-Path $ArchiveDir "prospective_run_history.csv")
        Copy-Item -Force $LatestStatus (Join-Path $ArchiveDir "latest_status.json")
        Write-RunLog "Archived current prospective artifacts to $ArchiveDir"
    }
    else {
        Write-RunLog "Archive directory not configured; local artifacts retained"
    }

    # Optional n8n/Cloudflare ingress. No URL/token is stored in the repository.
    $WebhookUrl = Get-ConfiguredEnvironmentVariable "GOLDMICRO_N8N_WEBHOOK_URL"
    if ($WebhookUrl) {
        $headers = @{}
        $WebhookToken = Get-ConfiguredEnvironmentVariable "GOLDMICRO_N8N_WEBHOOK_TOKEN"
        if ($WebhookToken) {
            $headers["Authorization"] = "Bearer $WebhookToken"
        }
        $payload = @{
            source = "goldmicro-v5-prospective"
            generated_at = $RunTime
            host = $env:COMPUTERNAME
            git_head = $GitHead
            state = $ReportJson.status
            fresh_rows = $ReportJson.fresh_rows
            fresh_setup_events = $ReportJson.fresh_setup_events
            matured_setup_events = $ReportJson.matured_setup_events
            per_block_matured = $ReportJson.block_event_counts
            economic_outcomes = "HIDDEN / NOT EVALUATED"
            promotion = "DISABLED"
        } | ConvertTo-Json -Depth 5
        Invoke-RestMethod -Method Post -Uri $WebhookUrl -Headers $headers -ContentType "application/json" -Body $payload | Out-Null
        Write-RunLog "Posted readiness summary to configured webhook"
    }

    Write-RunLog "GOLDmicro V5 automated prospective run SUCCESS"
    exit 0
}
catch {
    Write-RunLog "FAILED: $($_.Exception.Message)"
    exit 1
}
