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
    $collector = & $Python ".\scripts\collect_goldmicro_smc_setup_v5_m15.py" --manifest $Manifest 2>&1
    $collector | Tee-Object -FilePath $LogPath -Append | Write-Host
    if ($LASTEXITCODE -ne 0) { throw "Collector failed with exit code $LASTEXITCODE" }

    Write-RunLog "Running blind readiness checker"
    $readiness = & $Python ".\scripts\check_goldmicro_smc_setup_v5_readiness.py" --manifest $Manifest --snapshot $Snapshot 2>&1
    $readiness | Tee-Object -FilePath $LogPath -Append | Write-Host
    if ($LASTEXITCODE -ne 0) { throw "Readiness checker failed with exit code $LASTEXITCODE" }

    $Report = ".\models\reports\event_edge_v4_20260916_132710\smc_setup_v5_prospective_readiness.json"
    if (-not (Test-Path $Report)) { throw "Readiness report missing after run: $Report" }
    $ReportJson = Get-Content -Raw -Path $Report | ConvertFrom-Json

    # Optional archive: point GOLDMICRO_ARCHIVE_DIR to a Google Drive Desktop-synced folder.
    if ($env:GOLDMICRO_ARCHIVE_DIR) {
        $ArchiveDir = $env:GOLDMICRO_ARCHIVE_DIR
        New-Item -ItemType Directory -Force -Path $ArchiveDir | Out-Null
        Copy-Item -Force $Report (Join-Path $ArchiveDir "smc_setup_v5_prospective_readiness.json")
        if (Test-Path $Snapshot) {
            Copy-Item -Force $Snapshot (Join-Path $ArchiveDir "market_snapshot_m15_v5_prospective.parquet")
        }
        $Meta = [System.IO.Path]::ChangeExtension($Snapshot, ".json")
        if (Test-Path $Meta) {
            Copy-Item -Force $Meta (Join-Path $ArchiveDir "market_snapshot_m15_v5_prospective.json")
        }
        Write-RunLog "Archived current prospective artifacts to $ArchiveDir"
    }

    # Optional n8n/Cloudflare ingress. No URL/token is stored in the repository.
    if ($env:GOLDMICRO_N8N_WEBHOOK_URL) {
        $headers = @{}
        if ($env:GOLDMICRO_N8N_WEBHOOK_TOKEN) {
            $headers["Authorization"] = "Bearer $($env:GOLDMICRO_N8N_WEBHOOK_TOKEN)"
        }
        $payload = @{
            source = "goldmicro-v5-prospective"
            generated_at = (Get-Date).ToString("o")
            host = $env:COMPUTERNAME
            state = $ReportJson.status
            fresh_rows = $ReportJson.fresh_rows
            fresh_setup_events = $ReportJson.fresh_setup_events
            matured_setup_events = $ReportJson.matured_setup_events
            per_block_matured = $ReportJson.block_event_counts
            economic_outcomes = "HIDDEN / NOT EVALUATED"
            promotion = "DISABLED"
        } | ConvertTo-Json -Depth 5
        Invoke-RestMethod -Method Post -Uri $env:GOLDMICRO_N8N_WEBHOOK_URL -Headers $headers -ContentType "application/json" -Body $payload | Out-Null
        Write-RunLog "Posted readiness summary to configured webhook"
    }

    Write-RunLog "GOLDmicro V5 automated prospective run SUCCESS"
    exit 0
}
catch {
    Write-RunLog "FAILED: $($_.Exception.Message)"
    exit 1
}
