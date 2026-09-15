[CmdletBinding()]
param(
    [int]$Limit = 24,
    [int]$Samples = 5,
    [int]$SampleStrideBars = 1000,
    [int]$TopK = 6,
    [switch]$PlanOnly,
    [switch]$SkipGitPull,
    [switch]$NoInstallPrompt
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvBase = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } else { $env:USERPROFILE }
$VenvDir = Join-Path $VenvBase "xaubot-ai\goldmicro-research-py311"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$Requirements = Join-Path $RepoRoot "requirements-goldmicro-research.txt"
$Pipeline = Join-Path $RepoRoot "scripts\run_goldmicro_challenger_research.py"
$PreflightScript = Join-Path $RepoRoot "scripts\goldmicro_research_preflight.py"

function Write-Step([string]$Text) {
    Write-Host "`n=== $Text ===" -ForegroundColor Cyan
}

function Test-Python311 {
    try {
        & py -3.11 -c "import sys; print(sys.executable); print(sys.version)" 2>$null
        return ($LASTEXITCODE -eq 0)
    }
    catch {
        return $false
    }
}

if ($Limit -lt 1) { throw "Limit must be >= 1" }
if ($Samples -lt 1) { throw "Samples must be >= 1" }
if ($SampleStrideBars -lt 1) { throw "SampleStrideBars must be >= 1" }

Set-Location $RepoRoot
Write-Step "GOLDmicro Research Environment"
Write-Host "Repo          : $RepoRoot"
Write-Host "Venv          : $VenvDir"
Write-Host "Configurations: $Limit"
Write-Host "Samples/config: $Samples"
Write-Host "Training jobs : $Limit x $Samples = $($Limit * $Samples)"
Write-Host "Sample stride : $SampleStrideBars M15 bars"
Write-Host "Shortlist     : $TopK"
Write-Host "Mode          : $(if ($PlanOnly) { 'PLAN ONLY' } else { 'EXECUTE NON-LIVE' })"
Write-Host "Promotion     : DISABLED"

if (-not (Test-Path (Join-Path $RepoRoot ".git"))) {
    throw "Repository metadata not found at $RepoRoot"
}

if (-not $SkipGitPull) {
    Write-Step "Fast-forward repository"
    & git pull --ff-only
    if ($LASTEXITCODE -ne 0) {
        throw "git pull --ff-only failed; resolve Git state before research training"
    }
}

Write-Step "Check Python 3.11"
if (-not (Test-Python311)) {
    Write-Host "Python 3.11 is not registered with the Python launcher." -ForegroundColor Yellow
    Write-Host "The GOLDmicro research environment is intentionally pinned to Python 3.11 to match CI and avoid Python 3.14/hmmlearn build friction." -ForegroundColor Yellow

    if ($NoInstallPrompt) {
        throw "Python 3.11 missing. Install Python.Python.3.11 and rerun this script."
    }

    $Winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $Winget) {
        throw "Python 3.11 is missing and winget is unavailable. Install Python 3.11 (64-bit) then rerun."
    }

    $answer = Read-Host "Install Python 3.11 for the current user now via winget? [Y/N]"
    if ($answer -notmatch '^[Yy]$') {
        throw "Python 3.11 installation was not approved. No environment changes were made."
    }

    Write-Step "Install Python 3.11"
    & winget install --exact --id Python.Python.3.11 --scope user --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        throw "winget could not install Python 3.11"
    }

    if (-not (Test-Python311)) {
        throw "Python 3.11 was installed but is not visible to 'py -3.11' yet. Open a new PowerShell window and rerun this script."
    }
}

Write-Step "Create isolated Python 3.11 venv"
if (-not (Test-Path $VenvPython)) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $VenvDir) | Out-Null
    & py -3.11 -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create $VenvDir"
    }
}
& $VenvPython --version

Write-Step "Install research dependencies"
& $VenvPython -m pip install --disable-pip-version-check --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "pip bootstrap failed" }
& $VenvPython -m pip install --disable-pip-version-check -r $Requirements
if ($LASTEXITCODE -ne 0) { throw "research dependency installation failed" }

Write-Step "Dependency preflight"
if (-not (Test-Path $PreflightScript)) {
    throw "Dependency preflight script not found: $PreflightScript"
}
& $VenvPython $PreflightScript
if ($LASTEXITCODE -ne 0) {
    throw "Dependency preflight failed"
}

Write-Step "Run GOLDmicro 24 x N chronological research matrix"
$GitSha = (& git rev-parse HEAD).Trim()
$Args = @(
    $Pipeline,
    "--limit", "$Limit",
    "--samples", "$Samples",
    "--sample-stride-bars", "$SampleStrideBars",
    "--top-k", "$TopK",
    "--git-sha", $GitSha
)
if (-not $PlanOnly) {
    $Args += "--execute"
}

& $VenvPython @Args
$ExitCode = $LASTEXITCODE
if ($ExitCode -ne 0) {
    throw "GOLDmicro research pipeline failed with exit code $ExitCode"
}

Write-Step "Complete"
Write-Host "Python env : $VenvPython"
Write-Host "Git SHA    : $GitSha"
Write-Host "Matrix     : $Limit x $Samples = $($Limit * $Samples) jobs"
Write-Host "Live model : UNCHANGED"
Write-Host "Promotion  : DISABLED"
Write-Host "Next       : inspect multisample_screening.json and strategy_oos_queue.json, then run PF/DD/cost OOS validation."
