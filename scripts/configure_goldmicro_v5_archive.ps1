param(
    [string]$ArchiveSubfolder = "GOLDmicro Model Lifecycle\V5 Prospective Archive"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Get-CandidateDriveRoots {
    $roots = New-Object System.Collections.Generic.List[string]

    foreach ($drive in Get-PSDrive -PSProvider FileSystem) {
        if ($drive.Root -and (Test-Path $drive.Root)) {
            $roots.Add($drive.Root)
        }
    }

    $common = @(
        (Join-Path $env:USERPROFILE "Google Drive"),
        (Join-Path $env:USERPROFILE "My Drive"),
        (Join-Path $env:USERPROFILE "Drive")
    )
    foreach ($path in $common) {
        if (Test-Path $path) { $roots.Add($path) }
    }

    return $roots | Select-Object -Unique
}

function Find-GoogleDriveMyDrive {
    foreach ($root in Get-CandidateDriveRoots) {
        $directCandidates = @(
            (Join-Path $root "My Drive"),
            (Join-Path $root "Google Drive\My Drive")
        )
        foreach ($candidate in $directCandidates) {
            if (Test-Path $candidate) { return (Resolve-Path $candidate).Path }
        }

        try {
            $marker = Get-ChildItem -Path $root -Directory -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -eq "My Drive" } |
                Select-Object -First 1
            if ($marker) { return $marker.FullName }
        }
        catch { }
    }
    return $null
}

$myDrive = Find-GoogleDriveMyDrive
if (-not $myDrive) {
    Write-Host "Google Drive Desktop 'My Drive' folder was not auto-detected."
    Write-Host "No environment variable was changed."
    Write-Host "If Drive Desktop is installed later, rerun this script."
    exit 2
}

$archive = Join-Path $myDrive $ArchiveSubfolder
New-Item -ItemType Directory -Force -Path $archive | Out-Null
[Environment]::SetEnvironmentVariable("GOLDMICRO_ARCHIVE_DIR", $archive, "User")
$env:GOLDMICRO_ARCHIVE_DIR = $archive

Write-Host "=== GOLDmicro V5 Drive Archive Configured ==="
Write-Host "My Drive : $myDrive"
Write-Host "Archive  : $archive"
Write-Host "Variable : GOLDMICRO_ARCHIVE_DIR (User)"
Write-Host "Mode     : local Google Drive Desktop sync; no broker/order credentials"
