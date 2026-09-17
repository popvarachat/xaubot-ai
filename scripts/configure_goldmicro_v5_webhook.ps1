param(
    [Parameter(Mandatory = $true)][string]$WebhookUrl
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if (-not [Uri]::IsWellFormedUriString($WebhookUrl, [UriKind]::Absolute)) {
    throw "WebhookUrl must be an absolute URL"
}
$uri = [Uri]$WebhookUrl
if ($uri.Scheme -ne "https") {
    throw "WebhookUrl must use HTTPS"
}

$SecretDir = Join-Path $env:LOCALAPPDATA "xaubot-ai\secrets"
New-Item -ItemType Directory -Force -Path $SecretDir | Out-Null
$SecretPath = Join-Path $SecretDir "goldmicro-v5-webhook-token.clixml"

$Token = Read-Host "Enter n8n/Cloudflare webhook bearer token" -AsSecureString
if ($Token.Length -lt 32) {
    throw "Webhook bearer token must be at least 32 characters"
}

# Export-Clixml protects SecureString material with Windows DPAPI for this user/machine.
$Token | Export-Clixml -Path $SecretPath
[Environment]::SetEnvironmentVariable("GOLDMICRO_N8N_WEBHOOK_URL", $WebhookUrl, "User")
[Environment]::SetEnvironmentVariable("GOLDMICRO_N8N_WEBHOOK_TOKEN_FILE", $SecretPath, "User")

Write-Host "=== GOLDmicro V5 Webhook Client Configured ==="
Write-Host "Webhook : $WebhookUrl"
Write-Host "Token   : stored as DPAPI-protected SecureString"
Write-Host "File    : $SecretPath"
Write-Host "Scope   : current Windows user / machine"
Write-Host "Note    : no token value was written to the repository or console"
