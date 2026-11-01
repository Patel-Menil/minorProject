<#
  status-demo.ps1 - state and health of the three demo containers, the frontend URL and the loaded checkpoint
#>
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
cmd /c "docker version --format {{.Server.Version}} >nul 2>&1"
if ($LASTEXITCODE -ne 0) { Write-Host "Docker is not available (start Docker Desktop)." -ForegroundColor Red; exit 1 }

foreach ($s in "model", "backend", "frontend") {
    $state = cmd /c "docker inspect -f `"{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{else}}-{{end}}`" minorproject-$s 2>nul"
    if (-not $state) { $state = "not created" }
    Write-Host ("  {0,-9} {1}" -f $s, $state)
}
$port = $null
if (Test-Path ".env.runtime") { $port = (@(Get-Content ".env.runtime") -match '^FRONTEND_PORT=' -replace 'FRONTEND_PORT=', '') }
$published = cmd /c "docker port minorproject-frontend 80 2>nul"
if ($published) { Write-Host "  published: $published (frontend only)" }
if (-not $port) { Write-Host "  no .env.runtime yet - start with .\run-demo.ps1"; exit 0 }
Write-Host "  URL      : http://127.0.0.1:$port"
try {
    $h = Invoke-WebRequest "http://127.0.0.1:$port/api/health" -UseBasicParsing -TimeoutSec 15
    $m = ($h.Content | ConvertFrom-Json).model
    $manifest = Get-Content "results\minor_project_sepsis\final_model_manifest.json" -Raw | ConvertFrom-Json
    $expected = $manifest.models.($manifest.primary_model).sha256.Substring(0, 12)
    $ok = if ($m.checkpoint_sha256_prefix -eq $expected) { "expected checkpoint" } else { "UNEXPECTED checkpoint" }
    Write-Host "  model    : ready=$($m.ready), $($m.checkpoint) round $($m.checkpoint_round), SHA-256 $($m.checkpoint_sha256_prefix)... ($ok)"
} catch {
    Write-Host "  API      : not reachable or not ready ($($_.Exception.Message))" -ForegroundColor Yellow
}
