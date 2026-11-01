<#
  stop-demo.ps1 - stop and remove ONLY this project's containers and network (Compose project "minorproject-demo").
  Images, data (processed\), checkpoints and results are not touched; no volumes are deleted (no -v).
#>
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$envArg = if (Test-Path ".env.runtime") { "--env-file `".env.runtime`"" } else { "" }
cmd /c "docker compose -p minorproject-demo -f compose.yaml $envArg down 2>&1"
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: docker compose down failed (is Docker Desktop running?)" -ForegroundColor Red; exit 1 }
Write-Host "Stopped the minorproject-demo containers. Local data, checkpoints and results are unchanged." -ForegroundColor Green
