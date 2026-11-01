<#
  logs-demo.ps1 - recent container logs (method / path / status lines; no request bodies or patient data)

    powershell -ExecutionPolicy Bypass -File .\logs-demo.ps1            # all three services
    powershell -ExecutionPolicy Bypass -File .\logs-demo.ps1 model      # one service: model | backend | frontend
    powershell -ExecutionPolicy Bypass -File .\logs-demo.ps1 model -Tail 300 -Follow
#>
param([ValidateSet("", "model", "backend", "frontend")][string]$Service = "", [int]$Tail = 100, [switch]$Follow)
Set-Location $PSScriptRoot
$envArg = if (Test-Path ".env.runtime") { "--env-file `".env.runtime`"" } else { "" }
$f = if ($Follow) { "--follow" } else { "" }
cmd /c "docker compose -p minorproject-demo -f compose.yaml $envArg logs --no-color --tail $Tail $f $Service 2>&1"
