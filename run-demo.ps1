<#
  run-demo.ps1 - start the three-container Minor Project demo (frontend, backend, model)

    powershell -ExecutionPolicy Bypass -File .\run-demo.ps1
    powershell -ExecutionPolicy Bypass -File .\run-demo.ps1 -Port 8770 -StrictPort
    powershell -ExecutionPolicy Bypass -File .\run-demo.ps1 -Rebuild -NoBrowser

  - checks Docker Desktop, Docker Compose, the local data and the trained checkpoint (SHA-256)
  - picks a free host port for the frontend: MINOR_PROJECT_PORT or -Port, else 8765, 8766, ... up to -MaxPort
    (never stops or kills whatever already uses a port)
  - writes the port to .env.runtime, starts the containers, waits until all three are healthy,
    checks the API and opens http://127.0.0.1:<port>
  No training happens; the model service only loads the saved checkpoint.
#>
param(
    [int]$Port = 0,
    [int]$MaxPort = 8795,
    [switch]$StrictPort,
    [switch]$Rebuild,
    [switch]$NoBrowser,
    [int]$HealthTimeoutSec = 420
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$Project = "minorproject-demo"
$EnvFile = Join-Path $PSScriptRoot ".env.runtime"
$Services = @("model", "backend", "frontend")

function Fail($msg) { Write-Host "`nERROR: $msg" -ForegroundColor Red; exit 1 }
function Step($msg) { Write-Host "== $msg" -ForegroundColor Cyan }

# Native command with stdout+stderr captured as text (no PowerShell error records)
function Invoke-Native([string]$cmdline) {
    $out = cmd /c "$cmdline 2>&1"
    return @{ Code = $LASTEXITCODE; Text = ($out | Out-String) }
}
function Compose([string]$args_) { return Invoke-Native "docker compose -p $Project -f compose.yaml --env-file `"$EnvFile`" $args_" }

function Test-PortFree([int]$p) {
    # 1) nothing listening according to Windows, 2) a real bind on 127.0.0.1 succeeds
    try {
        if (Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue) { return $false }
    } catch { }
    $listener = $null
    try {
        $listener = New-Object System.Net.Sockets.TcpListener ([System.Net.IPAddress]::Loopback), $p
        $listener.ExclusiveAddressUse = $true
        $listener.Start()
        return $true
    } catch { return $false }
    finally { if ($listener) { $listener.Stop() } }
}

function Select-Port([int]$preferred, [int[]]$exclude) {
    if ($preferred -gt 0 -and -not ($exclude -contains $preferred)) {
        if (Test-PortFree $preferred) { return $preferred }
        if ($StrictPort) { Fail "port $preferred is in use and -StrictPort was given. Free it or choose another port." }
        Write-Host "   port $preferred is in use (not touched); looking for another" -ForegroundColor Yellow
    }
    $start = 8765
    for ($p = $start; $p -le $MaxPort; $p++) {
        if ($exclude -contains $p -or $p -eq $preferred) { continue }
        if (Test-PortFree $p) { return $p }
        Write-Host "   port $p is in use (not touched)"
    }
    Fail "no free port between $start and $MaxPort. Close something that uses these ports or pass -MaxPort."
}

function Get-Health([string]$container) {
    $r = Invoke-Native "docker inspect -f `"{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}`" $container"
    if ($r.Code -ne 0) { return "missing" }
    return $r.Text.Trim()
}

# --- 1. prerequisites ---
Step "Checking Docker"
$r = Invoke-Native "docker version --format `"{{.Server.Version}}`""
if ($r.Code -ne 0) { Fail "Docker is not available. Start Docker Desktop and wait until it says 'Engine running'.`n$($r.Text)" }
Write-Host "   Docker engine $($r.Text.Trim())"
$r = Invoke-Native "docker compose version --short"
if ($r.Code -ne 0) { Fail "the Docker Compose plugin is missing (docker compose). Update Docker Desktop." }
Write-Host "   Docker Compose $($r.Text.Trim())"

Step "Checking local data and the trained checkpoint"
$need = @("compose.yaml", "processed\hospitals", "processed\hospital_split.json", "processed\sepsis_labels.parquet",
          "processed\timeseries_stats.json", "results\minor_project_sepsis\final_model_manifest.json")
foreach ($f in $need) { if (-not (Test-Path $f)) { Fail "required local file or folder missing: $f (see README, section 6b)" } }
$manifest = Get-Content "results\minor_project_sepsis\final_model_manifest.json" -Raw | ConvertFrom-Json
$entry = $manifest.models.($manifest.primary_model)
$ckpt = Join-Path $PSScriptRoot $entry.checkpoint
if (-not (Test-Path $ckpt)) { Fail "checkpoint missing: $($entry.checkpoint)" }
$hash = (Get-FileHash $ckpt -Algorithm SHA256).Hash.ToLower()
if ($hash -ne $entry.sha256) { Fail "checkpoint $($entry.checkpoint) does not match the SHA-256 in its manifest" }
Write-Host "   $($entry.checkpoint) verified (SHA-256 $($hash.Substring(0,12))..., round $($entry.training.selected_round))"
if (-not (Test-Path "demo_artifacts")) { Write-Host "   note: demo_artifacts\ not found; the synthetic demo tabs will show an error (run scripts\run_demo.py all to create it)" -ForegroundColor Yellow }

# --- 2. already running? ---
$states = $Services | ForEach-Object { Get-Health "minorproject-$_" }
if (($states | Where-Object { $_ -like "running*" }).Count -gt 0) {
    $oldPort = $null
    if (Test-Path $EnvFile) { $oldPort = (@(Get-Content $EnvFile) -match '^FRONTEND_PORT=' -replace 'FRONTEND_PORT=', '') }
    if (($states | Where-Object { $_ -eq "running healthy" }).Count -eq 3 -and $oldPort -and -not $Rebuild) {
        try {
            $h = Invoke-WebRequest "http://127.0.0.1:$oldPort/api/health" -UseBasicParsing -TimeoutSec 15
            if ($h.StatusCode -eq 200) {
                Write-Host "`nThe demo is already running and healthy (not restarted):" -ForegroundColor Green
                Write-Host "   http://127.0.0.1:$oldPort" -ForegroundColor Green
                Write-Host "   To restart it: .\stop-demo.ps1, then .\run-demo.ps1"
                if (-not $NoBrowser) { Start-Process "http://127.0.0.1:$oldPort" }
                exit 0
            }
        } catch { }
    }
    Write-Host "   containers of this project exist but are not all healthy: $($Services -join ', ') = $($states -join ' | ')" -ForegroundColor Yellow
    Write-Host "   they will be updated by 'docker compose up' (only this project's containers)"
}

# --- 3. port + start (retry if the port is taken between check and bind) ---
$preferred = if ($Port -gt 0) { $Port } elseif ($env:MINOR_PROJECT_PORT) { [int]$env:MINOR_PROJECT_PORT } else { 8765 }
$tried = @()
$chosen = $null
for ($attempt = 1; $attempt -le 4; $attempt++) {
    Step "Selecting a frontend port (attempt $attempt)"
    $chosen = Select-Port $preferred $tried
    Set-Content -Path $EnvFile -Value "FRONTEND_PORT=$chosen" -Encoding ascii
    Write-Host "   using port $chosen (written to .env.runtime)"
    Step "Starting containers (first start builds the images; this can take several minutes)"
    $build = if ($Rebuild) { "--build" } else { "" }
    $r = Compose "up -d $build"
    if ($r.Code -eq 0) { break }
    if ($r.Text -match "port is already allocated|address already in use|Only one usage of each socket address|bind: ") {
        Write-Host "   port $chosen became unavailable while starting; trying another port" -ForegroundColor Yellow
        $tried += $chosen; $preferred = 0; $chosen = $null
        continue
    }
    Write-Host $r.Text
    foreach ($s in $Services) {
        if ((Get-Health "minorproject-$s") -match "unhealthy|exited|dead") {
            Write-Host "`n--- last log lines of $s ---" -ForegroundColor Yellow
            (Compose "logs --no-color --tail 25 $s").Text | Write-Host
        }
    }
    Fail "docker compose up failed (see above). Fix the cause, then run .\stop-demo.ps1 and .\run-demo.ps1 again."
}
if (-not $chosen) { Fail "could not bind a frontend port after 4 attempts. Ports tried: $($tried -join ', ')" }

# --- 4. wait for health ---
Step "Waiting for the services to become healthy (model loads and verifies the checkpoint first)"
$deadline = (Get-Date).AddSeconds($HealthTimeoutSec)
$last = ""
while ($true) {
    $states = $Services | ForEach-Object { Get-Health "minorproject-$_" }
    $line = ($Services | ForEach-Object -Begin { $i = 0 } -Process { "$_=$($states[$i])"; $i++ }) -join "  "
    if ($line -ne $last) { Write-Host "   $line"; $last = $line }
    if (($states | Where-Object { $_ -eq "running healthy" }).Count -eq 3) { break }
    $bad = @()
    for ($i = 0; $i -lt 3; $i++) { if ($states[$i] -match "unhealthy|exited|dead|missing") { $bad += $Services[$i] } }
    if ($bad.Count -gt 0 -or (Get-Date) -gt $deadline) {
        $show = if ($bad.Count -gt 0) { $bad } else { $Services }
        foreach ($s in $show) { Write-Host "`n--- last log lines of $s ---" -ForegroundColor Yellow; (Compose "logs --no-color --tail 40 $s").Text | Write-Host }
        Fail "service(s) $($show -join ', ') did not become healthy. Fix the cause above, then run .\stop-demo.ps1 and .\run-demo.ps1 again."
    }
    Start-Sleep -Seconds 3
}

# --- 5. API check + browser ---
$url = "http://127.0.0.1:$chosen"
try {
    $h = Invoke-WebRequest "$url/api/health" -UseBasicParsing -TimeoutSec 30
    $model = ($h.Content | ConvertFrom-Json).model
    Write-Host "   API ready; model checkpoint $($model.checkpoint) (SHA-256 $($model.checkpoint_sha256_prefix)..., round $($model.checkpoint_round))"
} catch { Fail "the frontend is up but $url/api/health failed: $_" }

Write-Host "`nMinor Project demo is running:  $url" -ForegroundColor Green
Write-Host "   Interactive prediction tab = trained mortality + sepsis checkpoint. Stop with .\stop-demo.ps1"
if (-not $NoBrowser) { Start-Process $url }
