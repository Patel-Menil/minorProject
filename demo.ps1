<#
  demo.ps1 - one-command synthetic viva demo (Windows PowerShell, CPU only)

    powershell -ExecutionPolicy Bypass -File .\demo.ps1            # set up if needed, run demo, start UI
    powershell -ExecutionPolicy Bypass -File .\demo.ps1 -Test      # also run the test suite first
    powershell -ExecutionPolicy Bypass -File .\demo.ps1 -NoServe   # demo in the console only

  First run needs internet once (pip and npm packages). The demo itself is offline:
  synthetic data only, no eICU files, no cloud services, no API keys.
#>
param([switch]$Test, [switch]$NoServe, [int]$Port = 8765)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

function Step($msg) { Write-Host "`n== $msg" -ForegroundColor Cyan }
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE)" } }

if (-not (Test-Path $py)) {
    Step "Creating project virtual environment (.venv) - your system Python is not changed"
    python -m venv .venv; Check "python -m venv"
    & $py -m pip install --upgrade pip; Check "pip upgrade"
    & $py -m pip install -r requirements.txt; Check "pip install -r requirements.txt"
}

if ($Test) {
    Step "Running tests"
    & $py -m pytest tests -q; Check "pytest"
}

Step "Synthetic demo: generate data, federated training, inference, checkpoint check"
& $py scripts\run_demo.py all; Check "run_demo.py all"

if (-not (Test-Path "frontend-playground\dist\index.html")) {
    if (Get-Command npm -ErrorAction SilentlyContinue) {
        Step "Building the frontend (npm ci + npm run build)"
        Push-Location frontend-playground
        try { npm ci --no-audit --no-fund; Check "npm ci"; npm run build; Check "npm run build" } finally { Pop-Location }
    } else {
        Write-Warning "Node.js/npm not found: the API will run, but the web UI needs 'npm run build' once."
    }
}

if (-not $NoServe) {
    Step "Starting the demo server - open http://127.0.0.1:$Port in a browser (Ctrl+C to stop)"
    & $py scripts\run_demo.py serve --port $Port
}
