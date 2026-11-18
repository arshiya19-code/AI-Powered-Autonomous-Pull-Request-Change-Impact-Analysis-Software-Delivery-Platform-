param(
    [string]$Python = '',
    [switch]$Frontend
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path -LiteralPath '.env')) {
    Copy-Item -LiteralPath '.env.example' -Destination '.env'
    Write-Host 'Created .env from placeholders. Set your local credentials there; they persist across terminals.'
}
if ($Frontend) {
    & npm.cmd --prefix frontend run dev -- --host 127.0.0.1
    exit $LASTEXITCODE
}
if (-not $Python) {
    foreach ($candidate in '.venv-ci/Scripts/python.exe', '.venv/Scripts/python.exe') {
        if (Test-Path -LiteralPath $candidate) {
            try { & $candidate -c 'import sys' 2>$null; if ($LASTEXITCODE -eq 0) { $Python=$candidate; break } } catch {}
        }
    }
}
if (-not $Python) { throw 'No working Python environment. Follow docs/CI_TOOLING.md, then pass -Python <path-to-python.exe>.' }
@'
import dotenv, fastapi, uvicorn, pytest, bandit
import backend.app
print("Startup dependencies and .env loaded; no secrets displayed.")
'@ | & $Python -
if ($LASTEXITCODE -ne 0) { throw 'Install requirements.txt into the selected Python environment first.' }
$env:PATH = "$(Split-Path -Parent (Resolve-Path -LiteralPath $Python));$env:PATH"
& $Python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
exit $LASTEXITCODE
