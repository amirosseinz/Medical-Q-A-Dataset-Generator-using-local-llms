# Open a shell in a running container
# Usage: .\scripts\shell.ps1 [backend|celery_worker|frontend|redis]
param([string]$Service = "backend")
$ProjectDir = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectDir
Write-Host "[INFO] Opening shell in $Service..." -ForegroundColor Cyan
docker compose exec $Service /bin/sh
Pop-Location
