# Stop all services
$ProjectDir = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectDir
Write-Host "[INFO] Stopping services..." -ForegroundColor Cyan
docker compose down
Write-Host "[INFO] All services stopped." -ForegroundColor Green
Pop-Location
