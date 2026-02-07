# Start all services via Docker Compose
$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectDir

# Create .env from example if it doesn't exist
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "[INFO] Created .env from .env.example - review and update before first run." -ForegroundColor Yellow
}

Write-Host "[INFO] Building and starting services..." -ForegroundColor Cyan
docker compose up --build -d

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Medical Q&A Dataset Generator" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Frontend:  http://localhost:3000" -ForegroundColor White
Write-Host "  API:       http://localhost:8000" -ForegroundColor White
Write-Host "  API Docs:  http://localhost:8000/docs" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Run 'docker compose logs -f' to follow logs."

Pop-Location
