# Reset everything — stop services, remove volumes, delete data
$ProjectDir = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectDir
$confirm = Read-Host "This will remove all data, containers, and volumes. Are you sure? (y/N)"
if ($confirm -eq "y" -or $confirm -eq "Y") {
    docker compose down -v --remove-orphans
    Write-Host "[INFO] Reset complete." -ForegroundColor Green
} else {
    Write-Host "[INFO] Aborted." -ForegroundColor Yellow
}
Pop-Location
