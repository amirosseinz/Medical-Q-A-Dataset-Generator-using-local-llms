# Follow logs from all services
$ProjectDir = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectDir
docker compose logs -f @args
Pop-Location
