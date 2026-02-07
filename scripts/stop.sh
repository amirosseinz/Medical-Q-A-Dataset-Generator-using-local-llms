#!/usr/bin/env bash
# Stop all services
set -euo pipefail
cd "$(dirname "$(dirname "${BASH_SOURCE[0]}")")"
echo "[INFO] Stopping services..."
docker compose down
echo "[INFO] All services stopped."
