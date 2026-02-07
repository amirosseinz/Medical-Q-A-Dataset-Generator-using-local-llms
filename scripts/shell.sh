#!/usr/bin/env bash
# Open a shell in a running container
# Usage: ./scripts/shell.sh [backend|celery_worker|frontend|redis]
set -euo pipefail
cd "$(dirname "$(dirname "${BASH_SOURCE[0]}")")"
SERVICE="${1:-backend}"
echo "[INFO] Opening shell in $SERVICE..."
docker compose exec "$SERVICE" /bin/sh
