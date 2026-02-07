#!/usr/bin/env bash
# Reset everything — stop services, remove volumes, delete data
set -euo pipefail
cd "$(dirname "$(dirname "${BASH_SOURCE[0]}")")"
echo "[WARN] This will remove all data, containers, and volumes."
read -p "Are you sure? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker compose down -v --remove-orphans
    echo "[INFO] Reset complete."
else
    echo "[INFO] Aborted."
fi
