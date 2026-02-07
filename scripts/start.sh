#!/usr/bin/env bash
# Start all services via Docker Compose
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Create .env from example if it doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
    echo "[INFO] Created .env from .env.example — review and update before first run."
fi

echo "[INFO] Building and starting services..."
docker compose up --build -d

echo ""
echo "========================================"
echo "  Medical Q&A Dataset Generator"
echo "========================================"
echo "  Frontend:  http://localhost:3000"
echo "  API:       http://localhost:8000"
echo "  API Docs:  http://localhost:8000/docs"
echo "========================================"
echo ""
echo "Run 'docker compose logs -f' to follow logs."
