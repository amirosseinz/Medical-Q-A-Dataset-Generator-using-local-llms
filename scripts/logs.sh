#!/usr/bin/env bash
# Follow logs from all services
cd "$(dirname "$(dirname "${BASH_SOURCE[0]}")")"
docker compose logs -f "$@"
