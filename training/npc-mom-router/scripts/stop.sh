#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "Stopping NPC MoM..."

COMPOSE_FILES=(-f docker-compose.yml)
if [[ "${FALLBACK:-0}" == "1" ]]; then
    COMPOSE_FILES+=(-f docker-compose.fallback.yml)
fi

docker compose "${COMPOSE_FILES[@]}" down

echo "All services stopped."
