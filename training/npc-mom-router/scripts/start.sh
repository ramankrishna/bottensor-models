#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== NPC Mixture of Models ==="
echo "Starting NPC Fast 1.7B on port 8001..."
echo "Starting NPC Fin 32B  on port 8002..."
echo "Starting MoM Router  on port 8000..."
echo ""

# Choose compose files — FALLBACK=1 swaps vLLM npc-fin for the transformers worker
COMPOSE_FILES=(-f docker-compose.yml)
if [[ "${FALLBACK:-0}" == "1" ]]; then
    echo "(transformers fallback enabled for npc-fin)"
    COMPOSE_FILES+=(-f docker-compose.fallback.yml)
fi

docker compose "${COMPOSE_FILES[@]}" up -d

echo ""
echo "Waiting for models to load (cold start: 2–5 min for npc-fast, 5–10 min for npc-fin)..."
echo ""

# Wait for NPC Fast
deadline=$(( $(date +%s) + 900 ))   # 15 min budget
while ! curl -sf http://localhost:8001/health >/dev/null 2>&1; do
    if (( $(date +%s) > deadline )); then
        echo "  ✗ NPC Fast did not come up within 15 minutes"
        docker compose "${COMPOSE_FILES[@]}" logs --tail=80 npc-fast
        exit 1
    fi
    echo "  NPC Fast loading..."
    sleep 10
done
echo "  ✓ NPC Fast loaded"

# Wait for NPC Fin
deadline=$(( $(date +%s) + 1800 ))  # 30 min budget (32B in 4-bit takes a while)
while ! curl -sf http://localhost:8002/health >/dev/null 2>&1; do
    if (( $(date +%s) > deadline )); then
        echo "  ✗ NPC Fin did not come up within 30 minutes"
        docker compose "${COMPOSE_FILES[@]}" logs --tail=120 npc-fin
        exit 1
    fi
    echo "  NPC Fin loading..."
    sleep 15
done
echo "  ✓ NPC Fin loaded"

# Wait for router (short wait — it's ready the moment uvicorn binds)
deadline=$(( $(date +%s) + 60 ))
while ! curl -sf http://localhost:8000/health >/dev/null 2>&1; do
    if (( $(date +%s) > deadline )); then
        echo "  ✗ Router did not come up within 60 seconds"
        docker compose "${COMPOSE_FILES[@]}" logs --tail=80 router
        exit 1
    fi
    echo "  Router starting..."
    sleep 2
done
echo "  ✓ Router ready"

echo ""
echo "=== NPC MoM is LIVE ==="
echo ""
echo "Endpoint: http://localhost:8000/v1/chat/completions"
echo ""
echo "Models:"
echo "  npc         — Auto-routing (NPC Fast decides)"
echo "  npc-fast    — Direct to NPC Fast 1.7B"
echo "  npc-fin-32b — Direct to NPC Fin 32B (backward compatible)"
echo ""
echo "Health:  http://localhost:8000/health"
echo "Models:  http://localhost:8000/v1/models"
