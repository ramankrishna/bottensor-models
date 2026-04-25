#!/usr/bin/env bash
set -u

AUTH_KEY="${AUTH_KEY:-test-key}"
ROUTER_URL="${ROUTER_URL:-http://localhost:8000}"

pass() { printf "  \033[32m✓\033[0m %s\n" "$1"; }
fail() { printf "  \033[31m✗\033[0m %s\n" "$1"; FAIL=1; }
FAIL=0

echo "=== NPC MoM health check ==="
echo ""

echo "[1/5] NPC Fast upstream (port 8001)"
if curl -sf http://localhost:8001/health >/dev/null 2>&1; then
    pass "npc-fast healthy"
else
    fail "npc-fast unreachable or unhealthy"
fi

echo ""
echo "[2/5] NPC Fin upstream (port 8002)"
if curl -sf http://localhost:8002/health >/dev/null 2>&1; then
    pass "npc-fin healthy"
else
    fail "npc-fin unreachable or unhealthy"
fi

echo ""
echo "[3/5] Router (port 8000)"
if curl -sf "$ROUTER_URL/health" >/dev/null 2>&1; then
    pass "router healthy"
    curl -s "$ROUTER_URL/health" | python3 -m json.tool || true
else
    fail "router unreachable"
fi

echo ""
echo "[4/5] Auto-route (model=npc, simple math → expect npc-fast)"
resp=$(curl -sS -D /tmp/npc_hdrs1.txt "$ROUTER_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${AUTH_KEY}" \
    -d '{
      "model": "npc",
      "messages": [{"role": "user", "content": "What is 2+2?"}],
      "max_tokens": 50,
      "temperature": 0
    }')
echo "$resp" | python3 -m json.tool || { fail "auto-route response not JSON"; echo "$resp"; }
echo "  headers:"
grep -E "^(X-NPC|X-Request|X-RateLimit)" /tmp/npc_hdrs1.txt || true

echo ""
echo "[5/5] Direct npc-fin-32b (backward compat)"
resp=$(curl -sS -D /tmp/npc_hdrs2.txt "$ROUTER_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${AUTH_KEY}" \
    -d '{
      "model": "npc-fin-32b",
      "messages": [{"role": "user", "content": "Briefly outline a DCF valuation."}],
      "max_tokens": 100,
      "temperature": 0.2
    }')
echo "$resp" | python3 -m json.tool || { fail "npc-fin-32b response not JSON"; echo "$resp"; }
echo "  headers:"
grep -E "^(X-NPC|X-Request|X-RateLimit)" /tmp/npc_hdrs2.txt || true

echo ""
if (( FAIL == 0 )); then
    echo "=== ALL CHECKS PASSED ==="
    exit 0
else
    echo "=== SOME CHECKS FAILED ==="
    exit 1
fi
