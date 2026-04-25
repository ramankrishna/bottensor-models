"""
NPC Mixture of Models — Configuration
Bottensor API | dude.npc
"""
import os
from dotenv import load_dotenv

load_dotenv()


CONFIG = {
    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    "npc_fast_model": os.getenv("NPC_FAST_MODEL", "ramankrishna10/npc-fast-1.7b"),
    "npc_fin_model":  os.getenv("NPC_FIN_MODEL",  "ramankrishna10/npc-fin-32b-sft"),

    # Internal vLLM endpoints (docker-compose hostnames by default)
    "npc_fast_url": os.getenv("NPC_FAST_URL", "http://localhost:8001"),
    "npc_fin_url":  os.getenv("NPC_FIN_URL",  "http://localhost:8002"),

    # Upstream API keys (forwarded as `Authorization: Bearer <key>` on each
    # proxied request). Set one or both when the underlying vLLM servers were
    # launched with `--api-key`. Empty strings disable the header.
    "npc_fast_upstream_key": os.getenv("NPC_FAST_UPSTREAM_KEY", ""),
    "npc_fin_upstream_key":  os.getenv("NPC_FIN_UPSTREAM_KEY",  ""),

    # External MoM router API
    "api_port": int(os.getenv("API_PORT", "8000")),
    "api_host": os.getenv("API_HOST", "0.0.0.0"),

    # ------------------------------------------------------------------
    # MongoDB (API-key auth)
    # ------------------------------------------------------------------
    "mongo_uri":        os.getenv("MONGO_URI", "mongodb://localhost:27017"),
    "mongo_db":         os.getenv("MONGO_DB", "bottensor"),
    "mongo_collection": os.getenv("MONGO_COLLECTION", "api_keys"),
    "auth_cache_ttl_s": int(os.getenv("AUTH_CACHE_TTL_S", "300")),   # 5 min
    # For local dev: if MONGO_URI is unreachable, allow a sentinel test key
    "dev_test_key":     os.getenv("DEV_TEST_KEY", "test-key"),
    "dev_test_tier":    os.getenv("DEV_TEST_TIER", "enterprise"),
    "dev_test_owner":   os.getenv("DEV_TEST_OWNER", "dev@bottensor.ai"),
    "allow_dev_key":    os.getenv("ALLOW_DEV_KEY", "false").lower() == "true",

    # ------------------------------------------------------------------
    # HuggingFace
    # ------------------------------------------------------------------
    "hf_token": os.getenv("HF_TOKEN"),

    # ------------------------------------------------------------------
    # Timeouts (seconds)
    # ------------------------------------------------------------------
    "router_timeout_s":     float(os.getenv("ROUTER_TIMEOUT_S", "30")),
    "generation_timeout_s": float(os.getenv("GENERATION_TIMEOUT_S", "120")),
    "healthcheck_timeout_s": float(os.getenv("HEALTHCHECK_TIMEOUT_S", "5")),

    # Router call should be sub-200ms at SLO; guard with a tight budget
    "router_soft_budget_ms": int(os.getenv("ROUTER_SOFT_BUDGET_MS", "200")),

    # ------------------------------------------------------------------
    # Model name mapping  (public API name → internal disposition)
    # ------------------------------------------------------------------
    "model_map": {
        "npc":         "mom",          # Mixture of Models routing
        "npc-fast":    "direct_fast",
        "npc-fin-32b": "direct_fin",   # BACKWARD COMPATIBLE — existing prod name
    },

    # ------------------------------------------------------------------
    # Rate-limit tiers
    # ------------------------------------------------------------------
    "tiers": {
        "free":       {"rpm": 10,  "tpm": 10_000,    "max_context": 4096},
        "pro":        {"rpm": 60,  "tpm": 100_000,   "max_context": 16384},
        "enterprise": {"rpm": 300, "tpm": 1_000_000, "max_context": 16384},
    },

    # ------------------------------------------------------------------
    # Branding
    # ------------------------------------------------------------------
    "brand": {
        "api_name":    "Bottensor API",
        "company":     "Bottensor (a Falcon Hash company)",
        "creator":     "dude.npc",
        "owned_by":    "bottensor",
    },

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    "log_file": os.getenv("LOG_FILE", "/var/log/npc-mom/requests.log"),
    "log_level": os.getenv("LOG_LEVEL", "INFO"),
}


# Expose the public model list in a stable order (used by /v1/models)
PUBLIC_MODELS = ["npc", "npc-fast", "npc-fin-32b"]
