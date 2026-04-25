"""
MongoDB-backed API-key authentication with in-process TTL cache.

Cache semantics:
  - Validated keys are cached for `auth_cache_ttl_s` seconds.
  - Cache stores (owner, tier, active) so rate-limiter can read tier without
    re-querying Mongo.
  - Negative lookups (key not found / inactive) are cached for 30s to throttle
    brute-force attempts without being user-hostile.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from pymongo import MongoClient
from pymongo.errors import PyMongoError

from config import CONFIG

log = logging.getLogger("npc-mom")


@dataclass(frozen=True)
class KeyInfo:
    owner: str
    tier: str
    active: bool


class AuthError(Exception):
    """Raised when an API key is invalid, inactive, or missing."""

    def __init__(self, message: str, status: int = 401):
        super().__init__(message)
        self.status = status


class AuthService:
    """
    Thread-safe wrapper around Mongo key lookup.

    pymongo is synchronous, so calls are run in the default thread-pool via
    `asyncio.to_thread`. This keeps the event loop responsive under load.
    """

    _NEGATIVE_TTL_S = 30.0

    def __init__(self):
        self._client: Optional[MongoClient] = None
        self._cache: Dict[str, Tuple[float, Optional[KeyInfo]]] = {}
        self._lock = asyncio.Lock()
        self._ttl = CONFIG["auth_cache_ttl_s"]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def connect(self) -> None:
        if self._client is not None:
            return
        try:
            self._client = MongoClient(
                CONFIG["mongo_uri"],
                serverSelectionTimeoutMS=3000,
                connectTimeoutMS=3000,
            )
            # Touch the server to fail fast if unreachable
            self._client.admin.command("ping")
            log.info("mongo-connected", extra={"uri": _safe_uri(CONFIG["mongo_uri"])})
        except PyMongoError as exc:
            log.warning("mongo-unreachable", extra={"error": str(exc)})
            self._client = None

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    async def validate(self, api_key: Optional[str]) -> KeyInfo:
        """
        Returns a KeyInfo on success. Raises AuthError otherwise.
        """
        if not api_key:
            raise AuthError("Missing Authorization: Bearer <key>")

        # Dev key shortcut (only if explicitly allowed)
        if CONFIG["allow_dev_key"] and api_key == CONFIG["dev_test_key"]:
            return KeyInfo(
                owner=CONFIG["dev_test_owner"],
                tier=CONFIG["dev_test_tier"],
                active=True,
            )

        # Cache hit?
        now = time.monotonic()
        cached = self._cache.get(api_key)
        if cached is not None:
            expires_at, info = cached
            if expires_at > now:
                if info is None or not info.active:
                    raise AuthError("Invalid or inactive API key")
                return info

        # Cache miss → Mongo lookup
        info = await asyncio.to_thread(self._lookup, api_key)
        ttl = self._ttl if info is not None and info.active else self._NEGATIVE_TTL_S
        async with self._lock:
            self._cache[api_key] = (now + ttl, info)

        if info is None or not info.active:
            raise AuthError("Invalid or inactive API key")
        return info

    def invalidate(self, api_key: str) -> None:
        """Remove a key from the cache (e.g. after admin revocation)."""
        self._cache.pop(api_key, None)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _lookup(self, api_key: str) -> Optional[KeyInfo]:
        if self._client is None:
            # Try to (re)connect lazily. If Mongo is down, fail closed.
            self.connect()
        if self._client is None:
            log.error("mongo-lookup-failed", extra={"reason": "no-connection"})
            return None

        try:
            coll = self._client[CONFIG["mongo_db"]][CONFIG["mongo_collection"]]
            doc = coll.find_one(
                {"key": api_key},
                {"owner": 1, "tier": 1, "active": 1, "_id": 0},
            )
        except PyMongoError as exc:
            log.error("mongo-lookup-error", extra={"error": str(exc)})
            return None

        if not doc:
            return None

        tier = doc.get("tier", "free")
        if tier not in CONFIG["tiers"]:
            tier = "free"

        return KeyInfo(
            owner=str(doc.get("owner", "unknown")),
            tier=tier,
            active=bool(doc.get("active", True)),
        )


def _safe_uri(uri: str) -> str:
    """Redact credentials from a Mongo URI for logging."""
    if "@" in uri and "://" in uri:
        scheme, rest = uri.split("://", 1)
        _, host = rest.split("@", 1)
        return f"{scheme}://***@{host}"
    return uri


# Module-level singleton
auth_service = AuthService()
