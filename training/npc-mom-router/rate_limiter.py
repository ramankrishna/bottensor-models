"""
Per-API-key sliding-window rate limiter.

Tracks:
  - RPM  (requests per minute)
  - TPM  (tokens per minute — consumed at request accept time using
          max_tokens + input token estimate; refunded if a request fails
          before reaching upstream)

Sliding windows use timestamp deques (one per key) — O(1) amortized per check.

The limiter is in-process only (single-worker deployment). For multi-worker
scale-out, swap to Redis; the interface stays the same.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

from config import CONFIG

WINDOW_SECONDS = 60.0


class RateLimitExceeded(Exception):
    def __init__(self, retry_after: float, reason: str):
        super().__init__(reason)
        self.retry_after = max(1, int(round(retry_after)))
        self.reason = reason


@dataclass
class TierLimits:
    rpm: int
    tpm: int
    max_context: int


@dataclass
class LimitState:
    # deque of (timestamp_monotonic, token_count)
    #   token_count is 0 for RPM-only entries (but we always store both together)
    events: Deque[Tuple[float, int]]


class RateLimiter:
    def __init__(self):
        self._state: Dict[str, LimitState] = {}
        self._lock = asyncio.Lock()

    def tier_limits(self, tier: str) -> TierLimits:
        cfg = CONFIG["tiers"].get(tier) or CONFIG["tiers"]["free"]
        return TierLimits(
            rpm=cfg["rpm"],
            tpm=cfg["tpm"],
            max_context=cfg["max_context"],
        )

    # ------------------------------------------------------------------
    # Check + reserve
    # ------------------------------------------------------------------
    async def check_and_reserve(
        self,
        key: str,
        tier: str,
        estimated_tokens: int,
    ) -> "ReservationReceipt":
        """
        Atomically verifies that the request fits within RPM/TPM windows and
        reserves `estimated_tokens` tokens + 1 request slot.

        Returns a receipt with remaining counters (for response headers).

        Raises RateLimitExceeded on violation.
        """
        limits = self.tier_limits(tier)
        now = time.monotonic()
        cutoff = now - WINDOW_SECONDS

        async with self._lock:
            state = self._state.get(key)
            if state is None:
                state = LimitState(events=deque())
                self._state[key] = state

            # Evict expired entries
            q = state.events
            while q and q[0][0] < cutoff:
                q.popleft()

            # Current usage in the window
            used_req = len(q)
            used_tok = sum(t for _, t in q)

            # RPM check
            if used_req >= limits.rpm:
                oldest = q[0][0]
                retry_after = max(0.0, WINDOW_SECONDS - (now - oldest))
                raise RateLimitExceeded(
                    retry_after=retry_after,
                    reason=f"rpm-exceeded: {used_req}/{limits.rpm}",
                )

            # TPM check
            if used_tok + estimated_tokens > limits.tpm:
                # Find the earliest event whose expiry frees enough headroom.
                need = (used_tok + estimated_tokens) - limits.tpm
                freed = 0
                retry_at = now
                for ts, tok in q:
                    freed += tok
                    retry_at = ts + WINDOW_SECONDS
                    if freed >= need:
                        break
                retry_after = max(0.0, retry_at - now)
                raise RateLimitExceeded(
                    retry_after=retry_after,
                    reason=f"tpm-exceeded: {used_tok + estimated_tokens}/{limits.tpm}",
                )

            # Reserve
            q.append((now, estimated_tokens))
            return ReservationReceipt(
                limiter=self,
                key=key,
                reservation=(now, estimated_tokens),
                limits=limits,
                used_req_after=used_req + 1,
                used_tok_after=used_tok + estimated_tokens,
                window_reset_at_epoch=int(time.time() + WINDOW_SECONDS),
            )

    async def refund(self, key: str, reservation: Tuple[float, int]) -> None:
        """Remove a previously-inserted reservation (used on upstream failure)."""
        async with self._lock:
            state = self._state.get(key)
            if state is None:
                return
            try:
                state.events.remove(reservation)
            except ValueError:
                pass

    async def adjust_actual(
        self,
        key: str,
        reservation: Tuple[float, int],
        actual_tokens: int,
    ) -> None:
        """
        After upstream returns, adjust the reserved token count to the actual
        usage. Replaces the reservation entry in-place.
        """
        async with self._lock:
            state = self._state.get(key)
            if state is None:
                return
            events = state.events
            for i, entry in enumerate(events):
                if entry == reservation:
                    events[i] = (entry[0], max(0, actual_tokens))
                    return


@dataclass
class ReservationReceipt:
    """Returned to callers so they can emit rate-limit headers."""
    limiter: RateLimiter
    key: str
    reservation: Tuple[float, int]
    limits: TierLimits
    used_req_after: int
    used_tok_after: int
    window_reset_at_epoch: int

    @property
    def remaining_requests(self) -> int:
        return max(0, self.limits.rpm - self.used_req_after)

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.limits.tpm - self.used_tok_after)

    def headers(self) -> Dict[str, str]:
        return {
            "X-RateLimit-Limit-Requests": str(self.limits.rpm),
            "X-RateLimit-Limit-Tokens":   str(self.limits.tpm),
            "X-RateLimit-Remaining-Requests": str(self.remaining_requests),
            "X-RateLimit-Remaining-Tokens":   str(self.remaining_tokens),
            # Keep a single composite header too for simple clients
            "X-RateLimit-Remaining": str(self.remaining_requests),
            "X-RateLimit-Reset": str(self.window_reset_at_epoch),
        }

    async def refund(self) -> None:
        await self.limiter.refund(self.key, self.reservation)

    async def finalize(self, actual_tokens: Optional[int]) -> None:
        if actual_tokens is None:
            return
        await self.limiter.adjust_actual(self.key, self.reservation, actual_tokens)


# Module-level singleton
rate_limiter = RateLimiter()
