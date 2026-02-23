# L104 Sovereign Node — Shared Runtime State
# Mutable singletons shared across routers. Import from here, never re-create.

from collections import defaultdict
from datetime import datetime
from typing import Optional
from config import UTC

import httpx

# ─── Request / Response Counters ─────────────────────────────────────────────
app_metrics: dict = {
    "requests_total": 0,
    "requests_success": 0,
    "requests_error": 0,
    "api_calls": 0,
    "upstream_429": 0,
    "upstream_5xx": 0,
    "uptime_start": datetime.now(UTC),
}

rate_limit_store: defaultdict = defaultdict(list)
responder_counts: defaultdict = defaultdict(int)

# ─── HTTP Client (shared, lazy-initialized) ───────────────────────────────────
_http_client: Optional[httpx.AsyncClient] = None


async def get_http_client() -> httpx.AsyncClient:
    """Return (or create) the shared async HTTP client."""
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0))
    return _http_client


async def close_http_client() -> None:
    """Close the shared HTTP client on shutdown."""
    global _http_client
    if _http_client:
        await _http_client.aclose()
        _http_client = None


# ─── Model-Rotation State (used by streaming / ai router) ────────────────────
_current_model_index: int = 0
_model_cooldowns: dict = {}        # model_name → cooldown_end_time (float)
_quota_exhausted_until: float = 0  # global quota cooldown timestamp
_consecutive_quota_failures: int = 0
