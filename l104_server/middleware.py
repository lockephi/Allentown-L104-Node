# L104 Server — FastAPI Middleware
# EVO_61: Rate limiting and request metrics

import os
import time
import logging
from collections import defaultdict
from typing import Callable

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# Rate limiting state (shared with state.py)
_rate_limit_last_prune = 0.0


async def rate_limit_middleware(request: Request, call_next: Callable) -> JSONResponse:
    """Rate limit middleware with automatic stale IP pruning.

    - Limits requests per IP within a time window
    - Prunes stale IPs every 60s to prevent unbounded dict growth
    - Tracks request metrics for monitoring
    """
    from state import app_metrics, rate_limit_store
    from config import RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW, DISABLE_RATE_LIMIT_ENV

    global _rate_limit_last_prune
    now = time.time()

    # Prune stale IPs every 60s
    if now - _rate_limit_last_prune > 60:
        _rate_limit_last_prune = now
        stale = [ip for ip, ts in rate_limit_store.items() if not ts or ts[-1] < now - RATE_LIMIT_WINDOW * 2]
        for ip in stale:
            del rate_limit_store[ip]

    # Check rate limit if not disabled
    if not os.getenv(DISABLE_RATE_LIMIT_ENV):
        client_ip = request.client.host if request.client else "unknown"
        window_start = now - RATE_LIMIT_WINDOW
        # Initialize if not present
        if client_ip not in rate_limit_store:
            rate_limit_store[client_ip] = []
        rate_limit_store[client_ip] = [t for t in rate_limit_store[client_ip] if t > window_start]
        if len(rate_limit_store[client_ip]) >= RATE_LIMIT_REQUESTS:
            return JSONResponse(status_code=429, content={"error": "Rate limit exceeded"})
        rate_limit_store[client_ip].append(now)

    # Track metrics
    app_metrics["requests_total"] += 1
    response = await call_next(request)
    if response.status_code < 400:
        app_metrics["requests_success"] += 1
    else:
        app_metrics["requests_error"] += 1

    return response


__all__ = ["rate_limit_middleware"]