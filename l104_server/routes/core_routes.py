"""
Core Routes — Root-level endpoints (favicon, landing, health, chat, synergy)

Extracted from app.py during EVO_78 refactoring.
Contains: /favicon.ico, /landing, /, /health, /metrics, /chat, /self/heal
"""

import time
import logging
from datetime import datetime, timezone
from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse

logger = logging.getLogger("L104_FAST")

router = APIRouter(tags=["core"])

# Import intellect
try:
    from l104_server.learning import intellect
    INTELLECT_AVAILABLE = True
except ImportError:
    intellect = None
    INTELLECT_AVAILABLE = False
    logger.warning("⚠️ [CORE] Intellect not available")

# Import constants
try:
    from l104_server.constants import FAST_SERVER_VERSION, VOID_CONSTANT, ZENITH_HZ, UUC
except ImportError:
    FAST_SERVER_VERSION = "5.0.0"
    VOID_CONSTANT = 1.0416180339887497
    ZENITH_HZ = 528.0
    UUC = 104.0


# ═══════════════════════════════════════════════════════════════════
#  STATIC/LANDING ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.get("/favicon.ico")
async def favicon():
    """Serve favicon."""
    return {"status": "ok"}


@router.get("/landing")
async def landing():
    """Serve landing page data."""
    return {
        "status": "ONLINE",
        "version": FAST_SERVER_VERSION,
        "features": ["quantum", "consciousness", "learning", "agents"],
        "resonance": intellect.current_resonance if INTELLECT_AVAILABLE else 1.0
    }


@router.get("/")
async def root():
    """Root endpoint - server status."""
    return {
        "status": "ONLINE",
        "version": FAST_SERVER_VERSION,
        "mode": "SOVEREIGN_FAST_LEARNING",
        "constants": {
            "GOD_CODE": 527.5184818492612,
            "PHI": 1.618033988749895,
            "VOID_CONSTANT": VOID_CONSTANT
        }
    }


@router.get("/market")
async def market():
    """Market status endpoint."""
    return {"status": "ACTIVE", "market": "SOVEREIGN"}


@router.get("/intricate")
async def intricate_root():
    """Intricate UI root endpoint."""
    return {"status": "ACTIVE", "ui_engine": "V1.0"}


@router.get("/intricate/{subpath:path}")
async def intricate_subpath(subpath: str):
    """Intricate UI subpath endpoint."""
    return {"status": "ACTIVE", "subpath": subpath}


@router.get("/WHITE_PAPER.md")
async def white_paper():
    """Serve white paper."""
    return {"status": "See docs/WHITE_PAPER.md"}


# ═══════════════════════════════════════════════════════════════════
#  HEALTH ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "HEALTHY",
        "version": FAST_SERVER_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get("/health/resilience")
async def health_resilience():
    """Resilience health check."""
    try:
        from l104_resilience import get_resilience_report
        report = get_resilience_report()
        return {"status": "HEALTHY", "resilience": report}
    except ImportError:
        return {"status": "HEALTHY", "resilience": {"available": False}}
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}


# ═══════════════════════════════════════════════════════════════════
#  SELF-HEALING
# ═══════════════════════════════════════════════════════════════════

@router.post("/self/heal")
async def self_heal():
    """Trigger self-healing cycle."""
    try:
        # Basic self-healing: clear caches, reset state
        if INTELLECT_AVAILABLE and hasattr(intellect, 'clear_cache'):
            intellect.clear_cache()
        return {"status": "HEALED", "timestamp": datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}


# ═══════════════════════════════════════════════════════════════════
#  METRICS ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.get("/metrics")
async def metrics():
    """Get server metrics."""
    if not INTELLECT_AVAILABLE:
        return {"status": "UNAVAILABLE", "error": "Intellect not available"}

    try:
        stats = intellect.get_stats() if hasattr(intellect, 'get_stats') else {}
        return {
            "status": "ACTIVE",
            "metrics": {
                "memories": stats.get("memories", 0),
                "clusters": stats.get("clusters", 0),
                "skills": stats.get("skills", 0),
                "resonance": intellect.current_resonance if hasattr(intellect, 'current_resonance') else 1.0
            }
        }
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}


@router.get("/system/capacity")
async def system_capacity():
    """Get system capacity."""
    import os
    import platform

    try:
        return {
            "status": "ACTIVE",
            "capacity": {
                "cpu_count": os.cpu_count() or 1,
                "platform": platform.platform(),
                "machine": platform.machine()
            }
        }
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}


# ═══════════════════════════════════════════════════════════════════
#  CHAT ENDPOINT
# ═══════════════════════════════════════════════════════════════════

@router.post("/api/v6/chat")
async def chat_v6(request: Request):
    """Chat endpoint - v6 API."""
    if not INTELLECT_AVAILABLE:
        return {"error": "Intellect not available", "status": "UNAVAILABLE"}

    try:
        data = await request.json()
        message = data.get("message", data.get("query", ""))
        context = data.get("context", {})

        # Process through intellect
        response = intellect.process(message, context) if hasattr(intellect, 'process') else message

        return {
            "status": "SUCCESS",
            "response": response,
            "resonance": intellect.current_resonance if hasattr(intellect, 'current_resonance') else 1.0
        }
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}


# ═══════════════════════════════════════════════════════════════════
#  SYNERGY ENDPOINT
# ═══════════════════════════════════════════════════════════════════

@router.post("/api/v10/synergy/execute")
async def synergy_execute(request: Request):
    """Execute synergy operation - AI-powered task execution."""
    if not INTELLECT_AVAILABLE:
        return {"error": "Intellect not available", "status": "UNAVAILABLE"}

    try:
        data = await request.json()
        task = data.get("task", "")

        if not task:
            return {"status": "ERROR", "error": "No task provided"}

        # Boost resonance for complex synergy
        if hasattr(intellect, 'boost_resonance'):
            intellect.boost_resonance(0.2)

        logger.info(f"⚡ [SYNERGY] Executing Sovereign Task: {task[:50]}...")

        # Process through intellect
        context = f"Execute this task with full capability: {task}"
        response = intellect.process(context) if hasattr(intellect, 'process') else f"Task: {task}"

        return {
            "status": "SUCCESS",
            "result": response,
            "task": task,
            "resonance": intellect.current_resonance if hasattr(intellect, 'current_resonance') else 1.0
        }
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}


__all__ = ['router']