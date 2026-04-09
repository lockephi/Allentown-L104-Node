"""
v14 Stats Routes — Stats, consolidate, intellect status endpoints

Extracted from app.py during EVO_78 refactoring.
Contains: /api/v14/stats, /api/v14/consolidate, /api/v14/intellect endpoints
"""

import time
import logging
from typing import Dict, Any
from fastapi import APIRouter, BackgroundTasks

logger = logging.getLogger("L104_FAST")

router = APIRouter(prefix="/api/v14", tags=["v14-stats"])

# Import intellect
try:
    from l104_server.learning import intellect
    INTELLECT_AVAILABLE = True
except ImportError:
    intellect = None
    INTELLECT_AVAILABLE = False
    logger.warning("⚠️ [STATS] Intellect not available")

# Import performance metrics
try:
    from l104_server.engines_infra import performance_metrics
except ImportError:
    performance_metrics = None

# Start time for uptime calculation
START_TIME = time.time()


# ═══════════════════════════════════════════════════════════════════
#  INTELLECT STATUS
# ═══════════════════════════════════════════════════════════════════

async def _get_intellect_status():
    """Core intellect status logic."""
    if not INTELLECT_AVAILABLE:
        return {"error": "Intellect not available", "status": "UNAVAILABLE"}
    try:
        stats = intellect.get_stats() if hasattr(intellect, 'get_stats') else {}
        return {
            "status": "ACTIVE",
            "resonance": intellect.current_resonance if hasattr(intellect, 'current_resonance') else 1.0,
            "stats": stats
        }
    except Exception as e:
        return {"error": str(e), "status": "ERROR"}


@router.get("/intellect")
async def get_intellect_v14():
    """Return intellect status via v14 API."""
    return await _get_intellect_status()


# ═══════════════════════════════════════════════════════════════════
#  CONSOLIDATE
# ═══════════════════════════════════════════════════════════════════

async def _trigger_consolidate(background_tasks: BackgroundTasks):
    """Core consolidation logic."""
    if not INTELLECT_AVAILABLE:
        return {"error": "Intellect not available", "status": "UNAVAILABLE"}

    logger.info("🧠 [API] Consolidation triggered via API.")
    background_tasks.add_task(intellect.consolidate)
    return {
        "status": "SUCCESS",
        "message": "Consolidation initiated",
        "resonance": intellect.current_resonance if hasattr(intellect, 'current_resonance') else 1.0
    }


@router.post("/consolidate")
async def consolidate_v14(background_tasks: BackgroundTasks):
    """Trigger intellect consolidation via v14 API."""
    return await _trigger_consolidate(background_tasks)


# ═══════════════════════════════════════════════════════════════════
#  STATS
# ═══════════════════════════════════════════════════════════════════

async def _get_stats():
    """Core stats logic."""
    if not INTELLECT_AVAILABLE:
        return {"error": "Intellect not available", "status": "UNAVAILABLE"}

    try:
        stats = intellect.get_stats() if hasattr(intellect, 'get_stats') else {}
        perf = performance_metrics.get_performance_report() if performance_metrics and hasattr(performance_metrics, 'get_performance_report') else {}
        return {
            "status": "SUCCESS",
            "intellect": stats,
            "performance": perf,
            "resonance": intellect.current_resonance if hasattr(intellect, 'current_resonance') else 1.0,
            "uptime": time.time() - START_TIME
        }
    except Exception as e:
        return {"error": str(e), "status": "ERROR"}


@router.get("/stats")
async def stats_v14():
    """Return intellect stats via v14 API."""
    return await _get_stats()


__all__ = ['router']