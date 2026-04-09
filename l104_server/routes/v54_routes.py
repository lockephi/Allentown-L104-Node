"""
v54 Meta-Cognitive Routes — Meta-cognitive monitoring and knowledge bridge

Extracted from app.py during EVO_78 refactoring.
Contains: /api/v54/* endpoints (meta-cognitive, knowledge-bridge)
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, Request

logger = logging.getLogger("L104_FAST")

router = APIRouter(prefix="/api/v54", tags=["v54-meta-cognitive"])

# Import meta-cognitive
try:
    from l104_meta_cognitive import meta_cognitive
    META_COG_AVAILABLE = True
except ImportError:
    meta_cognitive = None
    META_COG_AVAILABLE = False
    logger.warning("⚠️ [META_COG] Meta-cognitive not available")

# Import knowledge bridge
try:
    from l104_knowledge_bridge import knowledge_bridge as kb_bridge
    KB_AVAILABLE = True
except ImportError:
    kb_bridge = None
    KB_AVAILABLE = False
    logger.warning("⚠️ [KB_BRIDGE] Knowledge bridge not available")


# ═══════════════════════════════════════════════════════════════════
#  META-COGNITIVE ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.get("/meta-cognitive/load-balancer")
async def meta_cognitive_load_balancer():
    """Load balancer state and Thompson sampling arms."""
    if not META_COG_AVAILABLE:
        return {"error": "Meta-cognitive not available", "status": "UNAVAILABLE"}
    try:
        return meta_cognitive.load_balancer.get_status() if hasattr(meta_cognitive, 'load_balancer') else {}
    except Exception as e:
        return {"error": str(e)}


@router.get("/meta-cognitive/learning-velocity")
async def meta_cognitive_learning_velocity():
    """Learning velocity and plateau detection report."""
    if not META_COG_AVAILABLE:
        return {"velocity": 0, "is_plateau": False}
    try:
        return meta_cognitive.learning_velocity.get_report() if hasattr(meta_cognitive, 'learning_velocity') else {}
    except Exception as e:
        return {"error": str(e)}


@router.get("/meta-cognitive/strategy-report")
async def meta_cognitive_strategy_report():
    """Thompson sampling strategy optimizer report."""
    if not META_COG_AVAILABLE:
        return {"strategies": {}}
    try:
        return meta_cognitive.strategy_optimizer.get_report() if hasattr(meta_cognitive, 'strategy_optimizer') else {}
    except Exception as e:
        return {"error": str(e)}


@router.get("/meta-cognitive/diagnostics")
async def meta_cognitive_diagnostics():
    """Pipeline diagnostics — cache rates, latency percentiles, bottlenecks."""
    if not META_COG_AVAILABLE:
        return {"diagnostics": {}}
    try:
        return meta_cognitive.diagnostics.diagnose() if hasattr(meta_cognitive, 'diagnostics') else {}
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════
#  KNOWLEDGE BRIDGE ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.get("/knowledge-bridge/status")
async def knowledge_bridge_status():
    """Knowledge bridge status — adapter states, query stats, gap detection."""
    if not KB_AVAILABLE:
        return {"status": "UNAVAILABLE", "error": "KnowledgeBridge module not loaded"}
    try:
        return {"status": "ACTIVE", **kb_bridge.status()}
    except Exception as e:
        return {"error": str(e)}


@router.post("/knowledge-bridge/query")
async def knowledge_bridge_query(req: Request):
    """Query all knowledge stores via the unified bridge."""
    if not KB_AVAILABLE:
        return {"status": "UNAVAILABLE", "results": []}
    try:
        data = await req.json()
    except Exception:
        data = {}

    topic = data.get("topic", data.get("query", ""))
    depth = data.get("depth", 2)

    if not topic:
        return {"status": "ERROR", "error": "topic is required"}

    result = await kb_bridge.query(topic, depth=depth, max_results=20) if hasattr(kb_bridge, 'query') else {}
    return {"status": "SUCCESS", **result}


@router.get("/knowledge-bridge/gaps")
async def knowledge_bridge_gaps():
    """Top knowledge gaps — topics the system lacks knowledge about."""
    if not KB_AVAILABLE:
        return {"gaps": []}

    try:
        gaps = kb_bridge.get_knowledge_gaps(20) if hasattr(kb_bridge, 'get_knowledge_gaps') else []
        return {
            "gaps": [{"topic": t, "miss_count": c} for t, c in gaps],
            "miss_rate": round(kb_bridge.gap_detector.get_miss_rate(), 4) if hasattr(kb_bridge, 'gap_detector') else 0.0,
        }
    except Exception as e:
        return {"error": str(e), "gaps": []}


__all__ = ['router']