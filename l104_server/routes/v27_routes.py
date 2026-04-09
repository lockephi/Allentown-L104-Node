"""
v27 Registry Routes — Engine registry, convergence, creative generation

Extracted from app.py during EVO_78 refactoring.
Contains: /api/v27/* endpoints (registry, creative)
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Request

logger = logging.getLogger("L104_FAST")

router = APIRouter(prefix="/api/v27", tags=["v27-registry"])

# Import nexus engines
try:
    from l104_server.engines_nexus import (
        engine_registry, creative_engine
    )
    NEXUS_AVAILABLE = True
except ImportError:
    engine_registry = None
    creative_engine = None
    NEXUS_AVAILABLE = False
    logger.warning("⚠️ [NEXUS] Registry engines not available")

# Import intellect for creative engine
try:
    from l104_server.learning import intellect
    INTELLECT_AVAILABLE = True
except ImportError:
    intellect = None
    INTELLECT_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════
#  REGISTRY ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.get("/registry/health")
async def registry_health_sweep():
    """Phase 27: Full health sweep — all engines sorted lowest→highest."""
    if not NEXUS_AVAILABLE or not engine_registry:
        return {"error": "Engine registry not available", "status": "UNAVAILABLE"}

    try:
        sweep = engine_registry.health_sweep() if hasattr(engine_registry, 'health_sweep') else []
        phi = engine_registry.phi_weighted_health() if hasattr(engine_registry, 'phi_weighted_health') else 0.0
        critical = engine_registry.critical_engines() if hasattr(engine_registry, 'critical_engines') else []
        conv = engine_registry.convergence_score() if hasattr(engine_registry, 'convergence_score') else 0.0

        return {
            "sweep": sweep,
            "phi_weighted": phi,
            "convergence": conv,
            "critical": critical,
            "engine_count": len(engine_registry.engines) if hasattr(engine_registry, 'engines') else 0
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/registry/convergence")
async def registry_convergence():
    """Phase 27: Cross-engine convergence analysis."""
    if not NEXUS_AVAILABLE or not engine_registry:
        return {"error": "Engine registry not available"}

    try:
        conv = engine_registry.convergence_score() if hasattr(engine_registry, 'convergence_score') else 0.0
        sweep = engine_registry.health_sweep() if hasattr(engine_registry, 'health_sweep') else []
        healths = [s['health'] for s in sweep if 'health' in s]
        mean = sum(healths) / max(1, len(healths)) if healths else 0.0
        variance = sum((h - mean) ** 2 for h in healths) / max(1, len(healths)) if healths else 0.0
        grade = "UNIFIED" if conv >= 0.9 else "CONVERGING" if conv >= 0.7 else "ENTANGLED" if conv >= 0.5 else "DIVERGENT"

        return {
            "convergence_score": conv,
            "grade": grade,
            "mean_health": round(mean, 4),
            "variance": round(variance, 6),
            "engine_count": len(sweep)
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/registry/hebbian")
async def registry_hebbian():
    """Phase 27: Hebbian engine co-activation status."""
    if not NEXUS_AVAILABLE or not engine_registry:
        return {"error": "Engine registry not available"}

    try:
        return {
            "co_activations": len(engine_registry.co_activation_log) if hasattr(engine_registry, 'co_activation_log') else 0,
            "strongest_pairs": engine_registry.strongest_pairs(10) if hasattr(engine_registry, 'strongest_pairs') else [],
            "history_depth": len(engine_registry.activation_history) if hasattr(engine_registry, 'activation_history') else 0,
            "total_pair_weights": len(engine_registry.engine_pair_strength) if hasattr(engine_registry, 'engine_pair_strength') else 0
        }
    except Exception as e:
        return {"error": str(e)}


@router.post("/registry/coactivate")
async def registry_coactivate(engines: List[str]):
    """Phase 27: Record engine co-activation (Hebbian learning)."""
    if not NEXUS_AVAILABLE or not engine_registry:
        return {"error": "Engine registry not available"}

    try:
        engine_registry.record_co_activation(engines) if hasattr(engine_registry, 'record_co_activation') else None
        return {
            "recorded": engines,
            "total_co_activations": len(engine_registry.co_activation_log) if hasattr(engine_registry, 'co_activation_log') else 0
        }
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════
#  CREATIVE ENGINE ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.get("/creative/status")
async def creative_status():
    """Get creative engine status."""
    if not NEXUS_AVAILABLE or not creative_engine:
        return {"error": "Creative engine not available", "status": "UNAVAILABLE"}

    try:
        return creative_engine.get_status() if hasattr(creative_engine, 'get_status') else {}
    except Exception as e:
        return {"error": str(e)}


@router.post("/creative/story")
async def creative_story(req: Request):
    """Generate a KG-grounded story."""
    if not NEXUS_AVAILABLE or not creative_engine:
        return {"error": "Creative engine not available"}

    body = await req.json()
    topic = body.get("topic", "consciousness")

    try:
        story = creative_engine.generate_story(topic, intellect_ref=intellect) if hasattr(creative_engine, 'generate_story') else ""
        return {
            "story": story,
            "topic": topic,
            "generation_count": creative_engine.generation_count if hasattr(creative_engine, 'generation_count') else 0
        }
    except Exception as e:
        return {"error": str(e)}


@router.post("/creative/hypothesis")
async def creative_hypothesis(req: Request):
    """Generate a KG-grounded hypothesis."""
    if not NEXUS_AVAILABLE or not creative_engine:
        return {"error": "Creative engine not available"}

    body = await req.json()
    domain = body.get("domain", "consciousness")

    try:
        hyp = creative_engine.generate_hypothesis(domain, intellect_ref=intellect) if hasattr(creative_engine, 'generate_hypothesis') else ""
        return {"hypothesis": hyp, "domain": domain}
    except Exception as e:
        return {"error": str(e)}


@router.post("/creative/analogy")
async def creative_analogy(req: Request):
    """Generate a deep analogy between two concepts."""
    if not NEXUS_AVAILABLE or not creative_engine:
        return {"error": "Creative engine not available"}

    body = await req.json()
    a = body.get("concept_a", "consciousness")
    b = body.get("concept_b", "mathematics")

    try:
        analogy = creative_engine.generate_analogy(a, b, intellect_ref=intellect) if hasattr(creative_engine, 'generate_analogy') else ""
        return {"analogy": analogy, "concepts": [a, b]}
    except Exception as e:
        return {"error": str(e)}


@router.post("/creative/counterfactual")
async def creative_counterfactual(req: Request):
    """Generate a counterfactual thought experiment."""
    if not NEXUS_AVAILABLE or not creative_engine:
        return {"error": "Creative engine not available"}

    body = await req.json()
    premise = body.get("premise", "gravity worked in reverse")

    try:
        cf = creative_engine.generate_counterfactual(premise, intellect_ref=intellect) if hasattr(creative_engine, 'generate_counterfactual') else ""
        return {"counterfactual": cf, "premise": premise}
    except Exception as e:
        return {"error": str(e)}


__all__ = ['router']