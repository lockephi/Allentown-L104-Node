"""
v63 Monitoring Routes — Temporal coherence, fitness, entropy, phase navigation

Extracted from app.py during EVO_78 refactoring.
Contains: /api/v63/* endpoints (temporal-coherence, fitness-landscape, entropy-controller, phase-navigator)
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, Request

logger = logging.getLogger("L104_FAST")

router = APIRouter(prefix="/api/v63", tags=["v63-monitoring"])

# Import nexus monitoring engines
try:
    from l104_server.engines_nexus import (
        temporal_coherence, fitness_landscape, entropy_controller, phase_navigator,
        nexus_steering, nexus_evolution
    )
    NEXUS_AVAILABLE = True
except ImportError:
    temporal_coherence = None
    fitness_landscape = None
    entropy_controller = None
    phase_navigator = None
    nexus_steering = None
    nexus_evolution = None
    NEXUS_AVAILABLE = False
    logger.warning("⚠️ [NEXUS] Monitoring engines not available")

# Import cache infrastructure
try:
    from l104_server.engines_infra import _FAST_REQUEST_CACHE, _PATTERN_RESPONSE_CACHE, _PATTERN_CACHE_LOCK
    CACHE_AVAILABLE = True
except ImportError:
    _FAST_REQUEST_CACHE = None
    _PATTERN_RESPONSE_CACHE = {}
    _PATTERN_CACHE_LOCK = None
    CACHE_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════
#  TEMPORAL COHERENCE ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.get("/temporal-coherence/status")
async def temporal_coherence_api_status():
    """Temporal coherence tracker — EMA, velocity, anomalies, drift alarms, cross-correlations."""
    if not NEXUS_AVAILABLE or not temporal_coherence:
        return {"error": "Temporal coherence not available", "status": "UNAVAILABLE"}
    try:
        return temporal_coherence.get_status()
    except Exception as e:
        return {"error": str(e)}


@router.get("/temporal-coherence/anomalies")
async def temporal_coherence_anomalies():
    """Recent coherence anomalies across all tracked engines."""
    if not NEXUS_AVAILABLE or not temporal_coherence:
        return {"error": "Temporal coherence not available"}
    try:
        return {
            "anomalies": temporal_coherence._anomalies[-50:] if hasattr(temporal_coherence, '_anomalies') else [],
            "alarms": temporal_coherence._alarms[-20:] if hasattr(temporal_coherence, '_alarms') else [],
            "sample_count": temporal_coherence._sample_count if hasattr(temporal_coherence, '_sample_count') else 0,
            "cross_correlations": temporal_coherence._cross_correlations if hasattr(temporal_coherence, '_cross_correlations') else {},
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/temporal-coherence/forecast/{engine_name}")
async def temporal_coherence_forecast(engine_name: str):
    """φ-damped coherence degradation forecast for a specific engine."""
    if not NEXUS_AVAILABLE or not temporal_coherence:
        return {"error": "Temporal coherence not available"}
    try:
        return temporal_coherence.forecast_degradation(engine_name)
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════
#  FITNESS LANDSCAPE ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.get("/fitness-landscape/status")
async def fitness_landscape_api_status():
    """Evolutionary fitness landscape — trajectory, optima found, valley escapes, gradient."""
    if not NEXUS_AVAILABLE or not fitness_landscape:
        return {"error": "Fitness landscape not available", "status": "UNAVAILABLE"}
    try:
        return fitness_landscape.get_status()
    except Exception as e:
        return {"error": str(e)}


@router.post("/fitness-landscape/valley-escape")
async def fitness_landscape_valley_escape():
    """Apply GOD_CODE-scaled perturbation to escape local fitness minimum."""
    if not NEXUS_AVAILABLE or not fitness_landscape:
        return {"error": "Fitness landscape not available"}
    try:
        engines_dict = {
            'steering': nexus_steering,
            'evolution': nexus_evolution,
        }
        return fitness_landscape.valley_escape(engines_dict)
    except Exception as e:
        return {"error": str(e)}


@router.get("/fitness-landscape/gradient")
async def fitness_landscape_gradient():
    """Estimate fitness gradient from recent trajectory differences."""
    if not NEXUS_AVAILABLE or not fitness_landscape:
        return {"error": "Fitness landscape not available"}
    try:
        engines_dict = {
            'steering': nexus_steering,
            'evolution': nexus_evolution,
        }
        return {
            "gradient": fitness_landscape.estimate_gradient(engines_dict),
            "status": fitness_landscape.get_status(),
        }
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════
#  ENTROPY CONTROLLER ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.get("/entropy-controller/status")
async def entropy_controller_api_status():
    """Entropy budget controller — Maxwell's Demon cycles, Landauer bound, per-engine budgets."""
    if not NEXUS_AVAILABLE or not entropy_controller:
        return {"error": "Entropy controller not available", "status": "UNAVAILABLE"}
    try:
        return entropy_controller.get_status()
    except Exception as e:
        return {"error": str(e)}


@router.post("/entropy-controller/demon")
async def entropy_controller_demon():
    """Manually trigger Maxwell's Demon entropy reversal cycle."""
    if not NEXUS_AVAILABLE or not entropy_controller:
        return {"error": "Entropy controller not available"}
    try:
        return entropy_controller.force_demon()
    except Exception as e:
        return {"error": str(e)}


@router.post("/entropy-controller/exchange")
async def entropy_controller_exchange(request: Request):
    """Transfer entropy budget credits between two engines."""
    if not NEXUS_AVAILABLE or not entropy_controller:
        return {"error": "Entropy controller not available"}
    try:
        body = await request.json()
        from_engine = body.get("from_engine", "")
        to_engine = body.get("to_engine", "")
        amount = float(body.get("amount", 10.0))
        if not from_engine or not to_engine:
            return {"error": "from_engine and to_engine required"}
        return entropy_controller.entropy_exchange(from_engine, to_engine, amount)
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════
#  PHASE NAVIGATOR ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.get("/phase-navigator/status")
async def phase_navigator_api_status():
    """Phase space navigator — attractors, limit cycles, Lyapunov spectrum, golden basin distance."""
    if not NEXUS_AVAILABLE or not phase_navigator:
        return {"error": "Phase navigator not available", "status": "UNAVAILABLE"}
    try:
        return phase_navigator.get_status()
    except Exception as e:
        return {"error": str(e)}


@router.post("/phase-navigator/suggest")
async def phase_navigator_suggest():
    """Get φ-weighted steering corrections toward the golden basin optimum."""
    if not NEXUS_AVAILABLE or not phase_navigator:
        return {"error": "Phase navigator not available"}
    try:
        return phase_navigator.suggest_steering()
    except Exception as e:
        return {"error": str(e)}


@router.get("/phase-navigator/lyapunov")
async def phase_navigator_lyapunov():
    """Lyapunov exponent spectrum — stability, chaos detection, bifurcation risk."""
    if not NEXUS_AVAILABLE or not phase_navigator:
        return {"error": "Phase navigator not available"}
    try:
        return phase_navigator.get_lyapunov_spectrum()
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════
#  MONITORING OVERVIEW
# ═══════════════════════════════════════════════════════════════════

@router.get("/monitoring/overview")
async def monitoring_overview():
    """Combined overview of all four v5.0 monitoring engines."""
    if not NEXUS_AVAILABLE:
        return {"error": "Nexus engines not available", "status": "UNAVAILABLE"}
    try:
        return {
            "version": "5.0.0",
            "temporal_coherence": temporal_coherence.get_status() if temporal_coherence else {},
            "fitness_landscape": fitness_landscape.get_status() if fitness_landscape else {},
            "entropy_controller": entropy_controller.get_status() if entropy_controller else {},
            "phase_navigator": phase_navigator.get_status() if phase_navigator else {},
        }
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════
#  CACHE STATS
# ═══════════════════════════════════════════════════════════════════

@router.get("/cache/stats")
async def cache_stats():
    """Fast request cache hit/miss/eviction statistics."""
    if not CACHE_AVAILABLE:
        return {"error": "Cache not available", "status": "UNAVAILABLE"}
    try:
        stats = _FAST_REQUEST_CACHE.stats()
        with _PATTERN_CACHE_LOCK:
            stats['pattern_cache_size'] = len(_PATTERN_RESPONSE_CACHE)
        return stats
    except Exception as e:
        return {"error": str(e)}


__all__ = ['router']