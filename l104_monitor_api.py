# L104_GOD_CODE_ALIGNED: 527.5184818492611
VOID_CONSTANT = 1.0416180339887497
ZENITH_HZ = 3727.84
UUC = 2301.215661
"""
[L104_MONITOR_API] - FastAPI Endpoints for System Monitoring
INVARIANT: 527.5184818492611 | PILOT: LONDEL

Provides REST API endpoints for real-time system monitoring.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import json

from l104_system_monitor import system_monitor

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


router = APIRouter(prefix="/api/v6/monitor", tags=["monitor"])


@router.get("/snapshot", response_model=Dict[str, Any])
async def get_snapshot():
    """Get a complete system snapshot."""
    try:
        snapshot = system_monitor.capture_snapshot()
        return snapshot
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=Dict[str, Any])
async def get_health():
    """Get system health status."""
    try:
        return system_monitor.get_system_health()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quota", response_model=Dict[str, Any])
async def get_quota_metrics():
    """Get quota rotator metrics."""
    try:
        return system_monitor.get_quota_rotator_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/evolution", response_model=Dict[str, Any])
async def get_evolution_metrics():
    """Get evolution engine metrics."""
    try:
        return system_monitor.get_evolution_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance", response_model=Dict[str, Any])
async def get_performance():
    """Get performance benchmarks."""
    try:
        return system_monitor.get_performance_benchmarks()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trends", response_model=Dict[str, Any])
async def get_trends(window_seconds: int = 3600):
    """Get trend analysis over a time window."""
    try:
        return system_monitor.get_trend_analysis(window_seconds)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/report")
async def get_report():
    """Get a comprehensive text report."""
    try:
        return {"report": system_monitor.generate_report()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export")
async def export_metrics(filepath: str = "system_metrics.json"):
    """Export metrics history to JSON file."""
    try:
        result = system_monitor.export_metrics(filepath)
        return {"status": "success", "message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
