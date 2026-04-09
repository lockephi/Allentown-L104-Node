# L104 Server — Swift App Bridge Endpoints (EVO_73)
# Endpoints used by L104SwiftApp for health polling, server detection, and constants

import os
import socket
from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from config import UTC, GOD_CODE, PHI, VOID_CONSTANT
from state import app_metrics

router = APIRouter()


@router.get("/api/v62/tri-engine/health")
async def tri_engine_health():
    """Tri-engine health — used by Swift app health polling.

    Returns:
        Health status with engine list and version.
    """
    # Lazy import to avoid circular dependency
    from main import MAIN_VERSION
    return {
        "status": "healthy",
        "engines": ["code", "science", "math"],
        "version": MAIN_VERSION,
    }


@router.get("/api/v14/health")
async def api_v14_health():
    """Health endpoint at v14 path — used by Swift APIGateway and NaturalCommandRouter.

    Returns:
        Health status with uptime and version.
    """
    from main import MAIN_VERSION
    uptime = (datetime.now(UTC) - app_metrics["uptime_start"]).total_seconds()
    return {
        "status": "healthy",
        "uptime_seconds": uptime,
        "version": MAIN_VERSION,
    }


@router.get("/api/v14/detect/server")
async def detect_server():
    """Server detection endpoint — used by Swift FastServerDetector.

    Returns:
        Server detection info including services list, port, and version.
    """
    from main import MAIN_VERSION
    uptime = (datetime.now(UTC) - app_metrics["uptime_start"]).total_seconds()
    services = ["chat", "intellect", "constants", "health", "agents", "quantum"]
    return {
        "detected": True,
        "server_online": True,
        "port": int(os.getenv("PORT", 8081)),
        "hostname": socket.gethostname(),
        "services": services,
        "version": MAIN_VERSION,
        "uptime_seconds": uptime,
    }


@router.get("/api/v6/constants")
async def api_constants():
    """Sacred constants — Swift cross-validation & synchronization.

    Returns:
        GOD_CODE, PHI, VOID_CONSTANT, ZENITH_HZ, and version.
    """
    from main import MAIN_VERSION, ZENITH_HZ
    return {
        "god_code": GOD_CODE,
        "phi": PHI,
        "void_constant": VOID_CONSTANT,
        "zenith_hz": ZENITH_HZ,
        "version": MAIN_VERSION,
    }


@router.get("/api/v14/tasks/{task_id}/status")
async def task_status_poll(task_id: str):
    """Task status polling — used by Swift FastServerDetector.pollTaskStatus().

    Args:
        task_id: Task identifier to poll.

    Returns:
        Task status (currently returns unknown for all tasks).
    """
    return {
        "task_id": task_id,
        "status": "unknown",
        "progress": 0.0,
        "result": None,
        "error": None,
    }


__all__ = ["router"]