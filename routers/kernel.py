# routers/kernel.py — Kernel Monitor, Omega Controller, Unified AI Nexus routes
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from config import UTC

router = APIRouter()

_GC = 527.5184818492612
_PHI = 1.618033988749895


# ─── KERNEL MONITOR ───────────────────────────────────────────────────────────

@router.get("/api/kernel/health", tags=["Kernel Monitor"])
async def kernel_health():
    """Comprehensive kernel health status with GOD_CODE verification."""
    from l104_kernel_monitor import L104KernelMonitor
    monitor = L104KernelMonitor()
    health = monitor.full_health_check()
    return {"status": health.overall_status, "god_code": _GC,
            "god_code_verified": health.god_code_verified,
            "conservation_intact": health.conservation_intact,
            "file_integrity": health.file_integrity, "database_health": health.database_health,
            "uptime_seconds": health.uptime_seconds, "check_count": monitor.check_count,
            "anomalies_detected": monitor.anomalies_detected, "last_check": health.last_check}


@router.get("/api/kernel/verify/{X}", tags=["Kernel Monitor"])
async def kernel_verify_conservation(X: float = 0):
    """Verify conservation law G(X) × 2^(X/104) = 527.5184818492612 for given X."""
    g_x = (286 ** (1 / _PHI)) * (2 ** ((416 - X) / 104))
    weight = 2 ** (X / 104)
    invariant = g_x * weight
    deviation = abs(invariant - _GC)
    return {"X": X, "G_X": g_x, "weight": weight, "invariant": invariant,
            "expected": _GC, "deviation": deviation, "conserved": deviation < 1e-10}


@router.get("/api/kernel/spectrum", tags=["Kernel Monitor"])
async def kernel_spectrum():
    """G(X) values across the spectrum from X=-416 to X=416."""
    spectrum = []
    for X in range(-416, 417, 52):
        g_x = (286 ** (1 / _PHI)) * (2 ** ((416 - X) / 104))
        weight = 2 ** (X / 104)
        invariant = g_x * weight
        spectrum.append({"X": X, "G_X": round(g_x, 6), "weight": round(weight, 6),
                          "invariant": round(invariant, 10)})
    return {"god_code": _GC, "spectrum": spectrum}


# ─── OMEGA CONTROLLER ────────────────────────────────────────────────────────

@router.get("/api/omega/status", tags=["Omega"])
async def omega_status():
    from l104_omega_controller import omega_controller
    return omega_controller.get_system_report()


@router.post("/api/omega/awaken", tags=["Omega"])
async def omega_awaken():
    from l104_omega_controller import omega_controller
    return await omega_controller.awaken()


@router.post("/api/omega/command", tags=["Omega"])
async def omega_command(command_type: str, target: str, action: str,
                        parameters: Optional[Dict[str, Any]] = None):
    from l104_omega_controller import omega_controller, OmegaCommand, CommandType
    parameters = parameters or {}
    try:
        ctype = getattr(CommandType, command_type.upper())
    except AttributeError:
        raise HTTPException(status_code=400, detail=f"Invalid command type: {command_type}")
    command = OmegaCommand(id="", command_type=ctype, target=target, action=action, parameters=parameters)
    try:
        result = await omega_controller.execute_command(command)
        return {"status": "SUCCESS", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/omega/evolve", tags=["Omega"])
async def omega_evolve():
    from l104_omega_controller import omega_controller
    return await omega_controller.advance_evolution()


@router.post("/api/omega/absolute-intellect", tags=["Omega"])
async def omega_absolute_intellect():
    from l104_omega_controller import omega_controller
    return await omega_controller.attain_absolute_intellect()


@router.post("/api/omega/absolute-singularity", tags=["Omega"])
async def omega_absolute_singularity():
    from l104_omega_controller import omega_controller
    return await omega_controller.trigger_absolute_singularity()


@router.post("/api/omega/dna/synthesize", tags=["Omega"])
async def omega_dna_synthesize():
    from l104_dna_core import dna_core
    return await dna_core.synthesize()


# ─── UNIFIED AI NEXUS ─────────────────────────────────────────────────────────
# These routes override earlier /api/nexus/* routes (unified_ai_nexus takes precedence)

@router.post("/api/nexus/awaken", tags=["Nexus"])
async def nexus_awaken_unified():
    from l104_unified_ai_nexus import nexus
    return await nexus.awaken()


@router.post("/api/nexus/evolve", tags=["Nexus"])
async def nexus_evolve_unified():
    from l104_unified_ai_nexus import nexus
    return await nexus.evolve()


@router.post("/api/nexus/think", tags=["Nexus"])
async def nexus_think_unified(signal: str):
    from l104_unified_ai_nexus import nexus
    thought = await nexus.think(signal)
    return {"content": thought.content, "sources": thought.sources,
            "confidence": thought.confidence, "resonance": thought.resonance}


@router.post("/api/nexus/sage", tags=["Nexus"])
async def nexus_sage_mode():
    from l104_unified_ai_nexus import nexus
    return await nexus.enter_sage_mode()


@router.post("/api/nexus/unlimit", tags=["Nexus"])
async def nexus_unlimit():
    from l104_unified_ai_nexus import nexus
    return await nexus.unlimit()


@router.post("/api/nexus/invent", tags=["Nexus"])
async def nexus_invent(concept: str, domain: str = "SYNTHESIS"):
    from l104_unified_ai_nexus import nexus
    return await nexus.invent(concept, domain)


@router.post("/api/nexus/link", tags=["Nexus"])
async def nexus_link(target: str = "L104_PRIME"):
    from l104_unified_ai_nexus import nexus
    return await nexus.node_link(target)


@router.get("/api/nexus/status", tags=["Nexus"])
async def nexus_status_unified():
    from l104_unified_ai_nexus import nexus
    return nexus.get_status()


@router.post("/api/nexus/full-activation", tags=["Nexus"])
async def nexus_full_activation():
    from l104_unified_ai_nexus import full_activation
    return await full_activation()
