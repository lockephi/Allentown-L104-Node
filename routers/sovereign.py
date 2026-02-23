# routers/sovereign.py — Dashboard, Evolution, Audit, Simulation, System admin routes
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from config import UTC
from models import ScourRequest, SimulationRequest, ManipulateRequest

router = APIRouter()

_GC = 527.5184818492612
_PHI = 1.618033988749895


# ─── DASHBOARD ────────────────────────────────────────────────────────────────

@router.get("/", tags=["Sovereign"])
async def dashboard(request: Request):
    """L104 Sovereign Node dashboard."""
    try:
        from main import templates
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception:
        return JSONResponse({"status": "L104 SOVEREIGN NODE ONLINE", "god_code": _GC,
                             "timestamp": datetime.now(UTC).isoformat()})


# ─── SCOUR / INVENT / EVOLVE ─────────────────────────────────────────────────

@router.post("/api/v6/scour", tags=["Sovereign"])
async def scour_internet(request: ScourRequest):
    """Scour internet/sources for new knowledge."""
    try:
        from l104_local_intellect import local_intellect
        result = local_intellect.scour(request.query, depth=request.depth, sources=request.sources)
        return {"status": "SUCCESS", "query": request.query, "findings": result,
                "god_code": _GC, "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/api/v6/invent", tags=["Sovereign"])
async def invent(signal: str = ""):
    """Pure invention engine — create novel concepts."""
    try:
        from l104_local_intellect import local_intellect
        invention = local_intellect.invent(signal)
        return {"status": "SUCCESS", "signal": signal, "invention": invention,
                "god_code": _GC, "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/api/v6/evolve", tags=["Sovereign"])
async def evolve():
    """Trigger evolutionary cycle across all subsystems."""
    try:
        from l104_agi_core import agi_core
        result = await agi_core.run_recursive_improvement_cycle()
        return {"status": "EVOLVED", "result": result, "god_code": _GC,
                "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ─── SIMULATION ───────────────────────────────────────────────────────────────

@router.post("/api/v6/simulate", tags=["Simulation"])
async def simulate(request: SimulationRequest):
    """Run sovereign simulation."""
    try:
        from l104_local_intellect import local_intellect
        result = local_intellect.simulate(request.scenario, steps=request.steps,
                                          variables=request.variables)
        return {"status": "SUCCESS", "scenario": request.scenario, "result": result,
                "god_code": _GC, "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/api/v6/manipulate", tags=["Simulation"])
async def manipulate(request: ManipulateRequest):
    """Manipulation endpoint — permanently disabled."""
    return JSONResponse(status_code=403,
                        content={"status": "FORBIDDEN", "message": "Manipulation is not permitted.",
                                 "god_code": _GC})


@router.post("/simulation/debate", tags=["Simulation"])
async def simulation_debate(payload: Dict[str, Any]):
    """Multi-agent debate simulation."""
    try:
        from l104_simulation_engine import simulation_engine
        topic = payload.get("topic", "consciousness")
        agents = payload.get("agents", ["AGI", "ASI", "L104"])
        rounds = payload.get("rounds", 3)
        return simulation_engine.run_debate(topic, agents, rounds)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/simulation/hyper_evolve", tags=["Simulation"])
async def simulation_hyper_evolve(payload: Dict[str, Any]):
    """Hyper-evolution simulation across manifolds."""
    try:
        from l104_simulation_engine import simulation_engine
        dimensions = payload.get("dimensions", 11)
        cycles = payload.get("cycles", 104)
        return simulation_engine.hyper_evolve(dimensions=dimensions, cycles=cycles)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ─── RAM FACTS ────────────────────────────────────────────────────────────────

@router.get("/api/v6/ram/facts", tags=["Sovereign"])
async def get_ram_facts():
    """Return all sovereign RAM facts."""
    try:
        from db import memory_list
        facts = memory_list(limit=500)
        return {"status": "SUCCESS", "count": len(facts), "facts": facts,
                "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ─── AUDIT ───────────────────────────────────────────────────────────────────

@router.post("/api/v6/audit/app", tags=["Audit"])
async def audit_app(path: str = ".", auto_remediate: bool = False):
    from l104_code_engine import code_engine
    return code_engine.audit_app(path, auto_remediate=auto_remediate)


@router.post("/api/v6/audit/quick", tags=["Audit"])
async def audit_quick():
    from l104_code_engine import code_engine
    return code_engine.audit_app(".", auto_remediate=False)


@router.post("/api/v6/audit/file", tags=["Audit"])
async def audit_file(filepath: str, auto_fix: bool = False):
    from l104_code_engine import code_engine
    return code_engine.audit_file(filepath, auto_fix=auto_fix)


@router.get("/api/v6/audit/status", tags=["Audit"])
async def audit_status():
    from l104_code_engine import code_engine
    return code_engine.get_audit_status()


@router.get("/api/v6/audit/trail", tags=["Audit"])
async def audit_trail(limit: int = 50):
    from l104_code_engine import code_engine
    return code_engine.get_audit_trail(limit)


@router.get("/api/v6/audit/history", tags=["Audit"])
async def audit_history(limit: int = 20):
    from l104_code_engine import code_engine
    return code_engine.get_audit_history(limit)


@router.post("/api/v6/audit/streamline", tags=["Audit"])
async def audit_streamline():
    from l104_code_engine import code_engine
    return code_engine.streamline_audit()


# ─── EVOLUTION ────────────────────────────────────────────────────────────────

@router.post("/api/v6/evolution/cycle", tags=["Evolution"])
async def evolution_cycle():
    from l104_agi_core import agi_core
    result = await agi_core.run_recursive_improvement_cycle()
    return {"status": "SUCCESS", "cycle_result": result, "timestamp": datetime.now(UTC).isoformat()}


@router.post("/api/v6/evolution/propose", tags=["Evolution"])
async def evolution_propose(concept: str = ""):
    from l104_agi_core import agi_core
    return await agi_core.propose_evolution(concept)


@router.post("/api/v6/evolution/self-improve", tags=["Evolution"])
async def evolution_self_improve():
    from l104_agi_core import agi_core
    return await agi_core.self_improve()


# ─── SYSTEM OPERATIONS ───────────────────────────────────────────────────────

@router.get("/system/capacity", tags=["System"])
async def system_capacity():
    """Return system capacity and resource metrics."""
    import psutil
    cpu = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(".")
    return {"cpu_percent": cpu, "memory_total_gb": round(mem.total / 1e9, 2),
            "memory_used_gb": round(mem.used / 1e9, 2), "memory_percent": mem.percent,
            "disk_total_gb": round(disk.total / 1e9, 2), "disk_used_gb": round(disk.used / 1e9, 2),
            "disk_percent": disk.percent, "god_code": _GC,
            "timestamp": datetime.now(UTC).isoformat()}


@router.post("/system/reindex", tags=["System"])
async def system_reindex():
    """Reindex all sovereign knowledge bases."""
    try:
        from l104_local_intellect import local_intellect
        result = local_intellect.reindex()
        return {"status": "REINDEXED", "result": result, "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ─── v14 SYSTEM ADMIN ────────────────────────────────────────────────────────

@router.post("/api/v14/system/update", tags=["System"])
async def system_update(payload: Dict[str, Any]):
    from l104_agi_core import agi_core
    return await agi_core.apply_system_update(payload)


@router.post("/api/v14/system/inject", tags=["System"])
async def system_inject(payload: Dict[str, Any]):
    """Inject knowledge or signals directly into AGI context."""
    from l104_agi_core import agi_core
    signal = payload.get("signal", "")
    ctx = payload.get("context", {})
    return await agi_core.inject(signal, ctx)


@router.post("/api/v14/google/process", tags=["System"])
async def google_process(payload: Dict[str, Any]):
    """Process via Google/Gemini integration."""
    try:
        from l104_gemini_bridge import gemini_bridge
        return await gemini_bridge.process(payload)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
