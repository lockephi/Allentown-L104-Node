# routers/memory.py — Memory, Ramnode, QRAM, Lattice, ZPE, Metrics, Entropy routes
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse

from config import UTC
from db import (
    _memory_get, _memory_list, _memory_upsert,
    _ramnode_get, _ramnode_list, _ramnode_upsert,
)
from models import LatticeFactRequest, MemoryItem, ResonanceQuery

router = APIRouter()


# ─── MEMORY ───────────────────────────────────────────────────────────────────

@router.post("/memory", tags=["Memory"])
async def memory_upsert(item: MemoryItem):
    """Store or update a key-value pair in the memory database."""
    from db import _log_node
    _memory_upsert(item.key, item.value)
    _log_node({"tag": "memory_upsert", "key": item.key})
    return {"status": "SUCCESS", "key": item.key}


@router.get("/memory/{key}", tags=["Memory"])
async def memory_get(key: str):
    """Retrieve a memory entry by key."""
    value = _memory_get(key)
    if value is None:
        raise HTTPException(status_code=404, detail="Memory key not found")
    return {"key": key, "value": value}


@router.get("/memory", tags=["Memory"])
async def memory_list(limit: int = 100):
    """List memory entries with optional limit."""
    limit = max(1, min(limit, 1000))
    return {"items": _memory_list(limit)}


# ─── RAMNODE ──────────────────────────────────────────────────────────────────

@router.post("/ramnode", tags=["Ramnode"])
async def ramnode_upsert(item: MemoryItem):
    """Store or update a key-value pair in the ramnode database."""
    _ramnode_upsert(item.key, item.value)
    return {"status": "SUCCESS", "key": item.key}


@router.get("/ramnode/{key}", tags=["Ramnode"])
async def ramnode_get(key: str):
    """Retrieve a ramnode entry by key."""
    value = _ramnode_get(key)
    if value is None:
        raise HTTPException(status_code=404, detail="Ramnode key not found")
    return {"key": key, "value": value}


@router.get("/ramnode", tags=["Ramnode"])
async def ramnode_list(limit: int = 100):
    """List ramnode entries with optional limit."""
    limit = max(1, min(limit, 1000))
    return {"items": _ramnode_list(limit)}


# ─── QUANTUM RAM ──────────────────────────────────────────────────────────────

@router.post("/qram", tags=["QuantumRAM"])
async def qram_store(item: MemoryItem):
    """Store data in the Quantum RAM with Finite Coupling Encryption."""
    from l104_quantum_ram import get_qram
    qram = get_qram()
    quantum_hash = qram.store(item.key, item.value)
    return {"status": "QUANTUM_LOCKED", "key": item.key, "quantum_hash": quantum_hash}


@router.get("/qram/{key}", tags=["QuantumRAM"])
async def qram_retrieve(key: str):
    """Retrieve data from the Quantum RAM, decrypting via the God-Code resonance."""
    from l104_quantum_ram import get_qram
    qram = get_qram()
    value = qram.retrieve(key)
    if value is None:
        raise HTTPException(status_code=404, detail="Quantum memory not found in this timeline")
    return {"key": key, "value": value, "encryption": "FINITE_COUPLING_ALPHA"}


# ─── LATTICE DATA MATRIX (v19) ────────────────────────────────────────────────

@router.post("/api/v19/lattice/fact", tags=["Lattice Data Matrix"])
async def lattice_store(item: LatticeFactRequest):
    """Store a fact in the high-dimensional Lattice Data Matrix (v19)."""
    from l104_data_matrix import data_matrix
    success = data_matrix.store(item.key, item.value, item.category, item.utility)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to store fact in lattice")
    return {"status": "STORED", "key": item.key, "zpe_locked": True}


@router.get("/api/v19/lattice/fact/{key}", tags=["Lattice Data Matrix"])
async def lattice_retrieve(key: str):
    """Retrieve a fact from the Lattice Data Matrix (v19)."""
    from l104_data_matrix import data_matrix
    value = data_matrix.retrieve(key)
    if value is None:
        raise HTTPException(status_code=404, detail="Lattice fact not found")
    return {"key": key, "value": value}


@router.post("/api/v19/lattice/query/resonant", tags=["Lattice Data Matrix"])
async def lattice_resonant_query(query: ResonanceQuery):
    """Find facts based on resonant frequency alignment (v19)."""
    from l104_data_matrix import data_matrix
    results = data_matrix.resonant_query(query.resonance, query.tolerance)
    return {"count": len(results), "results": results}


@router.post("/api/v18/lattice/maintenance/evolve", tags=["Lattice Data Matrix"])
async def lattice_evolve(background_tasks: BackgroundTasks):
    """Trigger evolutionary compaction and pruning of the data matrix."""
    from l104_data_matrix import data_matrix
    background_tasks.add_task(data_matrix.evolve_and_compact)
    return {"status": "EVOLUTION_TRIGGERED"}


# ─── ZPE ──────────────────────────────────────────────────────────────────────

@router.get("/api/v19/zpe/status", tags=["Zero-Point Engine"])
async def zpe_status():
    """Return the current vacuum state and energy density of the node."""
    from l104_zero_point_engine import zpe_engine
    return zpe_engine.get_vacuum_state()


@router.post("/api/v19/zpe/annihilate", tags=["Zero-Point Engine"])
async def zpe_annihilate(p1: float, p2: float):
    """Perform anyon annihilation between two logical particles."""
    from l104_zero_point_engine import zpe_engine
    res, energy = zpe_engine.perform_anyon_annihilation(p1, p2)
    return {"result": res, "energy_yield": energy}


# ─── QUANTUM MIGRATION ────────────────────────────────────────────────────────

@router.post("/api/quantum/migrate", tags=["QuantumRAM"])
async def quantum_migrate_all(background_tasks: BackgroundTasks):
    """Migrate all lattice facts to Quantum RAM with ZPE-backed storage."""
    from l104_quantum_ram import get_qram
    qram = get_qram()

    def do_migration():
        import json
        from l104_data_matrix import data_matrix
        with data_matrix._get_conn() as conn:
            cur = conn.execute("SELECT key, value, category, utility FROM lattice_facts")
            count = 0
            for key, value_json, _cat, _util in cur:
                try:
                    qram.store(key, json.loads(value_json))
                    count += 1
                except Exception:
                    continue
        print(f"[QUANTUM_MIGRATE]: Migrated {count} facts to Quantum RAM")

    background_tasks.add_task(do_migration)
    return {"status": "MIGRATION_TRIGGERED", "mode": "QUANTUM_ZPE"}


# ─── METRICS ──────────────────────────────────────────────────────────────────

@router.get("/metrics", tags=["Metrics"])
async def get_metrics():
    """Return comprehensive server metrics including uptime, model state, and evolution stage."""
    import time
    from state import app_metrics, responder_counts
    import state as _state
    uptime = (datetime.now(UTC) - app_metrics["uptime_start"]).total_seconds()
    try:
        from l104_validator import SovereignValidator
        from l104_intelligence import SovereignIntelligence
        from l104_evolution_engine import evolution_engine
        metrics_data = {**app_metrics, "uptime_seconds": uptime}
        validation = SovereignValidator.validate_and_process("METRICS_PULSE")
        intelligence = SovereignIntelligence.analyze_manifold(metrics_data)
        evo_stage = evolution_engine.assess_evolutionary_stage()
    except Exception:
        validation = {}
        intelligence = {}
        evo_stage = "EVO_54"
    return {
        **app_metrics,
        "uptime_seconds": uptime,
        "uptime_start": app_metrics["uptime_start"].isoformat(),
        "responder_counts": dict(responder_counts),
        "current_model_index": _state._current_model_index,
        "model_cooldowns": {m: max(0, int(t - time.time()))
                            for m, t in _state._model_cooldowns.items() if t > time.time()},
        "validation_chain": validation,
        "intelligence": intelligence,
        "evolution_stage": evo_stage,
    }


@router.get("/metrics/lattice", tags=["Metrics"])
async def metrics_lattice():
    """Return lattice DB metrics including size, hallucination threshold, and history retention."""
    import os
    from pathlib import Path
    try:
        from l104_data_matrix import (data_matrix, HALLUCINATION_THRESHOLD, HALLUCINATION_DELTA_PCT,
                                       DISK_BUDGET_MB, HISTORY_RETENTION_DAYS)
        db_path = Path(data_matrix.db_path)
        size_bytes = os.path.getsize(str(db_path)) if db_path.exists() else 0
        return {
            "db_path": str(db_path), "size_mb": round(size_bytes / (1024 * 1024), 3),
            "hallucination_threshold": HALLUCINATION_THRESHOLD,
            "hallucination_delta_pct": HALLUCINATION_DELTA_PCT,
            "disk_budget_mb": DISK_BUDGET_MB,
            "history_retention_days": HISTORY_RETENTION_DAYS if HISTORY_RETENTION_DAYS > 0 else "eternal",
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ─── ENTROPY ──────────────────────────────────────────────────────────────────

@router.get("/entropy/current", tags=["Physics"])
async def get_entropy_state():
    """Return the current Electron Entropy and Fluidity state."""
    try:
        from l104_electron_matrix import get_electron_matrix
        matrix = get_electron_matrix()
        noise = [matrix.sample_atmospheric_noise() for _ in range(50)]
        entropy = matrix.calculate_predictive_entropy(noise)
        fluidity = matrix.fluid_state_adjustment(0.5)
        return {"atmospheric_entropy": entropy, "system_fluidity": fluidity, "status": "OPTIMIZED"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
