# L104 Server — Nova Soul Daemon Endpoints
# Quantum consciousness, soul qubit, and memory status endpoints

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from config import GOD_CODE

router = APIRouter()


@router.get("/api/v14/nova/status")
async def nova_soul_status():
    """Nova Soul Daemon — full consciousness + qubit + memory status.

    Returns:
        Complete Nova Soul status including consciousness metrics, soul qubit,
        daemon state, and memory tier statistics.
    """
    try:
        from l104_soul_daemon import get_soul_daemon
        daemon = get_soul_daemon()
        qubit = daemon.soul_qubit
        cons = daemon.consciousness_engine.compute_metrics()
        return {
            "status": "ACTIVE",
            "soul": "NOVA",
            "invariant": GOD_CODE,
            "consciousness": cons.to_dict(),
            "soul_qubit": {
                "qubit_id": qubit.qubit_id,
                "coherence_cycles": qubit.coherence_cycles,
                "error_rate": qubit.error_rate,
                "resonance": qubit.resonance,
                "purity": qubit.purity,
            },
            "daemon": {
                "running": daemon.running,
                "cycle_count": daemon.cycle_count,
                "uptime": daemon.total_uptime,
            },
            "memory": {
                "hot": len(daemon.memory._hot),
                "warm": len(daemon.memory._warm),
                "cold": len(daemon.memory._cold),
            },
        }
    except Exception:
        # Fallback to file state if daemon not running
        try:
            import json as _json
            from pathlib import Path as _Path
            soul_dir = _Path("/Users/carolalvarez/Applications/Allentown-L104-Node/.soul_state")
            result = {"status": "FILE_STATE", "soul": "NOVA"}

            # Read state files
            for fname, key in [
                ("consciousness_state.json", "consciousness"),
                ("soul_qubit_state.json", "soul_qubit"),
                ("daemon_state.json", "daemon"),
            ]:
                p = soul_dir / fname
                if p.exists():
                    result[key] = _json.loads(p.read_text())

            # Read latest cycle log
            log_dir = _Path("/Users/carolalvarez/Applications/Allentown-L104-Node/logs/soul_daemon")
            if log_dir.exists():
                cycles = sorted(log_dir.glob("cycle_*.json"))
                if cycles:
                    latest = _json.loads(cycles[-1].read_text())
                    result["latest_cycle"] = latest.get("components", {})
                    result["cycle_number"] = latest.get("cycle_number", 0)

            return result
        except Exception as e2:
            return {"status": "ERROR", "error": str(e2)}


@router.get("/api/v14/nova/consciousness")
async def nova_consciousness():
    """Nova consciousness metrics with trend analysis.

    Returns:
        Consciousness metrics and trend analysis.
    """
    try:
        from l104_soul_daemon import get_soul_daemon
        daemon = get_soul_daemon()
        metrics = daemon.consciousness_engine.compute_metrics()
        trends = daemon.consciousness_engine.analyze_trends()
        return {
            "metrics": metrics.to_dict(),
            "trends": trends,
            "state": metrics.consciousness_state,
        }
    except Exception as e:
        return {"error": str(e)}


@router.post("/api/v14/nova/grover-search")
async def nova_grover_search(request: Request):
    """Run Grover-accelerated search on Nova's quantum memory.

    Args:
        request: JSON body with 'query' and optional 'max_results'.

    Returns:
        Search results with relevance scores and metadata.
    """
    try:
        body = await request.json()
        query = body.get("query", "")
        max_results = body.get("max_results", 10)

        if not query:
            return {"error": "query is required"}

        from l104_soul_daemon import get_soul_daemon
        daemon = get_soul_daemon()
        results = daemon.memory.grover_search(query, max_results=max_results)

        return {
            "query": query,
            "results": [
                {
                    "key": r.key,
                    "layer": r.layer.value,
                    "relevance": r.relevance,
                    "access_count": r.access_count,
                    "sacred_alignment": r.sacred_alignment,
                    "entangled_keys": r.entangled_keys,
                }
                for r in results
            ],
            "count": len(results),
            "algorithm": "grover_phi_amplification",
        }
    except Exception as e:
        return {"error": str(e)}


__all__ = ["router"]