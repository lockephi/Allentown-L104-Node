# routers/sync.py — Bidirectional Swift <-> Server sync endpoints
# v23.4: Fire-and-forget ingestion — respond immediately, process in background
import asyncio
import logging
import concurrent.futures
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse

from config import UTC

router = APIRouter()
_logger = logging.getLogger("L104_SYNC")

# Shared thread pool (reused across requests, avoids per-request overhead)
_sync_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="sync")

# Track background ingestion progress
_sync_state = {
    "pending_count": 0,
    "total_ingested": 0,
    "last_chunk": {},
    "errors": [],
}


def _background_ingest(swift_knowledge: list, swift_convos: list, swift_evo: dict):
    """Heavy ingestion work — runs in background thread after response is sent."""
    try:
        from l104_local_intellect import local_intellect
    except Exception as e:
        _sync_state["errors"].append(f"import: {e}")
        _sync_state["pending_count"] = 0
        return

    ingested = 0

    # ── Ingest knowledge (bulk — skip FT vectors on first pass for speed) ──
    if swift_knowledge:
        has_ft = (hasattr(local_intellect, "_ft_engine")
                  and getattr(local_intellect, "_ft_init_done", False))

        for entry in swift_knowledge[:500]:
            topic = entry.get("prompt", entry.get("topic", ""))
            content = entry.get("completion", entry.get("content", ""))
            if not (topic and content):
                continue
            try:
                local_intellect.retrain_memory(topic, content)
                if hasattr(local_intellect, "knowledge"):
                    local_intellect.knowledge[topic] = content
                ingested += 1
            except Exception:
                pass

        # FT engine batch (best-effort, non-blocking)
        if has_ft and ingested > 0:
            ft = local_intellect._ft_engine
            for entry in swift_knowledge[:ingested]:
                try:
                    content = entry.get("completion", entry.get("content", ""))
                    if not content:
                        continue
                    vec = local_intellect._text_to_ft_vector(content[:500])
                    ft.attention.add_pattern(vec)
                    ft.memory.store(vec, label=entry.get("prompt", "")[:30])
                    tokens = [w.lower() for w in content.split() if len(w) > 2][:80]
                    if tokens:
                        ft.tfidf.add_document(tokens)
                except Exception:
                    pass

    # ── Ingest conversations ──
    if swift_convos:
        for convo in swift_convos[:50]:
            q = convo.get("query", "")
            r = convo.get("response", "")
            if q and r:
                try:
                    local_intellect.retrain_memory(q, r)
                    ingested += 1
                except Exception:
                    pass

    # ── Max-merge evolution state ──
    if swift_evo:
        try:
            for key in ["quantum_interactions", "autonomous_improvements"]:
                if key in swift_evo:
                    server_val = local_intellect._evolution_state.get(key, 0)
                    swift_val = swift_evo[key]
                    if isinstance(swift_val, (int, float)) and swift_val > server_val:
                        local_intellect._evolution_state[key] = int(swift_val)
        except Exception:
            pass

    _sync_state["total_ingested"] += ingested
    _sync_state["pending_count"] = max(0, _sync_state["pending_count"] - 1)
    _logger.info(f"Background sync: ingested {ingested} entries (total: {_sync_state['total_ingested']})")


@router.post("/api/v6/sync", tags=["Sync"])
async def unified_sync(payload: Dict[str, Any] = None, background_tasks: BackgroundTasks = None):
    """
    Unified bidirectional sync for Swift iOS app <-> L104 Server.

    v23.4: Fire-and-forget — accepts payload immediately, ingests in background.
    Swift gets instant 200 with current server state; ingestion happens async.
    """
    try:
        from l104_local_intellect import local_intellect

        payload = payload or {}
        sync_meta = payload.get("sync_meta", {})

        # Queue heavy ingestion work in background (does NOT block response)
        swift_knowledge = payload.get("swift_knowledge", [])
        swift_convos = payload.get("swift_conversations", [])
        swift_evo = payload.get("swift_evolution", {})

        if swift_knowledge or swift_convos or swift_evo:
            _sync_state["pending_count"] += 1
            _sync_state["last_chunk"] = sync_meta
            background_tasks.add_task(
                _background_ingest, swift_knowledge, swift_convos, swift_evo
            )

        # ── Build response immediately with current server state ──
        evo = local_intellect._evolution_state

        recent_insights = []
        try:
            pm = evo.get("permanent_memory", {})
            for k in sorted(pm.keys(), reverse=True)[:10]:
                v = pm[k]
                if isinstance(v, dict):
                    recent_insights.append({"key": k, "value": str(v.get("value", v.get("insight", "")))[:200]})
                elif isinstance(v, str):
                    recent_insights.append({"key": k, "value": v[:200]})
        except Exception:
            pass

        ft_status = {}
        try:
            if hasattr(local_intellect, "_ft_engine") and local_intellect._ft_init_done:
                ft_status = {
                    "attn_patterns": getattr(local_intellect._ft_engine.attention, "pattern_count", 0),
                    "mem_stored": getattr(local_intellect._ft_engine.memory, "stored_count", 0),
                    "tfidf_vocab": getattr(local_intellect._ft_engine.tfidf, "vocab_size", 0),
                }
        except Exception:
            pass

        return {
            "status": "SUCCESS",
            "ingested_count": len(swift_knowledge),  # accepted count (processing async)
            "pending_background": _sync_state["pending_count"],
            "total_ingested": _sync_state["total_ingested"],
            "sync_meta": sync_meta,
            "evolution_state": {
                "quantum_interactions": evo.get("quantum_interactions", 0),
                "autonomous_improvements": evo.get("autonomous_improvements", 0),
                "quantum_data_mutations": evo.get("quantum_data_mutations", 0),
                "wisdom_quotient": evo.get("wisdom_quotient", 0),
                "logic_depth_reached": evo.get("logic_depth_reached", 0),
                "mutation_dna": evo.get("mutation_dna", "")[:16],
                "total_runs": evo.get("total_runs", 0),
                "cross_references": len(evo.get("cross_references", {})),
                "concept_evolution_count": len(evo.get("concept_evolution", {})),
                "permanent_memory_count": len(evo.get("permanent_memory", {})),
            },
            "training_count": len(local_intellect.training_data) if hasattr(local_intellect, "training_data") else 0,
            "conversation_memory_size": len(local_intellect.conversation_memory),
            "ft_status": ft_status,
            "recent_insights": recent_insights,
            "resonance": local_intellect._calculate_resonance(),
            "god_code": 527.5184818492612,
            "sync_timestamp": datetime.now(UTC).isoformat(),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.get("/api/v6/sync/status", tags=["Sync"])
async def sync_status():
    """Quick sync health check for Swift app — lightweight polling endpoint."""
    try:
        from l104_local_intellect import local_intellect
        evo = local_intellect._evolution_state
        return {
            "status": "ONLINE",
            "qi": evo.get("quantum_interactions", 0),
            "auto": evo.get("autonomous_improvements", 0),
            "training": len(local_intellect.training_data) if hasattr(local_intellect, "training_data") else 0,
            "dna": evo.get("mutation_dna", "")[:8],
            "resonance": local_intellect._calculate_resonance(),
            "pending_sync": _sync_state["pending_count"],
            "total_ingested": _sync_state["total_ingested"],
            "timestamp": datetime.now(UTC).isoformat(),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})
