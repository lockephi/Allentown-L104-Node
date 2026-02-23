# routers/sync.py — Bidirectional Swift ↔ Server sync endpoints
import asyncio
import concurrent.futures
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from config import UTC

router = APIRouter()


@router.post("/api/v6/sync", tags=["Sync"])
async def unified_sync(payload: Dict[str, Any] = None):
    """
    Unified bidirectional sync for Swift iOS app ↔ L104 Server.

    Swift sends (optional):
      - swift_knowledge: [{prompt, completion, source}]
      - swift_conversations: [{query, response}]
      - swift_evolution: {qi, auto_improvements, ...}
      - swift_concepts: [str]

    Server returns:
      - evolution_state, training_count, ft_status, recent_insights, sync_timestamp
    """
    try:
        from l104_local_intellect import local_intellect
        from l104_quantum_ram import get_qram

        qram = get_qram()
        payload = payload or {}
        ingested_count = 0

        # ── Ingest Swift knowledge ──
        swift_knowledge = payload.get("swift_knowledge", [])
        if swift_knowledge:
            def _ingest_knowledge():
                count = 0
                for entry in swift_knowledge[:50]:
                    topic = entry.get("prompt", entry.get("topic", ""))
                    content = entry.get("completion", entry.get("content", ""))
                    if topic and content:
                        local_intellect.retrain_memory(topic, content)
                        if hasattr(local_intellect, "knowledge"):
                            local_intellect.knowledge[topic] = content
                        if hasattr(local_intellect, "_ft_engine") and local_intellect._ft_init_done:
                            try:
                                vec = local_intellect._text_to_ft_vector(content[:500])
                                local_intellect._ft_engine.attention.add_pattern(vec)
                                local_intellect._ft_engine.memory.store(vec, label=topic[:30])
                                tokens = [w.lower() for w in content.split() if len(w) > 2][:80]
                                if tokens:
                                    local_intellect._ft_engine.tfidf.add_document(tokens)
                            except Exception:
                                pass
                        count += 1
                return count

            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                ingested_count = await loop.run_in_executor(pool, _ingest_knowledge)

        # ── Ingest Swift conversations ──
        swift_convos = payload.get("swift_conversations", [])
        if swift_convos:
            def _ingest_convos():
                count = 0
                for convo in swift_convos[:20]:
                    q = convo.get("query", "")
                    r = convo.get("response", "")
                    if q and r:
                        local_intellect.retrain_memory(q, r)
                        count += 1
                return count

            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                ingested_count += await loop.run_in_executor(pool, _ingest_convos)

        # ── Max-merge Swift evolution state ──
        swift_evo = payload.get("swift_evolution", {})
        if swift_evo:
            for key in ["quantum_interactions", "autonomous_improvements"]:
                if key in swift_evo:
                    server_val = local_intellect._evolution_state.get(key, 0)
                    swift_val = swift_evo[key]
                    if isinstance(swift_val, (int, float)) and swift_val > server_val:
                        local_intellect._evolution_state[key] = int(swift_val)

        # ── Build response ──
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
            "ingested_count": ingested_count,
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
            "timestamp": datetime.now(UTC).isoformat(),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})
