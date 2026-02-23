# routers/intellect.py — Local Intellect stats, training, export/import, v14 endpoints
import asyncio
import concurrent.futures
import math
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from config import UTC

router = APIRouter()


@router.get("/api/v6/intellect/stats", tags=["Intellect"])
async def intellect_stats():
    """Get Local Intellect statistics — evolution state, quantum memory, learning metrics."""
    try:
        from l104_local_intellect import local_intellect
        from l104_quantum_ram import get_qram
        qram = get_qram()
        stats = {
            "status": "SOVEREIGN_ACTIVE",
            "resonance": local_intellect._calculate_resonance(),
            "god_code": 527.5184818492612,
            "conversation_memory_size": len(local_intellect.conversation_memory),
            "knowledge_topics": len(local_intellect.knowledge) if hasattr(local_intellect, "knowledge") else 0,
            "quantum_memory_entries": len(qram.memory_manifold),
            "evolution_state": local_intellect.get_evolution_state()
                if hasattr(local_intellect, "get_evolution_state") else {"learning_cycles": 0, "insights_accumulated": 0},
            "timestamp": datetime.now(UTC).isoformat(),
        }
        return {"status": "SUCCESS", "stats": stats}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v6/intellect/resonate", tags=["Intellect"])
async def intellect_resonate(payload: Dict[str, Any] = None):
    """Trigger a resonance calibration cycle for the Local Intellect."""
    try:
        from l104_local_intellect import local_intellect
        from l104_quantum_ram import get_qram
        qram = get_qram()
        old_resonance = local_intellect._calculate_resonance()
        qram.store("resonance_event", {
            "timestamp": datetime.now(UTC).isoformat(),
            "resonance": old_resonance,
            "trigger": payload.get("trigger", "manual") if payload else "manual",
        })
        if hasattr(local_intellect, "evolve_patterns"):
            local_intellect.evolve_patterns()
        new_resonance = local_intellect._calculate_resonance()
        return {
            "status": "SUCCESS",
            "previous_resonance": old_resonance,
            "new_resonance": new_resonance,
            "delta": new_resonance - old_resonance,
            "quantum_stored": True,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v6/intellect/train", tags=["Intellect"])
async def intellect_train(payload: Dict[str, Any]):
    """
    Train the Local Intellect with new knowledge.
    Accepts server style {topic, content} or Swift style {query, response, quality}.
    """
    try:
        from l104_local_intellect import local_intellect
        from l104_quantum_ram import get_qram
        import time

        topic = payload.get("topic") or payload.get("query", "general")
        content = payload.get("content") or payload.get("response", "")
        quality = payload.get("quality", 1.0)

        if not content:
            return JSONResponse(status_code=400, content={"status": "ERROR", "error": "Content required"})

        qram = get_qram()
        qram.store(f"training_{topic}_{int(time.time())}", {
            "topic": topic, "content": content, "quality": quality,
            "source": "swift_sync" if "query" in payload else "api",
            "timestamp": datetime.now(UTC).isoformat(),
        })
        if hasattr(local_intellect, "knowledge"):
            local_intellect.knowledge[topic] = content
        if hasattr(local_intellect, "record_learning"):
            local_intellect.record_learning(topic, content)

        embedding_norm = 0.0
        learning_quality = quality
        try:
            loop = asyncio.get_running_loop()

            def _train_sync():
                local_intellect.retrain_memory(topic, content)
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
                if quality >= 1.5 and hasattr(local_intellect, "remember_permanently"):
                    local_intellect.remember_permanently(
                        f"swift_train_{topic[:30]}",
                        {"topic": topic, "content": content[:500], "quality": quality},
                        importance=quality / 2.0,
                    )

            with concurrent.futures.ThreadPoolExecutor() as pool:
                await loop.run_in_executor(pool, _train_sync)
            embedding_norm = math.log2(max(1, len(content))) * quality / 10.0
            learning_quality = quality * (1.0 + embedding_norm)
        except Exception:
            pass

        return {
            "status": "SUCCESS",
            "topic": topic,
            "content_length": len(content),
            "stored": True,
            "embedding_norm": round(embedding_norm, 4),
            "learning_quality": round(learning_quality, 4),
            "qi": local_intellect._evolution_state.get("quantum_interactions", 0),
            "auto_improvements": local_intellect._evolution_state.get("autonomous_improvements", 0),
            "training_count": len(local_intellect.training_data) if hasattr(local_intellect, "training_data") else 0,
            "timestamp": datetime.now(UTC).isoformat(),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.get("/api/v14/intellect/export", tags=["Intellect"])
async def intellect_export():
    """Export all Local Intellect knowledge and evolution state."""
    try:
        from l104_local_intellect import local_intellect
        from l104_quantum_ram import get_qram
        qram = get_qram()
        return {
            "status": "SUCCESS",
            "export": {
                "knowledge": dict(local_intellect.knowledge) if hasattr(local_intellect, "knowledge") else {},
                "conversation_memory": list(local_intellect.conversation_memory)
                    if hasattr(local_intellect, "conversation_memory") else [],
                "quantum_memory_keys": list(qram.memory_manifold.keys()),
                "evolution_state": local_intellect.get_evolution_state()
                    if hasattr(local_intellect, "get_evolution_state") else {},
                "resonance": local_intellect._calculate_resonance(),
                "exported_at": datetime.now(UTC).isoformat(),
            },
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v14/intellect/import", tags=["Intellect"])
async def intellect_import(payload: Dict[str, Any]):
    """Import knowledge and state into Local Intellect."""
    try:
        from l104_local_intellect import local_intellect
        from l104_quantum_ram import get_qram
        qram = get_qram()
        imported_count = 0
        if "knowledge" in payload and hasattr(local_intellect, "knowledge"):
            for topic, content in payload["knowledge"].items():
                local_intellect.knowledge[topic] = content
                qram.store(f"imported_{topic}", content)
                imported_count += 1
        if "evolution_state" in payload and hasattr(local_intellect, "set_evolution_state"):
            local_intellect.set_evolution_state(payload["evolution_state"])
        return {"status": "SUCCESS", "imported_topics": imported_count,
                "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})
