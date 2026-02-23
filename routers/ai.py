# routers/ai.py — Chat, Streaming, Gemini, Local, Scribe, Research, Code Analysis routes
import asyncio
import concurrent.futures
from collections import Counter
import math
from datetime import datetime

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from config import UTC
from models import ChatRequest, ScribeIngestRequest, StreamRequest
from state import app_metrics

router = APIRouter()


# ─── SCRIBE ───────────────────────────────────────────────────────────────────

@router.post("/api/v6/scribe/ingest", tags=["Scribe"])
async def scribe_ingest(req: ScribeIngestRequest):
    """Ingest intelligence from a provider into the Universal Scribe."""
    try:
        from l104_sage_core import sage_core
        from l104_agi_core import agi_core
        sage_core.scribe_ingest(req.provider, req.data)
        state = sage_core.get_state()
        if hasattr(agi_core, "save"):
            agi_core.save()
        return {
            "status": "SUCCESS",
            "provider": req.provider,
            "saturation": state["scribe"]["knowledge_saturation"],
            "linked_count": state["scribe"]["linked_count"],
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v6/scribe/synthesize", tags=["Scribe"])
async def scribe_synthesize():
    """Synthesize the ingested knowledge into Sovereign DNA."""
    try:
        from l104_sage_core import sage_core
        from l104_agi_core import agi_core
        sage_core.scribe_synthesize()
        state = sage_core.get_state()
        if hasattr(agi_core, "save"):
            agi_core.save()
        return {
            "status": "SUCCESS",
            "dna": state["scribe"]["sovereign_dna"],
            "saturation": state["scribe"]["knowledge_saturation"],
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.get("/api/v6/scribe/status", tags=["Scribe"])
async def scribe_status():
    """Get the current state of the Universal Scribe."""
    try:
        from l104_sage_core import sage_core
        state = sage_core.get_state()
        return {"status": "SUCCESS", "state": state["scribe"]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


# ─── CHAT ─────────────────────────────────────────────────────────────────────

@router.post("/api/v6/chat", tags=["AI"])
async def ai_chat(req: ChatRequest):
    """Direct AI chat — unified local path, full neural processing."""

    def unified_local_think(message: str) -> dict:
        from l104_local_intellect import local_intellect
        import time as _time

        t_start = _time.time()
        response = local_intellect.think(message)
        think_ms = (_time.time() - t_start) * 1000

        sage_meta = {}
        try:
            from const import sage_logic_gate, quantum_logic_gate
            char_counts = Counter(response.lower())
            total = max(len(response), 1)
            raw_entropy = -sum(
                (c / total) * math.log2(c / total)
                for c in char_counts.values() if c > 0
            )
            gated = sage_logic_gate(raw_entropy, "chat_response")
            q_amp = quantum_logic_gate(gated, depth=2)
            sage_meta = {
                "raw_entropy": round(raw_entropy, 4),
                "gated_entropy": round(gated, 4),
                "quantum_amplified": round(q_amp, 4),
                "entropy_reduction": round(max(0, raw_entropy - gated), 4),
            }
        except Exception:
            pass

        metrics = {}
        learned = False
        novelty = 0.0
        try:
            if hasattr(local_intellect, "_last_response_metrics"):
                m = local_intellect._last_response_metrics
                metrics = {
                    "qi": m.get("qi", 0),
                    "auto_improvements": m.get("auto_improvements", 0),
                    "mutations": m.get("mutations", 0),
                    "training_count": m.get("training_count", 0),
                    "ft_attn_patterns": m.get("ft_attn_patterns", 0),
                    "ft_mem_stored": m.get("ft_mem_stored", 0),
                    "ft_tfidf_vocab": m.get("ft_tfidf_vocab", 0),
                    "permanent_memory_count": m.get("permanent_memory_count", 0),
                    "latency_ms": round(think_ms, 1),
                }
                learned = m.get("learned", False)
                novelty = m.get("novelty", 0.0)
        except Exception:
            pass

        return {
            "status": "SUCCESS",
            "response": response,
            "model": "L104_UNIFIED_ASI",
            "mode": "sovereign",
            "sage_logic_gate": sage_meta,
            "latency_ms": round(think_ms, 1),
            "learned": learned,
            "novelty": round(novelty, 4),
            "metrics": metrics,
        }

    try:
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            result = await loop.run_in_executor(pool, unified_local_think, req.message)
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v6/chat/deep", tags=["AI"])
async def ai_chat_deep(req: ChatRequest):
    """Deep AI chat — full 6-stage neural processing."""
    try:
        from l104_local_intellect import local_intellect
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None, lambda: local_intellect.think(req.message)
        )
        return {"status": "SUCCESS", "response": response, "model": "L104_DEEP", "mode": "deep"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v6/chat/enhanced", tags=["AI"])
async def ai_chat_enhanced(req: ChatRequest):
    """Enhanced AI chat with Gemini fallback."""
    try:
        from l104_gemini_real import gemini_real
        if gemini_real.is_connected or gemini_real.connect():
            loop = asyncio.get_running_loop()
            try:
                response = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: gemini_real.generate(req.message)),
                    timeout=30.0,
                )
                if response and len(response) > 20:
                    return {"status": "SUCCESS", "response": response,
                            "model": f"GEMINI_{gemini_real.model_name}", "mode": "enhanced"}
            except asyncio.TimeoutError:
                pass
        from l104_local_intellect import local_intellect
        return {"status": "SUCCESS", "response": local_intellect.think(req.message),
                "model": "L104_LOCAL", "mode": "fallback"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


# ─── STREAMING ────────────────────────────────────────────────────────────────

async def _stream_generator(message: str, signal: str = ""):
    """Yield streaming tokens from local intellect."""
    import time
    from l104_local_intellect import local_intellect
    try:
        response = await asyncio.get_running_loop().run_in_executor(
            None, lambda: local_intellect.think(message or signal)
        )
        words = response.split()
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            yield f"data: {chunk}\n\n"
            await asyncio.sleep(0.01)
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: ERROR: {e}\n\n"
        yield "data: [DONE]\n\n"


@router.post("/api/v6/stream", tags=["AI"])
async def ai_stream(req: StreamRequest):
    """Streaming AI response using Server-Sent Events."""
    return StreamingResponse(
        _stream_generator(req.message, req.signal),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/api/stream", tags=["AI"])
async def ai_stream_v2(req: StreamRequest):
    """Alternate streaming endpoint (same backend as /api/v6/stream)."""
    return StreamingResponse(
        _stream_generator(req.message, req.signal),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/api/local/chat", tags=["AI"])
async def local_chat(req: ChatRequest):
    """Direct local inference endpoint — no cloud routing."""
    try:
        from l104_local_intellect import local_intellect
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, lambda: local_intellect.think(req.message))
        return {"status": "SUCCESS", "response": response, "model": "L104_LOCAL", "mode": "local"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.get("/debug/upstream", tags=["AI"])
async def debug_upstream():
    """Debug endpoint — shows model pool, cooldowns, quota state."""
    import state as _state
    return {
        "current_model_index": _state._current_model_index,
        "model_cooldowns": _state._model_cooldowns,
        "quota_exhausted_until": _state._quota_exhausted_until,
        "consecutive_quota_failures": _state._consecutive_quota_failures,
        "timestamp": datetime.now(UTC).isoformat(),
    }


# ─── RESEARCH & CODE ANALYSIS ─────────────────────────────────────────────────

@router.post("/api/v6/research", tags=["AI"])
async def sovereign_research(payload: dict = None):
    """Run deep sovereign research query using local intellect."""
    payload = payload or {}
    topic = payload.get("topic", "")
    if not topic:
        return JSONResponse(status_code=400, content={"error": "topic required"})
    try:
        from l104_local_intellect import local_intellect
        result = local_intellect.deep_research(topic)
        return {"status": "SUCCESS", "research": result, "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/api/v6/analyze-code", tags=["AI"])
async def analyze_code(payload: dict = None):
    """Run code analysis through the Code Engine."""
    payload = payload or {}
    code = payload.get("code", "")
    filename = payload.get("filename", "snippet.py")
    if not code:
        return JSONResponse(status_code=400, content={"error": "code required"})
    try:
        from l104_code_engine import code_engine
        result = await code_engine.analyze(code, filename)
        return {"status": "SUCCESS", "analysis": result, "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
