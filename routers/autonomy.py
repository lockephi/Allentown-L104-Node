# routers/autonomy.py — Heart, Omni, Concept, Reality, Symmetry, Cloud, Audio, Choice, Storage routes
import base64
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()


# ─── HEART CORE v6 ───────────────────────────────────────────────────────────

@router.get("/api/v6/heart/status", tags=["Heart Core"])
async def heart_status():
    from l104_heart_core import heart_core
    return heart_core.get_status()


@router.post("/api/v6/heart/pulse", tags=["Heart Core"])
async def heart_pulse():
    from l104_heart_core import heart_core
    return heart_core.pulse()


@router.post("/api/v6/heart/resonate", tags=["Heart Core"])
async def heart_resonate(frequency: float = 527.5184818492612):
    from l104_heart_core import heart_core
    return heart_core.resonate(frequency)


@router.get("/api/v6/heart/rhythm", tags=["Heart Core"])
async def heart_rhythm():
    from l104_heart_core import heart_core
    return heart_core.get_rhythm()


# ─── OMNI CORE v7 ────────────────────────────────────────────────────────────

@router.get("/api/v7/omni/status", tags=["Omni Core"])
async def omni_status():
    from l104_omni_core import omni_core
    return omni_core.get_status()


@router.post("/api/v7/omni/perceive", tags=["Omni Core"])
async def omni_perceive(data: Dict[str, Any]):
    from l104_omni_core import omni_core
    return omni_core.perceive(data)


@router.post("/api/v7/omni/synthesize", tags=["Omni Core"])
async def omni_synthesize(signal: str = ""):
    from l104_omni_core import omni_core
    return omni_core.synthesize(signal)


@router.get("/api/v7/omni/awareness", tags=["Omni Core"])
async def omni_awareness():
    from l104_omni_core import omni_core
    return omni_core.get_awareness()


# ─── CONCEPT ENGINE v7 ───────────────────────────────────────────────────────

@router.post("/api/v7/concept/analyze", tags=["Concept Engine"])
async def concept_analyze(payload: Dict[str, Any]):
    from l104_concept_engine import concept_engine
    concept = payload.get("concept", "")
    if not concept:
        return JSONResponse(status_code=400, content={"error": "concept required"})
    return concept_engine.analyze(concept, payload.get("depth", 3))


# ─── REALITY CORE v8 ─────────────────────────────────────────────────────────

@router.post("/api/v8/reality/verify", tags=["Reality Core"])
async def reality_verify(payload: Dict[str, Any]):
    from l104_reality_core import reality_core
    proposition = payload.get("proposition", "")
    if not proposition:
        return JSONResponse(status_code=400, content={"error": "proposition required"})
    return reality_core.verify(proposition, payload.get("context", {}))


# ─── SYMMETRY CORE v8 ────────────────────────────────────────────────────────

@router.get("/api/v8/symmetry/status", tags=["Symmetry Core"])
async def symmetry_status():
    from l104_symmetry_core import symmetry_core
    return symmetry_core.get_status()


@router.post("/api/v8/symmetry/balance", tags=["Symmetry Core"])
async def symmetry_balance(payload: Dict[str, Any]):
    from l104_symmetry_core import symmetry_core
    return symmetry_core.balance(payload)


@router.get("/api/v8/symmetry/invariants", tags=["Symmetry Core"])
async def symmetry_invariants():
    from l104_symmetry_core import symmetry_core
    return symmetry_core.get_invariants()


# ─── REALITY BREACH v10 ──────────────────────────────────────────────────────

@router.post("/api/v10/reality/breach", tags=["Reality Breach"])
async def initiate_reality_breach(auth_token: str = "AUTH[LONDEL]"):
    from l104_reality_breach import reality_breach_engine
    return reality_breach_engine.initiate_breach(auth_token)


@router.get("/api/v10/reality/breach/status", tags=["Reality Breach"])
async def get_reality_breach_status():
    from l104_reality_breach import reality_breach_engine
    return reality_breach_engine.get_breach_status()


# ─── CLOUD AGENT v11 ─────────────────────────────────────────────────────────

@router.post("/api/v11/cloud/delegate", tags=["Cloud Agent"])
async def delegate_to_cloud_agent(task: Dict[str, Any]):
    import uuid
    from l104_cloud_agent_delegator import cloud_agent_delegator
    task_dict = {"type": task.get("type", ""), "data": task.get("data", {}),
                 "requirements": task.get("requirements") or [],
                 "id": task.get("id") or f"task_{uuid.uuid4().hex[:12]}"}
    return await cloud_agent_delegator.delegate(task_dict, task.get("agent"))


@router.get("/api/v11/cloud/status", tags=["Cloud Agent"])
async def get_cloud_agent_status():
    from l104_cloud_agent_delegator import cloud_agent_delegator
    return cloud_agent_delegator.get_status()


@router.post("/api/v11/cloud/register", tags=["Cloud Agent"])
async def register_cloud_agent(registration: Dict[str, Any]):
    from l104_cloud_agent_delegator import cloud_agent_delegator
    name = registration.get("name", "")
    config = {"endpoint": registration.get("endpoint", ""),
              "capabilities": registration.get("capabilities", []),
              "priority": registration.get("priority", 999),
              "enabled": registration.get("enabled", True),
              "client_id": registration.get("client_id")}
    success = cloud_agent_delegator.register_agent(name, config)
    if success:
        return {"status": "SUCCESS", "message": f"Cloud agent '{name}' registered successfully", "agent": name}
    raise HTTPException(status_code=500, detail=f"Failed to register cloud agent '{name}'")


@router.get("/api/v11/cloud/agents", tags=["Cloud Agent"])
async def list_cloud_agents():
    from l104_cloud_agent_delegator import cloud_agent_delegator
    return {"agents": cloud_agent_delegator.agents, "count": len(cloud_agent_delegator.agents)}


# ─── AUDIO ANALYSIS v6 ───────────────────────────────────────────────────────

@router.post("/api/v6/audio/analyze", tags=["Autonomy"])
async def analyze_audio(payload: Dict[str, Any]):
    """Analyze audio for resonance and tuning verification against 527.5184818492612 Hz."""
    from l104_audio_resonance import analyze_audio_resonance
    audio_source = payload.get("audio_source", "")
    check_tuning = payload.get("check_tuning", True)
    if not audio_source:
        return JSONResponse(status_code=400, content={"error": "audio_source required"})
    try:
        result = await analyze_audio_resonance(audio_source, check_tuning)
        if result.get("success"):
            return result
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {result.get('error')}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze audio: {str(e)}")


# ─── CLOUD DELEGATION v6 ─────────────────────────────────────────────────────

@router.post("/api/v6/cloud/delegate", tags=["Autonomy"])
async def delegate_task_v6(task: Dict[str, Any]):
    from l104_cloud_agent_v6 import delegate_to_cloud_agent_v6
    task_dict = {"type": task.get("task_type", ""), "payload": task.get("payload", {}),
                 "priority": task.get("priority", "normal")}
    result = await delegate_to_cloud_agent_v6(task_dict)
    if result.get("success"):
        return result
    if result.get("fallback_to_local"):
        return {"status": "Cloud delegation failed, fallback available", "cloud_result": result,
                "local_processing": True, "message": "Task can be processed locally if needed"}
    raise HTTPException(status_code=500, detail=f"Cloud delegation failed: {result.get('error')}")


# ─── AUTONOMY STATUS v6 ───────────────────────────────────────────────────────

@router.get("/api/v6/autonomy/status", tags=["Autonomy"])
async def get_autonomy_status():
    import os
    from datetime import datetime
    from config import UTC, CLOUD_AGENT_URL, CLOUD_AGENT_KEY, AUTONOMY_ENABLED, ENABLE_AUTO_APPROVE, AUTO_APPROVE_MODE
    cloud_agent_ready = bool(CLOUD_AGENT_URL)
    cloud_agent_configured = bool(CLOUD_AGENT_URL and CLOUD_AGENT_KEY)
    return {"autonomy_enabled": AUTONOMY_ENABLED,
            "auto_approve": {"enabled": ENABLE_AUTO_APPROVE, "mode": AUTO_APPROVE_MODE,
                              "description": "Controls automatic approval of autonomous commits"},
            "cloud_agent": {"configured": cloud_agent_configured, "url": CLOUD_AGENT_URL or None,
                             "ready": cloud_agent_ready,
                             "description": "Ready if URL configured; fully configured if both URL and KEY provided"},
            "sovereign_commit": {"available": True, "requires": ["GITHUB_PAT environment variable"],
                                  "auto_approve_default": ENABLE_AUTO_APPROVE},
            "timestamp": datetime.now(UTC).isoformat()}


# ─── CHOICE ENGINE v10 ───────────────────────────────────────────────────────

@router.post("/api/v10/choice/reflective", tags=["Choice Engine"])
async def trigger_reflective_choice():
    from l104_choice_engine import choice_engine
    return await choice_engine.evaluate_and_act()


@router.get("/api/v10/choice/status", tags=["Choice Engine"])
async def get_choice_status():
    from l104_choice_engine import choice_engine
    return {"is_autonomous": choice_engine.autonomous_active,
            "current_intention": choice_engine.current_intention,
            "history_count": len(choice_engine.history)}


# ─── STORAGE MASTERY v8 ──────────────────────────────────────────────────────

@router.post("/api/v8/storage/mastery/compress", tags=["Storage Mastery"])
async def storage_mastery_compress(data: str):
    from l104_disk_compression_mastery import compression_mastery
    compressed_bytes, stats = compression_mastery.mastery_compress(data.encode("utf-8"))
    return {"compressed_data_b64": base64.b64encode(compressed_bytes).decode("utf-8"),
            "stats": {**stats, "topological_state": "ANYON_BRAIDED"}}


@router.post("/api/v8/storage/mastery/decompress", tags=["Storage Mastery"])
async def storage_mastery_decompress(compressed_data_b64: str):
    from l104_disk_compression_mastery import compression_mastery
    compressed_bytes = base64.b64decode(compressed_data_b64)
    original_bytes = compression_mastery.mastery_decompress(compressed_bytes)
    return {"original_data": original_bytes.decode("utf-8")}
