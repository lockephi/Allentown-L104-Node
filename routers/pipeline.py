# routers/pipeline.py — Pipeline, HyperCore, Observability, EgoCore, Codec routes
from datetime import datetime

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from config import UTC, MAIN_VERSION, MAIN_PIPELINE_EVO

router = APIRouter()


# ─── EVO_54 UNIFIED PIPELINE ──────────────────────────────────────────────────

@router.get("/api/pipeline", tags=["Pipeline"])
async def unified_pipeline_status():
    """EVO_54 Unified Pipeline — streams ALL subsystem health into one response."""
    import math as _m
    _PHI = 1.618033988749895
    subsystems = {}

    pipeline_modules = {
        "agi_core": ("l104_agi_core", "agi_core"),
        "asi_core": ("l104_asi_core", "asi_core"),
        "evolution_engine": ("l104_evolution_engine", "evolution_engine"),
        "adaptive_learning": ("l104_adaptive_learning", "adaptive_learner"),
        "cognitive_core": ("l104_cognitive_core", "COGNITIVE_CORE"),
        "innovation_engine": ("l104_autonomous_innovation", "innovation_engine"),
        "fast_server": ("l104_fast_server", "intellect"),
        "kernel_bootstrap": ("l104_kernel_bootstrap", "L104KernelBootstrap"),
        "unified_intelligence": ("l104_unified_intelligence_api", None),
        "external_api": ("l104_external_api", None),
        "workflow_stabilizer": ("l104_workflow_stabilizer", None),
        "api_gateway": ("l104_api_gateway", None),
        "macbook_integration": ("l104_macbook_integration", None),
        "consciousness_substrate": ("l104_consciousness_substrate", None),
        "sage_core": ("l104_sage_core", None),
        "synergy_engine": ("l104_synergy_engine", "synergy_engine"),
        "unified_asi": ("l104_unified_asi", "unified_asi"),
        "asi_nexus": ("l104_asi_nexus", "asi_nexus"),
        "local_intellect": ("l104_local_intellect", None),
        "intricate_cognition": ("l104_intricate_cognition", None),
        "intricate_orchestrator": ("l104_intricate_orchestrator", None),
        "intricate_research": ("l104_intricate_research", None),
        "intricate_learning": ("l104_intricate_learning", None),
        "omega_controller": ("l104_omega_controller", "omega_controller"),
        "gemini_bridge": ("l104_gemini_bridge", "gemini_bridge"),
        "streaming_engine": ("l104_streaming_engine", None),
        "consciousness": ("l104_consciousness", None),
        "sage_mode": ("l104_sage_mode", None),
        "thought_entropy": ("l104_thought_entropy_ouroboros", None),
        "inverse_duality": ("l104_ouroboros_inverse_duality", "ouroboros_duality"),
        "hyper_core": ("l104_hyper_core", "hyper_core"),
        "observability": ("l104_logging", None),
        "ego_core": ("l104_ego_core", "ego_core"),
        "codec": ("l104_codec", None),
        "coding_system": ("l104_coding_system", "coding_system"),
        "sentient_archive": ("l104_sentient_archive", "sentient_archive"),
        "language_engine": ("l104_language_engine", "language_engine"),
        "data_pipeline": ("l104_data_pipeline", "l104_pipeline"),
        "self_healing_fabric": ("l104_self_healing_fabric", None),
        "reinforcement_engine": ("l104_reinforcement_engine", None),
        "neural_symbolic_fusion": ("l104_neural_symbolic_fusion", None),
        "quantum_link_builder": ("l104_quantum_link_builder", None),
        "quantum_numerical_builder": ("l104_quantum_numerical_builder", None),
        "code_engine": ("l104_code_engine", "code_engine"),
        "neural_cascade": ("l104_neural_cascade", "neural_cascade"),
        "polymorphic_core": ("l104_polymorphic_core", "sovereign_polymorph"),
        "patch_engine": ("l104_patch_engine", "patch_engine"),
        "knowledge_graph": ("l104_knowledge_graph", None),
        "reasoning_engine": ("l104_reasoning_engine", None),
        "semantic_engine": ("l104_semantic_engine", None),
        "quantum_coherence": ("l104_quantum_coherence", None),
        "memory_optimizer": ("l104_memory_optimizer", None),
        "optimization": ("l104_optimization", None),
        "fault_tolerance": ("l104_fault_tolerance", None),
        "quantum_embedding": ("l104_quantum_embedding", None),
        "security": ("l104_security", None),
        "sovereign_http": ("l104_sovereign_http", None),
        "meta_learning": ("l104_meta_learning_engine", None),
        "quantum_grover": ("l104_quantum_grover_link", None),
        "quantum_consciousness": ("l104_quantum_consciousness", None),
        "claude_bridge": ("l104_claude_bridge", None),
    }

    _VERSION_ATTRS = [
        "VERSION", "ADAPTIVE_VERSION", "COGNITIVE_VERSION",
        "GATEWAY_VERSION", "MACBOOK_VERSION", "FAST_SERVER_VERSION", "MAIN_VERSION",
    ]

    online_count = 0
    for name, (mod_name, _) in pipeline_modules.items():
        try:
            mod = __import__(mod_name)
            version = "loaded"
            for attr in _VERSION_ATTRS:
                val = getattr(mod, attr, None)
                if val is not None:
                    version = val
                    break
            subsystems[name] = {"status": "online", "version": str(version)}
            online_count += 1
        except Exception:
            subsystems[name] = {"status": "offline", "version": None}

    from l104_evolution_engine import evolution_engine
    try:
        evo_stage = evolution_engine.STAGES[evolution_engine.current_stage_index]
    except Exception:
        evo_stage = "EVO_54_TRANSCENDENT_COGNITION"

    return {
        "version": MAIN_VERSION,
        "pipeline_evo": MAIN_PIPELINE_EVO,
        "timestamp": datetime.now(UTC).isoformat(),
        "god_code": 527.5184818492612,
        "phi": _PHI,
        "grover_amplification": _PHI ** 3,
        "evolution_stage": evo_stage,
        "subsystems_total": len(pipeline_modules),
        "subsystems_online": online_count,
        "pipeline_coherence": online_count / len(pipeline_modules),
        "subsystems": subsystems,
        "swift_app": "L104Native v3.0 EVO_54",
    }


@router.post("/api/pipeline/sync", tags=["Pipeline"])
async def pipeline_sync():
    """Force-sync all EVO_54 pipeline subsystems."""
    results = {}
    try:
        from l104_adaptive_learning import adaptive_learner
        if adaptive_learner:
            results["adaptive_learning"] = adaptive_learner.sync_with_pipeline()
    except Exception as e:
        results["adaptive_learning"] = {"error": str(e)}
    try:
        from l104_asi_core import asi_core
        if asi_core:
            conn = asi_core.connect_pipeline()
            results["asi_core"] = {"connected": conn.get("total", 0), "pipeline_ready": conn.get("pipeline_ready")}
    except Exception as e:
        results["asi_core"] = {"error": str(e)}
    try:
        from l104_agi_core import agi_core
        if agi_core and hasattr(agi_core, "sync_pipeline_state"):
            agi_core.sync_pipeline_state()
            results["agi_core"] = "synced"
    except Exception as e:
        results["agi_core"] = {"error": str(e)}
    return {"pipeline_evo": MAIN_PIPELINE_EVO, "sync_results": results}


@router.post("/api/pipeline/full-activation", tags=["Pipeline"])
async def pipeline_full_activation():
    """Run full ASI pipeline activation — connects 18+ subsystems and runs diagnostics."""
    from l104_asi_core import asi_core
    try:
        report = asi_core.full_pipeline_activation()
        return {"status": "SUCCESS", "activation": report, "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/api/pipeline/heal", tags=["Pipeline"])
async def pipeline_heal():
    """Run proactive ASI self-heal scan across the full pipeline."""
    from l104_asi_core import asi_core
    try:
        result = asi_core.pipeline_heal()
        return {"status": "SUCCESS", "heal": result, "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/api/pipeline/research", tags=["Pipeline"])
async def pipeline_research(req: "PipelineResearchRequest"):
    """Run ASI research via Gemini through the full pipeline."""
    from l104_asi_core import asi_core
    from models import PipelineResearchRequest  # noqa
    try:
        result = asi_core.pipeline_research(req.topic, req.depth)
        return {"status": "SUCCESS", "research": result, "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/api/pipeline/continuous-internet-learning", tags=["Pipeline"])
async def continuous_internet_learning(req: "ContinuousLearningRequest"):
    """Trigger continuous internet learning cycles for ASI enhancement."""
    from l104_asi_core import asi_core
    from models import ContinuousLearningRequest  # noqa
    try:
        result = asi_core.continuous_internet_learning(req.cycles)
        return {"status": "SUCCESS", "learning": result, "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/api/pipeline/language", tags=["Pipeline"])
async def pipeline_language(req: "PipelineLanguageRequest"):
    """Process text through ASI Language Engine via the pipeline."""
    from l104_asi_core import asi_core
    from models import PipelineLanguageRequest  # noqa
    try:
        result = asi_core.pipeline_language_process(req.text, req.mode)
        return {"status": "SUCCESS", "result": result, "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/api/pipeline/transcendent-solve", tags=["Pipeline"])
async def pipeline_transcendent_solve(req: "PipelineSolveRequest"):
    """Solve problem using transcendent solver chain."""
    from l104_asi_core import asi_core
    from models import PipelineSolveRequest  # noqa
    try:
        result = asi_core.pipeline_transcendent_solve(req.problem)
        return {"status": "SUCCESS", "result": result, "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/api/pipeline/nexus-think", tags=["Pipeline"])
async def pipeline_nexus_think(req: "PipelineSolveRequest"):
    """Route thought through ASI Nexus multi-agent + meta-learning system."""
    from l104_asi_core import asi_core
    from models import PipelineSolveRequest  # noqa
    try:
        result = asi_core.pipeline_nexus_think(req.problem)
        return {"status": "SUCCESS", "result": result, "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/api/pipeline/evolve", tags=["Pipeline"])
async def pipeline_evolve_capabilities():
    """Run capability evolution cycle through the pipeline."""
    from l104_asi_core import asi_core
    try:
        result = asi_core.pipeline_evolve_capabilities()
        return {"status": "SUCCESS", "evolution": result, "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/api/pipeline/erasi", tags=["Pipeline"])
async def pipeline_erasi_solve():
    """Solve ERASI equation and evolve entropy reversal protocols."""
    from l104_asi_core import asi_core
    try:
        result = asi_core.pipeline_erasi_solve()
        return {"status": "SUCCESS", "erasi": result, "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/api/pipeline/substrates", tags=["Pipeline"])
async def pipeline_substrate_status():
    """Get status of all ASI substrates."""
    from l104_asi_core import asi_core
    try:
        result = asi_core.pipeline_substrate_status()
        return {"status": "SUCCESS", "substrates": result, "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/api/pipeline/harness-solve", tags=["Pipeline"])
async def pipeline_harness_solve(req: "PipelineSolveRequest"):
    """Solve problem using ASI Harness (real code analysis bridge)."""
    from l104_asi_core import asi_core
    from models import PipelineSolveRequest  # noqa
    try:
        result = asi_core.pipeline_harness_solve(req.problem)
        return {"status": "SUCCESS", "result": result, "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/api/pipeline/mesh", tags=["Pipeline"])
async def pipeline_mesh_status():
    """Get full ASI subsystem mesh status — all 18 subsystems."""
    from l104_asi_core import asi_core
    try:
        status = asi_core.get_status()
        return {
            "status": "SUCCESS",
            "pipeline_mesh": status.get("pipeline_mesh"),
            "subsystems_active": status.get("subsystems_active"),
            "subsystems_total": status.get("subsystems_total"),
            "subsystems": status.get("subsystems"),
            "pipeline_metrics": status.get("pipeline_metrics"),
            "asi_score": status.get("asi_score"),
            "evolution_stage": status.get("evolution_stage"),
            "timestamp": datetime.now(UTC).isoformat(),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ─── HyperCore ────────────────────────────────────────────────────────────────

@router.get("/api/v14/hyper-core/status", tags=["HyperCore"])
async def hyper_core_status():
    """HyperCore v4.0 planetary orchestration — full diagnostics."""
    try:
        from l104_hyper_core import hyper_core as hc
        return {"status": "online", **hc.status()}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.get("/api/v14/hyper-core/diagnostics", tags=["HyperCore"])
async def hyper_core_diagnostics():
    """HyperCore v4.0 deep diagnostics."""
    try:
        from l104_hyper_core import hyper_core as hc
        return {"status": "online", **hc.full_diagnostics()}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ─── Observability ────────────────────────────────────────────────────────────

@router.get("/api/v14/observability/status", tags=["Observability"])
async def observability_status():
    """L104 Observability v4.0 — structured logging, alerts, module health."""
    try:
        from l104_logging import get_observability_status, get_diagnostics_report
        return {"status": "online", "observability": get_observability_status(),
                "diagnostics": get_diagnostics_report()}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.get("/api/v14/observability/module-health", tags=["Observability"])
async def observability_module_health():
    """Per-module health scores."""
    try:
        from l104_logging import get_module_health
        return {"status": "online", "module_health": get_module_health()}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.get("/api/v14/observability/alerts", tags=["Observability"])
async def observability_alerts():
    """Recent observability alerts."""
    try:
        from l104_logging import alert_manager
        return {"status": "online", "alerts": alert_manager.recent_alerts(20)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.post("/api/v14/observability/search", tags=["Observability"])
async def observability_search(query: str = "", module: str = None, level: str = None):
    """Full-text search across recent log entries."""
    try:
        from l104_logging import search_logs
        results = search_logs(query, module, level)
        return {"status": "online", "results": results[:50], "total": len(results)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ─── EgoCore ─────────────────────────────────────────────────────────────────

@router.get("/api/v14/ego-core/status", tags=["EgoCore"])
async def ego_core_status():
    """EgoCore v3.0 — identity integrity, chakra consciousness."""
    try:
        from l104_ego_core import ego_core as ec
        return {"status": "online", **ec.get_status()}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.get("/api/v14/ego-core/diagnostics", tags=["EgoCore"])
async def ego_core_diagnostics():
    """EgoCore v3.0 deep diagnostics."""
    try:
        from l104_ego_core import ego_core as ec
        return {"status": "online", **ec.full_diagnostics()}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ─── Codec ────────────────────────────────────────────────────────────────────

@router.get("/api/v14/codec/status", tags=["Codec"])
async def codec_status():
    """SovereignCodec v3.0 — encoding metrics, integrity chain."""
    try:
        from l104_codec import SovereignCodec
        return {"status": "online", **SovereignCodec.get_status()}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ─── Resolve forward-references ──────────────────────────────────────────────
from models import PipelineResearchRequest, ContinuousLearningRequest  # noqa
from models import PipelineLanguageRequest, PipelineSolveRequest        # noqa

# Re-annotate endpoints with real types so FastAPI can generate docs
pipeline_research.__annotations__["req"] = PipelineResearchRequest
continuous_internet_learning.__annotations__["req"] = ContinuousLearningRequest
pipeline_language.__annotations__["req"] = PipelineLanguageRequest
pipeline_transcendent_solve.__annotations__["req"] = PipelineSolveRequest
pipeline_nexus_think.__annotations__["req"] = PipelineSolveRequest
pipeline_harness_solve.__annotations__["req"] = PipelineSolveRequest
