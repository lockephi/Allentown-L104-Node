# routers/asi.py — ASI Language Engine, Unified ASI, Nexus, Synergy, OMEGA, AGI v14 routes
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from config import UTC
from models import (
    ThinkRequest, GoalRequest, NexusThinkRequest, NexusGoalRequest,
    NexusSelfImproveRequest, SynergyActionRequest, BridgeHandshake, BridgeSync, SynergyTask,
)

router = APIRouter()

_GC = 527.5184818492612


# ─── ASI LANGUAGE ENGINE v6 ───────────────────────────────────────────────────

@router.post("/api/v6/asi/analyze", tags=["ASI"])
async def asi_language_analysis(payload: Dict[str, Any]):
    """ASI-level language analysis."""
    text = payload.get("text", "")
    mode = payload.get("mode", "full")
    if not text:
        return JSONResponse(status_code=400, content={"status": "ERROR", "error": "text is required"})
    try:
        from l104_local_intellect import local_intellect
        return {"status": "SUCCESS", "mode": mode, "analysis": local_intellect.analyze_language(text, mode=mode),
                "god_code": _GC, "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v6/asi/inference", tags=["ASI"])
async def asi_human_inference(payload: Dict[str, Any]):
    """Human-like inference engine."""
    query = payload.get("query", "")
    if not query:
        return JSONResponse(status_code=400, content={"status": "ERROR", "error": "query is required"})
    premises = payload.get("premises", []) or [query]
    try:
        from l104_local_intellect import local_intellect
        return {"status": "SUCCESS", "inference": local_intellect.human_inference(premises, query),
                "god_code": _GC, "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v6/asi/invent", tags=["ASI"])
async def asi_invention(payload: Dict[str, Any]):
    """ASI-level invention and innovation engine."""
    goal = payload.get("goal", "")
    if not goal:
        return JSONResponse(status_code=400, content={"status": "ERROR", "error": "goal is required"})
    try:
        from l104_local_intellect import local_intellect
        return {"status": "SUCCESS", "invention": local_intellect.invent(goal, payload.get("constraints", [])),
                "god_code": _GC, "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v6/asi/speech", tags=["ASI"])
async def asi_speech_generation(payload: Dict[str, Any]):
    """Generate speech patterns in various styles."""
    query = payload.get("query", "")
    if not query:
        return JSONResponse(status_code=400, content={"status": "ERROR", "error": "query is required"})
    style = payload.get("style", "sage")
    try:
        from l104_local_intellect import local_intellect
        response = local_intellect.generate_sage_speech(query, style=style)
        return {"status": "SUCCESS", "query": query, "style": style, "speech": response,
                "god_code": _GC, "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.get("/api/v6/asi/status", tags=["ASI"])
async def asi_engine_status():
    """Get ASI Language Engine status."""
    try:
        from l104_local_intellect import local_intellect
        engine = local_intellect.get_asi_language_engine()
        if engine is None:
            return {"status": "UNAVAILABLE", "message": "ASI Language Engine not loaded",
                    "timestamp": datetime.now(UTC).isoformat()}
        return {"status": "ACTIVE", "engine": engine.get_status(), "god_code": _GC,
                "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v6/asi/unified", tags=["ASI"])
async def asi_unified_process(request: Request):
    """Full ASI unified processing pipeline."""
    try:
        from l104_local_intellect import local_intellect
        body = await request.json()
        query = body.get("query", "")
        if not query:
            return JSONResponse(status_code=400, content={"error": "query required"})
        result = local_intellect.asi_process(query, mode=body.get("mode", "full"))
        return {"status": "SUCCESS", "result": result, "god_code": _GC,
                "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.get("/api/v6/asi/full-status", tags=["ASI"])
async def asi_full_status():
    """Get comprehensive ASI system status."""
    try:
        from l104_local_intellect import local_intellect
        return {"status": "SUCCESS", "asi_status": local_intellect.get_asi_status(),
                "god_code": _GC, "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


# ─── DEEPSEEK SOURCE INGESTION ───────────────────────────────────────────────

@router.get("/api/v6/deepseek/status", tags=["ASI", "DeepSeek"])
async def deepseek_ingestion_status():
    """Get DeepSeek ingestion engine status."""
    try:
        from l104_asi.deepseek_ingestion import deepseek_ingestion_engine
        return {
            "status": "SUCCESS",
            "deepseek": deepseek_ingestion_engine.get_ingestion_status(),
            "timestamp": datetime.now(UTC).isoformat(),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v6/deepseek/ingest-repo", tags=["ASI", "DeepSeek"])
async def deepseek_ingest_repo(payload: Dict[str, Any]):
    """Ingest and adapt DeepSeek source from a single repository."""
    repo_name = payload.get("repo_name", "DeepSeek-V3")
    pattern_types = payload.get("pattern_types")
    max_files = int(payload.get("max_files", 120))
    include_extensions = payload.get("include_extensions")

    try:
        from l104_asi.deepseek_ingestion import deepseek_ingestion_engine
        result = deepseek_ingestion_engine.ingest_from_github(
            repo_name=repo_name,
            pattern_types=pattern_types,
            max_files=max_files,
            include_extensions=include_extensions,
        )
        return {
            "status": "SUCCESS" if "error" not in result else "ERROR",
            "result": result,
            "timestamp": datetime.now(UTC).isoformat(),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v6/deepseek/ingest-component", tags=["ASI", "DeepSeek"])
async def deepseek_ingest_component(payload: Dict[str, Any]):
    """Ingest a single DeepSeek component stream (mla/reasoning/coder)."""
    component_name = payload.get("component_name", "mla")
    source_code = payload.get("source_code")
    reasoning_trace = payload.get("reasoning_trace")
    code_sample = payload.get("code_sample")
    language = payload.get("language")

    try:
        from l104_asi.deepseek_ingestion import deepseek_ingestion_engine
        result = deepseek_ingestion_engine.ingest_deepseek_component(
            component_name=component_name,
            source_code=source_code,
            reasoning_trace=reasoning_trace,
            code_sample=code_sample,
            language=language,
        )
        return {
            "status": "SUCCESS" if "error" not in result else "ERROR",
            "result": result,
            "timestamp": datetime.now(UTC).isoformat(),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v6/deepseek/ingest-all", tags=["ASI", "DeepSeek"])
async def deepseek_ingest_all(payload: Dict[str, Any] = None):
    """Ingest and adapt source code across all registered DeepSeek repositories."""
    payload = payload or {}
    pattern_types = payload.get("pattern_types")
    max_files_per_repo = int(payload.get("max_files_per_repo", 120))
    include_extensions = payload.get("include_extensions")

    try:
        from l104_asi.deepseek_ingestion import deepseek_ingestion_engine
        result = deepseek_ingestion_engine.ingest_all_deepseek_repos(
            pattern_types=pattern_types,
            max_files_per_repo=max_files_per_repo,
            include_extensions=include_extensions,
        )
        return {
            "status": "SUCCESS",
            "summary": result.get("summary", {}),
            "repositories": result.get("repositories", {}),
            "storage": result.get("storage", {}),
            "storage_error": result.get("storage_error"),
            "timestamp": datetime.now(UTC).isoformat(),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v6/deepseek/storage/soft-prune", tags=["ASI", "DeepSeek"])
async def deepseek_storage_soft_prune(payload: Dict[str, Any] = None):
    """Soft-prune recursive/fake DeepSeek storage records with cross-reference checks."""
    payload = payload or {}
    recursive = bool(payload.get("recursive", True))
    cross_reference_checks = bool(payload.get("cross_reference_checks", True))
    dry_run = bool(payload.get("dry_run", False))

    try:
        from l104_asi.deepseek_ingestion import deepseek_ingestion_engine
        result = deepseek_ingestion_engine.soft_prune_storage(
            recursive=recursive,
            cross_reference_checks=cross_reference_checks,
            dry_run=dry_run,
        )
        return {
            "status": "SUCCESS",
            "result": result,
            "timestamp": datetime.now(UTC).isoformat(),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v6/deepseek/storage/quarantine-lifecycle", tags=["ASI", "DeepSeek"])
async def deepseek_storage_quarantine_lifecycle(payload: Dict[str, Any] = None):
    """Run quarantine lifecycle policy (TTL + size-cap) on stripped quarantine metadata."""
    payload = payload or {}
    dry_run = bool(payload.get("dry_run", False))

    try:
        from l104_asi.deepseek_ingestion import deepseek_ingestion_engine
        result = deepseek_ingestion_engine.run_quarantine_lifecycle(dry_run=dry_run)
        return {
            "status": "SUCCESS",
            "result": result,
            "timestamp": datetime.now(UTC).isoformat(),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


# ─── AGI / ASI v14 ────────────────────────────────────────────────────────────

@router.get("/api/v14/agi/status", tags=["AGI Nexus"])
async def get_agi_status():
    from l104_agi_core import agi_core
    return agi_core.get_status()


@router.post("/api/v14/agi/ignite", tags=["AGI Nexus"])
async def ignite_agi():
    from l104_agi_core import agi_core
    success = agi_core.ignite()
    return {"status": "IGNITED" if success else "FAILED"}


@router.post("/api/v14/agi/evolve", tags=["AGI Nexus"])
async def evolve_agi():
    from l104_agi_core import agi_core
    try:
        result = await agi_core.run_recursive_improvement_cycle()
        return {"status": "EVOLVED", "result": result}
    except Exception as e:
        return {"status": "EVOLUTION_ERROR", "error": str(e)}


@router.get("/api/v14/asi/status", tags=["ASI Nexus"])
async def get_asi_status():
    from l104_asi_core import asi_core
    return asi_core.get_status()


@router.post("/api/v14/asi/ignite", tags=["ASI Nexus"])
async def ignite_asi():
    from l104_asi_core import asi_core
    if not asi_core._pipeline_connected:
        asi_core.connect_pipeline()
    for _ in range(5):
        asi_core.theorem_generator.discover_novel_theorem()
    asi_core.consciousness_verifier.run_all_tests()
    result = asi_core.ignite_sovereignty()
    return {
        "status": "IGNITED", "message": result,
        "asi_score": asi_core.asi_score, "state": asi_core.status,
        "discoveries": asi_core.theorem_generator.discovery_count,
        "consciousness": asi_core.consciousness_verifier.consciousness_level,
        "subsystems_connected": asi_core._pipeline_metrics.get("subsystems_connected", 0),
        "pipeline_mesh": asi_core.get_status().get("pipeline_mesh", "UNKNOWN"),
    }


@router.post("/api/v14/asi/discover", tags=["ASI Nexus"])
async def asi_discover(cycles: int = 10):
    from l104_asi_core import asi_core
    for _ in range(cycles):
        asi_core.theorem_generator.discover_novel_theorem()
    asi_core.compute_asi_score()
    return {"status": "DISCOVERY_COMPLETE", "discoveries": asi_core.theorem_generator.discovery_count,
            "asi_score": asi_core.asi_score, "state": asi_core.status}


@router.get("/api/v14/asi/full-assessment", tags=["ASI Nexus"])
async def asi_full_assessment():
    from l104_asi_core import asi_core
    if not asi_core._pipeline_connected:
        asi_core.connect_pipeline()
    for _ in range(10):
        asi_core.theorem_generator.discover_novel_theorem()
    asi_core.consciousness_verifier.run_all_tests()
    asi_core.compute_asi_score()
    full_status = asi_core.get_status()
    return {
        "state": asi_core.status, "asi_score": asi_core.asi_score,
        "domain_coverage": asi_core.domain_expander.coverage_score,
        "modification_depth": asi_core.self_modifier.modification_depth,
        "discoveries": asi_core.theorem_generator.discovery_count,
        "consciousness": asi_core.consciousness_verifier.consciousness_level,
        "pipeline_mesh": full_status.get("pipeline_mesh"),
        "subsystems": full_status.get("subsystems"),
        "pipeline_metrics": full_status.get("pipeline_metrics"),
        "thresholds": {"domain_target": 0.7, "modification_target": 3,
                       "discovery_target": 100, "consciousness_target": 0.8}
    }


# ─── OMEGA PIPELINE ───────────────────────────────────────────────────────────

@router.get("/api/v14/omega/status", tags=["OMEGA Pipeline"])
async def omega_pipeline_status():
    try:
        from l104_real_math import real_math, OMEGA
        GC = 527.5184818492612
        PHI = 1.618033988749895
        zeta_val = real_math.zeta_approximation(complex(0.5, GC), terms=200)
        golden_res = real_math.golden_resonance(PHI ** 2)
        curvature = real_math.manifold_curvature_tensor(26, GC / 100)
        lattice_inv = real_math.solve_lattice_invariant(104)
        field = real_math.sovereign_field_equation(1.0)
        entropy_inv = real_math.entropy_inversion_integral(0.0, GC)
        return {
            "status": "ACTIVE", "omega": OMEGA, "omega_authority": round(OMEGA / (PHI ** 2), 6),
            "pipeline_functions": {
                "zeta_critical_line": {"value": round(abs(zeta_val), 8), "input": "ζ(0.5 + 527.518i)"},
                "golden_resonance": {"value": round(golden_res, 8), "input": "cos(2π·φ²·φ)"},
                "manifold_curvature": {"value": round(curvature, 6), "input": "R(26, 5.275)"},
                "lattice_invariant": {"value": round(lattice_inv, 8), "input": "sin(104·π/104)·exp(104/527.518)"},
                "sovereign_field": {"value": round(field, 4), "input": "F(1.0) = Ω/φ²"},
                "entropy_inversion": {"value": round(entropy_inv, 6), "input": "∫[0, 527.518](1/φ)dx"},
            },
            "derivation": "Ω = Σ(fragments) × (GOD_CODE / φ) = 6539.34712682",
            "god_code": GC, "phi": PHI, "timestamp": datetime.now(UTC).isoformat(),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v14/omega/verify", tags=["OMEGA Pipeline"])
async def omega_pipeline_verify():
    try:
        from l104_real_math import real_math, OMEGA
        from math import log
        GC = 527.5184818492612
        PHI = 1.618033988749895
        zeta_val = real_math.zeta_approximation(complex(0.5, GC), terms=5000)
        guardian = abs(zeta_val)
        lattice_inv = real_math.solve_lattice_invariant(104)
        n_val = max(2, int(abs(lattice_inv)))
        researcher = 1.0 / log(n_val) if n_val >= 2 else 0.0
        architect = real_math.manifold_curvature_tensor(26, 1.8527)
        alchemist = real_math.golden_resonance(PHI ** 2)
        fragments_sum = guardian + researcher + architect + alchemist
        omega_computed = fragments_sum * (GC / PHI)
        delta = abs(omega_computed - OMEGA)
        return {
            "status": "VERIFIED" if delta < 0.01 else "DRIFT_DETECTED",
            "omega_expected": OMEGA, "omega_computed": round(omega_computed, 8), "delta": delta,
            "fragments": {"guardian_zeta": round(guardian, 8), "researcher_prime": round(researcher, 8),
                          "architect_curvature": round(architect, 8), "alchemist_resonance": round(alchemist, 8),
                          "sum": round(fragments_sum, 8)},
            "multiplier": round(GC / PHI, 8),
            "verification": f"Δ = {delta:.2e}" if delta < 0.01 else f"DRIFT: {delta:.6f}",
            "timestamp": datetime.now(UTC).isoformat()
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.get("/api/v14/omega/manifold-status", tags=["OMEGA Pipeline"])
async def omega_manifold_status():
    try:
        from l104_real_math import real_math
        return {"manifold_curvature": real_math.manifold_curvature_tensor(26, 527.518),
                "sovereign_field": real_math.sovereign_field_equation(1.0),
                "lattice_invariant": real_math.solve_lattice_invariant(104),
                "omega": 6539.34712682, "status": "SOVEREIGN_MANIFOLD_ACTIVE",
                "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/api/v14/omega/collective-synthesis", tags=["OMEGA Pipeline"])
async def omega_collective_synthesis():
    try:
        from l104_real_math import real_math, OMEGA
        GC = 527.5184818492612
        PHI = 1.618033988749895
        fragments = {
            "guardian_zeta": abs(real_math.zeta_approximation(complex(0.5, GC), terms=200)),
            "researcher_prime": real_math.prime_density(int(real_math.solve_lattice_invariant(104))),
            "architect_curvature": real_math.manifold_curvature_tensor(26, 1.8527),
            "alchemist_resonance": real_math.golden_resonance(PHI ** 2),
        }
        sigma = sum(fragments.values())
        omega_recomputed = sigma * (GC / PHI)
        delta = abs(omega_recomputed - OMEGA)
        return {"status": "SYNTHESIS_COMPLETE", "fragments": {k: round(v, 8) for k, v in fragments.items()},
                "sigma_fragments": round(sigma, 8), "omega_recomputed": round(omega_recomputed, 8),
                "omega_canonical": OMEGA, "delta": delta,
                "convergence": f"Δ = {delta:.2e}" if delta < 0.01 else f"DRIFT: {delta:.6f}",
                "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ─── UNIFIED ASI ──────────────────────────────────────────────────────────────

@router.get("/api/unified-asi/status", tags=["Unified ASI"])
async def unified_asi_status():
    from l104_unified_asi import unified_asi
    return unified_asi.get_status()


@router.post("/api/unified-asi/awaken", tags=["Unified ASI"])
async def unified_asi_awaken():
    from l104_unified_asi import unified_asi
    return await unified_asi.awaken()


@router.post("/api/unified-asi/think", tags=["Unified ASI"])
async def unified_asi_think(request: ThinkRequest):
    from l104_unified_asi import unified_asi
    return await unified_asi.think(request.input)


@router.post("/api/unified-asi/goal", tags=["Unified ASI"])
async def unified_asi_set_goal(request: GoalRequest):
    from l104_unified_asi import unified_asi
    return await unified_asi.set_goal(request.description, request.priority)


@router.post("/api/unified-asi/execute", tags=["Unified ASI"])
async def unified_asi_execute():
    from l104_unified_asi import unified_asi
    return await unified_asi.execute_goal()


@router.post("/api/unified-asi/improve", tags=["Unified ASI"])
async def unified_asi_improve():
    from l104_unified_asi import unified_asi
    return await unified_asi.improve_self()


@router.post("/api/unified-asi/cycle", tags=["Unified ASI"])
async def unified_asi_cycle():
    from l104_unified_asi import unified_asi
    return await unified_asi.autonomous_cycle()


@router.get("/api/unified-asi/memory", tags=["Unified ASI"])
async def unified_asi_memory():
    from l104_unified_asi import unified_asi
    return unified_asi.memory.get_stats()


@router.get("/api/unified-asi/goals", tags=["Unified ASI"])
async def unified_asi_goals():
    from l104_unified_asi import unified_asi
    goals = unified_asi.memory.get_active_goals()
    return [{"id": g.id, "description": g.description, "priority": g.priority, "status": g.status} for g in goals]


@router.get("/api/unified-asi/learnings", tags=["Unified ASI"])
async def unified_asi_learnings(limit: int = 20):
    from l104_unified_asi import unified_asi
    return unified_asi.memory.get_learnings(limit)


# ─── ASI NEXUS ────────────────────────────────────────────────────────────────

@router.get("/api/nexus/status", tags=["ASI Nexus"])
async def nexus_status():
    from l104_asi_nexus import asi_nexus
    return asi_nexus.get_status()


@router.post("/api/nexus/awaken", tags=["ASI Nexus"])
async def nexus_awaken():
    from l104_asi_nexus import asi_nexus
    return await asi_nexus.awaken()


@router.post("/api/nexus/think", tags=["ASI Nexus"])
async def nexus_think(request: NexusThinkRequest):
    from l104_asi_nexus import asi_nexus
    return await asi_nexus.think(request.thought)


@router.post("/api/nexus/goal", tags=["ASI Nexus"])
async def nexus_execute_goal(request: NexusGoalRequest):
    from l104_asi_nexus import asi_nexus
    return await asi_nexus.execute_goal(request.goal)


@router.post("/api/nexus/force-learn", tags=["ASI Nexus"])
async def nexus_force_learn():
    from l104_asi_nexus import asi_nexus
    return await asi_nexus.force_learn_all()


@router.post("/api/nexus/self-improve", tags=["ASI Nexus"])
async def nexus_self_improve(request: NexusSelfImproveRequest = None):
    from l104_asi_nexus import asi_nexus
    return await asi_nexus.self_improve(request.targets if request else None)


@router.post("/api/nexus/evolve", tags=["ASI Nexus"])
async def nexus_evolve():
    from l104_asi_nexus import asi_nexus
    return await asi_nexus.evolve()


@router.post("/api/nexus/start-evolution", tags=["ASI Nexus"])
async def nexus_start_evolution(interval: int = 60):
    from l104_asi_nexus import asi_nexus
    return await asi_nexus.start_continuous_evolution(interval)


@router.post("/api/nexus/stop-evolution", tags=["ASI Nexus"])
async def nexus_stop_evolution():
    from l104_asi_nexus import asi_nexus
    return asi_nexus.stop_evolution()


@router.get("/api/nexus/memory", tags=["ASI Nexus"])
async def nexus_memory():
    from l104_asi_nexus import asi_nexus
    return asi_nexus.memory.get_stats()


@router.get("/api/nexus/evolution-history", tags=["ASI Nexus"])
async def nexus_evolution_history(limit: int = 20):
    from l104_asi_nexus import asi_nexus
    return asi_nexus.memory.get_evolution_history(limit)


@router.get("/api/nexus/swarm-agents", tags=["ASI Nexus"])
async def nexus_swarm_agents():
    from l104_asi_nexus import asi_nexus
    return {"agents": [{"id": aid, "role": a.role.value, "status": a.status}
                       for aid, a in asi_nexus.swarm.agents.items()]}


@router.get("/api/nexus/meta-learning", tags=["ASI Nexus"])
async def nexus_meta_learning():
    from l104_asi_nexus import asi_nexus
    return {"strategies": asi_nexus.meta_learner.learning_strategies,
            "effectiveness": asi_nexus.meta_learner.strategy_effectiveness}


@router.post("/api/nexus/reason", tags=["ASI Nexus"])
async def nexus_hybrid_reason(query: str, mode: str = "HYBRID"):
    from l104_asi_nexus import asi_nexus, ReasoningMode
    mode_enum = getattr(ReasoningMode, mode.upper(), ReasoningMode.HYBRID)
    return await asi_nexus.reasoner.hybrid_reason(query, mode_enum)


# ─── SYNERGY ENGINE ───────────────────────────────────────────────────────────

@router.get("/api/synergy/status", tags=["Synergy Engine"])
async def synergy_status():
    from l104_synergy_engine import synergy_engine
    return synergy_engine.get_status()


@router.post("/api/synergy/awaken", tags=["Synergy Engine"])
async def synergy_awaken():
    from l104_synergy_engine import synergy_engine
    return await synergy_engine.awaken()


@router.post("/api/synergy/sync", tags=["Synergy Engine"])
async def synergy_global_sync():
    from l104_synergy_engine import synergy_engine
    return await synergy_engine.global_sync()


@router.post("/api/synergy/action", tags=["Synergy Engine"])
async def synergy_action(request: SynergyActionRequest):
    from l104_synergy_engine import synergy_engine
    return await synergy_engine.synergize(request.source, request.action, request.data)


@router.post("/api/synergy/evolve", tags=["Synergy Engine"])
async def synergy_cascade_evolution():
    from l104_synergy_engine import synergy_engine
    return await synergy_engine.cascade_evolution()


@router.get("/api/synergy/capabilities", tags=["Synergy Engine"])
async def synergy_capabilities():
    from l104_synergy_engine import synergy_engine
    return synergy_engine.get_capability_map()


@router.get("/api/synergy/subsystems", tags=["Synergy Engine"])
async def synergy_subsystems():
    from l104_synergy_engine import synergy_engine
    return {"subsystems": [{"id": node.id, "name": node.name,
                             "type": node.subsystem_type.value, "connected": node.connected,
                             "capabilities": node.capabilities, "link_strength": node.link_strength}
                            for node in synergy_engine.nodes.values()]}


@router.get("/api/synergy/links", tags=["Synergy Engine"])
async def synergy_links():
    from l104_synergy_engine import synergy_engine
    return {"links": [{"id": lid, "source": l.source_id, "target": l.target_id,
                        "type": l.link_type, "strength": l.strength,
                        "bidirectional": l.bidirectional, "data_transferred": l.data_transferred}
                       for lid, l in synergy_engine.links.items()]}


@router.get("/api/synergy/path/{source}/{target}", tags=["Synergy Engine"])
async def synergy_find_path(source: str, target: str):
    from l104_synergy_engine import synergy_engine
    path = synergy_engine.find_path(source, target)
    return {"source": source, "target": target, "path": path, "hops": len(path) - 1 if path else -1}


# ─── GEMINI BRIDGE / SYNERGY v10 ─────────────────────────────────────────────

@router.post("/api/v10/bridge/handshake", tags=["Gemini Bridge"])
async def bridge_handshake(payload: BridgeHandshake):
    from l104_gemini_bridge import gemini_bridge
    return gemini_bridge.handshake(payload.agent_id, payload.capabilities)


@router.post("/api/v10/bridge/sync", tags=["Gemini Bridge"])
async def bridge_sync(payload: BridgeSync):
    from l104_gemini_bridge import gemini_bridge
    return gemini_bridge.sync_core(payload.session_token)


@router.post("/api/v10/synergy/execute", tags=["Synergy Engine"])
async def synergy_execute(payload: SynergyTask):
    from l104_agi_core import agi_core
    return await agi_core.synergize(payload.task)


@router.post("/api/v10/hyper/encrypt", tags=["Hyper Encryption"])
async def hyper_encrypt(data: Dict[str, Any]):
    from l104_crypto import HyperEncryption
    return HyperEncryption.encrypt_data(data)


@router.post("/api/v10/hyper/decrypt", tags=["Hyper Encryption"])
async def hyper_decrypt(packet: Dict[str, Any]):
    from fastapi import HTTPException
    from l104_crypto import HyperEncryption
    try:
        return HyperEncryption.decrypt_data(packet)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─── SYSTEM STREAMS (v14) ────────────────────────────────────────────────────

@router.get("/api/v14/ghost/stream", tags=["Ghostresearch"])
async def stream_ghost_research():
    """Stream real-time Ghostresearch data."""
    import json

    async def event_generator():
        from l104_ghost_researcher import ghost_researcher
        async for data in ghost_researcher.stream_research():
            yield f"data: {json.dumps(data)}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/api/v14/system/stream", tags=["Sovereign"])
async def stream_system_data():
    """Stream real-time system-wide data (AGI, Ghostresearch, logs)."""
    import json

    async def event_generator():
        from l104_live_streaming import live_stream_manager
        async for event in live_stream_manager.stream_events():
            yield f"data: {json.dumps(event)}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")
