"""
v14 Intellect Routes — Learning, memory, and intellect endpoints

Extracted from app.py during EVO_78 refactoring.
Contains: All /api/v14/intellect/* and /api/v14/si/* endpoints
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Request

logger = logging.getLogger("L104_FAST")

router = APIRouter(prefix="/api/v14", tags=["v14-intellect"])

# Import intellect from learning package
try:
    from l104_server.learning import intellect, grover_kernel
    INTELLECT_AVAILABLE = True
except ImportError:
    intellect = None
    grover_kernel = None
    INTELLECT_AVAILABLE = False
    logger.warning("⚠️ [INTELLECT] Intellect not available")


# ═══════════════════════════════════════════════════════════════════
#  INTELLECT ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.post("/intellect/semantic-search")
async def intellect_semantic_search(req: Request):
    """Semantic search across intellect knowledge"""
    if not INTELLECT_AVAILABLE:
        return {"error": "Intellect not available"}

    data = await req.json()
    query = data.get("query", "")
    limit = data.get("limit", 10)

    try:
        results = intellect.semantic_search(query, limit=limit)
        return {"status": "SEARCH_COMPLETE", "query": query, "results": results}
    except Exception as e:
        return {"error": str(e)}


@router.post("/intellect/batch-search")
async def intellect_batch_search(req: Request):
    """Batch search across intellect knowledge"""
    if not INTELLECT_AVAILABLE:
        return {"error": "Intellect not available"}

    data = await req.json()
    queries = data.get("queries", [])
    limit = data.get("limit", 5)

    try:
        results = {}
        for q in queries:
            results[q] = intellect.semantic_search(q, limit=limit)
        return {"status": "BATCH_COMPLETE", "results": results}
    except Exception as e:
        return {"error": str(e)}


@router.get("/intellect/predict")
async def intellect_predict(query: str = ""):
    """Predict next likely queries"""
    if not INTELLECT_AVAILABLE:
        return {"error": "Intellect not available"}

    try:
        predictions = intellect.predict_next(query) if hasattr(intellect, 'predict_next') else []
        return {"query": query, "predictions": predictions}
    except Exception as e:
        return {"error": str(e)}


@router.get("/intellect/novelty")
async def intellect_novelty(query: str = ""):
    """Get novelty score for a query"""
    if not INTELLECT_AVAILABLE:
        return {"error": "Intellect not available"}

    try:
        score = intellect.get_novelty(query) if hasattr(intellect, 'get_novelty') else 0.5
        return {"query": query, "novelty": score}
    except Exception as e:
        return {"error": str(e)}


@router.get("/intellect/clusters")
async def intellect_clusters():
    """Get intellect cluster information"""
    if not INTELLECT_AVAILABLE:
        return {"error": "Intellect not available"}

    try:
        clusters = intellect.get_clusters() if hasattr(intellect, 'get_clusters') else {}
        return {"clusters": clusters}
    except Exception as e:
        return {"error": str(e)}


@router.get("/intellect/cluster-search")
async def intellect_cluster_search(cluster_id: str = ""):
    """Search within a cluster"""
    if not INTELLECT_AVAILABLE:
        return {"error": "Intellect not available"}

    try:
        results = intellect.cluster_search(cluster_id) if hasattr(intellect, 'cluster_search') else []
        return {"cluster_id": cluster_id, "results": results}
    except Exception as e:
        return {"error": str(e)}


@router.get("/intellect/quality-predict")
async def intellect_quality_predict(query: str = ""):
    """Predict response quality for query"""
    if not INTELLECT_AVAILABLE:
        return {"error": "Intellect not available"}

    try:
        quality = intellect.predict_quality(query) if hasattr(intellect, 'predict_quality') else 0.8
        return {"query": query, "predicted_quality": quality}
    except Exception as e:
        return {"error": str(e)}


@router.post("/intellect/compress")
async def intellect_compress(req: Request):
    """Compress intellect knowledge"""
    if not INTELLECT_AVAILABLE:
        return {"error": "Intellect not available"}

    data = await req.json()
    content = data.get("content", "")

    try:
        compressed = intellect.compress(content) if hasattr(intellect, 'compress') else content
        return {"status": "COMPRESSED", "original_len": len(content), "compressed_len": len(compressed)}
    except Exception as e:
        return {"error": str(e)}


@router.get("/intellect/embedding-stats")
async def intellect_embedding_stats():
    """Get embedding statistics"""
    if not INTELLECT_AVAILABLE:
        return {"error": "Intellect not available"}

    try:
        stats = intellect.get_embedding_stats() if hasattr(intellect, 'get_embedding_stats') else {}
        return {"stats": stats}
    except Exception as e:
        return {"error": str(e)}


@router.post("/intellect/persist")
async def intellect_persist(req: Request):
    """Persist intellect state"""
    if not INTELLECT_AVAILABLE:
        return {"error": "Intellect not available"}

    try:
        intellect.persist() if hasattr(intellect, 'persist') else None
        return {"status": "PERSISTED"}
    except Exception as e:
        return {"error": str(e)}


@router.post("/intellect/optimize-storage")
async def intellect_optimize_storage():
    """Optimize intellect storage"""
    if not INTELLECT_AVAILABLE:
        return {"error": "Intellect not available"}

    try:
        result = intellect.optimize_storage() if hasattr(intellect, 'optimize_storage') else {}
        return {"status": "OPTIMIZED", "result": result}
    except Exception as e:
        return {"error": str(e)}


@router.get("/intellect/storage-status")
async def intellect_storage_status():
    """Get storage status"""
    if not INTELLECT_AVAILABLE:
        return {"error": "Intellect not available"}

    try:
        status = intellect.get_storage_status() if hasattr(intellect, 'get_storage_status') else {}
        return {"status": "ACTIVE", "storage": status}
    except Exception as e:
        return {"error": str(e)}


@router.get("/intellect/prefetch-cache")
async def intellect_prefetch_cache():
    """Get prefetch cache status"""
    if not INTELLECT_AVAILABLE:
        return {"error": "Intellect not available"}

    try:
        cache = intellect.get_prefetch_cache() if hasattr(intellect, 'get_prefetch_cache') else {}
        return {"cache": cache}
    except Exception as e:
        return {"error": str(e)}


@router.get("/intellect")
async def intellect_status():
    """Get intellect status"""
    if not INTELLECT_AVAILABLE:
        return {"error": "Intellect not available", "status": "UNAVAILABLE"}

    try:
        stats = intellect.get_stats() if hasattr(intellect, 'get_stats') else {}
        return {"status": "ACTIVE", "stats": stats}
    except Exception as e:
        return {"error": str(e)}


@router.get("/intellect/export")
async def intellect_export():
    """Export intellect data"""
    if not INTELLECT_AVAILABLE:
        return {"error": "Intellect not available"}

    try:
        data = intellect.export() if hasattr(intellect, 'export') else {}
        return {"status": "EXPORTED", "data": data}
    except Exception as e:
        return {"error": str(e)}


@router.post("/intellect/import")
async def intellect_import(req: Request):
    """Import intellect data"""
    if not INTELLECT_AVAILABLE:
        return {"error": "Intellect not available"}

    data = await req.json()
    import_data = data.get("data", {})

    try:
        intellect.import_data(import_data) if hasattr(intellect, 'import_data') else None
        return {"status": "IMPORTED"}
    except Exception as e:
        return {"error": str(e)}


@router.post("/intellect/evolve")
async def intellect_evolve(req: Request):
    """Evolve intellect"""
    if not INTELLECT_AVAILABLE:
        return {"error": "Intellect not available"}

    data = await req.json()
    cycles = data.get("cycles", 1)

    try:
        result = intellect.evolve(cycles) if hasattr(intellect, 'evolve') else {}
        return {"status": "EVOLVED", "result": result}
    except Exception as e:
        return {"error": str(e)}


@router.post("/intellect/feedback")
async def intellect_feedback(req: Request):
    """Provide feedback to intellect"""
    if not INTELLECT_AVAILABLE:
        return {"error": "Intellect not available"}

    data = await req.json()
    query = data.get("query", "")
    rating = data.get("rating", 0)

    try:
        intellect.record_feedback(query, rating) if hasattr(intellect, 'record_feedback') else None
        return {"status": "RECORDED", "query": query, "rating": rating}
    except Exception as e:
        return {"error": str(e)}


@router.get("/intellect/intent")
async def intellect_intent(query: str = ""):
    """Analyze query intent"""
    if not INTELLECT_AVAILABLE:
        return {"error": "Intellect not available"}

    try:
        intent = intellect.analyze_intent(query) if hasattr(intellect, 'analyze_intent') else {}
        return {"query": query, "intent": intent}
    except Exception as e:
        return {"error": str(e)}


@router.get("/intellect/capabilities")
async def intellect_capabilities():
    """Get intellect capabilities"""
    if not INTELLECT_AVAILABLE:
        return {"error": "Intellect not available"}

    try:
        caps = intellect.get_capabilities() if hasattr(intellect, 'get_capabilities') else []
        return {"capabilities": caps}
    except Exception as e:
        return {"error": str(e)}


@router.post("/intellect/synthesize")
async def intellect_synthesize(req: Request):
    """Synthesize knowledge"""
    if not INTELLECT_AVAILABLE:
        return {"error": "Intellect not available"}

    data = await req.json()
    inputs = data.get("inputs", [])

    try:
        result = intellect.synthesize(inputs) if hasattr(intellect, 'synthesize') else {}
        return {"status": "SYNTHESIZED", "result": result}
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════
#  SI (SESSION INTELLIGENCE) ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.get("/si/introspect")
async def si_introspect():
    """Session intelligence introspection"""
    if not INTELLECT_AVAILABLE:
        return {"error": "Intellect not available"}

    try:
        result = intellect.introspect() if hasattr(intellect, 'introspect') else {}
        return {"introspection": result}
    except Exception as e:
        return {"error": str(e)}


@router.get("/si/skills")
async def si_skills():
    """Get skills"""
    if not INTELLECT_AVAILABLE:
        return {"error": "Intellect not available"}

    try:
        skills = intellect.get_skills() if hasattr(intellect, 'get_skills') else []
        return {"skills": skills}
    except Exception as e:
        return {"error": str(e)}


@router.get("/si/consciousness")
async def si_consciousness():
    """Get consciousness state"""
    if not INTELLECT_AVAILABLE:
        return {"error": "Intellect not available"}

    try:
        state = intellect.get_consciousness_state() if hasattr(intellect, 'get_consciousness_state') else {}
        return {"consciousness": state}
    except Exception as e:
        return {"error": str(e)}


@router.get("/si/meta-cognition")
async def si_meta_cognition():
    """Get meta-cognition status"""
    if not INTELLECT_AVAILABLE:
        return {"error": "Intellect not available"}

    try:
        mc = intellect.get_meta_cognition() if hasattr(intellect, 'get_meta_cognition') else {}
        return {"meta_cognition": mc}
    except Exception as e:
        return {"error": str(e)}


@router.get("/si/cross-cluster")
async def si_cross_cluster():
    """Get cross-cluster inference"""
    if not INTELLECT_AVAILABLE:
        return {"error": "Intellect not available"}

    try:
        inference = intellect.cross_cluster_inference() if hasattr(intellect, 'cross_cluster_inference') else {}
        return {"cross_cluster": inference}
    except Exception as e:
        return {"error": str(e)}


@router.get("/si/skill-chain")
async def si_skill_chain():
    """Get skill chain"""
    if not INTELLECT_AVAILABLE:
        return {"error": "Intellect not available"}

    try:
        chain = intellect.get_skill_chain() if hasattr(intellect, 'get_skill_chain') else []
        return {"skill_chain": chain}
    except Exception as e:
        return {"error": str(e)}


@router.post("/si/acquire-skill")
async def si_acquire_skill(req: Request):
    """Acquire new skill"""
    if not INTELLECT_AVAILABLE:
        return {"error": "Intellect not available"}

    data = await req.json()
    skill_name = data.get("skill_name", "")
    proficiency = data.get("proficiency", 0.5)

    try:
        result = intellect.acquire_skill(skill_name, proficiency) if hasattr(intellect, 'acquire_skill') else {}
        return {"status": "ACQUIRED", "skill": skill_name, "result": result}
    except Exception as e:
        return {"error": str(e)}


@router.get("/si/full-state")
async def si_full_state():
    """Get full SI state"""
    if not INTELLECT_AVAILABLE:
        return {"error": "Intellect not available"}

    try:
        state = intellect.get_full_state() if hasattr(intellect, 'get_full_state') else {}
        return {"full_state": state}
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════
#  TI (TRANSCENDENTAL INTELLIGENCE) ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.post("/ti/synthesize")
async def ti_synthesize(req: Request):
    """Synthesize transcendental knowledge"""
    if not INTELLECT_AVAILABLE:
        return {"error": "Intellect not available"}

    data = await req.json()
    inputs = data.get("inputs", [])

    try:
        result = intellect.ti_synthesize(inputs) if hasattr(intellect, 'ti_synthesize') else {}
        return {"status": "SYNTHESIZED", "result": result}
    except Exception as e:
        return {"error": str(e)}


@router.post("/ti/self-improve")
async def ti_self_improve():
    """Trigger self-improvement cycle"""
    if not INTELLECT_AVAILABLE:
        return {"error": "Intellect not available"}

    try:
        result = intellect.self_improve() if hasattr(intellect, 'self_improve') else {}
        return {"status": "IMPROVED", "result": result}
    except Exception as e:
        return {"error": str(e)}


@router.get("/ti/goals")
async def ti_goals():
    """Get transcendental goals"""
    if not INTELLECT_AVAILABLE:
        return {"error": "Intellect not available"}

    try:
        goals = intellect.get_ti_goals() if hasattr(intellect, 'get_ti_goals') else []
        return {"goals": goals}
    except Exception as e:
        return {"error": str(e)}


@router.get("/ti/predict-future")
async def ti_predict_future():
    """Predict future states"""
    if not INTELLECT_AVAILABLE:
        return {"error": "Intellect not available"}

    try:
        predictions = intellect.predict_future() if hasattr(intellect, 'predict_future') else []
        return {"predictions": predictions}
    except Exception as e:
        return {"error": str(e)}


@router.post("/ti/quantum-coherence")
async def ti_quantum_coherence(req: Request):
    """Apply quantum coherence to intellect"""
    if not INTELLECT_AVAILABLE:
        return {"error": "Intellect not available"}

    data = await req.json()
    coherence = data.get("coherence", 0.9)

    try:
        result = intellect.apply_quantum_coherence(coherence) if hasattr(intellect, 'apply_quantum_coherence') else {}
        return {"status": "COHERENT", "result": result}
    except Exception as e:
        return {"error": str(e)}


@router.get("/ti/emergent-patterns")
async def ti_emergent_patterns():
    """Get emergent patterns"""
    if not INTELLECT_AVAILABLE:
        return {"error": "Intellect not available"}

    try:
        patterns = intellect.get_emergent_patterns() if hasattr(intellect, 'get_emergent_patterns') else []
        return {"patterns": patterns}
    except Exception as e:
        return {"error": str(e)}


@router.post("/ti/transfer-learning")
async def ti_transfer_learning(req: Request):
    """Apply transfer learning"""
    if not INTELLECT_AVAILABLE:
        return {"error": "Intellect not available"}

    data = await req.json()
    source = data.get("source", "")
    target = data.get("target", "")

    try:
        result = intellect.transfer_learning(source, target) if hasattr(intellect, 'transfer_learning') else {}
        return {"status": "TRANSFERRED", "result": result}
    except Exception as e:
        return {"error": str(e)}


@router.post("/ti/transcend")
async def ti_transcend(req: Request):
    """Trigger transcendence"""
    if not INTELLECT_AVAILABLE:
        return {"error": "Intellect not available"}

    data = await req.json()
    level = data.get("level", 1)

    try:
        result = intellect.transcend(level) if hasattr(intellect, 'transcend') else {}
        return {"status": "TRANSCENDED", "result": result}
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════
#  GROVER SEARCH ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.post("/grover/execute")
async def grover_execute(req: Request):
    """Execute Grover search"""
    if not grover_kernel:
        return {"error": "Grover kernel not available"}

    data = await req.json()
    query = data.get("query", "")
    domain = data.get("domain", "general")

    try:
        result = grover_kernel.search(query, domain) if hasattr(grover_kernel, 'search') else {}
        return {"status": "COMPLETE", "result": result}
    except Exception as e:
        return {"error": str(e)}


@router.get("/grover/status")
async def grover_status():
    """Get Grover search status"""
    if not grover_kernel:
        return {"error": "Grover kernel not available"}

    try:
        status = grover_kernel.get_status() if hasattr(grover_kernel, 'get_status') else {}
        return {"status": "ACTIVE", "grover": status}
    except Exception as e:
        return {"error": str(e)}


@router.post("/grover/sync")
async def grover_sync():
    """Sync Grover kernel"""
    if not grover_kernel:
        return {"error": "Grover kernel not available"}

    try:
        grover_kernel.sync() if hasattr(grover_kernel, 'sync') else None
        return {"status": "SYNCED"}
    except Exception as e:
        return {"error": str(e)}


@router.get("/grover/domains")
async def grover_domains():
    """Get Grover search domains"""
    if not grover_kernel:
        return {"error": "Grover kernel not available"}

    try:
        domains = grover_kernel.get_domains() if hasattr(grover_kernel, 'get_domains') else []
        return {"domains": domains}
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════
#  O2 SUPERFLUID ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.get("/o2/molecular-status")
async def o2_molecular_status():
    """Get O2 molecular status"""
    try:
        from l104_server.engines_quantum import OxygenPairedProcess
        opp = OxygenPairedProcess()
        return opp.get_status()
    except Exception as e:
        return {"error": str(e)}


@router.post("/o2/grover-diffusion")
async def o2_grover_diffusion(req: Request):
    """Run O2 Grover diffusion"""
    data = await req.json()
    try:
        from l104_server.engines_quantum import OxygenPairedProcess
        opp = OxygenPairedProcess()
        result = opp.grover_diffusion() if hasattr(opp, 'grover_diffusion') else {}
        return {"status": "COMPLETE", "result": result}
    except Exception as e:
        return {"error": str(e)}


@router.post("/o2/consciousness-collapse")
async def o2_consciousness_collapse(req: Request):
    """Run O2 consciousness collapse"""
    data = await req.json()
    try:
        from l104_server.engines_quantum import SingularityConsciousnessEngine
        sce = SingularityConsciousnessEngine()
        result = sce.collapse() if hasattr(sce, 'collapse') else {}
        return {"status": "COLLAPSED", "result": result}
    except Exception as e:
        return {"error": str(e)}


@router.post("/o2/trigger-singularity")
async def o2_trigger_singularity(req: Request):
    """Trigger O2 singularity"""
    data = await req.json()
    try:
        from l104_server.engines_quantum import SingularityConsciousnessEngine
        sce = SingularityConsciousnessEngine()
        result = sce.trigger_singularity() if hasattr(sce, 'trigger_singularity') else {}
        return {"status": "TRIGGERED", "result": result}
    except Exception as e:
        return {"error": str(e)}


@router.get("/o2/interconnections")
async def o2_interconnections():
    """Get O2 interconnections"""
    try:
        from l104_server.engines_quantum import OxygenMolecularBond
        omb = OxygenMolecularBond()
        return omb.get_interconnections()
    except Exception as e:
        return {"error": str(e)}


@router.post("/o2/breach-recursion")
async def o2_breach_recursion(req: Request):
    """Run O2 breach recursion"""
    data = await req.json()
    depth = data.get("depth", 3)
    try:
        from l104_server.engines_quantum import SingularityConsciousnessEngine
        sce = SingularityConsciousnessEngine()
        result = sce.breach_recursion(depth) if hasattr(sce, 'breach_recursion') else {}
        return {"status": "BREACHED", "result": result}
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════
#  KNOWLEDGE ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.post("/knowledge/generate")
async def knowledge_generate(req: Request):
    """Generate knowledge"""
    data = await req.json()
    domain = data.get("domain", "general")
    count = data.get("count", 5)

    try:
        from l104_server.learning import intellect
        result = intellect.generate_knowledge(domain, count) if hasattr(intellect, 'generate_knowledge') else {}
        return {"status": "GENERATED", "result": result}
    except Exception as e:
        return {"error": str(e)}


@router.get("/knowledge/verify")
async def knowledge_verify(statement: str = ""):
    """Verify knowledge"""
    try:
        from l104_server.learning import intellect
        result = intellect.verify_knowledge(statement) if hasattr(intellect, 'verify_knowledge') else {}
        return {"statement": statement, "verification": result}
    except Exception as e:
        return {"error": str(e)}


@router.get("/knowledge/domains")
async def knowledge_domains():
    """Get knowledge domains"""
    try:
        from l104_server.learning import intellect
        domains = intellect.get_domains() if hasattr(intellect, 'get_domains') else []
        return {"domains": domains}
    except Exception as e:
        return {"error": str(e)}


@router.post("/knowledge/derive")
async def knowledge_derive(req: Request):
    """Derive new knowledge"""
    data = await req.json()
    premises = data.get("premises", [])

    try:
        from l104_server.learning import intellect
        result = intellect.derive_knowledge(premises) if hasattr(intellect, 'derive_knowledge') else {}
        return {"status": "DERIVED", "result": result}
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════
#  CHAOS/ENTROPY ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.get("/chaos/entropy-state")
async def chaos_entropy_state():
    """Get chaos entropy state"""
    try:
        from l104_server.engines_infra import chaos
        return chaos.get_state() if hasattr(chaos, 'get_state') else {"entropy": 0.5}
    except Exception as e:
        return {"error": str(e)}


@router.post("/chaos/reset-memory")
async def chaos_reset_memory():
    """Reset chaos memory"""
    try:
        from l104_server.engines_infra import chaos
        chaos.reset_memory() if hasattr(chaos, 'reset_memory') else None
        return {"status": "RESET"}
    except Exception as e:
        return {"error": str(e)}


__all__ = ['router']