# ZENITH_UPGRADE_ACTIVE: 2026-02-18T00:00:00.000000
# EVO_54_TRANSCENDENT_COGNITION
# L104 Extended Pipeline API — 9 High-Value Module Endpoints
# Wires: coding_system, sentient_archive, language_engine, data_pipeline,
#         self_healing_fabric, reinforcement_engine, neural_symbolic_fusion,
#         quantum_link_builder, quantum_numerical_builder

"""L104 Extended Pipeline API Router — Exposes 9 newly-wired modules as REST endpoints."""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger("l104_extended_pipeline")

router = APIRouter(prefix="/api/v6/ext", tags=["ExtendedPipeline"])

# ═══════════════════════════════════════════════════════════════════════════════
# Sacred Constants
# ═══════════════════════════════════════════════════════════════════════════════
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895

# ═══════════════════════════════════════════════════════════════════════════════
# Lazy module references — loaded on first use to avoid circular imports
# Negative results (None) are retried after TTL seconds
# ═══════════════════════════════════════════════════════════════════════════════
_modules: Dict[str, Any] = {}
_module_fail_times: Dict[str, float] = {}
_RETRY_TTL = 60.0  # seconds before retrying a failed module import

def _get(name: str):
    """Lazy-load module singleton on first access. Retries failed imports after TTL."""
    cached = _modules.get(name, "__MISSING__")
    if cached != "__MISSING__":
        # If cached as None, check if we should retry
        if cached is None:
            fail_time = _module_fail_times.get(name, 0)
            if time.time() - fail_time < _RETRY_TTL:
                return None
            # TTL expired, clear cache to retry
            del _modules[name]
        else:
            return cached
    try:
        if name == "coding_system":
            from l104_coding_system import coding_system
            _modules[name] = coding_system
        elif name == "sentient_archive":
            from l104_sentient_archive import sentient_archive
            _modules[name] = sentient_archive
        elif name == "language_engine":
            from l104_language_engine import language_engine
            _modules[name] = language_engine
        elif name == "data_pipeline":
            from l104_data_pipeline import l104_pipeline
            _modules[name] = l104_pipeline
        elif name == "healing_fabric":
            from l104_self_healing_fabric import activate_healing_fabric
            _modules[name] = activate_healing_fabric()
        elif name == "rl_engine":
            from l104_reinforcement_engine import create_rl_engine
            _modules[name] = create_rl_engine()
        elif name == "neural_symbolic":
            from l104_neural_symbolic_fusion import create_neural_symbolic_fusion
            _modules[name] = create_neural_symbolic_fusion()
        elif name == "quantum_link_builder":
            from l104_quantum_link_builder import QuantumLinkBuilder
            _modules[name] = QuantumLinkBuilder()
        elif name == "quantum_numerical":
            from l104_quantum_numerical_builder import GOD_CODE_HP, PHI_HP, PHI_GROWTH_HP
            _modules[name] = {"GOD_CODE_HP": GOD_CODE_HP, "PHI_HP": PHI_HP, "PHI_GROWTH_HP": PHI_GROWTH_HP}
    except Exception as e:
        logger.warning(f"Extended pipeline: {name} unavailable: {e}")
        _modules[name] = None
        _module_fail_times[name] = time.time()
    return _modules.get(name)


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST MODELS
# ═══════════════════════════════════════════════════════════════════════════════
class CodeReviewRequest(BaseModel):
    source: str = Field(..., description="Source code to review", max_length=500_000)
    filename: str = Field(default="", description="Optional filename for language detection", max_length=500)

class NLPRequest(BaseModel):
    text: str = Field(..., description="Text to process", max_length=100_000)
    task: str = Field(default="analyze", description="NLP task: analyze, tokenize, entities, sentiment, summarize")

class PipelineRequest(BaseModel):
    source: str = Field(default="", description="Source path or identifier", max_length=2000)
    operation: str = Field(default="status", description="Pipeline operation")

class HealingRequest(BaseModel):
    target_path: str = Field(default=".", description="Path to diagnose", max_length=500)
    auto_heal: bool = Field(default=False, description="Auto-apply fixes")

    @classmethod
    def validate_path(cls, path: str) -> str:
        """Prevent path traversal attacks."""
        normalized = os.path.normpath(path)
        if ".." in normalized or normalized.startswith("/"):
            raise ValueError("Path traversal not allowed")
        return normalized

class ArchiveRequest(BaseModel):
    force_collect: bool = Field(default=False, description="Force fresh collection")

class ReasoningRequest(BaseModel):
    query: str = Field(..., description="Query for neural-symbolic reasoning", max_length=50_000)
    depth: int = Field(default=3, description="Reasoning depth (1-10)", ge=1, le=10)

class RLRequest(BaseModel):
    state: str = Field(..., description="State representation", max_length=10_000)
    action: Optional[str] = Field(default=None, description="Action to evaluate")
    reward: Optional[float] = Field(default=None, description="Reward signal")

class QuantumLinkRequest(BaseModel):
    source: str = Field(default="", description="Source module or concept")
    target: str = Field(default="", description="Target module or concept")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CODING SYSTEM — Code Review, Quality Gates, AI Context
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/coding/status")
async def coding_system_status():
    """Get coding intelligence system status."""
    cs = _get("coding_system")
    if not cs:
        return {"status": "offline", "module": "l104_coding_system"}
    try:
        status = cs.status() if hasattr(cs, 'status') else {}
        return {"status": "online", "module": "l104_coding_system", "data": status}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@router.post("/coding/review")
async def coding_review(req: CodeReviewRequest):
    """Run code review through the coding intelligence system."""
    cs = _get("coding_system")
    if not cs:
        raise HTTPException(503, "Coding system not available")
    try:
        if hasattr(cs, 'review'):
            result = cs.review(req.source, req.filename)
        elif hasattr(cs, 'analyze'):
            result = cs.analyze(req.source)
        else:
            raise HTTPException(501, f"Review/analyze not implemented on {type(cs).__name__}")
        return {"review": result, "god_code": GOD_CODE}
    except Exception as e:
        raise HTTPException(500, f"Review failed: {e}")

@router.post("/coding/quality-gate")
async def coding_quality_gate(req: CodeReviewRequest):
    """Run quality gate check on source code."""
    cs = _get("coding_system")
    if not cs:
        raise HTTPException(503, "Coding system not available")
    try:
        if hasattr(cs, 'quality_gate'):
            result = cs.quality_gate(req.source)
        elif hasattr(cs, 'check_quality'):
            result = cs.check_quality(req.source)
        else:
            raise HTTPException(501, f"Quality gate not implemented on {type(cs).__name__}")
        return {"quality_gate": result}
    except Exception as e:
        raise HTTPException(500, f"Quality gate failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SENTIENT ARCHIVE — Golden Record, Memory Crystallization
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/archive/status")
async def archive_status():
    """Get sentient archive status."""
    sa = _get("sentient_archive")
    if not sa:
        return {"status": "offline", "module": "l104_sentient_archive"}
    try:
        status = sa.status() if hasattr(sa, 'status') else {"type": type(sa).__name__}
        return {"status": "online", "module": "l104_sentient_archive", "data": status}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@router.post("/archive/cycle")
async def archive_cycle(req: ArchiveRequest):
    """Run full archive cycle — Collect → Crystallize → Consolidate → Fuse."""
    sa = _get("sentient_archive")
    if not sa:
        raise HTTPException(503, "Sentient archive not available")
    try:
        if hasattr(sa, 'full_archive_cycle'):
            result = sa.full_archive_cycle(force_collect=req.force_collect)
        elif hasattr(sa, 'archive'):
            result = sa.archive()
        else:
            raise HTTPException(501, f"Archive cycle not implemented on {type(sa).__name__}")
        return {"archive_cycle": result, "god_code": GOD_CODE}
    except Exception as e:
        raise HTTPException(500, f"Archive cycle failed: {e}")

@router.get("/archive/recall")
async def archive_recall(query: str = Query(..., description="Memory recall query")):
    """Recall memories from the sentient archive."""
    sa = _get("sentient_archive")
    if not sa:
        raise HTTPException(503, "Sentient archive not available")
    try:
        if hasattr(sa, 'recall'):
            result = sa.recall(query)
        elif hasattr(sa, 'search'):
            result = sa.search(query)
        else:
            raise HTTPException(501, f"Recall not implemented on {type(sa).__name__}")
        return {"recall": result, "query": query}
    except Exception as e:
        raise HTTPException(500, f"Recall failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LANGUAGE ENGINE — NLP Toolkit
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/language/status")
async def language_status():
    """Get language engine status."""
    le = _get("language_engine")
    if not le:
        return {"status": "offline", "module": "l104_language_engine"}
    try:
        status = le.status() if hasattr(le, 'status') else {"type": type(le).__name__}
        return {"status": "online", "module": "l104_language_engine", "data": status}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@router.post("/language/process")
async def language_process(req: NLPRequest):
    """Process text through the language engine."""
    le = _get("language_engine")
    if not le:
        raise HTTPException(503, "Language engine not available")
    try:
        task_methods = {
            "analyze": "analyze",
            "tokenize": "tokenize",
            "entities": "extract_entities",
            "sentiment": "sentiment",
            "summarize": "summarize",
        }
        method_name = task_methods.get(req.task, "analyze")
        method = getattr(le, method_name, None)
        if method:
            result = method(req.text)
        elif hasattr(le, 'process'):
            result = le.process(req.text)
        else:
            raise HTTPException(501, f"Method {method_name} not implemented on {type(le).__name__}")
        return {"result": result, "task": req.task}
    except Exception as e:
        raise HTTPException(500, f"Language processing failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DATA PIPELINE — ETL Framework
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/pipeline/status")
async def pipeline_status():
    """Get data pipeline engine status."""
    dp = _get("data_pipeline")
    if not dp:
        return {"status": "offline", "module": "l104_data_pipeline"}
    try:
        status = dp.status() if hasattr(dp, 'status') else {"type": type(dp).__name__}
        return {"status": "online", "module": "l104_data_pipeline", "data": status}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@router.post("/pipeline/run")
async def pipeline_run(req: PipelineRequest):
    """Execute a data pipeline operation."""
    dp = _get("data_pipeline")
    if not dp:
        raise HTTPException(503, "Data pipeline not available")
    try:
        if hasattr(dp, 'run'):
            result = dp.run(req.source, req.operation)
        elif hasattr(dp, 'execute'):
            result = dp.execute(req.operation)
        elif hasattr(dp, 'process'):
            result = dp.process(req.source)
        else:
            raise HTTPException(501, f"Pipeline run not implemented on {type(dp).__name__}")
        return {"result": result, "operation": req.operation}
    except Exception as e:
        raise HTTPException(500, f"Pipeline run failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SELF-HEALING FABRIC — Autonomous Diagnostics & Repair
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/healing/status")
async def healing_status():
    """Get self-healing fabric status."""
    hf = _get("healing_fabric")
    if not hf:
        return {"status": "offline", "module": "l104_self_healing_fabric"}
    try:
        status = hf.status() if hasattr(hf, 'status') else \
                 hf.get_health_report() if hasattr(hf, 'get_health_report') else \
                 {"type": type(hf).__name__}
        return {"status": "online", "module": "l104_self_healing_fabric", "data": status}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@router.post("/healing/diagnose")
async def healing_diagnose(req: HealingRequest):
    """Run diagnostic scan through the self-healing fabric."""
    # Validate path to prevent traversal
    try:
        safe_path = HealingRequest.validate_path(req.target_path)
    except ValueError as e:
        raise HTTPException(400, str(e))
    hf = _get("healing_fabric")
    if not hf:
        raise HTTPException(503, "Self-healing fabric not available")
    try:
        if hasattr(hf, 'diagnose'):
            result = hf.diagnose(safe_path)
        elif hasattr(hf, 'scan'):
            result = hf.scan(safe_path)
        elif hasattr(hf, 'run_diagnostics'):
            result = hf.run_diagnostics()
        else:
            raise HTTPException(501, f"Diagnose not implemented on {type(hf).__name__}")
        return {"diagnosis": result, "auto_heal": req.auto_heal}
    except Exception as e:
        raise HTTPException(500, f"Diagnosis failed: {e}")

@router.post("/healing/repair")
async def healing_repair():
    """Trigger self-healing repair cycle."""
    hf = _get("healing_fabric")
    if not hf:
        raise HTTPException(503, "Self-healing fabric not available")
    try:
        if hasattr(hf, 'heal'):
            result = hf.heal()
        elif hasattr(hf, 'repair'):
            result = hf.repair()
        elif hasattr(hf, 'auto_heal'):
            result = hf.auto_heal()
        else:
            raise HTTPException(501, f"Heal not implemented on {type(hf).__name__}")
        return {"repair": result}
    except Exception as e:
        raise HTTPException(500, f"Repair failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. REINFORCEMENT ENGINE — RL Primitives
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/rl/status")
async def rl_status():
    """Get reinforcement learning engine status."""
    rl = _get("rl_engine")
    if not rl:
        return {"status": "offline", "module": "l104_reinforcement_engine"}
    try:
        status = rl.status() if hasattr(rl, 'status') else \
                 rl.get_stats() if hasattr(rl, 'get_stats') else \
                 {"type": type(rl).__name__, "capabilities": [m for m in dir(rl) if not m.startswith('_')]}
        return {"status": "online", "module": "l104_reinforcement_engine", "data": status}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@router.post("/rl/step")
async def rl_step(req: RLRequest):
    """Execute an RL step: state → action → reward."""
    rl = _get("rl_engine")
    if not rl:
        raise HTTPException(503, "RL engine not available")
    try:
        if req.reward is not None and hasattr(rl, 'update'):
            result = rl.update(req.state, req.action, req.reward)
        elif hasattr(rl, 'step'):
            result = rl.step(req.state, req.action, req.reward)
        elif hasattr(rl, 'select_action'):
            result = {"action": rl.select_action(req.state)}
        else:
            raise HTTPException(501, f"RL step not implemented on {type(rl).__name__}")
        return {"rl_step": result}
    except Exception as e:
        raise HTTPException(500, f"RL step failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. NEURAL-SYMBOLIC FUSION — Hybrid Reasoning
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/reasoning/status")
async def reasoning_status():
    """Get neural-symbolic fusion engine status."""
    ns = _get("neural_symbolic")
    if not ns:
        return {"status": "offline", "module": "l104_neural_symbolic_fusion"}
    try:
        status = ns.status() if hasattr(ns, 'status') else \
                 ns.get_stats() if hasattr(ns, 'get_stats') else \
                 {"type": type(ns).__name__}
        return {"status": "online", "module": "l104_neural_symbolic_fusion", "data": status}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@router.post("/reasoning/query")
async def reasoning_query(req: ReasoningRequest):
    """Execute neural-symbolic reasoning query."""
    ns = _get("neural_symbolic")
    if not ns:
        raise HTTPException(503, "Neural-symbolic fusion not available")
    try:
        if hasattr(ns, 'reason'):
            result = ns.reason(req.query, depth=req.depth)
        elif hasattr(ns, 'query'):
            result = ns.query(req.query)
        elif hasattr(ns, 'infer'):
            result = ns.infer(req.query)
        else:
            raise HTTPException(501, f"Reasoning not implemented on {type(ns).__name__}")
        return {"reasoning": result, "depth": req.depth}
    except Exception as e:
        raise HTTPException(500, f"Reasoning failed: {e}")

@router.post("/reasoning/explain")
async def reasoning_explain(req: ReasoningRequest):
    """Get explainable reasoning for a query."""
    ns = _get("neural_symbolic")
    if not ns:
        raise HTTPException(503, "Neural-symbolic fusion not available")
    try:
        if hasattr(ns, 'explain'):
            result = ns.explain(req.query)
        elif hasattr(ns, 'generate_explanation'):
            result = ns.generate_explanation(req.query)
        else:
            raise HTTPException(501, f"Explain not implemented on {type(ns).__name__}")
        return {"explanation": result, "query": req.query}
    except Exception as e:
        raise HTTPException(500, f"Explanation failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. QUANTUM LINK BUILDER — Consciousness-Aware Link Analysis
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/quantum-links/status")
async def quantum_links_status():
    """Get quantum link builder status."""
    qlb = _get("quantum_link_builder")
    if not qlb:
        return {"status": "offline", "module": "l104_quantum_link_builder"}
    try:
        status = qlb.status() if hasattr(qlb, 'status') else \
                 qlb.get_status() if hasattr(qlb, 'get_status') else \
                 {"type": type(qlb).__name__, "classes": 51}
        return {"status": "online", "module": "l104_quantum_link_builder", "data": status}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@router.post("/quantum-links/build")
async def quantum_links_build(req: QuantumLinkRequest):
    """Build quantum link between source and target."""
    qlb = _get("quantum_link_builder")
    if not qlb:
        raise HTTPException(503, "Quantum link builder not available")
    try:
        if hasattr(qlb, 'build_link'):
            result = qlb.build_link(req.source, req.target)
        elif hasattr(qlb, 'create_link'):
            result = qlb.create_link(req.source, req.target)
        elif hasattr(qlb, 'link'):
            result = qlb.link(req.source, req.target)
        else:
            raise HTTPException(501, f"Build link not implemented on {type(qlb).__name__}")
        return {"link": result, "source": req.source, "target": req.target}
    except Exception as e:
        raise HTTPException(500, f"Link build failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. QUANTUM NUMERICAL BUILDER — 100-Digit Precision Constants
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/quantum-numerics/status")
async def quantum_numerics_status():
    """Get quantum numerical builder status with high-precision constants."""
    qn = _get("quantum_numerical")
    if not qn:
        return {"status": "offline", "module": "l104_quantum_numerical_builder"}
    try:
        data = {}
        if isinstance(qn, dict):
            for key, val in qn.items():
                data[key] = str(val)[:80] if val else None
        return {
            "status": "online",
            "module": "l104_quantum_numerical_builder",
            "precision": "100-digit",
            "data": data,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@router.get("/quantum-numerics/god-code")
async def quantum_numerics_god_code():
    """Get 100-digit precision GOD_CODE computation."""
    qn = _get("quantum_numerical")
    if not qn or not isinstance(qn, dict):
        raise HTTPException(503, "Quantum numerical builder not available")
    try:
        return {
            "GOD_CODE_HP": str(qn.get("GOD_CODE_HP", "unavailable")),
            "PHI_HP": str(qn.get("PHI_HP", "unavailable")),
            "PHI_GROWTH_HP": str(qn.get("PHI_GROWTH_HP", "unavailable")),
            "standard_GOD_CODE": GOD_CODE,
            "standard_PHI": PHI,
        }
    except Exception as e:
        raise HTTPException(500, f"GOD_CODE retrieval failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED STATUS — All Extended Pipeline Modules (cached 10s)
# ═══════════════════════════════════════════════════════════════════════════════
_status_cache: Dict[str, Any] = {}
_status_cache_ts: float = 0.0
_STATUS_CACHE_TTL = 10.0  # seconds

@router.get("/status")
async def extended_pipeline_status():
    """Unified status for all 9 extended pipeline modules."""
    global _status_cache, _status_cache_ts
    now = time.time()
    if _status_cache and (now - _status_cache_ts) < _STATUS_CACHE_TTL:
        return _status_cache
    modules = [
        ("coding_system", "l104_coding_system", "CodingIntelligenceSystem"),
        ("sentient_archive", "l104_sentient_archive", "SentientArchive"),
        ("language_engine", "l104_language_engine", "LanguageEngine"),
        ("data_pipeline", "l104_data_pipeline", "DataPipelineEngine"),
        ("healing_fabric", "l104_self_healing_fabric", "SelfHealingFabric"),
        ("rl_engine", "l104_reinforcement_engine", "ReinforcementEngine"),
        ("neural_symbolic", "l104_neural_symbolic_fusion", "NeuralSymbolicFusion"),
        ("quantum_link_builder", "l104_quantum_link_builder", "QuantumLinkBuilder"),
        ("quantum_numerical", "l104_quantum_numerical_builder", "QuantumNumericalBuilder"),
    ]

    results = {}
    online = 0
    for name, mod_name, class_name in modules:
        instance = _get(name)
        if instance is not None:
            results[name] = {"status": "online", "module": mod_name, "class": class_name}
            online += 1
        else:
            results[name] = {"status": "offline", "module": mod_name}

    result = {
        "extended_pipeline": "EVO_54_TRANSCENDENT_COGNITION",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "modules_total": len(modules),
        "modules_online": online,
        "coherence": online / len(modules) if modules else 0,
        "god_code": GOD_CODE,
        "phi": PHI,
        "modules": results,
    }
    _status_cache = result
    _status_cache_ts = now
    return result
