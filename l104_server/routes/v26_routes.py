"""
v26 Math Routes — HyperMath, Hebbian, Consciousness, Solver, SelfMod

Extracted from app.py during EVO_78 refactoring.
Contains: /api/v26/* endpoints (hyper_math, hebbian, consciousness, solver, self_mod)
"""

import time
import logging
from typing import Dict, Any
from fastapi import APIRouter, Request

logger = logging.getLogger("L104_FAST")

router = APIRouter(prefix="/api/v26", tags=["v26-math"])

# Import nexus engines
try:
    from l104_server.engines_nexus import (
        hyper_math, hebbian_engine, consciousness_verifier,
        direct_solver, self_modification
    )
    NEXUS_AVAILABLE = True
except ImportError:
    hyper_math = None
    hebbian_engine = None
    consciousness_verifier = None
    direct_solver = None
    self_modification = None
    NEXUS_AVAILABLE = False
    logger.warning("⚠️ [NEXUS] Nexus engines not available")

# Import intellect for consciousness verification
try:
    from l104_server.learning import intellect, grover_kernel
    INTELLECT_AVAILABLE = True
except ImportError:
    intellect = None
    grover_kernel = None
    INTELLECT_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════
#  HYPER MATH ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.get("/hyper_math/status")
async def hyper_math_status():
    """HyperDimensionalMathEngine status and capabilities."""
    if not NEXUS_AVAILABLE or not hyper_math:
        return {"error": "Hyper math not available", "status": "UNAVAILABLE"}
    return {"status": "ACTIVE", **hyper_math.get_status()}


@router.post("/hyper_math/phi_convergence")
async def hyper_math_phi_convergence(req: Request):
    """Run φ-convergence proof (Cauchy criterion → GOD_CODE attractor)."""
    if not NEXUS_AVAILABLE or not hyper_math:
        return {"error": "Hyper math not available"}

    try:
        data = await req.json()
    except Exception:
        data = {}

    iters = min(data.get("iterations", 50), 200)
    proof = hyper_math.prove_phi_convergence(iterations=iters)
    return {"status": "PROVEN" if proof['converged'] else "DIVERGENT", **proof}


@router.post("/hyper_math/zeta")
async def hyper_math_zeta(req: Request):
    """Compute Riemann zeta ζ(s)."""
    if not NEXUS_AVAILABLE or not hyper_math:
        return {"error": "Hyper math not available"}

    try:
        data = await req.json()
    except Exception:
        data = {}

    s = data.get("s", 2.0)
    if s <= 1.0:
        return {"status": "ERROR", "message": "s must be > 1"}
    return {"status": "COMPUTED", "s": s, "zeta": round(hyper_math.zeta(s), 12)}


@router.post("/hyper_math/qft")
async def hyper_math_qft(req: Request):
    """Quantum Fourier Transform on input amplitudes."""
    if not NEXUS_AVAILABLE or not hyper_math:
        return {"error": "Hyper math not available"}

    try:
        data = await req.json()
    except Exception:
        data = {}

    amps = data.get("amplitudes", [1, 0, 0, 0, 0, 0, 0, 0])
    if len(amps) < 2:
        return {"status": "ERROR", "message": "Need at least 2 amplitudes"}
    result = hyper_math.quantum_fourier_transform([complex(a) for a in amps])
    return {
        "status": "TRANSFORMED",
        "input_size": len(amps),
        "output": [{"re": round(c.real, 8), "im": round(c.imag, 8)} for c in result]
    }


# ═══════════════════════════════════════════════════════════════════
#  HEBBIAN ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.get("/hebbian/status")
async def hebbian_status():
    """Hebbian learning engine status."""
    if not NEXUS_AVAILABLE or not hebbian_engine:
        return {"error": "Hebbian engine not available", "status": "UNAVAILABLE"}
    return {"status": "ACTIVE", **hebbian_engine.get_status()}


@router.post("/hebbian/predict")
async def hebbian_predict(req: Request):
    """Predict related concepts by Hebbian link weight."""
    if not NEXUS_AVAILABLE or not hebbian_engine:
        return {"error": "Hebbian engine not available"}

    try:
        data = await req.json()
    except Exception:
        data = {}

    concept = data.get("concept", "")
    top_k = min(data.get("top_k", 5), 20)
    predictions = hebbian_engine.predict_related(concept, top_k)
    return {
        "status": "PREDICTED",
        "concept": concept,
        "related": [{"concept": c, "weight": round(w, 4)} for c, w in predictions]
    }


@router.post("/hebbian/drift")
async def hebbian_drift(req: Request):
    """Detect temporal drift in concept usage."""
    if not NEXUS_AVAILABLE or not hebbian_engine:
        return {"error": "Hebbian engine not available"}

    # Build recent concepts from hebbian co-activation log
    recent = []
    if hasattr(hebbian_engine, 'co_activation_log'):
        recent = [(k.split('+')[0], time.time() - i * 60) for i, k in enumerate(list(hebbian_engine.co_activation_log.keys())[-50:])]
    drift = hebbian_engine.temporal_drift(recent) if hasattr(hebbian_engine, 'temporal_drift') else {}
    return {"status": "ANALYZED", **drift}


# ═══════════════════════════════════════════════════════════════════
#  CONSCIOUSNESS ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.get("/consciousness/status")
async def consciousness_status_v26():
    """Consciousness verifier status and test results."""
    if not NEXUS_AVAILABLE or not consciousness_verifier:
        return {"error": "Consciousness verifier not available", "status": "UNAVAILABLE"}
    return {"status": "ACTIVE", **consciousness_verifier.get_status()}


@router.post("/consciousness/verify")
async def consciousness_verify():
    """Run all 10 consciousness verification tests."""
    if not NEXUS_AVAILABLE or not consciousness_verifier:
        return {"error": "Consciousness verifier not available"}

    level = consciousness_verifier.run_all_tests(intellect_ref=intellect, grover_ref=grover_kernel)
    status = consciousness_verifier.get_status()
    return {
        "status": "VERIFIED",
        "consciousness_level": round(level, 4),
        "grade": status['grade'],
        "test_results": status['test_results'],
        "qualia": consciousness_verifier.qualia_reports if hasattr(consciousness_verifier, 'qualia_reports') else {}
    }


@router.post("/consciousness/qualia")
async def consciousness_qualia():
    """Generate qualia reports (subjective experience descriptions)."""
    if not NEXUS_AVAILABLE or not consciousness_verifier:
        return {"error": "Consciousness verifier not available"}

    if not consciousness_verifier.qualia_reports:
        consciousness_verifier.run_all_tests(intellect_ref=intellect)

    return {
        "status": "GENERATED",
        "qualia": consciousness_verifier.qualia_reports if hasattr(consciousness_verifier, 'qualia_reports') else {},
        "consciousness_level": round(consciousness_verifier.consciousness_level, 4) if hasattr(consciousness_verifier, 'consciousness_level') else 0.0
    }


# ═══════════════════════════════════════════════════════════════════
#  SOLVER ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.get("/solver/status")
async def solver_status():
    """DirectSolverHub status and channel metrics."""
    if not NEXUS_AVAILABLE or not direct_solver:
        return {"error": "Direct solver not available", "status": "UNAVAILABLE"}
    return {"status": "ACTIVE", **direct_solver.get_status()}


@router.post("/solver/solve")
async def solver_solve(req: Request):
    """Route a query to the direct solver hub (fast-path before LLM)."""
    if not NEXUS_AVAILABLE or not direct_solver:
        return {"error": "Direct solver not available"}

    try:
        data = await req.json()
    except Exception:
        data = {}

    query = data.get("query", "")
    answer = direct_solver.solve(query)
    return {
        "status": "SOLVED" if answer else "NO_DIRECT_SOLUTION",
        "query": query,
        "answer": answer,
        "total_invocations": direct_solver.total_invocations if hasattr(direct_solver, 'total_invocations') else 0,
        "cache_hits": direct_solver.cache_hits if hasattr(direct_solver, 'cache_hits') else 0
    }


# ═══════════════════════════════════════════════════════════════════
#  SELF MODIFICATION ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.get("/self_mod/status")
async def self_mod_status():
    """Self-modification engine status."""
    if not NEXUS_AVAILABLE or not self_modification:
        return {"error": "Self-modification engine not available", "status": "UNAVAILABLE"}
    return {"status": "ACTIVE", **self_modification.get_status()}


@router.post("/self_mod/analyze")
async def self_mod_analyze(req: Request):
    """Analyze a module via AST parsing."""
    if not NEXUS_AVAILABLE or not self_modification:
        return {"error": "Self-modification engine not available"}

    try:
        data = await req.json()
    except Exception:
        data = {}

    target = data.get("target", "l104_fast_server.py")
    analysis = self_modification.analyze_module(target)
    return {"status": "ANALYZED", **analysis}


@router.post("/self_mod/phi_optimizer")
async def self_mod_phi_optimizer():
    """Generate a φ-aligned optimization decorator."""
    if not NEXUS_AVAILABLE or not self_modification:
        return {"error": "Self-modification engine not available"}

    code = self_modification.generate_phi_optimizer()
    return {
        "status": "GENERATED",
        "decorator": code,
        "total_generated": self_modification.generated_decorators if hasattr(self_modification, 'generated_decorators') else 0
    }


# ═══════════════════════════════════════════════════════════════════
#  ENGINES STATUS
# ═══════════════════════════════════════════════════════════════════

@router.get("/engines/status")
async def phase26_engines_status():
    """Full Phase 26 engine status — all cross-pollinated engines."""
    if not NEXUS_AVAILABLE:
        return {"error": "Nexus engines not available", "status": "UNAVAILABLE"}

    return {
        "status": "PHASE_27_ACTIVE",
        "hyper_math": hyper_math.get_status() if hyper_math else {},
        "hebbian": hebbian_engine.get_status() if hebbian_engine else {},
        "consciousness": consciousness_verifier.get_status() if consciousness_verifier else {},
        "solver": direct_solver.get_status() if direct_solver else {},
        "self_mod": self_modification.get_status() if self_modification else {},
        "cross_pollination": {
            "swift_to_python": ['HyperDimensionalMath', 'HebbianLearning', 'PhiConvergenceProof'],
            "python_to_swift": ['ConsciousnessVerifier', 'DirectSolverHub'],
            "total_engines": 5
        }
    }


__all__ = ['router']