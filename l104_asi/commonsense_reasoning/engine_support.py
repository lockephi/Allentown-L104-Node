"""Lazy-loaded engine support for Commonsense Reasoning Engine.

Provides cached singletons for cross-engine integration:
Science, Math, Quantum Gate, Quantum Math Core, Dual-Layer, Local Intellect,
and Quantum Probability.
"""

from __future__ import annotations


# ── Engine Support (lazy-loaded for physics intuition + reasoning depth) ────
def _get_science_engine():
    """Lazy-load ScienceEngine for physics-based reasoning support."""
    try:
        from l104_science_engine import ScienceEngine
        return ScienceEngine()
    except Exception:
        return None

def _get_math_engine():
    """Lazy-load MathEngine for quantitative reasoning support."""
    try:
        from l104_math_engine import MathEngine
        return MathEngine()
    except Exception:
        return None

def _get_quantum_gate_engine():
    """Lazy-load l104_quantum_gate_engine for quantum circuit probability computation."""
    try:
        from l104_quantum_gate_engine import get_engine
        return get_engine()
    except Exception:
        return None

def _get_quantum_math_core():
    """Lazy-load QuantumMathCore for Grover amplitude amplification + quantum scoring."""
    try:
        from l104_quantum_engine import QuantumMathCore
        return QuantumMathCore
    except Exception:
        return None

def _get_dual_layer_engine():
    """Lazy-load Dual-Layer Engine for Thought + Physics layer integration."""
    try:
        from l104_asi import dual_layer_engine
        return dual_layer_engine
    except Exception:
        return None

_science_engine_cache = None
_math_engine_cache = None
_quantum_gate_engine_cache = None
_quantum_math_core_cache = None
_dual_layer_engine_cache = None
_local_intellect_cache = None

def _get_cached_local_intellect():
    """Lazy-load local_intellect singleton for KB augmentation.

    Local Intellect has 5000+ BM25-indexed training entries, knowledge
    manifold, and knowledge vault. QUOTA_IMMUNE — runs entirely locally.
    """
    global _local_intellect_cache
    if _local_intellect_cache is None:
        try:
            from l104_intellect import local_intellect
            _local_intellect_cache = local_intellect
        except Exception:
            pass
    return _local_intellect_cache

def _get_cached_science_engine():
    global _science_engine_cache
    if _science_engine_cache is None:
        _science_engine_cache = _get_science_engine()
    return _science_engine_cache

def _get_cached_math_engine():
    global _math_engine_cache
    if _math_engine_cache is None:
        _math_engine_cache = _get_math_engine()
    return _math_engine_cache

def _get_cached_quantum_gate_engine():
    global _quantum_gate_engine_cache
    if _quantum_gate_engine_cache is None:
        _quantum_gate_engine_cache = _get_quantum_gate_engine()
    return _quantum_gate_engine_cache

def _get_cached_quantum_math_core():
    global _quantum_math_core_cache
    if _quantum_math_core_cache is None:
        _quantum_math_core_cache = _get_quantum_math_core()
    return _quantum_math_core_cache

def _get_cached_dual_layer_engine():
    global _dual_layer_engine_cache
    if _dual_layer_engine_cache is None:
        _dual_layer_engine_cache = _get_dual_layer_engine()
    return _dual_layer_engine_cache

# ── Quantum Probability — wave collapse for MCQ selection ─────────────────────
_quantum_probability_cache = None

def _get_cached_quantum_probability():
    """Lazy-load QuantumProbability for Born-rule measurement collapse."""
    global _quantum_probability_cache
    if _quantum_probability_cache is None:
        try:
            from l104_probability_engine import QuantumProbability
            _quantum_probability_cache = QuantumProbability
        except Exception:
            pass
    return _quantum_probability_cache
