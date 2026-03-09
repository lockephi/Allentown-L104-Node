from __future__ import annotations

import logging

_log = logging.getLogger('l104.language_comprehension')

def _get_science_engine():
    """Lazy-load ScienceEngine for entropy-based confidence calibration."""
    try:
        from l104_science_engine import ScienceEngine
        return ScienceEngine()
    except Exception:
        return None

def _get_math_engine():
    """Lazy-load MathEngine for mathematical domain support."""
    try:
        from l104_math_engine import MathEngine
        return MathEngine()
    except Exception:
        return None

def _get_code_engine():
    """Lazy-load CodeEngine for code-domain knowledge support."""
    try:
        from l104_code_engine import code_engine
        return code_engine
    except Exception:
        return None

def _get_quantum_gate_engine():
    """Lazy-load l104_quantum_gate_engine for quantum circuit-based probability scoring."""
    try:
        from l104_quantum_gate_engine import get_engine
        return get_engine()
    except Exception:
        return None

def _get_quantum_math_core():
    """Lazy-load QuantumMathCore for Grover amplitude amplification + entanglement measures."""
    try:
        from l104_quantum_engine import QuantumMathCore
        return QuantumMathCore
    except Exception:
        return None

def _get_dual_layer_engine():
    """Lazy-load Dual-Layer Engine for physics-grounded confidence calibration."""
    try:
        from l104_asi.dual_layer import dual_layer_engine, DUAL_LAYER_AVAILABLE
        if DUAL_LAYER_AVAILABLE and dual_layer_engine is not None:
            return dual_layer_engine
        return None
    except Exception:
        return None

def _get_formal_logic_engine():
    """Lazy-load FormalLogicEngine for deductive question support."""
    try:
        from l104_asi.formal_logic import FormalLogicEngine
        return FormalLogicEngine()
    except Exception:
        return None

def _get_deep_nlu_engine():
    """Lazy-load DeepNLUEngine for discourse-level comprehension."""
    try:
        from l104_asi.deep_nlu import DeepNLUEngine
        return DeepNLUEngine()
    except Exception:
        return None

_science_engine_cache = None
_math_engine_cache = None
_code_engine_cache = None
_quantum_gate_engine_cache = None
_quantum_math_core_cache = None
_dual_layer_cache = None
_formal_logic_cache = None
_deep_nlu_cache = None
_local_intellect_cache = None
_search_engine_cache = None
_precognition_engine_cache = None
_three_engine_hub_cache = None
_precog_synthesis_cache = None

def _get_cached_local_intellect():
    """Lazy-load local_intellect singleton for KB augmentation.

    Local Intellect has 5000+ BM25-indexed training entries including
    1600+ MMLU academic facts, knowledge manifold, and knowledge vault.
    QUOTA_IMMUNE — runs entirely locally with no API calls.
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

def _get_cached_code_engine():
    global _code_engine_cache
    if _code_engine_cache is None:
        _code_engine_cache = _get_code_engine()
    return _code_engine_cache

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

def _get_cached_dual_layer():
    global _dual_layer_cache
    if _dual_layer_cache is None:
        _dual_layer_cache = _get_dual_layer_engine()
    return _dual_layer_cache

def _get_cached_formal_logic():
    global _formal_logic_cache
    if _formal_logic_cache is None:
        _formal_logic_cache = _get_formal_logic_engine()
    return _formal_logic_cache

def _get_cached_deep_nlu():
    global _deep_nlu_cache
    if _deep_nlu_cache is None:
        _deep_nlu_cache = _get_deep_nlu_engine()
    return _deep_nlu_cache

def _get_cached_search_engine():
    """Lazy-load L104SearchEngine (7 sovereign search algorithms)."""
    global _search_engine_cache
    if _search_engine_cache is None:
        try:
            from l104_search_algorithms import search_engine
            _search_engine_cache = search_engine
        except Exception:
            pass
    return _search_engine_cache

def _get_cached_precognition_engine():
    """Lazy-load L104PrecognitionEngine (7 precognitive algorithms)."""
    global _precognition_engine_cache
    if _precognition_engine_cache is None:
        try:
            from l104_data_precognition import precognition_engine
            _precognition_engine_cache = precognition_engine
        except Exception:
            pass
    return _precognition_engine_cache

def _get_cached_three_engine_hub():
    """Lazy-load ThreeEngineSearchPrecog (5 integrated pipelines)."""
    global _three_engine_hub_cache
    if _three_engine_hub_cache is None:
        try:
            from l104_three_engine_search_precog import three_engine_hub
            _three_engine_hub_cache = three_engine_hub
        except Exception:
            pass
    return _three_engine_hub_cache

def _get_cached_precog_synthesis():
    """Lazy-load PrecogSynthesisIntelligence (HD fusion + manifold + 5D projection)."""
    global _precog_synthesis_cache
    if _precog_synthesis_cache is None:
        try:
            from l104_precog_synthesis import precog_synthesis
            _precog_synthesis_cache = precog_synthesis
        except Exception:
            pass
    return _precog_synthesis_cache

# ── Probability Engine — full ProbabilityEngine hub for hybrid comprehension ──
_probability_engine_cache = None

def _get_cached_probability_engine():
    """Lazy-load ProbabilityEngine singleton for hybrid comprehension.

    v10.0 Hybrid Comprehension Integration:
    Returns the full ProbabilityEngine hub with DataIngestor,
    ASIInsightSynthesis, Bayesian update, Grover amplification,
    GateProbabilityBridge, and GOD_CODE quantum algorithm.
    """
    global _probability_engine_cache
    if _probability_engine_cache is None:
        try:
            from l104_probability_engine import probability_engine
            _probability_engine_cache = probability_engine
        except Exception:
            pass
    return _probability_engine_cache


# ── Quantum Reasoning / Probability — wave collapse for MCQ selection ─────────
_quantum_reasoning_cache = None
_quantum_probability_cache = None

def _get_cached_quantum_reasoning():
    """Lazy-load QuantumReasoningEngine for wave-collapse MCQ selection."""
    global _quantum_reasoning_cache
    if _quantum_reasoning_cache is None:
        try:
            from l104_quantum_reasoning import QuantumReasoningEngine
            _quantum_reasoning_cache = QuantumReasoningEngine()
        except Exception:
            pass
    return _quantum_reasoning_cache

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


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYER 1: TOKENIZER — BPE-style subword tokenization
# ═══════════════════════════════════════════════════════════════════════════════
