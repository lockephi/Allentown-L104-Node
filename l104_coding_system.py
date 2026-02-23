#!/usr/bin/env python3
"""
L104 CODING INTELLIGENCE SYSTEM v3.0.0 - Backward-Compatibility Shim

All code has been migrated to the l104_code_engine package.
This module re-exports everything for backward compatibility.

Canonical imports (preferred):
    from l104_code_engine import coding_system
    from l104_code_engine import code_engine

Legacy imports (still work via this shim):
    from l104_coding_system import coding_system

INVARIANT: GOD_CODE = 527.5184818492612 | PHI = 1.618033988749895
"""

import warnings as _warnings

_warnings.warn(
    "l104_coding_system is now a backward-compatibility shim. "
    "Import from l104_code_engine instead: "
    "from l104_code_engine import coding_system",
    DeprecationWarning,
    stacklevel=2,
)

# RE-EXPORT: Sacred Constants
from l104_code_engine.constants import (
    PHI, GOD_CODE, TAU, VOID_CONSTANT, FEIGENBAUM,
    ALPHA_FINE, PLANCK_SCALE, BOLTZMANN_K, QISKIT_AVAILABLE,
    CODING_SYSTEM_NAME as SYSTEM_NAME,
    CODING_SYSTEM_VERSION as VERSION,
    FIBONACCI_7,
    _HARMONIC_BASE, _L104_CONST, _OCTAVE_REF, _GOD_CODE_BASE,
    _god_code_at, _god_code_tuned, _conservation_check,
    _quantum_amplify, _resonance_frequency,
)

# RE-EXPORT: All Classes
from l104_code_engine.analyzer import ProjectAnalyzer
from l104_code_engine.audit import CodeReviewPipeline, QualityGateEngine
from l104_code_engine.ai_context import AIContextBridge
from l104_code_engine.asi_intelligence import SelfReferentialEngine, ASICodeIntelligence
from l104_code_engine.synthesis import CodingSuggestionEngine
from l104_code_engine.session_intelligence import SessionIntelligence
from l104_code_engine.training_kernel import DynamicCodeHarvester, QuantumCodeTrainingKernel
from l104_code_engine.hub import CodingIntelligenceSystem, primal_calculus, resolve_non_dual_logic

# RE-EXPORT: Singletons
from l104_code_engine import coding_system, code_engine

# RE-EXPORT: Lazy ASI Module Getters
from l104_code_engine._lazy_imports import (
    _get_code_engine, _get_neural_cascade, _get_evolution_engine,
    _get_self_optimizer, _get_innovation_engine, _get_consciousness,
    _get_reasoning, _get_knowledge_graph, _get_polymorph,
)

import logging
logger = logging.getLogger("l104_coding_system")
