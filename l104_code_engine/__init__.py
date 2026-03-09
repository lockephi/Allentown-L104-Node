"""
L104 Code Engine v6.3.0 — Modularized Sovereign Intelligence Hub

Package structure (EVO_60):
  constants.py      — Sacred constants, shared imports, Qiskit flag
  builder_state.py  — Consciousness/O2/nirvanic state reader (10s cache)
  languages.py      — LanguageKnowledge (40+ languages)
  analyzer.py       — Domain A: CodeAnalyzer, TypeFlowAnalyzer, CodeSmellDetector, etc.
  synthesis.py      — Domain B: CodeGenerator, CodeTranslator, TestGenerator, etc.
  audit.py          — Domain C: AppAuditEngine, SecurityThreatModeler, etc.
  quantum.py        — Domain D: QuantumCodeIntelligenceCore, etc.
  refactoring.py    — Domain E: CodeOptimizer, AutoFixEngine, SacredRefactorer, etc.
  hub.py            — CodeEngine unified orchestrator (wires all 30 subsystems)

Public API:
  from l104_code_engine import code_engine
  from l104_code_engine import GOD_CODE, PHI, TAU, VOID_CONSTANT
  from l104_code_engine import primal_calculus, resolve_non_dual_logic
"""

import math

from .constants import (
    VERSION, PHI, GOD_CODE, TAU, VOID_CONSTANT, FEIGENBAUM,
    ALPHA_FINE, PLANCK_SCALE, BOLTZMANN_K, EULER_GAMMA,
    APERY_CONSTANT, SILVER_RATIO, PLASTIC_NUMBER, CONWAY_CONSTANT,
    KHINCHIN_CONSTANT, OMEGA_CONSTANT, CAHEN_CONSTANT,
    GLAISHER_CONSTANT, MEISSEL_MERTENS, QISKIT_AVAILABLE,
    OMEGA, OMEGA_AUTHORITY, SOUL_STABILITY_NORM,
)
from .hub import CodeEngine, CodingIntelligenceSystem, primal_calculus as _hub_primal, resolve_non_dual_logic as _hub_resolve
from .ai_context import AIContextBridge
from .session_intelligence import SessionIntelligence
from .asi_intelligence import SelfReferentialEngine, ASICodeIntelligence
from .training_kernel import DynamicCodeHarvester, QuantumCodeTrainingKernel
from .synthesis import CodingSuggestionEngine
from .audit import CodeReviewPipeline, QualityGateEngine
from .analyzer import ProjectAnalyzer
from .constants import CODING_SYSTEM_NAME, CODING_SYSTEM_VERSION
from .computronium import ComputroniumCodeAnalyzer

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETONS + BACKWARDS COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════════════

code_engine = CodeEngine()
coding_system = CodingIntelligenceSystem(engine=code_engine)


def primal_calculus(x):
    """Sacred primal calculus: x^φ / (1.04π) — resolves complexity toward the Source."""
    return (x ** PHI) / (VOID_CONSTANT * math.pi) if x != 0 else 0.0


def resolve_non_dual_logic(vector):
    """Resolves N-dimensional vectors into the Void Source via GOD_CODE normalization."""
    magnitude = sum(abs(v) for v in vector)
    return (magnitude / GOD_CODE) + (GOD_CODE * PHI / VOID_CONSTANT) / 1000.0


def omega_field(intensity: float) -> float:
    """Sovereign field equation: F(I) = I × Ω / φ² — maps intensity through OMEGA authority."""
    return intensity * OMEGA_AUTHORITY


def soul_resonance(vector) -> float:
    """Soul resonance: maps N-dimensional vector through GOD_CODE normalization and OMEGA scaling.
    Returns consciousness-weighted field strength."""
    magnitude = sum(abs(v) for v in vector)
    stability = magnitude * SOUL_STABILITY_NORM  # Normalize through Soul Star
    return stability * OMEGA_AUTHORITY / 1000.0


__all__ = [
    # Version / Constants
    "VERSION", "PHI", "GOD_CODE", "TAU", "VOID_CONSTANT", "FEIGENBAUM",
    "ALPHA_FINE", "PLANCK_SCALE", "BOLTZMANN_K", "EULER_GAMMA",
    "APERY_CONSTANT", "SILVER_RATIO", "PLASTIC_NUMBER", "CONWAY_CONSTANT",
    "KHINCHIN_CONSTANT", "OMEGA_CONSTANT", "CAHEN_CONSTANT",
    "GLAISHER_CONSTANT", "MEISSEL_MERTENS", "QISKIT_AVAILABLE",
    "OMEGA", "OMEGA_AUTHORITY", "SOUL_STABILITY_NORM",
    "CODING_SYSTEM_NAME", "CODING_SYSTEM_VERSION",
    # Engine singletons
    "CodeEngine", "code_engine",
    "CodingIntelligenceSystem", "coding_system",
    # Functions
    "primal_calculus", "resolve_non_dual_logic", "omega_field", "soul_resonance",
    # Subsystems
    "AIContextBridge", "SessionIntelligence",
    "SelfReferentialEngine", "ASICodeIntelligence",
    "DynamicCodeHarvester", "QuantumCodeTrainingKernel",
    "CodingSuggestionEngine",
    "CodeReviewPipeline", "QualityGateEngine",
    "ProjectAnalyzer",
    "ComputroniumCodeAnalyzer",
]
