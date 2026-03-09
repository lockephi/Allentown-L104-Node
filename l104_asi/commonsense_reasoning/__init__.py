"""
L104 ASI COMMONSENSE REASONING ENGINE v2.1.0
═══════════════════════════════════════════════════════════════════════════════
Decomposed package — was a 9,783-line monolith, now split into 11 modules.

Architecture (8-layer pipeline):
  Layer 1: CONCEPT ONTOLOGY     — ontology.py
  Layer 2: CAUSAL REASONING     — causal.py
  Layer 3: PHYSICAL INTUITION   — physical_intuition.py
  Layer 4: TEMPORAL REASONING   — temporal.py
  Layer 5: ANALOGICAL ENGINE    — analogical.py
  Layer 7: MCQ ELIMINATION      — mcq_solver.py
  Layer 8: CROSS-VERIFICATION   — cross_verification.py
  Bridge:  SCIENCE ENGINE v2.0  — science_bridge.py
  Engine:  UNIFIED ENGINE       — engine.py
  Support: ENGINE LOADERS       — engine_support.py
  Constants: SACRED CONSTANTS   — constants.py
"""

from .constants import PHI, GOD_CODE, VOID_CONSTANT, TAU

from .engine_support import (
    _get_cached_local_intellect,
    _get_cached_science_engine,
    _get_cached_math_engine,
    _get_cached_quantum_gate_engine,
    _get_cached_quantum_math_core,
    _get_cached_dual_layer_engine,
    _get_cached_quantum_probability,
)

from .science_bridge import ScienceEngineBridge, _get_cached_science_bridge
from .ontology import Concept, ConceptOntology
from .causal import CausalRule, CausalReasoningEngine
from .physical_intuition import PhysicalIntuition
from .temporal import TemporalSequence, TemporalReasoningEngine
from .analogical import AnalogicalReasoner
from .cross_verification import CrossVerificationEngine
from .mcq_solver import CommonsenseMCQSolver
from .engine import CommonsenseReasoningEngine

__all__ = [
    # Constants
    "PHI", "GOD_CODE", "VOID_CONSTANT", "TAU",
    # Core classes
    "CommonsenseReasoningEngine",
    "CommonsenseMCQSolver",
    "ScienceEngineBridge",
    "ConceptOntology",
    "Concept",
    "CausalRule",
    "CausalReasoningEngine",
    "PhysicalIntuition",
    "TemporalSequence",
    "TemporalReasoningEngine",
    "AnalogicalReasoner",
    "CrossVerificationEngine",
]
