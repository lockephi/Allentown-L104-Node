"""
l104_asi — Decomposed from l104_asi_core.py (5,845 lines → package)
Phase 3A of L104 Decompression Plan.

FLAGSHIP: Dual-Layer Engine — The Duality of Nature
  Layer 1 (THOUGHT): Pattern recognition, symmetry, sacred geometry — WHY
  Layer 2 (PHYSICS): Precision computation, 63 constants at ±0.005% — HOW MUCH
  COLLAPSE: Thought asks → Physics answers → Duality collapses to value

Re-exports ALL public symbols so that:
    from l104_asi import X
works identically to the original:
    from l104_asi_core import X
"""
# Constants and flags
from .constants import (
    ASI_CORE_VERSION, ASI_PIPELINE_EVO,
    PHI, GOD_CODE, TAU, PHI_CONJUGATE, VOID_CONSTANT, FEIGENBAUM,
    OMEGA_AUTHORITY, PLANCK_CONSCIOUSNESS, ALPHA_FINE,
    TORCH_AVAILABLE, TENSORFLOW_AVAILABLE, PANDAS_AVAILABLE,
    QISKIT_AVAILABLE,
    # v7.1 Dual-Layer Flagship Constants
    DUAL_LAYER_VERSION, GOD_CODE_V3,
    DUAL_LAYER_PRECISION_TARGET, DUAL_LAYER_CONSTANTS_COUNT,
    DUAL_LAYER_INTEGRITY_CHECKS, DUAL_LAYER_GRID_REFINEMENT,
    PRIME_SCAFFOLD, QUANTIZATION_GRAIN,
)

# ★ FLAGSHIP: Dual-Layer Engine — The Duality of Nature ★
from .dual_layer import (
    DualLayerEngine, dual_layer_engine, DUAL_LAYER_AVAILABLE,
    NATURES_DUALITIES, CONSCIOUSNESS_TO_PHYSICS_BRIDGE,
)

# Domain
from .domain import DomainKnowledge, GeneralDomainExpander, Theorem

# Theorem generation
from .theorem_gen import NovelTheoremGenerator

# Self-modification
from .self_mod import SelfModificationEngine

# Consciousness verification
from .consciousness import ConsciousnessVerifier

# Pipeline
from .pipeline import (SolutionChannel, DirectSolutionHub, PipelineTelemetry,
                       SoftmaxGatingRouter, AdaptivePipelineRouter)

# Reasoning
from .reasoning import (TreeOfThoughts, MultiHopReasoningChain,
                        SolutionEnsembleEngine, PipelineHealthDashboard,
                        PipelineReplayBuffer)

# Quantum
from .quantum import QuantumComputationCore

# Core + singleton
from .core import ASICore, asi_core, main, get_current_parameters, update_parameters

# KerasASIModel is defined conditionally inside ASICore — re-export if available
try:
    from .core import KerasASIModel
except ImportError:
    pass
