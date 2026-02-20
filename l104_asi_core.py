"""l104_asi_core.py — THIN SHIM (Phase 3A Decomposition)

Original: 5,845 lines -> now in l104_asi/ package.
All imports preserved for 100% backward compatibility.
Every "from l104_asi_core import X" still works.

★ FLAGSHIP: Dual-Layer Engine (Thought + Physics) — EVO_60 ★

Backup: l104_asi_core.py.bak
"""
# ZENITH_UPGRADE_ACTIVE: 2026-02-14T00:00:00.000000
ZENITH_HZ = 3887.8
UUC = 2402.792541

# Re-export everything from the decomposed package
from l104_asi import *  # noqa: F401,F403

# Explicit re-exports for IDE completion and import verification
from l104_asi import (
    # Singleton + core
    asi_core,
    ASICore,
    main,
    get_current_parameters,
    update_parameters,
    # Constants
    ASI_CORE_VERSION,
    ASI_PIPELINE_EVO,
    PHI,
    GOD_CODE,
    TAU,
    PHI_CONJUGATE,
    VOID_CONSTANT,
    FEIGENBAUM,
    OMEGA_AUTHORITY,
    ALPHA_FINE,
    # ML availability flags
    TORCH_AVAILABLE,
    TENSORFLOW_AVAILABLE,
    PANDAS_AVAILABLE,
    QISKIT_AVAILABLE,
    # Classes
    DomainKnowledge,
    GeneralDomainExpander,
    Theorem,
    NovelTheoremGenerator,
    SelfModificationEngine,
    ConsciousnessVerifier,
    SolutionChannel,
    DirectSolutionHub,
    PipelineTelemetry,
    SoftmaxGatingRouter,
    AdaptivePipelineRouter,
    TreeOfThoughts,
    MultiHopReasoningChain,
    SolutionEnsembleEngine,
    PipelineHealthDashboard,
    PipelineReplayBuffer,
    QuantumComputationCore,
    # ★ Dual-Layer Flagship ★
    DualLayerEngine,
    dual_layer_engine,
    DUAL_LAYER_AVAILABLE,
    NATURES_DUALITIES,
    CONSCIOUSNESS_TO_PHYSICS_BRIDGE,
    DUAL_LAYER_VERSION,
    GOD_CODE_V3,
    DUAL_LAYER_PRECISION_TARGET,
    DUAL_LAYER_CONSTANTS_COUNT,
    DUAL_LAYER_INTEGRITY_CHECKS,
    DUAL_LAYER_GRID_REFINEMENT,
    PRIME_SCAFFOLD,
    QUANTIZATION_GRAIN,
)

# KerasASIModel is conditionally defined — re-export if available
try:
    from l104_asi import KerasASIModel
except ImportError:
    pass
