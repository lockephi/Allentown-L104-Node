"""
l104_asi — Decomposed from l104_asi_core.py (5,845 lines → package)
Phase 3A of L104 Decompression Plan.

FLAGSHIP: Dual-Layer Engine — The Duality of Nature
  Layer 1 (THOUGHT): Pattern recognition, symmetry, sacred geometry — WHY
  Layer 2 (PHYSICS): Precision computation, 63 constants at ±0.005% — HOW MUCH
  COLLAPSE: Thought asks → Physics answers → Duality collapses to value

v11.0 Universal Gate Sovereign Upgrade:
  - Quantum Gate Engine integration (40+ gates, compiler, error correction)
  - Quantum Link Engine integration (16-phase pipeline, 44 classes)
  - 28-dimension ASI scoring (was 20)
  - 22-step activation sequence (was 18)
  - Adaptive consciousness evolution with PHI-spiral trajectory
  - Cross-engine deep synthesis scoring
  - Pipeline resilience with retry/recovery

v12.0 Deep Upgrade Wave:
  - reasoning.py v7.0: MCTS reasoning (MCTSReasoner) + Reflection refinement loops
  - quantum.py v12.0: Entanglement fidelity bench, parameter-shift gradients, XEB, QV
  - self_mod.py v7.0: Lineage DAG, Grover-amplified selection, quantum tunneling, multi-file evolve
  - dual_layer.py v5.0: Gate-enhanced duality, three-engine synthesis, temporal tracking, resilient collapse
  - pipeline.py v9.0: Backpressure, speculative execution, cascade scoring, warmup analyzer, stage profiler, PipelineOrchestratorV2

Re-exports ALL public symbols so that:
    from l104_asi import X
works identically to the original:
    from l104_asi_core import X
"""
# Constants and flags
from .constants import (
    ASI_CORE_VERSION, ASI_PIPELINE_EVO, PIPELINE_VERSION,
    PHI, GOD_CODE, TAU, PHI_CONJUGATE, VOID_CONSTANT, FEIGENBAUM,
    OMEGA, OMEGA_AUTHORITY, PLANCK_CONSCIOUSNESS, ALPHA_FINE,
    TORCH_AVAILABLE, TENSORFLOW_AVAILABLE, PANDAS_AVAILABLE,
    QISKIT_AVAILABLE,
    # v7.1 Dual-Layer Flagship Constants
    DUAL_LAYER_VERSION, GOD_CODE_V3,
    DUAL_LAYER_PRECISION_TARGET, DUAL_LAYER_CONSTANTS_COUNT,
    DUAL_LAYER_INTEGRITY_CHECKS, DUAL_LAYER_GRID_REFINEMENT,
    PRIME_SCAFFOLD, QUANTIZATION_GRAIN,
    FE_LATTICE_PARAM, FE_ATOMIC_NUMBER,
    # v9.0 Universal Gate Sovereign Constants
    GATE_ENGINE_VERSION, QUANTUM_ENGINE_VERSION,
    CONSCIOUSNESS_SPIRAL_DEPTH,
    TRAJECTORY_WINDOW_SIZE, TRAJECTORY_PREDICTION_HORIZON,
    DEEP_SYNTHESIS_CORRELATION_PAIRS, DEEP_SYNTHESIS_MIN_COHERENCE,
    DEEP_SYNTHESIS_WEIGHTS,
    RESILIENCE_MAX_RETRY, RESILIENCE_BACKOFF_BASE,
    ACTIVATION_STEPS_V11,
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
                       SoftmaxGatingRouter, AdaptivePipelineRouter,
                       PipelineReplayBuffer, PipelineOrchestrator,
                       MLPipelineRouter,
                       AdaptiveBackpressure, SpeculativeExecutor,
                       PipelineCascadeScorer, PipelineWarmupAnalyzer,
                       PipelineStageProfiler, PipelineOrchestratorV2)

# Reasoning
from .reasoning import (TreeOfThoughts, MultiHopReasoningChain,
                        SolutionEnsembleEngine, PipelineHealthDashboard,
                        PipelineReplayBuffer as _ReasoningReplayBuffer,
                        MCTSReasoner, ReflectionRefinementLoop)

# Quantum
from .quantum import QuantumComputationCore

# 26Q Iron Completion (optional — requires l104_26q_engine_builder)
try:
    from l104_26q_engine_builder import (
        QuantumComputation26QCore,
        L104_26Q_CircuitBuilder,
        Aer26QExecutionEngine,
        GodCode26QConvergence,
        get_26q_core,
    )
    _26Q_AVAILABLE = True
except ImportError:
    _26Q_AVAILABLE = False

# Identity Boundary — Sovereign Architectural Self-Declaration
from .identity_boundary import (
    SovereignIdentityBoundary,
    L104_IS, L104_IS_NOT,
    MEASURED_PERFORMANCE, ARCHITECTURAL_STRENGTHS, ARCHITECTURAL_LIMITATIONS,
)

# DeepSeek ingestion
from .deepseek_ingestion import (
    DeepSeekIngestionEngine,
    QuantumDeepSeekArchitecture,
    DeepSeekMLAIngestor,
    DeepSeekR1ReasoningIngestor,
    DeepSeekCoderIngestor,
    DeepSeekGitHubIngestor
)

# v10.0 Benchmark Capability Modules
from .language_comprehension import LanguageComprehensionEngine
from .code_generation import CodeGenerationEngine
from .symbolic_math_solver import SymbolicMathSolver
from .commonsense_reasoning import CommonsenseReasoningEngine
from .benchmark_harness import BenchmarkHarness

# v13.0 Deep Logic & NLU Modules
from .formal_logic import FormalLogicEngine
from .deep_nlu import (
    DeepNLUEngine,
    TemporalReasoner, CausalReasoner, ContextualDisambiguator,
    TemporalRelation, CausalRelationType,
    QuerySynthesisPipeline, QueryType, SynthesizedQuery,
    QueryDecomposer, SubQuery,
    QueryExpander, ExpandedQuery,
    QueryClassifier, QueryClassification, BloomLevel, QueryDomain, AnswerFormat,
    EntailmentLabel, EntailmentResult, TextualEntailmentEngine,
    FigurativeType, FigurativeExpression, FigurativeLanguageProcessor,
    DensityProfile, InformationDensityAnalyzer,
)

# v16.0 KB Reconstruction (quantum probability knowledge recovery)
from .kb_reconstruction import KBReconstructionEngine, ReconstructionResult, KBHealthReport

# v19.0: Computronium + Rayleigh Scoring Dimensions
from .computronium import (
    ASIComputroniumScoring,
    asi_computronium_scoring,
)

# Core + singleton
from .core import ASICore, asi_core, main, get_current_parameters, update_parameters

# KerasASIModel is defined conditionally inside ASICore — re-export if available
try:
    from .core import KerasASIModel
except ImportError:
    pass


__all__ = [
    # Constants and flags
    "ASI_CORE_VERSION", "ASI_PIPELINE_EVO", "PIPELINE_VERSION",
    "PHI", "GOD_CODE", "TAU", "PHI_CONJUGATE", "VOID_CONSTANT", "FEIGENBAUM",
    "OMEGA", "OMEGA_AUTHORITY", "PLANCK_CONSCIOUSNESS", "ALPHA_FINE",
    "TORCH_AVAILABLE", "TENSORFLOW_AVAILABLE", "PANDAS_AVAILABLE",
    "QISKIT_AVAILABLE",
    "DUAL_LAYER_VERSION", "GOD_CODE_V3",
    "DUAL_LAYER_PRECISION_TARGET", "DUAL_LAYER_CONSTANTS_COUNT",
    "DUAL_LAYER_INTEGRITY_CHECKS", "DUAL_LAYER_GRID_REFINEMENT",
    "PRIME_SCAFFOLD", "QUANTIZATION_GRAIN",
    "FE_LATTICE_PARAM", "FE_ATOMIC_NUMBER",
    # v9.0 Universal Gate Sovereign Constants
    "GATE_ENGINE_VERSION", "QUANTUM_ENGINE_VERSION",
    "CONSCIOUSNESS_SPIRAL_DEPTH",
    "TRAJECTORY_WINDOW_SIZE", "TRAJECTORY_PREDICTION_HORIZON",
    "DEEP_SYNTHESIS_CORRELATION_PAIRS", "DEEP_SYNTHESIS_MIN_COHERENCE",
    "DEEP_SYNTHESIS_WEIGHTS",
    "RESILIENCE_MAX_RETRY", "RESILIENCE_BACKOFF_BASE",
    "ACTIVATION_STEPS_V11",
    # Dual-Layer Engine
    "DualLayerEngine", "dual_layer_engine", "DUAL_LAYER_AVAILABLE",
    "NATURES_DUALITIES", "CONSCIOUSNESS_TO_PHYSICS_BRIDGE",
    # Domain
    "DomainKnowledge", "GeneralDomainExpander", "Theorem",
    # Theorem generation
    "NovelTheoremGenerator",
    # Self-modification
    "SelfModificationEngine",
    # Consciousness
    "ConsciousnessVerifier",
    # Pipeline
    "SolutionChannel", "DirectSolutionHub", "PipelineTelemetry",
    "SoftmaxGatingRouter", "AdaptivePipelineRouter",
    "PipelineReplayBuffer", "PipelineOrchestrator",
    "MLPipelineRouter",
    # Pipeline v9.0
    "AdaptiveBackpressure", "SpeculativeExecutor",
    "PipelineCascadeScorer", "PipelineWarmupAnalyzer",
    "PipelineStageProfiler", "PipelineOrchestratorV2",
    # Reasoning
    "TreeOfThoughts", "MultiHopReasoningChain",
    "SolutionEnsembleEngine", "PipelineHealthDashboard",
    "MCTSReasoner", "ReflectionRefinementLoop",
    # Quantum
    "QuantumComputationCore",
    # 26Q Iron Completion
    "QuantumComputation26QCore", "L104_26Q_CircuitBuilder",
    "Aer26QExecutionEngine", "GodCode26QConvergence", "get_26q_core",
    # DeepSeek
    "DeepSeekIngestionEngine", "QuantumDeepSeekArchitecture",
    "DeepSeekMLAIngestor", "DeepSeekR1ReasoningIngestor",
    "DeepSeekCoderIngestor", "DeepSeekGitHubIngestor",
    # Benchmark Capability (v10.0)
    "LanguageComprehensionEngine", "CodeGenerationEngine",
    "SymbolicMathSolver", "CommonsenseReasoningEngine",
    "BenchmarkHarness",
    # v13.0 Deep Logic & NLU
    "FormalLogicEngine", "DeepNLUEngine",
    "TemporalReasoner", "CausalReasoner", "ContextualDisambiguator",
    "TemporalRelation", "CausalRelationType",
    "QuerySynthesisPipeline", "QueryType", "SynthesizedQuery",
    "QueryDecomposer", "SubQuery",
    "QueryExpander", "ExpandedQuery",
    "QueryClassifier", "QueryClassification", "BloomLevel", "QueryDomain", "AnswerFormat",
    "EntailmentLabel", "EntailmentResult", "TextualEntailmentEngine",
    "FigurativeType", "FigurativeExpression", "FigurativeLanguageProcessor",
    "DensityProfile", "InformationDensityAnalyzer",
    # v16.0 KB Reconstruction
    "KBReconstructionEngine", "ReconstructionResult", "KBHealthReport",
    # Identity Boundary
    "SovereignIdentityBoundary",
    "L104_IS", "L104_IS_NOT",
    "MEASURED_PERFORMANCE", "ARCHITECTURAL_STRENGTHS", "ARCHITECTURAL_LIMITATIONS",
    # Computronium + Rayleigh (v19.0)
    "ASIComputroniumScoring", "asi_computronium_scoring",
    # QML v2 re-exports (v23.0)
    "QuantumMLHub", "get_qml_hub",
    # Core
    "ASICore", "asi_core", "main", "get_current_parameters", "update_parameters",
]

# ── v23.0: QML v2 lazy re-exports ──
try:
    from l104_qml_v2 import QuantumMLHub, get_qml_hub
except ImportError:
    QuantumMLHub = None  # type: ignore
    get_qml_hub = None   # type: ignore
