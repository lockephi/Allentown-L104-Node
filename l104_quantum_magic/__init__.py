"""
l104_quantum_magic v3.0.0 — Fully Decomposed Quantum Magic Package

Decomposed from l104_quantum_magic.py (5,726 lines, 51 classes) into 9 domain modules:

  constants.py            — Sacred constants and standard library imports
  quantum_primitives.py   — Qubit, QuantumGates, QuantumRegister (+ fallbacks)
  hyperdimensional.py     — VectorType, Hypervector, HypervectorFactory, HDCAlgebra,
                            AssociativeMemory, SequenceEncoder
  cognitive.py            — ReasoningStrategy, Observation, Hypothesis, ContextualMemory,
                            QuantumInferenceEngine, AdaptiveLearner, PatternRecognizer,
                            MetaCognition, PredictiveReasoner (EVO_52)
  advanced_reasoning.py   — CausalLink, CausalReasoner, CounterfactualEngine, Goal,
                            GoalPlanner, AttentionMechanism, AbductiveReasoner,
                            CreativeInsight, TemporalReasoner, EmotionalResonance (EVO_53)
  neural_consciousness.py — QuantumNeuralLayer, QuantumNeuralNetwork,
                            ConsciousContent, ConsciousnessSimulator, LogicalTerm,
                            LogicalClause, SymbolicReasoner, WorkingMemory, Episode,
                            EpisodicMemory, IntuitionEngine (EVO_54)
  social_evolution.py     — Agent, SocialIntelligence, DreamState, Individual,
                            EvolutionaryOptimizer, CognitiveControl (EVO_54)
  synthesizer.py          — IntelligentSynthesizer (master intelligence)
  magic.py                — SuperpositionMagic, EntanglementMagic, WaveFunctionMagic,
                            HyperdimensionalMagic, QuantumMagicSynthesizer

Usage:
    from l104_quantum_magic import QuantumMagicSynthesizer
    from l104_quantum_magic import QuantumInferenceEngine
    from l104_quantum_magic import CausalReasoner, CounterfactualEngine
"""

# ── Constants ─────────────────────────────────────────────────────────────────
from .constants import (  # noqa: F401
    PHI, GOD_CODE, PHI_CONJUGATE, PLANCK, HBAR, FE_LATTICE,
    VOID_CONSTANT, ZENITH_HZ, UUC,
)

# ── Quantum Primitives ───────────────────────────────────────────────────────
from .quantum_primitives import (  # noqa: F401
    QUANTUM_AVAILABLE, _QUANTUM_RUNTIME_AVAILABLE,
    Qubit, QuantumGates, QuantumRegister,
)

# ── Hyperdimensional Computing ────────────────────────────────────────────────
from .hyperdimensional import (  # noqa: F401
    HDC_AVAILABLE,
    VectorType, Hypervector, HypervectorFactory, HDCAlgebra,
    AssociativeMemory, SequenceEncoder,
)

# ── Cognitive Intelligence (EVO_52) ──────────────────────────────────────────
from .cognitive import (  # noqa: F401
    ReasoningStrategy, Observation, Hypothesis,
    ContextualMemory, QuantumInferenceEngine,
    AdaptiveLearner, PatternRecognizer,
    MetaCognition, PredictiveReasoner,
)

# ── Advanced Reasoning (EVO_53) ──────────────────────────────────────────────
from .advanced_reasoning import (  # noqa: F401
    CausalLink, CausalReasoner, CounterfactualEngine,
    Goal, GoalPlanner, AttentionMechanism,
    AbductiveReasoner, CreativeInsight,
    TemporalReasoner, EmotionalResonance,
)

# ── Neural & Consciousness (EVO_54) ─────────────────────────────────────────
from .neural_consciousness import (  # noqa: F401
    QuantumNeuralLayer, QuantumNeuralNetwork,
    ConsciousContent, ConsciousnessSimulator,
    LogicalTerm, LogicalClause, SymbolicReasoner,
    WorkingMemory, Episode, EpisodicMemory,
    IntuitionEngine,
)

# ── Social & Evolution (EVO_54) ──────────────────────────────────────────────
from .social_evolution import (  # noqa: F401
    Agent, SocialIntelligence, DreamState,
    Individual, EvolutionaryOptimizer, CognitiveControl,
)

# ── Orchestrators ────────────────────────────────────────────────────────────
from .synthesizer import IntelligentSynthesizer  # noqa: F401
from .magic import (  # noqa: F401
    SuperpositionMagic, EntanglementMagic,
    WaveFunctionMagic, HyperdimensionalMagic,
    QuantumMagicSynthesizer,
)

__version__ = "3.0.0"
__all__ = [
    # Constants
    "PHI", "GOD_CODE", "PHI_CONJUGATE", "PLANCK", "HBAR", "FE_LATTICE",
    "VOID_CONSTANT", "ZENITH_HZ", "UUC",
    "QUANTUM_AVAILABLE", "HDC_AVAILABLE",
    # Quantum Primitives
    "Qubit", "QuantumGates", "QuantumRegister",
    # Hyperdimensional Computing
    "VectorType", "Hypervector", "HypervectorFactory", "HDCAlgebra",
    "AssociativeMemory", "SequenceEncoder",
    # Cognitive Intelligence (EVO_52)
    "ReasoningStrategy", "Observation", "Hypothesis",
    "ContextualMemory", "QuantumInferenceEngine",
    "AdaptiveLearner", "PatternRecognizer",
    "MetaCognition", "PredictiveReasoner",
    # Advanced Reasoning (EVO_53)
    "CausalLink", "CausalReasoner", "CounterfactualEngine",
    "Goal", "GoalPlanner", "AttentionMechanism",
    "AbductiveReasoner", "CreativeInsight",
    "TemporalReasoner", "EmotionalResonance",
    # Neural & Consciousness (EVO_54)
    "QuantumNeuralLayer", "QuantumNeuralNetwork",
    "ConsciousContent", "ConsciousnessSimulator",
    "LogicalTerm", "LogicalClause", "SymbolicReasoner",
    "WorkingMemory", "Episode", "EpisodicMemory",
    "IntuitionEngine",
    # Social & Evolution (EVO_54)
    "Agent", "SocialIntelligence", "DreamState",
    "Individual", "EvolutionaryOptimizer", "CognitiveControl",
    # Orchestrators
    "IntelligentSynthesizer",
    "SuperpositionMagic", "EntanglementMagic",
    "WaveFunctionMagic", "HyperdimensionalMagic",
    "QuantumMagicSynthesizer",
]
