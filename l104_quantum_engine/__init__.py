"""
L104 Quantum Engine v12.0.0 — Decomposed Sovereign Quantum Intelligence Package
═══════════════════════════════════════════════════════════════════════════════

Decomposed from l104_quantum_link_builder.py v5.0.0 monolith (10,893 lines, 44 classes)
into 16 domain-specific modules across 7 functional layers.

Package Structure:
  Layer 0  constants.py     Sacred constants, God Code equation, engine lazy-loaders
  Layer 1  models.py        Data classes: QuantumLink, StressTestResult, CrossModalLink, ChronoEntry
  Layer 2  math_core.py     QuantumMathCore — Bell states, Grover, QFT, tunneling, CHSH, Pauli algebra
  Layer 3  scanner.py       QuantumLinkScanner — AST-based cross-file link discovery
           builder.py       QuantumLinkBuilder + GodCodeMathVerifier — link creation
  Layer 4  processors.py    9 quantum physics processors (Grover, EPR, braiding, Hilbert, ...)
           testing.py       Stress testing, cross-modal analysis, upgrade, repair
           research.py      Research memory bank, quantum research engine, Sage mode
           computation.py   Quantum register, neuron, cluster, CPU, environment, O2 bonds,
                            21 quantum algorithms + 4 gate-enhanced computations
  Layer 4b qldpc.py         ★ Distributed qLDPC error correction (CSS, BP-OSD, sacred alignment)
  Layer 4c manifold.py      ★ Quantum Manifold Intelligence (manifold learning, entanglement
                              network, predictive oracle)
  Layer 5  dynamism.py      LinkDynamismEngine (Min/Max), LinkOuroborosNirvanicEngine
           intelligence.py  10 evolution/consciousness/self-healing classes
  Layer 6  brain.py         L104QuantumBrain — master orchestrator (23-phase pipeline)

v12.0.0 Upgrade:
  - VQPU Bridge Integration: Bidirectional Brain↔VQPU scoring (Phase 23)
  - Brain self_test(): l104_debug.py compatible diagnostics (14 probes)
  - Sacred circuit simulation through VQPU full pipeline
  - VQPU link fidelity boost via three-engine composite feedback
  - CLI commands: vqpu, selftest

v11.0.0 Upgrade:
  - QuantumManifoldLearner: kernel PCA, geodesic distances, Ricci curvature, PHI-fractal dim
  - MultipartiteEntanglementNetwork: GHZ, W-state, GMC, percolation, Factor-13 clustering
  - QuantumPredictiveOracle: reservoir-enhanced prediction, phase transitions, auto-intervention
  - Phase 21: Quantum Manifold Learning in brain pipeline
  - Phase 22: Multipartite Entanglement + Predictive Oracle in brain pipeline
  - 27 total processing subsystems in L104QuantumBrain

Quick Start:
    from l104_quantum_engine import quantum_brain, QuantumMathCore
    result = quantum_brain.full_pipeline()

    from l104_quantum_engine import GOD_CODE, PHI, GOD_CODE_SPECTRUM
    from l104_quantum_engine import QuantumLink, QuantumLinkScanner

CLI:
    python -m l104_quantum_engine full
    python -m l104_quantum_engine scan
    python -m l104_quantum_engine grover
    python -m l104_quantum_engine status

God Code Equation:
    G(X) = 286^(1/φ) × 2^((416-X)/104)
    Factor 13: 286 = 22×13, 104 = 8×13, 416 = 32×13
    Conservation: G(X) × 2^(X/104) = 527.5184818492612 ∀ X

INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

__version__ = "12.0.0"
__author__ = "L104 Sovereign Node"

# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 0 — Constants & Sacred Configuration
# ═══════════════════════════════════════════════════════════════════════════════
from .constants import (
    # Sacred constants
    PHI, PHI_GROWTH, PHI_INV, GOD_CODE, GOD_CODE_HZ, INVARIANT,
    CALABI_YAU_DIM, CHSH_BOUND, GROVER_AMPLIFICATION,
    CONSCIOUSNESS_THRESHOLD, COHERENCE_MINIMUM,
    # Evolution state
    EVOLUTION_STAGE, EVOLUTION_INDEX, EVOLUTION_TOTAL_STAGES,
    O2_SUPERPOSITION_STATES,
    # God Code spectrum & workspace
    GOD_CODE_SPECTRUM, LINK_DRIFT_ENVELOPE, LINK_SACRED_DYNAMIC_BOUNDS,
    QUANTUM_LINKED_FILES, ALL_REPO_FILES, WORKSPACE_ROOT, STATE_FILE,
    # Qiskit availability
    QISKIT_AVAILABLE,
    # Version
    VERSION,
    # 4D God Code equation
    god_code_4d,
    # Engine lazy-loaders
    _get_science_engine, _get_math_engine, _get_code_engine,
    _get_asi_core, _get_agi_core, _get_gate_engine,
)

# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — Data Models
# ═══════════════════════════════════════════════════════════════════════════════
from .models import QuantumLink, StressTestResult, CrossModalLink, ChronoEntry

# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — Quantum Math Core
# ═══════════════════════════════════════════════════════════════════════════════
from .math_core import QuantumMathCore

# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — Discovery & Building
# ═══════════════════════════════════════════════════════════════════════════════
from .scanner import QuantumLinkScanner
from .builder import QuantumLinkBuilder, GodCodeMathVerifier

# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 4 — Quantum Physics Processors & Research
# ═══════════════════════════════════════════════════════════════════════════════
from .processors import (
    GroverQuantumProcessor, QuantumTunnelingAnalyzer, EPREntanglementVerifier,
    DecoherenceShieldTester, TopologicalBraidingTester, HilbertSpaceNavigator,
    QuantumFourierLinkAnalyzer, GodCodeResonanceVerifier,
    EntanglementDistillationEngine,
)
from .testing import (
    QuantumStressTestEngine, CrossModalAnalyzer,
    QuantumUpgradeEngine, QuantumRepairEngine,
)
from .research import ResearchMemoryBank, QuantumResearchEngine, ProbabilityWaveCollapseResearch, L104WeakMeasurement, SageModeInference
from .sage_circuits import (
    NoiseFloorSuppressionCircuit, DemonDenoiseCircuit,
    ZNERecoveryCircuit, EntropyCascadeCircuit, SageNDECircuit,
)
from .quantum_deep_link import (
    EPRConsensusTeleporter, GroverKnowledgeExtractor, PhaseKickbackScorer,
    EntanglementSwapBridge, DensityMatrixFusion, ErrorCorrectedConsensus,
    SacredResonanceHarmonizer, VQEConsensusOptimizer, QuantumWalkKBSearch,
    RecursiveFeedbackLoop, QuantumDeepLink, CoherenceProtectedDeepLink,
)
from .discoveries import QuantumDiscoveryEngine, DISCOVERY_CATALOG
from .genetic_refiner import (
    L104GeneticRefiner, GeneticIndividual,
    god_code_4d, abcd_to_x, x_to_abcd,
    genetic_refine_from_wave_collapse,
)
from .computation import (
    QuantumRegister, QuantumNeuron, QuantumCluster, QuantumCPU,
    QuantumEnvironment, O2MolecularBondProcessor, QuantumLinkComputationEngine,
)

# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 4b — Quantum LDPC Error Correction
# ═══════════════════════════════════════════════════════════════════════════════
from .qldpc import (
    CSSCode, TannerGraph, DecodingResult,
    CSSCodeConstructor,
    BeliefPropagationDecoder, BPOSDDecoder,
    DistributedSyndromeExtractor,
    LogicalErrorRateEstimator,
    QuantumLDPCSacredIntegration,
    create_qldpc_code, full_qldpc_pipeline,
)

# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 4c — Pauli Algebra & Quantum Gates (from math_core)
# ═══════════════════════════════════════════════════════════════════════════════
from .math_core import (
    PAULI_I, PAULI_X, PAULI_Y, PAULI_Z, PAULI_SET,
    HADAMARD, PHASE_S, T_GATE, CNOT_MATRIX,
)

# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 4d — Quantum Manifold Intelligence (v11.0.0)
# ═══════════════════════════════════════════════════════════════════════════════
from .manifold import (
    QuantumManifoldLearner, MultipartiteEntanglementNetwork,
    QuantumPredictiveOracle,
)

# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 5 — Dynamism & Intelligence
# ═══════════════════════════════════════════════════════════════════════════════
from .dynamism import LinkDynamismEngine, LinkOuroborosNirvanicEngine
from .intelligence import (
    EvolutionTracker, AgenticLoop, StochasticLinkResearchLab,
    LinkChronolizer, ConsciousnessO2LinkEngine, LinkTestGenerator,
    QuantumLinkCrossPollinationEngine, InterBuilderFeedbackBus,
    QuantumLinkSelfHealer, LinkTemporalMemoryBank,
)

# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 6 — Orchestrator (Brain)
# ═══════════════════════════════════════════════════════════════════════════════
from .brain import L104QuantumBrain, main

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════
quantum_brain = L104QuantumBrain()

# ═══════════════════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY ALIASES
# ═══════════════════════════════════════════════════════════════════════════════

# Old monolith class names → new package locations
QuantumLinkBuilderMonolith = QuantumLinkBuilder  # was the primary class
QuantumBrain = L104QuantumBrain

# Convenience: god_code function re-export
from .constants import god_code
from .constants import god_code_4d

# ═══════════════════════════════════════════════════════════════════════════════
# __all__ — Public API
# ═══════════════════════════════════════════════════════════════════════════════
__all__ = [
    # Meta
    "__version__",
    # Constants — Sacred
    "PHI", "PHI_GROWTH", "PHI_INV", "GOD_CODE", "GOD_CODE_HZ", "INVARIANT",
    "CALABI_YAU_DIM", "CHSH_BOUND", "GROVER_AMPLIFICATION",
    "CONSCIOUSNESS_THRESHOLD", "COHERENCE_MINIMUM",
    "EVOLUTION_STAGE", "EVOLUTION_INDEX", "EVOLUTION_TOTAL_STAGES",
    "O2_SUPERPOSITION_STATES",
    "GOD_CODE_SPECTRUM", "LINK_DRIFT_ENVELOPE", "LINK_SACRED_DYNAMIC_BOUNDS",
    "QUANTUM_LINKED_FILES", "ALL_REPO_FILES", "WORKSPACE_ROOT", "STATE_FILE",
    "QISKIT_AVAILABLE", "VERSION",
    "god_code",
    # Data Models
    "QuantumLink", "StressTestResult", "CrossModalLink", "ChronoEntry",
    # Math Core
    "QuantumMathCore",
    # Discovery
    "QuantumLinkScanner",
    "QuantumLinkBuilder", "GodCodeMathVerifier",
    # Processors
    "GroverQuantumProcessor", "QuantumTunnelingAnalyzer", "EPREntanglementVerifier",
    "DecoherenceShieldTester", "TopologicalBraidingTester", "HilbertSpaceNavigator",
    "QuantumFourierLinkAnalyzer", "GodCodeResonanceVerifier",
    "EntanglementDistillationEngine",
    # Testing
    "QuantumStressTestEngine", "CrossModalAnalyzer",
    "QuantumUpgradeEngine", "QuantumRepairEngine",
    # Research
    "ResearchMemoryBank", "QuantumResearchEngine", "ProbabilityWaveCollapseResearch", "SageModeInference",
    # Genetic Refiner
    "L104GeneticRefiner", "GeneticIndividual",
    "god_code_4d", "abcd_to_x", "x_to_abcd",
    "genetic_refine_from_wave_collapse",
    # Computation
    "QuantumRegister", "QuantumNeuron", "QuantumCluster", "QuantumCPU",
    "QuantumEnvironment", "O2MolecularBondProcessor", "QuantumLinkComputationEngine",
    # Dynamism
    "LinkDynamismEngine", "LinkOuroborosNirvanicEngine",
    # Intelligence
    "EvolutionTracker", "AgenticLoop", "StochasticLinkResearchLab",
    "LinkChronolizer", "ConsciousnessO2LinkEngine", "LinkTestGenerator",
    "QuantumLinkCrossPollinationEngine", "InterBuilderFeedbackBus",
    "QuantumLinkSelfHealer", "LinkTemporalMemoryBank",
    # Brain
    "L104QuantumBrain", "quantum_brain", "main",
    # Quantum LDPC Error Correction
    "CSSCode", "TannerGraph", "DecodingResult",
    "CSSCodeConstructor",
    "BeliefPropagationDecoder", "BPOSDDecoder",
    "DistributedSyndromeExtractor",
    "LogicalErrorRateEstimator",
    "QuantumLDPCSacredIntegration",
    "create_qldpc_code", "full_qldpc_pipeline",
    # Pauli Algebra & Quantum Gates
    "PAULI_I", "PAULI_X", "PAULI_Y", "PAULI_Z", "PAULI_SET",
    "HADAMARD", "PHASE_S", "T_GATE", "CNOT_MATRIX",
    # Backward compat
    "QuantumLinkBuilderMonolith", "QuantumBrain",
    # Sage NDE Quantum Circuits
    "NoiseFloorSuppressionCircuit", "DemonDenoiseCircuit",
    "ZNERecoveryCircuit", "EntropyCascadeCircuit", "SageNDECircuit",
    # Quantum Deep Link — Brain ↔ Sage ↔ Intellect Entanglement
    "EPRConsensusTeleporter", "GroverKnowledgeExtractor", "PhaseKickbackScorer",
    "EntanglementSwapBridge", "DensityMatrixFusion", "ErrorCorrectedConsensus",
    "SacredResonanceHarmonizer", "VQEConsensusOptimizer", "QuantumWalkKBSearch",
    "RecursiveFeedbackLoop", "QuantumDeepLink", "CoherenceProtectedDeepLink",
    # Quantum Discovery Engine
    "QuantumDiscoveryEngine", "DISCOVERY_CATALOG",
    # Quantum Manifold Intelligence (v11.0.0)
    "QuantumManifoldLearner", "MultipartiteEntanglementNetwork",
    "QuantumPredictiveOracle",
    # Engine lazy-loaders
    "_get_science_engine", "_get_math_engine", "_get_code_engine",
    "_get_asi_core", "_get_agi_core", "_get_gate_engine",
]
