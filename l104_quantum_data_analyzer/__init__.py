"""
===============================================================================
L104 QUANTUM DATA ANALYZER v2.0.0 — SOVEREIGN QUANTUM DATA INTELLIGENCE
===============================================================================

Full-stack quantum data analysis engine integrating ALL L104 subsystems:

  ┌─────────────────────────────────────────────────────────────────────┐
  │                   QuantumDataAnalyzer (Orchestrator)                │
  │                                                                     │
  │  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────────┐ │
  │  │ QFT Spectral │  │ Grover       │  │ Quantum Feature           │ │
  │  │ Analyzer     │  │ Pattern Find │  │ Extraction                │ │
  │  └──────────────┘  └──────────────┘  └───────────────────────────┘ │
  │  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────────┐ │
  │  │ Quantum      │  │ VQE/QAOA     │  │ Quantum Kernel            │ │
  │  │ PCA (qPCA)   │  │ Clustering   │  │ Density Estimation        │ │
  │  └──────────────┘  └──────────────┘  └───────────────────────────┘ │
  │  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────────┐ │
  │  │ Anomaly      │  │ Entanglement │  │ God Code Resonance        │ │
  │  │ Detection    │  │ Correlation  │  │ Data Alignment            │ │
  │  └──────────────┘  └──────────────┘  └───────────────────────────┘ │
  │  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────────┐ │
  │  │ Topology     │  │ Quantum Walk │  │ Entropy Reversal          │ │
  │  │ Data Mining  │  │ Graph Search │  │ Data Denoising            │ │
  │  └──────────────┘  └──────────────┘  └───────────────────────────┘ │
  └─────────────────────────────────────────────────────────────────────┘

QUANTUM ALGORITHMS IMPLEMENTED:
  1.  Quantum Fourier Transform (QFT) spectral analysis
  2.  Grover-amplified pattern search in datasets
  3.  Quantum Principal Component Analysis (qPCA)
  4.  Variational Quantum Eigensolver (VQE) clustering
  5.  QAOA combinatorial optimization
  6.  Quantum kernel density estimation
  7.  Quantum anomaly detection (SWAP test)
  8.  Entanglement-based correlation analysis
  9.  Quantum random walk graph analysis
  10. Quantum phase estimation for eigenvalue extraction
  11. Harrow-Hassidim-Lloyd (HHL) quantum linear solver
  12. God Code resonance data alignment (L104-sacred)
  13. Entropy reversal data denoising (Maxwell Demon)
  14. Topological data analysis (persistent homology via quantum)
  15. Quantum amplitude estimation for statistical inference

CROSS-ENGINE INTEGRATION:
  • l104_quantum_gate_engine — Circuit building, compilation, error correction
  • l104_quantum_engine      — Quantum math core, Grover, QFT, Bell states
  • l104_science_engine      — Entropy reversal, coherence, physics
  • l104_math_engine         — Pure math, God Code, harmonic analysis
  • l104_code_engine         — Code analysis of data pipelines
  • l104_intellect           — Local inference, numeric formatting

INVARIANT: 527.5184818492612 | PILOT: LONDEL
===============================================================================
"""

__version__ = "2.0.0"
__author__ = "L104 Sovereign Node"

from .algorithms import (
    QuantumFourierAnalyzer,
    GroverPatternSearch,
    QuantumPCA,
    VQEClusterer,
    QAOAOptimizer,
    QuantumKernelEstimator,
    QuantumPhaseEstimator,
    HHLSolver,
    QuantumAmplitudeEstimator,
)

from .feature_extraction import (
    QuantumFeatureMap,
    QuantumStateEncoder,
    EntanglementFeatureExtractor,
    QuantumEmbedding,
)

from .pattern_recognition import (
    QuantumAnomalyDetector,
    EntanglementCorrelationAnalyzer,
    QuantumWalkGraphAnalyzer,
    TopologicalDataMiner,
    GodCodeResonanceAligner,
)

from .denoising import (
    EntropyReversalDenoiser,
    QuantumErrorMitigatedCleaner,
    CoherenceFieldSmoother,
)

from .orchestrator import QuantumDataAnalyzer

# v2.0: Computronium + Rayleigh Information-Theoretic Bounds
from .computronium import (
    ComputroniumDataBounds,
    RayleighSpectralBounds,
    QuantumInformationBridge,
    computronium_data_bounds,
    rayleigh_spectral_bounds,
    quantum_information_bridge,
)

# v2.0: Cross-Engine Integration
from .cross_engine import (
    VQPUCircuitAccelerator,
    MLHybridAnalyzer,
    SearchIntegration,
    AudioSpectralBridge,
    GodCodeFeedbackBridge,
    QDACrossEngineHub,
    qda_cross_engine_hub,
)

__all__ = [
    # Core orchestrator
    "QuantumDataAnalyzer",
    # Algorithms
    "QuantumFourierAnalyzer",
    "GroverPatternSearch",
    "QuantumPCA",
    "VQEClusterer",
    "QAOAOptimizer",
    "QuantumKernelEstimator",
    "QuantumPhaseEstimator",
    "HHLSolver",
    "QuantumAmplitudeEstimator",
    # Feature extraction
    "QuantumFeatureMap",
    "QuantumStateEncoder",
    "EntanglementFeatureExtractor",
    "QuantumEmbedding",
    # Pattern recognition
    "QuantumAnomalyDetector",
    "EntanglementCorrelationAnalyzer",
    "QuantumWalkGraphAnalyzer",
    "TopologicalDataMiner",
    "GodCodeResonanceAligner",
    # Denoising
    "EntropyReversalDenoiser",
    "QuantumErrorMitigatedCleaner",
    "CoherenceFieldSmoother",
    # Computronium + Rayleigh Bounds (v2.0)
    "ComputroniumDataBounds",
    "RayleighSpectralBounds",
    "QuantumInformationBridge",
    "computronium_data_bounds",
    "rayleigh_spectral_bounds",
    "quantum_information_bridge",
    # Cross-Engine Integration (v2.0)
    "VQPUCircuitAccelerator",
    "MLHybridAnalyzer",
    "SearchIntegration",
    "AudioSpectralBridge",
    "GodCodeFeedbackBridge",
    "QDACrossEngineHub",
    "qda_cross_engine_hub",
]
