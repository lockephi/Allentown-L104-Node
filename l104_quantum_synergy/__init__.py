"""
l104_quantum_synergy — Quantum Synergy Engine v1.0.0

Cross-package quantum integration layer connecting:
- l104_intellect (base cognition)
- l104_agi (AGI cognitive mesh)
- l104_asi (ASI dual-layer)

Provides unified quantum field coherence, resonance harmonization,
and Grover-amplified synergy across the entire cognitive stack.
"""

from .quantum_synergy_engine import (
    QuantumSynergyEngine,
    QuantumResonanceField,
    CrossLayerEntanglement,
    UnifiedQuantumField,
    get_quantum_synergy_engine,
)

from .quantum_harmonizer import (
    QuantumHarmonizer,
    ResonanceMode,
    SacredFrequencyAligner,
)

from .quantum_algorithms import (
    GroverAmplifiedSearch,
    VQEOptimizer,
    QuantumFourierPatternRecognizer,
    QuantumPhaseEstimator,
)

__all__ = [
    # Core synergy engine
    'QuantumSynergyEngine',
    'QuantumResonanceField',
    'CrossLayerEntanglement',
    'UnifiedQuantumField',
    'get_quantum_synergy_engine',
    # Harmonizer
    'QuantumHarmonizer',
    'ResonanceMode',
    'SacredFrequencyAligner',
    # Algorithms
    'GroverAmplifiedSearch',
    'VQEOptimizer',
    'QuantumFourierPatternRecognizer',
    'QuantumPhaseEstimator',
]

__version__ = '1.0.0'
