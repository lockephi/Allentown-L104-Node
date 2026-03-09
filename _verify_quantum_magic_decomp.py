#!/usr/bin/env python3
"""Verify all l104_quantum_magic imports work after decomposition."""

# Test all external imports that exist in the codebase
from l104_quantum_magic import QuantumInferenceEngine
from l104_quantum_magic import CausalReasoner
from l104_quantum_magic import CounterfactualEngine
from l104_quantum_magic import PatternRecognizer
from l104_quantum_magic import MetaCognition
from l104_quantum_magic import PredictiveReasoner
from l104_quantum_magic import GoalPlanner
from l104_quantum_magic import AttentionMechanism
from l104_quantum_magic import AbductiveReasoner
from l104_quantum_magic import AdaptiveLearner
from l104_quantum_magic import ContextualMemory
from l104_quantum_magic import Observation
from l104_quantum_magic import QuantumMagicSynthesizer
from l104_quantum_magic import SuperpositionMagic
from l104_quantum_magic import EntanglementMagic
from l104_quantum_magic import WaveFunctionMagic
from l104_quantum_magic import HyperdimensionalMagic
from l104_quantum_magic import ReasoningStrategy
from l104_quantum_magic import GOD_CODE, PHI

print("All 19 imports OK")
print(f"  GOD_CODE: {GOD_CODE}")
print(f"  PHI: {PHI}")
print(f"  QuantumInferenceEngine: {QuantumInferenceEngine}")
print(f"  QuantumMagicSynthesizer: {QuantumMagicSynthesizer}")
print(f"  CausalReasoner: {CausalReasoner}")
print(f"  Strategies: {len(list(ReasoningStrategy))}")

# Quick functional test
synth = QuantumMagicSynthesizer()
result = synth.synthesize_with_intelligence()
print(f"  Synthesis discoveries: {result['num_discoveries']}")
print(f"  Magic quotient: {result['magic_quotient']:.4f}")
print("PASS")
