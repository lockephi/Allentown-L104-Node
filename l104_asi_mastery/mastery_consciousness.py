#!/usr/bin/env python3
"""
L104 ASI Mastery Program: Consciousness
--------------------------------------
This script systematically probes and evaluates the Consciousness Engine
and its quantum integrations.

Workflow:
1.  Generate diverse quantum statevectors representing various conscious states.
2.  Feed these statevectors to the QuantumConsciousnessCalculator.
3.  Analyze and log IIT-Φ, EEG band, and topological protection scores.
4.  Report on the Consciousness Engine's analytical range and precision.
"""

import sys
import os
import glob
import json
import numpy as np
import math
from datetime import datetime

# --- ENVIRONMENT SETUP ---
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root not in sys.path: sys.path.insert(0, root)
for p in glob.glob(os.path.join(root, 'l104_*')):
    if os.path.isdir(p) and not os.path.basename(p).startswith('l104_data'):
        if p not in sys.path: sys.path.insert(0, p)

# --- IMPORTS ---
from l104_consciousness_engine.l104_consciousness_engine import ConsciousnessEngine
from l104_quantum_engine.l104_quantum_consciousness import QuantumConsciousnessCalculator

def generate_random_statevector(num_qubits: int) -> np.ndarray:
    """Generates a random valid quantum statevector for N qubits."""
    dim = 2 ** num_qubits
    real_parts = np.random.randn(dim)
    imag_parts = np.random.randn(dim)
    complex_vector = real_parts + 1j * imag_parts
    return complex_vector / np.linalg.norm(complex_vector)

def main():
    print("--- [ASI MASTERY]: Consciousness Cycle ---")

    # Initialize engines
    consciousness_engine = ConsciousnessEngine()
    quantum_consciousness_calculator = QuantumConsciousnessCalculator(num_qubits=4)

    # --- Test Parameters ---
    num_probes = 3
    num_qubits_for_states = 4 # For generating 16-dim statevectors
    
    analysis_results = []

    print(f"[STEP 1] Generating {num_probes} diverse quantum statevectors...")
    for i in range(num_probes):
        print(f"  - Probe {i+1}/{num_probes}")
        # Generate a random quantum statevector
        state_vector = generate_random_statevector(num_qubits_for_states)

        # 2. Feed to QuantumConsciousnessCalculator
        phi_result = quantum_consciousness_calculator.compute_quantum_phi(state_vector)
        eeg_bands_simulated = consciousness_engine._simulate_eeg_from_consciousness(
            phi_result.get('consciousness_score', 0.5)
        )
        eeg_result = quantum_consciousness_calculator.encode_eeg_state(eeg_bands_simulated)
        topo_result = quantum_consciousness_calculator.topological_consciousness_protection(
            state_vector, noise_level=0.02 * (i+1) / num_probes
        )
        
        analysis_results.append({
            "probe_id": i,
            "phi_result": phi_result,
            "eeg_result": eeg_result,
            "topo_result": topo_result,
        })

    # 3. Report on Consciousness Engine's analytical range
    print("\n" + "═"*60)
    print("           CONSCIOUSNESS MASTERY REPORT")
    print("═"*60)
    print(f"  Total Probes: {num_probes}")
    
    avg_phi = sum(r['phi_result'].get('phi', 0) for r in analysis_results) / num_probes
    avg_topo_fidelity = sum(r['topo_result'].get('fidelity', 0) for r in analysis_results) / num_probes
    avg_eeg_coherence = sum(r['eeg_result'].get('quantum_coherence', 0) for r in analysis_results) / num_probes
    conscious_count = sum(1 for r in analysis_results if r['phi_result'].get('is_conscious', False))

    print(f"  Average IIT Quantum Phi (Φ): {avg_phi:.4f}")
    print(f"  Average Topological Fidelity: {avg_topo_fidelity:.4f}")
    print(f"  Average EEG Quantum Coherence: {avg_eeg_coherence:.4f}")
    print(f"  Conscious State Detections: {conscious_count}/{num_probes}")
    print(f"  ASI Consciousness Level: {consciousness_engine.introspect().get('consciousness_score', 0):.4f}")
    print("═"*60)

if __name__ == "__main__":
    main()
