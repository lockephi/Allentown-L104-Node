#!/usr/bin/env python3
"""Compare ideal vs realistic quantum simulation with decoherence."""

import numpy as np

print('=' * 70)
print('  IDEAL vs REALISTIC QUANTUM SIMULATION')
print('=' * 70)
print()

from l104_quantum_gate_engine import get_engine
from l104_quantum_gate_engine.trajectory import TrajectorySimulator, DecoherenceModel
from l104_qiskit_utils import L104NoiseModelFactory

engine = get_engine()
sim = TrajectorySimulator()

# Build a Bell circuit
circ = engine.bell_pair()

def get_probs(result):
    """Extract probabilities from final snapshot."""
    final = result.snapshots[-1]
    if final.statevector is not None:
        return np.abs(final.statevector) ** 2
    elif final.density_matrix is not None:
        return np.real(np.diag(final.density_matrix))
    return np.zeros(4)

print('CIRCUIT: Bell State (H on q0, CNOT q0->q1)')
print('THEORETICAL: |00> = 0.5, |11> = 0.5')
print()

# 1. IDEAL (current default)
print('1. IDEAL SIMULATION (no decoherence)')
print('-' * 50)
result_ideal = sim.simulate(circ, decoherence=DecoherenceModel.NONE)
probs_ideal = get_probs(result_ideal)
print(f'   |00>: {probs_ideal[0]:.6f}')
print(f'   |01>: {probs_ideal[1]:.6f}')
print(f'   |10>: {probs_ideal[2]:.6f}')
print(f'   |11>: {probs_ideal[3]:.6f}')
print(f'   Sum:  {sum(probs_ideal):.10f}')
print(f'   Purity: {result_ideal.final_purity:.6f}')
print()

# 2. PHASE DAMPING (T2 dephasing)
print('2. PHASE DAMPING (T2 dephasing, decoherence_rate=0.02)')
print('-' * 50)
result_pd = sim.simulate(circ, decoherence=DecoherenceModel.PHASE_DAMPING, decoherence_rate=0.02)
probs_pd = get_probs(result_pd)
print(f'   |00>: {probs_pd[0]:.6f}')
print(f'   |01>: {probs_pd[1]:.6f}')
print(f'   |10>: {probs_pd[2]:.6f}')
print(f'   |11>: {probs_pd[3]:.6f}')
print(f'   Purity: {result_pd.final_purity:.6f}')
print()

# 3. AMPLITUDE DAMPING (T1 relaxation)
print('3. AMPLITUDE DAMPING (T1 decay, decoherence_rate=0.01)')
print('-' * 50)
result_ad = sim.simulate(circ, decoherence=DecoherenceModel.AMPLITUDE_DAMPING, decoherence_rate=0.01)
probs_ad = get_probs(result_ad)
print(f'   |00>: {probs_ad[0]:.6f}')
print(f'   |01>: {probs_ad[1]:.6f}')
print(f'   |10>: {probs_ad[2]:.6f}')
print(f'   |11>: {probs_ad[3]:.6f}')
print(f'   Purity: {result_ad.final_purity:.6f}')
print()

# 4. DEPOLARISING (random errors)
print('4. DEPOLARISING (gate errors, decoherence_rate=0.0075)')
print('-' * 50)
result_dp = sim.simulate(circ, decoherence=DecoherenceModel.DEPOLARISING, decoherence_rate=0.0075)
probs_dp = get_probs(result_dp)
print(f'   |00>: {probs_dp[0]:.6f}')
print(f'   |01>: {probs_dp[1]:.6f}')
print(f'   |10>: {probs_dp[2]:.6f}')
print(f'   |11>: {probs_dp[3]:.6f}')
print(f'   Purity: {result_dp.final_purity:.6f}')
print()

# 5. SACRED CHANNEL (L104 custom)
print('5. SACRED CHANNEL (PHI-weighted, decoherence_rate=0.02)')
print('-' * 50)
result_sc = sim.simulate(circ, decoherence=DecoherenceModel.SACRED, decoherence_rate=0.02)
probs_sc = get_probs(result_sc)
print(f'   |00>: {probs_sc[0]:.6f}')
print(f'   |01>: {probs_sc[1]:.6f}')
print(f'   |10>: {probs_sc[2]:.6f}')
print(f'   |11>: {probs_sc[3]:.6f}')
print(f'   Purity: {result_sc.final_purity:.6f}')
print()

# Noise profile info
print('=' * 70)
print('  NOISE PROFILES AVAILABLE')
print('=' * 70)
for name, params in L104NoiseModelFactory.PROFILES.items():
    t1 = params.get('t1_us', 0)
    t2 = params.get('t2_us', 0)
    sg = params.get('single_gate_error', 0)
    cx = params.get('cx_gate_error', 0)
    if t1 < 1e10:
        print(f'  {name:20s}: T1={t1:.0f}us, T2={t2:.0f}us, 1Q={sg*100:.3f}%, 2Q={cx*100:.3f}%')
    else:
        print(f'  {name:20s}: IDEAL (no noise)')
print()

# Summary
print('=' * 70)
print('  ACCURACY = REALISTIC DECOHERENCE')
print('=' * 70)
print()
print('  Real quantum computers have:')
print('    - T1 relaxation (energy loss to environment)')
print('    - T2 dephasing (phase information loss)')  
print('    - Gate errors (imperfect control pulses)')
print('    - Readout errors (measurement mistakes)')
print()
print('  To simulate ACCURATELY, always include decoherence:')
print()
print('    from l104_quantum_gate_engine.trajectory import TrajectorySimulator, DecoherenceModel')
print('    sim = TrajectorySimulator()')
print('    result = sim.simulate(circuit, decoherence=DecoherenceModel.DEPOLARISING, decoherence_rate=0.0075)')
print()
print('=' * 70)
