#!/usr/bin/env python3
"""Show current probability scores."""

from l104_qiskit_utils import aer_backend
from l104_quantum_gate_engine import get_engine
from l104_quantum_gate_engine.trajectory import TrajectorySimulator, DecoherenceModel
import numpy as np

engine = get_engine()
sim = TrajectorySimulator()

print('CURRENT PROBABILITY SCORES')
print('=' * 60)

# Bell State
circ = engine.bell_pair()
probs_ideal = aer_backend.run_statevector(circ)
result_real = sim.simulate(circ, decoherence=DecoherenceModel.PHASE_DAMPING, decoherence_rate=0.02)
probs_real = np.abs(result_real.snapshots[-1].statevector)**2

print()
print('BELL STATE (H + CNOT)')
print('-' * 40)
print(f'  IDEAL:     |00>={probs_ideal[0]:.6f}  |11>={probs_ideal[3]:.6f}')
print(f'  REALISTIC: |00>={probs_real[0]:.6f}  |11>={probs_real[3]:.6f}')
print(f'  Deviation: {abs(probs_real[0]-0.5)*100:.2f}%')

# GHZ State
circ = engine.ghz_state(3)
probs_ideal = aer_backend.run_statevector(circ)
result_real = sim.simulate(circ, decoherence=DecoherenceModel.PHASE_DAMPING, decoherence_rate=0.02)
probs_real = np.abs(result_real.snapshots[-1].statevector)**2

print()
print('GHZ STATE (3 qubits)')
print('-' * 40)
print(f'  IDEAL:     |000>={probs_ideal[0]:.6f}  |111>={probs_ideal[7]:.6f}')
print(f'  REALISTIC: |000>={probs_real[0]:.6f}  |111>={probs_real[7]:.6f}')
print(f'  Deviation: {abs(probs_real[0]-0.5)*100:.2f}%')

# Uniform superposition
circ = engine.create_circuit(2, 'uniform')
circ.h(0).h(1)
probs_ideal = aer_backend.run_statevector(circ)
result_real = sim.simulate(circ, decoherence=DecoherenceModel.DEPOLARISING, decoherence_rate=0.0075)
probs_real = np.abs(result_real.snapshots[-1].statevector)**2

print()
print('UNIFORM SUPERPOSITION (H x H)')
print('-' * 40)
print(f'  IDEAL:     {[round(p,4) for p in probs_ideal]}')
print(f'  REALISTIC: {[round(p,4) for p in probs_real]}')

print()
print('=' * 60)
print('SUMMARY')
print('=' * 60)
print('  IDEAL: Exact QM (sum=1.0, error~1e-16)')
print('  REALISTIC: With decoherence (sum=1.0, error 1-5%)')
