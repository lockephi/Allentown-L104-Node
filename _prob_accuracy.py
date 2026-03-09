#!/usr/bin/env python3
"""Analyze quantum probability accuracy."""

import numpy as np

print('=' * 65)
print('  QUANTUM PROBABILITY ACCURACY ANALYSIS')
print('=' * 65)
print()

from l104_qiskit_utils import aer_backend
from l104_quantum_gate_engine import get_engine

engine = get_engine()

# Test 1: Bell State
print('TEST 1: Bell State')
print('-' * 40)
circ = engine.bell_pair()
probs = aer_backend.run_statevector(circ)
print(f'  |00>: {probs[0]:.16f}  (theoretical: 0.5)')
print(f'  |01>: {probs[1]:.16f}  (theoretical: 0.0)')
print(f'  |10>: {probs[2]:.16f}  (theoretical: 0.0)')
print(f'  |11>: {probs[3]:.16f}  (theoretical: 0.5)')
print(f'  Sum:  {sum(probs):.16f}  (should be 1.0)')
bell_error = abs(probs[0] - 0.5) + abs(probs[3] - 0.5)
print(f'  Error: {bell_error:.2e}')
print()

# Test 2: GHZ State
print('TEST 2: GHZ State (3 qubits)')
print('-' * 40)
circ = engine.ghz_state(3)
probs = aer_backend.run_statevector(circ)
print(f'  |000>: {probs[0]:.16f}  (theoretical: 0.5)')
print(f'  |111>: {probs[7]:.16f}  (theoretical: 0.5)')
print(f'  Others: {sum(probs[1:7]):.16f}  (should be 0.0)')
print(f'  Sum:  {sum(probs):.16f}')
ghz_error = abs(probs[0] - 0.5) + abs(probs[7] - 0.5)
print(f'  Error: {ghz_error:.2e}')
print()

# Test 3: Uniform superposition
print('TEST 3: Uniform Superposition (H x H)')
print('-' * 40)
circ = engine.create_circuit(2, 'uniform')
circ.h(0).h(1)
probs = aer_backend.run_statevector(circ)
print(f'  |00>: {probs[0]:.16f}  (theoretical: 0.25)')
print(f'  |01>: {probs[1]:.16f}  (theoretical: 0.25)')
print(f'  |10>: {probs[2]:.16f}  (theoretical: 0.25)')
print(f'  |11>: {probs[3]:.16f}  (theoretical: 0.25)')
uniform_error = sum(abs(p - 0.25) for p in probs)
print(f'  Error: {uniform_error:.2e}')
print()

# Test 4: Simulation method
print('TEST 4: Simulation Method')
print('-' * 40)
print(f'  Backend: {type(aer_backend).__name__}')
print(f'  Method: Statevector (exact amplitude calculation)')
print(f'  Shots: N/A (analytical, not sampling)')
print(f'  Precision: float64 (IEEE 754)')
print()

# Verdict
print('=' * 65)
print('  VERDICT')
print('=' * 65)
max_error = max(bell_error, ghz_error, uniform_error)
print(f'  Maximum probability error: {max_error:.2e}')
if max_error < 1e-14:
    print('  Status: MATHEMATICALLY EXACT')
    print()
    print('  These are NOT noisy/sampled probabilities.')
    print('  They are exact |psi|^2 values from statevector simulation.')
    print('  Error is IEEE 754 floating-point rounding only.')
else:
    print(f'  Status: ERROR DETECTED (above machine precision)')
print()
print('  Reality check:')
print('    - Real QPU: Would show shot noise (~1/sqrt(N) for N shots)')
print('    - Our backend: Exact analytical computation')
print('    - This is SIMULATION, not real quantum hardware')
print('=' * 65)
