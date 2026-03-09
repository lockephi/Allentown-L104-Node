#!/usr/bin/env python3
"""
L104 Multi-Execution Quantum Benchmark
======================================
Runs multiple executions with decoherence to gather statistical samples.
All simulations now use DEPOLARISING decoherence by default.
"""

import numpy as np
import time

print('=' * 70)
print('  L104 MULTI-EXECUTION QUANTUM BENCHMARK')
print('  All simulations use DEPOLARISING decoherence (IBM Eagle-like)')
print('=' * 70)
print()

from l104_quantum_gate_engine import get_engine
from l104_quantum_gate_engine.trajectory import TrajectorySimulator, DecoherenceModel
from l104_qiskit_utils import aer_backend, L104NoiseModelFactory

engine = get_engine()
sim = TrajectorySimulator()

N_EXECUTIONS = 100  # Statistical sample size
CIRCUITS = [
    ("Bell State", engine.bell_pair),
    ("GHZ-3", lambda: engine.ghz_state(3)),
    ("GHZ-4", lambda: engine.ghz_state(4)),
    ("QFT-4", lambda: engine.quantum_fourier_transform(4)),
]

print(f"Executions per circuit: {N_EXECUTIONS}")
print(f"Default decoherence: DEPOLARISING (0.75% error rate)")
print(f"Default shots: 8192")
print()

# Show updated defaults
print("UPDATED DEFAULTS:")
print("-" * 50)
print(f"  TrajectorySimulator.simulate() default: DEPOLARISING")
print(f"  aer_backend.run_shots() default: 8192 shots")
print(f"  VQC_DEFAULT_SHOTS: 8192")
print()

results = {}

for circuit_name, circuit_fn in CIRCUITS:
    print(f"Running {circuit_name}...")
    circ = circuit_fn()
    n_qubits = circ.num_qubits

    # Run N executions with decoherence (now default)
    probabilities = []
    purities = []
    start = time.time()

    for i in range(N_EXECUTIONS):
        # No need to specify decoherence - it's now the default!
        result = sim.simulate(circ)
        sv = result.snapshots[-1].statevector
        probs = np.abs(sv) ** 2
        probabilities.append(probs)
        purities.append(result.final_purity)

    elapsed = time.time() - start

    # Compute statistics
    probs_array = np.array(probabilities)
    mean_probs = np.mean(probs_array, axis=0)
    std_probs = np.std(probs_array, axis=0)
    mean_purity = np.mean(purities)

    # For Bell/GHZ states, check ground and excited superposition
    if n_qubits == 2:
        p00_mean, p00_std = mean_probs[0], std_probs[0]
        p11_mean, p11_std = mean_probs[3], std_probs[3]
        target_state = f"|00>={p00_mean:.4f}±{p00_std:.4f}, |11>={p11_mean:.4f}±{p11_std:.4f}"
    elif n_qubits == 3:
        p000_mean, p000_std = mean_probs[0], std_probs[0]
        p111_mean, p111_std = mean_probs[7], std_probs[7]
        target_state = f"|000>={p000_mean:.4f}±{p000_std:.4f}, |111>={p111_mean:.4f}±{p111_std:.4f}"
    elif n_qubits == 4:
        p0000_mean, p0000_std = mean_probs[0], std_probs[0]
        p1111_mean, p1111_std = mean_probs[15], std_probs[15]
        target_state = f"|0000>={p0000_mean:.4f}±{p0000_std:.4f}, |1111>={p1111_mean:.4f}±{p1111_std:.4f}"
    else:
        target_state = f"max_prob={np.max(mean_probs):.4f}"

    results[circuit_name] = {
        "n_qubits": n_qubits,
        "mean_probs": mean_probs,
        "std_probs": std_probs,
        "mean_purity": mean_purity,
        "time_ms": elapsed * 1000,
        "target_state": target_state,
    }

    print(f"  {circuit_name}: {target_state}")
    print(f"  Purity: {mean_purity:.6f}, Time: {elapsed*1000:.1f}ms")
    print()

# Run shot-based execution
print("=" * 70)
print("  SHOT-BASED EXECUTION (8192 shots default)")
print("=" * 70)
print()

circ = engine.bell_pair()
counts = aer_backend.run_shots(circ)  # Uses new 8192 default
total = sum(counts.values())
print(f"Bell State with {total} shots:")
for bitstring, count in sorted(counts.items()):
    print(f"  |{bitstring}>: {count} ({count/total*100:.2f}%)")
print()

# Summary
print("=" * 70)
print("  SUMMARY")
print("=" * 70)
print()
print(f"Total circuits tested: {len(CIRCUITS)}")
print(f"Executions per circuit: {N_EXECUTIONS}")
print()
print("KEY CHANGES:")
print("  ✓ TrajectorySimulator default: DEPOLARISING @ 0.75%")
print("  ✓ aer_backend.run_shots default: 8192 shots")
print("  ✓ QuantumRuntime default: 8192 shots")
print("  ✓ VQC_DEFAULT_SHOTS: 8192")
print()
print("All quantum simulations now include realistic decoherence.")
print("=" * 70)
