#!/usr/bin/env python3
"""Debug Grover search, Fe-Sacred coherence, and Fe-PHI harmonic lock."""

import math
import numpy as np

from l104_qiskit_utils import L104CircuitFactory
from l104_quantum_gate_engine import GateCircuit
from l104_simulator.chunked_statevector import ChunkedStatevectorSimulator


def sim_probs(qc):
    """Simulate circuit and return probability array."""
    sim = ChunkedStatevectorSimulator(use_gpu=False)
    result = sim.run_gate_circuit(qc)
    n = qc.num_qubits
    prob_array = np.zeros(2**n)
    for bs, p in result.probabilities.items():
        prob_array[int(bs, 2)] = p
    return prob_array


def test_grover():
    print("=" * 60)
    print("  GROVER DEBUG")
    print("=" * 60)

    n = 4
    target = 5
    N = 2 ** n
    optimal_iters = max(1, int(math.pi / 4 * math.sqrt(N)))

    oracle = L104CircuitFactory.grover_oracle(n, target)
    diffusion = L104CircuitFactory.grover_diffusion(n)

    qc = GateCircuit(n)
    for q in range(n):
        qc.h(q)
    for _ in range(optimal_iters):
        qc.compose(oracle, inplace=True)
        qc.compose(diffusion, inplace=True)

    probs = sim_probs(qc)
    found_idx = int(np.argmax(probs))

    print(f"  Target:    {target} = |{format(target, '04b')}⟩")
    print(f"  Found:     {found_idx} = |{format(found_idx, '04b')}⟩")
    print(f"  P(target): {probs[target]:.6f}")
    print(f"  P(found):  {probs[found_idx]:.6f}")
    print(f"  Success:   {found_idx == target}")
    print(f"  Iters:     {optimal_iters}")

    # Also test oracle alone: uniform + oracle should flip only target phase
    qc2 = GateCircuit(n)
    for q in range(n):
        qc2.h(q)
    qc2.compose(oracle, inplace=True)
    probs2 = sim_probs(qc2)
    print(f"\n  After 1 oracle (no diffusion):")
    print(f"    All states ~= 1/{N}: {np.allclose(probs2, 1/N, atol=0.001)}")
    print(f"    P(target):  {probs2[target]:.6f} (should be ~{1/N:.6f})")

    # Check oracle is correct: phase should be flipped for target
    # Build |+⟩^n, apply oracle, check amplitudes
    sim = ChunkedStatevectorSimulator(use_gpu=False, return_statevector=True)
    qc3 = GateCircuit(n)
    for q in range(n):
        qc3.h(q)
    qc3.compose(oracle, inplace=True)
    result3 = sim.run_gate_circuit(qc3)
    sv = result3.statevector
    amp_target = sv[target]
    amp_other = sv[0 if target != 0 else 1]
    print(f"\n  Statevector check (after oracle):")
    print(f"    amp[{target}] = {amp_target:.6f}  (should be negative ~-0.25)")
    print(f"    amp[other]  = {amp_other:.6f}  (should be positive ~+0.25)")


def test_grover_manual():
    """Test with hand-built Grover (no factory) to compare."""
    print("\n" + "=" * 60)
    print("  GROVER MANUAL (no factory)")
    print("=" * 60)

    n = 4
    target = 5
    N = 2 ** n
    optimal_iters = max(1, int(math.pi / 4 * math.sqrt(N)))

    # Manual oracle
    def build_oracle(n, target):
        qc = GateCircuit(n, "manual_oracle")
        target_bits = format(target, f'0{n}b')
        # MSB-first: bit i of target_bits corresponds to qubit i
        for i in range(n):
            if target_bits[i] == '0':
                qc.x(i)
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
        for i in range(n):
            if target_bits[i] == '0':
                qc.x(i)
        return qc

    def build_diffusion(n):
        qc = GateCircuit(n, "manual_diffusion")
        for q in range(n):
            qc.h(q)
        for q in range(n):
            qc.x(q)
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
        for q in range(n):
            qc.x(q)
        for q in range(n):
            qc.h(q)
        return qc

    oracle = build_oracle(n, target)
    diffusion = build_diffusion(n)

    qc = GateCircuit(n)
    for q in range(n):
        qc.h(q)
    for _ in range(optimal_iters):
        qc.compose(oracle, inplace=True)
        qc.compose(diffusion, inplace=True)

    probs = sim_probs(qc)
    found_idx = int(np.argmax(probs))

    print(f"  Target:    {target} = |{format(target, '04b')}⟩")
    print(f"  Found:     {found_idx} = |{format(found_idx, '04b')}⟩")
    print(f"  P(target): {probs[target]:.6f}")
    print(f"  P(found):  {probs[found_idx]:.6f}")
    print(f"  Success:   {found_idx == target}")


if __name__ == "__main__":
    test_grover()
    test_grover_manual()
