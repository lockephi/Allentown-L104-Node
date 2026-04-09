#!/usr/bin/env python3
"""
Demonstration of shot‑based sampling and statistical analysis utilities.

Shows how to use the new functions in `advanced_circuits` to:
  - sample circuits with finite shots,
  - compute empirical statistics,
  - compare distributions,
  - run comprehensive shot‑based benchmarks.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from l104_simulator.advanced_circuits import (
    surface_code_plaquette,
    qft_circuit,
    sacred_variational_ansatz,
    sample_circuit,
    statistical_analysis,
    compare_shots,
    run_shot_benchmark,
)

def demo_basic_sampling():
    """Sample a simple circuit and print results."""
    print("=== 1. Basic circuit sampling ===")
    qc = qft_circuit(n_qubits=3)
    print(f"Circuit: {qc.name}, {qc.n_qubits} qubits")

    result = sample_circuit(qc, shots=1000, seed=42)
    print(f"  Shots: {result['shots']}")
    print(f"  Unique outcomes: {len(result['counts'])}")
    print(f"  Total counts: {result['total_counts']}")

    # Show top‑3 most frequent bitstrings
    sorted_counts = sorted(result['counts'].items(), key=lambda kv: kv[1], reverse=True)
    for bitstr, cnt in sorted_counts[:3]:
        print(f"    {bitstr}: {cnt} (prob {cnt/result['shots']:.4f})")

    print(f"  Per‑qubit ⟨Z⟩: {[f'{z:.3f}' for z in result['expectation_z']]}")
    print(f"  Shot entropy: {result['shot_entropy']:.3f} bits")
    print()

def demo_statistical_analysis():
    """Compute detailed statistics from counts."""
    print("=== 2. Statistical analysis ===")
    qc = sacred_variational_ansatz(n_qubits=2, layers=1)
    sample = sample_circuit(qc, shots=800, seed=123)
    counts = sample["counts"]

    stats = statistical_analysis(counts)
    print(f"Circuit: {qc.name}")
    print(f"  Total shots: {stats['total_shots']}")
    print(f"  Unique outcomes: {stats['unique_outcomes']}")
    print(f"  Maximum probability: {stats['max_prob']:.4f} (outcome {stats['max_outcome']})")
    print(f"  Entropy: {stats['entropy']:.3f} bits")

    # Example observable: parity (XOR of bits)
    observable = {k: (int(k[0]) ^ int(k[1])) for k in counts.keys()}
    stats_obs = statistical_analysis(counts, observable=observable)
    print(f"  Observable 'parity' expectation: {stats_obs.get('expectation', 0):.4f}")
    print(f"  Observable variance: {stats_obs.get('variance', 0):.4f}")
    print()

def demo_comparison():
    """Compare two similar circuits via shot distributions."""
    print("=== 3. Distribution comparison ===")
    # Create two circuits that differ slightly
    qc1 = surface_code_plaquette(rows=2, cols=2)
    qc2 = surface_code_plaquette(rows=2, cols=2)
    # Modify qc2 by adding an extra H gate on qubit 0
    qc2.h(0)

    sample1 = sample_circuit(qc1, shots=500, seed=1)
    sample2 = sample_circuit(qc2, shots=500, seed=2)

    dist = compare_shots(sample1["counts"], sample2["counts"])
    print(f"Circuit A: {qc1.name}")
    print(f"Circuit B: {qc2.name} (with extra H on q0)")
    print(f"  Total variation distance: {dist['total_variation_distance']:.4f}")
    print(f"  Hellinger distance: {dist['hellinger_distance']:.4f}")
    print(f"  Fidelity: {dist['fidelity']:.4f}")
    print(f"  Chi‑squared: {dist['chi_squared']:.4f}")
    print()

def demo_shot_benchmark():
    """Run a full shot‑based benchmark."""
    print("=== 4. Comprehensive shot benchmark ===")
    qc = sacred_variational_ansatz(n_qubits=3, layers=2)
    bench = run_shot_benchmark(qc, shots=2000, seed=999)

    print(f"Circuit: {bench['circuit_name']}")
    print(f"  Qubits: {bench['n_qubits']}, shots: {bench['shots']}")
    print(f"  Depth: {qc.depth}, gate count: {qc.gate_count}")
    print(f"  Unique outcomes: {bench['unique_outcomes']}")
    print(f"  Entropy: {bench['entropy']:.3f} bits")
    print(f"  Per‑qubit ⟨Z⟩: {[f'{z:.3f}' for z in bench['expectation_z']]}")
    print(f"  Max probability outcome: {bench['max_outcome']} ({bench['max_prob']:.4f})")
    print()

def main():
    print("Demonstration of shot‑based sampling and statistical analysis")
    print("=" * 60)
    demo_basic_sampling()
    demo_statistical_analysis()
    demo_comparison()
    demo_shot_benchmark()
    print("All demonstrations completed successfully.")

if __name__ == "__main__":
    main()