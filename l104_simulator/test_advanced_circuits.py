#!/usr/bin/env python3
"""
Quick test for the advanced circuit library.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from l104_simulator.advanced_circuits import (
    surface_code_plaquette,
    toric_code_plaquette,
    color_code_hexagonal,
    repetition_code,
    shor_code_full,
    hardware_efficient_ansatz,
    sacred_variational_ansatz,
    qft_circuit,
    ErrorCorrectingCode,
    VariationalAnsatzLibrary,
    run_and_benchmark,
    sample_circuit,
    statistical_analysis,
    compare_shots,
    run_shot_benchmark,
)

from l104_simulator.simulator import Simulator

def test_circuit_creation():
    """Test that each circuit can be created without error."""
    circuits = []

    # Surface code
    qc1 = surface_code_plaquette(rows=2, cols=2)
    circuits.append(("surface_code_2x2", qc1))

    # Toric code
    qc_toric = toric_code_plaquette(L=2)
    circuits.append(("toric_code_2", qc_toric))

    # Color code
    qc_color = color_code_hexagonal(distance=3)
    circuits.append(("color_code_3", qc_color))

    # Repetition code
    qc2 = repetition_code(n_physical=5)
    circuits.append(("repetition_code_5", qc2))

    # Shor code
    qc3 = shor_code_full()
    circuits.append(("shor_code_full", qc3))

    # Hardware efficient ansatz
    qc4 = hardware_efficient_ansatz(n_qubits=4, layers=2, entanglement="linear")
    circuits.append(("hea_4q", qc4))

    # Sacred variational ansatz
    qc5 = sacred_variational_ansatz(n_qubits=3, layers=1)
    circuits.append(("sacred_var_3q", qc5))

    # QFT
    qc6 = qft_circuit(n_qubits=3, inverse=False)
    circuits.append(("qft_3", qc6))

    # Inverse QFT
    qc7 = qft_circuit(n_qubits=3, inverse=True)
    circuits.append(("iqft_3", qc7))

    # ErrorCorrectingCode class
    qc8 = ErrorCorrectingCode.bit_flip(n_physical=3)
    circuits.append(("bit_flip_3", qc8))

    qc9 = ErrorCorrectingCode.phase_flip(n_physical=3)
    circuits.append(("phase_flip_3", qc9))

    qc10 = ErrorCorrectingCode.steane()
    circuits.append(("steane_7", qc10))

    # VariationalAnsatzLibrary
    qc11 = VariationalAnsatzLibrary.hea(n_qubits=5, layers=2)
    circuits.append(("hea_5q", qc11))

    qc12 = VariationalAnsatzLibrary.sacred(n_qubits=2, layers=1)
    circuits.append(("sacred_2q", qc12))

    print("✓ Created", len(circuits), "circuits")

    # Quick simulation (just run each circuit)
    sim = Simulator()
    for name, qc in circuits:
        try:
            result = sim.run(qc)
            print(f"  {name}: {qc.n_qubits} qubits, depth {qc.depth}, gates {qc.gate_count} — OK")
        except Exception as e:
            print(f"  {name}: ERROR {e}")
            return False

    return True

def test_run_and_benchmark():
    """Test the convenience runner."""
    qc = surface_code_plaquette(rows=2, cols=2)
    metrics = run_and_benchmark(qc, shots=100)
    assert "circuit_name" in metrics
    assert "n_qubits" in metrics
    assert "depth" in metrics
    print("✓ run_and_benchmark works")
    return True

def test_shot_based_utilities():
    """Test the shot‑based sampling and analysis functions."""
    qc = surface_code_plaquette(rows=2, cols=2)

    # 1. sample_circuit
    sample = sample_circuit(qc, shots=500, seed=42)
    assert "counts" in sample
    assert "empirical_probabilities" in sample
    assert "expectation_z" in sample
    assert "variance_z" in sample
    assert sample["shots"] == 500
    assert sample["n_qubits"] == qc.n_qubits
    total = sum(sample["counts"].values())
    assert total == 500, f"Total counts {total} != 500"
    # Expectation Z should be list length n_qubits
    assert len(sample["expectation_z"]) == qc.n_qubits
    print("✓ sample_circuit works")

    # 2. statistical_analysis
    stats = statistical_analysis(sample["counts"])
    assert "total_shots" in stats
    assert "unique_outcomes" in stats
    assert "entropy" in stats
    assert stats["total_shots"] == 500
    # Entropy non‑negative
    assert stats["entropy"] >= 0.0
    print("✓ statistical_analysis works")

    # 3. compare_shots (compare same distribution should give zero distance)
    dist = compare_shots(sample["counts"], sample["counts"])
    assert dist["total_variation_distance"] < 1e-10
    assert abs(dist["hellinger_distance"]) < 1e-10
    assert abs(dist["fidelity"] - 1.0) < 1e-10
    print("✓ compare_shots works")

    # 4. run_shot_benchmark
    bench = run_shot_benchmark(qc, shots=300, seed=123)
    assert "circuit_name" in bench
    assert "counts" in bench
    assert "expectation_z" in bench
    assert "entropy" in bench
    assert bench["shots"] == 300
    print("✓ run_shot_benchmark works")

    return True


if __name__ == "__main__":
    print("Testing advanced circuit library...")
    if test_circuit_creation():
        print("Circuit creation and simulation PASSED")
    else:
        print("Circuit creation and simulation FAILED")
        sys.exit(1)

    if test_run_and_benchmark():
        print("Benchmark runner PASSED")
    else:
        print("Benchmark runner FAILED")
        sys.exit(1)

    if test_shot_based_utilities():
        print("Shot‑based utilities PASSED")
    else:
        print("Shot‑based utilities FAILED")
        sys.exit(1)

    print("\nAll tests passed!")