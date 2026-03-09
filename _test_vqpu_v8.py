#!/usr/bin/env python3
"""VQPU v8.0 Validation Test Suite."""
import sys
import numpy as np

passed = 0
failed = 0

def test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  \u2713 {name}")
        passed += 1
    except Exception as e:
        print(f"  \u2717 {name}: {e}")
        failed += 1

print("\u2550" * 60)
print("  VQPU v8.0 VALIDATION SUITE")
print("\u2550" * 60)

# -- 1. Imports --
print("\n-- Imports --")
from l104_vqpu import (
    VQPUBridge,
    QuantumInformationMetrics,
    QuantumStateTomography,
    HamiltonianSimulator,
    ScoringCache,
    ThreeEngineQuantumScorer,
    EngineIntegration,
    CircuitTranspiler,
)
print("  \u2713 All v8.0 classes imported")
passed += 1

# -- 2. ScoringCache --
print("\n-- ScoringCache --")
test("stats()", lambda: ScoringCache.stats())
test("clear()", lambda: ScoringCache.clear())

def _test_harmonic_cache():
    ScoringCache.clear()
    v1 = ScoringCache.get_harmonic(lambda: 0.85)
    v2 = ScoringCache.get_harmonic(lambda: 999)  # should return cached 0.85
    assert v1 == v2 == 0.85, f"Cache miss: {v1} != {v2}"
    s = ScoringCache.stats()
    assert s["harmonic_cached"], "Harmonic should be cached"
test("harmonic caching", _test_harmonic_cache)

def _test_wave_cache():
    v1 = ScoringCache.get_wave(lambda: 0.72)
    v2 = ScoringCache.get_wave(lambda: 999)
    assert v1 == v2 == 0.72
    s = ScoringCache.stats()
    assert s["wave_cached"], "Wave should be cached"
test("wave caching", _test_wave_cache)
ScoringCache.clear()

# -- 3. Quantum Information Metrics --
print("\n-- Quantum Information Metrics --")
bell_sv = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
product_sv = np.array([1, 0, 0, 0], dtype=complex)

def _test_mutual_info():
    result = QuantumInformationMetrics.quantum_mutual_information(bell_sv, 2)
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    mi = result["mutual_information"]
    assert mi > 0, f"MI should be > 0 for Bell state, got {mi}"
test("mutual information", _test_mutual_info)

def _test_qfi():
    gen = [{"gate": "Rz", "qubits": [0], "parameters": [1.0]}]
    result = QuantumInformationMetrics.quantum_fisher_information(bell_sv, gen, 2)
    assert isinstance(result, dict), f"QFI should return dict"
    assert "qfi" in result
    assert isinstance(result["qfi"], float)
test("quantum Fisher information", _test_qfi)

def _test_topo():
    sv4 = np.zeros(16, dtype=complex)
    sv4[0] = 1/np.sqrt(2)
    sv4[15] = 1/np.sqrt(2)
    t = QuantumInformationMetrics.topological_entanglement_entropy(sv4, 4)
    assert "topological_entropy" in t, f"Keys: {list(t.keys())}"
    assert "quantum_dimension_estimate" in t
test("topological entanglement entropy", _test_topo)

def _test_berry():
    svs = [bell_sv, product_sv, bell_sv]
    bp = QuantumInformationMetrics.berry_phase(svs, 2)
    assert isinstance(bp, dict)
    assert "berry_phase" in bp
test("Berry phase", _test_berry)

def _test_relative_entropy():
    re = QuantumInformationMetrics.quantum_relative_entropy(bell_sv, product_sv, 2)
    assert "relative_entropy" in re
    assert "fidelity" in re
    assert "trace_distance" in re
test("relative entropy + fidelity", _test_relative_entropy)

def _test_loschmidt():
    ham_ops = [{"gate": "Z", "qubits": [0], "parameters": [0.1]}]
    pert_ops = [{"gate": "X", "qubits": [0], "parameters": [0.05]}]
    le = QuantumInformationMetrics.loschmidt_echo(bell_sv, ham_ops, pert_ops, 2)
    assert "echo_values" in le
    assert "decay_rate" in le
test("Loschmidt echo", _test_loschmidt)

def _test_full_metrics():
    sv4 = np.zeros(16, dtype=complex)
    sv4[0] = 1/np.sqrt(2)
    sv4[15] = 1/np.sqrt(2)
    fm = QuantumInformationMetrics.full_metrics(sv4, 4)
    assert "mutual_information" in fm
    assert isinstance(fm["mutual_information"], dict)
test("full_metrics()", _test_full_metrics)

# -- 4. Quantum State Tomography --
print("\n-- Quantum State Tomography --")

def _test_pauli_measure():
    pm = QuantumStateTomography.measure_in_pauli_bases(bell_sv, 2, shots=256)
    assert isinstance(pm, dict)
    assert len(pm) > 0
test("Pauli measurement", _test_pauli_measure)

def _test_reconstruct():
    pm = QuantumStateTomography.measure_in_pauli_bases(bell_sv, 2, shots=512)
    recon = QuantumStateTomography.reconstruct_density_matrix(pm, 2)
    assert "purity" in recon, f"Missing 'purity' in {list(recon.keys())}"
test("density matrix reconstruction", _test_reconstruct)

def _test_fidelity():
    f = QuantumStateTomography.state_fidelity(bell_sv, bell_sv, 2)
    assert "fidelity" in f
    assert abs(f["fidelity"] - 1.0) < 1e-6, f"Self-fidelity should be 1, got {f['fidelity']}"
test("state fidelity", _test_fidelity)

def _test_swap_circuit():
    ops = QuantumStateTomography.swap_test_circuit(2)
    assert len(ops) > 0
test("SWAP test circuit", _test_swap_circuit)

def _test_full_tomo():
    ft = QuantumStateTomography.full_tomography(bell_sv, 2, shots=256)
    assert "purity" in ft
    assert "rank" in ft
test("full_tomography()", _test_full_tomo)

# -- 5. Hamiltonian Simulator --
# NOTE: hamiltonian_terms format is (coefficient, pauli_string)
print("\n-- Hamiltonian Simulator --")

def _test_trotter():
    ham = [(0.5, "ZZ"), (0.1, "ZI")]
    evo = HamiltonianSimulator.trotter_evolution(ham, 2, total_time=1.0, trotter_steps=4)
    assert "error" not in evo, f"Trotter failed: {evo.get('error')}"
    assert evo["num_qubits"] == 2
test("Trotter-Suzuki evolution", _test_trotter)

def _test_trotter_2nd_order():
    ham = [(0.5, "ZZ"), (0.2, "XI")]
    evo = HamiltonianSimulator.trotter_evolution(ham, 2, total_time=0.5, trotter_steps=2, order=2)
    assert "error" not in evo, f"Trotter-2 failed: {evo.get('error')}"
    assert evo["trotter_order"] == 2
test("2nd-order Trotter", _test_trotter_2nd_order)

def _test_adiabatic():
    target_ham = [(1.0, "ZI"), (0.5, "ZZ")]
    ad = HamiltonianSimulator.adiabatic_preparation(target_ham, 2, adiabatic_steps=5)
    assert "error" not in ad, f"Adiabatic failed: {ad.get('error')}"
    assert ad["num_qubits"] == 2
test("adiabatic preparation", _test_adiabatic)

def _test_iron_lattice():
    fe = HamiltonianSimulator.iron_lattice_circuit(4, trotter_steps=3)
    assert "error" not in fe, f"Iron lattice failed: {fe.get('error')}"
    assert fe["lattice_sites"] == 4
    assert "magnetization" in fe
    assert "zz_correlations" in fe
    assert abs(fe["coupling_j"] - 527.5184818492612 / 1000) < 1e-10
test("iron lattice Fe(26)", _test_iron_lattice)

# -- 6. VQPUBridge Integration --
print("\n-- VQPUBridge Methods --")

with VQPUBridge() as bridge:
    def _test_bridge_tomo():
        job = bridge.bell_pair(shots=256)
        r = bridge.run_tomography(job, shots=256)
        assert "purity" in r
    test("bridge.run_tomography()", _test_bridge_tomo)

    def _test_bridge_qi():
        job = bridge.ghz_state(3)
        r = bridge.quantum_information_metrics(job)
        assert "information_metrics" in r or "error" not in r
    test("bridge.quantum_information_metrics()", _test_bridge_qi)

    def _test_bridge_fidelity():
        a = bridge.bell_pair()
        b = bridge.bell_pair()
        r = bridge.quantum_fidelity(a, b)
        assert "fidelity" in r
    test("bridge.quantum_fidelity()", _test_bridge_fidelity)

    def _test_bridge_swap():
        a = bridge.bell_pair(shots=256)
        b = bridge.bell_pair(shots=256)
        r = bridge.swap_test(a, b, shots=512)
        assert "estimated_fidelity" in r, f"Missing estimated_fidelity, keys: {list(r.keys())}"
    test("bridge.swap_test()", _test_bridge_swap)

    def _test_bridge_ham():
        r = bridge.run_hamiltonian_evolution(
            [(0.5, "ZZ"), (0.1, "ZI")],
            num_qubits=2, total_time=1.0, trotter_steps=4,
        )
        assert "error" not in r, f"Failed: {r.get('error')}"
    test("bridge.run_hamiltonian_evolution()", _test_bridge_ham)

    def _test_bridge_adiabatic():
        r = bridge.run_adiabatic_preparation(
            [(1.0, "ZI")], num_qubits=2, adiabatic_steps=5,
        )
        assert "error" not in r, f"Failed: {r.get('error')}"
    test("bridge.run_adiabatic_preparation()", _test_bridge_adiabatic)

    def _test_bridge_iron():
        r = bridge.run_iron_lattice(n_sites=4, trotter_steps=3)
        assert "magnetization" in r
    test("bridge.run_iron_lattice()", _test_bridge_iron)

    def _test_bridge_cache_stats():
        r = bridge.scoring_cache_stats()
        assert "harmonic_cached" in r
    test("bridge.scoring_cache_stats()", _test_bridge_cache_stats)

    def _test_bridge_cache_clear():
        bridge.clear_scoring_cache()
        s = bridge.scoring_cache_stats()
        assert not s["harmonic_cached"]
    test("bridge.clear_scoring_cache()", _test_bridge_cache_clear)

    def _test_bridge_status():
        s = bridge.status()
        assert s["version"] == "8.0.0", f"Version should be 8.0.0, got {s['version']}"
    test("bridge.status() version 8.0.0", _test_bridge_status)

# -- Summary --
print()
print("\u2550" * 60)
total = passed + failed
print(f"  VQPU v8.0 VALIDATION: {passed}/{total} passed, {failed} failed")
if failed == 0:
    print("  ALL TESTS PASSED")
else:
    print("  SOME TESTS FAILED")
print("\u2550" * 60)

sys.exit(0 if failed == 0 else 1)
