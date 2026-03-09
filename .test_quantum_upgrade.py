#!/usr/bin/env python3
"""L104 Quantum Upgrade — Comprehensive smoke test for all upgraded modules."""
import os
import sys

# Workaround for OpenMP duplicate library issue (libiomp5 vs libomp)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("=" * 70)
print("L104 QUANTUM UPGRADE — COMPREHENSIVE IMPORT + SMOKE TEST")
print("=" * 70)

passed = 0
failed = 0

# ═══ TEST 1: l104_qiskit_utils imports ═══
print("\n[1/6] l104_qiskit_utils.py — import + factories")
try:
    from l104_qiskit_utils import (
        L104NoiseModelFactory, L104CircuitFactory, L104AerBackend,
        L104ErrorMitigation, L104ObservableFactory, L104Transpiler,
        aer_backend, aer_backend_noisy, circuit_factory,
        quick_statevector, quick_shots, build_vqe, build_qaoa, build_grover,
    )
    print(f"  NoiseModelFactory profiles: {list(L104NoiseModelFactory.PROFILES.keys())}")
    print(f"  AerBackend ideal: {aer_backend is not None}")
    print(f"  AerBackend noisy: {aer_backend_noisy is not None}")
    from l104_quantum_gate_engine.quantum_info import ParameterVector
    vqe_circ, vqe_pv = L104CircuitFactory.vqe_ansatz(3, depth=3)
    print(f"  VQE ansatz (3q, depth=3): {vqe_circ.num_qubits} qubits, {vqe_circ.depth()} depth, {len(vqe_pv)} params")
    sv = quick_statevector(vqe_circ.assign_parameters({vqe_pv[i]: 0.5 for i in range(len(vqe_pv))}))
    print(f"  Statevector result: shape={sv.shape}, norm={abs(sum(abs(x)**2 for x in sv)):.4f}")
    print("  PASS")
    passed += 1
except Exception as e:
    print(f"  FAIL: {e}")
    failed += 1

# ═══ TEST 2: l104_quantum_runtime imports ═══
print("\n[2/6] l104_quantum_runtime.py — v2.0.0 import")
try:
    from l104_quantum_runtime import get_runtime, ExecutionMode
    rt = get_runtime()
    status = rt.get_status()
    print(f"  Version: {status.get('version', '?')}")
    print(f"  L104 Utils: {status.get('l104_utils_available', False)}")
    print(f"  EstimatorV2: {status.get('estimator_v2_enabled', False)}")
    caps = status.get("capabilities", {})
    if isinstance(caps, dict):
        print(f"  Capabilities: {list(caps.keys())}")
    print("  PASS")
    passed += 1
except Exception as e:
    print(f"  FAIL: {e}")
    failed += 1

# ═══ TEST 3: l104_asi/quantum.py — v9.0.0 ═══
print("\n[3/6] l104_asi/quantum.py — v9.0.0 import")
try:
    from l104_asi.quantum import QuantumComputationCore
    qc = QuantumComputationCore()
    qs = qc.status()
    print(f"  Version: {qs.get('version', '?')}")
    qu = qs.get("qiskit_upgrade", {})
    print(f"  Upgrade caps: {list(qu.get('capabilities', {}).keys())}")
    print("  PASS")
    passed += 1
except Exception as e:
    print(f"  FAIL: {e}")
    failed += 1

# ═══ TEST 4: l104_quantum_coherence.py — v5.0.0 ═══
print("\n[4/6] l104_quantum_coherence.py — v5.0.0 import")
try:
    from l104_quantum_coherence import QuantumCoherenceEngine
    engine = QuantumCoherenceEngine()
    # Force local simulator to avoid QPU usage-limit hang
    engine.set_real_qpu(False)
    es = engine.get_status()
    print(f"  Version: {es.get('version', '?')}")
    eu = es.get("qiskit_upgrade", {})
    print(f"  L104 Utils: {eu.get('l104_utils_available', False)}")
    print(f"  Aer Noise: {eu.get('aer_noise_available', False)}")
    print(f"  Upgrade caps: {list(eu.get('capabilities', {}).keys())}")
    print("  PASS")
    passed += 1
except Exception as e:
    print(f"  FAIL: {e}")
    failed += 1

# ═══ TEST 5: Grover search (uses L104CircuitFactory) ═══
print("\n[5/6] Grover search (L104CircuitFactory path)")
try:
    result = engine.grover_search(target_index=5, search_space_qubits=4)
    print(f"  Target: {result['target_index']}, Found: {result['found_index']}")
    print(f"  Success: {result['success']}")
    print(f"  Target prob: {result['target_probability']:.4f}")
    assert result["success"], "Grover failed to find target"
    print("  PASS")
    passed += 1
except Exception as e:
    print(f"  FAIL: {e}")
    failed += 1

# ═══ TEST 6: QAOA + VQE + Kernel smoke ═══
print("\n[6/6] QAOA + VQE + Kernel smoke tests")
try:
    qaoa = engine.qaoa_maxcut([(0, 1), (1, 2), (2, 3), (3, 0)], p=1)
    print(f"  QAOA cut: {qaoa['cut_value']}/{qaoa['max_possible_cut']} ratio={qaoa['approximation_ratio']}")

    vqe = engine.vqe_optimize(num_qubits=3, max_iterations=10)
    print(f"  VQE energy error: {vqe['energy_error']:.4f}")

    kern = engine.quantum_kernel([1.0, 0.5, 0.3], [1.0, 0.5, 0.3])
    print(f"  Kernel self-similarity: {kern['kernel_value']:.4f}")
    print("  PASS")
    passed += 1
except Exception as e:
    print(f"  FAIL: {e}")
    failed += 1

print("\n" + "=" * 70)
if failed == 0:
    print(f"ALL {passed} TESTS PASSED — QUANTUM UPGRADE VERIFIED")
else:
    print(f"{passed} PASSED / {failed} FAILED")
print("=" * 70)
sys.exit(1 if failed else 0)
