#!/usr/bin/env python3
"""End-to-end test for quantum runtime bridge across all ASI subsystems."""
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def test_runtime_bridge():
    print("=== TEST 1: Runtime Bridge ===")
    from l104_quantum_runtime import get_runtime, ExecutionMode
    rt = get_runtime()
    status = rt.get_status()
    print(f"  Connected: {status.get('connected', False)}")
    print(f"  Mode: {status.get('execution_mode', 'unknown')}")
    print(f"  Backend: {status.get('default_backend', 'none')}")
    assert 'connected' in status, "Runtime status missing 'connected'"
    return True

def test_coherence_engine():
    print("\n=== TEST 2: Quantum Coherence Engine ===")
    from l104_quantum_coherence import QuantumCoherenceEngine
    e = QuantumCoherenceEngine()
    print(f"  Execution Mode: {e.execution_mode}")
    print(f"  Has _runtime: {e._runtime is not None}")
    print(f"  Use real QPU: {e._use_real_qpu}")
    assert hasattr(e, '_execute_circuit'), "Missing _execute_circuit method"
    assert hasattr(e, 'execution_mode'), "Missing execution_mode property"
    return True

def test_grover():
    print("\n=== TEST 3: Grover Search ===")
    from l104_quantum_coherence import QuantumCoherenceEngine
    e = QuantumCoherenceEngine()
    r = e.grover_search(target_index=3, search_space_qubits=3)
    print(f"  Algorithm: {r.get('algorithm')}")
    print(f"  Success: {r.get('success')}")
    print(f"  Found: {r.get('found_index')}")
    exec_info = r.get('execution', {})
    print(f"  Exec Mode: {exec_info.get('mode', 'unknown')}")
    print(f"  Backend: {exec_info.get('backend', 'unknown')}")
    assert r.get('algorithm') == 'grover_search', "Not a Grover result"
    assert 'execution' in r, "Grover result missing 'execution' metadata"
    return True

def test_asi_quantum():
    print("\n=== TEST 4: ASI Quantum Core ===")
    from l104_asi.quantum import QuantumComputationCore
    qc = QuantumComputationCore()
    qs = qc.status()
    print(f"  Version: {qs.get('version')}")
    print(f"  Real QPU: {qs.get('real_qpu_enabled')}")
    print(f"  Runtime Bridge: {'connected' if qs.get('runtime_bridge') else 'none'}")
    assert qs.get('version', '').startswith('7'), f"Expected v7.x, got {qs.get('version')}"
    return True

def test_code_engine_quantum():
    print("\n=== TEST 5: Code Engine Quantum ===")
    from l104_code_engine.quantum import QuantumCodeIntelligenceCore
    qci = QuantumCodeIntelligenceCore()
    qs = qci.status()
    print(f"  Version: {qs.get('version')}")
    print(f"  Real QPU: {qs.get('real_qpu_enabled', 'N/A')}")
    assert 'real_qpu_enabled' in qs, "Missing real_qpu_enabled in code engine quantum status"
    return True

if __name__ == "__main__":
    passed = 0
    failed = 0
    tests = [test_runtime_bridge, test_coherence_engine, test_grover, test_asi_quantum, test_code_engine_quantum]
    for t in tests:
        try:
            t()
            passed += 1
            print("  ✓ PASSED")
        except Exception as ex:
            failed += 1
            print(f"  ✗ FAILED: {ex}")
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed == 0:
        print("=== ALL TESTS PASSED ===")
    sys.exit(failed)
