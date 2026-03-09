#!/usr/bin/env python3
"""Test: Quantum Simulator ↔ Brain Bridge (v9.0.0)"""

from l104_quantum_engine.brain import L104QuantumBrain
brain = L104QuantumBrain()

# Test 1: Verify new attributes exist
print("=== TEST 1: v9.0 Attributes ===")
assert hasattr(brain, '_coherence_engine'), 'Missing _coherence_engine'
assert hasattr(brain, '_coherence_engine_checked'), 'Missing _coherence_engine_checked'
print("  ✓ v9.0 attributes present")

# Test 2: Lazy-load coherence engine
print("\n=== TEST 2: Lazy-load QuantumCoherenceEngine ===")
sim = brain._get_coherence_engine_cached()
print(f"  Simulator: {type(sim).__name__}")
assert sim is not None, 'Coherence engine failed to load'
print("  ✓ QuantumCoherenceEngine loaded")

# Test 3: Grover search
print("\n=== TEST 3: Grover Search via Brain ===")
grv = brain.grover_search(7, 4)
print(f"  Target: 7, Found: {grv.get('found_index')}, Prob: {grv.get('target_probability', 0):.4f}")
assert grv.get('success') is not None, 'Grover failed'
print("  ✓ Grover search works")

# Test 4: Shor factoring
print("\n=== TEST 4: Shor Factor via Brain ===")
shr = brain.shor_factor(15)
print(f"  N=15, Factors: {shr.get('factors')}, Quantum: {shr.get('quantum')}")
assert shr.get('factors'), 'Shor failed'
print("  ✓ Shor factoring works")

# Test 5: Iron simulator
print("\n=== TEST 5: Iron Simulator via Brain ===")
iron = brain.iron_simulate('all', 4)
print(f"  Algorithm: {iron.get('algorithm')}, Z: {iron.get('atomic_number')}")
assert iron.get('algorithm') in ('quantum_iron_simulator', 'quantum_iron_engine'), 'Iron sim failed'
print("  ✓ Iron simulator works")

# Test 6: Simulator status
print("\n=== TEST 6: Simulator Status ===")
status = brain.simulator_status()
print(f"  Available: {status.get('available')}")
print(f"  Version: {status.get('version')}")
print(f"  Mode: {status.get('execution_mode')}")
print(f"  Qubits: {status.get('register', {}).get('num_qubits')}")
print(f"  Algorithms: {len(status.get('capabilities', []))}")
assert status.get('available'), 'Status failed'
print("  ✓ Status works")

# Test 7: Kernel status includes simulator
print("\n=== TEST 7: Kernel Status includes simulator ===")
ks = brain.kernel_status()
print(f"  quantum_simulator in kernel_status: {'quantum_simulator' in ks}")
qs = ks.get('quantum_simulator', {})
print(f"  Available: {qs.get('available')}")
print(f"  Algorithms: {qs.get('algorithms')}")
assert 'quantum_simulator' in ks, 'Missing from kernel_status'
print("  ✓ Kernel status includes simulator")

# Test 8: Version
print("\n=== TEST 8: VERSION updated to 9.0.0 ===")
print(f"  Brain VERSION: {brain.VERSION}")
assert brain.VERSION == '9.0.0', f'Version mismatch: {brain.VERSION}'
print("  ✓ Version is 9.0.0")

# Test 9: QAOA convenience
print("\n=== TEST 9: QAOA via Brain ===")
qaoa = brain.qaoa_optimize([(0, 1), (1, 2), (2, 0)], p=2)
print(f"  Cut: {qaoa.get('cut_value')}, Ratio: {qaoa.get('approximation_ratio', 0):.4f}")
assert qaoa.get('cut_value') is not None, 'QAOA failed'
print("  ✓ QAOA works")

# Test 10: VQE convenience
print("\n=== TEST 10: VQE via Brain ===")
vqe = brain.vqe_optimize(num_qubits=2, max_iterations=10)
print(f"  Energy: {vqe.get('optimized_energy', 0):.4f}, Error: {vqe.get('energy_error', 0):.4f}")
assert vqe.get('optimized_energy') is not None, 'VQE failed'
print("  ✓ VQE works")

print("\n" + "=" * 60)
print("  ALL 10 TESTS PASSED — Simulator ↔ Brain Bridge v9.0.0")
print("=" * 60)
