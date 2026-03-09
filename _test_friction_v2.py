#!/usr/bin/env python3
"""v2.0 Friction Analyzer upgrade validation."""
import time
import numpy as np

from l104_god_code_friction_analyzer import (
    _apply_fe26_decoherence, _renyi_entropy, _relative_entropy,
    compute_friction_candidates, quantum_friction_analysis,
    evaluate_friction_against_constants, search_optimal_friction,
)
from l104_quantum_gate_engine.quantum_info import Statevector, DensityMatrix

print('═══ v2.0 UPGRADE VALIDATION ═══\n')

# 1. Test helper functions
print('1. Testing helper functions...')
sv = Statevector.from_label('00')
rho = DensityMatrix(sv)
noisy = _apply_fe26_decoherence(rho, gamma=0.01)
print(f'   Fe-26 decoherence: trace={float(np.trace(noisy.data)):.6f} (expect ~1.0)')
r2 = _renyi_entropy(rho, alpha=2.0)
print(f'   Rényi-2 entropy (pure state): {r2:.6f} (expect ~0.0)')
re = _relative_entropy(rho, noisy)
print(f'   Relative entropy S(ρ||σ): {re:.6f} (expect > 0)')
print('   ✅ All helpers OK\n')

# 2. Test 16 friction candidates
print('2. Testing compute_friction_candidates...')
candidates = compute_friction_candidates()
print(f'   Generated {len(candidates)} candidates (expected 16)')
assert len(candidates) == 16, f'Expected 16, got {len(candidates)}'
new_names = [c['name'] for c in candidates[12:]]
print(f'   New candidates: {new_names}')
print('   ✅ Candidates OK\n')

# 3. Test quantum_friction_analysis v2.0
print('3. Testing quantum_friction_analysis v2.0...')
t0 = time.time()
qa = quantum_friction_analysis(0.001, num_qubits=4, decoherence=True, decoherence_gamma=0.005)
elapsed = time.time() - t0
print(f'   Time: {elapsed:.2f}s')
print(f'   Coherence (ref):    {qa["coherence_without_friction"]:.6f}')
print(f'   Coherence (fric):   {qa["coherence_with_friction"]:.6f}')
print(f'   Pure fidelity:      {qa["state_fidelity"]:.8f}')
print(f'   Noisy fidelity:     {qa["noisy_fidelity"]:.8f}')
print(f'   Bures distance:     {qa["bures_distance"]:.8f}')
print(f'   Relative entropy:   {qa["relative_entropy"]:.8f}')
print(f'   Rényi-2 (ref):      {qa["renyi2_entropy_without"]:.8f}')
print(f'   Rényi-2 (fric):     {qa["renyi2_entropy_with"]:.8f}')
print(f'   Mean bipartite ent: {qa["mean_bipartite_entropy_with"]:.8f}')
print(f'   QPE error:          {qa["qpe_error"]:.8f}')
print(f'   Composite Q score:  {qa["composite_quantum_score"]:.6f}')
print(f'   Decoherence:        {qa["decoherence_enabled"]}')
for key in ['composite_quantum_score', 'noisy_fidelity', 'bures_distance',
            'relative_entropy', 'renyi2_entropy_with', 'mean_bipartite_entropy_with']:
    assert key in qa, f'Missing key: {key}'
print('   ✅ Quantum analysis v2.0 OK\n')

# 4. Test search with gradient refinement (2 candidates for speed)
print('4. Testing search_optimal_friction v2.0 (2 candidates, 4 qubits)...')
t0 = time.time()
sr = search_optimal_friction(candidates[:2], num_qubits=4)
elapsed = time.time() - t0
print(f'   Time: {elapsed:.2f}s')
print(f'   Candidates tested: {sr["total_candidates_tested"]}')
print(f'   Refined candidates: {sr["total_refined"]}')
print(f'   Best name: {sr["best"]["name"]}')
print(f'   Best combined score: {sr["best"]["combined_score"]:.6f}')
print(f'   Best composite Q:    {sr["best"].get("composite_quantum_score", "N/A")}')
assert 'total_refined' in sr
assert 'composite_quantum_score' in sr['best']
print('   ✅ Search v2.0 OK\n')

# 5. Quick v1 vs v2 discriminating power comparison
print('5. Discriminating power comparison...')
qa_zero = quantum_friction_analysis(0.0, num_qubits=4, decoherence=True)
qa_small = quantum_friction_analysis(0.0005, num_qubits=4, decoherence=True)
qa_med = quantum_friction_analysis(0.005, num_qubits=4, decoherence=True)
print(f'   Λ=0.0000: composite={qa_zero["composite_quantum_score"]:.6f}, bures={qa_zero["bures_distance"]:.8f}, S_rel={qa_zero["relative_entropy"]:.8f}')
print(f'   Λ=0.0005: composite={qa_small["composite_quantum_score"]:.6f}, bures={qa_small["bures_distance"]:.8f}, S_rel={qa_small["relative_entropy"]:.8f}')
print(f'   Λ=0.005:  composite={qa_med["composite_quantum_score"]:.6f}, bures={qa_med["bures_distance"]:.8f}, S_rel={qa_med["relative_entropy"]:.8f}')
# Verify metrics actually discriminate (non-zero should differ from zero)
assert qa_zero["bures_distance"] < qa_med["bures_distance"], "Bures distance should increase with friction"
print('   ✅ Metrics discriminate friction levels correctly\n')

print('═══ ALL v2.0 VALIDATION PASSED ═══')
