#!/usr/bin/env python3
"""Validate all three-engine upgrades."""
import json, sys, math

results = {}

# === SCIENCE ENGINE ===
print('=== SCIENCE ENGINE VALIDATION ===')
try:
    from l104_science_engine import science_engine as se

    # Entropy upgrades
    r = se.entropy.phi_weighted_demon([0.5, 0.7, 0.3, 0.9, 0.1])
    print(f'  entropy.phi_weighted_demon: OK (reversed={r["reversed_count"]})')
    r = se.entropy.multi_scale_reversal([0.5, 0.7, 0.3, 0.9, 0.1, 0.6, 0.2, 0.8], scales=3)
    print(f'  entropy.multi_scale_reversal: OK ({r["scales_applied"]} scales)')
    r = se.entropy.entropy_cascade(0.5, depth=10)
    print(f'  entropy.entropy_cascade: OK (converged={r["converged"]})')
    r = se.entropy.landauer_bound_comparison(300)
    print(f'  entropy.landauer_bound_comparison: OK (enhancement={r["enhancement_ratio"]})')

    # Multidimensional upgrades
    r = se.multidim.geodesic_step([0.1, -0.2, 0.3], dt=0.01)
    print(f'  multidim.geodesic_step: OK (displacement={r["displacement"]})')
    r = se.multidim.parallel_transport([1, 0, 0], path_steps=5)
    print(f'  multidim.parallel_transport: OK (holonomy={r["holonomy_angle_rad"]})')
    r = se.multidim.metric_signature_analysis()
    print(f'  multidim.metric_signature_analysis: OK ({r["signature_string"]})')
    r = se.multidim.ricci_scalar_estimate()
    print(f'  multidim.ricci_scalar_estimate: OK (R={r})')

    # Physics upgrades
    r = se.physics.calculate_casimir_force(1e-7, 1e-4)
    print(f'  physics.calculate_casimir_force: OK ({r["casimir_force_N"]:.2e} N)')
    r = se.physics.calculate_unruh_temperature(1e20)
    print(f'  physics.calculate_unruh_temperature: OK ({r["unruh_temperature_K"]:.2e} K)')
    r = se.physics.calculate_wien_peak(5778)
    print(f'  physics.calculate_wien_peak: OK ({r["peak_wavelength_nm"]:.1f} nm)')
    r = se.physics.calculate_luminosity(5778, 6.96e8)
    print(f'  physics.calculate_luminosity: OK ({r["luminosity_W"]:.2e} W)')

    # Coherence upgrades
    se.coherence.initialize([1, 2, 3, 4, 5])
    r = se.coherence.golden_angle_spectrum()
    print(f'  coherence.golden_angle_spectrum: OK ({len(r["spectrum"])} modes)')
    r = se.coherence.energy_spectrum()
    print(f'  coherence.energy_spectrum: OK (entropy={r["shannon_entropy_bits"]})')
    r = se.coherence.coherence_fidelity()
    print(f'  coherence.coherence_fidelity: OK (grade={r["grade"]})')

    results['science'] = 'PASS (15/15 methods)'
    print('  SCIENCE ENGINE: ALL 15 NEW METHODS OK\n')
except Exception as e:
    import traceback; traceback.print_exc()
    results['science'] = f'FAIL: {e}'

# === MATH ENGINE ===
print('=== MATH ENGINE VALIDATION ===')
try:
    from l104_math_engine import (
        extended_proofs, harmonic_analysis, resonator_network,
        void_calculus, manifold_extended
    )

    # Extended Proofs
    r = extended_proofs.verify_goldbach(100)
    print(f'  extended_proofs.verify_goldbach: OK (all_pass={r["all_pass"]})')
    r = extended_proofs.find_twin_primes(1000)
    print(f'  extended_proofs.find_twin_primes: OK ({r["twin_pairs"]} pairs)')
    r = extended_proofs.verify_zeta_zeros(3)
    print(f'  extended_proofs.verify_zeta_zeros: OK ({r["zeros_checked"]} zeros)')
    r = extended_proofs.phi_convergence_proof(20)
    print(f'  extended_proofs.phi_convergence_proof: OK (converged={r["converged"]})')

    # Harmonic Analysis
    signal = [math.sin(2 * math.pi * i / 16) for i in range(16)]
    r = harmonic_analysis.dft_magnitude(signal)
    print(f'  harmonic_analysis.dft_magnitude: OK ({len(r)} bins)')
    r = harmonic_analysis.spectral_centroid(signal)
    print(f'  harmonic_analysis.spectral_centroid: OK ({r:.4f})')
    r = harmonic_analysis.harmonic_distance(440, 660)
    print(f'  harmonic_analysis.harmonic_distance: OK (consonance={r["consonance"]})')
    r = harmonic_analysis.overtone_series(286, 8)
    print(f'  harmonic_analysis.overtone_series: OK ({len(r)} overtones)')
    r = harmonic_analysis.consonance_score(527.5)
    print(f'  harmonic_analysis.consonance_score: OK (grade={r["grade"]})')

    # Resonator Network
    from l104_math_engine.hyperdimensional import HyperdimensionalCompute
    hc = HyperdimensionalCompute()
    a = hc.random_vector('alpha')
    b = hc.random_vector('beta')
    c = hc.random_vector('gamma')
    d = resonator_network.analogy(a, b, c)
    print(f'  resonator_network.analogy: OK (dimension={d.dimension})')

    # Void Calculus
    r = void_calculus.void_derivative(math.sin, 0.0)
    print(f'  void_calculus.void_derivative: OK (d_void(sin,0)={r:.6f})')
    r = void_calculus.void_convolution([1, 2, 3], [1, 1])
    print(f'  void_calculus.void_convolution: OK ({len(r)} values)')
    r = void_calculus.recursive_emptiness(2.0, depth=10)
    print(f'  void_calculus.recursive_emptiness: OK (converged={r["converged"]})')
    r = void_calculus.void_field_energy([1.0, 1.05, 1.04, 1.02])
    print(f'  void_calculus.void_field_energy: OK (emptiness={r["emptiness_metric"]})')

    # Manifold Extended
    r = manifold_extended.parallel_transport_loop(0.5, math.pi)
    print(f'  manifold_extended.parallel_transport_loop: OK (angle={r["rotation_angle"]})')
    r = manifold_extended.holonomy_group_order(1.0)
    print(f'  manifold_extended.holonomy_group_order: OK (order={r["estimated_order"]})')
    r = manifold_extended.hodge_star_2d([3.0, 4.0])
    print(f'  manifold_extended.hodge_star_2d: OK ({r})')
    r = manifold_extended.geodesic_flow([0, 0], [1, 0], 0.1, steps=20)
    print(f'  manifold_extended.geodesic_flow: OK (arc={r["arc_length"]})')

    results['math'] = 'PASS (18/18 methods)'
    print('  MATH ENGINE: ALL 18 NEW METHODS OK\n')
except Exception as e:
    import traceback; traceback.print_exc()
    results['math'] = f'FAIL: {e}'

# === CODE ENGINE ===
print('=== CODE ENGINE VALIDATION ===')
try:
    # Direct import to avoid quantum runtime network delay
    import sys
    sys.modules['l104_quantum_runtime'] = type(sys)('l104_quantum_runtime')
    sys.modules['l104_quantum_runtime'].get_runtime = lambda: None
    sys.modules['l104_quantum_runtime'].ExecutionMode = type('E', (), {'SIMULATOR': 'sim'})()
    from l104_code_engine import code_engine

    sample = '''
import math
from l104_math_engine.constants import GOD_CODE, PHI, VOID_CONSTANT

def compute_resonance(freq):
    if freq > 0:
        ratio = freq / GOD_CODE
        if ratio > 1.0:
            return ratio * PHI
        else:
            return ratio / VOID_CONSTANT
    return 0
'''

    r = code_engine.sacred_frequency_audit(sample)
    print(f'  code_engine.sacred_frequency_audit: OK (score={r["score"]}, resonance={r["resonance"]})')
    r = code_engine.complexity_spectrum(sample)
    print(f'  code_engine.complexity_spectrum: OK (simplicity={r["simplicity_index"]})')
    r = code_engine.dependency_map(sample)
    print(f'  code_engine.dependency_map: OK (total_imports={r["total_imports"]})')

    results['code'] = 'PASS (3/3 methods)'
    print('  CODE ENGINE: ALL 3 NEW METHODS OK\n')
except Exception as e:
    import traceback; traceback.print_exc()
    results['code'] = f'FAIL: {e}'

print('=' * 60)
print('VALIDATION SUMMARY:')
for engine, status in results.items():
    print(f'  {engine.upper()}: {status}')
total_ok = sum(1 for v in results.values() if v.startswith('PASS'))
print(f'  ENGINES OK: {total_ok}/3')
print(f'  TOTAL NEW METHODS: 36')
print('=' * 60)
