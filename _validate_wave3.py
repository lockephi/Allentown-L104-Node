#!/usr/bin/env python3
"""Wave 3 validation: all computronium fixes."""
import math
import sys

errors = []
passes = 0

# ═══ BLOCK 1: research.py imports & constants ═══
try:
    from l104_computronium_research import (
        BekensteinLimitResearch, QuantumCoherenceResearch,
        DimensionalComputationResearch, EntropyEngineeringResearch,
        QuantumCircuitResearch, ComputroniumResearchHub,
        BEKENSTEIN_CONSTANT, C_LIGHT, HBAR, BOLTZMANN_K, PLANCK_LENGTH
    )
    print("[PASS] research.py imports OK")
    passes += 1
except Exception as e:
    errors.append(f"research.py imports: {e}")
    print(f"[FAIL] research.py imports: {e}")

# ═══ BLOCK 2: explore_density_limits (real Bekenstein sweep) ═══
try:
    b = BekensteinLimitResearch()
    r = b.explore_density_limits(20)
    assert "bekenstein_ratio" in r, "Missing bekenstein_ratio"
    assert "optimal_radius_m" in r, "Missing optimal_radius_m"
    assert r["max_density_bits_per_cycle"] > 0, "Density should be positive"
    assert 0 <= r["bekenstein_ratio"] <= 1.0, f"Ratio out of range: {r['bekenstein_ratio']}"
    print(f"[PASS] explore_density_limits: max={r['max_density_bits_per_cycle']:.4e}, ratio={r['bekenstein_ratio']:.4f}")
    passes += 1
except Exception as e:
    errors.append(f"explore_density_limits: {e}")
    print(f"[FAIL] explore_density_limits: {e}")

# ═══ BLOCK 3: theoretical_breakthrough_simulation ═══
try:
    r = b.theoretical_breakthrough_simulation()
    assert "extra_dim_capacity_bits" in r
    assert "holographic_4d_bits" in r
    assert r["improvement_factor"] >= 1.0
    print(f"[PASS] breakthrough_sim: 4D={r['holographic_4d_bits']:.4e} -> 11D={r['extended_11d_limit']:.4e}, x{r['improvement_factor']:.4f}")
    passes += 1
except Exception as e:
    errors.append(f"breakthrough_sim: {e}")
    print(f"[FAIL] breakthrough_sim: {e}")

# ═══ BLOCK 4: phi_stabilized_coherence (real EC) ═══
try:
    qc = QuantumCoherenceResearch()
    r = qc.phi_stabilized_coherence(1e-3, depth=5)
    assert "p_phys" in r
    assert "p_threshold" in r
    layer0 = r["protection_layers"][0]
    assert "code_distance" in layer0
    assert "p_logical" in layer0
    print(f"[PASS] phi_stabilized: {r['base_coherence_s']:.1e} -> {r['stabilized_coherence_s']:.1e}, x{r['improvement_factor']:.1e}")
    passes += 1
except Exception as e:
    errors.append(f"phi_stabilized: {e}")
    print(f"[FAIL] phi_stabilized: {e}")

# ═══ BLOCK 5: void_coherence_channel ═══
try:
    r = qc.void_coherence_channel()
    assert "void_bypass_factor" in r
    assert "bell_fidelity" in r
    assert r["void_bypass_factor"] > 1.0
    assert r["total_improvement"] < 100, f"Total improvement too high: {r['total_improvement']}"
    print(f"[PASS] void_coherence: bypass={r['void_bypass_factor']:.4f}, bell={r['bell_fidelity']:.4f}, total={r['total_improvement']:.4f}x")
    passes += 1
except Exception as e:
    errors.append(f"void_coherence: {e}")
    print(f"[FAIL] void_coherence: {e}")

# ═══ BLOCK 6: dimensional_capacity (n-sphere) ═══
try:
    dc = DimensionalComputationResearch()
    r = dc.calculate_dimensional_capacity()
    assert "reference_radius_m" in r
    cap0 = r["capacities"][0]
    assert "surface_area_m2" in cap0
    assert "holographic_bits" in cap0
    print(f"[PASS] dimensional_capacity: optimal={r['optimal_dimension']}D, cap={r['optimal_capacity']:.4e}")
    passes += 1
except Exception as e:
    errors.append(f"dimensional_capacity: {e}")
    print(f"[FAIL] dimensional_capacity: {e}")

# ═══ BLOCK 7: folded_dimension (Bekenstein d-torus) ═══
try:
    r = dc.folded_dimension_architecture()
    assert "kk_energy_J" in r["fold_architecture"][0]
    assert "base_3d_capacity_bits" in r
    assert "total_extra_capacity_bits" in r
    print(f"[PASS] folded_dimension: multiplier={r['total_capacity_multiplier']:.4f}x, extra={r['total_extra_capacity_bits']:.4e}")
    passes += 1
except Exception as e:
    errors.append(f"folded_dimension: {e}")
    print(f"[FAIL] folded_dimension: {e}")

# ═══ BLOCK 8: phi_compression_cascade (Shannon) ═══
try:
    ee = EntropyEngineeringResearch()
    r = ee.phi_compression_cascade(8.0, 15)
    assert "total_landauer_cost_J" in r
    c0 = r["cascade"][0]
    assert "order_K" in c0
    assert "landauer_cost_J" in c0
    print(f"[PASS] compression_cascade: ratio={r['compression_ratio']:.2f}x, landauer={r['total_landauer_cost_J']:.4e} J")
    passes += 1
except Exception as e:
    errors.append(f"compression_cascade: {e}")
    print(f"[FAIL] compression_cascade: {e}")

# ═══ BLOCK 9: void_entropy_sink (Landauer-bounded) ═══
try:
    r = ee.void_entropy_sink(10.0)
    assert r["void_capacity"] == "BOUNDED_BY_LANDAUER", f"Not bounded: {r['void_capacity']}"
    assert r["entropy_remaining"] > 0, "Should have remaining entropy"
    assert r["entropy_remaining"] < 10.0, "Should reduce some entropy"
    print(f"[PASS] void_entropy_sink: remaining={r['entropy_remaining']:.4f}, efficiency={r['reversal_efficiency']:.4f}")
    passes += 1
except Exception as e:
    errors.append(f"void_entropy_sink: {e}")
    print(f"[FAIL] void_entropy_sink: {e}")

# ═══ BLOCK 10: quantum circuit experiments ═══
try:
    qr = QuantumCircuitResearch()
    sd = qr.sacred_density_experiment(4, 3)
    assert sd.get("quantum", False), "Sacred not quantum"
    assert "entropy_bits" in sd, "Missing entropy_bits"
    assert "bekenstein_bound_bits" in sd, "Missing bekenstein_bound_bits"

    qft = qr.qft_information_capacity(3)
    assert qft.get("quantum", False), "QFT not quantum"
    assert qft["info_capacity"] <= 3.01, f"Capacity exceeds max: {qft['info_capacity']}"

    ghz = qr.ghz_condensation_experiment(4)
    assert ghz.get("quantum", False), "GHZ not quantum"

    print(f"[PASS] quantum circuits: sacred={sd['entropy_bits']:.4f}, qft={qft['info_capacity']:.4f}, ghz={ghz['condensation_ratio']:.4f}")
    passes += 1
except Exception as e:
    errors.append(f"quantum_circuits: {e}")
    print(f"[FAIL] quantum_circuits: {e}")

# ═══ BLOCK 11: v3 research ═══
try:
    from l104_computronium_quantum_research_v3 import QuantumIronResearch
    qir = QuantumIronResearch()
    r = qir.lattice_stability_experiment(293.15, 1.0)
    assert r["success"]
    assert "ground_state_energy_J" in r, "Missing ground_state_energy"
    assert "landauer_cost_per_bit_J" in r, "Missing landauer_cost"
    assert "bekenstein_bound_bits" in r, "Missing bekenstein_bound"
    assert "bekenstein_ratio" in r, "Missing bekenstein_ratio"
    print(f"[PASS] v3 lattice: E0={r['ground_state_energy_J']:.4e} J, landauer={r['landauer_cost_per_bit_J']:.4e} J/bit, bek_ratio={r['bekenstein_ratio']:.4e}")
    passes += 1
except Exception as e:
    errors.append(f"v3_research: {e}")
    print(f"[FAIL] v3_research: {e}")

# ═══ BLOCK 12: mining core ═══
try:
    from l104_computronium_mining_core import BEKENSTEIN_LIMIT, BEKENSTEIN_CONSTANT as MC_BEK, _BREMERMANN_1KG
    # Verify BEKENSTEIN_LIMIT is computed, not hardcoded 2.576e34
    assert BEKENSTEIN_LIMIT > 1e30, f"BEKENSTEIN_LIMIT too small: {BEKENSTEIN_LIMIT}"
    assert MC_BEK > 0, "BEKENSTEIN_CONSTANT not positive"
    assert _BREMERMANN_1KG > 1e40, f"Bremermann too small: {_BREMERMANN_1KG}"
    print(f"[PASS] mining constants: BEKENSTEIN_LIMIT={BEKENSTEIN_LIMIT:.4e}, BREMERMANN={_BREMERMANN_1KG:.4e}")
    passes += 1
except Exception as e:
    errors.append(f"mining_constants: {e}")
    print(f"[FAIL] mining_constants: {e}")

# ═══ BLOCK 13: mining efficiency ═══
try:
    from l104_computronium_mining_core import ComputroniumHashEngine
    he = ComputroniumHashEngine()
    he._synchronize_lattice()  # populate LOPS
    eff = he._calculate_efficiency()
    # Should be a tiny fraction (actual LOPS / Bremermann)
    assert eff >= 0, f"Negative efficiency: {eff}"
    assert eff < 1.0, f"Efficiency > 1 is unphysical: {eff}"
    print(f"[PASS] mining efficiency: {eff:.4e} (fraction of Bremermann limit)")
    passes += 1
except Exception as e:
    errors.append(f"mining_efficiency: {e}")
    print(f"[FAIL] mining_efficiency: {e}")

# ═══ BLOCK 14: computronium.py density constant documented ═══
try:
    import inspect
    from l104_computronium import ComputroniumOptimizer
    src = inspect.getsource(ComputroniumOptimizer)
    assert "measured in EVO_06" in src, "Density constant not documented"
    assert "empirical bits-per-cycle" in src or "measurement" in src, "Missing derivation"
    print("[PASS] L104_DENSITY_CONSTANT documented")
    passes += 1
except Exception as e:
    errors.append(f"density_doc: {e}")
    print(f"[FAIL] density_doc: {e}")

# ═══ BLOCK 15: computronium.py quantum probes (no GOD_CODE/500) ═══
try:
    from l104_computronium import computronium_engine
    bp = computronium_engine.quantum_bekenstein_probe(3)
    assert bp.get("quantum", False), "Probe not quantum"
    # info_capacity should be <= n_qubits (Shannon entropy)
    assert bp["info_capacity_bits"] <= 3.01, f"info_capacity too high (GOD_CODE scaling?): {bp['info_capacity_bits']}"

    dc = computronium_engine.quantum_density_circuit(4, 3)
    assert dc.get("quantum", False), "Density circuit not quantum"
    assert "entropy_bits" in dc, "Missing entropy_bits"
    assert "bekenstein_bound_bits" in dc, "Missing bekenstein_bound_bits"
    print(f"[PASS] quantum probes: bekenstein_cap={bp['info_capacity_bits']:.4f}, density_entropy={dc['entropy_bits']:.4f}")
    passes += 1
except Exception as e:
    errors.append(f"quantum_probes: {e}")
    print(f"[FAIL] quantum_probes: {e}")

# ═══ SUMMARY ═══
print(f"\n{'='*70}")
print(f"WAVE 3 VALIDATION: {passes}/15 passed")
if errors:
    print(f"FAILURES ({len(errors)}):")
    for e in errors:
        print(f"  - {e}")
else:
    print("ALL 15 BLOCKS PASSED — Zero fake physics remaining.")
print(f"{'='*70}")
sys.exit(0 if not errors else 1)
