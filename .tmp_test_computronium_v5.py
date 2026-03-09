#!/usr/bin/env python3
"""Computronium v5.0 Upgrade Validation"""
import time, json

from l104_computronium import computronium_engine as ce

print("=" * 70)
print("COMPUTRONIUM v5.0 — PHASE 5 RESEARCH VALIDATION")
print("=" * 70)

passed = 0
failed = 0

# Test 1: Landauer Temperature Sweep (I-5-01)
print("\n[1/6] I-5-01: Landauer Temperature Sweep...")
try:
    t0 = time.time()
    sweep = ce.landauer_temperature_sweep(n_points=15)
    assert sweep["optimal_temperature_K"] > 0
    assert sweep["optimal_throughput_bits"] > 0
    assert len(sweep["sweep"]) == 15
    print(f"  Optimal T: {sweep['optimal_temperature_K']:.4f} K")
    print(f"  Optimal throughput: {sweep['optimal_throughput_bits']:.4e} bits")
    print(f"  Cryo vs Room: {sweep['cryo_vs_room_advantage']:.2f}x")
    print(f"  PASS ({(time.time()-t0)*1000:.1f}ms)")
    passed += 1
except Exception as e:
    print(f"  FAIL: {e}")
    failed += 1

# Test 2: Decoherence Topography (I-5-02)
print("\n[2/6] I-5-02: Decoherence Topography Probe...")
try:
    t0 = time.time()
    topo = ce.decoherence_topography_probe(n_qubits=4, noise_levels=10)
    assert len(topo["topography"]) == 11  # 0..10 inclusive
    assert topo["steane_overhead_factor"] == 21.0
    print(f"  EC break-even: {topo['ec_break_even_noise']}")
    print(f"  QA threshold: {topo['quantum_advantage_threshold']}")
    print(f"  Steane overhead: {topo['steane_overhead_factor']}x")
    print(f"  Real circuit: {bool(topo.get('real_circuit'))}")
    print(f"  PASS ({(time.time()-t0)*1000:.1f}ms)")
    passed += 1
except Exception as e:
    print(f"  FAIL: {e}")
    failed += 1

# Test 3: Bremermann Saturation (I-5-03)
print("\n[3/6] I-5-03: Bremermann Saturation Analysis...")
try:
    t0 = time.time()
    sat = ce.bremermann_saturation_analysis(scale_factors=15)
    assert sat["equivalent_mass_kg"] > 0
    assert sat["planck_mass_kg"] > 0
    assert len(sat["saturation_curve"]) == 15
    print(f"  Throughput: {sat['current_throughput_bits_s']:.4e} bits/s")
    print(f"  Equiv mass: {sat['equivalent_mass_kg']:.4e} kg (log10={sat['equivalent_mass_log10']})")
    print(f"  Planck mass: {sat['planck_mass_kg']:.4e} kg")
    print(f"  PASS ({(time.time()-t0)*1000:.1f}ms)")
    passed += 1
except Exception as e:
    print(f"  FAIL: {e}")
    failed += 1

# Test 4: Entropy Lifecycle (I-5-04)
print("\n[4/6] I-5-04: Entropy Lifecycle Pipeline...")
try:
    t0 = time.time()
    lc = ce.entropy_lifecycle_pipeline(data_size_bytes=5000)
    net = lc["phases"]["6_net_accounting"]
    assert net["net_energy_J"] > 0
    assert lc["total_bits"] == 40000
    print(f"  Total bits: {lc['total_bits']}")
    print(f"  Lifecycle eff: {net['lifecycle_efficiency']:.8f}")
    print(f"  Net energy: {net['net_energy_J']:.4e} J")
    print(f"  Vs ideal: {net['efficiency_vs_ideal']:.8f}")
    print(f"  Demon: {lc['phases']['4_reversal']['demon_available']}")
    print(f"  PASS ({(time.time()-t0)*1000:.1f}ms)")
    passed += 1
except Exception as e:
    print(f"  FAIL: {e}")
    failed += 1

# Test 5: Cross-Engine Synthesis
print("\n[5/6] Cross-Engine Computronium Synthesis...")
try:
    t0 = time.time()
    synth = ce.cross_engine_computronium_synthesis()
    assert synth["engines_available"] >= 1  # at least computronium itself
    assert synth["synthesis_score"] >= 0
    print(f"  Engines: {synth['engines_available']}/{synth['total_engines']}")
    print(f"  Score: {synth['synthesis_score']:.4f}")
    for nm, dt in synth["engines"].items():
        avail = dt.get("available", False) if dt else False
        print(f"    {nm}: {'OK' if avail else 'UNAVAIL'}")
    print(f"  PASS ({(time.time()-t0)*1000:.1f}ms)")
    passed += 1
except Exception as e:
    print(f"  FAIL: {e}")
    failed += 1

# Test 6: Full Assessment
print("\n[6/6] Full Computronium Assessment...")
try:
    t0 = time.time()
    assess = ce.full_computronium_assessment()
    assert assess["phases_completed"] >= 5  # at least 5 of 8 phases should work
    assert assess["version"] == "5.0.0"
    print(f"  Phases: {assess['phases_completed']}/{assess['phases_total']}")
    print(f"  Errors: {assess['errors_count']}")
    for nm in assess["phases"]:
        print(f"    {nm}: OK")
    for e in assess.get("errors", []):
        print(f"    ERROR: {e}")
    print(f"  PASS ({assess['total_latency_ms']:.0f}ms)")
    passed += 1
except Exception as e:
    print(f"  FAIL: {e}")
    failed += 1

# Solve routing tests
print("\n--- Solve Routing (v5 keywords) ---")
for q in ["temperature sweep analysis", "decoherence topography", "bremermann saturation",
           "entropy lifecycle accounting", "cross-engine synthesis", "full assessment pipeline"]:
    try:
        sol = ce.solve({"query": q})
        print(f"  '{q[:30]}...' -> {sol['source']} | {sol['solution'][:60]}...")
    except Exception as e:
        print(f"  '{q}' FAIL: {e}")

print("\n" + "=" * 70)
print(f"RESULTS: {passed}/{passed+failed} PASSED | {failed} FAILED")
print(f"VERSION: {ce.VERSION} | GOD_CODE: {ce.GOD_CODE}")
print("=" * 70)
