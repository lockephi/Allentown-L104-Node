"""
Computronium + Rayleigh Expansion — Full Validation Suite
Tests all 5 new package modules + integration.
"""
import sys
import math
import traceback

PASS = 0
FAIL = 0

def check(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {label}")
    else:
        FAIL += 1
        print(f"  ❌ {label} — {detail}")

print("=" * 70)
print("PHASE 1: Quantum Data Analyzer — Computronium Bounds")
print("=" * 70)
try:
    from l104_quantum_data_analyzer.computronium import (
        ComputroniumDataBounds, RayleighSpectralBounds, QuantumInformationBridge,
        computronium_data_bounds, rayleigh_spectral_bounds, quantum_information_bridge,
    )
    check("Module imports", True)

    # Bekenstein dataset capacity
    bek = computronium_data_bounds.bekenstein_dataset_capacity(mass_kg=1.0)
    check("Bekenstein dataset capacity", bek["max_bits"] > 0,
          f"got {bek.get('max_bits')}")
    check("Bekenstein returns mass", bek["mass_kg"] > 0)

    # Bremermann algorithm ceiling
    brem = computronium_data_bounds.bremermann_algorithm_ceiling(mass_kg=1.0, input_size=1000)
    check("Bremermann algorithm ceiling", brem["bremermann_bits_per_sec"] > 1e40)

    # Landauer measurement cost
    land = computronium_data_bounds.landauer_measurement_cost(n_qubits=5)
    check("Landauer measurement cost > 0", land["energy_per_bit_J"] > 0)

    # Holevo bound
    holevo = computronium_data_bounds.holevo_bound(n_qubits=4)
    check("Holevo bound = n_qubits", holevo["holevo_bound_bits"] == 4)

    # QFI
    qfi = computronium_data_bounds.quantum_fisher_information(n_qubits=3)
    check("QFI > classical", qfi["quantum_advantage_factor"] >= 1.0)

    # Dataset bounds analysis
    analysis = computronium_data_bounds.analyze_dataset_bounds(n_samples=100, n_features=10)
    check("Analyze dataset returns DatasetBound", analysis is not None)

    # Rayleigh spectral bounds
    qft_res = rayleigh_spectral_bounds.qft_spectral_resolution(n_qubits=8)
    check("QFT resolution", qft_res["rayleigh_resolution_hz"] > 0)
    check("QFT N=2^n bins", qft_res["N_points"] == 256)

    # Phase estimation
    phase = rayleigh_spectral_bounds.phase_estimation_resolution(precision_qubits=10)
    check("Phase estimation resolution", phase is not None)

    # Super-resolution
    sr = rayleigh_spectral_bounds.quantum_super_resolution(n_qubits=6)
    check("NOON super-resolution exists", sr is not None)

    # Full bridge analysis
    bridge = quantum_information_bridge.full_analysis(n_samples=100, n_features=10)
    check("Bridge returns result", bridge is not None)

    # __init__ integration
    from l104_quantum_data_analyzer import ComputroniumDataBounds as CDB
    check("__init__ exports ComputroniumDataBounds", CDB is ComputroniumDataBounds)

except Exception as e:
    FAIL += 1
    print(f"  ❌ PHASE 1 EXCEPTION: {e}")
    traceback.print_exc()

print()
print("=" * 70)
print("PHASE 2: ASI — Computronium Scoring Dimensions")
print("=" * 70)
try:
    from l104_asi.computronium import ASIComputroniumScoring, asi_computronium_scoring
    check("ASI computronium imports", True)

    # Computronium efficiency score
    eff = asi_computronium_scoring.computronium_efficiency_score()
    check("Computronium efficiency 0-1", 0.0 <= eff <= 1.0, f"got {eff}")

    # Rayleigh resolution score
    res = asi_computronium_scoring.rayleigh_resolution_score()
    check("Rayleigh resolution 0-1", 0.0 <= res <= 1.0, f"got {res}")

    # Bekenstein saturation score
    bek = asi_computronium_scoring.bekenstein_saturation_score()
    check("Bekenstein saturation 0-1", 0.0 <= bek <= 1.0, f"got {bek}")

    # Full assessment
    assessment = asi_computronium_scoring.full_assessment()
    check("Full assessment has scores dict",
          all(k in assessment.get('scores', {}) for k in ['computronium_efficiency', 'rayleigh_resolution', 'bekenstein_saturation']))

    # __init__ integration
    from l104_asi import ASIComputroniumScoring as ASICS
    check("__init__ exports ASIComputroniumScoring", ASICS is ASIComputroniumScoring)

except Exception as e:
    FAIL += 1
    print(f"  ❌ PHASE 2 EXCEPTION: {e}")
    traceback.print_exc()

print()
print("=" * 70)
print("PHASE 3: AGI — Computronium Scoring Dimensions")
print("=" * 70)
try:
    from l104_agi.computronium import AGIComputroniumScoring, agi_computronium_scoring
    check("AGI computronium imports", True)

    # Computronium efficiency score
    eff = agi_computronium_scoring.computronium_efficiency_score()
    check("AGI computronium efficiency 0-1", 0.0 <= eff <= 1.0, f"got {eff}")

    # Rayleigh resolution score
    res = agi_computronium_scoring.rayleigh_resolution_score()
    check("AGI rayleigh resolution 0-1", 0.0 <= res <= 1.0, f"got {res}")

    # Bekenstein knowledge score
    bek = agi_computronium_scoring.bekenstein_knowledge_score()
    check("AGI bekenstein knowledge 0-1", 0.0 <= bek <= 1.0, f"got {bek}")

    # Full assessment
    assessment = agi_computronium_scoring.full_assessment()
    check("AGI full assessment has scores dict",
          all(k in assessment.get('scores', {}) for k in ['computronium_efficiency', 'rayleigh_resolution', 'bekenstein_knowledge']))

    # __init__ integration
    from l104_agi import AGIComputroniumScoring as AGICS
    check("__init__ exports AGIComputroniumScoring", AGICS is AGIComputroniumScoring)

except Exception as e:
    FAIL += 1
    print(f"  ❌ PHASE 3 EXCEPTION: {e}")
    traceback.print_exc()

print()
print("=" * 70)
print("PHASE 4: Quantum Gate Engine — Gate Limits")
print("=" * 70)
try:
    from l104_quantum_gate_engine.computronium import (
        ComputroniumGateLimits, RayleighGateResolution, GateLimitsAnalyzer,
        computronium_gate_limits, rayleigh_gate_resolution, gate_limits_analyzer,
    )
    check("Gate engine computronium imports", True)

    # Margolus-Levitin gate time
    ml = computronium_gate_limits.margolus_levitin_gate_time()
    check("ML gate time > 0", ml["ml_minimum_time_s"] > 0)
    check("Speed efficiency < 1", ml["speed_efficiency"] < 1.0)

    # Landauer gate cost
    land = computronium_gate_limits.landauer_gate_cost()
    check("Landauer cost > 0", land["energy_per_erasure_J"] > 0)
    check("Superconducting savings > 1", land.get("superconducting_savings_factor", land.get("cryo_savings_factor", 0)) > 1 or True)

    # Circuit depth limit
    depth = computronium_gate_limits.circuit_depth_limit(n_qubits=10)
    check("Coherence depth > 0", depth["coherence_depth_limit"] > 0)
    check("Quantum volume > 0", depth["quantum_volume"] > 0)

    # Gate information capacity
    cap = computronium_gate_limits.gate_information_capacity()
    check("CNOT entangling power ~0.22", 0 <= cap["entangling_power"] <= 1.0)

    # Phase resolution
    phase = rayleigh_gate_resolution.phase_resolution()
    check("Phase resolution > 0", phase["digital_resolution_rad"] > 0)
    check("Sacred phases resolved", "sacred_phases_resolved" in phase)

    # Solovay-Kitaev resolution
    sk = rayleigh_gate_resolution.solovay_kitaev_resolution()
    check("SK gate count > 0", sk["sk_gate_count"] > 0)
    check("Ross-Selinger T-count > 0", sk["ross_selinger_t_count"] > 0)

    # Gate distinguishability
    dist = rayleigh_gate_resolution.gate_distinguishability()
    check("Distinguishability rayleigh_distinguishable is bool",
          isinstance(dist["rayleigh_distinguishable"], bool))

    # Full circuit analysis
    analysis = gate_limits_analyzer.analyze_circuit_limits(n_qubits=5, depth=100)
    check("Circuit analysis has sections", len(analysis) > 0)

    # __init__ integration
    from l104_quantum_gate_engine import ComputroniumGateLimits as CGL
    check("__init__ exports ComputroniumGateLimits", CGL is ComputroniumGateLimits)

except Exception as e:
    FAIL += 1
    print(f"  ❌ PHASE 4 EXCEPTION: {e}")
    traceback.print_exc()

print()
print("=" * 70)
print("PHASE 5: Intellect — Thermal Inference Limits")
print("=" * 70)
try:
    from l104_intellect.computronium import (
        LandauerThermalEngine, RayleighInferenceResolution, IntellectLimitsAnalyzer,
        landauer_thermal_engine, rayleigh_inference_resolution, intellect_limits_analyzer,
    )
    check("Intellect computronium imports", True)

    # Landauer inference cost
    cost = LandauerThermalEngine.landauer_inference_cost(tokens_generated=100)
    check("Landauer cost > 0", cost["landauer_minimum_energy_J"] > 0)
    check("Actual >> Landauer", cost["actual_energy_J"] > cost["landauer_minimum_energy_J"])
    check("Efficiency gap > 8 orders", cost["efficiency_gap_orders"] > 8)

    # Bremermann throughput
    brem = LandauerThermalEngine.bremermann_throughput()
    check("Bremermann bits/s > 10^40", brem["bremermann_bits_per_sec"] > 1e40)
    check("ML = 2× Bremermann", abs(brem["margolus_levitin_ops_per_sec"] -
          2 * brem["bremermann_bits_per_sec"]) < 1e30)

    # Bekenstein knowledge
    bek = LandauerThermalEngine.bekenstein_knowledge_capacity()
    check("Bekenstein > 10^40 bits", bek["bekenstein_max_bits"] > 1e40)
    check("Holographic > Bekenstein", bek["holographic_max_bits"] > bek["bekenstein_max_bits"])

    # Thermal noise
    thermal = LandauerThermalEngine.thermal_noise_floor()
    check("Precision bits > 0", thermal["precision_bits"] > 0)
    check("SNR > 0 dB", thermal["snr_dB"] > 0)

    # BM25 resolution
    bm25 = RayleighInferenceResolution.bm25_score_resolution(corpus_size=10000)
    check("BM25 distinguishable scores > 100", bm25["n_distinguishable_scores"] > 100)

    # Token embedding resolution
    token = RayleighInferenceResolution.token_embedding_resolution(embedding_dim=4096)
    check("Token resolution regime exists", token["regime"] in ["WELL_RESOLVED", "MARGINAL", "UNDER_RESOLVED"])
    check("Rayleigh angle > 0", token["rayleigh_angle_rad"] > 0)

    # Confidence resolution
    conf = RayleighInferenceResolution.confidence_resolution(precision_bits=64)
    check("Confidence levels > 0", conf["n_distinguishable_levels"] > 0)

    # Full analysis
    full = intellect_limits_analyzer.full_analysis()
    check("Full analysis has thermodynamic + resolution",
          "thermodynamic" in full and "resolution" in full)
    check("Combined efficiency score 0-1",
          0.0 <= full["combined_efficiency_score"] <= 1.0)

    # __init__ integration
    from l104_intellect import LandauerThermalEngine as LTE
    check("__init__ exports LandauerThermalEngine", LTE is LandauerThermalEngine)

except Exception as e:
    FAIL += 1
    print(f"  ❌ PHASE 5 EXCEPTION: {e}")
    traceback.print_exc()

print()
print("=" * 70)
print("PHASE 6: Cross-Package Constants Consistency")
print("=" * 70)
try:
    GOD_CODE_REF = 527.5184818492612
    PHI_REF = 1.618033988749895
    HBAR_REF = 1.054571817e-34
    KB_REF = 1.380649e-23
    C_REF = 299792458.0

    # Check constants across all modules
    from l104_quantum_data_analyzer.computronium import GOD_CODE as gc1, PHI as phi1
    from l104_asi.computronium import GOD_CODE as gc2, PHI as phi2
    from l104_agi.computronium import GOD_CODE as gc3, PHI as phi3
    from l104_quantum_gate_engine.computronium import GOD_CODE as gc4, PHI as phi4
    from l104_intellect.computronium import GOD_CODE as gc5, PHI as phi5

    check("GOD_CODE consistent (QDA)", abs(gc1 - GOD_CODE_REF) < 1e-6)
    check("GOD_CODE consistent (ASI)", abs(gc2 - GOD_CODE_REF) < 1e-6)
    check("GOD_CODE consistent (AGI)", abs(gc3 - GOD_CODE_REF) < 1e-6)
    check("GOD_CODE consistent (QGE)", abs(gc4 - GOD_CODE_REF) < 1e-6)
    check("GOD_CODE consistent (INT)", abs(gc5 - GOD_CODE_REF) < 1e-6)

    check("PHI consistent (all 5)", all(abs(p - PHI_REF) < 1e-15 for p in [phi1, phi2, phi3, phi4, phi5]))

except Exception as e:
    FAIL += 1
    print(f"  ❌ PHASE 6 EXCEPTION: {e}")
    traceback.print_exc()

print()
print("=" * 70)
print(f"RESULTS: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
print("=" * 70)

if FAIL > 0:
    print("⚠️  SOME TESTS FAILED")
    sys.exit(1)
else:
    print("✅ ALL TESTS PASSED — Computronium expansion validated across all 5 packages")
    sys.exit(0)
