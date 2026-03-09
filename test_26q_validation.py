#!/usr/bin/env python3
"""L104 26Q Engine Builder — Comprehensive Validation Test"""
import os
import sys
import time

# Fix OpenMP duplicate library issue on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("=" * 80)
print("L104 26Q ENGINE BUILDER — COMPREHENSIVE VALIDATION")
print("=" * 80)

# ═══ TEST 1: Module Import ═══
print("\n=== TEST 1: Module Import ===")
try:
    from l104_26q_engine_builder import (
        L104_26Q_CircuitBuilder, Aer26QExecutionEngine,
        QuantumComputation26QCore, GodCode26QConvergence,
        get_26q_core,
        N_QUBITS_26, STATEVECTOR_MB_26, HILBERT_DIM_26,
        IRON_COMPLETION_FACTOR, GOD_CODE_26Q_RATIO,
        GOD_CODE, PHI, VOID_CONSTANT,
    )
    print(f"  ✓ Module imported: N_QUBITS={N_QUBITS_26}, MEM={STATEVECTOR_MB_26}MB")
    print(f"  ✓ Iron completion: {IRON_COMPLETION_FACTOR}")
    print(f"  ✓ GOD_CODE/1024 ratio: {GOD_CODE_26Q_RATIO:.10f}")
    print(f"  ✓ Hilbert dim: {HILBERT_DIM_26:,}")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# ═══ TEST 2: Convergence Analysis ═══
print("\n=== TEST 2: GOD_CODE ↔ 26Q Convergence ===")
try:
    conv = GodCode26QConvergence.analyze()
    print(f"  Ratio 25Q: GOD_CODE/512  = {conv['ratio_25q']:.10f}")
    print(f"  Ratio 26Q: GOD_CODE/1024 = {conv['ratio_26q']:.10f}")
    print(f"  Octave invariant: {conv['octave_invariant']}")
    print(f"  Iron bridge delta: {conv['iron_completion']['bridge_delta']}")
    assert conv['octave_invariant'], "Octave invariance FAILED"
    assert conv['iron_completion']['bridge_delta'] == 0, "Iron bridge should be 0"
    print("  ✓ Convergence analysis PASSED")
except Exception as e:
    print(f"  ✗ Convergence failed: {e}")

# ═══ TEST 3: ASI Integration ═══
print("\n=== TEST 3: ASI Integration ===")
try:
    from l104_asi.quantum import QuantumComputationCore, _26Q_BUILDER_AVAILABLE
    print(f"  26Q builder available in ASI: {_26Q_BUILDER_AVAILABLE}")
    qcc = QuantumComputationCore()
    print(f"  ✓ QuantumComputationCore instantiated")

    # Test 26Q bridge methods exist
    assert hasattr(qcc, 'build_26q_full_circuit'), "Missing build_26q_full_circuit"
    assert hasattr(qcc, 'execute_26q_circuit'), "Missing execute_26q_circuit"
    assert hasattr(qcc, 'execute_26q_validation'), "Missing execute_26q_validation"
    assert hasattr(qcc, 'iron_26q_convergence'), "Missing iron_26q_convergence"
    assert hasattr(qcc, 'report_26q'), "Missing report_26q"
    print("  ✓ All 26Q bridge methods present on QuantumComputationCore")
except Exception as e:
    print(f"  ✗ ASI integration: {e}")

# ═══ TEST 4: __init__ Re-export ═══
print("\n=== TEST 4: Package Re-export ===")
try:
    from l104_asi import QuantumComputation26QCore as QC26
    print(f"  ✓ Re-exported from l104_asi: {QC26.__name__}")
except Exception as e:
    print(f"  ✗ Re-export: {e}")

# ═══ TEST 5: Builder Instantiation ═══
print("\n=== TEST 5: 26Q Builder Instantiation ===")
try:
    import l104_26q_engine_builder as _mod
    _mod._singleton_26q = None  # Reset singleton for fresh config
    core = get_26q_core(noise_profile="ideal", shots=1024)
    print(f"  ✓ Singleton core created: {type(core).__name__}")
    status = core.status()
    print(f"  Version: {status['version']}")
    print(f"  Qubits: {status['n_qubits']}")
    print(f"  Iron completion: {status['iron_completion']}")
    print(f"  Capabilities: {len(status['capabilities'])}")
except Exception as e:
    print(f"  ✗ Builder instantiation: {e}")

# ═══ TEST 6: Circuit Building + Aer Execution ═══
print("\n=== TEST 6: Circuit Building + Aer Execution ===")
test_circuits = [
    "ghz_iron", "qrng_iron", "bernstein_vazirani",
    "sacred_resonance", "fe_orbital", "consciousness",
]
try:
    core._ensure_builder()
    builder = core._builder
    # Map circuit names to builder methods
    circuit_map = {
        "ghz_iron": builder.build_ghz_iron,
        "qrng_iron": builder.build_qrng_iron,
        "bernstein_vazirani": builder.build_bernstein_vazirani,
        "sacred_resonance": builder.build_sacred_resonance,
        "fe_orbital": builder.build_fe_orbital_mapping,
        "consciousness": builder.build_consciousness_circuit,
    }
    # First verify all circuits BUILD correctly
    for name in test_circuits:
        t0 = time.time()
        build_fn = circuit_map.get(name)
        if build_fn is None:
            print(f"  ✗ {name}: no builder found")
            continue
        qc = build_fn()
        elapsed = time.time() - t0
        print(f"  ✓ {name}: {qc.num_qubits}Q, depth={qc.depth()}, "
              f"gates={sum(qc.count_ops().values())}, build={elapsed:.3f}s")

    # Then test one Aer execution (ghz_iron — fast, ideal noise, DD/ZNE disabled)
    print("  --- Aer execution test (ghz_iron, 1024 shots, ideal noise) ---")
    t0 = time.time()
    result = core.execute_circuit("ghz_iron", mode="shots",
                                   apply_dd=False, apply_zne=False,
                                   apply_noise=False)
    elapsed = time.time() - t0
    if result.get("success"):
        print(f"  ✓ Aer: outcomes={result.get('unique_outcomes', '?')}, "
              f"max_p={result.get('max_probability', '?')}, time={elapsed:.2f}s")
    else:
        print(f"  ✗ Aer: {result.get('error', 'unknown')}")
        if result.get("traceback"):
            print(f"      {result['traceback'][:200]}")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"  ✗ Circuit building: {e}")

# ═══ TEST 7: Full Iron Circuit ═══
print("\n=== TEST 7: Full 26Q Iron Completion Circuit ===")
try:
    result = core.build_full_circuit()
    if result.get("quantum"):
        print(f"  ✓ Full circuit: {result['qubits']}Q, "
              f"depth={result['depth']}, gates={result['gates']}")
        report = result.get('report', {})
        print(f"  Registers: {list(report.get('registers', {}).keys())}")
    else:
        print(f"  ✗ Full circuit: {result.get('error')}")
except Exception as e:
    print(f"  ✗ Full circuit: {e}")

# ═══ TEST 8: Circuit Catalog ═══
print("\n=== TEST 8: Circuit Catalog ===")
try:
    catalog = core.get_circuit_catalog()
    print(f"  ✓ {len(catalog)} circuit types available:")
    for name, info in sorted(catalog.items()):
        print(f"    • {name}: {info['type']} — {info['desc']}")
except Exception as e:
    print(f"  ✗ Catalog: {e}")

# ═══ TEST 9: ASI Bridge Execution ═══
print("\n=== TEST 9: ASI Bridge Execution ===")
try:
    qcc = QuantumComputationCore()
    conv = qcc.iron_26q_convergence()
    print(f"  ✓ iron_26q_convergence: octave_invariant={conv.get('octave_invariant')}")

    full_status = qcc.full_circuit_status()
    print(f"  ✓ 26Q in full_circuit_status: available={full_status.get('builder_26q', {}).get('available')}")
except Exception as e:
    print(f"  ✗ ASI bridge: {e}")

print("\n" + "=" * 80)
print("26Q IRON COMPLETION — VALIDATION COMPLETE")
print("=" * 80)
