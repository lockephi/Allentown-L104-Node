#!/usr/bin/env python3
"""Debug Phase 16: Quantum Link Computation Engine — all 16+ algorithms."""

import traceback
import time
import sys

print("=" * 70)
print("  PHASE 16 DEBUG: Quantum Link Computation Engine")
print("=" * 70)

# ── Boot ──
t0 = time.time()
try:
    from l104_quantum_engine.computation import QuantumLinkComputationEngine
    print(f"\n  ✓ Import OK ({time.time()-t0:.3f}s)")
except Exception as e:
    print(f"\n  ✗ Import FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

qce = QuantumLinkComputationEngine()
print(f"  ✓ Engine init: {type(qce).__name__}")
print(f"    qmath: {type(qce.qmath).__name__}")
print(f"    gate_engine_checked: {qce._gate_engine_checked}")

# ── Test gate engine availability ──
print(f"\n  ── Gate Engine Probe ──")
ge = qce._get_gate_engine_cached()
if ge is not None:
    print(f"  ✓ Gate engine available: {type(ge).__name__}")
else:
    print(f"  ⚠ Gate engine NOT available (graceful fallback)")

# ── Run each of the 16 core algorithms ──
print(f"\n  ── Core Algorithms (16) ──")
algos = [
    ("1.  Error Correction",        lambda: qce.quantum_error_correction()),
    ("2.  Channel Capacity",        lambda: qce.quantum_channel_capacity()),
    ("3.  BB84 Key Distribution",   lambda: qce.bb84_key_distribution()),
    ("4.  State Tomography",        lambda: qce.quantum_state_tomography()),
    ("5.  Quantum Walk",            lambda: qce.quantum_walk_link_graph()),
    ("6.  Variational Optimizer",   lambda: qce.variational_link_optimizer()),
    ("7.  Process Tomography",      lambda: qce.quantum_process_tomography()),
    ("8.  Zeno Stabilizer",         lambda: qce.quantum_zeno_stabilizer()),
    ("9.  Adiabatic Evolution",     lambda: qce.adiabatic_link_evolution()),
    ("10. Quantum Metrology",       lambda: qce.quantum_metrology()),
    ("11. Reservoir Computing",     lambda: qce.quantum_reservoir_computing()),
    ("12. Approximate Counting",    lambda: qce.quantum_approximate_counting()),
    ("13. Lindblad Decoherence",    lambda: qce.lindblad_decoherence_model()),
    ("14. Entanglement Distillation", lambda: qce.entanglement_distillation()),
    ("15. Fe Lattice Simulation",   lambda: qce.fe_lattice_simulation()),
    ("16. HHL Linear Solver",       lambda: qce.hhl_link_linear_solver()),
]

passed = 0
failed = 0
errors = []

for name, fn in algos:
    t1 = time.time()
    try:
        result = fn()
        dt = time.time() - t1
        if isinstance(result, dict) and "error" in result:
            print(f"  ⚠ {name}: error in result → {str(result['error'])[:100]}")
            failed += 1
            errors.append((name, result["error"]))
        else:
            algo_label = result.get("algorithm", "?") if isinstance(result, dict) else "?"
            gc = result.get("god_code_alignment",
                 result.get("god_code_resonance", "n/a")) if isinstance(result, dict) else "?"
            gc_str = f"{gc:.6f}" if isinstance(gc, float) else str(gc)
            print(f"  ✓ {name}: {algo_label} | GC={gc_str} ({dt:.3f}s)")
            passed += 1
    except Exception as e:
        dt = time.time() - t1
        print(f"  ✗ {name}: EXCEPTION ({dt:.3f}s)")
        print(f"    → {type(e).__name__}: {e}")
        traceback.print_exc()
        failed += 1
        errors.append((name, str(e)))

# ── Gate engine enhanced algorithms ──
if ge is not None:
    print(f"\n  ── Gate Engine Enhanced Algorithms (4) ──")
    gate_algos = [
        ("G1. Grover Circuit",      lambda: qce._gate_grover_circuit(ge)),
        ("G2. QFT Analysis",        lambda: qce._gate_qft_analysis(ge)),
        ("G3. Bell Verification",   lambda: qce._gate_bell_verification(ge)),
        ("G4. Sacred Alignment",    lambda: qce._gate_sacred_alignment(ge)),
    ]
    for name, fn in gate_algos:
        t1 = time.time()
        try:
            result = fn()
            dt = time.time() - t1
            if isinstance(result, dict) and "error" in result:
                print(f"  ⚠ {name}: error → {str(result['error'])[:100]}")
                failed += 1
                errors.append((name, result["error"]))
            else:
                algo_label = result.get("algorithm", "?") if isinstance(result, dict) else "?"
                gc = result.get("god_code_alignment", "n/a") if isinstance(result, dict) else "?"
                gc_str = f"{gc:.6f}" if isinstance(gc, float) else str(gc)
                print(f"  ✓ {name}: {algo_label} | GC={gc_str} ({dt:.3f}s)")
                passed += 1
        except Exception as e:
            dt = time.time() - t1
            print(f"  ✗ {name}: EXCEPTION ({dt:.3f}s)")
            print(f"    → {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
            errors.append((name, str(e)))

# ── Full analysis test ──
print(f"\n  ── Full Quantum Analysis ──")
t2 = time.time()
try:
    full = qce.full_quantum_analysis()
    dt = time.time() - t2
    comps = full.get("computations", {})
    comp_ok = sum(1 for v in comps.values() if isinstance(v, dict) and "error" not in v)
    comp_err = sum(1 for v in comps.values() if isinstance(v, dict) and "error" in v)
    print(f"  ✓ full_quantum_analysis OK ({dt:.3f}s)")
    print(f"    computations: {comp_ok} ok, {comp_err} errors, {len(comps)} total")
    print(f"    composite_coherence: {full.get('composite_coherence', 0):.6f}")
    print(f"    gate_engine_active: {full.get('gate_engine_active', False)}")
    print(f"    total_computations: {full.get('total_computations', 0)}")
    if comp_err > 0:
        print(f"  ── Errored computations ──")
        for k, v in comps.items():
            if isinstance(v, dict) and "error" in v:
                print(f"    ✗ {k}: {v['error'][:120]}")
except Exception as e:
    dt = time.time() - t2
    print(f"  ✗ full_quantum_analysis FAILED ({dt:.3f}s): {e}")
    traceback.print_exc()

# ── Summary ──
total_time = time.time() - t0
print(f"\n{'='*70}")
print(f"  PHASE 16 RESULTS: {passed} passed, {failed} failed")
print(f"  Computation count: {qce.computation_count}")
print(f"  Total time: {total_time:.3f}s")
if errors:
    print(f"\n  ── ERRORS REQUIRING ATTENTION ──")
    for name, err in errors:
        print(f"    ✗ {name}: {str(err)[:120]}")
print(f"{'='*70}")
