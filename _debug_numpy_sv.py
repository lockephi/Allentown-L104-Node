#!/usr/bin/env python3
"""Quick debug of _execute_local_numpy statevector fallback."""
import sys
import numpy as np
import traceback

from l104_quantum_gate_engine import get_engine, ExecutionTarget
from l104_quantum_gate_engine.orchestrator import CrossSystemOrchestrator

engine = get_engine()
orch = CrossSystemOrchestrator()

PASS = FAIL = 0
def check(label, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ✓ {label}")
    else:
        FAIL += 1; print(f"  ✗ {label}  {detail}")

print("=" * 60)
print("DEBUG: _execute_local_numpy statevector")
print("=" * 60)

# --- Bell pair ---
try:
    circ = engine.bell_pair()
    r = orch._execute_local_numpy(circ)
    check("Bell pair runs", r.probabilities is not None, r.metadata.get("error",""))
    if r.probabilities:
        p00 = r.probabilities.get("00", 0)
        p11 = r.probabilities.get("11", 0)
        check(f"P(00)={p00:.6f} ≈ 0.5", abs(p00 - 0.5) < 0.01)
        check(f"P(11)={p11:.6f} ≈ 0.5", abs(p11 - 0.5) < 0.01)
        check(f"statevector shape = {r.statevector.shape}", r.statevector is not None and r.statevector.shape == (4,))
    else:
        print(f"  ERROR: {r.metadata}")
except Exception as e:
    traceback.print_exc()
    check("Bell pair", False, str(e))

# --- GHZ-5 ---
try:
    ghz = engine.ghz_state(5)
    r2 = orch._execute_local_numpy(ghz)
    check("GHZ-5 runs", r2.probabilities is not None, r2.metadata.get("error",""))
    if r2.probabilities:
        p0 = r2.probabilities.get("00000", 0)
        p1 = r2.probabilities.get("11111", 0)
        check(f"GHZ P(0^5)={p0:.6f} ≈ 0.5", abs(p0 - 0.5) < 0.01)
        check(f"GHZ P(1^5)={p1:.6f} ≈ 0.5", abs(p1 - 0.5) < 0.01)
except Exception as e:
    traceback.print_exc()
    check("GHZ-5", False, str(e))

# --- 10Q QFT ---
try:
    qft = engine.quantum_fourier_transform(4)
    r3 = orch._execute_local_numpy(qft)
    check("QFT-4 runs", r3.probabilities is not None, r3.metadata.get("error",""))
    if r3.probabilities:
        check(f"QFT-4 nstates={len(r3.probabilities)}", len(r3.probabilities) > 0)
        total_p = sum(r3.probabilities.values())
        check(f"QFT-4 total_prob={total_p:.6f} ≈ 1", abs(total_p - 1.0) < 0.01)
except Exception as e:
    traceback.print_exc()
    check("QFT-4", False, str(e))

# --- Compare with ChunkedStatevectorSimulator ---
try:
    from l104_simulator.chunked_statevector import ChunkedStatevectorSimulator
    sim = ChunkedStatevectorSimulator(use_gpu=False, return_statevector=True)
    circ8 = engine.ghz_state(8)
    r_chunked = sim.run_gate_circuit(circ8)
    r_numpy = orch._execute_local_numpy(circ8)

    if r_numpy.statevector is not None and r_chunked.statevector is not None:
        diff = np.max(np.abs(r_numpy.statevector - r_chunked.statevector))
        check(f"8Q GHZ: numpy vs chunked max diff = {diff:.2e}", diff < 1e-10)
    elif r_numpy.probabilities and r_chunked.probabilities:
        # Compare probabilities
        keys = set(r_numpy.probabilities) | set(r_chunked.probabilities)
        max_pdiff = max(abs(r_numpy.probabilities.get(k,0) - r_chunked.probabilities.get(k,0)) for k in keys)
        check(f"8Q GHZ: numpy vs chunked max prob diff = {max_pdiff:.2e}", max_pdiff < 1e-10)
    else:
        check("8Q comparison", False, f"numpy probs={r_numpy.probabilities is not None} chunked probs={r_chunked.probabilities is not None}")
except Exception as e:
    traceback.print_exc()
    check("Chunked comparison", False, str(e))

# --- Sacred circuit ---
try:
    sacred = engine.sacred_circuit(3, depth=2)
    r4 = orch._execute_local_numpy(sacred)
    check("Sacred 3Q runs", r4.probabilities is not None, r4.metadata.get("error",""))
    if r4.probabilities:
        total = sum(r4.probabilities.values())
        check(f"Sacred total_prob={total:.6f}", abs(total - 1.0) < 0.01)
except Exception as e:
    traceback.print_exc()
    check("Sacred circuit", False, str(e))

# --- Edge case: 1-qubit ---
try:
    c1 = engine.create_circuit(1, "single")
    c1.h(0)
    r5 = orch._execute_local_numpy(c1)
    check("1Q H|0⟩ runs", r5.probabilities is not None, r5.metadata.get("error",""))
    if r5.probabilities:
        p0 = r5.probabilities.get("0", 0)
        p1 = r5.probabilities.get("1", 0)
        check(f"1Q P(0)={p0:.4f}, P(1)={p1:.4f}", abs(p0-0.5)<0.01 and abs(p1-0.5)<0.01)
except Exception as e:
    traceback.print_exc()
    check("1Q circuit", False, str(e))

print()
print("=" * 60)
print(f"RESULTS: {PASS}/{PASS+FAIL} passed, {FAIL} failed")
if FAIL == 0:
    print("★ numpy_gatewise statevector debug: ALL CORRECT")
print("=" * 60)
sys.exit(0 if FAIL == 0 else 1)
