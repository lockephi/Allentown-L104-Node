"""Verify _embed_gate correctness, then run Phase 1F."""
import sys, time, traceback, numpy as np
sys.path.insert(0, '.')

# ── Part 1: Verify _embed_gate produces correct matrices ──────────────────────
print('=== EMBED_GATE CORRECTNESS CHECK ===', flush=True)

from l104_quantum_gate_engine.circuit import GateCircuit
from l104_quantum_gate_engine import H as H_GATE, CNOT as CNOT_GATE

# Test 1: H on qubit 0 of 2-qubit system
circ = GateCircuit(2, "test")
m = circ._embed_gate(H_GATE.matrix, (0,))
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
I2 = np.eye(2)
expected = np.kron(I2, H)
ok = np.allclose(m, expected)
print(f'  [1] H on q0 of 2q: {"OK" if ok else "FAIL"}', flush=True)
if not ok:
    print(f'    got:\n{np.round(m, 3)}', flush=True)
    print(f'    exp:\n{np.round(expected, 3)}', flush=True)

# Test 2: H on qubit 1 of 2-qubit system
m = circ._embed_gate(H_GATE.matrix, (1,))
expected = np.kron(H, I2)
ok = np.allclose(m, expected)
print(f'  [2] H on q1 of 2q: {"OK" if ok else "FAIL"}', flush=True)
if not ok:
    print(f'    got:\n{np.round(m, 3)}', flush=True)
    print(f'    exp:\n{np.round(expected, 3)}', flush=True)

# Test 3: CNOT(1,0) — control=q1 (MSB), target=q0 (LSB) → standard CNOT
m = circ._embed_gate(CNOT_GATE.matrix, (1, 0))
expected_cnot = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
ok = np.allclose(m, expected_cnot)
print(f'  [3] CNOT(1,0) 2q: {"OK" if ok else "FAIL"}', flush=True)
if not ok:
    print(f'    got:\n{np.round(m, 3)}', flush=True)

# Test 4: CNOT(0,1) — control=q0 (LSB), target=q1 (MSB) → swapped
m = circ._embed_gate(CNOT_GATE.matrix, (0, 1))
expected_swapped = np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]], dtype=complex)
ok = np.allclose(m, expected_swapped)
print(f'  [4] CNOT(0,1) 2q: {"OK" if ok else "FAIL"}', flush=True)
if not ok:
    print(f'    got:\n{np.round(m, 3)}', flush=True)

# Test 5: H on qubit 1 of 3-qubit system
circ3 = GateCircuit(3, "test3")
m = circ3._embed_gate(H_GATE.matrix, (1,))
expected = np.kron(np.kron(I2, H), I2)
ok = np.allclose(m, expected)
print(f'  [5] H on q1 of 3q: {"OK" if ok else "FAIL"}', flush=True)

# Test 6: Full Bell circuit unitary
circ_bell = GateCircuit(2, "bell")
circ_bell.h(0)
circ_bell.cx(0, 1)
U = circ_bell.unitary()
U_expected = expected_swapped @ np.kron(I2, H)
ok = np.allclose(U, U_expected)
print(f'  [6] Bell unitary: {"OK" if ok else "FAIL"}', flush=True)

# Test 7: Performance — 8 qubit embed
circ8 = GateCircuit(8, "perf")
t0 = time.time()
for _ in range(100):
    circ8._embed_gate(H_GATE.matrix, (4,))
t1 = time.time()
print(f'  [7] 8q embed: {(t1-t0)/100*1000:.2f}ms/gate', flush=True)

# ── Part 2: Phase 1F ─────────────────────────────────────────────────────────
print('\n=== PHASE 1F TEST ===', flush=True)
from l104_quantum_engine import quantum_brain

gate_engine = quantum_brain._get_gate_engine_cached()
if gate_engine is None:
    print('FAIL: gate engine not available', flush=True)
    sys.exit(1)

print(f'  gate_engine: {type(gate_engine).__name__}', flush=True)
n_links = len(getattr(quantum_brain, 'links', []))
print(f'  n_links: {n_links}', flush=True)

t0 = time.time()
try:
    results = quantum_brain._run_gate_engine_phase(gate_engine, n_links)
    elapsed = time.time() - t0
    print(f'  Phase 1F completed in {elapsed:.2f}s', flush=True)
    print(f'  Status: {results.get("status")}', flush=True)
    for k, v in sorted(results.items()):
        if k == 'status':
            continue
        if isinstance(v, dict):
            err = v.get('error')
            if err:
                print(f'  FAIL {k}: {err}', flush=True)
            else:
                keys_shown = [sk for sk in v if not isinstance(v[sk], (dict, list)) or len(str(v[sk])) < 50]
                summary = {sk: v[sk] for sk in keys_shown[:6]}
                print(f'  OK   {k}: {summary}', flush=True)
except Exception as e:
    elapsed = time.time() - t0
    print(f'  EXCEPTION after {elapsed:.2f}s: {e}', flush=True)
    traceback.print_exc()

print('\n=== DONE ===', flush=True)
