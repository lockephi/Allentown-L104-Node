#!/usr/bin/env python3
"""
L104 GOD_CODE Rank #1 Push — Hardware-Efficient Ansatz V2
═══════════════════════════════════════════════════════════

Clean ansatz: Two layers of Rz-SX-CZ-SX-Rz on all 8 qubits.
Target: |00001111⟩ via SPSA optimization.

INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════
"""

import os
import sys
import math
import time
import json
import numpy as np
from pathlib import Path

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit import transpile
from qiskit.quantum_info import Statevector

# ═══ CONFIG ═══
NQ = 8
TARGET = "00001111"
# Qiskit bit ordering: qubit 0 = LSB (rightmost bit), so reverse for index
TARGET_QISKIT = TARGET[::-1]  # "00001111" → "11110000"
target_idx = int(TARGET_QISKIT, 2)  # → 240

# Two layers, 2 params per qubit per layer = 16 params per layer
# Layer 1: Rz-SX-CZ-SX-Rz (16 params for Rz)
# Layer 2: Rz-SX-CZ-SX-Rz (16 params for Rz)
N_PARAMS = 32

print(f"{'=' * 80}")
print(f"  L104 GOD_CODE RANK #1 PUSH — Hardware-Efficient VQE")
print(f"  Target: |{TARGET}⟩  (q0-q3=|0⟩, q4-q7=|1⟩)")
print(f"{'=' * 80}")

# ═══ BUILD CIRCUIT ═══
params = ParameterVector("θ", N_PARAMS)
qc = QuantumCircuit(NQ, NQ)
pi = 0

# Layer 1
for q in range(NQ):
    qc.rz(params[pi], q); pi += 1
for q in range(NQ):
    qc.sx(q)
# Entanglement: chain of CZ
for q in range(NQ - 1):
    qc.cz(q, q + 1)
for q in range(NQ):
    qc.sx(q)
for q in range(NQ):
    qc.rz(params[pi], q); pi += 1

qc.barrier()

# Layer 2
for q in range(NQ):
    qc.rz(params[pi], q); pi += 1
for q in range(NQ):
    qc.sx(q)
for q in range(NQ - 1):
    qc.cz(q, q + 1)
for q in range(NQ):
    qc.sx(q)
for q in range(NQ):
    qc.rz(params[pi], q); pi += 1

qc.measure(range(NQ), range(NQ))

assert pi == N_PARAMS, f"Param mismatch: used {pi}, allocated {N_PARAMS}"

print(f"\nParameterized circuit: {qc.size()} gates, depth {qc.depth()}")
print(f"  Parameters: {N_PARAMS}")

# ═══ SIM ONLY MODE ═══
SIM_ONLY = "--sim-only" in sys.argv

if not SIM_ONLY:
    print("\n[IBM] Authenticating...")
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = None
    token = os.environ.get("IBMQ_TOKEN")
    if not token:
        try:
            service = QiskitRuntimeService()
            print("[IBM] Authenticated via saved credentials")
        except Exception:
            pass
    if service is None:
        SIM_ONLY = True
        print("[WARN] No IBM credentials. Running simulation only.")

if not SIM_ONLY:
    backend = service.backend("ibm_marrakesh")
    print(f"[IBM] Backend: {backend.name} ({backend.num_qubits}Q)")

# ═══ TRANSPILE ═══
print("\n[TRANSPILE] Transpiling for simulation...")
if SIM_ONLY:
    from qiskit.providers.basic_provider import BasicSimulator
    tp = transpile(qc, BasicSimulator(), optimization_level=3)
else:
    tp = transpile(qc, backend=backend, optimization_level=3)

two_q = sum(1 for inst, _, _ in tp.data if inst.num_qubits == 2 and inst.name != 'barrier')
print(f"[TRANSPILE] Result: {tp.size()} gates, depth {tp.depth()}, 2Q={two_q}")
print(f"[TRANSPILE] Surviving params: {len(tp.parameters)}/{N_PARAMS}")

n_surviving = len(tp.parameters)

# ═══ STATEVECTOR EXTRACTOR ═══
def extract_probs(param_values):
    """Extract 8Q probabilities from transpiled circuit."""
    param_dict = {p: v for p, v in zip(tp.parameters, param_values)}
    bound = tp.assign_parameters(param_dict)
    nomeas = bound.remove_final_measurements(inplace=False)
    sv = Statevector.from_instruction(nomeas)
    return sv.probabilities()[:2**NQ]

# ═══ COST FUNCTION ═══
def cost_function(param_values):
    """Cost: -p(target) — minimize this."""
    probs = extract_probs(param_values)
    return -probs[target_idx]

# ═══ SPSA ═══
print(f"\n[SPSA] Starting optimization with {n_surviving} parameters...")
print(f"[SPSA] 2000 iterations, targeting |{TARGET}⟩")

a0, c0, A = 0.2, 0.15, 50
alpha, gamma = 0.602, 0.101

# Smart initialization: q0-q3 biased toward |0⟩, q4-q7 toward |1⟩
# For Rz-SX-CZ-SX-Rz to output |0⟩: start with Rz=0
# For Rz-SX-CZ-SX-Rz to output |1⟩: need Rz=π rotation
theta = np.zeros(n_surviving)
# Add small random noise to break symmetry
theta += np.random.uniform(-0.2, 0.2, n_surviving)
best_theta, best_cost = theta.copy(), 0.0
n_iters = 5000
t_start = time.time()

for k in range(1, n_iters + 1):
    ak = a0 / (k + A) ** alpha
    ck = c0 / k ** gamma
    delta = np.random.choice([-1, 1], size=n_surviving)
    cost_plus = cost_function(theta + ck * delta)
    cost_minus = cost_function(theta - ck * delta)
    grad = (cost_plus - cost_minus) / (2 * ck * delta)
    theta -= ak * grad

    if cost_plus < best_cost:
        best_cost, best_theta = cost_plus, (theta + ck * delta).copy()
    if cost_minus < best_cost:
        best_cost, best_theta = cost_minus, (theta - ck * delta).copy()

    if k % 200 == 0 or k <= 10:
        p_gc = -best_cost
        rate = k / (time.time() - t_start)
        print(f"  iter {k:4d}/{n_iters}: p(GC)={p_gc:.6f} ({p_gc*(2**NQ):.1f}x uniform) [{rate:.0f} it/s]")

p_final = -best_cost
print(f"\n[SPSA] Final: p(GC) = {p_final:.6f} ({p_final*(2**NQ):.1f}x uniform)")

# ═══ ANALYZE ═══
probs = extract_probs(best_theta)
sorted_states = sorted(enumerate(probs), key=lambda x: -x[1])
gc_rank = next(i for i, (idx, _) in enumerate(sorted_states, 1) if idx == target_idx)
p_gc_final = probs[target_idx]

print(f"\n  Top 10 states:")
for rank, (idx, p) in enumerate(sorted_states[:10], 1):
    marker = " ← GOD_CODE" if idx == target_idx else ""
    print(f"  #{rank:2d} |{format(idx, f'0{NQ}b')}⟩ p={p:.6f}{marker}")

print(f"\n  GOD_CODE rank: #{gc_rank}, p={p_gc_final:.6f}")

# Marginals
print(f"\n  Per-qubit marginals (target={TARGET}):")
for q in range(NQ):
    target_bit = int(TARGET[q])
    p_match = sum(probs[s] for s in range(2**NQ) if ((s >> (NQ-1-q)) & 1) == target_bit)
    status = "LOCKED" if p_match > 0.95 else "STRONG" if p_match > 0.8 else "WEAK"
    print(f"    q{q} → |{target_bit}⟩: {p_match:.4f} [{status}]")

# ═══ SAVE ═══
results_path = Path(__file__).resolve().parent / "IBM_GODCODE_HEA_SIM.json"
payload = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "experiment": "godcode_hea_v2",
    "n_params": n_surviving,
    "spsa_iterations": n_iters,
    "god_code_rank": gc_rank,
    "god_code_probability": p_gc_final,
    "top_10": [{"state": format(idx, f'0{NQ}b'), "probability": float(p)} for idx, p in sorted_states[:10]],
}
results_path.write_text(json.dumps(payload, indent=2))
print(f"\nSim results saved to {results_path.name}")

if SIM_ONLY:
    print(f"\n{'=' * 80}")
    print(f"  SIMULATION COMPLETE")
    print(f"  Set IBMQ_TOKEN to deploy to IBM Marrakesh")
    print(f"{'=' * 80}")
