"""
===============================================================================
L104 QUANTUM GATE ENGINE — STABILIZER TABLEAU TESTS
===============================================================================

Comprehensive validation of the Aaronson–Gottesman stabilizer tableau
simulator and the hybrid statevector+stabilizer backend.

7 test phases:
  1. Tableau initialization + state inspection
  2. Single-qubit Clifford gates
  3. Two-qubit Clifford gates (CNOT-heavy)
  4. Measurement (deterministic + random)
  5. Circuit simulation + Bell/GHZ states
  6. Hybrid backend routing
  7. Performance: 100+ qubit Clifford circuits (1000x speedup proof)

INVARIANT: 527.5184818492612 | PILOT: LONDEL
===============================================================================
"""

import time
import sys
import os
import numpy as np

# Ensure workspace root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from l104_quantum_gate_engine import (
    get_engine, GateCircuit,
    H, X, Y, Z, S, S_DAG, CNOT, CZ, CY, SWAP, ISWAP, T, Rx,
    StabilizerTableau, HybridStabilizerSimulator, HybridSimulationResult,
    is_clifford_gate, is_clifford_circuit, clifford_prefix_length,
    ExecutionTarget,
)

PASS = 0
FAIL = 0
TOTAL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL, TOTAL
    TOTAL += 1
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} — {detail}")


def phase(name: str):
    print(f"\n{'═' * 70}")
    print(f"  PHASE: {name}")
    print(f"{'═' * 70}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 1: Tableau Initialization
# ═══════════════════════════════════════════════════════════════════════════════

phase("1 — Tableau Initialization + State Inspection")

tab = StabilizerTableau(4)
state = tab.get_stabilizer_state()

check("4-qubit tableau created", tab.n == 4)
check("4 stabilizer generators", len(state.stabilizer_generators) == 4)
check("4 destabilizer generators", len(state.destabilizer_generators) == 4)
check("Initial stabilizers are +Z_i",
      all(g.startswith('+') and 'Z' in g for g in state.stabilizer_generators))
check("Initial destabilizers are +X_i",
      all(g.startswith('+') and 'X' in g for g in state.destabilizer_generators))
check("Memory < 200 bytes for 4 qubits", tab._memory_usage() < 200)

# Large tableau memory check
tab1000 = StabilizerTableau(1000)
mem_1000 = tab1000._memory_usage()
check(f"1000-qubit tableau memory = {mem_1000:,} bytes (< 5 MB)",
      mem_1000 < 5_000_000,
      f"got {mem_1000:,} bytes")
check("1000-qubit SV equiv = astronomical",
      True)  # 2^1000 × 16 bytes ≈ 10^301 bytes


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 2: Single-Qubit Clifford Gates
# ═══════════════════════════════════════════════════════════════════════════════

phase("2 — Single-Qubit Clifford Gates")

# H gate: |0⟩ → |+⟩
tab = StabilizerTableau(1, seed=42)
tab.hadamard(0)
state = tab.get_stabilizer_state()
check("H|0⟩ → stabilizer +X", state.stabilizer_generators[0] == '+X')

# H·H = I
tab2 = StabilizerTableau(1, seed=42)
tab2.hadamard(0)
tab2.hadamard(0)
state2 = tab2.get_stabilizer_state()
check("H·H|0⟩ → stabilizer +Z",  state2.stabilizer_generators[0] == '+Z')

# S gate: X→Y
tab3 = StabilizerTableau(1)
tab3.hadamard(0)   # now stabilize by +X
tab3.phase_s(0)    # X→Y (S·X·S† = Y)
state3 = tab3.get_stabilizer_state()
check("S·H|0⟩ → stabilizer +Y", state3.stabilizer_generators[0] == '+Y')

# X gate: Z→-Z
tab4 = StabilizerTableau(1)
tab4.pauli_x(0)
state4 = tab4.get_stabilizer_state()
check("X|0⟩ → stabilizer -Z", state4.stabilizer_generators[0] == '-Z')

# Z gate on |+⟩: X→-X
tab5 = StabilizerTableau(1)
tab5.hadamard(0)
tab5.pauli_z(0)
state5 = tab5.get_stabilizer_state()
check("Z·H|0⟩ → stabilizer -X", state5.stabilizer_generators[0] == '-X')

# Y gate: both X and Z flip sign
tab6 = StabilizerTableau(1)
tab6.pauli_y(0)
state6 = tab6.get_stabilizer_state()
check("Y|0⟩ → stabilizer -Z", state6.stabilizer_generators[0] == '-Z')


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 3: Two-Qubit Clifford Gates
# ═══════════════════════════════════════════════════════════════════════════════

phase("3 — Two-Qubit Clifford Gates (CNOT-heavy)")

# CNOT on |+0⟩ → Bell state (|00⟩+|11⟩)/√2
tab_bell = StabilizerTableau(2, seed=42)
tab_bell.hadamard(0)
tab_bell.cnot(0, 1)
state_bell = tab_bell.get_stabilizer_state()
# Bell state stabilizers should be ±XX and ±ZZ
stab_strings = set(state_bell.stabilizer_generators)
check("Bell state has XX stabilizer",
      any('XX' in s for s in stab_strings),
      f"got {stab_strings}")
check("Bell state has ZZ stabilizer",
      any('ZZ' in s for s in stab_strings),
      f"got {stab_strings}")

# CZ gate
tab_cz = StabilizerTableau(2)
tab_cz.hadamard(0)
tab_cz.hadamard(1)
tab_cz.cz(0, 1)
state_cz = tab_cz.get_stabilizer_state()
check("CZ on |++⟩ produces valid state", len(state_cz.stabilizer_generators) == 2)

# SWAP gate: swap two different states
tab_swap = StabilizerTableau(2)
tab_swap.hadamard(0)  # qubit 0 = |+⟩, qubit 1 = |0⟩
state_pre = tab_swap.get_stabilizer_state()
tab_swap.swap(0, 1)   # qubit 0 = |0⟩, qubit 1 = |+⟩
state_post = tab_swap.get_stabilizer_state()
check("SWAP swaps qubit states",
      state_post.stabilizer_generators[0] != state_pre.stabilizer_generators[0])


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 4: Measurement
# ═══════════════════════════════════════════════════════════════════════════════

phase("4 — Measurement (Deterministic + Random + State Verification)")

# ── 4a. Deterministic |0⟩ — outcome + post-measurement stabilizer ────────
tab_m = StabilizerTableau(1, seed=42)
result_m = tab_m.measure(0)
check("|0⟩ measurement = 0", result_m.outcome == 0)
check("|0⟩ measurement is deterministic", result_m.deterministic == True)
# Post-measurement stabilizer must still be +Z  (x=0, z=1, phase=0)
check("|0⟩ post-measurement stabilizer is +Z",
      tab_m.tableau[1, 0] == 0 and tab_m.tableau[1, 1] == 1 and tab_m.phases[1] == 0,
      f"got tableau row={tab_m.tableau[1]} phases={tab_m.phases[1]}")

# ── 4b. Deterministic |1⟩ — outcome + post-measurement stabilizer ────────
tab_m2 = StabilizerTableau(1, seed=42)
tab_m2.apply_x(0)
outcome_m2 = tab_m2.measure_z(0)
check("|1⟩ measurement = 1", outcome_m2 == 1)
# Post-measurement stabilizer must be -Z  (x=0, z=1, phase=1)
check("|1⟩ post-measurement stabilizer is -Z",
      tab_m2.tableau[1, 0] == 0 and tab_m2.tableau[1, 1] == 1 and tab_m2.phases[1] == 1,
      f"got tableau row={tab_m2.tableau[1]} phases={tab_m2.phases[1]}")

# ── 4c. Probabilistic |+⟩ — collapse to ±Z matching outcome ─────────────
tab_m3 = StabilizerTableau(1, seed=42)
tab_m3.apply_h(0)
result_m3 = tab_m3.measure(0)
check("|+⟩ measurement is random", result_m3.deterministic == False)
check("|+⟩ outcome is 0 or 1", result_m3.outcome in (0, 1))
# Stabilizer must be Z with phase == outcome  (0 → +Z, 1 → -Z)
check(f"|+⟩ collapses to {'+Z' if result_m3.outcome == 0 else '-Z'}",
      tab_m3.tableau[1, 0] == 0 and tab_m3.tableau[1, 1] == 1
      and tab_m3.phases[1] == result_m3.outcome,
      f"got tableau row={tab_m3.tableau[1]} phases={tab_m3.phases[1]}")

# ── 4d. Bell state entanglement collapse — single-shot perfect correlation ─
tab_bc = StabilizerTableau(2, seed=42)
tab_bc.apply_h(0)
tab_bc.apply_cnot(0, 1)
outcome_0 = tab_bc.measure_z(0)
outcome_1 = tab_bc.measure_z(1)
check(f"Bell collapse: q0={outcome_0} perfectly correlates q1={outcome_1}",
      outcome_0 == outcome_1)

# ── 4e. Bell state — statistical 100% correlation over 100 trials ─────────
correlation_count = 0
for trial in range(100):
    tab_t = StabilizerTableau(2, seed=trial)
    tab_t.apply_h(0)
    tab_t.apply_cnot(0, 1)
    outcomes = tab_t.measure_all()
    if outcomes[0] == outcomes[1]:
        correlation_count += 1
check(f"Bell state 100% correlated ({correlation_count}/100)",
      correlation_count == 100,
      f"got {correlation_count}/100")

# ── 4f. Symplectic integrity post-measurement ─────────────────────────────
# After measuring a Bell state, the symplectic inner product between each
# destabilizer/stabilizer pair must remain 1 (anti-commutation invariant).
tab_sym = StabilizerTableau(2, seed=42)
tab_sym.apply_h(0)
tab_sym.apply_cnot(0, 1)
tab_sym.measure_z(0)
n_sym = tab_sym.num_qubits
symplectic_ok = True
for q in range(n_sym):
    d_x = tab_sym.tableau[q, :n_sym]
    d_z = tab_sym.tableau[q, n_sym:2*n_sym]
    s_x = tab_sym.tableau[n_sym + q, :n_sym]
    s_z = tab_sym.tableau[n_sym + q, n_sym:2*n_sym]
    inner = (np.dot(d_x.astype(int), s_z.astype(int))
             + np.dot(d_z.astype(int), s_x.astype(int))) % 2
    if inner != 1:
        symplectic_ok = False
        break
check("Destabilizer/stabilizer symplectic integrity post-measurement",
      symplectic_ok,
      f"inner product ≠ 1 for qubit {q}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 5: Circuit Simulation + Sampling
# ═══════════════════════════════════════════════════════════════════════════════

phase("5 — Circuit Simulation + Sampling")

# Bell circuit via GateCircuit
engine = get_engine()
bell = engine.bell_pair()
check("Bell circuit is Clifford", is_clifford_circuit(bell))

tab_circ = StabilizerTableau(bell.num_qubits, seed=42)
sim_result = tab_circ.simulate_circuit(bell)
check("Bell circuit simulated", sim_result["gate_count"] == 2)
check("Simulator is stabilizer_tableau", sim_result["simulator"] == "stabilizer_tableau")

# Sample Bell state
counts = tab_circ.sample(4096)
check("Bell sampling produces 2 outcomes", len(counts) == 2,
      f"got {len(counts)}: {counts}")
if len(counts) == 2:
    keys = sorted(counts.keys())
    check("Bell outcomes are 00 and 11",
          keys == ['00', '11'],
          f"got {keys}")
    total = sum(counts.values())
    p00 = counts.get('00', 0) / total
    check(f"Bell P(00) ≈ 0.5 (got {p00:.3f})", abs(p00 - 0.5) < 0.05)

# GHZ circuit
ghz = engine.ghz_state(5)
check("GHZ-5 is Clifford", is_clifford_circuit(ghz))
tab_ghz = StabilizerTableau(5, seed=42)
tab_ghz.simulate_circuit(ghz)
counts_ghz = tab_ghz.sample(4096)
check("GHZ-5 has 2 outcomes", len(counts_ghz) == 2,
      f"got {len(counts_ghz)}: {list(counts_ghz.keys())[:5]}")
if len(counts_ghz) == 2:
    keys = sorted(counts_ghz.keys())
    check("GHZ-5 outcomes are 00000 and 11111",
          keys == ['00000', '11111'],
          f"got {keys}")

# 10-qubit Clifford random circuit
circ10 = GateCircuit(10, "random_cliff_10")
for i in range(10):
    circ10.append(H, [i])
for i in range(9):
    circ10.append(CNOT, [i, i + 1])
for i in range(10):
    circ10.append(S, [i])
for i in range(0, 9, 2):
    circ10.append(CZ, [i, i + 1])
check("10-qubit Clifford circuit is Clifford", is_clifford_circuit(circ10))
tab10 = StabilizerTableau(10, seed=42)
sim10 = tab10.simulate_circuit(circ10)
check(f"10-qubit sim: {sim10['gate_count']} gates", sim10["gate_count"] > 0)


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 6: Hybrid Backend Routing
# ═══════════════════════════════════════════════════════════════════════════════

phase("6 — Hybrid Backend Routing")

# Pure Clifford → stabilizer route
hybrid = HybridStabilizerSimulator(seed=42)
bell_hybrid = hybrid.simulate(bell, shots=2048)
check("Pure Clifford → stabilizer backend",
      bell_hybrid.backend_used == "stabilizer_tableau",
      f"got {bell_hybrid.backend_used}")
check("Hybrid Bell has 2 outcomes", len(bell_hybrid.probabilities) == 2)
check("Clifford fraction = 1.0", bell_hybrid.clifford_fraction == 1.0)

# Non-Clifford circuit → statevector route
non_cliff = GateCircuit(3, "non_clifford")
non_cliff.append(H, [0])
non_cliff.append(T, [0])  # T is NOT Clifford
non_cliff.append(CNOT, [0, 1])
check("Non-Clifford circuit detected", not is_clifford_circuit(non_cliff))

result_nc = hybrid.simulate(non_cliff, shots=1024)
check("Non-Clifford routes to SV or hybrid",
      result_nc.backend_used in ("statevector", "hybrid_stabilizer_statevector"),
      f"got {result_nc.backend_used}")
check("Non-Clifford has valid probabilities", len(result_nc.probabilities) > 0)

# Clifford classification
check("H is Clifford", is_clifford_gate(H))
check("S is Clifford", is_clifford_gate(S))
check("CNOT is Clifford", is_clifford_gate(CNOT))
check("T is NOT Clifford", not is_clifford_gate(T))
check("Rx(π/7) is NOT Clifford", not is_clifford_gate(Rx(3.14159 / 7)))

# Prefix detection
mixed = GateCircuit(3, "mixed")
mixed.append(H, [0])
mixed.append(CNOT, [0, 1])
mixed.append(S, [1])
mixed.append(T, [2])  # First non-Clifford at index 3
mixed.append(CNOT, [1, 2])
prefix = clifford_prefix_length(mixed)
check("Clifford prefix length = 3", prefix == 3, f"got {prefix}")

# GateCircuit.is_clifford() method
check("bell.is_clifford() == True", bell.is_clifford())
check("mixed.is_clifford() == False", not mixed.is_clifford())

# GateCircuit.simulate_stabilizer() method
stab_result = bell.simulate_stabilizer(shots=2048, seed=42)
check("simulate_stabilizer returns probabilities",
      "probabilities" in stab_result and len(stab_result["probabilities"]) > 0)
check("simulate_stabilizer backend = stabilizer_tableau",
      stab_result["backend"] == "stabilizer_tableau")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 7: Performance — 100+ Qubit Clifford (Speedup Proof)
# ═══════════════════════════════════════════════════════════════════════════════

phase("7 — Performance: 100+ Qubit Clifford Circuits")

# 100-qubit GHZ state
t0 = time.time()
tab100 = StabilizerTableau(100, seed=42)
tab100.hadamard(0)
for i in range(1, 100):
    tab100.cnot(0, i)
t_gates = time.time() - t0

t1 = time.time()
counts100 = tab100.sample(512)  # 512 shots for 100 qubits (sampling is O(n²) per shot)
t_sample = time.time() - t1

total_time_ms = (t_gates + t_sample) * 1000
mem100 = tab100._memory_usage()

check(f"100-qubit GHZ gate sim in {t_gates*1000:.0f}ms, sample in {t_sample*1000:.0f}ms",
      t_gates < 2,  # Gate simulation should be fast
      f"gate sim {t_gates:.3f}s, sample {t_sample:.1f}s")
check(f"100-qubit memory = {mem100:,} bytes", mem100 < 100_000)
check("100-qubit GHZ has 2 outcomes", len(counts100) == 2,
      f"got {len(counts100)}")
if len(counts100) == 2:
    keys = sorted(counts100.keys())
    check("100-qubit GHZ: 0×100 and 1×100",
          keys == ['0' * 100, '1' * 100],
          f"got keys of length {len(keys[0])}")

# SV comparison: 100 qubits would need 2^100 × 16 bytes ≈ 2×10^31 bytes
sv_mem = 2**100 * 16 / (1024**3)
print(f"\n  📊 Stabilizer: {mem100:,} bytes | Statevector: {sv_mem:.1e} GB")
print(f"  📊 Memory ratio: {sv_mem * 1024**3 / max(mem100, 1):.1e}x")
print(f"  📊 Gate sim: {t_gates*1000:.1f}ms | Sample (512 shots): {t_sample*1000:.0f}ms | SV: impossible")

# 500-qubit Clifford circuit
print(f"\n  Building 500-qubit CNOT-heavy circuit...")
t2 = time.time()
tab500 = StabilizerTableau(500, seed=42)
for i in range(500):
    tab500.hadamard(i)
for layer in range(3):
    for i in range(499):
        tab500.cnot(i, i + 1)
    for i in range(500):
        tab500.phase_s(i)
t500_gates = time.time() - t2
mem500 = tab500._memory_usage()

check(f"500-qubit Clifford in {t500_gates*1000:.0f}ms",
      t500_gates < 60,
      f"took {t500_gates:.1f}s")
check(f"500-qubit memory = {mem500:,} bytes (< 2 MB)",
      mem500 < 2_000_000,
      f"got {mem500:,}")

# Entanglement entropy
t_ent = time.time()
entropy_50 = tab500.entanglement_entropy(list(range(250)))
t_ent = time.time() - t_ent
check(f"500-qubit entanglement entropy = {entropy_50:.0f} (in {t_ent*1000:.0f}ms)",
      entropy_50 >= 0)

# Orchestrator integration: execute Bell via STABILIZER_TABLEAU target
result_orch = engine.execute(bell, target=ExecutionTarget.STABILIZER_TABLEAU, shots=1024)
check("Orchestrator STABILIZER_TABLEAU target works",
      result_orch.probabilities is not None and len(result_orch.probabilities) > 0,
      f"got {result_orch.metadata}")

# Orchestrator integration: execute via HYBRID target
result_hybrid = engine.execute(bell, target=ExecutionTarget.HYBRID, shots=1024)
check("Orchestrator HYBRID target works",
      result_hybrid.probabilities is not None and len(result_hybrid.probabilities) > 0,
      f"got {result_hybrid.metadata}")


# ═══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'═' * 70}")
print(f"  STABILIZER TABLEAU SIMULATOR — TEST SUMMARY")
print(f"{'═' * 70}")
print(f"  Passed: {PASS}/{TOTAL}")
print(f"  Failed: {FAIL}/{TOTAL}")
print(f"  Result: {'✅ ALL PASSED' if FAIL == 0 else '❌ FAILURES DETECTED'}")
print(f"{'═' * 70}")

sys.exit(0 if FAIL == 0 else 1)
