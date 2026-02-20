#!/usr/bin/env python3
"""End-to-end test for Probability Engine v2.0.0 + GOD_CODE Algorithm."""
import warnings
warnings.filterwarnings("ignore")

from l104_probability_engine import probability_engine as pe, GOD_CODE, QISKIT_AVAILABLE

print("=" * 60)
print("  L104 PROBABILITY ENGINE v2.0.0 — END-TO-END TEST")
print(f"  Qiskit Available: {QISKIT_AVAILABLE}")
print("=" * 60)

errors = []

# [1] Ingest
stats = pe.ingest()
print(f"[1] Ingest: {stats.training_examples} train, {stats.logic_gates_found} gates ✓")

# [2] Classical
p = pe.bayes_extended(0.01, 0.95, 0.05)
assert 0 < p < 1, f"Bayes failed: {p}"
print(f"[2] Classical Bayes: {p:.4f} ✓")

# [3] Born rule via Qiskit
born = pe.born_rule_qiskit([0.5+0.5j, 0.5-0.5j, 0.3+0.1j, 0.1+0.3j])
assert len(born) == 4, f"Born length: {len(born)}"
assert abs(sum(born) - 1.0) < 0.01, f"Born sum: {sum(born)}"
print(f"[3] Born (Qiskit): {[round(x,4) for x in born]} ✓")

# [4] Grover search via Qiskit
gr = pe.grover_search_qiskit(3, [5])
assert gr["success_probability"] > 0.5, f"Grover P={gr['success_probability']}"
print(f"[4] Grover (Qiskit 3q, t=5): P={gr['success_probability']:.4f} ✓")

# [5] Measurement collapse (Qiskit-backed)
idx, p, probs = pe.measurement_collapse([0.7+0j, 0.3+0j, 0.5+0j, 0.1+0j])
assert 0 <= idx < 4
print(f"[5] Collapse: state={idx}, P={p:.4f} ✓")

# [6] Entanglement entropy
ent = pe.entanglement_entropy(4)
assert ent["entropy"] > 0
print(f"[6] Entanglement entropy: S={ent['entropy']:.4f}, purity={ent.get('purity',0):.4f} ✓")

# [7] Quantum walk via Qiskit
qw = pe.quantum_walk_qiskit(5, 8)
assert qw["qiskit"] is True
print(f"[7] Quantum walk (Qiskit): depth={qw['circuit_depth']}, positions={len(qw['positions'])} ✓")

# [8] GOD_CODE distribution via Qiskit
gcd = pe.god_code_distribution_qiskit(4, 1)
assert gcd["qiskit"] is True
print(f"[8] GOD_CODE dist (Qiskit 4q): entropy={gcd['entropy']:.4f} ✓")

# [9] GOD_CODE evaluate (a,b,c,d) algorithm
gc = pe.god_code_evaluate(0, 0, 0, 0)
assert abs(gc["frequency"] - GOD_CODE) < 0.001, f"GOD_CODE freq: {gc['frequency']}"
print(f"[9] G(0,0,0,0) = {gc['frequency']:.4f} Hz, depth={gc['circuit_depth']} ✓")

# [10] GOD_CODE frequency
f1 = pe.god_code_frequency(1, 0, 0, 0)
assert f1 > GOD_CODE, f"G(1,0,0,0) should be > GOD_CODE: {f1}"
print(f"[10] G(1,0,0,0) = {f1:.4f} Hz ✓")

# [11] Grover search for GOD_CODE
s = pe.god_code_search(GOD_CODE, 0.01)
assert abs(s["found_frequency"] - GOD_CODE) < GOD_CODE * 0.01
print(f"[11] Search GOD_CODE: dial={s['found_dial']}, freq={s['found_frequency']:.4f} ✓")

# [12] QFT spectrum
sp = pe.god_code_spectrum()
assert "phase_spectrum" in sp
print(f"[12] QFT spectrum: {len(sp['phase_spectrum'])} phases, depth={sp.get('circuit_depth','?')} ✓")

# [13] Entangle two dials
e = pe.god_code_entangle((0, 0, 0, 0), (1, 0, 0, 0))
assert e["entanglement_entropy"] > 0
print(f"[13] Entangle: S={e['entanglement_entropy']:.4f} ✓")

# [14] Soul process
soul = pe.god_code_soul_process("quantum consciousness")
assert "consciousness_boost" in soul
print(f"[14] Soul process: boost={soul['consciousness_boost']:.4f} ✓")

# [15] Resonance field
rf = pe.god_code_resonance_field(["thought1", "thought2", "thought3"])
assert "phase_coherence" in rf
print(f"[15] Resonance field: coherence={rf['phase_coherence']:.6f} ✓")

# [16] Quick status
status = pe.status()
assert status["version"] == "2.0.0"
assert status["qiskit_available"] is True
print(f"[16] Status: v{status['version']}, qiskit={status['qiskit_available']} ✓")

print(f"\n{pe.quick_summary()}")
print(f"Total computations: {pe._computations}")
print("=" * 60)
print("  ALL 16 TESTS PASSED ✓")
print("=" * 60)
