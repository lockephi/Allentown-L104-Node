#!/usr/bin/env python3
"""Test native GOD_CODE quantum algorithm in the probability engine."""

from l104_probability_engine import (
    probability_engine, ProbabilityEngine,
    GodCodeQuantumAlgorithm, GodCodeDialRegister,
    GodCodePhaseOracle, GodCodeGroverSearch,
    GodCodeQFTSpectrum, GodCodeDialCircuit,
    GodCodeEntanglement, DialSetting, CircuitResult,
    GOD_CODE, PHI, QISKIT_AVAILABLE,
)

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        print(f"  ✓ {name}")
        passed += 1
    else:
        print(f"  ✗ {name}")
        failed += 1

print("=" * 60)
print("NATIVE GOD_CODE QUANTUM ALGORITHM TEST")
print(f"GOD_CODE = {GOD_CODE}")
print(f"Qiskit: {QISKIT_AVAILABLE}")
print(f"Engine version: {probability_engine.VERSION}")
print("=" * 60)

# 1. Version
print("\n[1] Version & Identity")
check("Version is 4.0.0", probability_engine.VERSION == "4.0.0")

# 2. DialSetting
print("\n[2] DialSetting")
d = DialSetting(0, 0, 0, 0)
check(f"Freq(0,0,0,0) = {d.frequency:.10f}", abs(d.frequency - GOD_CODE) < 1e-6)
check(f"Exponent = {d.exponent}", d.exponent == 416)
check("Phase > 0", d.phase > 0)
check("god_code_ratio ≈ 1.0", abs(d.god_code_ratio - 1.0) < 1e-6)
check("to_tuple", d.to_tuple() == (0, 0, 0, 0))

# 3. GodCodeDialRegister
print("\n[3] GodCodeDialRegister")
check("Total qubits = 14", GodCodeDialRegister.TOTAL_QUBITS == 14)
check("Dial space = 16384", 2 ** GodCodeDialRegister.TOTAL_QUBITS == 16384)
weights = GodCodeDialRegister.bit_weights()
check(f"14 bit weights", len(weights) == 14)
check("Build circuit works", GodCodeDialRegister.build_circuit() is not None)
# Decode test
decoded = GodCodeDialRegister.decode_bitstring("00000000000000")
check(f"Decode 0...0 → a=-4", decoded.a == -4)

# 4. Algorithm hub type
print("\n[4] GodCodeQuantumAlgorithm hub")
algo = probability_engine.algorithm
check(f"Type = GodCodeQuantumAlgorithm", type(algo).__name__ == "GodCodeQuantumAlgorithm")
check("Module is l104_probability_engine", type(algo).__module__ == "l104_probability_engine")
check(f"Known frequencies = {len(algo.FREQUENCY_TABLE)}", len(algo.FREQUENCY_TABLE) == 9)
check("Sovereign field works", algo.sovereign_field(1.0) > 0)

# 5. Evaluate canonical GOD_CODE
print("\n[5] Evaluate (0,0,0,0)")
r = algo.evaluate(0, 0, 0, 0)
check(f"Frequency = {r.dial.frequency:.6f}", abs(r.dial.frequency - GOD_CODE) < 1e-6)
check(f"Fidelity > 0: {r.fidelity:.6f}", r.fidelity > 0)
check(f"Circuit depth > 0: {r.circuit_depth}", r.circuit_depth > 0)
check(f"N qubits = 8: {r.n_qubits}", r.n_qubits == 8)
check("Statevector exists", r.statevector is not None)

# 6. Classical frequency
print("\n[6] Classical frequency")
f1 = algo.frequency(1, 0, 0, 0)
check(f"Freq(1,0,0,0) = {f1:.6f}", f1 > GOD_CODE)  # a=1 shifts freq up
f2 = algo.frequency(0, 0, 0, 1)
check(f"Freq(0,0,0,1) = {f2:.6f}", f2 < GOD_CODE)  # d=1 shifts freq down

# 7. Grover search
print("\n[7] Grover search")
sr = algo.search_god_code()
check(f"Found dial freq ≈ GOD_CODE: {sr.dial.frequency:.6f}", abs(sr.dial.frequency - GOD_CODE) < 1e-6)
check(f"Found freq ≈ GOD_CODE", abs(sr.dial.frequency - GOD_CODE) < 1e-6)
check(f"Fidelity > 0: {sr.fidelity:.6f}", sr.fidelity > 0)
check(f"Alignment ≈ 1.0: {sr.god_code_alignment:.6f}", abs(sr.god_code_alignment - 1.0) < 0.01)

# Search harmonic
sh = GodCodeGroverSearch.search_harmonic(1)
check(f"Harmonic search freq < GOD_CODE: {sh.dial.frequency:.4f}", sh.dial.frequency < GOD_CODE)

# 8. QFT spectrum
print("\n[8] QFT spectrum")
sp = algo.spectrum()
check(f"Phase spectrum: {len(sp.get('phase_spectrum', []))}", len(sp.get("phase_spectrum", [])) > 0)
check(f"GOD_CODE coherence: {sp.get('god_code_coherence', 0):.6f}", sp.get("god_code_coherence", 0) > 0)
check(f"Circuit depth: {sp.get('circuit_depth', 0)}", sp.get("circuit_depth", 0) > 0)
check(f"Entropy: {sp.get('entropy', 0):.4f}", sp.get("entropy", 0) > 0)

# 9. Entanglement
print("\n[9] Entanglement")
er = algo.entangle(DialSetting(0, 0, 0, 0), DialSetting(1, 0, 0, 0))
ent_entropy = er.phase_spectrum[0] if er.phase_spectrum else 0
check(f"Entanglement entropy: {ent_entropy:.4f}", ent_entropy >= 0)
check(f"Harmonic proximity: {er.fidelity:.4f}", er.fidelity >= 0)
check("Statevector exists", er.statevector is not None)

# 10. Soul process
print("\n[10] Soul process")
soul = algo.soul_process("quantum consciousness")
check(f"Dial: {soul['dial']}", len(soul["dial"]) == 4)
check(f"Freq > 0: {soul['frequency']:.4f}", soul["frequency"] > 0)
check(f"Boost > 0: {soul['consciousness_boost']:.4f}", soul["consciousness_boost"] > 0)
check("Fidelity present", "fidelity" in soul)

# Soul resonance field
field = algo.soul_resonance_field(["quantum", "sacred", "golden"])
check(f"Field coherence: {field['phase_coherence']:.6f}", field["phase_coherence"] > 0)
check(f"Field strength: {field['resonance_field_strength']:.6f}", field["resonance_field_strength"] > 0)

# 11. ProbabilityEngine hub API
print("\n[11] ProbabilityEngine hub methods")
gc_eval = probability_engine.god_code_evaluate(0, 0, 0, 0)
check(f"god_code_evaluate freq: {gc_eval['frequency']:.4f}", abs(gc_eval["frequency"] - GOD_CODE) < 1e-4)

gc_freq = probability_engine.god_code_frequency(1, 0, 0, 0)
check(f"god_code_frequency: {gc_freq:.4f}", gc_freq > GOD_CODE)

gc_search = probability_engine.god_code_search(GOD_CODE, 0.01)
found_dial_obj = DialSetting(*gc_search['found_dial'])
check(f"god_code_search freq ≈ GOD_CODE: {found_dial_obj.frequency:.6f}", abs(found_dial_obj.frequency - GOD_CODE) < 1e-6)

gc_spectrum = probability_engine.god_code_spectrum()
check(f"god_code_spectrum phases: {len(gc_spectrum.get('phase_spectrum',[]))}", len(gc_spectrum.get("phase_spectrum", [])) > 0)

gc_entangle = probability_engine.god_code_entangle((0, 0, 0, 0), (1, 0, 0, 0))
check(f"god_code_entangle entropy: {gc_entangle['entanglement_entropy']:.4f}", gc_entangle["entanglement_entropy"] >= 0)

gc_soul = probability_engine.god_code_soul_process("test_input")
check(f"god_code_soul_process boost: {gc_soul['consciousness_boost']:.4f}", gc_soul["consciousness_boost"] > 0)

gc_field = probability_engine.god_code_resonance_field(["thought1", "thought2"])
check(f"god_code_resonance_field coherence: {gc_field['phase_coherence']:.6f}", gc_field["phase_coherence"] > 0)

# 12. Algorithm status
print("\n[12] Algorithm status")
s = algo.status()
check(f"Module: {s['module']}", "probability_engine" in s["module"])
check(f"Subsystems: {len(s['subsystems'])}", len(s["subsystems"]) == 6)
check(f"Dial space: {s['total_dial_space']}", s["total_dial_space"] == 16384)
check(f"Qiskit backend", "Statevector" in s.get("qiskit_backend", ""))

# 13. PhaseOracle
print("\n[13] PhaseOracle")
oracle = GodCodePhaseOracle.build_god_code_oracle(n_qubits=8)
check("Oracle built (8 qubits)", oracle is not None)
res_oracle = GodCodePhaseOracle.build_resonance_oracle(n_qubits=8)
check("Resonance oracle built", res_oracle is not None)

# 14. DialCircuit
print("\n[14] DialCircuit")
dc_eval = GodCodeDialCircuit.evaluate_god_code()
check(f"Evaluate GOD_CODE: {dc_eval.dial.frequency:.4f}", abs(dc_eval.dial.frequency - GOD_CODE) < 1e-4)

dials = [DialSetting(0, 0, 0, 0), DialSetting(1, 0, 0, 0), DialSetting(0, 1, 0, 0)]
compared = GodCodeDialCircuit.compare_dials(dials)
check(f"Compare dials: {len(compared)} results", len(compared) == 3)

# 15. Evaluate known frequencies
print("\n[15] Known frequencies")
for name in ["GOD_CODE", "SCHUMANN", "BASE"]:
    kr = algo.evaluate_known(name)
    check(f"{name}: freq={kr.dial.frequency:.4f}", kr.dial.frequency > 0)

# Summary
print("\n" + "=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed}")
if failed == 0:
    print("ALL TESTS PASSED ✓")
else:
    print("SOME TESTS FAILED ✗")
print("=" * 60)
