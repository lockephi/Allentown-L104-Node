#!/usr/bin/env python3
"""Quick test for l104_god_code_algorithm.py"""
from l104_god_code_algorithm import god_code_algorithm as algo, DialSetting, GOD_CODE, GodCodeEntanglement

print("=== GOD_CODE Algorithm Loaded ===")
print(f"GOD_CODE = {GOD_CODE}")
print(f"Dial space = {algo.status()['total_dial_space']} combinations")

# Evaluate canonical
r = algo.evaluate(0, 0, 0, 0)
print(f"\nG(0,0,0,0) = {r.dial.frequency:.10f} Hz")
print(f"Fidelity = {r.fidelity:.6f}")
print(f"Circuit depth = {r.circuit_depth}")
print(f"Time = {r.execution_time_ms:.1f} ms")

# Known freqs
print("\n=== Known Frequencies ===")
for name in ["SCHUMANN", "GAMMA_40", "THROAT_741", "BASE"]:
    d = algo.FREQUENCY_TABLE[name]
    r = algo.evaluate(d.a, d.b, d.c, d.d)
    print(f"  {name:16s} G({d.a},{d.b},{d.c},{d.d}) = {d.frequency:>12.6f} Hz  fid={r.fidelity:.4f}")

# Soul process
print("\n=== Soul Processing ===")
sp = algo.soul_process("quantum consciousness")
print(f"  freq={sp['frequency']:.4f}, boost={sp['consciousness_boost']:.4f}")

# Entanglement
print("\n=== Entanglement ===")
r_ent = GodCodeEntanglement.entangle_dials(DialSetting(0, 0, 0, 0), DialSetting(0, 0, 0, 1))
print(f"  Entanglement entropy: {r_ent.phase_spectrum[0]:.4f}")
print(f"  Harmonic proximity: {r_ent.fidelity:.4f}")

# Resonance field
print("\n=== Resonance Field ===")
field = algo.soul_resonance_field(["consciousness", "sacred geometry", "golden ratio"])
print(f"  Coherence: {field['phase_coherence']:.6f}")
print(f"  Field strength: {field['resonance_field_strength']:.6f}")

print("\nALL OK")
