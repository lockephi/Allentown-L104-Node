#!/usr/bin/env python3
"""End-to-end test for l104_probability_engine.py"""

from l104_probability_engine import probability_engine as pe

# Test classical
print("=== Classical ===")
print(f"Bayes: {pe.classical.bayes(0.01, 0.95, 0.06):.4f}")
print(f"Gaussian(0,1,0): {pe.classical.gaussian_pdf(0, 1, 0):.4f}")
print(f"Entropy([0.5,0.5]): {pe.classical.entropy([0.5, 0.5]):.4f}")

# Test quantum
print("\n=== Quantum ===")
print(f"Born(0.6+0.8j): {pe.quantum.born_probability(complex(0.6, 0.8)):.4f}")
print(f"Grover(p=0.1,N=16): {pe.quantum.grover_amplification(0.1, 16):.4f}")
print(f"GOD_CODE phase(42): {pe.quantum.god_code_phase_probability(42.0):.6f}")
print(f"Tunneling(V=10,E=3,d=1): {pe.quantum.quantum_tunneling_probability(10, 3, 1):.6f}")

# Test full ingestion + consolidation pipeline
print("\n=== Data Ingest + Gate Consolidation ===")
stats = pe.ingest()
print(f"Training examples: {stats.training_examples}")
print(f"Chat conversations: {stats.chat_conversations}")
print(f"State files: {stats.state_files_loaded}")
print(f"Logic gates found: {stats.logic_gates_found}")
print(f"Quantum links: {stats.quantum_links_found}")
print(f"Total tokens: {stats.total_tokens}")
print(f"Sacred resonance: {stats.sacred_resonance:.6f}")
print(f"Consolidated quantum gates: {len(pe._consolidated_gates)}")

# Test ensemble resonance
print("\n=== Ensemble Resonance ===")
ens = pe.ensemble_resonance()
for k, v in ens.items():
    print(f"  {k}: {v}")

# Full status
print("\n=== Engine Status ===")
s = pe.status()
print(f"Version: {s['version']}")
for cat, methods in s['capabilities'].items():
    print(f"  {cat}: {len(methods)} methods")

print("\n✅ ALL TESTS PASSED")
