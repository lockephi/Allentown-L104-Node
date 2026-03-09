#!/usr/bin/env python3
"""Debug probability and unitarity in L104 quantum system."""
from l104_quantum_gate_engine import get_engine, GateCircuit
from l104_qiskit_utils import aer_backend
import numpy as np

engine = get_engine()
bell = engine.bell_pair()

# Raw statevector from L104
sv = aer_backend.run_statevector(bell)
print("=" * 60)
print("  L104 STATEVECTOR DEBUG")
print("=" * 60)
print(f"\nRaw L104 statevector: {sv}")
print(f"Type: {type(sv)}")
print(f"Shape: {sv.shape if hasattr(sv, 'shape') else len(sv)}")
print(f"Sum of |amplitudes|^2: {np.sum(np.abs(sv)**2)}")

# Compare with Qiskit directly
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
sv_qiskit = Statevector.from_instruction(qc)

print(f"\n--- Qiskit Reference ---")
print(f"Amplitudes: {sv_qiskit.data}")
print(f"Probabilities: {sv_qiskit.probabilities()}")
print(f"Sum of probs: {sum(sv_qiskit.probabilities())}")

# Check if L104 returns probabilities instead of amplitudes
print(f"\n--- Analysis ---")
print(f"L104 values look like: {'probabilities' if np.max(sv) <= 1 and np.min(sv) >= 0 else 'amplitudes'}")
print(f"If L104 returns probs, sum = {np.sum(sv)}")

# Correct interpretation: L104 might return probabilities directly
print(f"\n--- Corrected Metrics ---")
if np.sum(sv) < 1.5:  # Looks like probabilities
    print(f"Interpreting as PROBABILITIES:")
    print(f"  Sum: {np.sum(sv):.15f}")
    print(f"  |00>: {sv[0]:.6f}, |11>: {sv[3]:.6f}")
else:  # Looks like amplitudes
    print(f"Interpreting as AMPLITUDES:")
    probs = np.abs(sv) ** 2
    print(f"  Sum of |amp|^2: {np.sum(probs):.15f}")
    print(f"  |00>: {probs[0]:.6f}, |11>: {probs[3]:.6f}")

# Unitarity via gate reversibility (not statevector norm)
print(f"\n--- UNITARITY (Gate Reversibility) ---")
qc_full = GateCircuit(2)
qc_full.h(0).cx(0, 1).rz(0.5, 0).ry(0.3, 1)  # Forward
qc_full.ry(-0.3, 1).rz(-0.5, 0).cx(0, 1).h(0)  # Inverse
sv_final = aer_backend.run_statevector(qc_full)
# After U†U, should return to |00⟩
return_prob = sv_final[0] if np.sum(sv_final) < 1.5 else np.abs(sv_final[0])**2
print(f"Return to |00⟩: {return_prob:.15f}")
print(f"Unitarity error: {abs(1.0 - return_prob):.2e}")
print(f"VERDICT: {'UNITARY' if abs(1.0 - return_prob) < 1e-10 else 'NON-UNITARY'}")
