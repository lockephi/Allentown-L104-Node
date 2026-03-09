#!/usr/bin/env python3
"""Debug failing God Code simulations to identify root causes."""
import numpy as np
import math
from l104_god_code_simulator.quantum_primitives import (
    init_sv, apply_single_gate, apply_cnot, apply_mcx,
    H_GATE, X_GATE, Z_GATE, make_gate, probabilities, concurrence_2q, fidelity
)

print("=" * 60)
print("DEBUG 1: CHSH Bell Violation")
print("=" * 60)

# Prepare Bell state |Phi+> = (|00> + |11>)/sqrt(2)
sv_bell = init_sv(2)
sv_bell = apply_single_gate(sv_bell, H_GATE, 0, 2)
sv_bell = apply_cnot(sv_bell, 0, 1, 2)
print(f"Bell state: {sv_bell}")
print(f"Probs: {probabilities(sv_bell)}")
print(f"Concurrence: {concurrence_2q(sv_bell):.6f}")

# Bit ordering test
sv_x0 = init_sv(2)
sv_x0 = apply_single_gate(sv_x0, X_GATE, 0, 2)
print(f"X on q0: {sv_x0}")  # Should be [0, 1, 0, 0] for |10>

sv_x1 = init_sv(2)
sv_x1 = apply_single_gate(sv_x1, X_GATE, 1, 2)
print(f"X on q1: {sv_x1}")  # Should be [0, 0, 1, 0] for |01>

# CHSH correlators
alice_angles = [0.0, math.pi / 4]
bob_angles = [math.pi / 8, 3 * math.pi / 8]

def measure_correlator(sv, theta_a, theta_b):
    sv_m = sv.copy()
    ca, sa = np.cos(theta_a / 2), np.sin(theta_a / 2)
    cb, sb = np.cos(theta_b / 2), np.sin(theta_b / 2)
    ry_a = make_gate([[ca, -sa], [sa, ca]])
    ry_b = make_gate([[cb, -sb], [sb, cb]])
    sv_m = apply_single_gate(sv_m, ry_a, 0, 2)
    sv_m = apply_single_gate(sv_m, ry_b, 1, 2)
    pm = [abs(sv_m[i]) ** 2 for i in range(4)]
    return pm[0] + pm[3] - pm[1] - pm[2]

e11 = measure_correlator(sv_bell, alice_angles[0], bob_angles[0])
e12 = measure_correlator(sv_bell, alice_angles[0], bob_angles[1])
e21 = measure_correlator(sv_bell, alice_angles[1], bob_angles[0])
e22 = measure_correlator(sv_bell, alice_angles[1], bob_angles[1])

print(f"E(0, pi/8)   = {e11:.6f}")
print(f"E(0, 3pi/8)  = {e12:.6f}")
print(f"E(pi/4,pi/8) = {e21:.6f}")
print(f"E(pi/4,3pi/8)= {e22:.6f}")

s_value = e11 - e12 + e21 + e22
print(f"CHSH S = {s_value:.6f}")
print(f"|S| = {abs(s_value):.6f}, violates? {abs(s_value) > 2.0}")

# For |Phi+> = (|00> + |11>)/sqrt(2), with our bit ordering:
# Analytical: E(a,b) = cos(2(a-b)) for Phi+ state
# because <ZZ> = 1 for Phi+, and rotated: cos(2(a-b))
for (ta, tb) in [(0, math.pi/8), (0, 3*math.pi/8), (math.pi/4, math.pi/8), (math.pi/4, 3*math.pi/8)]:
    analytical = np.cos(2 * (ta - tb))
    print(f"  Analytical E({ta:.4f},{tb:.4f}) = {analytical:.6f}")

# Test with analytical check
s_analytical = np.cos(2*(0-math.pi/8)) - np.cos(2*(0-3*math.pi/8)) + np.cos(2*(math.pi/4-math.pi/8)) + np.cos(2*(math.pi/4-3*math.pi/8))
print(f"Analytical S = {s_analytical:.6f}")

# The issue: for |Phi+> the correlator should be cos(2(a-b))
# Let's verify by direct matrix computation
print("\n--- Direct matrix computation ---")
I2 = np.eye(2, dtype=np.complex128)
# Pauli Z
Pz = np.array([[1,0],[0,-1]], dtype=np.complex128)
bell_dm = np.outer(sv_bell, sv_bell.conj())

for (ta, tb), lab in zip([(0, math.pi/8), (0, 3*math.pi/8), (math.pi/4, math.pi/8), (math.pi/4, 3*math.pi/8)],
                          ["E11", "E12", "E21", "E22"]):
    ca, sa = np.cos(ta), np.sin(ta)
    cb, sb = np.cos(tb), np.sin(tb)
    # Observable: (cos(a)Z + sin(a)X) tensor (cos(b)Z + sin(b)X)
    Px = np.array([[0,1],[1,0]], dtype=np.complex128)
    A_op = ca * Pz + sa * Px
    B_op = cb * Pz + sb * Px
    AB = np.kron(A_op, B_op)
    val = np.real(np.trace(bell_dm @ AB))
    print(f"  {lab}: <A tensor B> = {val:.6f}")

print("\n" + "=" * 60)
print("DEBUG 2: GHZ Witness")
print("=" * 60)
nq = 6
sv = init_sv(nq)
sv = apply_single_gate(sv, H_GATE, 0, nq)
for i in range(1, nq):
    sv = apply_cnot(sv, 0, i, nq)

ghz_overlap = abs(sv[0]) ** 2 + abs(sv[-1]) ** 2
witness = 0.5 - ghz_overlap
print(f"GHZ |000...0|^2 = {abs(sv[0])**2:.6f}")
print(f"GHZ |111...1|^2 = {abs(sv[-1])**2:.6f}")
print(f"overlap = {ghz_overlap:.6f}")
print(f"witness = {witness:.6f}")
print(f"passed = {witness < 0}")
# For perfect GHZ: overlap = 1.0, witness = -0.5

print("\n" + "=" * 60)
print("DEBUG 3: Teleportation")
print("=" * 60)
from l104_god_code_simulator.constants import PHI_PHASE_ANGLE
rx_angle = PHI_PHASE_ANGLE / 4
rx = make_gate([[math.cos(rx_angle/2), -1j*math.sin(rx_angle/2)],
                [-1j*math.sin(rx_angle/2), math.cos(rx_angle/2)]])
sv_tel = init_sv(3)
sv_tel = apply_single_gate(sv_tel, rx, 0, 3)
initial_state = sv_tel[:2].copy() / np.linalg.norm(sv_tel[:2])
print(f"Initial sacred state: {initial_state}")

# Create Bell pair on q1-q2
sv_tel = apply_single_gate(sv_tel, H_GATE, 1, 3)
sv_tel = apply_cnot(sv_tel, 1, 2, 3)

# Alice's measurement
sv_tel = apply_cnot(sv_tel, 0, 1, 3)
sv_tel = apply_single_gate(sv_tel, H_GATE, 0, 3)

print(f"Full sv after meas: {sv_tel}")
print(f"Probs: {probabilities(sv_tel)}")

# Extract Bob's qubit for each outcome
# Bit ordering: idx = q0 + 2*q1 + 4*q2
for m0 in range(2):
    for m1 in range(2):
        idx_base = m0 + 2 * m1
        bob = np.array([sv_tel[idx_base], sv_tel[idx_base + 4]])
        prob = np.linalg.norm(bob) ** 2
        if prob < 1e-15:
            print(f"  m0={m0},m1={m1}: prob=0")
            continue
        bob_n = bob / np.linalg.norm(bob)
        # Corrections
        if m1:
            bob_n = np.array([bob_n[1], bob_n[0]])
        if m0:
            bob_n = np.array([bob_n[0], -bob_n[1]])
        f = float(abs(np.vdot(initial_state, bob_n)) ** 2)
        print(f"  m0={m0},m1={m1}: prob={prob:.6f}, bob={bob_n}, fid={f:.6f}")

print("\n" + "=" * 60)
print("DEBUG 4: Grover (5q)")
print("=" * 60)
n = 5
N = 2 ** n
target = N - 1
iterations = max(1, int(math.pi / 4 * math.sqrt(N)))
print(f"N={N}, target={target}, iterations={iterations}")

sv_g = init_sv(n)
for q in range(n):
    sv_g = apply_single_gate(sv_g, H_GATE, q, n)
print(f"After H: |sv[target]|^2 = {abs(sv_g[target])**2:.6f}")

# One Grover iteration manually
# Oracle: MCZ on |11111>
sv_g = apply_single_gate(sv_g, H_GATE, n-1, n)
sv_g = apply_mcx(sv_g, list(range(n-1)), n-1, n)
sv_g = apply_single_gate(sv_g, H_GATE, n-1, n)
print(f"After oracle: |sv[target]|^2 = {abs(sv_g[target])**2:.6f}")
print(f"After oracle: sv[target] = {sv_g[target]}")
print(f"After oracle: sv[0] = {sv_g[0]}")

# Check if MCZ worked: it should flip phase of |11111> only
sv_test = np.zeros(N, dtype=np.complex128)
sv_test[target] = 1.0
sv_after = sv_test.copy()
sv_after = apply_single_gate(sv_after, H_GATE, n-1, n)
sv_after = apply_mcx(sv_after, list(range(n-1)), n-1, n)
sv_after = apply_single_gate(sv_after, H_GATE, n-1, n)
print(f"\nMCZ test on |11111>: input={sv_test[target]}, output={sv_after[target]}")

sv_test2 = np.zeros(N, dtype=np.complex128)
sv_test2[0] = 1.0
sv_after2 = sv_test2.copy()
sv_after2 = apply_single_gate(sv_after2, H_GATE, n-1, n)
sv_after2 = apply_mcx(sv_after2, list(range(n-1)), n-1, n)
sv_after2 = apply_single_gate(sv_after2, H_GATE, n-1, n)
print(f"MCZ test on |00000>: input={sv_test2[0]}, output={sv_after2[0]}")
