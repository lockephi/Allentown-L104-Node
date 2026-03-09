#!/usr/bin/env python3
"""
L104 Canonical GOD_CODE Qubit — Stress Test Suite
═══════════════════════════════════════════════════════════════════════════════

Comprehensive stress test of the QPU-verified GOD_CODE qubit across:

  Phase 1:  Qubit Integrity          — singleton, phases, unitarity, eigenspectrum
  Phase 2:  Gate Algebra              — Rz composition, decomposition, inverse, powers
  Phase 3:  State Preparation         — |0⟩, |1⟩, |+⟩, |−⟩ bases, Bloch vectors
  Phase 4:  Ramsey Interferometry     — Phase readout via H-Rz-H sequences
  Phase 5:  Dial System Sweep         — All 4 dials (a,b,c,d) parametric sweep
  Phase 6:  Multi-Qubit Circuits      — 2Q entanglement, 3Q sacred, 4Q GHZ
  Phase 7:  Noise Resilience          — Depolarizing, amplitude damping, readout
  Phase 8:  QPE Phase Extraction      — 4-bit, 6-bit, 8-bit precision
  Phase 9:  Conservation Laws         — Phase additivity, unitary product, Rz† identity
  Phase 10: Cross-Package Consistency — All packages use identical phase values

═══════════════════════════════════════════════════════════════════════════════
"""

import cmath
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── Imports ──────────────────────────────────────────────────────────────────
from l104_god_code_simulator.god_code_qubit import (
    GodCodeQubit, GOD_CODE_QUBIT,
    GOD_CODE_PHASE, GOD_CODE_RZ, GOD_CODE_P,
    IRON_PHASE, OCTAVE_PHASE, PHI_CONTRIBUTION,
    PHI_PHASE, VOID_PHASE,
    IRON_RZ, PHI_RZ, OCTAVE_RZ,
    QPU_DATA, _rz, _phase_gate,
)
from l104_god_code_simulator.quantum_primitives import (
    GOD_CODE_GATE, H_GATE, X_GATE, Z_GATE,
    PHI_GATE, VOID_GATE, IRON_GATE,
    init_sv, apply_single_gate, apply_cnot, apply_cp,
    apply_swap, build_unitary,
    probabilities, entanglement_entropy, concurrence_2q,
    fidelity, bloch_vector,
)
from l104_god_code_simulator.constants import (
    GOD_CODE, PHI, TAU, VOID_CONSTANT, BASE,
    QUANTIZATION_GRAIN, OCTAVE_OFFSET,
    GOD_CODE_PHASE_ANGLE, PHI_PHASE_ANGLE, IRON_PHASE_ANGLE,
)

T0 = time.perf_counter()
PASS = 0
FAIL = 0
RESULTS = []  # (phase, name, pass, detail, ms)


def check(phase: int, name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    ms = (time.perf_counter() - T0) * 1000
    if condition:
        PASS += 1
        RESULTS.append((phase, name, True, detail, ms))
        print(f"  ✅ {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAIL += 1
        RESULTS.append((phase, name, False, detail, ms))
        print(f"  ❌ {name}" + (f"  — {detail}" if detail else ""))

TAU_2PI = 2.0 * math.pi

print("╔═══════════════════════════════════════════════════════════════════════╗")
print("║   L104 CANONICAL GOD_CODE QUBIT — STRESS TEST                       ║")
print("║   QPU-verified: ibm_torino (Heron r2) | Fidelity: 0.975             ║")
print("╚═══════════════════════════════════════════════════════════════════════╝")
print()

# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 1: QUBIT INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════════
print("═══ Phase 1: Qubit Integrity ═══")

check(1, "Singleton exists", GOD_CODE_QUBIT is not None)
check(1, "Is GodCodeQubit instance", isinstance(GOD_CODE_QUBIT, GodCodeQubit))
check(1, "GOD_CODE = 527.518...", abs(GOD_CODE - 527.5184818492612) < 1e-8)
check(1, "Phase = GOD_CODE mod 2π",
      abs(GOD_CODE_PHASE - (GOD_CODE % TAU_2PI)) < 1e-14,
      f"θ = {GOD_CODE_PHASE:.15f}")
check(1, "Phase ∈ (0, 2π)", 0 < GOD_CODE_PHASE < TAU_2PI)
check(1, "Phase ≈ 6.014 rad", abs(GOD_CODE_PHASE - 6.014) < 0.001)
check(1, "Phase ≈ 344.58°", abs(math.degrees(GOD_CODE_PHASE) - 344.58) < 0.01)

# Full verify()
v = GOD_CODE_QUBIT.verify()
check(1, "verify() — PASS", v["PASS"] is True)
check(1, "verify() — is_unitary", v["is_unitary"])
check(1, "verify() — all_on_unit_circle", v["all_on_unit_circle"])
check(1, "verify() — god_code_phase_detected", v["god_code_phase_detected"])
check(1, "verify() — decomposition conserved", v["decomposition"]["conserved"])
check(1, "verify() — QPU verified", v["qpu_verified"])
check(1, "verify() — unitarity error < 1e-14",
      v["unitarity_error"] < 1e-14,
      f"err = {v['unitarity_error']:.2e}")
check(1, "verify() — decomp matrix error < 1e-14",
      v["decomposition"]["matrix_error"] < 1e-14,
      f"err = {v['decomposition']['matrix_error']:.2e}")

# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 2: GATE ALGEBRA
# ═══════════════════════════════════════════════════════════════════════════════
print("\n═══ Phase 2: Gate Algebra ═══")

U = GOD_CODE_RZ
I2 = np.eye(2, dtype=np.complex128)

# Unitarity
check(2, "U†U = I (machine precision)",
      np.allclose(U.conj().T @ U, I2, atol=1e-15))
check(2, "UU† = I (machine precision)",
      np.allclose(U @ U.conj().T, I2, atol=1e-15))
check(2, "|det(U)| = 1.0",
      abs(abs(np.linalg.det(U)) - 1.0) < 1e-14)
check(2, "det(U) real part ≈ 1 (SU(2))",
      abs(np.linalg.det(U).real - 1.0) < 1e-10,
      f"det = {np.linalg.det(U)}")

# Decomposition: Rz(Fe) · Rz(φ) · Rz(oct) = Rz(Fe + φ + oct) = Rz(θ_GC)
product = IRON_RZ @ PHI_RZ @ OCTAVE_RZ
# Allow global phase
gp = product[0, 0] / U[0, 0] if abs(U[0, 0]) > 1e-10 else 1.0
check(2, "Rz(Fe)·Rz(φ)·Rz(oct) = Rz(θ_GC)",
      np.allclose(product / gp, U, atol=1e-14))

# Inverse
check(2, "U · U† = I",
      np.allclose(U @ U.conj().T, I2, atol=1e-15))
check(2, "Rz(θ) · Rz(−θ) = I",
      np.allclose(U @ _rz(-GOD_CODE_PHASE), I2, atol=1e-14))

# Powers: Rz(nθ) = Rz(θ)^n
for n in [2, 3, 5, 10]:
    U_n = np.linalg.matrix_power(U, n)
    U_n_direct = _rz(n * GOD_CODE_PHASE)
    # up to global phase
    if abs(U_n[0, 0]) > 1e-10:
        gp_n = U_n_direct[0, 0] / U_n[0, 0]
        ok = np.allclose(U_n_direct / gp_n, U_n, atol=1e-12)
    else:
        ok = np.allclose(U_n_direct, U_n, atol=1e-12)
    check(2, f"Rz(θ)^{n} = Rz({n}θ)", ok)

# Phase gate equivalence: P(θ) = e^{iθ/2} · Rz(θ)
P = GOD_CODE_P
expected_P = np.exp(1j * GOD_CODE_PHASE / 2) * U
check(2, "P(θ) = e^{iθ/2} · Rz(θ)",
      np.allclose(P, expected_P, atol=1e-14))

# Eigenvalue spectrum
eigvals = np.linalg.eigvals(U)
phases = sorted([cmath.phase(e) for e in eigvals])
check(2, "Eigenvalue phases = ±θ/2",
      abs(abs(phases[0]) - GOD_CODE_PHASE / 2) < 1e-12 and
      abs(abs(phases[1]) - GOD_CODE_PHASE / 2) < 1e-12,
      f"phases = {phases[0]:.6f}, {phases[1]:.6f}")

# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 3: STATE PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n═══ Phase 3: State Preparation ═══")

for label in ["|0>", "|1>", "|+>", "|->",]:
    sv = GOD_CODE_QUBIT.prepare(label)
    norm = np.linalg.norm(sv)
    check(3, f"prepare({label}) norm = 1.0", abs(norm - 1.0) < 1e-14,
          f"norm = {norm}")

# |0⟩: Rz|0⟩ = e^{-iθ/2}|0⟩ (only global phase, still |0⟩ basis)
sv0 = GOD_CODE_QUBIT.prepare("|0>")
check(3, "Rz|0⟩ → P(|0⟩) = 1.0", abs(abs(sv0[0])**2 - 1.0) < 1e-14)
check(3, "Rz|0⟩ → P(|1⟩) = 0.0", abs(sv0[1])**2 < 1e-28)

# |1⟩: Rz|1⟩ = e^{+iθ/2}|1⟩
sv1 = GOD_CODE_QUBIT.prepare("|1>")
check(3, "Rz|1⟩ → P(|0⟩) = 0.0", abs(sv1[0])**2 < 1e-28)
check(3, "Rz|1⟩ → P(|1⟩) = 1.0", abs(abs(sv1[1])**2 - 1.0) < 1e-14)

# |+⟩: Rz|+⟩ stays 50/50 but gains relative phase
svp = GOD_CODE_QUBIT.prepare("|+>")
check(3, "Rz|+⟩ → P(|0⟩) = 0.5", abs(abs(svp[0])**2 - 0.5) < 1e-14)
check(3, "Rz|+⟩ → P(|1⟩) = 0.5", abs(abs(svp[1])**2 - 0.5) < 1e-14)
rel_phase = cmath.phase(svp[1]) - cmath.phase(svp[0])
check(3, "Rz|+⟩ relative phase = θ_GC",
      abs(abs(rel_phase) - GOD_CODE_PHASE) < 1e-12,
      f"rel_phase = {rel_phase:.6f}")

# |−⟩
svm = GOD_CODE_QUBIT.prepare("|->")
check(3, "Rz|−⟩ → P(|0⟩) = 0.5", abs(abs(svm[0])**2 - 0.5) < 1e-14)
check(3, "Rz|−⟩ → P(|1⟩) = 0.5", abs(abs(svm[1])**2 - 0.5) < 1e-14)

# Bloch vectors
bx, by, bz = GOD_CODE_QUBIT.bloch
check(3, "Bloch |0⟩ state at pole (z=1)",
      abs(bz - 1.0) < 1e-12 and abs(bx) < 1e-12,
      f"Bloch = ({bx:.6f}, {by:.6f}, {bz:.6f})")

# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 4: RAMSEY INTERFEROMETRY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n═══ Phase 4: Ramsey Interferometry ═══")

# H → Rz(θ) → H readout: P(|0⟩) = cos²(θ/2)
state_0 = np.array([1, 0], dtype=np.complex128)
for mult, label in [(1, "1×θ"), (2, "2×θ"), (0.5, "θ/2"), (3, "3×θ"), (7, "7×θ")]:
    theta = mult * GOD_CODE_PHASE
    sv = H_GATE @ state_0
    sv = _rz(theta) @ sv
    sv = H_GATE @ sv
    p0 = abs(sv[0])**2
    expected_p0 = math.cos(theta / 2)**2
    ok = abs(p0 - expected_p0) < 1e-12
    check(4, f"Ramsey {label}: P(0) = cos²(θ/2) = {expected_p0:.6f}",
          ok, f"got {p0:.6f}")

# Multiple sequential Rz gates = Rz(n×θ)
for n in [1, 2, 5, 10, 83, 104]:  # 104 = quantization grain
    sv = H_GATE @ state_0
    for _ in range(n):
        sv = GOD_CODE_RZ @ sv
    sv = H_GATE @ sv
    p0 = abs(sv[0])**2
    expected_p0 = math.cos(n * GOD_CODE_PHASE / 2)**2
    check(4, f"Ramsey {n}× sequential Rz",
          abs(p0 - expected_p0) < 1e-10,
          f"P(0) = {p0:.8f} vs {expected_p0:.8f}")

# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 5: DIAL SYSTEM SWEEP
# ═══════════════════════════════════════════════════════════════════════════════
print("\n═══ Phase 5: Dial System Sweep ═══")

# Origin dial
g0 = GOD_CODE_QUBIT.dial(0, 0, 0, 0)
check(5, "dial(0,0,0,0) = GOD_CODE",
      abs(g0 - GOD_CODE) < 1e-10,
      f"{g0:.10f}")

# Sweep each dial individually
n_ok = 0
n_total = 0
for a in range(8):
    freq = GOD_CODE_QUBIT.dial(a=a)
    E = 8 * a + OCTAVE_OFFSET
    expected = BASE * (2.0 ** (E / QUANTIZATION_GRAIN))
    n_total += 1
    if abs(freq - expected) < 1e-8:
        n_ok += 1
check(5, f"dial(a=0..7): {n_ok}/{n_total} correct", n_ok == n_total)

n_ok = 0
n_total = 0
for b in range(16):
    freq = GOD_CODE_QUBIT.dial(b=b)
    E = OCTAVE_OFFSET - b
    expected = BASE * (2.0 ** (E / QUANTIZATION_GRAIN))
    n_total += 1
    if abs(freq - expected) < 1e-8:
        n_ok += 1
check(5, f"dial(b=0..15): {n_ok}/{n_total} correct", n_ok == n_total)

n_ok = 0
n_total = 0
for c in range(8):
    freq = GOD_CODE_QUBIT.dial(c=c)
    E = OCTAVE_OFFSET - 8 * c
    expected = BASE * (2.0 ** (E / QUANTIZATION_GRAIN))
    n_total += 1
    if abs(freq - expected) < 1e-8:
        n_ok += 1
check(5, f"dial(c=0..7): {n_ok}/{n_total} correct", n_ok == n_total)

n_ok = 0
n_total = 0
for d in range(16):
    freq = GOD_CODE_QUBIT.dial(d=d)
    E = OCTAVE_OFFSET - QUANTIZATION_GRAIN * d
    expected = BASE * (2.0 ** (E / QUANTIZATION_GRAIN))
    n_total += 1
    if abs(freq - expected) < 1e-8:
        n_ok += 1
check(5, f"dial(d=0..15): {n_ok}/{n_total} correct", n_ok == n_total)

# Phase for every dial → all must produce valid Rz gates
n_valid = 0
n_tested = 0
for a in range(4):
    for d in range(4):
        phase = GOD_CODE_QUBIT.dial_phase(a=a, d=d)
        gate = GOD_CODE_QUBIT.dial_gate(a=a, d=d)
        UdU = gate.conj().T @ gate
        n_tested += 1
        if np.allclose(UdU, I2, atol=1e-14) and 0 <= phase < TAU_2PI:
            n_valid += 1
check(5, f"dial_gate unitarity: {n_valid}/{n_tested} valid", n_valid == n_tested)

# Full 14-qubit dial space sample (random 100)
rng = np.random.default_rng(104)
n_sampled = 0
n_unitary = 0
for _ in range(100):
    a = int(rng.integers(0, 8))
    b = int(rng.integers(0, 16))
    c = int(rng.integers(0, 8))
    d = int(rng.integers(0, 16))
    freq = GOD_CODE_QUBIT.dial(a, b, c, d)
    gate = GOD_CODE_QUBIT.dial_gate(a, b, c, d)
    UdU = gate.conj().T @ gate
    n_sampled += 1
    if np.allclose(UdU, I2, atol=1e-14) and freq > 0:
        n_unitary += 1
check(5, f"Random 100 dial configs: {n_unitary}/{n_sampled} unitary",
      n_unitary == n_sampled)

# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 6: MULTI-QUBIT CIRCUITS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n═══ Phase 6: Multi-Qubit Circuits ═══")

# 2-qubit Bell + GOD_CODE phase
sv2 = init_sv(2)
sv2 = apply_single_gate(sv2, H_GATE, 0, 2)
sv2 = apply_cnot(sv2, 0, 1, 2)
# Bell state: (|00⟩ + |11⟩)/√2
check(6, "Bell state fidelity",
      abs(abs(sv2[0])**2 - 0.5) < 1e-14 and abs(abs(sv2[3])**2 - 0.5) < 1e-14)
check(6, "Bell concurrence = 1.0",
      abs(concurrence_2q(sv2) - 1.0) < 1e-12)

# Apply GOD_CODE to qubit 0
sv2_gc = apply_single_gate(sv2, GOD_CODE_GATE, 0, 2)
check(6, "Bell + GOD_CODE norm preserved",
      abs(np.linalg.norm(sv2_gc) - 1.0) < 1e-14)
check(6, "Bell + GOD_CODE remains maximally entangled",
      abs(concurrence_2q(sv2_gc) - 1.0) < 1e-10)

# 3-qubit sacred circuit: H(0) → CNOT(0,1) → CNOT(0,2) → GOD_CODE(0) → GOD_CODE(1) → GOD_CODE(2)
sv3 = init_sv(3)
sv3 = apply_single_gate(sv3, H_GATE, 0, 3)
sv3 = apply_cnot(sv3, 0, 1, 3)
sv3 = apply_cnot(sv3, 0, 2, 3)
for q in range(3):
    sv3 = apply_single_gate(sv3, GOD_CODE_GATE, q, 3)
check(6, "3Q sacred circuit norm preserved",
      abs(np.linalg.norm(sv3) - 1.0) < 1e-14)
entropy_3q = entanglement_entropy(sv3, 3, partition=1)
check(6, "3Q sacred entanglement entropy > 0",
      entropy_3q > 0.01,
      f"S = {entropy_3q:.6f}")

# 4-qubit GHZ + GOD_CODE
sv4 = init_sv(4)
sv4 = apply_single_gate(sv4, H_GATE, 0, 4)
for i in range(3):
    sv4 = apply_cnot(sv4, i, i+1, 4)
# GHZ: (|0000⟩ + |1111⟩)/√2
check(6, "4Q GHZ: P(|0000⟩) ≈ 0.5",
      abs(abs(sv4[0])**2 - 0.5) < 1e-14)
check(6, "4Q GHZ: P(|1111⟩) ≈ 0.5",
      abs(abs(sv4[15])**2 - 0.5) < 1e-14)

# Apply GOD_CODE to all 4 qubits
for q in range(4):
    sv4 = apply_single_gate(sv4, GOD_CODE_GATE, q, 4)
check(6, "4Q GHZ + GOD_CODE norm preserved",
      abs(np.linalg.norm(sv4) - 1.0) < 1e-14)
check(6, "4Q GHZ + GOD_CODE still 50/50",
      abs(abs(sv4[0])**2 - 0.5) < 1e-10 and abs(abs(sv4[15])**2 - 0.5) < 1e-10)

# Entanglement entropy of 4Q
ent_4q = entanglement_entropy(sv4, 4, partition=2)
check(6, "4Q entanglement entropy = 1.0 bit",
      abs(ent_4q - 1.0) < 1e-10,
      f"S = {ent_4q:.6f}")

# 5-qubit chain: sequential entanglement + GOD_CODE
sv5 = init_sv(5)
sv5 = apply_single_gate(sv5, H_GATE, 0, 5)
for i in range(4):
    sv5 = apply_cnot(sv5, i, i+1, 5)
for q in range(5):
    sv5 = apply_single_gate(sv5, GOD_CODE_GATE, q, 5)
check(6, "5Q chain + GOD_CODE norm preserved",
      abs(np.linalg.norm(sv5) - 1.0) < 1e-14)

# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 7: NOISE RESILIENCE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n═══ Phase 7: Noise Resilience ═══")

try:
    from l104_god_code_simulator.qpu_verification import (
        depolarizing_channel_1q, amplitude_damping_channel,
        apply_readout_noise, simulate_with_noise,
        HERON_NOISE_PARAMS,
    )

    # Depolarizing noise: Monte-Carlo over many trials for statistical fidelity
    N_TRIALS = 500
    sv_ref = GOD_CODE_QUBIT.prepare("|+>")  # Use |+⟩ so phase effects are visible

    for p_dep, label, min_fid in [(0.0, "p=0", 0.999), (0.001, "p=0.001", 0.99),
                                   (0.01, "p=0.01", 0.95), (0.05, "p=0.05", 0.80)]:
        fids = []
        for _ in range(N_TRIALS):
            sv_noisy = depolarizing_channel_1q(p_dep, sv_ref.copy(), 0, 1)
            fids.append(fidelity(sv_noisy, sv_ref))
        mean_fid = np.mean(fids)
        check(7, f"Depolarizing {label}: mean fidelity ≥ {min_fid}",
              mean_fid >= min_fid, f"F = {mean_fid:.6f}")

    # Amplitude damping on |1⟩ state (where damping is visible)
    sv_1 = GOD_CODE_QUBIT.prepare("|1>")
    for gamma, label, min_fid in [(0.0, "γ=0", 0.999), (0.01, "γ=0.01", 0.98),
                                   (0.1, "γ=0.1", 0.85)]:
        sv_damped = amplitude_damping_channel(gamma, sv_1.copy(), 0, 1)
        f_ad = fidelity(sv_damped, sv_1)
        check(7, f"Amplitude damping {label}: fidelity ≥ {min_fid}",
              f_ad >= min_fid, f"F = {f_ad:.6f}")

    # Heron noise parameters check
    check(7, "Heron 1Q error rate = 0.00025",
          abs(HERON_NOISE_PARAMS["depolarizing_1q"] - 0.00025) < 1e-6)

    # Readout noise on distribution
    dist = {"0": 0.5, "1": 0.5}
    noisy_dist = apply_readout_noise(dist, p10=0.01, p01=0.02)
    check(7, "Readout noise preserves total probability",
          abs(sum(noisy_dist.values()) - 1.0) < 1e-10)

    # simulate_with_noise: full circuit
    sv_noisy = init_sv(1)
    gate_ops = [("H", 0), ("GATE", (GOD_CODE_GATE, 0)), ("H", 0)]
    sv_noisy = simulate_with_noise(sv_noisy, 1, gate_ops, noise_level=0.001)
    check(7, "simulate_with_noise: norm preserved",
          abs(np.linalg.norm(sv_noisy) - 1.0) < 0.01,
          f"norm = {np.linalg.norm(sv_noisy):.6f}")

except ImportError as e:
    check(7, "QPU verification module available", False, str(e))

# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 8: QPE PHASE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n═══ Phase 8: QPE Phase Extraction ═══")

for n_bits in [4, 6, 8]:
    n_total = n_bits + 1
    sv = init_sv(n_total)

    # Target |1⟩
    sv = apply_single_gate(sv, X_GATE, n_bits, n_total)

    # Hadamard on ancilla
    for i in range(n_bits):
        sv = apply_single_gate(sv, H_GATE, i, n_total)

    # Controlled-U^(2^k):  CP(GOD_CODE_PHASE × 2^k) on (ancilla_k, target)
    for k in range(n_bits):
        angle = GOD_CODE_PHASE * (2 ** k)
        sv = apply_cp(sv, angle, k, n_bits, n_total)

    # Inverse QFT on ancilla register
    for i in range(n_bits // 2):
        sv = apply_swap(sv, i, n_bits - 1 - i, n_total)
    for i in range(n_bits):
        for j in range(i):
            angle = -math.pi / (2 ** (i - j))
            sv = apply_cp(sv, angle, j, i, n_total)
        sv = apply_single_gate(sv, H_GATE, i, n_total)

    # Extract probabilities from ancilla register
    probs = probabilities(sv)
    # Find most probable ancilla state
    ancilla_probs = {}
    for bitstr, p in probs.items():
        ancilla = bitstr[:n_bits]
        ancilla_probs[ancilla] = ancilla_probs.get(ancilla, 0) + p

    best = max(ancilla_probs, key=ancilla_probs.get)
    best_val = int(best, 2)
    extracted_phase = (best_val / (2 ** n_bits)) * TAU_2PI

    # The QPE extracts θ/2π, so extracted_phase should be close to GOD_CODE_PHASE
    phase_error = abs(extracted_phase - GOD_CODE_PHASE)
    phase_error = min(phase_error, TAU_2PI - phase_error)
    max_expected_error = TAU_2PI / (2 ** n_bits)  # 1× resolution limit
    # QPE phase lands on nearest integer/(2^n), so real error can be up to ½ bin
    # For non-dyadic phases, spectral leakage adds ~1 more bin of error
    max_allowed = max(max_expected_error * 6, 0.15)  # generous for irrational phase

    check(8, f"QPE {n_bits}-bit: peak = |{best}⟩ (P = {ancilla_probs[best]:.4f})",
          ancilla_probs[best] > 0.3,
          f"phase_err = {phase_error:.6f} rad")
    check(8, f"QPE {n_bits}-bit: error < {max_allowed:.4f} rad",
          phase_error < max_allowed,
          f"extracted = {extracted_phase:.6f}, target = {GOD_CODE_PHASE:.6f}")

# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 9: CONSERVATION LAWS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n═══ Phase 9: Conservation Laws ═══")

# Phase additivity: θ_Fe + θ_φ + θ_oct = θ_GC
sum_phase = (IRON_PHASE + PHI_CONTRIBUTION + OCTAVE_PHASE) % TAU_2PI
target_phase = GOD_CODE_PHASE % TAU_2PI
check(9, "Phase additivity: Fe + φ + oct = θ_GC",
      abs(sum_phase - target_phase) < 1e-14,
      f"err = {abs(sum_phase - target_phase):.2e}")

# Matrix product conservation
decomp_product = IRON_RZ @ PHI_RZ @ OCTAVE_RZ
direct = GOD_CODE_RZ
gp = decomp_product[0, 0] / direct[0, 0]
check(9, "Matrix product: Rz(Fe)·Rz(φ)·Rz(oct) ≡ Rz(θ_GC)",
      np.allclose(decomp_product / gp, direct, atol=1e-14))

# Round-trip: apply → inverse → identity
for basis in ["|0>", "|1>", "|+>", "|->",]:
    sv = GOD_CODE_QUBIT.prepare(basis)
    sv_inv = _rz(-GOD_CODE_PHASE) @ sv
    # Should return to original basis state
    if basis == "|0>":
        original = np.array([1, 0], dtype=np.complex128)
    elif basis == "|1>":
        original = np.array([0, 1], dtype=np.complex128)
    elif basis == "|+>":
        original = H_GATE @ np.array([1, 0], dtype=np.complex128)
    else:
        original = H_GATE @ np.array([0, 1], dtype=np.complex128)

    f_round = fidelity(sv_inv, original)
    check(9, f"Round-trip {basis}: F = {f_round:.6f}",
          abs(f_round - 1.0) < 1e-12)

# Energy conservation in multi-qubit: unitary preserves trace
for nq in [2, 3, 4]:
    sv = init_sv(nq)
    sv = apply_single_gate(sv, H_GATE, 0, nq)
    for i in range(nq - 1):
        sv = apply_cnot(sv, i, i + 1, nq)
    for q in range(nq):
        sv = apply_single_gate(sv, GOD_CODE_GATE, q, nq)
    check(9, f"{nq}Q circuit ‖ψ‖² = 1.0",
          abs(np.linalg.norm(sv)**2 - 1.0) < 1e-14)

# Commutativity of diagonal gates: Rz(a)·Rz(b) = Rz(b)·Rz(a)
check(9, "Rz(Fe)·Rz(oct) = Rz(oct)·Rz(Fe)",
      np.allclose(IRON_RZ @ OCTAVE_RZ, OCTAVE_RZ @ IRON_RZ, atol=1e-15))

# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 10: CROSS-PACKAGE CONSISTENCY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n═══ Phase 10: Cross-Package Consistency ═══")

# god_code_simulator constants
check(10, "Simulator GOD_CODE_PHASE_ANGLE = GOD_CODE mod 2π",
      abs(GOD_CODE_PHASE_ANGLE - (GOD_CODE % TAU_2PI)) < 1e-14)
check(10, "Simulator PHI_PHASE_ANGLE = 2π/φ",
      abs(PHI_PHASE_ANGLE - TAU_2PI / PHI) < 1e-14)
check(10, "Simulator IRON_PHASE_ANGLE = π/2",
      abs(IRON_PHASE_ANGLE - math.pi / 2) < 1e-14)

# quantum_gate_engine constants
from l104_quantum_gate_engine.constants import (
    GOD_CODE_PHASE_ANGLE as QGE_PHASE,
    PHI_PHASE_ANGLE as QGE_PHI,
    IRON_PHASE_ANGLE as QGE_IRON,
)
check(10, "QGE phase = Simulator phase",
      abs(QGE_PHASE - GOD_CODE_PHASE_ANGLE) < 1e-14)
check(10, "QGE PHI phase = Simulator PHI phase",
      abs(QGE_PHI - PHI_PHASE_ANGLE) < 1e-14)
check(10, "QGE IRON phase = Simulator IRON phase",
      abs(QGE_IRON - IRON_PHASE_ANGLE) < 1e-14)

# Bridge singletons
from l104_quantum_gate_engine import GOD_CODE_QUBIT as QGE_Q
from l104_science_engine import GOD_CODE_QUBIT as SE_Q
from l104_god_code_simulator import GOD_CODE_QUBIT as SIM_Q

check(10, "QGE singleton is same object",
      QGE_Q is GOD_CODE_QUBIT)
check(10, "SciEngine singleton is same object",
      SE_Q is GOD_CODE_QUBIT)
check(10, "Simulator singleton is same object",
      SIM_Q is GOD_CODE_QUBIT)

# QGE sacred gate phase consistency
from l104_quantum_gate_engine.gates import GOD_CODE_PHASE as QGE_GC_GATE
qge_relative = abs(np.angle(QGE_GC_GATE.matrix[1,1]) - np.angle(QGE_GC_GATE.matrix[0,0]))
canonical_relative = abs(np.angle(GOD_CODE_RZ[1,1]) - np.angle(GOD_CODE_RZ[0,0]))
check(10, "QGE gate relative phase = canonical relative phase",
      abs(qge_relative - canonical_relative) < 1e-12,
      f"QGE = {qge_relative:.6f}, canonical = {canonical_relative:.6f}")

# QPU data consistency
check(10, "QPU backend = ibm_torino", QPU_DATA["backend"] == "ibm_torino")
check(10, "QPU mean fidelity ≥ 0.95", QPU_DATA["mean_fidelity"] >= 0.95)
check(10, "QPU has all 6 circuits",
      len(QPU_DATA["circuits"]) == 6)
check(10, "All QPU fidelities > 0.90",
      all(c["fidelity"] > 0.90 for c in QPU_DATA["circuits"].values()))

# ═══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
elapsed = time.perf_counter() - T0
print()
print("═" * 72)
print(f"  L104 GOD_CODE QUBIT STRESS TEST — RESULTS")
print("═" * 72)
print()

# Phase breakdown
for phase_num in range(1, 11):
    phase_results = [r for r in RESULTS if r[0] == phase_num]
    p = sum(1 for r in phase_results if r[2])
    f = sum(1 for r in phase_results if not r[2])
    phase_names = {
        1: "Qubit Integrity", 2: "Gate Algebra", 3: "State Preparation",
        4: "Ramsey Interferometry", 5: "Dial System Sweep",
        6: "Multi-Qubit Circuits", 7: "Noise Resilience",
        8: "QPE Phase Extraction", 9: "Conservation Laws",
        10: "Cross-Package Consistency",
    }
    status = "✅" if f == 0 else "❌"
    print(f"  {status}  Phase {phase_num:2d}: {phase_names[phase_num]:<28s} — {p}/{p+f}")

print()
print(f"  Total: {PASS} passed, {FAIL} failed, {PASS+FAIL} checks")
print(f"  Time:  {elapsed*1000:.0f} ms")
print()
print(f"  GOD_CODE     = {GOD_CODE}")
print(f"  θ_GC         = {GOD_CODE_PHASE:.15f} rad")
print(f"  θ_GC         = {math.degrees(GOD_CODE_PHASE):.10f}°")
print(f"  QPU Backend  = {QPU_DATA['backend']} ({QPU_DATA['processor']})")
print(f"  QPU Fidelity = {QPU_DATA['mean_fidelity']}")
print(f"  Decomposition: Fe({IRON_PHASE:.4f}) + φ({PHI_CONTRIBUTION:.4f}) + oct({OCTAVE_PHASE:.4f}) = {IRON_PHASE+PHI_CONTRIBUTION+OCTAVE_PHASE:.4f}")
print()
print("═" * 72)

if FAIL == 0:
    print("  ✅  ALL CHECKS PASSED — Canonical GOD_CODE Qubit is SOVEREIGN")
else:
    print(f"  ⚠️   {FAIL} CHECK(S) FAILED")

print("═" * 72)

sys.exit(0 if FAIL == 0 else 1)
