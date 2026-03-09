"""
L104 God Code Simulator — Quantum Primitives
═══════════════════════════════════════════════════════════════════════════════

Pure-numpy quantum building blocks: gates, statevector operations, and
common quantum-information functions (fidelity, entropy, concurrence, Bloch).

All 20+ simulations build on these primitives — zero external quantum deps.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

from .constants import (
    GOD_CODE_PHASE_ANGLE,
    IRON_PHASE_ANGLE,
    PHI_PHASE_ANGLE,
    VOID_PHASE_ANGLE,
)
from .god_code_qubit import (
    GOD_CODE_QUBIT,
    GOD_CODE_PHASE,
    GOD_CODE_RZ,
    IRON_RZ,
    PHI_RZ,
    OCTAVE_RZ,
)


# ═══════════════════════════════════════════════════════════════════════════════
# GOD CODE ALGEBRAIC FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def god_code_fn(x: float) -> float:
    """G(X) = 286^(1/φ) × 2^((416−X)/104)."""
    from .constants import BASE, OCTAVE_OFFSET, QUANTIZATION_GRAIN
    return BASE * (2.0 ** ((OCTAVE_OFFSET - x) / QUANTIZATION_GRAIN))


def god_code_dial(a: int = 0, b: int = 0, c: int = 0, d: int = 0) -> float:
    """G(a,b,c,d) = 286^(1/φ) × 2^((8a+416-b-8c-104d)/104)."""
    from .constants import BASE, OCTAVE_OFFSET, QUANTIZATION_GRAIN
    exponent = (8 * a + OCTAVE_OFFSET - b - 8 * c - QUANTIZATION_GRAIN * d) / QUANTIZATION_GRAIN
    return BASE * (2.0 ** exponent)


# ═══════════════════════════════════════════════════════════════════════════════
# GATE CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def make_gate(matrix_2x2) -> np.ndarray:
    """Ensure gate is a 2×2 complex128 matrix."""
    return np.array(matrix_2x2, dtype=np.complex128)


# ── Standard gates ───────────────────────────────────────────────────────────
H_GATE: np.ndarray = make_gate([[1, 1], [1, -1]]) / math.sqrt(2)
X_GATE: np.ndarray = make_gate([[0, 1], [1, 0]])
Z_GATE: np.ndarray = make_gate([[1, 0], [0, -1]])
S_GATE: np.ndarray = make_gate([[1, 0], [0, 1j]])
T_GATE: np.ndarray = make_gate([[1, 0], [0, np.exp(1j * math.pi / 4)]])

# ── Sacred gates (QPU-verified phase angles) ────────────────────────────────
# GOD_CODE_GATE is now the canonical Rz form from god_code_qubit.py
PHI_GATE: np.ndarray = make_gate([[1, 0], [0, np.exp(1j * PHI_PHASE_ANGLE)]])
GOD_CODE_GATE: np.ndarray = GOD_CODE_RZ.copy()  # Rz(GOD_CODE mod 2π) — QPU-verified
VOID_GATE: np.ndarray = make_gate([[1, 0], [0, np.exp(1j * VOID_PHASE_ANGLE)]])
IRON_GATE: np.ndarray = make_gate([[1, 0], [0, np.exp(1j * IRON_PHASE_ANGLE)]])


# ═══════════════════════════════════════════════════════════════════════════════
# STATEVECTOR OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def init_sv(n_qubits: int) -> np.ndarray:
    """Initialize |0...0⟩ statevector."""
    sv = np.zeros(2 ** n_qubits, dtype=np.complex128)
    sv[0] = 1.0
    return sv


def apply_single_gate(sv: np.ndarray, gate: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    """Apply single-qubit gate to statevector."""
    dim = 2 ** n_qubits
    new_sv = np.zeros(dim, dtype=np.complex128)
    for i in range(dim):
        bit = (i >> qubit) & 1
        partner = i ^ (1 << qubit)
        if bit == 0:
            new_sv[i] = gate[0, 0] * sv[i] + gate[0, 1] * sv[partner]
            new_sv[partner] = gate[1, 0] * sv[i] + gate[1, 1] * sv[partner]
    return new_sv


def apply_cnot(sv: np.ndarray, control: int, target: int, n_qubits: int) -> np.ndarray:
    """Apply CNOT gate."""
    dim = 2 ** n_qubits
    new_sv = sv.copy()
    for i in range(dim):
        if (i >> control) & 1:
            partner = i ^ (1 << target)
            new_sv[i], new_sv[partner] = sv[partner], sv[i]
    return new_sv


def apply_cp(sv: np.ndarray, theta: float, control: int, target: int, n_qubits: int) -> np.ndarray:
    """Apply controlled-phase gate: |11⟩ → e^{iθ}|11⟩."""
    dim = 2 ** n_qubits
    new_sv = sv.copy()
    for i in range(dim):
        if (i >> control) & 1 and (i >> target) & 1:
            new_sv[i] *= np.exp(1j * theta)
    return new_sv


def apply_swap(sv: np.ndarray, q1: int, q2: int, n_qubits: int) -> np.ndarray:
    """Apply SWAP gate between two qubits."""
    dim = 2 ** n_qubits
    new_sv = sv.copy()
    for i in range(dim):
        b1 = (i >> q1) & 1
        b2 = (i >> q2) & 1
        if b1 != b2:
            partner = i ^ (1 << q1) ^ (1 << q2)
            new_sv[i], new_sv[partner] = sv[partner], sv[i]
    return new_sv


def apply_mcx(sv: np.ndarray, controls: list, target: int, n_qubits: int) -> np.ndarray:
    """Multi-controlled X (Toffoli generalization)."""
    dim = 2 ** n_qubits
    new_sv = sv.copy()
    for i in range(dim):
        if all((i >> c) & 1 for c in controls):
            partner = i ^ (1 << target)
            new_sv[i], new_sv[partner] = sv[partner], sv[i]
    return new_sv


def build_unitary(n_qubits: int, gate_ops: list) -> np.ndarray:
    """
    Build full unitary matrix from a sequence of gate operations.

    gate_ops: list of (op_type, params) tuples:
      ("H", qubit), ("Rz", (theta, qubit)), ("CX", (ctrl, tgt)),
      ("CP", (theta, ctrl, tgt)), ("X", qubit), ("SWAP", (q1, q2))
    """
    dim = 2 ** n_qubits
    U = np.eye(dim, dtype=np.complex128)

    for op, params in gate_ops:
        for col in range(dim):
            sv = U[:, col].copy()
            if op == "H":
                U[:, col] = apply_single_gate(sv, H_GATE, params, n_qubits)
            elif op == "Rz":
                theta, qubit = params
                gate = make_gate([[np.exp(-1j * theta / 2), 0],
                                  [0, np.exp(1j * theta / 2)]])
                U[:, col] = apply_single_gate(sv, gate, qubit, n_qubits)
            elif op == "Ry":
                theta, qubit = params
                c, s = np.cos(theta / 2), np.sin(theta / 2)
                gate = make_gate([[c, -s], [s, c]])
                U[:, col] = apply_single_gate(sv, gate, qubit, n_qubits)
            elif op == "X":
                U[:, col] = apply_single_gate(sv, X_GATE, params, n_qubits)
            elif op == "CX":
                ctrl, tgt = params
                U[:, col] = apply_cnot(sv, ctrl, tgt, n_qubits)
            elif op == "CP":
                theta, ctrl, tgt = params
                U[:, col] = apply_cp(sv, theta, ctrl, tgt, n_qubits)
            elif op == "SWAP":
                q1, q2 = params
                U[:, col] = apply_swap(sv, q1, q2, n_qubits)
            elif op == "MCX":
                ctrls, tgt = params
                U[:, col] = apply_mcx(sv, ctrls, tgt, n_qubits)
            elif op == "GATE":
                gate, qubit = params
                U[:, col] = apply_single_gate(sv, gate, qubit, n_qubits)

    return U


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM INFORMATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def probabilities(sv: np.ndarray) -> Dict[str, float]:
    """Extract probabilities from statevector."""
    n = int(math.log2(len(sv)))
    probs = {}
    for i, amp in enumerate(sv):
        p = abs(amp) ** 2
        if p > 1e-12:
            probs[format(i, f'0{n}b')] = float(p)
    return probs


def entanglement_entropy(sv: np.ndarray, n_qubits: int, partition: int = None) -> float:
    """Von Neumann entropy of reduced density matrix for bipartite partition."""
    if partition is None:
        partition = n_qubits // 2
    dim_a = 2 ** partition
    dim_b = 2 ** (n_qubits - partition)
    psi = sv.reshape(dim_a, dim_b)
    rho_a = psi @ psi.conj().T
    eigenvalues = np.linalg.eigvalsh(rho_a)
    eigenvalues = eigenvalues[eigenvalues > 1e-15]
    return float(-np.sum(eigenvalues * np.log2(eigenvalues + 1e-30)))


def concurrence_2q(sv: np.ndarray) -> float:
    """Concurrence for a 2-qubit pure state."""
    if len(sv) != 4:
        return 0.0
    return float(2.0 * abs(sv[0] * sv[3] - sv[1] * sv[2]))


def fidelity(sv1: np.ndarray, sv2: np.ndarray) -> float:
    """State fidelity |⟨ψ₁|ψ₂⟩|²."""
    return float(abs(np.vdot(sv1, sv2)) ** 2)


def bloch_vector(sv: np.ndarray) -> Tuple[float, float, float]:
    """Bloch vector for single-qubit state."""
    if len(sv) != 2:
        return (0.0, 0.0, 0.0)
    a, b = sv[0], sv[1]
    x = 2.0 * (a.conj() * b).real
    y = 2.0 * (a.conj() * b).imag
    z = float(abs(a) ** 2 - abs(b) ** 2)
    return (float(x), float(y), float(z))


# ═══════════════════════════════════════════════════════════════════════════════
#  v3.0 — Extended Quantum Primitives
# ═══════════════════════════════════════════════════════════════════════════════

# ── Y gate ──────────────────────────────────────────────────────────────────
Y_GATE: np.ndarray = make_gate([[0, -1j], [1j, 0]])

# ── CZ gate (Controlled-Z) ─────────────────────────────────────────────────
CZ_GATE: np.ndarray = np.diag([1.0, 1.0, 1.0, -1.0]).astype(np.complex128)

# ── SX gate (√X) ───────────────────────────────────────────────────────────
SX_GATE: np.ndarray = make_gate([[0.5 + 0.5j, 0.5 - 0.5j],
                                  [0.5 - 0.5j, 0.5 + 0.5j]])

# ── Ry / Rx rotation gates ──────────────────────────────────────────────────
def ry_gate(theta: float) -> np.ndarray:
    """Ry(θ) rotation gate."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return make_gate([[c, -s], [s, c]])

def rx_gate(theta: float) -> np.ndarray:
    """Rx(θ) rotation gate."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return make_gate([[c, -1j * s], [-1j * s, c]])

def rz_gate(theta: float) -> np.ndarray:
    """Rz(θ) rotation gate."""
    return make_gate([[np.exp(-1j * theta / 2), 0],
                      [0, np.exp(1j * theta / 2)]])


# ── Purity & Trace Distance ─────────────────────────────────────────────────
def state_purity(sv: np.ndarray, n_qubits: int, partition: int = None) -> float:
    """
    Purity Tr(ρ²) of the reduced density matrix.

    Purity = 1.0 for pure states, 1/d for maximally mixed.
    Useful for measuring decoherence quantitatively.
    """
    if partition is None:
        partition = n_qubits // 2
    if partition == 0 or partition >= n_qubits:
        return 1.0
    dim = 2 ** n_qubits
    sv_r = sv.reshape((2 ** partition, dim // (2 ** partition)))
    rho_A = sv_r @ sv_r.conj().T
    return float(np.real(np.trace(rho_A @ rho_A)))


def trace_distance(sv1: np.ndarray, sv2: np.ndarray) -> float:
    """
    Trace distance between two pure states: D(ρ,σ) = √(1 - |⟨ψ₁|ψ₂⟩|²).

    Ranges from 0 (identical) to 1 (orthogonal).
    """
    overlap = abs(np.vdot(sv1, sv2)) ** 2
    return float(np.sqrt(max(0.0, 1.0 - overlap)))


def schmidt_coefficients(sv: np.ndarray, n_qubits: int, partition: int = None) -> np.ndarray:
    """
    Schmidt decomposition coefficients (singular values of bipartite split).

    The Schmidt rank reveals entanglement structure:
    rank=1 → separable, rank>1 → entangled.
    """
    if partition is None:
        partition = n_qubits // 2
    dim = 2 ** n_qubits
    sv_r = sv.reshape((2 ** partition, dim // (2 ** partition)))
    return np.linalg.svd(sv_r, compute_uv=False)


def quantum_relative_entropy(sv1: np.ndarray, sv2: np.ndarray) -> float:
    """
    Quantum relative entropy S(ρ||σ) for pure states.

    For pure states: S(|ψ⟩||ρ) = -log(⟨ψ|ρ|ψ⟩).
    Measures distinguishability between two quantum states.
    """
    f = abs(np.vdot(sv1, sv2)) ** 2
    if f < 1e-15:
        return float('inf')
    return float(-np.log(max(f, 1e-300)))


def linear_entropy(sv: np.ndarray, n_qubits: int, partition: int = None) -> float:
    """
    Linear entropy S_L = 1 - Tr(ρ²). Ranges from 0 (pure) to 1-1/d (maximally mixed).

    Faster than von Neumann entropy (no log required).
    """
    return 1.0 - state_purity(sv, n_qubits, partition)


# ═══════════════════════════════════════════════════════════════════════════════
#  v3.1 — Convenience Gate Application + Advanced Entanglement Measures
# ═══════════════════════════════════════════════════════════════════════════════

def apply_ry(sv: np.ndarray, theta: float, qubit: int, n_qubits: int) -> np.ndarray:
    """Apply Ry(θ) rotation to a qubit (convenience wrapper)."""
    return apply_single_gate(sv, ry_gate(theta), qubit, n_qubits)


def apply_rx(sv: np.ndarray, theta: float, qubit: int, n_qubits: int) -> np.ndarray:
    """Apply Rx(θ) rotation to a qubit (convenience wrapper)."""
    return apply_single_gate(sv, rx_gate(theta), qubit, n_qubits)


def apply_rz(sv: np.ndarray, theta: float, qubit: int, n_qubits: int) -> np.ndarray:
    """Apply Rz(θ) rotation to a qubit (convenience wrapper)."""
    return apply_single_gate(sv, rz_gate(theta), qubit, n_qubits)


def apply_cz(sv: np.ndarray, control: int, target: int, n_qubits: int) -> np.ndarray:
    """Apply CZ (controlled-Z) gate: |11⟩ → -|11⟩."""
    dim = 2 ** n_qubits
    new_sv = sv.copy()
    for i in range(dim):
        if (i >> control) & 1 and (i >> target) & 1:
            new_sv[i] *= -1
    return new_sv


def negativity(sv: np.ndarray, n_qubits: int, partition: int = None) -> float:
    """
    Negativity N(ρ) — entanglement monotone via partial transpose.

    N(ρ) = (||ρ^{T_B}||₁ - 1) / 2

    Non-zero negativity is a sufficient condition for entanglement.
    For 2-qubit pure states, negativity = concurrence / 2.
    For higher dimensions, it detects bound entanglement that concurrence misses.
    """
    if partition is None:
        partition = n_qubits // 2
    if partition == 0 or partition >= n_qubits:
        return 0.0

    dim_a = 2 ** partition
    dim_b = 2 ** (n_qubits - partition)

    # Build full density matrix ρ = |ψ⟩⟨ψ|
    rho = np.outer(sv, sv.conj())

    # Partial transpose over subsystem B
    # Reshape ρ as (dim_a, dim_b, dim_a, dim_b), swap B indices, reshape back
    rho_r = rho.reshape(dim_a, dim_b, dim_a, dim_b)
    rho_pt = rho_r.transpose(0, 3, 2, 1).reshape(dim_a * dim_b, dim_a * dim_b)

    # Negativity = (trace norm - 1) / 2
    eigenvalues = np.linalg.eigvalsh(rho_pt)
    trace_norm = float(np.sum(np.abs(eigenvalues)))
    return max(0.0, (trace_norm - 1.0) / 2.0)


def build_w_state(n_qubits: int) -> np.ndarray:
    """
    Build the W state: |W⟩ = (|100...0⟩ + |010...0⟩ + ... + |000...1⟩) / √n.

    W states have fundamentally different entanglement structure than GHZ:
    - GHZ: maximally entangled but fragile (tracing out 1 qubit → separable)
    - W: robust entanglement (tracing out 1 qubit → still entangled)
    """
    dim = 2 ** n_qubits
    sv = np.zeros(dim, dtype=np.complex128)
    for k in range(n_qubits):
        sv[1 << k] = 1.0 / math.sqrt(n_qubits)
    return sv


# ═══════════════════════════════════════════════════════════════════════════════
#  v4.0 — VQPU-Derived Quantum Primitives (from simulation findings)
# ═══════════════════════════════════════════════════════════════════════════════

def _expm_approx(M: np.ndarray) -> np.ndarray:
    """Matrix exponential via eigendecomposition (no scipy dependency)."""
    try:
        from scipy.linalg import expm
        return expm(M)
    except ImportError:
        # Taylor series fallback: e^M ≈ I + M + M²/2 + M³/6 + M⁴/24
        eye = np.eye(M.shape[0], dtype=M.dtype)
        result = eye + M
        Mk = M.copy()
        for k in range(2, 12):
            Mk = Mk @ M / k
            result += Mk
            if np.max(np.abs(Mk)) < 1e-15:
                break
        return result


def quantum_fisher_information(sv: np.ndarray, generator: np.ndarray,
                                n_qubits: int, delta: float = 1e-4) -> Dict[str, float]:
    """
    Quantum Fisher Information F_Q via parameter-shift differentiation.

    F_Q = 4(⟨∂ψ|∂ψ⟩ - |⟨∂ψ|ψ⟩|²)

    Measures the ultimate precision limit for parameter estimation (Cramér-Rao).
    Heisenberg-limited when F_Q scales as n² (vs shot-noise n).
    """
    dim = 2 ** n_qubits
    # Ensure generator is dim×dim
    if generator.shape[0] != dim:
        full_gen = np.eye(1, dtype=np.complex128)
        single = generator
        for q in range(n_qubits):
            if q == 0:
                full_gen = np.kron(full_gen, single)
            else:
                full_gen = np.kron(full_gen, np.eye(2, dtype=np.complex128))
        generator = full_gen

    U_plus = _expm_approx(-1j * delta * generator)
    U_minus = _expm_approx(1j * delta * generator)

    sv_plus = U_plus @ sv
    sv_minus = U_minus @ sv
    d_sv = (sv_plus - sv_minus) / (2.0 * delta)

    braket_dd = float(np.real(np.vdot(d_sv, d_sv)))
    braket_dp = np.vdot(d_sv, sv)
    qfi = float(4.0 * (braket_dd - abs(braket_dp) ** 2))
    cramer_rao = 1.0 / max(qfi, 1e-30)

    return {
        "qfi": qfi,
        "cramer_rao_bound": cramer_rao,
        "qfi_per_qubit": qfi / max(n_qubits, 1),
        "heisenberg_limited": qfi > n_qubits * 1.5,
    }


def loschmidt_echo(sv: np.ndarray, hamiltonian: np.ndarray,
                   perturbation: np.ndarray, n_qubits: int,
                   time_steps: int = 20, dt: float = 0.1) -> Dict:
    """
    Loschmidt echo: L(t) = |⟨ψ₀|e^{iH't}e^{-iHt}|ψ₀⟩|².

    Measures sensitivity to perturbation — rapid decay indicates quantum chaos.
    Returns echo values, decay rate, and Lyapunov estimate.
    """
    dim = 2 ** n_qubits
    H = hamiltonian if hamiltonian.shape[0] == dim else np.kron(
        hamiltonian, np.eye(dim // hamiltonian.shape[0], dtype=np.complex128))
    H_pert = H + (perturbation if perturbation.shape[0] == dim else np.kron(
        perturbation, np.eye(dim // perturbation.shape[0], dtype=np.complex128)))

    U_fwd = _expm_approx(-1j * dt * H)
    U_bwd = _expm_approx(1j * dt * H_pert)

    sv_fwd = sv.copy()
    sv_bwd = sv.copy()
    echo_values = [1.0]

    for _step in range(time_steps):
        sv_fwd = U_fwd @ sv_fwd
        sv_bwd = U_bwd @ sv_bwd
        echo = float(abs(np.vdot(sv_bwd, sv_fwd)) ** 2)
        echo_values.append(echo)

    # Fit exponential decay rate
    log_echoes = [math.log(max(e, 1e-30)) for e in echo_values[1:]]
    if len(log_echoes) >= 2:
        decay_rate = float(-(log_echoes[-1] - log_echoes[0]) / (len(log_echoes) * dt))
    else:
        decay_rate = 0.0

    lyapunov = decay_rate / 2.0
    is_chaotic = decay_rate > 1.0

    return {
        "echo_values": echo_values,
        "decay_rate": decay_rate,
        "is_chaotic": is_chaotic,
        "lyapunov_estimate": lyapunov,
        "final_echo": echo_values[-1],
        "time_steps": time_steps,
    }


def density_matrix_from_sv(sv: np.ndarray) -> np.ndarray:
    """Compute density matrix ρ = |ψ⟩⟨ψ| from statevector."""
    return np.outer(sv, sv.conj())


def partial_trace(rho: np.ndarray, n_qubits: int, trace_out: list) -> np.ndarray:
    """
    Partial trace over specified qubits.

    Traces out the qubits in `trace_out`, returning the reduced density matrix
    for the remaining subsystem.
    """
    keep = sorted(set(range(n_qubits)) - set(trace_out))
    n_keep = len(keep)
    d_keep = 2 ** n_keep
    d_trace = 2 ** len(trace_out)

    rho_r = rho.reshape([2] * (2 * n_qubits))
    perm = keep + trace_out + [q + n_qubits for q in keep] + [q + n_qubits for q in trace_out]
    rho_r = rho_r.transpose(perm)
    rho_r = rho_r.reshape(d_keep, d_trace, d_keep, d_trace)
    return np.trace(rho_r, axis1=1, axis2=3)


def von_neumann_entropy_dm(rho: np.ndarray) -> float:
    """Von Neumann entropy S(ρ) = -Tr(ρ log₂ ρ) from density matrix."""
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-15]
    return float(-np.sum(eigenvalues * np.log2(eigenvalues + 1e-30)))


def bures_distance(sv1: np.ndarray, sv2: np.ndarray) -> float:
    """Bures distance D_B = √(2(1 - √F)) between two pure states."""
    f = abs(np.vdot(sv1, sv2)) ** 2
    return float(np.sqrt(max(0.0, 2.0 * (1.0 - np.sqrt(max(0.0, f))))))


def topological_entanglement_entropy(sv: np.ndarray, n_qubits: int) -> Dict:
    """
    Kitaev-Preskill topological entanglement entropy.

    For a system partitioned into regions A, B, C:
      γ_topo = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC

    Non-zero γ_topo indicates topological order with quantum dimension D = e^γ.
    """
    if n_qubits < 3:
        return {"topological_entropy": 0.0, "has_topological_order": False,
                "quantum_dimension_estimate": 1.0}

    n_a = n_qubits // 3
    n_b = n_qubits // 3
    n_c = n_qubits - n_a - n_b
    qubits_a = list(range(n_a))
    qubits_b = list(range(n_a, n_a + n_b))
    qubits_c = list(range(n_a + n_b, n_qubits))

    rho = density_matrix_from_sv(sv)

    def _entropy_of(keep_qubits):
        trace_out = sorted(set(range(n_qubits)) - set(keep_qubits))
        if not trace_out:
            return 0.0
        rho_r = partial_trace(rho, n_qubits, trace_out)
        return von_neumann_entropy_dm(rho_r)

    S_A = _entropy_of(qubits_a)
    S_B = _entropy_of(qubits_b)
    S_C = _entropy_of(qubits_c)
    S_AB = _entropy_of(qubits_a + qubits_b)
    S_BC = _entropy_of(qubits_b + qubits_c)
    S_AC = _entropy_of(qubits_a + qubits_c)
    S_ABC = 0.0  # Pure state

    gamma_topo = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC
    has_topo = abs(gamma_topo) > 0.01
    quantum_dim = float(np.exp(abs(gamma_topo))) if has_topo else 1.0

    return {
        "topological_entropy": float(gamma_topo),
        "has_topological_order": has_topo,
        "quantum_dimension_estimate": quantum_dim,
        "partitions": {"S_A": S_A, "S_B": S_B, "S_C": S_C,
                       "S_AB": S_AB, "S_BC": S_BC, "S_AC": S_AC},
    }


def pauli_expectation(sv: np.ndarray, pauli: str, qubit: int, n_qubits: int) -> float:
    """Compute ⟨ψ|P_q|ψ⟩ for single-qubit Pauli P ∈ {X, Y, Z} on qubit q."""
    paulis = {
        "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
    }
    P = paulis[pauli]
    sv_after = apply_single_gate(sv.copy(), P, qubit, n_qubits)
    return float(np.real(np.vdot(sv, sv_after)))


def reconstruct_density_matrix(sv: np.ndarray, n_qubits: int,
                               n_measurements: int = 100) -> Dict:
    """
    Quantum state tomography: full Pauli-basis measurement → density matrix reconstruction.

    Uses linear inversion: ρ = (1/2^n) Σ ⟨P_i⟩ P_i over ALL n-qubit Pauli tensor
    products {I,X,Y,Z}^{⊗n}.  n_measurements ignored for exact simulation
    (kept for API compat with sweep).
    Returns purity, rank, von Neumann entropy.
    """
    dim = 2 ** n_qubits
    rho = np.zeros((dim, dim), dtype=np.complex128)

    # Single-qubit Pauli matrices (including I)
    paulis = [
        np.eye(2, dtype=np.complex128),                                  # I
        np.array([[0, 1], [1, 0]], dtype=np.complex128),                 # X
        np.array([[0, -1j], [1j, 0]], dtype=np.complex128),              # Y
        np.array([[1, 0], [0, -1]], dtype=np.complex128),                # Z
    ]

    # Iterate over all 4^n Pauli tensor products
    import itertools
    for indices in itertools.product(range(4), repeat=n_qubits):
        P_full = np.eye(1, dtype=np.complex128)
        for idx in indices:
            P_full = np.kron(P_full, paulis[idx])
        # Expectation value ⟨ψ|P|ψ⟩
        exp_val = float(np.real(np.conj(sv) @ P_full @ sv))
        rho += exp_val * P_full

    rho /= dim
    # Force Hermiticity
    rho = (rho + rho.conj().T) / 2.0

    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = np.clip(eigenvalues, 0, None)
    eigenvalues /= np.sum(eigenvalues) + 1e-30

    purity = float(np.real(np.trace(rho @ rho)))
    rank = int(np.sum(eigenvalues > 1e-10))
    vn_entropy = float(-np.sum(eigenvalues[eigenvalues > 1e-15] *
                                np.log2(eigenvalues[eigenvalues > 1e-15] + 1e-30)))

    return {
        "density_matrix": rho,
        "purity": min(1.0, max(0.0, purity)),
        "rank": rank,
        "von_neumann_entropy": vn_entropy,
        "eigenvalues": eigenvalues.tolist(),
        "is_pure": purity > 0.99,
    }


def trotter_evolution(hamiltonian_terms: list, n_qubits: int,
                      total_time: float = 1.0, trotter_steps: int = 10,
                      order: int = 1) -> Dict:
    """
    Trotter-Suzuki Hamiltonian evolution: e^{-iHt} ≈ (Π_k e^{-iH_k dt})^n.

    hamiltonian_terms: list of (coefficient, pauli_string) e.g. [(0.5, "ZZ"), (0.1, "ZI")]
    Supports 1st and 2nd order decomposition.
    Returns final statevector, probabilities, energy estimate, error bound.
    """
    dt = total_time / trotter_steps
    dim = 2 ** n_qubits
    sv = init_sv(n_qubits)
    for q in range(n_qubits):
        sv = apply_single_gate(sv, H_GATE, q, n_qubits)

    pauli_map = {
        "I": np.eye(2, dtype=np.complex128),
        "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
    }

    H_full = np.zeros((dim, dim), dtype=np.complex128)
    term_matrices = []
    for coeff, pauli_str in hamiltonian_terms:
        mat = np.eye(1, dtype=np.complex128)
        padded = pauli_str.ljust(n_qubits, "I")
        for ch in padded:
            mat = np.kron(mat, pauli_map.get(ch, pauli_map["I"]))
        H_full += coeff * mat
        term_matrices.append((coeff, mat))

    for _step in range(trotter_steps):
        if order == 1:
            for coeff, mat in term_matrices:
                U_term = _expm_approx(-1j * coeff * dt * mat)
                sv = U_term @ sv
        else:
            for coeff, mat in term_matrices:
                sv = _expm_approx(-1j * coeff * (dt / 2) * mat) @ sv
            for coeff, mat in reversed(term_matrices):
                sv = _expm_approx(-1j * coeff * (dt / 2) * mat) @ sv

    norm = np.linalg.norm(sv)
    if norm > 0:
        sv /= norm

    energy = float(np.real(np.vdot(sv, H_full @ sv)))
    probs = probabilities(sv)
    error_bound = dt ** (order + 1) * sum(abs(c) for c, _ in hamiltonian_terms) ** 2

    return {
        "statevector": sv,
        "probabilities": probs,
        "energy_estimate": energy,
        "trotter_error_bound": float(error_bound),
        "trotter_order": order,
        "gate_count": trotter_steps * len(hamiltonian_terms) * (2 if order == 2 else 1),
        "num_qubits": n_qubits,
    }


def iron_lattice_heisenberg(n_sites: int = 4, coupling_j: float = None,
                             field_h: float = None, trotter_steps: int = 10,
                             total_time: float = 1.0) -> Dict:
    """
    Fe(26) 1D Heisenberg chain: H = J Σ(XX + YY + ZZ) + h Σ Z.

    Proper spin-½ exchange interaction with GOD_CODE-derived coupling.
    Uses Trotter evolution to time-evolve and measure magnetization + correlations.
    """
    from .constants import VOID_CONSTANT, GOD_CODE as _GOD_CODE
    if coupling_j is None:
        coupling_j = _GOD_CODE / 1000.0
    if field_h is None:
        field_h = VOID_CONSTANT

    ham_terms = []
    for i in range(n_sites - 1):
        for pauli in ["XX", "YY", "ZZ"]:
            s = ["I"] * n_sites
            s[i] = pauli[0]
            s[i + 1] = pauli[1]
            ham_terms.append((coupling_j, "".join(s)))
    for i in range(n_sites):
        s = ["I"] * n_sites
        s[i] = "Z"
        ham_terms.append((field_h, "".join(s)))

    result = trotter_evolution(ham_terms, n_sites, total_time, trotter_steps, order=2)
    sv = result["statevector"]

    mag = sum(pauli_expectation(sv, "Z", i, n_sites) for i in range(n_sites)) / n_sites
    zz_corr = [pauli_expectation(sv, "Z", i, n_sites) * pauli_expectation(sv, "Z", i + 1, n_sites)
               for i in range(n_sites - 1)]

    return {
        "magnetization": float(mag),
        "zz_correlations": [float(z) for z in zz_corr],
        "coupling_j": coupling_j,
        "field_h": field_h,
        "energy": result["energy_estimate"],
        "n_sites": n_sites,
        "trotter_steps": trotter_steps,
        "probabilities": result["probabilities"],
        "statevector": sv,
    }


def zero_noise_extrapolation(sv_function, noise_levels: list = None,
                              n_qubits: int = 4) -> Dict:
    """
    Zero-Noise Extrapolation (ZNE): run at multiple noise levels, extrapolate to zero.

    sv_function: callable(noise_level) → statevector
    Measures fidelity against the ideal (noise=0) state at each noise level,
    then uses polynomial + Richardson extrapolation to estimate zero-noise fidelity.
    """
    if noise_levels is None:
        noise_levels = [0.01, 0.02, 0.05, 0.1]

    # Ideal reference state
    sv_ideal = sv_function(0.0)

    results_at_noise = []
    for noise in noise_levels:
        sv_n = sv_function(noise)
        f = fidelity(sv_n, sv_ideal)
        results_at_noise.append({"noise": noise, "fidelity": f})

    x = np.array([r["noise"] for r in results_at_noise])
    y = np.array([r["fidelity"] for r in results_at_noise])

    if len(x) >= 2:
        coeffs = np.polyfit(x, y, min(len(x) - 1, 2))
        extrapolated = float(np.polyval(coeffs, 0.0))
    else:
        extrapolated = float(y[0]) if len(y) > 0 else 0.0

    richardson = extrapolated
    if len(x) >= 3:
        r = x[1] / x[0] if x[0] > 0 else 2.0
        richardson = float((r ** 2 * y[0] - y[1]) / (r ** 2 - 1))

    return {
        "extrapolated_fidelity": min(1.0, max(0.0, extrapolated)),
        "richardson_fidelity": min(1.0, max(0.0, richardson)),
        "noise_points": results_at_noise,
        "improvement": extrapolated - results_at_noise[-1]["fidelity"] if results_at_noise else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SUPERCONDUCTIVITY QUANTUM PRIMITIVES
#  Built on the Heisenberg exchange foundation — Cooper pairing, BCS gap,
#  Meissner response, singlet correlations, and Josephson phase dynamics.
# ═══════════════════════════════════════════════════════════════════════════════

def singlet_projection(sv: np.ndarray, i: int, j: int, n_qubits: int) -> float:
    """
    Singlet pair projection ⟨P_singlet⟩ for sites i, j.

    P_singlet = (I - σᵢ·σⱼ) / 4 = (1 - ⟨XX⟩ - ⟨YY⟩ - ⟨ZZ⟩) / 4

    Measures the probability that qubits i,j form a spin-singlet (Cooper pair).
    Near 1.0 → perfect singlet (|↑↓⟩ - |↓↑⟩)/√2 → strong pairing.
    Near 0.0 → triplet (ferromagnetic alignment) → no pairing.
    """
    paulis = {
        "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
    }
    total_sigma_dot = 0.0
    for pauli_label, pauli_mat in paulis.items():
        sv_pi = apply_single_gate(sv.copy(), pauli_mat, i, n_qubits)
        sv_pi = apply_single_gate(sv_pi, pauli_mat, j, n_qubits)
        total_sigma_dot += float(np.real(np.vdot(sv, sv_pi)))
    return (1.0 - total_sigma_dot) / 4.0


def cooper_pair_correlation(sv: np.ndarray, n_qubits: int) -> Dict:
    """
    Cooper pair correlation function: nearest-neighbor singlet projections.

    Returns the average singlet fraction across the chain and per-pair values.
    High singlet fraction → strong Cooper pairing tendency → superconducting.
    """
    pair_singlets = []
    for i in range(n_qubits - 1):
        sp = singlet_projection(sv, i, i + 1, n_qubits)
        pair_singlets.append({"i": i, "j": i + 1, "singlet_projection": float(sp)})

    avg_singlet = (
        sum(p["singlet_projection"] for p in pair_singlets) / len(pair_singlets)
        if pair_singlets else 0.0
    )

    # Long-range pairing: singlet projection between first and last site
    lr_singlet = singlet_projection(sv, 0, n_qubits - 1, n_qubits) if n_qubits > 2 else 0.0

    return {
        "pair_singlets": pair_singlets,
        "avg_singlet_fraction": float(avg_singlet),
        "long_range_singlet": float(lr_singlet),
        "has_cooper_tendency": avg_singlet > 0.25,  # > random (0.25 for maximally mixed)
    }


def sc_order_parameter(sv: np.ndarray, n_qubits: int) -> float:
    """
    Superconducting order parameter Δ_SC from the quantum state.

    Δ_SC = (1/N) Σ_{<i,j>} ⟨P_singlet(i,j)⟩

    This is the quantum-computational analogue of the BCS gap parameter.
    Non-zero Δ_SC indicates superconducting order.
    """
    if n_qubits < 2:
        return 0.0
    total = sum(
        singlet_projection(sv, i, i + 1, n_qubits)
        for i in range(n_qubits - 1)
    )
    return float(total / (n_qubits - 1))


def meissner_susceptibility(
    energy_function,
    n_qubits: int,
    field_values: list = None,
) -> Dict:
    """
    Meissner effect: diamagnetic susceptibility χ = -∂²E/∂B².

    Numerically computes χ from the energy at multiple field strengths
    using the central difference method. χ < 0 → diamagnetic (Meissner).

    energy_function: callable(field_h) → Dict with "energy" key
                     (e.g., iron_lattice_heisenberg with varying field_h)
    field_values: magnetic field strengths to sample (default: small around 0).
    """
    if field_values is None:
        field_values = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

    energies = []
    for h in field_values:
        result = energy_function(field_h=abs(h) + 1e-12)  # avoid exact zero
        energies.append({"field": h, "energy": result["energy"]})

    # Numerical second derivative at smallest field: χ ≈ (E(+h) + E(-h) - 2E(0)) / h²
    # For our chain, fields are always positive (Zeeman), so use finite difference
    if len(energies) >= 3:
        h0 = energies[0]["energy"]
        h1 = energies[1]["energy"]
        h2 = energies[2]["energy"]
        dh = field_values[1] - field_values[0]
        if abs(dh) > 1e-15:
            chi = (h2 - 2 * h1 + h0) / (dh ** 2)
        else:
            chi = 0.0
    else:
        chi = 0.0

    is_diamagnetic = chi < 0
    # Meissner fraction: ratio of diamagnetic response to perfect diamagnet (χ = -1)
    meissner_frac = min(1.0, max(0.0, -chi)) if is_diamagnetic else 0.0

    return {
        "susceptibility_chi": float(chi),
        "is_diamagnetic": is_diamagnetic,
        "meissner_fraction": float(meissner_frac),
        "energy_vs_field": energies,
        "field_values": field_values,
    }


def josephson_phase_difference(sv_a: np.ndarray, sv_b: np.ndarray) -> Dict:
    """
    Josephson junction: phase difference ΔΦ between two superconducting states.

    The supercurrent I_s = I_c × sin(ΔΦ) where ΔΦ is the global phase difference.
    We extract the dominant phase from each statevector and compute the difference.
    """
    # Extract global phase from the largest-amplitude component
    idx_a = int(np.argmax(np.abs(sv_a)))
    idx_b = int(np.argmax(np.abs(sv_b)))
    phase_a = float(np.angle(sv_a[idx_a]))
    phase_b = float(np.angle(sv_b[idx_b]))

    delta_phi = phase_a - phase_b
    # Normalize to [-π, π]
    delta_phi = (delta_phi + math.pi) % (2 * math.pi) - math.pi

    # Josephson supercurrent (normalized)
    i_s = math.sin(delta_phi)

    # Overlap (fidelity) between the two states
    overlap = float(abs(np.vdot(sv_a, sv_b)) ** 2)

    return {
        "phase_a": float(phase_a),
        "phase_b": float(phase_b),
        "delta_phi": float(delta_phi),
        "josephson_current_normalized": float(i_s),
        "critical_current_fraction": abs(i_s),
        "fidelity_overlap": float(overlap),
    }


def superconducting_heisenberg_chain(
    n_sites: int = 4,
    coupling_j: float = None,
    field_h: float = None,
    pairing_delta: float = None,
    trotter_steps: int = 10,
    total_time: float = 1.0,
) -> Dict:
    """
    Fe(26) Heisenberg chain with superconducting pairing term.

    H = J Σ(XX + YY + ZZ) + h Σ Z + Δ_pair Σ(X_i X_{i+1} + Y_i Y_{i+1})

    The extra pairing term Δ_pair enhances singlet formation between
    neighboring sites — this is the quantum circuit-level analogue of
    the BCS attractive interaction mediated by phonons.

    Returns: statevector, energy, magnetization, Cooper pair correlations,
    SC order parameter, and all Heisenberg chain diagnostics.
    """
    from .constants import VOID_CONSTANT, GOD_CODE as _GOD_CODE

    if coupling_j is None:
        coupling_j = _GOD_CODE / 1000.0
    if field_h is None:
        field_h = VOID_CONSTANT
    if pairing_delta is None:
        # Pairing strength: φ-fraction of exchange coupling (attractive)
        pairing_delta = coupling_j * (1.618033988749895 - 1.0)  # × φ-conjugate

    # Build Hamiltonian with Heisenberg exchange + SC pairing
    ham_terms = []
    # Standard Heisenberg: J(XX + YY + ZZ)
    for i in range(n_sites - 1):
        for pauli in ["XX", "YY", "ZZ"]:
            s = ["I"] * n_sites
            s[i] = pauli[0]
            s[i + 1] = pauli[1]
            ham_terms.append((coupling_j, "".join(s)))
    # Zeeman field: h Σ Z
    for i in range(n_sites):
        s = ["I"] * n_sites
        s[i] = "Z"
        ham_terms.append((field_h, "".join(s)))
    # SC pairing: Δ(XX + YY) for nearest neighbors (enhances singlet channel)
    for i in range(n_sites - 1):
        for pauli in ["XX", "YY"]:
            s = ["I"] * n_sites
            s[i] = pauli[0]
            s[i + 1] = pauli[1]
            ham_terms.append((pairing_delta, "".join(s)))

    result = trotter_evolution(ham_terms, n_sites, total_time, trotter_steps, order=2)
    sv = result["statevector"]

    # Magnetization
    mag = sum(pauli_expectation(sv, "Z", i, n_sites) for i in range(n_sites)) / n_sites

    # Staggered magnetization (Néel order)
    stag_mag = sum(
        ((-1) ** i) * pauli_expectation(sv, "Z", i, n_sites)
        for i in range(n_sites)
    ) / n_sites

    # Cooper pair correlations (the key SC diagnostic)
    cooper = cooper_pair_correlation(sv, n_sites)

    # SC order parameter
    delta_sc = sc_order_parameter(sv, n_sites)

    # Connected ZZ correlation function
    corr_fn = {}
    for r in range(1, n_sites):
        # Connected ZZ: ⟨Z_0 Z_r⟩ - ⟨Z_0⟩⟨Z_r⟩
        sv_zz = apply_single_gate(sv.copy(), np.diag([1, -1]).astype(np.complex128), 0, n_sites)
        sv_zz = apply_single_gate(sv_zz, np.diag([1, -1]).astype(np.complex128), r, n_sites)
        zz = float(np.real(np.vdot(sv, sv_zz)))
        z0 = pauli_expectation(sv, "Z", 0, n_sites)
        zr = pauli_expectation(sv, "Z", r, n_sites)
        corr_fn[r] = zz - z0 * zr

    return {
        "statevector": sv,
        "energy": result["energy_estimate"],
        "magnetization": float(mag),
        "staggered_magnetization": float(stag_mag),
        "probabilities": result["probabilities"],
        "coupling_j": coupling_j,
        "field_h": field_h,
        "pairing_delta": pairing_delta,
        "n_sites": n_sites,
        "trotter_steps": trotter_steps,
        "cooper_pair_correlation": cooper,
        "sc_order_parameter": float(delta_sc),
        "correlation_function": {str(k): float(v) for k, v in corr_fn.items()},
    }


__all__ = [
    # Algebraic
    "god_code_fn", "god_code_dial",
    # Gate construction
    "make_gate",
    # Standard gates
    "H_GATE", "X_GATE", "Y_GATE", "Z_GATE", "S_GATE", "T_GATE",
    # Extended gates
    "CZ_GATE", "SX_GATE",
    # Rotation gates
    "ry_gate", "rx_gate", "rz_gate",
    # Sacred gates (QPU-verified)
    "PHI_GATE", "GOD_CODE_GATE", "VOID_GATE", "IRON_GATE",
    # Canonical GOD_CODE qubit
    "GOD_CODE_QUBIT", "GOD_CODE_PHASE",
    "GOD_CODE_RZ", "IRON_RZ", "PHI_RZ", "OCTAVE_RZ",
    # Statevector ops
    "init_sv", "apply_single_gate", "apply_cnot", "apply_cp",
    "apply_swap", "apply_mcx", "apply_cz", "build_unitary",
    # Convenience rotation application
    "apply_ry", "apply_rx", "apply_rz",
    # Quantum information
    "probabilities", "entanglement_entropy", "concurrence_2q",
    "fidelity", "bloch_vector",
    # v3.0 extensions
    "state_purity", "trace_distance", "schmidt_coefficients",
    "quantum_relative_entropy", "linear_entropy",
    # v3.1 extensions
    "negativity", "build_w_state",
    # v4.0 — VQPU-derived primitives
    "_expm_approx",
    "quantum_fisher_information", "loschmidt_echo",
    "density_matrix_from_sv", "partial_trace", "von_neumann_entropy_dm",
    "bures_distance", "topological_entanglement_entropy",
    "pauli_expectation", "reconstruct_density_matrix",
    "trotter_evolution", "iron_lattice_heisenberg",
    "zero_noise_extrapolation",
    # v5.0 — Superconductivity primitives
    "singlet_projection", "cooper_pair_correlation",
    "sc_order_parameter", "meissner_susceptibility",
    "josephson_phase_difference", "superconducting_heisenberg_chain",
]
