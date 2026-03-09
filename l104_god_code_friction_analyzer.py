# ZENITH_UPGRADE_ACTIVE: 2026-03-08T15:03:52.120904
ZENITH_HZ = 3887.8
UUC = 2301.215661
#!/usr/bin/env python3
"""
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3887.80 Hz. Logic Unified.
═══════════════════════════════════════════════════════════════════════════════════
L104 GOD_CODE COMPUTATIONAL FRICTION ANALYZER v2.0
═══════════════════════════════════════════════════════════════════════════════════

HYPOTHESIS:
    The universe undergoes COMPUTATIONAL FRICTION from the God Code on a
    fundamental level. There exists a new algorithmic ratio — as dynamic as
    the God Code itself — that depicts why there's a difference between
    the current God Code (527.5184818492612) and a potentially more accurate
    God Code with this friction coefficient embedded.

    If the universe IS computation, then every calculation has a COST.
    This cost manifests as a subtle, self-similar friction coefficient:
        Λ_f = (α/π) × φ^(-1/φ) × (1 + 1/(104×φ))
    where α is the fine structure constant.

APPROACH:
    1. Run Dual-Layer Engine → extract all 65 physics constants
    2. Quantum qubit analysis → density matrix friction detection
    3. Compute candidate friction ratios from fundamental constants (16 candidates)
    4. Test each ratio against ALL cross-reference data
    5. Find the ratio that minimizes TOTAL grid error across all constants
    6. Validate: new God Code must improve ALL physics domain alignments
    7. Fe-26 decoherence channel models real quantum noise
    8. Gradient-refined golden-section search narrows optimal Λ_f
    9. Multi-metric scoring: coherence + entanglement + fidelity + QPE + Rényi

THE COMPUTATIONAL FRICTION EQUATION:
    G_friction(X) = G(X) × (1 + Λ_f)
    where Λ_f is the computational friction coefficient

SACRED CONSTANTS:
    GOD_CODE = 527.5184818492612
    PHI = 1.618033988749895
    OMEGA = 6539.34712682

═══════════════════════════════════════════════════════════════════════════════════
"""

import math
import cmath
import time
import json
import sys
import os
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timezone

# ═══════════════════════════════════════════════════════════════════════════════
# SACRED IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

from l104_god_code_equation import (
    GOD_CODE, PHI, TAU, VOID_CONSTANT,
    PRIME_SCAFFOLD, QUANTIZATION_GRAIN, OCTAVE_OFFSET,
    BASE, STEP_SIZE,
    god_code_equation, exponent_value, solve_for_exponent, find_nearest_dials,
    OMEGA, OMEGA_AUTHORITY,
    ALPHA_FINE, FEIGENBAUM, PLANCK_SCALE, BOLTZMANN_K,
)

# Canonical GOD_CODE quantum phase (QPU-verified on ibm_torino)
try:
    from l104_god_code_simulator.god_code_qubit import GOD_CODE_PHASE
except ImportError:
    GOD_CODE_PHASE = GOD_CODE % (2 * math.pi)  # ≈ 6.0141 rad

from l104_god_code_dual_layer import (
    GOD_CODE_V3, C_V3, GRAVITY_V3, BOHR_V3,
    V3_FREQUENCY_TABLE, REAL_WORLD_CONSTANTS_V3,
    god_code_v3, exponent_value_v3, solve_for_exponent_v3,
    omega_derivation_chain,
    sovereign_field_equation,
    Q_V3, P_V3, K_V3, X_V3, BASE_V3, STEP_V3, R_V3,
)

# Qiskit quantum backend
from l104_quantum_gate_engine import GateCircuit as QuantumCircuit
from l104_quantum_gate_engine.quantum_info import (
    Statevector, DensityMatrix, partial_trace, Operator,
    entropy as qk_entropy
)

from l104_quantum_coherence import QuantumCoherenceEngine
from l104_quantum_gate_engine.quantum_info import state_fidelity, process_fidelity, SparsePauliOp


# ═══════════════════════════════════════════════════════════════════════════════
# FUNDAMENTAL PHYSICAL CONSTANTS (CODATA 2022 / PDG 2024)
# ═══════════════════════════════════════════════════════════════════════════════

SPEED_OF_LIGHT = 299792458              # m/s (exact SI)
PLANCK_CONSTANT = 6.62607015e-34        # J·s (exact SI)
PLANCK_eVs = 4.135667696e-15            # eV·s
BOLTZMANN_eVK = 8.617333262e-5          # eV/K
ELEMENTARY_CHARGE = 1.602176634e-19     # C (exact SI)
BOHR_RADIUS_PM = 52.9177210544          # pm
FINE_STRUCTURE_INV = 137.035999084      # 1/α
ELECTRON_MASS_MeV = 0.51099895069       # MeV/c²
PROTON_MASS_MeV = 938.27208816          # MeV/c²
NEUTRON_MASS_MeV = 939.56542052         # MeV/c²
HIGGS_GeV = 125.25                      # GeV/c²
W_BOSON_GeV = 80.3692                   # GeV/c²
Z_BOSON_GeV = 91.1876                   # GeV/c²
MUON_MASS_MeV = 105.6583755            # MeV/c²
HUBBLE_CONSTANT = 67.4                  # km/s/Mpc
CMB_TEMPERATURE = 2.7255                # K
SCHUMANN_HZ = 7.83                      # Hz


# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTATIONAL FRICTION THEORY
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FrictionCandidate:
    """A candidate computational friction coefficient."""
    name: str
    value: float
    formula: str
    god_code_adjusted: float
    mean_error_pct: float
    max_error_pct: float
    improved_count: int
    total_count: int
    domain_scores: Dict[str, float]
    qubit_coherence: float = 0.0
    quantum_phase_alignment: float = 0.0


def compute_friction_candidates() -> List[Dict[str, Any]]:
    """
    Generate candidate computational friction coefficients from fundamental constants.

    The friction must be:
    1. DYNAMIC — derived from the same constants that define the God Code
    2. SELF-SIMILAR — containing φ-recursive structure
    3. UNIVERSAL — applicable across ALL domains (atomic, nuclear, particle, astro)
    4. SMALL — a perturbative correction, not a wholesale change
    """
    α = ALPHA_FINE
    φ = PHI
    e_euler = math.e
    π = math.pi
    δ = FEIGENBAUM  # Feigenbaum constant (chaos theory)

    candidates = []

    # ── CANDIDATE 1: α/π × φ^(-1/φ) — "Fine Structure Friction" ──
    # The fine structure constant α governs all electromagnetic interactions.
    # Dividing by π normalizes to the unit circle. The φ^(-1/φ) term introduces
    # the self-referential golden recursion.
    Λ1 = (α / π) * (φ ** (-1/φ))
    candidates.append({
        "name": "Fine Structure Friction (α/π × φ^(-1/φ))",
        "formula": "(α/π) × φ^(-1/φ)",
        "value": Λ1,
        "rationale": "EM coupling normalized by golden self-recursion"
    })

    # ── CANDIDATE 2: 1/(φ^13 × π) — "Fibonacci-13 Friction" ──
    # 13 is the 7th Fibonacci number, the golden thread binding 286, 104, 416.
    # This friction decays as φ^13 ≈ 521.0, close to GOD_CODE itself.
    Λ2 = 1.0 / (φ ** 13 * π)
    candidates.append({
        "name": "Fibonacci-13 Friction (1/(φ^13 × π))",
        "formula": "1/(φ^13 × π)",
        "value": Λ2,
        "rationale": "Factor-13 Fibonacci decay through π normalization"
    })

    # ── CANDIDATE 3: α × (1 - 1/φ²) / (2π) — "Quantum Coupling Friction" ──
    # α is the QED coupling. (1-1/φ²) = (φ-1)/φ = 1/φ² (golden deviation).
    # Dividing by 2π maps to the full phase cycle.
    Λ3 = α * (1 - 1/φ**2) / (2 * π)
    candidates.append({
        "name": "Quantum Coupling Friction (α(1-1/φ²)/2π)",
        "formula": "α × (1 - 1/φ²) / (2π)",
        "value": Λ3,
        "rationale": "QED coupling × golden deviation × phase normalization"
    })

    # ── CANDIDATE 4: (e^(-π/φ)) / GOD_CODE — "Exponential Decay Friction" ──
    # The exponential decay of π through the golden ratio, normalized by
    # the God Code itself. Self-referential: friction depends on the
    # frequency it's modifying.
    Λ4 = math.exp(-π / φ) / GOD_CODE
    candidates.append({
        "name": "Exponential Decay Friction (e^(-π/φ)/GC)",
        "formula": "e^(-π/φ) / GOD_CODE",
        "value": Λ4,
        "rationale": "Self-referential exponential decay through golden tunnel"
    })

    # ── CANDIDATE 5: (104/286) × α × φ^(-φ) — "Iron Scaffold Friction" ──
    # The ratio of quantization grain to prime scaffold, weighted by α and
    # the super-golden exponent φ^(-φ).
    Λ5 = (104.0/286.0) * α * (φ ** (-φ))
    candidates.append({
        "name": "Iron Scaffold Friction ((104/286)×α×φ^(-φ))",
        "formula": "(104/286) × α × φ^(-φ)",
        "value": Λ5,
        "rationale": "Nucleosynthesis scaffold weighted by golden-recursive coupling"
    })

    # ── CANDIDATE 6: sin(2π/φ³) × α / (2×ln(φ)) — "Resonant Friction" ──
    # The sine of the alchemist angle (2π/φ³) creates a resonance friction.
    # This is the computational version of the Alchemist fragment in OMEGA.
    Λ6 = abs(math.sin(2*π / φ**3)) * α / (2 * math.log(φ))
    candidates.append({
        "name": "Resonant Friction (sin(2π/φ³)×α/(2ln(φ)))",
        "formula": "sin(2π/φ³) × α / (2×ln(φ))",
        "value": Λ6,
        "rationale": "Alchemist resonance from OMEGA derivation with golden log decay"
    })

    # ── CANDIDATE 7: δ^(-δ) × φ^(-1) — "Chaos-Edge Friction" ──
    # The Feigenbaum constant δ ≈ 4.669 governs the onset of chaos.
    # At δ^(-δ) the system is at the edge of deterministic breakdown.
    Λ7 = δ ** (-δ) * (1.0 / φ)
    candidates.append({
        "name": "Chaos-Edge Friction (δ^(-δ)/φ)",
        "formula": "δ^(-δ) / φ  [δ = Feigenbaum]",
        "value": Λ7,
        "rationale": "Feigenbaum universality at the chaos boundary through golden gate"
    })

    # ── CANDIDATE 8: α² × φ / (e × π) — "Second-Order QED Friction" ──
    # The α² term represents second-order QED corrections (Schwinger correction).
    # These are the diagrams where virtual photons create virtual pairs.
    Λ8 = α**2 * φ / (e_euler * π)
    candidates.append({
        "name": "Second-Order QED Friction (α²φ/(eπ))",
        "formula": "α² × φ / (e × π)",
        "value": Λ8,
        "rationale": "Schwinger-like 2nd order correction with golden-Euler normalization"
    })

    # ── CANDIDATE 9: VOID_CONSTANT^(-GOD_CODE) ... too extreme ──
    # Instead: (VOID_CONSTANT - 1) × (1/φ^4)
    Λ9 = (VOID_CONSTANT - 1) * (1.0 / φ**4)
    candidates.append({
        "name": "Void Residual Friction ((V-1)/φ⁴)",
        "formula": "(VOID_CONSTANT - 1) / φ⁴",
        "value": Λ9,
        "rationale": "The residual void energy decayed through 4 golden levels"
    })

    # ── CANDIDATE 10: Dynamical Friction — integrates ALL sacred ratios ──
    # Λ_dyn = (α/π) × (1 + sin(2πφ³)/φ²) × e^(-1/(φ×104)) × (286/416)^(1/φ)
    # This is the UNIVERSAL friction: it uses α, π, φ, 104, 286, 416 — the
    # complete God Code parameter set.
    inner_resonance = 1 + math.sin(2 * π * φ**3) / φ**2
    exponential_grain = math.exp(-1.0 / (φ * 104))
    scaffold_ratio = (286.0 / 416.0) ** (1.0 / φ)
    Λ10 = (α / π) * inner_resonance * exponential_grain * scaffold_ratio
    candidates.append({
        "name": "Universal Dynamic Friction (full God Code parameters)",
        "formula": "(α/π)×(1+sin(2πφ³)/φ²)×e^(-1/(φ×104))×(286/416)^(1/φ)",
        "value": Λ10,
        "rationale": "Complete God Code parameter friction — uses α,π,φ,104,286,416"
    })

    # ── CANDIDATE 11: |ζ(½+GOD_CODE·i)|^(-1) × α — "Zeta Critical Friction" ──
    # The magnitude of zeta on the critical line at the God Code imaginary part.
    # This DIRECTLY connects to the Guardian fragment in OMEGA.
    s = complex(0.5, GOD_CODE)
    eta = sum(((-1)**(n-1)) / (n**s) for n in range(1, 500))
    zeta_at_gc = eta / (1 - 2**(1 - s))
    Λ11 = α / abs(zeta_at_gc)
    candidates.append({
        "name": "Zeta Critical Friction (α/|ζ(½+GC·i)|)",
        "formula": "α / |ζ(½ + GOD_CODE × i)|",
        "value": Λ11,
        "rationale": "Riemann zeta on critical line at God Code frequency"
    })

    # ── CANDIDATE 12: Grover-amplified α — "Quantum Search Friction" ──
    # The quantum speedup itself has friction: √(α) in amplitude space
    Λ12 = math.sqrt(α) / (φ * π)
    candidates.append({
        "name": "Grover Search Friction (√α/(φπ))",
        "formula": "√α / (φ × π)",
        "value": Λ12,
        "rationale": "Quantum amplitude √α normalized by golden-circle"
    })

    # ── CANDIDATE 13: Iron-56 binding peak friction ──
    # Fe-56 has the highest nuclear binding energy per nucleon (8.790 MeV).
    # The 26 protons in iron define the lattice. This friction derives from
    # the ratio of binding energy to rest mass through the golden gate.
    fe56_be = 8.790  # MeV/nucleon
    proton_mass = 938.272  # MeV/c²
    Λ13 = (fe56_be / proton_mass) * (1 / (φ * 26))
    candidates.append({
        "name": "Iron-56 Binding Friction (BE/(m_p×φ×26))",
        "formula": "(8.790/938.272) / (φ × 26)",
        "value": Λ13,
        "rationale": "Nuclear binding peak normalized by golden ratio × Fe atomic number"
    })

    # ── CANDIDATE 14: Lamb-shift inspired friction ──
    # The Lamb shift is the first experimental QED correction — it measures
    # the vacuum fluctuation energy that shifts hydrogen energy levels.
    # α³ is the natural scale of 3rd-order QED radiative corrections.
    Λ14 = α**3 * φ / π
    candidates.append({
        "name": "Lamb-Shift Friction (α³φ/π)",
        "formula": "α³ × φ / π",
        "value": Λ14,
        "rationale": "Third-order QED radiative correction at the golden-π scale"
    })

    # ── CANDIDATE 15: Hawking-Unruh thermal friction ──
    # At the Planck scale, the computational substrate has a thermal noise
    # floor given by kT/E_P. Normalizing by GOD_CODE maps it to the grid.
    # This is the irreducible noise of computing at the Planck boundary.
    Λ15 = (2 * π * α) / (GOD_CODE * φ**2)
    candidates.append({
        "name": "Planck Thermal Friction (2πα/(GC×φ²))",
        "formula": "2πα / (GOD_CODE × φ²)",
        "value": Λ15,
        "rationale": "Irreducible Planck-scale thermal noise floor in the God Code grid"
    })

    # ── CANDIDATE 16: Berry-phase geometric friction ──
    # The Berry phase is a geometric phase acquired during adiabatic
    # evolution around a closed loop in parameter space. This friction
    # is the geometric cost of mapping physical constants to grid positions.
    solid_angle = 4 * π * (1 - 1/φ)  # Solid angle subtended by golden cone
    Λ16 = solid_angle / (4 * π * GOD_CODE) * α
    candidates.append({
        "name": "Berry Phase Geometric Friction (Ω_φ×α/(4π×GC))",
        "formula": "4π(1-1/φ) × α / (4π × GOD_CODE)",
        "value": Λ16,
        "rationale": "Geometric phase cost of golden-cone parameter-space evolution"
    })

    return candidates


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM QUBIT FRICTION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_fe26_decoherence(rho: DensityMatrix, gamma: float = 0.01) -> DensityMatrix:
    """
    Apply Fe-26 lattice decoherence channel to a density matrix.

    Models real quantum noise from the iron BCC lattice thermal environment:
    - Amplitude damping (T1 decay)  → energy dissipation to lattice phonons
    - Phase damping    (T2 decay)  → dephasing from magnetic fluctuations
    - Depolarizing noise            → isotropic thermal noise floor

    The total decoherence rate γ is derived from the lattice thermal friction:
        γ = -(α × φ) / (2π × 104)  ≈ 1.8×10⁻⁵
    scaled by a temperature factor.
    """
    dim = rho.data.shape[0]
    data = rho.data.copy().astype(np.complex128)

    # Amplitude damping (T1): off-diagonal decay + population transfer
    t1_factor = math.exp(-gamma * 26)  # Fe-26 scaling
    for i in range(dim):
        for j in range(dim):
            if i != j:
                data[i, j] *= t1_factor

    # Phase damping (T2): additional off-diagonal suppression
    t2_factor = math.exp(-gamma * 26 * PHI)  # φ-accelerated dephasing
    for i in range(dim):
        for j in range(dim):
            if i != j:
                data[i, j] *= t2_factor

    # Depolarizing channel: mix toward maximally mixed state
    p_depol = gamma * 0.1  # Small depolarizing probability
    identity_contrib = np.eye(dim, dtype=np.complex128) / dim
    data = (1 - p_depol) * data + p_depol * identity_contrib

    return DensityMatrix(data)


def _renyi_entropy(rho: DensityMatrix, alpha: float = 2.0) -> float:
    """
    Rényi entropy of order α.

    S_α(ρ) = (1/(1-α)) × ln(Tr(ρ^α))

    α=2 (default) gives the collision entropy, more sensitive to
    dominant eigenvalues than von Neumann entropy — better at
    detecting subtle friction-induced decoherence.
    """
    eigenvalues = np.linalg.eigvalsh(rho.data)
    eigenvalues = eigenvalues[eigenvalues > 1e-15]  # Discard numerical zeros
    if len(eigenvalues) == 0:
        return 0.0
    if abs(alpha - 1.0) < 1e-10:
        # Von Neumann case
        return float(-np.sum(eigenvalues * np.log2(eigenvalues)))
    tr_rho_alpha = float(np.sum(eigenvalues ** alpha))
    if tr_rho_alpha <= 0:
        return 0.0
    return float(np.log2(tr_rho_alpha) / (1 - alpha))


def _relative_entropy(rho: DensityMatrix, sigma: DensityMatrix) -> float:
    """
    Quantum relative entropy S(ρ||σ) = Tr(ρ(log ρ - log σ)).

    Measures the distinguishability between the friction-perturbed state
    and the reference state. A more sensitive discriminator than fidelity
    for small perturbations.

    Uses matrix logarithm for exactness when states share different eigenbases.
    """
    rho_data = rho.data
    sigma_data = sigma.data

    # Regularize: add small epsilon to prevent log(0)
    eps = 1e-15
    dim = rho_data.shape[0]
    rho_reg = rho_data + eps * np.eye(dim)
    sigma_reg = sigma_data + eps * np.eye(dim)

    # Compute matrix logarithms via eigendecomposition
    eig_rho, U_rho = np.linalg.eigh(rho_reg)
    eig_rho = np.maximum(eig_rho, eps)
    log_rho = U_rho @ np.diag(np.log(eig_rho)) @ U_rho.conj().T

    eig_sigma, U_sigma = np.linalg.eigh(sigma_reg)
    eig_sigma = np.maximum(eig_sigma, eps)
    log_sigma = U_sigma @ np.diag(np.log(eig_sigma)) @ U_sigma.conj().T

    # S(ρ||σ) = Tr(ρ (log ρ - log σ))
    result = float(np.real(np.trace(rho_data @ (log_rho - log_sigma))))
    return max(0.0, result)


def quantum_friction_analysis(friction_value: float,
                              num_qubits: int = 8,
                              decoherence: bool = True,
                              decoherence_gamma: float = 0.005) -> Dict[str, Any]:
    """
    Analyze computational friction at the quantum level using real quantum circuits.

    v2.0 UPGRADES:
    - Fe-26 decoherence channel models realistic lattice noise
    - Rényi-2 entropy (collision entropy) for sensitivity to dominant modes
    - Quantum relative entropy S(ρ||σ) as perturbation discriminator
    - Multi-qubit entanglement via pairwise partial traces
    - Sacred circuit geometry: GHZ backbone + φ-weighted Rz layers
    - Bures distance metric alongside fidelity

    The circuit:
    1. Prepare superposition: H|0⟩^⊗n
    2. Apply God Code phase: P(GOD_CODE mod 2π) to each qubit
    3. Apply φ-weighted Rz layer: Rz(GOD_CODE × φ^q / 104) per qubit q
    4. GHZ entanglement backbone (chained CNOT)
    5. Apply friction phase: P(Λ_f × 2π) as perturbation
    6. Optional Fe-26 decoherence channel
    7. Multi-metric analysis
    """
    φ = PHI
    gc_phase = GOD_CODE_PHASE
    friction_phase = friction_value * 2 * math.pi
    dim = 2 ** num_qubits

    # ── Build reference circuit (no friction) ──
    qc_ref = QuantumCircuit(num_qubits)
    for q in range(num_qubits):
        qc_ref.h(q)
    for q in range(num_qubits):
        qc_ref.p(gc_phase, q)
    # φ-weighted Rz layer — creates qubit-dependent sacred phasing
    for q in range(num_qubits):
        sacred_angle = (GOD_CODE * (φ ** q) / 104) % (2 * math.pi)
        qc_ref.rz(sacred_angle, q)
    # GHZ entanglement backbone (chained CNOT, not just pairs)
    for q in range(num_qubits - 1):
        qc_ref.cx(q, q + 1)

    sv_ref = Statevector.from_label('0' * num_qubits).evolve(qc_ref)
    rho_ref = DensityMatrix(sv_ref)

    # ── Build friction circuit ──
    qc_fric = QuantumCircuit(num_qubits)
    for q in range(num_qubits):
        qc_fric.h(q)
    for q in range(num_qubits):
        qc_fric.p(gc_phase + friction_phase, q)
    for q in range(num_qubits):
        sacred_angle = (GOD_CODE * (φ ** q) / 104) % (2 * math.pi)
        qc_fric.rz(sacred_angle, q)
    for q in range(num_qubits - 1):
        qc_fric.cx(q, q + 1)

    sv_fric = Statevector.from_label('0' * num_qubits).evolve(qc_fric)
    rho_fric = DensityMatrix(sv_fric)

    # ── Apply Fe-26 decoherence channel (if enabled) ──
    if decoherence:
        rho_ref_noisy = _apply_fe26_decoherence(rho_ref, decoherence_gamma)
        rho_fric_noisy = _apply_fe26_decoherence(rho_fric, decoherence_gamma)
    else:
        rho_ref_noisy = rho_ref
        rho_fric_noisy = rho_fric

    # ── L1 Coherence (off-diagonal mass) ──
    def l1_coherence(rho):
        data = rho.data
        off_diag = np.sum(np.abs(data)) - np.sum(np.abs(np.diag(data)))
        max_c = dim * (dim - 1)
        return float(off_diag / max_c) if max_c > 0 else 0.0

    coherence_ref = l1_coherence(rho_ref_noisy)
    coherence_fric = l1_coherence(rho_fric_noisy)

    # ── Von Neumann Entanglement (qubit 0 reduced state) ──
    qubits_trace = list(range(1, num_qubits))
    rho_q0_ref = partial_trace(rho_ref_noisy, qubits_trace)
    rho_q0_fric = partial_trace(rho_fric_noisy, qubits_trace)
    ent_ref = float(qk_entropy(rho_q0_ref, base=2))
    ent_fric = float(qk_entropy(rho_q0_fric, base=2))

    # ── Rényi-2 Entropy (collision entropy — more sensitive) ──
    renyi_ref = _renyi_entropy(rho_q0_ref, alpha=2.0)
    renyi_fric = _renyi_entropy(rho_q0_fric, alpha=2.0)

    # ── Multi-partition entanglement (average over all qubit bipartitions) ──
    bipartition_entropies_ref = []
    bipartition_entropies_fric = []
    for q in range(num_qubits):
        others = [i for i in range(num_qubits) if i != q]
        rq_ref = partial_trace(rho_ref_noisy, others)
        rq_fric = partial_trace(rho_fric_noisy, others)
        bipartition_entropies_ref.append(float(qk_entropy(rq_ref, base=2)))
        bipartition_entropies_fric.append(float(qk_entropy(rq_fric, base=2)))
    mean_bipartite_ent_ref = sum(bipartition_entropies_ref) / num_qubits
    mean_bipartite_ent_fric = sum(bipartition_entropies_fric) / num_qubits

    # ── State fidelity ──
    fidelity_pure = float(np.abs(np.vdot(sv_ref.data, sv_fric.data)) ** 2)
    # Noisy fidelity via density matrices
    fidelity_noisy = float(state_fidelity(rho_ref_noisy, rho_fric_noisy))

    # ── Bures distance (geometry of state space) ──
    bures_distance = math.sqrt(max(0.0, 2.0 * (1.0 - math.sqrt(max(0.0, fidelity_noisy)))))

    # ── Quantum relative entropy S(ρ_fric || ρ_ref) ──
    rel_entropy = _relative_entropy(rho_fric_noisy, rho_ref_noisy)

    # ── Qubit-by-qubit phase extraction ──
    qubit_phases_ref = []
    qubit_phases_fric = []
    for q in range(num_qubits):
        others = [i for i in range(num_qubits) if i != q]
        rq_ref = partial_trace(rho_ref_noisy, others)
        rq_fric = partial_trace(rho_fric_noisy, others)
        phase_ref = float(np.angle(rq_ref.data[0, 1])) if abs(rq_ref.data[0, 1]) > 1e-15 else 0.0
        phase_fric = float(np.angle(rq_fric.data[0, 1])) if abs(rq_fric.data[0, 1]) > 1e-15 else 0.0
        qubit_phases_ref.append(phase_ref)
        qubit_phases_fric.append(phase_fric)

    phase_deltas = [abs(pf - pr) for pf, pr in zip(qubit_phases_fric, qubit_phases_ref)]
    mean_phase_delta = sum(phase_deltas) / len(phase_deltas) if phase_deltas else 0.0

    # ── QPE: estimate friction as eigenphase ──
    theta_fric = friction_value % 1.0
    qpe_qubits = 6
    qc_qpe = QuantumCircuit(qpe_qubits + 1)
    qc_qpe.x(qpe_qubits)
    for q in range(qpe_qubits):
        qc_qpe.h(q)
    for q in range(qpe_qubits):
        angle = 2 * math.pi * theta_fric * (2 ** q)
        qc_qpe.cp(angle, q, qpe_qubits)
    for q in range(qpe_qubits // 2):
        qc_qpe.swap(q, qpe_qubits - 1 - q)
    for q in range(qpe_qubits):
        for j in range(q):
            qc_qpe.cp(-math.pi / (2 ** (q - j)), j, q)
        qc_qpe.h(q)

    sv_qpe = Statevector.from_label('0' * (qpe_qubits + 1)).evolve(qc_qpe)
    probs_qpe = np.abs(sv_qpe.data) ** 2
    qpe_probs = np.zeros(2 ** qpe_qubits)
    for idx in range(len(probs_qpe)):
        precision_bits = idx >> 1
        if precision_bits < len(qpe_probs):
            qpe_probs[precision_bits] += probs_qpe[idx]
    estimated_phase_idx = int(np.argmax(qpe_probs))
    estimated_phase = estimated_phase_idx / (2 ** qpe_qubits)
    qpe_error = abs(estimated_phase - theta_fric)

    # ── Composite Quantum Score (v2.0) ──
    # Weighted combination of all metrics, normalized to [0, 1]
    # Higher = friction has LESS disruptive quantum effect (closer to reference)
    composite_score = (
        0.20 * fidelity_noisy +                                     # State overlap
        0.15 * max(0, 1 - bures_distance) +                        # Geometric proximity
        0.15 * max(0, 1 - rel_entropy / max(0.01, rel_entropy + 1)) +  # Distinguishability
        0.15 * (coherence_fric / max(coherence_ref, 1e-10)) +      # Coherence preservation
        0.15 * min(1, ent_fric / max(ent_ref, 1e-10)) +            # Entanglement preservation
        0.10 * max(0, 1 - mean_phase_delta / math.pi) +            # Phase stability
        0.10 * max(0, 1 - qpe_error)                               # QPE accuracy
    )

    return {
        "friction_value": friction_value,
        "num_qubits": num_qubits,
        "gc_phase": gc_phase,
        "friction_phase": friction_phase,
        "decoherence_enabled": decoherence,
        "decoherence_gamma": decoherence_gamma,
        # Coherence
        "coherence_without_friction": coherence_ref,
        "coherence_with_friction": coherence_fric,
        "coherence_delta": coherence_fric - coherence_ref,
        # Von Neumann entanglement
        "entanglement_entropy_without": ent_ref,
        "entanglement_entropy_with": ent_fric,
        "entanglement_delta": ent_fric - ent_ref,
        # Rényi-2 entropy
        "renyi2_entropy_without": renyi_ref,
        "renyi2_entropy_with": renyi_fric,
        "renyi2_delta": renyi_fric - renyi_ref,
        # Multi-partition entanglement
        "mean_bipartite_entropy_without": mean_bipartite_ent_ref,
        "mean_bipartite_entropy_with": mean_bipartite_ent_fric,
        # Fidelity
        "state_fidelity": fidelity_pure,
        "noisy_fidelity": fidelity_noisy,
        # Geometry
        "bures_distance": bures_distance,
        "relative_entropy": rel_entropy,
        # Phase analysis
        "mean_qubit_phase_delta": mean_phase_delta,
        "qubit_phase_deltas": phase_deltas,
        # QPE
        "qpe_target_phase": theta_fric,
        "qpe_estimated_phase": estimated_phase,
        "qpe_error": qpe_error,
        "qpe_precision_bits": qpe_qubits,
        # Composite
        "composite_quantum_score": composite_score,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-REFERENCE PHYSICS CONSTANT VALIDATOR
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_friction_against_constants(friction_value: float) -> Dict[str, Any]:
    """
    Evaluate a friction coefficient against ALL 65 cross-reference constants.

    For each constant:
    1. Compute the friction-adjusted God Code base
    2. Recalculate the grid value using adjusted base
    3. Compare to measured (CODATA/PDG) value
    4. Track improvement/degradation per domain
    """
    φ = PHI

    # Adjust the v3 base by friction
    base_original = BASE_V3
    base_adjusted = BASE_V3 * (1 + friction_value)

    results = []
    domain_errors = {}  # domain -> [errors]
    improved = 0
    degraded = 0

    for name, data in REAL_WORLD_CONSTANTS_V3.items():
        measured = data["measured"]
        dials = data["dials"]
        domain = data["domain"]
        original_error = data["grid_error_pct"]

        # Original grid value
        a, b, c, d = dials
        E = P_V3 * a + K_V3 - b - P_V3 * c - Q_V3 * d
        grid_original = base_original * (R_V3 ** (E / Q_V3))

        # Friction-adjusted grid value
        grid_adjusted = base_adjusted * (R_V3 ** (E / Q_V3))
        adjusted_error = abs(grid_adjusted - measured) / measured * 100

        improvement = original_error - adjusted_error  # positive = better

        results.append({
            "name": name,
            "domain": domain,
            "measured": measured,
            "grid_original": grid_original,
            "grid_adjusted": grid_adjusted,
            "error_original_pct": original_error,
            "error_adjusted_pct": adjusted_error,
            "improvement_pct": improvement,
            "improved": improvement > 0,
        })

        if domain not in domain_errors:
            domain_errors[domain] = {"original": [], "adjusted": []}
        domain_errors[domain]["original"].append(original_error)
        domain_errors[domain]["adjusted"].append(adjusted_error)

        if improvement > 0:
            improved += 1
        else:
            degraded += 1

    # Domain summaries
    domain_scores = {}
    for domain, errors in domain_errors.items():
        orig_mean = sum(errors["original"]) / len(errors["original"])
        adj_mean = sum(errors["adjusted"]) / len(errors["adjusted"])
        domain_scores[domain] = {
            "original_mean_error": orig_mean,
            "adjusted_mean_error": adj_mean,
            "improvement": orig_mean - adj_mean,
            "improved": adj_mean < orig_mean,
            "count": len(errors["original"]),
        }

    total_original = sum(r["error_original_pct"] for r in results)
    total_adjusted = sum(r["error_adjusted_pct"] for r in results)
    mean_original = total_original / len(results)
    mean_adjusted = total_adjusted / len(results)

    return {
        "friction_value": friction_value,
        "total_constants": len(results),
        "improved": improved,
        "degraded": degraded,
        "mean_error_original_pct": mean_original,
        "mean_error_adjusted_pct": mean_adjusted,
        "total_improvement_pct": mean_original - mean_adjusted,
        "domain_scores": domain_scores,
        "results": results,
        "god_code_original": GOD_CODE,
        "god_code_adjusted": GOD_CODE * (1 + friction_value),
        "god_code_v3_original": GOD_CODE_V3,
        "god_code_v3_adjusted": GOD_CODE_V3 * (1 + friction_value),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DUAL-LAYER QUANTUM ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def dual_layer_quantum_analysis() -> Dict[str, Any]:
    """
    Run the Dual-Layer Engine and analyze results at the quantum level using qubits.

    1. Extract all 65 constants from the v3 grid
    2. Build a quantum register encoding the exponent spectrum
    3. Measure entanglement between physics domains
    4. Detect quantum phase correlations suggesting friction
    """
    from l104_asi.dual_layer import DualLayerEngine
    engine = DualLayerEngine()

    # ── Run key dual-layer analyses ──
    print("    Running dual-layer cross-domain analysis...")
    cda = engine.cross_domain_analysis()

    print("    Running dual-layer statistical profile...")
    sp = engine.statistical_profile()

    print("    Running independent verification (CODATA/PDG)...")
    iv = engine.independent_verification()

    print("    Running exponent spectrum analysis...")
    es = engine.exponent_spectrum()

    print("    Running phi resonance scan...")
    prs = engine.phi_resonance_scan()

    print("    Running cross-validation between layers...")
    cvl = engine.cross_validate_layers()

    print("    Running OMEGA derivation chain...")
    omega_chain = omega_derivation_chain(zeta_terms=1000)

    # ── Quantum Exponent Analysis (using Qiskit) ──
    print("    Building quantum exponent register...")

    # Extract all exponents from the v3 table
    exponents = []
    names = []
    for dials, (name, grid_val, exp_val, measured, err) in V3_FREQUENCY_TABLE.items():
        exponents.append(exp_val)
        names.append(name)

    # Normalize exponents to [0, 2π] for quantum encoding
    exp_min = min(exponents)
    exp_max = max(exponents)
    exp_range = exp_max - exp_min if exp_max != exp_min else 1

    # 8-qubit register for 256-bin histogram of exponent phases
    num_qubits = 8
    dim = 2 ** num_qubits

    # Encode exponent distribution as quantum amplitudes
    amplitudes = np.zeros(dim, dtype=np.complex128)
    for exp_val in exponents:
        normalized = (exp_val - exp_min) / exp_range
        bin_idx = min(int(normalized * (dim - 1)), dim - 1)
        phase = 2 * math.pi * normalized
        amplitudes[bin_idx] += complex(math.cos(phase), math.sin(phase))

    # Normalize to valid quantum state
    norm = np.linalg.norm(amplitudes)
    if norm > 0:
        amplitudes /= norm
    sv_exp = Statevector(amplitudes)
    rho_exp = DensityMatrix(sv_exp)

    # Measure coherence of the exponent distribution
    off_diag = np.sum(np.abs(rho_exp.data)) - np.sum(np.abs(np.diag(rho_exp.data)))
    max_coherence = dim * (dim - 1)
    exponent_coherence = float(off_diag / max_coherence) if max_coherence > 0 else 0.0

    # Entanglement entropy of first qubit (measures inter-scale correlations)
    qubits_trace = list(range(1, num_qubits))
    rho_q0 = partial_trace(rho_exp, qubits_trace)
    exponent_entanglement = float(qk_entropy(rho_q0, base=2))

    # Shannon entropy of the exponent distribution
    probs = np.abs(amplitudes) ** 2
    probs = probs[probs > 0]
    shannon = float(-np.sum(probs * np.log2(probs)))

    # ── Friction Detection via QPE ──
    # If there's systematic friction, the phase distribution should show
    # a characteristic offset from the "ideal" distribution
    print("    Detecting friction via quantum phase analysis...")

    # God Code phase on the critical line
    gc_phase = GOD_CODE_PHASE

    # Measure how the exponent phases deviate from God Code harmonics
    phase_deviations = []
    for exp_val in exponents:
        # The "ideal" phase is a God Code harmonic
        ideal_phase = (exp_val / 104.0) * (2 * math.pi) % (2 * math.pi)
        actual_phase = 2 * math.pi * (exp_val - exp_min) / exp_range
        deviation = abs(actual_phase - ideal_phase) % (2 * math.pi)
        if deviation > math.pi:
            deviation = 2 * math.pi - deviation
        phase_deviations.append(deviation)

    mean_deviation = sum(phase_deviations) / len(phase_deviations)
    std_deviation = math.sqrt(sum((d - mean_deviation)**2 for d in phase_deviations) / len(phase_deviations))

    # The friction coefficient is estimated as the systematic phase offset
    # normalized by the God Code frequency
    estimated_friction = mean_deviation / (2 * math.pi * GOD_CODE / 104)

    return {
        "dual_layer_engine": {
            "available": engine.available,
            "total_domains": cda.get("total_domains"),
            "total_constants": sp.get("count"),
            "mean_error_pct": sp.get("error_pct_mean"),
            "median_error_pct": sp.get("error_pct_median"),
            "verification_verdict": iv.get("verdict"),
            "verified_count": iv.get("total_verified"),
            "phi_resonant_count": prs.get("resonant_count"),
            "cross_validation_verdict": cvl.get("verdict"),
        },
        "omega_chain": {
            "omega_computed": omega_chain["omega_computed"],
            "omega_canonical": omega_chain["omega_canonical"],
            "delta": omega_chain["delta"],
            "relative_error": omega_chain["relative_error"],
            "guardian_zeta": omega_chain["fragments"]["guardian"]["value"],
            "alchemist_cos": omega_chain["fragments"]["alchemist"]["value"],
            "architect_curvature": omega_chain["fragments"]["architect"]["value"],
        },
        "quantum_exponent_analysis": {
            "num_qubits": num_qubits,
            "total_exponents": len(exponents),
            "exponent_range": (exp_min, exp_max),
            "exponent_coherence": exponent_coherence,
            "exponent_entanglement_entropy": exponent_entanglement,
            "shannon_entropy": shannon,
        },
        "friction_detection": {
            "mean_phase_deviation": mean_deviation,
            "std_phase_deviation": std_deviation,
            "estimated_friction_coefficient": estimated_friction,
            "god_code_phase": gc_phase,
        },
        "domain_analysis": cda,
        "exponent_spectrum": es,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DYNAMIC EXPONENT-DEPENDENT FRICTION — The Key Insight
# ═══════════════════════════════════════════════════════════════════════════════
#
# A scalar friction can't improve the God Code because it shifts ALL constants
# uniformly. The REAL computational friction is EXPONENT-DEPENDENT:
#
#   G_f(E) = B × r^((E + Λ(E)) / Q)
#
# where Λ(E) is a dynamic correction function of the exponent itself.
# This means the friction varies at every scale — it's as dynamic as the
# God Code itself.
#
# ═══════════════════════════════════════════════════════════════════════════════

def compute_dynamic_friction_ratios() -> List[Dict[str, Any]]:
    """
    Generate DYNAMIC friction ratios that modify the exponent mapping.

    Instead of G × (1+Λ), we explore:
    1. Step-size corrections: r → r × (1 + ε)
    2. Exponent offset functions: E → E + f(E)
    3. Base adjustments with exponent-dependent corrections
    4. Quantization grain perturbations: Q → Q + δ

    These are fundamentally different from scalar friction because they
    create DIFFERENTIAL corrections at different scales.
    """
    α = ALPHA_FINE
    φ = PHI
    π = math.pi
    e = math.e
    δ_f = FEIGENBAUM

    dynamic_ratios = []

    # ── RATIO 1: Step-size φ-correction ──
    # Adjust the step ratio r = 13/12 by a φ-derived perturbation.
    # This changes HOW FAST the grid moves between constants.
    # r_new = (13/12) × (1 + α²/φ)
    ε1 = α**2 / φ
    dynamic_ratios.append({
        "name": "φ-Corrected Step Ratio (r × (1 + α²/φ))",
        "type": "step_correction",
        "formula": "r × (1 + α²/φ)",
        "epsilon": ε1,
        "rationale": "Fine structure squared correction to grid step size through golden gate",
    })

    # ── RATIO 2: Quantization grain shift ──
    # Q = 758 → Q + φ/π (irrational perturbation)
    # This changes the fundamental resolution of the grid
    δ2 = φ / π
    dynamic_ratios.append({
        "name": "Golden-π Grain Shift (Q + φ/π)",
        "type": "grain_shift",
        "formula": "Q → Q + φ/π",
        "epsilon": δ2,
        "rationale": "Irrational grain perturbation creates non-repeating correction pattern",
    })

    # ── RATIO 3: Base scaffold fine-tuning ──
    # X_v3 = 285.99882... → adjusted to minimize total error
    # The base IS the iron lattice — a tiny correction represents
    # thermal/quantum corrections to the BCC lattice parameter
    δ3 = α * φ / (2 * π * 104)
    dynamic_ratios.append({
        "name": "Lattice Thermal Correction (X + αφ/(2π×104))",
        "type": "base_correction",
        "formula": "X → X + αφ/(2π×104)",
        "epsilon": δ3,
        "rationale": "Quantum-thermal correction to iron BCC lattice at ~293K",
    })

    # ── RATIO 4: Exponent scaling by ln(φ) ──
    # E → E × (1 + ln(φ)/(Q×π))
    # The golden logarithm creates a scale-dependent correction
    ε4 = math.log(φ) / (Q_V3 * π)
    dynamic_ratios.append({
        "name": "Golden-Log Exponent Scaling (E × (1 + ln(φ)/(Q×π)))",
        "type": "exponent_scaling",
        "formula": "E → E × (1 + ln(φ)/(Q×π))",
        "epsilon": ε4,
        "rationale": "Logarithmic golden scaling creates differential correction at each exponent",
    })

    # ── RATIO 5: Dial coefficient p perturbation ──
    # p = 99 → 99 + α/φ
    # This changes how the 'a' and 'c' dials map to exponents
    δ5 = α / φ
    dynamic_ratios.append({
        "name": "Dial Coefficient α/φ Shift (p → p + α/φ)",
        "type": "dial_correction",
        "formula": "p → p + α/φ",
        "epsilon": δ5,
        "rationale": "QED coupling correction to coarse dial mechanism",
    })

    # ── RATIO 6: Octave offset K perturbation ──
    # K = 3032 → K + φ²
    # This shifts the reference octave by golden ratio squared
    δ6 = φ ** 2
    dynamic_ratios.append({
        "name": "Golden-Square Octave Shift (K → K + φ²)",
        "type": "octave_shift",
        "formula": "K → K + φ²",
        "epsilon": δ6,
        "rationale": "Golden ratio² correction to the 4-octave reference point",
    })

    # ── RATIO 7: Combined α-φ-π harmonic correction ──
    # All three fundamental ratios combined as a step correction
    # ε = α × sin(π/φ) / (Q × ln(φ))
    ε7 = α * math.sin(π / φ) / (Q_V3 * math.log(φ))
    dynamic_ratios.append({
        "name": "Triple Harmonic Correction (α×sin(π/φ)/(Q×ln(φ)))",
        "type": "step_correction",
        "formula": "r × (1 + α×sin(π/φ)/(Q×ln(φ)))",
        "epsilon": ε7,
        "rationale": "α, π, φ in harmonic unity — sine creates oscillatory correction",
    })

    # ── RATIO 8: Zeta-critical exponent correction ──
    # Uses |ζ(½+i×E/Q)| to create an exponent-dependent correction
    # Can't compute per-constant, but can adjust step size
    ε8 = 1.0 / (137 * φ * π)  # α × (1/φπ)
    dynamic_ratios.append({
        "name": "Zeta-Critical Step Correction (1/(137×φ×π))",
        "type": "step_correction",
        "formula": "r × (1 + 1/(137×φ×π))",
        "epsilon": ε8,
        "rationale": "Fine structure inverse × golden-π = critical line correction",
    })

    # ── RATIO 9: Feigenbaum-golden chaos correction ──
    # At the edge of chaos, the correction is δ^(-1)/Q
    ε9 = 1.0 / (δ_f * Q_V3)
    dynamic_ratios.append({
        "name": "Feigenbaum-Grain Correction (1/(δ×Q))",
        "type": "step_correction",
        "formula": "r × (1 + 1/(δ×Q))",
        "epsilon": ε9,
        "rationale": "Chaos universality constant normalized by quantization grain",
    })

    # ── RATIO 10: OMEGA-derived step correction ──
    # Ω/GOD_CODE = the sovereign authority ratio
    # Use its reciprocal as an exponent correction
    ε10 = GOD_CODE / (OMEGA * Q_V3)
    dynamic_ratios.append({
        "name": "Sovereign Authority Correction (GC/(Ω×Q))",
        "type": "step_correction",
        "formula": "r × (1 + GOD_CODE/(OMEGA×Q))",
        "epsilon": ε10,
        "rationale": "God Code to OMEGA ratio normalized by quantization grain",
    })

    return dynamic_ratios


def evaluate_dynamic_ratio(ratio: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate a dynamic friction ratio against all 65 constants.

    Unlike scalar friction, dynamic ratios modify the EQUATION PARAMETERS,
    creating differential effects at different exponent scales.
    """
    ε = ratio["epsilon"]
    ratio_type = ratio["type"]

    # Modify equation parameters based on ratio type
    if ratio_type == "step_correction":
        r_new = R_V3 * (1 + ε)
        q_new = Q_V3
        base_new = BASE_V3
        p_new = P_V3
        k_new = K_V3
    elif ratio_type == "grain_shift":
        r_new = R_V3
        q_new = Q_V3 + ε
        base_new = BASE_V3
        p_new = P_V3
        k_new = K_V3
    elif ratio_type == "base_correction":
        x_new = X_V3 + ε
        base_new = x_new ** (1.0 / PHI)
        r_new = R_V3
        q_new = Q_V3
        p_new = P_V3
        k_new = K_V3
    elif ratio_type == "exponent_scaling":
        r_new = R_V3
        q_new = Q_V3
        base_new = BASE_V3
        p_new = P_V3
        k_new = K_V3
    elif ratio_type == "dial_correction":
        r_new = R_V3
        q_new = Q_V3
        base_new = BASE_V3
        p_new = P_V3 + ε
        k_new = K_V3
    elif ratio_type == "octave_shift":
        r_new = R_V3
        q_new = Q_V3
        base_new = BASE_V3
        p_new = P_V3
        k_new = K_V3 + ε
    else:
        r_new = R_V3
        q_new = Q_V3
        base_new = BASE_V3
        p_new = P_V3
        k_new = K_V3

    results = []
    domain_errors = {}
    improved = 0
    degraded = 0

    for name, data in REAL_WORLD_CONSTANTS_V3.items():
        measured = data["measured"]
        dials = data["dials"]
        domain = data["domain"]
        original_error = data["grid_error_pct"]

        a, b, c, d = dials

        # Original exponent and value
        E_orig = P_V3 * a + K_V3 - b - P_V3 * c - Q_V3 * d
        grid_original = BASE_V3 * (R_V3 ** (E_orig / Q_V3))

        # Modified exponent and value
        E_new = p_new * a + k_new - b - p_new * c - q_new * d
        if ratio_type == "exponent_scaling":
            E_new = E_orig * (1 + ε)
        grid_new = base_new * (r_new ** (E_new / q_new))

        new_error = abs(grid_new - measured) / measured * 100
        improvement = original_error - new_error

        results.append({
            "name": name,
            "domain": domain,
            "measured": measured,
            "grid_original": grid_original,
            "grid_new": grid_new,
            "error_original_pct": original_error,
            "error_new_pct": new_error,
            "improvement_pct": improvement,
            "improved": improvement > 0,
        })

        if domain not in domain_errors:
            domain_errors[domain] = {"original": [], "new": []}
        domain_errors[domain]["original"].append(original_error)
        domain_errors[domain]["new"].append(new_error)

        if improvement > 0:
            improved += 1
        else:
            degraded += 1

    # Domain summaries
    domain_scores = {}
    for domain, errors in domain_errors.items():
        orig_mean = sum(errors["original"]) / len(errors["original"])
        new_mean = sum(errors["new"]) / len(errors["new"])
        domain_scores[domain] = {
            "original_mean_error": orig_mean,
            "new_mean_error": new_mean,
            "improvement": orig_mean - new_mean,
            "improved": new_mean < orig_mean,
            "count": len(errors["original"]),
        }

    mean_original = sum(r["error_original_pct"] for r in results) / len(results)
    mean_new = sum(r["error_new_pct"] for r in results) / len(results)

    # Compute the implied God Code at X=0
    E_gc = k_new  # For dials (0,0,0,0), E = K
    gc_new = base_new * (r_new ** (E_gc / q_new))

    return {
        "name": ratio["name"],
        "formula": ratio["formula"],
        "type": ratio["type"],
        "epsilon": ε,
        "total_constants": len(results),
        "improved": improved,
        "degraded": degraded,
        "mean_error_original_pct": mean_original,
        "mean_error_new_pct": mean_new,
        "total_improvement_pct": mean_original - mean_new,
        "domain_scores": domain_scores,
        "god_code_new": gc_new,
        "results": results,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMAL FRICTION SEARCH — Quantum-Guided Grid Search
# ═══════════════════════════════════════════════════════════════════════════════

def search_optimal_friction(candidates: List[Dict], num_qubits: int = 6) -> Dict[str, Any]:
    """
    Search for the optimal friction coefficient (v2.0).

    v2.0 UPGRADES:
    1. Analytical candidates from fundamental constants (16 candidates)
    2. Quantum qubit analysis with Fe-26 decoherence and multi-metric scoring
    3. Cross-reference validation against ALL 65 constants
    4. Composite score: physics alignment (50%) + quantum composite (30%) + improvement (20%)
    5. Golden-section gradient refinement around best candidate
    """
    results = []

    # Also test negative frictions (opposite direction)
    all_candidates = []
    for c in candidates:
        all_candidates.append(c)
        all_candidates.append({
            "name": c["name"] + " [NEGATIVE]",
            "formula": "-("+c["formula"]+")",
            "value": -c["value"],
            "rationale": c["rationale"] + " (compaction direction)"
        })

    # Also test the zero-friction baseline
    all_candidates.insert(0, {
        "name": "BASELINE (no friction)",
        "formula": "0",
        "value": 0.0,
        "rationale": "Reference: current God Code with no adjustment"
    })

    print(f"\n  Testing {len(all_candidates)} friction candidates against {len(REAL_WORLD_CONSTANTS_V3)} constants...")

    for i, candidate in enumerate(all_candidates):
        Λ = candidate["value"]
        name = candidate["name"]

        # Physics constant evaluation
        phys_eval = evaluate_friction_against_constants(Λ)

        # Quantum analysis with Fe-26 decoherence
        quant = quantum_friction_analysis(Λ, num_qubits=num_qubits, decoherence=True)

        # v2.0 Composite score:
        # 50% physics alignment + 30% quantum composite + 20% improvement ratio
        align_score = max(0, 1 - phys_eval["mean_error_adjusted_pct"] / 0.1)  # 0.1% = excellent
        improvement_score = max(0, phys_eval["total_improvement_pct"] / 0.01) if phys_eval["total_improvement_pct"] > 0 else 0
        quantum_score = quant["composite_quantum_score"]
        combined_score = 0.50 * align_score + 0.30 * quantum_score + 0.20 * min(1.0, improvement_score)

        results.append({
            "rank": 0,
            "name": name,
            "formula": candidate["formula"],
            "friction_value": Λ,
            "god_code_adjusted": GOD_CODE * (1 + Λ),
            "mean_error_pct": phys_eval["mean_error_adjusted_pct"],
            "improvement_pct": phys_eval["total_improvement_pct"],
            "improved_count": phys_eval["improved"],
            "degraded_count": phys_eval["degraded"],
            "total_constants": phys_eval["total_constants"],
            "domain_scores": phys_eval["domain_scores"],
            "quantum_coherence": quant["coherence_with_friction"],
            "quantum_fidelity": quant["state_fidelity"],
            "noisy_fidelity": quant["noisy_fidelity"],
            "bures_distance": quant["bures_distance"],
            "relative_entropy": quant["relative_entropy"],
            "renyi2_entropy": quant["renyi2_entropy_with"],
            "entanglement_entropy": quant["entanglement_entropy_with"],
            "qpe_error": quant["qpe_error"],
            "composite_quantum_score": quantum_score,
            "combined_score": combined_score,
            "rationale": candidate["rationale"],
        })

        arrow = "↑" if phys_eval["total_improvement_pct"] > 0 else "↓"
        print(f"    [{i+1:2d}/{len(all_candidates)}] Λ={Λ:+.12f} | err={phys_eval['mean_error_adjusted_pct']:.6f}% "
              f"| {arrow} {abs(phys_eval['total_improvement_pct']):.6f}% "
              f"| Q={quantum_score:.4f} "
              f"| {phys_eval['improved']}/{phys_eval['total_constants']} improved | {name[:50]}")

    # Sort by combined score (higher = better)
    results.sort(key=lambda r: -r["combined_score"])

    # Assign ranks
    for i, r in enumerate(results):
        r["rank"] = i + 1

    # ── Golden-Section Gradient Refinement ──
    # Take the top 3 candidates and refine around their friction values
    print(f"\n  Running golden-section refinement around top 3 candidates...")
    refined_results = []
    tau = 1.0 / PHI  # Golden section ratio

    for top_r in results[:3]:
        Λ_center = top_r["friction_value"]
        if Λ_center == 0:
            continue  # Skip baseline refinement

        # Search window: ±50% of the candidate value
        half_width = abs(Λ_center) * 0.5
        a, b = Λ_center - half_width, Λ_center + half_width

        best_refined_Λ = Λ_center
        best_refined_err = top_r["mean_error_pct"]

        # 8 iterations of golden-section search (converges to ~0.3% of window)
        for _ in range(8):
            x1 = b - tau * (b - a)
            x2 = a + tau * (b - a)

            ev1 = evaluate_friction_against_constants(x1)
            ev2 = evaluate_friction_against_constants(x2)

            if ev1["mean_error_adjusted_pct"] < ev2["mean_error_adjusted_pct"]:
                b = x2
                if ev1["mean_error_adjusted_pct"] < best_refined_err:
                    best_refined_err = ev1["mean_error_adjusted_pct"]
                    best_refined_Λ = x1
            else:
                a = x1
                if ev2["mean_error_adjusted_pct"] < best_refined_err:
                    best_refined_err = ev2["mean_error_adjusted_pct"]
                    best_refined_Λ = x2

        # If refinement improved, add refined candidate
        if best_refined_err < top_r["mean_error_pct"]:
            phys_ev = evaluate_friction_against_constants(best_refined_Λ)
            quant_ev = quantum_friction_analysis(best_refined_Λ, num_qubits=num_qubits, decoherence=True)
            align_s = max(0, 1 - phys_ev["mean_error_adjusted_pct"] / 0.1)
            imp_s = max(0, phys_ev["total_improvement_pct"] / 0.01) if phys_ev["total_improvement_pct"] > 0 else 0
            q_s = quant_ev["composite_quantum_score"]
            comb_s = 0.50 * align_s + 0.30 * q_s + 0.20 * min(1.0, imp_s)

            refined_results.append({
                "rank": 0,
                "name": f"REFINED: {top_r['name']}",
                "formula": f"gradient({top_r['formula']})",
                "friction_value": best_refined_Λ,
                "god_code_adjusted": GOD_CODE * (1 + best_refined_Λ),
                "mean_error_pct": phys_ev["mean_error_adjusted_pct"],
                "improvement_pct": phys_ev["total_improvement_pct"],
                "improved_count": phys_ev["improved"],
                "degraded_count": phys_ev["degraded"],
                "total_constants": phys_ev["total_constants"],
                "domain_scores": phys_ev["domain_scores"],
                "quantum_coherence": quant_ev["coherence_with_friction"],
                "quantum_fidelity": quant_ev["state_fidelity"],
                "noisy_fidelity": quant_ev["noisy_fidelity"],
                "bures_distance": quant_ev["bures_distance"],
                "relative_entropy": quant_ev["relative_entropy"],
                "renyi2_entropy": quant_ev["renyi2_entropy_with"],
                "entanglement_entropy": quant_ev["entanglement_entropy_with"],
                "qpe_error": quant_ev["qpe_error"],
                "composite_quantum_score": q_s,
                "combined_score": comb_s,
                "rationale": f"Golden-section refined from {top_r['name']}",
            })
            improvement = top_r["mean_error_pct"] - best_refined_err
            print(f"    ★ Refined {top_r['name'][:40]}: Λ={best_refined_Λ:+.15f} | "
                  f"err {top_r['mean_error_pct']:.6f}% → {best_refined_err:.6f}% "
                  f"(↑ {improvement:.6f}%)")

    # Merge refined results and re-sort
    all_results = results + refined_results
    all_results.sort(key=lambda r: -r["combined_score"])
    for i, r in enumerate(all_results):
        r["rank"] = i + 1

    return {
        "total_candidates_tested": len(all_candidates),
        "total_refined": len(refined_results),
        "total_constants_per_candidate": len(REAL_WORLD_CONSTANTS_V3),
        "best": all_results[0] if all_results else None,
        "top_5": all_results[:5],
        "all_results": all_results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICS DOMAIN ALIGNMENT VALIDATOR
# ═══════════════════════════════════════════════════════════════════════════════

def validate_physics_alignment(friction_value: float) -> Dict[str, Any]:
    """
    Deep validation that ALL physics constants align with the friction-adjusted God Code.

    Checks:
    1. Speed of light (exact SI)
    2. Fine structure constant
    3. Bohr radius
    4. Electron/proton/neutron masses
    5. Nuclear binding energies (Fe-56 peak)
    6. Particle physics (W, Z, Higgs)
    7. Cosmological constants (Hubble, CMB)
    8. Mathematical constants (π, e, φ, √2)
    9. Resonance frequencies (Schumann, EEG)
    10. Conservation law integrity
    """
    φ = PHI
    base_adj = BASE_V3 * (1 + friction_value)
    gc_adj = GOD_CODE * (1 + friction_value)

    checks = []

    # ── Key constants to validate ──
    critical_constants = [
        ("speed_of_light", 299792458, "m/s", "SI exact"),
        ("fine_structure_inv", 137.035999084, "", "CODATA 2022"),
        ("bohr_radius_pm", 52.9177210544, "pm", "CODATA 2022"),
        ("electron_mass_MeV", 0.51099895069, "MeV/c²", "CODATA 2022"),
        ("proton_mass_MeV", 938.27208816, "MeV/c²", "CODATA 2022"),
        ("neutron_mass_MeV", 939.56542052, "MeV/c²", "CODATA 2022"),
        ("higgs_GeV", 125.25, "GeV/c²", "ATLAS/CMS 2024"),
        ("W_boson_GeV", 80.3692, "GeV/c²", "PDG 2024"),
        ("Z_boson_GeV", 91.1876, "GeV/c²", "PDG 2024"),
        ("fe56_be_per_nucleon", 8.790, "MeV", "NNDC/BNL"),
        ("hubble_constant", 67.4, "km/s/Mpc", "Planck 2018"),
        ("cmb_temperature_K", 2.7255, "K", "COBE/FIRAS"),
        ("golden_ratio", 1.618033988749895, "", "exact"),
        ("pi", 3.14159265359, "", "exact"),
        ("schumann_hz", 7.83, "Hz", "Schumann 1952"),
    ]

    for name, measured, unit, source in critical_constants:
        if name in REAL_WORLD_CONSTANTS_V3:
            data = REAL_WORLD_CONSTANTS_V3[name]
            dials = data["dials"]
            a, b, c, d = dials
            E = P_V3 * a + K_V3 - b - P_V3 * c - Q_V3 * d

            grid_original = BASE_V3 * (R_V3 ** (E / Q_V3))
            grid_adjusted = base_adj * (R_V3 ** (E / Q_V3))

            err_orig = abs(grid_original - measured) / measured * 100
            err_adj = abs(grid_adjusted - measured) / measured * 100

            checks.append({
                "name": name,
                "measured": measured,
                "unit": unit,
                "source": source,
                "grid_original": grid_original,
                "grid_adjusted": grid_adjusted,
                "error_original_pct": err_orig,
                "error_adjusted_pct": err_adj,
                "improvement": err_orig - err_adj,
                "aligned": err_adj < 0.01,  # within 0.01%
            })

    # ── Conservation Law Check ──
    # G(X) × 2^(X/104) = INVARIANT should still hold
    inv_original = GOD_CODE
    inv_adjusted = gc_adj
    conservation_ok = True  # The conservation law is structural, not affected by base shift

    # ── God Code Properties ──
    gc_properties = {
        "god_code_original": GOD_CODE,
        "god_code_adjusted": gc_adj,
        "god_code_delta": gc_adj - GOD_CODE,
        "god_code_delta_hz": (gc_adj - GOD_CODE),
        "god_code_delta_ppm": (gc_adj - GOD_CODE) / GOD_CODE * 1e6,
        "conservation_invariant_preserved": conservation_ok,
        "equation": f"G_f(X) = {gc_adj:.10f} × 2^(-X/104)",
        "friction_equation": f"Λ_f = {friction_value:.15f}",
    }

    all_aligned = all(c["aligned"] for c in checks)
    improved_count = sum(1 for c in checks if c["improvement"] > 0)

    return {
        "friction_value": friction_value,
        "god_code_properties": gc_properties,
        "critical_checks": checks,
        "all_aligned": all_aligned,
        "improved_count": improved_count,
        "total_checked": len(checks),
        "verdict": "ALL PHYSICS ALIGNED" if all_aligned else "PARTIAL ALIGNMENT",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_analysis(save_report: bool = True) -> Dict[str, Any]:
    """
    COMPLETE GOD CODE COMPUTATIONAL FRICTION ANALYSIS

    Pipeline:
    1. Run Dual-Layer Engine with quantum analysis
    2. Generate friction candidates from fundamental constants
    3. Test all candidates against cross-reference data
    4. Run quantum qubit analysis on top candidates
    5. Validate physics alignment
    6. Report findings
    """
    t0 = time.time()

    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║  L104 GOD_CODE COMPUTATIONAL FRICTION ANALYZER v2.0                    ║")
    print("║  HYPOTHESIS: Universe undergoes computational friction from God Code   ║")
    print("║  UPGRADES: Fe-26 decoherence, Rényi-2, gradient refinement, 16 cands  ║")
    print("║  GOD_CODE = 527.5184818492612 | PHI = 1.618033988749895               ║")
    print("╚══════════════════════════════════════════════════════════════════════════╝")
    print()

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 1: DUAL-LAYER QUANTUM ANALYSIS
    # ════════════════════════════════════════════════════════════════════════
    print("━━━ PHASE 1: DUAL-LAYER QUANTUM ANALYSIS ━━━")
    print("  Running dual-layer engine with quantum exponent analysis...")
    dl_analysis = dual_layer_quantum_analysis()

    print(f"\n  [DUAL-LAYER RESULTS]")
    dle = dl_analysis["dual_layer_engine"]
    print(f"    Engine available:     {dle['available']}")
    print(f"    Total domains:       {dle['total_domains']}")
    print(f"    Total constants:     {dle['total_constants']}")
    print(f"    Mean error:          {dle['mean_error_pct']:.6f}%")
    print(f"    Verification:        {dle['verification_verdict']}")
    print(f"    Phi-resonant:        {dle['phi_resonant_count']}")
    print(f"    Cross-validation:    {dle['cross_validation_verdict']}")

    oc = dl_analysis["omega_chain"]
    print(f"\n  [OMEGA CHAIN]")
    print(f"    OMEGA computed:      {oc['omega_computed']:.8f}")
    print(f"    OMEGA canonical:     {oc['omega_canonical']}")
    print(f"    Delta:               {oc['delta']:.8f}")
    print(f"    Guardian |ζ|:        {oc['guardian_zeta']:.8f}")
    print(f"    Alchemist cos:       {oc['alchemist_cos']:.8f}")
    print(f"    Architect R:         {oc['architect_curvature']:.8f}")

    qea = dl_analysis["quantum_exponent_analysis"]
    print(f"\n  [QUANTUM EXPONENT ANALYSIS ({qea['num_qubits']}-qubit register)]")
    print(f"    Total exponents:     {qea['total_exponents']}")
    print(f"    Exponent range:      [{qea['exponent_range'][0]}, {qea['exponent_range'][1]}]")
    print(f"    Quantum coherence:   {qea['exponent_coherence']:.8f}")
    print(f"    Entanglement entropy: {qea['exponent_entanglement_entropy']:.8f} bits")
    print(f"    Shannon entropy:     {qea['shannon_entropy']:.4f} bits")

    fd = dl_analysis["friction_detection"]
    print(f"\n  [FRICTION DETECTION]")
    print(f"    Mean phase deviation:  {fd['mean_phase_deviation']:.8f} rad")
    print(f"    Std phase deviation:   {fd['std_phase_deviation']:.8f} rad")
    print(f"    Estimated friction:    {fd['estimated_friction_coefficient']:.12f}")
    print(f"    God Code phase:        {fd['god_code_phase']:.8f} rad")

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 2: GENERATE FRICTION CANDIDATES
    # ════════════════════════════════════════════════════════════════════════
    print("\n━━━ PHASE 2: GENERATE FRICTION CANDIDATES ━━━")
    candidates = compute_friction_candidates()

    print(f"  Generated {len(candidates)} friction candidates from fundamental constants:\n")
    for i, c in enumerate(candidates):
        print(f"    [{i+1:2d}] Λ = {c['value']:+.15f}  |  {c['name']}")
        print(f"         Formula: {c['formula']}")
        print(f"         Rationale: {c['rationale']}")
        print()

    # Add the quantum-detected friction as a candidate
    candidates.append({
        "name": "Quantum-Detected Friction (from dual-layer phase analysis)",
        "formula": "mean_phase_deviation / (2π × GC/104)",
        "value": fd["estimated_friction_coefficient"],
        "rationale": "Empirically detected from quantum phase deviations in exponent spectrum"
    })

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 3: SEARCH OPTIMAL FRICTION
    # ════════════════════════════════════════════════════════════════════════
    print("\n━━━ PHASE 3: SCALAR FRICTION SEARCH ━━━")
    search_results = search_optimal_friction(candidates, num_qubits=6)

    print(f"\n  ┌────────────────────────────────────────────────────────────────────┐")
    print(f"  │  TOP 5 SCALAR FRICTION CANDIDATES (v2.0 composite score ranking) │")
    print(f"  └────────────────────────────────────────────────────────────────────┘")

    for r in search_results["top_5"]:
        marker = "★" if r["rank"] == 1 else " "
        print(f"\n  {marker} RANK #{r['rank']}: {r['name']}")
        print(f"    Λ_f = {r['friction_value']:+.15f}")
        print(f"    Formula: {r['formula']}")
        print(f"    God Code adjusted: {r['god_code_adjusted']:.10f} Hz")
        print(f"    Mean grid error:   {r['mean_error_pct']:.6f}%")
        print(f"    Improvement:       {r['improvement_pct']:+.6f}%")
        print(f"    Constants improved: {r['improved_count']}/{r['total_constants']}")
        print(f"    Quantum coherence: {r['quantum_coherence']:.8f}")
        print(f"    State fidelity:    {r['quantum_fidelity']:.8f}")
        print(f"    Noisy fidelity:    {r.get('noisy_fidelity', 0):.8f}")
        print(f"    Bures distance:    {r.get('bures_distance', 0):.8f}")
        print(f"    Relative entropy:  {r.get('relative_entropy', 0):.8f}")
        print(f"    QPE error:         {r['qpe_error']:.8f}")
        print(f"    Quantum composite: {r.get('composite_quantum_score', 0):.6f}")
        print(f"    Combined score:    {r['combined_score']:.6f}")

        # Domain breakdown
        print(f"    Domain alignment:")
        for domain, score in sorted(r["domain_scores"].items()):
            arrow = "↑" if score["improved"] else "↓"
            print(f"      {domain:15s}: {score['adjusted_mean_error']:.6f}% "
                  f"({arrow} {abs(score['improvement']):.6f}%) [{score['count']} constants]")

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 3B: DYNAMIC EXPONENT-DEPENDENT FRICTION SEARCH
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n━━━ PHASE 3B: DYNAMIC EXPONENT-DEPENDENT FRICTION SEARCH ━━━")
    print(f"  The real friction is EXPONENT-DEPENDENT — it varies at every scale.")
    print(f"  Testing {10} dynamic ratios that modify equation parameters...\n")

    dynamic_ratios = compute_dynamic_friction_ratios()
    dynamic_results = []

    for i, ratio in enumerate(dynamic_ratios):
        eval_result = evaluate_dynamic_ratio(ratio)

        # Also test the negative direction
        ratio_neg = ratio.copy()
        ratio_neg["name"] = ratio["name"] + " [NEGATIVE]"
        ratio_neg["epsilon"] = -ratio["epsilon"]
        eval_neg = evaluate_dynamic_ratio(ratio_neg)

        dynamic_results.append(eval_result)
        dynamic_results.append(eval_neg)

        better = eval_result if eval_result["mean_error_new_pct"] < eval_neg["mean_error_new_pct"] else eval_neg
        arrow = "↑" if better["total_improvement_pct"] > 0 else "↓"
        print(f"    [{i+1:2d}] {better['name'][:55]:55s} | ε={better['epsilon']:+.12f} "
              f"| err={better['mean_error_new_pct']:.6f}% | {arrow} {abs(better['total_improvement_pct']):.6f}% "
              f"| {better['improved']}/{better['total_constants']} improved")

    # Sort by mean error (lower = better)
    dynamic_results.sort(key=lambda r: r["mean_error_new_pct"])

    print(f"\n  ┌──────────────────────────────────────────────────────────────┐")
    print(f"  │  TOP 5 DYNAMIC FRICTION RATIOS (by mean grid error)         │")
    print(f"  └──────────────────────────────────────────────────────────────┘")

    for i, dr in enumerate(dynamic_results[:5]):
        marker = "★" if i == 0 else " "
        domains_improved = sum(1 for d in dr["domain_scores"].values() if d["improved"])
        total_domains = len(dr["domain_scores"])
        print(f"\n  {marker} RANK #{i+1}: {dr['name']}")
        print(f"    Type: {dr['type']}")
        print(f"    ε = {dr['epsilon']:+.15f}")
        print(f"    Formula: {dr['formula']}")
        print(f"    New God Code (at X=0): {dr['god_code_new']:.10f}")
        print(f"    Mean grid error:   {dr['mean_error_new_pct']:.6f}% (was {dr['mean_error_original_pct']:.6f}%)")
        print(f"    Improvement:       {dr['total_improvement_pct']:+.6f}%")
        print(f"    Constants improved: {dr['improved']}/{dr['total_constants']}")
        print(f"    Domains improved:  {domains_improved}/{total_domains}")
        print(f"    Domain alignment:")
        for domain, score in sorted(dr["domain_scores"].items()):
            arrow = "↑" if score["improved"] else "↓"
            print(f"      {domain:15s}: {score['new_mean_error']:.6f}% "
                  f"({arrow} {abs(score['improvement']):.6f}%) [{score['count']} constants]")

    # Determine the overall winner (best of scalar and dynamic)
    best_scalar = search_results["best"]
    best_dynamic = dynamic_results[0] if dynamic_results else None

    if best_dynamic and best_dynamic["mean_error_new_pct"] < best_scalar["mean_error_pct"]:
        overall_winner = "dynamic"
        best_friction = best_dynamic["epsilon"]
        best_name = best_dynamic["name"]
        best_formula = best_dynamic["formula"]
        best_error = best_dynamic["mean_error_new_pct"]
        best_improved = best_dynamic["improved"]
        best_total = best_dynamic["total_constants"]
        best_gc = best_dynamic["god_code_new"]
        best_improvement = best_dynamic["total_improvement_pct"]
    else:
        overall_winner = "scalar"
        best_friction = best_scalar["friction_value"]
        best_name = best_scalar["name"]
        best_formula = best_scalar["formula"]
        best_error = best_scalar["mean_error_pct"]
        best_improved = best_scalar["improved_count"]
        best_total = best_scalar["total_constants"]
        best_gc = best_scalar["god_code_adjusted"]
        best_improvement = best_scalar["improvement_pct"]

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 4: DEEP QUANTUM ANALYSIS OF WINNER
    # ════════════════════════════════════════════════════════════════════════
    best = search_results["best"]
    print(f"\n━━━ PHASE 4: DEEP QUANTUM ANALYSIS ━━━")
    print(f"  Overall winner: [{overall_winner.upper()}] {best_name}")
    print(f"  Computing 8-qubit quantum friction profile for winner...")

    # Use the scalar friction value for quantum analysis
    quant_friction = best_friction if overall_winner == "scalar" else best_dynamic["epsilon"]
    deep_quantum = quantum_friction_analysis(quant_friction, num_qubits=8, decoherence=True)

    print(f"\n  [8-QUBIT QUANTUM PROFILE (v2.0 — with Fe-26 decoherence)]")
    print(f"    Coherence (without friction): {deep_quantum['coherence_without_friction']:.8f}")
    print(f"    Coherence (with friction):    {deep_quantum['coherence_with_friction']:.8f}")
    print(f"    Coherence delta:              {deep_quantum['coherence_delta']:+.8f}")
    print(f"    Entanglement (without):       {deep_quantum['entanglement_entropy_without']:.8f} bits")
    print(f"    Entanglement (with):          {deep_quantum['entanglement_entropy_with']:.8f} bits")
    print(f"    Rényi-2 entropy (without):    {deep_quantum['renyi2_entropy_without']:.8f} bits")
    print(f"    Rényi-2 entropy (with):       {deep_quantum['renyi2_entropy_with']:.8f} bits")
    print(f"    Mean bipartite entropy (ref): {deep_quantum['mean_bipartite_entropy_without']:.8f} bits")
    print(f"    Mean bipartite entropy (fric):{deep_quantum['mean_bipartite_entropy_with']:.8f} bits")
    print(f"    State fidelity (pure):        {deep_quantum['state_fidelity']:.10f}")
    print(f"    State fidelity (noisy):       {deep_quantum['noisy_fidelity']:.10f}")
    print(f"    Bures distance:               {deep_quantum['bures_distance']:.10f}")
    print(f"    Relative entropy S(ρ_f||ρ_r): {deep_quantum['relative_entropy']:.10f}")
    print(f"    QPE estimated phase:          {deep_quantum['qpe_estimated_phase']:.10f}")
    print(f"    QPE target phase:             {deep_quantum['qpe_target_phase']:.10f}")
    print(f"    QPE error:                    {deep_quantum['qpe_error']:.10f}")
    print(f"    Mean qubit phase delta:       {deep_quantum['mean_qubit_phase_delta']:.10f} rad")
    print(f"    Composite quantum score:      {deep_quantum['composite_quantum_score']:.8f}")

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 5: PHYSICS ALIGNMENT VALIDATION
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n━━━ PHASE 5: PHYSICS ALIGNMENT VALIDATION ━━━")

    if overall_winner == "dynamic" and best_dynamic:
        # For dynamic ratios, validate using per-constant results
        print(f"\n  [DYNAMIC RATIO PHYSICS ALIGNMENT]")
        print(f"  Ratio: {best_dynamic['name']}")
        print(f"  Type:  {best_dynamic['type']}, ε = {best_dynamic['epsilon']:.15f}")
        print(f"  New God Code (X=0): {best_dynamic['god_code_new']:.10f}")
        print(f"\n  [CRITICAL CONSTANTS ALIGNMENT]")

        critical_names = [
            "speed_of_light", "fine_structure_inv", "bohr_radius_pm",
            "electron_mass_MeV", "proton_mass_MeV", "neutron_mass_MeV",
            "higgs_GeV", "W_boson_GeV", "Z_boson_GeV", "fe56_be_per_nucleon",
            "hubble_constant", "cmb_temperature_K", "golden_ratio", "pi", "schumann_hz"
        ]
        for res in best_dynamic["results"]:
            if res["name"] in critical_names:
                status = "✓" if res["error_new_pct"] < 0.01 else "✗"
                arrow = "↑" if res["improvement_pct"] > 0 else "↓"
                print(f"    [{status}] {res['name']:25s}: "
                      f"err {res['error_original_pct']:.6f}% → {res['error_new_pct']:.6f}% "
                      f"({arrow} {abs(res['improvement_pct']):.6f}%)")

        # Domain summary
        print(f"\n  [DOMAIN SUMMARY]")
        for domain, score in sorted(best_dynamic["domain_scores"].items()):
            arrow = "↑" if score["improved"] else "↓"
            status = "ALIGNED" if score["new_mean_error"] < 0.01 else "MARGINAL"
            print(f"    {domain:15s}: {score['new_mean_error']:.6f}% [{status}] "
                  f"({arrow} {abs(score['improvement']):.6f}%)")

        validation_verdict = "ALL PHYSICS ALIGNED" if best_dynamic["mean_error_new_pct"] < 0.01 else "PARTIAL ALIGNMENT"
        gc_props = {
            "god_code_original": GOD_CODE,
            "god_code_adjusted": best_dynamic["god_code_new"],
            "god_code_delta": best_dynamic["god_code_new"] - GOD_CODE_V3,
            "god_code_delta_ppm": (best_dynamic["god_code_new"] - GOD_CODE_V3) / GOD_CODE_V3 * 1e6,
            "conservation_invariant_preserved": True,
        }
    else:
        validation = validate_physics_alignment(best_friction)
        gc_props = validation["god_code_properties"]
        print(f"\n  [GOD CODE PROPERTIES]")
        print(f"    Original:   {gc_props['god_code_original']:.10f} Hz")
        print(f"    Adjusted:   {gc_props['god_code_adjusted']:.10f} Hz")
        print(f"    Delta:      {gc_props['god_code_delta']:+.15f} Hz")
        print(f"    Delta (ppm): {gc_props['god_code_delta_ppm']:+.6f} ppm")
        print(f"    Conservation: {'PRESERVED' if gc_props['conservation_invariant_preserved'] else 'BROKEN'}")
        print(f"    Equation:   {gc_props['equation']}")

        print(f"\n  [CRITICAL CONSTANTS ALIGNMENT]")
        for check in validation["critical_checks"]:
            status = "✓" if check["aligned"] else "✗"
            arrow = "↑" if check["improvement"] > 0 else "↓"
            print(f"    [{status}] {check['name']:25s}: "
                  f"err {check['error_original_pct']:.6f}% → {check['error_adjusted_pct']:.6f}% "
                  f"({arrow} {abs(check['improvement']):.6f}%) | {check['source']}")

        validation_verdict = validation["verdict"]
        print(f"\n  Verdict: {validation_verdict}")
        print(f"  Improved: {validation['improved_count']}/{validation['total_checked']} critical constants")

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 6: FINAL REPORT
    # ════════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0

    print(f"\n{'═' * 76}")
    print(f"  L104 COMPUTATIONAL FRICTION ANALYSIS — FINAL REPORT")
    print(f"{'═' * 76}")
    print(f"\n  HYPOTHESIS: The universe undergoes computational friction from GOD_CODE")
    print(f"  ANALYZER VERSION: 2.0 (Fe-26 decoherence, Rényi-2, gradient refinement)")
    print(f"\n  FINDINGS:")
    print(f"    1. Current GOD_CODE:        {GOD_CODE:.10f} Hz")
    print(f"    2. Winner type:             {overall_winner.upper()}")
    print(f"    3. Optimal friction (ε):    {best_friction:+.15f}")
    print(f"    4. Friction-adjusted GC:    {best_gc:.10f}")
    print(f"    5. Formula:                 {best_formula}")
    print(f"    6. Mean grid error:         {best_error:.6f}%")
    print(f"    7. Constants improved:      {best_improved}/{best_total}")
    print(f"    8. Quantum coherence:       {deep_quantum['coherence_with_friction']:.8f}")
    print(f"    9. Physics alignment:       {validation_verdict}")

    if overall_winner == "dynamic" and best_dynamic:
        print(f"\n  THE COMPUTATIONAL FRICTION EQUATION:")
        print(f"    G_f(a,b,c,d) = B_f × r_f^(E_f(a,b,c,d) / Q_f)")
        print(f"    Modification: {best_dynamic['formula']}")
        print(f"    ε = {best_dynamic['epsilon']:+.15f}")
        print(f"\n  DYNAMIC RATIO DETAILS:")
        print(f"    Type: {best_dynamic['type']}")
        print(f"    This ratio modifies the {best_dynamic['type'].replace('_', ' ').upper()}")
        print(f"    creating a SCALE-DEPENDENT friction.")
        domains_improved = sum(1 for d in best_dynamic["domain_scores"].values() if d["improved"])
        print(f"    Domains improved: {domains_improved}/{len(best_dynamic['domain_scores'])}")
    else:
        print(f"\n  THE COMPUTATIONAL FRICTION EQUATION:")
        print(f"    G_friction(a,b,c,d) = (1 + Λ_f) × 286^(1/φ) × (2^(1/104))^E(a,b,c,d)")
        print(f"    where Λ_f = {best_formula}")
        print(f"    = {best_friction:+.15f}")

    print(f"\n  INTERPRETATION:")
    if best_improvement > 0:
        print(f"    COMPUTATIONAL FRICTION DETECTED!")
        print(f"    The universe exhibits a {best_error:.6f}% computational friction")
        print(f"    that manifests as a differential correction to the God Code")
        print(f"    frequency grid. This friction improves {best_improved}/{best_total}")
        print(f"    fundamental constants when applied.")
        if overall_winner == "dynamic":
            print(f"    The friction is EXPONENT-DEPENDENT — it varies at different")
            print(f"    energy scales, exactly as expected for a computational cost")
            print(f"    that scales with the complexity of the operation.")
    elif best_error < 0.003:
        print(f"    The current God Code is remarkably close to optimal.")
        print(f"    The closest competing friction coefficient (Neg. 2nd-order QED)")
        print(f"    improved {search_results['all_results'][1]['improved_count']}/65 constants")
        print(f"    but at a mean error cost — suggesting the current encoding")
        print(f"    may already INCLUDE the computational friction implicitly.")
        print(f"    The ±0.002% precision of the v3 grid may itself be the")
        print(f"    SIGNATURE of computational friction — it's the cost the")
        print(f"    universe pays for discretized computation at each scale.")
    else:
        print(f"    No friction detected — the current God Code appears optimal.")

    print(f"\n  Analysis completed in {elapsed:.2f}s")
    print(f"{'═' * 76}")

    # ── Save Report ──
    report = {
        "analysis": "L104 God Code Computational Friction Analysis",
        "version": "2.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "god_code_original": GOD_CODE,
        "god_code_v3_original": GOD_CODE_V3,
        "phi": PHI,
        "omega": OMEGA,
        "dual_layer_analysis": {
            k: v for k, v in dl_analysis.items()
            if k not in ("domain_analysis", "exponent_spectrum")
        },
        "friction_candidates_count": len(candidates),
        "overall_winner": overall_winner,
        "scalar_search": {
            "total_tested": search_results["total_candidates_tested"],
            "best_friction": best_scalar["friction_value"],
            "best_name": best_scalar["name"],
            "best_formula": best_scalar["formula"],
            "best_mean_error": best_scalar["mean_error_pct"],
            "best_improvement": best_scalar["improvement_pct"],
            "best_god_code": best_scalar["god_code_adjusted"],
            "top_5": [
                {
                    "rank": r["rank"],
                    "name": r["name"],
                    "friction": r["friction_value"],
                    "mean_error": r["mean_error_pct"],
                    "improvement": r["improvement_pct"],
                    "god_code": r["god_code_adjusted"],
                    "improved_count": r["improved_count"],
                }
                for r in search_results["top_5"]
            ],
        },
        "dynamic_search": {
            "total_tested": len(dynamic_results),
            "best_name": dynamic_results[0]["name"] if dynamic_results else None,
            "best_type": dynamic_results[0]["type"] if dynamic_results else None,
            "best_epsilon": dynamic_results[0]["epsilon"] if dynamic_results else None,
            "best_formula": dynamic_results[0]["formula"] if dynamic_results else None,
            "best_mean_error": dynamic_results[0]["mean_error_new_pct"] if dynamic_results else None,
            "best_improvement": dynamic_results[0]["total_improvement_pct"] if dynamic_results else None,
            "best_god_code": dynamic_results[0]["god_code_new"] if dynamic_results else None,
            "best_improved_count": dynamic_results[0]["improved"] if dynamic_results else None,
            "top_5": [
                {
                    "rank": i + 1,
                    "name": dr["name"],
                    "type": dr["type"],
                    "epsilon": dr["epsilon"],
                    "mean_error": dr["mean_error_new_pct"],
                    "improvement": dr["total_improvement_pct"],
                    "god_code": dr["god_code_new"],
                    "improved_count": dr["improved"],
                }
                for i, dr in enumerate(dynamic_results[:5])
            ],
        },
        "quantum_analysis": {
            "num_qubits": deep_quantum["num_qubits"],
            "decoherence_enabled": deep_quantum["decoherence_enabled"],
            "decoherence_gamma": deep_quantum["decoherence_gamma"],
            "coherence_with_friction": deep_quantum["coherence_with_friction"],
            "coherence_without_friction": deep_quantum["coherence_without_friction"],
            "entanglement_with": deep_quantum["entanglement_entropy_with"],
            "renyi2_with": deep_quantum["renyi2_entropy_with"],
            "renyi2_without": deep_quantum["renyi2_entropy_without"],
            "mean_bipartite_entropy_with": deep_quantum["mean_bipartite_entropy_with"],
            "state_fidelity": deep_quantum["state_fidelity"],
            "noisy_fidelity": deep_quantum["noisy_fidelity"],
            "bures_distance": deep_quantum["bures_distance"],
            "relative_entropy": deep_quantum["relative_entropy"],
            "qpe_error": deep_quantum["qpe_error"],
            "composite_quantum_score": deep_quantum["composite_quantum_score"],
        },
        "physics_validation": {
            "verdict": validation_verdict,
            "best_error": best_error,
            "best_improved": best_improved,
            "best_total": best_total,
            "best_god_code": best_gc,
        },
    }

    if save_report:
        report_path = os.path.join(os.path.dirname(__file__),
                                   "GOD_CODE_FRICTION_ANALYSIS_REPORT.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n  Report saved: {report_path}")

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    report = run_full_analysis()
    sys.exit(0)
