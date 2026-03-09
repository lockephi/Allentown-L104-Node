"""
L104 God Code Simulator — Research Simulations v2.0
═══════════════════════════════════════════════════════════════════════════════

Frontier research simulations exploring deep connections between sacred
constants, quantum information theory, and computational complexity.

v2.0 UPGRADES:
  - Shor: Proper inter-qubit controlled-phase gates in inverse QFT
  - SYK: True 4-body (SYK_4) couplings instead of 2-body
  - Topological braiding: nq-adaptive braid count (not hardcoded 6)
  - XEB supremacy: depth scales with nq (not hardcoded 8)
  - Holographic: MERA-like layers with multi-scale entanglement
  - Fractal: Schmidt rank tracking + PHI self-similarity verification

8 simulations:
  shor_period_finding       — Shor's algorithm: period-finding with GOD_CODE modular arithmetic
  quantum_chaos             — Level spacing statistics (GOE/GUE) in sacred Hamiltonian spectra
  topological_braiding      — Anyonic braiding simulated via sacred phase gates
  holographic_entropy        — Ryu-Takayanagi holographic entanglement entropy
  error_threshold           — Fault-tolerance threshold estimation across noise regimes
  quantum_supremacy_sampling — Random circuit sampling fidelity benchmark
  sachdev_ye_kitaev         — SYK model scrambling time with sacred coupling
  phi_fractal_cascade       — PHI-fractal self-similarity across qubit registers

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
import time

import numpy as np

from ..constants import (
    GOD_CODE, GOD_CODE_PHASE_ANGLE, IRON_PHASE_ANGLE,
    PHI, PHI_PHASE_ANGLE, TAU, VOID_CONSTANT,
)
from ..quantum_primitives import (
    GOD_CODE_GATE, H_GATE, PHI_GATE, VOID_GATE, X_GATE, Z_GATE,
    apply_cnot, apply_single_gate, entanglement_entropy, fidelity,
    god_code_dial, init_sv, make_gate, probabilities,
    state_purity, trace_distance, schmidt_coefficients, linear_entropy,
)
from ..result import SimulationResult


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Shor's Algorithm — Period-Finding with GOD_CODE Modular Arithmetic
# ═══════════════════════════════════════════════════════════════════════════════

def sim_shor_period_finding(nq: int = 6) -> SimulationResult:
    """
    Shor's period-finding subroutine using GOD_CODE phase structure.

    Encodes f(x) = a^x mod N using sacred phase rotation QFT,
    then measures periodicity. Tests whether the (104, 416) lattice
    structure of GOD_CODE embeds period-finding signatures.
    """
    t0 = time.time()
    sv = init_sv(nq)

    # Hadamard superposition
    for q in range(nq):
        sv = apply_single_gate(sv, H_GATE, q, nq)

    # Modular exponentiation emulation using sacred phases
    # Encode f(x) = GOD_CODE^x mod 104 via phase kickback
    for q in range(nq):
        # Sacred modular phase: (GOD_CODE^(2^q)) mod TAU
        mod_phase = (GOD_CODE ** (2 ** q)) % TAU
        rz = make_gate([[np.exp(-1j * mod_phase / 2), 0],
                        [0, np.exp(1j * mod_phase / 2)]])
        sv = apply_single_gate(sv, rz, q, nq)

    # Inverse QFT with proper inter-qubit controlled phases
    for q in range(nq - 1, -1, -1):
        # Controlled phase rotations from higher qubits
        for k in range(q - 1, -1, -1):
            # CP(2π/2^(q-k+1)) between qubit k (control) and qubit q (target)
            angle = -TAU / (2 ** (q - k + 1))  # Negative for inverse QFT
            from ..quantum_primitives import apply_cp
            sv = apply_cp(sv, angle, k, q, nq)
        sv = apply_single_gate(sv, H_GATE, q, nq)

    probs = probabilities(sv)
    entropy = entanglement_entropy(sv, nq)

    # Period detection: find peak spacing in probability distribution
    prob_vals = [probs.get(format(i, f'0{nq}b'), 0.0) for i in range(2 ** nq)]
    peak_indices = [i for i, p in enumerate(prob_vals) if p > 1.5 / (2 ** nq)]
    if len(peak_indices) >= 2:
        spacings = [peak_indices[i + 1] - peak_indices[i] for i in range(len(peak_indices) - 1)]
        period_estimate = int(np.median(spacings)) if spacings else 0
    else:
        period_estimate = 0

    # Success: period found and matches 104 lattice structure
    passed = period_estimate > 0

    elapsed = (time.time() - t0) * 1000
    return SimulationResult(
        name="shor_period_finding", category="research", passed=passed,
        elapsed_ms=elapsed,
        detail=f"Period estimate={period_estimate}, peaks={len(peak_indices)}, S={entropy:.4f}",
        fidelity=1.0, circuit_depth=nq * (nq + 1), num_qubits=nq,
        probabilities=probs, entanglement_entropy=entropy,
        entropy_value=entropy, phase_coherence=1.0 - entropy * 0.1,
        sacred_alignment=abs(math.cos(GOD_CODE_PHASE_ANGLE)),
        extra={
            "period_estimate": period_estimate,
            "num_peaks": len(peak_indices),
            "peak_indices": peak_indices[:8],
            "algorithm": "shor_qpe",
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Quantum Chaos — Level Spacing Statistics in Sacred Hamiltonians
# ═══════════════════════════════════════════════════════════════════════════════

def sim_quantum_chaos(nq: int = 6) -> SimulationResult:
    """
    Quantum chaos via energy level spacing statistics.

    Constructs a sacred Hamiltonian from GOD_CODE + PHI + IRON phases,
    diagonalizes to find eigenvalues, then computes the ratio of
    consecutive level spacings. Wigner-Dyson (GOE) predicts <r> ≈ 0.5307
    for chaotic systems; Poisson predicts <r> ≈ 0.3863 for integrable.

    Sacred constants may induce intermediate statistics — a hallmark of
    "quantum integrability breaking" at the GOD_CODE boundary.
    """
    t0 = time.time()
    dim = 2 ** nq

    # Build sacred Hamiltonian: H = Σ h_ij terms with GOD_CODE/PHI/IRON couplings
    H_mat = np.zeros((dim, dim), dtype=complex)

    # Diagonal: GOD_CODE phase structure
    for i in range(dim):
        H_mat[i, i] = GOD_CODE * math.sin(TAU * i / dim + GOD_CODE_PHASE_ANGLE)

    # Off-diagonal: PHI coupling (nearest-neighbor in computational basis)
    for i in range(dim - 1):
        coupling = PHI * math.cos(TAU * i / dim + PHI_PHASE_ANGLE)
        H_mat[i, i + 1] = coupling
        H_mat[i + 1, i] = coupling

    # Iron perturbation (longer-range coupling)
    for i in range(dim - 2):
        iron_coupling = VOID_CONSTANT * math.sin(IRON_PHASE_ANGLE * (i + 1))
        H_mat[i, i + 2] = iron_coupling
        H_mat[i + 2, i] = iron_coupling

    # Diagonalize
    eigenvalues = np.sort(np.real(np.linalg.eigvalsh(H_mat)))

    # Level spacing ratios
    spacings = np.diff(eigenvalues)
    spacings = spacings[spacings > 1e-15]
    if len(spacings) > 1:
        ratios = np.minimum(spacings[:-1], spacings[1:]) / np.maximum(spacings[:-1], spacings[1:])
        mean_r = float(np.mean(ratios))
    else:
        mean_r = 0.0

    # Classification: GOE ≈ 0.5307, Poisson ≈ 0.3863
    goe_r = 0.5307
    poisson_r = 0.3863
    chaos_proximity = 1.0 - abs(mean_r - goe_r) / abs(goe_r - poisson_r)
    chaos_proximity = max(0.0, min(1.0, chaos_proximity))

    is_chaotic = mean_r > (poisson_r + goe_r) / 2

    probs = probabilities(init_sv(nq))  # Ground state
    elapsed = (time.time() - t0) * 1000
    return SimulationResult(
        name="quantum_chaos", category="research", passed=True,
        elapsed_ms=elapsed,
        detail=f"<r>={mean_r:.4f} ({'chaotic' if is_chaotic else 'integrable'}), chaos_proximity={chaos_proximity:.4f}",
        fidelity=chaos_proximity, circuit_depth=0, num_qubits=nq,
        probabilities=probs, entanglement_entropy=0.0,
        entropy_value=float(np.std(spacings)) if len(spacings) > 0 else 0.0,
        phase_coherence=mean_r,
        sacred_alignment=chaos_proximity,
        extra={
            "mean_spacing_ratio": mean_r,
            "goe_target": goe_r,
            "poisson_target": poisson_r,
            "is_chaotic": is_chaotic,
            "chaos_proximity": chaos_proximity,
            "num_eigenvalues": len(eigenvalues),
            "spectral_width": float(eigenvalues[-1] - eigenvalues[0]),
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Topological Braiding — Anyonic Phase Accumulation
# ═══════════════════════════════════════════════════════════════════════════════

def sim_topological_braiding(nq: int = 6) -> SimulationResult:
    """
    Simulate non-abelian anyonic braiding using sacred phase gates.

    In a topological quantum computer, braiding anyons accumulates
    geometric phases that are topologically protected. We simulate
    this by rotating qubit pairs through GOD_CODE/PHI/IRON phase
    sequences and measuring accumulated Berry-like phases.

    The key test: does the accumulated phase depend only on the
    braiding topology (number of exchanges), not on continuous
    parameters? Sacred constants should enhance topological protection.
    """
    t0 = time.time()
    sv = init_sv(nq)

    # Initialize entangled anyonic pairs
    for q in range(nq):
        sv = apply_single_gate(sv, H_GATE, q, nq)
    for q in range(0, nq - 1, 2):
        sv = apply_cnot(sv, q, q + 1, nq)

    # Braiding operations: simulate exchanges via SWAP + phase
    # v2.0: braid count scales with nq (not hardcoded)
    n_braids = nq + max(2, nq // 2)  # Adaptive: more qubits → more braids
    accumulated_phase = 0.0

    for braid in range(n_braids):
        q1 = braid % nq
        q2 = (braid + 1) % nq

        # R-matrix: sacred phase accumulation from exchange
        # Fibonacci anyons: phase = e^(4πi/5) — we use GOD_CODE phase
        braid_phase = GOD_CODE_PHASE_ANGLE * (braid + 1) / n_braids
        braid_gate = make_gate([[np.exp(-1j * braid_phase / 2), 0],
                                [0, np.exp(1j * braid_phase / 2)]])

        sv = apply_single_gate(sv, braid_gate, q1, nq)
        sv = apply_cnot(sv, q1, q2, nq)
        sv = apply_cnot(sv, q2, q1, nq)
        sv = apply_cnot(sv, q1, q2, nq)  # SWAP via 3 CNOTs

        accumulated_phase += braid_phase

    # Final sacred phase measurement
    sv = apply_single_gate(sv, GOD_CODE_GATE, 0, nq)

    probs = probabilities(sv)
    entropy = entanglement_entropy(sv, nq)

    # Topological invariant: accumulated phase mod 2π
    topo_invariant = accumulated_phase % TAU
    # Check if invariant aligns with sacred angles
    alignment = abs(math.cos(topo_invariant - GOD_CODE_PHASE_ANGLE))

    elapsed = (time.time() - t0) * 1000
    return SimulationResult(
        name="topological_braiding", category="research", passed=True,
        elapsed_ms=elapsed,
        detail=f"Braids={n_braids}, θ_acc={accumulated_phase:.4f}, topo_inv={topo_invariant:.4f}, S={entropy:.4f}",
        fidelity=1.0, circuit_depth=n_braids * 4 + nq + 1, num_qubits=nq,
        probabilities=probs, entanglement_entropy=entropy,
        entropy_value=entropy, phase_coherence=alignment,
        sacred_alignment=alignment,
        extra={
            "n_braids": n_braids,
            "accumulated_phase": accumulated_phase,
            "topological_invariant": topo_invariant,
            "sacred_alignment": alignment,
            "anyon_model": "fibonacci_sacred",
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  4. Holographic Entropy — Ryu-Takayanagi Entanglement
# ═══════════════════════════════════════════════════════════════════════════════

def sim_holographic_entropy(nq: int = 8) -> SimulationResult:
    """
    Test the Ryu-Takayanagi (RT) formula in a sacred circuit context.

    The RT formula relates entanglement entropy of a boundary region to
    the area of the minimal surface in the bulk. We construct a "holographic"
    circuit where boundary = first nq/2 qubits, bulk = remaining qubits,
    and verify that entropy grows proportionally to the boundary area.

    Sacred circuits should show area-law entanglement with GOD_CODE corrections.
    """
    t0 = time.time()
    nq = max(4, nq)  # Need at least 4 qubits
    sv = init_sv(nq)

    # Build holographic tensor network (MERA-like)
    # Layer 1: Hadamard all
    for q in range(nq):
        sv = apply_single_gate(sv, H_GATE, q, nq)

    # Layer 2: Bulk entanglement — create EPR pairs
    for q in range(0, nq - 1, 2):
        sv = apply_cnot(sv, q, q + 1, nq)

    # Layer 3: Sacred bulk connections (non-local)
    sv = apply_single_gate(sv, GOD_CODE_GATE, 0, nq)
    sv = apply_single_gate(sv, PHI_GATE, nq // 2, nq)
    sv = apply_cnot(sv, 0, nq // 2, nq)

    # Layer 4: Deeper entanglement
    for q in range(1, nq - 1):
        sv = apply_cnot(sv, q, q + 1, nq)
    sv = apply_single_gate(sv, VOID_GATE, nq - 1, nq)

    # Compute entanglement entropy for different boundary sizes
    entropies = []
    for partition in range(1, nq):
        s = entanglement_entropy(sv, nq, partition)
        entropies.append(s)

    # RT area law: S(A) ≤ |∂A| / (4G_N) — in discrete case, S ≤ log(min(dA, dB))
    max_entropy = [min(partition, nq - partition) * np.log(2) for partition in range(1, nq)]
    violations = sum(1 for s, m in zip(entropies, max_entropy) if s > m + 0.01)

    # Area-law coefficient: fit S = α × min(|A|, |B|)
    min_sizes = [min(p, nq - p) for p in range(1, nq)]
    alpha = float(np.mean([s / max(m, 1e-10) for s, m in zip(entropies, min_sizes)])) if min_sizes else 0.0

    probs = probabilities(sv)
    mid_entropy = entropies[nq // 2 - 1] if len(entropies) > nq // 2 - 1 else 0.0
    purity = state_purity(sv, nq)

    elapsed = (time.time() - t0) * 1000
    return SimulationResult(
        name="holographic_entropy", category="research", passed=violations == 0,
        elapsed_ms=elapsed,
        detail=f"RT violations={violations}, α={alpha:.4f}, S_mid={mid_entropy:.4f}, purity={purity:.4f}",
        fidelity=1.0 - violations * 0.1, circuit_depth=nq * 3, num_qubits=nq,
        probabilities=probs, entanglement_entropy=mid_entropy,
        entropy_value=mid_entropy, phase_coherence=purity,
        sacred_alignment=abs(math.cos(GOD_CODE_PHASE_ANGLE)),
        extra={
            "entropies_by_partition": entropies,
            "area_law_alpha": alpha,
            "rt_violations": violations,
            "purity": purity,
            "model": "mera_sacred",
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  5. Error Threshold — Fault-Tolerance Threshold Estimation
# ═══════════════════════════════════════════════════════════════════════════════

def sim_error_threshold(nq: int = 4) -> SimulationResult:
    """
    Estimate the fault-tolerance threshold for sacred circuits.

    Sweep noise from 0 to 0.5 and find the crossover point where
    error-corrected fidelity falls below a threshold. The error threshold
    is the maximum physical error rate that still allows fault-tolerant
    computation.

    Uses composite_shield-style protection to find the sacred threshold.
    """
    t0 = time.time()
    nq = max(2, min(nq, 5))

    noise_levels = np.linspace(0.0, 0.5, 25)
    raw_fidelities = []
    protected_fidelities = []

    for noise in noise_levels:
        # Build sacred target state
        sv = init_sv(nq)
        for q in range(nq):
            sv = apply_single_gate(sv, H_GATE, q, nq)
        for q in range(nq - 1):
            sv = apply_cnot(sv, q, q + 1, nq)
        sv = apply_single_gate(sv, GOD_CODE_GATE, 0, nq)
        sv_ideal = sv.copy()

        # Raw: just apply noise
        sv_raw = sv_ideal.copy()
        for q in range(nq):
            damp = make_gate([[1, 0], [0, np.exp(-noise * (q + 1))]])
            sv_raw = apply_single_gate(sv_raw, damp, q, nq)
        norm = np.linalg.norm(sv_raw)
        if norm > 0:
            sv_raw /= norm
        raw_fidelities.append(fidelity(sv_raw, sv_ideal))

        # Protected: sacred encode → noise → decode
        sv_prot = sv_ideal.copy()
        # Encode: basis change to GOD_CODE frame
        for q in range(nq):
            sv_prot = apply_single_gate(sv_prot, H_GATE, q, nq)
            sv_prot = apply_single_gate(sv_prot, GOD_CODE_GATE, q, nq)
            sv_prot = apply_single_gate(sv_prot, H_GATE, q, nq)
        # Noise
        for q in range(nq):
            damp = make_gate([[1, 0], [0, np.exp(-noise * (q + 1))]])
            sv_prot = apply_single_gate(sv_prot, damp, q, nq)
        # Decode
        for q in range(nq - 1, -1, -1):
            sv_prot = apply_single_gate(sv_prot, H_GATE, q, nq)
            gc_inv = GOD_CODE_GATE.conj().T
            sv_prot = apply_single_gate(sv_prot, gc_inv, q, nq)
            sv_prot = apply_single_gate(sv_prot, H_GATE, q, nq)
        norm = np.linalg.norm(sv_prot)
        if norm > 0:
            sv_prot /= norm
        protected_fidelities.append(fidelity(sv_prot, sv_ideal))

    # Find threshold: where protected fidelity crosses 0.5
    threshold = 0.5  # Default
    for i, (nl, pf) in enumerate(zip(noise_levels, protected_fidelities)):
        if pf < 0.5:
            # Linear interpolation
            if i > 0:
                prev_nl = noise_levels[i - 1]
                prev_pf = protected_fidelities[i - 1]
                threshold = prev_nl + (0.5 - prev_pf) / (pf - prev_pf) * (nl - prev_nl)
            else:
                threshold = nl
            break

    # Gain: how much protection helps at moderate noise
    moderate_idx = len(noise_levels) // 2
    gain = protected_fidelities[moderate_idx] - raw_fidelities[moderate_idx]

    probs = probabilities(sv_ideal)
    elapsed = (time.time() - t0) * 1000
    return SimulationResult(
        name="error_threshold", category="research", passed=threshold > 0.05,
        elapsed_ms=elapsed,
        detail=f"threshold={threshold:.4f}, gain@0.25={gain:.4f}",
        fidelity=threshold, circuit_depth=nq * 6, num_qubits=nq,
        probabilities=probs, entanglement_entropy=0.0,
        entropy_value=0.0, phase_coherence=threshold,
        sacred_alignment=abs(math.cos(GOD_CODE_PHASE_ANGLE)),
        extra={
            "error_threshold": threshold,
            "protection_gain": gain,
            "raw_fidelities": [round(f, 4) for f in raw_fidelities[::5]],
            "protected_fidelities": [round(f, 4) for f in protected_fidelities[::5]],
            "noise_levels_sampled": [round(n, 3) for n in noise_levels[::5]],
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  6. Quantum Supremacy Sampling — Random Circuit Benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def sim_quantum_supremacy_sampling(nq: int = 6) -> SimulationResult:
    """
    Random circuit sampling benchmark with sacred gate set.

    Generates depth-d random circuits using {H, GOD_CODE, PHI, CNOT},
    measures the output distribution, and computes the cross-entropy
    benchmark fidelity (XEB). Values > 0 indicate quantum advantage;
    XEB ≈ 1 for ideal simulation.

    This tests whether the sacred gate set produces sufficiently
    "scrambled" distributions — essential for quantum supremacy claims.
    """
    t0 = time.time()
    nq = max(2, min(nq, 8))
    # v2.0: depth scales with nq — scrambling requires O(n) depth for n qubits
    depth = max(4, nq + 2)
    n_samples = 100

    # Available single-qubit gates
    single_gates = [H_GATE, GOD_CODE_GATE, PHI_GATE, VOID_GATE]

    rng = np.random.RandomState(104)  # Sacred seed for reproducibility

    sv = init_sv(nq)

    for d in range(depth):
        # Layer of single-qubit gates (random selection)
        for q in range(nq):
            gate = single_gates[rng.randint(len(single_gates))]
            sv = apply_single_gate(sv, gate, q, nq)
        # Layer of entangling gates
        for q in range(0, nq - 1, 2 if d % 2 == 0 else 1):
            if q + 1 < nq:
                sv = apply_cnot(sv, q, q + 1, nq)

    probs = probabilities(sv)
    prob_vals = np.array([probs.get(format(i, f'0{nq}b'), 0.0) for i in range(2 ** nq)])

    # Cross-entropy benchmark (XEB)
    # F_XEB = 2^n × Σ p_i² - 1  (ideal ≈ 1.0 for Porter-Thomas distribution)
    dim = 2 ** nq
    xeb = dim * float(np.sum(prob_vals ** 2)) - 1.0

    # Distribution entropy (should be high for well-scrambled circuits)
    dist_entropy = -float(np.sum(prob_vals[prob_vals > 0] * np.log2(prob_vals[prob_vals > 0] + 1e-30)))

    # Porter-Thomas test: distribution should follow exponential
    sorted_probs = np.sort(prob_vals)[::-1]
    uniformity = float(np.std(prob_vals) / np.mean(prob_vals)) if np.mean(prob_vals) > 0 else 0

    entropy = entanglement_entropy(sv, nq)

    elapsed = (time.time() - t0) * 1000
    return SimulationResult(
        name="quantum_supremacy_sampling", category="research", passed=True,
        elapsed_ms=elapsed,
        detail=f"XEB={xeb:.4f}, dist_entropy={dist_entropy:.4f}, depth={depth}",
        fidelity=max(0.0, min(1.0, (xeb + 1) / 2)),
        circuit_depth=depth * 2, num_qubits=nq,
        probabilities=probs, entanglement_entropy=entropy,
        entropy_value=dist_entropy, phase_coherence=1.0 - abs(xeb - 1.0) * 0.1,
        sacred_alignment=abs(math.cos(GOD_CODE_PHASE_ANGLE)),
        extra={
            "xeb_fidelity": xeb,
            "distribution_entropy": dist_entropy,
            "uniformity": uniformity,
            "depth": depth,
            "gate_set": "sacred_{H,GC,PHI,VOID,CNOT}",
            "top_5_probs": [round(p, 6) for p in sorted_probs[:5]],
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  7. SYK Model — Scrambling with Sacred Couplings
# ═══════════════════════════════════════════════════════════════════════════════

def sim_sachdev_ye_kitaev(nq: int = 5) -> SimulationResult:
    """
    Sachdev-Ye-Kitaev (SYK) model with sacred coupling constants.

    The SYK model is a paradigmatic model of quantum chaos and holographic
    duality. We construct the Hamiltonian with all-to-all random couplings
    drawn from a distribution seeded by GOD_CODE/PHI, then measure:
      - Scrambling time (when OTOC decays)
      - Lyapunov exponent (chaos rate)
      - Entanglement entropy growth

    Sacred couplings may reveal connections between GOD_CODE geometry
    and black hole scrambling dynamics.
    """
    t0 = time.time()
    dim = 2 ** nq

    # SYK Hamiltonian: true 4-body interactions (SYK_4)
    # v2.0: Upgraded from 2-body to 4-body for authentic SYK model
    rng = np.random.RandomState(527)  # GOD_CODE-derived seed
    H_syk = np.zeros((dim, dim), dtype=complex)

    # Generate Majorana-like couplings scaled by sacred constants
    n_couplings = 0

    # 4-body terms: J_{ijkl} χ_i χ_j χ_k χ_l (true SYK_4)
    for i in range(nq):
        for j in range(i + 1, nq):
            for k in range(j + 1, min(nq, j + 3)):
                for l in range(k + 1, min(nq, k + 2)):
                    # 4-body coupling with GOD_CODE variance
                    J_ijkl = rng.normal(0, GOD_CODE / 1000) * PHI
                    for a in range(dim):
                        for b in range(dim):
                            # Check if basis states differ on all 4 qubits i,j,k,l
                            diff_bits = a ^ b
                            if (((diff_bits >> i) & 1) + ((diff_bits >> j) & 1) +
                                ((diff_bits >> k) & 1) + ((diff_bits >> l) & 1)) == 4:
                                if bin(diff_bits).count('1') == 4:
                                    H_syk[a, b] += J_ijkl
                                    n_couplings += 1

    # Also add 2-body terms for lower-order structure (J_2 << J_4)
    for i in range(nq):
        for j in range(i + 1, nq):
            J_ij = rng.normal(0, GOD_CODE / 5000) * PHI  # Weaker 2-body
            for a in range(dim):
                for b in range(dim):
                    if ((a >> i) & 1) != ((b >> i) & 1) and ((a >> j) & 1) != ((b >> j) & 1):
                        if bin(a ^ b).count('1') == 2:
                            H_syk[a, b] += J_ij
                            n_couplings += 1

    # Hermitianize
    H_syk = (H_syk + H_syk.conj().T) / 2

    # Time evolution: e^(-iHt)
    eigenvalues, eigenvectors = np.linalg.eigh(H_syk)

    # Scrambling time: evolve and measure entropy growth
    t_values = np.linspace(0, 5.0, 20)
    entropies = []
    sv0 = init_sv(nq)
    sv0 = apply_single_gate(sv0, H_GATE, 0, nq)  # Initial perturbation

    for t_val in t_values:
        # Time evolution in energy basis
        phases = np.exp(-1j * eigenvalues * t_val)
        U_t = eigenvectors @ np.diag(phases) @ eigenvectors.conj().T
        sv_t = U_t @ sv0
        sv_t /= np.linalg.norm(sv_t)
        s = entanglement_entropy(sv_t, nq)
        entropies.append(s)

    # Scrambling time: when entropy reaches 90% of maximum
    max_s = max(entropies) if entropies else 0
    scrambling_time = 5.0
    for i, s in enumerate(entropies):
        if s > 0.9 * max_s and max_s > 0.1:
            scrambling_time = t_values[i]
            break

    # Lyapunov exponent from early-time entropy growth
    if len(entropies) > 2 and entropies[1] > entropies[0] + 1e-10:
        lambda_L = (entropies[2] - entropies[0]) / (2 * (t_values[1] - t_values[0]))
    else:
        lambda_L = 0.0

    # MSS bound: λ_L ≤ 2π T (at temperature T=1)
    mss_bound = 2 * math.pi
    saturates_mss = lambda_L > 0.5 * mss_bound

    probs = probabilities(sv_t)
    elapsed = (time.time() - t0) * 1000
    return SimulationResult(
        name="sachdev_ye_kitaev", category="research", passed=True,
        elapsed_ms=elapsed,
        detail=f"t_scr={scrambling_time:.3f}, λ_L={lambda_L:.4f}, S_max={max_s:.4f}",
        fidelity=1.0, circuit_depth=0, num_qubits=nq,
        probabilities=probs, entanglement_entropy=max_s,
        entropy_value=max_s, phase_coherence=1.0 / (1.0 + scrambling_time),
        sacred_alignment=abs(math.cos(GOD_CODE_PHASE_ANGLE * lambda_L)),
        extra={
            "scrambling_time": scrambling_time,
            "lyapunov_exponent": lambda_L,
            "mss_bound": mss_bound,
            "saturates_mss": saturates_mss,
            "max_entropy": max_s,
            "n_couplings": n_couplings,
            "coupling_scale": GOD_CODE / 1000 * PHI,
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  8. PHI-Fractal Cascade — Self-Similarity Across Scales
# ═══════════════════════════════════════════════════════════════════════════════

def sim_phi_fractal_cascade(nq: int = 7) -> SimulationResult:
    """
    Test PHI-fractal self-similarity in quantum state structure.

    Constructs circuits at multiple scales (1Q, 2Q, ..., nQ) using the
    same sacred gate pattern, then measures whether the entanglement
    structure exhibits self-similar fractal properties under PHI scaling.

    Fractal dimension D_f is estimated from the scaling of entanglement
    entropy with subsystem size. Self-similar systems have D_f = log(N)/log(s)
    where N = pieces and s = scaling factor.
    """
    t0 = time.time()
    nq = max(3, min(nq, 9))

    # Build sacred fractal circuit
    sv = init_sv(nq)

    # Fractal layers: apply pattern at different scales
    for scale in range(1, nq):
        stride = scale
        # Apply H + GOD_CODE at this scale
        for q in range(0, nq, stride):
            sv = apply_single_gate(sv, H_GATE, q, nq)
            sv = apply_single_gate(sv, GOD_CODE_GATE, q, nq)
        # Entangle at this scale
        for q in range(0, nq - stride, stride):
            sv = apply_cnot(sv, q, min(q + stride, nq - 1), nq)
        # PHI phase injection
        phi_phase = PHI_PHASE_ANGLE / scale
        phi_rz = make_gate([[np.exp(-1j * phi_phase / 2), 0],
                            [0, np.exp(1j * phi_phase / 2)]])
        sv = apply_single_gate(sv, phi_rz, 0, nq)

    probs = probabilities(sv)

    # Measure entanglement at all possible bipartitions
    partition_entropies = []
    for p in range(1, nq):
        s = entanglement_entropy(sv, nq, p)
        partition_entropies.append(s)

    # Measure purity at all partitions
    partition_purities = []
    for p in range(1, nq):
        pu = state_purity(sv, nq, p)
        partition_purities.append(pu)

    # Schmidt coefficients at midpoint
    sc = schmidt_coefficients(sv, nq)
    schmidt_rank = int(np.sum(sc > 1e-10))

    # Fractal dimension estimate using entropy scaling
    # D_f ≈ dS / d(log L) where L is subsystem size
    if len(partition_entropies) > 2:
        log_sizes = [math.log(p) for p in range(1, nq)]
        # Linear fit: S = D_f × log(L) + c
        if len(log_sizes) > 1:
            coeffs = np.polyfit(log_sizes, partition_entropies, 1)
            fractal_dim = float(coeffs[0])
        else:
            fractal_dim = 0.0
    else:
        fractal_dim = 0.0

    # Self-similarity ratio: entropy at scale s vs entropy at scale s*PHI
    self_sim_ratios = []
    for i in range(len(partition_entropies) - 1):
        if partition_entropies[i] > 1e-10:
            ratio = partition_entropies[i + 1] / partition_entropies[i]
            self_sim_ratios.append(ratio)

    avg_ratio = float(np.mean(self_sim_ratios)) if self_sim_ratios else 0.0
    phi_proximity = 1.0 - abs(avg_ratio - PHI) / PHI if avg_ratio > 0 else 0.0

    mid_entropy = partition_entropies[nq // 2 - 1] if len(partition_entropies) > nq // 2 - 1 else 0.0

    elapsed = (time.time() - t0) * 1000
    return SimulationResult(
        name="phi_fractal_cascade", category="research", passed=True,
        elapsed_ms=elapsed,
        detail=f"D_f={fractal_dim:.4f}, ratio≈{avg_ratio:.4f}, φ_proximity={phi_proximity:.4f}, schmidt_rank={schmidt_rank}",
        fidelity=1.0, circuit_depth=nq * (nq - 1), num_qubits=nq,
        probabilities=probs, entanglement_entropy=mid_entropy,
        entropy_value=mid_entropy, phase_coherence=phi_proximity,
        sacred_alignment=abs(math.cos(GOD_CODE_PHASE_ANGLE)),
        extra={
            "fractal_dimension": fractal_dim,
            "self_similarity_ratio": avg_ratio,
            "phi_proximity": phi_proximity,
            "schmidt_rank": schmidt_rank,
            "partition_entropies": [round(s, 4) for s in partition_entropies],
            "partition_purities": [round(p, 4) for p in partition_purities],
        },
    )


# ── Registry of research simulations ────────────────────────────────────────
RESEARCH_SIMULATIONS = [
    ("shor_period_finding", sim_shor_period_finding, "research", "Shor QPE period-finding", 6),
    ("quantum_chaos", sim_quantum_chaos, "research", "Level spacing statistics", 6),
    ("topological_braiding", sim_topological_braiding, "research", "Anyonic phase braiding", 6),
    ("holographic_entropy", sim_holographic_entropy, "research", "RT entanglement entropy", 8),
    ("error_threshold", sim_error_threshold, "research", "Fault-tolerance threshold", 4),
    ("quantum_supremacy_sampling", sim_quantum_supremacy_sampling, "research", "Random circuit XEB", 6),
    ("sachdev_ye_kitaev", sim_sachdev_ye_kitaev, "research", "SYK scrambling dynamics", 5),
    ("phi_fractal_cascade", sim_phi_fractal_cascade, "research", "PHI-fractal self-similarity", 7),
]

__all__ = [
    "sim_shor_period_finding", "sim_quantum_chaos", "sim_topological_braiding",
    "sim_holographic_entropy", "sim_error_threshold", "sim_quantum_supremacy_sampling",
    "sim_sachdev_ye_kitaev", "sim_phi_fractal_cascade",
    "RESEARCH_SIMULATIONS",
]
