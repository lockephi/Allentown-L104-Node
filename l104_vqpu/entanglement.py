"""L104 VQPU v13.2 — Entanglement Quantification + Quantum Information Metrics.

v13.2 GOD_CODE QUBIT UPGRADE:
  - Sacred entanglement now uses canonical GOD_CODE_PHASE (QPU-verified)
  - Phase decomposition resonance (IRON+PHI+OCTAVE) in Schmidt analysis
  - Berry phase alignment uses exact GOD_CODE_PHASE instead of GOD_CODE/100
  - QPU fidelity calibration for sacred scoring thresholds
"""

import math
import numpy as np

from .constants import GOD_CODE, PHI, VOID_CONSTANT

# v13.2: Import canonical phase angles for QPU-calibrated sacred scoring
try:
    from l104_god_code_simulator.god_code_qubit import (
        GOD_CODE_PHASE as _GC_PHASE,
        IRON_PHASE as _IRON_PHASE,
        PHI_CONTRIBUTION as _PHI_CONTRIB,
        OCTAVE_PHASE as _OCTAVE_PHASE,
        PHI_PHASE as _PHI_PHASE,
        QPU_DATA as _QPU_DATA,
    )
except ImportError:
    _GC_PHASE = GOD_CODE % (2.0 * math.pi)
    _IRON_PHASE = math.pi / 2.0
    _OCTAVE_PHASE = (4.0 * math.log(2.0)) % (2.0 * math.pi)
    _PHI_CONTRIB = (_GC_PHASE - _IRON_PHASE - _OCTAVE_PHASE) % (2.0 * math.pi)
    _PHI_PHASE = 2.0 * math.pi / PHI
    _QPU_DATA = {"mean_fidelity": 0.97475930}

_TWO_PI = 2.0 * math.pi

__all__ = ["EntanglementQuantifier", "QuantumInformationMetrics"]


class EntanglementQuantifier:
    """
    Quantifies entanglement in quantum states using information-theoretic
    measures, providing formal metrics beyond the basic CircuitAnalyzer.

    Metrics computed:
      - Von Neumann entropy:  S(ρ_A) for bipartition entanglement
      - Concurrence:          Two-qubit entanglement measure (0=separable, 1=Bell)
      - Schmidt rank:         Number of non-zero Schmidt coefficients
      - Entanglement spectrum: Full Schmidt coefficient distribution
      - Sacred entanglement:  GOD_CODE resonance in entanglement structure
    """

    @staticmethod
    def von_neumann_entropy(statevector, num_qubits: int,
                            partition: int = None) -> float:
        """
        Compute von Neumann entropy S(ρ_A) for a bipartition of the state.

        Traces out subsystem B (qubits >= partition) to get reduced density
        matrix ρ_A, then computes S = -Tr(ρ_A log₂ ρ_A).

        Args:
            statevector: Complex amplitude array of length 2^n
            num_qubits:  Total qubit count
            partition:   Qubit index to split at (default: n//2)

        Returns:
            Von Neumann entropy in bits. 0 = product state, log₂(d) = maximally entangled.
        """
        if partition is None:
            partition = max(1, num_qubits // 2)
        partition = max(1, min(partition, num_qubits - 1))

        sv = np.array(statevector, dtype=np.complex128)
        dim = 1 << num_qubits
        if len(sv) < dim:
            sv = np.pad(sv, (0, dim - len(sv)))

        d_a = 1 << partition
        d_b = 1 << (num_qubits - partition)

        # v12.2: SVD-based VNE — avoids building d_a × d_a density matrix.
        # eigenvalues(ρ_A) = singular_values(ψ)² — SVD on (d_a, d_b) matrix
        # is O(d_a × d_b × min(d_a,d_b)) vs eigvalsh on (d_a × d_a) which
        # is O(d_a³). For balanced partitions this is the same, but for
        # unbalanced ones (d_b << d_a) it's much faster and avoids the
        # d_a × d_a matmul entirely.
        psi = sv.reshape(d_a, d_b)
        S = np.linalg.svd(psi, compute_uv=False)
        eigenvalues = (S * S)  # λ_i = σ_i²
        eigenvalues = eigenvalues[eigenvalues > 1e-15]

        if len(eigenvalues) == 0:
            return 0.0

        # Von Neumann entropy: S = -Σ λ log₂(λ)
        entropy = -float(np.sum(eigenvalues * np.log2(eigenvalues)))
        return max(0.0, entropy)

    @staticmethod
    def concurrence(statevector, qubit_a: int = 0, qubit_b: int = 1,
                    num_qubits: int = 2) -> float:
        """
        Compute concurrence for a two-qubit subsystem.

        Concurrence C = max(0, λ₁ - λ₂ - λ₃ - λ₄) where λᵢ are the
        square roots of eigenvalues of ρ·ρ̃ in decreasing order,
        with ρ̃ = (σ_y ⊗ σ_y) ρ* (σ_y ⊗ σ_y).

        Returns: 0.0 (separable) to 1.0 (maximally entangled Bell state).
        """
        sv = np.array(statevector, dtype=np.complex128)
        dim = 1 << num_qubits

        if len(sv) < dim:
            sv = np.pad(sv, (0, dim - len(sv)))
        norm = np.linalg.norm(sv)
        if norm > 0:
            sv = sv / norm

        # For 2-qubit pure states, concurrence = 2|ad - bc| where ψ = a|00⟩+b|01⟩+c|10⟩+d|11⟩
        if num_qubits == 2:
            a, b, c, d = sv[0], sv[1], sv[2], sv[3]
            return min(1.0, float(2.0 * abs(a * d - b * c)))

        # For larger systems, compute reduced 2-qubit density matrix
        # v12.2: Vectorized partial trace via reshape — O(4·2^n) replaces O(n·4^n)
        n = num_qubits
        psi = sv.reshape([2] * n)
        # Move target qubits to front: axes=[qubit_a, qubit_b, ...rest...]
        other_qubits = [q for q in range(n) if q != qubit_a and q != qubit_b]
        psi_ordered = np.transpose(psi, [qubit_a, qubit_b] + other_qubits)
        # Reshape to (4, 2^(n-2))
        d_rest = 1 << len(other_qubits)
        psi_2d = psi_ordered.reshape(4, d_rest)
        # Reduced density matrix: ρ_AB = psi_2d @ psi_2d†
        rho_2q = psi_2d @ psi_2d.conj().T

        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        yy = np.kron(sigma_y, sigma_y)
        rho_tilde = yy @ rho_2q.conj() @ yy
        R = rho_2q @ rho_tilde
        eigenvalues = np.sort(np.sqrt(np.maximum(np.linalg.eigvals(R).real, 0)))[::-1]
        concurrence_val = max(0.0, float(eigenvalues[0] - sum(eigenvalues[1:])))
        return min(1.0, concurrence_val)

    @staticmethod
    def schmidt_decomposition(statevector, num_qubits: int,
                              partition: int = None) -> dict:
        """
        Compute the Schmidt decomposition of a bipartite state.

        Returns Schmidt coefficients (singular values), rank, and
        sacred alignment of the entanglement spectrum.
        """
        if partition is None:
            partition = max(1, num_qubits // 2)

        sv = np.array(statevector, dtype=np.complex128)
        dim = 1 << num_qubits
        if len(sv) < dim:
            sv = np.pad(sv, (0, dim - len(sv)))
        norm = np.linalg.norm(sv)
        if norm > 0:
            sv = sv / norm

        d_a = 1 << partition
        d_b = 1 << (num_qubits - partition)
        psi = sv.reshape(d_a, d_b)

        _, S, _ = np.linalg.svd(psi, full_matrices=False)
        coefficients = S[S > 1e-12].tolist()
        rank = len(coefficients)

        probs = [c ** 2 for c in coefficients]
        entropy = -sum(p * math.log2(p) for p in probs if p > 1e-15)

        phi_resonance = 0.0
        if len(coefficients) >= 2:
            ratio = coefficients[0] / coefficients[1] if coefficients[1] > 1e-12 else 0
            phi_resonance = max(0.0, 1.0 - abs(ratio - PHI) / PHI)

        # ── v13.2: Phase decomposition resonance ──────────────────────
        # Check if entanglement spectrum encodes the 3-rotation structure
        # of the GOD_CODE qubit (IRON π/2, PHI ~1.67, OCTAVE ~2.77).
        phase_decomp_resonance = 0.0
        if len(coefficients) >= 3:
            # Map first 3 Schmidt coefficients to phase-like values
            c_phases = [c * _TWO_PI for c in coefficients[:3]]
            # Score alignment of (c0,c1,c2) with (IRON,PHI,OCTAVE) phases
            _targets = [_IRON_PHASE / _TWO_PI, _PHI_CONTRIB / _TWO_PI,
                        _OCTAVE_PHASE / _TWO_PI]
            _coefs_norm = [c / sum(coefficients[:3]) for c in coefficients[:3]]
            _deviations = [abs(cn - t) for cn, t in zip(_coefs_norm, _targets)]
            phase_decomp_resonance = max(0.0, 1.0 - sum(_deviations) / 1.5)

        return {
            "schmidt_coefficients": [round(c, 8) for c in coefficients[:10]],
            "schmidt_rank": rank,
            "entanglement_entropy": round(entropy, 6),
            "max_entropy": round(math.log2(min(d_a, d_b)), 6) if min(d_a, d_b) > 0 else 0,
            "is_entangled": rank > 1,
            "is_maximally_entangled": abs(entropy - math.log2(min(d_a, d_b))) < 0.01 if min(d_a, d_b) > 1 else False,
            "phi_resonance": round(phi_resonance, 6),
            "phase_decomposition_resonance": round(phase_decomp_resonance, 6),
            "partition": partition,
        }

    @staticmethod
    def purity(statevector, num_qubits: int, partition: int = None) -> float:
        """
        Compute purity Tr(ρ_A²) of the reduced density matrix.

        Purity = 1 for pure states, 1/d for maximally mixed states.
        Related to linear entropy: S_L = 1 - Tr(ρ²).

        Uses SVD-based approach: Tr(ρ_A²) = Σ σ_i⁴ where σ_i are the
        singular values of the bipartite reshaping.

        Args:
            statevector: Complex amplitude array
            num_qubits:  Total qubit count
            partition:   Qubit index to split at (default: n//2)

        Returns:
            Purity value in [1/d_A, 1.0]
        """
        if partition is None:
            partition = max(1, num_qubits // 2)
        partition = max(1, min(partition, num_qubits - 1))

        sv = np.array(statevector, dtype=np.complex128)
        dim = 1 << num_qubits
        if len(sv) < dim:
            sv = np.pad(sv, (0, dim - len(sv)))
        norm = np.linalg.norm(sv)
        if norm > 0:
            sv = sv / norm

        d_a = 1 << partition
        d_b = 1 << (num_qubits - partition)
        psi = sv.reshape(d_a, d_b)

        # SVD: singular values give Schmidt coefficients
        S = np.linalg.svd(psi, compute_uv=False)
        # Purity = Σ σ_i⁴  (eigenvalues of ρ_A are σ_i²)
        lambdas = S * S
        return float(np.sum(lambdas * lambdas))

    @staticmethod
    def logarithmic_negativity(statevector, num_qubits: int,
                                partition: int = None) -> float:
        """
        Compute logarithmic negativity E_N = log₂ ||ρ^{T_B}||₁.

        The logarithmic negativity is an entanglement monotone computable
        from the partial transpose. For pure bipartite states:
        E_N = log₂(Σ σ_i)² where σ_i are Schmidt coefficients, which
        equals log₂(Tr(√(ρ_A)))².

        E_N = 0 for separable states, > 0 for entangled states.
        Upper bound: log₂(d) where d = min(d_A, d_B).

        Args:
            statevector: Complex amplitude array
            num_qubits:  Total qubit count
            partition:   Qubit index to split at (default: n//2)

        Returns:
            Logarithmic negativity (non-negative float)
        """
        if partition is None:
            partition = max(1, num_qubits // 2)
        partition = max(1, min(partition, num_qubits - 1))

        sv = np.array(statevector, dtype=np.complex128)
        dim = 1 << num_qubits
        if len(sv) < dim:
            sv = np.pad(sv, (0, dim - len(sv)))
        norm = np.linalg.norm(sv)
        if norm > 0:
            sv = sv / norm

        d_a = 1 << partition
        d_b = 1 << (num_qubits - partition)
        psi = sv.reshape(d_a, d_b)

        # For pure states: E_N = 2 log₂(Σ σ_i) = 2 log₂(||ψ||_tr)
        # where ||ψ||_tr = Σ σ_i (trace norm = sum of Schmidt coefficients)
        S = np.linalg.svd(psi, compute_uv=False)
        S = S[S > 1e-15]
        trace_norm = float(np.sum(S))

        if trace_norm <= 1.0 + 1e-12:
            return 0.0  # Separable (or nearly so)

        return float(2.0 * math.log2(trace_norm))

    @staticmethod
    def full_analysis(statevector, num_qubits: int) -> dict:
        """Complete entanglement analysis with all metrics."""
        result = {"num_qubits": num_qubits}
        result["von_neumann_entropy"] = round(
            EntanglementQuantifier.von_neumann_entropy(statevector, num_qubits), 6)
        result["schmidt"] = EntanglementQuantifier.schmidt_decomposition(
            statevector, num_qubits)
        if num_qubits >= 2:
            result["concurrence_01"] = round(
                EntanglementQuantifier.concurrence(statevector, 0, 1, num_qubits), 6)

        # Purity and logarithmic negativity (v12.3)
        result["purity"] = round(
            EntanglementQuantifier.purity(statevector, num_qubits), 8)
        result["linear_entropy"] = round(1.0 - result["purity"], 8)
        result["logarithmic_negativity"] = round(
            EntanglementQuantifier.logarithmic_negativity(statevector, num_qubits), 6)

        vne = result["von_neumann_entropy"]
        max_ent = math.log2(min(1 << (num_qubits // 2), 1 << ((num_qubits + 1) // 2)))
        ent_frac = vne / max_ent if max_ent > 0 else 0
        phi_res = result["schmidt"].get("phi_resonance", 0)
        phase_res = result["schmidt"].get("phase_decomposition_resonance", 0)

        # v13.2: Sacred score now includes phase decomposition resonance
        # and QPU fidelity calibration. The three components are:
        #   1. Entanglement fraction (weighted by φ)
        #   2. PHI resonance in Schmidt spectrum
        #   3. Phase decomposition resonance (IRON+PHI+OCTAVE)
        # QPU mean fidelity scales the score to account for real hardware.
        qpu_fid = _QPU_DATA.get("mean_fidelity", 0.975) if isinstance(_QPU_DATA, dict) else 0.975
        w_ent = PHI / (1 + PHI + 0.5)      # ~0.618/2.118 ≈ 0.292
        w_phi = 1.0 / (1 + PHI + 0.5)      # ~1/2.118     ≈ 0.472
        w_phase = 0.5 / (1 + PHI + 0.5)    # ~0.5/2.118   ≈ 0.236
        raw_score = w_ent * ent_frac + w_phi * phi_res + w_phase * phase_res
        result["sacred_entanglement_score"] = round(raw_score * qpu_fid, 6)
        result["qpu_calibrated"] = True
        return result


# ═══════════════════════════════════════════════════════════════════
# QUANTUM INFORMATION METRICS (v8.0) — Advanced Quantum Equations
# ═══════════════════════════════════════════════════════════════════

class QuantumInformationMetrics:
    """
    Advanced quantum information-theoretic metrics for circuit analysis.

    v8.0 NEW — Six sacred-aligned quantum equations:

    1. Quantum Fisher Information (QFI):
       F_Q = 4(⟨∂ψ|∂ψ⟩ - |⟨∂ψ|ψ⟩|²)
       Bounds parameter estimation precision via Cramér-Rao: Δθ ≥ 1/√(N·F_Q)

    2. Berry Phase (Geometric Phase):
       γ = -Im Σ ln⟨ψ(k)|ψ(k+1)⟩
       Discrete approximation of ∮⟨ψ|∇|ψ⟩·dR over parameter cycle

    3. Quantum Mutual Information:
       I(A:B) = S(ρ_A) + S(ρ_B) - S(ρ_AB)
       Total classical + quantum correlations between subsystems

    4. Quantum Relative Entropy:
       S(ρ||σ) = Tr[ρ(log ρ - log σ)]
       Distinguishability between quantum states (asymmetric)

    5. Loschmidt Echo:
       L(t) = |⟨ψ₀|e^{iH't}e^{-iHt}|ψ₀⟩|²
       Quantum chaos / sensitivity to Hamiltonian perturbation

    6. Topological Entanglement Entropy:
       γ_topo = S_total - Σ S_boundary
       Detects topological order beyond local entanglement

    All metrics are PHI-aligned: sacred resonance is computed for each.
    """

    @staticmethod
    def quantum_fisher_information(statevector, generator_ops: list,
                                    num_qubits: int,
                                    delta: float = 1e-4) -> dict:
        """
        Compute Quantum Fisher Information for a parameterised state.

        Uses numerical differentiation:
        F_Q(θ) = 4[⟨∂_θψ|∂_θψ⟩ - |⟨∂_θψ|ψ⟩|²]

        The QFI determines the ultimate precision limit for estimating
        the parameter θ encoded in the state |ψ(θ)⟩.

        Args:
            statevector:   Current state |ψ(θ)⟩
            generator_ops: List of gate operations that define dψ/dθ
                           (the parameterised layer with angle θ)
            num_qubits:    Total qubit count
            delta:         Finite-difference step size

        Returns:
            dict with 'qfi', 'cramer_rao_bound', 'sacred_alignment'
        """
        sv = np.array(statevector, dtype=np.complex128)
        dim = 1 << num_qubits
        if len(sv) < dim:
            sv = np.pad(sv, (0, dim - len(sv)))
        norm = np.linalg.norm(sv)
        if norm > 0:
            sv = sv / norm

        # Build generator matrix from ops (sum of Pauli generators)
        gen_matrix = np.zeros((dim, dim), dtype=np.complex128)
        for op in generator_ops:
            gate = op.get("gate", "Z") if isinstance(op, dict) else "Z"
            qubits = op.get("qubits", [0]) if isinstance(op, dict) else [0]
            param_val = 1.0  # coefficient
            if gate in ("Rz", "Rz"):
                # Generator is Z/2 on target qubit
                z = np.array([[1, 0], [0, -1]], dtype=np.complex128) / 2.0
                mat = np.eye(1, dtype=np.complex128)
                for q in range(num_qubits):
                    mat = np.kron(mat, z if q == qubits[0] else np.eye(2, dtype=np.complex128))
                gen_matrix += param_val * mat
            elif gate in ("Ry",):
                y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128) / 2.0
                mat = np.eye(1, dtype=np.complex128)
                for q in range(num_qubits):
                    mat = np.kron(mat, y if q == qubits[0] else np.eye(2, dtype=np.complex128))
                gen_matrix += param_val * mat
            elif gate in ("Rx",):
                x = np.array([[0, 1], [1, 0]], dtype=np.complex128) / 2.0
                mat = np.eye(1, dtype=np.complex128)
                for q in range(num_qubits):
                    mat = np.kron(mat, x if q == qubits[0] else np.eye(2, dtype=np.complex128))
                gen_matrix += param_val * mat

        # |∂ψ/∂θ⟩ = -i·G|ψ⟩
        d_sv = -1j * gen_matrix @ sv

        # QFI = 4[⟨∂ψ|∂ψ⟩ - |⟨∂ψ|ψ⟩|²]
        inner_dd = float(np.real(np.dot(d_sv.conj(), d_sv)))
        inner_ds = np.dot(d_sv.conj(), sv)
        qfi = 4.0 * (inner_dd - float(np.abs(inner_ds) ** 2))
        qfi = max(0.0, qfi)

        # Cramér-Rao bound: Δθ ≥ 1/√(N×QFI) for N=1 shot
        cramer_rao = 1.0 / math.sqrt(qfi) if qfi > 1e-15 else float('inf')

        # Sacred alignment: QFI normalized by dim, compared to PHI
        qfi_norm = qfi / dim if dim > 0 else 0
        phi_dev = abs(qfi_norm - PHI) / PHI if PHI > 0 else 1
        sacred_res = max(0.0, 1.0 - phi_dev)

        return {
            "qfi": round(qfi, 8),
            "cramer_rao_bound": round(cramer_rao, 8) if cramer_rao < 1e6 else float('inf'),
            "qfi_per_qubit": round(qfi / max(num_qubits, 1), 8),
            "heisenberg_limited": qfi > num_qubits,
            "sacred_alignment": round(sacred_res, 6),
            "god_code": GOD_CODE,
            "num_qubits": num_qubits,
        }

    @staticmethod
    def berry_phase(statevectors: list, num_qubits: int) -> dict:
        """
        Compute Berry (geometric) phase from a cyclic evolution of states.

        γ = -Im Σ_k ln⟨ψ(k)|ψ(k+1)⟩

        The Berry phase is a topological invariant that depends only on
        the path in parameter space, not the speed of traversal.

        Args:
            statevectors: List of statevectors along a parameter cycle
                          (first and last should be close for closed loop)
            num_qubits:   Qubit count

        Returns:
            dict with 'berry_phase', 'geometric_phase_mod_2pi', 'sacred_alignment'
        """
        if len(statevectors) < 3:
            return {"berry_phase": 0.0, "error": "need_at_least_3_states"}

        dim = 1 << num_qubits
        phase_sum = 0.0 + 0.0j

        for k in range(len(statevectors)):
            sv_k = np.array(statevectors[k], dtype=np.complex128)
            sv_next = np.array(statevectors[(k + 1) % len(statevectors)], dtype=np.complex128)
            if len(sv_k) < dim:
                sv_k = np.pad(sv_k, (0, dim - len(sv_k)))
            if len(sv_next) < dim:
                sv_next = np.pad(sv_next, (0, dim - len(sv_next)))

            overlap = np.dot(sv_k.conj(), sv_next)
            if abs(overlap) > 1e-15:
                phase_sum += np.log(overlap)

        berry = -float(np.imag(phase_sum))
        berry_mod = berry % (2 * math.pi)

        # Sacred alignment: Berry phase vs PHI-scaled π
        phi_pi = PHI * math.pi
        sacred_dev = abs(berry_mod - phi_pi % (2 * math.pi)) / (2 * math.pi)
        sacred_res = max(0.0, 1.0 - sacred_dev * 2)

        # v13.2: GOD_CODE phase alignment uses canonical QPU-verified phase
        gc_phase = _GC_PHASE  # canonical GOD_CODE mod 2π ≈ 6.0141
        gc_dev = abs(berry_mod - gc_phase) / _TWO_PI
        gc_alignment = max(0.0, 1.0 - gc_dev * 2)

        # v13.2: Iron lattice phase alignment (IRON = π/2)
        iron_dev = abs(berry_mod - _IRON_PHASE) / _TWO_PI
        iron_alignment = max(0.0, 1.0 - iron_dev * 4)

        return {
            "berry_phase": round(berry, 8),
            "geometric_phase_mod_2pi": round(berry_mod, 8),
            "states_in_cycle": len(statevectors),
            "sacred_alignment": round(sacred_res, 6),
            "god_code_alignment": round(gc_alignment, 6),
            "iron_lattice_alignment": round(iron_alignment, 6),
            "phi_pi_target": round(phi_pi % _TWO_PI, 8),
            "god_code_phase_target": round(_GC_PHASE, 8),
            "god_code": GOD_CODE,
        }

    @staticmethod
    def quantum_mutual_information(statevector, num_qubits: int,
                                    partition_a: int = None,
                                    partition_b: int = None) -> dict:
        """
        Compute quantum mutual information I(A:B) = S(A) + S(B) - S(AB).

        Measures total correlations (classical + quantum) between
        two subsystems A and B of a bipartite quantum state.

        Args:
            statevector: State amplitude vector
            num_qubits:  Total qubits
            partition_a: Last qubit index of subsystem A (default: n//3)
            partition_b: First qubit index of subsystem B (default: 2n//3)

        Returns:
            dict with 'mutual_information', 'S_A', 'S_B', 'S_AB', 'sacred_alignment'
        """
        if partition_a is None:
            partition_a = max(1, num_qubits // 3)
        if partition_b is None:
            partition_b = max(partition_a + 1, 2 * num_qubits // 3)
        partition_b = min(partition_b, num_qubits)

        # S(A) - entropy of subsystem A
        s_a = EntanglementQuantifier.von_neumann_entropy(
            statevector, num_qubits, partition=partition_a)

        # S(B) - entropy of subsystem B
        s_b = EntanglementQuantifier.von_neumann_entropy(
            statevector, num_qubits, partition=partition_b)

        # S(AB) - entropy of full system (0 for pure states)
        sv = np.array(statevector, dtype=np.complex128)
        dim = 1 << num_qubits
        if len(sv) < dim:
            sv = np.pad(sv, (0, dim - len(sv)))
        norm = np.linalg.norm(sv)
        # For pure states, S(AB) = 0
        s_ab = 0.0 if abs(norm - 1.0) < 1e-10 else s_a  # approximate

        mutual_info = s_a + s_b - s_ab

        # Sacred alignment: MI compared to GOD_CODE harmonic
        gc_harmonic = (GOD_CODE / 1000.0) * num_qubits
        mi_ratio = mutual_info / gc_harmonic if gc_harmonic > 0 else 0
        sacred_res = max(0.0, 1.0 - abs(mi_ratio - VOID_CONSTANT + 1.0) * 5)

        return {
            "mutual_information": round(mutual_info, 8),
            "S_A": round(s_a, 8),
            "S_B": round(s_b, 8),
            "S_AB": round(s_ab, 8),
            "partition_a": partition_a,
            "partition_b": partition_b,
            "is_correlated": mutual_info > 1e-6,
            "sacred_alignment": round(sacred_res, 6),
            "god_code": GOD_CODE,
        }

    @staticmethod
    def quantum_relative_entropy(statevector_rho, statevector_sigma,
                                  num_qubits: int) -> dict:
        """
        Compute quantum relative entropy S(ρ||σ) = Tr[ρ(log ρ - log σ)].

        Measures the distinguishability of two quantum states.
        S(ρ||σ) ≥ 0, with equality iff ρ = σ.

        Args:
            statevector_rho: State vector for ρ
            statevector_sigma: State vector for σ
            num_qubits: Qubit count

        Returns:
            dict with 'relative_entropy', 'fidelity', 'trace_distance', 'sacred_alignment'
        """
        dim = 1 << num_qubits

        sv_rho = np.array(statevector_rho, dtype=np.complex128)
        sv_sigma = np.array(statevector_sigma, dtype=np.complex128)
        if len(sv_rho) < dim:
            sv_rho = np.pad(sv_rho, (0, dim - len(sv_rho)))
        if len(sv_sigma) < dim:
            sv_sigma = np.pad(sv_sigma, (0, dim - len(sv_sigma)))

        norm_r = np.linalg.norm(sv_rho)
        norm_s = np.linalg.norm(sv_sigma)
        if norm_r > 0:
            sv_rho /= norm_r
        if norm_s > 0:
            sv_sigma /= norm_s

        # Pure-state density matrices
        rho = np.outer(sv_rho, sv_rho.conj())
        sigma = np.outer(sv_sigma, sv_sigma.conj())

        # Fidelity F = |⟨ψ|φ⟩|²
        fidelity = float(np.abs(np.dot(sv_rho.conj(), sv_sigma)) ** 2)

        # Trace distance T = ½||ρ - σ||₁
        diff = rho - sigma
        eigenvalues = np.linalg.eigvalsh(diff).real
        trace_distance = float(0.5 * np.sum(np.abs(eigenvalues)))

        # Relative entropy for pure states:
        # S(ρ||σ) = -log F(ρ, σ) when both are pure
        if fidelity > 1e-15:
            relative_entropy = -math.log(fidelity)
        else:
            relative_entropy = float('inf')

        # Sacred alignment: fidelity to PHI ratio
        phi_fid = abs(fidelity - 1.0 / PHI)
        sacred_res = max(0.0, 1.0 - phi_fid * PHI)

        return {
            "relative_entropy": round(relative_entropy, 8) if relative_entropy < 1e6 else float('inf'),
            "fidelity": round(fidelity, 8),
            "trace_distance": round(trace_distance, 8),
            "states_distinguishable": trace_distance > 0.01,
            "sacred_alignment": round(sacred_res, 6),
            "god_code": GOD_CODE,
        }

    @staticmethod
    def loschmidt_echo(statevector, hamiltonian_ops: list,
                       perturbation_ops: list, num_qubits: int,
                       time_steps: int = 20,
                       dt: float = 0.1) -> dict:
        """
        Compute Loschmidt echo (fidelity decay) for quantum chaos detection.

        L(t) = |⟨ψ₀|e^{iH't}·e^{-iHt}|ψ₀⟩|²

        Forward-evolves under H, then backward-evolves under H' = H + ε·V.
        Rapid decay indicates quantum chaos; slow decay indicates integrability.

        Args:
            statevector:      Initial state |ψ₀⟩
            hamiltonian_ops:  Original Hamiltonian as gate operations
            perturbation_ops: Perturbation V as gate operations
            num_qubits:       Qubit count
            time_steps:       Number of Trotterized time steps
            dt:               Time step size (controls evolution speed)

        Returns:
            dict with 'echo_values', 'decay_rate', 'is_chaotic',
            'lyapunov_estimate', 'sacred_alignment'
        """
        dim = 1 << num_qubits
        sv = np.array(statevector, dtype=np.complex128)
        if len(sv) < dim:
            sv = np.pad(sv, (0, dim - len(sv)))
        norm = np.linalg.norm(sv)
        if norm > 0:
            sv = sv / norm

        psi_0 = sv.copy()
        echo_values = [1.0]  # L(0) = 1 always

        # Build H and H' as matrices (small qubit counts)
        def _build_hamiltonian(ops, nq):
            d = 1 << nq
            H = np.zeros((d, d), dtype=np.complex128)
            paulis = {'I': np.eye(2), 'X': np.array([[0, 1], [1, 0]]),
                      'Y': np.array([[0, -1j], [1j, 0]]),
                      'Z': np.array([[1, 0], [0, -1]])}
            for op in ops:
                gate = op.get("gate", "Z") if isinstance(op, dict) else "Z"
                qubits = op.get("qubits", [0]) if isinstance(op, dict) else [0]
                params = op.get("parameters", [1.0]) if isinstance(op, dict) else [1.0]
                coeff = params[0] if params else 1.0
                if gate in paulis:
                    mat = np.eye(1, dtype=np.complex128)
                    for q in range(nq):
                        mat = np.kron(mat, paulis[gate] if q in qubits else paulis['I'])
                    H += coeff * mat
                elif gate == "ZZ":
                    mat = np.eye(1, dtype=np.complex128)
                    for q in range(nq):
                        mat = np.kron(mat, paulis['Z'] if q in qubits else paulis['I'])
                    H += coeff * mat
            return H

        H = _build_hamiltonian(hamiltonian_ops, num_qubits)
        V = _build_hamiltonian(perturbation_ops, num_qubits)
        H_prime = H + V

        # Forward evolution under H, backward under H'
        # Using second-order Trotter (Strang splitting): e^{-iHdt} ≈ e^{-iH dt/2} · e^{-iH dt/2}
        # This gives O(dt³) error per step vs O(dt²) for first-order
        U_fwd = np.eye(dim, dtype=np.complex128)
        U_bwd = np.eye(dim, dtype=np.complex128)

        # Pre-compute time-evolution operators using exact matrix exponential.
        # The Padé approximant via scipy.linalg.expm is unitary to machine
        # precision, unlike Taylor truncation which was non-unitary.
        try:
            from scipy.linalg import expm as _expm
        except ImportError:
            # Fallback: eigendecomposition-based expm
            def _expm(A):
                eigvals, V = np.linalg.eig(A)
                return (V * np.exp(eigvals)) @ np.linalg.inv(V)

        # Cache single-step propagators (they're reused every step)
        U_step_fwd = _expm(-1j * dt * H)          # e^{-iHdt}
        U_step_bwd = _expm(1j * dt * H_prime)      # e^{+iH'dt}

        for step in range(1, time_steps + 1):
            U_fwd = U_step_fwd @ U_fwd
            U_bwd = U_step_bwd @ U_bwd

            # L(t) = |⟨ψ₀|U_bwd·U_fwd|ψ₀⟩|²
            evolved = U_bwd @ U_fwd @ psi_0
            echo = float(np.abs(np.dot(psi_0.conj(), evolved)) ** 2)
            echo_values.append(min(1.0, echo))

        # Analyze decay
        echoes = np.array(echo_values)
        # Exponential fit: L(t) ≈ e^{-λt} → ln(L) ≈ -λt
        valid = echoes > 1e-10
        if np.sum(valid) > 2:
            log_echoes = np.log(echoes[valid])
            times = np.arange(len(echoes))[valid] * dt
            if len(times) > 1:
                # Linear fit to log(L) vs t
                coeffs = np.polyfit(times, log_echoes, 1)
                decay_rate = -float(coeffs[0])
            else:
                decay_rate = 0.0
        else:
            decay_rate = float('inf')

        # Lyapunov exponent estimate (quantum analog)
        lyapunov = decay_rate / 2.0 if decay_rate < 100 else float('inf')

        # Chaos classification
        is_chaotic = decay_rate > PHI  # PHI as chaos threshold

        # Sacred alignment: decay rate vs GOD_CODE harmonic
        gc_rate = GOD_CODE / 1000.0
        sacred_dev = abs(decay_rate - gc_rate) / gc_rate if gc_rate > 0 else 1
        sacred_res = max(0.0, 1.0 - sacred_dev)

        return {
            "echo_values": [round(e, 8) for e in echo_values],
            "decay_rate": round(decay_rate, 8) if decay_rate < 1e6 else float('inf'),
            "lyapunov_estimate": round(lyapunov, 8) if lyapunov < 1e6 else float('inf'),
            "is_chaotic": is_chaotic,
            "chaos_threshold": round(PHI, 6),
            "final_echo": round(float(echo_values[-1]), 8),
            "time_steps": time_steps,
            "dt": dt,
            "sacred_alignment": round(sacred_res, 6),
            "god_code": GOD_CODE,
        }

    @staticmethod
    def topological_entanglement_entropy(statevector, num_qubits: int) -> dict:
        """
        Estimate topological entanglement entropy γ_topo.

        Uses the Kitaev-Preskill construction:
        γ_topo = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC

        For a state with topological order, γ_topo = -log(D) where D is
        the total quantum dimension of the anyonic excitations.

        Args:
            statevector: State amplitude vector
            num_qubits:  Qubit count (should be ≥ 4 for meaningful result)

        Returns:
            dict with 'topological_entropy', 'has_topological_order',
            'quantum_dimension_estimate', 'sacred_alignment'
        """
        if num_qubits < 4:
            return {"topological_entropy": 0.0,
                    "error": "need_at_least_4_qubits",
                    "has_topological_order": False}

        # Partition into 3 regions: A, B, C (contiguous qubit blocks)
        # Kitaev-Preskill: γ = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC
        n_a = num_qubits // 3
        n_b = num_qubits // 3
        n_c = num_qubits - n_a - n_b

        # Compute entropies for all subsets using bipartite VNE
        # S(X) = VNE of reduced density matrix obtained by tracing out complement of X
        vne = EntanglementQuantifier.von_neumann_entropy

        # S_A: trace out B and C (partition at qubit n_a)
        s_a = vne(statevector, num_qubits, partition=n_a)
        # S_B: for middle block, use S_AB - S_A (subadditivity relation)
        s_ab = vne(statevector, num_qubits, partition=n_a + n_b)
        # S_C: trace out A and B (partition at n_a + n_b gives S_AB, complement is S_C)
        # For pure state: S(C) = S(AB)
        s_c = s_ab
        # S_B: use S(A ∪ C) = S(B) for pure state
        # S(B) via tracing out A and C: reorder qubits conceptually
        # For contiguous partition: S_B ≈ S(partition=n_a..n_a+n_b)
        # Approximate with subadditivity: S_B ≤ S_AB
        s_b = max(0.0, s_ab - s_a)
        # S_BC: trace out A (same as s_ab partition from A's perspective but on BC)
        # For pure state: S(BC) = S(A) = s_a
        s_bc = s_a
        # S_AC: for pure state: S(AC) = S(B)
        s_ac = s_b
        s_abc = 0.0  # Pure state: S(ABC) = 0

        gamma_topo = s_a + s_b + s_c - s_ab - s_bc - s_ac + s_abc
        gamma_topo = abs(gamma_topo)

        # Quantum dimension estimate: D ≈ exp(γ_topo)
        quantum_dim = math.exp(gamma_topo) if gamma_topo < 20 else float('inf')

        # Has topological order if γ_topo significantly > 0
        has_topo = gamma_topo > 0.1

        # Sacred alignment: γ_topo vs VOID_CONSTANT
        void_dev = abs(gamma_topo - (VOID_CONSTANT - 1.0)) * 10
        sacred_res = max(0.0, 1.0 - void_dev)

        return {
            "topological_entropy": round(gamma_topo, 8),
            "has_topological_order": has_topo,
            "quantum_dimension_estimate": round(quantum_dim, 6),
            "region_entropies": {
                "S_A": round(s_a, 6), "S_B": round(s_b, 6), "S_C": round(s_c, 6),
                "S_AB": round(s_ab, 6), "S_BC": round(s_bc, 6), "S_AC": round(s_ac, 6),
            },
            "sacred_alignment": round(sacred_res, 6),
            "void_constant_target": round(VOID_CONSTANT - 1.0, 8),
            "god_code": GOD_CODE,
        }

    @staticmethod
    def full_metrics(statevector, num_qubits: int,
                     generator_ops: list = None) -> dict:
        """
        Compute all quantum information metrics for a state.

        Returns a comprehensive quantum information profile including
        QFI, mutual information, and topological entropy.
        """
        result = {"num_qubits": num_qubits, "god_code": GOD_CODE}

        # Quantum mutual information
        result["mutual_information"] = QuantumInformationMetrics \
            .quantum_mutual_information(statevector, num_qubits)

        # Topological entanglement entropy
        if num_qubits >= 4:
            result["topological"] = QuantumInformationMetrics \
                .topological_entanglement_entropy(statevector, num_qubits)

        # QFI (if generator ops provided)
        if generator_ops:
            result["fisher_information"] = QuantumInformationMetrics \
                .quantum_fisher_information(statevector, generator_ops, num_qubits)

        return result
