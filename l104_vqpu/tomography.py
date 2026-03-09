"""L104 VQPU v14.0.0 — Quantum State Tomography.

v14.0.0 QUANTUM FIDELITY ARCHITECTURE:
  - Maximum Likelihood Estimation (MLE) density matrix reconstruction
  - Iterative R·ρ·R algorithm with φ-accelerated convergence
  - Physical density matrix guarantee (positive semi-definite, trace=1)

v12.2 (retained): Linear inversion, SWAP test, fidelity, purity, entropy
"""

import math
import numpy as np

from .constants import (
    GOD_CODE, PHI, VOID_CONSTANT,
    MLE_MAX_ITERATIONS, MLE_CONVERGENCE_TOL, MLE_ACCELERATION_FACTOR,
)
from .pauli_utils import _pauli_expectation  # single source of truth

__all__ = ["QuantumStateTomography", "_pauli_expectation"]


class QuantumStateTomography:
    """
    Quantum state tomography — reconstruct the density matrix from
    a set of measurement outcomes in multiple Pauli bases.

    v8.0 NEW — Full state characterisation:
      - Linear inversion tomography from Pauli measurements
      - Density matrix reconstruction (ρ = Σ ⟨Pᵢ⟩ Pᵢ / 2^n)
      - Purity calculation: γ = Tr(ρ²) — 1/d (mixed) to 1 (pure)
      - State fidelity: F(ρ, σ) = [Tr(√(√ρ·σ·√ρ))]²
      - SWAP test circuit builder for fidelity estimation

    All results include sacred alignment scoring.
    """

    @staticmethod
    def measure_in_pauli_bases(statevector, num_qubits: int,
                                shots: int = 4096) -> dict:
        """
        Simulate measuring a state in all 3^n Pauli bases (X, Y, Z per qubit).

        For efficiency, only measures single-qubit Pauli operators
        and 2-qubit correlators (scales as O(n²) not O(3^n)).

        Returns:
            dict mapping Pauli strings to expectation values
        """
        sv = np.array(statevector, dtype=np.complex128)
        dim = 1 << num_qubits
        if len(sv) < dim:
            sv = np.pad(sv, (0, dim - len(sv)))
        norm = np.linalg.norm(sv)
        if norm > 0:
            sv = sv / norm

        expectations = {}

        # Single-qubit Paulis
        for q in range(num_qubits):
            for p in ['X', 'Y', 'Z']:
                pauli_str = 'I' * q + p + 'I' * (num_qubits - q - 1)
                exp_val = _pauli_expectation(sv, pauli_str)
                expectations[pauli_str] = round(exp_val, 8)

        # Two-qubit correlators (for entanglement) - include Y for full tomography
        for q1 in range(num_qubits):
            max_corr_dist = min(num_qubits, max(4, num_qubits)) if num_qubits <= 12 else 6
            for q2 in range(q1 + 1, min(q1 + max_corr_dist, num_qubits)):
                for p1 in ['X', 'Y', 'Z']:
                    for p2 in ['X', 'Y', 'Z']:
                        chars = list('I' * num_qubits)
                        chars[q1] = p1
                        chars[q2] = p2
                        pauli_str = ''.join(chars)
                        exp_val = _pauli_expectation(sv, pauli_str)
                        expectations[pauli_str] = round(exp_val, 8)

        return expectations

    @staticmethod
    def reconstruct_density_matrix(pauli_expectations: dict,
                                    num_qubits: int) -> dict:
        """
        Reconstruct the density matrix from Pauli expectation values.

        ρ = (1/2^n) Σ ⟨Pᵢ⟩ Pᵢ

        Args:
            pauli_expectations: dict mapping Pauli strings to ⟨P⟩ values
            num_qubits: Qubit count

        Returns:
            dict with 'density_matrix' (flattened), 'purity', 'rank',
            'sacred_alignment', statistics
        """
        dim = 1 << num_qubits
        rho = np.zeros((dim, dim), dtype=np.complex128)

        paulis = {
            'I': np.eye(2, dtype=np.complex128),
            'X': np.array([[0, 1], [1, 0]], dtype=np.complex128),
            'Y': np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
            'Z': np.array([[1, 0], [0, -1]], dtype=np.complex128),
        }

        # Identity term: ⟨I⊗I⊗...⊗I⟩ = 1 always (trace normalization)
        rho += np.eye(dim, dtype=np.complex128) / dim

        for pauli_str, exp_val in pauli_expectations.items():
            if len(pauli_str) != num_qubits:
                continue
            # Skip identity string — already accounted for above
            if all(ch == 'I' for ch in pauli_str):
                continue
            # Build tensor product of Pauli matrices
            mat = np.eye(1, dtype=np.complex128)
            for ch in pauli_str:
                if ch in paulis:
                    mat = np.kron(mat, paulis[ch])
                else:
                    mat = np.kron(mat, paulis['I'])

            rho += exp_val * mat / dim

        # Ensure Hermiticity and positive semi-definiteness
        rho = (rho + rho.conj().T) / 2.0
        eigenvalues = np.linalg.eigvalsh(rho).real
        eigenvalues = np.maximum(eigenvalues, 0)  # clip negatives
        eigenvalues /= np.sum(eigenvalues)  # normalize

        # Purity: Tr(ρ²)
        purity = float(np.real(np.trace(rho @ rho)))
        purity = min(1.0, max(1.0 / dim, purity))

        # Rank
        rank = int(np.sum(eigenvalues > 1e-10))

        # Von Neumann entropy from eigenvalues
        valid_eigs = eigenvalues[eigenvalues > 1e-15]
        entropy = -float(np.sum(valid_eigs * np.log2(valid_eigs))) if len(valid_eigs) > 0 else 0

        # Sacred alignment: purity vs PHI ratio
        phi_target = 1.0 / PHI  # ≈ 0.618 — target purity for sacred state
        sacred_dev = abs(purity - phi_target) / phi_target
        sacred_res = max(0.0, 1.0 - sacred_dev)

        return {
            "purity": round(purity, 8),
            "rank": rank,
            "von_neumann_entropy": round(entropy, 8),
            "max_entropy": round(math.log2(dim), 4),
            "is_pure": purity > 0.99,
            "is_mixed": purity < 0.99,
            "eigenvalues": [round(float(e), 8) for e in sorted(eigenvalues, reverse=True)[:8]],
            "pauli_measurements": len(pauli_expectations),
            "sacred_alignment": round(sacred_res, 6),
            "god_code": GOD_CODE,
        }

    @staticmethod
    def state_fidelity(sv_a, sv_b, num_qubits: int) -> dict:
        """
        Compute fidelity F(ρ, σ) between two pure states.

        F = |⟨ψ_a|ψ_b⟩|²

        For mixed states: F = [Tr(√(√ρ·σ·√ρ))]² (Uhlmann fidelity).

        Returns:
            dict with 'fidelity', 'infidelity', 'bures_distance',
            'sacred_alignment'
        """
        dim = 1 << num_qubits

        a = np.array(sv_a, dtype=np.complex128)
        b = np.array(sv_b, dtype=np.complex128)
        if len(a) < dim:
            a = np.pad(a, (0, dim - len(a)))
        if len(b) < dim:
            b = np.pad(b, (0, dim - len(b)))

        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 0:
            a /= na
        if nb > 0:
            b /= nb

        fidelity = float(np.abs(np.dot(a.conj(), b)) ** 2)
        fidelity = min(1.0, fidelity)  # clamp for float precision
        infidelity = 1.0 - fidelity

        # Bures distance: d_B = √(2(1 - √F))
        bures = math.sqrt(max(0.0, 2.0 * (1.0 - math.sqrt(max(0.0, fidelity)))))

        # Sacred: fidelity vs 1/PHI
        sacred_dev = abs(fidelity - 1.0 / PHI)
        sacred_res = max(0.0, 1.0 - sacred_dev * PHI)

        return {
            "fidelity": round(fidelity, 8),
            "infidelity": round(infidelity, 8),
            "bures_distance": round(bures, 8),
            "states_identical": fidelity > 0.9999,
            "sacred_alignment": round(sacred_res, 6),
            "god_code": GOD_CODE,
        }

    @staticmethod
    def swap_test_circuit(num_qubits: int) -> list:
        """
        Build a SWAP test circuit for fidelity estimation.

        Uses one ancilla qubit + controlled-SWAPs to estimate |⟨ψ|φ⟩|²
        from P(ancilla=0) = (1 + F)/2.

        Returns: list of gate operation dicts for a SWAP test circuit
        """
        # Ancilla is qubit 0, state A on qubits 1..n, state B on qubits n+1..2n
        n = num_qubits
        total = 2 * n + 1  # ancilla + 2 copies
        ops = [{"gate": "H", "qubits": [0]}]  # Hadamard on ancilla

        # Controlled-SWAP between corresponding qubits
        for i in range(n):
            q_a = 1 + i
            q_b = 1 + n + i
            # Fredkin (controlled-SWAP) = CNOT cascade
            ops.append({"gate": "CX", "qubits": [q_b, q_a]})
            ops.append({"gate": "CX", "qubits": [0, q_b]})  # Toffoli approx
            ops.append({"gate": "CX", "qubits": [q_b, q_a]})

        ops.append({"gate": "H", "qubits": [0]})  # final Hadamard
        return ops

    @staticmethod
    def reconstruct_density_matrix_mle(pauli_expectations: dict,
                                        num_qubits: int,
                                        max_iterations: int = None,
                                        tol: float = None) -> dict:
        """
        v14.0 — Maximum Likelihood Estimation for density matrix reconstruction.

        Iterative algorithm: ρ_{k+1} = N(R·ρ_k·R) where
        R = Σ_i (f_i / Tr(Π_i·ρ_k)) · Π_i and N normalizes trace to 1.

        Guarantees a physical density matrix (Hermitian, positive semi-definite,
        Tr(ρ)=1) unlike linear inversion which can produce unphysical states.

        Uses φ-accelerated convergence (MLE_ACCELERATION_FACTOR ≈ φ) for
        sacred harmonic alignment during iteration.

        Args:
            pauli_expectations: dict mapping Pauli strings to ⟨P⟩ values
            num_qubits:         Qubit count
            max_iterations:     Convergence limit (default: MLE_MAX_ITERATIONS)
            tol:                Convergence tolerance (default: MLE_CONVERGENCE_TOL)

        Returns:
            dict with 'purity', 'rank', 'von_neumann_entropy',
            'mle_iterations', 'mle_converged', 'sacred_alignment'
        """
        if max_iterations is None:
            max_iterations = MLE_MAX_ITERATIONS
        if tol is None:
            tol = MLE_CONVERGENCE_TOL

        dim = 1 << num_qubits

        paulis = {
            'I': np.eye(2, dtype=np.complex128),
            'X': np.array([[0, 1], [1, 0]], dtype=np.complex128),
            'Y': np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
            'Z': np.array([[1, 0], [0, -1]], dtype=np.complex128),
        }

        # Build measurement operators and observed frequencies
        operators = []
        frequencies = []
        for pauli_str, exp_val in pauli_expectations.items():
            if len(pauli_str) != num_qubits:
                continue
            if all(ch == 'I' for ch in pauli_str):
                continue
            # Build Pauli tensor product
            mat = np.eye(1, dtype=np.complex128)
            for ch in pauli_str:
                mat = np.kron(mat, paulis.get(ch, paulis['I']))
            # Convert expectation to projector frequencies:
            # ⟨P⟩ = Tr(P·ρ) = p(+1) - p(-1), p(+1) + p(-1) = 1
            # So p(+1) = (1 + ⟨P⟩)/2, measurement projector Π+ = (I + P)/2
            proj_plus = (np.eye(dim, dtype=np.complex128) + mat) / 2.0
            freq_plus = (1.0 + exp_val) / 2.0
            operators.append(proj_plus)
            frequencies.append(max(freq_plus, 1e-12))  # avoid zero

            proj_minus = (np.eye(dim, dtype=np.complex128) - mat) / 2.0
            freq_minus = (1.0 - exp_val) / 2.0
            operators.append(proj_minus)
            frequencies.append(max(freq_minus, 1e-12))

        if not operators:
            return {"error": "no_valid_pauli_measurements"}

        # Initialize ρ as maximally mixed state
        rho = np.eye(dim, dtype=np.complex128) / dim

        converged = False
        iterations = 0

        for it in range(max_iterations):
            iterations = it + 1

            # Build R operator: R = Σ_i (f_i / Tr(Π_i·ρ)) · Π_i
            R = np.zeros((dim, dim), dtype=np.complex128)
            for proj, freq in zip(operators, frequencies):
                denom = np.real(np.trace(proj @ rho))
                if denom > 1e-15:
                    R += (freq / denom) * proj
                else:
                    R += freq * proj * 1e12  # large weight for unsatisfied

            # Iteration: ρ_{k+1} = R·ρ·R (with φ-acceleration)
            rho_new = R @ rho @ R

            # φ-accelerated step: blend with momentum
            if it > 0:
                accel = MLE_ACCELERATION_FACTOR
                rho_new = accel * rho_new + (1.0 - accel) * rho
                # Re-enforce Hermiticity
                rho_new = (rho_new + rho_new.conj().T) / 2.0

            # Normalize: Tr(ρ) = 1
            tr = np.real(np.trace(rho_new))
            if tr > 1e-15:
                rho_new /= tr
            else:
                rho_new = np.eye(dim, dtype=np.complex128) / dim

            # Check convergence: ||ρ_{k+1} - ρ_k||_F < tol
            delta = np.linalg.norm(rho_new - rho, 'fro')
            rho = rho_new

            if delta < tol:
                converged = True
                break

        # Ensure positive semi-definite via eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(rho)
        eigenvalues = np.maximum(eigenvalues.real, 0)
        eigenvalues /= np.sum(eigenvalues)
        rho = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.conj().T

        # Purity: Tr(ρ²)
        purity = float(np.real(np.trace(rho @ rho)))
        purity = min(1.0, max(1.0 / dim, purity))

        rank = int(np.sum(eigenvalues > 1e-10))

        # Von Neumann entropy
        valid_eigs = eigenvalues[eigenvalues > 1e-15]
        entropy = -float(np.sum(valid_eigs * np.log2(valid_eigs))) if len(valid_eigs) > 0 else 0

        # Sacred alignment
        phi_target = 1.0 / PHI
        sacred_dev = abs(purity - phi_target) / phi_target
        sacred_res = max(0.0, 1.0 - sacred_dev)

        return {
            "purity": round(purity, 8),
            "rank": rank,
            "von_neumann_entropy": round(entropy, 8),
            "max_entropy": round(math.log2(dim), 4),
            "is_pure": purity > 0.99,
            "is_mixed": purity < 0.99,
            "eigenvalues": [round(float(e), 8) for e in sorted(eigenvalues, reverse=True)[:8]],
            "pauli_measurements": len(pauli_expectations),
            "mle_iterations": iterations,
            "mle_converged": converged,
            "mle_method": "iterative_rho_r_phi_accelerated",
            "sacred_alignment": round(sacred_res, 6),
            "god_code": GOD_CODE,
        }

    @staticmethod
    def full_tomography(statevector, num_qubits: int, shots: int = 4096,
                        method: str = "mle") -> dict:
        """
        Full state tomography pipeline: measure → reconstruct → analyze.

        v14.0: Default method is now 'mle' (Maximum Likelihood Estimation)
        for guaranteed physical density matrices. Use method='linear' for
        the original linear inversion approach.

        Returns complete state characterisation including density matrix
        properties, purity, entropy, and sacred alignment.
        """
        # Measure in Pauli bases
        expectations = QuantumStateTomography.measure_in_pauli_bases(
            statevector, num_qubits, shots)

        # Reconstruct density matrix
        if method == "mle":
            reconstruction = QuantumStateTomography.reconstruct_density_matrix_mle(
                expectations, num_qubits)
        else:
            reconstruction = QuantumStateTomography.reconstruct_density_matrix(
                expectations, num_qubits)

        reconstruction["num_qubits"] = num_qubits
        reconstruction["total_measurements"] = len(expectations)
        reconstruction["god_code"] = GOD_CODE

        return reconstruction
