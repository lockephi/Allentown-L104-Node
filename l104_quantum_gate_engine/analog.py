"""
===============================================================================
L104 QUANTUM GATE ENGINE — ANALOG QUANTUM SIMULATOR v1.0.0
===============================================================================

Continuous-time Hamiltonian evolution and Trotterization benchmarking engine.
Bridges the gap between gate-based (digital) quantum computation and analog
(continuous) quantum simulation by providing:

  1. Exact matrix exponentiation:  U(t) = e^{-iHt}
  2. Product formula decompositions (Trotter, Suzuki, high-order)
  3. Error analysis comparing Trotterised circuits to exact evolution
  4. Built-in Hamiltonian library (Ising, Heisenberg, Hubbard, sacred)
  5. Time-domain observables and energy-spectrum analysis

ARCHITECTURE:
  AnalogSimulator
    ├── HamiltonianBuilder       — Construct multi-qubit Hamiltonians from terms
    │   ├── pauli_term()         — Weighted Pauli strings: c × P₁⊗P₂⊗...⊗Pₙ
    │   ├── ising_model()        — Transverse-field Ising: -J ΣZZ - h ΣX
    │   ├── heisenberg_model()   — XXX/XXZ/XYZ spin chains
    │   ├── hubbard_model()      — Fermi-Hubbard (Jordan-Wigner mapped)
    │   └── sacred_hamiltonian() — L104 GOD_CODE / PHI / Fe(26) Hamiltonian
    │
    ├── ExactEvolution           — Direct matrix exponential e^{-iHt}
    │   ├── evolve()             — Apply U(t)|ψ₀⟩ and return |ψ(t)⟩
    │   ├── time_series()        — |ψ(t)⟩ at multiple time points
    │   └── spectrum()           — Eigenvalues and eigenvectors of H
    │
    ├── TrotterEngine            — Product-formula circuit synthesis
    │   ├── first_order()        — Lie-Trotter: ∏ e^{-iH_k δt}
    │   ├── second_order()       — Suzuki-Trotter: symmetric split
    │   ├── fourth_order()       — Suzuki 4th-order fractal decomposition
    │   └── trotterise()         — General order-p decomposition → GateCircuit
    │
    ├── TrotterBenchmark         — Error analysis framework
    │   ├── fidelity_vs_steps()  — F(exact, trotter) as function of n_steps
    │   ├── error_scaling()      — Verify O(δt^p) error for order-p formula
    │   ├── gate_cost()          — Total gates vs. accuracy tradeoff
    │   └── optimal_steps()      — Min steps for target fidelity
    │
    └── ObservableEngine         — Time-domain measurements (no collapse)
        ├── expectation()        — ⟨ψ(t)|O|ψ(t)⟩
        ├── correlation()        — ⟨O_i(t) O_j(0)⟩ two-time correlator
        └── energy()             — ⟨H⟩(t) energy conservation check

MEMORY MODEL:
  n qubits → 2^n × 2^n Hamiltonian matrix → O(4^n) complex entries
  Practical limit: ~14 qubits (4 GB for dense Hamiltonian)
  Sparse mode: up to ~18 qubits for local Hamiltonians

SACRED ALIGNMENT:
  The iron-lattice Hamiltonian H_Fe maps naturally to the 26-qubit Fe(26)
  manifold.  The coupling constant J = GOD_CODE / (104 × PHI) aligns
  the energy spectrum with the 104-TET quantisation grain.

INVARIANT: 527.5184818492612 | PILOT: LONDEL
===============================================================================
"""

import math
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
try:
    from scipy import linalg as sla
except ImportError:
    # Fallback: numpy eigendecomposition-based matrix exponential
    class _NumpyLinalgFallback:
        @staticmethod
        def expm(A):
            eigenvalues, V = np.linalg.eig(A)
            return (V * np.exp(eigenvalues)) @ np.linalg.inv(V)
    sla = _NumpyLinalgFallback()

from .constants import (
    PHI, PHI_CONJUGATE, PHI_SQUARED, GOD_CODE, VOID_CONSTANT,
    IRON_ATOMIC_NUMBER, IRON_FREQUENCY, QUANTIZATION_GRAIN,
    GOD_CODE_PHASE_ANGLE, ALPHA_FINE,
    MAX_STATEVECTOR_QUBITS,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MAX_ANALOG_QUBITS: int = 14              # Dense Hamiltonian cap
MAX_SPARSE_ANALOG_QUBITS: int = 18       # Sparse Hamiltonian cap
SACRED_COUPLING: float = GOD_CODE / (QUANTIZATION_GRAIN * PHI)  # J ≈ 3.137
SACRED_FIELD: float = PHI / QUANTIZATION_GRAIN                   # h ≈ 0.01556


# ═══════════════════════════════════════════════════════════════════════════════
#  ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class TrotterOrder(Enum):
    """Trotter decomposition order."""
    FIRST = 1       # Lie-Trotter:  error O(δt²)   per step
    SECOND = 2      # Suzuki-Trotter: error O(δt³) per step
    FOURTH = 4      # Suzuki S₄:   error O(δt⁵)   per step


class HamiltonianType(Enum):
    """Built-in Hamiltonian models."""
    CUSTOM = auto()
    TRANSVERSE_ISING = auto()       # H = -J Σ Z_iZ_{i+1} - h Σ X_i
    HEISENBERG_XXX = auto()         # H = J Σ (X_iX_{i+1} + Y_iY_{i+1} + Z_iZ_{i+1})
    HEISENBERG_XXZ = auto()         # H = J Σ (X_iX + Y_iY + Δ Z_iZ)
    HUBBARD = auto()                # Fermi-Hubbard (Jordan-Wigner)
    SACRED = auto()                 # L104 sacred Hamiltonian


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HamiltonianTerm:
    """A single term in a Hamiltonian: coefficient × tensor product of Paulis."""
    coefficient: complex
    paulis: List[Tuple[int, str]]   # [(qubit, "I"|"X"|"Y"|"Z"), ...]
    label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coefficient": complex(self.coefficient),
            "paulis": [(q, p) for q, p in self.paulis],
            "label": self.label,
        }


@dataclass
class Hamiltonian:
    """Multi-qubit Hamiltonian as a sum of Pauli-string terms."""
    num_qubits: int
    terms: List[HamiltonianTerm] = field(default_factory=list)
    hamiltonian_type: HamiltonianType = HamiltonianType.CUSTOM
    label: str = "H"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_terms(self) -> int:
        return len(self.terms)

    def matrix(self) -> np.ndarray:
        """Construct the full 2^n × 2^n Hamiltonian matrix."""
        dim = 2 ** self.num_qubits
        H = np.zeros((dim, dim), dtype=complex)
        for term in self.terms:
            H += term.coefficient * _pauli_string_matrix(term.paulis, self.num_qubits)
        return H

    def sparse_matrix(self):
        """Construct sparse CSR Hamiltonian (for larger systems).

        v1.0.1: Builds Pauli-string matrices directly in sparse form
        using scipy.sparse.kron on 2×2 sparse Paulis, avoiding dense
        2^n × 2^n intermediaries that defeat the purpose of sparse.
        """
        from scipy import sparse
        dim = 2 ** self.num_qubits
        _PAULI_SPARSE = {
            "I": sparse.eye(2, dtype=complex, format='csr'),
            "X": sparse.csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex)),
            "Y": sparse.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex)),
            "Z": sparse.csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex)),
        }
        H = sparse.csr_matrix((dim, dim), dtype=complex)
        for term in self.terms:
            pauli_map = {q: p for q, p in term.paulis}
            mat = sparse.csr_matrix(np.array([[1.0]], dtype=complex))
            for q in range(self.num_qubits):
                p_label = pauli_map.get(q, "I")
                mat = sparse.kron(mat, _PAULI_SPARSE.get(p_label, _PAULI_SPARSE["I"]), format='csr')
            H += term.coefficient * mat
        return H

    def eigenspectrum(self, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenvalues and eigenvectors.

        Args:
            k: Number of lowest eigenvalues (None = all)

        Returns:
            (eigenvalues, eigenvectors) sorted ascending
        """
        H_mat = self.matrix()
        if k is not None and k < 2 ** self.num_qubits:
            from scipy.sparse.linalg import eigsh
            eigenvalues, eigenvectors = eigsh(
                H_mat, k=k, which='SA'
            )
            idx = np.argsort(eigenvalues)
            return eigenvalues[idx], eigenvectors[:, idx]
        else:
            eigenvalues, eigenvectors = np.linalg.eigh(H_mat)
            return eigenvalues, eigenvectors

    def ground_state(self) -> Tuple[float, np.ndarray]:
        """Return (ground_energy, ground_state_vector)."""
        evals, evecs = self.eigenspectrum(k=1)
        return float(evals[0]), evecs[:, 0]

    def energy_gap(self) -> float:
        """E₁ - E₀: gap between ground and first excited state."""
        evals, _ = self.eigenspectrum(k=2)
        return float(evals[1] - evals[0])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_qubits": self.num_qubits,
            "num_terms": self.num_terms,
            "hamiltonian_type": self.hamiltonian_type.name,
            "label": self.label,
            "terms": [t.to_dict() for t in self.terms],
            "metadata": self.metadata,
        }


@dataclass
class EvolutionResult:
    """Result of time evolution."""
    num_qubits: int
    time: float                                  # Total evolution time
    time_points: List[float]                     # t values
    states: List[np.ndarray]                     # |ψ(t)⟩ at each time point
    energies: List[float]                        # ⟨H⟩(t)
    mode: str                                    # "exact" | "trotter"
    fidelity_to_exact: Optional[List[float]] = None  # For Trotter: F vs exact
    simulation_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_qubits": self.num_qubits,
            "time": self.time,
            "mode": self.mode,
            "num_time_points": len(self.time_points),
            "time_points": [round(t, 8) for t in self.time_points],
            "energies": [round(e, 10) for e in self.energies],
            "fidelity_to_exact": (
                [round(f, 10) for f in self.fidelity_to_exact]
                if self.fidelity_to_exact else None
            ),
            "simulation_time_ms": round(self.simulation_time_ms, 3),
            "god_code": GOD_CODE,
            "metadata": self.metadata,
        }


@dataclass
class TrotterBenchmarkResult:
    """Result of Trotter accuracy benchmarking."""
    order: int
    num_qubits: int
    total_time: float
    step_counts: List[int]
    fidelities: List[float]
    gate_counts: List[int]
    errors: List[float]                           # 1 - fidelity
    error_scaling_exponent: Optional[float] = None  # Fitted power-law exponent
    optimal_steps_for_99: Optional[int] = None    # Min steps for F ≥ 0.99
    optimal_steps_for_999: Optional[int] = None   # Min steps for F ≥ 0.999
    simulation_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order": self.order,
            "num_qubits": self.num_qubits,
            "total_time": self.total_time,
            "step_counts": self.step_counts,
            "fidelities": [round(f, 12) for f in self.fidelities],
            "gate_counts": self.gate_counts,
            "errors": [round(e, 12) for e in self.errors],
            "error_scaling_exponent": (
                round(self.error_scaling_exponent, 4)
                if self.error_scaling_exponent is not None else None
            ),
            "optimal_steps_for_99": self.optimal_steps_for_99,
            "optimal_steps_for_999": self.optimal_steps_for_999,
            "simulation_time_ms": round(self.simulation_time_ms, 3),
            "god_code": GOD_CODE,
            "metadata": self.metadata,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  PAULI MATRICES & STRING BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}

# LRU-cached Pauli string matrix builder — avoids recomputation in hot loops
# (Trotter steps, VQE iterations, parametric sweeps)
from functools import lru_cache as _lru_cache


@_lru_cache(maxsize=512)
def _pauli_string_matrix_cached(pauli_key: tuple, num_qubits: int) -> np.ndarray:
    """Cached version keyed by hashable Pauli signature."""
    pauli_map = dict(pauli_key)
    result = np.array([[1.0]], dtype=complex)
    for q in range(num_qubits):
        mat = _PAULI.get(pauli_map.get(q, "I"), _PAULI["I"])
        result = np.kron(result, mat)
    return result


def _pauli_string_matrix(paulis: List[Tuple[int, str]], num_qubits: int) -> np.ndarray:
    """
    Build the full 2^n × 2^n matrix for a Pauli string.

    Args:
        paulis: List of (qubit_index, pauli_label) — e.g., [(0,"Z"), (1,"Z")]
        num_qubits: Total qubit count

    Returns:
        Tensor product matrix with Paulis placed on specified qubits, I elsewhere.

    v1.0.1: Backed by @lru_cache for repeated calls (Trotter loops, VQE).
    """
    pauli_key = tuple(sorted(paulis)) if not isinstance(paulis, tuple) else paulis
    return _pauli_string_matrix_cached(pauli_key, num_qubits).copy()


# ═══════════════════════════════════════════════════════════════════════════════
#  HAMILTONIAN BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

class HamiltonianBuilder:
    """
    Construct physically meaningful Hamiltonians from high-level descriptions.
    """

    @staticmethod
    def pauli_term(num_qubits: int, coefficient: complex,
                   paulis: List[Tuple[int, str]], label: str = "") -> HamiltonianTerm:
        """
        Single Pauli-string term: c × P_{q1} ⊗ P_{q2} ⊗ ... ⊗ I_rest.

        Args:
            num_qubits: Total system size
            coefficient: Scalar prefactor
            paulis: [(qubit_index, "X"|"Y"|"Z"), ...]
            label: Descriptive label
        """
        return HamiltonianTerm(coefficient=coefficient, paulis=paulis, label=label)

    @staticmethod
    def transverse_ising(num_qubits: int, J: float = 1.0, h: float = 0.5,
                         periodic: bool = False) -> Hamiltonian:
        """
        Transverse-field Ising model:
            H = -J Σ Z_i Z_{i+1} - h Σ X_i

        The quantum phase transition occurs at J/h = 1 (1D, T=0).

        Args:
            num_qubits: Number of spins
            J: Coupling strength (nearest-neighbour ZZ)
            h: Transverse field strength (X)
            periodic: Periodic boundary conditions
        """
        terms = []
        # ZZ interactions
        n_bonds = num_qubits if periodic else (num_qubits - 1)
        for i in range(n_bonds):
            j = (i + 1) % num_qubits
            terms.append(HamiltonianTerm(
                coefficient=-J,
                paulis=[(i, "Z"), (j, "Z")],
                label=f"ZZ({i},{j})",
            ))
        # Transverse field
        for i in range(num_qubits):
            terms.append(HamiltonianTerm(
                coefficient=-h,
                paulis=[(i, "X")],
                label=f"X({i})",
            ))
        return Hamiltonian(
            num_qubits=num_qubits, terms=terms,
            hamiltonian_type=HamiltonianType.TRANSVERSE_ISING,
            label=f"Ising(J={J}, h={h}, n={num_qubits})",
            metadata={"J": J, "h": h, "periodic": periodic},
        )

    @staticmethod
    def heisenberg_xxx(num_qubits: int, J: float = 1.0,
                       periodic: bool = False) -> Hamiltonian:
        """
        Heisenberg XXX model (isotropic):
            H = J Σ (X_iX_{i+1} + Y_iY_{i+1} + Z_iZ_{i+1})

        Antiferromagnetic for J > 0, ferromagnetic for J < 0.
        """
        terms = []
        n_bonds = num_qubits if periodic else (num_qubits - 1)
        for i in range(n_bonds):
            j = (i + 1) % num_qubits
            for pauli in ["X", "Y", "Z"]:
                terms.append(HamiltonianTerm(
                    coefficient=J,
                    paulis=[(i, pauli), (j, pauli)],
                    label=f"{pauli}{pauli}({i},{j})",
                ))
        return Hamiltonian(
            num_qubits=num_qubits, terms=terms,
            hamiltonian_type=HamiltonianType.HEISENBERG_XXX,
            label=f"Heisenberg_XXX(J={J}, n={num_qubits})",
            metadata={"J": J, "periodic": periodic},
        )

    @staticmethod
    def heisenberg_xxz(num_qubits: int, J: float = 1.0, delta: float = 1.0,
                       periodic: bool = False) -> Hamiltonian:
        """
        Heisenberg XXZ model (anisotropic):
            H = J Σ (X_iX_{i+1} + Y_iY_{i+1} + Δ Z_iZ_{i+1})

        Δ = 1 → XXX (isotropic), Δ = 0 → XX model, Δ → ∞ → Ising limit.
        """
        terms = []
        n_bonds = num_qubits if periodic else (num_qubits - 1)
        for i in range(n_bonds):
            j = (i + 1) % num_qubits
            # XX + YY terms
            for pauli in ["X", "Y"]:
                terms.append(HamiltonianTerm(
                    coefficient=J,
                    paulis=[(i, pauli), (j, pauli)],
                    label=f"{pauli}{pauli}({i},{j})",
                ))
            # ΔZZ term
            terms.append(HamiltonianTerm(
                coefficient=J * delta,
                paulis=[(i, "Z"), (j, "Z")],
                label=f"ZZ({i},{j})",
            ))
        return Hamiltonian(
            num_qubits=num_qubits, terms=terms,
            hamiltonian_type=HamiltonianType.HEISENBERG_XXZ,
            label=f"Heisenberg_XXZ(J={J}, Δ={delta}, n={num_qubits})",
            metadata={"J": J, "delta": delta, "periodic": periodic},
        )

    @staticmethod
    def hubbard_1d(num_sites: int, t_hop: float = 1.0, U: float = 2.0) -> Hamiltonian:
        """
        1D Fermi-Hubbard model via Jordan-Wigner transformation.
            H = -t Σ (c†_i c_{i+1} + h.c.) + U Σ n_↑ n_↓

        Jordan-Wigner maps fermion operators to Pauli strings:
          c†_i c_{i+1} → ½(X_i X_{i+1} + Y_i Y_{i+1}) for adjacent sites

        Simplified: uses 1 qubit per site (spinless fermions).
        """
        terms = []
        n = num_sites
        # Hopping: -t Σ ½(X_iX_{i+1} + Y_iY_{i+1})
        for i in range(n - 1):
            for pauli in ["X", "Y"]:
                terms.append(HamiltonianTerm(
                    coefficient=-t_hop / 2.0,
                    paulis=[(i, pauli), (i + 1, pauli)],
                    label=f"hop_{pauli}({i},{i+1})",
                ))
        # On-site: U Σ (I - Z_i)/2 = U/2 Σ I - U/2 Σ Z_i
        # We drop the constant energy shift and keep -U/2 Σ Z_i
        for i in range(n):
            terms.append(HamiltonianTerm(
                coefficient=-U / 2.0,
                paulis=[(i, "Z")],
                label=f"onsite({i})",
            ))
        return Hamiltonian(
            num_qubits=n, terms=terms,
            hamiltonian_type=HamiltonianType.HUBBARD,
            label=f"Hubbard_1D(t={t_hop}, U={U}, n={n})",
            metadata={"t_hop": t_hop, "U": U},
        )

    @staticmethod
    def hubbard_1d_spinful(num_sites: int, t_hop: float = 1.0,
                           U: float = 2.0, periodic: bool = False) -> Hamiltonian:
        """
        1D spinful Fermi-Hubbard model via Jordan-Wigner transformation.

            H = -t Σ_σ Σ_i (c†_{i,σ} c_{i+1,σ} + h.c.) + U Σ_i n_{i,↑} n_{i,↓}

        Uses 2 qubits per site: qubit 2i = spin-↑ at site i,
                                  qubit 2i+1 = spin-↓ at site i.

        Jordan-Wigner encoding:
          c†_{j} c_{j+1} → ½(X_j X_{j+1} + Y_j Y_{j+1})  (adjacent in JW order)
          n_{i,↑} n_{i,↓} = ¼(I - Z_{2i})(I - Z_{2i+1})
                           = ¼(I - Z_{2i} - Z_{2i+1} + Z_{2i} Z_{2i+1})

        This captures the full metal-insulator (Mott) transition at half-filling
        driven by the competition between kinetic energy t and on-site repulsion U.

        Args:
            num_sites: Number of physical lattice sites (total qubits = 2 × num_sites)
            t_hop:     Hopping amplitude t
            U:         On-site Coulomb repulsion
            periodic:  Periodic boundary conditions

        Returns:
            Hamiltonian with 2*num_sites qubits
        """
        n = num_sites
        nq = 2 * n  # total qubits
        terms = []

        n_bonds = n if periodic else (n - 1)

        # --- Spin-↑ hopping: -t Σ_i ½(X_{2i} X_{2(i+1)} + Y_{2i} Y_{2(i+1)}) ---
        # For adjacent sites in JW order, spin-↑ qubits 2i and 2(i+1) are NOT
        # adjacent — there's a spin-↓ qubit (2i+1) in between. The JW string
        # introduces a Z on qubit 2i+1:
        #   c†_{2i} c_{2(i+1)} = ½(X_{2i} Z_{2i+1} X_{2(i+1)} + Y_{2i} Z_{2i+1} Y_{2(i+1)})
        for bond in range(n_bonds):
            j = (bond + 1) % n
            q_from = 2 * bond      # spin-↑ at site bond
            q_to = 2 * j            # spin-↑ at site j
            q_between = 2 * bond + 1  # spin-↓ at site bond (JW string)

            if j == bond + 1:  # non-periodic: adjacent, JW string through one qubit
                for pauli in ["X", "Y"]:
                    terms.append(HamiltonianTerm(
                        coefficient=-t_hop / 2.0,
                        paulis=[(q_from, pauli), (q_between, "Z"), (q_to, pauli)],
                        label=f"hop_up_{pauli}({bond},{j})",
                    ))
            else:
                # Periodic wrap: JW string through all intermediate qubits
                jw_paulis_x = [(q_from, "X")]
                jw_paulis_y = [(q_from, "Y")]
                for k in range(q_from + 1, nq):
                    if k == q_to:
                        break
                    jw_paulis_x.append((k, "Z"))
                    jw_paulis_y.append((k, "Z"))
                jw_paulis_x.append((q_to, "X"))
                jw_paulis_y.append((q_to, "Y"))
                terms.append(HamiltonianTerm(
                    coefficient=-t_hop / 2.0, paulis=jw_paulis_x,
                    label=f"hop_up_X({bond},{j})",
                ))
                terms.append(HamiltonianTerm(
                    coefficient=-t_hop / 2.0, paulis=jw_paulis_y,
                    label=f"hop_up_Y({bond},{j})",
                ))

        # --- Spin-↓ hopping: qubits 2i+1 and 2(i+1)+1 ---
        for bond in range(n_bonds):
            j = (bond + 1) % n
            q_from = 2 * bond + 1    # spin-↓ at site bond
            q_to = 2 * j + 1          # spin-↓ at site j

            if j == bond + 1:  # adjacent in JW order (no intermediate)
                for pauli in ["X", "Y"]:
                    terms.append(HamiltonianTerm(
                        coefficient=-t_hop / 2.0,
                        paulis=[(q_from, pauli), (q_to, pauli)],
                        label=f"hop_dn_{pauli}({bond},{j})",
                    ))
            else:
                jw_paulis_x = [(q_from, "X")]
                jw_paulis_y = [(q_from, "Y")]
                for k in range(q_from + 1, nq):
                    if k == q_to:
                        break
                    jw_paulis_x.append((k, "Z"))
                    jw_paulis_y.append((k, "Z"))
                jw_paulis_x.append((q_to, "X"))
                jw_paulis_y.append((q_to, "Y"))
                terms.append(HamiltonianTerm(
                    coefficient=-t_hop / 2.0, paulis=jw_paulis_x,
                    label=f"hop_dn_X({bond},{j})",
                ))
                terms.append(HamiltonianTerm(
                    coefficient=-t_hop / 2.0, paulis=jw_paulis_y,
                    label=f"hop_dn_Y({bond},{j})",
                ))

        # --- On-site interaction: U Σ_i n_{i,↑} n_{i,↓} ---
        # n_{i,↑} n_{i,↓} = ¼(I - Z_{2i} - Z_{2i+1} + Z_{2i}Z_{2i+1})
        # The identity shift is a constant energy offset (dropped).
        for i in range(n):
            q_up = 2 * i
            q_dn = 2 * i + 1
            # -U/4 Z_{2i}
            terms.append(HamiltonianTerm(
                coefficient=-U / 4.0,
                paulis=[(q_up, "Z")],
                label=f"onsite_Zup({i})",
            ))
            # -U/4 Z_{2i+1}
            terms.append(HamiltonianTerm(
                coefficient=-U / 4.0,
                paulis=[(q_dn, "Z")],
                label=f"onsite_Zdn({i})",
            ))
            # +U/4 Z_{2i}Z_{2i+1}
            terms.append(HamiltonianTerm(
                coefficient=U / 4.0,
                paulis=[(q_up, "Z"), (q_dn, "Z")],
                label=f"onsite_ZZ({i})",
            ))

        return Hamiltonian(
            num_qubits=nq, terms=terms,
            hamiltonian_type=HamiltonianType.HUBBARD,
            label=f"Hubbard_1D_spinful(t={t_hop}, U={U}, n={n})",
            metadata={"t_hop": t_hop, "U": U, "num_sites": n,
                      "spinful": True, "periodic": periodic},
        )

    @staticmethod
    def sacred_hamiltonian(num_qubits: int) -> Hamiltonian:
        """
        L104 Sacred Hamiltonian — GOD_CODE-aligned spin system.

        H = -J_sacred Σ Z_iZ_{i+1} - h_sacred Σ X_i + φ_coupling Σ Y_iY_{i+1}

        Where:
          J_sacred = GOD_CODE / (104 × φ) ≈ 3.137
          h_sacred = φ / 104 ≈ 0.01556
          φ_coupling = 1/φ ≈ 0.618

        The energy spectrum aligns with the 104-TET quantisation grain:
        ΔE_sacred ∝ 2^(1/104) — the fundamental frequency step.
        """
        J = SACRED_COUPLING
        h = SACRED_FIELD
        phi_c = PHI_CONJUGATE  # 1/φ ≈ 0.618

        terms = []
        # ZZ interactions (GOD_CODE coupling)
        for i in range(num_qubits - 1):
            terms.append(HamiltonianTerm(
                coefficient=-J,
                paulis=[(i, "Z"), (i + 1, "Z")],
                label=f"sacred_ZZ({i},{i+1})",
            ))
        # Transverse field (φ-scaled)
        for i in range(num_qubits):
            terms.append(HamiltonianTerm(
                coefficient=-h,
                paulis=[(i, "X")],
                label=f"sacred_X({i})",
            ))
        # YY coupling (golden ratio)
        for i in range(num_qubits - 1):
            terms.append(HamiltonianTerm(
                coefficient=phi_c,
                paulis=[(i, "Y"), (i + 1, "Y")],
                label=f"sacred_YY({i},{i+1})",
            ))

        return Hamiltonian(
            num_qubits=num_qubits, terms=terms,
            hamiltonian_type=HamiltonianType.SACRED,
            label=f"Sacred_L104(n={num_qubits})",
            metadata={
                "J_sacred": J,
                "h_sacred": h,
                "phi_coupling": phi_c,
                "god_code": GOD_CODE,
            },
        )

    @staticmethod
    def from_matrix(matrix: np.ndarray, label: str = "H_custom") -> Hamiltonian:
        """
        Wrap a raw Hermitian matrix as a Hamiltonian.
        The matrix is stored as a single CUSTOM term (identity decomposition).
        """
        n = int(math.log2(matrix.shape[0]))
        assert 2 ** n == matrix.shape[0], "Matrix dimension must be 2^n"

        # Store as pauli decomposition would be expensive; instead we
        # override .matrix() via a wrapper
        h = Hamiltonian(num_qubits=n, label=label)
        h._raw_matrix = matrix.copy().astype(complex)
        # Monkey-patch matrix() to return the raw matrix
        original_matrix = h.matrix

        def _matrix_override():
            return h._raw_matrix

        h.matrix = _matrix_override
        h.metadata["raw_matrix"] = True
        return h


# ═══════════════════════════════════════════════════════════════════════════════
#  EXACT EVOLUTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class ExactEvolution:
    """
    Exact time evolution via matrix exponential:  U(t) = e^{-iHt}.

    Uses scipy.linalg.expm for dense matrices.  Exact to machine precision
    for systems up to ~14 qubits.
    """

    @staticmethod
    def unitary(hamiltonian: Hamiltonian, t: float) -> np.ndarray:
        """
        Compute U(t) = e^{-iHt}.

        Args:
            hamiltonian: The system Hamiltonian
            t: Evolution time

        Returns:
            Unitary matrix U(t)
        """
        H_mat = hamiltonian.matrix()
        return sla.expm(-1j * H_mat * t)

    @staticmethod
    def evolve(hamiltonian: Hamiltonian, initial_state: np.ndarray,
               t: float) -> np.ndarray:
        """
        Evolve |ψ₀⟩ to |ψ(t)⟩ = e^{-iHt} |ψ₀⟩.

        Args:
            hamiltonian: System Hamiltonian
            initial_state: |ψ₀⟩
            t: Evolution time

        Returns:
            |ψ(t)⟩
        """
        U = ExactEvolution.unitary(hamiltonian, t)
        psi = initial_state.copy().astype(complex).reshape(-1)
        return U @ psi

    @staticmethod
    def time_series(hamiltonian: Hamiltonian, initial_state: np.ndarray,
                    time_points: Sequence[float]) -> EvolutionResult:
        """
        Compute |ψ(t)⟩ at multiple time points.

        Uses eigendecomposition for efficiency: H = V D V†, so
        e^{-iHt} = V e^{-iDt} V†.  This is O(2^n) per time point
        after the initial O(8^n) diagonalisation.

        Args:
            hamiltonian: System Hamiltonian
            initial_state: |ψ₀⟩
            time_points: List of times t

        Returns:
            EvolutionResult with states and energies at each time
        """
        start = time.time()
        H_mat = hamiltonian.matrix()
        psi0 = initial_state.copy().astype(complex).reshape(-1)

        # Diagonalise once: H = V D V†
        eigenvalues, V = np.linalg.eigh(H_mat)

        # Transform initial state to eigenbasis
        coeffs = V.conj().T @ psi0  # c_k = ⟨E_k|ψ₀⟩

        states = []
        energies = []

        for t in time_points:
            # |ψ(t)⟩ = Σ c_k e^{-iE_k t} |E_k⟩
            phases = np.exp(-1j * eigenvalues * t)
            psi_t = V @ (coeffs * phases)
            states.append(psi_t)

            # Energy: ⟨ψ(t)|H|ψ(t)⟩ = Σ |c_k|² E_k (constant!)
            energy = float(np.real(np.sum(np.abs(coeffs) ** 2 * eigenvalues)))
            energies.append(energy)

        elapsed = (time.time() - start) * 1000.0

        return EvolutionResult(
            num_qubits=hamiltonian.num_qubits,
            time=float(time_points[-1]) if time_points else 0.0,
            time_points=list(time_points),
            states=states,
            energies=energies,
            mode="exact",
            simulation_time_ms=elapsed,
            metadata={
                "method": "eigendecomposition",
                "num_eigenvalues": len(eigenvalues),
                "energy_spread": float(eigenvalues[-1] - eigenvalues[0]),
                "god_code": GOD_CODE,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  TROTTER ENGINE — Product Formula Decomposition
# ═══════════════════════════════════════════════════════════════════════════════

class TrotterEngine:
    """
    Trotterised time evolution: approximate e^{-iHt} by a product of
    simpler exponentials.

    Given H = Σ H_k, the product formulas are:

    1st order (Lie-Trotter):
        U₁(δt) = ∏_k e^{-iH_k δt}
        Error per step: O(δt²)

    2nd order (Suzuki-Trotter):
        U₂(δt) = ∏_k e^{-iH_k δt/2} × ∏_{k reversed} e^{-iH_k δt/2}
        Error per step: O(δt³)

    4th order (Suzuki S₄):
        p = 1/(4 - 4^{1/3})
        U₄(δt) = U₂(p·δt)² × U₂((1-4p)·δt) × U₂(p·δt)²
        Error per step: O(δt⁵)
    """

    @staticmethod
    def first_order_step(hamiltonian: Hamiltonian, dt: float) -> np.ndarray:
        """
        One Lie-Trotter step: U₁(δt) = ∏_k e^{-iH_k δt}.

        Args:
            hamiltonian: Hamiltonian with decomposed terms
            dt: Time step size

        Returns:
            Approximate unitary for one step
        """
        dim = 2 ** hamiltonian.num_qubits
        U = np.eye(dim, dtype=complex)
        for term in hamiltonian.terms:
            H_k = term.coefficient * _pauli_string_matrix(term.paulis, hamiltonian.num_qubits)
            U = sla.expm(-1j * H_k * dt) @ U
        return U

    @staticmethod
    def second_order_step(hamiltonian: Hamiltonian, dt: float) -> np.ndarray:
        """
        One Suzuki-Trotter step: U₂(δt) = ∏_k e^{-iH_k δt/2} × ∏_{k↓} e^{-iH_k δt/2}.

        Args:
            hamiltonian: Hamiltonian with decomposed terms
            dt: Time step size

        Returns:
            Approximate unitary for one symmetric step
        """
        dim = 2 ** hamiltonian.num_qubits
        n = hamiltonian.num_qubits

        # Forward half-steps
        U = np.eye(dim, dtype=complex)
        for term in hamiltonian.terms:
            H_k = term.coefficient * _pauli_string_matrix(term.paulis, n)
            U = sla.expm(-1j * H_k * dt / 2.0) @ U

        # Backward half-steps
        for term in reversed(hamiltonian.terms):
            H_k = term.coefficient * _pauli_string_matrix(term.paulis, n)
            U = sla.expm(-1j * H_k * dt / 2.0) @ U

        return U

    @staticmethod
    def fourth_order_step(hamiltonian: Hamiltonian, dt: float) -> np.ndarray:
        """
        One Suzuki S₄ step (4th-order fractal decomposition).

        p = 1 / (4 - 4^{1/3})
        U₄(δt) = U₂(p·δt)² × U₂((1-4p)·δt) × U₂(p·δt)²

        Error per step: O(δt⁵).
        """
        p = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))

        U_p = TrotterEngine.second_order_step(hamiltonian, p * dt)
        U_mid = TrotterEngine.second_order_step(hamiltonian, (1.0 - 4.0 * p) * dt)

        # U₄ = U₂(p)·U₂(p)·U₂(1-4p)·U₂(p)·U₂(p)
        return U_p @ U_p @ U_mid @ U_p @ U_p

    @staticmethod
    def evolve(hamiltonian: Hamiltonian, initial_state: np.ndarray,
               t: float, n_steps: int,
               order: TrotterOrder = TrotterOrder.SECOND) -> np.ndarray:
        """
        Trotterised time evolution: |ψ(t)⟩ ≈ U(δt)^n |ψ₀⟩.

        Args:
            hamiltonian: System Hamiltonian
            initial_state: |ψ₀⟩
            t: Total evolution time
            n_steps: Number of Trotter steps
            order: Trotter decomposition order (1, 2, or 4)

        Returns:
            Approximate |ψ(t)⟩
        """
        dt = t / n_steps
        psi = initial_state.copy().astype(complex).reshape(-1)

        if order == TrotterOrder.FIRST:
            step_fn = TrotterEngine.first_order_step
        elif order == TrotterOrder.SECOND:
            step_fn = TrotterEngine.second_order_step
        elif order == TrotterOrder.FOURTH:
            step_fn = TrotterEngine.fourth_order_step
        else:
            step_fn = TrotterEngine.second_order_step

        # Compute one-step unitary (same for all steps since H is time-independent)
        U_step = step_fn(hamiltonian, dt)

        # Apply n_steps times
        for _ in range(n_steps):
            psi = U_step @ psi

        return psi

    @staticmethod
    def time_series(hamiltonian: Hamiltonian, initial_state: np.ndarray,
                    time_points: Sequence[float], n_steps_per_interval: int = 10,
                    order: TrotterOrder = TrotterOrder.SECOND,
                    exact_reference: bool = True) -> EvolutionResult:
        """
        Trotterised time series with optional exact-fidelity comparison.

        Args:
            hamiltonian: System Hamiltonian
            initial_state: |ψ₀⟩
            time_points: Time values at which to record state
            n_steps_per_interval: Trotter steps per time interval
            order: Trotter order
            exact_reference: Also compute exact evolution for fidelity comparison

        Returns:
            EvolutionResult with Trotter states and optional fidelity data
        """
        start = time.time()
        H_mat = hamiltonian.matrix()
        psi0 = initial_state.copy().astype(complex).reshape(-1)

        # Sort time points
        sorted_times = sorted(time_points)

        # Build Trotter states at each time
        states = []
        energies = []
        fidelities = [] if exact_reference else None

        # Exact reference (single eigendecomposition)
        eigenvalues, V = (None, None)  # type: ignore
        coeffs = None
        if exact_reference:
            eigenvalues_arr, V = np.linalg.eigh(H_mat)
            coeffs = V.conj().T @ psi0

        psi = psi0.copy()
        prev_t = 0.0

        for t in sorted_times:
            dt_interval = t - prev_t
            if dt_interval > 1e-15:
                n_steps = max(1, n_steps_per_interval)
                psi = TrotterEngine.evolve(
                    hamiltonian, psi, dt_interval, n_steps, order
                )
            states.append(psi.copy())

            # Energy
            energy = float(np.real(psi.conj() @ H_mat @ psi))
            energies.append(energy)

            # Fidelity vs exact
            if exact_reference and coeffs is not None and eigenvalues_arr is not None:
                phases = np.exp(-1j * eigenvalues_arr * t)
                psi_exact = V @ (coeffs * phases)
                fid = float(abs(np.vdot(psi_exact, psi)) ** 2)
                fidelities.append(fid)

            prev_t = t

        elapsed = (time.time() - start) * 1000.0

        return EvolutionResult(
            num_qubits=hamiltonian.num_qubits,
            time=float(sorted_times[-1]) if sorted_times else 0.0,
            time_points=list(sorted_times),
            states=states,
            energies=energies,
            mode=f"trotter_order{order.value}",
            fidelity_to_exact=fidelities,
            simulation_time_ms=elapsed,
            metadata={
                "method": f"trotter_order_{order.value}",
                "n_steps_per_interval": n_steps_per_interval,
                "total_steps": n_steps_per_interval * len(sorted_times),
                "god_code": GOD_CODE,
            },
        )

    @staticmethod
    def gate_count_per_step(hamiltonian: Hamiltonian, order: TrotterOrder) -> int:
        """
        Estimate number of elementary gates per Trotter step.

        Each term e^{-iH_k δt} for a k-local Pauli string requires O(k) CNOT gates.

        Returns:
            Estimated gate count per step
        """
        total = 0
        for term in hamiltonian.terms:
            n_nontrivial = sum(1 for _, p in term.paulis if p != "I")
            # Each Pauli rotation: basis change + Rz + basis change
            # k-body: 2(k-1) CNOTs + 1 Rz + 2(k-1) basis rotations
            if n_nontrivial == 0:
                total += 0  # Global phase — skip
            elif n_nontrivial == 1:
                total += 1  # Single Pauli rotation
            else:
                total += 2 * (n_nontrivial - 1) + 1  # CNOT ladder + Rz

        multiplier = {
            TrotterOrder.FIRST: 1,
            TrotterOrder.SECOND: 2,   # Forward + backward
            TrotterOrder.FOURTH: 10,  # 5 × second_order
        }
        return total * multiplier.get(order, 1)


# ═══════════════════════════════════════════════════════════════════════════════
#  TROTTER BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════

class TrotterBenchmark:
    """
    Benchmarks Trotter decomposition accuracy against exact evolution.

    Measures:
      - Fidelity F = |⟨ψ_exact|ψ_trotter⟩|² as function of step count
      - Error scaling exponent (should be p+1 for order-p formula)
      - Gate cost vs. accuracy tradeoff
      - Minimum steps for target fidelity
    """

    @staticmethod
    def fidelity_vs_steps(
        hamiltonian: Hamiltonian,
        initial_state: np.ndarray,
        total_time: float,
        step_counts: Sequence[int],
        order: TrotterOrder = TrotterOrder.SECOND,
    ) -> TrotterBenchmarkResult:
        """
        Sweep step counts and measure fidelity at each.

        Args:
            hamiltonian: System Hamiltonian
            initial_state: |ψ₀⟩
            total_time: Total evolution time t
            step_counts: List of step counts to benchmark
            order: Trotter order

        Returns:
            TrotterBenchmarkResult with fidelity/error curves
        """
        start = time.time()
        psi0 = initial_state.copy().astype(complex).reshape(-1)

        # Exact evolution (reference)
        psi_exact = ExactEvolution.evolve(hamiltonian, psi0, total_time)

        fidelities = []
        errors = []
        gate_counts_list = []
        gates_per_step = TrotterEngine.gate_count_per_step(hamiltonian, order)

        for n_steps in step_counts:
            psi_trotter = TrotterEngine.evolve(
                hamiltonian, psi0, total_time, n_steps, order
            )
            fid = float(abs(np.vdot(psi_exact, psi_trotter)) ** 2)
            fidelities.append(fid)
            errors.append(1.0 - fid)
            gate_counts_list.append(gates_per_step * n_steps)

        # Fit error scaling: error ∝ (1/n)^α → log(error) = -α·log(n) + const
        scaling_exp = None
        valid_errors = [(n, e) for n, e in zip(step_counts, errors) if e > 1e-14]
        if len(valid_errors) >= 2:
            log_n = np.log([n for n, _ in valid_errors])
            log_e = np.log([e for _, e in valid_errors])
            if len(log_n) >= 2:
                # Linear fit
                coeffs = np.polyfit(log_n, log_e, 1)
                scaling_exp = float(-coeffs[0])

        # Optimal steps for target fidelities
        opt_99 = None
        opt_999 = None
        for n, f in zip(step_counts, fidelities):
            if f >= 0.99 and opt_99 is None:
                opt_99 = n
            if f >= 0.999 and opt_999 is None:
                opt_999 = n

        elapsed = (time.time() - start) * 1000.0

        return TrotterBenchmarkResult(
            order=order.value,
            num_qubits=hamiltonian.num_qubits,
            total_time=total_time,
            step_counts=list(step_counts),
            fidelities=fidelities,
            gate_counts=gate_counts_list,
            errors=errors,
            error_scaling_exponent=scaling_exp,
            optimal_steps_for_99=opt_99,
            optimal_steps_for_999=opt_999,
            simulation_time_ms=elapsed,
            metadata={
                "order": order.value,
                "gates_per_step": gates_per_step,
                "hamiltonian_type": hamiltonian.hamiltonian_type.name,
                "god_code": GOD_CODE,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  OBSERVABLE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class ObservableEngine:
    """
    Compute expectation values and correlations over time-evolved states.
    No wavefunction collapse — purely Hermitian sandwich ⟨ψ|O|ψ⟩.
    """

    @staticmethod
    def expectation(state: np.ndarray, observable: np.ndarray) -> float:
        """⟨ψ|O|ψ⟩ — expectation value of observable O in state |ψ⟩."""
        psi = state.reshape(-1)
        return float(np.real(psi.conj() @ observable @ psi))

    @staticmethod
    def expectation_time_series(
        evolution_result: EvolutionResult,
        observable: np.ndarray,
    ) -> List[float]:
        """⟨O⟩(t) at each time point in an evolution result."""
        return [
            ObservableEngine.expectation(state, observable)
            for state in evolution_result.states
        ]

    @staticmethod
    def pauli_expectation(state: np.ndarray, qubit: int,
                          pauli: str, num_qubits: int) -> float:
        """⟨ψ|P_i|ψ⟩ for Pauli P on qubit i."""
        mat = _pauli_string_matrix([(qubit, pauli)], num_qubits)
        return ObservableEngine.expectation(state, mat)

    @staticmethod
    def magnetisation(state: np.ndarray, num_qubits: int, axis: str = "Z") -> float:
        """
        Total magnetisation: M = (1/n) Σ ⟨P_i⟩ for axis P ∈ {X, Y, Z}.
        """
        total = sum(
            ObservableEngine.pauli_expectation(state, q, axis, num_qubits)
            for q in range(num_qubits)
        )
        return total / num_qubits

    @staticmethod
    def two_point_correlator(state: np.ndarray, qubit_i: int, qubit_j: int,
                              pauli: str, num_qubits: int) -> float:
        """
        Two-point correlator: ⟨P_i P_j⟩ - ⟨P_i⟩⟨P_j⟩ (connected correlation).
        """
        # ⟨P_i P_j⟩
        mat_ij = _pauli_string_matrix([(qubit_i, pauli), (qubit_j, pauli)], num_qubits)
        corr_ij = ObservableEngine.expectation(state, mat_ij)
        # ⟨P_i⟩ × ⟨P_j⟩
        exp_i = ObservableEngine.pauli_expectation(state, qubit_i, pauli, num_qubits)
        exp_j = ObservableEngine.pauli_expectation(state, qubit_j, pauli, num_qubits)
        return corr_ij - exp_i * exp_j

    @staticmethod
    def energy(state: np.ndarray, hamiltonian: Hamiltonian) -> float:
        """⟨ψ|H|ψ⟩ — total energy."""
        H_mat = hamiltonian.matrix()
        return ObservableEngine.expectation(state, H_mat)

    @staticmethod
    def energy_variance(state: np.ndarray, hamiltonian: Hamiltonian) -> float:
        """Var(H) = ⟨H²⟩ - ⟨H⟩². Measures energy fluctuations."""
        H_mat = hamiltonian.matrix()
        psi = state.reshape(-1)
        e_mean = float(np.real(psi.conj() @ H_mat @ psi))
        e2_mean = float(np.real(psi.conj() @ (H_mat @ H_mat) @ psi))
        return e2_mean - e_mean ** 2


# ═══════════════════════════════════════════════════════════════════════════════
#  ANALOG SIMULATOR — TOP-LEVEL ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class AnalogSimulator:
    """
    Top-level orchestrator for analog quantum simulation.

    Combines Hamiltonian building, exact evolution, Trotter decomposition,
    benchmarking, and observable measurement into a unified research interface.

    Usage:
        from l104_quantum_gate_engine.analog import AnalogSimulator

        sim = AnalogSimulator()
        H = sim.build.transverse_ising(4, J=1.0, h=0.5)

        # Exact evolution
        result = sim.exact_evolve(H, t=2.0, n_points=50)

        # Trotter benchmark
        bench = sim.benchmark_trotter(H, t=2.0, orders=[1, 2, 4])

        # Sacred Hamiltonian
        H_sacred = sim.build.sacred_hamiltonian(4)
        spectrum = H_sacred.eigenspectrum()
    """

    def __init__(self):
        self.build = HamiltonianBuilder()
        self.exact = ExactEvolution()
        self.trotter = TrotterEngine()
        self.benchmark = TrotterBenchmark()
        self.observe = ObservableEngine()
        self._metrics = {
            "hamiltonians_built": 0,
            "exact_evolutions": 0,
            "trotter_evolutions": 0,
            "benchmarks_run": 0,
        }

    def exact_evolve(
        self,
        hamiltonian: Hamiltonian,
        t: float = 1.0,
        n_points: int = 50,
        initial_state: Optional[np.ndarray] = None,
    ) -> EvolutionResult:
        """
        Exact time evolution over a uniform time grid.

        Args:
            hamiltonian: System Hamiltonian
            t: Total evolution time
            n_points: Number of time points
            initial_state: |ψ₀⟩ (default: |0...0⟩)

        Returns:
            EvolutionResult with exact states at each time
        """
        self._metrics["exact_evolutions"] += 1
        dim = 2 ** hamiltonian.num_qubits
        if initial_state is None:
            initial_state = np.zeros(dim, dtype=complex)
            initial_state[0] = 1.0

        time_points = np.linspace(0, t, n_points).tolist()
        return ExactEvolution.time_series(hamiltonian, initial_state, time_points)

    def trotter_evolve(
        self,
        hamiltonian: Hamiltonian,
        t: float = 1.0,
        n_steps: int = 20,
        order: Union[int, TrotterOrder] = TrotterOrder.SECOND,
        n_points: int = 50,
        initial_state: Optional[np.ndarray] = None,
    ) -> EvolutionResult:
        """
        Trotterised time evolution with exact-fidelity comparison.

        Args:
            hamiltonian: System Hamiltonian
            t: Total evolution time
            n_steps: Total Trotter steps (across all time points)
            order: Trotter order (1, 2, or 4)
            n_points: Number of time points
            initial_state: |ψ₀⟩ (default: |0...0⟩)

        Returns:
            EvolutionResult with Trotter states and fidelity vs exact
        """
        self._metrics["trotter_evolutions"] += 1
        dim = 2 ** hamiltonian.num_qubits
        if initial_state is None:
            initial_state = np.zeros(dim, dtype=complex)
            initial_state[0] = 1.0

        if isinstance(order, int):
            order = TrotterOrder(order)

        time_points = np.linspace(0, t, n_points).tolist()
        steps_per_interval = max(1, n_steps // max(1, n_points - 1))

        return TrotterEngine.time_series(
            hamiltonian, initial_state, time_points,
            n_steps_per_interval=steps_per_interval,
            order=order,
            exact_reference=True,
        )

    def benchmark_trotter(
        self,
        hamiltonian: Hamiltonian,
        t: float = 1.0,
        orders: Optional[List[int]] = None,
        step_counts: Optional[List[int]] = None,
        initial_state: Optional[np.ndarray] = None,
    ) -> Dict[int, TrotterBenchmarkResult]:
        """
        Comprehensive Trotter benchmarking across multiple orders.

        Args:
            hamiltonian: System Hamiltonian
            t: Total evolution time
            orders: List of Trotter orders to benchmark (default: [1, 2, 4])
            step_counts: Step counts to sweep (default: [1, 2, 5, 10, 20, 50, 100])
            initial_state: |ψ₀⟩

        Returns:
            Dict mapping order → TrotterBenchmarkResult
        """
        self._metrics["benchmarks_run"] += 1
        dim = 2 ** hamiltonian.num_qubits
        if initial_state is None:
            initial_state = np.zeros(dim, dtype=complex)
            initial_state[0] = 1.0

        if orders is None:
            orders = [1, 2, 4]
        if step_counts is None:
            step_counts = [1, 2, 5, 10, 20, 50, 100]

        results = {}
        for o in orders:
            trotter_order = TrotterOrder(o)
            results[o] = TrotterBenchmark.fidelity_vs_steps(
                hamiltonian, initial_state, t, step_counts, order=trotter_order
            )
        return results

    def sacred_analysis(self, num_qubits: int = 4, t: float = 2.0) -> Dict[str, Any]:
        """
        Full sacred Hamiltonian analysis: spectrum, evolution, benchmarking.

        Args:
            num_qubits: System size
            t: Evolution time

        Returns:
            Dict with spectrum, evolution, and benchmark data
        """
        H = HamiltonianBuilder.sacred_hamiltonian(num_qubits)
        dim = 2 ** num_qubits
        psi0 = np.zeros(dim, dtype=complex)
        psi0[0] = 1.0

        # Spectrum
        evals, evecs = H.eigenspectrum()
        gap = float(evals[1] - evals[0]) if len(evals) >= 2 else 0.0

        # Exact evolution
        exact_result = self.exact_evolve(H, t=t, n_points=30, initial_state=psi0)

        # Benchmark
        bench = self.benchmark_trotter(
            H, t=t, orders=[1, 2, 4],
            step_counts=[1, 2, 5, 10, 20, 50],
            initial_state=psi0,
        )

        # Magnetisation time series
        mag_z = [
            ObservableEngine.magnetisation(s, num_qubits, "Z")
            for s in exact_result.states
        ]

        # Sacred alignment: check if gap aligns with 104-TET
        fundamental_step = 2 ** (1.0 / QUANTIZATION_GRAIN)
        gap_ratio = gap / (GOD_CODE / QUANTIZATION_GRAIN) if gap > 0 else 0
        sacred_aligned = abs(gap_ratio - round(gap_ratio)) < 0.1

        return {
            "hamiltonian": H.to_dict(),
            "eigenvalues": evals.tolist()[:16],
            "energy_gap": gap,
            "ground_energy": float(evals[0]),
            "sacred_coupling": SACRED_COUPLING,
            "sacred_field": SACRED_FIELD,
            "gap_104tet_ratio": round(gap_ratio, 6),
            "sacred_aligned": sacred_aligned,
            "exact_evolution": exact_result.to_dict(),
            "magnetisation_z": [round(m, 8) for m in mag_z],
            "benchmark_order1": bench[1].to_dict(),
            "benchmark_order2": bench[2].to_dict(),
            "benchmark_order4": bench[4].to_dict(),
            "god_code": GOD_CODE,
        }

    @property
    def metrics(self) -> Dict[str, Any]:
        return dict(self._metrics)


# ═══════════════════════════════════════════════════════════════════════════════
#  TROTTERISED CIRCUIT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def trotterise_to_circuit(
    hamiltonian: Hamiltonian,
    t: float,
    n_steps: int,
    order: TrotterOrder = TrotterOrder.SECOND,
) -> 'GateCircuit':
    """
    Convert Trotterised evolution to a GateCircuit.

    Decomposes each e^{-iH_k δt} into native gate operations:
      - Single-qubit Pauli rotation → Rz/Rx/Ry
      - Two-qubit ZZ term → CNOT + Rz + CNOT
      - k-body terms → CNOT ladder + Rz

    Args:
        hamiltonian: System Hamiltonian
        t: Total evolution time
        n_steps: Number of Trotter steps
        order: Product formula order

    Returns:
        GateCircuit implementing the Trotterised evolution
    """
    from .circuit import GateCircuit
    from .gates import Rx, Ry, Rz, CNOT

    n = hamiltonian.num_qubits
    circ = GateCircuit(n, name=f"trotter_o{order.value}_s{n_steps}")
    dt = t / n_steps

    def _append_term(circuit: 'GateCircuit', term: HamiltonianTerm, angle: float):
        """Append a single Pauli-rotation term to the circuit."""
        nontrivial = [(q, p) for q, p in term.paulis if p != "I"]
        if not nontrivial:
            return  # Global phase — skip

        coeff_real = float(np.real(term.coefficient))
        theta = 2.0 * coeff_real * angle  # Factor of 2 from Rz convention

        if len(nontrivial) == 1:
            q, p = nontrivial[0]
            if p == "Z":
                circuit.append(Rz(theta), [q])
            elif p == "X":
                circuit.append(Rx(theta), [q])
            elif p == "Y":
                circuit.append(Ry(theta), [q])
        else:
            # Multi-qubit: basis change → CNOT ladder → Rz → CNOT ladder† → basis change†
            qubits = [q for q, _ in nontrivial]
            pauli_labels = [p for _, p in nontrivial]

            # Basis change: X→H, Y→Rx(π/2)†, Z→identity
            for q, p in zip(qubits, pauli_labels):
                if p == "X":
                    circuit.append(Ry(-math.pi / 2), [q])
                elif p == "Y":
                    circuit.append(Rx(math.pi / 2), [q])

            # CNOT ladder
            for i in range(len(qubits) - 1):
                circuit.append(CNOT, [qubits[i], qubits[i + 1]])

            # Rz on last qubit
            circuit.append(Rz(theta), [qubits[-1]])

            # Reverse CNOT ladder
            for i in range(len(qubits) - 2, -1, -1):
                circuit.append(CNOT, [qubits[i], qubits[i + 1]])

            # Undo basis change
            for q, p in zip(qubits, pauli_labels):
                if p == "X":
                    circuit.append(Ry(math.pi / 2), [q])
                elif p == "Y":
                    circuit.append(Rx(-math.pi / 2), [q])

    def _append_first_order(circuit, full_dt):
        for term in hamiltonian.terms:
            _append_term(circuit, term, full_dt)

    def _append_second_order(circuit, full_dt):
        for term in hamiltonian.terms:
            _append_term(circuit, term, full_dt / 2.0)
        for term in reversed(hamiltonian.terms):
            _append_term(circuit, term, full_dt / 2.0)

    def _append_fourth_order(circuit, full_dt):
        p = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
        for _ in range(2):
            _append_second_order(circuit, p * full_dt)
        _append_second_order(circuit, (1.0 - 4.0 * p) * full_dt)
        for _ in range(2):
            _append_second_order(circuit, p * full_dt)

    for _ in range(n_steps):
        if order == TrotterOrder.FIRST:
            _append_first_order(circ, dt)
        elif order == TrotterOrder.SECOND:
            _append_second_order(circ, dt)
        elif order == TrotterOrder.FOURTH:
            _append_fourth_order(circ, dt)
        circ.barrier()

    return circ


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE-LEVEL SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_simulator_instance: Optional[AnalogSimulator] = None


def get_analog_simulator() -> AnalogSimulator:
    """Get or create the singleton AnalogSimulator."""
    global _simulator_instance
    if _simulator_instance is None:
        _simulator_instance = AnalogSimulator()
    return _simulator_instance
