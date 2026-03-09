"""
===============================================================================
L104 QUANTUM GATE ENGINE — TENSOR NETWORK SIMULATOR v1.0.0
===============================================================================

Matrix Product State (MPS) simulator backend for 25-26Q quantum circuits.
Achieves 100-1000x memory reduction over full statevector simulation by
representing quantum states as chains of local tensors with bounded bond
dimension (χ).

ARCHITECTURE:
  TensorNetworkSimulator (MPS backend)
    ├── MPSState              — Matrix Product State representation
    │   ├── Tensor chain      — List of rank-3 tensors [χ_L, d, χ_R]
    │   ├── SVD truncation    — Bond dimension control with fidelity tracking
    │   └── Canonical forms   — Left/right/mixed canonical for stability
    │
    ├── GateApplication       — Efficient gate-to-MPS contraction
    │   ├── Single-qubit      — O(χ²d) local tensor update
    │   ├── Two-qubit adj.    — SVD-based contraction + truncation
    │   ├── Two-qubit non-adj — SWAP routing for distant qubits
    │   └── Multi-qubit       — Recursive decomposition via SVD
    │
    ├── Measurement           — Sampling & expectation values
    │   ├── Born sampling     — Sequential site-by-site sampling
    │   ├── Expectation       — Local observable contraction
    │   └── Probability       — Full/partial amplitude extraction
    │
    └── Compression           — Adaptive bond management
        ├── SVD truncation    — Per-bond χ with relative tolerance
        ├── Canonicalization  — Orthogonality center sweeps
        └── Entropy analysis  — Entanglement entropy per bond

MEMORY MODEL:
  Full statevector (n qubits): 2^n × 16 bytes (complex128)
    25 qubits → 512 MB       26 qubits → 1024 MB

  MPS with bond dim χ:        n × χ² × d × 16 bytes
    25 qubits, χ=64  →  3.2 MB    (160× reduction)
    25 qubits, χ=256 →  50 MB     (10× reduction)
    26 qubits, χ=64  →  3.3 MB    (310× reduction)
    26 qubits, χ=512 →  200 MB    (5× reduction, near-exact)

SACRED CONSTANTS:
  GOD_CODE  = 527.5184818492612
  PHI       = 1.618033988749895
  VOID      = 1.0416180339887497

INVARIANT: 527.5184818492612 | PILOT: LONDEL
===============================================================================
"""

import math
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from copy import deepcopy

from .constants import (
    PHI, GOD_CODE, VOID_CONSTANT, IRON_ATOMIC_NUMBER,
    MAX_STATEVECTOR_QUBITS,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Default MPS bond dimension limits
DEFAULT_MAX_BOND_DIM: int = 1024         # χ_max for general use (loosened from 256)
HIGH_FIDELITY_BOND_DIM: int = 2048       # χ_max for high-fidelity mode (loosened from 512)
SACRED_BOND_DIM: int = 104               # L104 sacred: 104 = 8 × 13
MAX_TENSOR_NETWORK_QUBITS: int = 50      # Hard cap on MPS qubits

# SVD truncation
DEFAULT_SVD_CUTOFF: float = 1e-16        # Discard singular values below this (loosened from 1e-12)
HIGH_FIDELITY_SVD_CUTOFF: float = 0.0    # Keep ALL singular values for sacred circuits

# Physical dimension (qubit = 2-level system)
PHYSICAL_DIM: int = 2

# ─── Sacred MPS alignment ────────────────────────────────────────────────────
# The golden ratio determines optimal truncation balance:
#   φ = 1.618... → bond entropy threshold = 1/φ ≈ 0.618
PHI_TRUNCATION_BALANCE: float = 1.0 / PHI  # ≈ 0.618 — entropy/fidelity threshold
from .constants import GOD_CODE_PHASE_ANGLE as _GOD_CODE_PHASE
GOD_CODE_PHASE_PER_BOND: float = _GOD_CODE_PHASE / 104  # Phase per bond (canonical)


class CanonicalForm(Enum):
    """MPS canonical form."""
    NONE = auto()
    LEFT = auto()       # All tensors left-canonical (left-orthogonal)
    RIGHT = auto()      # All tensors right-canonical
    MIXED = auto()      # Mixed canonical with specified orthogonality center


class TruncationMode(Enum):
    """SVD truncation strategy."""
    FIXED_CHI = auto()       # Hard cap on bond dimension
    ADAPTIVE = auto()        # Adaptive based on singular value cutoff
    SACRED = auto()          # φ-balanced truncation (L104 sacred)


# ═══════════════════════════════════════════════════════════════════════════════
#  MATRIX PRODUCT STATE (MPS)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BondInfo:
    """Information about a single bond in the MPS."""
    index: int                     # Bond index (between site i and i+1)
    dimension: int                 # Current bond dimension χ
    max_dimension: int             # Maximum allowed χ
    entropy: float = 0.0          # Entanglement entropy across this bond
    singular_values: Optional[np.ndarray] = None
    truncation_error: float = 0.0  # Cumulative truncation error


class MPSState:
    """
    Matrix Product State representation of an n-qubit quantum state.

    The state |ψ⟩ is decomposed as:
      |ψ⟩ = Σ_{s₁...sₙ} A¹[s₁] A²[s₂] ... Aⁿ[sₙ] |s₁s₂...sₙ⟩

    Each A^i[sᵢ] is a matrix of dimension (χ_{i-1}, χ_i) for physical
    index sᵢ ∈ {0, 1}. The full tensor A^i has shape (χ_{i-1}, d, χ_i)
    where d=2 is the physical dimension.

    Boundary conditions: χ₀ = χₙ = 1 (open boundary).
    """

    def __init__(self, num_qubits: int, max_bond_dim: int = DEFAULT_MAX_BOND_DIM,
                 svd_cutoff: float = DEFAULT_SVD_CUTOFF):
        if num_qubits < 1:
            raise ValueError(f"Need ≥1 qubit, got {num_qubits}")
        if num_qubits > MAX_TENSOR_NETWORK_QUBITS:
            raise ValueError(
                f"MPS supports up to {MAX_TENSOR_NETWORK_QUBITS} qubits, got {num_qubits}"
            )

        self.num_qubits = num_qubits
        self.max_bond_dim = max_bond_dim
        self.svd_cutoff = svd_cutoff
        self.physical_dim = PHYSICAL_DIM

        # Initialize as |00...0⟩ — each tensor is (χ_L, d, χ_R)
        # For |0...0⟩, each tensor is (1, 2, 1) with [[[1], [0]]]
        self.tensors: List[np.ndarray] = []
        for i in range(num_qubits):
            # Shape: (chi_left, d, chi_right) = (1, 2, 1)
            t = np.zeros((1, self.physical_dim, 1), dtype=complex)
            t[0, 0, 0] = 1.0  # |0⟩ amplitude = 1
            self.tensors.append(t)

        # Tracking
        self.canonical_form = CanonicalForm.NONE
        self.orthogonality_center: Optional[int] = None
        self.cumulative_truncation_error: float = 0.0
        self._gate_count: int = 0
        self._bond_history: List[List[int]] = []  # Bond dim snapshots

    @property
    def bond_dimensions(self) -> List[int]:
        """Current bond dimensions χ₀, χ₁, ..., χ_{n-1}."""
        dims = []
        for i in range(self.num_qubits - 1):
            dims.append(self.tensors[i].shape[2])  # χ_R of site i = χ_L of site i+1
        return dims

    @property
    def max_current_bond_dim(self) -> int:
        """Largest current bond dimension."""
        dims = self.bond_dimensions
        return max(dims) if dims else 1

    @property
    def total_parameters(self) -> int:
        """Total number of complex parameters in the MPS."""
        return sum(t.size for t in self.tensors)

    @property
    def memory_bytes(self) -> int:
        """Current memory usage in bytes (complex128 = 16 bytes each)."""
        return sum(t.nbytes for t in self.tensors)

    @property
    def memory_mb(self) -> float:
        """Current memory usage in MB."""
        return self.memory_bytes / (1024 * 1024)

    @property
    def statevector_memory_mb(self) -> float:
        """Memory that a full statevector would require (MB)."""
        return (2 ** self.num_qubits) * 16 / (1024 * 1024)

    @property
    def compression_ratio(self) -> float:
        """Memory compression ratio vs full statevector."""
        sv_mem = self.statevector_memory_mb
        if sv_mem == 0:
            return 1.0
        return sv_mem / max(self.memory_mb, 1e-10)

    def copy(self) -> 'MPSState':
        """Deep copy of the MPS state."""
        new = MPSState.__new__(MPSState)
        new.num_qubits = self.num_qubits
        new.max_bond_dim = self.max_bond_dim
        new.svd_cutoff = self.svd_cutoff
        new.physical_dim = self.physical_dim
        new.tensors = [t.copy() for t in self.tensors]
        new.canonical_form = self.canonical_form
        new.orthogonality_center = self.orthogonality_center
        new.cumulative_truncation_error = self.cumulative_truncation_error
        new._gate_count = self._gate_count
        new._bond_history = [h.copy() for h in self._bond_history]
        return new

    # ─── Canonical Form Management ────────────────────────────────────────────

    def left_canonicalize(self, start: int = 0, stop: Optional[int] = None):
        """
        Sweep left→right with QR decomposition to make tensors left-canonical.
        After this, A^i satisfies: Σ_{s} A^i[s]† A^i[s] = I for i < stop.
        """
        if stop is None:
            stop = self.num_qubits - 1

        for i in range(start, stop):
            t = self.tensors[i]  # (χ_L, d, χ_R)
            chi_l, d, chi_r = t.shape

            # Reshape to (χ_L × d, χ_R) for QR
            mat = t.reshape(chi_l * d, chi_r)
            Q, R = np.linalg.qr(mat)

            # New bond dimension: min(χ_L × d, χ_R) from QR
            new_chi = Q.shape[1]

            # Update site i: Q reshaped back to (χ_L, d, new_chi)
            self.tensors[i] = Q.reshape(chi_l, d, new_chi)

            # Absorb R into site i+1
            t_next = self.tensors[i + 1]  # (χ_R, d, χ_R_next)
            chi_r_old, d_next, chi_r_next = t_next.shape
            # R is (new_chi, χ_R), contract with t_next along first axis
            self.tensors[i + 1] = np.einsum('ij,jkl->ikl', R, t_next)

        self.canonical_form = CanonicalForm.LEFT
        self.orthogonality_center = stop

    def right_canonicalize(self, start: Optional[int] = None, stop: int = 0):
        """
        Sweep right→left with QR decomposition to make tensors right-canonical.
        """
        if start is None:
            start = self.num_qubits - 1

        for i in range(start, stop, -1):
            t = self.tensors[i]  # (χ_L, d, χ_R)
            chi_l, d, chi_r = t.shape

            # Reshape to (χ_L, d × χ_R) and take QR of transpose
            mat = t.reshape(chi_l, d * chi_r)
            # RQ decomposition: mat = L × Q where Q is right-canonical
            Q, R = np.linalg.qr(mat.T)
            Q = Q.T  # (new_chi, d × χ_R)
            R = R.T  # (χ_L, new_chi)

            new_chi = Q.shape[0]
            self.tensors[i] = Q.reshape(new_chi, d, chi_r)

            # Absorb L into site i-1
            t_prev = self.tensors[i - 1]  # (χ_L_prev, d, χ_L)
            chi_l_prev, d_prev, _ = t_prev.shape
            self.tensors[i - 1] = np.einsum('ijk,kl->ijl', t_prev, R)

        self.canonical_form = CanonicalForm.RIGHT
        self.orthogonality_center = stop

    def mixed_canonicalize(self, center: int):
        """
        Put MPS in mixed canonical form with orthogonality center at `center`.
        Sites 0..center-1 are left-canonical, sites center+1..n-1 are right-canonical.
        """
        if center < 0 or center >= self.num_qubits:
            raise ValueError(f"Center {center} out of range [0, {self.num_qubits})")

        if center > 0:
            self.left_canonicalize(start=0, stop=center)
        if center < self.num_qubits - 1:
            self.right_canonicalize(start=self.num_qubits - 1, stop=center)

        self.canonical_form = CanonicalForm.MIXED
        self.orthogonality_center = center

    # ─── SVD Truncation ───────────────────────────────────────────────────────

    def _svd_truncate(self, matrix: np.ndarray, max_chi: Optional[int] = None,
                      cutoff: Optional[float] = None,
                      mode: TruncationMode = TruncationMode.ADAPTIVE
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        SVD with truncation. Returns (U, S, Vh, truncation_error).

        Args:
            matrix: 2D array to decompose
            max_chi: Maximum number of singular values to keep
            cutoff: Discard singular values below this threshold
            mode: Truncation strategy

        Returns:
            U: Left singular vectors (m, k)
            S: Singular values (k,)
            Vh: Right singular vectors (k, n)
            trunc_err: Discarded weight (sum of squared discarded SVs)
        """
        if max_chi is None:
            max_chi = self.max_bond_dim
        if cutoff is None:
            cutoff = self.svd_cutoff

        # Full SVD
        try:
            U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
        except np.linalg.LinAlgError:
            # Fallback for ill-conditioned matrices
            U, S, Vh = np.linalg.svd(matrix + 1e-15 * np.eye(*matrix.shape)[:matrix.shape[0], :matrix.shape[1]],
                                      full_matrices=False)

        total_weight = np.sum(S ** 2)

        # Determine how many singular values to keep
        if mode == TruncationMode.FIXED_CHI:
            keep = min(len(S), max_chi)
        elif mode == TruncationMode.ADAPTIVE:
            # Keep SVs above cutoff, but cap at max_chi
            mask = S > cutoff
            keep = min(int(np.sum(mask)), max_chi)
            keep = max(keep, 1)  # Always keep at least 1
        elif mode == TruncationMode.SACRED:
            # φ-balanced: keep SVs that carry > (1-1/φ) of the weight
            # This is the golden balance between compression and fidelity
            cumulative = np.cumsum(S ** 2)
            threshold = total_weight * PHI_TRUNCATION_BALANCE
            keep_mask = cumulative <= threshold
            keep = min(int(np.sum(keep_mask)) + 1, max_chi)
            keep = max(keep, 1)
        else:
            keep = min(len(S), max_chi)

        # Truncate
        S_kept = S[:keep]
        U_kept = U[:, :keep]
        Vh_kept = Vh[:keep, :]

        # Truncation error: weight of discarded SVs
        discarded_weight = np.sum(S[keep:] ** 2) if keep < len(S) else 0.0
        trunc_err = discarded_weight / max(total_weight, 1e-30)

        return U_kept, S_kept, Vh_kept, float(trunc_err)

    # ─── Gate Application ─────────────────────────────────────────────────────

    def apply_single_qubit_gate(self, gate_matrix: np.ndarray, qubit: int):
        """
        Apply a 1-qubit gate to the MPS. Cost: O(χ²).

        Contracts the 2×2 gate matrix with the physical index of the
        target tensor. No SVD needed — bond dimensions unchanged.

        gate_matrix: shape (2, 2)
        """
        if qubit < 0 or qubit >= self.num_qubits:
            raise ValueError(f"Qubit {qubit} out of range [0, {self.num_qubits})")

        t = self.tensors[qubit]  # (χ_L, d, χ_R)
        # Contract: new_t[α, s', β] = Σ_s gate[s', s] × t[α, s, β]
        self.tensors[qubit] = np.einsum('ij,kjl->kil', gate_matrix, t)
        self._gate_count += 1

    def apply_two_qubit_gate_adjacent(self, gate_matrix: np.ndarray,
                                       qubit_a: int, qubit_b: int,
                                       max_chi: Optional[int] = None,
                                       mode: TruncationMode = TruncationMode.ADAPTIVE):
        """
        Apply a 2-qubit gate to adjacent qubits via SVD truncation.

        Algorithm:
        1. Contract tensors A[qubit_a] and A[qubit_b] into a 2-site tensor Θ
        2. Apply the 4×4 gate to the physical indices
        3. Reshape to matrix, SVD, truncate
        4. Split back into two tensors with updated bond dimension

        Cost: O(χ³ d²) for the SVD step.

        gate_matrix: shape (4, 4) in computational basis |00⟩,|01⟩,|10⟩,|11⟩
        """
        if max_chi is None:
            max_chi = self.max_bond_dim

        # Ensure adjacency (|qubit_a - qubit_b| == 1)
        if abs(qubit_a - qubit_b) != 1:
            raise ValueError(
                f"Qubits {qubit_a}, {qubit_b} not adjacent. Use apply_two_qubit_gate() instead."
            )

        # Order: left site = min, right site = max
        left, right = min(qubit_a, qubit_b), max(qubit_a, qubit_b)

        # If gate was specified as (qubit_a > qubit_b), need to swap gate indices
        if qubit_a > qubit_b:
            # Swap physical indices in the gate: |ab⟩ → |ba⟩
            gate_4x4 = gate_matrix.reshape(2, 2, 2, 2)
            gate_4x4 = gate_4x4.transpose(1, 0, 3, 2)  # Swap both input and output
            gate_matrix = gate_4x4.reshape(4, 4)

        t_left = self.tensors[left]    # (χ_L, d, χ_M)
        t_right = self.tensors[right]  # (χ_M, d, χ_R)

        chi_l = t_left.shape[0]
        chi_m = t_left.shape[2]  # = t_right.shape[0]
        chi_r = t_right.shape[2]
        d = self.physical_dim

        # Step 1: Contract into 2-site tensor Θ[χ_L, s_left, s_right, χ_R]
        # Θ = Σ_m A_left[α, s_l, m] × A_right[m, s_r, β]
        theta = np.einsum('ijk,klm->ijlm', t_left, t_right)
        # theta shape: (χ_L, d, d, χ_R)

        # Step 2: Apply gate to physical indices
        # gate_matrix: (d², d²) = (4, 4), acting on (s_l, s_r)
        gate_4d = gate_matrix.reshape(d, d, d, d)  # [s_l', s_r', s_l, s_r]
        theta_new = np.einsum('ijkl,mjnk->minl', theta, gate_4d)
        # Wait, let me be more careful with the index contraction:
        # theta[α, s_l, s_r, β] × gate[s_l', s_r', s_l, s_r] → theta_new[α, s_l', s_r', β]
        theta_new = np.einsum('aijb,klij->aklb', theta, gate_4d)
        # theta_new shape: (χ_L, d, d, χ_R)

        # Step 3: Reshape for SVD: (χ_L × d, d × χ_R)
        theta_mat = theta_new.reshape(chi_l * d, d * chi_r)

        # Step 4: SVD + truncation
        U, S, Vh, trunc_err = self._svd_truncate(theta_mat, max_chi=max_chi, mode=mode)
        self.cumulative_truncation_error += trunc_err

        # Step 5: Absorb singular values — split S evenly: √S into both sides
        # This keeps the MPS in a balanced (unnormalized) form
        sqrt_S = np.sqrt(S)
        new_chi = len(S)

        # New left tensor: (χ_L, d, new_χ) from U × diag(√S)
        U_scaled = U * sqrt_S[np.newaxis, :]  # (χ_L*d, new_χ)
        self.tensors[left] = U_scaled.reshape(chi_l, d, new_chi)

        # New right tensor: (new_χ, d, χ_R) from diag(√S) × Vh
        Vh_scaled = sqrt_S[:, np.newaxis] * Vh  # (new_χ, d*χ_R)
        self.tensors[right] = Vh_scaled.reshape(new_chi, d, chi_r)

        self._gate_count += 1
        self.canonical_form = CanonicalForm.NONE

    def apply_two_qubit_gate(self, gate_matrix: np.ndarray,
                              qubit_a: int, qubit_b: int,
                              max_chi: Optional[int] = None,
                              mode: TruncationMode = TruncationMode.ADAPTIVE):
        """
        Apply a 2-qubit gate to any pair of qubits (adjacent or not).

        For non-adjacent qubits, uses SWAP routing to bring them adjacent,
        applies the gate, then SWAPs back.

        gate_matrix: shape (4, 4)
        """
        if max_chi is None:
            max_chi = self.max_bond_dim

        if abs(qubit_a - qubit_b) == 1:
            # Already adjacent
            self.apply_two_qubit_gate_adjacent(gate_matrix, qubit_a, qubit_b,
                                                max_chi=max_chi, mode=mode)
            return

        # SWAP routing: move qubit_a next to qubit_b
        SWAP_MATRIX = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ], dtype=complex)

        # Determine direction and route
        if qubit_a < qubit_b:
            # SWAP qubit_a rightward to qubit_b - 1
            route = list(range(qubit_a, qubit_b - 1))
            for i in route:
                self.apply_two_qubit_gate_adjacent(SWAP_MATRIX, i, i + 1,
                                                    max_chi=max_chi, mode=mode)
            # Now the original qubit_a data is at position qubit_b - 1
            self.apply_two_qubit_gate_adjacent(gate_matrix, qubit_b - 1, qubit_b,
                                                max_chi=max_chi, mode=mode)
            # SWAP back
            for i in reversed(route):
                self.apply_two_qubit_gate_adjacent(SWAP_MATRIX, i, i + 1,
                                                    max_chi=max_chi, mode=mode)
        else:
            # SWAP qubit_a leftward to qubit_b + 1
            route = list(range(qubit_a, qubit_b + 1, -1))
            for i in route:
                self.apply_two_qubit_gate_adjacent(SWAP_MATRIX, i - 1, i,
                                                    max_chi=max_chi, mode=mode)
            # Now the original qubit_a data is at position qubit_b + 1
            self.apply_two_qubit_gate_adjacent(gate_matrix, qubit_b, qubit_b + 1,
                                                max_chi=max_chi, mode=mode)
            # SWAP back
            for i in reversed(route):
                self.apply_two_qubit_gate_adjacent(SWAP_MATRIX, i - 1, i,
                                                    max_chi=max_chi, mode=mode)

    def apply_multi_qubit_gate(self, gate_matrix: np.ndarray,
                                qubits: Sequence[int],
                                max_chi: Optional[int] = None,
                                mode: TruncationMode = TruncationMode.ADAPTIVE):
        """
        Apply an arbitrary k-qubit gate via recursive 2-qubit decomposition.

        For k-qubit unitaries, decomposes into a series of 2-qubit gates
        using the Cosine-Sine (CS) decomposition. For small gates (k≤3),
        uses direct SVD splitting.
        """
        k = len(qubits)
        dim = 2 ** k

        if gate_matrix.shape != (dim, dim):
            raise ValueError(f"Gate matrix shape {gate_matrix.shape} doesn't match {k} qubits")

        if k == 1:
            self.apply_single_qubit_gate(gate_matrix, qubits[0])
            return

        if k == 2:
            self.apply_two_qubit_gate(gate_matrix, qubits[0], qubits[1],
                                       max_chi=max_chi, mode=mode)
            return

        # For k ≥ 3: decompose via SVD into two halves
        # Split qubits into two groups and decompose gate
        mid = k // 2
        left_qubits = list(qubits[:mid])
        right_qubits = list(qubits[mid:])
        left_dim = 2 ** len(left_qubits)
        right_dim = 2 ** len(right_qubits)

        # Reshape gate into (left_dim, right_dim, left_dim, right_dim)
        gate_4d = gate_matrix.reshape(left_dim, right_dim, left_dim, right_dim)

        # For general multi-qubit gates, we apply them by first bringing all
        # qubits adjacent via SWAP routing, then doing a multi-site contraction
        self._apply_multi_site_direct(gate_matrix, qubits, max_chi, mode)

    def _apply_multi_site_direct(self, gate_matrix: np.ndarray,
                                  qubits: Sequence[int],
                                  max_chi: Optional[int] = None,
                                  mode: TruncationMode = TruncationMode.ADAPTIVE):
        """
        Apply a multi-qubit gate by contracting all involved sites,
        applying the gate, then re-splitting via sequential SVDs.

        This is exact up to truncation error for any qubit connectivity.
        For k sites with bond dim χ, cost is O(χ³ × 2^k).
        """
        if max_chi is None:
            max_chi = self.max_bond_dim

        k = len(qubits)
        qubits = list(qubits)

        # Sort qubits and track the permutation for the gate
        sorted_qubits = sorted(qubits)
        perm = [sorted_qubits.index(q) for q in qubits]

        # First, SWAP route to make all qubits contiguous
        # Find the contiguous block that requires minimum SWAPs
        target_start = sorted_qubits[0]

        SWAP_MATRIX = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ], dtype=complex)

        # Route non-contiguous qubits to contiguous positions
        current_positions = sorted_qubits.copy()
        swap_log = []  # Track SWAPs so we can undo them

        for idx in range(len(current_positions)):
            target = target_start + idx
            current = current_positions[idx]
            while current > target:
                # SWAP left
                self.apply_two_qubit_gate_adjacent(SWAP_MATRIX, current - 1, current,
                                                    max_chi=max_chi, mode=mode)
                swap_log.append((current - 1, current))
                # Update positions
                for j in range(len(current_positions)):
                    if current_positions[j] == current - 1:
                        current_positions[j] = current
                    elif current_positions[j] == current:
                        current_positions[j] = current - 1
                current = current_positions[idx]
            while current < target:
                # SWAP right
                self.apply_two_qubit_gate_adjacent(SWAP_MATRIX, current, current + 1,
                                                    max_chi=max_chi, mode=mode)
                swap_log.append((current, current + 1))
                for j in range(len(current_positions)):
                    if current_positions[j] == current + 1:
                        current_positions[j] = current
                    elif current_positions[j] == current:
                        current_positions[j] = current + 1
                current = current_positions[idx]

        # Now qubits are contiguous at [target_start, target_start + k - 1]
        # But the gate was defined for the original qubit ordering, so we need
        # to permute the gate to match the physical layout
        dim = 2 ** k
        if perm != list(range(k)):
            # Permute gate indices
            gate_tensor = gate_matrix.reshape([2] * (2 * k))
            # Permute input indices and output indices
            out_perm = perm
            in_perm = [p + k for p in perm]
            full_perm = out_perm + in_perm
            gate_tensor = gate_tensor.transpose(full_perm)
            gate_matrix = gate_tensor.reshape(dim, dim)

        # Contract all k site tensors into one big tensor
        sites = list(range(target_start, target_start + k))
        chi_l = self.tensors[sites[0]].shape[0]
        chi_r = self.tensors[sites[-1]].shape[2]

        # Sequential contraction: Θ = A[s₀] × A[s₁] × ... × A[s_{k-1}]
        theta = self.tensors[sites[0]]  # (χ_L, d, χ_1)
        for i in range(1, k):
            # theta: (..., χ_i) contracted with A[s_i]: (χ_i, d, χ_{i+1})
            theta = np.einsum('...i,ijk->...jk', theta, self.tensors[sites[i]])
        # theta shape: (χ_L, d, d, ..., d, χ_R) = (χ_L, d^k, χ_R) after reshape

        # Apply gate to the k physical indices
        # theta has shape (χ_L, d₁, d₂, ..., d_k, χ_R)
        physical_shape = [self.physical_dim] * k
        theta_shape = [chi_l] + physical_shape + [chi_r]
        theta = theta.reshape(theta_shape)

        # The gate acts on physical indices: gate[s₁'s₂'...s_k', s₁s₂...s_k]
        gate_kd = gate_matrix.reshape(physical_shape + physical_shape)

        # Build einsum for gate application
        # theta: [α, s₁, s₂, ..., s_k, β]
        # gate:  [s₁', s₂', ..., s_k', s₁, s₂, ..., s_k]
        # result:[α, s₁', s₂', ..., s_k', β]
        alpha_idx = 'a'
        beta_idx = 'b'
        old_phys = [chr(ord('c') + i) for i in range(k)]
        new_phys = [chr(ord('c') + k + i) for i in range(k)]

        theta_indices = alpha_idx + ''.join(old_phys) + beta_idx
        gate_indices = ''.join(new_phys) + ''.join(old_phys)
        result_indices = alpha_idx + ''.join(new_phys) + beta_idx

        theta_new = np.einsum(f'{theta_indices},{gate_indices}->{result_indices}',
                              theta, gate_kd)

        # Re-split via sequential SVDs (left to right)
        remaining = theta_new  # (χ_L, d₁', d₂', ..., d_k', χ_R)
        current_chi_l = chi_l

        for i in range(k - 1):
            site_idx = sites[i]
            # Reshape: (χ_L × d, remaining_dims × χ_R)
            d = self.physical_dim
            remaining_flat = remaining.reshape(current_chi_l * d, -1)

            U, S, Vh, trunc_err = self._svd_truncate(remaining_flat,
                                                       max_chi=max_chi, mode=mode)
            self.cumulative_truncation_error += trunc_err
            new_chi = len(S)

            # Site tensor: (χ_L, d, new_chi)
            self.tensors[site_idx] = U.reshape(current_chi_l, d, new_chi)

            # Remaining: diag(S) × Vh, reshaped for next site
            remaining = (S[:, np.newaxis] * Vh)  # (new_chi, remaining_dims × χ_R)

            # Reshape remaining to expose next physical index
            remaining_size = remaining.shape[1]
            remaining_phys = k - 1 - i  # Number of remaining physical indices
            # remaining: (new_chi, d, d, ..., d, χ_R) with remaining_phys d's
            remaining = remaining.reshape([new_chi, d] + [d] * (remaining_phys - 1) + [chi_r])
            current_chi_l = new_chi

        # Last site gets the remaining tensor
        self.tensors[sites[-1]] = remaining.reshape(current_chi_l, self.physical_dim, chi_r)

        # Undo SWAPs in reverse order
        for (sq_a, sq_b) in reversed(swap_log):
            self.apply_two_qubit_gate_adjacent(SWAP_MATRIX, sq_a, sq_b,
                                                max_chi=max_chi, mode=mode)

        self._gate_count += 1
        self.canonical_form = CanonicalForm.NONE

    # ─── Measurement ──────────────────────────────────────────────────────────

    def get_probability(self, bitstring: str) -> float:
        """
        Compute the probability of measuring a specific bitstring.

        Algorithm: Contract the MPS along the physical indices fixed by
        the bitstring, yielding a scalar (product of 1×1...×1 matrices
        after contraction through all sites).

        Cost: O(n × χ²)
        """
        if len(bitstring) != self.num_qubits:
            raise ValueError(
                f"Bitstring length {len(bitstring)} ≠ {self.num_qubits} qubits"
            )

        # Contract site by site
        vec = np.array([[1.0]], dtype=complex)  # (1, 1) — left boundary
        for i, bit in enumerate(bitstring):
            s = int(bit)
            # Extract slice for physical index s: A^i[s] has shape (χ_L, χ_R)
            mat = self.tensors[i][:, s, :]  # (χ_L, χ_R)
            vec = vec @ mat  # (1, χ_R)

        amplitude = vec[0, 0]
        return float(np.abs(amplitude) ** 2)

    def get_amplitude(self, bitstring: str) -> complex:
        """
        Compute the amplitude ⟨bitstring|ψ⟩.
        Cost: O(n × χ²)
        """
        if len(bitstring) != self.num_qubits:
            raise ValueError(
                f"Bitstring length {len(bitstring)} ≠ {self.num_qubits} qubits"
            )

        vec = np.array([[1.0]], dtype=complex)
        for i, bit in enumerate(bitstring):
            s = int(bit)
            mat = self.tensors[i][:, s, :]
            vec = vec @ mat

        return complex(vec[0, 0])

    def sample(self, shots: int = 1024, seed: Optional[int] = None) -> Dict[str, int]:
        """
        Sample measurement outcomes using sequential Born rule sampling.

        Algorithm (site-by-site):
        1. Compute reduced density matrix ρ₁ for site 0 → sample s₀
        2. Condition on s₀, compute ρ₂ for site 1 → sample s₁
        3. Repeat for all sites → one bitstring

        Cost per shot: O(n × χ² × d) — vastly cheaper than full statevector.

        Optimized: vectorized batch sampling with pre-computed left-canonical form.
        """
        rng = np.random.default_rng(seed)
        counts: Dict[str, int] = {}

        # Pre-compute left-canonical form once (amortized across all shots)
        mps_lc = self.copy()
        mps_lc.left_canonicalize()

        # Batch sampling: process multiple shots in parallel chunks
        # Using vectorized operations for the per-site probability computation
        batch_size = min(shots, 1024)  # Process in batches for memory efficiency (was 256)
        remaining = shots

        while remaining > 0:
            current_batch = min(batch_size, remaining)
            bitstrings = mps_lc._sample_batch(current_batch, rng)
            for bs in bitstrings:
                counts[bs] = counts.get(bs, 0) + 1
            remaining -= current_batch

        return counts

    def _sample_batch(self, batch_size: int, rng: np.random.Generator) -> List[str]:
        """Sample a batch of bitstrings using vectorized conditional sampling."""
        n = self.num_qubits
        d = self.physical_dim

        # Each sample maintains a boundary vector; batch into a matrix
        # vecs shape: (batch_size, chi) — chi starts at 1
        vecs = np.ones((batch_size, 1), dtype=complex)
        result_bits = [[] for _ in range(batch_size)]

        for i in range(n):
            t = self.tensors[i]  # (chi_l, d, chi_r)
            chi_l, _, chi_r = t.shape

            # For each physical state s, compute: new_vecs_s = vecs @ t[:, s, :]
            # t[:, s, :] has shape (chi_l, chi_r)
            # vecs has shape (batch, chi_l)
            # Compute probabilities for all s simultaneously
            probs = np.zeros((batch_size, d), dtype=float)
            new_vecs_all = np.zeros((d, batch_size, chi_r), dtype=complex)

            for s in range(d):
                mat = t[:, s, :]  # (chi_l, chi_r)
                nv = vecs @ mat   # (batch, chi_r)
                new_vecs_all[s] = nv
                probs[:, s] = np.real(np.sum(nv.conj() * nv, axis=1))

            # Normalize probabilities per sample
            totals = np.sum(probs, axis=1, keepdims=True)
            mask = totals.flatten() < 1e-30
            totals[mask] = 1.0
            probs[mask.flatten()] = 1.0 / d
            probs = probs / totals

            # Sample physical states for each batch element
            # Vectorized multinomial sampling
            cumprobs = np.cumsum(probs, axis=1)
            randoms = rng.random(batch_size)[:, np.newaxis]
            choices = np.argmax(cumprobs >= randoms, axis=1)  # (batch_size,)

            # Update boundary vectors based on chosen states
            new_vecs = np.zeros((batch_size, chi_r), dtype=complex)
            for s in range(d):
                mask_s = choices == s
                if np.any(mask_s):
                    new_vecs[mask_s] = new_vecs_all[s][mask_s]

            # Normalize
            norms = np.sqrt(np.real(np.sum(new_vecs.conj() * new_vecs, axis=1, keepdims=True)))
            norms = np.maximum(norms, 1e-30)
            vecs = new_vecs / norms

            # Record bits
            for b in range(batch_size):
                result_bits[b].append(str(choices[b]))

        return [''.join(bits) for bits in result_bits]

    def _sample_single(self, rng: np.random.Generator) -> str:
        """Sample a single bitstring via sequential conditional sampling (fallback)."""
        bits = []
        vec = np.array([1.0], dtype=complex)

        for i in range(self.num_qubits):
            t = self.tensors[i]
            chi_l, d, chi_r = t.shape

            probs = np.zeros(d, dtype=float)
            right_vecs = []

            for s in range(d):
                mat = t[:, s, :]
                new_vec = vec @ mat
                probs[s] = float(np.real(np.dot(new_vec.conj(), new_vec)))
                right_vecs.append(new_vec)

            total = np.sum(probs)
            if total < 1e-30:
                probs = np.ones(d) / d
                total = 1.0
            else:
                probs /= total

            s = int(rng.choice(d, p=probs))
            bits.append(str(s))

            vec = right_vecs[s]
            norm = np.sqrt(float(np.real(np.dot(vec.conj(), vec))))
            if norm > 1e-30:
                vec = vec / norm

        return ''.join(bits)

    def probabilities(self, threshold: float = 1e-10) -> Dict[str, float]:
        """
        Compute all probabilities above threshold.

        Warning: For n > 20, this enumerates 2^n states and may be slow.
        For large systems, use sample() instead.

        Cost: O(2^n × n × χ²)
        """
        if self.num_qubits > 24:
            raise ValueError(
                f"Full probability enumeration for {self.num_qubits} qubits "
                f"would require {2**self.num_qubits} amplitude evaluations. "
                f"Use sample() for large systems."
            )

        prob_dict: Dict[str, float] = {}
        for state_idx in range(2 ** self.num_qubits):
            bitstring = format(state_idx, f'0{self.num_qubits}b')
            p = self.get_probability(bitstring)
            if p > threshold:
                prob_dict[bitstring] = p

        return prob_dict

    def expectation_value(self, operator_matrices: List[Tuple[int, np.ndarray]]) -> complex:
        """
        Compute ⟨ψ|O|ψ⟩ for a product of local operators.

        operator_matrices: List of (qubit, 2×2 matrix) pairs.
        Qubits not in the list are acted on by identity.

        Cost: O(n × χ²) for product operators.
        """
        # Create a copy with operators applied
        bra = self.copy()
        for qubit, op_matrix in operator_matrices:
            bra.apply_single_qubit_gate(op_matrix, qubit)

        # Contract ⟨ψ_bra|ψ_ket⟩ = overlap
        return self._overlap(bra)

    def _overlap(self, other: 'MPSState') -> complex:
        """
        Compute inner product ⟨self|other⟩ by contracting the transfer matrices.

        Cost: O(n × χ² × χ'²) where χ, χ' are the bond dims of self and other.
        """
        if self.num_qubits != other.num_qubits:
            raise ValueError("MPS states must have same number of qubits")

        # Transfer matrix contraction, left to right
        # T[i] = Σ_s conj(A_self[s]) ⊗ A_other[s]
        # Start with (1, 1) boundary
        transfer = np.array([[1.0]], dtype=complex)  # (χ_self, χ_other)

        for i in range(self.num_qubits):
            t_self = self.tensors[i]    # (χ_L_s, d, χ_R_s)
            t_other = other.tensors[i]  # (χ_L_o, d, χ_R_o)

            # Contract: transfer[α,β] × Σ_s conj(t_self[α,s,α']) × t_other[β,s,β']
            # Result: new_transfer[α', β']
            new_transfer = np.einsum('ab,asc,bsd->cd',
                                      transfer,
                                      t_self.conj(),
                                      t_other)
            transfer = new_transfer

        return complex(transfer[0, 0])

    def norm(self) -> float:
        """Compute the norm ⟨ψ|ψ⟩ of the MPS state."""
        return float(np.sqrt(np.abs(self._overlap(self))))

    def normalize(self):
        """Normalize the MPS state to unit norm."""
        n = self.norm()
        if n > 1e-30:
            # Distribute normalization across first tensor
            self.tensors[0] = self.tensors[0] / n

    # ─── Entanglement Analysis ────────────────────────────────────────────────

    def bond_entropy(self, bond: int) -> float:
        """
        Compute the von Neumann entanglement entropy across bond `bond`
        (between sites bond and bond+1).

        Requires mixed canonical form with center at `bond`.

        S = -Σᵢ λᵢ² log(λᵢ²)
        """
        if bond < 0 or bond >= self.num_qubits - 1:
            raise ValueError(f"Bond {bond} out of range [0, {self.num_qubits - 2}]")

        # Put in mixed canonical form at the bond
        self.mixed_canonicalize(bond)

        # SVD of the center tensor to get Schmidt values
        t = self.tensors[bond]  # (χ_L, d, χ_R)
        chi_l, d, chi_r = t.shape
        mat = t.reshape(chi_l * d, chi_r)

        _, S, _ = np.linalg.svd(mat, full_matrices=False)

        # Normalize singular values
        S_sq = S ** 2
        total = np.sum(S_sq)
        if total < 1e-30:
            return 0.0
        S_sq /= total

        # Von Neumann entropy
        entropy = 0.0
        for s in S_sq:
            if s > 1e-30:
                entropy -= s * np.log2(s)

        return float(entropy)

    def all_bond_entropies(self) -> List[float]:
        """Compute entanglement entropy for every bond."""
        entropies = []
        for bond in range(self.num_qubits - 1):
            entropies.append(self.bond_entropy(bond))
        return entropies

    def entanglement_profile(self) -> Dict[str, Any]:
        """
        Full entanglement analysis of the MPS state.
        """
        entropies = self.all_bond_entropies()
        bond_dims = self.bond_dimensions

        return {
            "bond_entropies": entropies,
            "bond_dimensions": bond_dims,
            "max_entropy": max(entropies) if entropies else 0.0,
            "mean_entropy": sum(entropies) / len(entropies) if entropies else 0.0,
            "max_bond_dim": max(bond_dims) if bond_dims else 1,
            "mean_bond_dim": sum(bond_dims) / len(bond_dims) if bond_dims else 1,
            "total_parameters": self.total_parameters,
            "memory_mb": self.memory_mb,
            "compression_ratio": self.compression_ratio,
        }

    # ─── State Extraction ─────────────────────────────────────────────────────

    def to_statevector(self) -> np.ndarray:
        """
        Convert MPS back to full statevector (for verification, small systems only).

        Warning: O(2^n) memory — only for n ≤ 25.
        """
        if self.num_qubits > 25:
            raise ValueError(
                f"Cannot convert {self.num_qubits}-qubit MPS to statevector "
                f"(would need {2**self.num_qubits * 16 / 1e9:.1f} GB)"
            )

        dim = 2 ** self.num_qubits
        sv = np.zeros(dim, dtype=complex)

        for state_idx in range(dim):
            bitstring = format(state_idx, f'0{self.num_qubits}b')
            sv[state_idx] = self.get_amplitude(bitstring)

        return sv

    def to_dict(self) -> Dict[str, Any]:
        """Serialize MPS metadata (not tensor data)."""
        return {
            "num_qubits": self.num_qubits,
            "max_bond_dim": self.max_bond_dim,
            "current_bond_dims": self.bond_dimensions,
            "max_current_bond_dim": self.max_current_bond_dim,
            "total_parameters": self.total_parameters,
            "memory_mb": round(self.memory_mb, 4),
            "statevector_memory_mb": round(self.statevector_memory_mb, 2),
            "compression_ratio": round(self.compression_ratio, 1),
            "cumulative_truncation_error": self.cumulative_truncation_error,
            "gate_count": self._gate_count,
            "canonical_form": self.canonical_form.name,
        }

    def __repr__(self) -> str:
        dims = self.bond_dimensions
        max_chi = max(dims) if dims else 1
        return (f"MPSState(n={self.num_qubits}, χ_max={max_chi}, "
                f"params={self.total_parameters}, "
                f"mem={self.memory_mb:.2f}MB, "
                f"trunc_err={self.cumulative_truncation_error:.2e})")


# ═══════════════════════════════════════════════════════════════════════════════
#  TENSOR NETWORK SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TNSimulationResult:
    """Result of a tensor network simulation."""
    probabilities: Optional[Dict[str, float]] = None
    counts: Optional[Dict[str, int]] = None
    statevector: Optional[np.ndarray] = None
    mps_state: Optional[MPSState] = None
    execution_time: float = 0.0
    shots: int = 0
    fidelity_estimate: float = 1.0
    truncation_error: float = 0.0
    bond_dimensions: Optional[List[int]] = None
    max_bond_dim_reached: int = 1
    memory_mb: float = 0.0
    compression_ratio: float = 1.0
    entanglement_entropy: Optional[List[float]] = None
    sacred_alignment: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_time_ms": round(self.execution_time * 1000, 2),
            "shots": self.shots,
            "fidelity_estimate": round(self.fidelity_estimate, 8),
            "truncation_error": self.truncation_error,
            "bond_dimensions": self.bond_dimensions,
            "max_bond_dim_reached": self.max_bond_dim_reached,
            "memory_mb": round(self.memory_mb, 4),
            "compression_ratio": round(self.compression_ratio, 1),
            "num_probabilities": len(self.probabilities) if self.probabilities else 0,
            "sacred_alignment": self.sacred_alignment,
            "metadata": self.metadata,
        }


class TensorNetworkSimulator:
    """
    MPS-based quantum circuit simulator for the L104 Quantum Gate Engine.

    Enables simulation of 25-26 qubit circuits with 100-1000x memory reduction
    compared to full statevector. Integrates with the CrossSystemOrchestrator
    as a new ExecutionTarget.

    Features:
    - Adaptive bond dimension management
    - Sacred truncation mode (φ-balanced fidelity/compression)
    - Real-time entanglement tracking
    - Born rule sampling for measurement
    - Fidelity estimation via truncation error tracking
    - GOD_CODE sacred alignment scoring

    Usage:
        from l104_quantum_gate_engine.tensor_network import TensorNetworkSimulator
        sim = TensorNetworkSimulator(max_bond_dim=1024)
        result = sim.simulate(circuit, shots=1024)
    """

    VERSION = "1.0.0"

    def __init__(self, max_bond_dim: int = DEFAULT_MAX_BOND_DIM,
                 svd_cutoff: float = DEFAULT_SVD_CUTOFF,
                 truncation_mode: TruncationMode = TruncationMode.ADAPTIVE,
                 sacred_mode: bool = False):
        """
        Args:
            max_bond_dim: Maximum bond dimension χ (higher = more accurate, more memory)
            svd_cutoff: Minimum singular value threshold
            truncation_mode: SVD truncation strategy
            sacred_mode: Enable GOD_CODE-aligned truncation at χ=104
        """
        if sacred_mode:
            self.max_bond_dim = SACRED_BOND_DIM  # 104
            self.truncation_mode = TruncationMode.SACRED
        else:
            self.max_bond_dim = max_bond_dim
            self.truncation_mode = truncation_mode

        self.svd_cutoff = svd_cutoff

        # Telemetry
        self._simulations_run: int = 0
        self._total_gates_applied: int = 0
        self._total_truncation_error: float = 0.0

    def simulate(self, circuit, shots: int = 1024,
                 return_mps: bool = False,
                 return_statevector: bool = False,
                 compute_entanglement: bool = True,
                 seed: Optional[int] = None) -> TNSimulationResult:
        """
        Simulate a GateCircuit using MPS tensor network contraction.

        Args:
            circuit: GateCircuit to simulate
            shots: Number of measurement shots for sampling
            return_mps: Include the final MPS state in results
            return_statevector: Convert MPS to statevector (small circuits only)
            compute_entanglement: Compute bond entanglement entropies
            seed: Random seed for reproducible sampling

        Returns:
            TNSimulationResult with probabilities, counts, and diagnostics
        """
        start_time = time.time()

        # Initialize MPS in |00...0⟩
        mps = MPSState(
            num_qubits=circuit.num_qubits,
            max_bond_dim=self.max_bond_dim,
            svd_cutoff=self.svd_cutoff,
        )

        # Apply gates sequentially
        for op in circuit.operations:
            if op.label == "BARRIER":
                continue
            self._apply_gate_operation(mps, op)

        # Normalize
        mps.normalize()

        # Build result
        result = TNSimulationResult()
        result.mps_state = mps if return_mps else None
        result.truncation_error = mps.cumulative_truncation_error
        result.fidelity_estimate = max(0.0, 1.0 - mps.cumulative_truncation_error)
        result.bond_dimensions = mps.bond_dimensions
        result.max_bond_dim_reached = mps.max_current_bond_dim
        result.memory_mb = mps.memory_mb
        result.compression_ratio = mps.compression_ratio

        # Sample measurements
        if shots > 0:
            counts = mps.sample(shots=shots, seed=seed)
            result.counts = counts
            result.shots = shots
            total = sum(counts.values())
            result.probabilities = {k: v / total for k, v in counts.items()}

        # Statevector conversion (small circuits)
        if return_statevector and circuit.num_qubits <= 25:
            try:
                result.statevector = mps.to_statevector()
            except Exception:
                pass

        # Entanglement analysis
        if compute_entanglement and circuit.num_qubits > 1:
            try:
                result.entanglement_entropy = mps.all_bond_entropies()
            except Exception:
                pass

        # Sacred alignment
        result.sacred_alignment = self._compute_sacred_alignment(circuit, mps)

        result.execution_time = time.time() - start_time
        result.metadata = {
            "simulator": "l104_tensor_network_mps_v1.0.0",
            "max_bond_dim_setting": self.max_bond_dim,
            "svd_cutoff": self.svd_cutoff,
            "truncation_mode": self.truncation_mode.name,
            "num_qubits": circuit.num_qubits,
            "num_gates": circuit.num_operations,
            "circuit_depth": circuit.depth,
            "god_code": GOD_CODE,
        }

        # Update telemetry
        self._simulations_run += 1
        self._total_gates_applied += circuit.num_operations
        self._total_truncation_error += mps.cumulative_truncation_error

        return result

    def simulate_probabilities(self, circuit,
                                threshold: float = 1e-10) -> Dict[str, float]:
        """
        Compute exact probabilities (no sampling noise) for small circuits.

        For circuits ≤ 24 qubits, enumerates all amplitudes.
        For larger circuits, uses high-shot sampling.
        """
        mps = MPSState(
            num_qubits=circuit.num_qubits,
            max_bond_dim=self.max_bond_dim,
            svd_cutoff=self.svd_cutoff,
        )

        for op in circuit.operations:
            if op.label == "BARRIER":
                continue
            self._apply_gate_operation(mps, op)

        mps.normalize()

        if circuit.num_qubits <= 24:
            return mps.probabilities(threshold=threshold)
        else:
            # Use high-shot sampling for approximation
            counts = mps.sample(shots=100_000)
            total = sum(counts.values())
            return {k: v / total for k, v in counts.items()}

    def _apply_gate_operation(self, mps: MPSState, op):
        """Apply a GateOperation to the MPS state."""
        gate = op.gate
        qubits = op.qubits
        k = gate.num_qubits

        if k == 1:
            mps.apply_single_qubit_gate(gate.matrix, qubits[0])
        elif k == 2:
            mps.apply_two_qubit_gate(gate.matrix, qubits[0], qubits[1],
                                      mode=self.truncation_mode)
        else:
            mps.apply_multi_qubit_gate(gate.matrix, qubits,
                                        mode=self.truncation_mode)

    def _compute_sacred_alignment(self, circuit, mps: MPSState) -> Dict[str, float]:
        """Compute L104 sacred alignment scores for the MPS simulation."""
        scores = {
            "phi_alignment": 0.0,
            "god_code_alignment": 0.0,
            "iron_alignment": 0.0,
            "bond_dim_sacred": 0.0,
            "entropy_golden_ratio": 0.0,
            "total_sacred_resonance": 0.0,
        }

        # Bond dimension alignment with sacred numbers
        max_chi = mps.max_current_bond_dim
        if max_chi == 104 or max_chi == 26 or max_chi == 13:
            scores["bond_dim_sacred"] = 1.0
        elif max_chi % 13 == 0:
            scores["bond_dim_sacred"] = 0.8
        elif max_chi % 8 == 0:
            scores["bond_dim_sacred"] = 0.5

        # Entropy golden ratio check
        try:
            entropies = mps.all_bond_entropies()
            if entropies:
                max_ent = max(entropies)
                mean_ent = sum(entropies) / len(entropies)
                if max_ent > 0:
                    ratio = mean_ent / max_ent
                    # How close is the ratio to 1/φ ≈ 0.618?
                    phi_deviation = abs(ratio - PHI_TRUNCATION_BALANCE)
                    scores["entropy_golden_ratio"] = max(0, 1.0 - phi_deviation * 5)
        except Exception:
            pass

        # Sacred gate density
        sacred_count = sum(1 for op_item in circuit.operations
                          if hasattr(op_item, 'gate') and
                          hasattr(op_item.gate, 'gate_type') and
                          str(op_item.gate.gate_type) == 'GateType.SACRED')

        if circuit.num_operations > 0:
            scores["phi_alignment"] = min(1.0, sacred_count / max(1, circuit.num_operations) * PHI)
            scores["god_code_alignment"] = min(1.0, sacred_count / max(1, circuit.num_operations))

        # Iron alignment: 26-qubit circuits are Fe(26) aligned
        if circuit.num_qubits == IRON_ATOMIC_NUMBER:
            scores["iron_alignment"] = 1.0
        elif circuit.num_qubits == 25:
            scores["iron_alignment"] = 0.9  # Close to Fe(26)

        scores["total_sacred_resonance"] = (
            scores["phi_alignment"] * 0.2 +
            scores["god_code_alignment"] * 0.2 +
            scores["iron_alignment"] * 0.2 +
            scores["bond_dim_sacred"] * 0.2 +
            scores["entropy_golden_ratio"] * 0.2
        )

        return scores

    # ─── Utility Methods ──────────────────────────────────────────────────────

    def estimate_memory(self, num_qubits: int, max_bond_dim: Optional[int] = None) -> Dict[str, float]:
        """
        Estimate memory usage for an MPS simulation.

        Returns comparison of MPS vs full statevector memory.
        """
        chi = max_bond_dim or self.max_bond_dim
        d = PHYSICAL_DIM

        # MPS memory: n sites × chi² × d × 16 bytes (complex128)
        mps_params = num_qubits * chi * chi * d
        mps_bytes = mps_params * 16
        mps_mb = mps_bytes / (1024 * 1024)

        # Statevector memory: 2^n × 16 bytes
        sv_states = 2 ** num_qubits
        sv_bytes = sv_states * 16
        sv_mb = sv_bytes / (1024 * 1024)

        return {
            "num_qubits": num_qubits,
            "max_bond_dim": chi,
            "mps_parameters": mps_params,
            "mps_memory_mb": round(mps_mb, 4),
            "statevector_states": sv_states,
            "statevector_memory_mb": round(sv_mb, 2),
            "compression_ratio": round(sv_mb / max(mps_mb, 0.001), 1),
            "feasible_mps": mps_mb < 4096,      # Under 4GB
            "feasible_statevector": sv_mb < 4096, # Under 4GB
        }

    def benchmark_bond_dims(self, circuit) -> List[Dict[str, Any]]:
        """
        Run simulation at multiple bond dimensions to show accuracy/speed tradeoff.
        """
        results = []
        test_dims = [8, 16, 32, 64, 128, 256]
        test_dims = [d for d in test_dims if d <= self.max_bond_dim * 2]

        for chi in test_dims:
            sim = TensorNetworkSimulator(
                max_bond_dim=chi,
                svd_cutoff=self.svd_cutoff,
                truncation_mode=self.truncation_mode,
            )
            r = sim.simulate(circuit, shots=0, return_mps=False,
                             compute_entanglement=False)
            results.append({
                "max_bond_dim": chi,
                "actual_max_bond_dim": r.max_bond_dim_reached,
                "truncation_error": r.truncation_error,
                "fidelity_estimate": r.fidelity_estimate,
                "memory_mb": r.memory_mb,
                "execution_time_ms": round(r.execution_time * 1000, 2),
            })

        return results

    def status(self) -> Dict[str, Any]:
        """Simulator status and configuration."""
        return {
            "version": self.VERSION,
            "max_bond_dim": self.max_bond_dim,
            "svd_cutoff": self.svd_cutoff,
            "truncation_mode": self.truncation_mode.name,
            "max_qubits_supported": MAX_TENSOR_NETWORK_QUBITS,
            "simulations_run": self._simulations_run,
            "total_gates_applied": self._total_gates_applied,
            "total_truncation_error": self._total_truncation_error,
            "memory_estimates": {
                "25q_chi64": self.estimate_memory(25, 64),
                "25q_chi256": self.estimate_memory(25, 256),
                "26q_chi64": self.estimate_memory(26, 64),
                "26q_chi256": self.estimate_memory(26, 256),
            },
            "god_code": GOD_CODE,
            "sacred_bond_dim": SACRED_BOND_DIM,
        }

    def __repr__(self) -> str:
        return (f"TensorNetworkSimulator(v{self.VERSION}, "
                f"χ_max={self.max_bond_dim}, "
                f"mode={self.truncation_mode.name}, "
                f"runs={self._simulations_run})")


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE-LEVEL CONVENIENCE
# ═══════════════════════════════════════════════════════════════════════════════

# Singleton instances for common configurations
_default_simulator: Optional[TensorNetworkSimulator] = None
_sacred_simulator: Optional[TensorNetworkSimulator] = None
_high_fidelity_simulator: Optional[TensorNetworkSimulator] = None


def get_simulator(mode: str = "default") -> TensorNetworkSimulator:
    """
    Get a singleton TensorNetworkSimulator instance.

    Modes:
        "default"       — χ=256, adaptive truncation
        "sacred"        — χ=104, φ-balanced truncation
        "high_fidelity" — χ=512, tight SVD cutoff
        "fast"          — χ=64, for rapid prototyping
    """
    global _default_simulator, _sacred_simulator, _high_fidelity_simulator

    if mode == "default":
        if _default_simulator is None:
            _default_simulator = TensorNetworkSimulator(
                max_bond_dim=DEFAULT_MAX_BOND_DIM,
                truncation_mode=TruncationMode.ADAPTIVE,
            )
        return _default_simulator
    elif mode == "sacred":
        if _sacred_simulator is None:
            _sacred_simulator = TensorNetworkSimulator(sacred_mode=True)
        return _sacred_simulator
    elif mode == "high_fidelity":
        if _high_fidelity_simulator is None:
            _high_fidelity_simulator = TensorNetworkSimulator(
                max_bond_dim=HIGH_FIDELITY_BOND_DIM,
                svd_cutoff=HIGH_FIDELITY_SVD_CUTOFF,
                truncation_mode=TruncationMode.ADAPTIVE,
            )
        return _high_fidelity_simulator
    elif mode == "fast":
        return TensorNetworkSimulator(max_bond_dim=64)
    else:
        raise ValueError(f"Unknown simulator mode: {mode}. Use default/sacred/high_fidelity/fast")
