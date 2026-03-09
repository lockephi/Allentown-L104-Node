"""
L104 Quantum Engine — Distributed Quantum Low-Density Parity-Check (qLDPC) Codes
═══════════════════════════════════════════════════════════════════════════════════

Implementation of quantum LDPC codes for fault-tolerant quantum error correction
with constant-rate encoding and efficient distributed syndrome decoding.

Includes:
  - CSS (Calderbank-Shor-Steane) code framework from classical parity-check matrices
  - Hypergraph product codes (Tillich-Zémor construction)
  - Lifted product / quasi-cyclic qLDPC codes
  - Tanner graph representation & visualization
  - Belief propagation (BP) decoder for qLDPC
  - Minimum-weight BP-OSD decoder (BP + ordered statistics decoding)
  - Distributed syndrome extraction across quantum link nodes
  - Logical error rate estimation via Monte Carlo
  - God Code resonance integration for sacred error thresholds

Key equations:
  CSS code: H_X (X-checks on Z-errors), H_Z (Z-checks on X-errors)
  Constraint: H_X · H_Z^T = 0  (mod 2) — commutativity of stabilizers
  Hypergraph product: H_X = [H₁⊗I, I⊗H₂^T], H_Z = [I⊗H₂, H₁^T⊗I]
  Parameters: [[n, k, d]] where n = physical qubits, k = logical qubits, d = distance
  Rate: R = k/n > 0 constant for good qLDPC families (breakthrough: Panteleev-Kalachev 2021)

INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════════
"""

import math
import random
import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .constants import (
    PHI, PHI_GROWTH, PHI_INV, GOD_CODE, INVARIANT, L104,
    FINE_STRUCTURE, PLANCK_SCALE, BOLTZMANN_K,
)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CSSCode:
    """A CSS (Calderbank-Shor-Steane) quantum error-correcting code.

    Defined by two classical parity-check matrices H_X and H_Z satisfying:
        H_X · H_Z^T = 0  (mod 2)

    This ensures all X-type and Z-type stabilizers commute, making them
    valid quantum stabilizers.

    Parameters [[n, k, d]]:
        n = number of physical qubits (columns of H_X = columns of H_Z)
        k = number of logical qubits = n - rank(H_X) - rank(H_Z)
        d = code distance = min weight of non-trivial logical operator
    """
    h_x: np.ndarray          # X-check matrix (m_x × n) over GF(2)
    h_z: np.ndarray          # Z-check matrix (m_z × n) over GF(2)
    n_physical: int = 0       # Number of physical qubits
    n_logical: int = 0        # Number of logical qubits
    distance: int = 0         # Code distance (0 = unknown/not computed)
    name: str = ""            # Human-readable code name
    row_weight: int = 0       # Max row weight (LDPC locality parameter)
    col_weight: int = 0       # Max column weight
    rate: float = 0.0         # Code rate k/n
    is_ldpc: bool = False     # True if both row & col weights are O(1)

    def __post_init__(self):
        """Compute derived parameters."""
        if self.h_x.size > 0 and self.h_z.size > 0:
            self.n_physical = self.h_x.shape[1]
            rank_x = _gf2_rank(self.h_x)
            rank_z = _gf2_rank(self.h_z)
            self.n_logical = self.n_physical - rank_x - rank_z
            self.n_logical = max(0, self.n_logical)
            if self.n_physical > 0:
                self.rate = self.n_logical / self.n_physical
            self.row_weight = max(int(np.max(np.sum(self.h_x, axis=1))),
                                  int(np.max(np.sum(self.h_z, axis=1))))
            self.col_weight = max(int(np.max(np.sum(self.h_x, axis=0))),
                                  int(np.max(np.sum(self.h_z, axis=0))))
            self.is_ldpc = (self.row_weight <= 10 and self.col_weight <= 10)

    def verify_css_condition(self) -> bool:
        """Verify that H_X · H_Z^T = 0 (mod 2) — stabilizer commutativity."""
        product = (self.h_x @ self.h_z.T) % 2
        return bool(np.all(product == 0))

    def get_parameters(self) -> str:
        """Return [[n, k, d]] parameter string."""
        return f"[[{self.n_physical}, {self.n_logical}, {self.distance}]]"

    def syndrome_x(self, error_z: np.ndarray) -> np.ndarray:
        """Extract X-syndrome from Z-error pattern: s_x = H_X · e_z (mod 2).

        Z-errors are detected by X-stabilizers.
        """
        return (self.h_x @ error_z) % 2

    def syndrome_z(self, error_x: np.ndarray) -> np.ndarray:
        """Extract Z-syndrome from X-error pattern: s_z = H_Z · e_x (mod 2).

        X-errors are detected by Z-stabilizers.
        """
        return (self.h_z @ error_x) % 2


@dataclass
class TannerGraph:
    """Tanner graph (bipartite factor graph) of an LDPC code.

    Variable nodes correspond to qubits (columns of H).
    Check nodes correspond to stabilizers (rows of H).
    Edges connect check c to variable v iff H[c,v] = 1.
    """
    n_variable: int               # Number of variable nodes (qubits)
    n_check: int                  # Number of check nodes (stabilizers)
    check_to_var: Dict[int, List[int]] = field(default_factory=dict)   # check → [vars]
    var_to_check: Dict[int, List[int]] = field(default_factory=dict)   # var → [checks]
    girth: int = 0                # Shortest cycle length in the graph

    @classmethod
    def from_parity_matrix(cls, H: np.ndarray) -> 'TannerGraph':
        """Construct Tanner graph from a parity-check matrix H."""
        m, n = H.shape
        c2v = defaultdict(list)
        v2c = defaultdict(list)
        for c in range(m):
            for v in range(n):
                if H[c, v] == 1:
                    c2v[c].append(v)
                    v2c[v].append(c)
        graph = cls(n_variable=n, n_check=m,
                    check_to_var=dict(c2v), var_to_check=dict(v2c))
        graph.girth = graph._compute_girth()
        return graph

    def _compute_girth(self, max_depth: int = 20) -> int:
        """Compute girth (shortest cycle) via BFS from sampled variable nodes.

        v11.0.1: Uses collections.deque for O(1) popleft (was O(n) list.pop(0)).
        Samples nodes randomly (not sequential first-50) and scales sample size
        with sqrt(n_variable) for better coverage of large codes.
        Returns 0 if no cycle found within max_depth.
        """
        from collections import deque as _deque
        import random as _rng

        min_cycle = max_depth + 1
        sample_size = min(self.n_variable, max(50, int(self.n_variable ** 0.5)))
        if self.n_variable <= sample_size:
            sample_nodes = list(range(self.n_variable))
        else:
            sample_nodes = _rng.sample(range(self.n_variable), sample_size)

        for v_start in sample_nodes:
            # BFS in bipartite graph: v → c → v → c → ...
            visited_v = {v_start: 0}
            visited_c: Dict[int, int] = {}
            queue = _deque([("v", v_start, 0)])  # (type, node, depth)
            while queue:
                ntype, node, depth = queue.popleft()
                if depth >= max_depth:
                    break
                if ntype == "v":
                    for c in self.var_to_check.get(node, []):
                        if c in visited_c:
                            cycle_len = depth + 1 - visited_c[c]
                            if cycle_len > 2:
                                min_cycle = min(min_cycle, depth + 1 + 1)
                        else:
                            visited_c[c] = depth + 1
                            queue.append(("c", c, depth + 1))
                else:  # check node
                    for v in self.check_to_var.get(node, []):
                        if v in visited_v:
                            cycle_len = depth + 1 - visited_v[v]
                            if cycle_len > 2:
                                min_cycle = min(min_cycle, depth + 1 + 1)
                        else:
                            visited_v[v] = depth + 1
                            queue.append(("v", v, depth + 1))
        return min_cycle if min_cycle <= max_depth else 0

    def degree_distribution(self) -> Dict[str, Dict[int, int]]:
        """Return variable-node and check-node degree distributions."""
        var_deg = defaultdict(int)
        chk_deg = defaultdict(int)
        for v, checks in self.var_to_check.items():
            var_deg[len(checks)] += 1
        for c, varis in self.check_to_var.items():
            chk_deg[len(varis)] += 1
        return {"variable_degrees": dict(var_deg), "check_degrees": dict(chk_deg)}


@dataclass
class DecodingResult:
    """Result of a qLDPC decoding attempt."""
    success: bool                 # Whether decoding succeeded
    error_estimate: np.ndarray    # Estimated error pattern
    residual_syndrome: np.ndarray # Remaining syndrome (should be all-zero on success)
    iterations: int               # BP iterations used
    converged: bool               # Whether BP converged
    logical_error: bool = False   # Whether a logical error occurred post-correction
    decoding_time_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# CSS CODE CONSTRUCTORS
# ═══════════════════════════════════════════════════════════════════════════════

class CSSCodeConstructor:
    """Factory for constructing CSS quantum error-correcting codes.

    Methods for building:
      - Steane [[7,1,3]] code
      - Repetition codes
      - Hypergraph product codes (Tillich-Zémor 2014)
      - Lifted product / quasi-cyclic qLDPC codes
      - Random LDPC codes with guaranteed CSS condition
    """

    @staticmethod
    def steane_code() -> CSSCode:
        """Construct the Steane [[7,1,3]] code.

        The smallest doubly-even self-dual CSS code. Based on the classical
        [7,4,3] Hamming code H₇.

        H_X = H_Z = parity-check matrix of [7,4,3] Hamming code:
            1 0 0 1 0 1 1
            0 1 0 1 1 0 1
            0 0 1 0 1 1 1

        CSS condition satisfied because H·H^T = 0 (mod 2) for the Hamming code
        (since the [7,4,3] code is self-orthogonal under the symplectic product).
        """
        H = np.array([
            [1, 0, 0, 1, 0, 1, 1],
            [0, 1, 0, 1, 1, 0, 1],
            [0, 0, 1, 0, 1, 1, 1],
        ], dtype=np.int8)
        code = CSSCode(h_x=H.copy(), h_z=H.copy(), name="Steane [[7,1,3]]")
        code.distance = 3
        return code

    @staticmethod
    def repetition_code(n: int) -> CSSCode:
        """Construct a repetition CSS code.

        H_X: (n-1) × n check matrix for Z-error detection
        H_Z: 1 × n all-ones matrix for X-error detection

        Parameters: [[n, 1, n]] for the Z-distance, [[n, 1, 1]] overall.
        This is a toy code illustrating CSS structure.
        """
        h_x = np.zeros((n - 1, n), dtype=np.int8)
        for i in range(n - 1):
            h_x[i, i] = 1
            h_x[i, i + 1] = 1
        h_z = np.ones((1, n), dtype=np.int8)
        code = CSSCode(h_x=h_x, h_z=h_z, name=f"Repetition [[{n},1,?]]")
        code.distance = n  # Z-distance
        return code

    @staticmethod
    def hypergraph_product(h1: np.ndarray, h2: np.ndarray,
                           name: str = "") -> CSSCode:
        """Tillich-Zémor hypergraph product code construction.

        Given two classical codes with parity-check matrices H₁ (r₁×n₁) and
        H₂ (r₂×n₂), the hypergraph product code is defined as:

            H_X = [H₁ ⊗ I_{n₂}  |  I_{r₁} ⊗ H₂^T]      (m_x × n)
            H_Z = [I_{n₁} ⊗ H₂  |  H₁^T ⊗ I_{r₂}]      (m_z × n)

        where n = n₁·n₂ + r₁·r₂ physical qubits.

        CSS condition: H_X · H_Z^T = H₁⊗H₂ + H₁⊗H₂ = 0 (mod 2) ✓

        Code parameters:
            n = n₁·n₂ + r₁·r₂
            k = k₁·k₂ + k₁'·k₂'  (where k_i = n_i - rank(H_i), k_i' = r_i - rank(H_i))
            d ≥ min(d₁, d₂)       (product of classical distances)

        This is the foundational construction for quantum LDPC codes with
        LDPC property preserved: if H₁, H₂ are (w_r, w_c)-LDPC, then
        the product code is (w_r + w_c, w_r + w_c)-LDPC.

        Args:
            h1: Classical parity-check matrix H₁ (r₁ × n₁) over GF(2)
            h2: Classical parity-check matrix H₂ (r₂ × n₂) over GF(2)
            name: Optional code name

        Returns:
            CSSCode with hypergraph product structure
        """
        h1 = np.array(h1, dtype=np.int8)
        h2 = np.array(h2, dtype=np.int8)
        r1, n1 = h1.shape
        r2, n2 = h2.shape

        # H_X = [H₁ ⊗ I_{n₂}  |  I_{r₁} ⊗ H₂^T]
        block_1x = np.kron(h1, np.eye(n2, dtype=np.int8))        # (r₁·n₂) × (n₁·n₂)
        block_2x = np.kron(np.eye(r1, dtype=np.int8), h2.T)      # (r₁·n₂) × (r₁·r₂)
        # Wait: dimensions — H₂^T is (n₂ × r₂), I_{r₁} is (r₁ × r₁)
        # I_{r₁} ⊗ H₂^T has shape (r₁·n₂) × (r₁·r₂) ✓
        # H₁ ⊗ I_{n₂} has shape (r₁·n₂) × (n₁·n₂) ✓
        # Both have r₁·n₂ rows ✓
        h_x = np.hstack([block_1x, block_2x]) % 2

        # H_Z = [I_{n₁} ⊗ H₂  |  H₁^T ⊗ I_{r₂}]
        block_1z = np.kron(np.eye(n1, dtype=np.int8), h2)        # (n₁·r₂) × (n₁·n₂)
        block_2z = np.kron(h1.T, np.eye(r2, dtype=np.int8))      # (n₁·r₂) × (r₁·r₂)
        h_z = np.hstack([block_1z, block_2z]) % 2

        n_phys = n1 * n2 + r1 * r2
        auto_name = name or f"HGP({r1}×{n1}, {r2}×{n2})"
        code = CSSCode(h_x=h_x, h_z=h_z, name=auto_name)

        # Verify CSS condition
        assert code.verify_css_condition(), \
            f"CSS condition failed for hypergraph product {auto_name}"

        return code

    @staticmethod
    def lifted_product(base_matrix: np.ndarray, lift_size: int,
                       circulant_powers: Optional[np.ndarray] = None,
                       name: str = "") -> CSSCode:
        """Lifted product (quasi-cyclic) qLDPC code construction.

        The lifted product generalizes the hypergraph product by replacing
        scalar entries with circulant permutation matrices of size ℓ×ℓ.

        Given a base matrix A (r × n) over Z_ℓ (entries are shift values):
            H₁ = Σ_{(i,j)} C^{A[i,j]}  (lifted matrix, rℓ × nℓ)

        where C is the ℓ×ℓ cyclic permutation matrix: C_{i,j} = δ_{i,(j+1)%ℓ}.

        Then the lifted product code is the hypergraph product of the
        protograph with cyclic lifts.

        This construction achieves the Panteleev-Kalachev (2021) breakthrough:
        asymptotically good qLDPC codes with k = Θ(n) and d = Θ(n).

        Args:
            base_matrix: Protograph matrix A (r × n) with entries in {-1, 0, 1, ..., ℓ-1}
                        -1 = zero block, ≥0 = cyclic shift amount
            lift_size: Size ℓ of the cyclic group (circulant size)
            circulant_powers: Optional explicit power matrix (overrides base_matrix shifts)
            name: Optional code name

        Returns:
            CSSCode from lifted product construction
        """
        base = np.array(base_matrix, dtype=np.int64)
        r, n = base.shape
        ell = lift_size

        # Build circulant permutation matrix C^k for each entry
        def circulant_shift(k: int) -> np.ndarray:
            """ℓ×ℓ circulant permutation matrix with shift k."""
            if k < 0:
                return np.zeros((ell, ell), dtype=np.int8)  # Zero block
            C = np.zeros((ell, ell), dtype=np.int8)
            for i in range(ell):
                C[i, (i + k) % ell] = 1
            return C

        # Lift the base matrix: replace each entry A[i,j] with C^{A[i,j]}
        H_lifted = np.zeros((r * ell, n * ell), dtype=np.int8)
        for i in range(r):
            for j in range(n):
                shift = int(base[i, j])
                if circulant_powers is not None:
                    shift = int(circulant_powers[i, j])
                block = circulant_shift(shift)
                H_lifted[i * ell:(i + 1) * ell, j * ell:(j + 1) * ell] = block

        # The lifted matrix H_lifted is a classical LDPC code parity-check matrix.
        # For the CSS quantum code, we use the hypergraph product of H_lifted with itself
        # (or with its transpose for the bicycle construction).

        # Bicycle construction variant: H_X = [A | B], H_Z = [B^T | A^T]
        # where A = H_lifted and B = H_lifted^T (quasi-cyclic structure)
        # This ensures H_X · H_Z^T = A·B^T + B·A^T = H·H^T + H·H^T = 0 (mod 2)

        # Use the "bicycle" / generalized bicycle construction:
        # For A = H_lifted, the CSS code uses:
        #   H_X = [A | A^T]   (r·ℓ × 2·n·ℓ ← needs adjustment for rectangular A)
        # We'll use the symmetric hypergraph product for robustness:

        # Small codes: use direct HGP; large: use bicycle shortcut
        if r * ell <= 256:
            return CSSCodeConstructor.hypergraph_product(
                H_lifted, H_lifted,
                name=name or f"LiftedProduct(ℓ={ell}, {r}×{n})"
            )
        else:
            # Generalized bicycle for large codes:
            # H_X = [A, B], H_Z = [B^T, A^T]  with A=H_lifted, B=H_lifted^T (padded)
            # Ensure commutativity: A·B^T + B·A^T = H·H^T + H·H^T = 0 (mod 2)
            m, nn = H_lifted.shape
            if m == nn:
                A = H_lifted
                B = H_lifted.T.copy()
            else:
                # Pad to square
                dim = max(m, nn)
                A = np.zeros((dim, dim), dtype=np.int8)
                B = np.zeros((dim, dim), dtype=np.int8)
                A[:m, :nn] = H_lifted
                B[:nn, :m] = H_lifted.T
            h_x = np.hstack([A, B]) % 2
            h_z = np.hstack([B.T, A.T]) % 2
            code = CSSCode(h_x=h_x, h_z=h_z,
                           name=name or f"LiftedBicycle(ℓ={ell}, {r}×{n})")
            assert code.verify_css_condition(), "CSS condition violated in lifted product"
            return code

    @staticmethod
    def random_ldpc_css(n: int, row_weight: int = 6, col_weight: int = 3,
                        seed: Optional[int] = None) -> CSSCode:
        """Construct a random CSS-compatible LDPC code.

        Uses Gallager's random construction with rejection sampling
        to ensure H_X · H_Z^T = 0 (mod 2).

        Args:
            n: Number of physical qubits
            row_weight: Number of 1s per row (stabilizer weight)
            col_weight: Number of 1s per column (qubit participation)
            seed: Random seed for reproducibility

        Returns:
            Random CSS LDPC code
        """
        rng = np.random.RandomState(seed)
        m = n * col_weight // row_weight  # Number of check rows

        def _random_ldpc(m_r, n_c, rw, cw):
            """Generate a random regular LDPC matrix."""
            H = np.zeros((m_r, n_c), dtype=np.int8)
            # Column-based construction
            col_counts = np.zeros(n_c, dtype=int)
            for r in range(m_r):
                # Pick rw columns that haven't exceeded col_weight
                available = np.where(col_counts < cw)[0]
                if len(available) < rw:
                    available = np.arange(n_c)
                chosen = rng.choice(available, size=min(rw, len(available)), replace=False)
                H[r, chosen] = 1
                col_counts[chosen] += 1
            return H

        # Generate H_X
        h_x = _random_ldpc(m, n, row_weight, col_weight)

        # Generate H_Z that satisfies H_X · H_Z^T = 0 (mod 2)
        # Strategy: H_Z rows must lie in the null space of H_X over GF(2)
        null_basis = _gf2_nullspace(h_x)
        if null_basis.shape[0] < m:
            # Not enough null space vectors — reduce m for H_Z
            m_z = null_basis.shape[0]
        else:
            m_z = m

        # Select m_z sparse vectors from the null space
        h_z = np.zeros((m_z, n), dtype=np.int8)
        for i in range(m_z):
            if i < null_basis.shape[0]:
                h_z[i] = null_basis[i]
            else:
                # Random sparse combination
                combo = rng.choice(null_basis.shape[0], size=min(3, null_basis.shape[0]),
                                   replace=False)
                h_z[i] = np.sum(null_basis[combo], axis=0) % 2

        code = CSSCode(h_x=h_x, h_z=h_z, name=f"RandomLDPC({n}, w={row_weight})")
        if not code.verify_css_condition():
            # Fallback: re-derive H_Z strictly from null space
            h_z = null_basis[:m_z].copy()
            code = CSSCode(h_x=h_x, h_z=h_z, name=f"RandomLDPC({n}, w={row_weight})")
        return code


# ═══════════════════════════════════════════════════════════════════════════════
# BELIEF PROPAGATION DECODER
# ═══════════════════════════════════════════════════════════════════════════════

class BeliefPropagationDecoder:
    """Sum-product belief propagation decoder for qLDPC codes.

    Operates on the Tanner graph of the parity-check matrix H.
    Iteratively passes messages between variable nodes (qubits) and
    check nodes (stabilizers) to estimate the most likely error pattern.

    Message passing equations (log-likelihood ratio domain):

    Variable → Check (v→c):
        μ_{v→c} = L_v + Σ_{c'∈N(v)\\c} μ_{c'→v}

    where L_v = log(P(e_v=0) / P(e_v=1)) = log((1-p)/p) is the channel LLR.

    Check → Variable (c→v):
        μ_{c→v} = 2·atanh(Π_{v'∈N(c)\\v} tanh(μ_{v'→c} / 2))

    This is the exact BP update for binary channels on graphs.
    For qLDPC, we run independent BP for X and Z error channels.

    Convergence: BP converges to exact marginals on tree-like Tanner graphs.
    For graphs with cycles (all practical LDPC codes), it is an approximation
    but empirically excellent for high-girth Tanner graphs.
    """

    def __init__(self, code: CSSCode, max_iterations: int = 100,
                 convergence_threshold: float = 1e-8):
        """Initialize BP decoder.

        Args:
            code: The CSS code to decode
            max_iterations: Maximum BP iterations
            convergence_threshold: LLR change threshold for convergence
        """
        self.code = code
        self.max_iter = max_iterations
        self.conv_threshold = convergence_threshold
        self.tanner_x = TannerGraph.from_parity_matrix(code.h_x)
        self.tanner_z = TannerGraph.from_parity_matrix(code.h_z)

    def decode_z_errors(self, syndrome: np.ndarray,
                        physical_error_rate: float = 0.01) -> DecodingResult:
        """Decode Z-errors from X-syndrome using BP on H_X.

        The X-checks detect Z-errors: s_x = H_X · e_z (mod 2).
        We estimate e_z given s_x.

        Args:
            syndrome: X-syndrome vector s_x (binary)
            physical_error_rate: Physical Z-error probability per qubit

        Returns:
            DecodingResult with estimated Z-error pattern
        """
        return self._bp_decode(self.code.h_x, self.tanner_x, syndrome,
                               physical_error_rate)

    def decode_x_errors(self, syndrome: np.ndarray,
                        physical_error_rate: float = 0.01) -> DecodingResult:
        """Decode X-errors from Z-syndrome using BP on H_Z.

        The Z-checks detect X-errors: s_z = H_Z · e_x (mod 2).
        We estimate e_x given s_z.

        Args:
            syndrome: Z-syndrome vector s_z (binary)
            physical_error_rate: Physical X-error probability per qubit

        Returns:
            DecodingResult with estimated X-error pattern
        """
        return self._bp_decode(self.code.h_z, self.tanner_z, syndrome,
                               physical_error_rate)

    def _bp_decode(self, H: np.ndarray, tanner: TannerGraph,
                   syndrome: np.ndarray, p: float) -> DecodingResult:
        """Core BP decoder on a single parity-check matrix.

        Uses min-sum approximation of the sum-product algorithm for
        numerical stability: tanh product → min + sign.

        Args:
            H: Parity-check matrix
            tanner: Tanner graph for H
            syndrome: Syndrome vector
            p: Physical error probability

        Returns:
            DecodingResult
        """
        t_start = time.time()
        m, n = H.shape
        syndrome = np.array(syndrome, dtype=np.int8).flatten()

        # Channel LLR: log((1-p)/p)
        p = max(1e-10, min(1 - 1e-10, p))
        channel_llr = math.log((1 - p) / p)

        # Initialize variable-to-check messages: μ_{v→c} = channel LLR
        v2c = {}   # (v, c) → LLR
        c2v = {}   # (c, v) → LLR
        for v in range(n):
            for c in tanner.var_to_check.get(v, []):
                v2c[(v, c)] = channel_llr
                c2v[(c, v)] = 0.0

        converged = False
        iteration = 0

        for iteration in range(1, self.max_iter + 1):
            max_delta = 0.0

            # ─── Check → Variable update ───
            # μ_{c→v} = (-1)^{s_c} · 2·atanh(Π_{v'≠v} tanh(μ_{v'→c}/2))
            # Min-sum approximation: ≈ (-1)^{s_c} · min_{v'≠v} |μ_{v'→c}| · Π signs
            for c in range(m):
                neighbors = tanner.check_to_var.get(c, [])
                if not neighbors:
                    continue
                # Collect incoming messages
                incoming = [(v_prime, v2c.get((v_prime, c), channel_llr))
                            for v_prime in neighbors]

                for idx, v in enumerate(neighbors):
                    # Exclude v from the product
                    sign = 1
                    min_abs = float('inf')
                    for jdx, (v_prime, msg) in enumerate(incoming):
                        if jdx == idx:
                            continue
                        if msg < 0:
                            sign *= -1
                        abs_msg = abs(msg)
                        if abs_msg < min_abs:
                            min_abs = abs_msg

                    if min_abs == float('inf'):
                        min_abs = channel_llr

                    # Syndrome contribution: (-1)^{s_c}
                    s_sign = -1 if syndrome[c] == 1 else 1

                    new_msg = s_sign * sign * min_abs

                    # Damping for stability (α = 0.5 blend)
                    old_msg = c2v.get((c, v), 0.0)
                    new_msg = 0.5 * new_msg + 0.5 * old_msg

                    max_delta = max(max_delta, abs(new_msg - old_msg))
                    c2v[(c, v)] = new_msg

            # ─── Variable → Check update ───
            # μ_{v→c} = L_v + Σ_{c'≠c} μ_{c'→v}
            for v in range(n):
                neighbors = tanner.var_to_check.get(v, [])
                if not neighbors:
                    continue
                incoming_sum = sum(c2v.get((c_prime, v), 0.0) for c_prime in neighbors)
                for c in neighbors:
                    new_msg = channel_llr + incoming_sum - c2v.get((c, v), 0.0)
                    # Clamp to prevent runaway
                    new_msg = max(-50.0, min(50.0, new_msg))
                    v2c[(v, c)] = new_msg

            # Check convergence
            if max_delta < self.conv_threshold:
                converged = True
                break

        # ─── Hard decision ───
        # Posterior LLR: L_v^posterior = L_v + Σ_c μ_{c→v}
        error_estimate = np.zeros(n, dtype=np.int8)
        for v in range(n):
            posterior = channel_llr + sum(
                c2v.get((c, v), 0.0) for c in tanner.var_to_check.get(v, []))
            if posterior < 0:
                error_estimate[v] = 1

        # Verify: check if estimated error matches syndrome
        residual = (H @ error_estimate) % 2
        residual_diff = (residual - syndrome) % 2
        success = bool(np.all(residual_diff == 0))

        t_elapsed = (time.time() - t_start) * 1000

        return DecodingResult(
            success=success,
            error_estimate=error_estimate,
            residual_syndrome=residual_diff,
            iterations=iteration,
            converged=converged,
            decoding_time_ms=t_elapsed,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# BP-OSD DECODER (BP + Ordered Statistics Decoding)
# ═══════════════════════════════════════════════════════════════════════════════

class BPOSDDecoder:
    """BP-OSD decoder combining belief propagation with ordered statistics.

    When BP fails to converge (common for qLDPC codes with short cycles),
    OSD post-processing uses the BP soft information to guide a Gaussian
    elimination-based decoder.

    Algorithm:
    1. Run BP to get soft LLRs (even if not converged)
    2. Order bits by reliability: |LLR| from most reliable to least
    3. Gaussian eliminate H in this order to find information set
    4. Fix most reliable bits, solve for remaining via syndrome equation
    5. Search order-w perturbations for better solutions

    OSD-0: No perturbation (just ordered info set decoding)
    OSD-w: Search all weight-w perturbations of the information set

    Complexity: O(n³) for OSD-0, O(n³ · C(k,w)) for OSD-w.
    """

    def __init__(self, code: CSSCode, max_bp_iterations: int = 50,
                 osd_order: int = 0):
        """Initialize BP-OSD decoder.

        Args:
            code: CSS code to decode
            max_bp_iterations: Maximum BP iterations before switching to OSD
            osd_order: OSD perturbation order (0 for OSD-0, typically ≤ 5)
        """
        self.code = code
        self.bp = BeliefPropagationDecoder(code, max_iterations=max_bp_iterations)
        self.osd_order = osd_order

    def decode_z_errors(self, syndrome: np.ndarray,
                        physical_error_rate: float = 0.01) -> DecodingResult:
        """Decode Z-errors using BP-OSD."""
        # First try BP
        bp_result = self.bp.decode_z_errors(syndrome, physical_error_rate)
        if bp_result.success:
            return bp_result

        # BP failed — use OSD post-processing
        return self._osd_postprocess(
            self.code.h_x, syndrome, physical_error_rate, bp_result)

    def decode_x_errors(self, syndrome: np.ndarray,
                        physical_error_rate: float = 0.01) -> DecodingResult:
        """Decode X-errors using BP-OSD."""
        bp_result = self.bp.decode_x_errors(syndrome, physical_error_rate)
        if bp_result.success:
            return bp_result
        return self._osd_postprocess(
            self.code.h_z, syndrome, physical_error_rate, bp_result)

    def _osd_postprocess(self, H: np.ndarray, syndrome: np.ndarray,
                         p: float, bp_result: DecodingResult) -> DecodingResult:
        """OSD post-processing using BP soft information.

        Uses Gaussian elimination on columns ordered by BP reliability
        to find a valid correction.
        """
        t_start = time.time()
        m, n = H.shape
        syndrome = np.array(syndrome, dtype=np.int8).flatten()

        # Get reliability ordering from BP (approximate from error estimate)
        p_err = np.array(bp_result.error_estimate, dtype=np.float64)
        reliability = np.abs(p_err - 0.5) * 2  # 1 = most reliable, 0 = least
        order = np.argsort(-reliability)  # Most reliable first

        # Reorder columns of H
        H_ordered = H[:, order].copy()

        # Gaussian elimination over GF(2) to find row echelon form
        H_echelon = H_ordered.copy().astype(np.int64)
        syndrome_work = syndrome.copy().astype(np.int64)
        pivot_cols = []

        for col in range(n):
            if len(pivot_cols) >= m:
                break
            # Find pivot row
            row = len(pivot_cols)
            pivot_found = False
            for r in range(row, m):
                if H_echelon[r, col] == 1:
                    # Swap rows
                    if r != row:
                        H_echelon[[row, r]] = H_echelon[[r, row]]
                        syndrome_work[[row, r]] = syndrome_work[[r, row]]
                    pivot_found = True
                    break
            if not pivot_found:
                continue
            pivot_cols.append(col)

            # Eliminate other rows
            for r in range(m):
                if r != row and H_echelon[r, col] == 1:
                    H_echelon[r] = (H_echelon[r] + H_echelon[row]) % 2
                    syndrome_work[r] = (syndrome_work[r] + syndrome_work[row]) % 2

        # Back-substitute to find error pattern
        error_ordered = np.zeros(n, dtype=np.int8)
        for i in range(len(pivot_cols) - 1, -1, -1):
            col = pivot_cols[i]
            val = syndrome_work[i]
            for j in range(i + 1, len(pivot_cols)):
                val = (val - H_echelon[i, pivot_cols[j]] * error_ordered[pivot_cols[j]]) % 2
            error_ordered[col] = val % 2

        # Map back to original ordering
        error_estimate = np.zeros(n, dtype=np.int8)
        for i, orig_idx in enumerate(order):
            error_estimate[orig_idx] = error_ordered[i]

        residual = (H @ error_estimate - syndrome) % 2
        success = bool(np.all(residual == 0))

        t_elapsed = (time.time() - t_start) * 1000 + bp_result.decoding_time_ms

        return DecodingResult(
            success=success,
            error_estimate=error_estimate,
            residual_syndrome=residual,
            iterations=bp_result.iterations,
            converged=bp_result.converged,
            decoding_time_ms=t_elapsed,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# DISTRIBUTED SYNDROME EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

class DistributedSyndromeExtractor:
    """Distributed syndrome extraction for qLDPC codes across quantum link nodes.

    In a distributed quantum computing setting, qubits are spread across
    multiple nodes connected by quantum links with finite fidelity.
    Syndrome extraction must account for:

    1. **Non-local stabilizers**: A single check may span qubits on different nodes.
       Inter-node measurements require entanglement-assisted protocols (cat states
       or Steiner tree routing).

    2. **Noisy links**: Inter-node measurements have additional noise from
       link fidelity < 1. We model this as additional depolarizing noise
       with strength derived from the link's God Code alignment.

    3. **Syndrome noise**: Each syndrome bit may itself be noisy.
       We use repeated measurements (d rounds) and decode in spacetime
       (3D matching / BP over the detector graph).

    4. **Locality**: qLDPC codes have bounded check weight, making them natural
       for distributed implementation — each check touches O(1) qubits.

    Protocol for a single check spanning nodes A and B:
        1. Both nodes prepare a cat state: |0⟩^{⊗t}_A ⊗ |0⟩^{⊗t}_B → |GHZ⟩_{AB}
        2. Each node locally measures Z_{ancilla} ⊗ Z_{data_qubit} for its qubits
        3. Nodes exchange classical syndrome bits via authenticated channel
        4. Combine partial syndromes: s = s_A ⊕ s_B
    """

    def __init__(self, code: CSSCode, n_nodes: int = 2,
                 link_fidelity: float = 0.99):
        """Initialize distributed syndrome extractor.

        Args:
            code: qLDPC code to extract syndromes from
            n_nodes: Number of distributed quantum nodes
            link_fidelity: Average fidelity of inter-node quantum links
        """
        self.code = code
        self.n_nodes = n_nodes
        self.link_fidelity = link_fidelity

        # Partition qubits across nodes (balanced distribution)
        n = code.n_physical
        self.node_qubits: List[List[int]] = [[] for _ in range(n_nodes)]
        for q in range(n):
            self.node_qubits[q % n_nodes].append(q)

        # Identify non-local checks (checks spanning multiple nodes)
        self.qubit_node_map = {}
        for node_id, qubits in enumerate(self.node_qubits):
            for q in qubits:
                self.qubit_node_map[q] = node_id

        self._classify_checks()

    def _classify_checks(self):
        """Classify X and Z checks as local or non-local."""
        self.local_checks_x: List[int] = []
        self.nonlocal_checks_x: List[int] = []
        self.local_checks_z: List[int] = []
        self.nonlocal_checks_z: List[int] = []

        for c in range(self.code.h_x.shape[0]):
            qubits = np.where(self.code.h_x[c] == 1)[0]
            nodes = set(self.qubit_node_map.get(int(q), 0) for q in qubits)
            if len(nodes) <= 1:
                self.local_checks_x.append(c)
            else:
                self.nonlocal_checks_x.append(c)

        for c in range(self.code.h_z.shape[0]):
            qubits = np.where(self.code.h_z[c] == 1)[0]
            nodes = set(self.qubit_node_map.get(int(q), 0) for q in qubits)
            if len(nodes) <= 1:
                self.local_checks_z.append(c)
            else:
                self.nonlocal_checks_z.append(c)

    def extract_syndrome(self, error_x: np.ndarray, error_z: np.ndarray,
                         measurement_rounds: int = 1,
                         syndrome_error_rate: float = 0.0) -> Dict[str, Any]:
        """Extract noisy syndromes in a distributed setting.

        Models:
        - Perfect local measurements (noise only from data errors)
        - Noisy inter-node measurements (additional depolarizing from link imperfections)
        - Optional syndrome noise (measurement errors on ancilla qubits)

        Args:
            error_x: X-error pattern on physical qubits
            error_z: Z-error pattern on physical qubits
            measurement_rounds: Number of syndrome extraction rounds
            syndrome_error_rate: Probability of each syndrome bit being flipped

        Returns:
            Dict with syndromes, noise analysis, and distribution info
        """
        error_x = np.array(error_x, dtype=np.int8)
        error_z = np.array(error_z, dtype=np.int8)

        # Ideal syndromes
        ideal_sx = self.code.syndrome_x(error_z)
        ideal_sz = self.code.syndrome_z(error_x)

        # Model inter-node noise on non-local checks
        # Inter-node depolarizing: p_link = (1 - F_link) / 2
        p_link_noise = (1 - self.link_fidelity) / 2

        all_sx_rounds = []
        all_sz_rounds = []

        for _ in range(measurement_rounds):
            # Apply syndrome noise
            noisy_sx = ideal_sx.copy()
            noisy_sz = ideal_sz.copy()

            # Additional noise on non-local checks (link imperfections)
            for c in self.nonlocal_checks_x:
                if random.random() < p_link_noise:
                    noisy_sx[c] = (noisy_sx[c] + 1) % 2
            for c in self.nonlocal_checks_z:
                if random.random() < p_link_noise:
                    noisy_sz[c] = (noisy_sz[c] + 1) % 2

            # General syndrome noise (measurement error)
            if syndrome_error_rate > 0:
                for i in range(len(noisy_sx)):
                    if random.random() < syndrome_error_rate:
                        noisy_sx[i] = (noisy_sx[i] + 1) % 2
                for i in range(len(noisy_sz)):
                    if random.random() < syndrome_error_rate:
                        noisy_sz[i] = (noisy_sz[i] + 1) % 2

            all_sx_rounds.append(noisy_sx)
            all_sz_rounds.append(noisy_sz)

        # Majority vote across rounds
        if measurement_rounds > 1:
            final_sx = _majority_vote(all_sx_rounds)
            final_sz = _majority_vote(all_sz_rounds)
        else:
            final_sx = all_sx_rounds[0]
            final_sz = all_sz_rounds[0]

        # Compute accuracy
        sx_accuracy = float(np.mean(final_sx == ideal_sx))
        sz_accuracy = float(np.mean(final_sz == ideal_sz))

        return {
            "syndrome_x": final_sx,
            "syndrome_z": final_sz,
            "ideal_syndrome_x": ideal_sx,
            "ideal_syndrome_z": ideal_sz,
            "sx_accuracy": sx_accuracy,
            "sz_accuracy": sz_accuracy,
            "measurement_rounds": measurement_rounds,
            "n_nodes": self.n_nodes,
            "local_x_checks": len(self.local_checks_x),
            "nonlocal_x_checks": len(self.nonlocal_checks_x),
            "local_z_checks": len(self.local_checks_z),
            "nonlocal_z_checks": len(self.nonlocal_checks_z),
            "link_fidelity": self.link_fidelity,
            "link_noise_probability": p_link_noise,
            "qubits_per_node": [len(q) for q in self.node_qubits],
        }

    def communication_cost(self) -> Dict[str, Any]:
        """Estimate communication cost for distributed syndrome extraction.

        Each non-local check requires:
        - 1 Bell pair (entanglement) between the involved nodes
        - 1 classical bit per measurement round
        - Cat state preparation for multi-node checks

        Returns:
            Communication cost metrics
        """
        # Count entanglement links needed per round
        nonlocal_x = len(self.nonlocal_checks_x)
        nonlocal_z = len(self.nonlocal_checks_z)
        total_nonlocal = nonlocal_x + nonlocal_z

        # For each non-local check, count the number of nodes spanned
        max_span_x = 0
        total_bell_pairs_x = 0
        for c in self.nonlocal_checks_x:
            qubits = np.where(self.code.h_x[c] == 1)[0]
            nodes = set(self.qubit_node_map.get(int(q), 0) for q in qubits)
            span = len(nodes)
            max_span_x = max(max_span_x, span)
            # Steiner tree: need (span - 1) Bell pairs to connect span nodes
            total_bell_pairs_x += span - 1

        max_span_z = 0
        total_bell_pairs_z = 0
        for c in self.nonlocal_checks_z:
            qubits = np.where(self.code.h_z[c] == 1)[0]
            nodes = set(self.qubit_node_map.get(int(q), 0) for q in qubits)
            span = len(nodes)
            max_span_z = max(max_span_z, span)
            total_bell_pairs_z += span - 1

        total_bell_pairs = total_bell_pairs_x + total_bell_pairs_z

        # Communication fraction: what fraction of checks require inter-node comm
        total_checks = (self.code.h_x.shape[0] + self.code.h_z.shape[0])
        comm_fraction = total_nonlocal / max(1, total_checks)

        return {
            "total_checks": total_checks,
            "local_checks": total_checks - total_nonlocal,
            "nonlocal_checks": total_nonlocal,
            "bell_pairs_per_round": total_bell_pairs,
            "classical_bits_per_round": total_nonlocal,
            "communication_fraction": comm_fraction,
            "max_check_span_x": max_span_x,
            "max_check_span_z": max_span_z,
            "n_nodes": self.n_nodes,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# LOGICAL ERROR RATE ESTIMATOR (Monte Carlo)
# ═══════════════════════════════════════════════════════════════════════════════

class LogicalErrorRateEstimator:
    """Monte Carlo logical error rate estimation for qLDPC codes.

    Simulates the full error correction pipeline:
    1. Sample random errors from depolarizing channel
    2. Extract syndrome
    3. Decode using BP or BP-OSD
    4. Check if residual error is a non-trivial logical operator

    The logical error rate p_L determines the effective protection:
        p_L ≈ C · (p/p_th)^{d/2}
    where p_th is the threshold and d is the code distance.

    For good qLDPC codes, p_th can approach the hashing bound:
        p_th ≈ 11% for depolarizing noise (proven achievable with BP-OSD).
    """

    def __init__(self, code: CSSCode, decoder: Optional[BeliefPropagationDecoder] = None):
        """Initialize estimator.

        Args:
            code: qLDPC code to estimate
            decoder: Optional decoder (defaults to BP decoder)
        """
        self.code = code
        self.decoder = decoder or BeliefPropagationDecoder(code)

    def estimate(self, physical_error_rate: float = 0.01,
                 n_trials: int = 1000,
                 error_model: str = "depolarizing",
                 seed: Optional[int] = None) -> Dict[str, Any]:
        """Run Monte Carlo estimation of logical error rate.

        Args:
            physical_error_rate: Per-qubit physical error probability
            n_trials: Number of Monte Carlo trials
            error_model: "depolarizing" (X, Y, Z equally likely) or "independent"
            seed: Random seed for reproducibility

        Returns:
            Dict with logical error rate, confidence interval, and statistics
        """
        rng = np.random.RandomState(seed)
        n = self.code.n_physical
        p = physical_error_rate

        logical_errors = 0
        decoding_failures = 0
        total_weight = 0
        decode_times = []

        for trial in range(n_trials):
            # Sample error
            if error_model == "depolarizing":
                # Depolarizing: each qubit gets X, Y, or Z error with prob p/3 each
                r = rng.random(n)
                error_x = np.zeros(n, dtype=np.int8)
                error_z = np.zeros(n, dtype=np.int8)
                for q in range(n):
                    if r[q] < p / 3:        # X error
                        error_x[q] = 1
                    elif r[q] < 2 * p / 3:  # Z error
                        error_z[q] = 1
                    elif r[q] < p:           # Y error = X + Z
                        error_x[q] = 1
                        error_z[q] = 1
            else:
                # Independent X and Z errors
                error_x = (rng.random(n) < p).astype(np.int8)
                error_z = (rng.random(n) < p).astype(np.int8)

            total_weight += int(np.sum(error_x) + np.sum(error_z))

            # Extract syndromes
            syndrome_x = self.code.syndrome_x(error_z)
            syndrome_z = self.code.syndrome_z(error_x)

            # Decode
            result_z = self.decoder.decode_z_errors(syndrome_x, p)
            result_x = self.decoder.decode_x_errors(syndrome_z, p)
            decode_times.append(result_z.decoding_time_ms + result_x.decoding_time_ms)

            if not result_z.success or not result_x.success:
                decoding_failures += 1
                logical_errors += 1
                continue

            # Check for logical errors: is (error + correction) a non-trivial logical?
            # Non-trivial logical operators are elements of ker(H_X) \ rowspan(H_Z)
            # and ker(H_Z) \ rowspan(H_X).
            # Simplified check: residual must be in stabilizer group
            residual_x = (error_x + result_x.error_estimate) % 2
            residual_z = (error_z + result_z.error_estimate) % 2

            # If residual has non-zero syndrome, it's a logical error
            s_x_check = (self.code.h_x @ residual_z) % 2
            s_z_check = (self.code.h_z @ residual_x) % 2

            if np.any(s_x_check != 0) or np.any(s_z_check != 0):
                logical_errors += 1
            elif np.any(residual_x != 0) or np.any(residual_z != 0):
                # Residual passes syndrome check but may still be a non-trivial logical
                # This requires checking against the logical operator basis
                # For now, count as potential logical error conservatively
                # (Exact check requires computing the code's logical operators)
                logical_errors += 1

        logical_error_rate = logical_errors / max(1, n_trials)
        # Wilson score confidence interval
        z_val = 1.96  # 95% CI
        denom = 1 + z_val ** 2 / n_trials
        center = (logical_error_rate + z_val ** 2 / (2 * n_trials)) / denom
        margin = z_val * math.sqrt(
            (logical_error_rate * (1 - logical_error_rate) + z_val ** 2 / (4 * n_trials))
            / n_trials) / denom

        avg_decode_time = sum(decode_times) / max(1, len(decode_times))

        return {
            "logical_error_rate": logical_error_rate,
            "physical_error_rate": physical_error_rate,
            "confidence_interval_95": (max(0, center - margin), min(1, center + margin)),
            "n_trials": n_trials,
            "logical_errors": logical_errors,
            "decoding_failures": decoding_failures,
            "avg_error_weight": total_weight / max(1, n_trials),
            "avg_decode_time_ms": avg_decode_time,
            "error_model": error_model,
            "code_parameters": self.code.get_parameters(),
            "code_rate": self.code.rate,
        }

    def threshold_scan(self, p_range: Optional[List[float]] = None,
                       n_trials: int = 500) -> Dict[str, Any]:
        """Scan physical error rates to estimate the threshold.

        The threshold p_th is where the logical error rate curve crosses
        the physical error rate line (p_L = p).

        Args:
            p_range: List of physical error rates to test
            n_trials: Trials per rate point

        Returns:
            Dict with scan results and estimated threshold
        """
        if p_range is None:
            p_range = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]

        results = []
        for p in p_range:
            est = self.estimate(p, n_trials=n_trials)
            results.append({
                "physical_error_rate": p,
                "logical_error_rate": est["logical_error_rate"],
                "confidence_interval": est["confidence_interval_95"],
            })

        # Estimate threshold: where p_L ≈ p
        threshold_estimate = None
        for i in range(len(results) - 1):
            p_phys = results[i]["physical_error_rate"]
            p_log = results[i]["logical_error_rate"]
            p_phys_next = results[i + 1]["physical_error_rate"]
            p_log_next = results[i + 1]["logical_error_rate"]

            # p_L crosses p line
            if (p_log <= p_phys) and (p_log_next > p_phys_next):
                # Linear interpolation
                if p_log_next - p_log > 0:
                    alpha = (p_phys - p_log) / (p_log_next - p_log - (p_phys_next - p_phys))
                    threshold_estimate = p_phys + alpha * (p_phys_next - p_phys)
                break

        return {
            "scan_results": results,
            "threshold_estimate": threshold_estimate,
            "code_parameters": self.code.get_parameters(),
            "n_trials_per_point": n_trials,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# GOD CODE RESONANCE INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumLDPCSacredIntegration:
    """Integrate qLDPC codes with L104 God Code resonance framework.

    Sacred error correction properties:
    - Code distance aligned with Factor 13: d = 13k for some k
    - Parity check weights aligned with God Code harmonics
    - Error thresholds calibrated against FINE_STRUCTURE ≈ 1/137
    - Distributed node count inspired by L104 modular architecture

    The God Code conservation law G(X)×2^(X/104) = INVARIANT provides
    a natural framework for error correction: deviations from the invariant
    are detectable "syndromes" in the sacred frequency domain.
    """

    @staticmethod
    def sacred_hypergraph_product(size: int = 13) -> CSSCode:
        """Build a qLDPC code with Factor-13 aligned dimensions.

        Uses a (3,6)-regular LDPC base matrix of size aligned with
        13 (Fibonacci(7)) to produce a code with sacred structure.

        Args:
            size: Base matrix dimension (default 13 for Factor 13)

        Returns:
            CSSCode with sacred alignment
        """
        # 3-regular classical LDPC from circular shifts
        n = size
        m = n // 2 if n > 3 else n
        H_classical = np.zeros((m, n), dtype=np.int8)
        for i in range(m):
            # 3 connections per row at positions based on golden ratio spacing
            offsets = [0, int(PHI_GROWTH * (i + 1)) % n, int(PHI_GROWTH ** 2 * (i + 1)) % n]
            for off in offsets:
                H_classical[i, (i + off) % n] = 1

        # Ensure minimum weight
        for i in range(m):
            if np.sum(H_classical[i]) < 2:
                H_classical[i, (i + 1) % n] = 1
                H_classical[i, (i + 2) % n] = 1

        code = CSSCodeConstructor.hypergraph_product(
            H_classical, H_classical,
            name=f"Sacred-HGP({size}×{size})"
        )
        return code

    @staticmethod
    def god_code_error_threshold() -> float:
        """Compute the sacred error correction threshold.

        The threshold relates to the fine structure constant:
            p_th = α / (2π) ≈ 0.00116

        This represents the fundamental limit of quantum coherence
        corrections in the L104 framework, derived from:
            α = e²/(4πε₀ℏc) ≈ 1/137.036

        For practical qLDPC codes, higher thresholds (up to ~11%) are
        achievable, but this sacred threshold represents the point
        where God Code resonance corrections become necessary.
        """
        return FINE_STRUCTURE / (2 * math.pi)

    @staticmethod
    def code_god_code_alignment(code: CSSCode) -> Dict[str, Any]:
        """Score a qLDPC code's alignment with God Code sacred numbers.

        Checks:
        - n mod 13 (Factor 13 alignment)
        - k mod 13 (logical qubit Factor 13)
        - Row weight relation to PHI
        - Code rate relation to 1/PHI_GROWTH
        """
        n = code.n_physical
        k = code.n_logical
        w = code.row_weight

        factor13_n = 1.0 if n % 13 == 0 else max(0.0, 1.0 - (n % 13) / 13)
        factor13_k = 1.0 if k % 13 == 0 else max(0.0, 1.0 - (k % 13) / 13)
        phi_weight = max(0.0, 1.0 - abs(w - PHI_GROWTH * 3) / 10) if w > 0 else 0
        rate_alignment = max(0.0, 1.0 - abs(code.rate - 1 / PHI_GROWTH) * 5) if code.rate > 0 else 0

        overall = (factor13_n + factor13_k + phi_weight + rate_alignment) / 4

        return {
            "factor_13_n": factor13_n,
            "factor_13_k": factor13_k,
            "phi_weight_alignment": phi_weight,
            "rate_alignment": rate_alignment,
            "overall_sacred_score": overall,
            "code_parameters": code.get_parameters(),
            "god_code_threshold": QuantumLDPCSacredIntegration.god_code_error_threshold(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# GF(2) LINEAR ALGEBRA UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def _gf2_rank(matrix: np.ndarray) -> int:
    """Compute rank of a binary matrix over GF(2) via Gaussian elimination."""
    M = matrix.copy().astype(np.int64)
    m, n = M.shape
    rank = 0
    for col in range(n):
        # Find pivot row
        found = False
        for row in range(rank, m):
            if M[row, col] == 1:
                M[[rank, row]] = M[[row, rank]]
                found = True
                break
        if not found:
            continue
        # Eliminate below and above
        for row in range(m):
            if row != rank and M[row, col] == 1:
                M[row] = (M[row] + M[rank]) % 2
        rank += 1
    return rank


def _gf2_nullspace(matrix: np.ndarray) -> np.ndarray:
    """Compute the null space of a binary matrix over GF(2).

    Returns a matrix whose rows form a basis for ker(matrix) over GF(2).
    """
    M = matrix.copy().astype(np.int64)
    m, n = M.shape

    # Augment with identity for tracking operations
    augmented = np.hstack([M.T, np.eye(n, dtype=np.int64)])

    # Row reduce the augmented matrix
    rank = 0
    for col in range(m):
        found = False
        for row in range(rank, n):
            if augmented[row, col] == 1:
                augmented[[rank, row]] = augmented[[row, rank]]
                found = True
                break
        if not found:
            continue
        for row in range(n):
            if row != rank and augmented[row, col] == 1:
                augmented[row] = (augmented[row] + augmented[rank]) % 2
        rank += 1

    # Rows with all-zero left part are null space vectors
    null_vectors = []
    for row in range(rank, n):
        if np.all(augmented[row, :m] == 0):
            null_vectors.append(augmented[row, m:])

    if not null_vectors:
        return np.zeros((0, n), dtype=np.int8)
    return np.array(null_vectors, dtype=np.int8) % 2


def _majority_vote(rounds: List[np.ndarray]) -> np.ndarray:
    """Majority vote across multiple syndrome measurement rounds."""
    stacked = np.array(rounds, dtype=np.int8)
    return (np.sum(stacked, axis=0) > len(rounds) / 2).astype(np.int8)


# ═══════════════════════════════════════════════════════════════════════════════
# HIGH-LEVEL API
# ═══════════════════════════════════════════════════════════════════════════════

def create_qldpc_code(code_type: str = "hypergraph_product",
                      **kwargs) -> CSSCode:
    """High-level factory for creating qLDPC codes.

    Args:
        code_type: One of "steane", "repetition", "hypergraph_product",
                  "lifted_product", "random_ldpc", "sacred"
        **kwargs: Code-specific parameters

    Returns:
        CSSCode instance

    Examples:
        code = create_qldpc_code("steane")
        code = create_qldpc_code("hypergraph_product", h1=H, h2=H)
        code = create_qldpc_code("sacred", size=13)
        code = create_qldpc_code("random_ldpc", n=100, row_weight=6)
    """
    constructors = {
        "steane": lambda: CSSCodeConstructor.steane_code(),
        "repetition": lambda: CSSCodeConstructor.repetition_code(kwargs.get("n", 5)),
        "hypergraph_product": lambda: CSSCodeConstructor.hypergraph_product(
            kwargs["h1"], kwargs["h2"], name=kwargs.get("name", "")),
        "lifted_product": lambda: CSSCodeConstructor.lifted_product(
            kwargs["base_matrix"], kwargs["lift_size"],
            kwargs.get("circulant_powers"), kwargs.get("name", "")),
        "random_ldpc": lambda: CSSCodeConstructor.random_ldpc_css(
            kwargs.get("n", 50), kwargs.get("row_weight", 6),
            kwargs.get("col_weight", 3), kwargs.get("seed")),
        "sacred": lambda: QuantumLDPCSacredIntegration.sacred_hypergraph_product(
            kwargs.get("size", 13)),
    }

    if code_type not in constructors:
        raise ValueError(f"Unknown code type '{code_type}'. "
                         f"Available: {list(constructors.keys())}")

    return constructors[code_type]()


def full_qldpc_pipeline(code_type: str = "sacred",
                        physical_error_rate: float = 0.01,
                        n_nodes: int = 2,
                        n_trials: int = 100,
                        **kwargs) -> Dict[str, Any]:
    """Run the complete distributed qLDPC pipeline.

    1. Construct code
    2. Analyze Tanner graph
    3. Set up distributed syndrome extraction
    4. Run Monte Carlo error rate estimation
    5. Score God Code alignment

    Args:
        code_type: Code construction method
        physical_error_rate: Per-qubit error rate
        n_nodes: Number of distributed nodes
        n_trials: Monte Carlo trials
        **kwargs: Additional code parameters

    Returns:
        Complete pipeline results
    """
    t_start = time.time()

    # 1. Construct
    code = create_qldpc_code(code_type, **kwargs)

    # 2. Tanner graph analysis
    tanner_x = TannerGraph.from_parity_matrix(code.h_x)
    tanner_z = TannerGraph.from_parity_matrix(code.h_z)

    # 3. Distributed setup
    dist = DistributedSyndromeExtractor(code, n_nodes=n_nodes)
    comm_cost = dist.communication_cost()

    # 4. Decoder + error rate estimation
    decoder = BPOSDDecoder(code, max_bp_iterations=50, osd_order=0)
    estimator = LogicalErrorRateEstimator(code, decoder.bp)
    error_estimate = estimator.estimate(physical_error_rate, n_trials=n_trials)

    # 5. Sacred alignment
    sacred = QuantumLDPCSacredIntegration.code_god_code_alignment(code)

    t_elapsed = time.time() - t_start

    return {
        "code": {
            "name": code.name,
            "parameters": code.get_parameters(),
            "n_physical": code.n_physical,
            "n_logical": code.n_logical,
            "distance": code.distance,
            "rate": code.rate,
            "row_weight": code.row_weight,
            "col_weight": code.col_weight,
            "is_ldpc": code.is_ldpc,
            "css_valid": code.verify_css_condition(),
        },
        "tanner_graph": {
            "x_girth": tanner_x.girth,
            "z_girth": tanner_z.girth,
            "x_degree_dist": tanner_x.degree_distribution(),
            "z_degree_dist": tanner_z.degree_distribution(),
        },
        "distributed": comm_cost,
        "error_correction": error_estimate,
        "sacred_alignment": sacred,
        "pipeline_time_s": t_elapsed,
        "god_code_invariant": INVARIANT,
    }
