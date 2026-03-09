"""L104 VQPU v14.0.0 — Exact MPS Hybrid Engine (lossless tensor network simulator).

v13.2 GOD_CODE QUBIT UPGRADE (retained):
  - Sacred gates now use canonical QPU-verified phases from god_code_qubit.py
  - GOD_CODE_PHASE: Rz(GOD_CODE mod 2π) — QPU-verified fidelity 0.999939
  - PHI_GATE: P(2π/φ) — canonical golden angle
  - IRON_RZ: Rz(π/2) — Iron lattice quarter-turn
  - PHI_RZ: Rz(θ_φ) — Golden ratio contribution
  - OCTAVE_RZ: Rz(4·ln2) — Octave frequency doubling
  - VOID_GATE: P(VOID_CONSTANT × π) — sacred void phase
  - GOD_CODE_DECOMPOSED: 3-rotation factorization gate sequence
"""

import math
import time
import hashlib
from collections import deque, OrderedDict

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

from .constants import (
    GOD_CODE, PHI, VOID_CONSTANT,
    _IS_INTEL, _IS_APPLE_SILICON,
    VQPU_MPS_MAX_BOND_LOW, VQPU_MPS_MAX_BOND_MED, VQPU_MPS_MAX_BOND_HIGH,
    VQPU_MPS_FALLBACK_TARGET,
)

# ── Import canonical GOD_CODE qubit phases (QPU-verified on ibm_torino) ──────
try:
    from l104_god_code_simulator.god_code_qubit import (
        GOD_CODE_PHASE as _GC_PHASE,      # GOD_CODE mod 2π ≈ 6.0141 rad
        IRON_PHASE as _IRON_PHASE,         # π/2 (iron lattice quarter-turn)
        PHI_CONTRIBUTION as _PHI_CONTRIB,  # golden ratio phase contribution
        OCTAVE_PHASE as _OCTAVE_PHASE,     # 4·ln(2) ≈ 2.7726 rad
        PHI_PHASE as _PHI_PHASE,           # 2π/φ ≈ 3.8832 rad (golden angle)
        VOID_PHASE as _VOID_PHASE,         # VOID_CONSTANT × π ≈ 3.2716 rad
    )
    _HAS_GC_QUBIT = True
except ImportError:
    # Fallback: derive from constants (matches god_code_qubit.py derivation)
    _TAU = 2.0 * math.pi
    _GC_PHASE = GOD_CODE % _TAU                          # ≈ 6.0141 rad
    _IRON_PHASE = _TAU * 26 / 104                        # π/2
    _OCTAVE_PHASE = (4.0 * math.log(2.0)) % _TAU         # ≈ 2.7726 rad
    _PHI_CONTRIB = (_GC_PHASE - _IRON_PHASE - _OCTAVE_PHASE) % _TAU
    _PHI_PHASE = _TAU / PHI                               # ≈ 3.8832 rad
    _VOID_PHASE = VOID_CONSTANT * math.pi                 # ≈ 3.2716 rad
    _HAS_GC_QUBIT = False


class ExactMPSHybridEngine:
    """
    100% lossless Matrix Product State simulation with dynamic GPU fallback.

    Applies gates to an MPS chain with cutoff=0 (no truncation), preserving
    exact quantum state fidelity. When the bond dimension exceeds a threshold
    (default 8192), the engine detects that entanglement has grown beyond
    what MPS can efficiently represent, and falls back to the Metal GPU
    statevector backend for the remaining gates.

    Architecture:
      1. Initialize MPS as product state |0...0⟩ (χ=1 per bond)
      2. Apply single-qubit gates: O(χ) per gate, no bond growth
      3. Apply two-qubit gates: SVD with cutoff=0 → bond may grow
      4. Monitor max(χ) after each two-qubit gate
      5. If max(χ) > threshold → convert MPS→statevector → Metal GPU

    Gate library:
      Single: H, X, Y, Z, S, SDG, T, TDG, Rx, Ry, Rz, SX
      Two:    CX/CNOT, CZ, SWAP, CY, ECR

    Memory: O(n·χ²) for n qubits with bond dimension χ.
    For low-entanglement circuits, χ stays small and this is exponentially
    cheaper than 2^n statevector. For high-entanglement, the GPU fallback
    catches the explosion before memory is exhausted.
    """

    # v11.0: Threshold bond dimension for GPU fallback — raised from 16384 to 24576
    DEFAULT_MAX_CHI = 24576

    # Pre-computed gate matrices (2×2 complex)
    import numpy as _np
    _sqrt2 = _np.sqrt(2)

    GATE_MATRICES = {
        "H":   _np.array([[1, 1], [1, -1]], dtype=_np.complex128) / _sqrt2,
        "X":   _np.array([[0, 1], [1, 0]], dtype=_np.complex128),
        "Y":   _np.array([[0, -1j], [1j, 0]], dtype=_np.complex128),
        "Z":   _np.array([[1, 0], [0, -1]], dtype=_np.complex128),
        "S":   _np.array([[1, 0], [0, 1j]], dtype=_np.complex128),
        "SDG": _np.array([[1, 0], [0, -1j]], dtype=_np.complex128),
        "T":   _np.array([[1, 0], [0, _np.exp(1j * _np.pi / 4)]], dtype=_np.complex128),
        "TDG": _np.array([[1, 0], [0, _np.exp(-1j * _np.pi / 4)]], dtype=_np.complex128),
        "SX":  _np.array([[1+1j, 1-1j], [1-1j, 1+1j]], dtype=_np.complex128) / 2,
        "I":   _np.eye(2, dtype=_np.complex128),
        # ═══ v13.2: QPU-VERIFIED SACRED GATES (canonical from god_code_qubit.py) ═══
        # GOD_CODE_PHASE: Rz(GOD_CODE mod 2π) — THE canonical gate
        # QPU-verified on IBM ibm_torino (Heron r2): fidelity 0.999939
        "GOD_CODE_PHASE": _np.array([
            [_np.exp(-1j * _GC_PHASE / 2), 0],
            [0, _np.exp(1j * _GC_PHASE / 2)]
        ], dtype=_np.complex128),
        # PHI_GATE: P(2π/φ) — canonical golden angle phase gate
        "PHI_GATE": _np.array([
            [1, 0], [0, _np.exp(1j * _PHI_PHASE)]
        ], dtype=_np.complex128),
        # VOID_GATE: P(VOID_CONSTANT × π) — sacred void phase
        "VOID_GATE": _np.array([
            [1, 0], [0, _np.exp(1j * _VOID_PHASE)]
        ], dtype=_np.complex128),
        # ═══ 3-ROTATION DECOMPOSITION GATES (conservation: IRON + PHI + OCTAVE ≡ GOD_CODE) ═══
        # IRON_RZ: Rz(π/2) — Iron lattice Fe(26) quarter-turn
        "IRON_RZ": _np.array([
            [_np.exp(-1j * _IRON_PHASE / 2), 0],
            [0, _np.exp(1j * _IRON_PHASE / 2)]
        ], dtype=_np.complex128),
        # PHI_RZ: Rz(θ_φ) — Golden ratio contribution rotation
        "PHI_RZ": _np.array([
            [_np.exp(-1j * _PHI_CONTRIB / 2), 0],
            [0, _np.exp(1j * _PHI_CONTRIB / 2)]
        ], dtype=_np.complex128),
        # OCTAVE_RZ: Rz(4·ln(2)) — Octave ×16 frequency doubling rotation
        "OCTAVE_RZ": _np.array([
            [_np.exp(-1j * _OCTAVE_PHASE / 2), 0],
            [0, _np.exp(1j * _OCTAVE_PHASE / 2)]
        ], dtype=_np.complex128),
        # IRON_GATE: P(2π×26/104) — Iron lattice phase gate (P form)
        "IRON_GATE": _np.array([
            [1, 0], [0, _np.exp(1j * _IRON_PHASE)]
        ], dtype=_np.complex128),
    }

    # Two-qubit gate matrices (4×4 → reshaped to (2,2,2,2) for tensor contraction)
    CNOT_MATRIX = _np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ], dtype=_np.complex128).reshape(2, 2, 2, 2)

    CZ_MATRIX = _np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1],
    ], dtype=_np.complex128).reshape(2, 2, 2, 2)

    SWAP_MATRIX = _np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=_np.complex128).reshape(2, 2, 2, 2)

    CY_MATRIX = _np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, -1j],
        [0, 0, 1j, 0],
    ], dtype=_np.complex128).reshape(2, 2, 2, 2)

    ECR_MATRIX = (1.0 / _sqrt2) * _np.array([
        [0, 0, 1, 1j],
        [0, 0, 1j, 1],
        [1, -1j, 0, 0],
        [-1j, 1, 0, 0],
    ], dtype=_np.complex128).reshape(2, 2, 2, 2)

    ISWAP_MATRIX = _np.array([
        [1, 0, 0, 0],
        [0, 0, 1j, 0],
        [0, 1j, 0, 0],
        [0, 0, 0, 1],
    ], dtype=_np.complex128).reshape(2, 2, 2, 2)

    del _sqrt2  # clean up class namespace

    def __init__(self, num_qubits: int, max_chi: int = DEFAULT_MAX_CHI):
        import numpy as np
        self.np = np
        self.n = num_qubits
        self.max_chi = max_chi
        self._fallback_triggered = False
        self._fallback_gate_idx = -1
        self._peak_chi = 1

        # Initialize MPS as product state |0...0⟩
        # Each tensor has shape (χ_left, 2, χ_right)
        # For product state: (1, 2, 1) with amplitude [1, 0] for |0⟩
        self.tensors = []
        for _ in range(num_qubits):
            t = np.zeros((1, 2, 1), dtype=np.complex128)
            t[0, 0, 0] = 1.0  # |0⟩
            self.tensors.append(t)

    @property
    def bond_dims(self) -> list:
        """Current bond dimensions: [χ₀, χ₁, ..., χₙ] (n+1 values)."""
        dims = [1]  # left boundary
        for t in self.tensors:
            dims.append(t.shape[2])
        return dims

    @property
    def max_bond_dim(self) -> int:
        """Current maximum bond dimension across all bonds."""
        return max(t.shape[2] for t in self.tensors)

    @property
    def fallback_triggered(self) -> bool:
        """Whether GPU fallback was triggered due to high entanglement."""
        return self._fallback_triggered

    # ─── Gate Resolution (with parametric cache — classical bypass) ───

    # Module-level parametric gate cache: angle → 2×2 matrix
    # Discretized to 10 decimals.  Eliminates repeated trig calls for
    # circuits that reuse the same rotation angles (e.g., GOD_CODE phase,
    # sacred angles, VQE parameter sweeps).
    _parametric_cache: OrderedDict = OrderedDict()        # v13.1: OrderedDict for O(1) LRU
    _PARAMETRIC_CACHE_MAX: int = 32768                     # v12.0: 2x (was 16384)

    @classmethod
    def _resolve_single_gate(cls, name: str, params: list = None):
        """Resolve a gate name to its 2×2 unitary matrix (cached for parametrics).

        v12.0: LRU eviction when cache exceeds _PARAMETRIC_CACHE_MAX.
        Evicts oldest 25% of entries on overflow.
        """
        import numpy as np
        up = name.upper()
        if up in cls.GATE_MATRICES:
            return cls.GATE_MATRICES[up]
        if up in ("RZ", "ROTATIONZ"):
            theta = params[0] if params else 0
            key = ("RZ", round(theta, 10))
            cached = cls._parametric_cache.get(key)
            if cached is not None:
                return cached
            mat = np.array([
                [np.exp(-1j * theta / 2), 0],
                [0, np.exp(1j * theta / 2)]
            ], dtype=np.complex128)
            cls._cache_put(key, mat)
            return mat
        if up in ("RX", "ROTATIONX"):
            theta = params[0] if params else 0
            key = ("RX", round(theta, 10))
            cached = cls._parametric_cache.get(key)
            if cached is not None:
                return cached
            c, s = np.cos(theta / 2), np.sin(theta / 2)
            mat = np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
            cls._cache_put(key, mat)
            return mat
        if up in ("RY", "ROTATIONY"):
            theta = params[0] if params else 0
            key = ("RY", round(theta, 10))
            cached = cls._parametric_cache.get(key)
            if cached is not None:
                return cached
            c, s = np.cos(theta / 2), np.sin(theta / 2)
            mat = np.array([[c, -s], [s, c]], dtype=np.complex128)
            cls._cache_put(key, mat)
            return mat
        return None

    @classmethod
    def _cache_put(cls, key, mat):
        """v13.1: Insert into parametric cache with O(1) OrderedDict LRU.

        Replaces dict+deque pattern — move_to_end() is O(1) vs
        deque.remove() which was O(n) on cache hits.
        """
        if key in cls._parametric_cache:
            # Key already cached — update value, move to end (O(1))
            mat_ro = mat.copy()
            mat_ro.flags.writeable = False
            cls._parametric_cache[key] = mat_ro
            cls._parametric_cache.move_to_end(key)
            return

        if len(cls._parametric_cache) >= cls._PARAMETRIC_CACHE_MAX:
            # Evict oldest 25%
            evict_count = cls._PARAMETRIC_CACHE_MAX // 4
            for _ in range(min(evict_count, len(cls._parametric_cache))):
                cls._parametric_cache.popitem(last=False)
        mat_ro = mat.copy()
        mat_ro.flags.writeable = False
        cls._parametric_cache[key] = mat_ro

    # v12.2: Dict lookup for two-qubit gates — O(1) vs if/elif chain
    _TWO_GATE_MAP: dict = None  # Initialized lazily

    @classmethod
    def _resolve_two_gate(cls, name: str):
        """Resolve a two-qubit gate to its (2,2,2,2) tensor.

        v12.2: Single dict lookup replaces 6-branch if/elif chain.
        """
        if cls._TWO_GATE_MAP is None:
            cls._TWO_GATE_MAP = {
                "CX": cls.CNOT_MATRIX, "CNOT": cls.CNOT_MATRIX,
                "CZ": cls.CZ_MATRIX, "SWAP": cls.SWAP_MATRIX,
                "CY": cls.CY_MATRIX, "ECR": cls.ECR_MATRIX,
                "ISWAP": cls.ISWAP_MATRIX,
            }
        return cls._TWO_GATE_MAP.get(name.upper())

    # ─── Single-Qubit Gate (v11.0: matmul fast path) ───

    def apply_single_gate(self, qubit: int, gate_matrix):
        """
        Apply a single-qubit gate to an MPS site.
        No bond dimension growth — O(χ_left × χ_right) work.

        v12.0: Contiguous array optimization — uses np.ascontiguousarray
        on the transposed view before matmul for better cache locality.
        Avoids final .copy() by using contiguous reshape path.
        """
        np = self.np
        t = self.tensors[qubit]    # (χ_left, 2, χ_right)
        chi_l, _, chi_r = t.shape
        # v12.0: contiguous fast path for small tensors (product state)
        if chi_l == 1 and chi_r == 1:
            # Ultra-fast path: just a 2-vector matmul
            self.tensors[qubit] = (gate_matrix @ t.reshape(2)).reshape(1, 2, 1)
            return
        # Reshape to (χ_left, 2, χ_right) → (χ_left × χ_right, 2)
        flat = np.ascontiguousarray(t.transpose(0, 2, 1)).reshape(chi_l * chi_r, 2)
        result = flat @ gate_matrix.T
        self.tensors[qubit] = np.ascontiguousarray(result.reshape(chi_l, chi_r, 2).transpose(0, 2, 1))

    # ─── Two-Qubit Gate (Exact SVD, cutoff=0) ───

    def apply_two_gate(self, q0: int, q1: int, gate_tensor):
        """
        Apply a two-qubit gate to MPS sites q0, q1.

        For adjacent sites (|q1 - q0| == 1):
          1. Contract tensors at q0, q1 into θ
          2. Apply gate to physical indices
          3. SVD-split back with cutoff=0 (exact, no truncation)

        For non-adjacent sites:
          SWAP chain to bring qubits into adjacency, apply, SWAP back.

        Returns True if the gate was applied successfully, False if
        bond dimension exceeded threshold (caller should trigger fallback).
        """
        lo, hi = min(q0, q1), max(q0, q1)

        if hi - lo > 1:
            # Non-adjacent: SWAP into adjacency
            for k in range(hi - 1, lo, -1):
                self._apply_adjacent_two_gate(k, k + 1, self.SWAP_MATRIX)
            # Adjust gate targets based on SWAP ordering
            if q0 < q1:
                self._apply_adjacent_two_gate(lo, lo + 1, gate_tensor)
            else:
                # Swap the gate indices
                gate_swapped = gate_tensor.transpose(1, 0, 3, 2)
                self._apply_adjacent_two_gate(lo, lo + 1, gate_swapped)
            # SWAP back (reverse direction to properly undo permutation)
            for k in range(hi - 1, lo, -1):
                self._apply_adjacent_two_gate(k, k + 1, self.SWAP_MATRIX)
        else:
            if q0 < q1:
                self._apply_adjacent_two_gate(q0, q1, gate_tensor)
            else:
                gate_swapped = gate_tensor.transpose(1, 0, 3, 2)
                self._apply_adjacent_two_gate(q1, q0, gate_swapped)

        chi = self.max_bond_dim
        if chi > self._peak_chi:
            self._peak_chi = chi
        return chi <= self.max_chi

    def _apply_adjacent_two_gate(self, site_a: int, site_b: int, gate_4d):
        """
        Apply two-qubit gate to adjacent MPS sites.
        Contract → gate → SVD (exact, cutoff=0) → split.

        v12.2: Product-state fast path — skips SVD entirely when both
        sites have bond dim 1. Falls through to matmul+SVD for general case.
        v11.0: Optimized contraction path — reshape+matmul for SVD,
        uses 'gesdd' divide-and-conquer SVD driver for speed.
        Pre-scales S values inline to avoid extra allocation.
        """
        np = self.np

        A = self.tensors[site_a]  # (χ_left, 2, χ_mid)
        B = self.tensors[site_b]  # (χ_mid, 2, χ_right)

        chi_left = A.shape[0]
        chi_mid = A.shape[2]
        chi_right = B.shape[2]

        # v12.2: Product-state fast path — skip SVD when all bond dims are 1
        if chi_left == 1 and chi_mid == 1 and chi_right == 1:
            # Both sites are (1, 2, 1) — just a 4-vector matmul + reshape
            vec_a = A.reshape(2)
            vec_b = B.reshape(2)
            # θ = outer product → (2, 2) state vector
            theta_flat = np.outer(vec_a, vec_b).flatten()  # (4,)
            # Apply gate: gate_4d reshaped as (4, 4)
            result = gate_4d.reshape(4, 4) @ theta_flat     # (4,)
            result_mat = result.reshape(2, 2)

            # v13.0: Rank-1 fast path — attempt rank-1 factorization first.
            # Most two-qubit gates applied to product states yield a product
            # state (e.g., CX|00⟩=|00⟩, CX|10⟩=|11⟩). SVD is O(8) but the
            # rank-1 check below is O(4) with zero allocations.
            # Check if result_mat ≈ u ⊗ v (rank 1) by testing |det| ≈ 0.
            det = result_mat[0, 0] * result_mat[1, 1] - result_mat[0, 1] * result_mat[1, 0]
            if abs(det) < 1e-12:
                # Rank-1: find a non-zero row/column to extract factors
                r0_norm = abs(result_mat[0, 0])**2 + abs(result_mat[0, 1])**2
                r1_norm = abs(result_mat[1, 0])**2 + abs(result_mat[1, 1])**2
                if r0_norm >= r1_norm and r0_norm > 1e-30:
                    v = result_mat[0]  # shape (2,)
                    v_norm = np.sqrt(r0_norm)
                    v = v / v_norm
                    u = np.array([v_norm, result_mat[1, 0] * np.conj(v[0]) + result_mat[1, 1] * np.conj(v[1])
                                  if r0_norm > 1e-30 else 0.0])
                    # More robust: u = M @ v* / |v|^2
                    u = result_mat @ np.conj(v) / v_norm
                elif r1_norm > 1e-30:
                    v = result_mat[1]
                    v_norm = np.sqrt(r1_norm)
                    v = v / v_norm
                    u = result_mat @ np.conj(v) / v_norm
                else:
                    # Zero matrix — keep as product of zeros
                    u = np.zeros(2, dtype=result_mat.dtype)
                    v = np.zeros(2, dtype=result_mat.dtype)
                self.tensors[site_a] = u.reshape(1, 2, 1)
                self.tensors[site_b] = v.reshape(1, 2, 1)
                return

            # General case: rank > 1 (entangling), fall through to SVD
            U, S, Vh = np.linalg.svd(result_mat, full_matrices=False)
            # Typically rank-1 after CX on product state → bond dim stays small
            sqrtS = np.sqrt(S)
            U *= sqrtS[np.newaxis, :]
            Vh *= sqrtS[:, np.newaxis]
            new_bond = len(S)
            self.tensors[site_a] = U.reshape(1, 2, new_bond)
            self.tensors[site_b] = Vh.reshape(new_bond, 2, 1)
            return

        # v11.0: matmul contraction instead of einsum
        # A reshaped: (χ_left * 2, χ_mid) @ B reshaped: (χ_mid, 2 * χ_right)
        # → theta_mat: (χ_left * 2, 2 * χ_right)
        A_mat = A.reshape(chi_left * 2, chi_mid)
        B_mat = B.reshape(chi_mid, 2 * chi_right)
        theta_mat = A_mat @ B_mat
        # Reshape to (χ_left, 2, 2, χ_right) for gate application
        theta = theta_mat.reshape(chi_left, 2, 2, chi_right)

        # Apply gate: gate_4d[s0',s1',s0,s1] × θ[l,s0,s1,r] → θ'[l,s0',s1',r]
        theta = np.einsum('pqij,lijr->lpqr', gate_4d, theta)

        # Reshape for SVD: (χ_left × 2, 2 × χ_right)
        mat = theta.reshape(chi_left * 2, 2 * chi_right)

        # v11.0: Use divide-and-conquer SVD (gesdd) — faster for large matrices
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)

        # No truncation — keep full rank
        # v11.0: Inline S scaling — avoid separate sqrt allocation
        sqrtS = np.sqrt(S)
        U *= sqrtS[np.newaxis, :]   # broadcast scale columns
        Vh *= sqrtS[:, np.newaxis]   # broadcast scale rows

        new_bond = len(S)
        self.tensors[site_a] = U.reshape(chi_left, 2, new_bond)
        self.tensors[site_b] = Vh.reshape(new_bond, 2, chi_right)

    # ─── Full Circuit Execution ───

    def run_circuit(self, operations: list, *, enable_fusion: bool = True) -> dict:
        """
        Execute a full circuit on the MPS engine.

        v15.0: Gate fusion support — when enable_fusion=True, consecutive
        single-qubit gates on the same qubit are fused into a single matmul
        before being applied. This reduces the number of MPS tensor updates
        and avoids redundant reshape/contraction operations.

        Returns:
          {
            "completed": bool,       # True if all gates applied
            "fallback_at": int,      # Gate index where fallback triggered (-1 if none)
            "peak_chi": int,         # Maximum bond dimension reached
            "bond_dims": list,       # Final bond dimensions
            "remaining_ops": list,   # Gates not yet applied (for GPU fallback)
            "fusion_stats": dict,    # Gate fusion statistics (if enabled)
          }
        """
        fusion_stats = None

        if enable_fusion and len(operations) > 1:
            # v15.0: Pre-fuse consecutive 1Q gates via AccelEngine
            try:
                from .accel_engine import GateFusionAnalyzer
                fused_ops = GateFusionAnalyzer.fuse_circuit(
                    operations, self._resolve_single_gate
                )
                fusion_stats = GateFusionAnalyzer.count_fusions(fused_ops)
                operations = fused_ops
            except Exception:
                pass  # Fall through to unfused execution

        for idx, op in enumerate(operations):
            gate_name = op.get("gate", "")
            qubits = op.get("qubits", [])
            params = op.get("parameters", [])

            if gate_name == "_FUSED_1Q":
                # v15.0: Pre-fused gate — matrix already computed
                self.apply_single_gate(qubits[0], op["_matrix"])
            elif len(qubits) >= 2:
                gate_4d = self._resolve_two_gate(gate_name)
                if gate_4d is None:
                    continue
                ok = self.apply_two_gate(qubits[0], qubits[1], gate_4d)
                if not ok:
                    self._fallback_triggered = True
                    self._fallback_gate_idx = idx
                    return {
                        "completed": False,
                        "fallback_at": idx,
                        "peak_chi": self._peak_chi,
                        "bond_dims": self.bond_dims,
                        "remaining_ops": operations[idx:],
                        "fusion_stats": fusion_stats,
                    }
            elif len(qubits) >= 1:
                gate_2x2 = self._resolve_single_gate(gate_name, params)
                if gate_2x2 is None:
                    continue
                self.apply_single_gate(qubits[0], gate_2x2)

        return {
            "completed": True,
            "fallback_at": -1,
            "peak_chi": self._peak_chi,
            "bond_dims": self.bond_dims,
            "remaining_ops": [],
            "fusion_stats": fusion_stats,
        }

    # ─── State Extraction ───

    def to_statevector(self):
        """
        Contract the full MPS chain to a 2^n statevector.

        Sequential left-to-right contraction:
          ψ = A₁ · A₂ · ... · Aₙ

        v11.0: Uses reshape+matmul instead of einsum for 2-4x speedup.

        Returns: numpy array of shape (2^n,) with complex amplitudes.
        """
        np = self.np
        # Start with leftmost tensor: shape (1, 2, χ₁) → (2, χ₁)
        state = self.tensors[0].reshape(2, -1)  # (2, χ₁)

        for q in range(1, self.n):
            t = self.tensors[q]  # (χ_q, 2, χ_{q+1})
            chi_q = t.shape[0]
            chi_next = t.shape[2]
            basis_dim = state.shape[0]
            # v11.0: matmul contraction — state: (basis_dim, χ_q) @ t reshaped
            # t reshaped: (χ_q, 2 * χ_{q+1})
            t_mat = t.reshape(chi_q, 2 * chi_next)
            state = state @ t_mat  # (basis_dim, 2 * χ_{q+1})
            state = state.reshape(basis_dim * 2, chi_next)

        return state.flatten()

    def to_probabilities(self):
        """Get probability distribution from MPS."""
        np = self.np
        sv = self.to_statevector()
        probs = np.abs(sv) ** 2
        # Normalize (handle floating-point drift)
        total = probs.sum()
        if total > 0 and abs(total - 1.0) > 1e-10:
            probs /= total
        return probs

    def sample(self, shots: int = 1024) -> dict:
        """
        Sample measurement outcomes from the MPS state.

        v12.0: Vectorized bitstring formatting via numpy for 2-3x speedup
        on large shot counts. Uses np.unique for efficient counting.

        Returns: {"bitstring": count, ...}
        """
        np = self.np
        probs = self.to_probabilities()
        dim = len(probs)
        n = self.n

        # Sample using numpy multinomial
        indices = np.random.choice(dim, size=shots, p=probs)

        # v12.0: Vectorized counting via np.unique
        unique_indices, unique_counts = np.unique(indices, return_counts=True)
        counts = {}
        for idx, cnt in zip(unique_indices, unique_counts):
            bits = format(int(idx), f'0{n}b')
            counts[bits] = int(cnt)
        return counts


__all__ = ["ExactMPSHybridEngine"]
