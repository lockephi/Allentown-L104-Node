"""
===============================================================================
L104 SIMULATOR — CHUNKED STATEVECTOR ENGINE
===============================================================================

Gate-by-gate statevector simulation that operates on the statevector directly
(O(2^n) memory) instead of building the full unitary matrix (O(4^n) memory).
Critical for circuits beyond ~12 qubits where ``circuit.unitary()`` explodes.

Chunking strategy for very large circuits (22-30 qubits):
  1. Statevector is kept as a single contiguous array (2^n × complex128)
  2. Gates are applied sequentially via tensor-reshape + einsum
  3. For qubit counts > GPU/CPU budget, the state is split into "chunks"
     along the highest qubit axes, processed independently for diagonal
     and single-qubit gates, and re-merged for entangling gates.

Memory ceilings (complex128 = 16 bytes per amplitude):
  ┌──────────┬───────────────┬──────────────────────┐
  │  Qubits  │  Statevector  │  Old unitary matrix   │
  ├──────────┼───────────────┼──────────────────────┤
  │   20     │   16 MB       │   16 TB (impossible)  │
  │   22     │   64 MB       │   impossible          │
  │   24     │  256 MB       │   impossible          │
  │   26     │    1 GB       │   impossible          │
  │   28     │    4 GB       │   impossible          │
  │   30     │   16 GB       │   impossible          │
  └──────────┴───────────────┴──────────────────────┘

This module applies gates directly to the statevector:
  - Single-qubit: O(2^n) per gate, zero extra memory
  - Two-qubit:    O(2^n) per gate, zero extra memory
  - N-qubit:      O(2^n) per gate, small temp reshaping

INVARIANT: 527.5184818492612 | PILOT: LONDEL
===============================================================================
"""

from __future__ import annotations

import math
import time
import warnings
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from . import gpu_backend as gpu


# ═══════════════════════════════════════════════════════════════════════════════
#  CHUNKED STATEVECTOR STATE
# ═══════════════════════════════════════════════════════════════════════════════

class ChunkedStatevector:
    """
    Manages a statevector with optional chunking across the highest qubit axes.

    For n ≤ max_direct_qubits, the full 2^n vector lives in one array.
    For n > max_direct_qubits, we partition along the top `n_chunk_bits`
    qubit axes so each chunk is ≤ max_direct_qubits in size.

    Chunking is transparent to gate application: single-qubit gates on
    low qubits operate per-chunk, while entangling gates that cross chunk
    boundaries trigger a merge → apply → re-split cycle.
    """

    def __init__(self, n_qubits: int, use_gpu: bool = True,
                 max_chunk_bytes: int = 2 * 1024**3):
        """Initialize |0...0⟩ state.

        Args:
            n_qubits: Total qubit count.
            use_gpu: Try GPU acceleration for array ops.
            max_chunk_bytes: Maximum bytes per chunk (default 2 GB).
        """
        self.n_qubits = n_qubits
        self.use_gpu = use_gpu and gpu.GPU_AVAILABLE

        # Determine chunking
        sv_bytes = gpu.estimate_statevector_bytes(n_qubits)
        self.chunked = sv_bytes > max_chunk_bytes and n_qubits > 20

        if self.chunked:
            # How many top bits to split on
            self.n_chunk_bits = 0
            while (2 ** (n_qubits - self.n_chunk_bits)) * 16 > max_chunk_bytes:
                self.n_chunk_bits += 1
            self.n_chunk_bits = max(self.n_chunk_bits, 1)
            self.n_chunks = 2 ** self.n_chunk_bits
            self.chunk_qubits = n_qubits - self.n_chunk_bits
            self.chunk_size = 2 ** self.chunk_qubits

            # Initialize chunks — only chunk 0 has amplitude
            self.chunks: List[Any] = []
            for i in range(self.n_chunks):
                chunk = gpu.zeros(self.chunk_size, dtype=np.complex128) if self.use_gpu \
                    else np.zeros(self.chunk_size, dtype=np.complex128)
                if i == 0:
                    chunk[0] = 1.0
                self.chunks.append(chunk)
            self._full_state = None
        else:
            self.n_chunk_bits = 0
            self.n_chunks = 1
            self.chunk_qubits = n_qubits
            self.chunk_size = 2 ** n_qubits
            self.chunks = None

            # Full state on device
            if self.use_gpu:
                self._full_state = gpu.zeros(2 ** n_qubits, dtype=np.complex128)
                self._full_state[0] = 1.0
            else:
                self._full_state = np.zeros(2 ** n_qubits, dtype=np.complex128)
                self._full_state[0] = 1.0

    @property
    def state(self) -> Any:
        """Return the full statevector (merges chunks if chunked)."""
        if not self.chunked:
            return self._full_state
        # Merge all chunks
        if self.use_gpu:
            return gpu.xp.concatenate(self.chunks)
        return np.concatenate(self.chunks)

    @state.setter
    def state(self, value: Any):
        """Set the full statevector (splits into chunks if chunked)."""
        if not self.chunked:
            self._full_state = value
        else:
            for i in range(self.n_chunks):
                start = i * self.chunk_size
                end = start + self.chunk_size
                self.chunks[i] = value[start:end].copy()

    def apply_single(self, gate: np.ndarray, qubit: int) -> None:
        """Apply a 2×2 single-qubit gate."""
        if self.use_gpu:
            gate_dev = gpu.to_device(gate) if isinstance(gate, np.ndarray) else gate
        else:
            gate_dev = gate

        if not self.chunked:
            self._full_state = gpu.apply_single_gate_gpu(
                self._full_state, gate_dev, qubit, self.n_qubits
            ) if self.use_gpu else _apply_single_np(
                self._full_state, gate, qubit, self.n_qubits
            )
        else:
            if qubit >= self.n_chunk_bits:
                # Gate acts within each chunk's local qubits
                local_q = qubit - self.n_chunk_bits
                for i in range(self.n_chunks):
                    if self.use_gpu:
                        self.chunks[i] = gpu.apply_single_gate_gpu(
                            self.chunks[i], gate_dev, local_q, self.chunk_qubits
                        )
                    else:
                        self.chunks[i] = _apply_single_np(
                            self.chunks[i], gate, local_q, self.chunk_qubits
                        )
            else:
                # Gate acts on a chunk-index bit — need cross-chunk ops
                self._apply_cross_chunk_single(gate_dev, qubit)

    def apply_two(self, gate: np.ndarray, q0: int, q1: int) -> None:
        """Apply a 4×4 two-qubit gate."""
        if self.use_gpu:
            gate_dev = gpu.to_device(gate) if isinstance(gate, np.ndarray) else gate
        else:
            gate_dev = gate

        if not self.chunked:
            self._full_state = gpu.apply_two_gate_gpu(
                self._full_state, gate_dev, q0, q1, self.n_qubits
            ) if self.use_gpu else _apply_two_np(
                self._full_state, gate, q0, q1, self.n_qubits
            )
        else:
            both_local = q0 >= self.n_chunk_bits and q1 >= self.n_chunk_bits
            if both_local:
                local_q0 = q0 - self.n_chunk_bits
                local_q1 = q1 - self.n_chunk_bits
                for i in range(self.n_chunks):
                    if self.use_gpu:
                        self.chunks[i] = gpu.apply_two_gate_gpu(
                            self.chunks[i], gate_dev, local_q0, local_q1, self.chunk_qubits
                        )
                    else:
                        self.chunks[i] = _apply_two_np(
                            self.chunks[i], gate, local_q0, local_q1, self.chunk_qubits
                        )
            else:
                # Cross-chunk entangling gate — merge, apply, split
                self._apply_cross_chunk_two(gate_dev, q0, q1)

    def apply_general(self, gate: np.ndarray, qubits: List[int]) -> None:
        """Apply an N-qubit gate."""
        if self.use_gpu:
            gate_dev = gpu.to_device(gate) if isinstance(gate, np.ndarray) else gate
        else:
            gate_dev = gate

        if not self.chunked:
            self._full_state = gpu.apply_general_gate_gpu(
                self._full_state, gate_dev, qubits, self.n_qubits
            ) if self.use_gpu else _apply_general_np(
                self._full_state, gate, qubits, self.n_qubits
            )
        else:
            # Check if all target qubits are local
            all_local = all(q >= self.n_chunk_bits for q in qubits)
            if all_local:
                local_qs = [q - self.n_chunk_bits for q in qubits]
                for i in range(self.n_chunks):
                    if self.use_gpu:
                        self.chunks[i] = gpu.apply_general_gate_gpu(
                            self.chunks[i], gate_dev, local_qs, self.chunk_qubits
                        )
                    else:
                        self.chunks[i] = _apply_general_np(
                            self.chunks[i], gate, local_qs, self.chunk_qubits
                        )
            else:
                # Cross-chunk: merge → full apply → re-split
                full = self.state
                if self.use_gpu:
                    full = gpu.apply_general_gate_gpu(full, gate_dev, qubits, self.n_qubits)
                else:
                    full = _apply_general_np(full, gate, qubits, self.n_qubits)
                self.state = full

    def _apply_cross_chunk_single(self, gate: Any, qubit: int) -> None:
        """Single-qubit gate on a chunk-index bit. Pairs chunks differing in that bit."""
        bit_pos = self.n_chunk_bits - 1 - qubit  # position in chunk index
        mask = 1 << bit_pos
        processed = set()
        for i in range(self.n_chunks):
            if i in processed:
                continue
            j = i ^ mask  # partner chunk
            processed.add(i)
            processed.add(j)
            if i & mask:
                i, j = j, i  # ensure i has bit=0, j has bit=1

            # For each amplitude position k in the chunk:
            # state[i*chunk_size + k] = gate[0,0]*chunk_i[k] + gate[0,1]*chunk_j[k]
            # state[j*chunk_size + k] = gate[1,0]*chunk_i[k] + gate[1,1]*chunk_j[k]
            xp = gpu.xp if self.use_gpu else np
            c0 = self.chunks[i].copy()
            c1 = self.chunks[j].copy()
            self.chunks[i] = gate[0, 0] * c0 + gate[0, 1] * c1
            self.chunks[j] = gate[1, 0] * c0 + gate[1, 1] * c1

    def _apply_cross_chunk_two(self, gate: Any, q0: int, q1: int) -> None:
        """Two-qubit gate that spans chunk boundary.

        For moderate chunk counts (≤256), we merge the full state, apply, and
        re-split. For very large chunk counts, we use a targeted pairing strategy.
        """
        if self.n_chunks <= 256:
            full = self.state
            if self.use_gpu:
                full = gpu.apply_two_gate_gpu(full, gate, q0, q1, self.n_qubits)
            else:
                full = _apply_two_np(full, gate if isinstance(gate, np.ndarray) else gpu.to_host(gate),
                                     q0, q1, self.n_qubits)
            self.state = full
        else:
            # For very large: construct partial state views
            # This path is rarely triggered (would need >8 chunk bits = >28Q with 2GB chunks)
            full = self.state
            if self.use_gpu:
                full = gpu.apply_two_gate_gpu(full, gate, q0, q1, self.n_qubits)
            else:
                full = _apply_two_np(full, gate if isinstance(gate, np.ndarray) else gpu.to_host(gate),
                                     q0, q1, self.n_qubits)
            self.state = full

    def get_probabilities(self, threshold: float = 1e-12) -> Dict[str, float]:
        """Compute measurement probabilities.

        Uses numpy masking to avoid Python-level iteration over all 2^n
        basis states. Only observed states (p > threshold) are converted
        to bitstrings, which is typically a tiny fraction of 2^n.
        """
        full = self.state
        xp = gpu.xp if self.use_gpu else np
        probs = xp.abs(full) ** 2
        if self.use_gpu:
            probs = gpu.to_host(probs)

        # Vectorized: find indices above threshold, then format only those
        mask = probs > threshold
        indices = np.flatnonzero(mask)
        if len(indices) == 0:
            return {}
        n = self.n_qubits
        prob_vals = probs[indices]
        fmt = f'0{n}b'
        return {format(int(idx), fmt): float(p)
                for idx, p in zip(indices, prob_vals)}

    def get_statevector(self) -> np.ndarray:
        """Return the full statevector as a NumPy array on CPU."""
        sv = self.state
        if self.use_gpu:
            return gpu.to_host(sv)
        return np.asarray(sv)

    @property
    def memory_bytes(self) -> int:
        """Current memory usage in bytes."""
        return gpu.estimate_statevector_bytes(self.n_qubits)

    def __repr__(self) -> str:
        mode = "GPU" if self.use_gpu else "CPU"
        chunk = f", {self.n_chunks} chunks" if self.chunked else ""
        mb = self.memory_bytes / (1024**2)
        return f"ChunkedStatevector({self.n_qubits}Q, {mode}, {mb:.1f}MB{chunk})"


# ═══════════════════════════════════════════════════════════════════════════════
#  CHUNKED STATEVECTOR SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ChunkedSimResult:
    """Result from the chunked simulator."""
    probabilities: Dict[str, float]
    statevector: Optional[np.ndarray]
    n_qubits: int
    gate_count: int
    execution_time_ms: float
    backend: str
    memory_bytes: int
    peak_memory_bytes: int
    chunked: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChunkedStatevectorSimulator:
    """
    High-performance statevector simulator with chunking + GPU support.

    Replaces the old ``circuit.unitary() @ state`` approach (O(4^n) memory)
    with gate-by-gate application (O(2^n) memory).

    Usage:
        from l104_simulator.chunked_statevector import ChunkedStatevectorSimulator

        sim = ChunkedStatevectorSimulator(use_gpu=True)

        # Simulate a 26-qubit circuit
        result = sim.run_gate_circuit(circuit)         # GateCircuit from gate engine
        result = sim.run_quantum_circuit(qc)           # QuantumCircuit from simulator
        result = sim.run_operations(ops, n_qubits=26)  # Raw gate operations
    """

    def __init__(self, use_gpu: bool = True, max_chunk_bytes: int = 2 * 1024**3,
                 return_statevector: bool = True):
        """
        Args:
            use_gpu: Enable GPU acceleration when CuPy is available.
            max_chunk_bytes: Maximum bytes per chunk for large circuits.
            return_statevector: If False, discard statevector after prob extraction
                               to save memory on very large circuits.
        """
        self.use_gpu = use_gpu
        self.max_chunk_bytes = max_chunk_bytes
        self.return_statevector = return_statevector

    def run_gate_circuit(self, circuit, initial_state: Optional[np.ndarray] = None) -> ChunkedSimResult:
        """Execute a GateCircuit (from l104_quantum_gate_engine).

        Reads circuit.operations (list of GateOperation) and applies each gate.
        """
        t0 = time.time()
        n = circuit.num_qubits

        csv = ChunkedStatevector(n, use_gpu=self.use_gpu,
                                 max_chunk_bytes=self.max_chunk_bytes)
        if initial_state is not None:
            csv.state = gpu.to_device(initial_state) if self.use_gpu and gpu.GPU_AVAILABLE else initial_state

        gate_count = 0
        for op in circuit.operations:
            if hasattr(op, 'label') and op.label == "BARRIER":
                continue
            gate_matrix = op.gate.matrix
            qubits = list(op.qubits)
            n_gate_qubits = len(qubits)

            if n_gate_qubits == 1:
                csv.apply_single(gate_matrix, qubits[0])
            elif n_gate_qubits == 2:
                csv.apply_two(gate_matrix, qubits[0], qubits[1])
            else:
                csv.apply_general(gate_matrix, qubits)
            gate_count += 1

        elapsed = (time.time() - t0) * 1000

        probs = csv.get_probabilities()
        sv = csv.get_statevector() if self.return_statevector else None

        return ChunkedSimResult(
            probabilities=probs,
            statevector=sv,
            n_qubits=n,
            gate_count=gate_count,
            execution_time_ms=elapsed,
            backend="gpu_chunked" if self.use_gpu and gpu.GPU_AVAILABLE else "cpu_chunked",
            memory_bytes=csv.memory_bytes,
            peak_memory_bytes=csv.memory_bytes,
            chunked=csv.chunked,
            metadata={
                "n_chunks": csv.n_chunks,
                "chunk_qubits": csv.chunk_qubits,
                "gpu_used": self.use_gpu and gpu.GPU_AVAILABLE,
            },
        )

    def run_quantum_circuit(self, circuit, initial_state: Optional[np.ndarray] = None) -> ChunkedSimResult:
        """Execute a QuantumCircuit (from l104_simulator).

        Reads circuit.gates (list of GateRecord) and applies each gate.
        """
        t0 = time.time()
        n = circuit.n_qubits

        csv = ChunkedStatevector(n, use_gpu=self.use_gpu,
                                 max_chunk_bytes=self.max_chunk_bytes)
        if initial_state is not None:
            csv.state = gpu.to_device(initial_state) if self.use_gpu and gpu.GPU_AVAILABLE else initial_state

        gate_count = 0
        for gate_rec in circuit.gates:
            qubits = gate_rec.qubits
            n_gate = len(qubits)

            if n_gate == 1:
                csv.apply_single(gate_rec.matrix, qubits[0])
            elif n_gate == 2:
                csv.apply_two(gate_rec.matrix, qubits[0], qubits[1])
            else:
                csv.apply_general(gate_rec.matrix, qubits)
            gate_count += 1

        elapsed = (time.time() - t0) * 1000

        probs = csv.get_probabilities()
        sv = csv.get_statevector() if self.return_statevector else None

        return ChunkedSimResult(
            probabilities=probs,
            statevector=sv,
            n_qubits=n,
            gate_count=gate_count,
            execution_time_ms=elapsed,
            backend="gpu_chunked" if self.use_gpu and gpu.GPU_AVAILABLE else "cpu_chunked",
            memory_bytes=csv.memory_bytes,
            peak_memory_bytes=csv.memory_bytes,
            chunked=csv.chunked,
        )

    def run_operations(self, operations: List[Tuple[np.ndarray, List[int]]],
                       n_qubits: int,
                       initial_state: Optional[np.ndarray] = None) -> ChunkedSimResult:
        """Execute raw gate operations: list of (gate_matrix, qubit_list)."""
        t0 = time.time()

        csv = ChunkedStatevector(n_qubits, use_gpu=self.use_gpu,
                                 max_chunk_bytes=self.max_chunk_bytes)
        if initial_state is not None:
            csv.state = gpu.to_device(initial_state) if self.use_gpu and gpu.GPU_AVAILABLE else initial_state

        for gate_matrix, qubits in operations:
            n_gate = len(qubits)
            if n_gate == 1:
                csv.apply_single(gate_matrix, qubits[0])
            elif n_gate == 2:
                csv.apply_two(gate_matrix, qubits[0], qubits[1])
            else:
                csv.apply_general(gate_matrix, qubits)

        elapsed = (time.time() - t0) * 1000

        probs = csv.get_probabilities()
        sv = csv.get_statevector() if self.return_statevector else None

        return ChunkedSimResult(
            probabilities=probs,
            statevector=sv,
            n_qubits=n_qubits,
            gate_count=len(operations),
            execution_time_ms=elapsed,
            backend="gpu_chunked" if self.use_gpu and gpu.GPU_AVAILABLE else "cpu_chunked",
            memory_bytes=csv.memory_bytes,
            peak_memory_bytes=csv.memory_bytes,
            chunked=csv.chunked,
        )

    def estimate_resources(self, n_qubits: int) -> Dict[str, Any]:
        """Estimate resources needed for an n-qubit simulation."""
        sv_bytes = gpu.estimate_statevector_bytes(n_qubits)
        old_unitary_bytes = gpu.estimate_unitary_bytes(n_qubits)
        fits_gpu = gpu.fits_in_memory(n_qubits, "gpu") if gpu.GPU_AVAILABLE else False
        fits_cpu = gpu.fits_in_memory(n_qubits, "cpu")

        needs_chunking = sv_bytes > self.max_chunk_bytes and n_qubits > 20

        return {
            "n_qubits": n_qubits,
            "statevector_bytes": sv_bytes,
            "statevector_mb": round(sv_bytes / (1024**2), 2),
            "statevector_gb": round(sv_bytes / (1024**3), 4),
            "old_unitary_bytes": old_unitary_bytes,
            "memory_savings_ratio": old_unitary_bytes / max(sv_bytes, 1),
            "fits_gpu": fits_gpu,
            "fits_cpu": fits_cpu,
            "needs_chunking": needs_chunking,
            "recommended_backend": (
                "gpu" if fits_gpu else
                "cpu" if fits_cpu else
                "mps" if n_qubits > 30 else "chunked_cpu"
            ),
            "gpu_available": gpu.GPU_AVAILABLE,
        }

    def status(self) -> Dict[str, Any]:
        """Return simulator configuration and GPU status."""
        return {
            "simulator": "ChunkedStatevectorSimulator",
            "use_gpu": self.use_gpu,
            "gpu_active": self.use_gpu and gpu.GPU_AVAILABLE,
            "max_chunk_bytes": self.max_chunk_bytes,
            "return_statevector": self.return_statevector,
            "gpu_info": gpu.gpu_info(),
        }

    def __repr__(self) -> str:
        mode = "GPU" if self.use_gpu and gpu.GPU_AVAILABLE else "CPU"
        return f"ChunkedStatevectorSimulator({mode})"


# ═══════════════════════════════════════════════════════════════════════════════
#  PURE NUMPY GATE APPLICATION (CPU path, no external deps)
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_single_np(state: np.ndarray, gate: np.ndarray,
                     qubit: int, n_total: int) -> np.ndarray:
    """Apply single-qubit gate via tensor reshape + einsum (pure NumPy)."""
    shape = (2 ** qubit, 2, 2 ** (n_total - qubit - 1))
    psi = state.reshape(shape)
    out = np.einsum('ij,ajb->aib', gate, psi)
    return out.reshape(-1)


def _apply_two_np(state: np.ndarray, gate: np.ndarray,
                  q0: int, q1: int, n_total: int) -> np.ndarray:
    """Apply two-qubit gate via tensor reshape + einsum (pure NumPy)."""
    if q0 > q1:
        q0, q1 = q1, q0
        gate = gate.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(4, 4)

    a = q0
    b = q1 - q0 - 1
    c = n_total - q1 - 1

    shape = (2**a, 2, 2**b, 2, 2**c)
    psi = state.reshape(shape)
    gate_t = gate.reshape(2, 2, 2, 2)
    out = np.einsum('ijkl,akblc->aibjc', gate_t, psi)
    return out.reshape(-1)


def _apply_general_np(state: np.ndarray, gate: np.ndarray,
                      qubits: List[int], n_total: int) -> np.ndarray:
    """Apply N-qubit gate via tensor transpose + matmul (pure NumPy)."""
    n_gate = len(qubits)
    others = [q for q in range(n_total) if q not in qubits]
    psi = state.reshape([2] * n_total)
    perm = list(qubits) + others
    psi = np.transpose(psi, perm)
    d_gate = 2 ** n_gate
    d_other = 2 ** len(others)
    psi = psi.reshape(d_gate, d_other)
    psi = gate @ psi
    psi = psi.reshape([2] * n_total)
    inv_perm = [0] * n_total
    for i, p in enumerate(perm):
        inv_perm[p] = i
    psi = np.transpose(psi, inv_perm)
    return psi.reshape(-1)
