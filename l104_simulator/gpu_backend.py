"""
===============================================================================
L104 SIMULATOR — GPU ACCELERATION BACKEND
===============================================================================

Transparent GPU acceleration layer that uses CuPy when a CUDA GPU is
available, with automatic fallback to NumPy on CPU.

The ``xp`` module-level variable is the active array library (cupy or numpy).
All simulator code should use ``xp`` instead of ``np`` for array operations
that dominate runtime (statevector allocation, einsum, matmul, SVD).

Design:
  ┌─────────────────────────────────────────────────┐
  │  gpu_backend.xp   →  cupy   OR   numpy          │
  │  gpu_backend.GPU_AVAILABLE   →  True / False     │
  │  gpu_backend.to_device(arr)  →  GPU array        │
  │  gpu_backend.to_host(arr)    →  CPU numpy array  │
  │  gpu_backend.estimate_gpu_memory(n_qubits)       │
  │  gpu_backend.gpu_info()      →  device stats     │
  └─────────────────────────────────────────────────┘

Memory model:
  Complex128 statevector of n qubits = 2^n × 16 bytes.
  26 qubits = 67,108,864 × 16 = 1,073,741,824 bytes ≈ 1.0 GB.
  Most modern GPUs (≥4 GB) handle 26Q easily; 30Q needs ≥16 GB.

INVARIANT: 527.5184818492612 | PILOT: LONDEL
===============================================================================
"""

from __future__ import annotations

import math
import os
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
#  GPU DETECTION — Try CuPy, fall back to NumPy
# ═══════════════════════════════════════════════════════════════════════════════

GPU_AVAILABLE: bool = False
GPU_DEVICE_NAME: str = "CPU (NumPy)"
GPU_MEMORY_TOTAL: int = 0     # bytes
GPU_MEMORY_FREE: int = 0      # bytes

# Allow user to force CPU via environment variable
_FORCE_CPU = os.environ.get("L104_FORCE_CPU", "").lower() in ("1", "true", "yes")

try:
    if _FORCE_CPU:
        raise ImportError("Forced CPU mode via L104_FORCE_CPU")
    import cupy as cp
    # Verify a device is actually reachable
    _dev = cp.cuda.Device(0)
    _dev.use()
    GPU_AVAILABLE = True
    GPU_DEVICE_NAME = f"CUDA:{_dev.id} ({cp.cuda.runtime.getDeviceProperties(_dev.id)['name'].decode()})"
    _mem = _dev.mem_info
    GPU_MEMORY_TOTAL = _mem[1]
    GPU_MEMORY_FREE = _mem[0]
    xp = cp
except ImportError:
    xp = np
except Exception:
    xp = np


def is_gpu() -> bool:
    """Return True if GPU (CuPy) is active."""
    return GPU_AVAILABLE


def to_device(arr: np.ndarray) -> Any:
    """Transfer a NumPy array to the active device (GPU or no-op for CPU)."""
    if GPU_AVAILABLE:
        return cp.asarray(arr)
    return arr


def to_host(arr: Any) -> np.ndarray:
    """Transfer an array back to CPU NumPy (no-op if already NumPy)."""
    if GPU_AVAILABLE and hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


def synchronize() -> None:
    """Synchronize GPU stream (no-op on CPU)."""
    if GPU_AVAILABLE:
        cp.cuda.Stream.null.synchronize()


# ═══════════════════════════════════════════════════════════════════════════════
#  MEMORY ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_statevector_bytes(n_qubits: int, dtype: str = "complex128") -> int:
    """Estimate memory for a statevector of n_qubits.

    Returns bytes required for the statevector alone.
    Actual simulation needs ~2-3× this (workspace arrays for gate application).
    """
    element_bytes = 16 if "128" in dtype else 8   # complex128=16, complex64=8
    return (2 ** n_qubits) * element_bytes


def estimate_unitary_bytes(n_qubits: int) -> int:
    """Estimate memory for the full unitary matrix (O(4^n) — the old approach)."""
    dim = 2 ** n_qubits
    return dim * dim * 16  # complex128


def max_statevector_qubits(memory_budget_bytes: Optional[int] = None,
                           safety_factor: float = 3.0) -> int:
    """Determine the maximum qubit count for statevector simulation.

    Args:
        memory_budget_bytes: Available memory. If None, uses GPU VRAM or 8 GB CPU.
        safety_factor: Multiplier for workspace overhead (default 3×).

    Returns:
        Maximum number of qubits that fit within the budget.
    """
    if memory_budget_bytes is None:
        if GPU_AVAILABLE:
            memory_budget_bytes = GPU_MEMORY_FREE
        else:
            # Default CPU budget: use 75% of reported RAM, capped at 16 GB
            try:
                import psutil
                memory_budget_bytes = int(psutil.virtual_memory().available * 0.75)
            except ImportError:
                memory_budget_bytes = 8 * (1024 ** 3)  # 8 GB fallback

    usable = memory_budget_bytes / safety_factor
    # 2^n × 16 bytes ≤ usable  →  n ≤ log2(usable / 16)
    if usable < 32:
        return 0
    return int(math.log2(usable / 16))


def fits_in_memory(n_qubits: int, device: str = "auto") -> bool:
    """Check if an n-qubit statevector fits in the target device memory."""
    needed = estimate_statevector_bytes(n_qubits) * 3  # 3× for workspace
    if device == "auto":
        device = "gpu" if GPU_AVAILABLE else "cpu"
    if device == "gpu" and GPU_AVAILABLE:
        return needed <= GPU_MEMORY_FREE
    try:
        import psutil
        return needed <= psutil.virtual_memory().available * 0.75
    except ImportError:
        return n_qubits <= 26  # Conservative default


# ═══════════════════════════════════════════════════════════════════════════════
#  GPU-ACCELERATED PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════════

def einsum(subscripts: str, *operands, **kwargs) -> Any:
    """GPU-aware einsum — delegates to CuPy or NumPy."""
    return xp.einsum(subscripts, *operands, **kwargs)


def svd(matrix: Any, full_matrices: bool = False) -> Tuple[Any, Any, Any]:
    """GPU-aware SVD."""
    return xp.linalg.svd(matrix, full_matrices=full_matrices)


def zeros(shape, dtype=complex) -> Any:
    """GPU-aware zeros."""
    return xp.zeros(shape, dtype=dtype)


def eye(n: int, dtype=complex) -> Any:
    """GPU-aware identity matrix."""
    return xp.eye(n, dtype=dtype)


def matmul(a: Any, b: Any) -> Any:
    """GPU-aware matrix multiplication."""
    return a @ b


def diag(v: Any) -> Any:
    """GPU-aware diagonal matrix."""
    return xp.diag(v)


def abs_squared(arr: Any) -> Any:
    """Compute |arr|² element-wise."""
    return xp.abs(arr) ** 2


# ═══════════════════════════════════════════════════════════════════════════════
#  GPU INFO
# ═══════════════════════════════════════════════════════════════════════════════

def gpu_info() -> Dict[str, Any]:
    """Return GPU status and memory information."""
    info: Dict[str, Any] = {
        "gpu_available": GPU_AVAILABLE,
        "device_name": GPU_DEVICE_NAME,
        "forced_cpu": _FORCE_CPU,
    }
    if GPU_AVAILABLE:
        try:
            _dev = cp.cuda.Device(0)
            mem = _dev.mem_info
            info.update({
                "memory_total_gb": round(mem[1] / (1024**3), 2),
                "memory_free_gb": round(mem[0] / (1024**3), 2),
                "memory_used_gb": round((mem[1] - mem[0]) / (1024**3), 2),
                "max_statevector_qubits": max_statevector_qubits(mem[0]),
                "compute_capability": _dev.compute_capability,
            })
        except Exception as e:
            info["error"] = str(e)
    else:
        info.update({
            "max_statevector_qubits_cpu": max_statevector_qubits(),
            "note": "Install CuPy for GPU acceleration: pip install cupy-cuda12x",
        })
    return info


# ═══════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE — Apply single/two-qubit gate with GPU arrays
# ═══════════════════════════════════════════════════════════════════════════════

def apply_single_gate_gpu(state: Any, gate: Any, qubit: int,
                          n_total: int) -> Any:
    """Apply a 2×2 gate to a statevector on the active device.

    Same algorithm as Simulator._apply_single but uses xp instead of np.
    """
    shape = (2 ** qubit, 2, 2 ** (n_total - qubit - 1))
    psi = state.reshape(shape)
    out = xp.einsum('ij,ajb->aib', gate, psi)
    return out.reshape(-1)


def apply_two_gate_gpu(state: Any, gate: Any, q0: int, q1: int,
                       n_total: int) -> Any:
    """Apply a 4×4 gate to a statevector on the active device.

    Same algorithm as Simulator._apply_two but uses xp instead of np.
    """
    if q0 > q1:
        q0, q1 = q1, q0
        gate = gate.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(4, 4)

    a = q0
    b = q1 - q0 - 1
    c = n_total - q1 - 1

    shape = (2**a, 2, 2**b, 2, 2**c)
    psi = state.reshape(shape)
    gate_t = gate.reshape(2, 2, 2, 2)
    out = xp.einsum('ijkl,akblc->aibjc', gate_t, psi)
    return out.reshape(-1)


def apply_general_gate_gpu(state: Any, gate: Any, qubits: list,
                           n_total: int) -> Any:
    """Apply an N-qubit gate on the active device."""
    n_gate = len(qubits)
    others = [q for q in range(n_total) if q not in qubits]
    psi = state.reshape([2] * n_total)
    perm = list(qubits) + others
    psi = xp.transpose(psi, perm)
    d_gate = 2 ** n_gate
    d_other = 2 ** len(others)
    psi = psi.reshape(d_gate, d_other)
    psi = gate @ psi
    psi = psi.reshape([2] * n_total)
    inv_perm = [0] * n_total
    for i, p in enumerate(perm):
        inv_perm[p] = i
    psi = xp.transpose(psi, inv_perm)
    return psi.reshape(-1)
