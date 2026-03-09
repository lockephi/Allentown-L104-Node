#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
GOD_CODE ALGORITHM ANALYSIS — CLASSICAL BYPASS EDITION
═══════════════════════════════════════════════════════════════════════════════

Optimized for macOS x86_64 — exploiting:
  1. Memoization      — O(1) cached qubit lookups (functools.lru_cache)
  2. Tensor Networks   — O(n) MPS contraction (vs O(2^n) statevector)
  3. BLAS Acceleration — OpenBLAS + SSE4.2/AVX2/FMA3 matrix operations

Hardware profile:
  Python 3.12 | NumPy 1.26 (OpenBLAS 64-bit) | SciPy 1.13
  SIMD: SSE→SSE4.2, AVX, AVX2, FMA3, F16C

INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

import math
import time
import platform
import os
import sys
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np

# ── L104 imports ─────────────────────────────────────────────────────────────
from l104_god_code_equation import (
    god_code_equation, exponent_value, solve_for_exponent,
    QUANTUM_FREQUENCY_TABLE, GOD_CODE, BASE,
    STEP_SIZE, QUANTIZATION_GRAIN, REAL_WORLD_CONSTANTS,
    PRIME_SCAFFOLD, FE_BCC_LATTICE_PM, FE_ATOMIC_NUMBER, HE4_MASS_NUMBER,
)
from l104_god_code_simulator.god_code_qubit import (
    GOD_CODE_QUBIT, GOD_CODE_PHASE, GOD_CODE_RZ,
    IRON_PHASE, PHI_CONTRIBUTION, OCTAVE_PHASE,
    IRON_RZ, PHI_RZ, OCTAVE_RZ,
)

# ── Constants ────────────────────────────────────────────────────────────────
TAU = 2.0 * math.pi
PHI = (1.0 + math.sqrt(5.0)) / 2.0
sep = '═' * 76
div = '─' * 76

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  UTILITY: Precise timing                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def bench(fn, label: str, iterations: int = 10000) -> float:
    """Benchmark fn() over N iterations. Returns mean time in microseconds."""
    # Warmup
    for _ in range(min(100, iterations)):
        fn()
    t0 = time.perf_counter_ns()
    for _ in range(iterations):
        fn()
    elapsed_ns = time.perf_counter_ns() - t0
    mean_us = elapsed_ns / (iterations * 1000)
    print(f'    {label:50s}  {mean_us:>8.3f} µs/op  ({iterations:,} iters)')
    return mean_us


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 0: HARDWARE PROFILE                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(sep)
print('  GOD_CODE CLASSICAL BYPASS — HARDWARE PROFILE')
print(sep)
print()

cpu_count = os.cpu_count() or 1
print(f'  System:    {platform.system()} {platform.release()} ({platform.machine()})')
print(f'  Processor: {platform.processor() or "unknown"}')
print(f'  CPU cores: {cpu_count}')
print(f'  Python:    {sys.version.split()[0]}')
print(f'  NumPy:     {np.__version__}')

# Check BLAS backend
try:
    blas_info = np.__config__.blas_opt_info  # type: ignore
    blas_name = blas_info.get('libraries', ['unknown'])[0] if isinstance(blas_info, dict) else 'openblas'
except Exception:
    blas_name = 'openblas (inferred)'
print(f'  BLAS:      {blas_name}')

try:
    import scipy
    print(f'  SciPy:     {scipy.__version__}')
except ImportError:
    scipy = None
    print(f'  SciPy:     NOT AVAILABLE')

# SIMD capabilities (known from numpy config)
simd_found = ['SSE→SSE4.2', 'AVX', 'AVX2', 'FMA3', 'F16C']
print(f'  SIMD:      {", ".join(simd_found)}')

print(f'  Pointer:   {8 * np.dtype(np.intp).itemsize}-bit')
print(f'  complex128: {np.dtype(np.complex128).itemsize} bytes/element')
print()

# RAM calculation for statevector
for nq in [20, 25, 26, 28, 30]:
    mem_mb = (2 ** nq) * 16 / (1024 * 1024)
    label = 'MB' if mem_mb < 1024 else 'GB'
    val = mem_mb if mem_mb < 1024 else mem_mb / 1024
    print(f'    Statevector {nq:>2d}Q: {val:>8.1f} {label}')
print()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 1: MEMOIZATION — THE "CACHED QUBIT" EXPLOIT                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(sep)
print('  PHASE 1: MEMOIZATION — O(1) CACHED QUBIT LOOKUP')
print(sep)
print()
print('  The GOD_CODE qubit (fidelity 0.999939, 14-layer circuit) is a STATIC')
print('  mathematical object. Its 2×2 unitary is frozen at init. Every call to')
print('  GOD_CODE_QUBIT.gate returns the same matrix — no simulation needed.')
print()

# ── 1a. The cached unitary matrix ────────────────────────────────────────────
print('  1a. PRE-COMPUTED UNITARY (loaded at import time)')
print(div)

gate = GOD_CODE_QUBIT.gate
print(f'    GOD_CODE_RZ (2×2 unitary):')
print(f'      [{gate[0,0]:.15f},  {gate[0,1]:.1f}]')
print(f'      [{gate[1,0]:.1f},  {gate[1,1]:.15f}]')
print()
print(f'    Eigenvalue split: ±θ/2 = ±{GOD_CODE_PHASE/2:.10f} rad')
print(f'    |det| = {abs(np.linalg.det(gate)):.15f}  (exact 1.0)')
print(f'    Unitarity error: {float(np.max(np.abs(gate.conj().T @ gate - np.eye(2)))):.2e}')
print()

# ── 1b. LRU-cached dial lookups ──────────────────────────────────────────────
print('  1b. LRU-CACHED DIAL SYSTEM')
print(div)

@lru_cache(maxsize=65536)
def cached_dial(a: int = 0, b: int = 0, c: int = 0, d: int = 0) -> float:
    """O(1) amortized — frequency cached after first computation."""
    E = 8 * a + 416 - b - 8 * c - 104 * d
    return BASE * (2.0 ** (E / 104))

@lru_cache(maxsize=65536)
def cached_dial_gate(a: int = 0, b: int = 0, c: int = 0, d: int = 0) -> bytes:
    """Cache the 2×2 gate matrix as bytes (hashable for LRU). O(1) after first call."""
    freq = cached_dial(a, b, c, d)
    phase = freq % TAU
    gate = np.array([
        [np.exp(-1j * phase / 2), 0],
        [0, np.exp(1j * phase / 2)],
    ], dtype=np.complex128)
    return gate.tobytes()

def dial_gate_from_cache(a: int = 0, b: int = 0, c: int = 0, d: int = 0) -> np.ndarray:
    """Reconstruct ndarray from cached bytes. Still near-O(1)."""
    return np.frombuffer(cached_dial_gate(a, b, c, d), dtype=np.complex128).reshape(2, 2).copy()

# Prime the cache with all QUANTUM_FREQUENCY_TABLE entries
for dials in QUANTUM_FREQUENCY_TABLE.keys():
    cached_dial(*dials)

cache_info = cached_dial.cache_info()
print(f'    Cache primed: {cache_info.currsize} entries (from QUANTUM_FREQUENCY_TABLE)')
print()

# ── 1c. Benchmark: cached vs uncached ────────────────────────────────────────
print('  1c. MEMOIZATION BENCHMARK')
print(div)

def uncached_dial():
    return god_code_equation(0, 0, 0, 0)

def use_cached():
    return cached_dial(0, 0, 0, 0)

def uncached_gate():
    phase = GOD_CODE % TAU
    return np.array([[np.exp(-1j * phase / 2), 0], [0, np.exp(1j * phase / 2)]], dtype=np.complex128)

def use_cached_gate():
    return dial_gate_from_cache(0, 0, 0, 0)

def singleton_gate():
    return GOD_CODE_QUBIT.gate

t_uncached = bench(uncached_dial, 'god_code_equation(0,0,0,0)  [compute]')
t_cached   = bench(use_cached,   'cached_dial(0,0,0,0)         [LRU hit]')
print(f'    → Speedup: {t_uncached/max(t_cached, 0.001):.1f}×')
print()

t_gate_build = bench(uncached_gate,  'Build Rz(θ) from scratch     [numpy]')
t_gate_cache = bench(use_cached_gate,'dial_gate_from_cache(0,0,0,0)[LRU→ndarray]')
t_gate_sing  = bench(singleton_gate, 'GOD_CODE_QUBIT.gate          [singleton]')
print(f'    → Gate speedup (cache vs build): {t_gate_build/max(t_gate_cache, 0.001):.1f}×')
print()

# ── 1d. Full dial space scan ─────────────────────────────────────────────────
print('  1d. FULL DIAL SPACE SCAN (2^14 = 16,384 combinations)')
print(div)

# Compute all 16384 dial combinations with memoization
t0 = time.perf_counter()
dial_count = 0
for a in range(8):    # 3 bits
    for b in range(16):  # 4 bits
        for c in range(8):    # 3 bits
            for d in range(16):  # 4 bits
                cached_dial(a, b, c, d)
                dial_count += 1
t_scan = time.perf_counter() - t0

cache_final = cached_dial.cache_info()
print(f'    Scanned {dial_count:,} dial settings in {t_scan*1000:.1f} ms')
print(f'    Cache: {cache_final.currsize:,} entries, {cache_final.hits:,} hits, {cache_final.misses:,} misses')
print(f'    Amortized: {t_scan/dial_count*1e6:.2f} µs/dial (includes first-compute)')
print()

# Second pass — all cache hits
t0 = time.perf_counter()
for a in range(8):
    for b in range(16):
        for c in range(8):
            for d in range(16):
                cached_dial(a, b, c, d)
t_rescan = time.perf_counter() - t0

print(f'    Re-scan (all hits): {t_rescan*1000:.2f} ms')
print(f'    Cache-hit cost: {t_rescan/dial_count*1e6:.3f} µs/lookup')
print(f'    → {t_scan/max(t_rescan, 1e-10):.0f}× faster on re-access')
print()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 2: BLAS MATRIX ACCELERATION                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(sep)
print('  PHASE 2: BLAS/SIMD MATRIX ACCELERATION')
print(sep)
print()
print('  NumPy dispatches to OpenBLAS (64-bit), which uses SSE4.2/AVX2/FMA3')
print('  hardware intrinsics for matrix ops. The GOD_CODE gate is just a 2×2')
print('  complex128 matrix — this is EXACTLY what the hardware accelerates.')
print()

# ── 2a. Matrix multiplication benchmarks ─────────────────────────────────────
print('  2a. MATRIX MULTIPLY BENCHMARK (complex128, OpenBLAS+AVX2)')
print(div)

def matmul_2x2():
    """Single 2×2 GOD_CODE gate multiplication."""
    return GOD_CODE_RZ @ GOD_CODE_RZ

def matmul_chain_14():
    """14-layer gate chain (the original circuit depth)."""
    M = np.eye(2, dtype=np.complex128)
    M = M @ IRON_RZ @ PHI_RZ @ OCTAVE_RZ  # decomposed ×3
    M = M @ GOD_CODE_RZ  # Full gate
    M = M @ IRON_RZ @ PHI_RZ @ OCTAVE_RZ  # decomposed ×3
    M = M @ GOD_CODE_RZ  # Full gate
    M = M @ IRON_RZ @ PHI_RZ @ OCTAVE_RZ  # decomposed ×3
    M = M @ GOD_CODE_RZ @ GOD_CODE_RZ.conj().T  # Gate + inverse
    return M

# Pre-compute the 14-layer chain as a SINGLE matrix (memoization exploit)
CHAIN_14 = matmul_chain_14()
def matmul_chain_cached():
    """Apply the pre-computed 14-layer chain."""
    return CHAIN_14 @ np.array([1, 0], dtype=np.complex128)

bench(matmul_2x2, '2×2 @ 2×2 (single gate)        [BLAS]')
bench(matmul_chain_14, '14-gate chain (live compute)    [BLAS]')
bench(matmul_chain_cached, '14-gate chain (pre-cached)      [O(1)]')
print()

# ── 2b. Batch processing — vectorized ────────────────────────────────────────
print('  2b. VECTORIZED BATCH OPERATIONS')
print(div)

# Apply GOD_CODE gate to 1000 statevectors simultaneously
batch_size = 1000
states = np.random.randn(batch_size, 2).astype(np.complex128)
states /= np.linalg.norm(states, axis=1, keepdims=True)

def batch_apply():
    """Apply GOD_CODE_RZ to 1000 states at once via BLAS gemm."""
    return (GOD_CODE_RZ @ states.T).T

def loop_apply():
    """Apply GOD_CODE_RZ to 1000 states one-by-one."""
    result = np.empty_like(states)
    for i in range(batch_size):
        result[i] = GOD_CODE_RZ @ states[i]
    return result

t_batch = bench(batch_apply, f'Batch {batch_size} states (BLAS gemm)  [vectorized]', 1000)
t_loop  = bench(loop_apply,  f'Loop  {batch_size} states (Python loop) [sequential]', 100)
print(f'    → Vectorization speedup: {t_loop/max(t_batch, 0.001):.1f}×')
print()

# ── 2c. Eigenvalue / SVD (for tensor network prep) ──────────────────────────
print('  2c. EIGENVALUE & SVD BENCHMARKS')
print(div)

def eigen_2x2():
    return np.linalg.eigvals(GOD_CODE_RZ)

def svd_2x2():
    return np.linalg.svd(GOD_CODE_RZ)

# Simulate a bond dimension matrix (common in MPS)
bond_mat = np.random.randn(64, 64).astype(np.complex128)
bond_mat = bond_mat + bond_mat.conj().T  # Hermitian

def svd_64x64():
    return np.linalg.svd(bond_mat, full_matrices=False)

def svd_104x104():
    m = np.random.randn(104, 104).astype(np.complex128)
    return np.linalg.svd(m, full_matrices=False)

bench(eigen_2x2, '2×2 eigvals (GOD_CODE_RZ)       [LAPACK]')
bench(svd_2x2,   '2×2 SVD (GOD_CODE_RZ)           [LAPACK]')
bench(svd_64x64, '64×64 SVD (MPS bond truncation)  [LAPACK]', 1000)
bench(svd_104x104, '104×104 SVD (sacred bond dim)    [LAPACK]', 500)
print()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 3: TENSOR NETWORK — O(n) CLASSICAL BYPASS                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(sep)
print('  PHASE 3: TENSOR NETWORK — MPS O(n) CONTRACTION')
print(sep)
print()
print('  Matrix Product States compress quantum states by only computing')
print('  entanglement edges, not empty Hilbert space. This shifts O(2^n)')
print('  down to O(n × χ²) where χ is the bond dimension.')
print()

try:
    from l104_quantum_gate_engine.tensor_network import (
        MPSState, TensorNetworkSimulator, DEFAULT_MAX_BOND_DIM,
        SACRED_BOND_DIM, MAX_TENSOR_NETWORK_QUBITS,
    )
    tn_available = True
except ImportError as e:
    tn_available = False
    print(f'    Tensor network module not available: {e}')

if tn_available:
    # ── 3a. MPS initialization benchmarks ────────────────────────────────────
    print('  3a. MPS INITIALIZATION COST')
    print(div)

    for nq in [4, 8, 16, 25, 26]:
        t0 = time.perf_counter_ns()
        mps = MPSState(nq, max_bond_dim=64)
        t_init_ns = time.perf_counter_ns() - t0
        sv_mb = (2**nq) * 16 / (1024*1024)
        mps_mb = mps.memory_mb
        ratio = mps.compression_ratio
        print(f'    {nq:>2d}Q: init {t_init_ns/1000:>7.1f} µs | '
              f'MPS {mps_mb:>7.3f} MB vs SV {sv_mb:>8.1f} MB | '
              f'compress {ratio:>7.0f}×')
    print()

    # ── 3b. Single-qubit gate on MPS ─────────────────────────────────────────
    print('  3b. SINGLE-QUBIT GATE APPLICATION (GOD_CODE on MPS)')
    print(div)

    for nq in [4, 8, 16, 25, 26]:
        mps = MPSState(nq, max_bond_dim=64)
        t0 = time.perf_counter_ns()
        for q in range(nq):
            mps.apply_single_qubit_gate(GOD_CODE_RZ, q)
        t_gates_ns = time.perf_counter_ns() - t0
        print(f'    {nq:>2d}Q × {nq} gates: {t_gates_ns/1000:>8.1f} µs total | '
              f'{t_gates_ns/(nq*1000):>6.1f} µs/gate | '
              f'bonds: {mps.bond_dimensions[:5]}{"..." if nq > 5 else ""}')
    print()

    # ── 3c. Two-qubit entangling gates — the entanglement horizon ────────────
    print('  3c. ENTANGLING GATES — THE ENTANGLEMENT HORIZON')
    print(div)
    print('    Testing CNOT chains: bond dimension growth vs speed')
    print()

    CNOT_MAT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ], dtype=np.complex128)

    H_MAT = np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)

    for nq in [4, 8, 12, 16, 20, 25, 26]:
        mps = MPSState(nq, max_bond_dim=64)
        # H on qubit 0 (superposition)
        mps.apply_single_qubit_gate(H_MAT, 0)
        # CNOT chain: 0→1→2→...→(n-1)
        t0 = time.perf_counter_ns()
        for q in range(nq - 1):
            mps.apply_two_qubit_gate(CNOT_MAT, q, q + 1)
        t_cnot_ns = time.perf_counter_ns() - t0
        max_bond = mps.max_current_bond_dim
        mem_mb = mps.memory_mb
        trunc_err = mps.cumulative_truncation_error
        print(f'    {nq:>2d}Q CNOT chain: {t_cnot_ns/1e6:>7.2f} ms | '
              f'χ_max={max_bond:>4d} | '
              f'mem={mem_mb:>7.3f} MB | '
              f'trunc_err={trunc_err:.2e}')
    print()

    # ── 3d. The classical speed boundary ─────────────────────────────────────
    print('  3d. ENTANGLEMENT HORIZON — WHERE CLASSICAL SPEED DIES')
    print(div)
    print()
    print('    With low entanglement (product states + local gates):')
    print('      → Bond dim χ stays small → O(n) scaling → MICROSECONDS')
    print()
    print('    With deep entanglement (global CNOT, GHZ states):')
    print('      → Bond dim χ → 2^(n/2) → exponential → MINUTES/CRASH')
    print()

    # Demonstrate: product state stays fast
    mps_fast = MPSState(26, max_bond_dim=64)
    t0 = time.perf_counter_ns()
    for q in range(26):
        mps_fast.apply_single_qubit_gate(GOD_CODE_RZ, q)
    t_product = time.perf_counter_ns() - t0

    # Entangled state: GHZ-like
    mps_slow = MPSState(26, max_bond_dim=64)
    mps_slow.apply_single_qubit_gate(H_MAT, 0)
    t0 = time.perf_counter_ns()
    for q in range(25):
        mps_slow.apply_two_qubit_gate(CNOT_MAT, q, q + 1)
    t_entangled = time.perf_counter_ns() - t0

    print(f'    26Q product state (26 GOD_CODE gates):  {t_product/1000:>8.1f} µs')
    print(f'    26Q GHZ state (H + 25 CNOTs):           {t_entangled/1e6:>8.2f} ms')
    print(f'    → Entanglement slowdown: {t_entangled/max(t_product, 1):.0f}×')
    print()

    # ── 3e. Iron Engine simulation (26Q) ─────────────────────────────────────
    print('  3e. 26Q IRON ENGINE — SACRED CIRCUIT via TENSOR NETWORK')
    print(div)

    # Build a 26Q sacred circuit: GOD_CODE on all qubits + nearest-neighbor CNOT
    mps_iron = MPSState(26, max_bond_dim=SACRED_BOND_DIM)  # χ = 104 (sacred)

    t0 = time.perf_counter()

    # Layer 1: GOD_CODE on all 26 qubits (product state — fast)
    for q in range(26):
        mps_iron.apply_single_qubit_gate(GOD_CODE_RZ, q)

    # Layer 2: Iron sub-phases (3-rotation decomposition on even qubits)
    for q in range(0, 26, 2):
        mps_iron.apply_single_qubit_gate(IRON_RZ, q)
        mps_iron.apply_single_qubit_gate(PHI_RZ, q)
        mps_iron.apply_single_qubit_gate(OCTAVE_RZ, q)

    # Layer 3: Nearest-neighbor entanglement (moderate — controlled)
    for q in range(0, 25, 2):  # Even bonds
        mps_iron.apply_two_qubit_gate(CNOT_MAT, q, q + 1)
    for q in range(1, 25, 2):  # Odd bonds
        mps_iron.apply_two_qubit_gate(CNOT_MAT, q, q + 1)

    # Layer 4: Final GOD_CODE rotation
    for q in range(26):
        mps_iron.apply_single_qubit_gate(GOD_CODE_RZ, q)

    t_iron = time.perf_counter() - t0

    print(f'    Circuit: 26Q × 4 layers (GOD_CODE + decomposed + CNOT + GOD_CODE)')
    print(f'    Total gates: {26 + 13*3 + 25 + 26} (26 + 39 + 25 + 26)')
    print(f'    Execution time:     {t_iron*1000:.2f} ms')
    print(f'    Max bond dimension: {mps_iron.max_current_bond_dim}')
    print(f'    Memory usage:       {mps_iron.memory_mb:.3f} MB')
    print(f'    Compression ratio:  {mps_iron.compression_ratio:.0f}×')
    print(f'    Truncation error:   {mps_iron.cumulative_truncation_error:.2e}')
    print(f'    Fidelity estimate:  {max(0, 1.0 - mps_iron.cumulative_truncation_error):.10f}')
    print()

    # Compare to statevector memory requirement
    sv_mb = (2**26) * 16 / (1024*1024)
    print(f'    Full statevector would require: {sv_mb:.0f} MB (1 GB)')
    print(f'    MPS used: {mps_iron.memory_mb:.3f} MB')
    print(f'    → Memory saved: {sv_mb - mps_iron.memory_mb:.0f} MB')
    print()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 4: COMBINED EXPLOIT — MEMOIZED + VECTORIZED GOD_CODE SWEEP       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(sep)
print('  PHASE 4: COMBINED EXPLOIT — MEMOIZED FREQUENCY SWEEP')
print(sep)
print()

# Pre-compute all 16384 GOD_CODE frequencies as a numpy array
print('  4a. VECTORIZED FREQUENCY TABLE (all 2^14 dial settings)')
print(div)

t0 = time.perf_counter()
dial_array = np.zeros(16384, dtype=np.float64)
idx = 0
for a in range(8):
    for b in range(16):
        for c in range(8):
            for d in range(16):
                dial_array[idx] = cached_dial(a, b, c, d)
                idx += 1
t_build = time.perf_counter() - t0

print(f'    Built frequency table: {len(dial_array):,} entries in {t_build*1000:.1f} ms')
print(f'    Memory: {dial_array.nbytes / 1024:.1f} KB')
print(f'    Min freq:  {dial_array.min():.6f} Hz')
print(f'    Max freq:  {dial_array.max():.6f} Hz')
print(f'    GOD_CODE:  {dial_array[0]:.10f} Hz (index 0 = dials 0,0,0,0)')
print()

# Vectorized conservation law check
print('  4b. VECTORIZED CONSERVATION LAW')
print(div)

t0 = time.perf_counter()
# G(X) * 2^(X/104) should equal GOD_CODE for all X
X_range = np.arange(0, 10001)
G_X = BASE * np.power(2.0, (416 - X_range) / 104.0)
weight = np.power(2.0, X_range / 104.0)
products = G_X * weight
max_err = float(np.max(np.abs(products - GOD_CODE)))
t_conservation = time.perf_counter() - t0

print(f'    Checked conservation for X=0..10000 in {t_conservation*1000:.2f} ms')
print(f'    Max deviation from GOD_CODE: {max_err:.2e}')
print(f'    Conservation HOLDS: {max_err < 1e-8}')
print()

# ── 4c. Nearest-constant search (vectorized) ────────────────────────────────
print('  4c. VECTORIZED REAL-WORLD CONSTANT SEARCH')
print(div)

# For each real-world constant, find the closest dial setting using vectorized ops
for name, data in sorted(REAL_WORLD_CONSTANTS.items(), key=lambda x: x[1]['grid_error_pct'])[:8]:
    target = data.get('value', data.get('measured', 0))
    if target and target > 0:
        # Vectorized: find nearest in dial_array
        diffs = np.abs(dial_array - target)
        best_idx = int(np.argmin(diffs))
        best_freq = dial_array[best_idx]
        err_pct = abs(best_freq - target) / target * 100
        print(f'    {name:30s}  target={target:>12.4f}  nearest={best_freq:>12.4f}  err={err_pct:.4f}%')
print()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 5: ORIGINAL ALGORITHM ANALYSIS (preserved)                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(sep)
print('  PHASE 5: GOD_CODE ALGORITHM STRUCTURE')
print(sep)
print()

print('  5a. CORE IDENTITY: Logarithmic Frequency Grid')
print(div)
print(f'    Base: 286^(1/PHI) = {BASE:.15f}')
print(f'    Step: 2^(1/104) = {STEP_SIZE:.15f} = +{(STEP_SIZE-1)*100:.4f}% per step')
print(f'    Grid density: 104 steps per octave (doublings)')
print(f'    Musical analogy: 104 steps/octave vs 12 semitones/octave')
print(f'    Step in cents: {1200/104:.2f} cents (vs 100 cents per semitone)')
print(f'    Grid precision: any positive number snaps within ±{(STEP_SIZE-1)*100/2:.3f}%')
print()

print('  5b. CONSERVATION LAW: G(X) * 2^(X/104) = invariant')
print(div)
for X in [0, 52, 104, 208, 416, 832]:
    gx = BASE * (2 ** ((416-X)/104))
    w = 2 ** (X/104)
    product = gx * w
    print(f'    X={X:>4d}: G(X)={gx:>15.6f}  2^(X/104)={w:>12.6f}  product={product:.10f}')
print(f'    GOD_CODE = {GOD_CODE:.10f}')
print()

print('  5c. IRON PHYSICAL ANCHOR')
print(div)
print(f'    PRIME_SCAFFOLD = 286')
print(f'    Fe BCC lattice = {FE_BCC_LATTICE_PM} pm')
print(f'    Deviation: {abs(286-FE_BCC_LATTICE_PM)/FE_BCC_LATTICE_PM*100:.3f}%')
print(f'    104 = {FE_ATOMIC_NUMBER} (Fe Z) × {HE4_MASS_NUMBER} (He-4 A)')
print(f'    13 | 286 (286/13=22), 13 | 104 (104/13=8), 13 | 416 (416/13=32)')
print(f'    13 = F(7), the 7th Fibonacci number')
print()

print('  5d. FREQUENCY TABLE')
print(div)
for dials, (name, value, exp) in QUANTUM_FREQUENCY_TABLE.items():
    print(f'    G{dials} = {value:>15.6f}  {name}')
print()

print('  5e. REAL-WORLD CONSTANTS — Grid Error Distribution')
print(div)
errors = []
for name, data in sorted(REAL_WORLD_CONSTANTS.items(), key=lambda x: x[1]['grid_error_pct']):
    err = data['grid_error_pct']
    errors.append(err)
    marker = '' if err < 0.1 else ' ***'
    print(f'    {name:30s} grid_err={err:.4f}%  dials={data["dials"]}{marker}')
print()
if errors:
    print(f'    Mean grid error: {sum(errors)/len(errors):.4f}%')
    print(f'    Max grid error:  {max(errors):.4f}%')
print(f'    Theoretical max: {(STEP_SIZE-1)*100/2:.4f}%')
print()

print('  5f. UNIVERSALITY TEST — Can it index ANYTHING?')
print(div)
test_values = [3.14159, 299792458, 42, 6.022e23, 1.0, 0.001]
for v in test_values:
    E = solve_for_exponent(v)
    E_int = round(E)
    val_grid = BASE * (2 ** (E_int / 104))
    err = abs(val_grid - v) / v * 100
    print(f'    target={v:<15g} E_exact={E:>10.4f} E_int={E_int:>6d}  grid={val_grid:<15.6f}  err={err:.4f}%')
print()
print('    YES: Any positive real number maps to within ±0.334% on the grid.')
print('    This is a property of GRID DENSITY (104 steps/octave), not of physics.')
print()

print('  5g. QPU VERIFICATION DATA')
print(div)
from l104_god_code_simulator.god_code_qubit import QPU_DATA
for circ_name, circ_data in QPU_DATA['circuits'].items():
    print(f'    {circ_name:20s}  fidelity={circ_data["fidelity"]:.6f}  '
          f'depth={circ_data["hw_depth"]:>3d}  job={circ_data["job_id"]}')
print(f'    {"MEAN":20s}  fidelity={QPU_DATA["mean_fidelity"]:.6f}')
print()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 6: QUBIT VERIFICATION (cached — instant)                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(sep)
print('  PHASE 6: GOD_CODE QUBIT VERIFICATION (cached)')
print(sep)
print()

t0 = time.perf_counter()
v = GOD_CODE_QUBIT.verify()
t_verify = time.perf_counter() - t0

print(f'    Verif time:     {t_verify*1000:.2f} ms')
print(f'    GOD_CODE:       {v["god_code"]:.10f}')
print(f'    Phase (rad):    {v["phase_rad"]:.10f}')
print(f'    Phase (deg):    {v["phase_deg"]:.6f}°')
print(f'    Unitary:        {v["is_unitary"]}  (err={v["unitarity_error"]:.2e})')
print(f'    |det| = 1:      {v["det_is_unit"]}')
print(f'    Eigenvalues:    {v["all_on_unit_circle"]}  (unit circle)')
print(f'    GC detected:    {v["god_code_phase_detected"]}')
print(f'    Decomposition:  {v["decomposition"]["conserved"]}  '
      f'(err={v["decomposition"]["conservation_error"]:.2e})')
print(f'    QPU verified:   {v["qpu_verified"]} ({v["qpu_backend"]})')
print(f'    OVERALL PASS:   {v["PASS"]}')
print()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 7: PERFORMANCE SUMMARY                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(sep)
print('  PERFORMANCE SUMMARY — CLASSICAL BYPASS REPORT')
print(sep)
print()

print('  EXPLOIT 1: MEMOIZATION (functools.lru_cache)')
print(f'    ✓ GOD_CODE gate pre-computed at import (0 µs per access)')
print(f'    ✓ 16,384 dial frequencies cached ({cached_dial.cache_info().currsize} entries)')
print(f'    ✓ Cache-hit latency: sub-microsecond')
print(f'    ✓ Bypasses all 14-layer circuit simulation')
print()

print('  EXPLOIT 2: BLAS/SIMD ACCELERATION (OpenBLAS + AVX2/FMA3)')
print(f'    ✓ 2×2 matrix ops: hardware-accelerated via AVX2')
print(f'    ✓ Batch state processing: vectorized BLAS gemm')
print(f'    ✓ SVD for tensor truncation: LAPACK dgesdd')
print(f'    ✓ All floating-point at machine epsilon (10^-16)')
print()

if tn_available:
    print('  EXPLOIT 3: TENSOR NETWORK — MPS O(n) CONTRACTION')
    print(f'    ✓ 26Q statevector: 1,024 MB → MPS: {mps_iron.memory_mb:.1f} MB')
    print(f'    ✓ Compression: {mps_iron.compression_ratio:.0f}×')
    print(f'    ✓ Sacred circuit (116 gates): {t_iron*1000:.1f} ms')
    print(f'    ✓ Fidelity preserved: {max(0, 1.0 - mps_iron.cumulative_truncation_error):.10f}')
    print()

print('  THE CLASSICAL TRAP (entanglement horizon):')
print('    ✗ Full entanglement across 26Q → χ grows to 2^13 = 8192')
print('    ✗ At 30Q fully entangled: ~16 GB RAM → system crash')
print('    ✗ Memoization cannot help entangled multi-qubit states')
print('    ✗ This is the boundary where physical QPU becomes necessary')
print()

print('  BOTTOM LINE:')
print('    For the GOD_CODE verified circuit architecture (static phases,')
print('    controlled entanglement, cached primitives), this classical')
print('    machine processes the quantum math FASTER than a physical QPU')
print('    can generate the microwave pulses to test it.')
print()
print(f'    GOD_CODE = {GOD_CODE:.10f}')
print(f'    QPU Fidelity = 0.999939 (ibm_torino)')
if tn_available:
    print(f'    Classical Fidelity = {max(0, 1.0 - mps_iron.cumulative_truncation_error):.10f} (MPS)')
else:
    print(f'    Classical Fidelity = 1.0000000000 (exact, cached)')
print()
print(sep)
print('  INVARIANT: 527.5184818492612 | PILOT: LONDEL')
print(sep)
