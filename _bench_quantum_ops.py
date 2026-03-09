#!/usr/bin/env python3
"""Benchmark CPU-bound quantum operations to find optimization targets."""
import numpy as np
import time
import os

print("=" * 60)
print("  QUANTUM OP BENCHMARKS")
print("=" * 60)

# Test 1: BLAS matmul speed
n = 256
A = np.random.randn(n, n).astype(np.complex128)
B = np.random.randn(n, n).astype(np.complex128)
_ = A @ B  # warmup
t0 = time.time()
for _ in range(50):
    C = A @ B
t1 = time.time()
print(f"  256x256 complex128 matmul: {(t1-t0)/50*1000:.2f} ms/op")

# Test 2: SVD speed
M = np.random.randn(32, 32).astype(np.complex128)
_ = np.linalg.svd(M)
t0 = time.time()
for _ in range(200):
    U, S, Vh = np.linalg.svd(M, full_matrices=False)
t1 = time.time()
print(f"  32x32 complex128 SVD: {(t1-t0)/200*1000:.3f} ms/op")

# Test 3: Matrix exponential
from scipy.linalg import expm
M16 = np.random.randn(16, 16).astype(np.complex128) * 0.1
_ = expm(M16)
t0 = time.time()
for _ in range(500):
    _ = expm(M16)
t1 = time.time()
print(f"  16x16 complex128 expm: {(t1-t0)/500*1000:.3f} ms/op")

# Test 4: Einsum
T1 = np.random.randn(2, 2, 2, 2).astype(np.complex128)
T2 = np.random.randn(32, 2, 2, 32).astype(np.complex128)
_ = np.einsum("pqij,lijr->lpqr", T1, T2)
t0 = time.time()
for _ in range(1000):
    _ = np.einsum("pqij,lijr->lpqr", T1, T2)
t1 = time.time()
print(f"  MPS 2Q gate einsum: {(t1-t0)/1000*1000:.3f} ms/op")

# Test 5: Reshape+matmul gate application (16q)
sv16 = np.random.randn(2**16).astype(np.complex128)
sv16 /= np.linalg.norm(sv16)
U2 = np.random.randn(4, 4).astype(np.complex128)
t0 = time.time()
for _ in range(500):
    sv_r = sv16.reshape(4, -1)
    result = U2 @ sv_r
    sv16 = result.reshape(-1)
t1 = time.time()
print(f"  16q 2Q-gate reshape+matmul: {(t1-t0)/500*1000:.3f} ms/op")

# Test 6: Vectorized probabilities
t0 = time.time()
for _ in range(1000):
    probs = np.abs(sv16) ** 2
t1 = time.time()
print(f"  16q probability extraction: {(t1-t0)/1000*1000:.3f} ms/op")

# Test 7: Batched expm (can we batch multiple at once?)
matrices = [np.random.randn(16, 16).astype(np.complex128) * 0.1 for _ in range(19)]
t0 = time.time()
for _ in range(50):
    results = [expm(m) for m in matrices]
t1 = time.time()
print(f"  19x 16x16 expm (serial): {(t1-t0)/50*1000:.2f} ms/batch")

# Test 8: Precomputed expm reuse (matvec only)
Us = [expm(m) for m in matrices]
sv4 = np.random.randn(16).astype(np.complex128)
sv4 /= np.linalg.norm(sv4)
t0 = time.time()
for _ in range(5000):
    for U in Us:
        sv4 = U @ sv4
t1 = time.time()
print(f"  19x 16x16 matvec (precomputed): {(t1-t0)/5000*1000:.3f} ms/step")

# Test 9: Fused matvec via combined unitary
U_combined = np.eye(16, dtype=np.complex128)
for U in Us:
    U_combined = U @ U_combined
t0 = time.time()
for _ in range(5000):
    sv4 = U_combined @ sv4
t1 = time.time()
print(f"  1x 16x16 fused matvec: {(t1-t0)/5000*1000:.4f} ms/step")

print()
print(f"  OpenBLAS threads: {os.environ.get('OPENBLAS_NUM_THREADS', 'auto')}")
print("=" * 60)
