#!/usr/bin/env python3
"""Validate VQPU Bridge v7.1 — Platform-aware Mac control upgrades."""

import sys
import time
import json

print("=" * 65)
print("  L104 VQPU BRIDGE v7.1 — VALIDATION")
print("=" * 65)

# ── 1. Platform Detection ──
print("\n=== 1. Platform Detection ===")
from l104_vqpu_bridge import (
    _PLATFORM, _IS_INTEL, _IS_APPLE_SILICON, _HAS_METAL_COMPUTE, _GPU_CLASS,
    VQPU_GPU_CROSSOVER, VQPU_MPS_FALLBACK_TARGET, VQPU_MAX_QUBITS,
    VQPU_BATCH_LIMIT, VQPU_MPS_MAX_BOND_HIGH,
    _HW_RAM_GB, _HW_CORES, _HW_PHYS_CORES
)
import os

print(f"  arch:               {_PLATFORM['arch']}")
print(f"  processor:          {_PLATFORM['processor']}")
print(f"  mac_ver:            {_PLATFORM['mac_ver']}")
print(f"  is_intel:           {_IS_INTEL}")
print(f"  is_apple_silicon:   {_IS_APPLE_SILICON}")
print(f"  gpu_class:          {_GPU_CLASS}")
print(f"  metal_family:       {_PLATFORM['metal_family']}")
print(f"  metal_compute:      {_HAS_METAL_COMPUTE}")
print(f"  simd:               {_PLATFORM['simd']}")
print(f"  has_amx:            {_PLATFORM['has_amx']}")
print(f"  has_neural_engine:  {_PLATFORM['has_neural_engine']}")

assert _PLATFORM['arch'] in ('x86_64', 'arm64'), f"Unknown arch: {_PLATFORM['arch']}"
if _IS_INTEL:
    assert not _IS_APPLE_SILICON
    assert _GPU_CLASS == "intel_igpu"
    assert not _HAS_METAL_COMPUTE, "Intel iGPU should NOT be metal_compute_capable"
    assert "AVX2" in _PLATFORM['simd'] or "SSE4.2" in _PLATFORM['simd'], "Intel should have SSE4.2+"
print("  [OK] Platform detection correct")

# ── 2. BLAS Thread Tuning ──
print("\n=== 2. BLAS Thread Tuning ===")
blas_threads = os.environ.get("OPENBLAS_NUM_THREADS", "unset")
mkl_threads = os.environ.get("MKL_NUM_THREADS", "unset")
omp_threads = os.environ.get("OMP_NUM_THREADS", "unset")
print(f"  OPENBLAS_NUM_THREADS: {blas_threads}")
print(f"  MKL_NUM_THREADS:     {mkl_threads}")
print(f"  OMP_NUM_THREADS:     {omp_threads}")
if _IS_INTEL:
    expected = str(max(1, (os.cpu_count() or 4) // 2))
    assert blas_threads == expected, f"Expected {expected}, got {blas_threads}"
    print(f"  [OK] BLAS threads set to {expected} (physical cores)")

# ── 3. Capacity Constants ──
print("\n=== 3. Platform-Aware Capacity ===")
print(f"  HW RAM:             {_HW_RAM_GB} GB")
print(f"  HW Cores:           {_HW_CORES} (phys: {_HW_PHYS_CORES})")
print(f"  Max qubits:         {VQPU_MAX_QUBITS}")
print(f"  Batch limit:        {VQPU_BATCH_LIMIT}")
print(f"  MPS bond high:      {VQPU_MPS_MAX_BOND_HIGH}")
print(f"  GPU crossover:      {VQPU_GPU_CROSSOVER}")
print(f"  MPS fallback:       {VQPU_MPS_FALLBACK_TARGET}")

if _IS_INTEL:
    # Intel: max qubits should be capped lower than Apple Silicon equivalents
    assert VQPU_MAX_QUBITS <= 32, f"Intel should cap at 32Q, got {VQPU_MAX_QUBITS}"
    assert VQPU_GPU_CROSSOVER >= 16, f"Intel CPU/MPS crossover should be >=16, got {VQPU_GPU_CROSSOVER}"
    assert VQPU_MPS_FALLBACK_TARGET == "chunked_cpu", f"Intel fallback should be chunked_cpu, got {VQPU_MPS_FALLBACK_TARGET}"
    print("  [OK] Intel-specific capacity limits applied")

# ── 4. Backend Routing ──
print("\n=== 4. Backend Routing ===")
from l104_vqpu_bridge import CircuitAnalyzer

# Small circuit → cpu_statevector
r1 = CircuitAnalyzer._recommend_backend(4, False, 0.5, 2)
print(f"  4Q, high-ent:               {r1}")
assert r1 == "cpu_statevector", f"Expected cpu_statevector, got {r1}"

# Clifford → stabilizer
r2 = CircuitAnalyzer._recommend_backend(100, True, 0.0, 0)
print(f"  100Q, Clifford:             {r2}")
assert r2 == "stabilizer_chp"

# Above crossover + high entanglement
r3 = CircuitAnalyzer._recommend_backend(20, False, 0.5, 10)
print(f"  20Q, high-ent:              {r3}")
if _IS_INTEL:
    if 20 < VQPU_GPU_CROSSOVER:
        assert r3 == "cpu_statevector", f"20Q under crossover={VQPU_GPU_CROSSOVER} should be cpu_sv"
    else:
        assert r3 != "metal_gpu", f"Intel should NEVER route to metal_gpu, got {r3}"
    print("  [OK] Intel never routes to metal_gpu")

# MPS range (above crossover, medium entanglement)
mps_q = VQPU_GPU_CROSSOVER + 2
r4 = CircuitAnalyzer._recommend_backend(mps_q, False, 0.15, 5)
print(f"  {mps_q}Q, med-ent:              {r4}")
assert r4 == "exact_mps_hybrid", f"Expected exact_mps_hybrid for {mps_q}Q med-ent, got {r4}"

# Low entanglement above crossover → tensor network MPS
r5 = CircuitAnalyzer._recommend_backend(mps_q, False, 0.05, 1)
print(f"  {mps_q}Q, low-ent:              {r5}")
assert r5 == "tensor_network_mps", f"Expected tensor_network_mps, got {r5}"

# High entanglement above crossover, within max → chunked_cpu (Intel) or metal_gpu (Apple)
r6 = CircuitAnalyzer._recommend_backend(mps_q, False, 0.5, 10)
print(f"  {mps_q}Q, high-ent:             {r6}")
if _IS_INTEL:
    assert r6 == "chunked_cpu", f"Intel high-ent should be chunked_cpu, got {r6}"
else:
    assert r6 == "metal_gpu", f"Apple Silicon high-ent should be metal_gpu, got {r6}"

print("  [OK] All routing paths verified")

# ── 5. VQPUBridge Boot + Status ──
print("\n=== 5. VQPUBridge Boot + Status ===")
from l104_vqpu_bridge import VQPUBridge, QuantumJob

bridge = VQPUBridge()
bridge.start()

status = bridge.status()
assert status["version"] == "7.1.0", f"Version should be 7.1.0, got {status['version']}"
assert "platform" in status, "Status should include platform info"
plat = status["platform"]
print(f"  version:            {status['version']}")
print(f"  active:             {status['active']}")
print(f"  platform.arch:      {plat['arch']}")
print(f"  platform.gpu_class: {plat['gpu_class']}")
print(f"  platform.metal_compute: {plat['metal_compute_capable']}")
print(f"  platform.simd:      {plat['simd']}")
print(f"  platform.blas_thr:  {plat['blas_threads']}")
print(f"  platform.gpu_cross: {plat['gpu_crossover']}")
print(f"  platform.mps_fb:    {plat['mps_fallback_target']}")

# Check features include v7.1 entries
features = status.get("process_features", [])
v71_features = [f for f in features if "v7.1" in f]
print(f"  v7.1 features:      {v71_features}")
assert len(v71_features) >= 5, f"Should have 5+ v7.1 features, got {len(v71_features)}"
print("  [OK] VQPUBridge v7.1 status complete")

# ── 6. run_simulation Test ──
print("\n=== 6. run_simulation Pipeline ===")
bell = bridge.bell_pair(shots=512)
sim = bridge.run_simulation(bell, compile=True, error_correct=False)

stages = sim["pipeline"]["stages_executed"]
print(f"  Stages:     {stages}")
print(f"  Total ms:   {sim['pipeline']['total_ms']:.2f}")

if "result" in sim:
    r = sim["result"]
    probs = r.get("probabilities", {})
    backend = r.get("backend", "unknown")
    print(f"  Backend:    {backend}")
    top = sorted(probs.items(), key=lambda x: -x[1])[:4]
    print(f"  Top probs:  {dict(top)}")
    if _IS_INTEL:
        assert "metal_gpu" not in backend, f"Intel should not use metal_gpu backend: {backend}"
        print("  [OK] No Metal GPU in execution path")

if "sacred" in sim:
    print(f"  Sacred:     {sim['sacred'].get('sacred_score', 'N/A')}")

# ── 7. Daemon Health ──
print("\n=== 7. Daemon Health ===")
health = bridge.daemon_health()
print(f"  daemon_running:     {health['daemon_running']}")
print(f"  pid:                {health.get('pid')}")
print(f"  bridge_path_exists: {health['bridge_path_exists']}")
print(f"  inbox_writable:     {health['inbox_writable']}")
print(f"  outbox_readable:    {health['outbox_readable']}")

bridge.stop()

print("\n" + "=" * 65)
print("  VQPU BRIDGE v7.1 — ALL VALIDATIONS PASSED")
print(f"  Platform: {'Intel x86_64' if _IS_INTEL else 'Apple Silicon'}")
print(f"  GPU: {_GPU_CLASS} (Metal compute: {_HAS_METAL_COMPUTE})")
print(f"  SIMD: {', '.join(_PLATFORM['simd'])}")
print(f"  Routing: Metal GPU {'DISABLED' if _IS_INTEL else 'ENABLED'}")
print(f"  Fallback: {VQPU_MPS_FALLBACK_TARGET}")
print(f"  INVARIANT: 527.5184818492612 | PILOT: LONDEL")
print("=" * 65)
