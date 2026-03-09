#!/usr/bin/env python3
"""L104 VQPU v11.0 Turbo Process Upgrade — Validation Script"""
import time, sys

try:
    from l104_vqpu_bridge import (
        VQPUBridge, QuantumJob, CircuitTranspiler, CircuitAnalyzer,
        ExactMPSHybridEngine, HardwareGovernor, CircuitCache,
        VQPU_MAX_QUBITS, VQPU_BATCH_LIMIT, VQPU_PIPELINE_WORKERS,
        VQPU_MPS_MAX_BOND_HIGH, VQPU_ADAPTIVE_SHOTS_MAX,
        _PLATFORM, _IS_APPLE_SILICON, _IS_INTEL, _HW_RAM_GB, _HW_CORES,
    )
except Exception as e:
    print(f"IMPORT ERROR: {e}")
    sys.exit(1)

import numpy as np

PASSED = 0
FAILED = 0

def check(label, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ✓ {label}" + (f"  ({detail})" if detail else ""))
    else:
        FAILED += 1
        print(f"  ✗ {label}" + (f"  ({detail})" if detail else ""))

print("═" * 65)
print("  L104 VQPU v11.0 TURBO PROCESS UPGRADE — VALIDATION")
print("═" * 65)

# ── Platform Info ──
arch = "Apple Silicon" if _IS_APPLE_SILICON else "Intel x86_64"
print(f"\n  Platform: {_PLATFORM['arch']} ({arch})")
print(f"  RAM: {_HW_RAM_GB} GB  |  Cores: {_HW_CORES}")
print()

# ── 1. Capacity Constants ──
print("  [1] Capacity Constants")
check("Max Qubits ≥ 24", VQPU_MAX_QUBITS >= 24, f"{VQPU_MAX_QUBITS}")
check("Batch Limit ≥ 128", VQPU_BATCH_LIMIT >= 128, f"{VQPU_BATCH_LIMIT}")
check("Pipeline Workers ≥ 2", VQPU_PIPELINE_WORKERS >= 2, f"{VQPU_PIPELINE_WORKERS}")
check("Bond HIGH ≥ 1024", VQPU_MPS_MAX_BOND_HIGH >= 1024, f"{VQPU_MPS_MAX_BOND_HIGH}")
check("Adaptive Shots ≥ 65536", VQPU_ADAPTIVE_SHOTS_MAX >= 65536, f"{VQPU_ADAPTIVE_SHOTS_MAX}")
check("MPS Max Chi = 24576", ExactMPSHybridEngine.DEFAULT_MAX_CHI == 24576, f"{ExactMPSHybridEngine.DEFAULT_MAX_CHI}")
check("Param Cache = 16384", ExactMPSHybridEngine._PARAMETRIC_CACHE_MAX == 16384, f"{ExactMPSHybridEngine._PARAMETRIC_CACHE_MAX}")
print()

# ── 2. Transpiler 10-Pass Pipeline ──
print("  [2] Transpiler 10-Pass Pipeline")
bell_ops = [
    {"gate": "H", "qubits": [0]},
    {"gate": "CX", "qubits": [0, 1]},
]
t0 = time.perf_counter_ns()
result = CircuitTranspiler.transpile(bell_ops)
t1 = time.perf_counter_ns()
check("Bell transpile OK", len(result) >= 1, f"{(t1-t0)/1e6:.3f} ms, {len(result)} ops")

# Peephole: H-Z-H → X
peephole_test = [
    {"gate": "H", "qubits": [0]},
    {"gate": "Z", "qubits": [0]},
    {"gate": "H", "qubits": [0]},
]
result_p = CircuitTranspiler._peephole_optimize(peephole_test)
gates_p = [r["gate"] for r in result_p]
check("Peephole H-Z-H → X", gates_p == ["X"], f"{gates_p}")

# Gate fusion: T+T → S
fusion_test = [
    {"gate": "T", "qubits": [0]},
    {"gate": "T", "qubits": [0]},
]
result_f = CircuitTranspiler._gate_fusion(fusion_test)
gates_f = [r["gate"] for r in result_f]
check("Gate fusion T+T → S", gates_f == ["S"], f"{gates_f}")

# Gate fusion: S+S → Z
fusion_test2 = [
    {"gate": "S", "qubits": [0]},
    {"gate": "S", "qubits": [0]},
]
result_f2 = CircuitTranspiler._gate_fusion(fusion_test2)
gates_f2 = [r["gate"] for r in result_f2]
check("Gate fusion S+S → Z", gates_f2 == ["Z"], f"{gates_f2}")

# Identity cancel: H-H → empty
cancel_test = [
    {"gate": "H", "qubits": [0]},
    {"gate": "H", "qubits": [0]},
]
result_c = CircuitTranspiler._peephole_optimize(cancel_test)
check("Peephole H-H → cancel", len(result_c) == 0, f"{[r['gate'] for r in result_c]}")
print()

# ── 3. MPS Engine (matmul fast path) ──
print("  [3] MPS Engine v11.0 (matmul fast path)")
engine = ExactMPSHybridEngine(4)
t0 = time.perf_counter_ns()
engine.run_circuit([
    {"gate": "H", "qubits": [0]},
    {"gate": "CX", "qubits": [0, 1]},
    {"gate": "CX", "qubits": [1, 2]},
    {"gate": "CX", "qubits": [2, 3]},
])
t1 = time.perf_counter_ns()
counts = engine.sample(4096)
check("4Q GHZ circuit OK", len(counts) > 0, f"{(t1-t0)/1e6:.3f} ms, {len(counts)} unique states")

# Verify GHZ state: should be ~50% |0000⟩ and ~50% |1111⟩
ghz_fidelity = (counts.get("0000", 0) + counts.get("1111", 0)) / 4096
check("GHZ fidelity > 0.95", ghz_fidelity > 0.95, f"{ghz_fidelity:.4f}")

# Statevector contraction (matmul path)
t0 = time.perf_counter_ns()
sv = engine.to_statevector()
t1 = time.perf_counter_ns()
check("Statevector dim = 16", len(sv) == 16, f"{(t1-t0)/1e6:.3f} ms, dim={len(sv)}")
check("Statevector normalized", abs(np.sum(np.abs(sv)**2) - 1.0) < 1e-10, f"norm={np.sum(np.abs(sv)**2):.12f}")

# Larger circuit benchmark
engine8 = ExactMPSHybridEngine(8)
ops8 = [{"gate": "H", "qubits": [0]}]
for i in range(7):
    ops8.append({"gate": "CX", "qubits": [i, i+1]})
t0 = time.perf_counter_ns()
engine8.run_circuit(ops8)
t1 = time.perf_counter_ns()
ms8 = (t1-t0) / 1e6
check("8Q GHZ < 500 ms", ms8 < 500, f"{ms8:.2f} ms")
print()

# ── 4. CircuitCache (bloom filter) ──
print("  [4] CircuitCache v11.0 (bloom filter)")
cache = CircuitCache()
fp = CircuitCache.fingerprint(bell_ops, 2, 1024)
cache.put(fp, {"cached": True})
t0 = time.perf_counter_ns()
hit = cache.get(fp)
t1 = time.perf_counter_ns()
check("Cache hit works", hit is not None and hit.get("cached") is True, f"{(t1-t0)/1e6:.4f} ms")

t0 = time.perf_counter_ns()
miss = cache.get("nonexistent_fingerprint_xyz")
t1 = time.perf_counter_ns()
check("Bloom miss fast", miss is None, f"{(t1-t0)/1e6:.4f} ms")
check("Cache max_size = 1024", cache._max_size == 1024, f"{cache._max_size}")
check("Bloom filter exists", hasattr(cache, "_bloom"), "set-based negative lookup")
print()

# ── 5. HardwareGovernor (thermal prediction) ──
print("  [5] HardwareGovernor v11.0 (thermal prediction)")
gov = HardwareGovernor()
check("Poll hot = 0.3s", gov._poll_hot == 0.3, f"{gov._poll_hot}")
check("Poll cool = 2.5s", gov._poll_cool == 2.5, f"{gov._poll_cool}")
check("Samples buffer = 180", gov._samples.maxlen == 180, f"{gov._samples.maxlen}")
check("Has predict flag", hasattr(gov, '_predict_throttle'), "predictive throttle")
print()

# ── 6. Boot Manager v3.0 ──
print("  [6] Boot Manager v3.0")
try:
    import l104_vqpu_boot_manager as bm_mod
    check("BootManager imports OK", True, "v3.0")
    check("Has tcp_probe", hasattr(bm_mod, '_tcp_probe'), "TCP health check")
    check("Has renice_critical", hasattr(bm_mod, 'renice_critical'), "process priority")
    check("Has _check_system_ram", hasattr(bm_mod, '_check_system_ram'), "RAM safety")
except Exception as e:
    check("BootManager import", False, str(e))
print()

# ── Summary ──
total = PASSED + FAILED
print("═" * 65)
if FAILED == 0:
    print(f"  ALL {PASSED}/{total} VALIDATIONS PASSED ✓")
else:
    print(f"  {PASSED}/{total} passed, {FAILED} FAILED ✗")
print("═" * 65)
sys.exit(0 if FAILED == 0 else 1)
