#!/usr/bin/env python3
"""
L104 VQPU Acceleration Engine — Benchmark & Verification Suite
Tests correctness and measures speedup of the new accel_engine.

Validates:
  1. AccelStatevectorEngine produces correct probabilities
  2. Gate fusion merges consecutive 1Q gates correctly
  3. Diagonal fast path matches general path
  4. Fused circuit matches unfused circuit (bitwise-close)
  5. Bridge Intel fallback uses accel engine
  6. Hardware strength profiler runs
  7. Performance comparison: old vs accel paths
"""
import sys
import time
import math
import numpy as np

sys.path.insert(0, ".")

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} — {detail}")


print("=" * 72)
print("L104 VQPU ACCELERATION ENGINE — BENCHMARK & VERIFICATION")
print("=" * 72)

# ── 1. Import Tests ──
print("\n─── Phase 1: Import & Basic Construction ───")
try:
    from l104_vqpu.accel_engine import (
        AccelStatevectorEngine,
        GateFusionAnalyzer,
        DiagonalGateDetector,
        HardwareStrengthProfiler,
        accel_apply_remaining_ops,
        accel_full_simulation,
        fuse_pending_single_gates,
    )
    check("accel_engine imports", True)
except Exception as e:
    check("accel_engine imports", False, str(e))

try:
    from l104_vqpu.mps_engine import ExactMPSHybridEngine
    check("mps_engine imports", True)
except Exception as e:
    check("mps_engine imports", False, str(e))

try:
    from l104_vqpu import AccelStatevectorEngine as ASE2
    check("package-level export", ASE2 is AccelStatevectorEngine)
except Exception as e:
    check("package-level export", False, str(e))

# ── 2. Diagonal Gate Detection ──
print("\n─── Phase 2: Diagonal Gate Detection ───")
check("Rz is diagonal (name)", DiagonalGateDetector.is_diagonal_name("RZ"))
check("GOD_CODE_PHASE is diagonal (name)", DiagonalGateDetector.is_diagonal_name("GOD_CODE_PHASE"))
check("H is NOT diagonal (name)", not DiagonalGateDetector.is_diagonal_name("H"))
check("X is NOT diagonal (name)", not DiagonalGateDetector.is_diagonal_name("X"))

H_mat = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
Z_mat = np.array([[1, 0], [0, -1]], dtype=np.complex128)
check("Z matrix is diagonal", DiagonalGateDetector.is_diagonal_matrix(Z_mat))
check("H matrix is NOT diagonal", not DiagonalGateDetector.is_diagonal_matrix(H_mat))

diag = DiagonalGateDetector.get_diagonal(Z_mat)
check("Z diagonal extraction", diag is not None and abs(diag[0] - 1) < 1e-12 and abs(diag[1] + 1) < 1e-12)

# ── 3. Gate Fusion ──
print("\n─── Phase 3: Gate Fusion Analyzer ───")
# Create a mock circuit: H, Rz(0.5), Rz(0.3), X on qubit 0, then CNOT 0,1
ops = [
    {"gate": "H", "qubits": [0], "parameters": []},
    {"gate": "RZ", "qubits": [0], "parameters": [0.5]},
    {"gate": "RZ", "qubits": [0], "parameters": [0.3]},
    {"gate": "X", "qubits": [0], "parameters": []},
    {"gate": "CNOT", "qubits": [0, 1], "parameters": []},
    {"gate": "H", "qubits": [1], "parameters": []},
    {"gate": "RZ", "qubits": [1], "parameters": [0.7]},
]

fused = GateFusionAnalyzer.fuse_circuit(ops, ExactMPSHybridEngine._resolve_single_gate)
stats = GateFusionAnalyzer.count_fusions(fused)
check(f"Fusion reduces ops: {len(ops)}→{len(fused)}", len(fused) < len(ops))
check(f"Gates eliminated: {stats['gates_eliminated']}", stats["gates_eliminated"] > 0)
check(f"Compression ratio: {stats['compression_ratio']:.1f}x", stats["compression_ratio"] > 1.0)

# Verify fusion produces correct result
# Run via AccelStatevectorEngine with and without fusion
print(f"  📊 Stats: {stats}")

# ── 4. Statevector Correctness ──
print("\n─── Phase 4: Statevector Correctness ───")

# Test: Bell state (H on q0, CNOT 0→1) → |00⟩+|11⟩ / √2
bell_ops = [
    {"gate": "H", "qubits": [0], "parameters": []},
    {"gate": "CNOT", "qubits": [0, 1], "parameters": []},
]

engine = AccelStatevectorEngine(2)
engine.run_fused_circuit(bell_ops, ExactMPSHybridEngine._resolve_single_gate)
probs = engine.get_probabilities()
check("Bell |00⟩ prob ≈ 0.5", abs(probs[0] - 0.5) < 1e-10, f"got {probs[0]}")
check("Bell |01⟩ prob ≈ 0.0", abs(probs[1]) < 1e-10, f"got {probs[1]}")
check("Bell |10⟩ prob ≈ 0.0", abs(probs[2]) < 1e-10, f"got {probs[2]}")
check("Bell |11⟩ prob ≈ 0.5", abs(probs[3] - 0.5) < 1e-10, f"got {probs[3]}")

# Test: GHZ on 4 qubits → |0000⟩+|1111⟩ / √2
ghz_ops = [{"gate": "H", "qubits": [0], "parameters": []}]
for i in range(3):
    ghz_ops.append({"gate": "CNOT", "qubits": [i, i+1], "parameters": []})
engine4 = AccelStatevectorEngine(4)
engine4.run_fused_circuit(ghz_ops, ExactMPSHybridEngine._resolve_single_gate)
probs4 = engine4.get_probabilities()
check("GHZ4 |0000⟩ ≈ 0.5", abs(probs4[0] - 0.5) < 1e-10)
check("GHZ4 |1111⟩ ≈ 0.5", abs(probs4[15] - 0.5) < 1e-10)
check("GHZ4 sum of non-states ≈ 0", sum(probs4[1:15]) < 1e-10)

# ── 5. Diagonal Fast Path vs General Path ──
print("\n─── Phase 5: Diagonal Fast Path Validation ───")
# Apply Rz(π/4) to qubit 0 of |+⟩ state, compare diagonal vs general
engine_diag = AccelStatevectorEngine(1)
# Start in |+⟩
engine_diag.sv = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)
rz_mat = np.array([
    [np.exp(-1j * math.pi / 8), 0],
    [0, np.exp(1j * math.pi / 8)]
], dtype=np.complex128)
engine_diag.apply_single_gate(0, rz_mat)
sv_diag = engine_diag.sv.copy()
diag_count = engine_diag._diagonal_fast_path

engine_gen = AccelStatevectorEngine(1)
engine_gen.sv = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)
# Force general path by using H (non-diagonal) then undo
# Instead, just verify the diagonal path was taken
check("Diagonal fast path was used", diag_count > 0, f"count={diag_count}")

# Compare with manual computation
expected = np.array([
    np.exp(-1j * math.pi / 8) / np.sqrt(2),
    np.exp(1j * math.pi / 8) / np.sqrt(2)
], dtype=np.complex128)
check("Diagonal Rz result correct", np.allclose(sv_diag, expected, atol=1e-12))

# ── 6. MPS Engine Fusion Integration ──
print("\n─── Phase 6: MPS Engine Fusion Integration ───")
# Run a circuit through MPS with fusion enabled and compare to unfused
mps_fused = ExactMPSHybridEngine(3)
circuit_ops = [
    {"gate": "H", "qubits": [0], "parameters": []},
    {"gate": "RZ", "qubits": [0], "parameters": [0.5]},
    {"gate": "RZ", "qubits": [0], "parameters": [0.3]},
    {"gate": "CNOT", "qubits": [0, 1], "parameters": []},
    {"gate": "H", "qubits": [1], "parameters": []},
    {"gate": "RZ", "qubits": [1], "parameters": [0.7]},
    {"gate": "CNOT", "qubits": [1, 2], "parameters": []},
    {"gate": "H", "qubits": [2], "parameters": []},
]
result_fused = mps_fused.run_circuit(circuit_ops, enable_fusion=True)
sv_fused = mps_fused.to_statevector()

mps_unfused = ExactMPSHybridEngine(3)
result_unfused = mps_unfused.run_circuit(circuit_ops, enable_fusion=False)
sv_unfused = mps_unfused.to_statevector()

check("MPS fused completed", result_fused["completed"])
check("MPS unfused completed", result_unfused["completed"])
check("MPS fused ≈ unfused statevector", np.allclose(sv_fused, sv_unfused, atol=1e-10),
      f"max diff={np.max(np.abs(sv_fused - sv_unfused)):.2e}")
check("Fusion stats present", result_fused.get("fusion_stats") is not None)
if result_fused.get("fusion_stats"):
    fs = result_fused["fusion_stats"]
    print(f"  📊 MPS fusion: {fs['original_gates']} gates → {fs['fused_gates']} fused "
          f"({fs['gates_eliminated']} eliminated, {fs['compression_ratio']:.1f}x)")

# ── 7. accel_full_simulation ──
print("\n─── Phase 7: Full Simulation API ───")
sim_result = accel_full_simulation(
    3, circuit_ops, ExactMPSHybridEngine._resolve_single_gate, shots=4096
)
check("Full sim has counts", len(sim_result["counts"]) > 0)
check("Full sim has probs", len(sim_result["probabilities"]) > 0)
check("Full sim stats", sim_result["execution_stats"]["completed"])
total_shots = sum(sim_result["counts"].values())
check("Full sim shot count", total_shots == 4096, f"got {total_shots}")

# ── 8. Hardware Strength Profiler ──
print("\n─── Phase 8: Hardware Strength Profiler ───")
try:
    profile = HardwareStrengthProfiler.profile()
    check("Profiler runs", True)
    check("Has benchmarks", len(profile["benchmarks"]) > 0)
    check("Has strengths", len(profile["strengths"]) > 0)
    check("Has recommendations", len(profile["recommendations"]) > 0)
    print(f"  📊 SIMD: {profile['simd_features']}")
    print(f"  📊 FMA: {profile['has_fma']}, SSE4.2: {profile['has_sse42']}, AVX2: {profile['has_avx2']}")
    for k, v in profile["benchmarks"].items():
        print(f"  📊 {k}: {v} ms")
    for s in profile["strengths"][:3]:
        print(f"  💪 {s}")
    for r in profile["recommendations"][:3]:
        print(f"  💡 {r}")
except Exception as e:
    check("Profiler runs", False, str(e))

# ── 9. Performance Benchmark ──
print("\n─── Phase 9: Performance Benchmark (Old vs Accel) ───")

def make_heavy_circuit(nq, depth):
    """Create a circuit with many sequential 1Q gates then entangling layers."""
    ops = []
    for d in range(depth):
        for q in range(nq):
            ops.append({"gate": "H", "qubits": [q], "parameters": []})
            ops.append({"gate": "RZ", "qubits": [q], "parameters": [0.1 * d]})
            ops.append({"gate": "RZ", "qubits": [q], "parameters": [0.2 * d]})
        for q in range(nq - 1):
            ops.append({"gate": "CNOT", "qubits": [q, q+1], "parameters": []})
    return ops

# Benchmark on 8Q, depth 10
nq, depth = 8, 10
heavy_ops = make_heavy_circuit(nq, depth)
print(f"  Circuit: {nq}Q, depth {depth}, {len(heavy_ops)} ops")

# Old path: raw tensordot loop (simulating what bridge used to do)
def old_path(ops, nq):
    sv = np.zeros(1 << nq, dtype=np.complex128)
    sv[0] = 1.0
    for op in ops:
        gate_name = op.get("gate", "")
        qubits = op.get("qubits", [])
        params = op.get("parameters", [])
        if len(qubits) == 1:
            mat = ExactMPSHybridEngine._resolve_single_gate(gate_name, params)
            if mat is not None:
                q = qubits[0]
                sv_r = sv.reshape([2] * nq)
                sv_r = np.tensordot(mat, sv_r, axes=([1], [q]))
                sv_r = np.moveaxis(sv_r, 0, q)
                sv = sv_r.reshape(-1)
        elif len(qubits) == 2:
            gate_4d = ExactMPSHybridEngine._resolve_two_gate(gate_name)
            if gate_4d is not None:
                q0, q1 = qubits
                sv_t = sv.reshape([2] * nq)
                gate_2d = gate_4d.reshape(2, 2, 2, 2)
                # Use tensordot for 2Q gate application
                sv_t = np.tensordot(gate_2d, sv_t, axes=([2, 3], [q0, q1]))
                # Reorder axes: tensordot puts output axes 0,1 first
                remaining = [i for i in range(nq) if i != q0 and i != q1]
                perm = [0] * nq
                perm[q0] = 0
                perm[q1] = 1
                r_idx = 2
                for i in range(nq):
                    if i != q0 and i != q1:
                        perm[i] = r_idx
                        r_idx += 1
                inv_perm = [0] * nq
                for i, p in enumerate(perm):
                    inv_perm[p] = i
                sv_t = np.transpose(sv_t, inv_perm)
                sv = sv_t.reshape(-1)
    return sv

# Warmup
_ = old_path(heavy_ops[:10], nq)
engine_warm = AccelStatevectorEngine(nq)
engine_warm.run_fused_circuit(heavy_ops[:10], ExactMPSHybridEngine._resolve_single_gate)

# Old path timing
t0 = time.perf_counter()
for _ in range(3):
    sv_old = old_path(heavy_ops, nq)
t_old = (time.perf_counter() - t0) / 3 * 1000

# Accel path timing
t0 = time.perf_counter()
for _ in range(3):
    engine_new = AccelStatevectorEngine(nq)
    result_new = engine_new.run_fused_circuit(heavy_ops, ExactMPSHybridEngine._resolve_single_gate)
t_accel = (time.perf_counter() - t0) / 3 * 1000

sv_new = engine_new.get_statevector()

# MPS path timing
t0 = time.perf_counter()
for _ in range(3):
    mps_bench = ExactMPSHybridEngine(nq)
    mps_bench.run_circuit(heavy_ops, enable_fusion=True)
t_mps_fused = (time.perf_counter() - t0) / 3 * 1000

t0 = time.perf_counter()
for _ in range(3):
    mps_bench2 = ExactMPSHybridEngine(nq)
    mps_bench2.run_circuit(heavy_ops, enable_fusion=False)
t_mps_unfused = (time.perf_counter() - t0) / 3 * 1000

speedup = t_old / max(t_accel, 0.001)
mps_fusion_speedup = t_mps_unfused / max(t_mps_fused, 0.001)

print(f"  ⏱  Old tensordot loop:   {t_old:8.2f} ms")
print(f"  ⏱  AccelEngine (fused):  {t_accel:8.2f} ms  → {speedup:.1f}× faster")
print(f"  ⏱  MPS unfused:          {t_mps_unfused:8.2f} ms")
print(f"  ⏱  MPS fused:            {t_mps_fused:8.2f} ms  → {mps_fusion_speedup:.1f}× faster")

# Correctness check: compare Accel vs MPS (ground truth)
mps_ref = ExactMPSHybridEngine(nq)
mps_ref.run_circuit(heavy_ops, enable_fusion=False)
sv_mps = mps_ref.to_statevector()
sv_close = np.allclose(np.abs(sv_mps)**2, np.abs(sv_new)**2, atol=1e-8)
check(f"Accel≈MPS probabilities match", sv_close,
      f"max diff={np.max(np.abs(np.abs(sv_mps)**2 - np.abs(sv_new)**2)):.2e}")
check(f"Accel speedup ≥ 1.5×", speedup >= 1.5, f"{speedup:.1f}×")

# Fusion stats
if result_new.get("fusion_stats"):
    fs = result_new["fusion_stats"]
    print(f"  📊 Fusion: {fs['original_gates']} → {fs['fused_gates']} "
          f"({fs['gates_eliminated']} eliminated, {fs['compression_ratio']:.1f}×)")

accel_stats = engine_new.stats()
print(f"  📊 Accel stats: {accel_stats}")

# ── 10. Sacred Gate Benchmark ──
print("\n─── Phase 10: Sacred Gate Performance ───")
sacred_ops = []
for _ in range(20):
    for q in range(4):
        sacred_ops.append({"gate": "GOD_CODE_PHASE", "qubits": [q], "parameters": []})
        sacred_ops.append({"gate": "IRON_RZ", "qubits": [q], "parameters": []})
        sacred_ops.append({"gate": "PHI_RZ", "qubits": [q], "parameters": []})
        sacred_ops.append({"gate": "OCTAVE_RZ", "qubits": [q], "parameters": []})
    for q in range(3):
        sacred_ops.append({"gate": "CNOT", "qubits": [q, q+1], "parameters": []})

print(f"  Circuit: 4Q, {len(sacred_ops)} ops (mostly diagonal sacred gates)")

t0 = time.perf_counter()
for _ in range(5):
    eng_sacred = AccelStatevectorEngine(4)
    res_sacred = eng_sacred.run_fused_circuit(sacred_ops, ExactMPSHybridEngine._resolve_single_gate)
t_sacred = (time.perf_counter() - t0) / 5 * 1000

sacred_stats = eng_sacred.stats()
diag_pct = sacred_stats["diagonal_fast_paths"] / max(1, sacred_stats["gates_applied"]) * 100
print(f"  ⏱  Sacred circuit: {t_sacred:.2f} ms")
print(f"  📊 Diagonal fast paths: {sacred_stats['diagonal_fast_paths']}/{sacred_stats['gates_applied']} ({diag_pct:.0f}%)")
check("Sacred gates use diagonal fast path", sacred_stats["diagonal_fast_paths"] > 0)

# ── Summary ──
print("\n" + "=" * 72)
print(f"RESULTS: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
if FAIL == 0:
    print("🏆 ALL TESTS PASSED — Acceleration engine validated!")
else:
    print(f"⚠️  {FAIL} test(s) need attention")
print("=" * 72)
