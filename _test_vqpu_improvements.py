#!/usr/bin/env python3
"""Quick validation of VQPU v13.0 improvements."""
import sys, os, time
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np

print("=" * 60)
print("  VQPU v13.0 Improvement Validation")
print("=" * 60)

PASS = 0
FAIL = 0

def ok(msg):
    global PASS; PASS += 1; print(f"  ✓ {msg}")

def fail(msg):
    global FAIL; FAIL += 1; print(f"  ✗ {msg}")


# ── 1. Package import (not monolith) ──
print("\n[1] Package Import")
try:
    from l104_vqpu import CircuitAnalyzer, ExactMPSHybridEngine, ScoringCache
    from l104_vqpu import DAEMON_MAX_ERROR_LOG, DAEMON_ERROR_THRESHOLD
    from l104_vqpu import BrainIntegration
    ok("l104_vqpu package imported (decomposed, not monolith)")
except Exception as e:
    fail(f"Import failed: {e}")
    sys.exit(1)


# ── 2. CircuitAnalyzer field aliases ──
print("\n[2] CircuitAnalyzer Field Aliases")
analysis = CircuitAnalyzer.analyze(
    [{"gate": "H", "qubits": [0]}, {"gate": "CX", "qubits": [0, 1]}], 2
)
if analysis.get("depth") == 2:
    ok(f"'depth' alias present = {analysis['depth']}")
else:
    fail(f"'depth' alias missing or wrong: {analysis.get('depth')}")

if analysis.get("two_qubit_gates") == 1:
    ok(f"'two_qubit_gates' alias present = {analysis['two_qubit_gates']}")
else:
    fail(f"'two_qubit_gates' alias missing or wrong: {analysis.get('two_qubit_gates')}")

# originals still work
if analysis.get("circuit_depth_est") == 2:
    ok("'circuit_depth_est' still present")
else:
    fail(f"'circuit_depth_est' missing: {analysis.get('circuit_depth_est')}")


# ── 3. MPS Rank-1 Fast Path ──
print("\n[3] MPS Rank-1 Fast Path")
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
CX = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=np.complex128).reshape(2,2,2,2)
H_gate = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)

# CX|10⟩ → |11⟩ (product → product, rank-1)
mps = ExactMPSHybridEngine(2)
mps.apply_single_gate(0, X)
mps._apply_adjacent_two_gate(0, 1, CX)
bond = mps.tensors[0].shape[2]
probs = mps.sample(4096)
total_shots_1 = sum(probs.values())
p11 = probs.get("11", 0) / total_shots_1 if total_shots_1 else 0

if bond == 1:
    ok(f"Rank-1: bond dim = 1 (SVD skipped)")
else:
    fail(f"Rank-1: bond dim = {bond} (expected 1)")

if p11 > 0.95:
    ok(f"CX|10⟩ = |11⟩ correct (p11={p11:.4f})")
else:
    fail(f"CX|10⟩ wrong: {probs}")

# CX|+0⟩ → Bell (product → entangled, rank-2)
mps2 = ExactMPSHybridEngine(2)
mps2.apply_single_gate(0, H_gate)
mps2._apply_adjacent_two_gate(0, 1, CX)
bond2 = mps2.tensors[0].shape[2]
probs2 = mps2.sample(4096)
p00 = probs2.get("00", 0)
p11b = probs2.get("11", 0)

if bond2 == 2:
    ok(f"Rank-2 (Bell): bond dim = 2 (SVD used)")
else:
    fail(f"Rank-2 (Bell): bond dim = {bond2} (expected 2)")

# sample() returns counts, normalize to probabilities
total_shots = sum(probs2.values())
p00 = probs2.get("00", 0) / total_shots if total_shots else 0
p11b = probs2.get("11", 0) / total_shots if total_shots else 0

if abs(p00 - 0.5) < 0.05 and abs(p11b - 0.5) < 0.05:
    ok(f"Bell state correct (p00={p00:.3f}, p11={p11b:.3f})")
else:
    fail(f"Bell state wrong: {probs2}")

# Speed benchmark
N = 10000
t0 = time.monotonic()
for _ in range(N):
    m = ExactMPSHybridEngine(2)
    m.apply_single_gate(0, X)
    m._apply_adjacent_two_gate(0, 1, CX)
ms = (time.monotonic() - t0) * 1000
print(f"  · Rank-1 fast path: {N} ops in {ms:.1f}ms ({ms/N:.3f}ms/op)")


# ── 4. ScoringCache LRU Eviction ──
print("\n[4] ScoringCache LRU Eviction")
from collections import OrderedDict

ScoringCache._asi_cache = OrderedDict()
old_max = ScoringCache._ASI_AGI_MAX
ScoringCache._ASI_AGI_MAX = 3

for i in range(3):
    ScoringCache.get_asi_score({}, num_qubits=i, entropy_bucket=0.5,
                                scorer_fn=lambda p, nq: {"score": nq})

if len(ScoringCache._asi_cache) == 3:
    ok("Cache filled to capacity (3)")
else:
    fail(f"Expected 3, got {len(ScoringCache._asi_cache)}")

# Overflow → evict oldest
ScoringCache.get_asi_score({}, num_qubits=99, entropy_bucket=0.5,
                            scorer_fn=lambda p, nq: {"score": nq})

if len(ScoringCache._asi_cache) == 3:
    ok("LRU eviction: size stayed at 3")
else:
    fail(f"Expected 3 after eviction, got {len(ScoringCache._asi_cache)}")

if (0, 0.5) not in ScoringCache._asi_cache:
    ok("Oldest entry (nq=0) correctly evicted")
else:
    fail("Oldest entry NOT evicted")

if (99, 0.5) in ScoringCache._asi_cache:
    ok("Newest entry (nq=99) present")
else:
    fail("Newest entry missing")

# LRU refresh: access nq=1, then add nq=100 → nq=2 should be evicted (not nq=1)
ScoringCache.get_asi_score({}, num_qubits=1, entropy_bucket=0.5,
                            scorer_fn=lambda p, nq: {"score": nq})  # refresh
ScoringCache.get_asi_score({}, num_qubits=100, entropy_bucket=0.5,
                            scorer_fn=lambda p, nq: {"score": nq})

if (2, 0.5) not in ScoringCache._asi_cache and (1, 0.5) in ScoringCache._asi_cache:
    ok("LRU refresh works: nq=2 evicted, nq=1 kept (was refreshed)")
else:
    fail(f"LRU refresh issue: keys={list(ScoringCache._asi_cache.keys())}")

ScoringCache._ASI_AGI_MAX = old_max
ScoringCache._asi_cache = OrderedDict()


# ── 5. Version strings ──
print("\n[5] Version Consistency")
from l104_vqpu import VERSION
from l104_vqpu.bridge import VQPUBridge
bridge = VQPUBridge.__new__(VQPUBridge)
# Check VERSION constant
if VERSION == "13.0.0":
    ok(f"l104_vqpu VERSION = {VERSION}")
else:
    fail(f"VERSION = {VERSION}, expected 13.0.0")


# ── Summary ──
print()
print("=" * 60)
total = PASS + FAIL
print(f"  Results: {PASS}/{total} passed, {FAIL} failed")
if FAIL == 0:
    print("  ALL IMPROVEMENTS VALIDATED ✓")
else:
    print(f"  {FAIL} FAILURE(S) — review above")
print("=" * 60)
sys.exit(1 if FAIL > 0 else 0)
