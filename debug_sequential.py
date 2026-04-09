#!/usr/bin/env python3
"""
L104 Sequential Debug Script v2
================================
Properly tests imports and boots engines sequentially.
"""

import sys
import time

print("=" * 70)
print("L104 SEQUENTIAL DEBUG v2.0")
print("=" * 70)

results = {"passed": [], "failed": []}

def test_import(name: str, import_stmt: str, test_stmt: str = None) -> bool:
    """Test a module import with proper isolation."""
    print(f"\n--- Testing {name} ---")
    start = time.time()
    try:
        module = __import__(import_stmt, fromlist=[''])
        elapsed = time.time() - start

        if test_stmt:
            result = eval(f"module.{test_stmt}")
            print(f"  ✅ {name} OK ({elapsed:.2f}s) - {test_stmt}={result}")
        else:
            print(f"  ✅ {name} OK ({elapsed:.2f}s)")

        results["passed"].append(name)
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"  ❌ {name} FAILED ({elapsed:.2f}s): {e}")
        results["failed"].append((name, str(e)))
        return False

# Test Sacred Algorithms
if test_import("Sacred Algorithms", "l104_sacred_algorithms", "GOD_CODE"):
    import l104_sacred_algorithms as sa
    print(f"     PHI={sa.PHI}")
    print(f"     TAU={sa.TAU}")
    print(f"     OMEGA={sa.OMEGA}")

# Test Math Engine
if test_import("Math Engine", "l104_math_engine", "MathEngine"):
    from l104_math_engine import MathEngine
    me = MathEngine()
    print(f"     GOD_CODE={me.god_code_value()}")

# Test Science Engine
if test_import("Science Engine", "l104_science_engine", "ScienceEngine"):
    from l104_science_engine import ScienceEngine
    se = ScienceEngine()
    print(f"     Initialized")

# Test Code Engine
if test_import("Code Engine", "l104_code_engine", "code_engine"):
    from l104_code_engine import code_engine
    print(f"     Loaded")

# Test AGI Core
if test_import("AGI Core", "l104_agi", "agi_core"):
    from l104_agi import agi_core
    print(f"     AGI ready")

# Test ASI Core (may take longer)
print("\n--- Testing ASI Core ---")
start = time.time()
try:
    from l104_asi import asi_core, ASICore
    elapsed = time.time() - start
    print(f"  ✅ ASI Core OK ({elapsed:.2f}s)")
    results["passed"].append("ASI Core")
except Exception as e:
    elapsed = time.time() - start
    print(f"  ❌ ASI Core FAILED ({elapsed:.2f}s): {e}")
    results["failed"].append(("ASI Core", str(e)))

# Test Local Intellect
if test_import("Local Intellect", "l104_intellect", "local_intellect"):
    from l104_intellect import local_intellect
    print(f"     Intellect ready")

# Test Quantum Engine
if test_import("Quantum Engine", "l104_quantum_engine", "quantum_brain"):
    from l104_quantum_engine import quantum_brain
    print(f"     Quantum brain ready")

# Test Quantum Gate Engine
if test_import("Quantum Gate Engine", "l104_quantum_gate_engine", "get_engine"):
    from l104_quantum_gate_engine import get_engine
    engine = get_engine()
    print(f"     Engine singleton created")

# Test VQPU
if test_import("VQPU", "l104_vqpu", "get_bridge"):
    from l104_vqpu import get_bridge
    bridge = get_bridge()
    print(f"     Bridge ready")

# Test Quantum Networker
if test_import("Quantum Networker", "l104_quantum_networker", "get_networker"):
    from l104_quantum_networker import get_networker
    net = get_networker()
    print(f"     Networker ready")

# Test Algorithmic Constants (NEW)
print("\n--- Testing Algorithmic Constants (EVO_71) ---")
start = time.time()
try:
    from l104_sacred_algorithms import (
        derive_timeout, derive_cache_size, derive_iterations,
        derive_threshold, derive_worker_threads, PHI, GOD_CODE
    )

    # Test the algorithmic functions
    timeout = derive_timeout(priority=5)
    cache = derive_cache_size(tier=1)
    iterations = derive_iterations(1.0)

    elapsed = time.time() - start
    print(f"  ✅ Algorithmic Constants OK ({elapsed:.2f}s)")
    print(f"     derive_timeout(5)={timeout:.2f}s")
    print(f"     derive_cache_size(1)={cache}")
    print(f"     derive_iterations(1.0)={iterations}")
    results["passed"].append("Algorithmic Constants")
except Exception as e:
    elapsed = time.time() - start
    print(f"  ❌ Algorithmic Constants FAILED ({elapsed:.2f}s): {e}")
    results["failed"].append(("Algorithmic Constants", str(e)))

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
total = len(results["passed"]) + len(results["failed"])
passed = len(results["passed"])
failed = len(results["failed"])

print(f"Total:  {total}")
print(f"Passed: {passed} ✅")
print(f"Failed: {failed} ❌")

if results["passed"]:
    print("\n✅ PASS:")
    for name in results["passed"]:
        print(f"   - {name}")

if results["failed"]:
    print("\n❌ FAIL:")
    for name, err in results["failed"]:
        print(f"   - {name}: {err[:80]}")

if failed == 0:
    print("\n🎉 All systems operational!")
    sys.exit(0)
else:
    print(f"\n⚠️ {failed} systems need attention")
    sys.exit(1)
