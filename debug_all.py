#!/usr/bin/env python3
"""
L104 Comprehensive Debug — Sequential Engine Boot Test
"""

import sys
import time

print("=" * 70)
print("L104 COMPREHENSIVE DEBUG v3.5")
print("=" * 70)
print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("")

results = {"passed": [], "failed": []}
total_start = time.time()

def test_import(name: str, import_stmt: str, test_attr: str = None) -> bool:
    """Test a module import and optional attribute access."""
    print(f"Testing {name}...", end=" ")
    start = time.time()
    try:
        namespace = {}
        exec(import_stmt, namespace)
        if test_attr and test_attr in namespace:
            val = namespace[test_attr]
            elapsed = time.time() - start
            print(f"✅ ({elapsed:.2f}s) - {test_attr}={val}")
        else:
            elapsed = time.time() - start
            print(f"✅ ({elapsed:.2f}s)")
        results["passed"].append(name)
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"❌ ({elapsed:.2f}s) - {str(e)[:60]}")
        results["failed"].append((name, str(e)))
        return False

# Section 1: Sacred Constants
print("━━━ SACRED FOUNDATION ━━━")
test_import("Sacred Algorithms", "from l104_sacred_algorithms import GOD_CODE, PHI, TAU, OMEGA", "GOD_CODE")

# Section 2: Core Math/Science
print("\n━━━ FOUNDATION ENGINES ━━━")
test_import("Math Engine", "from l104_math_engine import MathEngine", "MathEngine")
test_import("Science Engine", "from l104_science_engine import ScienceEngine", "ScienceEngine")
test_import("Numerical Engine", "from l104_numerical_engine import QuantumNumericalBuilder", "QuantumNumericalBuilder")

# Section 3: Code/Intellect
print("\n━━━ CODE & INTELLECT ━━━")
test_import("Code Engine", "from l104_code_engine import code_engine", "code_engine")
test_import("Local Intellect", "from l104_intellect import local_intellect", "local_intellect")

# Section 4: Quantum Stack
print("\n━━━ QUANTUM STACK ━━━")
test_import("Quantum Engine", "from l104_quantum_engine import quantum_brain", "quantum_brain")
test_import("Quantum Gate Engine", "from l104_quantum_gate_engine import get_engine", "get_engine")
test_import("God Code Simulator", "from l104_god_code_simulator import god_code_simulator", "god_code_simulator")
test_import("Quantum Networker", "from l104_quantum_networker import get_networker", "get_networker")
test_import("VQPU", "from l104_vqpu import get_bridge", "get_bridge")

# Section 5: ASI/AGI
print("\n━━━ ASI/AGI CORE ━━━")
test_import("AGI Core", "from l104_agi import agi_core", "agi_core")

# ASI Core (verbose)
print("Testing ASI Core...", end=" ")
asi_start = time.time()
try:
    from l104_asi import asi_core, ASICore
    asi_elapsed = time.time() - asi_start
    print(f"✅ ({asi_elapsed:.2f}s)")
    results["passed"].append("ASI Core")
except Exception as e:
    asi_elapsed = time.time() - asi_start
    print(f"❌ ({asi_elapsed:.2f}s) - {str(e)[:60]}")
    results["failed"].append(("ASI Core", str(e)))

# Section 6: Server/Infra
print("\n━━━ SERVER & INFRA ━━━")
test_import("Server", "from l104_server import intellect", "intellect")
test_import("ML Engine", "from l104_ml_engine import MLEngine", "MLEngine")

# Section 7: Daemons
print("\n━━━ DAEMONS ━━━")
test_import("Quantum AI Daemon", "from l104_quantum_ai_daemon import QuantumAIDaemon", "QuantumAIDaemon")

# Summary
total_elapsed = time.time() - total_start
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Total Time: {total_elapsed:.2f}s")
print(f"Passed: {len(results['passed'])} ✅")
print(f"Failed: {len(results['failed'])} ❌")

if results["passed"]:
    print("\n✅ OPERATIONAL:")
    for name in results["passed"]:
        print(f"   • {name}")

if results["failed"]:
    print("\n❌ FAILURES:")
    for name, err in results["failed"]:
        print(f"   • {name}: {err[:70]}")

if not results["failed"]:
    print("\n🎉 ALL SYSTEMS OPERATIONAL!")
    sys.exit(0)
else:
    print(f"\n⚠️ {len(results['failed'])} system(s) need attention")
    sys.exit(1)
