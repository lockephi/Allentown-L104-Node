#!/usr/bin/env python3
"""Quick verification of all benchmark fixes."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

tests = []
def test(name, fn):
    try:
        fn()
        tests.append((name, True))
        print(f"  ✅ {name}")
    except Exception as e:
        tests.append((name, False))
        print(f"  ❌ {name} — {e}")

def t1():
    from l104_asi import dual_layer_engine
    assert dual_layer_engine is not None
test("ASI: dual_layer_engine import", t1)

def t2():
    import importlib
    mod = importlib.import_module("l104_code_engine.constants")
    assert hasattr(mod, "GOD_CODE")
    assert hasattr(mod, "PHI")
    assert hasattr(mod, "VOID_CONSTANT")
test("l104_code_engine.constants (GOD_CODE, PHI, VOID)", t2)

def t3():
    import importlib
    mod = importlib.import_module("l104_asi.dual_layer")
    assert mod is not None
test("l104_asi.dual_layer module import", t3)

def t4():
    from l104_math_engine import MathEngine
    me = MathEngine()
    assert getattr(me, "math_4d", None) is not None
    assert getattr(me, "math_5d", None) is not None
    assert getattr(me, "hyperdimensional", None) is not None
test("Math layers: math_4d, math_5d, hyperdimensional", t4)

def t5():
    from l104_quantum_engine import QuantumLinkBuilder, QuantumMathCore
    qb = QuantumLinkBuilder(QuantumMathCore())
    assert qb is not None
test("QuantumLinkBuilder(QuantumMathCore())", t5)

def t6():
    from l104_reasoning_engine import L104ReasoningCoordinator
    assert L104ReasoningCoordinator is not None
test("l104_reasoning_engine.L104ReasoningCoordinator", t6)

def t7():
    from l104_polymorphic_core import SovereignPolymorph
    assert SovereignPolymorph is not None
test("l104_polymorphic_core.SovereignPolymorph", t7)

def t8():
    from l104_autonomous_executor import AutonomousTaskExecutor
    assert AutonomousTaskExecutor is not None
test("l104_autonomous_executor.AutonomousTaskExecutor", t8)

passed = sum(1 for _, ok in tests if ok)
total = len(tests)
print(f"\n{'=' * 50}")
print(f"  {passed}/{total} PASSED")
if passed == total:
    print("  ALL FIXES VERIFIED ✓")
else:
    print("  SOME FIXES FAILED ✗")
