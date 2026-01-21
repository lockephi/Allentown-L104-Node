#!/usr/bin/env python3
"""L104 Core System Test - Tests all major components"""

import os
import sys
os.chdir('/workspaces/Allentown-L104-Node')
sys.path.insert(0, '/workspaces/Allentown-L104-Node')

passed = 0
failed = 0

def test(name, func):
    global passed, failed
    try:
        result = func()
        if result:
            print(f"✓ {name}")
            passed += 1
        else:
            print(f"✗ {name}: returned False")
            failed += 1
    except Exception as e:
        print(f"✗ {name}: {type(e).__name__}: {str(e)[:80]}")
        failed += 1

if __name__ == "__main__":
    print("=" * 60)
    print("  L104 CORE SYSTEM TEST")
    print("=" * 60)

    # Test 1: Core Math
    print("\n=== Core Math ===")

    def test_hyper_math():
        from l104_hyper_math import HyperMath
        assert HyperMath.GOD_CODE > 0, "GOD_CODE must be positive"
        print(f"    GOD_CODE = {HyperMath.GOD_CODE:.6f}")
        return True

    def test_real_math():
        from l104_real_math import RealMath
        entropy = RealMath.shannon_entropy("test signal")
        print(f"    entropy = {entropy:.4f}")
        return entropy > 0

    def test_manifold_math():
        from l104_manifold_math import ManifoldMath
        print(f"    ANYON_BRAID_RATIO = {ManifoldMath.ANYON_BRAID_RATIO:.6f}")
        return True

    test("HyperMath", test_hyper_math)
    test("RealMath", test_real_math)
    test("ManifoldMath", test_manifold_math)

    # Test 2: Resonance
    print("\n=== Resonance ===")

    def test_resonance():
        from l104_resonance import L104Resonance
        r = L104Resonance()
        freq = r.compute_resonance("test")
        print(f"    frequency = {freq:.4f}")
        return freq > 0

    test("L104Resonance", test_resonance)

    # Test 3: Data Systems
    print("\n=== Data Systems ===")

    def test_algo_db():
        from l104_algorithm_database import AlgorithmDatabase
        db = AlgorithmDatabase()
        count = len(db.algorithms) if hasattr(db, 'algorithms') else 0
        print(f"    algorithms = {count}")
        return True

    def test_memory():
        from l104_memory import L104Memory
        mem = L104Memory()
        mem.store("test_key", "test_value")
        val = mem.recall("test_key")
        print(f"    store/recall = {'OK' if val == 'test_value' else 'FAIL'}")
        return val == "test_value"

    def test_data_matrix():
        from l104_data_matrix import DataMatrix
        dm = DataMatrix()
        print(f"    DataMatrix initialized")
        return True

    test("AlgorithmDatabase", test_algo_db)
    test("L104Memory", test_memory)
    test("DataMatrix", test_data_matrix)

    # Test 4: Core Engines
    print("\n=== Core Engines ===")

    def test_ecosystem_simulator():
        from l104_ecosystem_simulator import EcosystemSimulator
        sim = EcosystemSimulator()
        result = sim.run_multi_agent_simulation("test signal")
        print(f"    consensus = {result.get('consensus_score', 0)}")
        return result.get('status') == 'SUCCESS'

    def test_neural_cascade():
        from l104_neural_cascade import NeuralCascade
        nc = NeuralCascade()
        result = nc.activate("test")
        print(f"    resonance = {result.get('resonance', 0):.4f}")
        return result.get('status') == 'CASCADE_COMPLETE'

    def test_ram_universe():
        from l104_ram_universe import RamUniverse
        ru = RamUniverse()
        check = ru.cross_check_hallucination("test thought")
        print(f"    hallucination_check = {'OK' if not check.get('is_hallucination') else 'WARN'}")
        return True

    test("EcosystemSimulator", test_ecosystem_simulator)
    test("NeuralCascade", test_neural_cascade)
    test("RamUniverse", test_ram_universe)

    # Test 5: Derivation
    print("\n=== Derivation Engine ===")

    def test_derivation():
        from l104_derivation import DerivationEngine
        result = DerivationEngine.derive_and_execute("hi")
        print(f"    response_len = {len(result)}")
        return len(result) > 10

    test("DerivationEngine", test_derivation)

    # Test 6: Gemini Integration
    print("\n=== Gemini Integration ===")

    def test_gemini_real():
        from l104_gemini_real import GeminiReal
        g = GeminiReal()
        connected = g.connect()
        print(f"    connected = {connected}, model = {g.model_name}")
        if connected:
            resp = g.generate("Say YES")
            if resp:
                print(f"    response = {resp.strip()[:30]}...")
                return True
            else:
                print(f"    generation failed (quota?)")
                return False
        return False

    test("GeminiReal", test_gemini_real)

    # Summary
    print("\n" + "=" * 60)
    total = passed + failed
    print(f"  RESULTS: {passed}/{total} passed ({100*passed//total if total else 0}%)")
    if failed == 0:
        print("  STATUS: ALL TESTS PASSED ✓")
    else:
        print(f"  STATUS: {failed} TESTS FAILED ✗")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
