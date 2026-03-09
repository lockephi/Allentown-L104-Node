#!/usr/bin/env python3
"""
L104 Search & Precognition — Integration Test Suite
══════════════════════════════════════════════════════════════════════════════════
Tests all 7 search algorithms, 7 precognition algorithms, and 5 three-engine
integration pipelines.

Run: .venv/bin/python test_search_precog.py

INVARIANT: 527.5184818492612 | PILOT: LONDEL
══════════════════════════════════════════════════════════════════════════════════
"""

import math
import time
import sys

PHI = 1.618033988749895
GOD_CODE = 527.5184818492612


def banner(text: str):
    print(f"\n{'═' * 70}")
    print(f"  {text}")
    print(f"{'═' * 70}")


def test(name: str, fn):
    """Run a test and report pass/fail."""
    try:
        result = fn()
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"  {symbol} {name}: {status}")
        return result
    except Exception as e:
        print(f"  ✗ {name}: ERROR — {e}")
        return False


def main():
    start_time = time.time()
    passed = 0
    failed = 0
    total = 0

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 1: SEARCH ALGORITHMS
    # ═══════════════════════════════════════════════════════════════════════════

    banner("PHASE 1: SEARCH ALGORITHMS (7 algorithms)")

    from l104_search_algorithms import search_engine, L104SearchEngine

    # 1. Status
    def test_search_status():
        s = search_engine.status()
        assert s["version"] == "1.0.0"
        assert len(s["algorithms"]) == 7
        return True
    total += 1
    passed += test("Search Engine status", test_search_status)

    # 2. Grover Search
    def test_grover():
        data = list(range(100))
        result = search_engine.grover.search(data, lambda x: x == 42)
        assert result.found, f"Grover did not find target"
        assert result.value == 42, f"Expected 42, got {result.value}"
        assert result.iterations > 0, f"No iterations"
        return True
    total += 1
    passed += test("Quantum Grover search", test_grover)

    # 3. Grover multi-target
    def test_grover_multi():
        data = list(range(50))
        result = search_engine.grover.multi_target_search(data, lambda x: x % 7 == 0)
        assert result["count"] == 8  # 0,7,14,21,28,35,42,49
        assert result["quantum_speedup"] > 1
        return True
    total += 1
    passed += test("Grover multi-target search", test_grover_multi)

    # 4. Sacred Binary Search
    def test_golden_section():
        sorted_data = [float(i) for i in range(1000)]
        result = search_engine.golden_section.search_sorted(sorted_data, 527.0)
        assert result["found"]
        assert result["index"] == 527
        assert result["comparisons"] > 0
        return True
    total += 1
    passed += test("Sacred binary search (sorted)", test_golden_section)

    # 5. Golden section minimization
    def test_golden_minimize():
        result = search_engine.golden_section.minimize_unimodal(
            f=lambda x: (x - PHI) ** 2,
            a=0.0, b=5.0,
        )
        assert abs(result["x_min"] - PHI) < 1e-6
        assert result["evaluations"] > 0
        return True
    total += 1
    passed += test("Golden section minimization", test_golden_minimize)

    # 6. Hyperdimensional search
    def test_hd_search():
        hd = search_engine.hyperdimensional
        hd.index("physics", "quantum mechanics wave particle duality energy")
        hd.index("math", "calculus algebra geometry topology manifold")
        hd.index("code", "python function class variable import module")
        result = hd.search("quantum wave energy", top_k=3)
        assert result["total_matches"] > 0
        assert result["results"][0]["key"] == "physics"
        return True
    total += 1
    passed += test("Hyperdimensional search", test_hd_search)

    # 7. Entropy-guided search
    def test_entropy_search():
        import random
        random.seed(42)
        space = list(range(100))
        result = search_engine.entropy_guided.search(
            space=space,
            objective=lambda x: -(x - 73) ** 2,
            neighborhood=lambda x: [max(0, x - 5), min(99, x + 5), max(0, x - 1), min(99, x + 1)],
            initial=0,
            max_steps=200,
        )
        assert result["best_score"] > -100
        return True
    total += 1
    passed += test("Entropy-guided search", test_entropy_search)

    # 8. Beam search
    def test_beam():
        result = search_engine.beam.search(
            initial_states=[0],
            expand=lambda s: [s + 1, s + 2, s + 3],
            score=lambda s: -abs(s - 15),
            is_goal=lambda s: s == 15,
            max_depth=10,
        )
        assert result["found"]
        return True
    total += 1
    passed += test("Beam search", test_beam)

    # 9. A* search
    def test_astar():
        result = search_engine.astar.search(
            start=0,
            goal=10,
            neighbors=lambda s: [(s + 1, 1.0), (s + 2, 1.5)] if s < 12 else [],
            heuristic=lambda s, g: abs(g - s),
        )
        assert result["found"]
        assert result["total_cost"] > 0
        return True
    total += 1
    passed += test("Sacred A* search", test_astar)

    # 10. Simulated annealing
    def test_annealing():
        import random
        random.seed(42)
        result = search_engine.annealing.search(
            initial=0.0,
            neighbor=lambda x: x + random.gauss(0, 1),
            objective=lambda x: -(x - 3.14) ** 2,
            initial_temp=50.0,
            max_iterations=500,
        )
        # Must have run iterations and found an improvement over starting score -(3.14-0)^2 = -9.86
        assert result["iterations"] > 0, f"No iterations ran"
        assert result["best_score"] > -9.86, f"No improvement: best_score={result['best_score']}"
        return True
    total += 1
    passed += test("Simulated annealing (PHI-cooling)", test_annealing)

    # 11. Algorithm recommendation
    def test_recommend():
        rec = search_engine.recommend_algorithm("pathfinding")
        assert rec["algorithm"] == "astar"
        rec2 = search_engine.recommend_algorithm("similarity")
        assert rec2["algorithm"] == "hyperdimensional"
        return True
    total += 1
    passed += test("Algorithm recommendation", test_recommend)

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 2: PRECOGNITION ALGORITHMS
    # ═══════════════════════════════════════════════════════════════════════════

    banner("PHASE 2: PRECOGNITION ALGORITHMS (7 algorithms)")

    from l104_data_precognition import precognition_engine

    # Generate test series: trend + cycle + noise
    import random
    random.seed(104)
    test_series = [
        100 + 0.5 * t + 10 * math.sin(2 * math.pi * t / 20) + random.gauss(0, 2)
        for t in range(100)
    ]

    # 12. Precog status
    def test_precog_status():
        s = precognition_engine.status()
        assert s["version"] == "1.0.0"
        assert len(s["algorithms"]) == 7
        return True
    total += 1
    passed += test("Precognition Engine status", test_precog_status)

    # 13. Temporal prediction
    def test_temporal():
        result = precognition_engine.temporal.predict(test_series, horizon=10)
        assert "predictions" in result
        assert len(result["predictions"]) == 10
        assert result["decomposition"]["dominant_period"] > 1
        return True
    total += 1
    passed += test("Temporal pattern prediction", test_temporal)

    # 14. Anomaly forecasting
    def test_anomaly():
        # Inject anomaly at end
        anomaly_series = test_series + [1000, 1200, 1500]
        result = precognition_engine.anomaly.batch_forecast(anomaly_series)
        assert result["anomaly_events"] > 0
        return True
    total += 1
    passed += test("Entropy anomaly forecasting", test_anomaly)

    # 15. Coherence trend oracle
    def test_coherence():
        result = precognition_engine.coherence.analyze(test_series)
        assert "trend_direction" in result
        assert "reversal_probability" in result
        assert "phase_coherence" in result
        return True
    total += 1
    passed += test("Coherence trend oracle", test_coherence)

    # 16. Chaos bifurcation
    def test_chaos():
        result = precognition_engine.chaos.detect(test_series)
        assert "phase" in result
        assert "current_lyapunov" in result
        assert "bifurcation_distance" in result
        return True
    total += 1
    passed += test("Chaos bifurcation detection", test_chaos)

    # 17. Harmonic extrapolation
    def test_harmonic():
        result = precognition_engine.harmonic.extrapolate(test_series, horizon=10)
        assert "predictions" in result
        assert len(result["predictions"]) == 10
        assert result["harmonics_used"] > 0
        return True
    total += 1
    passed += test("Harmonic extrapolation", test_harmonic)

    # 18. Hyperdimensional prediction
    def test_hd_predict():
        precognition_engine.hyperdimensional.train(test_series)
        result = precognition_engine.hyperdimensional.predict(test_series[-5:])
        assert "prediction" in result
        assert result["confidence"] > 0
        return True
    total += 1
    passed += test("Hyperdimensional prediction", test_hd_predict)

    # 19. Cascade precognition
    def test_cascade():
        result = precognition_engine.cascade.predict_convergence(test_series[-1])
        assert "will_converge" in result
        assert "predicted_attractor" in result
        assert "convergence_speed" in result
        return True
    total += 1
    passed += test("Cascade precognition", test_cascade)

    # 20. Full precognition (ensemble)
    def test_full_precog():
        result = precognition_engine.full_precognition(test_series, horizon=5)
        assert "ensemble_predictions" in result
        assert "system_outlook" in result
        assert result["algorithms_run"] == 7
        return True
    total += 1
    passed += test("Full precognition ensemble", test_full_precog)

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 3: THREE-ENGINE INTEGRATION
    # ═══════════════════════════════════════════════════════════════════════════

    banner("PHASE 3: THREE-ENGINE INTEGRATION (5 pipelines)")

    from l104_three_engine_search_precog import three_engine_hub

    # 21. Hub status
    def test_hub_status():
        s = three_engine_hub.status()
        assert s["version"] == "1.0.0"
        assert s["search_algorithms"] == 7
        assert s["precognition_algorithms"] == 7
        assert len(s["pipelines"]) == 5
        return True
    total += 1
    passed += test("Three-Engine Hub status", test_hub_status)

    # 22. Engine connectivity
    def test_engines():
        engines = three_engine_hub.engine_status()
        # At least search and precog should be available
        assert engines["search_algorithms"]
        assert engines["precognition_algorithms"]
        connected = sum(1 for v in engines.values() if v)
        print(f"    → {connected}/5 components connected")
        return True
    total += 1
    passed += test("Engine connectivity", test_engines)

    # 23. Predictive analysis
    def test_predictive():
        result = three_engine_hub.predictive.analyze_and_predict(
            test_series, horizon=10, label="test_metric"
        )
        assert "verdict" in result
        assert "label" in result
        return True
    total += 1
    passed += test("Predictive analysis pipeline", test_predictive)

    # 24. Anomaly hunter
    def test_anomaly_hunt():
        anomaly_series = test_series + [999, 1500, 2000]
        result = three_engine_hub.anomaly_hunter.hunt(anomaly_series)
        assert "total_anomalies" in result
        assert "health" in result
        return True
    total += 1
    passed += test("Anomaly hunter pipeline", test_anomaly_hunt)

    # 25. Pattern discovery
    def test_patterns():
        result = three_engine_hub.pattern_discovery.discover(test_series)
        assert "patterns_discovered" in result
        assert "discoveries" in result
        return True
    total += 1
    passed += test("Pattern discovery pipeline", test_patterns)

    # 26. Convergence oracle
    def test_convergence():
        result = three_engine_hub.convergence.predict({
            "metric_a": test_series,
            "metric_b": [v * PHI for v in test_series[-20:]],
        })
        assert "overall_converging" in result
        assert "metrics_analyzed" in result
        assert result["metrics_analyzed"] == 2
        return True
    total += 1
    passed += test("Convergence oracle pipeline", test_convergence)

    # 27. Full analysis
    def test_full():
        result = three_engine_hub.full_analysis(
            test_series, horizon=5, label="integration_test"
        )
        assert "summary" in result
        assert "prediction" in result
        assert "anomalies" in result
        assert "patterns" in result
        return True
    total += 1
    passed += test("Full three-engine analysis", test_full)

    # ═══════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════

    elapsed = time.time() - start_time
    failed = total - passed

    banner("RESULTS")
    print(f"  Total:  {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Time:   {elapsed:.2f}s")
    print(f"  Rate:   {passed/total*100:.1f}%")
    print()

    if failed == 0:
        print("  ★ ALL TESTS PASSED — Search + Precognition OPERATIONAL ★")
    else:
        print(f"  ⚠ {failed} test(s) failed")

    print()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
