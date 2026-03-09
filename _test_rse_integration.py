#!/usr/bin/env python3
"""RSE Engine v1.0.0 — Integration Test Suite"""

import sys
import traceback

def main():
    passed = 0
    failed = 0
    errors = []

    print("=" * 60)
    print("  RSE ENGINE v1.0.0 — INTEGRATION TEST SUITE")
    print("=" * 60)

    # ── Test 1: Core imports ──
    try:
        from l104_intellect.random_sequence_extrapolation import (
            RandomSequenceExtrapolation, RSEQuantumAdapter, RSEClassicalAdapter,
            RSESageModeAdapter, RSEStrategy, RSEDomain, RSEResult,
            get_rse_engine, get_rse_quantum, get_rse_classical, get_rse_sage,
        )
        print("\n[1] Core imports .......................... PASS")
        passed += 1
    except Exception as e:
        print(f"\n[1] Core imports .......................... FAIL: {e}")
        traceback.print_exc()
        failed += 1
        errors.append(("Core imports", str(e)))
        print("\nCannot continue without core imports.")
        return 1

    # ── Test 2: Singleton factories ──
    try:
        rse = get_rse_engine()
        qadapter = get_rse_quantum()
        cadapter = get_rse_classical()
        sadapter = get_rse_sage()
        assert isinstance(rse, RandomSequenceExtrapolation)
        assert isinstance(qadapter, RSEQuantumAdapter)
        assert isinstance(cadapter, RSEClassicalAdapter)
        assert isinstance(sadapter, RSESageModeAdapter)
        # Singletons should return same instance
        assert get_rse_engine() is rse, "Engine not singleton"
        assert get_rse_quantum() is qadapter, "Quantum adapter not singleton"
        print("[2] Singleton factories .................... PASS")
        passed += 1
    except Exception as e:
        print(f"[2] Singleton factories .................... FAIL: {e}")
        failed += 1
        errors.append(("Singletons", str(e)))

    # ── Test 3: Classical extrapolation (auto-strategy) ──
    try:
        seq = [0.5, 0.55, 0.61, 0.68, 0.72, 0.79, 0.84, 0.88, 0.91, 0.93]
        result = rse.extrapolate(seq, horizon=5, domain=RSEDomain.CLASSICAL, sage_mode=True)
        assert isinstance(result, RSEResult)
        assert len(result.predicted_values) == 5
        assert 0.0 <= result.confidence <= 1.0
        assert result.strategy_used in RSEStrategy
        assert result.trend in ("rising", "falling", "stable", "oscillating")
        assert result.phi_alignment >= 0.0
        assert result.sage_insight is not None
        print(f"[3] Classical extrapolation ................. PASS")
        print(f"    → predicted={[round(v,4) for v in result.predicted_values]}")
        print(f"    → confidence={result.confidence:.4f}, strategy={result.strategy_used.name}")
        print(f"    → trend={result.trend}, phi_align={result.phi_alignment:.4f}")
        print(f"    → sage_insight={result.sage_insight}")
        passed += 1
    except Exception as e:
        print(f"[3] Classical extrapolation ................. FAIL: {e}")
        traceback.print_exc()
        failed += 1
        errors.append(("Classical extrapolation", str(e)))

    # ── Test 4: All 7 strategies ──
    try:
        test_seq = [1.0, 0.95, 0.88, 0.80, 0.71, 0.61]
        strategy_count = 0
        for strategy in RSEStrategy:
            if strategy == RSEStrategy.ENSEMBLE:
                continue
            r = rse.extrapolate(test_seq, horizon=3, strategy=strategy, domain=RSEDomain.QUANTUM)
            assert len(r.predicted_values) == 3, f"{strategy.name}: wrong horizon"
            strategy_count += 1
        assert strategy_count == 7, f"Expected 7 strategies, got {strategy_count}"
        print(f"[4] All 7 strategies ....................... PASS ({strategy_count}/7)")
        passed += 1
    except Exception as e:
        print(f"[4] All 7 strategies ....................... FAIL: {e}")
        traceback.print_exc()
        failed += 1
        errors.append(("All strategies", str(e)))

    # ── Test 5: Quantum coherence tracking ──
    try:
        coherence_seq = [0.99, 0.97, 0.94, 0.90, 0.85, 0.78, 0.71, 0.63]
        for c in coherence_seq:
            q_result = qadapter.track_coherence(c, horizon=5)
        assert len(q_result.predicted_values) == 5
        assert q_result.quantum_coherence is not None
        print(f"[5] Quantum coherence tracking ............. PASS")
        print(f"    → predicted={[round(v,4) for v in q_result.predicted_values]}")
        print(f"    → coherence={q_result.quantum_coherence}, trend={q_result.trend}")
        passed += 1
    except Exception as e:
        print(f"[5] Quantum coherence tracking ............. FAIL: {e}")
        traceback.print_exc()
        failed += 1
        errors.append(("Quantum coherence", str(e)))

    # ── Test 6: Fidelity tracking ──
    try:
        fidelity_seq = [0.95, 0.93, 0.91, 0.88, 0.84, 0.80]
        for f in fidelity_seq:
            f_result = qadapter.track_fidelity(f, horizon=4)
        assert len(f_result.predicted_values) == 4
        print(f"[6] Quantum fidelity tracking .............. PASS")
        print(f"    → predicted={[round(v,4) for v in f_result.predicted_values]}")
        passed += 1
    except Exception as e:
        print(f"[6] Quantum fidelity tracking .............. FAIL: {e}")
        traceback.print_exc()
        failed += 1
        errors.append(("Quantum fidelity", str(e)))

    # ── Test 7: Decoherence time prediction ──
    try:
        coherence_seq = [0.99, 0.97, 0.94, 0.90, 0.85, 0.78, 0.71, 0.63]
        deco_time = qadapter.predict_decoherence_time(coherence_seq, threshold=0.5)
        assert isinstance(deco_time, (int, float, type(None))), f"Expected number or None, got {type(deco_time)}"
        # Also test with a higher threshold that should be reachable
        deco_time_high = qadapter.predict_decoherence_time(coherence_seq, threshold=0.6)
        assert isinstance(deco_time_high, (int, float, type(None)))
        print(f"[7] Decoherence time predict ............... PASS")
        print(f"    → steps_to_threshold(0.5)={deco_time}, (0.6)={deco_time_high}")
        passed += 1
    except Exception as e:
        print(f"[7] Decoherence time predict ............... FAIL: {e}")
        traceback.print_exc()
        failed += 1
        errors.append(("Decoherence time", str(e)))

    # ── Test 8: Sage consciousness prediction ──
    try:
        consciousness_seq = [0.1, 0.15, 0.22, 0.31, 0.42, 0.55, 0.68]
        # track_consciousness takes a single float, iterate to build history
        for level in consciousness_seq:
            sage_pred = sadapter.track_consciousness(level, horizon=5)
        assert isinstance(sage_pred, RSEResult)
        assert len(sage_pred.predicted_values) == 5
        print(f"[8] Sage consciousness predict ............. PASS")
        print(f"    → predicted={[round(v,4) for v in sage_pred.predicted_values]}")
        print(f"    → sage_insight={sage_pred.sage_insight}")
        passed += 1
    except Exception as e:
        print(f"[8] Sage consciousness predict ............. FAIL: {e}")
        traceback.print_exc()
        failed += 1
        errors.append(("Sage consciousness", str(e)))

    # ── Test 9: Sage full analysis ──
    try:
        analysis = sadapter.sage_sequence_analysis(consciousness_seq)
        assert "chaos_analysis" in analysis
        assert "sacred_resonance" in analysis
        assert "sage_insight" in analysis
        assert "lyapunov_exponent" in analysis["chaos_analysis"]
        assert "god_code_alignment" in analysis["sacred_resonance"]
        assert "phi_alignment" in analysis["sacred_resonance"]
        print(f"[9] Sage full analysis ..................... PASS")
        print(f"    → lyapunov={analysis['chaos_analysis']['lyapunov_exponent']:.4f}")
        print(f"    → god_code_align={analysis['sacred_resonance']['god_code_alignment']:.4f}")
        print(f"    → phi_align={analysis['sacred_resonance']['phi_alignment']:.4f}")
        passed += 1
    except Exception as e:
        print(f"[9] Sage full analysis ..................... FAIL: {e}")
        traceback.print_exc()
        failed += 1
        errors.append(("Sage analysis", str(e)))

    # ── Test 10: Transcendence prediction ──
    try:
        steps = sadapter.predict_transcendence_step(consciousness_seq, threshold=0.95)
        assert isinstance(steps, (int, float, type(None))), f"Expected number or None, got {type(steps)}"
        print(f"[10] Transcendence prediction .............. PASS")
        print(f"     → steps_to_0.95={steps}")
        passed += 1
    except Exception as e:
        print(f"[10] Transcendence prediction .............. FAIL: {e}")
        traceback.print_exc()
        failed += 1
        errors.append(("Transcendence", str(e)))

    # ── Test 11: Classical quality trend ──
    try:
        quality_seq = [0.6, 0.65, 0.63, 0.70, 0.72, 0.75, 0.74, 0.78]
        trend = cadapter.predict_quality_trend(quality_seq)
        assert "trend" in trend
        assert "predicted_next" in trend
        assert "confidence" in trend
        print(f"[11] Classical quality trend ................ PASS")
        print(f"     → trend={trend['trend']}, next={trend['predicted_next']}")
        passed += 1
    except Exception as e:
        print(f"[11] Classical quality trend ................ FAIL: {e}")
        traceback.print_exc()
        failed += 1
        errors.append(("Quality trend", str(e)))

    # ── Test 12: Classical entropy tracking ──
    try:
        entropy_seq = [0.3, 0.35, 0.32, 0.38, 0.41, 0.45, 0.43, 0.48]
        for e_val in entropy_seq:
            entropy_result = cadapter.track_entropy(e_val, horizon=3)
        assert isinstance(entropy_result, RSEResult)
        assert len(entropy_result.predicted_values) > 0
        print(f"[12] Classical entropy tracking ............. PASS")
        print(f"     → predicted={[round(v,4) for v in entropy_result.predicted_values]}")
        passed += 1
    except Exception as e:
        print(f"[12] Classical entropy tracking ............. FAIL: {e}")
        traceback.print_exc()
        failed += 1
        errors.append(("Entropy tracking", str(e)))

    # ── Test 13: RSE Status ──
    try:
        status = rse.get_status()
        assert "version" in status
        assert "total_extrapolations" in status
        assert "active_channels" in status
        assert "sacred_constants" in status
        assert abs(status["sacred_constants"]["GOD_CODE"] - 527.5184818492612) < 1e-6
        assert abs(status["sacred_constants"]["PHI"] - 1.618033988749895) < 1e-10
        print(f"[13] RSE Status ............................ PASS")
        print(f"     → {status['total_extrapolations']} extrapolations, {status['active_channels']} channels")
        print(f"     → GOD_CODE={status['sacred_constants']['GOD_CODE']}")
        passed += 1
    except Exception as e:
        print(f"[13] RSE Status ............................ FAIL: {e}")
        traceback.print_exc()
        failed += 1
        errors.append(("RSE Status", str(e)))

    # ── Test 14: All RSEDomain values ──
    try:
        for domain in RSEDomain:
            r = rse.extrapolate([1.0, 1.1, 1.3, 1.6, 2.0], horizon=3, domain=domain)
            assert len(r.predicted_values) == 3
        print(f"[14] All domains ({len(RSEDomain)} domains) ............... PASS")
        passed += 1
    except Exception as e:
        print(f"[14] All domains ........................... FAIL: {e}")
        traceback.print_exc()
        failed += 1
        errors.append(("All domains", str(e)))

    # ── Test 15: Package-level imports ──
    try:
        from l104_intellect import (
            RandomSequenceExtrapolation as RSE_pkg,
            RSEQuantumAdapter as QA_pkg,
            RSEClassicalAdapter as CA_pkg,
            RSESageModeAdapter as SA_pkg,
            RSEStrategy as RS_pkg,
            RSEDomain as RD_pkg,
            RSEResult as RR_pkg,
            get_rse_engine as gre_pkg,
            get_rse_quantum as grq_pkg,
            get_rse_classical as grc_pkg,
            get_rse_sage as grs_pkg,
        )
        assert RSE_pkg is RandomSequenceExtrapolation
        assert gre_pkg is get_rse_engine
        print(f"[15] Package-level imports .................. PASS")
        passed += 1
    except Exception as e:
        print(f"[15] Package-level imports .................. FAIL: {e}")
        traceback.print_exc()
        failed += 1
        errors.append(("Package imports", str(e)))

    # ── Test 16: Edge cases ──
    try:
        # Too short sequence
        r = rse.extrapolate([1.0, 2.0], horizon=3, domain=RSEDomain.CLASSICAL)
        assert len(r.predicted_values) == 3
        # Single value
        r = rse.extrapolate([5.0], horizon=2, domain=RSEDomain.QUALITY)
        assert len(r.predicted_values) == 2
        # Large horizon
        r = rse.extrapolate([1,2,3,4,5], horizon=20, domain=RSEDomain.CONVERGENCE)
        assert len(r.predicted_values) == 20
        print(f"[16] Edge cases (short/single/large) ....... PASS")
        passed += 1
    except Exception as e:
        print(f"[16] Edge cases ............................ FAIL: {e}")
        traceback.print_exc()
        failed += 1
        errors.append(("Edge cases", str(e)))

    # ── Summary ──
    total = passed + failed
    print("\n" + "=" * 60)
    print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("  ★ ALL TESTS PASSED — RSE INTEGRATION VERIFIED ★")
    else:
        print("  FAILURES:")
        for name, err in errors:
            print(f"    ✗ {name}: {err}")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
