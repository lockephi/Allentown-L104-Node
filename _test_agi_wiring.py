#!/usr/bin/env python3
"""
L104 AGI Core Full Engine Wiring Validation — v58.3
Tests all 8 engine bridges, KB feed-back, intellect_think, write-back, and process_thought KB augmentation.
"""
import sys
import json

PASS = 0
FAIL = 0

def test(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} — {detail}")

def main():
    global PASS, FAIL
    print("=" * 70)
    print("  AGI CORE — FULL ENGINE WIRING VALIDATION v58.3")
    print("=" * 70)

    # ─── Phase 1: Import ───
    print("\n▸ Phase 1: AGI core import")
    from l104_agi import agi_core, AGICore
    test("AGI core imported", agi_core is not None)
    test("AGI core is AGICore", isinstance(agi_core, AGICore))

    # ─── Phase 2: Lazy getters (all 8 engines) ───
    print("\n▸ Phase 2: Lazy engine getters")
    se = agi_core._get_science_engine()
    test("ScienceEngine loaded", se is not None)
    me = agi_core._get_math_engine()
    test("MathEngine loaded", me is not None)
    ce = agi_core._get_code_engine()
    test("CodeEngine loaded", ce is not None)
    li = agi_core._get_local_intellect()
    test("LocalIntellect loaded", li is not None)
    qb = agi_core._get_quantum_brain()
    test("QuantumBrain loaded", qb is not None)
    qge = agi_core._get_quantum_gate_engine()
    test("QuantumGateEngine loaded", qge is not None)
    dl = agi_core._get_dual_layer_engine()
    test("DualLayerEngine loaded", dl is not None)
    so = agi_core._get_sage_orchestrator()
    test("SageModeOrchestrator loaded", so is not None)

    # ─── Phase 3: KB feed-back ───
    print("\n▸ Phase 3: KB feed-back")
    agi_core._feed_intellect_kb()
    test("KB feed-back completed", agi_core._intellect_kb_fed)
    # Verify entries exist
    if li is not None:
        agi_entries = [e for e in li.training_data if e.get("source") == "agi_kb_training"]
        test("AGI KB entries injected", len(agi_entries) >= 4, f"found {len(agi_entries)}")
    else:
        test("AGI KB entries injected", False, "LocalIntellect unavailable")

    # ─── Phase 4: intellect_think ───
    print("\n▸ Phase 4: intellect_think (QUOTA_IMMUNE)")
    result = agi_core.intellect_think("What is the AGI core scoring system?")
    test("intellect_think returns response", result.get("response") is not None, str(result.get("error", "")))
    test("intellect_think is quota_immune", result.get("quota_immune") is True)

    # ─── Phase 5: intellect_write_back ───
    print("\n▸ Phase 5: intellect_write_back")
    wb_result = agi_core.intellect_write_back()
    test("write_back succeeded", wb_result.get("entries_written", 0) > 0, str(wb_result))
    test("write_back has total count", wb_result.get("total_training_data", 0) > 0)

    # ─── Phase 6: full_engine_status ───
    print("\n▸ Phase 6: full_engine_status")
    status = agi_core.full_engine_status()
    engines = status.get("engines", {})
    test("status has engines", len(engines) >= 8, f"found {len(engines)}")
    all_wired = all(engines.values())
    test("all engines wired", all_wired, f"missing: {[k for k,v in engines.items() if not v]}")
    test("scoring_dimensions = 19", status.get("scoring_dimensions") == 19)

    # ─── Phase 7: get_status includes wiring ───
    print("\n▸ Phase 7: get_status engine_wiring")
    full_status = agi_core.get_status()
    wiring = full_status.get("engine_wiring", {})
    test("get_status has engine_wiring", len(wiring) > 0, "missing engine_wiring in get_status")
    wired_count = sum(1 for v in wiring.values() if v is True)
    test("wiring shows ≥7 engines", wired_count >= 7, f"only {wired_count} wired")

    # ─── Phase 8: process_thought KB augmentation ───
    print("\n▸ Phase 8: process_thought KB augmentation")
    thought_result = agi_core.process_thought("What engines are wired to AGI core?")
    test("process_thought returns result", thought_result is not None)

    # ─── Phase 9: Three-engine scoring still works ───
    print("\n▸ Phase 9: Three-engine scoring")
    ent_score = agi_core.three_engine_entropy_score()
    test("entropy_score returned", isinstance(ent_score, (int, float)))
    harm_score = agi_core.three_engine_harmonic_score()
    test("harmonic_score returned", isinstance(harm_score, (int, float)))
    wave_score = agi_core.three_engine_wave_coherence_score()
    test("wave_coherence_score returned", isinstance(wave_score, (int, float)))
    te_status = agi_core.three_engine_status()
    test("three_engine_status has data", isinstance(te_status, dict))

    # ─── Phase 10: kernel_status ───
    print("\n▸ Phase 10: kernel_status")
    ks = agi_core.kernel_status()
    test("kernel_status returned", isinstance(ks, dict))
    test("kernel_status has C", "c" in ks or "C" in str(ks))

    # ─── Summary ───
    print("\n" + "=" * 70)
    total = PASS + FAIL
    print(f"  RESULTS: {PASS}/{total} PASSED | {FAIL} FAILED")
    if FAIL == 0:
        print("  ✅ ALL AGI CORE WIRING TESTS PASSED — FULL ENGINE INTEGRATION CONFIRMED")
    else:
        print(f"  ⚠️  {FAIL} test(s) failed — review above")
    print("=" * 70)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
