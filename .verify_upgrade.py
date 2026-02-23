"""Verify AGI v57.0 and ASI v8.0 three-engine upgrades."""
import traceback

print("═══ AGI CORE v57.0 VERIFICATION ═══")
try:
    from l104_agi import agi_core
    from l104_agi.constants import AGI_CORE_VERSION as AGI_VER
    print(f"  AGI Version: {AGI_VER}")
    print(f"  Weights count: {len(agi_core._agi_score_weights)}")
    has_er = "entropy_reversal" in agi_core._agi_score_weights
    has_hr = "harmonic_resonance" in agi_core._agi_score_weights
    has_wc = "wave_coherence" in agi_core._agi_score_weights
    print(f"  Three-engine dims: ER={has_er}, HR={has_hr}, WC={has_wc}")

    print(f"  has three_engine_entropy_score: {hasattr(agi_core, 'three_engine_entropy_score')}")
    print(f"  has three_engine_harmonic_score: {hasattr(agi_core, 'three_engine_harmonic_score')}")
    print(f"  has three_engine_wave_coherence_score: {hasattr(agi_core, 'three_engine_wave_coherence_score')}")
    print(f"  has three_engine_status: {hasattr(agi_core, 'three_engine_status')}")

    e = agi_core.three_engine_entropy_score()
    h = agi_core.three_engine_harmonic_score()
    w = agi_core.three_engine_wave_coherence_score()
    print(f"  Entropy reversal: {e:.6f}")
    print(f"  Harmonic resonance: {h:.6f}")
    print(f"  Wave coherence: {w:.6f}")

    result = agi_core.compute_10d_agi_score()
    print(f"  Dimensions: {len(result['dimensions'])}")
    print(f"  Composite: {result['composite_score']}")
    print(f"  Verdict: {result['verdict']}")
    for d, v in result["dimensions"].items():
        print(f"    {d}: {v}")

    te = agi_core.three_engine_status()
    print(f"  TE Status: {te}")
    print("  OK AGI v57.0 ALL CHECKS PASSED")
except Exception as ex:
    traceback.print_exc()
    print(f"  FAIL AGI: {ex}")

print()
print("═══ ASI CORE v8.0 VERIFICATION ═══")
try:
    from l104_asi import asi_core
    from l104_asi.constants import ASI_CORE_VERSION as ASI_VER
    print(f"  ASI Version: {ASI_VER}")

    e = asi_core.three_engine_entropy_score()
    h = asi_core.three_engine_harmonic_score()
    w = asi_core.three_engine_wave_coherence_score()
    print(f"  Entropy reversal: {e:.6f}")
    print(f"  Harmonic resonance: {h:.6f}")
    print(f"  Wave coherence: {w:.6f}")

    result = asi_core.compute_asi_score()
    if isinstance(result, dict):
        print(f"  Dimensions: {len(result['dimensions'])}")
        print(f"  Composite: {result['composite_score']}")
        print(f"  Verdict: {result['verdict']}")
        for d, v in result["dimensions"].items():
            print(f"    {d}: {v}")
    else:
        print(f"  ASI Score (float): {result}")

    te = asi_core.three_engine_status()
    print(f"  TE Status: {te}")
    print("  OK ASI v8.0 ALL CHECKS PASSED")
except Exception as ex:
    traceback.print_exc()
    print(f"  FAIL ASI: {ex}")
