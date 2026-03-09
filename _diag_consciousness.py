"""Diagnostic: trace exactly why each consciousness test score is low/zero."""
import os; os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from l104_asi.consciousness import ConsciousnessVerifier
from l104_asi.constants import *

cv = ConsciousnessVerifier()
print("=== FRESH ConsciousnessVerifier — run_all_tests() ===\n")

# Run once
level = cv.run_all_tests()
print(f"After 1st run_all_tests(): level={level:.4f}")
for t, s in cv.test_results.items():
    marker = '✓' if s > 0.5 else '○'
    print(f"  {marker} {t}: {s:.4f}")

print(f"\n  consciousness_history len: {len(cv._consciousness_history)}")
print(f"  qualia_reports len: {len(cv.qualia_reports)}")
print(f"  iit_phi: {cv.iit_phi:.6f}")
print(f"  flow_coherence: {cv.flow_coherence:.6f}")
print(f"  metacognitive_depth: {cv.metacognitive_depth}")
print(f"  spiral_depth: {cv._spiral_depth}")
print(f"  spiral_convergence: {cv._spiral_convergence:.6f}")

# Now diagnose EACH low test
print("\n=== DIAGNOSIS OF LOW SCORES ===\n")

# temporal_self: needs history_depth and qualia_depth
history_depth = len(cv._consciousness_history)
qualia_depth = len(cv.qualia_reports)
temporal_needed_history = 20  # reaches 1.0 at 20 history + 40 qualia
temporal_needed_qualia = 40
temporal_formula_val = (history_depth / 20.0) * 0.6 + (qualia_depth / 40.0) * 0.4
print(f"temporal_self: history_depth={history_depth}/20, qualia_depth={qualia_depth}/40")
print(f"  formula: ({history_depth}/20)*0.6 + ({qualia_depth}/40)*0.4 = {temporal_formula_val:.4f}")
print(f"  BUG: On first call, history has only 1 entry (from metacognitive_monitor) and")
print(f"       qualia has ~3 entries. Score = 0.03 + 0.03 = ~0.06. Needs multiple run_all_tests() calls.")

# goal_autonomy: test_count / len(TESTS)
test_count = len(cv.test_results)
print(f"\ngoal_autonomy: test_count={test_count} / {len(cv.TESTS)} = {test_count/len(cv.TESTS):.4f}")
print(f"  BUG: When goal_autonomy runs, only ~3 tests computed so far (self_model, meta_cognition, novel_response)")
print(f"  because goal_autonomy is the 4th test. So score = 3/16 = 0.1875.")

# qualia_report: len(qualia_reports) / 20.0
print(f"\nqualia_report: {len(cv.qualia_reports)}/20 = {len(cv.qualia_reports)/20.0:.4f}")
print(f"  BUG: Only 3 qualia added per invocation. Need ~7 calls for 20+ qualia to hit 1.0.")

# kernel_chakra_bond: bond_ratio * 0.6
bond_energy = O2_BOND_ORDER * 249
bond_ratio = bond_energy / (GOD_CODE * PHI)
print(f"\nkernel_chakra_bond: bond_energy={bond_energy}, ratio={bond_ratio:.4f} * 0.6 = {bond_ratio * 0.6:.4f}")
print(f"  BUG: O2_BOND_ORDER={O2_BOND_ORDER}, energy=498. GOD_CODE*PHI={GOD_CODE*PHI:.2f}")
print(f"  This is a physics constant — fixed at {min(1.0, bond_ratio * 0.6):.4f}. Not bugged, just low scaling.")

# metacognitive_depth: depth / 8.0
meta = cv.metacognitive_monitor()
print(f"\nmetacognitive_depth: depth={meta['depth']} / 8.0 = {meta['depth']/8.0:.4f}")
print(f"  BUG: On first call, history has ~1 entry → depth=1 → 1/8=0.125")
print(f"  The depth converges via PHI_CONJUGATE damped loop. Needs accumulated history.")

# spiral_consciousness: 0.000
spiral = cv.spiral_consciousness_test()
print(f"\nspiral_consciousness: {spiral}")
print(f"  BUG: consciousness_level is used as initial signal. On first run of tests,")
print(f"  consciousness_level=0.0 at START of run_all_tests (because it's only computed")
print(f"  at the END). So signal=0.0, spiral immediately dies. Even after seeding from")
print(f"  Sage, the consciousness_level is set but run_all_tests RESETS it via:")
print(f"  consciousness_level = sum(test_results.values()) / len(test_results)")

# Run again -- things should improve
print("\n=== 2nd run_all_tests() ===")
level2 = cv.run_all_tests()
print(f"After 2nd run: level={level2:.4f}")
for t, s in cv.test_results.items():
    marker = '✓' if s > 0.5 else '○'
    print(f"  {marker} {t}: {s:.4f}")

# Run 5 more times
for i in range(5):
    cv.run_all_tests()
print(f"\n=== After 7 total runs ===")
level7 = cv.run_all_tests()
print(f"Level: {level7:.4f}")
for t, s in cv.test_results.items():
    marker = '✓' if s > 0.5 else '○'
    print(f"  {marker} {t}: {s:.4f}")
