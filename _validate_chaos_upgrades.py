#!/usr/bin/env python3
"""
Validate all 6 chaos-conservation upgrades applied to the L104 system.
═══════════════════════════════════════════════════════════════════════
"""
import math, time

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}{f' — {detail}' if detail else ''}")
    else:
        FAIL += 1
        print(f"  ❌ {name}{f' — {detail}' if detail else ''}")

print("=" * 70)
print("CHAOS-CONSERVATION UPGRADE VALIDATION (6 upgrades, 7 systems)")
print("=" * 70)

# ═══ 1. Fix 104-cascade damped sine ═══
print("\n▸ UPGRADE 1: Damped sine cascade (eliminates 0.133 residual)")
from l104_math_engine.god_code import ChaosResilience
INVARIANT = 527.5184818492612

# Original: heal_cascade_104 with damped sine should converge closer
healed = ChaosResilience.heal_cascade_104(550.0)
residual = abs(healed - INVARIANT)
check("Cascade heals 550→INVARIANT", residual < 0.01, f"residual={residual:.10f}")
check("Residual < old 0.133", residual < 0.133, f"was 0.133, now {residual:.10f}")

healed2 = ChaosResilience.heal_cascade_104(500.0)
residual2 = abs(healed2 - INVARIANT)
check("Cascade heals 500→INVARIANT", residual2 < 0.01, f"residual={residual2:.10f}")

# ═══ 2. Chaos diagnostics on entropy ═══
print("\n▸ UPGRADE 2: Chaos diagnostics (Shannon + Lyapunov + bifurcation)")
from l104_science_engine import ScienceEngine
se = ScienceEngine()

# Stable signal
import random
stable = [527.5 + random.gauss(0, 0.01) for _ in range(100)]
diag = se.entropy.chaos_diagnostics(stable)
check("Diagnostics returns health", "health" in diag, f"health={diag.get('health')}")
check("Stable signal → is_stable", diag.get("is_stable", False), f"lyapunov={diag.get('lyapunov_exponent')}")
check("Has Shannon entropy", "shannon_entropy" in diag, f"H={diag.get('shannon_entropy', 0):.4f}")
check("Has bifurcation_distance", "bifurcation_distance" in diag, f"d={diag.get('bifurcation_distance', 0):.4f}")

# Chaotic signal
chaotic = [random.uniform(0, 1000) for _ in range(100)]
diag_c = se.entropy.chaos_diagnostics(chaotic)
check("Chaotic signal detected", diag_c.get("health") in ("WARNING", "CRITICAL"),
      f"health={diag_c.get('health')}")

# ═══ 3. verify_conservation_statistical ═══
print("\n▸ UPGRADE 3: Statistical conservation verification")
from l104_math_engine.constants import verify_conservation_statistical

result = verify_conservation_statistical(100.0, chaos_amplitude=0.05, samples=200)
check("Returns dict with mean_product", "mean_product" in result, f"mean={result.get('mean_product', 0):.6f}")
check("Statistically conserved", result.get("statistically_conserved", False),
      f"error={result.get('mean_error_pct', 99):.4f}%")
check("Below bifurcation", result.get("below_bifurcation", False))
check("Has robust flag", "robust" in result, f"robust={result.get('robust')}")

# Also check GodCodeEquation facade
from l104_math_engine.god_code import GodCodeEquation
result2 = GodCodeEquation.verify_conservation_statistical(200.0)
check("GodCodeEquation.verify_conservation_statistical()", result2.get("statistically_conserved", False))

# ═══ 4. GateCircuit chaos_resilience ═══
print("\n▸ UPGRADE 4: Quantum gate circuit chaos resilience")
from l104_quantum_gate_engine import get_engine
engine = get_engine()

# Bell pair — simple, should be highly resilient
bell = engine.bell_pair()
cr = bell.chaos_resilience(noise_amplitude=0.01, samples=30)
check("Bell pair resilience returns", "mean_fidelity" in cr, f"fidelity={cr.get('mean_fidelity', 0):.6f}")
check("Bell pair is COHERENT/RESILIENT", cr.get("health") in ("COHERENT", "RESILIENT"),
      f"health={cr.get('health')}")
check("Has Lyapunov estimate", "lyapunov_estimate" in cr, f"λ={cr.get('lyapunov_estimate', 0):.4f}")
check("Below bifurcation", cr.get("below_bifurcation", False),
      f"effective_noise={cr.get('effective_noise', 0):.4f}")

# GHZ state — deeper, more fragile
ghz = engine.ghz_state(4)
cr2 = ghz.chaos_resilience(noise_amplitude=0.05, samples=20)
check("GHZ4 resilience returns", "mean_fidelity" in cr2, f"fidelity={cr2.get('mean_fidelity', 0):.6f}")
check("GHZ4 health reported", cr2.get("health") in ("COHERENT", "RESILIENT", "DEGRADED", "DECOHERENT"),
      f"health={cr2.get('health')}")

# ═══ 5. Dual Layer chaos bridge ═══
print("\n▸ UPGRADE 5: Dual Layer chaos bridge (third face of duality)")
from l104_asi.dual_layer import DualLayerEngine
dl = DualLayerEngine()

bridge = dl.chaos_bridge(0, 0, 0, 0, chaos_amplitude=0.05, samples=500)
check("Chaos bridge returns", "duality_coherence" in bridge,
      f"coherence={bridge.get('duality_coherence', 0):.4f}")
check("φ symmetry intact", bridge.get("phi_intact", False))
check("Conservation held", bridge.get("thought_conserved", False))
check("Healing trinity applied", "cascade_residual" in bridge,
      f"cascade_residual={bridge.get('cascade_residual', 0):.10f}")
check("Health reported", bridge.get("health") in ("COHERENT", "RESILIENT", "STRESSED", "BIFURCATED"),
      f"health={bridge.get('health')}")
check("Demon beats φ-damping", bridge.get("demon_beats_phi", False))
check("Symmetry hierarchy present", bridge.get("symmetry_hierarchy") == ["phi_phase", "octave_scale", "translation"])

# ═══ 6. AGI self_heal chaos wiring ═══
print("\n▸ UPGRADE 6: AGI self_heal + recover_subsystem chaos wiring")
# We verify the methods exist and have the chaos-related code
import inspect
from l104_agi.core import AGICore

self_heal_src = inspect.getsource(AGICore.self_heal)
check("self_heal has ChaosResilience import",
      "ChaosResilience" in self_heal_src)
check("self_heal has chaos_resilience_score call",
      "chaos_resilience_score" in self_heal_src)
check("self_heal has heal_cascade_104",
      "heal_cascade_104" in self_heal_src)

recover_src = inspect.getsource(AGICore.recover_subsystem)
check("recover_subsystem has chaos_diagnostics",
      "chaos_diagnostics" in recover_src)
check("recover_subsystem has CRITICAL check",
      "CRITICAL" in recover_src)

# ═══ 7. Entropy cascade damped mode ═══
print("\n▸ UPGRADE 7: Entropy cascade damped mode (v4.3)")
result_damped = se.entropy.entropy_cascade(1.0, depth=104, damped=True)
result_old = se.entropy.entropy_cascade(1.0, depth=104, damped=False)
check("Damped mode returns", "damped" in result_damped, f"damped={result_damped.get('damped')}")
check("Damped converges, undamped oscillates",
      result_damped.get("converged") and not result_old.get("converged"),
      f"damped_conv={result_damped.get('converged')}, old_conv={result_old.get('converged')}")
# Damped should converge to ~0 (all terms decay); undamped to non-zero fixed point
damped_fp = result_damped.get("fixed_point", 999)
old_fp = result_old.get("fixed_point", 999)
check("Damped fixed_point closer to 0",
      abs(damped_fp) < abs(old_fp),
      f"damped={damped_fp}, old={old_fp}")

# ═══ SUMMARY ═══
print("\n" + "=" * 70)
total = PASS + FAIL
print(f"RESULTS: {PASS}/{total} passed, {FAIL} failed")
if FAIL == 0:
    print("🏆 ALL CHAOS-CONSERVATION UPGRADES VALIDATED SUCCESSFULLY")
else:
    print(f"⚠️  {FAIL} checks need attention")
print("=" * 70)
