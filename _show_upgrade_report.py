#!/usr/bin/env python3
"""Display the full GOD CODE Quantum Brain Upgrade report with all metrics."""
import json

with open("god_code_brain_upgrade_report.json") as f:
    data = json.load(f)

print("═" * 80)
print("  GOD CODE QUANTUM BRAIN UPGRADE — FULL METRICS REPORT")
print("═" * 80)
print(f"  Timestamp: {data['timestamp']}")
print(f"  Total: {data['passed']}/{data['total_tests']} ({data['pass_rate']}%) in {data['elapsed_seconds']}s")
print()

results = data["all_results"]

# Map each result to its phase
phase_ranges = []
r_idx = 0
for pn, pd in data["phases"].items():
    count = pd["total"]
    phase_ranges.append((pn, results[r_idx:r_idx + count]))
    r_idx += count

for phase_name, phase_results in phase_ranges:
    pd = data["phases"][phase_name]
    pct = pd["score"] * 100
    print(f"  ┌─ {phase_name}: {pd['passed']}/{pd['total']} ({pct:.0f}%)")
    for r in phase_results:
        icon = "✓" if r["passed"] else "✗"
        detail = r["detail"]
        if len(detail) > 110:
            detail = detail[:110] + "…"
        print(f"  │  {icon} {r['name']}: {detail}")
    print("  └─")
    print()

# Key survivor/upgrade metrics
print("═" * 80)
print("  KEY SURVIVOR RATES & UPGRADE METRICS")
print("═" * 80)
key_names = [
    "cpu_processed", "cpu_energy_valid", "conservation_law_holds",
    "sage_verdict", "god_code_resonance", "fidelity_trend",
    "score_trend", "link_evolution", "asi_18d_score",
    "asi_fe_sacred_score", "asi_phi_lock_score", "asi_berry_phase_score",
    "iit_phi", "consciousness_multiplier", "demon_reversal",
    "wave_coherence_gc_286", "landauer_limit", "cpu_conservation",
    "cpu_void_energy", "cpu_three_engine",
    "auto_upgrade", "full_repair", "self_healer_heal",
    "spectrum_conservation", "post_upgrade_resonance",
    "quantum_zeno", "distillation",
]
for r in results:
    if r["name"] in key_names:
        print(f"    {r['name']}: {r['detail']}")

# Sacred constants
print()
print("═" * 80)
print("  SACRED CONSTANTS & CONSERVATION")
print("═" * 80)
print(f"  GOD_CODE  = {data['god_code']}")
print(f"  PHI       = {data['phi']}")
print(f"  VOID      = {data['void_constant']}")
print(f"  INVARIANT = {data['invariant']}")
print(f"  Conservation: G(X) × 2^(X/104) = {data['invariant']} (∀X)")
print()
print("═" * 80)
print("  FAILURES")
print("═" * 80)
if data["failures"]:
    for f in data["failures"]:
        print(f"    ✗ {f['name']}: {f['detail']}")
else:
    print("    NONE — 100% pass rate")
print("═" * 80)
