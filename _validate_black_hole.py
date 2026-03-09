#!/usr/bin/env python3
"""Validate black hole simulation results against known physics."""
import json, math, sys

PHI = (1 + math.sqrt(5)) / 2
PHI_CONJ = PHI - 1
GOD_CODE = 527.5184818492612
LN_GC = math.log(GOD_CODE)
TAU = 2 * math.pi

with open("_black_hole_simulation_data.json") as f:
    data = json.load(f)

sc = data["sacred_constants"]
sims = {s["name"]: s for s in data["simulations"]}
checks = []

# ── Sacred Constants ──
gc_phase = GOD_CODE % TAU
checks.append(("Schwarzschild phase = GC_phase/pi", abs(sc["schwarzschild_phase"] - gc_phase / math.pi) < 1e-8))
checks.append(("Hawking temp = phi^-1 / 4", abs(sc["hawking_temp_param"] - PHI_CONJ / 4) < 1e-10))
checks.append(("BH entropy coeff ~= 2pi (0.24%)", abs(sc["bh_entropy_coeff"] - TAU) / TAU < 0.003))
checks.append(("Penrose eff = 1 - 1/sqrt(2)", abs(sc["penrose_efficiency"] - (1 - 1 / math.sqrt(2))) < 1e-10))
checks.append(("Area quantum = 4*ln(phi)", abs(sc["area_quantum"] - 4 * math.log(PHI)) < 1e-10))
checks.append(("Scrambling rate = 2pi*T_H", abs(sc["scrambling_rate"] - TAU * PHI_CONJ / 4) < 1e-10))

# ── Schwarzschild ──
s = sims["schwarzschild_geometry"]
rt = s["extra"]["redshift_trace"]
checks.append(("Redshift decreasing toward horizon", all(rt[i] >= rt[i + 1] for i in range(len(rt) - 1))))
checks.append(("Tidal entanglement (S>0, purity<1)", s["extra"]["purity"] < 1.0 and s["entanglement_entropy"] > 0))

# ── Hawking ──
s = sims["hawking_radiation"]
checks.append(("Radiation entropy > 0", s["extra"]["radiation_entropy"] > 0))
checks.append(("Pairs = nq/2", s["extra"]["n_pairs"] == s["num_qubits"] // 2))
checks.append(("Purity <= 0.5 (thermal)", s["extra"]["purity"] <= 0.501))

# ── Information Paradox ──
s = sims["information_paradox"]
checks.append(("S_peak > S_final (Page curve)", s["extra"]["peak_entropy"] > s["extra"]["final_entropy"]))
checks.append(("Unitarity preserved", s["extra"]["unitarity_preserved"]))
checks.append(("Page time 0.1 < t < 0.7", 0.1 < s["extra"]["page_time_fraction"] < 0.7))

# ── Penrose ──
s = sims["penrose_process"]
checks.append(("Spin a/M = phi^-1", abs(s["extra"]["spin_parameter"] - PHI_CONJ) < 1e-4))
checks.append(("Penrose eff = Kerr limit", abs(s["extra"]["penrose_efficiency"] - (1 - 1 / math.sqrt(2))) < 1e-4))

# ── Scrambling ──
s = sims["horizon_scrambling"]
checks.append(("Scrambling fraction > 10%", s["extra"]["scrambling_fraction"] > 0.1))
checks.append(("S_BH = nq * ln(GC)", abs(s["extra"]["bh_entropy"] - s["num_qubits"] * LN_GC) < 0.01))

# ── BH Thermodynamics ──
s = sims["bh_thermodynamics"]
checks.append(("2nd law: entropy non-decreasing", s["extra"]["entropy_increasing"]))
al = s["extra"]["area_levels"]
aq = sc["area_quantum"]
checks.append(("Area quantization A_n = n*4ln(phi)", all(abs(a / aq - round(a / aq)) < 0.01 for a in al)))

# ── Probability normalization ──
for sim in data["simulations"]:
    cp = sim.get("coherence_payload", {}).get("probabilities", {})
    if cp:
        total = sum(cp.values())
        checks.append((f"{sim['name']} prob sum=1", abs(total - 1.0) < 1e-8))

# ── Engine payload keys ──
for sim in data["simulations"]:
    asi = sim.get("asi_scoring", {})
    checks.append((f"{sim['name']} ASI payload", all(k in asi for k in ["fidelity", "entanglement_entropy", "sacred_alignment", "conservation_verified"])))
    mv = sim.get("math_verification", {})
    checks.append((f"{sim['name']} Math payload", all(k in mv for k in ["god_code_measured", "phase_coherence", "sacred_alignment"])))

# ── Print ──
print("=" * 74)
print("  L104 BLACK HOLE SIMULATION — VALIDITY CHECK")
print("=" * 74)
for name, ok in checks:
    sym = "\u2713" if ok else "\u2717"
    print(f"  {sym}  {name}")
passed = sum(1 for _, ok in checks if ok)
total = len(checks)
print("=" * 74)
status = "ALL VALID" if passed == total else f"{total - passed} ISSUES"
print(f"  FINAL: {passed}/{total} checks passed — {status}")
print("=" * 74)
sys.exit(0 if passed == total else 1)
