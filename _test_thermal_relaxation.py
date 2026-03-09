"""
═══════════════════════════════════════════════════════════════════════════════
L104 THERMAL_RELAXATION VERIFICATION SUITE v2.0
═══════════════════════════════════════════════════════════════════════════════

Validates the composed T₁+T₂ Kraus decoherence model:
  • Kraus completeness (Σ K†K = I) across all gamma values
  • Realistic QPU probability scores on Bell, GHZ-3, GHZ-5
  • Ground-state bias scaling (more qubits → more bias)
  • Phase damping population-preservation physics
  • Model comparison across all 6 DecoherenceModel variants
  • Deterministic vs stochastic mode comparison
  • Unitarity (probability sum = 1.0) verification

Uses l104_qiskit_utils.aer_backend for ideal-state reference.

INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""
from l104_qiskit_utils import aer_backend
from l104_quantum_gate_engine import get_engine
from l104_quantum_gate_engine.trajectory import (
    TrajectorySimulator, DecoherenceModel, DecoherenceChannel,
)
import numpy as np

engine = get_engine()
sim = TrajectorySimulator(seed=104)

PASS = 0
FAIL = 0

def chk(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} — {detail}")

print()
print("╔" + "═" * 65 + "╗")
print("║" + "  L104 THERMAL_RELAXATION VERIFICATION SUITE v2.0".center(65) + "║")
print("╚" + "═" * 65 + "╝")
print()

# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 1: Kraus Completeness
# ═══════════════════════════════════════════════════════════════════════════════

print("┌" + "─" * 65 + "┐")
print("│  PHASE 1 — Kraus Operator Completeness (Σ K†K = I)".ljust(65) + " │")
print("└" + "─" * 65 + "┘")

for g in [0.0, 0.001, 0.01, 0.02, 0.05, 0.1, 0.3, 0.5, 1.0]:
    ops = DecoherenceChannel.thermal_relaxation(g)
    ok = DecoherenceChannel.verify_completeness(ops)
    chk(f"thermal_relaxation(γ={g:.3f}): Σ K†K = I", ok)

# Vary t2_factor
for t2f in [0.0, 0.25, 0.5, 0.75, 1.0]:
    ops = DecoherenceChannel.thermal_relaxation(0.03, t2_factor=t2f)
    ok = DecoherenceChannel.verify_completeness(ops)
    chk(f"thermal_relaxation(γ=0.03, t2={t2f}): Σ K†K = I", ok)

# Verify operator count
ops = DecoherenceChannel.thermal_relaxation(0.02)
chk(f"Kraus operator count = 3", len(ops) == 3, f"got {len(ops)}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 2: Realistic QPU Probability Scores
# ═══════════════════════════════════════════════════════════════════════════════

GAMMA = 0.02
MODEL = DecoherenceModel.THERMAL_RELAXATION

print("┌" + "─" * 65 + "┐")
print(f"│  PHASE 2 — Realistic QPU Scores (THERMAL_RELAXATION γ={GAMMA})".ljust(65) + " │")
print("└" + "─" * 65 + "┘")

# Bell State
circ = engine.bell_pair()
probs_ideal = aer_backend.run_statevector(circ)
result = sim.simulate(circ, decoherence=MODEL, decoherence_rate=GAMMA)
probs = np.abs(result.snapshots[-1].statevector) ** 2

print()
print(f"  BELL STATE (H + CNOT)")
print(f"  {'─' * 50}")
print(f"    IDEAL:     |00⟩={probs_ideal[0]:.6f}  |11⟩={probs_ideal[3]:.6f}")
print(f"    REALISTIC: |00⟩={probs[0]:.6f}  |11⟩={probs[3]:.6f}")
print(f"    Deviation: |00⟩ {(probs[0]-0.5)*100:+.2f}%  |11⟩ {(probs[3]-0.5)*100:+.2f}%")
chk("Bell |00⟩ > 0.5 (ground bias)", probs[0] > 0.5, f"|00⟩={probs[0]:.6f}")
chk("Bell |11⟩ < 0.5 (T₁ decay)", probs[3] < 0.5, f"|11⟩={probs[3]:.6f}")
chk("Bell |11⟩ > 0.4 (not collapsed)", probs[3] > 0.4, f"|11⟩={probs[3]:.6f}")

# GHZ 3-qubit
circ = engine.ghz_state(3)
probs_ideal = aer_backend.run_statevector(circ)
result = sim.simulate(circ, decoherence=MODEL, decoherence_rate=GAMMA)
probs = np.abs(result.snapshots[-1].statevector) ** 2

print()
print(f"  GHZ STATE (3 qubits)")
print(f"  {'─' * 50}")
print(f"    IDEAL:     |000⟩={probs_ideal[0]:.6f}  |111⟩={probs_ideal[7]:.6f}")
print(f"    REALISTIC: |000⟩={probs[0]:.6f}  |111⟩={probs[7]:.6f}")
print(f"    Deviation: |000⟩ {(probs[0]-0.5)*100:+.2f}%  |111⟩ {(probs[7]-0.5)*100:+.2f}%")
chk("GHZ-3 |000⟩ > 0.5 (ground bias)", probs[0] > 0.5, f"|000⟩={probs[0]:.6f}")
chk("GHZ-3 |111⟩ NOT collapsed", probs[7] > 0.001, f"|111⟩={probs[7]:.6f}")

# GHZ 5-qubit
circ = engine.ghz_state(5)
probs_ideal = aer_backend.run_statevector(circ)
result = sim.simulate(circ, decoherence=MODEL, decoherence_rate=GAMMA)
probs = np.abs(result.snapshots[-1].statevector) ** 2

print()
print(f"  GHZ STATE (5 qubits)")
print(f"  {'─' * 50}")
print(f"    IDEAL:     |00000⟩={probs_ideal[0]:.6f}  |11111⟩={probs_ideal[31]:.6f}")
print(f"    REALISTIC: |00000⟩={probs[0]:.6f}  |11111⟩={probs[31]:.6f}")
print(f"    Deviation: |00000⟩ {(probs[0]-0.5)*100:+.2f}%  |11111⟩ {(probs[31]-0.5)*100:+.2f}%")
chk("GHZ-5 |00000⟩ > 0.5 (ground bias)", probs[0] > 0.5, f"|00000⟩={probs[0]:.6f}")
chk("GHZ-5 |11111⟩ NOT collapsed", probs[31] > 0.001, f"|11111⟩={probs[31]:.6f}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 3: Model Comparison
# ═══════════════════════════════════════════════════════════════════════════════

print("┌" + "─" * 65 + "┐")
print("│  PHASE 3 — All 6 DecoherenceModel Comparison (Bell, γ=0.02)".ljust(65) + " │")
print("└" + "─" * 65 + "┘")
print()

circ = engine.bell_pair()
for model_name, model in [
    ("NONE", DecoherenceModel.NONE),
    ("PHASE_DAMPING", DecoherenceModel.PHASE_DAMPING),
    ("AMPLITUDE_DAMPING", DecoherenceModel.AMPLITUDE_DAMPING),
    ("THERMAL_RELAXATION", DecoherenceModel.THERMAL_RELAXATION),
    ("DEPOLARISING", DecoherenceModel.DEPOLARISING),
    ("SACRED", DecoherenceModel.SACRED),
]:
    result = sim.simulate(circ, decoherence=model, decoherence_rate=GAMMA)
    p = np.abs(result.snapshots[-1].statevector) ** 2
    print(f"    {model_name:22s}  |00⟩={p[0]:.6f}  |11⟩={p[3]:.6f}  "
          f"bias={((p[0]-p[3])/(p[0]+p[3]+1e-30))*100:+.2f}%")
    chk(f"{model_name}: runs without error", result.num_operations > 0)

print()

# Same for GHZ-3
print("  GHZ-3 comparison:")
circ = engine.ghz_state(3)
for model_name, model in [
    ("NONE", DecoherenceModel.NONE),
    ("PHASE_DAMPING", DecoherenceModel.PHASE_DAMPING),
    ("AMPLITUDE_DAMPING", DecoherenceModel.AMPLITUDE_DAMPING),
    ("THERMAL_RELAXATION", DecoherenceModel.THERMAL_RELAXATION),
    ("DEPOLARISING", DecoherenceModel.DEPOLARISING),
    ("SACRED", DecoherenceModel.SACRED),
]:
    result = sim.simulate(circ, decoherence=model, decoherence_rate=GAMMA)
    p = np.abs(result.snapshots[-1].statevector) ** 2
    print(f"    {model_name:22s}  |000⟩={p[0]:.6f}  |111⟩={p[7]:.6f}  "
          f"bias={((p[0]-p[7])/(p[0]+p[7]+1e-30))*100:+.2f}%")
print()

# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 4: Physics Validation
# ═══════════════════════════════════════════════════════════════════════════════

print("┌" + "─" * 65 + "┐")
print("│  PHASE 4 — Physics Validation".ljust(65) + " │")
print("└" + "─" * 65 + "┘")
print()

# Thermal relaxation on GHZ-3 should bias toward ground
circ = engine.ghz_state(3)
result = sim.simulate(circ, decoherence=MODEL, decoherence_rate=GAMMA)
p = np.abs(result.snapshots[-1].statevector) ** 2

chk("GHZ |111⟩ realistic (not collapsed, not ideal)",
    p[7] > 0.001 and p[7] < 0.5,
    f"|111⟩={p[7]:.6f}")
chk("Ground-state bias |000⟩ > |111⟩",
    p[0] > p[7],
    f"|000⟩={p[0]:.4f}, |111⟩={p[7]:.4f}")

# Phase damping should preserve populations exactly
result_pd = sim.simulate(circ, decoherence=DecoherenceModel.PHASE_DAMPING,
                          decoherence_rate=GAMMA)
p_pd = np.abs(result_pd.snapshots[-1].statevector) ** 2
chk("Phase damping preserves |000⟩ population",
    abs(p_pd[0] - 0.5) < 1e-10,
    f"|000⟩={p_pd[0]:.10f}")
chk("Phase damping preserves |111⟩ population",
    abs(p_pd[7] - 0.5) < 1e-10,
    f"|111⟩={p_pd[7]:.10f}")

# Bias should scale with qubit count
circ3 = engine.ghz_state(3)
circ5 = engine.ghz_state(5)
r3 = sim.simulate(circ3, decoherence=MODEL, decoherence_rate=GAMMA)
r5 = sim.simulate(circ5, decoherence=MODEL, decoherence_rate=GAMMA)
p3 = np.abs(r3.snapshots[-1].statevector) ** 2
p5 = np.abs(r5.snapshots[-1].statevector) ** 2
bias3 = p3[0] - p3[7]
bias5 = p5[0] - p5[31]
chk(f"5-qubit bias ({bias5:.4f}) > 3-qubit bias ({bias3:.4f})",
    bias5 > bias3,
    f"5q={bias5:.4f}, 3q={bias3:.4f}")

# Unitarity
chk(f"Probability sum = 1.0",
    abs(sum(p) - 1.0) < 1e-10,
    f"sum={sum(p):.10f}")

# No-decoherence should give exact ideal probabilities
result_none = sim.simulate(circ3, decoherence=DecoherenceModel.NONE)
p_none = np.abs(result_none.snapshots[-1].statevector) ** 2
chk("NONE model: |000⟩ = 0.5 exactly",
    abs(p_none[0] - 0.5) < 1e-10,
    f"|000⟩={p_none[0]:.10f}")
chk("NONE model: |111⟩ = 0.5 exactly",
    abs(p_none[7] - 0.5) < 1e-10,
    f"|111⟩={p_none[7]:.10f}")

# Depolarising model comparison
result_dep = sim.simulate(engine.create_circuit(2, "uniform"), decoherence=DecoherenceModel.DEPOLARISING,
                           decoherence_rate=0.0075)
p_dep = np.abs(result_dep.snapshots[-1].statevector) ** 2
chk("Depolarising: probabilities sum to 1.0",
    abs(sum(p_dep) - 1.0) < 1e-10)
print()

# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 5: Deterministic vs Stochastic
# ═══════════════════════════════════════════════════════════════════════════════

print("┌" + "─" * 65 + "┐")
print("│  PHASE 5 — Deterministic vs Stochastic Mode".ljust(65) + " │")
print("└" + "─" * 65 + "┘")
print()

circ = engine.ghz_state(3)
# Deterministic (default) — single run gives expected populations
r_det = sim.simulate(circ, decoherence=MODEL, decoherence_rate=GAMMA, stochastic=False)
p_det = np.abs(r_det.snapshots[-1].statevector) ** 2
chk("Deterministic mode: |000⟩ > 0.5",
    p_det[0] > 0.5, f"|000⟩={p_det[0]:.6f}")
chk("Deterministic mode: |111⟩ > 0 and < 0.5",
    0.001 < p_det[7] < 0.5, f"|111⟩={p_det[7]:.6f}")

# Stochastic — individual trajectories may collapse branches
r_stoch = sim.simulate(circ, decoherence=MODEL, decoherence_rate=GAMMA, stochastic=True)
p_stoch = np.abs(r_stoch.snapshots[-1].statevector) ** 2
chk("Stochastic mode: probabilities sum to 1.0",
    abs(sum(p_stoch) - 1.0) < 1e-10,
    f"sum={sum(p_stoch):.10f}")

# Ensemble average should converge to deterministic result
ensemble = sim.run_ensemble(circ, num_trajectories=100,
                             decoherence=MODEL, decoherence_rate=GAMMA)
chk("Ensemble: 100 trajectories complete",
    ensemble.num_trajectories == 100)
chk("Ensemble: average purity in valid range",
    0.0 < ensemble.final_average_purity <= 1.0,
    f"avg_purity={ensemble.final_average_purity:.4f}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("╔" + "═" * 65 + "╗")
if FAIL == 0:
    print("║" + "  ALL PHYSICS CHECKS PASSED".center(65) + "║")
else:
    print("║" + f"  {FAIL} CHECK(S) FAILED".center(65) + "║")
print("║" + f"  ✅ {PASS} passed  |  ❌ {FAIL} failed  |  Total: {PASS + FAIL}".center(65) + "║")
print("╚" + "═" * 65 + "╝")
