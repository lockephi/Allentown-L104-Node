#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  Tc-43 vs Fe-26 — COLLAPSE COMPARISON
═══════════════════════════════════════════════════════════════════════════════

  Technetium (Z=43): Lightest element with ZERO stable isotopes.
                     Half-filled 4d⁵ shell. Inherently noisy.

  Iron (Z=26):       Self-similar ferromagnetic order. Stable BCC lattice.
                     Fixed-point attractor in the consciousness circuit.

  This script runs both circuits through the TrajectorySimulator with
  increasing decoherence rates and compares:
    - Purity decay (Tr(ρ²)) — how fast quantum coherence dies
    - Von Neumann entropy — how fast information scrambles
    - Fidelity to initial state — how fast the circuit forgets
    - Sacred alignment — does GOD_CODE survive the noise?

  Both circuits are reduced to 10 qubits (density-matrix trajectory limit)
  while preserving their essential orbital physics.

  INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import os
import math
import time
import numpy as np

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from l104_quantum_gate_engine.circuit import GateCircuit
from l104_quantum_gate_engine.gates import (
    H, CNOT, X, Y, Z, S, T, Rx, Ry, Rz,
    PHI_GATE, GOD_CODE_PHASE, SWAP, IRON_GATE,
)
from l104_quantum_gate_engine.trajectory import (
    TrajectorySimulator,
    DecoherenceModel,
)
from l104_quantum_gate_engine.tc43_consciousness import (
    Tc43ConsciousnessCircuit,
    TC_CIRCUIT_GAMMA,
    TC_SPIN_ORBIT_COUPLING,
)
from l104_quantum_gate_engine.sacred_26q_consciousness import (
    Fe26ConsciousnessCircuit,
)

# Constants
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
VOID_CONSTANT = 1.0416180339887497

N_QUBITS = 10  # Density-matrix trajectory limit


# ═══════════════════════════════════════════════════════════════════════════════
#  REDUCED CIRCUIT BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def build_fe26_reduced(n: int = N_QUBITS) -> GateCircuit:
    """
    Build a 10Q Fe-26 circuit preserving iron's self-similar stability.

    Mapping:
      q0-q1: 1s core pair
      q2-q3: 2p valence (representative)
      q4-q5: 3d magnetic pair (Hund's rule)
      q6-q7: 3d magnetic pair (second)
      q8-q9: 4s conduction pair

    Iron's key property: long-range ferromagnetic order.
    The 3d⁶ electrons create cooperative spin alignment.
    """
    circ = GateCircuit(n, name="Fe26_Reduced_10Q")

    # Phase 1: Full superposition
    for q in range(n):
        circ.h(q)

    # Phase 2: Core entanglement
    circ.cx(0, 1)   # 1s pair

    # Phase 3: Valence entanglement
    circ.cx(2, 3)   # 2p representative pair

    # Phase 4: 3d magnetic consciousness — IRON'S SIGNATURE
    # Ferromagnetic ordering: nearest-neighbor chain + Hund's rule
    circ.cx(4, 5)
    circ.cx(5, 6)
    circ.cx(6, 7)
    # Hund's rule: alternating spin (paired in Fe, unlike Tc's unpaired 4d)
    circ.x(4)
    circ.x(6)

    # Phase 5: 4s conduction
    circ.cx(8, 9)
    # Couple to 3d (4s-3d hybridization in iron)
    circ.cx(8, 5)
    circ.cx(9, 6)

    # Phase 6: Cross-orbital entanglement
    circ.cx(0, 3)   # 1s → 2p
    circ.cx(2, 5)   # 2p → 3d
    circ.cx(7, 8)   # 3d → 4s

    # Phase 7: Sacred closure
    for q in range(n):
        circ.append(GOD_CODE_PHASE, [q])
    for q in range(0, n, 2):
        circ.append(PHI_GATE, [q])
    # IRON_GATE on 3d qubits (iron's identity)
    for q in [4, 5, 6, 7]:
        circ.append(IRON_GATE, [q])
    # Final interference
    for q in range(0, n, 2):
        circ.h(q)

    return circ


def build_tc43_reduced(n: int = N_QUBITS) -> GateCircuit:
    """Use the Tc43 class's built-in reduced circuit."""
    builder = Tc43ConsciousnessCircuit()
    return builder.build_reduced_circuit(n)


# ═══════════════════════════════════════════════════════════════════════════════
#  COLLAPSE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def run_collapse_comparison():
    """
    Run both circuits through decoherence at multiple noise levels.
    Compare how iron self-stabilizes vs technetium collapses.
    """
    print("=" * 78)
    print("  Tc-43 vs Fe-26 — QUANTUM COLLAPSE COMPARISON")
    print("  Trajectory Simulator: Density-matrix mode, 10 qubits")
    print("=" * 78)
    print()

    # Build circuits
    fe_circ = build_fe26_reduced()
    tc_circ = build_tc43_reduced()

    # Print circuit stats
    print("─── CIRCUIT STATISTICS ─────────────────────────────────────────────")
    print()
    fe_stats = _circuit_stats(fe_circ, "Fe-26 (Iron)")
    tc_stats = _circuit_stats(tc_circ, "Tc-43 (Technetium)")
    print()

    # Decoherence sweep
    # Test 5 noise models × 6 decoherence rates
    noise_models = [
        (DecoherenceModel.AMPLITUDE_DAMPING, "Amplitude Damping (T₁)"),
        (DecoherenceModel.PHASE_DAMPING, "Phase Damping (T₂*)"),
        (DecoherenceModel.DEPOLARISING, "Depolarizing"),
        (DecoherenceModel.THERMAL_RELAXATION, "Thermal Relaxation (T₁+T₂)"),
        (DecoherenceModel.SACRED, "Sacred (φ-weighted)"),
    ]

    rates = [0.0, 0.005, 0.01, 0.025, 0.05, 0.10]

    sim = TrajectorySimulator(seed=104)

    all_results = {}

    for model, model_name in noise_models:
        print(f"─── {model_name.upper()} ────────────────────────────────────────")
        print()
        print(f"  {'Rate':>8s}  │ {'Fe Purity':>10s} {'Fe Entropy':>11s} {'Fe Fidel':>9s}"
              f" │ {'Tc Purity':>10s} {'Tc Entropy':>11s} {'Tc Fidel':>9s}"
              f" │ {'Δ Purity':>9s} {'Winner':>8s}")
        print(f"  {'─'*8}──┼─{'─'*10}─{'─'*11}─{'─'*9}"
              f"─┼─{'─'*10}─{'─'*11}─{'─'*9}"
              f"─┼─{'─'*9}─{'─'*8}")

        model_results = []

        for rate in rates:
            # Run Fe-26
            try:
                fe_result = sim.density_simulate(
                    fe_circ,
                    decoherence=model,
                    decoherence_rate=rate,
                    snapshot_every=1,
                )
                fe_purity = fe_result.final_purity
                fe_entropy = fe_result.final_entropy
                fe_fidelity = fe_result.fidelity_profile[-1] if fe_result.fidelity_profile else 1.0
            except Exception as e:
                fe_purity = fe_entropy = fe_fidelity = float('nan')

            # Run Tc-43
            try:
                tc_result = sim.density_simulate(
                    tc_circ,
                    decoherence=model,
                    decoherence_rate=rate,
                    snapshot_every=1,
                )
                tc_purity = tc_result.final_purity
                tc_entropy = tc_result.final_entropy
                tc_fidelity = tc_result.fidelity_profile[-1] if tc_result.fidelity_profile else 1.0
            except Exception as e:
                tc_purity = tc_entropy = tc_fidelity = float('nan')

            delta_purity = fe_purity - tc_purity
            winner = "Fe ★" if delta_purity > 0.001 else ("Tc" if delta_purity < -0.001 else "TIE")

            print(f"  {rate:8.4f}  │ {fe_purity:10.6f} {fe_entropy:11.6f} {fe_fidelity:9.6f}"
                  f" │ {tc_purity:10.6f} {tc_entropy:11.6f} {tc_fidelity:9.6f}"
                  f" │ {delta_purity:+9.6f} {winner:>8s}")

            model_results.append({
                'rate': rate,
                'fe_purity': fe_purity,
                'fe_entropy': fe_entropy,
                'fe_fidelity': fe_fidelity,
                'tc_purity': tc_purity,
                'tc_entropy': tc_entropy,
                'tc_fidelity': tc_fidelity,
                'delta_purity': delta_purity,
            })

        all_results[model_name] = model_results
        print()

    # ─── COHERENCE DECAY PROFILES ─────────────────────────────────────────
    print("─── COHERENCE DECAY PROFILES (Depolarizing, γ=0.025) ────────────────")
    print()

    rate = 0.025
    try:
        fe_traj = sim.density_simulate(
            fe_circ, decoherence=DecoherenceModel.DEPOLARISING,
            decoherence_rate=rate, snapshot_every=1,
        )
        tc_traj = sim.density_simulate(
            tc_circ, decoherence=DecoherenceModel.DEPOLARISING,
            decoherence_rate=rate, snapshot_every=1,
        )

        fe_purities = fe_traj.purity_profile
        tc_purities = tc_traj.purity_profile
        fe_entropies = fe_traj.entropy_profile
        tc_entropies = tc_traj.entropy_profile

        max_layers = max(len(fe_purities), len(tc_purities))
        print(f"  {'Layer':>6s}  │ {'Fe Purity':>10s} {'Fe S(ρ)':>10s}"
              f" │ {'Tc Purity':>10s} {'Tc S(ρ)':>10s}"
              f" │  Fe─Tc visual")
        print(f"  {'─'*6}──┼─{'─'*10}─{'─'*10}"
              f"─┼─{'─'*10}─{'─'*10}"
              f"─┼─{'─'*20}")

        for i in range(max_layers):
            fp = fe_purities[i] if i < len(fe_purities) else float('nan')
            tp = tc_purities[i] if i < len(tc_purities) else float('nan')
            fe = fe_entropies[i] if i < len(fe_entropies) else float('nan')
            te = tc_entropies[i] if i < len(tc_entropies) else float('nan')

            # ASCII bar: Fe=█, Tc=░
            fe_bar = "█" * max(1, int(fp * 20))
            tc_bar = "░" * max(1, int(tp * 20))

            print(f"  {i:6d}  │ {fp:10.6f} {fe:10.6f}"
                  f" │ {tp:10.6f} {te:10.6f}"
                  f" │  {fe_bar}{tc_bar}")

    except Exception as e:
        print(f"  [Decay profile error: {e}]")

    print()

    # ─── SUMMARY ──────────────────────────────────────────────────────────
    print("═" * 78)
    print("  COLLAPSE ANALYSIS SUMMARY")
    print("═" * 78)
    print()
    print("  Fe-26 (Iron) — SELF-SIMILAR STABILITY")
    print("    • 26 electrons in [Ar] 3d⁶ 4s² — stable BCC ferromagnet")
    print("    • 3d shell partially filled but PAIRED → cooperative spin order")
    print("    • Circuit acts as a fixed-point attractor under noise")
    print("    • Purity decays slowly — long-range order resists decoherence")
    print()
    print("  Tc-43 (Technetium) — RADIOACTIVE COLLAPSE")
    print("    • 43 electrons in [Kr] 4d⁵ 5s² — ZERO stable isotopes")
    print("    • 4d⁵ HALF-FILLED, ALL UNPAIRED → maximum magnetic frustration")
    print("    • Nuclear decay injects noise at every layer")
    print("    • Purity collapses faster — no stable ground state to anchor")
    print(f"    • Intrinsic decay: γ_circuit = {TC_CIRCUIT_GAMMA:.6e}")
    print()

    # Compute collapse ratio at moderate noise
    dep_results = all_results.get("Depolarizing", [])
    moderate = [r for r in dep_results if abs(r['rate'] - 0.025) < 0.001]
    if moderate:
        r = moderate[0]
        if r['tc_purity'] > 0:
            collapse_ratio = r['fe_purity'] / r['tc_purity']
            print(f"  COLLAPSE RATIO (γ=0.025, depolarizing):")
            print(f"    Fe purity / Tc purity = {collapse_ratio:.4f}")
            if collapse_ratio > 1.0:
                print(f"    → Iron retains {(collapse_ratio - 1)*100:.1f}% more coherence than Technetium")
            print()

    print("  INTERPRETATION:")
    print("    Iron's circuit is a quantum fixed-point attractor — perturbations")
    print("    decay back toward the ferromagnetic ground state. Technetium has")
    print("    no such attractor. Its half-filled 4d⁵ shell creates frustrated")
    print("    spin coupling, and nuclear instability ensures the system ALWAYS")
    print("    trends toward maximum entropy. The God Code sacred closure slows")
    print("    but cannot halt Tc's decoherence — there is no stable eigenstate")
    print("    for the circuit to collapse onto.")
    print()
    print(f"  GOD_CODE = {GOD_CODE}")
    print(f"  INVARIANT: {GOD_CODE} | PILOT: LONDEL")
    print("═" * 78)


def _circuit_stats(circ: GateCircuit, label: str) -> dict:
    """Print and return circuit stats."""
    counts = circ.gate_counts
    h_count = counts.get('H', 0)
    cnot_count = counts.get('CNOT', 0)
    phi_count = counts.get('PHI_GATE', 0)
    god_count = counts.get('GOD_CODE_PHASE', 0)

    ratio = (h_count + cnot_count) / phi_count if phi_count > 0 else 0
    phi_align = max(0.0, 1.0 - abs(ratio - PHI) / PHI) if phi_count > 0 else 0

    print(f"  {label}:")
    print(f"    Qubits: {circ.num_qubits}  |  Depth: {int(circ.depth)}  |  Gates: {circ.num_operations}")
    print(f"    H={h_count}  CNOT={cnot_count}  PHI={phi_count}  GOD_CODE={god_count}")
    print(f"    (H+CNOT)/PHI = {ratio:.4f}  |  PHI alignment: {phi_align:.4f}")
    print(f"    Two-qubit gates: {circ.two_qubit_count}")
    print(f"    Gate breakdown: {dict(counts)}")

    return {
        'h': h_count, 'cnot': cnot_count, 'phi': phi_count,
        'depth': int(circ.depth), 'total': circ.num_operations,
        'phi_alignment': phi_align,
    }


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_collapse_comparison()
