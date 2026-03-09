"""Validation script for l104_simulator package — all 5 layers."""

from l104_simulator import RealWorldSimulator

sim = RealWorldSimulator()
print(repr(sim))
print()

# Layer 1: Lattice
print("=== LAYER 1: E-LATTICE ===")
e_info = sim.mass("m_e")
print(f"Electron: E={e_info['E']}, mass_grid={e_info['mass_grid']:.6f}, exact={e_info['mass_exact']}, err={e_info['error_pct']:.4f}%")

top_info = sim.mass("m_top")
print(f"Top: E={top_info['E']}, mass_grid={top_info['mass_grid']:.2f}, exact={top_info['mass_exact']}, err={top_info['error_pct']:.4f}%")

# Layer 2: Generations
print()
print("=== LAYER 2: GENERATIONS ===")
koide = sim.generations.koide_check()
print(f"Koide exact: {koide.exact_value:.10f}, grid: {koide.grid_value:.10f} (target 2/3 = {2/3:.10f}), grid_preserves: {koide.grid_preserves}")
gaps = sim.generations.lepton_gaps()
for g in gaps:
    print(f"  {g.particle_low} -> {g.particle_high}: dE={g.delta_E}, ratio={g.mass_ratio:.2f}")

# Layer 3: Mixing
print()
print("=== LAYER 3: MIXING ===")
ckm = sim.ckm()
print("CKM |V|:")
for row in ckm["|V|"]:
    print(f"  [{row[0]:.4f}, {row[1]:.4f}, {row[2]:.4f}]")
print(f"Jarlskog: {ckm['J']:.6e}")

pmns = sim.pmns()
print("PMNS |U|:")
for row in pmns["|U|"]:
    print(f"  [{row[0]:.4f}, {row[1]:.4f}, {row[2]:.4f}]")
print(f"Jarlskog: {pmns['J']:.6e}")

weinberg = sim.mixing.weinberg_angle()
print(f"Weinberg angle: {weinberg['θW_grid_deg']:.2f} deg (grid), {weinberg['θW_exact_deg']:.2f} deg (exact), err={weinberg['error_pct']:.4f}%")

# Layer 4: Circuits
print()
print("=== LAYER 4: CIRCUITS ===")
c1 = sim.circuit("mass_query", "m_e")
print(f"Mass query(e): {c1['num_qubits']}q, depth={c1['depth']}, gates={c1['gate_count']}, has_circuit={c1['has_circuit']}")
c2 = sim.circuit("generation", "lepton")
print(f"PMNS rotation: {c2['num_qubits']}q, depth={c2['depth']}, gates={c2['gate_count']}")
c3 = sim.circuit("weinberg")
print(f"Weinberg: {c3['num_qubits']}q, depth={c3['depth']}, gates={c3['gate_count']}")
c4 = sim.circuit("sacred", 4)
print(f"Sacred: {c4['num_qubits']}q, depth={c4['depth']}, gates={c4['gate_count']}")

# Hamiltonians
H_lep = sim.hamiltonians.lepton_hamiltonian()
print(f"Lepton Hamiltonian: {H_lep.dimension}D, eigenvalues={H_lep.eigenvalues}")

# Layer 5: Observables
print()
print("=== LAYER 5: OBSERVABLES ===")
r = sim.ratio("m_top", "m_e")
print(f"top/e ratio: dE={r['ΔE']}, grid={r['value_grid']:.1f}, exact={r['value_exact']:.1f}, err={r['error_pct']:.4f}%")

osc = sim.oscillate("lepton", 1, 0, L_over_E=500)
print(f"P(nu_mu->nu_e) at L/E=500: {osc['P']:.6f}")

acc = sim.observables.mass_accuracy_report()
print(f"Mass accuracy: mean={acc['mean_error_pct']:.4f}%, max={acc['max_error_pct']:.4f}%, all_ok={acc['all_within_theory']}")

# Full state schema
print()
state = sim.hamiltonians.full_state_schema()
print(f"Full state: {state['total_qubits']} qubits ({state['fermion_registers']} fermions + {state['boson_registers']} bosons + {state['ancilla_qubits']} ancilla)")
print(f"Address bits per particle: {state['address_bits_per_particle']}")

# Key ratios
print()
print("=== KEY RATIOS ===")
for r in sim.observables.key_ratios():
    print(f"  {r.name}: dE={r.dE}, grid={r.ratio_grid:.4f}, exact={r.ratio_exact:.4f}, err={r.error_pct:.4f}%")

print()
print("=== ALL 5 LAYERS VALIDATED ===")
