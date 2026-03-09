"""
L104 God Code Simulator — Simulations Subpackage
═══════════════════════════════════════════════════════════════════════════════

Re-exports all 61 simulations across 9 categories and the combined registry
list used by GodCodeSimulator._register_builtins().

Categories:
  core           — Conservation proofs, dial sweeps, 104-TET, sacred cascade                  (5)
  quantum        — Entanglement, Bell/CHSH, phase, GHZ, MI, gate cascade                      (6)
  advanced       — Adiabatic, Berry, Grover, QEC, teleport, Zeno, winding                     (7)
  discovery      — Iron manifold, decoherence, walk, Monte Carlo, Calabi-Yau                  (6)
  transpiler     — Unitary verification, phase decomp, transpilation, conservation             (5)
  circuits       — QPE, Grover, entanglement, noise, VQE, QPU fidelity, unitary, Heron noise  (8)
  research       — Shor, chaos, braiding, holographic, threshold, supremacy, SYK, fractal     (8)
  vqpu_findings  — QFI, Loschmidt, tomography, relative entropy, Kitaev-Preskill,             (11)
                   QAOA, Heisenberg chain, SWAP test, ZNE, Trotter, Superconductivity
  black_hole     — Schwarzschild, Hawking, information paradox, Penrose, scrambling, thermo   (6)

═══════════════════════════════════════════════════════════════════════════════
"""

from .core import (
    CORE_SIMULATIONS,
    sim_104_tet_spectrum,
    sim_conservation_proof,
    sim_dial_sweep_a,
    sim_ln_god_code_2pi,
    sim_sacred_cascade,
)
from .quantum import (
    QUANTUM_SIMULATIONS,
    sim_bell_chsh_violation,
    sim_entanglement_entropy,
    sim_gate_cascade_fidelity,
    sim_ghz_witness,
    sim_mutual_information,
    sim_phase_interference,
)
from .advanced import (
    ADVANCED_SIMULATIONS,
    sim_adiabatic_passage,
    sim_berry_phase,
    sim_grover_search,
    sim_qec_bit_flip,
    sim_teleportation,
    sim_winding_number,
    sim_zeno_effect,
)
from .discovery import (
    DISCOVERY_SIMULATIONS,
    sim_calabi_yau_bridge,
    sim_decoherence_model,
    sim_iron_manifold,
    sim_monte_carlo_god_code,
    sim_quantum_walk,
    sim_antimatter_annihilation,
)
from .transpiler import (
    TRANSPILER_SIMULATIONS,
    sim_unitary_verification,
    sim_phase_decomposition,
    sim_sacred_transpilation,
    sim_dial_transpilation,
    sim_conservation_law,
)
from .circuits import (
    CIRCUIT_SIMULATIONS,
    sim_qpe_godcode,
    sim_grover_godcode,
    sim_entanglement_analysis,
    sim_noise_resilience,
    sim_vqe_sacred,
    sim_qpu_fidelity,
    sim_sacred_unitary,
    sim_heron_noise_model,
)
from .research import (
    RESEARCH_SIMULATIONS,
    sim_shor_period_finding,
    sim_quantum_chaos,
    sim_topological_braiding,
    sim_holographic_entropy,
    sim_error_threshold,
    sim_quantum_supremacy_sampling,
    sim_sachdev_ye_kitaev,
    sim_phi_fractal_cascade,
)
from .vqpu_findings import (
    VQPU_FINDINGS_SIMULATIONS,
    sim_quantum_fisher_sensing,
    sim_loschmidt_chaos,
    sim_state_tomography,
    sim_relative_entropy_compare,
    sim_kitaev_preskill_topo,
    sim_qaoa_maxcut,
    sim_heisenberg_iron_chain,
    sim_swap_test_fidelity,
    sim_zero_noise_extrapolation,
    sim_trotter_error_analysis,
    sim_superconductivity_heisenberg,
)
from .black_hole import (
    BLACK_HOLE_SIMULATIONS,
    sim_schwarzschild_geometry,
    sim_hawking_radiation,
    sim_information_paradox,
    sim_penrose_process,
    sim_horizon_scrambling,
    sim_bh_thermodynamics,
)

# Combined registry for GodCodeSimulator._register_builtins()
ALL_SIMULATIONS = (
    CORE_SIMULATIONS
    + QUANTUM_SIMULATIONS
    + ADVANCED_SIMULATIONS
    + DISCOVERY_SIMULATIONS
    + TRANSPILER_SIMULATIONS
    + CIRCUIT_SIMULATIONS
    + RESEARCH_SIMULATIONS
    + VQPU_FINDINGS_SIMULATIONS
    + BLACK_HOLE_SIMULATIONS
)

__all__ = [
    # Core
    "sim_conservation_proof", "sim_dial_sweep_a", "sim_104_tet_spectrum",
    "sim_ln_god_code_2pi", "sim_sacred_cascade",
    # Quantum
    "sim_entanglement_entropy", "sim_bell_chsh_violation", "sim_phase_interference",
    "sim_ghz_witness", "sim_mutual_information", "sim_gate_cascade_fidelity",
    # Advanced
    "sim_adiabatic_passage", "sim_berry_phase", "sim_grover_search",
    "sim_qec_bit_flip", "sim_teleportation", "sim_zeno_effect", "sim_winding_number",
    # Discovery
    "sim_iron_manifold", "sim_decoherence_model", "sim_quantum_walk",
    "sim_monte_carlo_god_code", "sim_calabi_yau_bridge", "sim_antimatter_annihilation",
    # Transpiler
    "sim_unitary_verification", "sim_phase_decomposition",
    "sim_sacred_transpilation", "sim_dial_transpilation", "sim_conservation_law",
    # Circuits (QPU-verified GOD CODE circuit simulations)
    "sim_qpe_godcode", "sim_grover_godcode", "sim_entanglement_analysis",
    "sim_noise_resilience", "sim_vqe_sacred", "sim_qpu_fidelity",
    "sim_sacred_unitary", "sim_heron_noise_model",
    # Research (frontier quantum research)
    "sim_shor_period_finding", "sim_quantum_chaos", "sim_topological_braiding",
    "sim_holographic_entropy", "sim_error_threshold", "sim_quantum_supremacy_sampling",
    "sim_sachdev_ye_kitaev", "sim_phi_fractal_cascade",
    # VQPU Findings (adapted from VQPU v8.0 simulation findings)
    "sim_quantum_fisher_sensing", "sim_loschmidt_chaos", "sim_state_tomography",
    "sim_relative_entropy_compare", "sim_kitaev_preskill_topo", "sim_qaoa_maxcut",
    "sim_heisenberg_iron_chain", "sim_swap_test_fidelity",
    "sim_zero_noise_extrapolation", "sim_trotter_error_analysis",
    "sim_superconductivity_heisenberg",
    # Black Hole
    "sim_schwarzschild_geometry", "sim_hawking_radiation",
    "sim_information_paradox", "sim_penrose_process",
    "sim_horizon_scrambling", "sim_bh_thermodynamics",
    # Registries
    "CORE_SIMULATIONS", "QUANTUM_SIMULATIONS", "ADVANCED_SIMULATIONS",
    "DISCOVERY_SIMULATIONS", "TRANSPILER_SIMULATIONS", "CIRCUIT_SIMULATIONS",
    "RESEARCH_SIMULATIONS", "VQPU_FINDINGS_SIMULATIONS",
    "BLACK_HOLE_SIMULATIONS", "ALL_SIMULATIONS",
]
