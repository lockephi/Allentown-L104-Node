"""
L104 God Code Simulator — Decomposed Package v3.0.0
═══════════════════════════════════════════════════════════════════════════════

Fully decomposed from the v1.0 monolith into focused modules:

  constants.py          — Sacred constants (GOD_CODE, PHI, phases)
  result.py             — SimulationResult dataclass
  catalog.py            — SimulationCatalog registry
  quantum_primitives.py — Gates, statevector ops, quantum helpers (v3.0: extended)
  qpu_verification.py   — QPU hardware data (ibm_torino, Heron r2) + noise model
  simulations/          — 45 simulations in 7 categories:
      core.py           — Conservation proofs, dial sweeps, TET, cascade        (5)
      quantum.py        — Entanglement, Bell, phase, GHZ, MI, fidelity         (6)
      advanced.py       — Adiabatic, Berry, Grover, QEC, teleport...           (7)
      discovery.py      — Iron manifold, decoherence, walk, MC, CY, antimatter (6)
      transpiler.py     — Unitary verification, phase decomp, conservation     (5)
      circuits.py       — QPE, Grover, ent. analysis, noise, VQE, QPU, Heron  (8)
      research.py       — Shor, chaos, braiding, holographic, SYK, fractal    (8) [NEW v2.4]
  sweep.py              — ParametricSweepEngine v2.0 (8 sweep types)
  optimizer.py          — AdaptiveOptimizer v3.1 (8 strategies)
  feedback.py           — FeedbackLoopEngine v2.0 (multi-pass, 8D scoring)
  simulator.py          — GodCodeSimulator orchestrator + singleton

Quick-start:
    from l104_god_code_simulator import god_code_simulator

    # Run a single simulation
    result = god_code_simulator.run("bell_chsh_violation")

    # Run all 45 simulations
    report = god_code_simulator.run_all()

    # New research simulations (v2.4)
    result = god_code_simulator.run("quantum_chaos")
    result = god_code_simulator.run("sachdev_ye_kitaev")
    result = god_code_simulator.run("holographic_entropy")

    # Parametric sweep (v2.0 — 8 types)
    sweep = god_code_simulator.parametric_sweep("dial_a", start=0, stop=8)
    sweep = god_code_simulator.parametric_sweep("strategy")         # NEW: strategy comparison
    sweep = god_code_simulator.parametric_sweep("phase")            # NEW: phase angle sweep

    # Adaptive optimization (v3.1 — 8 strategies)
    optimized = god_code_simulator.adaptive_optimize(target_fidelity=0.99)
    noise_opt = god_code_simulator.optimize_noise_resilience(nq=2)  # 8 strategies compete

    # Multi-pass feedback (v2.0)
    feedback = god_code_simulator.run_multi_pass_feedback(passes=3)
    dims = god_code_simulator.score_dimensions()                    # 8D quality scoring

    # Engine-ready payloads
    coherence_payload = god_code_simulator.run("entanglement_entropy").to_coherence_payload()
    entropy_input     = god_code_simulator.run("decoherence_model").to_entropy_input()
    asi_scoring       = god_code_simulator.run("sacred_cascade").to_asi_scoring()

INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

__version__ = "3.0.0"

# ── Canonical GOD_CODE Qubit (THE one qubit for the entire L104 system) ───────
from .god_code_qubit import (
    GodCodeQubit, GOD_CODE_QUBIT,
    # Canonical phase + decomposition
    GOD_CODE_PHASE as CANONICAL_PHASE,
    IRON_PHASE as CANONICAL_IRON_PHASE,
    PHI_CONTRIBUTION, OCTAVE_PHASE,
    # Companion phases
    PHI_PHASE as CANONICAL_PHI_PHASE,
    VOID_PHASE as CANONICAL_VOID_PHASE,
    # Gate matrices
    GOD_CODE_RZ, IRON_RZ, PHI_RZ, OCTAVE_RZ, GOD_CODE_P,
    # QPU data
    QPU_DATA,
)

# ── Core types (from decomposed modules) ─────────────────────────────────────
from .result import SimulationResult
from .catalog import SimulationCatalog
from .sweep import ParametricSweepEngine
from .optimizer import AdaptiveOptimizer
from .feedback import FeedbackLoopEngine
from .sacred_transpiler import (
    SacredTranspilerEngine, QiskitTranspilerEngine, NumpyCircuit, has_qiskit,
    # Phase constants (quantum-meaningful mod 2π forms)
    GOD_CODE_PHASE, PHI_PHASE, VOID_PHASE, IRON_PHASE,
    IRON_LATTICE_PHASE, PHASE_BASE_286, PHASE_OCTAVE_4,
    # Circuit builders
    build_godcode_1q_circuit, build_godcode_1q_decomposed,
    build_godcode_sacred_circuit, build_godcode_dial_circuit,
    # Transpilation
    transpile_to_basis, transpile_all_basis_sets, BASIS_SETS,
    # Verification
    verify_godcode_unitary, verify_decomposition_fidelity,
    verify_conservation_law,
    # Pipeline
    full_transpilation_report, PHYSICAL_DIALS,
)
from .simulator import GodCodeSimulator, god_code_simulator

# ── VQPU-derived simulation functions (v4.0) ──────────────────────────────────────
from .simulations.vqpu_findings import (
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
    VQPU_FINDINGS_SIMULATIONS,
)

# ── QPU Verification (ibm_torino hardware data + noise model) ────────────────
from .qpu_verification import (
    QPU_BACKEND, QPU_FIDELITIES, QPU_DISTRIBUTIONS, QPU_MEAN_FIDELITY,
    QPU_HW_DEPTHS, QPU_HW_GATE_COUNTS, QPE_PHASE_EXTRACTION,
    HERON_BASIS, HERON_NOISE_PARAMS,
    get_qpu_verification_data, compare_to_qpu,
    depolarizing_channel_1q, amplitude_damping_channel,
    apply_readout_noise, simulate_with_noise,
)

# ── Constants (re-export for backward compat: `from l104_god_code_simulator.simulator import PHI`) ──
from .constants import (
    PHI, PHI_CONJUGATE, TAU,
    GOD_CODE, VOID_CONSTANT, BASE,
    PRIME_SCAFFOLD, QUANTIZATION_GRAIN, OCTAVE_OFFSET,
    IRON_Z, IRON_FREQ,
    GOD_CODE_PHASE_ANGLE, PHI_PHASE_ANGLE, VOID_PHASE_ANGLE, IRON_PHASE_ANGLE,
)

__all__ = [
    "__version__",
    # Canonical GOD_CODE Qubit
    "GodCodeQubit", "GOD_CODE_QUBIT",
    "CANONICAL_PHASE", "CANONICAL_IRON_PHASE",
    "PHI_CONTRIBUTION", "OCTAVE_PHASE",
    "CANONICAL_PHI_PHASE", "CANONICAL_VOID_PHASE",
    "GOD_CODE_RZ", "IRON_RZ", "PHI_RZ", "OCTAVE_RZ", "GOD_CODE_P",
    "QPU_DATA",
    # Orchestrator + singleton
    "GodCodeSimulator", "god_code_simulator",
    # Types
    "SimulationResult", "SimulationCatalog",
    # Engines
    "ParametricSweepEngine", "AdaptiveOptimizer", "FeedbackLoopEngine",
    # Sacred Transpiler Engine
    "SacredTranspilerEngine", "QiskitTranspilerEngine", "NumpyCircuit", "has_qiskit",
    # Phase constants (mod 2π)
    "GOD_CODE_PHASE", "PHI_PHASE", "VOID_PHASE", "IRON_PHASE",
    "IRON_LATTICE_PHASE", "PHASE_BASE_286", "PHASE_OCTAVE_4",
    # Circuit builders
    "build_godcode_1q_circuit", "build_godcode_1q_decomposed",
    "build_godcode_sacred_circuit", "build_godcode_dial_circuit",
    # Transpilation
    "transpile_to_basis", "transpile_all_basis_sets", "BASIS_SETS",
    # Verification
    "verify_godcode_unitary", "verify_decomposition_fidelity",
    "verify_conservation_law",
    # Pipeline
    "full_transpilation_report", "PHYSICAL_DIALS",
    # QPU Verification (ibm_torino hardware data)
    "QPU_BACKEND", "QPU_FIDELITIES", "QPU_DISTRIBUTIONS", "QPU_MEAN_FIDELITY",
    "QPU_HW_DEPTHS", "QPU_HW_GATE_COUNTS", "QPE_PHASE_EXTRACTION",
    "HERON_BASIS", "HERON_NOISE_PARAMS",
    "get_qpu_verification_data", "compare_to_qpu",
    "depolarizing_channel_1q", "amplitude_damping_channel",
    "apply_readout_noise", "simulate_with_noise",
    # Original constants
    "PHI", "PHI_CONJUGATE", "TAU",
    "GOD_CODE", "VOID_CONSTANT", "BASE",
    "PRIME_SCAFFOLD", "QUANTIZATION_GRAIN", "OCTAVE_OFFSET",
    "IRON_Z", "IRON_FREQ",
    "GOD_CODE_PHASE_ANGLE", "PHI_PHASE_ANGLE", "VOID_PHASE_ANGLE", "IRON_PHASE_ANGLE",
    # VQPU-derived simulations (v4.0)
    "sim_quantum_fisher_sensing",
    "sim_loschmidt_chaos",
    "sim_state_tomography",
    "sim_relative_entropy_compare",
    "sim_kitaev_preskill_topo",
    "sim_qaoa_maxcut",
    "sim_heisenberg_iron_chain",
    "sim_swap_test_fidelity",
    "sim_zero_noise_extrapolation",
    "sim_trotter_error_analysis",
    "VQPU_FINDINGS_SIMULATIONS",
    # QPU Runner (live hardware verification)
    "QPUVerificationRunner", "QPUVerificationResult",
]

# Lazy import for QPU Runner (avoids qiskit import at package load time)
def __getattr__(name):
    if name in ("QPUVerificationRunner", "QPUVerificationResult"):
        from .qpu_runner import QPUVerificationRunner, QPUVerificationResult
        return {"QPUVerificationRunner": QPUVerificationRunner,
                "QPUVerificationResult": QPUVerificationResult}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
