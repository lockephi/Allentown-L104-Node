"""
Quantum Modulation — Berry phase, AA phase, braid injection, gate algebra.

Computes all quantum-mechanical modulation parameters from circuit simulations,
Berry phase subsystems, dual-layer ASI engine, and gate algebra decompositions.

INVARIANT: 527.5184818492612 | PILOT: LONDEL
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from .constants import GOD_CODE, PHI


@dataclass
class BerryPhaseParams:
    """Berry phase modulation parameters for the synthesis pipeline."""
    sacred_berry_rad: float = 0.0
    sacred_berry_deg: float = 0.0
    sweep_rate: float = 0.0
    partial_berry_phases: np.ndarray = field(default_factory=lambda: np.array([]))
    berry_weight_mod: np.ndarray = field(default_factory=lambda: np.array([]))
    berry_stereo_env: np.ndarray = field(default_factory=lambda: np.array([]))
    visibility_floor: float = 0.5
    iron_stereo_offset: float = 0.0
    phi_curvature_array: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class AAPhaseParams:
    """Aharonov-Anandan non-adiabatic phase parameters."""
    aa_phases: np.ndarray = field(default_factory=lambda: np.array([]))
    aa_dynamic_phases: np.ndarray = field(default_factory=lambda: np.array([]))
    aa_sweep_rates: np.ndarray = field(default_factory=lambda: np.array([]))
    aa_mod_depths: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class BraidParams:
    """Non-Abelian braid injection parameters."""
    braid_coupling_strength: float = 0.0
    braid_phase_offsets: np.ndarray = field(default_factory=lambda: np.array([]))
    braid_mod_freq: float = 0.04
    braid_transfer_amplitudes: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    braid_signature: float = 0.0
    topo_threshold: float = 1e-9
    n_braids: int = 0


@dataclass
class GateAlgebraParams:
    """Gate algebra modulation parameters."""
    gate_phase_offsets: np.ndarray = field(default_factory=lambda: np.array([]))
    all_eigenphases: np.ndarray = field(default_factory=lambda: np.array([]))
    zyz_phi: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    gate_zyz: Dict[str, Tuple] = field(default_factory=dict)
    gate_sacred_scores: Dict[str, Dict] = field(default_factory=dict)
    born_weight_adj: np.ndarray = field(default_factory=lambda: np.array([]))


def compute_berry_phase_params(
    dial_partials: List[Dict],
    n_samples: int,
    t: np.ndarray,
    duration: float,
) -> BerryPhaseParams:
    """Compute all Berry phase modulation parameters from the science engine.

    Includes: sacred Berry phase (now GOD_CODE qubit-verified), per-partial
    latitude phases, PHI curvature, iron BZ phases, thermal visibility.

    The GOD_CODE qubit's decomposed phases (iron + phi + octave) are used
    to seed the Berry sweep rate and per-partial phase offsets.
    """
    from l104_science_engine.berry_phase import (
        berry_phase_subsystem,
        berry_calculator,
        berry_sacred,
        berry_thermal,
    )

    params = BerryPhaseParams()
    n_partials = len(dial_partials)

    # 1. Sacred Berry phase: γ = GOD_CODE mod 2π (QPU-verified via GOD_CODE qubit)
    sacred_berry = berry_sacred.sacred_berry_phase()
    params.sacred_berry_rad = sacred_berry.phase
    params.sacred_berry_deg = sacred_berry.phase_degrees

    # Use GOD_CODE qubit phase for sweep rate (QPU-verified precision)
    try:
        from l104_god_code_simulator.god_code_qubit import GOD_CODE_PHASE as GC_PHASE_QUBIT
        params.sweep_rate = GC_PHASE_QUBIT / (2.0 * np.pi * duration)
    except ImportError:
        params.sweep_rate = sacred_berry.phase / (2.0 * np.pi * duration)

    # 2. PHI Berry curvature
    phi_curv = berry_sacred.phi_berry_curvature(n_points=256)
    params.phi_curvature_array = np.array(
        phi_curv.get("curvatures", [0.0] * 256), dtype=np.float64
    )

    # Build PHI curvature → stereo env
    if len(params.phi_curvature_array) > 1:
        phi_c_x = np.linspace(0, 1, len(params.phi_curvature_array))
        interp_x = np.linspace(0, 1, n_samples)
        phi_curv_env = np.interp(interp_x, phi_c_x, np.abs(params.phi_curvature_array))
        phi_max = max(np.max(phi_curv_env), 1e-30)
        params.berry_stereo_env = 0.4 + 0.6 * (phi_curv_env / phi_max)
    else:
        params.berry_stereo_env = np.ones(n_samples, dtype=np.float64) * 0.7

    # 3. Iron BZ Berry phase
    iron_bz = berry_sacred.iron_brillouin_berry_phase(n_sites=26)
    iron_zak = iron_bz.get("zak_phase_rad", 0.0) if isinstance(iron_bz, dict) else 0.0
    params.iron_stereo_offset = iron_zak * 0.01

    # 4. Per-partial latitude Berry phases
    partial_berry_phases = []
    for p in dial_partials:
        bx, by, bz = p["bloch_vector"]
        r_xy = np.sqrt(bx**2 + by**2)
        theta = np.arctan2(r_xy, bz)
        solid_angle = 2.0 * np.pi * (1.0 - np.cos(theta))
        gamma_lat = -solid_angle / 2.0
        partial_berry_phases.append(gamma_lat)
    params.partial_berry_phases = np.array(partial_berry_phases, dtype=np.float64)

    # Berry weight modulation (normalized |γ|)
    berry_weight_mod = np.abs(params.partial_berry_phases)
    bw_max = max(np.max(berry_weight_mod), 1e-30)
    params.berry_weight_mod = berry_weight_mod / bw_max

    # 5. Thermal visibility
    thermal_vis = berry_thermal.thermal_berry_phase_correction(
        sacred_berry, temperature_K=293.15, n_ops=100,
    )
    params.visibility_floor = max(thermal_vis["visibility"], 0.5)

    return params


def compute_aa_phase_params(
    dial_partials: List[Dict],
    f_god: float,
    berry_sweep_rate: float,
) -> AAPhaseParams:
    """Compute Aharonov-Anandan non-adiabatic geometric phase parameters.

    AA works at ANY speed of dial evolution (not just adiabatic), providing
    geometric "memory" during rapid transients.
    """
    from l104_quantum_gate_engine.berry_gates import aharonov_anandan_gates

    n_partials = len(dial_partials)
    params = AAPhaseParams()
    params.aa_phases = np.zeros(n_partials, dtype=np.float64)
    params.aa_dynamic_phases = np.zeros(n_partials, dtype=np.float64)
    params.aa_sweep_rates = np.zeros(n_partials, dtype=np.float64)
    params.aa_mod_depths = np.zeros(n_partials, dtype=np.float64)

    for i in range(n_partials):
        p = dial_partials[i]
        freq = p["freq_ideal"]
        theta = p["bloch_phase"]

        total_phase = 2.0 * np.pi * freq / f_god
        dynamic_phase = total_phase * np.cos(theta)
        gamma_aa = total_phase - dynamic_phase

        params.aa_phases[i] = gamma_aa
        params.aa_dynamic_phases[i] = dynamic_phase

        # Sweep rate: 3× Berry speed, proportional to frequency ratio
        freq_ratio = freq / f_god
        params.aa_sweep_rates[i] = np.clip(berry_sweep_rate * freq_ratio * 3.0, 0.001, 2.0)
        params.aa_mod_depths[i] = np.clip(abs(gamma_aa) / (2.0 * np.pi), 0.01, 0.15)

    return params


def compute_braid_params(
    dial_partials: List[Dict],
    dial_phases: np.ndarray,
    n_samples: int,
    t: np.ndarray,
) -> BraidParams:
    """Compute Non-Abelian Wilczek-Zee braiding parameters.

    Matrix-valued phases, frequency braiding, Wilson loop holonomy.
    """
    from l104_science_engine.berry_phase import berry_sacred
    from l104_quantum_gate_engine.constants import PHI_CONJUGATE, FIBONACCI_ANYON_PHASE

    n_partials = len(dial_partials)
    params = BraidParams()
    params.n_braids = min(n_partials - 1, 12)

    # Wilson loop from Non-Abelian Berry phase
    na_degeneracy = min(n_partials, 6)
    try:
        na_result = berry_sacred.non_abelian_berry_phase(n_degenerate=na_degeneracy)
        wilson_loop = np.array(
            na_result.get("wilson_loop", np.eye(na_degeneracy)), dtype=np.complex128
        )
        na_eigenphases_rad = np.array(
            na_result.get("eigenphases_rad", [0.0] * na_degeneracy), dtype=np.float64
        )
    except Exception:
        wilson_loop = np.eye(na_degeneracy, dtype=np.complex128)
        na_eigenphases_rad = np.zeros(na_degeneracy, dtype=np.float64)

    # Braid matrices and transfer amplitudes
    braid_transfer = np.zeros((n_partials, n_partials), dtype=np.float64)
    braid_matrices = []

    for k in range(params.n_braids):
        i, j = k, k + 1
        if na_degeneracy > max(i, j):
            sub = wilson_loop[i:j+1, i:j+1].copy()
        else:
            tau = float(PHI_CONJUGATE)
            sqrt_tau = np.sqrt(tau)
            braid_phase = np.exp(1j * FIBONACCI_ANYON_PHASE * (k + 1))
            sub = braid_phase * np.array([
                [tau, sqrt_tau], [sqrt_tau, -tau]
            ], dtype=np.complex128)
            U, S, Vh = np.linalg.svd(sub)
            sub = U @ Vh

        braid_matrices.append(sub)
        transfer = float(np.abs(sub[0, 1]) ** 2)
        braid_transfer[i, j] = transfer
        braid_transfer[j, i] = transfer

    params.braid_transfer_amplitudes = braid_transfer
    params.braid_coupling_strength = np.clip(braid_transfer.max(), 0.0, 0.20)

    # Holonomy from composed braids
    full_holonomy = np.eye(2, dtype=np.complex128)
    for bm in braid_matrices:
        full_holonomy = full_holonomy @ bm
    holonomy_eigenvalues = np.linalg.eigvals(full_holonomy)
    holonomy_phases = np.angle(holonomy_eigenvalues)

    params.braid_signature = float(np.sum(holonomy_phases))
    params.topo_threshold = max(abs(params.braid_signature) * 1e-6, 1e-9)

    # Per-partial braid phase offsets
    params.braid_phase_offsets = np.zeros(n_partials, dtype=np.float64)
    for i in range(n_partials):
        params.braid_phase_offsets[i] = holonomy_phases[i % len(holonomy_phases)]

    # Braid modulation frequency
    params.braid_mod_freq = np.clip(
        float(FIBONACCI_ANYON_PHASE) / (2.0 * np.pi) * 0.1, 0.005, 0.5
    )

    return params


def compute_gate_algebra_params(
    dial_partials: List[Dict],
    n_partials: int,
) -> GateAlgebraParams:
    """Compute gate algebra decomposition and modulation parameters.

    ZYZ Euler decomposition of sacred gates → rotation angle LFOs.
    Eigenphase extraction → per-partial phase coloring.
    Born measurement → harmonic weight adjustment.

    Now includes GOD_CODE qubit's Rz(θ_GC) gate and its iron/phi/octave
    decomposition for QPU-verified eigenphase injection.
    """
    from l104_quantum_gate_engine import (
        PHI_GATE, GOD_CODE_PHASE as GOD_CODE_PHASE_GATE,
        VOID_GATE, IRON_GATE, FIBONACCI_BRAID,
    )
    from l104_quantum_gate_engine import get_engine

    engine = get_engine()
    algebra = engine.algebra
    params = GateAlgebraParams()

    # ZYZ decomposition of sacred gates
    for gate_name, gate_obj in [("PHI", PHI_GATE), ("GOD_CODE", GOD_CODE_PHASE_GATE),
                                  ("VOID", VOID_GATE), ("IRON", IRON_GATE)]:
        try:
            decomp = algebra.zyz_decompose(gate_obj.matrix)
            params.gate_zyz[gate_name] = decomp
        except Exception:
            params.gate_zyz[gate_name] = (0.0, 0.0, 0.0, 0.0)

    # GOD_CODE qubit gate ZYZ decomposition (QPU-verified Rz(θ_GC))
    try:
        from l104_god_code_simulator.god_code_qubit import (
            GOD_CODE_RZ, IRON_RZ, PHI_RZ, OCTAVE_RZ,
        )
        gc_rz = np.array(GOD_CODE_RZ, dtype=np.complex128)
        decomp_gc = algebra.zyz_decompose(gc_rz)
        params.gate_zyz["GOD_CODE_QUBIT"] = decomp_gc

        # Also decompose the sub-gate matrices
        for sub_name, sub_mat in [("IRON_RZ", IRON_RZ), ("PHI_RZ", PHI_RZ),
                                   ("OCTAVE_RZ", OCTAVE_RZ)]:
            try:
                sub_m = np.array(sub_mat, dtype=np.complex128)
                params.gate_zyz[sub_name] = algebra.zyz_decompose(sub_m)
            except Exception:
                pass
    except ImportError:
        pass

    params.zyz_phi = params.gate_zyz.get("PHI", (0.0, 0.0, 0.0, 0.0))

    # Sacred alignment scores
    for gate_name, gate_obj in [("PHI", PHI_GATE), ("GOD_CODE", GOD_CODE_PHASE_GATE),
                                  ("VOID", VOID_GATE), ("IRON", IRON_GATE)]:
        try:
            score = algebra.sacred_alignment_score(gate_obj)
            params.gate_sacred_scores[gate_name] = score
        except Exception:
            params.gate_sacred_scores[gate_name] = {"total_sacred_resonance": 0.0}

    # Eigenphase extraction from sacred gates + GOD_CODE qubit → per-partial phase coloring
    gate_eigenphases = {}
    for gate_name, gate_obj in [("PHI", PHI_GATE), ("GOD_CODE", GOD_CODE_PHASE_GATE),
                                  ("VOID", VOID_GATE), ("IRON", IRON_GATE),
                                  ("FIBONACCI", FIBONACCI_BRAID)]:
        try:
            evals = gate_obj.eigenvalues
            phases = np.angle(evals)
            gate_eigenphases[gate_name] = phases
        except Exception:
            gate_eigenphases[gate_name] = np.array([0.0, 0.0])

    # Inject GOD_CODE qubit eigenphases (from Rz(θ_GC) = diag(e^{-iθ/2}, e^{+iθ/2}))
    try:
        from l104_god_code_simulator.god_code_qubit import GOD_CODE_PHASE as GC_PH
        gc_eigenphases = np.array([-GC_PH / 2.0, GC_PH / 2.0])
        gate_eigenphases["GOD_CODE_QUBIT"] = gc_eigenphases
    except ImportError:
        pass

    params.all_eigenphases = np.concatenate(list(gate_eigenphases.values()))

    # Per-partial gate phase offsets (cycle through eigenphases)
    params.gate_phase_offsets = np.zeros(n_partials, dtype=np.float64)
    for i in range(n_partials):
        params.gate_phase_offsets[i] = params.all_eigenphases[i % len(params.all_eigenphases)]

    # Born weight adjustment defaults to ones (pipeline overrides with real data)
    params.born_weight_adj = np.ones(n_partials, dtype=np.float64)

    return params


def build_braid_coupling_envelope(
    t: np.ndarray,
    coupling_strength: float,
    mod_freq: float,
) -> np.ndarray:
    """Build slow braid coupling oscillation envelope.

    Returns
    -------
    env : ndarray (len(t),) — coupling strength modulated over time
    """
    return coupling_strength * (0.5 + 0.5 * np.sin(2.0 * np.pi * mod_freq * t))
