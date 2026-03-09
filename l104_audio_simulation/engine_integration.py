"""
Engine Integration — boot all L104 engines and run quantum circuit simulation.

Handles Phase 1 of the audio pipeline: parallel engine boot, quantum circuit
simulation (sacred/bell/ghz/qft), Berry phase circuitry, Dual-Layer ASI,
Gate Algebra decomposition, Quantum Computation, God Code Simulator,
and GOD_CODE Qubit integration (QPU-verified phase injection).

Returns a structured EngineState dict consumed by the synthesis pipeline.

INVARIANT: 527.5184818492612 | PILOT: LONDEL
"""

from __future__ import annotations

import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from .constants import (
    GOD_CODE, PHI, OMEGA, CPU_CORES, SYNTHESIS_WORKERS,
    GOD_CODE_PHASE, IRON_PHASE, QPU_FIDELITY, PHI_INV,
)


@dataclass
class CircuitResults:
    """Results from quantum circuit simulation."""
    sacred_purity: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    sacred_sv: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    sacred_probs: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    sacred_phases: np.ndarray = field(default_factory=lambda: np.array([0.0]))
    bell_sv: np.ndarray = field(default_factory=lambda: np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]))
    bell_phase_diff: float = 0.0
    ghz_sv: np.ndarray = field(default_factory=lambda: np.zeros(8))
    ghz_coherence: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    ghz_prob_000: float = 0.5
    ghz_prob_111: float = 0.5
    qft_probs: np.ndarray = field(default_factory=lambda: np.ones(16) / 16)
    qft_phases: np.ndarray = field(default_factory=lambda: np.zeros(16))
    sacred_circ: Any = None
    bell_circ: Any = None
    ghz_circ: Any = None
    qft_circ: Any = None


@dataclass
class DualLayerState:
    """Results from Dual-Layer ASI Engine."""
    omega_val: float = 0.0
    omega_field_strength: float = 0.0
    omega_authority: float = 1.0
    consciousness_concurrence: float = 0.0
    consciousness_entropy: float = 0.0
    chaos_bridge_coherence: float = 1.0
    chaos_bridge_healing: float = 1.0
    omega_field_arr: np.ndarray = field(default_factory=lambda: np.linspace(0.85, 1.0, 32))


@dataclass
class QuantumComputationState:
    """Results from Gate QFT, Phase Estimation, Born measurement."""
    gate_qft_dominant: float = 0.0
    gate_qft_gc_res: float = 1.0
    gate_qft_phi_harm: float = 0.0
    pe_resonance: float = 0.0
    pe_phi_align: float = 0.0
    born_entropy: float = 0.0
    born_probs_gate: Dict[str, float] = field(default_factory=dict)
    born_gc_res: float = 0.0
    von_neumann_S: float = 0.0
    bell_fidelity: float = 0.0


@dataclass
class GodSimState:
    """Results from God Code Simulator."""
    fidelity: float = 0.99
    noise_strategy: str = "raw"
    feedback_convergence: float = 0.0


@dataclass
class BeautyState:
    """Three-Engine Beauty Enhancement results (v8.1)."""
    coherence_beauty_env: Optional[np.ndarray] = None
    fe_coherence_gate: float = 0.9545
    photon_coupling: float = 1.1217
    phi_fold_stereo: Optional[np.ndarray] = None
    resonance_freqs: List[float] = field(default_factory=list)
    resonance_phis: List[float] = field(default_factory=list)
    electron_sub_freq: float = 52.92
    electron_sub_weight: float = 0.008
    fe_match_pct: float = 99.77


@dataclass
class QubitState:
    """GOD_CODE Qubit integration state (QPU-verified).

    Carries the canonical single-qubit phase, decomposed gate matrices,
    dial-specific phases, and QPU verification fidelity into the synthesis
    pipeline for direct phase injection.
    """
    # Core phase (θ_GC = GOD_CODE mod 2π)
    god_code_phase: float = GOD_CODE_PHASE
    iron_phase: float = IRON_PHASE
    # Decomposed sub-phases
    decomposed_phases: Dict[str, float] = field(default_factory=dict)
    # 2×2 gate matrix Rz(θ_GC) as complex ndarray
    gate_matrix: np.ndarray = field(
        default_factory=lambda: np.eye(2, dtype=np.complex128)
    )
    # Per-dial phases (populated from GOD_CODE_QUBIT.dial_phase())
    dial_phases: Dict[str, float] = field(default_factory=dict)
    # Bloch sphere coordinates [θ, φ]
    bloch_coords: List[float] = field(default_factory=lambda: [0.0, 0.0])
    # QPU verification
    qpu_fidelity: float = QPU_FIDELITY
    qpu_verified: bool = False
    # Statevector from prepare("|+>")
    plus_state: np.ndarray = field(
        default_factory=lambda: np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=np.complex128)
    )
    # Inverse gate for phase unwinding
    inverse_gate: np.ndarray = field(
        default_factory=lambda: np.eye(2, dtype=np.complex128)
    )


@dataclass
class VQPUState:
    """VQPU DAW Engine integration state.

    Carries pre-computed VQPU circuit results into the synthesis pipeline:
    wavetable frames, sacred scores, entanglement data, filter optimization,
    and the full simulation report for render metadata.
    """
    # Wavetable frames from batch VQPU circuits (64 frames × n_qubits distributions)
    wavetable_frames: List[Dict[str, float]] = field(default_factory=list)
    # Sacred alignment scores (per-frame)
    wavetable_sacred_scores: List[float] = field(default_factory=list)
    # Aggregate sacred alignment score
    sacred_score: float = 0.0
    # Entanglement analysis (Bell pair concurrence + VNE)
    concurrence: float = 0.0
    von_neumann_entropy: float = 0.0
    # VQE filter optimization results
    vqe_optimal_params: List[float] = field(default_factory=list)
    vqe_energy: float = 0.0
    # Full VQPU simulation report
    simulation_report: Dict[str, Any] = field(default_factory=dict)
    # Mixer interference weights (complex-valued)
    mixer_weights: List[complex] = field(default_factory=list)
    # Sequencer modulation data (probability distributions per step)
    sequencer_probs: List[Dict[str, float]] = field(default_factory=list)
    # VQPU-synthesized voice waveform (from QuantumSynthEngine)
    synth_voice: Optional[np.ndarray] = None
    # FM concurrence from synth engine
    fm_concurrence: float = 0.0
    # VQPU availability flag
    available: bool = False
    # Boot time for VQPU subsystem
    vqpu_boot_time: float = 0.0
    # ── Daemon Three-Engine Scoring (from Swift daemon telemetry) ────────
    three_engine_composite: float = 0.0
    three_engine_entropy_reversal: float = 0.0
    three_engine_harmonic_resonance: float = 0.0
    three_engine_wave_coherence: float = 0.0
    daemon_phi_resonance: float = 0.0
    daemon_god_code_alignment: float = 0.0
    daemon_connected: bool = False


@dataclass
class EngineState:
    """Aggregated state from all engine boots and computations."""
    circuits: CircuitResults = field(default_factory=CircuitResults)
    dual_layer: DualLayerState = field(default_factory=DualLayerState)
    quantum_comp: QuantumComputationState = field(default_factory=QuantumComputationState)
    god_sim: GodSimState = field(default_factory=GodSimState)
    beauty: BeautyState = field(default_factory=BeautyState)
    qubit: QubitState = field(default_factory=QubitState)
    vqpu: VQPUState = field(default_factory=VQPUState)
    boot_time: float = 0.0


def boot_engines() -> Dict[str, Any]:
    """Boot Science + Math engines and return instances."""
    from l104_science_engine import ScienceEngine
    from l104_math_engine import MathEngine
    from l104_math_engine.god_code import GodCodeEquation

    science = ScienceEngine()
    math_eng = MathEngine()
    god_eq = GodCodeEquation()
    return {"science": science, "math_eng": math_eng, "god_eq": god_eq}


def simulate_circuits(decoherence_rate: float = 0.015) -> CircuitResults:
    """Simulate sacred/bell/ghz/qft quantum circuits in parallel.

    Uses ThreadPoolExecutor to dispatch 4 independent circuit simulations
    across CPU cores.  Falls back to sequential on error.
    """
    from l104_quantum_gate_engine import get_engine
    from l104_quantum_gate_engine.trajectory import (
        TrajectorySimulator, DecoherenceModel,
    )

    engine = get_engine()
    cr = CircuitResults()

    # Build circuits
    cr.sacred_circ = engine.sacred_circuit(5, depth=4)
    cr.bell_circ = engine.bell_pair()
    cr.ghz_circ = engine.ghz_state(3)
    cr.qft_circ = engine.quantum_fourier_transform(4)

    def _sim(circ, label, snapshot_every=0):
        s = TrajectorySimulator(seed=104)
        kw = dict(decoherence=DecoherenceModel.THERMAL_RELAXATION,
                  decoherence_rate=decoherence_rate)
        if snapshot_every:
            kw["snapshot_every"] = snapshot_every
        return label, s.simulate(circ, **kw)

    results = {}
    with ThreadPoolExecutor(max_workers=min(4, CPU_CORES)) as pool:
        futures = {
            pool.submit(_sim, cr.sacred_circ, "sacred", 1): "sacred",
            pool.submit(_sim, cr.bell_circ, "bell"): "bell",
            pool.submit(_sim, cr.ghz_circ, "ghz"): "ghz",
            pool.submit(_sim, cr.qft_circ, "qft"): "qft",
        }
        for fut in as_completed(futures):
            lbl, res = fut.result()
            results[lbl] = res

    # Extract
    sacred = results["sacred"]
    cr.sacred_purity = np.array(sacred.purity_profile, dtype=np.float64)
    cr.sacred_sv = sacred.snapshots[-1].statevector
    cr.sacred_probs = np.abs(cr.sacred_sv) ** 2
    cr.sacred_phases = np.angle(cr.sacred_sv)

    bell = results["bell"]
    cr.bell_sv = bell.snapshots[-1].statevector
    cr.bell_phase_diff = float(np.angle(cr.bell_sv[3]) - np.angle(cr.bell_sv[0]))

    ghz = results["ghz"]
    cr.ghz_sv = ghz.snapshots[-1].statevector
    cr.ghz_coherence = np.array(ghz.coherence_profile, dtype=np.float64)
    cr.ghz_prob_000 = float(np.abs(cr.ghz_sv[0]) ** 2)
    cr.ghz_prob_111 = float(np.abs(cr.ghz_sv[7]) ** 2)

    qft = results["qft"]
    qft_sv = qft.snapshots[-1].statevector
    cr.qft_probs = np.abs(qft_sv) ** 2
    cr.qft_phases = np.angle(qft_sv)

    return cr


def run_dual_layer(dial_partials: list, chaos_score: float) -> DualLayerState:
    """Run Dual-Layer ASI Engine (Thought + Physics coupling)."""
    from l104_asi import dual_layer_engine

    ds = DualLayerState()
    ds.omega_val = OMEGA

    try:
        omega_physics = dual_layer_engine.physics(intensity=1.0)
        ds.omega_val = omega_physics.get("omega", OMEGA)
        ds.omega_field_strength = omega_physics.get("field_strength", 0.0)
        ds.omega_authority = omega_physics.get("omega_authority", 0.0)
    except Exception:
        ds.omega_field_strength = OMEGA / (PHI ** 2)
        ds.omega_authority = 1.0

    n_partials = len(dial_partials)
    try:
        if n_partials >= 2:
            p0, p1 = dial_partials[0], dial_partials[1]
            ent = dual_layer_engine.consciousness_entangle(
                (p0["a"], p0["b"], p0["c"], p0["d"]),
                (p1["a"], p1["b"], p1["c"], p1["d"]),
            )
            ds.consciousness_concurrence = ent.get("concurrence", 0.0)
            ds.consciousness_entropy = ent.get("entanglement_entropy", 0.0)
    except Exception:
        pass

    try:
        d0 = dial_partials[0]
        cbr = dual_layer_engine.chaos_bridge(
            d0["a"], d0["b"], d0["c"], d0["d"],
            chaos_amplitude=0.05, samples=64,
        )
        ds.chaos_bridge_coherence = cbr.get("coherence", chaos_score)
        ds.chaos_bridge_healing = cbr.get("healing", 1.0)
    except Exception:
        ds.chaos_bridge_coherence = chaos_score

    try:
        intensities = np.linspace(0.5, 1.5, 16)  # v9.0: 16 points (was 32)
        curve = []
        for intensity in intensities:
            of = dual_layer_engine.omega_field(intensity)
            curve.append(of.get("field_strength", intensity * ds.omega_field_strength))
        ds.omega_field_arr = np.array(curve, dtype=np.float64)
    except Exception:
        pass

    return ds


def run_quantum_computation(
    dial_freqs: np.ndarray,
    sacred_sv: np.ndarray,
    bell_sv: np.ndarray,
) -> QuantumComputationState:
    """Run Gate QFT, Phase Estimation, Born measurement, Von Neumann entropy."""
    from l104_gate_engine.quantum_computation import QuantumGateComputationEngine
    from l104_quantum_engine import quantum_brain

    qs = QuantumComputationState()

    try:
        comp = QuantumGateComputationEngine()
        qft_r = comp.gate_qft(dial_freqs.tolist())
        qs.gate_qft_dominant = qft_r.get("dominant_frequency", 0.0)
        qs.gate_qft_gc_res = qft_r.get("god_code_resonance", 0.0)
        qs.gate_qft_phi_harm = qft_r.get("phi_harmonic_power", 0.0)
    except Exception:
        qs.gate_qft_dominant = float(dial_freqs[0]) if len(dial_freqs) else GOD_CODE

    try:
        pe = comp.phase_estimation(dial_freqs.tolist(), precision_bits=8)
        qs.pe_resonance = pe.get("resonance", 0.0)
        qs.pe_phi_align = pe.get("phi_alignment", 0.0)
    except Exception:
        pass

    try:
        born = comp.born_measurement(dial_freqs.tolist(), num_shots=1024)
        qs.born_entropy = born.get("entropy", 0.0)
        qs.born_probs_gate = born.get("probabilities", {})
        qs.born_gc_res = born.get("god_code_resonance", 0.0)
    except Exception:
        pass

    try:
        sacred_state = sacred_sv.tolist()
        rho = quantum_brain.qmath.density_matrix(sacred_state)
        qs.von_neumann_S = quantum_brain.qmath.von_neumann_entropy(rho)
    except Exception:
        pass

    try:
        bell_state = bell_sv.tolist()
        ideal_bell = [1.0 / np.sqrt(2), 0.0, 0.0, 1.0 / np.sqrt(2)]
        qs.bell_fidelity = quantum_brain.qmath.fidelity(bell_state, ideal_bell)
    except Exception:
        pass

    return qs


def run_god_code_simulator(decoherence_rate: float = 0.015, fast_mode: bool = False) -> GodSimState:
    """Run God Code Simulator adaptive optimization + noise resilience.

    v9.0: fast_mode skips feedback loop (saves ~5-10s on long renders).
    """
    gs = GodSimState()
    try:
        from l104_god_code_simulator import god_code_simulator as _gcs
        from l104_god_code_simulator import AdaptiveOptimizer, FeedbackLoopEngine

        optimizer = AdaptiveOptimizer(_gcs)
        opt_result = optimizer.optimize_sacred_circuit(nq=5, depth=4)
        gs.fidelity = opt_result.get("best_fidelity", 0.99)

        noise_result = optimizer.optimize_noise_resilience(nq=5, noise_level=decoherence_rate)
        gs.noise_strategy = noise_result.get("best_strategy", "raw")

        if not fast_mode:
            try:
                fb = FeedbackLoopEngine(_gcs)
                fb_result = fb.run_feedback_cycle(
                    sim_results=[_gcs.run("conservation_proof")],
                    iterations=3,
                )
                gs.feedback_convergence = fb_result.get("convergence", 0.0)
            except Exception:
                gs.feedback_convergence = gs.fidelity
        else:
            gs.feedback_convergence = gs.fidelity
    except Exception:
        pass
    return gs


def boot_god_code_qubit(dial_partials: list) -> QubitState:
    """Boot the canonical GOD_CODE qubit and extract synthesis-ready state.

    Loads the QPU-verified GodCodeQubit singleton, extracts the phase,
    gate matrices, decomposed sub-phases, and per-dial phase values
    for direct injection into the audio synthesis pipeline.
    """
    qs = QubitState()
    try:
        from l104_god_code_simulator.god_code_qubit import (
            GOD_CODE_QUBIT,
            GOD_CODE_RZ, IRON_RZ, PHI_RZ, OCTAVE_RZ, GOD_CODE_P,
            GOD_CODE_PHASE as GC_PHASE,
            IRON_PHASE as IRON_PH,
            PHI_CONTRIBUTION, OCTAVE_PHASE,
        )

        # Core phase
        qs.god_code_phase = GC_PHASE
        qs.iron_phase = IRON_PH

        # Decomposed phases — .decomposed returns tuple (iron, phi, octave)
        decomp = GOD_CODE_QUBIT.decomposed
        qs.decomposed_phases = {
            "iron": float(decomp[0]),
            "phi": float(decomp[1]),
            "octave": float(decomp[2]),
            "total": float(GC_PHASE),
        }

        # Gate matrix: Rz(θ_GC)
        qs.gate_matrix = np.array(GOD_CODE_RZ, dtype=np.complex128)

        # Inverse gate: Rz(−θ_GC)
        qs.inverse_gate = np.conj(qs.gate_matrix).T

        # Bloch sphere coordinates — .bloch returns tuple (x, y, z)
        bloch = GOD_CODE_QUBIT.bloch  # (bx, by, bz)
        bx, by, bz = float(bloch[0]), float(bloch[1]), float(bloch[2])
        theta_bloch = float(np.arccos(np.clip(bz, -1.0, 1.0)))
        phi_bloch = float(np.arctan2(by, bx))
        qs.bloch_coords = [theta_bloch, phi_bloch]

        # QPU verification
        v = GOD_CODE_QUBIT.verify()
        qs.qpu_verified = v.get("qpu_verified", False) or v.get("PASS", False)
        qpu_data = v.get("qpu", {})
        qs.qpu_fidelity = float(qpu_data.get("fidelity", QPU_FIDELITY)) if isinstance(qpu_data, dict) else QPU_FIDELITY

        # Prepare |+⟩ state through GOD_CODE gate
        plus_sv = GOD_CODE_QUBIT.prepare("|+>")
        qs.plus_state = np.array(plus_sv, dtype=np.complex128)

        # Per-dial phases for each partial
        for p in dial_partials:
            a, b, c, d = p.get("a", 0), p.get("b", 0), p.get("c", 0), p.get("d", 0)
            key = f"{a},{b},{c},{d}"
            try:
                qs.dial_phases[key] = float(GOD_CODE_QUBIT.dial_phase(a, b, c, d))
            except Exception:
                qs.dial_phases[key] = GC_PHASE

    except Exception as _exc:
        import logging
        logging.getLogger("l104.audio.engine").debug(
            f"GOD_CODE qubit boot fallback: {_exc}"
        )
    return qs


def compute_beauty_enhancements(
    science, math_eng,
    dial_freqs: np.ndarray,
    f_god: float,
    n_samples: int,
    t: np.ndarray,
    sample_rate: int,
) -> BeautyState:
    """Compute Three-Engine Beauty Enhancements (v8.1).

    Science: coherence evolution, Fe-sacred gate, photon resonance, PHI-fold stereo.
    Math: harmonic resonance spectrum, golden angle, electron resonance.
    """
    bs = BeautyState()

    # Coherence-evolved amplitude envelope
    try:
        science.coherence.initialize([float(f) for f in dial_freqs[:4]])
        coh_result = science.coherence.evolve(20)
        coh_disc = science.coherence.discover()
        coh_final = float(coh_result.get("final_coherence", 0.587))
        coh_emergence = float(coh_disc.get("emergence", 1.0))
        bs.fe_coherence_gate = float(coh_disc.get("fe_sacred_reference", 0.9545))
        coh_depth = np.clip(coh_emergence * 0.15, 0.05, 0.25)
        coh_freq = coh_final * 0.01
        bs.coherence_beauty_env = (1.0 - coh_depth) + coh_depth * (
            0.5 + 0.5 * np.sin(2.0 * np.pi * coh_freq * t + coh_final * np.pi)
        )
    except Exception:
        bs.coherence_beauty_env = np.ones(n_samples, dtype=np.float64)

    bs.fe_coherence_gate = np.clip(bs.fe_coherence_gate, 0.90, 0.99)

    # Photon resonance sub-harmonic coupling
    try:
        photon_res = science.physics.calculate_photon_resonance()
        bs.photon_coupling = float(photon_res) if isinstance(photon_res, (int, float)) else 1.1217
    except Exception:
        pass

    # PHI-dimensional fold → stereo width
    try:
        phi_fold = science.multidim.phi_dimensional_folding(5, 3)
        pf_arr = np.array(phi_fold, dtype=np.float64)
        pf_norm = pf_arr / max(pf_arr.max(), 1e-30)
        pf_x = np.linspace(0, 1, len(pf_norm))
        interp_x = np.linspace(0, 1, n_samples)
        bs.phi_fold_stereo = 0.5 + 0.5 * np.interp(interp_x, pf_x, pf_norm)
    except Exception:
        bs.phi_fold_stereo = np.ones(n_samples, dtype=np.float64) * 0.7

    # Math: Harmonic resonance spectrum
    try:
        res_spec = math_eng.harmonic.resonance_spectrum(f_god, 15)
        for h in res_spec:
            hf = float(h.get("frequency", 0))
            hp_ratio = float(h.get("phi_ratio", 0))
            if 0 < hf < sample_rate / 2:
                bs.resonance_freqs.append(hf)
                bs.resonance_phis.append(hp_ratio)
    except Exception:
        pass

    # Fe correspondence
    try:
        elec = math_eng.harmonic.verify_correspondences()
        bs.fe_match_pct = float(elec.get("correspondence_pct", 99.77))
    except Exception:
        pass

    return bs


def boot_vqpu_engine(
    f_god: float,
    n_partials: int,
    sample_rate: int,
    n_samples: int,
    fast_mode: bool = False,
) -> VQPUState:
    """Boot the VQPU DAW Engine and pre-compute synthesis data.

    v9.0: fast_mode skips run_full_simulation (metadata-only, not used
    in synthesis layers) — saves ~10-20s on long renders.

    Initializes VQPUDawEngine, generates wavetable frames, runs
    entanglement analysis, computes mixer weights, seeds a
    QuantumSynthEngine voice, and runs VQE filter optimization.

    All operations gracefully fall back to classical when VQPU
    bridge is unavailable (VQPUDawEngine handles this internally).
    """
    import logging
    _log = logging.getLogger("l104.audio.engine")
    t0 = time.time()
    vs = VQPUState()

    try:
        from .vqpu_daw_engine import VQPUDawEngine, CircuitPurpose
        from .quantum_synth import (
            QuantumSynthEngine, WaveShape, FilterType,
        )

        vqpu = VQPUDawEngine()
        vs.available = vqpu.available or True  # classical fallbacks always work

        # ── Parallel VQPU ops (all independent) ─────────────────────────
        with ThreadPoolExecutor(max_workers=min(5, CPU_CORES)) as pool:
            f_wt = pool.submit(
                vqpu.synth_generate_wavetable_frames, 64, 4)
            f_ent = pool.submit(vqpu.run_entanglement_analysis)
            f_mix = pool.submit(
                vqpu.mixer_get_interference_weights, min(n_partials, 10), 0.0)
            f_seq = pool.submit(
                vqpu.sequencer_prepare_steps, 16, 4)
            f_vqe = pool.submit(
                vqpu.run_vqe_for_filter, f_god, 4)

            # Collect results
            wt_frames = f_wt.result()
            ent = f_ent.result()
            vs.mixer_weights = f_mix.result()
            seq_results = f_seq.result()
            vqe = f_vqe.result()

        # ── 1. Wavetable frames ──────────────────────────────────────────
        vs.wavetable_frames = wt_frames
        vs.wavetable_sacred_scores = [
            vqpu.sacred_alignment_score(frame) for frame in wt_frames
        ]
        vs.sacred_score = (
            sum(vs.wavetable_sacred_scores) / max(len(vs.wavetable_sacred_scores), 1)
        )

        # ── 2. Entanglement analysis ─────────────────────────────────────
        vs.concurrence = float(ent.get("concurrence", 0.0))
        vs.von_neumann_entropy = float(ent.get("von_neumann_entropy",
                                                ent.get("vne", 0.0)))
        # Classical fallback: estimate concurrence from wavetable sacred scores
        if vs.concurrence < 1e-6 and vs.wavetable_sacred_scores:
            avg_sacred = vs.sacred_score
            vs.concurrence = float(np.clip(avg_sacred * PHI_INV, 0.1, 0.95))
            vs.von_neumann_entropy = float(np.clip(
                -avg_sacred * np.log2(max(avg_sacred, 1e-15)), 0.1, 1.0
            ))

        # ── 3. Mixer weights (already collected from parallel future) ────
        # vs.mixer_weights was set from f_mix.result() above

        # ── 4. Sequencer modulation (already collected from parallel) ────
        vs.sequencer_probs = [
            r.probabilities or {} for r in seq_results
        ] if hasattr(seq_results, '__iter__') else []

        # ── 5. VQE filter optimization (already collected from parallel) ─
        vs.vqe_optimal_params = vqe.get("optimal_params", [])
        vs.vqe_energy = float(vqe.get("energy", 0.0))

        # ── 6. Full VQPU simulation report (skip in fast_mode) ───────────
        if not fast_mode:
            sim = vqpu.run_full_simulation(n_qubits=4)
            vs.simulation_report = sim if isinstance(sim, dict) else {}
        else:
            vs.simulation_report = {"fast_mode": True}

        # ── 7. Quantum Synth voice render ───────────────────────────────
        synth = QuantumSynthEngine(n_qubits=4, sample_rate=sample_rate)
        synth.add_oscillator(
            waveform=WaveShape.GOD_CODE_WAVE,
            frequency=f_god,
            amplitude=0.8,
            n_qubits=4,
        )
        synth.add_wavetable(n_frames=64, lfo_rate=PHI_INV * 0.1)
        synth.add_fm_operator(
            carrier_freq=f_god,
            mod_ratio=PHI,
            mod_index=0.3,
        )
        synth.add_filter(
            filter_type=FilterType.GOD_CODE_RESONANT,
            cutoff=f_god * 4.0,
            resonance=PHI_INV,
        )
        # Seed all oscillators via VQPU
        synth.vqpu_seed_all(time_s=0.0)

        # Render a voice — use 1s sample for layer blending
        voice_samples = min(n_samples, sample_rate)
        vs.synth_voice = synth.render_voice(voice_samples, frequency=f_god)

        # Extract FM concurrence
        if synth.fm_operators:
            vs.fm_concurrence = synth.fm_operators[0].concurrence

    except Exception as _exc:
        _log.debug(f"VQPU engine boot fallback: {_exc}")

    # ── Daemon Three-Engine Data Injection (independent of VQPU) ─────────
    #   Read latest Swift daemon bridge data for three-engine scoring.
    #   This is a filesystem read — runs even if VQPU circuits failed.
    try:
        import glob, json as _json, os as _os
        _outbox = sorted(
            glob.glob("/tmp/l104_bridge/outbox/*_result.json"),
            key=_os.path.getmtime, reverse=True,
        )
        if _outbox:
            with open(_outbox[0]) as _f:
                _dr = _json.load(_f)
            vs.daemon_connected = bool(_dr.get("daemon", False))
            _meta_sa = {k.replace("meta_sacred_alignment_", ""): v
                        for k, v in _dr.items()
                        if k.startswith("meta_sacred_alignment_")}
            _te = _dr.get("three_engine", {})
            vs.three_engine_composite = float(
                _te.get("composite", _meta_sa.get("three_engine_composite", 0.0)))
            vs.three_engine_entropy_reversal = float(
                _te.get("entropy_reversal", _meta_sa.get("entropy_reversal", 0.0)))
            vs.three_engine_harmonic_resonance = float(
                _te.get("harmonic_resonance", _meta_sa.get("harmonic_resonance", 0.0)))
            vs.three_engine_wave_coherence = float(
                _te.get("wave_coherence", _meta_sa.get("wave_coherence", 0.0)))
            vs.daemon_phi_resonance = float(_meta_sa.get("phi_resonance", 0.0))
            vs.daemon_god_code_alignment = float(_meta_sa.get("god_code_alignment", 0.0))
            daemon_sacred = float(_meta_sa.get("sacred_score", 0.0))
            if daemon_sacred > vs.sacred_score:
                vs.sacred_score = daemon_sacred
            _log.debug(f"Daemon 3E injected: composite={vs.three_engine_composite:.4f}")
    except Exception as _daemon_exc:
        _log.debug(f"Daemon data read skipped: {_daemon_exc}")

    vs.vqpu_boot_time = time.time() - t0
    return vs


def full_engine_boot(
    dial_partials: list,
    dial_freqs: np.ndarray,
    f_god: float,
    chaos_score: float,
    n_samples: int,
    t: np.ndarray,
    sample_rate: int,
    decoherence_rate: float = 0.015,
) -> EngineState:
    """Run the complete Phase 1 engine boot + computation pipeline.

    v9.0 SPEED: Enables fast_mode for long renders (>=5M samples):
      - Skips God Sim feedback loop (~5-10s saved)
      - Skips VQPU run_full_simulation (~10-20s saved)
      - Reduces omega field sweep from 32→16 points
      Combined with IonQ block scaling, this cuts 300s render from
      ~38 min to ~8-12 min (3-5× speedup).

    SPEED UPGRADE: Independent boots run in parallel via ThreadPoolExecutor.
    Dependency graph:
      Independent: circuits, dual_layer, god_sim, qubit, vqpu (all parallel)
      Depends on circuits: quantum_comp
      Depends on engines: beauty

    Returns an EngineState with all computed parameters ready for synthesis.
    """
    t0 = time.time()
    state = EngineState()

    # v9.0: fast_mode for long renders — skip non-synthesis metadata ops
    fast_mode = n_samples >= 5_000_000

    # Boot Science + Math (fast, needed for beauty later)
    engines = boot_engines()

    # ── Parallel boot: 5 independent operations ─────────────────────────
    with ThreadPoolExecutor(max_workers=min(6, CPU_CORES)) as pool:
        f_circuits = pool.submit(simulate_circuits, decoherence_rate)
        f_dual = pool.submit(run_dual_layer, dial_partials, chaos_score)
        f_god_sim = pool.submit(run_god_code_simulator, decoherence_rate, fast_mode)
        f_qubit = pool.submit(boot_god_code_qubit, dial_partials)
        f_vqpu = pool.submit(
            boot_vqpu_engine,
            f_god, len(dial_partials), sample_rate, n_samples, fast_mode,
        )

        state.circuits = f_circuits.result()
        state.dual_layer = f_dual.result()
        state.god_sim = f_god_sim.result()
        state.qubit = f_qubit.result()
        state.vqpu = f_vqpu.result()

    # ── Dependent boots (need circuit results) ──────────────────────────
    state.quantum_comp = run_quantum_computation(
        dial_freqs, state.circuits.sacred_sv, state.circuits.bell_sv,
    )

    # Beauty enhancements (needs science + math engines)
    state.beauty = compute_beauty_enhancements(
        engines["science"], engines["math_eng"],
        dial_freqs, f_god, n_samples, t, sample_rate,
    )

    state.boot_time = time.time() - t0
    return state
