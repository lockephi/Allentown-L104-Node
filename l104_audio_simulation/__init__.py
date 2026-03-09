"""
L104 Audio Simulation Suite — Decomposed Package v3.0.0
═══════════════════════════════════════════════════════════════════════════════

Refactored from the monolithic _gen_perfect_audio.py (v8.2, 3,231 lines),
_gen_decoherence_audio.py (145 lines), and l104_quantum_audio.py (605 lines)
into a clean, modular package with focused submodules.

v2.3.0: SPEED UNLIMITED + DAEMON THREE-ENGINE INJECTION
  - 17-layer pipeline (was 16): new Layer 17 daemon φ-harmonic reinforcement
  - Swift daemon three-engine data (sacred=0.749, composite=0.891) injected
  - Multi-core synthesis at 500K samples (was 4M) — 8× more renders go parallel
  - np.einsum fused weighted-sin for SIMD/BLAS acceleration
  - Duplicate VQPU boot calls eliminated (saves ~1s per boot)
  - Daemon wave_coherence + harmonic_resonance boost mixer / stereo accuracy
  - VQPUState now carries 7 daemon three-engine fields

Modules:
  constants.py            — Sacred audio constants, frequencies, intervals
  god_code_equation.py    — G(a,b,c,d) 4-dial system, partial table builders
  metal_gpu.py            — Apple Metal GPU acceleration (additive synthesis, envelopes)
  ionq_stabilizer.py      — IonQ trapped-ion stabilizers (171Yb+ bulk, 138Ba+ syndrome)
  synthesis.py            — Vectorized additive synthesis, multi-core dispatch
  envelopes.py            — Envelope generators (fade, purity, sovereign, primal, coherence)
  quantum_modulation.py   — Berry phase, AA phase, braid injection, gate algebra modulation
  engine_integration.py   — Science/Math/Dual-Layer/God-Sim engine boot + param extraction
  tone_generator.py       — QuantumPureToneGenerator class (sacred/quantum tones)
  decoherence.py          — Decoherence audio generator (RealisticNoiseEngine modulation)
  pipeline.py             — 17-layer VQPU-DAEMON synthesis pipeline orchestrator
  cli.py                  — CLI argument parsing + main entry point

  === Quantum DAW Modules (v2.0.0) ===
  quantum_sequencer.py    — Probabilistic superposition sequencer (pattern grid)
  quantum_mixer.py        — Interference-based mixing console (6 modes)
  quantum_synth.py        — Quantum synthesizer engines (oscillator + wavetable + FM + filter)
  track_entanglement.py   — Track-to-track quantum correlations (Bell/GHZ/W/spectral)
  vqpu_daw_engine.py      — VQPU circuit execution backbone for all DAW operations
  data_recorder.py        — Telemetry + ML training data collection
  daw_session.py          — Top-level orchestrator + ASI/AGI pipeline integration

Quick-start:
    from l104_audio_simulation import audio_suite

    # Run full 16-layer VQPU-linked pipeline (standard 3-file mode)
    audio_suite.generate_standard(duration=300.0, sample_rate=180000)

    # Quantum DAW session
    from l104_audio_simulation import quantum_daw
    tk = quantum_daw.add_track("BassLine", preset="god_code_wave")
    pat = quantum_daw.create_pattern(n_steps=16, n_qubits=4, bpm=120.0)
    quantum_daw.assign_pattern(tk, pat)
    audio = quantum_daw.render(duration=30.0)
    quantum_daw.export_wav("output.wav", audio)

    # Connect to ASI/AGI pipelines
    quantum_daw.connect_pipelines()

INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

__version__ = "3.0.0"

# ── Constants ────────────────────────────────────────────────────────────────
from .constants import (
    GOD_CODE, PHI, PHI_INV, VOID_CONSTANT,
    ZENITH_HZ, SCHUMANN_HZ, SACRED_INTERVALS,
    GOD_CODE_PHASE, IRON_PHASE, QPU_FIDELITY,
    DAEMON_WEIGHT_ENTROPY, DAEMON_WEIGHT_HARMONIC, DAEMON_WEIGHT_WAVE,
    DAEMON_SACRED_BASELINE, DAEMON_THREE_ENGINE_BASELINE,
)

# ── Tone Generator ──────────────────────────────────────────────────────────
from .tone_generator import (
    QuantumPureToneGenerator,
    QuantumTone,
    QuantumAudioResult,
    quantum_tone_generator,
)

# ── Core synthesis tools ────────────────────────────────────────────────────
from .metal_gpu import metal_additive_synthesis, metal_envelope_multiply
from .ionq_stabilizer import (
    ionq_stabilizer,
    ionq_signal_stabilizer,
    ionq_syndrome_extraction,
    ionq_ba_synthesize_partials,
    ionq_ba_synthesize_single,
)
from .synthesis import vectorized_additive, process_chunk
from .envelopes import make_fade_envelope, normalize_signal, write_wav
from .god_code_equation import evaluate_hp, build_dial_table, build_dial_partials

# ── Quantum modulation ──────────────────────────────────────────────────────
from .quantum_modulation import (
    BerryPhaseParams,
    AAPhaseParams,
    BraidParams,
    GateAlgebraParams,
    compute_berry_phase_params,
    compute_aa_phase_params,
    compute_braid_params,
    compute_gate_algebra_params,
)

# ── Engine integration ──────────────────────────────────────────────────────
from .engine_integration import EngineState, QubitState, VQPUState, full_engine_boot

# ── Decoherence ─────────────────────────────────────────────────────────────
from .decoherence import generate_decoherence_audio

# ── CLI ─────────────────────────────────────────────────────────────────────
from .cli import main as cli_main, build_parser

# ── Pipeline orchestrator ───────────────────────────────────────────────────
from .pipeline import AudioSimulationPipeline

# ── Quantum DAW Modules (v2.0.0) ───────────────────────────────────────────
from .quantum_sequencer import (
    ProbabilisticSequencer,
    QuantumPattern,
    SuperpositionStep,
    NoteState,
    CollapseMode,
    NoteEvent,
)
from .quantum_mixer import (
    QuantumInterferenceMixer,
    MixTrack,
    InterferenceMode,
    QuantumFader,
    QuantumPan,
)
from .quantum_synth import (
    QuantumSynthEngine,
    QuantumOscillator,
    SuperpositionWavetable,
    EntanglementFM,
    InterferenceFilter,
    WaveShape,
    FilterType,
)
from .track_entanglement import (
    TrackEntanglementManager,
    EntanglementBond,
    EntanglementType,
)
from .vqpu_daw_engine import (
    VQPUDawEngine,
    CircuitPurpose,
    VQPUCircuitRequest,
    VQPUBatchResult,
)
from .data_recorder import (
    DataRecorder,
    DAWEvent,
    EventCategory,
    EventSeverity,
    FeatureExtractor,
)
from .daw_session import (
    QuantumDAWSession,
    DAWTrack,
    DAWArrangement,
)
# ── Cross-Engine Integration (v3.0) ──────────────────────────────────────────
from .cross_engine import (
    AudioMLBridge,
    AudioQDABridge,
    AudioScienceBridge,
    AudioCrossEngineHub,
    audio_cross_engine_hub,
)
# ── Module-level orchestrator singletons ────────────────────────────────────
audio_suite = AudioSimulationPipeline()

from .daw_session import quantum_daw  # noqa: E402 — DAW singleton
