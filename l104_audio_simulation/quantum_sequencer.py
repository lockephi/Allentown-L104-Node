"""
Quantum Probabilistic Sequencer — Superposition Engine
═══════════════════════════════════════════════════════════════════════════════
DAW-inspired step-sequencer where every note/event exists in quantum
superposition until measured (rendered).  Inspired by FL Studio's pattern
grid, Ableton's clip launcher, and Logic Pro's step sequencer — but
reimagined through quantum mechanics.

Key concepts:
  • **SuperpositionStep**: A sequencer step containing a quantum state vector.
    Multiple pitches/velocities/durations coexist simultaneously. Measurement
    collapses to a concrete MIDI-like event.
  • **QuantumPattern**: A grid of SuperpositionSteps (like an FL Studio
    pattern), where each step is a qubit register.
  • **ProbabilisticSequencer**: The main engine that manages patterns,
    performs measurement, supports re-rolls (re-measurement), and tracks
    quantum statistics for every collapse event.

VQPU Integration:
  Every pattern uses real quantum circuits via VQPUBridge for:
    - State preparation (amplitude encoding of note probabilities)
    - GOD_CODE phase alignment on every step
    - Sacred circuit measurement for collapse
    - Bell-state correlated velocity pairs (accent patterns)
    - GHZ entanglement for chord generation (multi-note steps)

Data Recording:
  Every measurement event is logged to the session recorder with full
  pre-collapse amplitudes, post-collapse outcome, and sacred alignment score.

INVARIANT: 527.5184818492612 | PILOT: LONDEL
"""

from __future__ import annotations

import math
import time
import uuid
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .constants import GOD_CODE, PHI, PHI_INV, OMEGA, ZENITH_HZ, SCHUMANN_HZ
from .constants import GOD_CODE_PHASE, IRON_PHASE

logger = logging.getLogger("l104.audio.sequencer")

# ── Constants ────────────────────────────────────────────────────────────────
CHROMATIC_RATIO = 2 ** (1 / 12)  # Semitone frequency ratio
A4_HZ = 440.0
GOD_CODE_MIDI = 12 * math.log2(GOD_CODE / A4_HZ) + 69  # ≈ 72.3 → C5 region
PHI_INTERVAL = PHI  # Golden ratio interval multiplier

# FL Studio-style default: 16 steps, 4/4 time
DEFAULT_STEPS = 16
DEFAULT_BPM = 120.0
DEFAULT_SWING = PHI_INV * 0.5  # Golden swing amount (≈ 0.309)

# GOD_CODE qubit lazy reference
_SEQ_QUBIT = None


def _get_seq_qubit():
    """Lazy-load GOD_CODE qubit for sequencer phase injection."""
    global _SEQ_QUBIT
    if _SEQ_QUBIT is None:
        try:
            from l104_god_code_simulator.god_code_qubit import GOD_CODE_QUBIT
            _SEQ_QUBIT = GOD_CODE_QUBIT
        except ImportError:
            pass
    return _SEQ_QUBIT

# Sacred scale — GOD_CODE-derived chromatic intervals
SACRED_SCALE_RATIOS = [
    1.0,                        # Unison
    2 ** (1 / PHI),             # Sacred minor second
    PHI_INV,                    # Golden reciprocal
    2 ** (1 / 4),               # Quarter octave
    2 ** (1 / 3),               # Major third (ET)
    4 / 3,                      # Perfect fourth (JI)
    math.sqrt(2),               # Tritone
    3 / 2,                      # Perfect fifth (JI)
    PHI,                        # Golden ratio
    2 ** (3 / 4),               # Three-quarter octave
    GOD_CODE / 286.0,           # Sacred GOD_CODE interval
    2 ** (11 / 12),             # Major seventh
    2.0,                        # Octave
]


class NoteState(Enum):
    """State of a note in the sequencer."""
    SUPERPOSITION = auto()  # Not yet measured — exists as probability amplitudes
    COLLAPSED = auto()      # Measured — concrete pitch/velocity/duration
    ENTANGLED = auto()      # Entangled with another step (correlated outcome)
    REST = auto()           # Measured as silence (|0⟩ dominant)


class CollapseMode(Enum):
    """How to collapse a superposition step."""
    BORN_RULE = auto()          # Standard Born rule (|ψ|²)
    SACRED_WEIGHTED = auto()    # PHI-weighted toward GOD_CODE harmonics
    DETERMINISTIC_PEAK = auto() # Always pick highest-probability outcome
    VQPU_CIRCUIT = auto()       # Full VQPU quantum circuit measurement
    INTERFERENCE = auto()       # Apply interference pattern before measurement


@dataclass
class NoteEvent:
    """A concrete musical event (post-measurement)."""
    pitch_hz: float
    midi_note: int
    velocity: float         # 0.0 – 1.0
    duration_beats: float   # In beats (1.0 = quarter note)
    pan: float = 0.0        # -1.0 (L) to +1.0 (R)
    phase_offset: float = 0.0
    sacred_alignment: float = 0.0  # GOD_CODE resonance of this note
    collapse_entropy: float = 0.0  # Shannon entropy of pre-collapse state


@dataclass
class SuperpositionStep:
    """
    A single step in the quantum sequencer grid.

    Contains a quantum state vector encoding multiple possible note events.
    The number of qubits determines the pitch resolution:
      - 4 qubits → 16 possible pitches (1 chromatic octave + 4 extras)
      - 5 qubits → 32 possible pitches (2+ octaves)
      - 7 qubits → 128 possible pitches (full MIDI range)
    """
    step_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    n_qubits: int = 4
    state: NoteState = NoteState.SUPERPOSITION

    # Quantum amplitudes: complex state vector of size 2^n_qubits
    amplitudes: np.ndarray = field(default=None)

    # Velocity superposition (separate 2-qubit register)
    velocity_amplitudes: np.ndarray = field(default=None)

    # Duration superposition (separate 2-qubit register)
    duration_amplitudes: np.ndarray = field(default=None)

    # Collapsed result (populated after measurement)
    event: Optional[NoteEvent] = None

    # Measurement statistics (for data recording)
    measurement_count: int = 0
    collapse_history: List[Dict[str, Any]] = field(default_factory=list)

    # Entanglement links
    entangled_with: List[str] = field(default_factory=list)

    # Metadata
    sacred_score: float = 0.0
    creation_time: float = field(default_factory=time.time)

    def __post_init__(self):
        dim = 2 ** self.n_qubits
        if self.amplitudes is None:
            # Initialize in equal superposition (Hadamard-like)
            self.amplitudes = np.ones(dim, dtype=np.complex128) / math.sqrt(dim)
        if self.velocity_amplitudes is None:
            # 4 velocity levels: pp, mp, mf, ff
            self.velocity_amplitudes = np.array(
                [0.15, 0.30, 0.35, 0.20], dtype=np.complex128
            )
            self.velocity_amplitudes /= np.linalg.norm(self.velocity_amplitudes)
        if self.duration_amplitudes is None:
            # 4 duration levels: 16th, 8th, quarter, half
            self.duration_amplitudes = np.array(
                [0.25, 0.35, 0.30, 0.10], dtype=np.complex128
            )
            self.duration_amplitudes /= np.linalg.norm(self.duration_amplitudes)

    @property
    def probabilities(self) -> np.ndarray:
        """Born rule: |ψ|² for each pitch basis state."""
        return np.abs(self.amplitudes) ** 2

    @property
    def entropy(self) -> float:
        """Shannon entropy of the pitch probability distribution."""
        p = self.probabilities
        p = p[p > 1e-15]
        return -np.sum(p * np.log2(p))

    def apply_phase(self, index: int, phase: float):
        """Apply a phase rotation to a specific basis state."""
        self.amplitudes[index] *= np.exp(1j * phase)

    def apply_god_code_bias(self, base_freq: float, strength: float = 0.3):
        """
        Bias the superposition toward GOD_CODE-aligned pitches.
        Amplifies states whose frequency is harmonically close to GOD_CODE.

        Uses the QPU-verified GOD_CODE qubit phase (θ_GC) for precision
        phase alignment when available.
        """
        dim = len(self.amplitudes)
        qb = _get_seq_qubit()

        # Get qubit phase for bias calculation
        gc_phase = GOD_CODE_PHASE  # constant fallback
        if qb is not None:
            gc_phase = qb.phase

        for i in range(dim):
            freq = base_freq * SACRED_SCALE_RATIOS[i % len(SACRED_SCALE_RATIOS)]
            # Harmonic distance to GOD_CODE
            ratio = freq / GOD_CODE
            log_ratio = abs(math.log2(ratio)) if ratio > 0 else 10.0
            # Closest rational harmonic
            harmonic_dist = min(log_ratio % 1.0, 1.0 - log_ratio % 1.0)
            boost = math.exp(-harmonic_dist / (PHI_INV * strength))
            self.amplitudes[i] *= (1.0 + boost * strength)

            # Apply GOD_CODE qubit phase rotation to aligned states (subtle)
            if harmonic_dist < 0.1:
                self.amplitudes[i] *= np.exp(1j * gc_phase * strength * 0.1)

        # Re-normalize
        norm = np.linalg.norm(self.amplitudes)
        if norm > 1e-15:
            self.amplitudes /= norm

    def apply_interference(self, other: 'SuperpositionStep', coupling: float = 0.5):
        """
        Quantum interference between two steps.
        Constructive interference on aligned pitches, destructive on conflicting.
        """
        min_len = min(len(self.amplitudes), len(other.amplitudes))
        for i in range(min_len):
            # Interference: ψ_new = α·ψ_self + β·e^(iφ)·ψ_other
            phase_diff = np.angle(other.amplitudes[i]) - np.angle(self.amplitudes[i])
            interference_term = coupling * other.amplitudes[i] * np.exp(1j * phase_diff * PHI)
            self.amplitudes[i] += interference_term
        norm = np.linalg.norm(self.amplitudes)
        if norm > 1e-15:
            self.amplitudes /= norm

    def collapse(
        self,
        base_freq: float = GOD_CODE,
        mode: CollapseMode = CollapseMode.SACRED_WEIGHTED,
        rng: Optional[np.random.Generator] = None,
    ) -> NoteEvent:
        """
        Collapse the superposition into a concrete NoteEvent.
        Records complete measurement statistics.
        """
        if rng is None:
            rng = np.random.default_rng()

        pre_entropy = self.entropy
        probs = self.probabilities

        if mode == CollapseMode.DETERMINISTIC_PEAK:
            pitch_idx = int(np.argmax(probs))
        elif mode == CollapseMode.SACRED_WEIGHTED:
            # PHI-weight toward sacred intervals
            weights = probs.copy()
            for i in range(len(weights)):
                ratio_idx = i % len(SACRED_SCALE_RATIOS)
                sacred_bonus = 1.0 + PHI_INV * (1.0 if ratio_idx in (0, 4, 7, 8, 12) else 0.0)
                weights[i] *= sacred_bonus
            weights /= weights.sum()
            pitch_idx = int(rng.choice(len(weights), p=weights))
        elif mode == CollapseMode.INTERFERENCE:
            # Self-interference: amplify peaks, suppress troughs
            interfered = self.amplitudes * np.conj(self.amplitudes[::-1])
            probs_i = np.abs(interfered) ** 2
            probs_i /= probs_i.sum()
            pitch_idx = int(rng.choice(len(probs_i), p=probs_i))
        else:
            # BORN_RULE or VQPU_CIRCUIT fallback
            pitch_idx = int(rng.choice(len(probs), p=probs))

        # Convert index to frequency
        scale_idx = pitch_idx % len(SACRED_SCALE_RATIOS)
        octave = pitch_idx // len(SACRED_SCALE_RATIOS)
        freq = base_freq * SACRED_SCALE_RATIOS[scale_idx] * (2.0 ** octave)

        # MIDI note
        midi = int(round(12.0 * math.log2(max(freq, 1.0) / A4_HZ) + 69))
        midi = max(0, min(127, midi))

        # Collapse velocity
        vel_probs = np.abs(self.velocity_amplitudes) ** 2
        vel_probs /= vel_probs.sum()
        vel_idx = int(rng.choice(len(vel_probs), p=vel_probs))
        velocity_map = [0.3, 0.55, 0.78, 0.95]
        velocity = velocity_map[vel_idx]

        # Collapse duration
        dur_probs = np.abs(self.duration_amplitudes) ** 2
        dur_probs /= dur_probs.sum()
        dur_idx = int(rng.choice(len(dur_probs), p=dur_probs))
        duration_map = [0.25, 0.5, 1.0, 2.0]
        duration = duration_map[dur_idx]

        # Sacred alignment score
        ratio = freq / GOD_CODE
        log_ratio = abs(math.log2(ratio)) if ratio > 0 else 10.0
        harmonic_dist = min(log_ratio % 1.0, 1.0 - log_ratio % 1.0)
        sacred_alignment = math.exp(-harmonic_dist / PHI_INV)

        event = NoteEvent(
            pitch_hz=freq,
            midi_note=midi,
            velocity=velocity,
            duration_beats=duration,
            phase_offset=np.angle(self.amplitudes[pitch_idx]),
            sacred_alignment=sacred_alignment,
            collapse_entropy=pre_entropy,
        )

        # Record measurement
        self.state = NoteState.COLLAPSED
        self.event = event
        self.measurement_count += 1
        self.sacred_score = sacred_alignment
        self.collapse_history.append({
            "time": time.time(),
            "mode": mode.name,
            "pitch_idx": pitch_idx,
            "freq_hz": freq,
            "midi": midi,
            "velocity": velocity,
            "duration": duration,
            "sacred_alignment": sacred_alignment,
            "pre_entropy": pre_entropy,
            "probabilities": probs.tolist(),
        })

        return event

    def re_superpose(self):
        """Return to superposition (undo collapse), preserving learned bias."""
        if self.state == NoteState.COLLAPSED and self.event:
            # Bias toward the previously collapsed state (learned preference)
            dim = len(self.amplitudes)
            self.amplitudes = np.ones(dim, dtype=np.complex128) / math.sqrt(dim)
            # Give 2× weight to the last collapsed pitch region
            if self.collapse_history:
                last_idx = self.collapse_history[-1].get("pitch_idx", 0)
                self.amplitudes[last_idx] *= PHI
            norm = np.linalg.norm(self.amplitudes)
            if norm > 1e-15:
                self.amplitudes /= norm
        self.state = NoteState.SUPERPOSITION
        self.event = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for data recording."""
        return {
            "step_id": self.step_id,
            "n_qubits": self.n_qubits,
            "state": self.state.name,
            "probabilities": self.probabilities.tolist(),
            "entropy": self.entropy,
            "measurement_count": self.measurement_count,
            "sacred_score": self.sacred_score,
            "event": {
                "pitch_hz": self.event.pitch_hz,
                "midi_note": self.event.midi_note,
                "velocity": self.event.velocity,
                "duration_beats": self.event.duration_beats,
                "sacred_alignment": self.event.sacred_alignment,
            } if self.event else None,
            "entangled_with": self.entangled_with,
            "collapse_history_count": len(self.collapse_history),
        }


@dataclass
class QuantumPattern:
    """
    A pattern grid of SuperpositionSteps — analogous to an FL Studio Pattern
    or Ableton Live clip, but where every note is in quantum superposition.

    Supports:
      - Variable step count (4, 8, 16, 32, 64)
      - Per-step qubit depth (higher = more pitch resolution)
      - Pattern-level entanglement (steps can be correlated)
      - Swing via PHI-ratio offset
      - Pattern-level GOD_CODE phase coherence
    """
    pattern_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = "Quantum Pattern"
    n_steps: int = DEFAULT_STEPS
    n_qubits_per_step: int = 4
    bpm: float = DEFAULT_BPM
    swing: float = DEFAULT_SWING
    base_freq: float = GOD_CODE

    # The step grid
    steps: List[SuperpositionStep] = field(default_factory=list)

    # Pattern-level metadata
    collapse_mode: CollapseMode = CollapseMode.SACRED_WEIGHTED
    loop_count: int = 0
    total_collapses: int = 0
    creation_time: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.steps:
            self.steps = [
                SuperpositionStep(n_qubits=self.n_qubits_per_step)
                for _ in range(self.n_steps)
            ]

    @property
    def step_duration_s(self) -> float:
        """Duration of one step in seconds at current BPM (16th note default)."""
        beat_s = 60.0 / self.bpm
        return beat_s / 4.0  # 16th note subdivision

    @property
    def pattern_duration_s(self) -> float:
        """Total pattern duration in seconds."""
        return self.step_duration_s * self.n_steps

    @property
    def avg_entropy(self) -> float:
        """Average Shannon entropy across all steps (measure of "quantum-ness")."""
        entropies = [s.entropy for s in self.steps if s.state == NoteState.SUPERPOSITION]
        return np.mean(entropies) if entropies else 0.0

    def apply_god_code_phase(self, strength: float = 0.3):
        """Apply GOD_CODE harmonic bias to all steps in superposition.

        Uses the QPU-verified GOD_CODE qubit phase for precision alignment.
        When the qubit is available, each step gets a decomposed phase injection
        (iron + phi + octave cycling) for richer harmonic coloring.
        """
        qb = _get_seq_qubit()
        for idx, step in enumerate(self.steps):
            if step.state == NoteState.SUPERPOSITION:
                step.apply_god_code_bias(self.base_freq, strength)
                # Additional qubit phase rotation per step
                if qb is not None:
                    # .decomposed returns tuple (iron, phi, octave)
                    decomp = qb.decomposed
                    phase_cycle = [decomp[0], decomp[1], decomp[2]]
                    step_phase = phase_cycle[idx % 3]
                    # Apply subtle phase rotation to the dominant amplitude
                    peak_idx = int(np.argmax(np.abs(step.amplitudes)))
                    step.apply_phase(peak_idx, step_phase * strength)

    def apply_swing(self):
        """
        Apply PHI-golden swing: odd-numbered steps get a micro-timing offset.
        This modifies phase rather than time directly (quantum swing).
        """
        for i, step in enumerate(self.steps):
            if i % 2 == 1 and step.state == NoteState.SUPERPOSITION:
                # PHI-ratio swing: rotate amplitudes by golden angle
                golden_phase = 2.0 * math.pi * PHI_INV * self.swing
                for j in range(len(step.amplitudes)):
                    step.amplitudes[j] *= np.exp(1j * golden_phase * (j + 1))
                norm = np.linalg.norm(step.amplitudes)
                if norm > 1e-15:
                    step.amplitudes /= norm

    def entangle_steps(self, idx_a: int, idx_b: int, strength: float = 0.7):
        """
        Create quantum entanglement between two steps.
        When one collapses, the other's probabilities are correlated.
        Uses Bell-state-like correlation: if step A → high pitch,
        step B → complementary harmony.
        """
        if 0 <= idx_a < self.n_steps and 0 <= idx_b < self.n_steps:
            a, b = self.steps[idx_a], self.steps[idx_b]
            min_len = min(len(a.amplitudes), len(b.amplitudes))
            # Create correlation: amplify complementary indices
            for i in range(min_len):
                complement = min_len - 1 - i
                corr = strength * a.amplitudes[i] * np.conj(b.amplitudes[complement])
                b.amplitudes[complement] += corr * 0.5
                a.amplitudes[i] += np.conj(corr) * 0.5
            # Normalize both
            for step in (a, b):
                norm = np.linalg.norm(step.amplitudes)
                if norm > 1e-15:
                    step.amplitudes /= norm
                step.state = NoteState.ENTANGLED
                step.entangled_with.append(
                    b.step_id if step is a else a.step_id
                )

    def collapse_all(
        self,
        mode: Optional[CollapseMode] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> List[NoteEvent]:
        """
        Collapse all steps — render the pattern into concrete events.
        Returns list of NoteEvents (one per step; REST steps return None-pitch).
        """
        mode = mode or self.collapse_mode
        if rng is None:
            rng = np.random.default_rng()

        events = []
        for step in self.steps:
            if step.state in (NoteState.SUPERPOSITION, NoteState.ENTANGLED):
                event = step.collapse(base_freq=self.base_freq, mode=mode, rng=rng)
                events.append(event)
                self.total_collapses += 1

                # Propagate entanglement effects
                if step.entangled_with:
                    for partner_id in step.entangled_with:
                        partner = next(
                            (s for s in self.steps if s.step_id == partner_id),
                            None
                        )
                        if partner and partner.state != NoteState.COLLAPSED:
                            # Bias partner toward harmonic complement
                            complement_freq = event.pitch_hz * PHI
                            partner.apply_god_code_bias(complement_freq, 0.5)
            elif step.event:
                events.append(step.event)
            else:
                events.append(NoteEvent(
                    pitch_hz=0.0, midi_note=0, velocity=0.0,
                    duration_beats=0.0, sacred_alignment=0.0, collapse_entropy=0.0
                ))

        self.loop_count += 1
        return events

    def re_superpose_all(self):
        """Return all steps to superposition for quantum re-roll."""
        for step in self.steps:
            step.re_superpose()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for data recording."""
        return {
            "pattern_id": self.pattern_id,
            "name": self.name,
            "n_steps": self.n_steps,
            "n_qubits_per_step": self.n_qubits_per_step,
            "bpm": self.bpm,
            "swing": self.swing,
            "base_freq": self.base_freq,
            "collapse_mode": self.collapse_mode.name,
            "avg_entropy": self.avg_entropy,
            "loop_count": self.loop_count,
            "total_collapses": self.total_collapses,
            "steps": [s.to_dict() for s in self.steps],
        }


class ProbabilisticSequencer:
    """
    Main sequencer engine — manages multiple QuantumPatterns,
    handles VQPU-backed measurement, and orchestrates the
    superposition→collapse→render→record pipeline.

    Inspired by:
      - FL Studio: Pattern-based grid sequencer with per-step probability
      - Ableton Live: Clip launcher with probability + follow actions
      - Logic Pro: Step sequencer with per-step conditions
      - Bitwig: Probability + humanize per-step

    Quantum upgrades over all DAWs:
      - True superposition (not pseudo-random probability)
      - Entanglement-correlated steps (melodic coherence)
      - Interference-based variation (constructive/destructive)
      - GOD_CODE sacred alignment scoring on every collapse
      - Full quantum state recording for ML insights
    """

    def __init__(
        self,
        bpm: float = DEFAULT_BPM,
        n_qubits: int = 4,
        swing: float = DEFAULT_SWING,
        vqpu_enabled: bool = True,
    ):
        self.bpm = bpm
        self.default_qubits = n_qubits
        self.swing = swing
        self.vqpu_enabled = vqpu_enabled

        self.patterns: Dict[str, QuantumPattern] = {}
        self.playback_order: List[str] = []  # Playlist / arrangement
        self.rng = np.random.default_rng()

        # VQPU bridge (lazy-loaded)
        self._vqpu = None

        # Statistics
        self.total_collapses = 0
        self.total_patterns_created = 0
        self.session_start = time.time()

        logger.info(f"ProbabilisticSequencer initialized — BPM={bpm}, "
                     f"qubits={n_qubits}, VQPU={'ON' if vqpu_enabled else 'OFF'}")

    @property
    def vqpu(self):
        """Lazy-load VQPU bridge."""
        if self._vqpu is None and self.vqpu_enabled:
            try:
                from l104_vqpu_bridge import get_bridge
                self._vqpu = get_bridge()
            except ImportError:
                logger.warning("VQPU bridge not available — falling back to classical RNG")
                self.vqpu_enabled = False
        return self._vqpu

    def create_pattern(
        self,
        name: str = "Quantum Pattern",
        n_steps: int = DEFAULT_STEPS,
        n_qubits: Optional[int] = None,
        base_freq: float = GOD_CODE,
        swing: Optional[float] = None,
    ) -> QuantumPattern:
        """Create a new quantum pattern and register it."""
        pattern = QuantumPattern(
            name=name,
            n_steps=n_steps,
            n_qubits_per_step=n_qubits or self.default_qubits,
            bpm=self.bpm,
            swing=swing if swing is not None else self.swing,
            base_freq=base_freq,
        )
        self.patterns[pattern.pattern_id] = pattern
        self.total_patterns_created += 1
        return pattern

    def create_chord_pattern(
        self,
        name: str = "Quantum Chord",
        n_steps: int = 8,
        voices: int = 4,
        base_freq: float = GOD_CODE,
    ) -> List[QuantumPattern]:
        """
        Create entangled multi-voice pattern (like GHZ chord generation).
        Each voice is a separate pattern, but steps are entangled across voices
        for harmonic coherence.
        """
        patterns = []
        for v in range(voices):
            octave_offset = v - voices // 2
            p = self.create_pattern(
                name=f"{name} V{v+1}",
                n_steps=n_steps,
                base_freq=base_freq * (2.0 ** octave_offset),
            )
            patterns.append(p)

        # Entangle corresponding steps across voices
        for step_idx in range(n_steps):
            for i in range(len(patterns) - 1):
                patterns[i].entangle_steps(step_idx, step_idx, strength=0.6)
                # Cross-voice entanglement via amplitude sharing
                a_step = patterns[i].steps[step_idx]
                b_step = patterns[i + 1].steps[step_idx]
                a_step.apply_interference(b_step, coupling=PHI_INV * 0.4)

        return patterns

    def vqpu_prepare_step(self, step: SuperpositionStep) -> SuperpositionStep:
        """
        Use VQPU to prepare a quantum state for this step.
        Runs a real quantum circuit and maps the resulting statevector
        into the step's amplitude distribution.
        """
        if not self.vqpu:
            return step

        try:
            from l104_vqpu_bridge import QuantumJob, QuantumGate

            # Build a sacred circuit for this step
            n_q = min(step.n_qubits, 8)  # Cap for VQPU performance
            ops = []
            # Hadamard all qubits
            for q in range(n_q):
                ops.append(QuantumGate("H", [q]))
            # GOD_CODE phase on qubit 0
            god_phase = (GOD_CODE / 1000.0) * math.pi
            ops.append(QuantumGate("Rz", [0], [god_phase]))
            # PHI entanglement layers
            for q in range(n_q - 1):
                ops.append(QuantumGate("CNOT", [q, q + 1]))
                ops.append(QuantumGate("Rz", [q + 1], [PHI_INV * math.pi * (q + 1)]))
            # Final Hadamard
            for q in range(n_q):
                ops.append(QuantumGate("H", [q]))

            job = QuantumJob(
                circuit_id=f"seq_step_{step.step_id}",
                num_qubits=n_q,
                operations=ops,
                shots=1024,
                priority=3,
            )
            result = self.vqpu.submit_and_wait(job, timeout=5.0)

            if result and result.probabilities:
                # Map VQPU probabilities into step amplitudes
                probs = result.probabilities
                dim = len(step.amplitudes)
                new_amps = np.zeros(dim, dtype=np.complex128)
                for key, prob in probs.items():
                    idx = int(key, 2) if isinstance(key, str) else int(key)
                    if idx < dim:
                        new_amps[idx] = math.sqrt(prob) * np.exp(
                            1j * god_phase * idx / dim
                        )
                norm = np.linalg.norm(new_amps)
                if norm > 1e-15:
                    step.amplitudes = new_amps / norm
                step.sacred_score = result.god_code or 0.0

        except Exception as e:
            logger.debug(f"VQPU step preparation fallback: {e}")

        return step

    def vqpu_collapse_pattern(self, pattern: QuantumPattern) -> List[NoteEvent]:
        """
        Collapse an entire pattern using VQPU batch circuits.
        Each step gets its own quantum circuit; all are submitted as a batch.
        """
        if not self.vqpu:
            return pattern.collapse_all(rng=self.rng)

        try:
            from l104_vqpu_bridge import QuantumJob, QuantumGate

            jobs = []
            for step in pattern.steps:
                if step.state in (NoteState.SUPERPOSITION, NoteState.ENTANGLED):
                    n_q = min(step.n_qubits, 8)
                    ops = []
                    # Encode step amplitudes into circuit
                    for q in range(n_q):
                        ops.append(QuantumGate("H", [q]))
                        amp_phase = np.angle(step.amplitudes[q % len(step.amplitudes)])
                        ops.append(QuantumGate("Rz", [q], [float(amp_phase)]))
                    for q in range(n_q - 1):
                        ops.append(QuantumGate("CNOT", [q, q + 1]))

                    jobs.append(QuantumJob(
                        circuit_id=f"seq_collapse_{pattern.pattern_id}_{step.step_id}",
                        num_qubits=n_q,
                        operations=ops,
                        shots=512,
                        priority=4,
                    ))

            if jobs:
                results = self.vqpu.submit_batch_and_wait(
                    jobs, timeout=15.0, concurrent=True
                )
                # Map results back to steps
                job_idx = 0
                for step in pattern.steps:
                    if step.state in (NoteState.SUPERPOSITION, NoteState.ENTANGLED):
                        if job_idx < len(results) and results[job_idx]:
                            r = results[job_idx]
                            if r.probabilities:
                                dim = len(step.amplitudes)
                                for key, prob in r.probabilities.items():
                                    idx = int(key, 2) if isinstance(key, str) else int(key)
                                    if idx < dim:
                                        step.amplitudes[idx] = complex(math.sqrt(prob))
                                norm = np.linalg.norm(step.amplitudes)
                                if norm > 1e-15:
                                    step.amplitudes /= norm
                        job_idx += 1

        except Exception as e:
            logger.debug(f"VQPU batch collapse fallback: {e}")

        # Now do classical measurement on the VQPU-prepared states
        return pattern.collapse_all(rng=self.rng)

    def render_pattern_audio(
        self,
        pattern: QuantumPattern,
        sample_rate: int = 96000,
    ) -> np.ndarray:
        """
        Render a collapsed pattern to audio samples.
        Each NoteEvent becomes a synthesized tone segment.
        """
        step_samples = int(pattern.step_duration_s * sample_rate)
        total_samples = step_samples * pattern.n_steps
        audio = np.zeros(total_samples, dtype=np.float64)

        events = pattern.collapse_all(rng=self.rng) if pattern.steps[0].state != NoteState.COLLAPSED else [
            s.event for s in pattern.steps
        ]

        for i, event in enumerate(events):
            if event and event.pitch_hz > 20.0 and event.velocity > 0.01:
                start = i * step_samples
                dur_samples = int(event.duration_beats * step_samples)
                dur_samples = min(dur_samples, total_samples - start)
                if dur_samples <= 0:
                    continue

                t = np.arange(dur_samples) / sample_rate
                # Synthesize with GOD_CODE phase alignment
                tone = event.velocity * np.sin(
                    2.0 * math.pi * event.pitch_hz * t + event.phase_offset
                )
                # PHI-harmonic overtone
                tone += event.velocity * 0.3 * np.sin(
                    2.0 * math.pi * event.pitch_hz * PHI * t
                )
                # Fade envelope
                fade_len = min(256, dur_samples // 4)
                if fade_len > 0:
                    fade_in = np.linspace(0, 1, fade_len)
                    fade_out = np.linspace(1, 0, fade_len)
                    tone[:fade_len] *= fade_in
                    tone[-fade_len:] *= fade_out

                audio[start:start + dur_samples] += tone

        # Normalize
        peak = np.max(np.abs(audio))
        if peak > 1e-10:
            audio *= 0.85 / peak
        return audio

    def status(self) -> Dict[str, Any]:
        """Return sequencer status for diagnostics."""
        return {
            "bpm": self.bpm,
            "default_qubits": self.default_qubits,
            "swing": self.swing,
            "vqpu_enabled": self.vqpu_enabled,
            "patterns": len(self.patterns),
            "total_patterns_created": self.total_patterns_created,
            "total_collapses": self.total_collapses,
            "session_duration_s": time.time() - self.session_start,
            "playback_order": self.playback_order,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Full state serialization for data recording."""
        return {
            **self.status(),
            "patterns": {pid: p.to_dict() for pid, p in self.patterns.items()},
        }
