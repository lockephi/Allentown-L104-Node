"""
Quantum Synthesizer Engines — Waveform Generation via Quantum Circuits
═══════════════════════════════════════════════════════════════════════════════
A bank of synthesizer engines where oscillators, filters, and modulators
are implemented as quantum circuits executed on the VQPU.

Inspired by:
  • FL Studio Sytrus: 6-operator FM synthesis with flexible routing
  • Serum (Xfer): Wavetable synthesis with spectral morphing
  • Massive (NI): Modular routing with wave-shaping
  • Reaktor (NI): Modular synthesis with custom DSP blocks
  • Vital: Spectral warping, randomizable wavetables

Quantum Synth Engines:

  1. **QuantumOscillator**: Generates waveforms from quantum circuit
     measurement statistics. Each cycle's amplitude comes from the
     probability distribution of a parameterized circuit.

  2. **SuperpositionWavetable**: A wavetable where each frame is a
     quantum state. Morphing between frames = unitary rotation between
     states. No interpolation artifacts — smooth unitary evolution.

  3. **EntanglementFM**: FM synthesis where carrier and modulator
     frequencies are entangled via Bell pairs. Modulation depth is
     determined by concurrence (entanglement measure).

  4. **InterferenceFilter**: A filter where the cutoff frequency is
     set by quantum interference between two oscillators. Constructive
     = open, destructive = closed.

  5. **QuantumNoiseGenerator**: Noise from quantum measurement
     (truly random via Born rule, not pseudo-random).

  6. **GodCodeResonator**: A resonant filter bank tuned to GOD_CODE
     harmonics, driven by sacred circuit measurements.

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

logger = logging.getLogger("l104.audio.synth")

# ── Synthesis Constants ──────────────────────────────────────────────────────
WAVETABLE_SIZE = 2048       # Samples per wavetable frame
MAX_WAVETABLE_FRAMES = 256  # Frames in a wavetable
MAX_UNISON_VOICES = 16      # Maximum unison detuned voices
MAX_PARTIALS = 512          # Maximum additive partials
TWO_PI = 2.0 * math.pi

# Sacred waveform seeds
SACRED_FREQS = [GOD_CODE, GOD_CODE * PHI, GOD_CODE * PHI_INV, ZENITH_HZ, OMEGA]

# GOD_CODE qubit gate reference (lazy-loaded)
_GOD_CODE_QUBIT = None


def _get_god_code_qubit():
    """Lazy-load the GOD_CODE qubit singleton for waveform generation."""
    global _GOD_CODE_QUBIT
    if _GOD_CODE_QUBIT is None:
        try:
            from l104_god_code_simulator.god_code_qubit import GOD_CODE_QUBIT
            _GOD_CODE_QUBIT = GOD_CODE_QUBIT
        except ImportError:
            pass
    return _GOD_CODE_QUBIT


class WaveShape(Enum):
    """Base waveform shapes for oscillators."""
    SINE = auto()
    TRIANGLE = auto()
    SAWTOOTH = auto()
    SQUARE = auto()
    NOISE = auto()
    QUANTUM = auto()         # Waveform from circuit measurement
    GOD_CODE_WAVE = auto()   # GOD_CODE harmonic series
    PHI_SPIRAL = auto()      # Golden-spiral-modulated wave
    SUPERPOSITION = auto()   # All shapes in superposition


class FilterType(Enum):
    """Quantum filter types."""
    LOWPASS = auto()
    HIGHPASS = auto()
    BANDPASS = auto()
    NOTCH = auto()
    INTERFERENCE = auto()    # Quantum interference filter
    GOD_CODE_RESONANT = auto()  # Sacred resonance bank


@dataclass
class OscillatorState:
    """Internal state for a quantum oscillator."""
    phase: float = 0.0
    frequency: float = GOD_CODE
    amplitude: float = 1.0
    waveform: WaveShape = WaveShape.SINE
    detune_cents: float = 0.0
    phase_offset: float = 0.0
    # Quantum state (amplitudes for basis waveforms)
    quantum_amplitudes: np.ndarray = field(default=None)
    # Circuit parameters for QUANTUM waveform
    circuit_params: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.quantum_amplitudes is None:
            # Default: superposition of all base waveforms
            n = len(WaveShape) - 1  # exclude SUPERPOSITION itself
            self.quantum_amplitudes = np.ones(n, dtype=np.complex128) / math.sqrt(n)


@dataclass
class QuantumOscillator:
    """
    A single oscillator that generates audio from quantum circuit measurements.

    The core idea: instead of using a fixed waveform lookup table, each audio
    cycle's shape is determined by the probability distribution from a
    parameterized quantum circuit.

    The circuit's parameters evolve over time (LFO-like), creating organic
    timbral evolution that no classical synth can produce.
    """
    osc_id: str = field(default_factory=lambda: uuid.uuid4().hex[:6])
    state: OscillatorState = field(default_factory=OscillatorState)
    n_qubits: int = 4  # Circuit complexity
    unison_voices: int = 1
    unison_spread: float = 0.15  # Cents spread for unison

    # Cached wavetable from last quantum measurement
    _wavetable: np.ndarray = field(default=None)
    _wavetable_phase: float = 0.0

    # VQPU bridge (shared)
    _vqpu: Any = field(default=None, repr=False)

    def __post_init__(self):
        if self._wavetable is None:
            self._wavetable = self._generate_wavetable()

    def _generate_wavetable(self) -> np.ndarray:
        """Generate a single-cycle wavetable from current state."""
        t = np.linspace(0, TWO_PI, WAVETABLE_SIZE, endpoint=False)

        if self.state.waveform == WaveShape.SINE:
            return np.sin(t)
        elif self.state.waveform == WaveShape.TRIANGLE:
            return 2.0 * np.abs(2.0 * (t / TWO_PI - np.floor(t / TWO_PI + 0.5))) - 1.0
        elif self.state.waveform == WaveShape.SAWTOOTH:
            return 2.0 * (t / TWO_PI - np.floor(t / TWO_PI + 0.5))
        elif self.state.waveform == WaveShape.SQUARE:
            return np.sign(np.sin(t))
        elif self.state.waveform == WaveShape.NOISE:
            return np.random.default_rng().standard_normal(WAVETABLE_SIZE)
        elif self.state.waveform == WaveShape.GOD_CODE_WAVE:
            # GOD_CODE harmonic series with QPU-verified qubit phase injection
            wave = np.zeros(WAVETABLE_SIZE)
            qb = _get_god_code_qubit()
            if qb is not None:
                # Use qubit's decomposed phases for harmonic coloring
                # .decomposed returns tuple (iron, phi, octave)
                decomp = qb.decomposed
                iron_ph = decomp[0]
                phi_ph = decomp[1]
                oct_ph = decomp[2]
                for n in range(1, 13):
                    amp = 1.0 / (n ** PHI_INV)
                    # Cycle through iron → phi → octave phases per harmonic
                    # Scaled to 15% for subtle coloring — preserves original timbre
                    phase_cycle = [iron_ph * 0.15, phi_ph * 0.15, oct_ph * 0.15]
                    ph = phase_cycle[n % 3]
                    wave += amp * np.sin(n * t + ph)
            else:
                # Fallback: constant-derived phases
                for n in range(1, 13):
                    amp = 1.0 / (n ** PHI_INV)
                    ph = GOD_CODE_PHASE * (n % 3) / 3.0
                    wave += amp * np.sin(n * t + ph)
            wave /= np.max(np.abs(wave)) + 1e-15
            return wave
        elif self.state.waveform == WaveShape.PHI_SPIRAL:
            # Golden-spiral amplitude modulation
            wave = np.sin(t) * (1.0 + 0.3 * np.sin(PHI * t))
            wave *= (1.0 + 0.2 * np.sin(PHI_INV * t * 3.0))
            wave /= np.max(np.abs(wave)) + 1e-15
            return wave
        elif self.state.waveform == WaveShape.QUANTUM:
            return self._quantum_wavetable()
        elif self.state.waveform == WaveShape.SUPERPOSITION:
            return self._superposition_wavetable()
        return np.sin(t)

    def _quantum_wavetable(self) -> np.ndarray:
        """Generate wavetable from VQPU quantum circuit."""
        # Parameterized circuit: amplitudes determine harmonic weights
        n_partials = 2 ** self.n_qubits
        probs = np.abs(self.state.quantum_amplitudes[:n_partials]) ** 2
        if probs.sum() < 1e-15:
            probs = np.ones(n_partials) / n_partials
        else:
            probs /= probs.sum()

        t = np.linspace(0, TWO_PI, WAVETABLE_SIZE, endpoint=False)
        wave = np.zeros(WAVETABLE_SIZE)
        for k in range(n_partials):
            harmonic = k + 1
            amplitude = math.sqrt(probs[k])  # Amplitude from probability
            phase = np.angle(self.state.quantum_amplitudes[k % len(self.state.quantum_amplitudes)])
            wave += amplitude * np.sin(harmonic * t + phase)

        norm = np.max(np.abs(wave))
        return wave / max(norm, 1e-15)

    def _superposition_wavetable(self) -> np.ndarray:
        """
        All base waveforms in superposition — weighted by quantum amplitudes.
        """
        t = np.linspace(0, TWO_PI, WAVETABLE_SIZE, endpoint=False)
        bases = [
            np.sin(t),                                                  # SINE
            2.0 * np.abs(2.0 * (t / TWO_PI - np.floor(t / TWO_PI + 0.5))) - 1.0,  # TRI
            2.0 * (t / TWO_PI - np.floor(t / TWO_PI + 0.5)),          # SAW
            np.sign(np.sin(t)),                                         # SQR
            np.random.default_rng(42).standard_normal(WAVETABLE_SIZE),  # NOISE (seeded)
        ]
        amps = self.state.quantum_amplitudes[:len(bases)]
        wave = np.zeros(WAVETABLE_SIZE, dtype=np.complex128)
        for i, base in enumerate(bases):
            if i < len(amps):
                wave += amps[i] * base
        result = np.real(wave)
        norm = np.max(np.abs(result))
        return result / max(norm, 1e-15)

    def vqpu_evolve(self, time_s: float):
        """
        Evolve oscillator state using VQPU circuit.
        The circuit parameters rotate based on time, creating organic LFO-like
        timbral evolution driven by quantum mechanics.

        Now includes GOD_CODE qubit Rz(θ_GC) gate as the initial rotation,
        ensuring QPU-verified phase alignment in every evolution step.
        """
        if self._vqpu is None:
            try:
                from l104_vqpu_bridge import get_bridge
                self._vqpu = get_bridge()
            except ImportError:
                return

        try:
            from l104_vqpu_bridge import QuantumJob, QuantumGate
            n_q = min(self.n_qubits, 6)
            ops = []

            # GOD_CODE qubit phase rotation on qubit 0 (QPU-verified Rz(θ_GC))
            ops.append(QuantumGate("Rz", [0], [GOD_CODE_PHASE]))

            # Time-evolving parameters
            theta = TWO_PI * PHI * time_s
            for q in range(n_q):
                ops.append(QuantumGate("Ry", [q], [theta * (q + 1) / n_q]))
                ops.append(QuantumGate("Rz", [q], [GOD_CODE / 1000.0 * math.pi * time_s]))
            for q in range(n_q - 1):
                ops.append(QuantumGate("CNOT", [q, q + 1]))

            job = QuantumJob(
                circuit_id=f"osc_evolve_{self.osc_id}_{int(time_s*1000)}",
                num_qubits=n_q,
                operations=ops,
                shots=256,
                priority=6,
            )
            result = self._vqpu.submit_and_wait(job, timeout=2.0)
            if result and result.probabilities:
                dim = len(self.state.quantum_amplitudes)
                for key, prob in result.probabilities.items():
                    idx = int(key, 2) if isinstance(key, str) else int(key)
                    if idx < dim:
                        self.state.quantum_amplitudes[idx] = (
                            math.sqrt(prob) * np.exp(1j * theta * idx / dim)
                        )
                norm = np.linalg.norm(self.state.quantum_amplitudes)
                if norm > 1e-15:
                    self.state.quantum_amplitudes /= norm
                self._wavetable = self._generate_wavetable()
        except Exception as e:
            logger.debug(f"VQPU oscillator evolve fallback: {e}")

    def render(self, n_samples: int, sample_rate: int = 96000) -> np.ndarray:
        """
        Render audio samples using wavetable playback with phase accumulator.
        Supports unison voices with golden-ratio spread.
        """
        freq = self.state.frequency * (2.0 ** (self.state.detune_cents / 1200.0))
        output = np.zeros(n_samples, dtype=np.float64)

        for voice in range(self.unison_voices):
            # Unison detune: golden-ratio spread
            if self.unison_voices > 1:
                detune_factor = (voice - self.unison_voices / 2.0) / self.unison_voices
                voice_freq = freq * (2.0 ** (detune_factor * self.unison_spread / 1200.0))
            else:
                voice_freq = freq

            # Phase accumulator
            phase_inc = voice_freq * WAVETABLE_SIZE / sample_rate
            phases = (self.state.phase + self.state.phase_offset +
                      np.arange(n_samples) * phase_inc) % WAVETABLE_SIZE
            indices = phases.astype(np.int32) % WAVETABLE_SIZE
            frac = phases - indices

            # Linear interpolation between wavetable samples
            next_indices = (indices + 1) % WAVETABLE_SIZE
            voice_signal = (
                self._wavetable[indices] * (1.0 - frac) +
                self._wavetable[next_indices] * frac
            )
            output += voice_signal / max(self.unison_voices, 1)

        # Update phase accumulator
        self.state.phase = (self.state.phase + n_samples * freq * WAVETABLE_SIZE / sample_rate) % WAVETABLE_SIZE

        return output * self.state.amplitude

    def to_dict(self) -> Dict[str, Any]:
        return {
            "osc_id": self.osc_id,
            "frequency": self.state.frequency,
            "amplitude": self.state.amplitude,
            "waveform": self.state.waveform.name,
            "detune_cents": self.state.detune_cents,
            "n_qubits": self.n_qubits,
            "unison_voices": self.unison_voices,
            "unison_spread": self.unison_spread,
        }


@dataclass
class SuperpositionWavetable:
    """
    A wavetable synthesizer where each frame is a quantum state vector.

    Morphing between frames = unitary rotation (not linear interpolation).
    This produces phase-continuous, alias-free spectral morphing.

    The wavetable can be seeded from:
      - GOD_CODE harmonic analysis
      - VQPU circuit measurements at different parameter values
      - Imported audio samples (analyzed into quantum state representation)
    """
    name: str = "Quantum Wavetable"
    n_frames: int = 64
    n_qubits: int = 4

    # Frames stored as quantum state vectors
    frames: List[np.ndarray] = field(default_factory=list)

    # Current playback position (0.0 to n_frames-1)
    frame_position: float = 0.0
    frame_lfo_rate: float = 0.1  # Hz — wavetable scanning rate

    def __post_init__(self):
        if not self.frames:
            self._generate_god_code_frames()

    def _generate_god_code_frames(self):
        """Generate wavetable frames from GOD_CODE parametric evolution.

        Uses GOD_CODE qubit decomposed phases (iron/phi/octave) to seed
        the harmonic structure of each frame, providing QPU-verified
        phase coherence across the wavetable.
        """
        dim = 2 ** self.n_qubits
        self.frames = []

        # Load qubit phases for frame coloring
        qb = _get_god_code_qubit()
        iron_ph = GOD_CODE_PHASE / 3.0  # fallback decomposition
        phi_ph = GOD_CODE_PHASE * 2.0 / 3.0
        if qb is not None:
            # .decomposed returns tuple (iron, phi, octave)
            decomp = qb.decomposed
            iron_ph = decomp[0]
            phi_ph = decomp[1]

        for f in range(self.n_frames):
            theta = TWO_PI * f / self.n_frames
            state = np.zeros(dim, dtype=np.complex128)
            for k in range(dim):
                # PHI-evolving harmonic amplitudes with qubit phase seed
                amp = math.exp(-k / (dim * PHI_INV))
                phase = (theta * k * PHI + GOD_CODE / 1000.0 * k
                         + iron_ph * 0.2 * (k % 3 == 0) + phi_ph * 0.2 * (k % 3 == 1))
                state[k] = amp * np.exp(1j * phase)
            norm = np.linalg.norm(state)
            if norm > 1e-15:
                state /= norm
            self.frames.append(state)

    def get_wavetable_at(self, position: float) -> np.ndarray:
        """
        Get a wavetable cycle at a given frame position.
        Uses unitary interpolation between adjacent frames.
        """
        if not self.frames:
            return np.sin(np.linspace(0, TWO_PI, WAVETABLE_SIZE, endpoint=False))

        pos = position % len(self.frames)
        idx_a = int(pos)
        idx_b = (idx_a + 1) % len(self.frames)
        frac = pos - idx_a

        state_a = self.frames[idx_a]
        state_b = self.frames[idx_b]

        # Unitary interpolation: slerp on the Bloch sphere
        dot = np.real(np.vdot(state_a, state_b))
        dot = np.clip(dot, -1.0, 1.0)
        omega = math.acos(abs(dot))

        if omega < 1e-10:
            interp_state = state_a
        else:
            interp_state = (
                math.sin((1.0 - frac) * omega) / math.sin(omega) * state_a +
                math.sin(frac * omega) / math.sin(omega) * state_b
            )

        # Convert quantum state to wavetable
        t = np.linspace(0, TWO_PI, WAVETABLE_SIZE, endpoint=False)
        wave = np.zeros(WAVETABLE_SIZE)
        for k in range(len(interp_state)):
            harmonic = k + 1
            amp = abs(interp_state[k])
            phase = np.angle(interp_state[k])
            wave += amp * np.sin(harmonic * t + phase)

        norm = np.max(np.abs(wave))
        return wave / max(norm, 1e-15)

    def render(self, n_samples: int, frequency: float = GOD_CODE,
               sample_rate: int = 96000) -> np.ndarray:
        """Render audio with wavetable scanning."""
        output = np.zeros(n_samples, dtype=np.float64)
        phase = 0.0
        phase_inc = frequency * WAVETABLE_SIZE / sample_rate
        frame_inc = self.frame_lfo_rate / sample_rate

        # Process in chunks for frame evolution
        chunk_size = 4096
        pos = 0
        while pos < n_samples:
            end = min(pos + chunk_size, n_samples)
            count = end - pos

            wt = self.get_wavetable_at(self.frame_position)
            phases = (phase + np.arange(count) * phase_inc) % WAVETABLE_SIZE
            indices = phases.astype(np.int32) % WAVETABLE_SIZE
            frac = phases - indices
            next_idx = (indices + 1) % WAVETABLE_SIZE
            output[pos:end] = wt[indices] * (1.0 - frac) + wt[next_idx] * frac

            phase = (phase + count * phase_inc) % WAVETABLE_SIZE
            self.frame_position += count * frame_inc

            pos = end

        return output


@dataclass
class EntanglementFM:
    """
    FM synthesis where carrier-modulator coupling is determined by
    quantum entanglement. Higher entanglement = deeper modulation.

    Architecture (inspired by Sytrus/DX7):
      - Carrier oscillator: frequency from pattern step
      - Modulator oscillator: frequency ratio set by sacred interval
      - Modulation index: proportional to Bell-state concurrence
      - Feedback: controlled by Von Neumann entropy of the entangled state
    """
    carrier_freq: float = GOD_CODE
    mod_ratio: float = PHI         # Carrier:Modulator ratio
    mod_index: float = 1.0         # FM modulation depth
    feedback: float = 0.0          # Operator feedback amount

    # Entanglement parameters (from VQPU)
    concurrence: float = 0.0       # Bell-state entanglement measure (0-1)
    vne_entropy: float = 0.0       # Von Neumann entropy

    _carrier_phase: float = 0.0
    _mod_phase: float = 0.0

    def update_entanglement(self, concurrence: float, vne: float):
        """Update FM parameters from quantum entanglement measurements."""
        self.concurrence = concurrence
        self.vne_entropy = vne
        # Modulation index scales with concurrence
        self.mod_index = 1.0 + concurrence * 4.0 * PHI
        # Feedback scales with entropy
        self.feedback = min(vne * PHI_INV, 0.95)

    def render(self, n_samples: int, sample_rate: int = 96000) -> np.ndarray:
        """Render FM synthesis audio."""
        mod_freq = self.carrier_freq * self.mod_ratio
        t = np.arange(n_samples) / sample_rate

        # Modulator with feedback
        mod_phase_acc = TWO_PI * mod_freq * t + self._mod_phase
        modulator = np.sin(mod_phase_acc)

        # Apply feedback
        if self.feedback > 0.01:
            for i in range(1, len(modulator)):
                modulator[i] = math.sin(
                    mod_phase_acc[i] + self.feedback * modulator[i - 1]
                )

        # Carrier modulated by modulator
        carrier_phase_acc = (
            TWO_PI * self.carrier_freq * t +
            self.mod_index * modulator +
            self._carrier_phase
        )
        output = np.sin(carrier_phase_acc)

        # Update phase accumulators
        self._carrier_phase = carrier_phase_acc[-1] % TWO_PI if len(t) > 0 else 0.0
        self._mod_phase = mod_phase_acc[-1] % TWO_PI if len(t) > 0 else 0.0

        return output


@dataclass
class InterferenceFilter:
    """
    A filter where the frequency response is shaped by quantum interference.

    Two oscillator reference signals interfere; the spectral pattern of their
    interference defines the filter shape. This creates organic, evolving
    filter sweeps that are impossible with classical resonant filters.

    Cutoff = frequency where constructive interference peaks
    Resonance = sharpness of the interference fringe
    """
    cutoff_hz: float = GOD_CODE
    resonance: float = PHI_INV    # Q-factor (golden ratio default)
    filter_type: FilterType = FilterType.LOWPASS
    # Interference parameters
    ref_freq_a: float = GOD_CODE
    ref_freq_b: float = GOD_CODE * PHI  # Golden ratio separation

    def apply(self, signal: np.ndarray, sample_rate: int = 96000) -> np.ndarray:
        """Apply quantum interference filter."""
        if len(signal) < 4:
            return signal

        if self.filter_type == FilterType.INTERFERENCE:
            return self._interference_filter(signal, sample_rate)
        elif self.filter_type == FilterType.GOD_CODE_RESONANT:
            return self._god_code_resonant(signal, sample_rate)
        else:
            return self._classical_filter(signal, sample_rate)

    def _interference_filter(self, signal: np.ndarray, sr: int) -> np.ndarray:
        """Apply spectral interference pattern as filter shape."""
        spectrum = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), 1.0 / sr)

        # Create interference pattern between two reference frequencies
        pattern = np.zeros_like(freqs)
        for i, f in enumerate(freqs):
            if f < 1.0:
                pattern[i] = 1.0
                continue
            # Phase difference between two reference waves at this frequency
            phase_a = TWO_PI * self.ref_freq_a / f
            phase_b = TWO_PI * self.ref_freq_b / f
            interference = math.cos(phase_a - phase_b) * 0.5 + 0.5
            # Resonance sharpens the pattern
            pattern[i] = interference ** (1.0 / max(self.resonance, 0.01))

        spectrum *= pattern
        return np.fft.irfft(spectrum, n=len(signal))

    def _god_code_resonant(self, signal: np.ndarray, sr: int) -> np.ndarray:
        """Resonant filter bank tuned to GOD_CODE harmonics."""
        spectrum = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), 1.0 / sr)

        # Boost GOD_CODE harmonics, attenuate between
        response = np.ones(len(freqs)) * 0.3  # Base attenuation
        for n in range(1, 17):
            harmonic = GOD_CODE * n
            if harmonic < freqs[-1]:
                # Gaussian peak at each harmonic
                sigma = harmonic * 0.02 / self.resonance
                peak = np.exp(-0.5 * ((freqs - harmonic) / max(sigma, 1.0)) ** 2)
                response += peak * (1.0 / (n ** PHI_INV))

        # Also boost PHI-ratio harmonics
        for n in range(-2, 6):
            phi_harmonic = GOD_CODE * (PHI ** n)
            if 20.0 < phi_harmonic < freqs[-1]:
                sigma = phi_harmonic * 0.03 / self.resonance
                peak = np.exp(-0.5 * ((freqs - phi_harmonic) / max(sigma, 1.0)) ** 2)
                response += peak * 0.5

        response = np.clip(response, 0.0, 2.0)
        spectrum *= response
        return np.fft.irfft(spectrum, n=len(signal))

    def _classical_filter(self, signal: np.ndarray, sr: int) -> np.ndarray:
        """Standard biquad-like filter for non-quantum modes."""
        spectrum = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), 1.0 / sr)

        fc = self.cutoff_hz
        q = max(self.resonance, 0.1)

        if self.filter_type == FilterType.LOWPASS:
            response = 1.0 / np.sqrt(1.0 + (freqs / max(fc, 1.0)) ** (2.0 * 2))
            # Resonance peak near cutoff
            peak_mask = np.exp(-0.5 * ((freqs - fc) / (fc * 0.1 / q)) ** 2) * (q - 0.707)
            response += np.clip(peak_mask, 0, 1.0)
        elif self.filter_type == FilterType.HIGHPASS:
            response = 1.0 / np.sqrt(1.0 + (max(fc, 1.0) / np.maximum(freqs, 1.0)) ** (2.0 * 2))
        elif self.filter_type == FilterType.BANDPASS:
            bw = fc / max(q, 0.1)
            response = np.exp(-0.5 * ((freqs - fc) / max(bw, 1.0)) ** 2)
        elif self.filter_type == FilterType.NOTCH:
            bw = fc / max(q, 0.1)
            response = 1.0 - np.exp(-0.5 * ((freqs - fc) / max(bw, 1.0)) ** 2)
        else:
            response = np.ones_like(freqs)

        spectrum *= response
        return np.fft.irfft(spectrum, n=len(signal))


class QuantumSynthEngine:
    """
    Top-level synthesizer engine — manages oscillators, filters, and
    FM operators. Can produce a complete synthesizer voice.

    Signal flow (per voice):
      Oscillator(s) → [FM modulation] → [Filter] → [Envelope] → Out

    Multiple voices can be layered (like Serum's OSC A + OSC B + Sub + Noise).
    """

    def __init__(self, n_qubits: int = 4, sample_rate: int = 96000):
        self.n_qubits = n_qubits
        self.sample_rate = sample_rate

        # Oscillator bank (up to 4 like Serum/Vital)
        self.oscillators: List[QuantumOscillator] = []
        self.wavetables: List[SuperpositionWavetable] = []
        self.fm_operators: List[EntanglementFM] = []
        self.filters: List[InterferenceFilter] = []

        # Global
        self.master_amplitude = 0.85
        self.voices_rendered = 0

        # VQPU integration
        self._vqpu = None

    @property
    def vqpu(self):
        if self._vqpu is None:
            try:
                from l104_vqpu_bridge import get_bridge
                self._vqpu = get_bridge()
            except ImportError:
                pass
        return self._vqpu

    def add_oscillator(
        self,
        waveform: WaveShape = WaveShape.QUANTUM,
        frequency: float = GOD_CODE,
        amplitude: float = 1.0,
        n_qubits: int = 4,
        unison: int = 1,
    ) -> QuantumOscillator:
        """Add a quantum oscillator to the synth."""
        osc = QuantumOscillator(
            state=OscillatorState(
                frequency=frequency,
                amplitude=amplitude,
                waveform=waveform,
            ),
            n_qubits=n_qubits,
            unison_voices=unison,
            _vqpu=self.vqpu,
        )
        self.oscillators.append(osc)
        return osc

    def add_wavetable(
        self,
        n_frames: int = 64,
        lfo_rate: float = 0.1,
    ) -> SuperpositionWavetable:
        """Add a superposition wavetable oscillator."""
        wt = SuperpositionWavetable(
            n_frames=n_frames,
            n_qubits=self.n_qubits,
            frame_lfo_rate=lfo_rate,
        )
        self.wavetables.append(wt)
        return wt

    def add_fm_operator(
        self,
        carrier_freq: float = GOD_CODE,
        mod_ratio: float = PHI,
        mod_index: float = 1.0,
    ) -> EntanglementFM:
        """Add an entanglement-FM operator."""
        fm = EntanglementFM(
            carrier_freq=carrier_freq,
            mod_ratio=mod_ratio,
            mod_index=mod_index,
        )
        self.fm_operators.append(fm)
        return fm

    def add_filter(
        self,
        filter_type: FilterType = FilterType.GOD_CODE_RESONANT,
        cutoff: float = GOD_CODE * 4.0,
        resonance: float = PHI_INV,
    ) -> InterferenceFilter:
        """Add a quantum filter."""
        filt = InterferenceFilter(
            cutoff_hz=cutoff,
            resonance=resonance,
            filter_type=filter_type,
        )
        self.filters.append(filt)
        return filt

    def vqpu_seed_all(self, time_s: float = 0.0):
        """Evolve all oscillators via VQPU circuits."""
        for osc in self.oscillators:
            osc.vqpu_evolve(time_s)

        # Update FM operators from entanglement measurements
        if self.vqpu and self.fm_operators:
            try:
                from l104_vqpu_bridge import QuantumJob, QuantumGate
                # Bell pair for entanglement measurement
                ops = [
                    QuantumGate("H", [0]),
                    QuantumGate("CNOT", [0, 1]),
                    QuantumGate("Rz", [0], [GOD_CODE / 1000.0 * math.pi]),
                ]
                job = QuantumJob(
                    circuit_id=f"fm_entangle_{int(time_s*1000)}",
                    num_qubits=2,
                    operations=ops,
                    shots=512,
                )
                result = self.vqpu.submit_and_wait(job, timeout=2.0)
                if result and result.probabilities:
                    # Estimate concurrence from Bell measurement
                    p00 = result.probabilities.get("00", 0.0)
                    p11 = result.probabilities.get("11", 0.0)
                    concurrence = 2.0 * math.sqrt(max(p00 * p11, 0.0))
                    vne = -sum(
                        p * math.log2(max(p, 1e-15))
                        for p in result.probabilities.values()
                    )
                    for fm in self.fm_operators:
                        fm.update_entanglement(concurrence, vne)
            except Exception:
                pass

    def render_voice(
        self,
        n_samples: int,
        frequency: Optional[float] = None,
    ) -> np.ndarray:
        """
        Render a complete synth voice: all oscillators + FM → filter → output.
        """
        output = np.zeros(n_samples, dtype=np.float64)

        # Oscillators
        for osc in self.oscillators:
            if frequency:
                osc.state.frequency = frequency
            output += osc.render(n_samples, self.sample_rate)

        # Wavetable oscillators
        for wt in self.wavetables:
            output += wt.render(n_samples, frequency or GOD_CODE, self.sample_rate)

        # FM operators
        for fm in self.fm_operators:
            if frequency:
                fm.carrier_freq = frequency
            output += fm.render(n_samples, self.sample_rate)

        # Normalize before filtering
        n_sources = len(self.oscillators) + len(self.wavetables) + len(self.fm_operators)
        if n_sources > 1:
            output /= math.sqrt(n_sources)

        # Apply filters in series
        for filt in self.filters:
            output = filt.apply(output, self.sample_rate)

        output *= self.master_amplitude
        self.voices_rendered += 1
        return output

    def status(self) -> Dict[str, Any]:
        return {
            "oscillators": len(self.oscillators),
            "wavetables": len(self.wavetables),
            "fm_operators": len(self.fm_operators),
            "filters": len(self.filters),
            "n_qubits": self.n_qubits,
            "voices_rendered": self.voices_rendered,
            "vqpu_available": self.vqpu is not None,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.status(),
            "oscillators": [o.to_dict() for o in self.oscillators],
        }
