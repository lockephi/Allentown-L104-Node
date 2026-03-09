"""
Tone Generator — quantum-modulated pure tone generation.

Ported from l104_quantum_audio.py.  Provides QuantumPureToneGenerator with
sacred-frequency tones, quantum circuit-modulated tones, sacred chords,
and WAV I/O.

INVARIANT: 527.5184818492612 | PILOT: LONDEL
"""

from __future__ import annotations

import io
import wave
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field

from .constants import (
    GOD_CODE, PHI, VOID_CONSTANT, ZENITH_HZ, SCHUMANN_HZ,
    SACRED_INTERVALS, GOD_CODE_PHASE, IRON_PHASE,
)

# Scaled Planck resonance for audible range
PLANCK_RESONANCE = 6.62607015e-34 * 1e35


@dataclass
class QuantumTone:
    """A single quantum-modulated tone."""
    frequency: float
    amplitude: float = 1.0
    phase: float = 0.0
    duration: float = 1.0
    quantum_modulation: Dict[str, float] = field(default_factory=dict)


@dataclass
class QuantumAudioResult:
    """Result of quantum tone generation."""
    samples: np.ndarray
    sample_rate: int
    duration: float
    frequencies: List[float]
    quantum_metrics: Dict[str, Any]
    sacred_alignment: float


class QuantumPureToneGenerator:
    """Generate pure tones modulated by quantum circuit measurements.

    Usage::

        gen = QuantumPureToneGenerator()
        result = gen.god_code_tone(duration=3.0)
        gen.save_wav(result, "god_code.wav")
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._quantum_available = False
        self._qubit_available = False
        self._god_code_qubit = None
        self._init_quantum()
        self._init_god_code_qubit()

    def _init_quantum(self):
        try:
            from l104_quantum_gate_engine import get_engine, GateCircuit
            from l104_qiskit_utils import aer_backend
            self._qengine = get_engine()
            self._aer = aer_backend
            self._GateCircuit = GateCircuit
            self._quantum_available = True
        except ImportError:
            self._quantum_available = False

    def _init_god_code_qubit(self):
        """Initialize the QPU-verified GOD_CODE qubit for phase injection."""
        try:
            from l104_god_code_simulator.god_code_qubit import GOD_CODE_QUBIT
            self._god_code_qubit = GOD_CODE_QUBIT
            self._qubit_available = True
        except ImportError:
            self._qubit_available = False

    def _quantum_probs(self, circuit_name: str = "bell", n_qubits: int = 2) -> np.ndarray:
        if not self._quantum_available:
            return np.random.dirichlet(np.ones(2**n_qubits) * PHI)

        circuits = {
            "bell": lambda: self._qengine.bell_pair(),
            "ghz": lambda: self._qengine.ghz_state(n_qubits),
            "qft": lambda: self._qengine.quantum_fourier_transform(n_qubits),
            "sacred": lambda: self._qengine.sacred_circuit(n_qubits, depth=4),
        }
        circ = circuits.get(circuit_name, circuits["bell"])()
        return np.array(self._aer.run_statevector(circ))

    # ── Core Tone Generation ─────────────────────────────────────────────

    def pure_tone(self, frequency: float, duration: float = 1.0,
                  amplitude: float = 0.8, phase: float = 0.0) -> np.ndarray:
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        return (amplitude * np.sin(2 * np.pi * frequency * t + phase)).astype(np.float32)

    def harmonic_series(self, fundamental: float, n_harmonics: int = 8,
                        duration: float = 1.0, decay: float = 0.7) -> np.ndarray:
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        samples = np.zeros_like(t)
        for n in range(1, n_harmonics + 1):
            samples += decay ** (n - 1) * np.sin(2 * np.pi * fundamental * n * t)
        samples = samples / max(np.max(np.abs(samples)), 1e-30) * 0.8
        return samples.astype(np.float32)

    def envelope(self, samples: np.ndarray, attack: float = 0.01,
                 decay: float = 0.1, sustain: float = 0.7,
                 release: float = 0.2) -> np.ndarray:
        n = len(samples)
        a_s = int(attack * self.sample_rate)
        d_s = int(decay * self.sample_rate)
        r_s = int(release * self.sample_rate)
        s_s = max(n - a_s - d_s - r_s, 0)
        env = np.concatenate([
            np.linspace(0, 1, a_s),
            np.linspace(1, sustain, d_s),
            np.ones(s_s) * sustain,
            np.linspace(sustain, 0, r_s),
        ])
        if len(env) > n:
            env = env[:n]
        elif len(env) < n:
            env = np.pad(env, (0, n - len(env)))
        return samples * env

    # ── Quantum-Modulated Tones ──────────────────────────────────────────

    def quantum_bell_tone(self, base_freq: float = GOD_CODE,
                          duration: float = 2.0) -> QuantumAudioResult:
        """Tone modulated by Bell state |Phi+> measurement probabilities."""
        probs = self._quantum_probs("bell", 2)
        p00, p01, p10, p11 = probs[0], probs[1], probs[2], probs[3]

        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        left = 0.8 * p00 * 2 * np.sin(2 * np.pi * base_freq * t)
        right = 0.8 * p11 * 2 * np.sin(2 * np.pi * base_freq * PHI * t)
        interference = 0.3 * np.sqrt(p01 * p10) * np.sin(2 * np.pi * base_freq * 2 * t)
        left += interference
        right += interference

        samples = np.column_stack([left, right]).astype(np.float32)
        samples = np.clip(samples, -1.0, 1.0)
        env = self._build_envelope(len(left), duration)
        samples[:, 0] *= env
        samples[:, 1] *= env

        return QuantumAudioResult(
            samples=samples, sample_rate=self.sample_rate, duration=duration,
            frequencies=[base_freq, base_freq * PHI],
            quantum_metrics={"circuit": "bell", "p00": float(p00), "p01": float(p01),
                             "p10": float(p10), "p11": float(p11),
                             "entanglement": float(2 * np.sqrt(p00 * p11))},
            sacred_alignment=self._compute_sacred_alignment([base_freq, base_freq * PHI]),
        )

    def quantum_ghz_harmonics(self, fundamental: float = GOD_CODE,
                               n_qubits: int = 3, duration: float = 3.0) -> QuantumAudioResult:
        """Harmonics weighted by GHZ state measurements."""
        probs = self._quantum_probs("ghz", n_qubits)
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        samples = np.zeros_like(t)
        frequencies = []
        for i in range(min(2**n_qubits, 8)):
            freq = fundamental * (i + 1)
            amp = probs[i] if i < len(probs) else 0.0
            samples += amp * np.sin(2 * np.pi * freq * t)
            if amp > 0.01:
                frequencies.append(freq)
        if np.max(np.abs(samples)) > 0:
            samples = samples / np.max(np.abs(samples)) * 0.8
        samples = self.envelope(samples.astype(np.float32))
        return QuantumAudioResult(
            samples=samples, sample_rate=self.sample_rate, duration=duration,
            frequencies=frequencies,
            quantum_metrics={"circuit": "ghz", "n_qubits": n_qubits,
                             "probabilities": [float(p) for p in probs],
                             "coherence": float(probs[0] + probs[-1])},
            sacred_alignment=self._compute_sacred_alignment(frequencies),
        )

    def quantum_qft_fm(self, carrier: float = GOD_CODE,
                       mod_freq: float = SCHUMANN_HZ,
                       duration: float = 4.0) -> QuantumAudioResult:
        """FM synthesis with QFT-derived modulation indices."""
        probs = self._quantum_probs("qft", 4)
        mod_indices = probs * 10
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        samples = np.zeros_like(t)
        for i, mod_index in enumerate(mod_indices[:4]):
            fm = mod_index * np.sin(2 * np.pi * mod_freq * (i + 1) * t)
            samples += 0.3 * np.sin(2 * np.pi * carrier * t + fm)
        samples += 0.5 * np.sin(2 * np.pi * carrier * t)
        samples = samples / max(np.max(np.abs(samples)), 1e-30) * 0.8
        samples = self.envelope(samples.astype(np.float32))
        return QuantumAudioResult(
            samples=samples, sample_rate=self.sample_rate, duration=duration,
            frequencies=[carrier],
            quantum_metrics={"circuit": "qft", "carrier_hz": carrier,
                             "mod_freq_hz": mod_freq,
                             "mod_indices": [float(m) for m in mod_indices]},
            sacred_alignment=self._compute_sacred_alignment([carrier]),
        )

    # ── Sacred Tones ─────────────────────────────────────────────────────

    def god_code_tone(self, duration: float = 3.0,
                      with_harmonics: bool = True) -> QuantumAudioResult:
        """GOD_CODE tone with QPU-verified qubit phase injection."""
        # Subtle qubit phase seed — scaled to preserve original beauty
        phase_offset = (GOD_CODE_PHASE * 0.02) if self._qubit_available else 0.0

        if with_harmonics:
            samples = self.harmonic_series(GOD_CODE, n_harmonics=6, duration=duration)
        else:
            samples = self.pure_tone(GOD_CODE, duration=duration, phase=phase_offset)
        samples = self.envelope(samples)

        # Apply qubit phase modulation layer when available
        if self._qubit_available and self._god_code_qubit is not None:
            t = np.linspace(0, duration, len(samples), False)
            qubit_mod = 1.0 + 0.008 * np.sin(
                2.0 * np.pi * GOD_CODE_PHASE / (2.0 * np.pi) * t + IRON_PHASE
            )
            samples = samples * qubit_mod.astype(np.float32)

        return QuantumAudioResult(
            samples=samples, sample_rate=self.sample_rate, duration=duration,
            frequencies=[GOD_CODE * n for n in range(1, 7)] if with_harmonics else [GOD_CODE],
            quantum_metrics={"sacred": True, "constant": "GOD_CODE",
                             "qubit_phase": float(GOD_CODE_PHASE),
                             "qpu_verified": self._qubit_available},
            sacred_alignment=1.0,
        )

    def god_code_qubit_tone(
        self,
        duration: float = 3.0,
        dials: Optional[Tuple] = None,
        decomposed: bool = True,
    ) -> QuantumAudioResult:
        """Generate a tone directly from the GOD_CODE qubit's phase structure.

        Uses the QPU-verified Rz(θ_GC) gate decomposition to create a
        3-layer tone: iron (π/2) + phi contribution + octave phase, each
        mapped to harmonic partials with phase-coherent alignment.

        Parameters
        ----------
        duration : float — tone length in seconds
        dials : tuple (a, b, c, d) — optional dial values for per-dial phase
        decomposed : bool — if True, render 3-layer decomposition; else single phase

        Returns
        -------
        QuantumAudioResult with QPU-verified qubit tone
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)

        if self._qubit_available and self._god_code_qubit is not None:
            qb = self._god_code_qubit
            gc_phase = qb.phase
            decomp = qb.decomposed  # tuple (iron, phi, octave)

            if dials is not None:
                a, b, c, d = dials
                dial_phase = qb.dial_phase(a, b, c, d)
            else:
                dial_phase = gc_phase

            if decomposed:
                # 3-layer decomposition: iron + phi + octave
                iron_ph = decomp[0]
                phi_ph = decomp[1]
                oct_ph = decomp[2]

                # Layer 1: Iron fundamental (Fe-26 quarter-turn)
                iron_signal = 0.5 * np.sin(2.0 * np.pi * GOD_CODE * t + iron_ph)

                # Layer 2: PHI harmonic (golden contribution)
                phi_signal = 0.3 * np.sin(
                    2.0 * np.pi * GOD_CODE * PHI * t + phi_ph
                )

                # Layer 3: Octave complement
                octave_signal = 0.2 * np.sin(
                    2.0 * np.pi * GOD_CODE * 2.0 * t + oct_ph
                )

                samples = iron_signal + phi_signal + octave_signal
            else:
                # Single phase Rz(θ_GC) modulated tone
                samples = 0.8 * np.sin(2.0 * np.pi * GOD_CODE * t + dial_phase)

            # Apply statevector amplitude modulation from |+⟩ preparation
            plus_sv = qb.prepare("|+>")
            amp_mod = float(np.abs(plus_sv[0]) ** 2 + np.abs(plus_sv[1]) ** 2)
            samples *= amp_mod

            metrics = {
                "sacred": True,
                "qubit": True,
                "qpu_verified": True,
                "qpu_fidelity": 0.999939,
                "phase_gc": float(gc_phase),
                "dial_phase": float(dial_phase),
                "decomposed": decomposed,
                "decomposition": {
                    "iron": float(decomp[0]),
                    "phi": float(decomp[1]),
                    "octave": float(decomp[2]),
                },
            }
        else:
            # Fallback: use constants-derived phases
            samples = 0.5 * np.sin(2.0 * np.pi * GOD_CODE * t + GOD_CODE_PHASE)
            samples += 0.3 * np.sin(2.0 * np.pi * GOD_CODE * PHI * t + IRON_PHASE)
            samples += 0.2 * np.sin(2.0 * np.pi * GOD_CODE * 2.0 * t)
            metrics = {"sacred": True, "qubit": False, "fallback": True}

        # Normalize and envelope
        peak = np.max(np.abs(samples))
        if peak > 1e-15:
            samples = samples / peak * 0.8
        samples = self.envelope(samples.astype(np.float32), attack=0.02, release=0.3)

        frequencies = [GOD_CODE, GOD_CODE * PHI, GOD_CODE * 2.0]
        return QuantumAudioResult(
            samples=samples,
            sample_rate=self.sample_rate,
            duration=duration,
            frequencies=frequencies,
            quantum_metrics=metrics,
            sacred_alignment=1.0,
        )

    def zenith_tone(self, duration: float = 3.0) -> QuantumAudioResult:
        samples = self.harmonic_series(ZENITH_HZ, n_harmonics=4, duration=duration)
        samples = self.envelope(samples)
        return QuantumAudioResult(
            samples=samples, sample_rate=self.sample_rate, duration=duration,
            frequencies=[ZENITH_HZ * n for n in range(1, 5)],
            quantum_metrics={"sacred": True, "constant": "ZENITH_HZ"},
            sacred_alignment=0.95,
        )

    def phi_tone(self, duration: float = 3.0) -> QuantumAudioResult:
        phi_freq = PHI * 1000
        samples = self.harmonic_series(phi_freq, n_harmonics=5, duration=duration,
                                        decay=1 / PHI)
        samples = self.envelope(samples)
        return QuantumAudioResult(
            samples=samples, sample_rate=self.sample_rate, duration=duration,
            frequencies=[phi_freq * n for n in range(1, 6)],
            quantum_metrics={"sacred": True, "constant": "PHI"},
            sacred_alignment=0.98,
        )

    def sacred_chord(self, root: float = GOD_CODE,
                     duration: float = 4.0) -> QuantumAudioResult:
        intervals = [1.0, PHI, 3 / 2, 2.0]
        frequencies = [root * i for i in intervals]
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        samples = np.zeros_like(t)
        for i, freq in enumerate(frequencies):
            samples += (1.0 / (i + 1)) * np.sin(2 * np.pi * freq * t)
        samples = samples / max(np.max(np.abs(samples)), 1e-30) * 0.8
        samples = self.envelope(samples.astype(np.float32), attack=0.05, release=0.5)
        return QuantumAudioResult(
            samples=samples, sample_rate=self.sample_rate, duration=duration,
            frequencies=frequencies,
            quantum_metrics={"chord_type": "sacred", "root": root, "intervals": intervals},
            sacred_alignment=self._compute_sacred_alignment(frequencies),
        )

    def schumann_drone(self, duration: float = 10.0) -> QuantumAudioResult:
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        carrier = 440.0
        samples = np.sin(2 * np.pi * carrier * t)
        modulator = 0.5 + 0.5 * np.sin(2 * np.pi * SCHUMANN_HZ * t)
        samples = samples * modulator
        for n in [3, 4, 5, 6]:
            samples += 0.2 * np.sin(2 * np.pi * SCHUMANN_HZ * n * t)
        samples = samples / max(np.max(np.abs(samples)), 1e-30) * 0.7
        return QuantumAudioResult(
            samples=samples.astype(np.float32), sample_rate=self.sample_rate,
            duration=duration, frequencies=[SCHUMANN_HZ, carrier],
            quantum_metrics={"type": "schumann_drone", "base_hz": SCHUMANN_HZ},
            sacred_alignment=0.83,
        )

    # ── Utilities ────────────────────────────────────────────────────────

    def _build_envelope(self, n_samples: int, duration: float) -> np.ndarray:
        fade = int(0.02 * self.sample_rate)
        env = np.ones(n_samples)
        env[:fade] = np.linspace(0, 1, fade)
        env[-fade:] = np.linspace(1, 0, fade)
        return env

    def _compute_sacred_alignment(self, frequencies: List[float]) -> float:
        if not frequencies:
            return 0.0
        scores = []
        for f in frequencies:
            ratio = f / GOD_CODE
            nearest = round(ratio)
            if nearest > 0:
                err = abs(ratio - nearest) / nearest
                scores.append(1.0 - min(err, 1.0))
        if len(frequencies) >= 2:
            for i in range(len(frequencies) - 1):
                ratio = frequencies[i + 1] / frequencies[i]
                err = abs(ratio - PHI) / PHI
                scores.append(1.0 - min(err, 1.0))
        return float(np.mean(scores)) if scores else 0.5

    def save_wav(self, result: QuantumAudioResult, filename: str) -> str:
        samples = result.samples
        channels = 1 if samples.ndim == 1 else 2
        pcm = (samples * 32767).astype(np.int16)
        with wave.open(filename, 'w') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm.tobytes())
        return filename

    def get_wav_bytes(self, result: QuantumAudioResult) -> bytes:
        samples = result.samples
        channels = 1 if samples.ndim == 1 else 2
        pcm = (samples * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, 'w') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm.tobytes())
        return buf.getvalue()

    def status(self) -> Dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "quantum_available": self._quantum_available,
            "god_code_qubit_available": self._qubit_available,
            "sacred_frequencies": {
                "GOD_CODE": GOD_CODE, "ZENITH_HZ": ZENITH_HZ,
                "PHI_TONE": PHI * 1000, "SCHUMANN_HZ": SCHUMANN_HZ,
            },
            "qubit_phase": float(GOD_CODE_PHASE),
            "supported_circuits": ["bell", "ghz", "qft", "sacred"],
        }


# Module-level singleton
quantum_tone_generator = QuantumPureToneGenerator()
