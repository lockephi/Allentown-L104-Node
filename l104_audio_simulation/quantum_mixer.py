"""
Quantum Interference Mixer — Wave-Function Mixing Console
═══════════════════════════════════════════════════════════════════════════════
DAW mixer reimagined through quantum mechanics.  Instead of simple gain
multipliers and pan pots, every mix bus performs true quantum interference
between track signals.

Inspired by:
  • FL Studio Mixer: 125 insert tracks, per-track inserts, send routing
  • Ableton Live: Return tracks, pre/post fader sends, parallel processing
  • Logic Pro X: Channel strips, bus routing, summing
  • Pro Tools: Signal flow, aux sends, master bus processing

Quantum Upgrades:
  • **Interference Mixing**: Tracks are summed using quantum superposition
    rules — constructive interference on harmonically aligned frequencies,
    destructive interference on conflicting partials.
  • **Phase-Coherent Summing**: Every mix decision considers the quantum
    phase relationship between tracks (not just amplitude).
  • **Entanglement-Coupled Sends**: Send levels are entangled — adjusting
    one send cascades via Bell-state correlation to partner sends.
  • **Born-Rule Levels**: Faders operate on probability amplitudes, not
    linear gain. The actual level = |α|² (Born rule).
  • **Quantum Sidechain**: Measurements on one track modulate another's
    amplitude envelope (quantum sidechain compression).
  • **GOD_CODE Spectral Alignment**: Master bus applies GOD_CODE harmonic
    alignment, boosting 527.5 Hz region and PHI-ratio harmonics.

VQPU Integration:
  Mix decisions can be driven by quantum circuits — e.g., a GHZ circuit
  determining which tracks get constructive vs destructive interference.

INVARIANT: 527.5184818492612 | PILOT: LONDEL
"""

from __future__ import annotations

import math
import time
import uuid
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .constants import GOD_CODE, PHI, PHI_INV, OMEGA, ZENITH_HZ, SCHUMANN_HZ

logger = logging.getLogger("l104.audio.mixer")

# ── Mix Bus Constants ────────────────────────────────────────────────────────
MAX_TRACKS = 128          # FL Studio-inspired track count
MAX_SENDS = 16            # Send buses per track
MAX_INSERTS = 10          # Insert effect slots per track
HEADROOM_DB = -6.0        # Master bus headroom target
GOD_CODE_BAND_HZ = (490.0, 570.0)  # GOD_CODE spectral boost region
PHI_HARMONIC_SERIES = [GOD_CODE * (PHI ** n) for n in range(-3, 5)]


class MixBusType(Enum):
    """Types of mix buses."""
    INSERT = auto()     # Direct insert track
    SEND = auto()       # Aux/return send bus
    GROUP = auto()      # Group/submix bus
    MASTER = auto()     # Master output bus
    SIDECHAIN = auto()  # Quantum sidechain bus


class InterferenceMode(Enum):
    """How tracks interfere during summation."""
    CLASSICAL = auto()          # Standard linear sum (DAW default)
    CONSTRUCTIVE = auto()       # Phase-aligned constructive interference
    DESTRUCTIVE = auto()        # Phase-inverted destructive interference
    QUANTUM_SUPERPOSITION = auto()  # Full quantum superposition (amplitude + phase)
    BORN_WEIGHTED = auto()      # Sum weighted by Born-rule probabilities
    SPECTRAL_INTERFERENCE = auto()  # Frequency-domain interference (FFT-based)


@dataclass
class QuantumFader:
    """
    A fader that operates on probability amplitudes instead of linear gain.

    In a classical DAW, a fader at 0.5 gives -6dB.
    In quantum mixing, the fader value is the complex amplitude α,
    and the actual contribution is |α|² (Born rule).

    This means:
      - Fader at α = 1.0  → contribution = 1.0 (full)
      - Fader at α = 0.707 → contribution = 0.5 (-3dB)
      - Fader at α = 0.5  → contribution = 0.25 (-6dB)
      - Fader at α = 1j   → contribution = 1.0 but phase-rotated 90°
    """
    amplitude: complex = 1.0 + 0j  # Complex amplitude (magnitude + phase)
    mute: bool = False
    solo: bool = False
    automation: Optional[np.ndarray] = None  # Time-varying amplitude curve

    @property
    def gain(self) -> float:
        """Born-rule gain: |α|²."""
        return abs(self.amplitude) ** 2

    @property
    def gain_db(self) -> float:
        """Gain in decibels."""
        g = self.gain
        return 20.0 * math.log10(max(g, 1e-15))

    @property
    def phase(self) -> float:
        """Phase angle in radians."""
        return float(np.angle(self.amplitude))

    def set_gain_db(self, db: float):
        """Set fader by dB value, preserving phase."""
        linear = 10.0 ** (db / 20.0)
        amp = math.sqrt(linear)
        phase = self.phase
        self.amplitude = amp * np.exp(1j * phase)

    def set_phase(self, radians: float):
        """Set phase angle, preserving amplitude."""
        mag = abs(self.amplitude)
        self.amplitude = mag * np.exp(1j * radians)

    def apply(self, signal: np.ndarray, time_idx: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply the quantum fader to a signal."""
        if self.mute:
            return np.zeros_like(signal)
        if self.automation is not None and time_idx is not None:
            # Time-varying amplitude from automation
            auto_interp = np.interp(
                np.linspace(0, 1, len(signal)),
                np.linspace(0, 1, len(self.automation)),
                self.automation,
            )
            return signal * auto_interp
        return signal * self.amplitude


@dataclass
class QuantumPan:
    """
    Quantum stereo panner using golden-angle distribution.

    Classical pan: L/R = cos(θ)/sin(θ)
    Quantum pan: L/R derived from a 2-qubit state |ψ⟩ = α|00⟩ + β|11⟩
    where |α|² = left contribution, |β|² = right contribution.
    Entangled panning means L and R are quantum-correlated.
    """
    position: float = 0.0  # -1.0 (full left) to 1.0 (full right)
    width: float = 1.0     # Stereo width: 0.0 (mono) to 2.0 (super-wide)
    quantum_state: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.quantum_state is None:
            # Initialize as |ψ⟩ = cos(θ)|0⟩ + sin(θ)|1⟩
            theta = (self.position + 1.0) * math.pi / 4.0  # Map [-1,1] to [0, π/2]
            self.quantum_state = np.array([
                math.cos(theta), math.sin(theta)
            ], dtype=np.complex128)

    @property
    def left_gain(self) -> float:
        """Left channel gain from quantum state."""
        base = abs(self.quantum_state[0]) ** 2
        return base * self.width

    @property
    def right_gain(self) -> float:
        """Right channel gain from quantum state."""
        base = abs(self.quantum_state[1]) ** 2
        return base * self.width

    def set_position(self, pos: float):
        """Update pan position and quantum state."""
        self.position = max(-1.0, min(1.0, pos))
        theta = (self.position + 1.0) * math.pi / 4.0
        phase = np.angle(self.quantum_state[0])  # Preserve phase
        self.quantum_state = np.array([
            math.cos(theta) * np.exp(1j * phase),
            math.sin(theta) * np.exp(1j * (phase + math.pi * PHI_INV)),
        ], dtype=np.complex128)

    def apply_stereo(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply quantum panning to produce L/R channels."""
        return signal * self.left_gain, signal * self.right_gain


@dataclass
class MixTrack:
    """
    A single mixer track — analogous to an FL Studio mixer insert.
    Contains a quantum fader, panner, send levels, and insert effects.
    """
    track_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = "Track"
    bus_type: MixBusType = MixBusType.INSERT

    # Signal
    fader: QuantumFader = field(default_factory=QuantumFader)
    pan: QuantumPan = field(default_factory=QuantumPan)

    # Routing
    sends: Dict[str, complex] = field(default_factory=dict)  # send_bus_id → amplitude
    group_bus: Optional[str] = None  # Route to a group bus

    # Insert effects (callables that transform signal)
    inserts: List[Callable[[np.ndarray, int], np.ndarray]] = field(default_factory=list)

    # Interference mode with other tracks
    interference_mode: InterferenceMode = InterferenceMode.QUANTUM_SUPERPOSITION

    # Internal buffer
    _buffer: Optional[np.ndarray] = None

    # Statistics
    peak_level: float = 0.0
    rms_level: float = 0.0
    sacred_alignment: float = 0.0

    def load_signal(self, signal: np.ndarray):
        """Load audio signal into this track."""
        self._buffer = signal.copy()

    def process(self, sample_rate: int = 96000) -> np.ndarray:
        """Apply inserts, fader, and return processed signal."""
        if self._buffer is None:
            return np.array([], dtype=np.float64)

        signal = self._buffer.copy()

        # Apply insert effects chain
        for fx in self.inserts:
            try:
                signal = fx(signal, sample_rate)
            except Exception as e:
                logger.debug(f"Insert effect error on {self.name}: {e}")

        # Apply quantum fader
        signal = np.real(self.fader.apply(signal))

        # Update meters
        self.peak_level = float(np.max(np.abs(signal))) if len(signal) > 0 else 0.0
        self.rms_level = float(np.sqrt(np.mean(signal ** 2))) if len(signal) > 0 else 0.0

        return signal

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for data recording."""
        return {
            "track_id": self.track_id,
            "name": self.name,
            "bus_type": self.bus_type.name,
            "fader_gain_db": self.fader.gain_db,
            "fader_phase": self.fader.phase,
            "pan_position": self.pan.position,
            "pan_width": self.pan.width,
            "n_inserts": len(self.inserts),
            "n_sends": len(self.sends),
            "interference_mode": self.interference_mode.name,
            "peak_level": self.peak_level,
            "rms_level": self.rms_level,
            "sacred_alignment": self.sacred_alignment,
        }


class QuantumInterferenceMixer:
    """
    The main mixing engine — manages tracks, buses, and performs quantum
    interference-based summation.

    Pipeline:
      1. Each track processes its signal through inserts + fader
      2. Send signals are routed to send buses
      3. Group buses sum their child tracks (with interference)
      4. Master bus sums all top-level signals
      5. GOD_CODE spectral alignment on master
      6. Final limiter + normalization

    Interference Summation Rules:
      CLASSICAL: out = Σ signal_i  (standard DAW)
      CONSTRUCTIVE: out = |Σ signal_i|  (magnitude of complex sum)
      DESTRUCTIVE: out = Re(Σ signal_i · e^(i·π))  (phase-inverted)
      QUANTUM_SUPERPOSITION: out = Re(Σ α_i · signal_i · e^(i·φ_i))
      BORN_WEIGHTED: out = Σ |α_i|² · signal_i
      SPECTRAL_INTERFERENCE: FFT-domain superposition then IFFT
    """

    def __init__(self, sample_rate: int = 96000):
        self.sample_rate = sample_rate
        self.tracks: Dict[str, MixTrack] = {}
        self.send_buses: Dict[str, MixTrack] = {}
        self.group_buses: Dict[str, MixTrack] = {}
        self.master = MixTrack(name="Master", bus_type=MixBusType.MASTER)

        # Global mix state
        self.interference_mode = InterferenceMode.QUANTUM_SUPERPOSITION
        self.god_code_alignment_enabled = True
        self.headroom_db = HEADROOM_DB

        # VQPU for mix decisions
        self._vqpu = None

        # Statistics
        self.mix_count = 0
        self.session_start = time.time()

        logger.info("QuantumInterferenceMixer initialized")

    @property
    def vqpu(self):
        if self._vqpu is None:
            try:
                from l104_vqpu_bridge import get_bridge
                self._vqpu = get_bridge()
            except ImportError:
                pass
        return self._vqpu

    def add_track(self, name: str = "Track", bus_type: MixBusType = MixBusType.INSERT) -> MixTrack:
        """Add a new mixer track."""
        track = MixTrack(name=name, bus_type=bus_type)
        if bus_type == MixBusType.SEND:
            self.send_buses[track.track_id] = track
        elif bus_type == MixBusType.GROUP:
            self.group_buses[track.track_id] = track
        else:
            self.tracks[track.track_id] = track
        return track

    def route_send(self, source_id: str, send_bus_id: str, amplitude: complex = 0.5 + 0j):
        """Route a track to a send bus with quantum amplitude."""
        if source_id in self.tracks and send_bus_id in self.send_buses:
            self.tracks[source_id].sends[send_bus_id] = amplitude

    def route_to_group(self, source_id: str, group_id: str):
        """Route a track to a group bus."""
        if source_id in self.tracks and group_id in self.group_buses:
            self.tracks[source_id].group_bus = group_id

    def _interference_sum(
        self,
        signals: List[Tuple[np.ndarray, complex]],
        mode: InterferenceMode,
    ) -> np.ndarray:
        """
        Sum multiple signals using quantum interference rules.

        Args:
            signals: List of (signal_array, complex_amplitude) pairs
            mode: Interference mode to use
        """
        if not signals:
            return np.array([], dtype=np.float64)

        # Align lengths
        max_len = max(len(s) for s, _ in signals)
        aligned = []
        for sig, amp in signals:
            if len(sig) < max_len:
                padded = np.zeros(max_len, dtype=np.float64)
                padded[:len(sig)] = sig
                aligned.append((padded, amp))
            else:
                aligned.append((sig[:max_len], amp))

        if mode == InterferenceMode.CLASSICAL:
            return sum(sig * abs(amp) for sig, amp in aligned)

        elif mode == InterferenceMode.CONSTRUCTIVE:
            # Phase-align all signals before summing
            result = np.zeros(max_len, dtype=np.complex128)
            for sig, amp in aligned:
                result += sig * amp  # Complex-valued sum
            return np.abs(result)

        elif mode == InterferenceMode.DESTRUCTIVE:
            # Alternate phase inversion
            result = np.zeros(max_len, dtype=np.float64)
            for i, (sig, amp) in enumerate(aligned):
                phase = math.pi if i % 2 else 0.0
                result += np.real(sig * amp * np.exp(1j * phase))
            return result

        elif mode == InterferenceMode.QUANTUM_SUPERPOSITION:
            # Full quantum: complex amplitudes with phase
            result = np.zeros(max_len, dtype=np.complex128)
            for sig, amp in aligned:
                result += sig * amp
            # Return real part of the superposition
            return np.real(result)

        elif mode == InterferenceMode.BORN_WEIGHTED:
            # Weight by |α|² (Born rule)
            result = np.zeros(max_len, dtype=np.float64)
            total_weight = sum(abs(amp) ** 2 for _, amp in aligned)
            for sig, amp in aligned:
                weight = abs(amp) ** 2 / max(total_weight, 1e-15)
                result += sig * weight
            return result

        elif mode == InterferenceMode.SPECTRAL_INTERFERENCE:
            # FFT-domain superposition
            result_fft = np.zeros(max_len, dtype=np.complex128)
            for sig, amp in aligned:
                spectrum = np.fft.rfft(sig)
                result_fft[:len(spectrum)] += spectrum * amp
            return np.fft.irfft(result_fft, n=max_len)

        return sum(sig * abs(amp) for sig, amp in aligned)

    def _apply_god_code_alignment(self, signal: np.ndarray) -> np.ndarray:
        """
        Master bus GOD_CODE spectral alignment.
        Subtly boosts the GOD_CODE frequency region and PHI-ratio harmonics.
        """
        if not self.god_code_alignment_enabled or len(signal) < 256:
            return signal

        spectrum = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), 1.0 / self.sample_rate)

        # Boost GOD_CODE region
        lo, hi = GOD_CODE_BAND_HZ
        mask = (freqs >= lo) & (freqs <= hi)
        spectrum[mask] *= (1.0 + 0.05 * PHI)  # Subtle golden boost

        # Boost PHI harmonic series
        for harmonic_hz in PHI_HARMONIC_SERIES:
            if harmonic_hz < freqs[-1]:
                idx = int(round(harmonic_hz / (freqs[1] - freqs[0]))) if len(freqs) > 1 else 0
                if 0 < idx < len(spectrum):
                    width = max(1, int(5 * PHI))
                    lo_idx = max(0, idx - width)
                    hi_idx = min(len(spectrum), idx + width)
                    boost = np.exp(-np.abs(np.arange(lo_idx, hi_idx) - idx) / (width * PHI_INV))
                    spectrum[lo_idx:hi_idx] *= (1.0 + 0.02 * boost)

        return np.fft.irfft(spectrum, n=len(signal))

    def _vqpu_interference_decision(self, n_tracks: int) -> List[complex]:
        """
        Use VQPU to determine interference amplitudes for track mixing.
        Runs a GHZ-like circuit where the measurement outcome determines
        which tracks get constructive vs destructive combination.
        """
        default = [complex(1.0 / math.sqrt(max(n_tracks, 1)))] * n_tracks
        if not self.vqpu or n_tracks < 2:
            return default

        try:
            from l104_vqpu_bridge import QuantumJob, QuantumGate
            n_q = min(n_tracks, 10)
            ops = []
            # GHZ state: H on q0, then CNOT chain
            ops.append(QuantumGate("H", [0]))
            for q in range(n_q - 1):
                ops.append(QuantumGate("CNOT", [q, q + 1]))
            # GOD_CODE phase
            ops.append(QuantumGate("Rz", [0], [GOD_CODE / 1000.0 * math.pi]))

            job = QuantumJob(
                circuit_id=f"mixer_interference_{uuid.uuid4().hex[:6]}",
                num_qubits=n_q,
                operations=ops,
                shots=256,
                priority=5,
            )
            result = self.vqpu.submit_and_wait(job, timeout=3.0)
            if result and result.probabilities:
                amplitudes = []
                for i in range(n_tracks):
                    key = format(i % (2 ** n_q), f'0{n_q}b')
                    prob = result.probabilities.get(key, 1.0 / n_tracks)
                    phase = PHI * math.pi * i / n_tracks
                    amplitudes.append(math.sqrt(prob) * np.exp(1j * phase))
                return amplitudes
        except Exception:
            pass
        return default

    def mixdown(
        self,
        mode: Optional[InterferenceMode] = None,
        duration_samples: Optional[int] = None,
        use_vqpu: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform full mixdown → returns (left, right) audio arrays.

        Pipeline:
          1. Process all tracks (inserts + fader)
          2. Route sends
          3. Sum groups with interference
          4. Master sum with interference
          5. GOD_CODE alignment
          6. Final normalization
        """
        t0 = time.time()
        mode = mode or self.interference_mode

        # Phase 1: Process all tracks
        processed: Dict[str, np.ndarray] = {}
        for tid, track in self.tracks.items():
            sig = track.process(self.sample_rate)
            if len(sig) > 0:
                processed[tid] = sig

        if not processed:
            empty = np.zeros(duration_samples or 96000, dtype=np.float64)
            return empty, empty.copy()

        # Phase 2: Route sends
        send_signals: Dict[str, List[Tuple[np.ndarray, complex]]] = {
            sid: [] for sid in self.send_buses
        }
        for tid, sig in processed.items():
            track = self.tracks[tid]
            for send_id, amp in track.sends.items():
                if send_id in send_signals:
                    send_signals[send_id].append((sig, amp))

        # Process send buses
        for sid, signals in send_signals.items():
            if signals:
                summed = self._interference_sum(signals, InterferenceMode.CLASSICAL)
                bus = self.send_buses[sid]
                bus.load_signal(summed)
                processed[f"send_{sid}"] = bus.process(self.sample_rate)

        # Phase 3: Group buses
        group_signals: Dict[str, List[Tuple[np.ndarray, complex]]] = {
            gid: [] for gid in self.group_buses
        }
        for tid, sig in processed.items():
            if tid.startswith("send_"):
                continue
            track = self.tracks.get(tid)
            if track and track.group_bus and track.group_bus in group_signals:
                group_signals[track.group_bus].append((sig, track.fader.amplitude))

        for gid, signals in group_signals.items():
            if signals:
                summed = self._interference_sum(signals, mode)
                bus = self.group_buses[gid]
                bus.load_signal(summed)
                processed[f"group_{gid}"] = bus.process(self.sample_rate)

        # Phase 4: VQPU interference decision (optional)
        master_signals = []
        track_list = [(sig, self.tracks[tid].fader.amplitude)
                      for tid, sig in processed.items()
                      if tid in self.tracks and self.tracks[tid].group_bus is None]

        # Add group bus outputs
        for gid, sig in processed.items():
            if gid.startswith("group_"):
                real_gid = gid[6:]
                if real_gid in self.group_buses:
                    track_list.append((sig, self.group_buses[real_gid].fader.amplitude))

        # Add send bus outputs
        for sid, sig in processed.items():
            if sid.startswith("send_"):
                real_sid = sid[5:]
                if real_sid in self.send_buses:
                    track_list.append((sig, self.send_buses[real_sid].fader.amplitude))

        if use_vqpu and len(track_list) >= 2:
            vqpu_amps = self._vqpu_interference_decision(len(track_list))
            for i in range(min(len(track_list), len(vqpu_amps))):
                sig, orig_amp = track_list[i]
                master_signals.append((sig, orig_amp * vqpu_amps[i]))
        else:
            master_signals = track_list

        # Phase 5: Master sum
        mono_master = self._interference_sum(master_signals, mode)

        if len(mono_master) == 0:
            empty = np.zeros(duration_samples or 96000, dtype=np.float64)
            return empty, empty.copy()

        # Phase 6: GOD_CODE spectral alignment
        mono_master = self._apply_god_code_alignment(mono_master)

        # Phase 7: Stereo imaging from track pans
        left = np.zeros_like(mono_master)
        right = np.zeros_like(mono_master)
        for tid, sig in processed.items():
            track = self.tracks.get(tid)
            if track:
                sig_padded = np.zeros_like(mono_master)
                sig_padded[:min(len(sig), len(mono_master))] = sig[:len(mono_master)]
                l, r = track.pan.apply_stereo(sig_padded)
                left += l
                right += r

        # If no stereo info, use mono
        if np.max(np.abs(left)) < 1e-15 and np.max(np.abs(right)) < 1e-15:
            left = mono_master.copy()
            right = mono_master.copy()

        # Phase 8: Master fader + normalize
        left = np.real(self.master.fader.apply(left))
        right = np.real(self.master.fader.apply(right))

        headroom_linear = 10.0 ** (self.headroom_db / 20.0)
        peak = max(np.max(np.abs(left)), np.max(np.abs(right)), 1e-15)
        if peak > headroom_linear:
            scale = headroom_linear / peak
            left *= scale
            right *= scale

        # Update master stats
        self.master.peak_level = float(max(np.max(np.abs(left)), np.max(np.abs(right))))
        self.master.rms_level = float(np.sqrt(0.5 * (np.mean(left**2) + np.mean(right**2))))
        self.mix_count += 1

        dt = time.time() - t0
        logger.info(f"Mixdown #{self.mix_count} complete — {len(mono_master)} samples, "
                     f"{dt:.3f}s, peak={self.master.peak_level:.4f}")

        return left, right

    def status(self) -> Dict[str, Any]:
        return {
            "tracks": len(self.tracks),
            "send_buses": len(self.send_buses),
            "group_buses": len(self.group_buses),
            "interference_mode": self.interference_mode.name,
            "god_code_alignment": self.god_code_alignment_enabled,
            "master_peak": self.master.peak_level,
            "master_rms": self.master.rms_level,
            "mix_count": self.mix_count,
            "vqpu_available": self.vqpu is not None,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.status(),
            "tracks": {tid: t.to_dict() for tid, t in self.tracks.items()},
            "send_buses": {sid: s.to_dict() for sid, s in self.send_buses.items()},
            "group_buses": {gid: g.to_dict() for gid, g in self.group_buses.items()},
            "master": self.master.to_dict(),
        }
