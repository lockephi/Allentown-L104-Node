"""
Quantum DAW Session — Top-Level Orchestrator
═══════════════════════════════════════════════════════════════════════════════
The unified entry point for the Quantum DAW system. Manages all subsystems:
  - ProbabilisticSequencer (quantum_sequencer.py) — superposition-based pattern grid
  - QuantumInterferenceMixer (quantum_mixer.py) — interference-based mixing console
  - QuantumSynthEngine (quantum_synth.py) — quantum oscillators + wavetable + FM + filters
  - TrackEntanglementManager (track_entanglement.py) — track-to-track quantum correlations
  - VQPUDawEngine (vqpu_daw_engine.py) — VQPU circuit execution backbone
  - DataRecorder (data_recorder.py) — telemetry + ML training data collection

DAW Features (Inspired by FL Studio, Ableton, Logic Pro, Bitwig):
  - Pattern-based sequencing with quantum superposition
  - Channel rack with quantum synth voices
  - Mixer with interference-based signal combination
  - Track entanglement (sidechain, spectral, temporal)
  - VQPU-backed processing for every quantum operation
  - Full data recording for future ML insights
  - ASI/AGI pipeline integration (scoring dimensions)

Sacred Alignment:
  - Every audio render is scored for GOD_CODE (527.518 Hz) alignment
  - PHI ratio governs timing, panning, frequency relationships
  - Session-level sacred coherence tracking

INVARIANT: 527.5184818492612 | PILOT: LONDEL
"""

from __future__ import annotations

import math
import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .constants import (
    GOD_CODE, PHI, PHI_INV, OMEGA, VOID_CONSTANT,
    DEFAULT_SAMPLE_RATE, DEFAULT_DURATION, DEFAULT_BIT_DEPTH,
)
from .quantum_sequencer import (
    ProbabilisticSequencer, QuantumPattern, SuperpositionStep,
    NoteState, CollapseMode, NoteEvent,
)
from .quantum_mixer import (
    QuantumInterferenceMixer, MixTrack, InterferenceMode,
)
from .quantum_synth import (
    QuantumSynthEngine, WaveShape, FilterType,
)
from .track_entanglement import (
    TrackEntanglementManager, EntanglementType,
)
from .vqpu_daw_engine import (
    VQPUDawEngine, CircuitPurpose, VQPUCircuitRequest,
)
from .data_recorder import (
    DataRecorder, EventCategory, EventSeverity,
)

logger = logging.getLogger("l104.audio.daw_session")


@dataclass
class DAWTrack:
    """A single track in the quantum DAW session."""
    track_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = "Untitled"
    synth_preset: str = "quantum"
    patterns: List[str] = field(default_factory=list)  # Pattern IDs
    mute: bool = False
    solo: bool = False
    armed: bool = False
    color: str = "#00FF88"

    # Audio buffer (filled during render)
    audio: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "name": self.name,
            "synth_preset": self.synth_preset,
            "patterns": list(self.patterns),
            "mute": self.mute,
            "solo": self.solo,
            "armed": self.armed,
            "has_audio": self.audio is not None,
        }


@dataclass
class DAWArrangement:
    """Arrangement timeline — maps patterns to time positions."""
    clips: List[Dict[str, Any]] = field(default_factory=list)
    # Each clip: {"track_id": str, "pattern_id": str, "start_beat": float, "length_beats": float}

    def add_clip(
        self,
        track_id: str,
        pattern_id: str,
        start_beat: float = 0.0,
        length_beats: Optional[float] = None,
    ):
        self.clips.append({
            "track_id": track_id,
            "pattern_id": pattern_id,
            "start_beat": start_beat,
            "length_beats": length_beats,
        })

    def to_dict(self) -> Dict[str, Any]:
        return {"clips": list(self.clips), "total_clips": len(self.clips)}


class QuantumDAWSession:
    """
    The master orchestrator for the Quantum DAW system.

    Creates, manages, and renders quantum audio using all 6 subsystems.
    Integrates with the ASI/AGI pipeline as a scoring dimension.

    Usage:
        from l104_audio_simulation.daw_session import quantum_daw

        # Create tracks
        tk = quantum_daw.add_track("BassLine", preset="god_code_wave")

        # Create a pattern with superposition steps
        pat = quantum_daw.create_pattern(n_steps=16, n_qubits=4, bpm=120.0)

        # Assign pattern to track
        quantum_daw.assign_pattern(tk, pat)

        # Entangle two tracks (sidechain-like)
        quantum_daw.entangle_tracks(track_a, track_b, "bell")

        # Render to audio
        audio = quantum_daw.render(duration=30.0)

        # Export
        quantum_daw.export_wav("output.wav", audio)

        # Get ASI/AGI scoring data
        score = quantum_daw.asi_scoring_data()
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        bpm: float = 120.0,
        time_signature: Tuple[int, int] = (4, 4),
    ):
        self.session_id = uuid.uuid4().hex[:12]
        self.sample_rate = sample_rate
        self.bpm = bpm
        self.time_signature = time_signature
        self.created_at = time.time()

        # ── Subsystems ───────────────────────────────────────────────────
        self.sequencer = ProbabilisticSequencer(bpm=bpm)
        self.mixer = QuantumInterferenceMixer(sample_rate=sample_rate)
        self.synth = QuantumSynthEngine(sample_rate=sample_rate)
        self.entanglement = TrackEntanglementManager()
        self.vqpu = VQPUDawEngine()
        self.recorder = DataRecorder()
        self.arrangement = DAWArrangement()

        # ── Tracks ───────────────────────────────────────────────────────
        self.tracks: Dict[str, DAWTrack] = {}
        self._track_order: List[str] = []

        # ── Session state ────────────────────────────────────────────────
        self._render_count = 0
        self._total_render_time_ms = 0.0
        self._sacred_score_history: List[float] = []
        self._last_render_result: Optional[Dict[str, Any]] = None

        # ── ASI/AGI integration (lazy) ───────────────────────────────────
        self._asi_connected = False
        self._agi_connected = False

        # Record session creation
        self.recorder.record(
            EventCategory.SESSION,
            "session_created",
            data={
                "session_id": self.session_id,
                "sample_rate": sample_rate,
                "bpm": bpm,
                "time_signature": list(time_signature),
            },
        )

        logger.info(f"QuantumDAWSession created: {self.session_id}")

    # ── Track Management ─────────────────────────────────────────────────

    def add_track(
        self,
        name: str = "Untitled",
        preset: str = "quantum",
        color: str = "#00FF88",
    ) -> str:
        """Add a new track to the session. Returns track_id."""
        track = DAWTrack(name=name, synth_preset=preset, color=color)
        self.tracks[track.track_id] = track
        self._track_order.append(track.track_id)

        # Add corresponding mixer track
        mixer_track = self.mixer.add_track(name=track.track_id)

        self.recorder.record(
            EventCategory.SESSION,
            f"track_added_{track.track_id}",
            data={"track_id": track.track_id, "name": name, "preset": preset},
        )
        return track.track_id

    def remove_track(self, track_id: str):
        """Remove a track."""
        if track_id in self.tracks:
            del self.tracks[track_id]
            self._track_order = [t for t in self._track_order if t != track_id]

    def get_track(self, track_id: str) -> Optional[DAWTrack]:
        return self.tracks.get(track_id)

    # ── Pattern Management ───────────────────────────────────────────────

    def create_pattern(
        self,
        n_steps: int = 16,
        n_qubits: int = 4,
        bpm: Optional[float] = None,
        name: Optional[str] = None,
    ) -> str:
        """Create a quantum pattern with superposition steps. Returns pattern_id."""
        pattern = self.sequencer.create_pattern(
            name=name or "Quantum Pattern",
            n_steps=n_steps,
            n_qubits=n_qubits,
        )

        # VQPU-prepare all steps
        vqpu_results = self.vqpu.sequencer_prepare_steps(n_steps, n_qubits)
        for i, req in enumerate(vqpu_results):
            if req.probabilities and i < len(pattern.steps):
                step = pattern.steps[i]
                # Use VQPU probabilities to bias amplitudes
                for key, prob in req.probabilities.items():
                    idx = int(key, 2) % len(step.pitch_amplitudes)
                    step.pitch_amplitudes[idx] = complex(
                        math.sqrt(prob) * math.cos(PHI * idx),
                        math.sqrt(prob) * math.sin(PHI * idx),
                    )
                # Normalize
                norm = math.sqrt(sum(abs(a) ** 2 for a in step.pitch_amplitudes))
                if norm > 1e-10:
                    step.pitch_amplitudes = [a / norm for a in step.pitch_amplitudes]

        # Apply GOD_CODE phase
        pattern.apply_god_code_phase()

        # Record
        pattern_id = pattern.pattern_id
        self.recorder.record(
            EventCategory.COLLAPSE,
            f"pattern_created_{pattern_id}",
            data={
                "n_steps": n_steps,
                "n_qubits": n_qubits,
                "bpm": bpm or self.bpm,
            },
            sacred_score=self.vqpu.sacred_alignment_score(
                vqpu_results[0].probabilities if vqpu_results else {}
            ),
        )
        return pattern_id

    def assign_pattern(self, track_id: str, pattern_id: str, start_beat: float = 0.0):
        """Assign a pattern to a track in the arrangement."""
        track = self.tracks.get(track_id)
        if track:
            track.patterns.append(pattern_id)
            self.arrangement.add_clip(track_id, pattern_id, start_beat)

    # ── Entanglement Management ──────────────────────────────────────────

    def entangle_tracks(
        self,
        track_a: str,
        track_b: str,
        entangle_type: str = "bell",
        strength: float = 1.0,
    ) -> str:
        """Create an entanglement bond between two tracks. Returns bond_id."""
        # Map string to EntanglementType
        etype_map = {
            "bell": EntanglementType.BELL_PAIR,
            "ghz": EntanglementType.GHZ_GROUP,
            "w": EntanglementType.W_STATE,
            "spectral": EntanglementType.SPECTRAL,
            "sidechain": EntanglementType.SIDECHAIN,
            "temporal": EntanglementType.TEMPORAL,
        }
        etype = etype_map.get(entangle_type, EntanglementType.BELL_PAIR)

        if etype == EntanglementType.BELL_PAIR:
            bond_id = self.entanglement.create_bell_pair(track_a, track_b)
        elif etype == EntanglementType.GHZ_GROUP:
            bond_id = self.entanglement.create_ghz_group([track_a, track_b])
        elif etype == EntanglementType.W_STATE:
            bond_id = self.entanglement.create_w_state([track_a, track_b])
        elif etype == EntanglementType.SPECTRAL:
            bond_id = self.entanglement.create_spectral_pair(track_a, track_b)
        elif etype == EntanglementType.SIDECHAIN:
            bond_id = self.entanglement.create_sidechain(track_a, track_b)
        else:
            bond_id = self.entanglement.create_bell_pair(track_a, track_b)

        # VQPU refresh the new bond
        self.entanglement.vqpu_refresh_bonds()

        # Record
        self.recorder.record_entangle(
            bond_id=bond_id,
            action="create",
            bond_type=entangle_type,
            concurrence=strength,
        )
        return bond_id

    def entangle_group(self, track_ids: List[str], entangle_type: str = "ghz") -> str:
        """Entangle a group of tracks together."""
        if entangle_type == "ghz":
            bond_id = self.entanglement.create_ghz_group(track_ids)
        elif entangle_type == "w":
            bond_id = self.entanglement.create_w_state(track_ids)
        else:
            bond_id = self.entanglement.create_ghz_group(track_ids)

        self.recorder.record_entangle(
            bond_id=bond_id,
            action="create",
            bond_type=entangle_type,
        )
        return bond_id

    # ── Rendering ────────────────────────────────────────────────────────

    def render(
        self,
        duration: Optional[float] = None,
        collapse_mode: CollapseMode = CollapseMode.BORN_RULE,
    ) -> np.ndarray:
        """
        Render the entire session to audio.

        Pipeline:
          1. Collapse all patterns via sequencer (superposition → notes)
          2. Render each track through quantum synth
          3. Apply track entanglement propagation
          4. Mix all tracks through interference mixer
          5. Score sacred alignment
          6. Record all telemetry
        """
        t0 = time.time()
        dur = duration or DEFAULT_DURATION
        n_samples = int(self.sample_rate * dur)
        beats_per_second = self.bpm / 60.0
        total_beats = dur * beats_per_second

        self.recorder.record(
            EventCategory.SESSION,
            "render_started",
            data={
                "duration": dur,
                "n_tracks": len(self.tracks),
                "n_samples": n_samples,
                "collapse_mode": collapse_mode.name,
            },
        )

        # ── Phase 1: Collapse patterns ───────────────────────────────────
        collapsed_patterns = {}
        for pat_id, pattern in self.sequencer.patterns.items():
            # VQPU batch collapse
            self.vqpu.execute_batch([
                VQPUCircuitRequest(
                    purpose=CircuitPurpose.SEQUENCER_COLLAPSE,
                    n_qubits=pattern.n_qubits,
                    metadata={"pattern_id": pat_id, "step_index": i},
                )
                for i in range(pattern.n_steps)
            ])
            events = pattern.collapse_all(mode=collapse_mode)
            collapsed_patterns[pat_id] = events

            self.recorder.record_collapse(
                step_index=-1,  # All steps
                sacred_score=self.vqpu.sacred_alignment_score(
                    self.vqpu._classical_fallback(CircuitPurpose.SEQUENCER_COLLAPSE, pattern.n_qubits)
                ),
                n_steps=pattern.n_steps,
                pattern_id=pat_id,
            )

        # ── Phase 2: Synth render per track ──────────────────────────────
        track_audio = {}
        for track_id in self._track_order:
            track = self.tracks[track_id]
            if track.mute:
                track_audio[track_id] = np.zeros(n_samples, dtype=np.float64)
                continue

            # Render all assigned patterns
            audio = np.zeros(n_samples, dtype=np.float64)

            for pat_id in track.patterns:
                events = collapsed_patterns.get(pat_id, [])
                pat_audio = self._render_pattern_events(
                    events, n_samples, dur
                )
                audio += pat_audio

            # Normalize
            peak = np.max(np.abs(audio))
            if peak > 1e-10:
                audio /= peak

            track.audio = audio
            track_audio[track_id] = audio

        # ── Phase 3: Entanglement propagation ────────────────────────────
        for bond_id, bond in self.entanglement.bonds.items():
            track_ids = getattr(bond, 'track_ids', [])
            if len(track_ids) >= 2:
                try:
                    result = bond.measure()
                    outcome = result.get("outcome", 0) if isinstance(result, dict) else 0
                    for tid in track_ids:
                        if tid in track_audio:
                            prop = bond.propagate_spectral(outcome)
                            if isinstance(prop, dict) and "spectral_shift" in prop:
                                shift = prop["spectral_shift"]
                                track_audio[tid] *= (1.0 + 0.05 * shift)
                except Exception:
                    pass

        # ── Phase 4: Load audio into mixer tracks and mixdown ────────────
        for tid in self._track_order:
            track = self.tracks[tid]
            if not track.mute and tid in track_audio:
                mixer_tid = None
                for mt_id, mt in self.mixer.tracks.items():
                    if mt.name == tid:
                        mixer_tid = mt_id
                        break
                if mixer_tid and mixer_tid in self.mixer.tracks:
                    self.mixer.tracks[mixer_tid].load_signal(track_audio[tid])

        # VQPU-backed interference weights (informational)
        weights = self.vqpu.mixer_get_interference_weights(len(self._track_order))

        # Perform mixdown → returns (left, right)
        left, right = self.mixer.mixdown(duration_samples=n_samples)
        mix_result = np.column_stack([left, right])

        # ── Phase 5: Sacred scoring ──────────────────────────────────────
        sacred_score = self._compute_render_sacred_score(mix_result)
        self._sacred_score_history.append(sacred_score)

        # ── Phase 6: Record telemetry ────────────────────────────────────
        render_time_ms = (time.time() - t0) * 1000.0
        self._render_count += 1
        self._total_render_time_ms += render_time_ms

        self._last_render_result = {
            "duration": dur,
            "n_tracks": len(self.tracks),
            "n_patterns": len(collapsed_patterns),
            "sacred_score": sacred_score,
            "render_time_ms": render_time_ms,
            "vqpu_status": self.vqpu.status(),
        }

        self.recorder.record(
            EventCategory.SESSION,
            "render_complete",
            data=self._last_render_result,
            sacred_score=sacred_score,
            duration_ms=render_time_ms,
        )

        return mix_result

    def _render_pattern_events(
        self,
        events: List[NoteEvent],
        n_samples: int,
        duration: float,
    ) -> np.ndarray:
        """Render collapsed note events through the synth engine."""
        audio = np.zeros(n_samples, dtype=np.float64)
        beats_per_second = self.bpm / 60.0

        for i, event in enumerate(events):
            if event.pitch_hz <= 0.0 or event.velocity <= 0.0:
                continue

            # Calculate sample position (each event is a step in sequence)
            step_dur_s = 60.0 / self.bpm / 4.0  # 16th note
            start_sample = int(i * step_dur_s * self.sample_rate)
            dur_samples = int(event.duration_beats / beats_per_second * self.sample_rate)
            dur_samples = max(dur_samples, int(step_dur_s * self.sample_rate))
            end_sample = min(start_sample + dur_samples, n_samples)

            if start_sample >= n_samples or dur_samples <= 0:
                continue

            # Render voice through quantum synth
            voice_audio = self.synth.render_voice(
                dur_samples,
                frequency=event.pitch_hz,
            )

            # Apply velocity scaling
            voice_audio *= event.velocity

            # Fit to buffer
            actual_len = min(len(voice_audio), end_sample - start_sample)
            if actual_len > 0:
                audio[start_sample:start_sample + actual_len] += voice_audio[:actual_len]

        return audio

    def _compute_render_sacred_score(self, audio: np.ndarray) -> float:
        """Compute sacred alignment score for rendered audio."""
        if audio is None or len(audio) == 0:
            return 0.0

        # FFT analysis
        if audio.ndim == 2:
            mono = audio.mean(axis=1) if audio.shape[1] == 2 else audio[:, 0]
        else:
            mono = audio

        n = len(mono)
        if n < 1024:
            return 0.0

        spectrum = np.abs(np.fft.rfft(mono[:min(n, 65536)]))
        freqs = np.fft.rfftfreq(min(n, 65536), 1.0 / self.sample_rate)

        # Check energy around GOD_CODE frequency and PHI harmonics
        god_mask = (freqs >= GOD_CODE - 5) & (freqs <= GOD_CODE + 5)
        phi_harmonics = [GOD_CODE * PHI ** k for k in range(-2, 4)]
        harmonic_energy = 0.0
        total_energy = spectrum.sum() + 1e-15

        for fh in phi_harmonics:
            mask = (freqs >= fh - 5) & (freqs <= fh + 5)
            harmonic_energy += spectrum[mask].sum()

        god_energy = spectrum[god_mask].sum() if god_mask.any() else 0.0

        # Sacred score: weighted combination
        god_ratio = god_energy / total_energy
        harmonic_ratio = harmonic_energy / total_energy

        score = (god_ratio * PHI + harmonic_ratio) / (PHI + 1.0)
        return float(np.clip(score * 100, 0.0, 1.0))

    # ── Export ───────────────────────────────────────────────────────────

    def export_wav(self, filepath: str, audio: Optional[np.ndarray] = None):
        """Export audio to WAV file."""
        from .envelopes import write_wav, normalize_signal
        if audio is None:
            audio = self.render()

        if audio.ndim == 1:
            audio_norm = normalize_signal(audio, 0.95)
        else:
            audio_norm = audio

        write_wav(filepath, audio_norm, self.sample_rate, DEFAULT_BIT_DEPTH)

        self.recorder.record(
            EventCategory.SESSION,
            "export_wav",
            data={"filepath": filepath, "n_samples": len(audio)},
        )

    # ── ASI/AGI Pipeline Integration ─────────────────────────────────────

    def connect_asi_pipeline(self):
        """
        Connect to the ASI pipeline, registering quantum audio as a scoring dimension.
        This makes the DAW's sacred alignment scores contribute to ASI scoring.
        """
        try:
            from l104_asi import asi_core
            # Register quantum_audio subsystem
            if hasattr(asi_core, '_unified_state_bus') and asi_core._unified_state_bus:
                asi_core._unified_state_bus.register_subsystem(
                    'quantum_audio_daw', 1.0, 'ACTIVE'
                )
            self._asi_connected = True
            self.recorder.record(
                EventCategory.SESSION,
                "asi_pipeline_connected",
                data={"subsystem": "quantum_audio_daw"},
            )
            logger.info("Connected to ASI pipeline as quantum_audio_daw subsystem")
        except Exception as e:
            logger.debug(f"ASI pipeline connection: {e}")

    def connect_agi_pipeline(self):
        """
        Connect to the AGI pipeline, registering quantum audio as a domain
        for cross-domain knowledge fusion.
        """
        try:
            from l104_agi import agi_core
            # Register domain for cross-domain knowledge fusion
            agi_core.fusion_register_domain(
                "quantum_audio",
                [
                    "audio", "synthesis", "quantum", "superposition", "interference",
                    "entanglement", "wavetable", "sequencer", "mixer", "sacred",
                    "god_code", "harmonic", "frequency", "oscillator", "modulation",
                    "waveform", "filter", "resonance", "phase", "spectrum",
                ],
            )
            self._agi_connected = True
            self.recorder.record(
                EventCategory.SESSION,
                "agi_pipeline_connected",
                data={"domain": "quantum_audio"},
            )
            logger.info("Connected to AGI pipeline as quantum_audio domain")
        except Exception as e:
            logger.debug(f"AGI pipeline connection: {e}")

    def connect_pipelines(self):
        """Connect to both ASI and AGI pipelines."""
        self.connect_asi_pipeline()
        self.connect_agi_pipeline()

    def asi_scoring_data(self) -> Dict[str, Any]:
        """
        Generate ASI-compatible scoring data from the quantum DAW.
        This can be used as input for ASI scoring dimensions.
        """
        avg_sacred = (
            sum(self._sacred_score_history) / max(len(self._sacred_score_history), 1)
        )
        return {
            "quantum_audio_coherence": avg_sacred,
            "audio_synthesis_sacred_alignment": avg_sacred,
            "vqpu_daw_throughput": self.vqpu.total_circuits_completed,
            "entanglement_bonds": len(self.entanglement.bonds),
            "tracks_active": len(self.tracks),
            "render_count": self._render_count,
            "avg_render_time_ms": (
                self._total_render_time_ms / max(self._render_count, 1)
            ),
            "data_events_recorded": self.recorder._total_events,
            "sacred_resonance_count": self.recorder._sacred_resonance_count,
            "session_duration_s": time.time() - self.created_at,
        }

    def agi_scoring_data(self) -> Dict[str, Any]:
        """Generate AGI-compatible scoring data."""
        scoring = self.asi_scoring_data()
        scoring["quantum_audio_intelligence"] = min(1.0, (
            scoring["quantum_audio_coherence"] * 0.4 +
            min(1.0, scoring["vqpu_daw_throughput"] / 100.0) * 0.3 +
            min(1.0, scoring["entanglement_bonds"] / 10.0) * 0.2 +
            min(1.0, scoring["render_count"] / 10.0) * 0.1
        ))
        return scoring

    # ── Status ───────────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "bpm": self.bpm,
            "sample_rate": self.sample_rate,
            "time_signature": list(self.time_signature),
            "tracks": len(self.tracks),
            "patterns": len(self.sequencer.patterns),
            "entanglement_bonds": len(self.entanglement.bonds),
            "render_count": self._render_count,
            "avg_sacred_score": (
                sum(self._sacred_score_history) / max(len(self._sacred_score_history), 1)
            ),
            "asi_connected": self._asi_connected,
            "agi_connected": self._agi_connected,
            "vqpu": self.vqpu.status(),
            "recorder": self.recorder.statistics(),
            "session_duration_s": time.time() - self.created_at,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Full session state for serialization."""
        return {
            **self.status(),
            "tracks": {tid: t.to_dict() for tid, t in self.tracks.items()},
            "track_order": list(self._track_order),
            "arrangement": self.arrangement.to_dict(),
            "last_render": self._last_render_result,
        }

    def close(self):
        """Flush data and close the session."""
        self.recorder.save_snapshot(extra=self.to_dict())
        self.recorder.close()
        logger.info(f"QuantumDAWSession closed: {self.session_id}")


# ── Module-level singleton ───────────────────────────────────────────────────
quantum_daw = QuantumDAWSession()
