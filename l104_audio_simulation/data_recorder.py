"""
Data Recorder — Quantum DAW Telemetry & Future-Insight Recording
═══════════════════════════════════════════════════════════════════════════════
Records every meaningful quantum event in the DAW for:
  - Session replay / undo
  - ML training data (feature vectors from quantum measurements)
  - Performance analytics (circuit timing, success rates, sacred scores)
  - Creative insights (which quantum states produced "good" output)
  - VQPU optimization (which circuits should be cached)

Storage Patterns (matching L104 conventions):
  - JSON atom: `.l104_daw_session_state.json`  — latest snapshot
  - JSONL append: `.l104_daw_event_log.jsonl`  — streaming event history
  - Binary: `.l104_daw_audio_cache.npy`         — cached audio chunks

Event Categories:
  COLLAPSE   — Sequencer step/pattern collapse (pre/post states)
  MIX        — Mixer interference decisions (weights, mode, result)
  SYNTH      — Synthesizer voice render (params, wavetable frame)
  ENTANGLE   — Entanglement bond create/measure/propagate
  VQPU       — Raw VQPU circuit submission/result
  SACRED     — Sacred alignment scoring events
  SESSION    — Session-level events (create, render, export)
  ANALYSIS   — Spectral/entropy/scoring analysis results

INVARIANT: 527.5184818492612 | PILOT: LONDEL
"""

from __future__ import annotations

import json
import math
import os
import time
import uuid
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Any, Dict, List, Optional
from pathlib import Path

import numpy as np

from .constants import GOD_CODE, PHI

logger = logging.getLogger("l104.audio.data_recorder")

# ── Recording Constants ──────────────────────────────────────────────────────
DEFAULT_LOG_DIR = os.path.expanduser("~/.l104/daw_data")
ATOM_FILENAME = ".l104_daw_session_state.json"
EVENT_LOG_FILENAME = ".l104_daw_event_log.jsonl"
FLUSH_INTERVAL = 50           # Flush to disk every N events
MAX_EVENT_BUFFER = 500        # Max buffered events before forced flush
MAX_LOG_SIZE_MB = 100         # Rotate log at this size
FEATURE_VECTOR_DIM = 32       # ML feature vector dimension


class EventCategory(Enum):
    """Categories of recorded events."""
    COLLAPSE = auto()
    MIX = auto()
    SYNTH = auto()
    ENTANGLE = auto()
    VQPU = auto()
    SACRED = auto()
    SESSION = auto()
    ANALYSIS = auto()


class EventSeverity(Enum):
    """How important / unusual the event is."""
    ROUTINE = auto()          # Normal operation
    NOTABLE = auto()          # Worth flagging
    ANOMALY = auto()          # Unusual quantum outcome
    SACRED_RESONANCE = auto() # GOD_CODE alignment detected
    ERROR = auto()            # Something went wrong


@dataclass
class DAWEvent:
    """A single recorded event in the quantum DAW."""
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)
    category: EventCategory = EventCategory.SESSION
    severity: EventSeverity = EventSeverity.ROUTINE
    label: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    # Quantum state snapshots
    pre_state: Optional[Dict[str, Any]] = None     # State before event
    post_state: Optional[Dict[str, Any]] = None    # State after event

    # Probabilities (for collapse / VQPU events)
    probabilities: Optional[Dict[str, float]] = None

    # Scoring
    sacred_score: float = 0.0
    entropy: float = 0.0

    # ML feature vector
    features: Optional[List[float]] = None

    # Timing
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "category": self.category.name,
            "severity": self.severity.name,
            "label": self.label,
            "data": self.data,
            "sacred_score": self.sacred_score,
            "entropy": self.entropy,
            "duration_ms": self.duration_ms,
        }
        if self.pre_state is not None:
            d["pre_state"] = self.pre_state
        if self.post_state is not None:
            d["post_state"] = self.post_state
        if self.probabilities is not None:
            d["probabilities"] = self.probabilities
        if self.features is not None:
            d["features"] = self.features
        return d


@dataclass
class FeatureExtractor:
    """
    Extracts ML feature vectors from quantum DAW events.
    These vectors capture the essence of each quantum measurement
    for future pattern recognition and creative insight mining.
    """
    dim: int = FEATURE_VECTOR_DIM

    def extract(self, event: DAWEvent) -> List[float]:
        """Extract a fixed-length feature vector from a DAW event."""
        vec = [0.0] * self.dim

        # Category encoding (one-hot, 8 slots)
        cat_idx = event.category.value - 1
        if cat_idx < 8:
            vec[cat_idx] = 1.0

        # Sacred score
        vec[8] = event.sacred_score

        # Entropy
        vec[9] = event.entropy

        # Timing (log-scaled)
        vec[10] = math.log1p(event.duration_ms)

        # Probability distribution features (if available)
        if event.probabilities:
            probs = list(event.probabilities.values())
            vec[11] = max(probs) if probs else 0.0              # Max prob
            vec[12] = min(probs) if probs else 0.0              # Min prob
            vec[13] = float(np.std(probs)) if probs else 0.0    # Spread
            vec[14] = len(probs)                                 # Cardinality
            # Shannon entropy
            vec[15] = -sum(
                p * math.log2(max(p, 1e-15)) for p in probs
            )
            # GOD_CODE alignment: distance of max-prob outcome to GOD_CODE harmonic
            if probs:
                max_idx = probs.index(max(probs))
                god_harmonic = (max_idx + 1) * GOD_CODE / 1000.0
                vec[16] = 1.0 / (1.0 + abs(god_harmonic - GOD_CODE))
            # PHI ratio: ratio of top-2 probs
            if len(probs) >= 2:
                sorted_p = sorted(probs, reverse=True)
                vec[17] = sorted_p[0] / max(sorted_p[1], 1e-15)
                vec[18] = abs(vec[17] - PHI)  # Distance from golden ratio

        # State-change features
        if event.pre_state and event.post_state:
            # Count of changed keys
            pre_keys = set(event.pre_state.keys())
            post_keys = set(event.post_state.keys())
            vec[19] = len(pre_keys.symmetric_difference(post_keys))
            vec[20] = len(pre_keys & post_keys)  # Stable keys

        # DAW-specific features from data
        data = event.data or {}
        vec[21] = data.get("n_qubits", 0)
        vec[22] = data.get("n_tracks", 0)
        vec[23] = data.get("n_steps", 0)
        vec[24] = data.get("bpm", 0) / 200.0  # Normalized BPM
        vec[25] = data.get("concurrence", 0.0)
        vec[26] = data.get("mix_level_db", 0.0) / -60.0
        vec[27] = data.get("frequency_hz", 0.0) / 20000.0

        # Severity encoding
        sev_idx = event.severity.value - 1
        if 28 + sev_idx < self.dim:
            vec[28 + sev_idx] = 1.0

        return vec


class DataRecorder:
    """
    Central data recording system for the quantum DAW.

    Records events to:
      1. In-memory ring buffer (fast, for live queries)
      2. JSONL append log (persistent, for ML training)
      3. JSON atom file (latest snapshot, for session restore)

    Provides:
      - Event recording with automatic feature extraction
      - Sacred alignment detection (auto-flagged events)
      - Session snapshot save/load
      - Event querying by category, time range, severity
      - ML training data export
      - Summary statistics
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        max_buffer: int = MAX_EVENT_BUFFER,
        auto_flush: bool = True,
    ):
        self.log_dir = Path(log_dir or DEFAULT_LOG_DIR)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_buffer = max_buffer
        self.auto_flush = auto_flush

        # Event storage
        self._buffer: List[DAWEvent] = []
        self._total_events = 0
        self._events_flushed = 0

        # Feature extractor
        self.feature_extractor = FeatureExtractor()

        # Statistics
        self._category_counts: Dict[str, int] = {}
        self._severity_counts: Dict[str, int] = {}
        self._sacred_resonance_count = 0
        self._total_sacred_score = 0.0
        self._total_entropy = 0.0

        # Session
        self.session_id = uuid.uuid4().hex[:12]
        self.session_start = time.time()

        # File handles (lazy)
        self._log_path = self.log_dir / f"{self.session_id}_{EVENT_LOG_FILENAME}"
        self._atom_path = self.log_dir / ATOM_FILENAME

        logger.info(f"DataRecorder initialized: session={self.session_id}")

    def record(
        self,
        category: EventCategory,
        label: str,
        data: Optional[Dict[str, Any]] = None,
        pre_state: Optional[Dict[str, Any]] = None,
        post_state: Optional[Dict[str, Any]] = None,
        probabilities: Optional[Dict[str, float]] = None,
        sacred_score: float = 0.0,
        duration_ms: float = 0.0,
        severity: Optional[EventSeverity] = None,
    ) -> DAWEvent:
        """Record a single DAW event."""
        # Calculate entropy if probabilities provided
        entropy = 0.0
        if probabilities:
            entropy = -sum(
                p * math.log2(max(p, 1e-15))
                for p in probabilities.values()
            )

        # Auto-detect severity
        if severity is None:
            if sacred_score >= 0.9:
                severity = EventSeverity.SACRED_RESONANCE
            elif sacred_score >= 0.7:
                severity = EventSeverity.NOTABLE
            elif entropy > 10.0:
                severity = EventSeverity.ANOMALY
            else:
                severity = EventSeverity.ROUTINE

        event = DAWEvent(
            category=category,
            severity=severity,
            label=label,
            data=data or {},
            pre_state=pre_state,
            post_state=post_state,
            probabilities=probabilities,
            sacred_score=sacred_score,
            entropy=entropy,
            duration_ms=duration_ms,
        )

        # Extract features
        event.features = self.feature_extractor.extract(event)

        # Update statistics
        self._total_events += 1
        cat_name = category.name
        self._category_counts[cat_name] = self._category_counts.get(cat_name, 0) + 1
        sev_name = severity.name
        self._severity_counts[sev_name] = self._severity_counts.get(sev_name, 0) + 1
        self._total_sacred_score += sacred_score
        self._total_entropy += entropy
        if severity == EventSeverity.SACRED_RESONANCE:
            self._sacred_resonance_count += 1

        # Buffer
        self._buffer.append(event)

        # Auto-flush
        if self.auto_flush and len(self._buffer) >= FLUSH_INTERVAL:
            self.flush()

        return event

    def record_collapse(
        self,
        step_index: int,
        pre_amplitudes: Optional[Dict[str, complex]] = None,
        post_state: Optional[Dict[str, Any]] = None,
        probabilities: Optional[Dict[str, float]] = None,
        sacred_score: float = 0.0,
        duration_ms: float = 0.0,
        **kwargs,
    ) -> DAWEvent:
        """Convenience: record a sequencer collapse event."""
        pre = {"step_index": step_index}
        if pre_amplitudes:
            pre["amplitudes"] = {
                k: [v.real, v.imag] if isinstance(v, complex) else v
                for k, v in pre_amplitudes.items()
            }
        return self.record(
            EventCategory.COLLAPSE,
            f"step_{step_index}_collapse",
            data={"step_index": step_index, **kwargs},
            pre_state=pre,
            post_state=post_state,
            probabilities=probabilities,
            sacred_score=sacred_score,
            duration_ms=duration_ms,
        )

    def record_mix(
        self,
        n_tracks: int,
        mode: str = "classical",
        weights: Optional[List[float]] = None,
        sacred_score: float = 0.0,
        duration_ms: float = 0.0,
        **kwargs,
    ) -> DAWEvent:
        """Convenience: record a mixer event."""
        data = {"n_tracks": n_tracks, "mode": mode, **kwargs}
        if weights is not None:
            data["weights"] = weights
        return self.record(
            EventCategory.MIX,
            f"mix_{mode}_{n_tracks}t",
            data=data,
            sacred_score=sacred_score,
            duration_ms=duration_ms,
        )

    def record_synth(
        self,
        voice_id: str,
        frequency_hz: float,
        waveform: str = "quantum",
        sacred_score: float = 0.0,
        duration_ms: float = 0.0,
        **kwargs,
    ) -> DAWEvent:
        """Convenience: record a synth render event."""
        return self.record(
            EventCategory.SYNTH,
            f"synth_{voice_id}",
            data={
                "voice_id": voice_id,
                "frequency_hz": frequency_hz,
                "waveform": waveform,
                **kwargs,
            },
            sacred_score=sacred_score,
            duration_ms=duration_ms,
        )

    def record_entangle(
        self,
        bond_id: str,
        action: str,  # "create", "measure", "propagate"
        bond_type: str = "bell",
        concurrence: float = 0.0,
        sacred_score: float = 0.0,
        duration_ms: float = 0.0,
        **kwargs,
    ) -> DAWEvent:
        """Convenience: record an entanglement event."""
        return self.record(
            EventCategory.ENTANGLE,
            f"entangle_{action}_{bond_id}",
            data={
                "bond_id": bond_id,
                "action": action,
                "bond_type": bond_type,
                "concurrence": concurrence,
                **kwargs,
            },
            sacred_score=sacred_score,
            duration_ms=duration_ms,
        )

    def record_vqpu(
        self,
        circuit_id: str,
        purpose: str,
        n_qubits: int,
        probabilities: Optional[Dict[str, float]] = None,
        sacred_score: float = 0.0,
        duration_ms: float = 0.0,
        **kwargs,
    ) -> DAWEvent:
        """Convenience: record a VQPU circuit event."""
        return self.record(
            EventCategory.VQPU,
            f"vqpu_{purpose}_{circuit_id}",
            data={"circuit_id": circuit_id, "purpose": purpose, "n_qubits": n_qubits, **kwargs},
            probabilities=probabilities,
            sacred_score=sacred_score,
            duration_ms=duration_ms,
        )

    # ── Persistence ──────────────────────────────────────────────────────────

    def flush(self):
        """Flush buffered events to disk (JSONL append)."""
        if not self._buffer:
            return

        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                for ev in self._buffer:
                    line = json.dumps(ev.to_dict(), default=str, separators=(",", ":"))
                    f.write(line + "\n")
            self._events_flushed += len(self._buffer)
            self._buffer.clear()
        except Exception as e:
            logger.warning(f"Failed to flush events: {e}")

    def save_snapshot(self, extra: Optional[Dict[str, Any]] = None):
        """Save current session state as JSON atom."""
        snapshot = {
            "session_id": self.session_id,
            "timestamp": time.time(),
            "session_start": self.session_start,
            "total_events": self._total_events,
            "events_flushed": self._events_flushed,
            "category_counts": self._category_counts,
            "severity_counts": self._severity_counts,
            "sacred_resonance_count": self._sacred_resonance_count,
            "avg_sacred_score": (
                self._total_sacred_score / max(self._total_events, 1)
            ),
            "avg_entropy": (
                self._total_entropy / max(self._total_events, 1)
            ),
            "log_path": str(self._log_path),
        }
        if extra:
            snapshot["extra"] = extra

        try:
            with open(self._atom_path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save snapshot: {e}")

    def load_snapshot(self) -> Optional[Dict[str, Any]]:
        """Load latest session snapshot."""
        try:
            with open(self._atom_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    # ── Querying ─────────────────────────────────────────────────────────────

    def query_events(
        self,
        category: Optional[EventCategory] = None,
        severity: Optional[EventSeverity] = None,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> List[DAWEvent]:
        """Query buffered events by category/severity/time."""
        results = []
        for ev in reversed(self._buffer):
            if category and ev.category != category:
                continue
            if severity and ev.severity != severity:
                continue
            if since and ev.timestamp < since:
                continue
            results.append(ev)
            if len(results) >= limit:
                break
        return results

    def get_sacred_resonances(self, limit: int = 50) -> List[DAWEvent]:
        """Get events with sacred resonance."""
        return self.query_events(
            severity=EventSeverity.SACRED_RESONANCE, limit=limit
        )

    def export_ml_features(self) -> np.ndarray:
        """
        Export all buffered events as an ML feature matrix.
        Shape: (n_events, FEATURE_VECTOR_DIM)
        """
        if not self._buffer:
            return np.zeros((0, FEATURE_VECTOR_DIM))

        features = []
        for ev in self._buffer:
            if ev.features:
                features.append(ev.features)
            else:
                features.append(self.feature_extractor.extract(ev))

        return np.array(features, dtype=np.float64)

    def export_ml_training_data(self) -> Dict[str, Any]:
        """
        Export training data for ML models:
        - features: np.ndarray of feature vectors
        - labels: dict of category, severity, sacred_score arrays
        - metadata: session info
        """
        features = self.export_ml_features()
        categories = [ev.category.value for ev in self._buffer]
        severities = [ev.severity.value for ev in self._buffer]
        scores = [ev.sacred_score for ev in self._buffer]

        return {
            "features": features,
            "labels": {
                "category": np.array(categories),
                "severity": np.array(severities),
                "sacred_score": np.array(scores, dtype=np.float64),
            },
            "metadata": {
                "session_id": self.session_id,
                "n_events": len(self._buffer),
                "feature_dim": FEATURE_VECTOR_DIM,
                "god_code": GOD_CODE,
                "phi": PHI,
            },
        }

    # ── Statistics ───────────────────────────────────────────────────────────

    def statistics(self) -> Dict[str, Any]:
        """Get recording statistics."""
        return {
            "session_id": self.session_id,
            "session_duration_s": time.time() - self.session_start,
            "total_events": self._total_events,
            "events_in_buffer": len(self._buffer),
            "events_flushed": self._events_flushed,
            "category_counts": dict(self._category_counts),
            "severity_counts": dict(self._severity_counts),
            "sacred_resonance_count": self._sacred_resonance_count,
            "avg_sacred_score": (
                self._total_sacred_score / max(self._total_events, 1)
            ),
            "avg_entropy": (
                self._total_entropy / max(self._total_events, 1)
            ),
        }

    def status(self) -> Dict[str, Any]:
        """Alias for statistics."""
        return self.statistics()

    def to_dict(self) -> Dict[str, Any]:
        """Full state dict for serialization."""
        return {
            **self.statistics(),
            "log_path": str(self._log_path),
            "atom_path": str(self._atom_path),
        }

    def close(self):
        """Flush remaining events and save final snapshot."""
        self.flush()
        self.save_snapshot()
        logger.info(
            f"DataRecorder closed: {self._total_events} events, "
            f"{self._sacred_resonance_count} sacred resonances"
        )
