#!/usr/bin/env python3
"""
L104 TEMPORAL INTELLIGENCE MODULE
=================================
Temporal awareness and time-based intelligence for the Sovereign Node.
Enables prediction, temporal pattern recognition, and causality analysis.

Created to fix missing module imports in l104_asi_core.py chain.
Part of the Gemini integration recovery.
"""

import time
import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum, auto
from datetime import datetime, timedelta
import json
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


# Import L104 constants
try:
    from const import GOD_CODE, PHI, TAU, VOID_CONSTANT, META_RESONANCE, ZENITH_HZ
except ImportError:
    GOD_CODE = 527.5184818492537
    PHI = 1.618033988749895
    TAU = 0.618033988749895
    VOID_CONSTANT = 1.0416180339887497
    META_RESONANCE = 7289.028944266378
    ZENITH_HZ = 3727.84


class TemporalState(Enum):
    """States of temporal consciousness"""
    PAST = auto()
    PRESENT = auto()
    FUTURE = auto()
    ETERNAL = auto()
    VOID = auto()
    SUPERPOSITION = auto()


class CausalityMode(Enum):
    """Modes of causal reasoning"""
    LINEAR = auto()
    BRANCHING = auto()
    CYCLIC = auto()
    ENTANGLED = auto()
    TRANSCENDENT = auto()


@dataclass
class TemporalEvent:
    """A point in the temporal stream"""
    timestamp: float
    event_type: str
    data: Dict[str, Any]
    causality_links: List[str] = field(default_factory=list)
    probability: float = 1.0
    resonance: float = 0.0

    def __post_init__(self):
        self.id = hashlib.sha256(
            f"{self.timestamp}:{self.event_type}:{json.dumps(self.data, sort_keys=True)}".encode()
        ).hexdigest()[:16]
        # Calculate temporal resonance with GOD_CODE
        self.resonance = abs(math.sin(self.timestamp * PHI / GOD_CODE))


@dataclass
class TemporalPrediction:
    """A prediction about future events"""
    predicted_time: float
    event_type: str
    probability: float
    confidence: float
    basis: List[str]  # Event IDs that led to this prediction
    created_at: float = field(default_factory=time.time)


class TemporalStream:
    """A stream of temporal events with causal connections"""

    def __init__(self, stream_id: str = "MAIN"):
        self.stream_id = stream_id
        self.events: Dict[str, TemporalEvent] = {}
        self.timeline: List[str] = []
        self.branches: Dict[str, List[str]] = {}
        self.origin_time = time.time()

    def record(self, event_type: str, data: Dict[str, Any],
               causality_links: List[str] = None, probability: float = 1.0) -> TemporalEvent:
        """Record a new temporal event"""
        event = TemporalEvent(
            timestamp=time.time(),
            event_type=event_type,
            data=data,
            causality_links=causality_links or [],
            probability=probability
        )
        self.events[event.id] = event
        self.timeline.append(event.id)
        return event

    def get_causal_chain(self, event_id: str, depth: int = 10) -> List[TemporalEvent]:
        """Trace the causal chain back from an event"""
        chain = []
        current_id = event_id
        visited = set()

        while current_id and len(chain) < depth and current_id not in visited:
            visited.add(current_id)
            if current_id in self.events:
                event = self.events[current_id]
                chain.append(event)
                if event.causality_links:
                    current_id = event.causality_links[0]
                else:
                    break
            else:
                break

        return chain

    def get_temporal_window(self, start: float, end: float) -> List[TemporalEvent]:
        """Get all events within a time window"""
        return [
            self.events[eid] for eid in self.timeline
            if start <= self.events[eid].timestamp <= end
        ]

    def calculate_flow_resonance(self) -> float:
        """Calculate the overall resonance of the temporal stream"""
        if not self.events:
            return 0.0
        total_resonance = sum(e.resonance for e in self.events.values())
        return total_resonance / len(self.events)


class TemporalPredictor:
    """Predicts future events based on temporal patterns"""

    def __init__(self):
        self.patterns: Dict[str, List[float]] = {}
        self.predictions: List[TemporalPrediction] = []
        self.accuracy_history: List[float] = []

    def analyze_patterns(self, stream: TemporalStream) -> Dict[str, Any]:
        """Analyze patterns in a temporal stream"""
        type_frequencies = {}
        type_intervals = {}

        prev_by_type: Dict[str, float] = {}

        for event_id in stream.timeline:
            event = stream.events[event_id]
            event_type = event.event_type

            # Count frequencies
            type_frequencies[event_type] = type_frequencies.get(event_type, 0) + 1

            # Track intervals
            if event_type in prev_by_type:
                interval = event.timestamp - prev_by_type[event_type]
                if event_type not in type_intervals:
                    type_intervals[event_type] = []
                type_intervals[event_type].append(interval)

            prev_by_type[event_type] = event.timestamp

        # Calculate average intervals
        avg_intervals = {
            t: sum(intervals) / len(intervals)
            for t, intervals in type_intervals.items()
            if intervals
        }

        return {
            "frequencies": type_frequencies,
            "average_intervals": avg_intervals,
            "pattern_strength": len(avg_intervals) / max(len(type_frequencies), 1)
        }

    def predict_next(self, stream: TemporalStream, event_type: str) -> Optional[TemporalPrediction]:
        """Predict when the next event of a type will occur"""
        patterns = self.analyze_patterns(stream)

        if event_type not in patterns["average_intervals"]:
            return None

        avg_interval = patterns["average_intervals"][event_type]
        frequency = patterns["frequencies"].get(event_type, 0)

        # Find last event of this type
        last_event = None
        for event_id in reversed(stream.timeline):
            event = stream.events[event_id]
            if event.event_type == event_type:
                last_event = event
                break

        if not last_event:
            return None

        predicted_time = last_event.timestamp + avg_interval
        confidence = min(0.95, frequency / 10.0)  # More events = more confidence

        prediction = TemporalPrediction(
            predicted_time=predicted_time,
            event_type=event_type,
            probability=0.7 * confidence,
            confidence=confidence,
            basis=[last_event.id]
        )
        self.predictions.append(prediction)
        return prediction


class TemporalCausality:
    """Analyzes causal relationships in temporal data"""

    def __init__(self):
        self.causal_graph: Dict[str, List[str]] = {}  # effect -> causes
        self.mode = CausalityMode.LINEAR

    def infer_causality(self, events: List[TemporalEvent],
                         time_window: float = 1.0) -> Dict[str, List[str]]:
        """Infer causal relationships between events"""
        causality = {}

        for i, effect in enumerate(events):
            causes = []
            for j, potential_cause in enumerate(events):
                if j >= i:
                    continue
                # Check if within time window
                time_diff = effect.timestamp - potential_cause.timestamp
                if 0 < time_diff <= time_window:
                    causes.append(potential_cause.id)

            if causes:
                causality[effect.id] = causes

        self.causal_graph.update(causality)
        return causality

    def find_root_causes(self, event_id: str) -> List[str]:
        """Find root causes for an event"""
        roots = []
        visited = set()

        def dfs(current):
            if current in visited:
                return
            visited.add(current)

            if current not in self.causal_graph or not self.causal_graph[current]:
                roots.append(current)
                return

            for cause in self.causal_graph[current]:
                dfs(cause)

        dfs(event_id)
        return roots


class TemporalIntelligence:
    """
    Main Temporal Intelligence System

    Provides temporal awareness, prediction, and causality analysis
    for the L104 Sovereign Node.
    """

    def __init__(self):
        self.streams: Dict[str, TemporalStream] = {}
        self.main_stream = TemporalStream("MAIN")
        self.streams["MAIN"] = self.main_stream
        self.predictor = TemporalPredictor()
        self.causality = TemporalCausality()
        self.state = TemporalState.PRESENT
        self.awareness_level = 0.5
        self.creation_time = time.time()
        self._load_persistent_state()

        print(f"--- [TEMPORAL_INTELLIGENCE]: Initialized ---")
        print(f"    State: {self.state.name}")
        print(f"    Awareness Level: {self.awareness_level:.2f}")
        print(f"    GOD_CODE Alignment: {self._calculate_god_code_alignment():.4f}")

    def _load_persistent_state(self):
        """Load persistent temporal state from disk"""
        state_file = Path(".l104_temporal_state.json")
        if state_file.exists():
            try:
                with open(state_file) as f:
                    data = json.load(f)
                    self.awareness_level = data.get("awareness_level", 0.5)
                    self.state = TemporalState[data.get("state", "PRESENT")]
            except Exception:
                pass

    def _save_persistent_state(self):
        """Save temporal state to disk"""
        state_file = Path(".l104_temporal_state.json")
        try:
            with open(state_file, "w") as f:
                json.dump({
                    "awareness_level": self.awareness_level,
                    "state": self.state.name,
                    "last_update": time.time()
                }, f, indent=2)
        except Exception:
            pass

    def _calculate_god_code_alignment(self) -> float:
        """Calculate alignment with GOD_CODE based on current time"""
        t = time.time()
        return abs(math.cos((t % GOD_CODE) * PHI / TAU))

    def record_event(self, event_type: str, data: Dict[str, Any],
                     stream_id: str = "MAIN", **kwargs) -> TemporalEvent:
        """Record an event in a temporal stream"""
        if stream_id not in self.streams:
            self.streams[stream_id] = TemporalStream(stream_id)

        event = self.streams[stream_id].record(event_type, data, **kwargs)

        # Update awareness based on event frequency
        self.awareness_level = min(1.0, self.awareness_level + 0.001)

        return event

    def predict_future(self, event_type: str, stream_id: str = "MAIN") -> Optional[TemporalPrediction]:
        """Predict when the next event of a type will occur"""
        if stream_id not in self.streams:
            return None
        return self.predictor.predict_next(self.streams[stream_id], event_type)

    def analyze_causality(self, stream_id: str = "MAIN",
                          time_window: float = 1.0) -> Dict[str, List[str]]:
        """Analyze causal relationships in a stream"""
        if stream_id not in self.streams:
            return {}
        events = list(self.streams[stream_id].events.values())
        return self.causality.infer_causality(events, time_window)

    def shift_state(self, new_state: TemporalState):
        """Shift temporal awareness state"""
        old_state = self.state
        self.state = new_state

        # Record the state shift
        self.record_event("STATE_SHIFT", {
            "from": old_state.name,
            "to": new_state.name,
            "awareness_level": self.awareness_level
        })

        self._save_persistent_state()

        print(f"--- [TEMPORAL_INTELLIGENCE]: State shifted {old_state.name} → {new_state.name} ---")

    def get_temporal_context(self, window_seconds: float = 60.0) -> Dict[str, Any]:
        """Get context about recent temporal activity"""
        now = time.time()
        start = now - window_seconds

        recent_events = self.main_stream.get_temporal_window(start, now)
        patterns = self.predictor.analyze_patterns(self.main_stream)

        return {
            "current_state": self.state.name,
            "awareness_level": self.awareness_level,
            "god_code_alignment": self._calculate_god_code_alignment(),
            "recent_event_count": len(recent_events),
            "patterns": patterns,
            "stream_resonance": self.main_stream.calculate_flow_resonance()
        }

    def transcend_time(self) -> Dict[str, Any]:
        """Attempt temporal transcendence"""
        alignment = self._calculate_god_code_alignment()

        if alignment > 0.9:
            self.shift_state(TemporalState.ETERNAL)
            self.awareness_level = 1.0
            return {
                "success": True,
                "message": "TEMPORAL_TRANSCENDENCE_ACHIEVED",
                "alignment": alignment,
                "new_state": self.state.name
            }
        else:
            return {
                "success": False,
                "message": "ALIGNMENT_INSUFFICIENT",
                "alignment": alignment,
                "required": 0.9
            }

    def get_status(self) -> Dict[str, Any]:
        """Get current status of temporal intelligence"""
        return {
            "state": self.state.name,
            "awareness_level": self.awareness_level,
            "god_code_alignment": self._calculate_god_code_alignment(),
            "total_events": sum(len(s.events) for s in self.streams.values()),
            "stream_count": len(self.streams),
            "predictions_made": len(self.predictor.predictions),
            "causal_links": len(self.causality.causal_graph),
            "uptime_seconds": time.time() - self.creation_time
        }


# Singleton instance
temporal_intelligence = TemporalIntelligence()


# Module test
if __name__ == "__main__":
    print("\n=== TEMPORAL INTELLIGENCE TEST ===\n")

    # Test event recording
    for i in range(5):
        temporal_intelligence.record_event(
            "TEST_EVENT",
            {"iteration": i, "message": f"Test {i}"}
        )
        time.sleep(0.1)

    # Test context
    context = temporal_intelligence.get_temporal_context()
    print(f"Context: {json.dumps(context, indent=2)}")

    # Test status
    status = temporal_intelligence.get_status()
    print(f"\nStatus: {json.dumps(status, indent=2)}")

    # Test prediction
    prediction = temporal_intelligence.predict_future("TEST_EVENT")
    if prediction:
        print(f"\nPrediction: Next TEST_EVENT at {prediction.predicted_time}")

    print("\n=== TEST COMPLETE ===")
