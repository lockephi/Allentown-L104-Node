"""L104 Numerical Engine — Inter-Builder Feedback Bus.

Cross-builder event bus for inter-module communication.

PART V RESEARCH — l104_runtime_infrastructure_research.py:
  F83: Consciousness SOVEREIGN × Grover φ³ = φ⁴ announced via bus
  F78: Gate constants with GOD_CODE guards propagated through feedback
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _parse_timestamp(ts) -> float:
    """Convert a timestamp (float epoch or ISO string) to float epoch."""
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts)
            return dt.timestamp()
        except (ValueError, TypeError):
            return 0.0
    return 0.0


class InterBuilderFeedbackBus:
    """v3.0 — Cross-builder event bus for inter-module communication.
    Shared bus file: .l104_builder_feedback_bus.json
    """

    BUS_FILE = Path(".l104_builder_feedback_bus.json")
    MAX_MESSAGES = 200
    MESSAGE_TTL = 60  # seconds

    def __init__(self, builder_id: str = "numerical_builder"):
        self.builder_id = builder_id
        self.sent_count = 0
        self.received_count = 0

    def send(self, event_type: str, data: Dict) -> Dict:
        """Post a message to the shared bus."""
        self.sent_count += 1
        message = {
            "id": f"{self.builder_id}_{self.sent_count}_{int(time.time())}",
            "builder": self.builder_id,
            "event": event_type,
            "data": data,
            "timestamp": time.time(),
        }
        bus = self._read_bus()
        bus.append(message)
        now = time.time()
        bus = [m for m in bus if now - _parse_timestamp(m.get("timestamp", 0)) < self.MESSAGE_TTL]
        bus = bus[-self.MAX_MESSAGES:]
        try:
            self.BUS_FILE.write_text(json.dumps(bus, indent=2, default=str))
        except Exception:
            pass
        return message

    def receive(self, from_builder: str = None) -> List[Dict]:
        """Read messages from the bus, optionally filtering by builder."""
        bus = self._read_bus()
        now = time.time()
        valid = [m for m in bus if now - _parse_timestamp(m.get("timestamp", 0)) < self.MESSAGE_TTL]
        if from_builder:
            valid = [m for m in valid if m.get("builder") == from_builder]
        incoming = [m for m in valid if m.get("builder") != self.builder_id]
        self.received_count += len(incoming)
        return incoming

    def _read_bus(self) -> List[Dict]:
        """Read the shared bus file."""
        if not self.BUS_FILE.exists():
            return []
        try:
            raw = json.loads(self.BUS_FILE.read_text())
            # Support both formats:
            #   Legacy: [message, message, ...]
            #   v5.0+:  {"messages": [...], "last_updated": ...}
            if isinstance(raw, dict):
                raw = raw.get("messages", [])
            if not isinstance(raw, list):
                return []
            # Filter out corrupted non-dict entries
            return [m for m in raw if isinstance(m, dict)]
        except Exception:
            return []

    def announce_pipeline_complete(self, metrics: Dict) -> Dict:
        """Announce pipeline completion to other builders."""
        return self.send("pipeline_complete", {
            "builder": self.builder_id,
            "coherence": metrics.get("coherence", 0),
            "tokens": metrics.get("tokens", 0),
            "research_health": metrics.get("research_health", 0),
            "deep_math_engines": metrics.get("deep_math_engines", 0),
        })

    def announce_coherence_shift(self, old_val: float, new_val: float) -> Dict:
        """Announce a significant coherence change."""
        return self.send("coherence_shift", {
            "builder": self.builder_id,
            "old": old_val,
            "new": new_val,
            "delta": round(new_val - old_val, 6),
        })

    def status(self) -> Dict:
        bus = self._read_bus()
        now = time.time()
        active = [m for m in bus if now - _parse_timestamp(m.get("timestamp", 0)) < self.MESSAGE_TTL]
        return {
            "class": "InterBuilderFeedbackBus",
            "builder_id": self.builder_id,
            "sent": self.sent_count,
            "received": self.received_count,
            "active_messages": len(active),
            "bus_total": len(bus),
        }
