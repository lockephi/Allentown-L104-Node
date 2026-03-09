"""L104 Gate Engine — Inter-Builder Feedback Bus."""

import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from .constants import WORKSPACE_ROOT


class InterBuilderFeedbackBus:
    """
    Real-time message bus for inter-builder communication.

    Enables the three pillars of creation (gate, link, numerical builders)
    to exchange signals, discoveries, and state updates without direct imports.

    Communication via shared JSON file: .l104_builder_feedback_bus.json
    """

    BUS_FILE = WORKSPACE_ROOT / ".l104_builder_feedback_bus.json"
    MESSAGE_TTL = 60.0  # seconds
    MAX_MESSAGES = 200

    def __init__(self, builder_id: str = "gate_builder"):
        """Initialize feedback bus for this builder."""
        self.builder_id = builder_id
        self.sent_count = 0
        self.received_count = 0
        self._last_read_time = 0.0

    def send(self, msg_type: str, payload: Dict[str, Any]) -> None:
        """Send a message to the feedback bus."""
        msg = {
            "source": self.builder_id,
            "type": msg_type,
            "timestamp": time.time(),
            "iso_time": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }

        messages = self._read_bus()
        messages.append(msg)

        cutoff = time.time() - self.MESSAGE_TTL
        messages = [m for m in messages if m.get("timestamp", 0) > cutoff]

        if len(messages) > self.MAX_MESSAGES:
            messages = messages[-self.MAX_MESSAGES:]

        try:
            self.BUS_FILE.write_text(json.dumps(messages, indent=2, default=str))
            self.sent_count += 1
        except Exception:
            pass

    def receive(self, since: float = 0, msg_type: str = None,
                exclude_self: bool = True) -> List[Dict]:
        """Receive messages from other builders."""
        messages = self._read_bus()
        self._last_read_time = time.time()

        filtered = []
        for m in messages:
            if m.get("timestamp", 0) <= since:
                continue
            if exclude_self and m.get("source") == self.builder_id:
                continue
            if msg_type and m.get("type") != msg_type:
                continue
            filtered.append(m)

        self.received_count += len(filtered)
        return filtered

    def _read_bus(self) -> List[Dict]:
        """Read all messages from the bus file."""
        if not self.BUS_FILE.exists():
            return []
        try:
            data = json.loads(self.BUS_FILE.read_text())
            if isinstance(data, list):
                return data
        except Exception:
            pass
        return []

    def announce_pipeline_complete(self, results: Dict) -> None:
        """Announce pipeline completion to other builders."""
        self.send("EVOLUTION_MILESTONE", {
            "event": "pipeline_complete",
            "builder": self.builder_id,
            "gates_discovered": results.get("scan", {}).get("total_gates", 0),
            "coherence": results.get("gate_field", {}).get("phase_coherence", 0),
            "research_health": results.get("research", {}).get("research_health", 0),
        })

    def announce_coherence_shift(self, old_coherence: float, new_coherence: float) -> None:
        """Announce a significant coherence change."""
        delta = abs(new_coherence - old_coherence)
        if delta > 0.01:
            self.send("COHERENCE_SHIFT", {
                "old": old_coherence,
                "new": new_coherence,
                "delta": delta,
                "direction": "up" if new_coherence > old_coherence else "down",
            })

    def status(self) -> Dict[str, Any]:
        """Return bus status."""
        messages = self._read_bus()
        now = time.time()
        active = [m for m in messages if m.get("timestamp", 0) > now - self.MESSAGE_TTL]
        sources = set(m.get("source", "?") for m in active)
        return {
            "subsystem": "InterBuilderFeedbackBus",
            "builder_id": self.builder_id,
            "active_messages": len(active),
            "total_messages": len(messages),
            "active_builders": list(sources),
            "sent_count": self.sent_count,
            "received_count": self.received_count,
        }
