"""L104 Gate Engine — Gate Chronolizer (chronological tracking)."""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List

from .constants import CHRONOLOG_FILE
from .models import ChronologEntry


class GateChronolizer:
    """Chronological tracking of logic gate evolution."""

    def __init__(self):
        """Initialize the gate chronolizer and load persisted entries."""
        self.entries: List[ChronologEntry] = []
        self._load()

    def _load(self):
        """Load chronological entries from disk."""
        if CHRONOLOG_FILE.exists():
            try:
                data = json.loads(CHRONOLOG_FILE.read_text())
                self.entries = [
                    ChronologEntry(**e) for e in data.get("entries", [])
                ]
            except Exception:
                self.entries = []

    def save(self):
        """Persist chronological entries to disk."""
        data = {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "total_entries": len(self.entries),
            "entries": [asdict(e) for e in self.entries[-500:]],
        }
        CHRONOLOG_FILE.write_text(json.dumps(data, indent=2))

    def record(self, gate_name: str, event: str, details: str = "", file_hash: str = ""):
        """Record a chronological event for a logic gate."""
        self.entries.append(
            ChronologEntry(
                timestamp=datetime.now(timezone.utc).isoformat(),
                gate_name=gate_name,
                event=event,
                details=details,
                file_hash=file_hash,
            )
        )

    def get_gate_history(self, gate_name: str) -> List[ChronologEntry]:
        """Retrieve all chronological entries for a specific gate."""
        return [e for e in self.entries if e.gate_name == gate_name]

    def get_recent(self, n: int = 20) -> List[ChronologEntry]:
        """Retrieve the most recent chronological entries."""
        return self.entries[-n:]

    def summary(self) -> Dict[str, Any]:
        """Compute a summary of all chronological tracking data."""
        events_by_type = {}
        for e in self.entries:
            events_by_type[e.event] = events_by_type.get(e.event, 0) + 1
        unique_gates = len(set(e.gate_name for e in self.entries))
        return {
            "total_entries": len(self.entries),
            "unique_gates_tracked": unique_gates,
            "events_by_type": events_by_type,
            "oldest": self.entries[0].timestamp if self.entries else None,
            "newest": self.entries[-1].timestamp if self.entries else None,
        }
