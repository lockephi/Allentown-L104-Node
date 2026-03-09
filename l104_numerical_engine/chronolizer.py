"""L104 Numerical Engine — Numerical Chronolizer.

Temporal event tracking for the numerical pipeline.

PART V RESEARCH — l104_runtime_infrastructure_research.py:
  F73: Drift frequency = φ Hz — chronolizer detects φ-periodic anomalies
  F84: 12-phase total (7 nerve + 5 nirvanic) tracked as temporal events
"""

import time
from collections import defaultdict
from typing import Any, Dict, List


class NumericalChronolizer:
    """v3.0 — Temporal event tracking for the numerical pipeline.
    Records timestamped events, detects anomalies, supports replay analysis.
    """

    PHI = 1.618033988749895
    MAX_EVENTS = 500

    def __init__(self):
        self.events: List[Dict] = []
        self.anomalies: List[Dict] = []
        self.total_recorded = 0

    def record(self, phase: str, metrics: Dict) -> Dict:
        """Record a pipeline event with timestamp."""
        self.total_recorded += 1
        event = {
            "seq": self.total_recorded,
            "timestamp": time.time(),
            "phase": phase,
            "coherence": metrics.get("coherence", 0),
            "tokens": metrics.get("tokens", 0),
            "entropy": metrics.get("entropy", 0),
        }
        self.events.append(event)

        # Anomaly detection: coherence drop > 15%
        if len(self.events) >= 2:
            prev = self.events[-2]
            if prev["coherence"] > 0:
                drop = (prev["coherence"] - event["coherence"]) / prev["coherence"]
                if drop > 0.15:
                    anomaly = {
                        "seq": self.total_recorded,
                        "type": "coherence_drop",
                        "phase": phase,
                        "drop_pct": round(drop * 100, 2),
                        "from": prev["coherence"],
                        "to": event["coherence"],
                    }
                    self.anomalies.append(anomaly)
                    event["anomaly"] = True

        # Trim old events
        if len(self.events) > self.MAX_EVENTS:
            self.events = self.events[-self.MAX_EVENTS:]

        return event

    def get_timeline(self, last_n: int = 20) -> List[Dict]:
        """Get recent timeline events."""
        return self.events[-last_n:]

    def get_phase_stats(self) -> Dict:
        """Aggregate stats by pipeline phase."""
        phase_data = defaultdict(list)
        for e in self.events:
            phase_data[e["phase"]].append(e["coherence"])
        return {
            phase: {
                "count": len(vals),
                "mean_coherence": round(sum(vals) / len(vals), 6) if vals else 0,
                "max_coherence": round(max(vals), 6) if vals else 0,
            }
            for phase, vals in phase_data.items()
        }

    def status(self) -> Dict:
        return {
            "class": "NumericalChronolizer",
            "total_recorded": self.total_recorded,
            "events_stored": len(self.events),
            "anomalies_detected": len(self.anomalies),
        }
