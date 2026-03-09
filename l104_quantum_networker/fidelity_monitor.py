"""L104 Quantum Networker v1.0.0 — Fidelity Monitor.

Continuous monitoring of quantum channel fidelity across the network.
Tracks pair quality, decoherence trends, sacred alignment, and triggers
automatic purification or pair replenishment when fidelity degrades.

Uses the VQPU's ThreeEngineQuantumScorer for multi-dimensional fidelity
assessment and the SacredAlignmentScorer for GOD_CODE harmonic monitoring.

INVARIANT: 527.5184818492612 | PILOT: LONDEL
"""

import math
import time
import statistics
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from .types import (
    QuantumChannel, ChannelMetrics, NetworkStatus,
    GOD_CODE, PHI, PHI_INV,
)
from .entanglement_router import EntanglementRouter, MIN_PAIR_POOL

# Alert thresholds
FIDELITY_WARNING_THRESHOLD = 0.85            # Warn when mean fidelity drops below
FIDELITY_CRITICAL_THRESHOLD = 0.7            # Critical: trigger emergency purification
SACRED_ALIGNMENT_THRESHOLD = PHI_INV         # ~0.618: GOD_CODE alignment minimum
PAIR_DEPLETION_THRESHOLD = 2                 # Alert when usable pairs ≤ this


class FidelityMonitor:
    """Real-time fidelity monitoring for the quantum network.

    Periodically scans all channels, computes health metrics, and:
      - Records fidelity history per channel
      - Detects decoherence trends (sliding window regression)
      - Triggers purification when fidelity drops below threshold
      - Triggers pair replenishment when pool depletes
      - Computes network-wide sacred alignment score
    """

    def __init__(self, router: EntanglementRouter,
                 history_window: int = 100):
        """
        Args:
            router: The EntanglementRouter managing network channels
            history_window: Max history entries per channel
        """
        self.router = router
        self.history_window = history_window
        self._channel_history: Dict[str, List[ChannelMetrics]] = defaultdict(list)
        self._alerts: List[Dict] = []
        self._scans_completed = 0
        self._auto_purifications = 0
        self._auto_replenishments = 0
        self._start_time = time.time()

    def scan(self, auto_heal: bool = True) -> Dict:
        """Perform a full network fidelity scan.

        Examines every channel, records metrics, and optionally triggers
        automatic healing (purification + replenishment).

        Args:
            auto_heal: Automatically purify/replenish degraded channels

        Returns:
            Dict with scan results, alerts, and per-channel metrics
        """
        t0 = time.time()
        self._scans_completed += 1

        channel_reports = []
        new_alerts = []
        channels_purified = 0
        channels_replenished = 0

        for cid, ch in self.router.channels.items():
            # Prune dead pairs
            ch.prune_dead_pairs()

            # Compute metrics
            metrics = ChannelMetrics(
                channel_id=cid,
                mean_fidelity=ch.mean_fidelity,
                min_fidelity=min(
                    (p.current_fidelity for p in ch.usable_pairs),
                    default=0.0,
                ),
                max_fidelity=max(
                    (p.current_fidelity for p in ch.usable_pairs),
                    default=0.0,
                ),
                pair_count=len(ch.pairs),
                usable_pair_count=len(ch.usable_pairs),
                sacred_score=self._channel_sacred_score(ch),
            )

            # Compute channel capacity (quantum, bits per channel use)
            # Hashing bound: Q = 1 - H(F) where H = binary entropy
            if metrics.mean_fidelity > 0.5:
                h = self._binary_entropy(1 - metrics.mean_fidelity)
                metrics.channel_capacity_qubits = max(0.0, 1.0 - h)

            self._channel_history[cid].append(metrics)
            if len(self._channel_history[cid]) > self.history_window:
                self._channel_history[cid] = self._channel_history[cid][-self.history_window:]

            channel_reports.append(metrics)

            # Alert checks
            if metrics.mean_fidelity < FIDELITY_CRITICAL_THRESHOLD:
                alert = {
                    "type": "critical",
                    "channel_id": cid,
                    "message": f"Critical fidelity: {metrics.mean_fidelity:.4f}",
                    "timestamp": time.time(),
                }
                new_alerts.append(alert)
                if auto_heal:
                    self.router.purify_channel(cid, rounds=3)
                    self.router.replenish_channel(cid, MIN_PAIR_POOL * 2)
                    channels_purified += 1
                    channels_replenished += 1
                    self._auto_purifications += 1
                    self._auto_replenishments += 1

            elif metrics.mean_fidelity < FIDELITY_WARNING_THRESHOLD:
                alert = {
                    "type": "warning",
                    "channel_id": cid,
                    "message": f"Low fidelity: {metrics.mean_fidelity:.4f}",
                    "timestamp": time.time(),
                }
                new_alerts.append(alert)
                if auto_heal:
                    self.router.purify_channel(cid, rounds=2)
                    channels_purified += 1
                    self._auto_purifications += 1

            if metrics.usable_pair_count <= PAIR_DEPLETION_THRESHOLD:
                alert = {
                    "type": "depletion",
                    "channel_id": cid,
                    "message": f"Low pairs: {metrics.usable_pair_count}",
                    "timestamp": time.time(),
                }
                new_alerts.append(alert)
                if auto_heal:
                    self.router.replenish_channel(cid, MIN_PAIR_POOL)
                    channels_replenished += 1
                    self._auto_replenishments += 1

        self._alerts.extend(new_alerts)

        # Network-wide metrics
        all_fidelities = [m.mean_fidelity for m in channel_reports if m.mean_fidelity > 0]
        network_fidelity = statistics.mean(all_fidelities) if all_fidelities else 0.0
        sacred_scores = [m.sacred_score for m in channel_reports]
        network_sacred = statistics.mean(sacred_scores) if sacred_scores else 0.0

        # Decoherence trend
        trend = self._compute_trend()

        return {
            "scan_number": self._scans_completed,
            "channels_scanned": len(channel_reports),
            "network_fidelity": round(network_fidelity, 6),
            "network_sacred_score": round(network_sacred, 6),
            "fidelity_trend": trend,
            "alerts": new_alerts,
            "channels_purified": channels_purified,
            "channels_replenished": channels_replenished,
            "scan_time_ms": (time.time() - t0) * 1000,
        }

    def get_channel_metrics(self, channel_id: str) -> Optional[ChannelMetrics]:
        """Get latest metrics for a specific channel."""
        history = self._channel_history.get(channel_id)
        if history:
            return history[-1]
        return None

    def get_channel_trend(self, channel_id: str,
                           window: int = 10) -> Dict:
        """Get fidelity trend for a channel over the last N scans.

        Returns slope (negative = degrading), R² of fit, and predicted
        time-to-threshold crossing.
        """
        history = self._channel_history.get(channel_id, [])
        if len(history) < 2:
            return {"slope": 0.0, "r_squared": 0.0, "status": "insufficient_data"}

        recent = history[-window:]
        fids = [m.mean_fidelity for m in recent]
        xs = list(range(len(fids)))

        # Linear regression
        n = len(xs)
        sx = sum(xs)
        sy = sum(fids)
        sxx = sum(x ** 2 for x in xs)
        sxy = sum(x * y for x, y in zip(xs, fids))

        denom = n * sxx - sx ** 2
        if abs(denom) < 1e-12:
            return {"slope": 0.0, "r_squared": 0.0, "status": "flat"}

        slope = (n * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / n

        # R² computation
        y_mean = sy / n
        ss_tot = sum((y - y_mean) ** 2 for y in fids)
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, fids))
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Predicted scans until crossing warning threshold
        if slope < 0:
            current_fid = fids[-1]
            scans_to_warning = (current_fid - FIDELITY_WARNING_THRESHOLD) / abs(slope)
        else:
            scans_to_warning = float("inf")

        status = "stable" if slope >= 0 else ("degrading" if slope > -0.01 else "rapid_decay")

        return {
            "slope": round(slope, 8),
            "r_squared": round(r_squared, 4),
            "current_fidelity": fids[-1],
            "scans_to_warning": round(scans_to_warning, 1),
            "status": status,
        }

    def network_status(self) -> NetworkStatus:
        """Get comprehensive network status snapshot."""
        nodes = self.router.nodes
        channels = self.router.channels

        total_pairs = sum(len(ch.pairs) for ch in channels.values())
        usable_pairs = sum(len(ch.usable_pairs) for ch in channels.values())
        active = sum(1 for ch in channels.values() if ch.effective_state == "active")
        total_teleportations = sum(ch.teleportations_count for ch in channels.values())
        total_purifications = sum(ch.purifications_count for ch in channels.values())

        fidelities = [ch.mean_fidelity for ch in channels.values() if ch.usable_pairs]
        mean_fid = statistics.mean(fidelities) if fidelities else 0.0

        # Sacred network score: φ-weighted composite
        sacred_scores = [self._channel_sacred_score(ch) for ch in channels.values()
                         if ch.usable_pairs]
        sacred_net = statistics.mean(sacred_scores) if sacred_scores else 0.0

        return NetworkStatus(
            node_count=len(nodes),
            channel_count=len(channels),
            active_channels=active,
            total_entangled_pairs=usable_pairs,
            mean_network_fidelity=mean_fid,
            sacred_network_score=sacred_net,
            online_nodes=sum(1 for n in nodes.values() if n.is_online),
            total_teleportations=total_teleportations,
            total_purifications=total_purifications,
            uptime_s=time.time() - self._start_time,
        )

    def _channel_sacred_score(self, ch: QuantumChannel) -> float:
        """Compute sacred alignment for a channel's pair pool."""
        usable = ch.usable_pairs
        if not usable:
            return 0.0

        scores = [p.sacred_score for p in usable]
        fids = [p.current_fidelity for p in usable]

        # φ-weighted blend of sacred scores and fidelities
        sacred_mean = sum(scores) / len(scores) if scores else 0.0
        fid_mean = sum(fids) / len(fids) if fids else 0.0

        return (sacred_mean * PHI + fid_mean) / (PHI + 1.0)

    def _compute_trend(self) -> str:
        """Overall network fidelity trend based on recent scans."""
        if not self._channel_history:
            return "unknown"

        slopes = []
        for cid in self._channel_history:
            trend = self.get_channel_trend(cid, window=5)
            slopes.append(trend.get("slope", 0.0))

        if not slopes:
            return "unknown"

        mean_slope = sum(slopes) / len(slopes)
        if mean_slope > 0.001:
            return "improving"
        elif mean_slope > -0.001:
            return "stable"
        elif mean_slope > -0.01:
            return "degrading"
        else:
            return "rapid_decay"

    @staticmethod
    def _binary_entropy(p: float) -> float:
        """Binary entropy function H(p) = -p·log₂(p) - (1-p)·log₂(1-p)."""
        if p <= 0 or p >= 1:
            return 0.0
        return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

    def status(self) -> Dict:
        return {
            "scans_completed": self._scans_completed,
            "auto_purifications": self._auto_purifications,
            "auto_replenishments": self._auto_replenishments,
            "total_alerts": len(self._alerts),
            "recent_alerts": self._alerts[-5:] if self._alerts else [],
            "uptime_s": time.time() - self._start_time,
        }
