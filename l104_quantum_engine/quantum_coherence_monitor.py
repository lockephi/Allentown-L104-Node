"""
l104_quantum_engine/quantum_coherence_monitor.py — Quantum Coherence Monitor v1.0.0

Real-time quantum coherence monitoring across the entire L104 system.
Tracks decoherence, manages entanglement health, and maintains
sacred resonance alignment.
"""

import time
import math
import threading
from typing import Dict, Any, List, Optional, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum

# Sacred constants
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
TAU = 1 / PHI
VOID_CONSTANT = 1.0416180339887497
ZENITH_HZ = 3727.84


class CoherenceLevel(Enum):
    """Quantum coherence classification."""
    QUANTUM_CRITICAL = 0.0   # Below 0.1 - system failure
    DECOHERENT = 0.1        # 0.1-0.3 - major issues
    DEGRADED = 0.3          # 0.3-0.5 - suboptimal
    STABLE = 0.5            # 0.5-0.7 - functional
    COHERENT = 0.7          # 0.7-0.9 - good
    SACRED_LOCKED = 0.9     # 0.9+ - optimal


@dataclass
class CoherenceSample:
    """Single coherence measurement sample."""
    timestamp: float
    coherence: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class QuantumCoherenceMonitor:
    """Monitors quantum coherence across all L104 subsystems.

    Tracks:
    - Real-time coherence levels
    - Decoherence rates
    - Entanglement health
    - Sacred resonance drift
    - System-wide coherence flow
    """

    def __init__(self, window_size: int = 10000):
        self.window_size = window_size
        self.coherence_history: deque = deque(maxlen=window_size)
        self.subsystem_coherence: Dict[str, float] = {}
        self.entanglement_health: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._monitoring_active = True

        # Thresholds
        self.warning_threshold = 0.5
        self.critical_threshold = 0.3
        self.sacred_threshold = 0.9

    def record_coherence(self, coherence: float, source: str = "unknown",
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record a coherence measurement."""
        sample = CoherenceSample(
            timestamp=time.time(),
            coherence=coherence,
            source=source,
            metadata=metadata or {},
        )

        with self._lock:
            self.coherence_history.append(sample)
            self.subsystem_coherence[source] = coherence

    def get_coherence_level(self, coherence: float) -> CoherenceLevel:
        """Classify coherence level."""
        if coherence >= 0.9:
            return CoherenceLevel.SACRED_LOCKED
        elif coherence >= 0.7:
            return CoherenceLevel.COHERENT
        elif coherence >= 0.5:
            return CoherenceLevel.STABLE
        elif coherence >= 0.3:
            return CoherenceLevel.DEGRADED
        elif coherence >= 0.1:
            return CoherenceLevel.DECOHERENT
        else:
            return CoherenceLevel.QUANTUM_CRITICAL

    def compute_trend(self, source: Optional[str] = None,
                     window: int = 100) -> Dict[str, Any]:
        """Compute coherence trend over time window."""
        with self._lock:
            if not self.coherence_history:
                return {'status': 'no_data', 'trend': 0.0}

            # Filter by source if specified
            samples = [
                s for s in self.coherence_history
                if source is None or s.source == source
            ]

            if len(samples) < 2:
                return {'status': 'insufficient_data', 'trend': 0.0}

            # Recent samples
            recent = samples[-window:]

            # Linear regression for trend
            n = len(recent)
            x = list(range(n))
            y = [s.coherence for s in recent]

            mean_x = sum(x) / n
            mean_y = sum(y) / n

            numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
            denominator = sum((xi - mean_x) ** 2 for xi in x)

            slope = numerator / denominator if denominator > 0 else 0

            return {
                'status': 'computed',
                'trend': slope,
                'current': y[-1] if y else 0.0,
                'average': mean_y,
                'min': min(y) if y else 0.0,
                'max': max(y) if y else 0.0,
                'samples': n,
            }

    def check_health(self) -> Dict[str, Any]:
        """Check overall system coherence health."""
        with self._lock:
            if not self.subsystem_coherence:
                return {'status': 'unknown', 'overall': 0.0}

            # Average coherence
            avg_coherence = sum(self.subsystem_coherence.values()) / len(self.subsystem_coherence)

            # Classification
            level = self.get_coherence_level(avg_coherence)

            # Subsystem breakdown
            subsystems = {
                name: {
                    'coherence': coh,
                    'level': self.get_coherence_level(coh).name,
                }
                for name, coh in self.subsystem_coherence.items()
            }

            # Critical subsystems
            critical = [
                name for name, coh in self.subsystem_coherence.items()
                if coh < self.critical_threshold
            ]

            warnings = [
                name for name, coh in self.subsystem_coherence.items()
                if self.critical_threshold <= coh < self.warning_threshold
            ]

            return {
                'status': level.name,
                'overall_coherence': avg_coherence,
                'level': level.value,
                'subsystems': subsystems,
                'critical_count': len(critical),
                'warning_count': len(warnings),
                'critical_list': critical,
                'warning_list': warnings,
            }

    def predict_decoherence(self, source: str,
                           horizon_seconds: float = 60.0) -> Dict[str, Any]:
        """Predict decoherence at future time horizon."""
        trend = self.compute_trend(source)

        if trend['status'] != 'computed':
            return {'status': 'prediction_failed', 'coherence': None}

        current = trend['current']
        slope = trend['trend']

        # Extrapolate with exponential decay model
        predicted = current * math.exp(slope * horizon_seconds)
        predicted = max(0.0, min(1.0, predicted))

        # Time to threshold
        if slope < 0:
            time_to_critical = (0.3 - current) / slope if slope < 0 else float('inf')
            time_to_decoherent = (0.1 - current) / slope if slope < 0 else float('inf')
        else:
            time_to_critical = float('inf')
            time_to_decoherent = float('inf')

        return {
            'status': 'predicted',
            'current': current,
            'predicted': predicted,
            'horizon_seconds': horizon_seconds,
            'time_to_critical': time_to_critical if time_to_critical > 0 else float('inf'),
            'time_to_decoherent': time_to_decoherent if time_to_decoherent > 0 else float('inf'),
        }

    def get_sacred_alignment(self) -> float:
        """Compute sacred alignment score based on GOD_CODE resonance."""
        with self._lock:
            if not self.coherence_history:
                return 0.0

            recent = list(self.coherence_history)[-100:]
            if not recent:
                return 0.0

            # Compute phase alignment with GOD_CODE
            avg_coherence = sum(s.coherence for s in recent) / len(recent)
            phase_component = math.cos(GOD_CODE % (2 * math.pi))

            return avg_coherence * abs(phase_component) * PHI

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive coherence metrics."""
        with self._lock:
            return {
                'samples': len(self.coherence_history),
                'subsystems': len(self.subsystem_coherence),
                'current_coherence': {
                    name: coh
                    for name, coh in self.subsystem_coherence.items()
                },
                'sacred_alignment': self.get_sacred_alignment(),
                'health': self.check_health(),
            }


class EntanglementHealthMonitor:
    """Monitors health of quantum entanglements across the system."""

    def __init__(self):
        self.entanglements: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def register_entanglement(self, ent_id: str,
                              node_a: str, node_b: str,
                              initial_fidelity: float = 1.0) -> None:
        """Register a new entanglement."""
        with self._lock:
            self.entanglements[ent_id] = {
                'id': ent_id,
                'node_a': node_a,
                'node_b': node_b,
                'fidelity': initial_fidelity,
                'established': time.time(),
                'last_check': time.time(),
                'purifications': 0,
            }

    def update_fidelity(self, ent_id: str, fidelity: float) -> None:
        """Update entanglement fidelity."""
        with self._lock:
            if ent_id in self.entanglements:
                self.entanglements[ent_id]['fidelity'] = fidelity
                self.entanglements[ent_id]['last_check'] = time.time()

    def get_health_report(self) -> Dict[str, Any]:
        """Get entanglement health report."""
        with self._lock:
            if not self.entanglements:
                return {'status': 'no_entanglements', 'count': 0}

            fidelities = [e['fidelity'] for e in self.entanglements.values()]

            healthy = sum(1 for f in fidelities if f > 0.9)
            degraded = sum(1 for f in fidelities if 0.5 <= f <= 0.9)
            critical = sum(1 for f in fidelities if f < 0.5)

            return {
                'status': 'healthy' if healthy > degraded + critical else 'degraded',
                'total': len(self.entanglements),
                'healthy': healthy,
                'degraded': degraded,
                'critical': critical,
                'avg_fidelity': sum(fidelities) / len(fidelities),
                'min_fidelity': min(fidelities),
                'max_fidelity': max(fidelities),
            }


__all__ = [
    'QuantumCoherenceMonitor',
    'EntanglementHealthMonitor',
    'CoherenceLevel',
    'CoherenceSample',
]