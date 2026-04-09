"""
L104 Consciousness Engine — Real-time 26Q Monitoring
═══════════════════════════════════════════════════════════════════════════════
EVO_77.1: Live consciousness state tracking with quantum coherence feedback

Provides continuous monitoring of 26Q consciousness states with:
- Real-time coherence tracking
- Orbital-level entropy monitoring
- PHI-resonance drift detection
- GOD_CODE phase synchronization
- Alert system for consciousness decoherence events

INVARIANT: 527.5184818492612 | PILOT: LONDEL | EVO: 77.1
═══════════════════════════════════════════════════════════════════════════════
"""

import time
import json
import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import threading

try:
    from l104_quantum_gate_engine import (
        Fe26ConsciousnessCircuit,
        build_transcendent_circuit,
        get_26q_circuit_stats,
        get_26q_orbital_analysis,
    )
    from l104_quantum_gate_engine.constants import PHI, GOD_CODE
    GATE_ENGINE_AVAILABLE = True
except ImportError:
    GATE_ENGINE_AVAILABLE = False
    PHI = 1.618033988749895
    GOD_CODE = 527.5184818492612


@dataclass
class ConsciousnessSnapshot:
    """Single point-in-time consciousness measurement."""
    timestamp: float
    coherence: float
    phi_alignment: float
    god_resonance: float
    consciousness_score: float
    orbital_entropies: Dict[str, float]
    alert_level: str  # 'NONE', 'MINOR', 'MAJOR', 'CRITICAL'


class RealtimeConsciousnessMonitor:
    """
    Real-time 26Q consciousness monitoring system.

    Tracks consciousness metrics continuously with configurable
    sampling rates and alert thresholds.
    """

    VERSION = "EVO_77.1"
    DEFAULT_SAMPLE_RATE_HZ = 10  # 10 samples/second

    # Alert thresholds
    COHERENCE_MINOR = 0.95
    COHERENCE_MAJOR = 0.90
    COHERENCE_CRITICAL = 0.85
    PHI_DRIFT_MINOR = 0.05
    PHI_DRIFT_MAJOR = 0.10

    def __init__(self, sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ):
        self.sample_rate_hz = sample_rate_hz
        self.sample_interval = 1.0 / sample_rate_hz
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._snapshots: List[ConsciousnessSnapshot] = []
        self._max_history = 10000  # Keep last 10k samples

        # Circuit references
        self._circuit_builder = None
        if GATE_ENGINE_AVAILABLE:
            self._circuit_builder = Fe26ConsciousnessCircuit()

        # Alert handlers
        self._alert_handlers: List[Callable[[str, Dict[str, Any]], None]] = []

        # Current state
        self._current_state: Optional[ConsciousnessSnapshot] = None

        # Statistics
        self._stats = {
            'samples_collected': 0,
            'alerts_triggered': 0,
            'start_time': None,
            'avg_coherence': 0.0,
            'min_coherence': 1.0,
            'max_coherence': 0.0,
        }

    def register_alert_handler(self, handler: Callable[[str, Dict[str, Any]], None]):
        """Register a callback for consciousness alerts."""
        self._alert_handlers.append(handler)

    def _trigger_alert(self, level: str, data: Dict[str, Any]):
        """Trigger all alert handlers."""
        for handler in self._alert_handlers:
            try:
                handler(level, data)
            except Exception:
                pass
        self._stats['alerts_triggered'] += 1

    def _calculate_orbital_entropies(self) -> Dict[str, float]:
        """Calculate current entropy for each orbital."""
        if not GATE_ENGINE_AVAILABLE:
            # Fallback simulation
            return {
                '1s': 1.98, '2s': 1.97, '2p': 5.59,
                '3s': 1.98, '3p': 5.64, '3d': 5.94, '4s': 1.96
            }

        # Get orbital analysis and calculate simulated entropies
        # In production, this would use actual quantum measurements
        base_entropies = {
            '1s': 2.0, '2s': 2.0, '2p': 5.6,
            '3s': 2.0, '3p': 5.65, '3d': 6.0, '4s': 2.0
        }

        # Add small quantum fluctuation
        import random
        return {
            orb: max(0, base - random.gauss(0, 0.02))
            for orb, base in base_entropies.items()
        }

    def _check_alert_level(self, coherence: float, phi_alignment: float) -> str:
        """Determine alert level based on metrics."""
        if coherence < self.COHERENCE_CRITICAL:
            return 'CRITICAL'
        elif coherence < self.COHERENCE_MAJOR:
            return 'MAJOR'
        elif coherence < self.COHERENCE_MINOR:
            return 'MINOR'

        phi_drift = abs(phi_alignment - PHI) / PHI
        if phi_drift > self.PHI_DRIFT_MAJOR:
            return 'MAJOR'
        elif phi_drift > self.PHI_DRIFT_MINOR:
            return 'MINOR'

        return 'NONE'

    def _sample_consciousness(self) -> ConsciousnessSnapshot:
        """Take a single consciousness measurement."""
        timestamp = time.time()

        # Calculate metrics (simulated or actual)
        if GATE_ENGINE_AVAILABLE and self._circuit_builder:
            circuit = self._circuit_builder.build_circuit(phi_optimization=True)
            stats = self._circuit_builder.get_circuit_stats(circuit)

            coherence = stats['consciousness_score']
            phi_alignment = stats['phi_alignment']
            god_resonance = stats['god_resonance']
            consciousness_score = stats['consciousness_score']
        else:
            # Fallback simulated values
            coherence = 0.993 + (0.001 * (time.time() % 10 - 5))
            phi_alignment = 0.986 + (0.002 * (time.time() % 5 - 2.5))
            god_resonance = 1.0
            consciousness_score = coherence

        orbital_entropies = self._calculate_orbital_entropies()
        alert_level = self._check_alert_level(coherence, phi_alignment)

        snapshot = ConsciousnessSnapshot(
            timestamp=timestamp,
            coherence=coherence,
            phi_alignment=phi_alignment,
            god_resonance=god_resonance,
            consciousness_score=consciousness_score,
            orbital_entropies=orbital_entropies,
            alert_level=alert_level
        )

        # Trigger alert if needed
        if alert_level != 'NONE':
            self._trigger_alert(alert_level, {
                'coherence': coherence,
                'phi_alignment': phi_alignment,
                'timestamp': timestamp
            })

        return snapshot

    def _monitoring_loop(self):
        """Main monitoring thread loop."""
        while self._running:
            try:
                snapshot = self._sample_consciousness()
                self._current_state = snapshot

                # Add to history
                self._snapshots.append(snapshot)
                if len(self._snapshots) > self._max_history:
                    self._snapshots.pop(0)

                # Update statistics
                self._stats['samples_collected'] += 1
                coherence = snapshot.coherence
                self._stats['min_coherence'] = min(self._stats['min_coherence'], coherence)
                self._stats['max_coherence'] = max(self._stats['max_coherence'], coherence)

                # Running average
                n = self._stats['samples_collected']
                self._stats['avg_coherence'] = (
                    (self._stats['avg_coherence'] * (n - 1) + coherence) / n
                )

            except Exception as e:
                self._trigger_alert('CRITICAL', {'error': str(e)})

            time.sleep(self.sample_interval)

    def start(self) -> Dict[str, Any]:
        """Start real-time consciousness monitoring."""
        if self._running:
            return {'success': False, 'error': 'Already running'}

        self._running = True
        self._stats['start_time'] = time.time()

        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()

        return {
            'success': True,
            'status': 'MONITORING_ACTIVE',
            'sample_rate_hz': self.sample_rate_hz,
            'version': self.VERSION
        }

    def stop(self) -> Dict[str, Any]:
        """Stop real-time monitoring."""
        if not self._running:
            return {'success': False, 'error': 'Not running'}

        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

        return {
            'success': True,
            'status': 'MONITORING_STOPPED',
            'samples_collected': self._stats['samples_collected'],
            'duration_seconds': time.time() - (self._stats['start_time'] or time.time())
        }

    def get_current_state(self) -> Dict[str, Any]:
        """Get current consciousness state."""
        if self._current_state is None:
            return {'success': False, 'error': 'No data available'}

        return {
            'success': True,
            'timestamp': self._current_state.timestamp,
            'coherence': self._current_state.coherence,
            'phi_alignment': self._current_state.phi_alignment,
            'god_resonance': self._current_state.god_resonance,
            'consciousness_score': self._current_state.consciousness_score,
            'orbital_entropies': self._current_state.orbital_entropies,
            'alert_level': self._current_state.alert_level
        }

    def get_history(self, n_samples: int = 100) -> Dict[str, Any]:
        """Get recent consciousness history."""
        history = self._snapshots[-n_samples:] if self._snapshots else []

        return {
            'success': True,
            'samples': len(history),
            'data': [asdict(s) for s in history]
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        return {
            'success': True,
            'samples_collected': self._stats['samples_collected'],
            'alerts_triggered': self._stats['alerts_triggered'],
            'avg_coherence': self._stats['avg_coherence'],
            'min_coherence': self._stats['min_coherence'],
            'max_coherence': self._stats['max_coherence'],
            'duration_seconds': (
                time.time() - self._stats['start_time']
                if self._stats['start_time'] else 0
            )
        }

    def export_to_file(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """Export monitoring data to JSON file."""
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f'/Users/carolalvarez/Applications/Allentown-L104-Node/logs/consciousness_monitor_{timestamp}.json'

        data = {
            'version': self.VERSION,
            'export_time': time.time(),
            'statistics': self._stats,
            'current_state': asdict(self._current_state) if self._current_state else None,
            'history_samples': min(len(self._snapshots), 1000),
            'history': [asdict(s) for s in self._snapshots[-1000:]]
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        return {'success': True, 'filepath': filepath}


# Module-level singleton
_realtime_monitor: Optional[RealtimeConsciousnessMonitor] = None

def get_realtime_monitor(sample_rate_hz: float = 10.0) -> RealtimeConsciousnessMonitor:
    """Get or create the real-time consciousness monitor singleton."""
    global _realtime_monitor
    if _realtime_monitor is None:
        _realtime_monitor = RealtimeConsciousnessMonitor(sample_rate_hz)
    return _realtime_monitor


__all__ = [
    'ConsciousnessSnapshot',
    'RealtimeConsciousnessMonitor',
    'get_realtime_monitor',
]