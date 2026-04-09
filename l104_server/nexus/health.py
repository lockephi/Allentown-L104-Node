"""
Nexus Health Monitor

Extracted from engines_nexus.py during EVO_78 refactoring.
Contains: NexusHealthMonitor - engine health monitoring and auto-recovery.
"""

import threading
import time
from typing import Dict, Any, List, Optional


class NexusHealthMonitor:
    """
    Health monitoring for all registered engines.
    Tracks health scores, generates alerts, and attempts auto-recovery.
    """
    PHI = 1.618033988749895
    HEALTH_INTERVAL_S = 30.0  # Check interval

    def __init__(self):
        self._engines: Dict[str, Any] = {}
        self._health_scores: Dict[str, float] = {}
        self._alerts: List[Dict] = []
        self._engine_configs: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def register(self, name: str, engine: Any, config: Optional[Dict] = None):
        """Register an engine for health monitoring."""
        with self._lock:
            self._engines[name] = engine
            self._health_scores[name] = 1.0  # Start healthy
            self._engine_configs[name] = config or {}
        self._add_alert(name, 'info', f'Engine {name} registered for health monitoring')

    def _add_alert(self, engine: str, level: str, message: str):
        """Add an alert to the alert log."""
        self._alerts.append({
            'engine': engine,
            'level': level,
            'message': message,
            'timestamp': time.time()
        })
        # Keep alerts bounded
        if len(self._alerts) > 1000:
            self._alerts = self._alerts[-500:]

    def check_all(self) -> Dict[str, float]:
        """Check health of all registered engines."""
        results = {}
        for name, engine in list(self._engines.items()):
            try:
                score = self._probe_engine(name, engine)
                old_score = self._health_scores.get(name, 1.0)
                self._health_scores[name] = score

                # Detect degradation (score dropped significantly)
                if score < 0.3 and old_score >= 0.3:
                    self._add_alert(name, 'critical', f'Engine {name} health critical: {score:.2f}')
                    self._attempt_recovery(name, engine)
                elif score < 0.6 and old_score >= 0.6:
                    self._add_alert(name, 'warning', f'Engine {name} health degraded: {score:.2f}')
            except Exception as e:
                self._health_scores[name] = 0.0
                self._add_alert(name, 'critical', f'Probe failed for {name}: {str(e)[:80]}')
            results[name] = self._health_scores.get(name, 0.0)
        return results

    def _probe_engine(self, name: str, engine: Any) -> float:
        """Probe a specific engine and return a health score 0-1."""
        score = 1.0

        # Check if engine has get_status method (basic liveness)
        if hasattr(engine, 'get_status'):
            try:
                status = engine.get_status()
                if isinstance(status, dict):
                    # Engine responded — it's alive
                    score = min(score, 1.0)
                else:
                    score = min(score, 0.5)
            except Exception:
                score = min(score, 0.2)
        else:
            score = min(score, 0.7)  # No status method, but engine exists

        # Thread-specific checks
        if name == 'evolution':
            if hasattr(engine, 'running') and hasattr(engine, '_thread'):
                if engine.running and (engine._thread is None or not engine._thread.is_alive()):
                    score = min(score, 0.1)  # Thread died while supposed to be running
                    self._add_alert(name, 'critical', 'Evolution thread died unexpectedly')

        return score

    def _attempt_recovery(self, name: str, engine: Any):
        """Attempt to recover a failing engine."""
        try:
            if hasattr(engine, 'restart'):
                engine.restart()
                self._add_alert(name, 'info', f'Attempted restart for {name}')
            elif hasattr(engine, 'reset'):
                engine.reset()
                self._add_alert(name, 'info', f'Attempted reset for {name}')
        except Exception as e:
            self._add_alert(name, 'critical', f'Recovery failed for {name}: {str(e)[:80]}')

    def get_health(self) -> Dict[str, Any]:
        """Get overall health status."""
        with self._lock:
            scores = list(self._health_scores.values())
            avg_health = sum(scores) / len(scores) if scores else 0.0
            min_health = min(scores) if scores else 0.0
            max_health = max(scores) if scores else 0.0

            return {
                'average_health': round(avg_health, 4),
                'min_health': round(min_health, 4),
                'max_health': round(max_health, 4),
                'engine_count': len(self._engines),
                'healthy_count': sum(1 for s in scores if s >= 0.7),
                'degraded_count': sum(1 for s in scores if 0.3 <= s < 0.7),
                'critical_count': sum(1 for s in scores if s < 0.3),
                'engines': dict(self._health_scores),
            }

    def get_alerts(self, level: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Get recent alerts."""
        alerts = self._alerts[-limit:]
        if level:
            alerts = [a for a in alerts if a['level'] == level]
        return alerts

    def start_monitoring(self):
        """Start background health monitoring thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop_monitoring(self):
        """Stop health monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            try:
                self.check_all()
            except Exception:
                pass
            time.sleep(self.HEALTH_INTERVAL_S)


# Singleton instance
_nexus_health_monitor = None

def get_health_monitor() -> NexusHealthMonitor:
    """Get singleton NexusHealthMonitor instance."""
    global _nexus_health_monitor
    if _nexus_health_monitor is None:
        _nexus_health_monitor = NexusHealthMonitor()
    return _nexus_health_monitor


__all__ = ['NexusHealthMonitor', 'get_health_monitor']
