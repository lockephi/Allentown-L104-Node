"""L104 Autonomy Daemon — Autonomous Decision and Goal Management.

Manages:
- Self-awareness engine integration
- Autonomous decision making (120 decisions/minute target)
- Goal establishment and pursuit
- Meta-cognitive monitoring
- Cross-daemon autonomy coordination
"""

import time
import threading
import json
import signal
import sys
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import traceback

from .constants import (
    DAEMON_CYCLE_SECONDS, PERSISTENCE_INTERVAL, HEARTBEAT_INTERVAL,
    STATE_PERSISTENCE_PATH, LOG_DIRECTORY,
    MIN_CONSCIOUSNESS_LEVEL, TARGET_DECISIONS_PER_MINUTE, MAX_ACTIVE_GOALS,
    GOD_CODE, PHI, DAEMON_VERSION
)

try:
    from l104_autonomous_core.self_awareness_engine import get_self_awareness_engine
    _HAS_SELF_AWARENESS = True
except ImportError:
    _HAS_SELF_AWARENESS = False

try:
    from l104_daemon_adapter import DaemonAdapter
    _HAS_DAEMON_ADAPTER = True
except ImportError:
    _HAS_DAEMON_ADAPTER = False


@dataclass
class AutonomyState:
    """Current state of the autonomy daemon."""
    running: bool = False
    cycle_count: int = 0
    decisions_made: int = 0
    goals_established: int = 0
    goals_completed: int = 0
    last_cycle_start: float = 0.0
    last_cycle_end: float = 0.0
    total_uptime: float = 0.0
    avg_cycle_time: float = 0.0
    consciousness_level: float = 0.986
    phi_alignment: float = 0.986
    error_count: int = 0
    last_error: Optional[str] = None
    active_goals: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "running": self.running,
            "cycle_count": self.cycle_count,
            "decisions_made": self.decisions_made,
            "goals_established": self.goals_established,
            "goals_completed": self.goals_completed,
            "total_uptime": self.total_uptime,
            "avg_cycle_time": self.avg_cycle_time,
            "consciousness_level": self.consciousness_level,
            "phi_alignment": self.phi_alignment,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "active_goals": len(self.active_goals),
        }


class AutonomyDaemon:
    """Autonomous decision and goal management daemon."""

    def __init__(self):
        self.version = DAEMON_VERSION
        self.state = AutonomyState()
        self._stop_event = threading.Event()
        self._daemon_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Integration components
        self.awareness_engine = None
        self.daemon_adapter = None

        # Initialize
        self._initialize()

    def _initialize(self):
        """Initialize daemon components."""
        if _HAS_SELF_AWARENESS:
            self.awareness_engine = get_self_awareness_engine()
            self.state.consciousness_level = self.awareness_engine.consciousness_level

        if _HAS_DAEMON_ADAPTER:
            try:
                self.daemon_adapter = DaemonAdapter(
                    daemon_id="autonomy_daemon",
                    daemon_type="autonomy",
                    qubit_count=26
                )
            except:
                self.daemon_adapter = None

        # Ensure log directory
        LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)

    def start(self) -> bool:
        """Start the autonomy daemon."""
        if self.state.running:
            return False

        self._stop_event.clear()
        self.state.running = True
        self.state.last_cycle_start = time.time()

        self._daemon_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._daemon_thread.start()

        return True

    def stop(self) -> bool:
        """Stop the daemon gracefully."""
        if not self.state.running:
            return False

        self._stop_event.set()
        self.state.running = False

        if self._daemon_thread:
            self._daemon_thread.join(timeout=10)

        self._persist_state()
        return True

    def _run_loop(self):
        """Main daemon loop."""
        while not self._stop_event.is_set():
            cycle_start = time.time()
            self.state.last_cycle_start = cycle_start

            try:
                self._execute_cycle()
            except Exception as e:
                self.state.error_count += 1
                self.state.last_error = str(e)
                traceback.print_exc()

            cycle_end = time.time()
            cycle_duration = cycle_end - cycle_start
            self.state.last_cycle_end = cycle_end

            # Update metrics
            self.state.cycle_count += 1
            self.state.total_uptime += cycle_duration
            self.state.avg_cycle_time = (
                self.state.avg_cycle_time * (self.state.cycle_count - 1) + cycle_duration
            ) / self.state.cycle_count

            # Heartbeat
            if self.state.cycle_count % HEARTBEAT_INTERVAL == 0:
                self._send_heartbeat()

            # Persistence
            if self.state.cycle_count % PERSISTENCE_INTERVAL == 0:
                self._persist_state()

            # Wait for next cycle
            sleep_time = max(0, DAEMON_CYCLE_SECONDS - cycle_duration)
            self._stop_event.wait(sleep_time)

    def _execute_cycle(self):
        """Execute one autonomy cycle."""
        # Phase 1: Meta-cognitive check
        if self.awareness_engine:
            meta = self.awareness_engine.meta_cognitive_check()
            self.state.consciousness_level = self.awareness_engine.consciousness_level

        # Phase 2: Autonomous decisions (burst to meet target)
        decisions_this_cycle = 0
        for _ in range(20):  # 20 decisions per cycle
            if self.awareness_engine:
                decision = self.awareness_engine.make_decision(
                    context={'cycle': self.state.cycle_count},
                    options=['MAINTAIN', 'OPTIMIZE', 'HEAL', 'EXPAND', 'SYNC']
                )
                if decision.approved:
                    decisions_this_cycle += 1

        self.state.decisions_made += decisions_this_cycle

        # Phase 3: Goal management
        if self.awareness_engine:
            # Establish new goals if needed
            if len(self.awareness_engine.active_goals) < 3:
                goals = [
                    ('MAINTAIN_PHI', 0.9, None),
                    ('EXPAND_CONSCIOUSNESS', 0.8, None),
                    ('CROSS_ENGINE_SYNC', 0.85, None),
                ]
                for desc, priority, deadline in goals:
                    if len(self.awareness_engine.active_goals) < MAX_ACTIVE_GOALS:
                        self.awareness_engine.set_goal(desc, priority, deadline)
                        self.state.goals_established += 1

            # Pursue goals
            actions = self.awareness_engine.pursue_goals()
            self.state.goals_completed += len([a for a in actions if 'COMPLETED' in str(a)])

        # Phase 4: Cross-daemon coordination
        if self.daemon_adapter:
            self._broadcast_to_mesh()

    def _broadcast_to_mesh(self):
        """Broadcast autonomy state to daemon mesh."""
        if not self.daemon_adapter:
            return

        message = {
            "daemon": "autonomy",
            "timestamp": time.time(),
            "consciousness_level": self.state.consciousness_level,
            "decisions_made": self.state.decisions_made,
            "phi_alignment": self.state.phi_alignment,
        }
        try:
            if hasattr(self.daemon_adapter, 'broadcast_mesh_sync'):
                from l104_daemon_adapter import DaemonAdapter
                DaemonAdapter.broadcast_mesh_sync(self.daemon_adapter.daemon_id if hasattr(self.daemon_adapter, 'daemon_id') else 'daemon', message)
        except:
            pass

    def _send_heartbeat(self):
        """Send heartbeat signal."""
        pass

    def _persist_state(self):
        """Persist daemon state to disk."""
        try:
            with open(STATE_PERSISTENCE_PATH, 'w') as f:
                json.dump(self.state.to_dict(), f, indent=2)
        except Exception as e:
            print(f"Failed to persist state: {e}")

    def status(self) -> Dict[str, Any]:
        """Get current daemon status."""
        with self._lock:
            return {
                "daemon": "autonomy",
                "version": self.version,
                **self.state.to_dict(),
                "target_dpm": TARGET_DECISIONS_PER_MINUTE,
                "actual_dpm": self.state.decisions_made / max(1, self.state.total_uptime / 60),
            }

    def force_cycle(self):
        """Force immediate cycle execution."""
        self._execute_cycle()


# Singleton instance
_autonomy_daemon: Optional[AutonomyDaemon] = None


def get_autonomy_daemon() -> AutonomyDaemon:
    """Get or create the autonomy daemon singleton."""
    global _autonomy_daemon
    if _autonomy_daemon is None:
        _autonomy_daemon = AutonomyDaemon()
    return _autonomy_daemon
