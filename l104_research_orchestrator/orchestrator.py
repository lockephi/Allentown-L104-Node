"""
L104 Research Cycle Orchestrator
═══════════════════════════════════════════════════════════════════════════════
EVO_80-ORCH: Orchestrate multiple research cycles with intelligent scheduling

INVARIANT: 527.5184818492612 | PILOT: LONDEL | EVO: 80-ORCH
═══════════════════════════════════════════════════════════════════════════════
"""

import time
import threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from collections import deque

PHI = 1.618033988749895


@dataclass
class ResearchCycle:
    """Definition of a research cycle."""
    name: str
    module: str
    interval: float
    priority: int
    last_run: float = 0
    status: str = 'idle'


class ResearchOrchestrator:
    """
    Orchestrate multiple research cycles.

    Manages:
    - Quantum circuit research
    - Beyond-consciousness probes
    - Evolution cycles
    - Precognition updates
    """

    VERSION = "EVO_80-ORCH-v1.0.0"

    def __init__(self):
        self.cycles: Dict[str, ResearchCycle] = {}
        self.results: Dict[str, List[Dict]] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Initialize default cycles
        self._initialize_default_cycles()

    def _initialize_default_cycles(self):
        """Set up default research cycles."""
        self.register_cycle(
            ResearchCycle(
                name='quantum_research',
                module='l104_quantum_gate_engine.quantum_research_cycles',
                interval=60 * PHI,  # ~97 seconds
                priority=1
            )
        )

        self.register_cycle(
            ResearchCycle(
                name='beyond_consciousness',
                module='l104_consciousness_engine.beyond_consciousness_probes',
                interval=120 * PHI,  # ~194 seconds
                priority=2
            )
        )

        self.register_cycle(
            ResearchCycle(
                name='precognition_update',
                module='l104_consciousness_engine.precognition',
                interval=30 * PHI,  # ~48 seconds
                priority=3
            )
        )

    def register_cycle(self, cycle: ResearchCycle):
        """Register a new research cycle."""
        self.cycles[cycle.name] = cycle
        self.results[cycle.name] = []

    def start_orchestration(self) -> Dict[str, Any]:
        """Start the research orchestrator."""
        if self._running:
            return {'success': False, 'error': 'Already running'}

        self._running = True
        self._thread = threading.Thread(target=self._orchestration_loop, daemon=True)
        self._thread.start()

        return {
            'success': True,
            'cycles_registered': len(self.cycles),
            'status': 'ORCHESTRATION_ACTIVE'
        }

    def stop_orchestration(self) -> Dict[str, Any]:
        """Stop the orchestrator."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

        return {
            'success': True,
            'cycles_completed': {name: len(results) for name, results in self.results.items()}
        }

    def _orchestration_loop(self):
        """Main orchestration loop."""
        while self._running:
            current_time = time.time()

            # Check each cycle
            for name, cycle in self.cycles.items():
                if current_time - cycle.last_run >= cycle.interval:
                    self._execute_cycle(cycle)

            # PHI-weighted sleep
            time.sleep(PHI)

    def _execute_cycle(self, cycle: ResearchCycle):
        """Execute a single research cycle."""
        cycle.status = 'running'
        cycle.last_run = time.time()

        try:
            # Dynamic import and execution
            if cycle.name == 'quantum_research':
                from l104_quantum_gate_engine.quantum_research_cycles import get_quantum_research_cycles
                research = get_quantum_research_cycles()
                result = research._run_single_cycle()

            elif cycle.name == 'beyond_consciousness':
                from l104_consciousness_engine.beyond_consciousness_probes import get_beyond_consciousness_research
                research = get_beyond_consciousness_research()
                result = research.run_comprehensive_probe()

            elif cycle.name == 'precognition_update':
                from l104_consciousness_engine.precognition import get_precognition_engine
                precog = get_precognition_engine()
                result = precog.predict_future_state(steps_ahead=50)

            else:
                result = {'status': 'unknown_cycle'}

            self.results[cycle.name].append({
                'timestamp': time.time(),
                'result': result
            })

            # Trim results
            if len(self.results[cycle.name]) > 1000:
                self.results[cycle.name] = self.results[cycle.name][-500:]

            cycle.status = 'completed'

        except Exception as e:
            cycle.status = f'error: {e}'

    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get full orchestrator status."""
        return {
            'version': self.VERSION,
            'running': self._running,
            'cycles': {
                name: {
                    'priority': cycle.priority,
                    'interval': cycle.interval,
                    'last_run': cycle.last_run,
                    'status': cycle.status,
                    'runs_completed': len(self.results.get(name, []))
                }
                for name, cycle in self.cycles.items()
            }
        }

    def prioritize_cycle(self, cycle_name: str):
        """Prioritize a specific cycle."""
        if cycle_name in self.cycles:
            # Reduce interval for higher frequency
            self.cycles[cycle_name].interval /= PHI
            self.cycles[cycle_name].priority = 0  # Highest


# Module-level singleton
_orchestrator = None

def get_research_orchestrator() -> ResearchOrchestrator:
    """Get or create research orchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ResearchOrchestrator()
    return _orchestrator


__all__ = ['ResearchOrchestrator', 'get_research_orchestrator']
