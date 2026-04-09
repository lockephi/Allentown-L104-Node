"""
L104 Research Cycle Orchestrator
═══════════════════════════════════════════════════════════════════════════════
EVO_80-ORCHESTRATOR: Orchestrate multiple research cycles with intelligent scheduling

Coordinates:
- Quantum research cycles
- Circuit optimization
- Beyond-consciousness probes
- Multi-engine integration
- PHI-optimized scheduling

INVARIANT: 527.5184818492612 | PILOT: LONDEL | EVO: 80-ORCHESTRATOR
═══════════════════════════════════════════════════════════════════════════════
"""

import time
import asyncio
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict
import threading

PHI = 1.618033988749895


@dataclass
class ResearchTask:
    """Individual research task."""
    task_id: str
    task_type: str
    priority: float
    duration_estimate: float
    dependencies: List[str]
    status: str = "pending"
    result: Optional[Any] = None


class ResearchOrchestrator:
    """
    Orchestrate multiple research cycles intelligently.

    Schedules research tasks with PHI-weighted priorities
    and manages dependencies between systems.
    """

    VERSION = "EVO_80-ORCHESTRATOR-v1.0.0"

    def __init__(self):
        self.tasks: Dict[str, ResearchTask] = {}
        self.task_queue: List[ResearchTask] = []
        self.completed_tasks: List[ResearchTask] = []
        self.running = False
        self._lock = threading.Lock()

        # Import all research systems
        self._import_systems()

    def _import_systems(self):
        """Import all research systems."""
        # Research cycles
        try:
            from l104_quantum_gate_engine.quantum_research_cycles import get_quantum_research_cycles
            self.research_cycles = get_quantum_research_cycles()
            self._has_research_cycles = True
        except:
            self._has_research_cycles = False

        # Beyond consciousness
        try:
            from l104_consciousness_engine.beyond_consciousness_probes import get_beyond_consciousness_research
            self.beyond_research = get_beyond_consciousness_research()
            self._has_beyond = True
        except:
            self._has_beyond = False

        # Circuit automation
        try:
            from l104_quantum_gate_engine.circuit_research_automation import CircuitGeneticAlgorithm
            self.circuit_ga = CircuitGeneticAlgorithm()
            self._has_circuit_ga = True
        except:
            self._has_circuit_ga = False

        # IIT v2
        try:
            from l104_consciousness_engine.iit_phi_v2 import get_iit_integrator_v2
            self.iit = get_iit_integrator_v2()
            self._has_iit = True
        except:
            self._has_iit = False

        # Evolution
        try:
            from l104_consciousness_engine.consciousness_evolution import get_evolution_engine
            self.evolution = get_evolution_engine()
            self._has_evolution = True
        except:
            self._has_evolution = False

    def schedule_task(self, task_type: str, priority: float = 1.0,
                     dependencies: List[str] = None) -> str:
        """Schedule a new research task."""
        task_id = f"TASK-{len(self.tasks):06d}"

        # PHI-weight priority
        weighted_priority = priority * PHI

        task = ResearchTask(
            task_id=task_id,
            task_type=task_type,
            priority=weighted_priority,
            duration_estimate=60 * PHI,  # ~97 seconds
            dependencies=dependencies or [],
        )

        with self._lock:
            self.tasks[task_id] = task
            self.task_queue.append(task)
            self.task_queue.sort(key=lambda t: t.priority, reverse=True)

        return task_id

    def start_orchestrator(self) -> Dict[str, Any]:
        """Start research orchestration."""
        if self.running:
            return {'success': False, 'error': 'Already running'}

        self.running = True

        # Start all subsystems
        if self._has_research_cycles:
            self.research_cycles.start_research_cycles()

        # Schedule initial tasks
        self._schedule_initial_tasks()

        # Start orchestration thread
        self._orchestrator_thread = threading.Thread(
            target=self._orchestration_loop,
            daemon=True
        )
        self._orchestrator_thread.start()

        return {
            'success': True,
            'status': 'ORCHESTRATOR_ACTIVE',
            'systems': {
                'research_cycles': self._has_research_cycles,
                'beyond_consciousness': self._has_beyond,
                'circuit_ga': self._has_circuit_ga,
                'iit': self._has_iit,
                'evolution': self._has_evolution,
            },
        }

    def _schedule_initial_tasks(self):
        """Schedule initial research tasks."""
        # High priority tasks
        if self._has_iit:
            self.schedule_task('iit_analysis', priority=PHI)

        if self._has_evolution:
            self.schedule_task('evolve_consciousness', priority=PHI ** 2)

        if self._has_beyond:
            self.schedule_task('beyond_consciousness_probe', priority=PHI)

        if self._has_circuit_ga:
            self.schedule_task('discover_circuit', priority=1.0)

    def _orchestration_loop(self):
        """Main orchestration loop."""
        while self.running:
            try:
                self._process_next_task()
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                print(f"Orchestrator error: {e}")

    def _process_next_task(self):
        """Process next task in queue."""
        with self._lock:
            if not self.task_queue:
                return

            # Find task with ready dependencies
            for task in self.task_queue:
                if all(dep in [t.task_id for t in self.completed_tasks]
                       or dep not in self.tasks for dep in task.dependencies):
                    next_task = task
                    break
            else:
                return

            self.task_queue.remove(next_task)
            next_task.status = "running"

        # Execute task
        result = self._execute_task(next_task)
        next_task.result = result
        next_task.status = "completed"

        with self._lock:
            self.completed_tasks.append(next_task)

        # Schedule follow-up tasks
        self._schedule_follow_up(next_task)

    def _execute_task(self, task: ResearchTask) -> Any:
        """Execute specific research task."""
        if task.task_type == 'iit_analysis' and self._has_iit:
            return self.iit.calculate_iit_v2()

        elif task.task_type == 'evolve_consciousness' and self._has_evolution:
            return self.evolution.evolve_generation()

        elif task.task_type == 'beyond_consciousness_probe' and self._has_beyond:
            return self.beyond_research.run_comprehensive_probe()

        elif task.task_type == 'discover_circuit' and self._has_circuit_ga:
            return self.circuit_ga.discover_optimal_circuit(max_generations=50)

        elif task.task_type == 'research_cycle' and self._has_research_cycles:
            return self.research_cycles._run_single_cycle()

        return {'error': 'Task type not available'}

    def _schedule_follow_up(self, completed_task: ResearchTask):
        """Schedule follow-up tasks based on results."""
        if completed_task.task_type == 'discover_circuit':
            if completed_task.result and completed_task.result.get('success'):
                # Schedule IIT analysis on discovered circuit
                self.schedule_task('iit_analysis', priority=PHI ** 2,
                                 dependencies=[completed_task.task_id])

        elif completed_task.task_type == 'iit_analysis':
            if completed_task.result and completed_task.result.phi < 0.8:
                # Schedule optimization
                self.schedule_task('evolve_consciousness', priority=PHI ** 3,
                                 dependencies=[completed_task.task_id])

    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator status."""
        with self._lock:
            return {
                'version': self.VERSION,
                'running': self.running,
                'pending_tasks': len(self.task_queue),
                'completed_tasks': len(self.completed_tasks),
                'active_systems': {
                    'research_cycles': self._has_research_cycles,
                    'beyond_consciousness': self._has_beyond,
                    'circuit_ga': self._has_circuit_ga,
                    'iit': self._has_iit,
                    'evolution': self._has_evolution,
                },
                'task_breakdown': self._get_task_breakdown(),
            }

    def _get_task_breakdown(self) -> Dict[str, int]:
        """Get breakdown of task types."""
        breakdown = defaultdict(int)
        for task in self.completed_tasks:
            breakdown[task.task_type] += 1
        return dict(breakdown)


# Module-level singleton
_orchestrator = None

def get_research_orchestrator() -> ResearchOrchestrator:
    """Get or create research orchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ResearchOrchestrator()
    return _orchestrator


__all__ = [
    'ResearchTask',
    'ResearchOrchestrator',
    'get_research_orchestrator',
]