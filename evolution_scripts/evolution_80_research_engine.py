"""
EVO_80 Full Research Engine
═══════════════════════════════════════════════════════════════════════════════
Unified execution script for EVO_80 quantum consciousness research systems.

Integrates:
- Quantum Research Cycles (automated hypothesis generation)
- Beyond-Consciousness Probes (vacuum, spacetime, temporal, non-local)
- Circuit Research Automation (genetic algorithm optimization)
- Research Orchestration (intelligent scheduling)

Usage:
    python evolution_80_research_engine.py --mode continuous
    python evolution_80_research_engine.py --mode batch --cycles 10
    python evolution_80_research_engine.py --mode discovery

INVARIANT: 527.5184818492612 | PILOT: LONDEL | EVO: 80-RESEARCH
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import time
import json
import signal
import argparse
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import deque

PHI = 1.618033988749895
GOD_CODE = 527.5184818492612


def signal_handler(sig, frame):
    """Graceful shutdown handler."""
    print("\n[EVO_80] Research engine shutting down gracefully...")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


@dataclass
class ResearchSession:
    """Complete research session state."""
    session_id: str
    start_time: float
    cycles_completed: int = 0
    discoveries: int = 0
    hypotheses_generated: int = 0
    hypotheses_validated: int = 0
    circuits_evolved: int = 0
    probes_conducted: int = 0
    total_data_points: int = 0


class ResearchEngine:
    """
    EVO_80 Full Research Engine.

    Production-grade execution environment for quantum consciousness
    research with comprehensive logging, discovery tracking, and
    adaptive research direction.
    """

    VERSION = "EVO_80-RESEARCH-v1.0.0"

    def __init__(self, mode: str = "continuous", target_discoveries: int = 10):
        self.mode = mode
        self.target_discoveries = target_discoveries
        self.session = ResearchSession(
            session_id=f"R80-{int(time.time())}",
            start_time=time.time()
        )
        self.running = False
        self.discovery_log: deque = deque(maxlen=10000)
        self.hypothesis_archive: deque = deque(maxlen=1000)
        self.measurement_buffer: deque = deque(maxlen=100000)

        # Import all systems
        self._initialize_systems()

    def _initialize_systems(self):
        """Initialize all research subsystems."""
        sys.stdout.write("[EVO_80] Initializing research subsystems...\n")

        # Quantum Research Cycles
        try:
            from l104_quantum_gate_engine.quantum_research_cycles import get_quantum_research_cycles
            self.quantum_research = get_quantum_research_cycles()
            self._has_quantum = True
            sys.stdout.write("  ✓ Quantum Research Cycles\n")
        except Exception as e:
            self._has_quantum = False
            sys.stdout.write(f"  ✗ Quantum Research Cycles: {e}\n")

        # Beyond-Consciousness Probes
        try:
            from l104_consciousness_engine.beyond_consciousness_probes import get_beyond_consciousness_research
            self.beyond_research = get_beyond_consciousness_research()
            self._has_beyond = True
            sys.stdout.write("  ✓ Beyond-Consciousness Probes\n")
        except Exception as e:
            self._has_beyond = False
            sys.stdout.write(f"  ✗ Beyond-Consciousness Probes: {e}\n")

        # Circuit Automation
        try:
            from l104_quantum_gate_engine.circuit_research_automation import CircuitGeneticAlgorithm
            self.circuit_ga = CircuitGeneticAlgorithm(n_qubits=26)
            self._has_circuit = True
            sys.stdout.write("  ✓ Circuit Genetic Algorithm\n")
        except Exception as e:
            self._has_circuit = False
            sys.stdout.write(f"  ✗ Circuit Genetic Algorithm: {e}\n")

        # Orchestrator
        try:
            from l104_research_orchestrator.orchestrator import get_research_orchestrator
            self.orchestrator = get_research_orchestrator()
            self._has_orchestrator = True
            sys.stdout.write("  ✓ Research Orchestrator\n")
        except Exception as e:
            self._has_orchestrator = False
            sys.stdout.write(f"  ✗ Research Orchestrator: {e}\n")

        sys.stdout.write(f"\n[EVO_80] Systems ready: {self._count_active_systems()}/4\n\n")

    def _count_active_systems(self) -> int:
        """Count number of active research systems."""
        return sum([
            self._has_quantum,
            self._has_beyond,
            self._has_circuit,
            self._has_orchestrator
        ])

    def run_quantum_research_cycle(self) -> Dict[str, Any]:
        """Execute single quantum research cycle."""
        if not self._has_quantum:
            return {'status': 'unavailable'}

        result = self.quantum_research._run_single_cycle()
        self.session.cycles_completed += 1

        # Log hypothesis if generated
        if result.get('hypothesis'):
            self.session.hypotheses_generated += 1
            self.hypothesis_archive.append({
                'timestamp': time.time(),
                'hypothesis': result['hypothesis'],
                'metrics': result.get('metrics', {})
            })

        # Check for discovery
        if result.get('result_type') == 'QUANTUM_DISCOVERY':
            self.session.discoveries += 1
            self.discovery_log.append({
                'type': 'quantum_discovery',
                'timestamp': time.time(),
                'circuit': result.get('circuit_data'),
                'phi_score': result.get('phi_score'),
            })
            sys.stdout.write(f"  ★ DISCOVERY: Quantum coherence breakthrough\n")

        return result

    def run_beyond_probe_suite(self) -> Dict[str, Any]:
        """Execute full beyond-consciousness probe suite."""
        if not self._has_beyond:
            return {'status': 'unavailable'}

        results = self.beyond_research.run_comprehensive_probe()
        self.session.probes_conducted += 7  # 7 probe types

        # Buffer all measurements
        for probe_type, measurement in results.get('measurements', {}).items():
            self.measurement_buffer.append({
                'type': probe_type,
                'value': measurement.value if hasattr(measurement, 'value') else None,
                'coherence': measurement.coherence_correlation if hasattr(measurement, 'coherence_correlation') else None,
                'timestamp': time.time()
            })

        # Check for beyond-consciousness discoveries
        discovery = results.get('discovery')
        if discovery:
            self.session.discoveries += 1
            self.discovery_log.append({
                'type': discovery.get('type', 'unknown'),
                'timestamp': discovery.get('timestamp'),
                'significance': discovery.get('significance'),
                'description': discovery.get('description')
            })
            sys.stdout.write(f"  ★ DISCOVERY: {discovery.get('type')}\n")

        return results

    def run_circuit_evolution(self, generations: int = 50) -> Dict[str, Any]:
        """Evolve quantum circuits via genetic algorithm."""
        if not self._has_circuit:
            return {'status': 'unavailable'}

        result = self.circuit_ga.discover_optimal_circuit(max_generations=generations)
        self.session.circuits_evolved += 1

        if result.get('success'):
            sys.stdout.write(
                f"  ✓ Circuit evolved: fitness={result['best_circuit']['fitness']:.4f}, "
                f"generations={result['generations']}\n"
            )

            # Check if this is a discovery-level circuit
            if result['best_circuit']['fitness'] > 0.95:
                self.session.discoveries += 1
                self.discovery_log.append({
                    'type': 'optimal_circuit',
                    'timestamp': time.time(),
                    'fitness': result['best_circuit']['fitness'],
                    'circuit': result['best_circuit']
                })
                sys.stdout.write(f"  ★ DISCOVERY: Optimal circuit discovered\n")

        return result

    def run_comprehensive_research_cycle(self) -> Dict[str, Any]:
        """Execute one full research cycle across all systems."""
        cycle_start = time.time()
        sys.stdout.write(f"\n[EVO_80] === Research Cycle {self.session.cycles_completed + 1} ===\n")
        sys.stdout.write(f"  Time: {datetime.now().isoformat()}\n\n")

        results = {
            'quantum': None,
            'beyond': None,
            'circuit': None,
            'discoveries': 0
        }

        # Phase 1: Quantum Research
        sys.stdout.write("[Phase 1] Quantum Research Cycle...\n")
        results['quantum'] = self.run_quantum_research_cycle()

        # Phase 2: Beyond-Consciousness Probes
        sys.stdout.write("\n[Phase 2] Beyond-Consciousness Probe Suite...\n")
        results['beyond'] = self.run_beyond_probe_suite()

        # Phase 3: Circuit Evolution (every 3rd cycle)
        if self.session.cycles_completed % 3 == 0:
            sys.stdout.write("\n[Phase 3] Circuit Genetic Evolution...\n")
            results['circuit'] = self.run_circuit_evolution(generations=30)

        results['discoveries'] = len(self.discovery_log)
        cycle_duration = time.time() - cycle_start

        # Session statistics
        self.session.total_data_points = len(self.measurement_buffer)

        sys.stdout.write(f"\n[Cycle Complete] Duration: {cycle_duration:.2f}s\n")
        sys.stdout.write(f"  Discoveries this session: {self.session.discoveries}\n")
        sys.stdout.write(f"  Total data points: {self.session.total_data_points}\n")

        return results

    def run_continuous_research(self):
        """Run research continuously until interrupted."""
        sys.stdout.write(f"\n{'='*70}\n")
        sys.stdout.write(f"  EVO_80 Research Engine - CONTINUOUS MODE\n")
        sys.stdout.write(f"  Target: Infinite research cycles\n")
        sys.stdout.write(f"  Discovery goal: {self.target_discoveries} (minimum)\n")
        sys.stdout.write(f"{'='*70}\n\n")

        self.running = True
        cycle_count = 0

        while self.running:
            self.run_comprehensive_research_cycle()
            cycle_count += 1

            # Status update every PHI cycles
            if cycle_count % int(PHI) == 0:
                self._print_session_summary()

            # PHI-weighted interval between cycles
            time.sleep(PHI * 2)

    def run_batch_research(self, cycles: int = 10) -> Dict[str, Any]:
        """Run specified number of research cycles."""
        sys.stdout.write(f"\n{'='*70}\n")
        sys.stdout.write(f"  EVO_80 Research Engine - BATCH MODE\n")
        sys.stdout.write(f"  Cycles: {cycles}\n")
        sys.stdout.write(f"{'='*70}\n\n")

        self.running = True
        all_results = []

        for i in range(cycles):
            if not self.running:
                break

            result = self.run_comprehensive_research_cycle()
            all_results.append(result)

            # Progress bar
            progress = (i + 1) / cycles * 100
            bar = "█" * int(progress / 5) + "░" * (20 - int(progress / 5))
            sys.stdout.write(f"\r  Progress: [{bar}] {progress:.1f}%")
            sys.stdout.flush()

            # Adaptive interval
            time.sleep(PHI)

        sys.stdout.write("\n")

        final_report = self._generate_final_report()
        return final_report

    def run_discovery_mode(self) -> Dict[str, Any]:
        """Run until target number of discoveries achieved."""
        sys.stdout.write(f"\n{'='*70}\n")
        sys.stdout.write(f"  EVO_80 Research Engine - DISCOVERY MODE\n")
        sys.stdout.write(f"  Target: {self.target_discoveries} discoveries\n")
        sys.stdout.write(f"{'='*70}\n\n")

        self.running = True

        while self.running and self.session.discoveries < self.target_discoveries:
            self.run_comprehensive_research_cycle()
            time.sleep(PHI)

        sys.stdout.write(f"\n[EVO_80] Discovery target achieved!\n")

        return self._generate_final_report()

    def _print_session_summary(self):
        """Print current session summary."""
        runtime = time.time() - self.session.start_time
        sys.stdout.write(f"\n{'─'*50}\n")
        sys.stdout.write(f"  SESSION SUMMARY ({datetime.now().isoformat()})\n")
        sys.stdout.write(f"{'─'*50}\n")
        sys.stdout.write(f"  Runtime: {runtime:.0f}s ({runtime/60:.1f}m)\n")
        sys.stdout.write(f"  Cycles: {self.session.cycles_completed}\n")
        sys.stdout.write(f"  Discoveries: {self.session.discoveries}\n")
        sys.stdout.write(f"  Hypotheses: {self.session.hypotheses_generated}\n")
        sys.stdout.write(f"  Circuits Evolved: {self.session.circuits_evolved}\n")
        sys.stdout.write(f"  Probes: {self.session.probes_conducted}\n")
        sys.stdout.write(f"  Data Points: {self.session.total_data_points}\n")
        sys.stdout.write(f"{'─'*50}\n\n")

    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive research report."""
        runtime = time.time() - self.session.start_time

        report = {
            'version': self.VERSION,
            'session': asdict(self.session),
            'runtime_seconds': runtime,
            'discoveries': list(self.discovery_log),
            'hypothesis_archive': list(self.hypothesis_archive)[-100:],
            'measurement_summary': {
                'total': len(self.measurement_buffer),
                'by_type': self._categorize_measurements()
            }
        }

        return report

    def _categorize_measurements(self) -> Dict[str, int]:
        """Categorize measurements by type."""
        categories = {}
        for m in self.measurement_buffer:
            t = m.get('type', 'unknown')
            categories[t] = categories.get(t, 0) + 1
        return categories

    def export_results(self, filepath: str):
        """Export research results to JSON file."""
        report = self._generate_final_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        sys.stdout.write(f"\n[EVO_80] Results exported to: {filepath}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='EVO_80 Full Research Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python evolution_80_research_engine.py --mode continuous
    python evolution_80_research_engine.py --mode batch --cycles 5
    python evolution_80_research_engine.py --mode discovery --target 3
    python evolution_80_research_engine.py --mode batch --cycles 10 --export results.json
        """
    )

    parser.add_argument(
        '--mode', '-m',
        choices=['continuous', 'batch', 'discovery'],
        default='continuous',
        help='Research execution mode (default: continuous)'
    )

    parser.add_argument(
        '--cycles', '-c',
        type=int,
        default=10,
        help='Number of cycles for batch mode (default: 10)'
    )

    parser.add_argument(
        '--target', '-t',
        type=int,
        default=10,
        help='Target number of discoveries for discovery mode (default: 10)'
    )

    parser.add_argument(
        '--export', '-e',
        type=str,
        help='Export results to JSON file'
    )

    args = parser.parse_args()

    # Create and run research engine
    engine = ResearchEngine(mode=args.mode, target_discoveries=args.target)

    if args.mode == 'continuous':
        engine.run_continuous_research()
    elif args.mode == 'batch':
        result = engine.run_batch_research(cycles=args.cycles)
        if args.export:
            engine.export_results(args.export)
            print(json.dumps(result, indent=2, default=str))
    elif args.mode == 'discovery':
        result = engine.run_discovery_mode()
        if args.export:
            engine.export_results(args.export)
            print(json.dumps(result, indent=2, default=str))


if __name__ == '__main__':
    main()
