#!/usr/bin/env python3
"""
Demo: L104 Unified Daemon Orchestrator in Action
═════════════════════════════════════════════════════════════════════════

Shows:
  1. Starting the orchestrator
  2. Registering daemons
  3. Simulated daemon cycles with metrics
  4. Task scheduling & prioritization
  5. Event emission & subscription
  6. Health monitoring & degradation
"""

import sys
import time
import threading
import random
from pathlib import Path

# Ensure imports work from current directory
sys.path.insert(0, str(Path(__file__).parent))

from l104_daemon_orchestrator import (
    L104DaemonOrchestrator, Task, TaskPriority, DaemonType, HealthStatus
)
from l104_daemon_adapter import DaemonAdapter

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)

logger = logging.getLogger("DEMO")


class MockDaemon:
    """Mock daemon for demonstration."""

    def __init__(self, daemon_id: str, orchestrator, cycle_interval_s: float = 2.0):
        self.daemon_id = daemon_id
        self.orchestrator = orchestrator
        self.cycle_interval_s = cycle_interval_s
        self.adapter = DaemonAdapter(daemon_id, orchestrator)
        self.running = False

        # Subscribe to events from orchestrator
        self.adapter.subscribe_to_event("task_scheduled", self._on_task_scheduled)

    def _on_task_scheduled(self, event):
        logger.info(f"[{self.daemon_id}] Task scheduled: {event['payload']}")

    def run(self):
        """Main daemon loop."""
        self.running = True
        logger.info(f"[{self.daemon_id}] Starting daemon loop")

        cycle_count = 0
        while self.running:
            cycle_count += 1

            self.adapter.on_cycle_start()

            try:
                # Simulate cycle work
                duration_ms = random.uniform(100, 500)
                time.sleep(duration_ms / 1000.0)

                # Simulate random failures (10% chance)
                success = random.random() > 0.1

                if success:
                    # Simulate metrics
                    cpu_percent = random.uniform(10, 40)
                    memory_mb = random.uniform(50, 150)
                    fidelity = random.uniform(0.8, 1.0)

                    self.adapter.on_cycle_end(
                        success=True,
                        duration_ms=duration_ms,
                        cpu_percent=cpu_percent,
                        memory_mb=memory_mb
                    )

                    # Emit fidelity
                    trending = ["up", "stable", "down"][random.randint(0, 2)]
                    self.adapter.emit_fidelity_alert(
                        fidelity=fidelity,
                        trending=trending,
                        sim_count=random.randint(10, 50),
                        error_count=random.randint(0, 3)
                    )

                    logger.info(
                        f"[{self.daemon_id}] Cycle {cycle_count}: SUCCESS | "
                        f"{duration_ms:.0f}ms | CPU {cpu_percent:.1f}% | "
                        f"Fidelity {fidelity:.3f}"
                    )
                else:
                    error = f"Simulated error in cycle {cycle_count}"
                    self.adapter.on_cycle_end(success=False, duration_ms=duration_ms)
                    self.adapter.emit_error("cycle_failure", error, "error")
                    logger.warning(f"[{self.daemon_id}] Cycle {cycle_count}: FAILED")

            except Exception as e:
                logger.error(f"[{self.daemon_id}] Unexpected error: {e}", exc_info=True)
                self.adapter.on_cycle_end(success=False)

            time.sleep(self.cycle_interval_s)

    def stop(self):
        """Stop the daemon."""
        self.running = False
        logger.info(f"[{self.daemon_id}] Stopping daemon")


def print_status(orchestrator, title: str = "STATUS"):
    """Print formatted status."""
    status = orchestrator.status()

    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

    print(f"Health Status: {status['health_status']}")
    print(f"CPU: {status['cpu_percent']:.1f}% | Memory: {status['memory_percent']:.1f}%")
    print(f"Active Tasks: {status['active_tasks']} | Queued: {status['queued_tasks']}")
    print(f"Orchestration Cycles: {status['cycle_count']}")

    if status['daemon_metrics']:
        print("\nDaemon Metrics:")
        for daemon_id, metrics in status['daemon_metrics'].items():
            print(f"  {daemon_id}:")
            print(f"    Cycles: {metrics['cycles_completed']} ✓ / {metrics['cycles_failed']} ✗")
            print(f"    Avg Cycle: {metrics['avg_cycle_ms']:.1f}ms | Health: {metrics['health_score']:.3f}")
            print(f"    CPU: {metrics['cpu_percent']:.1f}% | Memory: {metrics['memory_mb']:.1f}MB")

    print("=" * 80 + "\n")


def main():
    """Run the demo."""
    print("\n" + "=" * 80)
    print("  L104 UNIFIED DAEMON ORCHESTRATOR - DEMO")
    print("=" * 80 + "\n")

    # Start orchestrator
    logger.info("Starting orchestrator...")
    orchestrator = L104DaemonOrchestrator()
    orchestrator.start()
    time.sleep(1)

    # Create mock daemons
    logger.info("Creating mock daemons...")
    daemons = {
        "vqpu_daemon": MockDaemon("vqpu_daemon", orchestrator, cycle_interval_s=1.5),
        "quantum_ai_daemon": MockDaemon("quantum_ai_daemon", orchestrator, cycle_interval_s=2.0),
        "soul_daemon": MockDaemon("soul_daemon", orchestrator, cycle_interval_s=2.5),
    }

    # Start daemon threads
    daemon_threads = {}
    for daemon_id, daemon in daemons.items():
        thread = threading.Thread(target=daemon.run, daemon=True, name=daemon_id)
        thread.start()
        daemon_threads[daemon_id] = thread
    time.sleep(1)

    # Submit some tasks
    logger.info("Submitting sample tasks...")
    tasks = [
        Task(
            daemon_id="vqpu_daemon",
            task_type="run_simulations",
            priority=TaskPriority.HIGH,
            deadline=time.time() + 60
        ),
        Task(
            daemon_id="quantum_ai_daemon",
            task_type="improve_code",
            priority=TaskPriority.NORMAL,
            deadline=time.time() + 120
        ),
        Task(
            daemon_id="soul_daemon",
            task_type="consciousness_check",
            priority=TaskPriority.NORMAL
        ),
    ]
    orchestrator.submit_batch(tasks)

    # Monitor for 30 seconds
    logger.info("Running demo for 30 seconds (watch the cycles)...\n")

    start_time = time.time()
    last_status_time = start_time

    try:
        while time.time() - start_time < 30:
            # Print status every 10 seconds
            if time.time() - last_status_time >= 10:
                print_status(orchestrator, f"STATUS @ {time.time() - start_time:.0f}s")
                last_status_time = time.time()

            time.sleep(0.5)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    # Final status
    print_status(orchestrator, "FINAL STATUS")

    # Shutdown
    logger.info("Shutting down daemons...")
    for daemon in daemons.values():
        daemon.stop()

    for thread in daemon_threads.values():
        thread.join(timeout=5)

    logger.info("Shutting down orchestrator...")
    orchestrator.stop()

    print("=" * 80)
    print("  DEMO COMPLETE")
    print("=" * 80)
    print("\nKey Observations:")
    print("  1. Three daemons run independently")
    print("  2. Orchestrator coordinates their scheduling")
    print("  3. Metrics are collected and aggregated")
    print("  4. Health status drives scheduling decisions")
    print("  5. State is persisted to .l104_daemon_orchestrator.json")
    print("\nNext Steps:")
    print("  1. Integrate adapters into real daemons")
    print("  2. Monitor status dashboard")
    print("  3. Subscribe to events from your application")
    print("  4. Tune resource quotas based on your system")


if __name__ == "__main__":
    main()
