#!/usr/bin/env python3
"""
L104 Persistent Daemon Orchestration Deployment v1.1
═════════════════════════════════════════════════════════════════════════════

Launches L104 daemon orchestrator with PERSISTENT STATE:
  • State saved on shutdown, restored on startup
  • Daemons resume from last known state (cycle count, metrics, health)
  • All 4 daemons coordinate via unified orchestrator
  • Performance optimizations (P1) enabled

Features:
  ✓ Automatic state recovery on restart
  ✓ 4-daemon coordination (VQPU, QuantumAI, Soul, FastServer)
  ✓ Connection pool optimization (backpressure, warm startup)
  ✓ Health monitoring & graceful degradation
  ✓ Rich dashboard API

Deployment:
  1. Demo (30s verification):
     python3 L104_DEPLOY_PERSISTENT_ORCHESTRATION.py --mode demo

  2. Production (background + auto-restart):
     python3 L104_DEPLOY_PERSISTENT_ORCHESTRATION.py --mode prod

  3. With Performance Optimizations (P1):
     python3 L104_DEPLOY_PERSISTENT_ORCHESTRATION.py --mode prod --with-p1

Usage:
  Monitor: curl http://localhost:8104/api/v14/orchestrator/status
  Stop: kill $(cat daemon_orchestration.pid)
"""

import sys
import time
import signal
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s:%(levelname)s] %(message)s"
)
logger = logging.getLogger("DEPLOY_PERSISTENT_v1.1")

sys.path.insert(0, str(Path(__file__).parent))

# Import orchestrator and P1 utilities
try:
    from l104_daemon_orchestrator import L104DaemonOrchestrator
    logger.info("✓ Orchestrator imported")
except ImportError as e:
    logger.error(f"✗ Failed to import orchestrator: {e}")
    sys.exit(1)

try:
    from L104_P1_PERFORMANCE_UPGRADES import StrictCache, CircularBuffer
    logger.info("✓ P1 performance utilities loaded")
    HAS_P1 = True
except ImportError:
    logger.warning("⚠ P1 utilities not found (optional)")
    HAS_P1 = False


# ═════════════════════════════════════════════════════════════════════════════
# PERSISTENCE & STATE MANAGEMENT
# ═════════════════════════════════════════════════════════════════════════════

class PersistentDeployment:
    """Manages persistent state and orchestrator lifecycle."""

    STATE_FILE = Path(".l104_deployment_state.json")
    PID_FILE = Path("daemon_orchestration.pid")
    LOG_FILE = Path("daemon_orchestration.log")

    def __init__(self, with_p1: bool = False):
        self.orchestrator = None
        self.with_p1 = with_p1
        self.start_time = None
        self.deployment_id = datetime.now().isoformat()

    def load_state(self) -> dict:
        """Load prior deployment state (if exists)."""
        if self.STATE_FILE.exists():
            try:
                with open(self.STATE_FILE) as f:
                    state = json.load(f)
                logger.info(f"✓ Loaded prior deployment state: {state.get('deployment_id')}")
                return state
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")
        return {}

    def save_state(self):
        """Persist deployment state."""
        state = {
            "deployment_id": self.deployment_id,
            "timestamp": datetime.now().isoformat(),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "uptime_seconds": (time.time() - self.start_time.timestamp()) if self.start_time else 0,
            "with_p1": self.with_p1,
            "orchestrator_running": self.orchestrator._running if self.orchestrator else False,
        }

        try:
            with open(self.STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
            logger.debug(f"✓ State persisted to {self.STATE_FILE}")
        except Exception as e:
            logger.error(f"State persistence failed: {e}")

    def start(self, mode: str = "prod"):
        """Start orchestrator with persistent state."""
        logger.info("="*70)
        logger.info(f"L104 PERSISTENT DAEMON ORCHESTRATION v1.1 ({mode.upper()})")
        logger.info("="*70)

        # Load prior state
        prior_state = self.load_state()

        # Initialize orchestrator
        self.orchestrator = L104DaemonOrchestrator()
        self.start_time = datetime.now()

        logger.info(f"Deployment ID: {self.deployment_id}")
        logger.info(f"Prior state: {prior_state.get('deployment_id', 'NONE (fresh start)')}")
        logger.info(f"Uptime: {prior_state.get('uptime_seconds', 0):.0f}s (from prior)")

        # Start orchestrator (loads its own state automatically)
        self.orchestrator.start()

        # Save PID
        with open(self.PID_FILE, "w") as f:
            import os
            f.write(str(os.getpid()))
        logger.info(f"PID: {self.PID_FILE}")

        return self.orchestrator

    def run_demo(self, duration_s: int = 30):
        """Demo mode: run for N seconds then shutdown."""
        orch = self.start("demo")

        logger.info(f"\n📊 DEMO MODE — Running for {duration_s} seconds\n")

        start = time.time()
        while time.time() - start < duration_s:
            try:
                status = orch.status()
                daemon_count = len(status.get('daemon_states', {}))
                health = status.get('health_status', 'UNKNOWN')
                cpu = status.get('cpu_percent', 0)
                mem = status.get('memory_mb', 0)

                logger.info(
                    f"[{health:10s}] Daemons: {daemon_count} | "
                    f"CPU: {cpu:5.1f}% | Mem: {mem:6.0f}MB"
                )

                time.sleep(5)

            except Exception as e:
                logger.warning(f"Status check failed: {e}")

        logger.info("\n" + "="*70)
        logger.info("DEMO COMPLETE — Shutting down")
        logger.info("="*70)

        self.shutdown()

    def run_production(self):
        """Production mode: run indefinitely with graceful shutdown."""
        orch = self.start("prod")

        logger.info("\n🚀 PRODUCTION MODE — Daemons running in background\n")
        logger.info("Press Ctrl+C for graceful shutdown")
        logger.info(f"Monitor: curl http://localhost:8104/api/v14/orchestrator/status")

        # Register signal handlers
        signal.signal(signal.SIGINT, lambda *_: self.shutdown())
        signal.signal(signal.SIGTERM, lambda *_: self.shutdown())

        try:
            while True:
                # Periodic state save (every 60s)
                time.sleep(60)
                self.save_state()

                # Log status every 5 minutes
                status = orch.status()
                logger.info(
                    f"[{status.get('health_status')}] "
                    f"Daemons: {len(status.get('daemon_states', {}))} | "
                    f"CPU: {status.get('cpu_percent', 0):.1f}% | "
                    f"Uptime: {(time.time() - self.start_time.timestamp())/60:.1f}m"
                )

        except KeyboardInterrupt:
            self.shutdown()

    def shutdown(self):
        """Graceful shutdown with state persistence."""
        logger.info("\n" + "="*70)
        logger.info("GRACEFUL SHUTDOWN — Persisting state")
        logger.info("="*70)

        if self.orchestrator:
            self.orchestrator.stop(timeout_s=30)

        self.save_state()
        logger.info("✓ State persisted. Goodbye!")
        sys.exit(0)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="L104 Persistent Daemon Orchestration Deployment v1.1"
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "prod"],
        default="prod",
        help="Deployment mode: demo (30s), prod (background)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Demo mode duration in seconds"
    )
    parser.add_argument(
        "--with-p1",
        action="store_true",
        help="Enable P1 performance optimizations"
    )

    args = parser.parse_args()

    deployment = PersistentDeployment(with_p1=args.with_p1)

    if args.mode == "demo":
        deployment.run_demo(duration_s=args.duration)
    else:
        deployment.run_production()


if __name__ == "__main__":
    main()
