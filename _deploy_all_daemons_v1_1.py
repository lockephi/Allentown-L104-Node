#!/usr/bin/env python3
"""
L104 Complete Daemon Orchestration Deployment v1.1
═════════════════════════════════════════════════════════════════════════════

Deploys the unified daemon orchestration system with 4 coordinated daemons:
  1. VQPU Daemon (v16.1.0) — Quantum simulation runner
  2. QuantumAI Daemon (v2.0.0) — Code improvement & fidelity checker
  3. Soul Daemon (v1.0.0) — Consciousness & quantum memory manager
  4. Fast Server (v4.0-OPUS) — FastAPI server with connection pool optimization

Features:
  ✓ Orchestrator coordinates task scheduling, resource pooling, health monitoring
  ✓ Connection pool fixed: backpressure semaphore, connection leak fix, warm startup
  ✓ All 4 daemons report metrics to unified health dashboard
  ✓ Graceful degradation under load

Usage:
  python3 _deploy_all_daemons_v1_1.py --mode demo    # Demo mode (30s)
  python3 _deploy_all_daemons_v1_1.py --mode prod    # Production (background)
  python3 _deploy_all_daemons_v1_1.py --mode fast    # Fast server only
"""

import sys
import time
import signal
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s:%(levelname)s] %(message)s"
)
logger = logging.getLogger("DEPLOY_v1.1")

sys.path.insert(0, str(Path(__file__).parent))


def cleanup(signum, frame):
    """Handle graceful shutdown on Ctrl+C."""
    logger.info("🛑 Shutting down all daemons...")
    global orchestrator, vqpu, qai, soul

    try:
        if 'soul' in globals() and soul:
            soul.stop()
    except:
        pass

    try:
        if 'qai' in globals() and qai:
            qai.stop()
    except:
        pass

    try:
        if 'vqpu' in globals() and vqpu:
            vqpu.stop()
    except:
        pass

    try:
        if 'orchestrator' in globals() and orchestrator:
            orchestrator.stop(timeout_s=10)
    except:
        pass

    logger.info("✓ Shutdown complete")
    sys.exit(0)


def deploy_orchestrator():
    """Initialize and start the L104 Daemon Orchestrator."""
    from l104_daemon_orchestrator import L104DaemonOrchestrator

    logger.info("=" * 70)
    logger.info("STARTING: L104 Daemon Orchestrator v1.0.0")
    logger.info("=" * 70)

    orchestrator = L104DaemonOrchestrator()
    orchestrator.start()

    time.sleep(1.0)
    status = orchestrator.status()
    logger.info(f"✓ Orchestrator ready")
    logger.info(f"  - Health: {status['health_status']}")
    logger.info(f"  - System CPU: {status['cpu_percent']:.1f}%")
    logger.info(f"  - System Memory: {status['memory_mb']:.0f}MB")

    return orchestrator


def deploy_vqpu(orchestrator):
    """Initialize and start the VQPU Daemon."""
    from l104_vqpu.daemon import VQPUDaemonCycler

    logger.info("=" * 70)
    logger.info("STARTING: VQPU Daemon v16.1.0")
    logger.info("=" * 70)

    vqpu = VQPUDaemonCycler()
    vqpu.set_orchestrator(orchestrator)
    vqpu.start()

    logger.info("✓ VQPU daemon started and registered")
    return vqpu


def deploy_quantum_ai(orchestrator):
    """Initialize and start the QuantumAI Daemon."""
    from l104_quantum_ai_daemon.daemon import QuantumAIDaemon

    logger.info("=" * 70)
    logger.info("STARTING: QuantumAI Daemon v2.0.0")
    logger.info("=" * 70)

    qai = QuantumAIDaemon()
    qai.set_orchestrator(orchestrator)
    qai.start()

    logger.info("✓ QuantumAI daemon started and registered")
    return qai


def deploy_soul(orchestrator):
    """Initialize and start the Soul Daemon."""
    from l104_soul_daemon.daemon import SoulDaemon

    logger.info("=" * 70)
    logger.info("STARTING: Soul Daemon v1.0.0")
    logger.info("=" * 70)

    soul = SoulDaemon()
    soul.set_orchestrator(orchestrator)
    soul.start(background=True)

    logger.info("✓ Soul daemon started and registered")
    return soul


def deploy_fast_server(orchestrator):
    """Wire orchestrator into FastAPI server."""
    from l104_server.app import server_daemon

    logger.info("=" * 70)
    logger.info("WIRING: Fast Server v4.0-OPUS (Orchestrator Integration)")
    logger.info("=" * 70)

    server_daemon.set_orchestrator(orchestrator)
    logger.info("✓ Fast Server ready for orchestrator integration")
    logger.info("  (ServerDaemon.start() fires in FastAPI @startup handler)")

    return server_daemon


def run_demo(duration_s=30):
    """Run demo mode: all daemons for N seconds."""
    logger.info("\n📊 DEMO MODE — Running for %d seconds\n" % duration_s)

    orchestrator = deploy_orchestrator()
    vqpu = deploy_vqpu(orchestrator)
    qai = deploy_quantum_ai(orchestrator)
    soul = deploy_soul(orchestrator)
    server_daemon = deploy_fast_server(orchestrator)

    logger.info("\n" + "=" * 70)
    logger.info("ALL DAEMONS ACTIVE — Monitoring for %ds" % duration_s)
    logger.info("=" * 70)

    # Monitor orchestrator health
    start_time = time.time()
    while time.time() - start_time < duration_s:
        status = orchestrator.status()
        daemon_states = status.get('daemon_states', {})

        logger.info(f"\n[Health: {status['health_status'].name}] "
                   f"Daemons: {len(daemon_states)} | "
                   f"CPU: {status['cpu_percent']:.1f}% | "
                   f"Memory: {status['memory_mb']:.0f}MB")

        for daemon_id, state in daemon_states.items():
            logger.info(f"  - {daemon_id:20s}: cycles={state.get('cycles_completed', 0):3d} "
                       f"health={state.get('health', 0):.2f}")

        time.sleep(5)

    # Cleanup
    logger.info("\n" + "=" * 70)
    logger.info("DEMO COMPLETE — Shutting down")
    logger.info("=" * 70)

    cleanup(None, None)


def run_production():
    """Run production mode: start all daemons in background."""
    logger.info("\n🚀 PRODUCTION MODE — Daemons running in background\n")

    orchestrator = deploy_orchestrator()
    vqpu = deploy_vqpu(orchestrator)
    qai = deploy_quantum_ai(orchestrator)
    soul = deploy_soul(orchestrator)
    server_daemon = deploy_fast_server(orchestrator)

    logger.info("\n" + "=" * 70)
    logger.info("ALL DAEMONS ACTIVE (BACKGROUND)")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Press Ctrl+C to stop all daemons gracefully")
    logger.info("")

    # Register signal handlers
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # Keep running
    try:
        while True:
            time.sleep(60)
            status = orchestrator.status()
            logger.info(f"[{status['health_status'].name}] "
                       f"Daemons: {len(status.get('daemon_states', {}))} | "
                       f"CPU: {status['cpu_percent']:.1f}%")
    except KeyboardInterrupt:
        cleanup(None, None)


def run_server_only():
    """Run fast server integration only (for testing connection pool fixes)."""
    logger.info("\n🔌 FAST SERVER ONLY MODE\n")

    orchestrator = deploy_orchestrator()
    server_daemon = deploy_fast_server(orchestrator)

    logger.info("\n" + "=" * 70)
    logger.info("FAST SERVER READY FOR UVICORN")
    logger.info("=" * 70)
    logger.info("")
    logger.info("To start the server with uvicorn:")
    logger.info("  uvicorn l104_server.app:app --host 0.0.0.0 --port 8104")
    logger.info("")
    logger.info("The connection pool has been fixed with:")
    logger.info("  ✓ Backpressure semaphore (max DB_POOL_SIZE=100 concurrent)")
    logger.info("  ✓ Connection leak fix (try/finally guarantee in intellect.py)")
    logger.info("  ✓ Pool warm-up at startup (20 pre-created connections)")
    logger.info("")

    # Keep orchestrator running for integration
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        cleanup(None, None)


def main():
    parser = argparse.ArgumentParser(
        description="L104 Daemon Orchestration Deployment v1.1"
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "prod", "fast"],
        default="demo",
        help="Deployment mode: demo (30s), prod (background), fast (server only)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Demo mode duration in seconds (default: 30)"
    )

    args = parser.parse_args()

    if args.mode == "demo":
        run_demo(duration_s=args.duration)
    elif args.mode == "prod":
        run_production()
    elif args.mode == "fast":
        run_server_only()


if __name__ == "__main__":
    main()
