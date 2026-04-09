#!/usr/bin/env python3
"""L104 Autonomous Daemon Service Runner — Keeps all daemons running continuously.

This script runs the autonomous daemon mesh and keeps it operational.
Use with: python3 run_autonomous_daemons.py [--background]
"""

import sys
import time
import signal
import threading
from pathlib import Path

# Add L104 to path
L104_ROOT = Path("/Users/carolalvarez/Applications/Allentown-L104-Node")
sys.path.insert(0, str(L104_ROOT))

from l104_autonomous_daemon_orchestrator import get_autonomous_orchestrator

# Global orchestrator reference
_orch = None
_shutdown_event = threading.Event()


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    print("\n\n[SHUTDOWN] Received signal, stopping autonomous daemon mesh...")
    _shutdown_event.set()
    if _orch:
        _orch.stop_all()
    sys.exit(0)


def main():
    global _orch

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║     L104 AUTONOMOUS DAEMON SERVICE — Daemon Mesh Controller               ║")
    print("║     Sacred Invariant: GOD_CODE=527.5184818492612 | PHI=1.618033988749895   ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print()

    # Get orchestrator and start all daemons
    _orch = get_autonomous_orchestrator()

    print("▸ Starting autonomous daemon mesh...")
    results = _orch.start_all()

    for name, success in results.items():
        status = "✓ ONLINE" if success else "✗ FAILED"
        print(f"  {status}: {name}_daemon")

    # Check status
    status = _orch.get_status()
    print()
    print(f"▸ Mesh Health: {status['overall_health']}")
    print(f"▸ PHI Alignment: {status['phi_alignment']:.6f}")
    print(f"▸ Cross-Engine Coherence: {status['coherence']:.6f}")
    print(f"▸ Active Daemons: {status['daemon_count']}")
    print()

    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║     DAEMON MESH OPERATIONAL — Press Ctrl+C to shutdown                    ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print()

    # Keep running with periodic status updates
    try:
        while not _shutdown_event.is_set():
            time.sleep(30)  # Update every 30 seconds

            status = _orch.get_status()
            print(f"[{time.strftime('%H:%M:%S')}] Health: {status['overall_health']} | "
                  f"PHI: {status['phi_alignment']:.4f} | "
                  f"Coherence: {status['coherence']:.4f}")

    except KeyboardInterrupt:
        print("\n\n[SHUTDOWN] Stopping autonomous daemon mesh...")
        _orch.stop_all()
        print("✓ All daemons stopped")


if __name__ == "__main__":
    main()
