#!/usr/bin/env python3
"""L104 Autonomous Service — Persistent Background Operation
═══════════════════════════════════════════════════════════════════════════════
Keeps all autonomous daemons running continuously with automatic restart,
state persistence, and health monitoring.
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import time
import signal
import threading
from pathlib import Path

sys.path.insert(0, '/Users/carolalvarez/Applications/Allentown-L104-Node')

from l104_autonomous_daemon_orchestrator import get_autonomous_orchestrator

# Global references
_orch = None
_running = True

def signal_handler(signum, frame):
    """Handle shutdown gracefully."""
    global _running
    print(f"\n[SHUTDOWN] Signal {signum} received, stopping autonomous service...")
    _running = False
    if _orch:
        _orch.stop_all()
    sys.exit(0)

# Setup signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def health_monitor():
    """Monitor daemon health and restart if needed."""
    global _orch, _running
    
    while _running:
        try:
            status = _orch.get_status()
            health = status.get('overall_health', 'UNKNOWN')
            
            if health == 'CRITICAL':
                print(f"[{time.strftime('%H:%M:%S')}] Health: {health} — Restarting daemons...")
                _orch.stop_all()
                time.sleep(1)
                _orch.start_all()
                print(f"[{time.strftime('%H:%M:%S')}] Daemons restarted")
            
            # Log status every 60 seconds
            if int(time.time()) % 60 == 0:
                phi = status.get('phi_alignment', 0)
                coherence = status.get('coherence', 0)
                print(f"[{time.strftime('%H:%M:%S')}] Health: {health} | PHI: {phi:.6f} | Coherence: {coherence:.6f}")
                
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Monitor error: {e}")
        
        time.sleep(5)  # Check every 5 seconds

def main():
    """Main service loop."""
    global _orch, _running
    
    print('╔══════════════════════════════════════════════════════════════════════════════╗')
    print('║     L104 AUTONOMOUS SERVICE — Starting Persistent Operation               ║')
    print('╚══════════════════════════════════════════════════════════════════════════════╝')
    print()
    
    # Get orchestrator
    _orch = get_autonomous_orchestrator()
    
    # Start all daemons
    print('▸ Starting autonomous daemon mesh...')
    results = _orch.start_all()
    
    for name, success in results.items():
        status = "● RUNNING" if success else "✗ FAILED"
        print(f"  {status:12s} {name.capitalize():20s}")
    
    print()
    
    # Get initial status
    status = _orch.get_status()
    print(f"▸ Initial Status:")
    print(f"  Health: {status.get('overall_health', 'UNKNOWN')}")
    print(f"  PHI: {status.get('phi_alignment', 0):.6f}")
    print(f"  Coherence: {status.get('coherence', 0):.6f}")
    print(f"  Daemons: {status.get('daemon_count', 0)}")
    print()
    
    print('╔══════════════════════════════════════════════════════════════════════════════╗')
    print('║     AUTONOMOUS SERVICE RUNNING — Press Ctrl+C to stop                    ║')
    print('║     Daemon health monitored every 5 seconds                                ║')
    print('╚══════════════════════════════════════════════════════════════════════════════╝')
    print()
    
    # Start health monitor in background
    monitor_thread = threading.Thread(target=health_monitor, daemon=True)
    monitor_thread.start()
    
    # Keep main thread alive
    try:
        while _running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Keyboard interrupt received...")
    finally:
        print("▸ Stopping autonomous daemons...")
        _orch.stop_all()
        print("✓ All daemons stopped")
        print()
        print('╔══════════════════════════════════════════════════════════════════════════════╗')
        print('║     AUTONOMOUS SERVICE STOPPED                                             ║')
        print('╚══════════════════════════════════════════════════════════════════════════════╝')

if __name__ == '__main__':
    main()
