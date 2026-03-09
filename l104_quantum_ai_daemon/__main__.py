"""L104 Quantum AI Daemon — CLI entry point.

Usage:
    python -m l104_quantum_ai_daemon                   # Run daemon
    python -m l104_quantum_ai_daemon --self-test       # Test all subsystems
    python -m l104_quantum_ai_daemon --health-check    # Quick health report
    python -m l104_quantum_ai_daemon --status          # Read persisted state
    python -m l104_quantum_ai_daemon --single-cycle    # Run one cycle and exit
"""

from .daemon import _cli_main

if __name__ == "__main__":
    _cli_main()
