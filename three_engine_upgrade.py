#!/usr/bin/env python3
"""
THREE-ENGINE UPGRADE (LEGACY WRAPPER)
=============================================================
This script is now a backward-compatible wrapper around the unified debug
framework in l104_debug.py (v2.0.0).

For the full-featured debug suite, use:
    python l104_debug.py                                             # All engines
    python l104_debug.py --engines code,science,math,asi,agi         # Upgrade targets
    python l104_debug.py --json --report three_engine_upgrade_report.json

Original scope: Uses Code + Science + Math engines to analyze and upgrade
ASI + AGI cores. Now covered by the unified self-test + cross-engine phases.

Run: .venv/bin/python three_engine_upgrade.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from l104_debug import run_debug

if __name__ == "__main__":
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║  LEGACY WRAPPER — Now using l104_debug.py v2.0.0                ║")
    print("║  For full options: python l104_debug.py --help                  ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")

    # Run the same scope as the original three_engine_upgrade:
    # Code + Science + Math for engine analysis
    # ASI + AGI for self-test and scoring verification
    diag = run_debug(
        engine_names=["code", "science", "math", "asi", "agi"],
        phases=None,  # all phases
        report_path=str(ROOT / "three_engine_upgrade_report.json"),
    )

    sys.exit(0 if diag.failed == 0 else 1)
