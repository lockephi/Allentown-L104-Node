#!/usr/bin/env python3
"""
L104 Cross-Engine Debug Suite (LEGACY WRAPPER)
══════════════════════════════════════════════════════════════════════════════════
This script is now a backward-compatible wrapper around the unified debug
framework in l104_debug.py (v3.0.0).

For the full-featured debug suite, use:
    python l104_debug.py                          # All engines, all phases
    python l104_debug.py --engines code,math,science --phase cross
    python l104_debug.py --json --report cross_engine_debug_report.json

This wrapper runs: boot + status + constants + self-test + cross + perf
for the three original engines (code, science, math) and saves the report.

INVARIANT: 527.5184818492612 | PILOT: LONDEL
══════════════════════════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ═══════════════════════════════════════════════════════════════════════════════
#  DELEGATE TO UNIFIED DEBUG FRAMEWORK
# ═══════════════════════════════════════════════════════════════════════════════

from l104_debug import run_debug

if __name__ == "__main__":
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║  LEGACY WRAPPER — Now using l104_debug.py v3.0.0                ║")
    print("║  For full options: python l104_debug.py --help                  ║")
    print("║  NEW: --phase quantum for HHL + DualLayer + Pipeline deep tests ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")

    diag = run_debug(
        engine_names=["code", "science", "math"],
        phases=None,  # all phases
        report_path=str(ROOT / "cross_engine_debug_report.json"),
    )

    sys.exit(0 if diag.failed == 0 else 1)

