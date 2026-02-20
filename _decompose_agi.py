#!/usr/bin/env python3
"""Phase 3B: Decompose l104_agi_core.py → l104_agi/ package.

Reads the original file and writes each class group to a separate module.
"""
import re

SRC = "l104_agi_core.py"

with open(SRC, "r") as f:
    lines = f.readlines()

total = len(lines)
print(f"Read {total} lines from {SRC}")

# ── Find class boundaries ──
class_starts = {}
for i, line in enumerate(lines):
    m = re.match(r'^class (\w+)', line)
    if m:
        class_starts[m.group(1)] = i

print("Classes found:", {k: v+1 for k, v in class_starts.items()})

# ── Define module assignments ──
# constants.py: lines 0 to first class (PipelineCircuitBreaker)
constants_end = class_starts["PipelineCircuitBreaker"]

# circuit_breaker.py: PipelineCircuitBreaker
cb_start = class_starts["PipelineCircuitBreaker"]
cb_end = class_starts["AGICore"]

# core.py: AGICore + module-level singleton + primal_calculus + resolve_non_dual_logic
core_start = class_starts["AGICore"]
core_end = total

# ── Write constants.py ──
with open("l104_agi/constants.py", "w") as f:
    f.write("".join(lines[:constants_end]))
print(f"  constants.py: lines 1-{constants_end} ({constants_end} lines)")

# ── Write circuit_breaker.py ──
cb_header = "from .constants import *\n"
cb_content = cb_header + "".join(lines[cb_start:cb_end])
with open("l104_agi/circuit_breaker.py", "w") as f:
    f.write(cb_content)
print(f"  circuit_breaker.py: lines {cb_start+1}-{cb_end} ({cb_end - cb_start} lines)")

# ── Write core.py ──
core_header = "from .constants import *\nfrom .circuit_breaker import PipelineCircuitBreaker\n"
core_content = core_header + "".join(lines[core_start:core_end])
with open("l104_agi/core.py", "w") as f:
    f.write(core_content)
print(f"  core.py: lines {core_start+1}-{core_end} ({core_end - core_start} lines)")

# ── Write __init__.py ──
init_content = '''"""
l104_agi — Decomposed from l104_agi_core.py (3,161 lines → package)
Phase 3B of L104 Decompression Plan.

Re-exports ALL public symbols so that:
    from l104_agi import X
works identically to the original:
    from l104_agi_core import X
"""
# Constants
from .constants import (
    AGI_CORE_VERSION, AGI_PIPELINE_EVO,
    PHI, GOD_CODE, TAU, FEIGENBAUM, ALPHA_FINE, VOID_CONSTANT,
    QISKIT_AVAILABLE,
)

# Circuit breaker
from .circuit_breaker import PipelineCircuitBreaker

# Core + singleton + module-level functions
from .core import AGICore, agi_core, primal_calculus, resolve_non_dual_logic

# Alias for backward compatibility (2 importers use L104AGICore)
L104AGICore = AGICore
'''

with open("l104_agi/__init__.py", "w") as f:
    f.write(init_content)
print("  __init__.py: re-export hub")

print(f"\nDone! Phase 3B decomposition complete.")
print(f"Total: {total} lines decomposed into 4 modules.")
