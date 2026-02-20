#!/usr/bin/env python3
"""Phase 3A: Decompose l104_asi_core.py → l104_asi/ package.

Reads the original file and writes each class group to a separate module.
Each module gets a header that imports from constants.py.
"""
import re

SRC = "l104_asi_core.py"

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
# constants.py: lines 0 to first class (DomainKnowledge)
constants_end = class_starts["DomainKnowledge"]

# domain.py: DomainKnowledge, GeneralDomainExpander, Theorem
domain_start = class_starts["DomainKnowledge"]
domain_end = class_starts["NovelTheoremGenerator"]

# theorem_gen.py: NovelTheoremGenerator
theorem_start = class_starts["NovelTheoremGenerator"]
theorem_end = class_starts["SelfModificationEngine"]

# self_mod.py: SelfModificationEngine
selfmod_start = class_starts["SelfModificationEngine"]
selfmod_end = class_starts["ConsciousnessVerifier"]

# consciousness.py: ConsciousnessVerifier
conscious_start = class_starts["ConsciousnessVerifier"]
conscious_end = class_starts["SolutionChannel"]

# pipeline.py: SolutionChannel, DirectSolutionHub, PipelineTelemetry, SoftmaxGatingRouter, AdaptivePipelineRouter
pipeline_start = class_starts["SolutionChannel"]
pipeline_end = class_starts["TreeOfThoughts"]

# reasoning.py: TreeOfThoughts, MultiHopReasoningChain, SolutionEnsembleEngine, PipelineHealthDashboard, PipelineReplayBuffer
reasoning_start = class_starts["TreeOfThoughts"]
reasoning_end = class_starts["QuantumComputationCore"]

# quantum.py: QuantumComputationCore
quantum_start = class_starts["QuantumComputationCore"]
quantum_end = class_starts["ASICore"]

# core.py: ASICore + main() + module-level stuff
core_start = class_starts["ASICore"]
core_end = total

# ── Write constants.py ──
with open("l104_asi/constants.py", "w") as f:
    f.write("".join(lines[:constants_end]))
print(f"  constants.py: lines 1-{constants_end} ({constants_end} lines)")

# ── Helper: write a module with constants import header ──
IMPORT_HEADER = """from .constants import *
"""

IMPORT_HEADER_WITH_DOMAIN = """from .constants import *
from .domain import DomainKnowledge, GeneralDomainExpander, Theorem
"""

def write_module(path, start, end, header=IMPORT_HEADER):
    content = header + "".join(lines[start:end])
    with open(path, "w") as f:
        f.write(content)
    print(f"  {path}: lines {start+1}-{end} ({end - start} lines)")

# ── Write domain.py ──
write_module("l104_asi/domain.py", domain_start, domain_end)

# ── Write theorem_gen.py ──
# NovelTheoremGenerator uses Theorem, constants
write_module("l104_asi/theorem_gen.py", theorem_start, theorem_end,
    header="from .constants import *\nfrom .domain import Theorem\n")

# ── Write self_mod.py ──
write_module("l104_asi/self_mod.py", selfmod_start, selfmod_end)

# ── Write consciousness.py ──
write_module("l104_asi/consciousness.py", conscious_start, conscious_end)

# ── Write pipeline.py ──
write_module("l104_asi/pipeline.py", pipeline_start, pipeline_end)

# ── Write reasoning.py ──
write_module("l104_asi/reasoning.py", reasoning_start, reasoning_end)

# ── Write quantum.py ──
write_module("l104_asi/quantum.py", quantum_start, quantum_end)

# ── Write core.py ──
# ASICore needs imports from all other modules
core_header = """from .constants import *
from .domain import DomainKnowledge, GeneralDomainExpander, Theorem
from .theorem_gen import NovelTheoremGenerator
from .self_mod import SelfModificationEngine
from .consciousness import ConsciousnessVerifier
from .pipeline import (SolutionChannel, DirectSolutionHub, PipelineTelemetry,
                       SoftmaxGatingRouter, AdaptivePipelineRouter)
from .reasoning import (TreeOfThoughts, MultiHopReasoningChain,
                        SolutionEnsembleEngine, PipelineHealthDashboard,
                        PipelineReplayBuffer)
from .quantum import QuantumComputationCore
"""
write_module("l104_asi/core.py", core_start, core_end, header=core_header)

# ── Write __init__.py ──
init_content = '''"""
l104_asi — Decomposed from l104_asi_core.py (5,845 lines → package)
Phase 3A of L104 Decompression Plan.

Re-exports ALL public symbols so that:
    from l104_asi import X
works identically to the original:
    from l104_asi_core import X
"""
# Constants and flags
from .constants import (
    ASI_CORE_VERSION, ASI_PIPELINE_EVO,
    PHI, GOD_CODE, TAU, PHI_CONJUGATE, VOID_CONSTANT, FEIGENBAUM,
    OMEGA_AUTHORITY, PLANCK_CONSCIOUSNESS, ALPHA_FINE,
    TORCH_AVAILABLE, TENSORFLOW_AVAILABLE, PANDAS_AVAILABLE,
    QISKIT_AVAILABLE,
)

# Domain
from .domain import DomainKnowledge, GeneralDomainExpander, Theorem

# Theorem generation
from .theorem_gen import NovelTheoremGenerator

# Self-modification
from .self_mod import SelfModificationEngine

# Consciousness verification
from .consciousness import ConsciousnessVerifier

# Pipeline
from .pipeline import (SolutionChannel, DirectSolutionHub, PipelineTelemetry,
                       SoftmaxGatingRouter, AdaptivePipelineRouter)

# Reasoning
from .reasoning import (TreeOfThoughts, MultiHopReasoningChain,
                        SolutionEnsembleEngine, PipelineHealthDashboard,
                        PipelineReplayBuffer)

# Quantum
from .quantum import QuantumComputationCore

# Core + singleton
from .core import ASICore, asi_core, main, get_current_parameters, update_parameters

# KerasASIModel is defined conditionally inside ASICore — re-export if available
try:
    from .core import KerasASIModel
except ImportError:
    pass
'''

with open("l104_asi/__init__.py", "w") as f:
    f.write(init_content)
print("  __init__.py: re-export hub")

print("\nDone! Phase 3A decomposition complete.")
print(f"Total: {total} lines decomposed into 10 modules.")
