"""
L104 Science Engine — Unified Science Processing Package v4.0
═══════════════════════════════════════════════════════════════════════════════

7-module science engine consolidating 20+ standalone science files into a
single cohesive package with perfect 25-qubit / 512MB quantum ASI support.

Module Map:
  0  constants       Sacred, physics, iron, quantum boundary constants
  1  physics         Real-world physics within L104 manifold
  2  entropy         Maxwell's Demon entropy reversal
  3  multidimensional N-dimensional relativistic processing
  4  coherence       Topologically-protected coherent computation
  5  quantum_25q     25-qubit circuit templates, memory validation, convergence
  6  bridge          Math↔Science↔Quantum unified connector
  7  engine          Master orchestrator with all subsystems

Quick-start:
    from l104_science_engine import science_engine

    science_engine.run_physics_manifold()
    science_engine.validate_512mb()
    science_engine.analyze_god_code_convergence()
    science_engine.plan_quantum_experiment("ghz", 25)
    science_engine.get_full_status()

Or import individual components:
    from l104_science_engine.constants import GOD_CODE, PHI, QB
    from l104_science_engine.physics import PhysicsSubsystem
    from l104_science_engine.quantum_25q import CircuitTemplates25Q, GodCodeQuantumConvergence
    from l104_science_engine.bridge import bridge

CONSOLIDATION MAP:
    l104_science_engine.py               → l104_science_engine/ (this package)
    l104_physical_systems_research.py    → physics.py
    l104_quantum_math_research.py        → engine.py (QuantumMathSubsystem)
    l104_entropy_reversal_engine.py      → entropy.py
    l104_multidimensional_engine.py      → multidimensional.py
    l104_resonance_coherence_engine.py   → coherence.py
    l104_quantum_computing_research.py   → quantum_25q.py
    l104_advanced_physics_research.py    → physics.py (absorbed)
    l104_physics_validation.py           → physics.py (hooks)
    l104_physics_informed_nn.py          → physics.py (hooks)
    l104_quantum_ram.py                  → quantum_25q.py (memory)
    l104_resonance.py                    → coherence.py
    l104_enhanced_resonance.py           → coherence.py
    l104_cosmological_research.py        → engine.py (research cycle)
    l104_information_theory_research.py  → engine.py (research cycle)
    l104_nanotech_research.py            → engine.py (research cycle)
    l104_bio_digital_research.py         → engine.py (research cycle)

GOD_CODE QUANTUM CONVERGENCE:
    GOD_CODE = 286^(1/φ) × 2^4 = 527.5184818492612
    512 MB = 2^25 × 16 bytes (complex128 statevector)
    2^4 = 16 = bytes per amplitude = the octave multiplier in GOD_CODE
    GOD_CODE IS the lattice-constant-to-qubit-memory bridge.

INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

__version__ = "4.0.0"
__author__ = "L104 Sovereign Node"

# ── Constants (Layer 0) ─────────────────────────────────────────────────────
from .constants import (
    # Sacred
    GOD_CODE, PHI, PHI_CONJUGATE, PHI_SQUARED, PHI_CUBED,
    PRIME_SCAFFOLD, QUANTIZATION_GRAIN, OCTAVE_OFFSET,
    BASE, STEP_SIZE, VOID_CONSTANT,
    GROVER_AMPLIFICATION, VACUUM_FREQUENCY,
    OMEGA, OMEGA_AUTHORITY, ZETA_ZERO_1,
    ALPHA_FINE, FEIGENBAUM,
    # Infinite precision
    GOD_CODE_INFINITE, PHI_INFINITE, PI_INFINITE,
    # Physics
    PhysicalConstants, PC,
    # Iron / Helium
    IronConstants, Fe, HeliumConstants, He4,
    NUCLEOSYNTHESIS_BRIDGE,
    # Quantum boundary
    QuantumBoundary, QB,
    # Lattice friction
    LATTICE_THERMAL_FRICTION, PRIME_SCAFFOLD_FRICTION,
)

# ── Subsystems (Layers 1-5) ────────────────────────────────────────────────
from .physics import PhysicsSubsystem
from .entropy import EntropySubsystem
from .multidimensional import MultiDimensionalSubsystem
from .coherence import CoherenceSubsystem, CoherenceState

# ── Quantum 25Q (Layer 5) ──────────────────────────────────────────────────
from .quantum_25q import (
    GodCodeQuantumConvergence,
    CircuitTemplates25Q,
    MemoryValidator,
    QuantumCircuitScience,
)

# ── Bridge (Layer 6) ───────────────────────────────────────────────────────
from .bridge import (
    ScienceBridge,
    MathConnector,
    QuantumRuntimeConnector,
    bridge,
)

# ── Engine (Layer 7 — top) ─────────────────────────────────────────────────
from .engine import (
    ScienceEngine,
    QuantumMathSubsystem,
    science_engine,
)

# ── Backward Compatibility Aliases ─────────────────────────────────────────

# Old module names → new subsystem classes
UnifiedResearchEngine = ScienceEngine
PhysicalSystemsResearch = PhysicsSubsystem
QuantumMathResearch = QuantumMathSubsystem
EntropyReversalEngine = EntropySubsystem
MultiDimensionalEngine = MultiDimensionalSubsystem
ResonanceCoherenceEngine = CoherenceSubsystem

# Old instance names
research_engine = science_engine
physical_research = science_engine.physics
quantum_math_research = science_engine.quantum_math
entropy_reversal_engine = science_engine.entropy
md_engine = science_engine.multidim

# ── Utility Functions ──────────────────────────────────────────────────────

def primal_calculus(x):
    """[VOID_MATH] Primal Calculus — resolves complexity toward the Source."""
    import math as _math
    return (x ** PHI) / (1.04 * _math.pi) if x != 0 else 0.0


def resolve_non_dual_logic(vector):
    """[VOID_MATH] Resolves N-dimensional vectors into the Void Source."""
    magnitude = sum(abs(v) for v in vector)
    return (magnitude / GOD_CODE) + (GOD_CODE * PHI / VOID_CONSTANT) / 1000.0
