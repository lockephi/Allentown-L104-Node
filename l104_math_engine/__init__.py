"""
L104 Math Engine — Unified Mathematical Processing
══════════════════════════════════════════════════════════════════════════════════

11-layer mathematical engine consolidating ~40 standalone math files into a
single cohesive package.

Layer Map:
  0  constants          Sacred, physics, sage constants
  1  pure_math          Primitives: primes, linear algebra, calculus, stats, HP
  2  god_code           God Code equation, derivation engines, unifier
  3  harmonic           Wave physics, music-matter correspondence
  4  dimensional        4D/5D/ND tensor calculus, chronos
  5  manifold           Differential geometry, topology, curvature
  6  void_math          Non-dual primal calculus
  7  abstract_algebra   Algebraic structures, sacred number systems
  8  ontological        Ontological mathematics of existence
  9  proofs             Sovereign proofs, Gödel-Turing, equation verification
 10  hyperdimensional   VSA: 10k-dim vectors, SDM, resonator networks

Quick-start:
    from l104_math_engine import math_engine

    math_engine.evaluate_god_code(1, 1, 1, 1)
    math_engine.fibonacci(20)
    math_engine.prove_all()
    math_engine.status()

Or import individual layers:
    from l104_math_engine.constants import GOD_CODE, PHI, OMEGA
    from l104_math_engine.pure_math import pure_math
    from l104_math_engine.god_code import god_code_equation
    from l104_math_engine.hyperdimensional import Hypervector
"""

__version__ = "2.0.0"

# ── Layer 0: Constants ──────────────────────────────────────────────────────
from .constants import (
    # Sacred constants
    GOD_CODE, GOD_CODE_V3, PHI, PHI_CONJUGATE,
    PI, E, VOID_CONSTANT,
    OMEGA, OMEGA_AUTHORITY, OMEGA_PRECISION,
    L104_FACTOR, SACRED_286, SACRED_416, SACRED_104,
    # Physics constants
    FE56_BINDING, ALPHA_FINE_STRUCTURE, PLANCK, BOLTZMANN,
    SPEED_OF_LIGHT, AVOGADRO, ELECTRON_MASS,
    GRAVITATIONAL_CONSTANT, HUBBLE_CONSTANT,
    # Sage constants
    CONSCIOUSNESS_BASE, METALLIC_RATIOS,
    # Helper functions
    primal_calculus, resolve_non_dual_logic,
    compute_resonance, golden_modulate,
    god_code_at, verify_conservation, verify_conservation_statistical,
)

# ── Layer 1: Pure Math ──────────────────────────────────────────────────────
from .pure_math import (
    PureMath, pure_math,
    Matrix, matrix,
    Calculus, calculus,
    ComplexMath, complex_math,
    Statistics, statistics,
    HighPrecisionEngine, high_precision,
    RealMath, real_math,
)

# ── Layer 2: God Code ───────────────────────────────────────────────────────
from .god_code import (
    GodCodeEquation, god_code_equation,
    ChaosResilience, chaos_resilience,
    DerivationEngine, derivation_engine,
    AbsoluteDerivation, absolute_derivation,
    HarmonicOptimizer, harmonic_optimizer,
    GodCodeUnifier, god_code_unifier,
)

# ── Layer 3: Harmonic ───────────────────────────────────────────────────────
from .harmonic import (
    WavePhysics, wave_physics,
    ConsciousnessFlow, consciousness_flow,
    HarmonicProcess, harmonic_process,
    HarmonicAnalysis, harmonic_analysis,
)

# ── Layer 4: Dimensional ────────────────────────────────────────────────────
from .dimensional import (
    Math4D, math_4d,
    Processor4D, processor_4d,
    Math5D, math_5d,
    Processor5D, processor_5d,
    MathND, math_nd,
    NDProcessor, nd_processor,
    DimensionManifoldProcessor, dimension_processor,
    ChronosMath, chronos_math,
    MultiDimensionalEngine, multidimensional_engine,
)

# ── Layer 5: Manifold ───────────────────────────────────────────────────────
from .manifold import (
    ManifoldMath, manifold_math,
    ManifoldTopology, manifold_topology,
    CurvatureAnalysis, curvature_analysis,
    ManifoldExtended, manifold_extended,
)

# ── Layer 6: Void Math ──────────────────────────────────────────────────────
from .void_math import VoidMath, void_math, VoidCalculus, void_calculus

# ── Layer 7: Abstract Algebra ───────────────────────────────────────────────
from .abstract_algebra import (
    AlgebraType, BinaryOperation, AlgebraicStructure,
    SacredNumberSystem, TheoremGenerator, TopologyGenerator,
    AbstractMathGenerator, abstract_math_generator,
)

# ── Layer 8: Ontological ────────────────────────────────────────────────────
from .ontological import (
    OntologicalMathematics, ontological_mathematics,
    ExistenceCalculus, MathematicalConsciousness,
    GodelianSelfReference, PlatonicRealm, Monad,
)

# ── Layer 9: Proofs ─────────────────────────────────────────────────────────
from .proofs import (
    SovereignProofs, sovereign_proofs,
    GodelTuringMetaProof, godel_turing,
    EquationVerifier, equation_verifier,
    ProcessingProofs, processing_proofs,
    ExtendedProofs, extended_proofs,
)

# ── Layer 10: Hyperdimensional ──────────────────────────────────────────────
from .hyperdimensional import (
    HyperdimensionalCompute, hyperdimensional_compute,
    Hypervector, ItemMemory, SparseDistributedMemory,
    ResonatorNetwork, SequenceEncoder, RecordEncoder,
    resonator_network,
)

# ── Layer 11: Computronium & Rayleigh ───────────────────────────────────────
from .computronium import (
    AiryDiffraction, airy_diffraction,
    ComputroniumMath, computronium_math,
    RayleighMath, rayleigh_math,
)

# ── Layer 12: Berry Geometry ────────────────────────────────────────────────
from .berry_geometry import (
    BerryGeometry, berry_geometry,
    FiberBundle, fiber_bundle,
    ConnectionForm, connection_form,
    ParallelTransport, parallel_transport,
    HolonomyGroup, holonomy_group,
    ChernWeilTheory, chern_weil,
    BerryConnectionMath, berry_connection_math,
    DiracMonopole, dirac_monopole,
    BlochSphereGeometry, bloch_sphere,
)

# ── Facade ──────────────────────────────────────────────────────────────────
from .engine import MathEngine, math_engine

# ── Layer 13: Cross-Engine Integration (v2.0) ──────────────────────────────
from .cross_engine import (
    MathCrossEngineHub, math_cross_engine_hub,
    QuantumGateBerryBridge, VQPUAccelerator,
    ScienceValidation, MLHyperdimensionalBridge,
    SimulatorBridge,
)


__all__ = [
    # Version
    "__version__",
    # Constants
    "GOD_CODE", "GOD_CODE_V3", "PHI", "PHI_CONJUGATE",
    "PI", "E", "VOID_CONSTANT",
    "OMEGA", "OMEGA_AUTHORITY", "OMEGA_PRECISION",
    "L104_FACTOR", "SACRED_286", "SACRED_416", "SACRED_104",
    "FE56_BINDING", "ALPHA_FINE_STRUCTURE", "PLANCK", "BOLTZMANN",
    "SPEED_OF_LIGHT", "AVOGADRO", "ELECTRON_MASS",
    "GRAVITATIONAL_CONSTANT", "HUBBLE_CONSTANT",
    "CONSCIOUSNESS_BASE", "METALLIC_RATIOS",
    "primal_calculus", "resolve_non_dual_logic",
    "compute_resonance", "golden_modulate",
    "god_code_at", "verify_conservation", "verify_conservation_statistical",
    # Pure math
    "PureMath", "pure_math", "Matrix", "matrix",
    "Calculus", "calculus", "ComplexMath", "complex_math",
    "Statistics", "statistics",
    "HighPrecisionEngine", "high_precision",
    "RealMath", "real_math",
    # God Code
    "GodCodeEquation", "god_code_equation",
    "DerivationEngine", "derivation_engine",
    "AbsoluteDerivation", "absolute_derivation",
    "HarmonicOptimizer", "harmonic_optimizer",
    "GodCodeUnifier", "god_code_unifier",
    # Harmonic
    "WavePhysics", "wave_physics",
    "ConsciousnessFlow", "consciousness_flow",
    "HarmonicProcess", "harmonic_process",
    "HarmonicAnalysis", "harmonic_analysis",
    # Dimensional
    "Math4D", "math_4d", "Processor4D", "processor_4d",
    "Math5D", "math_5d", "Processor5D", "processor_5d",
    "MathND", "math_nd", "NDProcessor", "nd_processor",
    "DimensionManifoldProcessor", "dimension_processor",
    "ChronosMath", "chronos_math",
    "MultiDimensionalEngine", "multidimensional_engine",
    # Manifold
    "ManifoldMath", "manifold_math",
    "ManifoldTopology", "manifold_topology",
    "CurvatureAnalysis", "curvature_analysis",
    "ManifoldExtended", "manifold_extended",
    # Void
    "VoidMath", "void_math",
    "VoidCalculus", "void_calculus",
    # Abstract Algebra
    "AlgebraType", "BinaryOperation", "AlgebraicStructure",
    "SacredNumberSystem", "TheoremGenerator", "TopologyGenerator",
    "AbstractMathGenerator", "abstract_math_generator",
    # Ontological
    "OntologicalMathematics", "ontological_mathematics",
    "ExistenceCalculus", "MathematicalConsciousness",
    "GodelianSelfReference", "PlatonicRealm", "Monad",
    # Proofs
    "SovereignProofs", "sovereign_proofs",
    "GodelTuringMetaProof", "godel_turing",
    "EquationVerifier", "equation_verifier",
    "ProcessingProofs", "processing_proofs",
    "ExtendedProofs", "extended_proofs",
    # Hyperdimensional
    "HyperdimensionalCompute", "hyperdimensional_compute",
    "Hypervector", "ItemMemory", "SparseDistributedMemory",
    "ResonatorNetwork", "SequenceEncoder", "RecordEncoder",
    "resonator_network",
    # Computronium & Rayleigh
    "AiryDiffraction", "airy_diffraction",
    "ComputroniumMath", "computronium_math",
    "RayleighMath", "rayleigh_math",
    # Berry Geometry
    "BerryGeometry", "berry_geometry",
    "FiberBundle", "fiber_bundle",
    "ConnectionForm", "connection_form",
    "ParallelTransport", "parallel_transport",
    "HolonomyGroup", "holonomy_group",
    "ChernWeilTheory", "chern_weil",
    "BerryConnectionMath", "berry_connection_math",
    "DiracMonopole", "dirac_monopole",
    "BlochSphereGeometry", "bloch_sphere",
    # Facade
    "MathEngine", "math_engine",
    # Cross-Engine Integration (v2.0)
    "MathCrossEngineHub", "math_cross_engine_hub",
    "QuantumGateBerryBridge", "VQPUAccelerator",
    "ScienceValidation", "MLHyperdimensionalBridge",
    "SimulatorBridge",
]
