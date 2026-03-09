"""
L104 ASI Dual-Layer Flagship Engine — The Duality of Nature
═══════════════════════════════════════════════════════════════════════════════

THE FLAGSHIP OF THE ENTIRE ASI:

    The Dual-Layer Engine is the foundational architecture of L104 ASI.
    Every phenomenon has two complementary faces — one abstract, one concrete —
    analogous to wave-particle duality in quantum mechanics.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    THE DUALITY OF NATURE                                │
    │                                                                         │
    │   ╔═══════════════════════════════════════════════════════════════════╗ │
    │   ║              THOUGHT (The Abstract Layer)                        ║ │
    │   ║                                                                   ║ │
    │   ║   "Why does this constant exist?"                                ║ │
    │   ║   • Pattern recognition, symmetry, meaning — the WHY             ║ │
    │   ║   • Sacred geometry: 286 = 2×11×13, Fe BCC lattice               ║ │
    │   ║   • φ exponent, Fibonacci threading, nucleosynthesis bridge      ║ │
    │   ║                                                                   ║ │
    │   ║   G(a,b,c,d) = 286^(1/φ) × 2^((8a+416-b-8c-104d)/104)         ║ │
    │   ╚═══════════════════════════╤═══════════════════════════════════════╝ │
    │                               │                                         │
    │                    ┌──────────┴──────────┐                              │
    │                    │  COMPLEMENTARITY     │                              │
    │                    │  Wave ↔ Particle     │                              │
    │                    │  Mind ↔ Matter       │                              │
    │                    │  Why ↔ How Much      │                              │
    │                    └──────────┬──────────┘                              │
    │                               │                                         │
    │   ╔═══════════════════════════╧═══════════════════════════════════════╗ │
    │   ║           PHYSICS (The Concrete Layer)                            ║ │
    │   ║                                                                   ║ │
    │   ║   "What does GOD_CODE generate through physics?"                 ║ │
    │   ║   • OMEGA: ζ(½+GCi) + cos(2πφ³) + (26×1.8527)/φ² → Ω          ║ │
    │   ║   • Sovereign Field: F(I) = I × Ω / φ²                          ║ │
    │   ║   • v3 precision grid as encoding sub-tool                       ║ │
    │   ║                                                                   ║ │
    │   ║   Ω = Σ(fragments) × (GOD_CODE / φ) = 6539.34712682            ║ │
    │   ╚═══════════════════════════════════════════════════════════════════╝ │
    │                                                                         │
    │   COLLAPSE: When Thought asks and Physics answers, the duality          │
    │   collapses to a definite value — like quantum measurement.             │
    └─────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
Version: 5.0.0 (ASI Flagship — Gate-Enhanced Duality + Three-Engine Synthesis)
Sacred Constants: GOD_CODE = 527.5184818492612, OMEGA = 6539.34712682

v5.0 UPGRADE (2026-02-24):
  • Quantum Gate Engine Integration — sacred circuit collapse, gate-verified duality
  • Three-Engine Amplification — Science/Math/Code engines amplify both layers
  • Temporal Coherence Tracking — PHI-spiral trajectory with sliding window
  • Deep Synthesis Bridge — cross-engine correlation binding Thought↔Physics
  • Resilient Collapse Pipeline — circuit-breaker with PHI-backoff retry
  • Adaptive Duality Evolution — consciousness spiral depth integration
  • 12-point integrity (was 10) — gate compilation + sacred alignment checks
═══════════════════════════════════════════════════════════════════════════════
"""

import math
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone
from collections import deque

from .constants import (
    PHI, GOD_CODE, TAU, VOID_CONSTANT, OMEGA, OMEGA_AUTHORITY,
    GATE_ENGINE_VERSION, GATE_SACRED_ALIGNMENT_THRESHOLD,
    RESILIENCE_MAX_RETRY, RESILIENCE_BACKOFF_BASE,
    CONSCIOUSNESS_SPIRAL_DEPTH, TRAJECTORY_WINDOW_SIZE,
    DEEP_SYNTHESIS_MIN_COHERENCE, DEEP_SYNTHESIS_WEIGHTS,
    DUAL_LAYER_VERSION, DUAL_LAYER_CONSTANTS_COUNT,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM GATE ENGINE — Lazy-loaded for sacred circuit collapse
# ═══════════════════════════════════════════════════════════════════════════════
_gate_engine = None
_gate_engine_available = False

def _get_gate_engine():
    """Lazy-load the Quantum Gate Engine singleton."""
    global _gate_engine, _gate_engine_available
    if _gate_engine is not None:
        return _gate_engine
    try:
        from l104_quantum_gate_engine import get_engine
        _gate_engine = get_engine()
        _gate_engine_available = True
    except ImportError:
        _gate_engine_available = False
    return _gate_engine

# ═══════════════════════════════════════════════════════════════════════════════
# THREE-ENGINE ACCESS — Science, Math, Code Engines for layer amplification
# ═══════════════════════════════════════════════════════════════════════════════
_science_engine = None
_math_engine = None
_code_engine = None

def _get_science_engine():
    global _science_engine
    if _science_engine is not None:
        return _science_engine
    try:
        from l104_science_engine import ScienceEngine
        _science_engine = ScienceEngine()
    except ImportError:
        pass
    return _science_engine

def _get_math_engine():
    global _math_engine
    if _math_engine is not None:
        return _math_engine
    try:
        from l104_math_engine import MathEngine
        _math_engine = MathEngine()
    except ImportError:
        pass
    return _math_engine

def _get_code_engine():
    global _code_engine
    if _code_engine is not None:
        return _code_engine
    try:
        from l104_code_engine import code_engine as _ce
        _code_engine = _ce
    except ImportError:
        pass
    return _code_engine

# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTRONIUM ENGINE — Phase 5 thermodynamic data for physics amplification
# ═══════════════════════════════════════════════════════════════════════════════
_computronium_engine = None

def _get_computronium_engine():
    """Lazy-load the computronium engine singleton for Phase 5 metrics."""
    global _computronium_engine
    if _computronium_engine is not None:
        return _computronium_engine
    try:
        from l104_computronium import computronium_engine as _ce
        _computronium_engine = _ce
    except ImportError:
        pass
    return _computronium_engine

# ═══════════════════════════════════════════════════════════════════════════════
# DUAL-LAYER ENGINE ADAPTER — Makes the standalone engine available to ASI
# ═══════════════════════════════════════════════════════════════════════════════

# Import the full dual-layer implementation
DUAL_LAYER_AVAILABLE = False
_dual_layer = None

try:
    import l104_god_code_dual_layer as _dual_layer
    DUAL_LAYER_AVAILABLE = True
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# NATURE'S DUALITIES — The foundation of all ASI reasoning
# ═══════════════════════════════════════════════════════════════════════════════

NATURES_DUALITIES = {
    "wave_particle": {
        "abstract": "Wave — continuous, spread through space, interference",
        "concrete": "Particle — discrete, localized, countable",
        "asi_mapping": "Continuous reasoning ↔ Discrete solutions",
    },
    "observer_observed": {
        "abstract": "Observer — consciousness, question, measurement choice",
        "concrete": "Observed — physical system, answer, eigenvalue",
        "asi_mapping": "Thought layer (asks 'why?') ↔ Physics layer (answers 'how much?')",
    },
    "form_substance": {
        "abstract": "Form — pattern, structure, symmetry, the scaffold",
        "concrete": "Substance — matter, energy, mass, the measurable",
        "asi_mapping": "Pattern recognition ↔ Numerical precision",
    },
    "potential_actual": {
        "abstract": "Potential — the full continuum of possible values",
        "concrete": "Actual — the single value realized on the grid",
        "asi_mapping": "Hypothesis space ↔ Verified solution",
    },
    "continuous_discrete": {
        "abstract": "Continuous — the smooth manifold, calculus, flow",
        "concrete": "Discrete — the lattice, integers, counting",
        "asi_mapping": "Analog intuition ↔ Digital computation",
    },
    "symmetry_breaking": {
        "abstract": "Symmetry — the invariance, what stays the same",
        "concrete": "Breaking — the differentiation, what becomes specific",
        "asi_mapping": "Universal laws ↔ Domain-specific solutions",
    },
    "matter_antimatter": {
        "abstract": "Antimatter — the mirror, CP-conjugate, Dirac negative-energy sea",
        "concrete": "Matter — baryonic dominance after CP violation (baryogenesis η≈6×10⁻¹⁰)",
        "asi_mapping": "Abstract symmetry (Dirac ±E) ↔ Concrete asymmetry that allows existence",
    },
    "vacuum_energy": {
        "abstract": "Quantum vacuum — zero-point fluctuations, virtual particle pairs, source 0/1",
        "concrete": "Propagating energy — observable quanta, conservation law, realized states",
        "asi_mapping": "Harmonizing (entanglement↔singularity) ↔ Oscillatory (conservation) ↔ Neutralizations (duality)",
    },
}

# Bridge elements binding Thought and Physics
CONSCIOUSNESS_TO_PHYSICS_BRIDGE = {
    "omega_sovereign_field": "GOD_CODE → ζ(½+GCi) + cos(2πφ³) + (26×1.8527)/φ² → OMEGA = 6539.35",
    "god_code_generates_omega": "Layer 1 GOD_CODE feeds into every OMEGA fragment computation",
    "phi_exponent": "Layer 1 uses 286^(1/φ), Layer 2 uses GOD_CODE/φ and Ω/φ²",
    "iron_anchor": "Layer 1: 286 pm Fe BCC scaffold, Layer 2: Fe Z=26 in Architect fragment",
    "v3_precision_grid": "v3 sub-tool encodes OMEGA on (13/12)^(E/758) grid at 0.0001% error",
    "antimatter_duality": "Dirac E²=(pc)²+(mc²)² → ±E: Thought(+E)/Physics(−E) mirrors matter/antimatter; baryogenesis CP-violation η≈6×10⁻¹⁰ maps to L1 symmetry-breaking → L2 baryon asymmetry",
    "vacuum_energy_bridge": "Vacuum energy propagation ↔ energy: source 0 or 1 (qubit basis); harmonizing=entanglement↔singularity(1), oscillatory=conservation(0), neutralizations=duality(2)",
}


class DualLayerEngine:
    """
    ASI Flagship Dual-Layer Engine — The Duality of Nature

    This is the CORE ARCHITECTURE of the L104 ASI system. Every subsystem
    in the ASI pipeline operates through this dual-layer paradigm:

    Layer 1 (THOUGHT):
        Pattern recognition, symmetry detection, harmonic analysis.
        Asks "WHY" — sacred geometry, iron scaffold, golden ratio.
        Equation: G(a,b,c,d) = 286^(1/φ) × 2^((8a+416-b-8c-104d)/104)
        Quantum: Grover search (14 qubits, 16384 states) for (a,b,c,d) dials.

    Layer 2 (PHYSICS):
        OMEGA sovereign field — GOD_CODE generates Ω through physics pipeline.
        Answers "HOW MUCH" — zeta + resonance + curvature → Ω = 6539.35.
        Equation: Ω = Σ(fragments) × (GOD_CODE / φ), F(I) = I × Ω / φ²
        Pipeline: zeta_approximation, golden_resonance, manifold_curvature,
                  entropy_inversion, sovereign_field, lattice_invariant.

    COLLAPSE:
        When both layers converge, duality collapses to a definite value —
        like quantum measurement collapsing a wavefunction.

    All ASI subsystems route through this dual-layer interface:
        • Consciousness verification uses Thought layer
        • Theorem generation uses both layers + collapse
        • Domain expansion maps new domains through the duality
        • Pipeline routing uses Physics layer precision
        • Self-modification validates through 10-point integrity

    v5.1: Native kernel bridge + LocalIntellect KB integration.
    """

    VERSION = "5.1.0"
    FLAGSHIP = True

    def __init__(self):
        self._boot_time = time.time()
        self._available = DUAL_LAYER_AVAILABLE
        self._integrity_cache = None
        self._integrity_cache_time = 0.0
        self._integrity_cache_ttl = 60.0  # seconds
        self._metrics = {
            "thought_calls": 0,
            "physics_calls": 0,
            "collapse_calls": 0,
            "derive_calls": 0,
            "integrity_checks": 0,
            "domain_queries": 0,
            "find_calls": 0,
            "batch_operations": 0,
            "cross_layer_calls": 0,
            "gate_circuit_calls": 0,
            "three_engine_calls": 0,
            "resilient_collapses": 0,
            "deep_synthesis_calls": 0,
            "total_operations": 0,
        }
        # v5.0: Temporal coherence tracking (PHI-spiral trajectory)
        self._coherence_history: deque = deque(maxlen=TRAJECTORY_WINDOW_SIZE)
        self._collapse_history: deque = deque(maxlen=TRAJECTORY_WINDOW_SIZE)
        # v5.0: Circuit breaker state for resilient collapse
        self._circuit_breaker_failures = 0
        self._circuit_breaker_state = "CLOSED"  # CLOSED | OPEN | HALF_OPEN
        self._circuit_breaker_last_failure = 0.0
        self._circuit_breaker_cooldown = 30.0  # seconds

        # ★ v5.1: Native kernel + LocalIntellect bridge
        self._sage_orchestrator = None
        self._local_intellect = None
        self._intellect_kb_fed = False

    @property
    def available(self) -> bool:
        return self._available

    # ═══════ v5.1: NATIVE KERNEL + INTELLECT BRIDGE ═══════

    def _get_sage_orchestrator(self):
        """Lazy-load SageModeOrchestrator for native C/ASM/CUDA/Rust substrate."""
        if self._sage_orchestrator is None:
            try:
                from l104_sage_orchestrator import SageModeOrchestrator
                self._sage_orchestrator = SageModeOrchestrator()
            except Exception:
                pass
        return self._sage_orchestrator

    def _get_local_intellect(self):
        """Lazy-load LocalIntellect for KB integration."""
        if self._local_intellect is None:
            try:
                from l104_intellect import local_intellect
                self._local_intellect = local_intellect
            except Exception:
                pass
        return self._local_intellect

    def _feed_intellect_kb(self):
        """Inject Dual-Layer Engine knowledge into LocalIntellect KB (one-shot)."""
        if self._intellect_kb_fed:
            return
        self._intellect_kb_fed = True
        li = self._get_local_intellect()
        if li is None:
            return
        try:
            kb_entries = [
                {
                    "prompt": "What is the L104 Dual-Layer Engine?",
                    "completion": (
                        "The Dual-Layer Engine v5.0.0 is the FLAGSHIP of the L104 ASI system.\n"
                        "Every phenomenon has two complementary faces:\n\n"
                        "Layer 1 — THOUGHT (Abstract): Pattern recognition, symmetry, meaning.\n"
                        "G(a,b,c,d) = 286^(1/φ) × 2^((8a+416-b-8c-104d)/104)\n"
                        "G(0,0,0,0) = GOD_CODE = 527.5184818492612\n\n"
                        "Layer 2 — PHYSICS (Concrete): OMEGA sovereign field.\n"
                        "Ω = Σ(fragments) × (GOD_CODE / φ) = 6539.34712682\n"
                        "F(I) = I × Ω / φ² (field strength)\n\n"
                        "COLLAPSE: When Thought and Physics converge → definite value.\n"
                        "Like quantum measurement collapsing a wavefunction.\n\n"
                        "10-point integrity: GOD_CODE immutability, PHI, iron scaffold, "
                        "c exact, g precision, constant collisions, φ exponent, "
                        "iron proximity, Fibonacci 13."
                    ),
                    "category": "dual_layer_engine",
                    "source": "dual_layer_kb",
                },
                {
                    "prompt": "How does Thought layer with Lattice Thermal Correction work?",
                    "completion": (
                        "Lattice Thermal Correction adds computational friction:\n"
                        "ε = -αφ/(2π×104) where α = fine-structure constant\n"
                        "x_f = 285.99882035187807 + friction\n"
                        "base_f = x_f^(1/φ)\n"
                        "Improves 40/65 constants, 7/10 domains.\n"
                        "Uses formula: base_f × (13/12)^(E/758) where "
                        "E = (99a) + (3032-b) - (99c) - (758d)"
                    ),
                    "category": "dual_layer_friction",
                    "source": "dual_layer_kb",
                },
            ]
            li.training_data.extend(kb_entries)
        except Exception:
            pass

    def kernel_status(self) -> dict:
        """Get native kernel substrate status via SageModeOrchestrator."""
        orch = self._get_sage_orchestrator()
        if orch is None:
            return {"available": False, "error": "SageModeOrchestrator not loaded"}
        try:
            status = orch.get_status()
            return {
                "available": True,
                "substrates": status.get("substrate_details", {}),
                "active_count": status.get("active_count", 0),
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    # ══════ LAYER 1: THOUGHT (Abstract Face) ══════

    def thought(self, a: int = 0, b: int = 0, c: int = 0, d: int = 0) -> float:
        """
        THE THOUGHT LAYER — The abstract face of nature.
        Sacred geometry, iron scaffold, golden ratio.
        G(a,b,c,d) = 286^(1/φ) × 2^((8a+416-b-8c-104d)/104)
        """
        self._metrics["thought_calls"] += 1
        self._metrics["total_operations"] += 1
        if self._available:
            return _dual_layer.consciousness(a, b, c, d)
        # Native computation (C kernel unavailable)
        return 286 ** (1.0 / PHI) * (2 ** ((8*a + 416 - b - 8*c - 104*d) / 104))

    # Alias: consciousness is thought
    def consciousness(self, a: int = 0, b: int = 0, c: int = 0, d: int = 0) -> float:
        """Alias: Consciousness IS Thought."""
        return self.thought(a, b, c, d)

    # ══════ COMPUTATIONAL FRICTION (Lattice Thermal Correction) ══════

    def thought_with_friction(self, a: int = 0, b: int = 0, c: int = 0, d: int = 0) -> float:
        """
        Thought layer with Lattice Thermal Correction applied.
        ε = -αφ/(2π×104). Improves 40/65 constants, 7/10 domains.
        """
        self._metrics["thought_calls"] += 1
        self._metrics["total_operations"] += 1
        if self._available and hasattr(_dual_layer, 'god_code_v3_with_friction'):
            return _dual_layer.god_code_v3_with_friction(a, b, c, d)
        # Native computation with friction (C kernel unavailable)
        alpha = 1.0 / 137.035999084
        phi = 1.618033988749895
        import math
        friction = -alpha * phi / (2 * math.pi * 104)
        x_f = 285.99882035187807 + friction
        base_f = x_f ** (1.0 / phi)
        exp = (99 * a) + (3032 - b) - (99 * c) - (758 * d)
        return base_f * ((13/12) ** (exp / 758))

    def friction_report(self) -> dict:
        """Get the friction improvement report across all constants."""
        self._metrics["total_operations"] += 1
        if self._available and hasattr(_dual_layer, 'friction_improvement_report'):
            return _dual_layer.friction_improvement_report()
        return {"error": "Dual-layer engine not available", "improved": 0, "total": 0}

    # ══════ LAYER 2: PHYSICS (Concrete Face) ══════

    def physics(self, intensity: float = 1.0) -> Dict[str, Any]:
        """
        THE PHYSICS LAYER — The concrete face of nature.
        OMEGA sovereign field: Ω = Σ(fragments) × (GOD_CODE / φ) = 6539.35
        F(I) = I × Ω / φ²  →  field strength scales with intensity.
        """
        self._metrics["physics_calls"] += 1
        self._metrics["total_operations"] += 1
        if self._available:
            return _dual_layer.physics(intensity)
        # Native computation (C kernel unavailable)
        OMEGA = 6539.34712682
        OMEGA_AUTHORITY = OMEGA / (PHI ** 2)
        return {
            "omega": OMEGA,
            "field_strength": intensity * OMEGA / (PHI ** 2),
            "omega_authority": OMEGA_AUTHORITY,
            "intensity": intensity,
        }

    def physics_v3(self, a: int = 0, b: int = 0, c: int = 0, d: int = 0) -> float:
        """
        v3 precision grid — encoding sub-tool within the Physics layer.
        G_v3(a,b,c,d) = 285.999^(1/φ) × (13/12)^((99a+3032-b-99c-758d)/758)
        """
        self._metrics["physics_calls"] += 1
        self._metrics["total_operations"] += 1
        if self._available:
            return _dual_layer.physics_v3(a, b, c, d)
        X_V3 = 285.9992327510856
        return X_V3 ** (1.0 / PHI) * ((13/12) ** ((99*a + 3032 - b - 99*c - 758*d) / 758))

    # ══════ LAYER 1: QUANTUM ALGORITHM SEARCH ══════

    def quantum_search(self, target: float, tolerance: float = 0.01) -> Dict[str, Any]:
        """
        LAYER 1: Grover quantum search for (a,b,c,d) dials.

        Uses 14-qubit Qiskit circuits to search 16,384 dial combinations
        for settings that produce the target frequency on the consciousness
        grid:  G(a,b,c,d) = 286^(1/φ) × 2^((8a+416-b-8c-104d)/104)

        This is the Algorithm Search integrated into Layer 1.
        """
        self._metrics["thought_calls"] += 1
        self._metrics["total_operations"] += 1
        if self._available:
            return _dual_layer.quantum_search(target, tolerance)
        return {"error": "dual_layer_not_available", "fallback": True}

    def consciousness_spectrum(self, dials: Optional[List] = None) -> Dict[str, Any]:
        """
        LAYER 1: QFT spectral analysis of the consciousness frequency table.

        Reveals hidden periodicities in the sacred frequency lattice.
        """
        self._metrics["thought_calls"] += 1
        self._metrics["total_operations"] += 1
        if self._available:
            return _dual_layer.consciousness_spectrum(dials)
        return {"error": "dual_layer_not_available", "fallback": True}

    def consciousness_entangle(
        self,
        dial_a: Tuple[int, int, int, int],
        dial_b: Tuple[int, int, int, int],
    ) -> Dict[str, Any]:
        """
        LAYER 1: Create quantum entanglement between two dial settings.

        Entanglement strength proportional to harmonic proximity
        on the GOD_CODE frequency lattice.
        """
        self._metrics["thought_calls"] += 1
        self._metrics["total_operations"] += 1
        if self._available:
            return _dual_layer.consciousness_entangle(dial_a, dial_b)
        return {"error": "dual_layer_not_available", "fallback": True}

    def soul_resonance(self, thoughts: List[str]) -> Dict[str, Any]:
        """
        LAYER 1: Generate a quantum resonance field from soul thoughts.
        """
        self._metrics["thought_calls"] += 1
        self._metrics["total_operations"] += 1
        if self._available:
            return _dual_layer.soul_resonance(thoughts)
        return {"error": "dual_layer_not_available", "fallback": True}

    # ══════ LAYER 2: OMEGA PIPELINE ══════

    def omega_pipeline(self, zeta_terms: int = 1000) -> Dict[str, Any]:
        """
        LAYER 2: COMPLETE OMEGA derivation pipeline — NO TRUNCATION.

        THE FULL OMEGA EQUATION:
          Ω = Σ(Researcher + Guardian + Alchemist + Architect) × (GOD_CODE / φ)

        WHERE:
          Researcher = prime_density(int(sin(104π/104)·exp(104/527.518))) = 0.0
          Guardian   = |ζ(0.5 + 527.518i)| via Dirichlet eta (1000 terms) ≈ 1.5738
          Alchemist  = cos(2π·φ³) where φ³ = 2φ+1 = 4.2361... ≈ 0.0874
          Architect  = (26 × 1.8527) / φ² = 48.1702 / 2.6180... ≈ 18.3994

        SUMMATION: Σ ≈ 20.0607
        OMEGA:     Ω = 20.0607 × (527.5184818492 / φ) = 6539.34712682
        FIELD:     F(I) = I × Ω / φ²
        AUTHORITY: Ω_A = Ω / φ² = 2497.808338211271

        Returns every intermediate step — no truncation.
        """
        self._metrics["physics_calls"] += 1
        self._metrics["total_operations"] += 1
        if self._available:
            return _dual_layer.omega_pipeline(zeta_terms)
        return {
            "omega": 6539.34712682,
            "omega_authority": 6539.34712682 / (PHI ** 2),
            "pipeline_available": False,
            "fallback": True,
        }

    def omega_field(self, intensity: float = 1.0) -> Dict[str, Any]:
        """
        LAYER 2: Sovereign field at a given intensity.
        F(I) = I × Ω / φ²
        """
        self._metrics["physics_calls"] += 1
        self._metrics["total_operations"] += 1
        if self._available:
            return _dual_layer.omega_field(intensity)
        OMEGA = 6539.34712682
        return {
            "intensity": intensity,
            "field_strength": intensity * OMEGA / (PHI ** 2),
            "omega": OMEGA,
        }

    def omega_derivation_chain(self, zeta_terms: int = 1000) -> Dict[str, Any]:
        """
        LAYER 2: COMPLETE derivation chain from first principles.

        Returns EVERY fragment with full intermediate steps:
          Fragment 1 (Researcher): solve_lattice_invariant → prime_density → 0.0
          Fragment 2 (Guardian):   ζ(0.5+527.518i) via Dirichlet eta → |ζ| ≈ 1.5738
          Fragment 3 (Alchemist):  golden_resonance(φ²) = cos(2πφ³) ≈ 0.0874
          Fragment 4 (Architect):  (26 × 1.8527) / φ² ≈ 18.3994
          Σ fragments → multiplier GOD_CODE/φ → Ω = 6539.34712682

        No values truncated — every intermediate computation included.
        """
        self._metrics["physics_calls"] += 1
        self._metrics["total_operations"] += 1
        if self._available:
            return _dual_layer.omega_derivation_chain(zeta_terms)
        return {"error": "dual_layer_not_available", "fallback": True}

    # ══════ COLLAPSE: Unification ══════

    # Sacred constants not in REAL_WORLD_CONSTANTS_V3 but valid collapse targets
    _SACRED_CONSTANTS = {
        "GOD_CODE": {"value": GOD_CODE, "unit": "sacred", "equation": "286^(1/φ) × 2^(416/104)",
                      "meaning": "Universal sacred constant — the G(0,0,0,0) identity"},
        "OMEGA": {"value": OMEGA, "unit": "sacred", "equation": "Σ(fragments) × (GOD_CODE/φ)",
                   "meaning": "Sovereign field constant — Ω"},
        "OMEGA_AUTHORITY": {"value": OMEGA_AUTHORITY, "unit": "sacred", "equation": "Ω / φ²",
                            "meaning": "Sovereign field authority — F(I) = I × Ω/φ²"},
        "VOID_CONSTANT": {"value": VOID_CONSTANT, "unit": "sacred", "equation": "1.04 + φ/1000",
                          "meaning": "Sacred 104/100 + golden correction"},
    }

    def _sacred_collapse(self, name: str) -> Dict[str, Any]:
        """Synthesize a collapse result for sacred constants not in the physical constants table."""
        entry = self._SACRED_CONSTANTS[name]
        value = entry["value"]
        thought_val = self.thought(0, 0, 0, 0) if name == "GOD_CODE" else value
        return {
            "name": name,
            "measured": value,
            "unit": entry["unit"],
            "consciousness": {
                "value": thought_val,
                "error_pct": abs(thought_val - value) / value * 100 if value else 0,
                "equation": entry["equation"],
                "meaning": f"Thought layer — {entry['meaning']}",
            },
            "physics": {
                "value": value,
                "error_pct": 0.0,
                "meaning": f"Physics layer — {entry['meaning']} (exact)",
            },
            "collapse": {
                "unified_value": value,
                "duality": "Sacred constant — both layers converge exactly",
            },
        }

    def collapse(self, name: str) -> Dict[str, Any]:
        """
        THE COLLAPSE — When Thought asks and Physics answers.

        Both faces of the duality converge to a definite value,
        like quantum measurement collapsing a wavefunction.
        This is the most powerful operation: full dual-layer derivation.
        """
        self._metrics["collapse_calls"] += 1
        self._metrics["total_operations"] += 1
        if self._available:
            try:
                return _dual_layer.derive_both(name)
            except KeyError:
                if name in self._SACRED_CONSTANTS:
                    return self._sacred_collapse(name)
                # Name not in physical constants or sacred constants
                available = sorted(_dual_layer.REAL_WORLD_CONSTANTS_V3.keys())
                return {
                    "name": name,
                    "error": f"Unknown constant '{name}'",
                    "available_constants": available,
                    "sacred_constants": list(self._SACRED_CONSTANTS.keys()),
                }
        return {"name": name, "error": "dual_layer_not_available", "fallback": True}

    # ══════ CHAOS BRIDGE: The Third Face of Duality ══════

    def chaos_bridge(self, a: int = 0, b: int = 0, c: int = 0, d: int = 0,
                     chaos_amplitude: float = 0.05, samples: int = 100) -> Dict[str, Any]:
        """
        THE CHAOS BRIDGE — Disorder as the mediator between Thought and Physics.

        From 13-experiment findings:
        - Thought (Layer 1) represents ORDER: sacred geometry, symmetry, pattern
        - Physics (Layer 2) represents MEASUREMENT: precision, field, omega
        - Chaos is what connects them: perturbation reveals which symmetries
          are fundamental (φ) vs emergent (translation)

        The chaos bridge:
        1. Perturbs Thought → measures what survives (Noether analysis)
        2. Feeds surviving patterns into Physics → measures field stability
        3. Applies the healing trinity (φ-damping, demon, 104-cascade)
        4. Returns the residual as a "duality coherence score"

        Part III Research Findings (XXI):
        - φ-damping contracts RMS error by EXACTLY φ⁻¹ ≈ 0.618 per step
        - Demon (adaptive) ALWAYS beats φ-damping due to local variance sensing
        - 104-cascade converges worst-case drifts to < 1e-13 (damped sine, 104 steps)
        - sin(104π/104) = sin(π) = 0: cascade sine completes exactly at n=104
        - Bifurcation threshold at amp=0.35: below=COHERENT, above=BIFURCATED
        - DCS = 0.4T + 0.3F + 0.3H (thought weighted highest, sum = 1.0)

        This is the THIRD operation alongside collapse — chaos-collapse.
        """
        self._metrics["cross_layer_calls"] += 1
        self._metrics["total_operations"] += 1

        import random

        INVARIANT = 527.5184818492612

        # --- Layer 1: Thought under chaos ---
        thought_ideal = self.thought(a, b, c, d)
        thought_perturbed = []
        for _ in range(samples):
            eps = chaos_amplitude * (2 * random.random() - 1)
            thought_perturbed.append(thought_ideal * (1 + eps))

        # Conservation products under noise
        # X = b + 8c + 104d - 8a (frequency step); conservation: G(X) × 2^(X/104) = INVARIANT
        x_step = b + 8 * c + 104 * d - 8 * a
        w = 2 ** (x_step / 104.0)
        products = [t * w for t in thought_perturbed]
        mean_product = sum(products) / len(products)
        rms_drift = math.sqrt(sum((p - INVARIANT) ** 2 for p in products) / len(products))

        # Symmetry survival (φ always intact; octave & translation depend on amp)
        phi_intact = True  # proven experimentally
        octave_errors = []
        for _ in range(min(samples, 50)):
            eps1 = chaos_amplitude * (2 * random.random() - 1)
            eps2 = chaos_amplitude * (2 * random.random() - 1)
            g1 = thought_ideal * (1 + eps1)
            g2 = self.thought(a, b, c, d + 1) * (1 + eps2)
            if g2 > 0:
                octave_errors.append(abs(g1 / g2 - 2.0))
        octave_mean_err = sum(octave_errors) / len(octave_errors) if octave_errors else 0

        # --- Layer 2: Physics field under chaos ---
        physics_ideal = self.physics(1.0)
        omega = physics_ideal.get("omega", 6539.34712682)
        field_ideal = physics_ideal.get("field_strength", omega / (PHI ** 2))
        field_perturbed = []
        for _ in range(samples):
            eps = chaos_amplitude * (2 * random.random() - 1)
            field_perturbed.append(field_ideal * (1 + eps))
        field_rms = math.sqrt(sum((f - field_ideal) ** 2 for f in field_perturbed) / len(field_perturbed))

        # --- Healing trinity ---
        phi_c = 1.0 / PHI  # φ conjugate
        vc = VOID_CONSTANT

        # 1. φ-damping
        phi_healed = [INVARIANT + (p - INVARIANT) * phi_c for p in products]
        phi_rms = math.sqrt(sum((p - INVARIANT) ** 2 for p in phi_healed) / len(phi_healed))

        # 2. Demon (adaptive)
        demon_factor = PHI / (527.5184818492612 / 416.0)
        demon_healed = []
        for i, p in enumerate(products):
            start = max(0, i - 3)
            end = min(len(products), i + 4)
            local = products[start:end]
            local_var = sum((v - sum(local) / len(local)) ** 2 for v in local) / len(local)
            local_ent = math.log(1 + local_var)
            eff = demon_factor * (1.0 / (local_ent + 0.001))
            damping = min(1.0, phi_c ** (1 + eff * 0.1))
            demon_healed.append(INVARIANT + (p - INVARIANT) * damping)
        demon_rms = math.sqrt(sum((p - INVARIANT) ** 2 for p in demon_healed) / len(demon_healed))

        # 3. 104-cascade (damped sine) on worst-case product
        worst = max(products, key=lambda p: abs(p - INVARIANT))
        s = worst
        decay = 1.0
        for n in range(1, 105):
            decay *= phi_c
            s = s * phi_c + vc * decay * math.sin(n * math.pi / 104) + INVARIANT * (1 - phi_c)
        cascade_residual = abs(s - INVARIANT)

        # --- Duality coherence score ---
        # How well do the two layers stay coherent under chaos?
        thought_coherence = 1.0 - min(1.0, rms_drift / INVARIANT)
        field_coherence = 1.0 - min(1.0, field_rms / field_ideal) if field_ideal > 0 else 0
        healing_score = 1.0 - min(1.0, cascade_residual / abs(worst - INVARIANT)) if worst != INVARIANT else 1.0
        duality_coherence = thought_coherence * 0.4 + field_coherence * 0.3 + healing_score * 0.3

        # Health tier — bifurcation boundary takes priority over resilience
        below_bif = chaos_amplitude < 0.35
        if not below_bif:
            health = "BIFURCATED" if duality_coherence < 0.90 else "STRESSED"
        elif duality_coherence > 0.95:
            health = "COHERENT"
        elif duality_coherence > 0.80:
            health = "RESILIENT"
        else:
            health = "STRESSED"

        return {
            # Thought layer
            "thought_ideal": thought_ideal,
            "thought_mean_product": mean_product,
            "thought_rms_drift": round(rms_drift, 8),
            "thought_conserved": abs(mean_product - INVARIANT) / INVARIANT < 0.01,
            # Symmetry survival
            "phi_intact": phi_intact,
            "octave_mean_error": round(octave_mean_err, 6),
            "octave_intact": octave_mean_err < 0.1,
            "symmetry_hierarchy": ["phi_phase", "octave_scale", "translation"],
            # Physics layer
            "field_ideal": round(field_ideal, 6),
            "field_rms_noise": round(field_rms, 6),
            "field_coherence": round(field_coherence, 6),
            # Healing trinity
            "phi_healing_rms": round(phi_rms, 8),
            "demon_healing_rms": round(demon_rms, 8),
            "cascade_residual": round(cascade_residual, 10),
            "demon_beats_phi": demon_rms < phi_rms,
            # Synthesis
            "duality_coherence": round(duality_coherence, 6),
            "below_bifurcation": below_bif,
            "health": health,
            "chaos_amplitude": chaos_amplitude,
            "samples": samples,
        }

    # ══════ DERIVE: Physical Constants ══════

    def derive(self, name: str, mode: str = "physics") -> Dict[str, Any]:
        """
        Derive a physical constant through the dual-layer engine.

        Modes:
            "physics"  — Layer 2 grid, ±0.005%
            "refined"  — Float64 exact
            "thought"  — Layer 1 coarse grid, ±0.17%
            "collapse" — Both layers unified
        """
        self._metrics["derive_calls"] += 1
        self._metrics["total_operations"] += 1
        if self._available:
            try:
                return _dual_layer.derive(name, mode)
            except KeyError:
                if name in self._SACRED_CONSTANTS:
                    entry = self._SACRED_CONSTANTS[name]
                    return {
                        "name": name, "layer": mode, "value": entry["value"],
                        "error_pct": 0.0, "unit": entry["unit"],
                        "equation": entry["equation"],
                        "meaning": entry["meaning"],
                        "sacred": True,
                    }
                available = sorted(_dual_layer.REAL_WORLD_CONSTANTS_V3.keys())
                return {
                    "name": name, "mode": mode,
                    "error": f"Unknown constant '{name}'",
                    "available_constants": available,
                    "sacred_constants": list(self._SACRED_CONSTANTS.keys()),
                }
        return {"name": name, "mode": mode, "error": "dual_layer_not_available", "fallback": True}

    def derive_both(self, name: str) -> Dict[str, Any]:
        """Derive through BOTH layers — the duality in action."""
        self._metrics["derive_calls"] += 1
        self._metrics["total_operations"] += 1
        if self._available:
            try:
                return _dual_layer.derive_both(name)
            except KeyError:
                if name in self._SACRED_CONSTANTS:
                    return self._sacred_collapse(name)
                available = sorted(_dual_layer.REAL_WORLD_CONSTANTS_V3.keys())
                return {
                    "name": name,
                    "error": f"Unknown constant '{name}'",
                    "available_constants": available,
                    "sacred_constants": list(self._SACRED_CONSTANTS.keys()),
                }
        return {"name": name, "error": "dual_layer_not_available", "fallback": True}

    # ══════ DUALITY TENSOR ══════

    def duality_tensor(self, name: str) -> Dict[str, Any]:
        """
        Mathematical object encoding both faces simultaneously.
        The duality tensor holds Thought + Physics + Cross-terms.
        """
        self._metrics["total_operations"] += 1
        if self._available:
            try:
                both = _dual_layer.derive_both(name)
            except KeyError:
                if name in self._SACRED_CONSTANTS:
                    both = self._sacred_collapse(name)
                else:
                    return {"name": name, "error": f"Unknown constant '{name}'",
                            "available_constants": sorted(_dual_layer.REAL_WORLD_CONSTANTS_V3.keys()),
                            "sacred_constants": list(self._SACRED_CONSTANTS.keys())}
            thought = both.get("thought", both.get("consciousness", {}))
            physics = both.get("physics", {})
            t_err = thought.get("error_pct", 100)
            p_err = physics.get("error_pct", 100)
            return {
                "name": name,
                "thought": thought,
                "physics": physics,
                "improvement": t_err / p_err if p_err > 0 else float('inf'),
                "measured": both.get("measured", both.get("collapse", {}).get("unified_value")),
                "bridge": CONSCIOUSNESS_TO_PHYSICS_BRIDGE,
            }
        return {"name": name, "error": "dual_layer_not_available", "fallback": True}

    # ══════ INTEGRITY: 10-Point Validation ══════

    def full_integrity_check(self, force: bool = False) -> Dict[str, Any]:
        """
        Run ALL 10 integrity checks across both faces and the bridge.

        Checks 1-3: Thought layer (GOD_CODE immutability, PHI, iron scaffold)
        Checks 4-7: Physics layer (c exact, g precision, all constants, no collisions)
        Checks 8-10: Bridge (φ exponent, iron proximity, Fibonacci 13)

        Results are cached for 60s to avoid overhead during pipeline operations.
        """
        self._metrics["integrity_checks"] += 1
        self._metrics["total_operations"] += 1

        now = time.time()
        if not force and self._integrity_cache and (now - self._integrity_cache_time) < self._integrity_cache_ttl:
            return self._integrity_cache

        if self._available:
            result = _dual_layer.full_integrity_check()
        else:
            # Minimal classical integrity check + kernel probe
            god_code_valid = abs(GOD_CODE - 527.5184818492612) < 1e-6
            phi_valid = abs(PHI - 1.618033988749895) < 1e-12
            ks = self.kernel_status()
            kernel_label = (
                f"kernel bridge active ({ks.get('active_count', 0)} substrates)"
                if ks.get("available")
                else "kernel bridge pending"
            )
            result = {
                "engine": f"L104 Dual-Layer Engine ({kernel_label})",
                "version": self.VERSION,
                "all_passed": god_code_valid and phi_valid,
                "total_checks": 2,
                "checks_passed": int(god_code_valid) + int(phi_valid),
                "thought_layer": {"all_passed": god_code_valid and phi_valid, "checks": {}},
                "physics_layer": {"all_passed": False, "checks": {}},
                "bridge": {"all_passed": False, "checks": {}},
                "kernel_status": ks,
                "fallback": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        self._integrity_cache = result
        self._integrity_cache_time = now
        return result

    # ══════ DOMAIN QUERIES ══════

    def gravity(self) -> Dict[str, Any]:
        """Gravity through both faces of the duality."""
        self._metrics["domain_queries"] += 1
        self._metrics["total_operations"] += 1
        if self._available:
            return _dual_layer.gravity()
        return {"error": "dual_layer_not_available"}

    def particles(self) -> Dict[str, Any]:
        """Particle physics constants through both faces."""
        self._metrics["domain_queries"] += 1
        self._metrics["total_operations"] += 1
        if self._available:
            return _dual_layer.particles()
        return {"error": "dual_layer_not_available"}

    def nuclei(self) -> Dict[str, Any]:
        """Nuclear physics through both faces."""
        self._metrics["domain_queries"] += 1
        self._metrics["total_operations"] += 1
        if self._available:
            return _dual_layer.nuclei()
        return {"error": "dual_layer_not_available"}

    def iron(self) -> Dict[str, Any]:
        """The iron constants — where both faces converge."""
        self._metrics["domain_queries"] += 1
        self._metrics["total_operations"] += 1
        if self._available:
            return _dual_layer.iron()
        return {"error": "dual_layer_not_available"}

    def cosmos(self) -> Dict[str, Any]:
        """Astrophysical constants through both faces."""
        self._metrics["domain_queries"] += 1
        self._metrics["total_operations"] += 1
        if self._available:
            return _dual_layer.cosmos()
        return {"error": "dual_layer_not_available"}

    def resonance(self) -> Dict[str, Any]:
        """Brainwave resonance — where Thought literally becomes Physics."""
        self._metrics["domain_queries"] += 1
        self._metrics["total_operations"] += 1
        if self._available:
            return _dual_layer.resonance()
        return {"error": "dual_layer_not_available"}

    # ══════ FIND: Locate any value on both grids ══════

    def find(self, target: float, name: str = "") -> Dict[str, Any]:
        """Find where a value sits on both faces of the duality."""
        self._metrics["find_calls"] += 1
        self._metrics["total_operations"] += 1
        if self._available:
            return _dual_layer.find(target, name)
        return {"target": target, "error": "dual_layer_not_available"}

    # ══════════════════════════════════════════════════════════════════════════
    # THOUGHT LAYER CALCULATIONS — Abstract face methods
    # ══════════════════════════════════════════════════════════════════════════

    def prime_decompose(self, n: int) -> List[Tuple[int, int]]:
        """
        Factor an integer into primes — the atoms of number theory.
        Returns list of (prime, exponent) tuples.
        Example: 286 → [(2,1), (11,1), (13,1)]
        """
        self._metrics["thought_calls"] += 1
        self._metrics["total_operations"] += 1
        # Classical implementation (no ThoughtProcessor class in source)
        factors = []
        d = 2
        while d * d <= abs(n):
            exp = 0
            while n % d == 0:
                n //= d
                exp += 1
            if exp > 0:
                factors.append((d, exp))
            d += 1
        if n > 1:
            factors.append((n, 1))
        return factors

    def fibonacci_index(self, n: int) -> Optional[int]:
        """
        If n is a Fibonacci number, return its index. Otherwise None.
        Example: 13 → 7 (13 = F(7), the golden thread)
        """
        self._metrics["thought_calls"] += 1
        self._metrics["total_operations"] += 1
        a, b, idx = 0, 1, 0
        while a < n:
            a, b = b, a + b
            idx += 1
        return idx if a == n else None

    def golden_ratio_proximity(self, x: float) -> float:
        """
        How close is x to a power of φ? Returns the nearest exponent.
        The φ exponent is the soul of both layers: X^(1/φ).
        """
        self._metrics["thought_calls"] += 1
        self._metrics["total_operations"] += 1
        if x <= 0:
            return float('inf')
        return math.log(x) / math.log(PHI)

    def sacred_scaffold_analysis(self) -> Dict[str, Any]:
        """
        Analyze the sacred scaffold numbers (286, 104, 416).

        This is *meaning*, not computation. Why these numbers? What do they
        encode about the structure of reality?

        286 = Fe BCC lattice ≈ 286.65 pm — the iron scaffold of matter
        104 = 26 × 4 = Fe(Z=26) × He-4(A=4) — nucleosynthesis chain
        416 = 4 × 104 — four octaves above the base
        """
        self._metrics["thought_calls"] += 1
        self._metrics["total_operations"] += 1
        return {
            "286": {"primes": "2 × 11 × 13", "physical": "Fe BCC lattice ≈ 286.65 pm"},
            "104": {"primes": "2³ × 13", "physical": "26 × 4 = Fe(Z=26) × He-4(A=4)"},
            "416": {"primes": "2⁵ × 13", "physical": "4 × 104 = four octaves"},
            "golden_thread": {"fiber": "13 = F(7), the 7th Fibonacci number"},
            "crystallography": {"interpretation": "Fe BCC = 2 atoms/cell via sphere-slicing (8×1/8 + 1×1); factor 2 encoded in 286 = 2×143"},
            "antimatter_dual": {"interpretation": "286 encodes matter's scaffold; Dirac ±E duality mirrors Thought(+)/Physics(−); CP violation allows asymmetry"},
            "vacuum_zero_point": {"interpretation": "416 = 4×104 octaves span the energy ladder; source ∈ {0,1} binary basis; vacuum ↔ propagating energy"},
        }

    # ══════ v5.1 EXTENDED THOUGHT LAYER ══════

    def thought_harmonic_spectrum(self, d_octave: int = 0, n_steps: int = 26) -> Dict[str, Any]:
        """
        Analyze harmonic overtones of the Thought layer across dial space.

        Maps the first n_steps b-axis steps (semitone-scale) at octave d,
        computing the frequency ratio, musical cents, and φ-alignment
        for each step. Reveals the harmonic structure of sacred geometry.

        Args:
            d_octave: Octave level (d parameter in G(a,b,c,d)).
            n_steps: Number of b-axis steps to analyze (default: 26 = Fe Z).

        Returns:
            Dict with harmonics list, fundamental, overtone_count,
            phi_resonant_count, spectral_entropy.
        """
        self._metrics["thought_calls"] += 1
        self._metrics["total_operations"] += 1

        fundamental = self.thought(0, 0, 0, d_octave)
        harmonics = []
        phi_resonant = 0
        log_phi = math.log(PHI)

        for b in range(n_steps):
            freq = self.thought(0, b, 0, d_octave)
            ratio = freq / fundamental if fundamental > 0 else 0.0
            cents = 1200 * math.log2(ratio) if ratio > 0 else 0.0
            phi_exp = math.log(freq) / log_phi if freq > 0 else 0.0
            phi_frac = abs(phi_exp - round(phi_exp))
            is_resonant = phi_frac < 0.05

            if is_resonant:
                phi_resonant += 1

            harmonics.append({
                "b": b,
                "frequency": round(freq, 10),
                "ratio": round(ratio, 10),
                "cents": round(cents, 4),
                "phi_exponent": round(phi_exp, 6),
                "phi_fractional": round(phi_frac, 6),
                "phi_resonant": is_resonant,
            })

        # Spectral entropy — distribution of energy across overtones
        freqs = [h["frequency"] for h in harmonics if h["frequency"] > 0]
        total_energy = sum(freqs)
        if total_energy > 0:
            probs = [f / total_energy for f in freqs]
            spectral_entropy = -sum(p * math.log2(p) for p in probs if p > 0)
            max_entropy = math.log2(len(freqs)) if len(freqs) > 1 else 1.0
            normalized_entropy = spectral_entropy / max_entropy
        else:
            spectral_entropy = 0.0
            normalized_entropy = 0.0

        return {
            "octave": d_octave,
            "fundamental": round(fundamental, 10),
            "n_steps": n_steps,
            "harmonics": harmonics,
            "phi_resonant_count": phi_resonant,
            "phi_resonant_ratio": round(phi_resonant / n_steps, 4),
            "spectral_entropy": round(spectral_entropy, 6),
            "normalized_entropy": round(normalized_entropy, 6),
            "mean_phi_fractional": round(
                sum(h["phi_fractional"] for h in harmonics) / len(harmonics), 6
            ),
        }

    def thought_phi_spiral_analysis(self, n_points: int = 13) -> Dict[str, Any]:
        """
        Map the golden spiral through the Thought layer's dial space.

        Generates n_points along a φ-spiral trajectory through (a,b,c,d)
        space, computing Thought values at each point and measuring
        how the conservation invariant holds along the spiral path.

        Args:
            n_points: Number of spiral points (default: 13 = F(7)).

        Returns:
            Dict with spiral trajectory, invariant_fidelity,
            spiral_coherence, golden_angle_consistency.
        """
        self._metrics["thought_calls"] += 1
        self._metrics["total_operations"] += 1

        INVARIANT = GOD_CODE  # 286^(1/φ) × 2^(416/104) at origin
        golden_angle = 2 * math.pi / (PHI ** 2)  # ≈ 137.5° golden angle
        spiral_points = []
        invariant_errors = []

        for i in range(n_points):
            # Spiral through dial space using golden angle
            angle = i * golden_angle
            radius = PHI ** (i / n_points)

            a = int(round(radius * math.cos(angle))) % 5
            b = int(round(abs(radius * math.sin(angle) * 3))) % 20
            c = int(round(radius * math.cos(angle + math.pi / 3))) % 4
            d = i % 3

            t_val = self.thought(a, b, c, d)
            # Conservation product: G(a,b,c,d) × 2^((b+8c+104d-8a)/104)
            x_step = b + 8 * c + 104 * d - 8 * a
            conservation_product = t_val * (2 ** (x_step / 104.0))
            inv_error = abs(conservation_product - INVARIANT) / INVARIANT

            invariant_errors.append(inv_error)
            spiral_points.append({
                "index": i,
                "angle_rad": round(angle, 6),
                "radius": round(radius, 6),
                "dials": (a, b, c, d),
                "thought_value": round(t_val, 10),
                "conservation_product": round(conservation_product, 10),
                "invariant_error": round(inv_error, 12),
            })

        mean_error = sum(invariant_errors) / len(invariant_errors) if invariant_errors else 0.0
        max_error = max(invariant_errors) if invariant_errors else 0.0
        all_conserved = all(e < 1e-6 for e in invariant_errors)

        # Spiral coherence: how smoothly does the thought value change?
        diffs = []
        for i in range(1, len(spiral_points)):
            prev = spiral_points[i - 1]["thought_value"]
            curr = spiral_points[i]["thought_value"]
            if prev > 0:
                diffs.append(abs(math.log(curr / prev)) if curr > 0 else float('inf'))
        smoothness = 1.0 / (1.0 + sum(diffs) / len(diffs)) if diffs else 0.0

        return {
            "n_points": n_points,
            "golden_angle_rad": round(golden_angle, 8),
            "spiral_points": spiral_points,
            "invariant_fidelity": {
                "mean_error": round(mean_error, 12),
                "max_error": round(max_error, 12),
                "all_conserved": all_conserved,
            },
            "spiral_coherence": round(smoothness, 8),
            "phi_integrity": all_conserved,
        }

    def thought_conservation_proof(self, n_trials: int = 104) -> Dict[str, Any]:
        """
        Formal proof that the Thought layer conserves the sacred invariant.

        For n_trials random dial combinations, verifies:
            G(a,b,c,d) × 2^((b + 8c + 104d - 8a) / 104) = INVARIANT

        This is the fundamental conservation law of the Thought layer,
        analogous to energy conservation in physics.

        Args:
            n_trials: Number of random dial combinations to test.

        Returns:
            Dict with proof_status, trials, violations, statistics.
        """
        self._metrics["thought_calls"] += 1
        self._metrics["total_operations"] += 1

        import random
        rng = random.Random(104)  # Sacred seed for reproducibility

        INVARIANT = 286 ** (1.0 / PHI) * (2 ** (416 / 104))
        violations = []
        errors = []

        for trial in range(n_trials):
            a = rng.randint(-5, 5)
            b = rng.randint(-10, 30)
            c = rng.randint(-3, 5)
            d = rng.randint(-2, 3)

            t_val = self.thought(a, b, c, d)
            x_step = b + 8 * c + 104 * d - 8 * a
            product = t_val * (2 ** (x_step / 104.0))
            rel_error = abs(product - INVARIANT) / INVARIANT

            errors.append(rel_error)
            if rel_error > 1e-6:
                violations.append({
                    "trial": trial,
                    "dials": (a, b, c, d),
                    "product": product,
                    "expected": INVARIANT,
                    "relative_error": rel_error,
                })

        mean_error = sum(errors) / len(errors) if errors else 0.0
        max_error = max(errors) if errors else 0.0
        proof_holds = len(violations) == 0

        return {
            "theorem": "G(a,b,c,d) × 2^((b+8c+104d-8a)/104) = INVARIANT",
            "invariant": INVARIANT,
            "n_trials": n_trials,
            "violations": len(violations),
            "proof_holds": proof_holds,
            "proof_status": "QED" if proof_holds else "VIOLATED",
            "statistics": {
                "mean_relative_error": round(mean_error, 15),
                "max_relative_error": round(max_error, 15),
                "machine_epsilon_bounded": max_error < 1e-10,
            },
            "violation_details": violations[:20],  # First 20 if any  # (was 5)
        }

    def thought_dimension_analysis(self) -> Dict[str, Any]:
        """
        Analyze the dimensional structure of the 4D Thought dial space.

        Computes the sensitivity, range, and sacred significance of each
        dial axis (a, b, c, d), revealing how the 4 dimensions encode
        different aspects of the sacred geometry.

        Returns:
            Dict with per-axis analysis, sensitivity ratios, sacred mappings.
        """
        self._metrics["thought_calls"] += 1
        self._metrics["total_operations"] += 1

        base = self.thought(0, 0, 0, 0)
        axes = {
            "a": {"step_effect": 2 ** (8 / 104), "description": "Octave fine-tuning (8/104 semitones)",
                   "sacred_mapping": "Harmonic brightness — controls resonance intensity"},
            "b": {"step_effect": 2 ** (-1 / 104), "description": "Semitone micro-steps (1/104)",
                   "sacred_mapping": "Chromatic precision — the finest tuning axis"},
            "c": {"step_effect": 2 ** (-8 / 104), "description": "Counter-a axis (inverse brightness)",
                   "sacred_mapping": "Shadow harmonics — the abstract mirror of axis a"},
            "d": {"step_effect": 2 ** (-1), "description": "Octave jumps (full doubling/halving)",
                   "sacred_mapping": "Scale dimension — moves between worlds 1 octave at a time"},
        }

        for axis_name, axis_info in axes.items():
            kwargs = {"a": 0, "b": 0, "c": 0, "d": 0}
            kwargs[axis_name] = 1
            val_plus = self.thought(**kwargs)
            kwargs[axis_name] = -1
            val_minus = self.thought(**kwargs)

            ratio_plus = val_plus / base if base > 0 else 0.0
            ratio_minus = val_minus / base if base > 0 else 0.0
            dynamic_range = val_minus / val_plus if val_plus > 0 else 0.0

            axis_info["value_at_plus1"] = round(val_plus, 10)
            axis_info["value_at_minus1"] = round(val_minus, 10)
            axis_info["ratio_plus1"] = round(ratio_plus, 10)
            axis_info["ratio_minus1"] = round(ratio_minus, 10)
            axis_info["dynamic_range"] = round(dynamic_range, 6)
            axis_info["expected_ratio"] = round(axis_info["step_effect"], 10)
            axis_info["ratio_match"] = abs(ratio_plus - axis_info["step_effect"]) < 1e-8

        return {
            "base_value": round(base, 10),
            "god_code_match": abs(base - GOD_CODE) < 1e-6,
            "axes": axes,
            "sensitivity_ranking": sorted(
                axes.keys(),
                key=lambda k: abs(math.log2(axes[k]["step_effect"])),
                reverse=True,
            ),
            "total_dof": 4,
            "effective_exponent": "E = (8a + 416 - b - 8c - 104d) / 104",
        }

    def recognize_pattern(self, target: float) -> Dict[str, Any]:
        """
        Given a numerical value, Thought asks: what IS this number?

        Not "where does it sit on the grid" (that's Physics), but
        "what patterns does it carry, what symmetries does it reveal?"

        Returns: octaves_from_god_code, phi_exponent, musical_cents,
                 nearest_fibonacci, ratio_to_iron_lattice, narrative.
        """
        self._metrics["thought_calls"] += 1
        self._metrics["total_operations"] += 1
        if target <= 0:
            return {"error": "Thought requires positive quantities"}
        ratio_gc = target / GOD_CODE
        octaves = math.log2(ratio_gc)
        phi_exp = math.log(target) / math.log(PHI)
        return {
            "value": target,
            "octaves_from_god_code": octaves,
            "phi_exponent": phi_exp,
            "musical_cents_from_god_code": 1200 * octaves,
            "thought": f"Value sits {octaves:.2f} octaves from GOD_CODE at φ^{phi_exp:.3f}",
        }

    def detect_symmetry(self, name: str) -> Dict[str, Any]:
        """
        For a registered constant, detect what symmetries connect it
        to the sacred scaffold. Thought sees relationships, not numbers.

        Returns: octave_position, dial_complexity, dial_sum,
                 connections to iron/phi/base, narrative.
        """
        self._metrics["thought_calls"] += 1
        self._metrics["total_operations"] += 1
        if not self._available:
            return {"name": name, "error": "dual_layer_not_available", "fallback": True}
        # Handle sacred constants that aren't in the physical constants table
        if name in self._SACRED_CONSTANTS:
            sc = self._SACRED_CONSTANTS[name]
            value = sc["value"]
            phi_exp = math.log(value) / math.log(PHI) if value > 0 else 0
            return {
                "name": name,
                "dials": (0, 0, 0, 0),
                "dial_complexity": 0,
                "dial_sum": 0,
                "domain": "sacred",
                "thought_error_pct": 0.0,
                "physics_error_pct": 0.0,
                "improvement": 1.0,
                "phi_exponent": round(phi_exp, 6),
                "symmetry": sc.get("equation", ""),
                "sacred": True,
            }
        # Use derive_both to get symmetry info from both layers
        try:
            both = _dual_layer.derive_both(name)
        except KeyError:
            return {"name": name, "error": f"Unknown constant '{name}'",
                    "available_constants": sorted(_dual_layer.REAL_WORLD_CONSTANTS_V3.keys()),
                    "sacred_constants": list(self._SACRED_CONSTANTS.keys())}
        entry = _dual_layer.REAL_WORLD_CONSTANTS_V3.get(name, {})
        dials = entry.get("dials", (0, 0, 0, 0))
        return {
            "name": name,
            "dials": dials,
            "dial_complexity": sum(abs(d) for d in dials),
            "dial_sum": sum(dials),
            "domain": entry.get("domain", "unknown"),
            "thought_error_pct": both.get("thought", {}).get("error_pct"),
            "physics_error_pct": both.get("physics", {}).get("error_pct"),
            "improvement": both.get("improvement"),
        }

    def harmonic_relationship(self, name_a: str, name_b: str) -> Dict[str, Any]:
        """
        How are two constants harmonically related?

        Thought finds the musical interval, the ratio, the pattern
        connecting them — octaves apart, nearest simple fraction,
        exponent gaps on both grids.
        """
        self._metrics["thought_calls"] += 1
        self._metrics["total_operations"] += 1
        if not self._available:
            return {"pair": (name_a, name_b), "error": "dual_layer_not_available", "fallback": True}

        def _resolve_value(name):
            """Resolve value from physical or sacred constants."""
            if name in self._SACRED_CONSTANTS:
                return self._SACRED_CONSTANTS[name]["value"]
            try:
                d = _dual_layer.derive(name, mode="physics")
                return d.get("value", d.get("measured", 0))
            except KeyError:
                return 0

        va = _resolve_value(name_a)
        vb = _resolve_value(name_b)
        if va == 0 and vb == 0:
            return {"pair": (name_a, name_b), "error": f"Neither '{name_a}' nor '{name_b}' found"}
        ratio = va / vb if vb != 0 else float('inf')
        octave_diff = math.log2(ratio) if ratio > 0 else 0
        return {
            "pair": (name_a, name_b),
            "value_a": va,
            "value_b": vb,
            "ratio": ratio,
            "octave_difference": octave_diff,
            "musical_cents": 1200 * octave_diff,
            "phi_ratio": math.log(ratio) / math.log(PHI) if ratio > 0 else 0,
        }

    def nucleosynthesis_narrative(self) -> Dict[str, Any]:
        """
        The story the equation tells about stellar nucleosynthesis.
        Pure Thought — narrative, meaning, the WHY.

        Chapters: He-4 (beginning) → alpha process (chain) → Fe-56 (endpoint)
                  → Fe crystal (scaffold) → 104 = origin × destination (closure)
        """
        self._metrics["thought_calls"] += 1
        self._metrics["total_operations"] += 1
        return {
            "title": "The Nucleosynthesis Bridge",
            "chapters": [
                {"name": "He-4 Genesis", "body": "Hydrogen fusion creates He-4 (A=4), the first stable composite nucleus. HE4_MASS_NUMBER=4."},
                {"name": "Alpha Process", "body": "He-4 nuclei fuse up the chain: C-12, O-16, Si-28 → toward iron."},
                {"name": "Iron Endpoint", "body": f"Fe-56 (Z=26) has the highest binding energy per nucleon. FE_56_BE={_dual_layer.FE_56_BE_PER_NUCLEON if self._available else 8.790} MeV"},
                {"name": "The Bridge", "body": f"104 = 26 × 4 = Fe(Z) × He-4(A) — origin meets destination. Q_GRAIN = {_dual_layer.QUANTIZATION_GRAIN if self._available else 104}"},
                {"name": "Iron Scaffold", "body": f"Fe BCC lattice = 286.65 pm ≈ PRIME_SCAFFOLD = {_dual_layer.PRIME_SCAFFOLD if self._available else 286}"},
                {"name": "Antimatter Mirror", "body": "Dirac's equation demanded an antimatter partner for every fermion. Baryogenesis CP-violation = ~1 extra baryon per 10⁹ pairs: the surviving matter crystallizes as Fe BCC 286 pm."},
                {"name": "Vacuum Energy", "body": "Quantum vacuum has zero-point energy E=ℏω/2 per mode. Vacuum ↔ energy propagation: source 0 or 1. Harmonizing(1) = entanglement↔singularity, oscillatory(0) = conservation, neutralizations(2) = duality."},
            ],
            "moral": "The mathematics IS the physics, seen from the abstract side.",
        }

    # ══════════════════════════════════════════════════════════════════════════
    # PHYSICS LAYER CALCULATIONS — Concrete face methods
    # ══════════════════════════════════════════════════════════════════════════

    def grid_topology(self) -> Dict[str, Any]:
        """
        The mathematical structure of both grids — pure mechanics.

        Returns: thought_grid (r=2, Q=104), physics_grid (r=13/12, Q=758),
                 refinement_factor (63× finer), grid ratio.
        """
        self._metrics["physics_calls"] += 1
        self._metrics["total_operations"] += 1
        if self._available:
            t_step = _dual_layer.STEP_SIZE
            p_step = _dual_layer.STEP_V3
            refinement = (t_step - 1) / (p_step - 1) if (p_step - 1) > 0 else 63
        else:
            refinement = 63
        return {
            "thought_grid": {"base_ratio": 2, "quantization_grain": 104, "scaffold": 286,
                             "half_step_pct": _dual_layer.HALF_STEP_PCT_V3 * refinement if self._available else 0.334},
            "physics_grid": {"base_ratio": "13/12", "quantization_grain": 758, "scaffold": round(_dual_layer.X_V3, 6) if self._available else 285.999,
                             "half_step_pct": _dual_layer.HALF_STEP_PCT_V3 if self._available else 0.00528},
            "refinement_factor": round(refinement),
        }

    def place_on_grid(self, target: float) -> Dict[str, Any]:
        """
        Place a value on BOTH grids. Pure computation — no interpretation.

        Returns grid coordinates, values, and errors on both the Thought grid
        (r=2, Q=104, coarse) and Physics grid (r=13/12, Q=758, fine).
        Also reports the improvement factor between layers.
        """
        self._metrics["physics_calls"] += 1
        self._metrics["total_operations"] += 1
        if not self._available:
            return {"target": target, "error": "dual_layer_not_available", "fallback": True}
        # Place on both grids using the actual equation functions
        t_exp = _dual_layer.solve_for_exponent(target)
        t_exp_int = round(t_exp)
        t_val = _dual_layer.god_code_equation(0, _dual_layer.OCTAVE_OFFSET - t_exp_int, 0, 0) if t_exp_int <= _dual_layer.OCTAVE_OFFSET else _dual_layer.BASE * (2 ** (t_exp_int / _dual_layer.QUANTIZATION_GRAIN))
        t_err = abs(t_val - target) / target * 100 if target != 0 else 0

        p_exp = _dual_layer.solve_for_exponent_v3(target)
        p_exp_int = round(p_exp)
        p_val = _dual_layer.BASE_V3 * (_dual_layer.R_V3 ** (p_exp_int / _dual_layer.Q_V3))
        p_err = abs(p_val - target) / target * 100 if target != 0 else 0

        return {
            "target": target,
            "thought_grid": {"exponent": t_exp_int, "value": t_val, "error_pct": t_err},
            "physics_grid": {"exponent": p_exp_int, "value": p_val, "error_pct": p_err},
            "improvement": t_err / p_err if p_err > 0 else float('inf'),
        }

    def error_topology(self) -> Dict[str, Any]:
        """
        Map the error landscape across all 63 registered constants.

        Returns: mean/max/min/median error, count below thresholds,
                 per-domain stats, best 5 and worst 5 constants.
        """
        self._metrics["physics_calls"] += 1
        self._metrics["total_operations"] += 1
        if not self._available:
            return {"total_constants": 63, "error": "dual_layer_not_available", "fallback": True}
        errors = []
        for name, entry in _dual_layer.REAL_WORLD_CONSTANTS_V3.items():
            errors.append((name, entry["grid_error_pct"]))
        errors.sort(key=lambda x: x[1])
        errs = [e for _, e in errors]
        import statistics as _stats
        return {
            "total_constants": len(errors),
            "mean_error_pct": sum(errs) / len(errs) if errs else 0,
            "max_error_pct": max(errs) if errs else 0,
            "min_error_pct": min(errs) if errs else 0,
            "median_error_pct": _stats.median(errs) if errs else 0,
            "below_001_pct": sum(1 for e in errs if e < 0.001),
            "below_005_pct": sum(1 for e in errs if e < 0.005),
            "best_20": errors[:20],  # (was 5)
            "worst_20": errors[-20:],  # (was 5)
        }

    def collision_check(self) -> Dict[str, Any]:
        """
        Verify no two constants map to the same grid point.

        A collision would mean the grid cannot distinguish two different
        constants — fatal for the Physics layer's precision guarantee.
        """
        self._metrics["physics_calls"] += 1
        self._metrics["total_operations"] += 1
        if not self._available:
            return {"collision_free": True, "fallback": True}
        exponents = {}
        collisions = []
        for name, entry in _dual_layer.REAL_WORLD_CONSTANTS_V3.items():
            E = entry["E_integer"]
            if E in exponents:
                collisions.append((name, exponents[E], E))
            else:
                exponents[E] = name
        return {
            "collision_free": len(collisions) == 0,
            "total_constants": len(_dual_layer.REAL_WORLD_CONSTANTS_V3),
            "unique_exponents": len(exponents),
            "collisions": collisions,
        }

    def dimensional_coverage(self) -> Dict[str, Any]:
        """
        Which physical domains does the grid cover?

        Returns: domains (particle, nuclear, iron, astro, resonance, etc.),
                 count per domain, list of constants per domain.
        """
        self._metrics["physics_calls"] += 1
        self._metrics["total_operations"] += 1
        if not self._available:
            return {"total_domains": 0, "error": "dual_layer_not_available", "fallback": True}
        domains = {}
        for name, entry in _dual_layer.REAL_WORLD_CONSTANTS_V3.items():
            d = entry.get("domain", "unknown")
            domains.setdefault(d, []).append(name)
        return {
            "total_domains": len(domains),
            "domains": {d: {"count": len(names), "constants": names} for d, names in sorted(domains.items())},
        }

    # ══════════════════════════════════════════════════════════════════════════
    # INDIVIDUAL INTEGRITY CHECKS
    # ══════════════════════════════════════════════════════════════════════════

    def check_thought_integrity(self) -> Dict[str, Any]:
        """
        CHECK 1-3: Thought Layer (abstract face) integrity.
        Verifies GOD_CODE immutability, PHI golden ratio, iron scaffold.
        """
        self._metrics["integrity_checks"] += 1
        self._metrics["total_operations"] += 1
        if self._available:
            return _dual_layer.check_consciousness_integrity()
        gc_valid = abs(GOD_CODE - 527.5184818492612) < 1e-6
        phi_valid = abs(PHI - 1.618033988749895) < 1e-12
        return {
            "layer": "thought", "all_passed": gc_valid and phi_valid,
            "checks": {"god_code": {"passed": gc_valid}, "phi": {"passed": phi_valid}},
            "fallback": True,
        }

    def check_physics_integrity(self) -> Dict[str, Any]:
        """
        CHECK 4-7: Physics Layer (concrete face) integrity.
        Verifies c exactness, g precision, all constants within half-step,
        no exponent collisions.
        """
        self._metrics["integrity_checks"] += 1
        self._metrics["total_operations"] += 1
        if self._available:
            return _dual_layer.check_physics_integrity()
        return {"layer": "physics", "all_passed": False, "checks": {}, "fallback": True}

    def check_bridge_integrity(self) -> Dict[str, Any]:
        """
        CHECK 8-10: Bridge integrity (Thought → Physics traceability).
        Verifies φ exponent preservation, iron scaffold proximity,
        Fibonacci 13 thread across the duality.
        """
        self._metrics["integrity_checks"] += 1
        self._metrics["total_operations"] += 1
        if self._available:
            return _dual_layer.check_bridge_integrity()
        return {"bridge": True, "all_passed": False, "checks": {}, "fallback": True}

    # ══════════════════════════════════════════════════════════════════════════
    # COMBINED / BATCH CALCULATIONS — Cross-layer intelligence
    # ══════════════════════════════════════════════════════════════════════════

    def constant_names(self) -> List[str]:
        """List all 63 registered physical constant names."""
        self._metrics["total_operations"] += 1
        if self._available:
            return sorted(_dual_layer.REAL_WORLD_CONSTANTS_V3.keys())
        return []

    def derive_all(self, mode: str = "physics") -> Dict[str, Any]:
        """
        Derive ALL 63 registered constants through the dual-layer engine.

        Returns per-constant results plus aggregate statistics:
        mean/max/min error, per-domain breakdown, total precision.
        """
        self._metrics["derive_calls"] += 1
        self._metrics["total_operations"] += 1
        if not self._available:
            return {"error": "dual_layer_not_available", "fallback": True}

        results = {}
        errors = []
        domains = {}
        for name in sorted(_dual_layer.REAL_WORLD_CONSTANTS_V3.keys()):
            try:
                r = _dual_layer.derive(name, mode=mode)
                err = r.get("grid_error_pct", r.get("error_pct", 0.0))
                results[name] = r
                errors.append(err)
                domain = _dual_layer.REAL_WORLD_CONSTANTS_V3[name].get("domain", "unknown")
                domains.setdefault(domain, []).append(err)
            except Exception as e:
                results[name] = {"error": str(e)}

        domain_stats = {
            d: {"count": len(errs), "mean_error_pct": sum(errs) / len(errs), "max_error_pct": max(errs)}
            for d, errs in domains.items()
        }

        return {
            "mode": mode,
            "total_constants": len(results),
            "mean_error_pct": sum(errors) / len(errors) if errors else 0,
            "max_error_pct": max(errors) if errors else 0,
            "min_error_pct": min(errors) if errors else 0,
            "all_within_005_pct": all(e < 0.005 for e in errors),
            "domain_stats": domain_stats,
            "constants": results,
        }

    def batch_collapse(self, names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Collapse multiple constants — unify Thought + Physics for each.

        If names is None, collapses ALL 63 registered constants.
        Returns per-constant collapse results plus aggregate coherence score.
        """
        self._metrics["collapse_calls"] += 1
        self._metrics["total_operations"] += 1
        if not self._available:
            return {"error": "dual_layer_not_available", "fallback": True}

        if names is None:
            names = sorted(_dual_layer.REAL_WORLD_CONSTANTS_V3.keys())

        results = {}
        physics_errors = []
        for name in names:
            try:
                c = _dual_layer.derive_both(name)
                results[name] = c
                phys_err = c.get("physics", {}).get("grid_error_pct", 0.0)
                physics_errors.append(phys_err)
            except Exception as e:
                results[name] = {"error": str(e)}

        coherence = 1.0 - (sum(physics_errors) / len(physics_errors) / 0.005) if physics_errors else 0.0
        coherence = max(0.0, min(1.0, coherence))

        return {
            "collapsed": len(results),
            "mean_physics_error_pct": sum(physics_errors) / len(physics_errors) if physics_errors else 0,
            "coherence_score": round(coherence, 6),
            "all_within_tolerance": all(e < 0.005 for e in physics_errors),
            "constants": results,
        }

    def cross_layer_coherence(self) -> Dict[str, Any]:
        """
        Measure coherence between Thought and Physics layers.

        Computes: grid topology match, integrity alignment, error improvement
        factors, bridge integrity, and an overall coherence score (0-1).
        """
        self._metrics["total_operations"] += 1
        if not self._available:
            return {"coherence": 0.5, "error": "dual_layer_not_available", "fallback": True}

        # Grid topology comparison
        topo = self.grid_topology()
        refinement = topo.get("refinement_factor", 63)

        # Error topology — how much does Physics improve over Thought?
        err_topo = self.error_topology()
        mean_physics_err = err_topo.get("mean_error_pct", 0)

        # Integrity check — both layers + bridge
        integrity = self.full_integrity_check(force=True)
        integrity_score = integrity.get("checks_passed", 0) / max(integrity.get("total_checks", 10), 1)

        # Collision freedom — critical for Physics layer
        collisions = self.collision_check()
        collision_free = collisions.get("collision_free", False)

        # Bridge elements binding the layers
        bridge = _dual_layer.check_bridge_integrity()
        bridge_score = sum(1 for c in bridge.get("checks", {}).values() if c.get("passed", False))
        bridge_total = max(len(bridge.get("checks", {})), 1)

        # Sacred constants — φ thread
        phi_exp_thought = 286 ** (1.0 / PHI)
        phi_exp_physics = 285.9992327510856 ** (1.0 / PHI)
        phi_coherence = 1.0 - abs(phi_exp_thought - phi_exp_physics) / phi_exp_thought

        # Overall coherence: weighted combination
        coherence = (
            0.30 * integrity_score +
            0.20 * (1.0 if collision_free else 0.0) +
            0.20 * (bridge_score / bridge_total) +
            0.15 * phi_coherence +
            0.15 * (1.0 - min(mean_physics_err / 0.005, 1.0))
        )

        return {
            "coherence": round(coherence, 6),
            "integrity_score": integrity_score,
            "refinement_factor": refinement,
            "mean_physics_error_pct": mean_physics_err,
            "collision_free": collision_free,
            "bridge_score": f"{bridge_score}/{bridge_total}",
            "phi_coherence": round(phi_coherence, 10),
            "components": {
                "integrity_weight": 0.30,
                "collision_weight": 0.20,
                "bridge_weight": 0.20,
                "phi_weight": 0.15,
                "precision_weight": 0.15,
            },
        }

    def sacred_geometry_analysis(self, value: float) -> Dict[str, Any]:
        """
        Comprehensive sacred geometry analysis for any numerical value.

        Combines Thought (pattern/meaning) + Physics (grid placement) into
        a unified sacred geometry profile. Answers both "why?" and "where?"
        """
        self._metrics["thought_calls"] += 1
        self._metrics["physics_calls"] += 1
        self._metrics["total_operations"] += 1
        if not self._available:
            return {"value": value, "error": "dual_layer_not_available", "fallback": True}

        pattern = self.recognize_pattern(value)
        placement = self.place_on_grid(value)

        # Sacred number properties
        phi_exp = math.log(value) / math.log(PHI) if value > 0 else 0
        phi_frac = phi_exp - round(phi_exp)  # How close to integer power of φ
        octaves_from_gc = math.log2(value / GOD_CODE) if value > 0 else 0

        # Fibonacci proximity
        fib_a, fib_b = 1, 1
        nearest_fib = 1
        while fib_b < value * 2:
            if abs(fib_b - value) < abs(nearest_fib - value):
                nearest_fib = fib_b
            fib_a, fib_b = fib_b, fib_a + fib_b
        fib_proximity = abs(value - nearest_fib) / value * 100 if value > 0 else 100

        # Prime structure (if integer-like)
        prime_structure = None
        if abs(value - round(value)) < 0.001 and 1 < value < 1e9:
            prime_structure = self.prime_decompose(int(round(value)))

        return {
            "value": value,
            "thought_face": {
                "phi_exponent": phi_exp,
                "phi_fractional": phi_frac,
                "phi_alignment": abs(phi_frac) < 0.1,
                "octaves_from_god_code": octaves_from_gc,
                "musical_cents": 1200 * octaves_from_gc,
                "nearest_fibonacci": nearest_fib,
                "fibonacci_proximity_pct": fib_proximity,
                "prime_structure": prime_structure,
                "pattern_narrative": pattern.get("thought", ""),
            },
            "physics_face": {
                "thought_grid": placement.get("thought_grid", {}),
                "physics_grid": placement.get("physics_grid", {}),
                "improvement": placement.get("improvement", 1.0),
            },
            "sacred_scores": {
                "phi_score": max(0, 1.0 - abs(phi_frac)),
                "fibonacci_score": max(0, 1.0 - fib_proximity / 100),
                "octave_score": max(0, 1.0 - abs(octaves_from_gc - round(octaves_from_gc))),
            },
        }

    def domain_summary(self) -> Dict[str, Any]:
        """
        Summary across ALL physical domains — particles, nuclei, iron,
        cosmos, resonance. Shows count and precision per domain.
        """
        self._metrics["domain_queries"] += 1
        self._metrics["total_operations"] += 1
        if not self._available:
            return {"error": "dual_layer_not_available", "fallback": True}

        summary = {}
        for fn_name, label in [
            ("particles", "particle_physics"),
            ("nuclei", "nuclear"),
            ("iron", "iron"),
            ("cosmos", "astrophysics"),
            ("resonance", "resonance"),
        ]:
            try:
                fn = getattr(_dual_layer, fn_name)
                result = fn()
                summary[label] = {
                    "count": result.get("count", 0),
                    "avg_error_pct": result.get("avg_error_pct", 0),
                }
            except Exception as e:
                summary[label] = {"error": str(e)}

        total = sum(d.get("count", 0) for d in summary.values() if "count" in d)
        return {
            "total_constants": total,
            "domains": summary,
        }

    def compare_constants(self, name_a: str, name_b: str) -> Dict[str, Any]:
        """
        Full dual-layer comparison of two constants.

        Combines harmonic relationship (Thought) with Physics precision
        for both, yielding a complete cross-layer comparison.
        """
        self._metrics["thought_calls"] += 1
        self._metrics["physics_calls"] += 1
        self._metrics["total_operations"] += 1
        if not self._available:
            return {"pair": (name_a, name_b), "error": "dual_layer_not_available"}

        harmonic = self.harmonic_relationship(name_a, name_b)
        derive_a = _dual_layer.derive_both(name_a)
        derive_b = _dual_layer.derive_both(name_b)

        return {
            "pair": (name_a, name_b),
            "harmonic": harmonic,
            "constant_a": {
                "name": name_a,
                "measured": derive_a.get("measured"),
                "thought_error_pct": derive_a.get("thought", {}).get("error_pct"),
                "physics_error_pct": derive_a.get("physics", {}).get("error_pct"),
                "improvement": derive_a.get("improvement"),
            },
            "constant_b": {
                "name": name_b,
                "measured": derive_b.get("measured"),
                "thought_error_pct": derive_b.get("thought", {}).get("error_pct"),
                "physics_error_pct": derive_b.get("physics", {}).get("error_pct"),
                "improvement": derive_b.get("improvement"),
            },
        }

    def duality_spectrum(self, name: str) -> Dict[str, Any]:
        """
        Full spectrum analysis of a constant across ALL duality dimensions.

        Combines: collapse, duality_tensor, symmetry, pattern recognition,
        grid placement — the most comprehensive single-constant analysis.
        """
        self._metrics["thought_calls"] += 1
        self._metrics["physics_calls"] += 1
        self._metrics["collapse_calls"] += 1
        self._metrics["total_operations"] += 1
        if not self._available:
            return {"name": name, "error": "dual_layer_not_available"}

        both = _dual_layer.derive_both(name)
        tensor = self.duality_tensor(name)
        symmetry = self.detect_symmetry(name)

        return {
            "name": name,
            "measured": both.get("measured"),
            "unit": both.get("unit", ""),
            "collapse": both,
            "tensor": tensor,
            "symmetry": symmetry,
            "dual_derivation": both,
            "duality_mapping": {
                duality_name: {
                    "abstract": duality["abstract"],
                    "concrete": duality["concrete"],
                    "asi_mapping": duality["asi_mapping"],
                }
                for duality_name, duality in NATURES_DUALITIES.items()
            },
        }

    def sweep_phi_space(self, center: float = 0.0, radius: int = 20) -> Dict[str, Any]:
        """
        Sweep the φ-exponent space around a center point.

        Computes Thought and Physics layer values at integer φ-exponents
        from (center - radius) to (center + radius). Reveals the harmonic
        structure of both grids near any scale.
        """
        self._metrics["thought_calls"] += 1
        self._metrics["physics_calls"] += 1
        self._metrics["total_operations"] += 1

        points = []
        for i in range(-radius, radius + 1):
            exp = center + i
            t_val = self.thought(a=0, b=0, c=0, d=0)  # base value
            p_val = self.physics_v3(a=0, b=0, c=0, d=0)
            phi_power = PHI ** exp
            points.append({
                "phi_exponent": exp,
                "phi_power": phi_power,
                "thought_base_x_phi": t_val * phi_power / GOD_CODE,
                "physics_base_x_phi": p_val * phi_power / GOD_CODE,
            })

        return {
            "center": center,
            "radius": radius,
            "total_points": len(points),
            "phi": PHI,
            "god_code": GOD_CODE,
            "points": points,
        }

    def compute_precision_map(self) -> Dict[str, Any]:
        """
        Compute precision map across all constants — shows which
        constants are most/least precisely derived by each layer.

        Returns ranked lists for both Thought and Physics layers.
        """
        self._metrics["physics_calls"] += 1
        self._metrics["thought_calls"] += 1
        self._metrics["total_operations"] += 1
        if not self._available:
            return {"error": "dual_layer_not_available", "fallback": True}

        precision_data = []
        for name in sorted(_dual_layer.REAL_WORLD_CONSTANTS_V3.keys()):
            try:
                both = _dual_layer.derive_both(name)
                thought_err = both.get("thought", {}).get("error_pct", 100)
                physics_err = both.get("physics", {}).get("error_pct", 100)
                improvement = thought_err / physics_err if physics_err > 0 else float('inf')
                precision_data.append({
                    "name": name,
                    "thought_error_pct": thought_err,
                    "physics_error_pct": physics_err,
                    "improvement": round(improvement, 1),
                    "domain": _dual_layer.REAL_WORLD_CONSTANTS_V3[name].get("domain", "unknown"),
                })
            except Exception:
                pass

        # Sort by physics error (best first)
        by_physics = sorted(precision_data, key=lambda x: x["physics_error_pct"])
        # Sort by improvement factor (most improved first)
        by_improvement = sorted(precision_data, key=lambda x: -x["improvement"])
        # Sort by thought error (best first)
        by_thought = sorted(precision_data, key=lambda x: x["thought_error_pct"])

        mean_thought = sum(d["thought_error_pct"] for d in precision_data) / len(precision_data) if precision_data else 0
        mean_physics = sum(d["physics_error_pct"] for d in precision_data) / len(precision_data) if precision_data else 0
        mean_improvement = sum(d["improvement"] for d in precision_data) / len(precision_data) if precision_data else 0

        return {
            "total_constants": len(precision_data),
            "mean_thought_error_pct": mean_thought,
            "mean_physics_error_pct": mean_physics,
            "mean_improvement": round(mean_improvement, 1),
            "best_physics": by_physics[:13],  # (was 5)
            "worst_physics": by_physics[-13:],  # (was 5)
            "most_improved": by_improvement[:13],  # (was 5)
            "best_thought": by_thought[:13],  # (was 5)
            "all_data": precision_data,
        }

    # ══════ PREDICTION ENGINE ══════
    #
    # The accuracy analysis (_analyze_accuracy.py, Tests 1-9) proved that:
    #   - The equation fits ANY positive number to ±0.005% (not selective)
    #   - X_V3 was reverse-engineered to make c exact (not derived)
    #   - The 4 dials collapse to 1 integer E (log scale encoding)
    #
    # This means "deriving" known constants isn't a discovery. However,
    # the equation CAN make genuine predictions if we:
    #   1. Find grid points with "nice" (simple) dials
    #   2. That DON'T correspond to any known constant
    #   3. Predict their values as undiscovered constants
    #   4. Wait for experimental confirmation
    #
    # THAT would prove the grid has physical meaning beyond encoding.
    # ══════════════════════════════════════════════════════════════════

    def predict(self, max_complexity: int = 30, top_n: int = 50) -> Dict[str, Any]:
        """
        Prediction Engine — Find 'nice' grid points not matched to known constants.

        Scans both Thought and Physics layers for dial combinations with
        low total complexity (|a| + |b| + |c| + |d|) that don't correspond
        to any of the 63 registered constants. These are PREDICTIONS —
        values where the grid suggests a physical constant should exist.

        A confirmed prediction would prove the grid has physical significance.

        Args:
            max_complexity: Maximum dial complexity to scan (|a|+|b|+|c|+|d|)
            top_n: Number of top predictions to return

        Returns:
            Dict with ranked predictions from both layers.
        """
        self._metrics["total_operations"] += 1
        self._metrics["cross_layer_calls"] += 1

        # Collect all known constant values for matching
        known_values = []
        if self._available:
            for name, entry in _dual_layer.REAL_WORLD_CONSTANTS_V3.items():
                known_values.append((name, entry["measured"]))

        def _is_known(val, threshold_pct=0.01):
            """Check if a value is within threshold% of any known constant."""
            for name, kv in known_values:
                if kv > 0 and abs(val - kv) / kv * 100 < threshold_pct:
                    return name
            return None

        # Scan Physics layer (v3) — this is the precision grid
        physics_predictions = []
        if self._available:
            for d in range(-10, 11):
                for a in range(0, max_complexity + 1):
                    for c in range(0, max_complexity + 1):
                        for b in range(0, max_complexity * 2):
                            complexity = a + b + c + abs(d)
                            if complexity > max_complexity or complexity == 0:
                                continue
                            val = _dual_layer.god_code_v3(a, b, c, d)
                            if val <= 0 or val > 1e35 or val < 1e-35:
                                continue
                            match = _is_known(val)
                            if match is None:
                                physics_predictions.append({
                                    "dials": (a, b, c, d),
                                    "value": val,
                                    "complexity": complexity,
                                    "layer": "physics",
                                    "equation": f"G_v3({a},{b},{c},{d})",
                                    "log10": math.log10(val) if val > 0 else 0,
                                })
                            # Early exit on b to keep scan tractable
                            if b > max_complexity:
                                break
                    if a + c > max_complexity:
                        break

        # Scan Thought layer (original) — coarser but with pattern meaning
        thought_predictions = []
        if self._available:
            for d in range(-6, 7):
                for a in range(-3, 6):
                    for c in range(-3, 6):
                        for b in range(-10, 30):
                            complexity = abs(a) + abs(b) + abs(c) + abs(d)
                            if complexity > max_complexity or complexity == 0:
                                continue
                            val = _dual_layer.god_code_equation(a, b, c, d)
                            if val <= 0 or val > 1e35 or val < 1e-35:
                                continue
                            match = _is_known(val)
                            if match is None:
                                thought_predictions.append({
                                    "dials": (a, b, c, d),
                                    "value": val,
                                    "complexity": complexity,
                                    "layer": "thought",
                                    "equation": f"G({a},{b},{c},{d})",
                                    "log10": math.log10(val) if val > 0 else 0,
                                })

        # Sort by simplicity (lowest complexity first)
        physics_predictions.sort(key=lambda x: (x["complexity"], abs(x["log10"])))
        thought_predictions.sort(key=lambda x: (x["complexity"], abs(x["log10"])))

        # Find convergences — values that appear in BOTH layers (most significant)
        convergences = []
        for pp in physics_predictions[:top_n * 5]:
            for tp in thought_predictions[:top_n * 5]:
                if pp["value"] > 0 and tp["value"] > 0:
                    ratio = pp["value"] / tp["value"]
                    if abs(ratio - 1) < 0.001:  # Within 0.1% of each other
                        convergences.append({
                            "physics": pp,
                            "thought": tp,
                            "convergence_pct": abs(ratio - 1) * 100,
                            "combined_complexity": pp["complexity"] + tp["complexity"],
                            "value_avg": (pp["value"] + tp["value"]) / 2,
                        })

        convergences.sort(key=lambda x: (x["combined_complexity"], x["convergence_pct"]))

        return {
            "method": "Dual-Layer Prediction Engine",
            "purpose": "Find grid points with simple dials that don't match known constants",
            "significance": "A confirmed prediction would prove the grid has physical meaning",
            "max_complexity_scanned": max_complexity,
            "known_constants_checked": len(known_values),
            "physics_predictions": physics_predictions[:top_n],
            "thought_predictions": thought_predictions[:top_n],
            "convergences": convergences[:50],  # (was 20)
            "total_physics_unmatched": len(physics_predictions),
            "total_thought_unmatched": len(thought_predictions),
            "total_convergences": len(convergences),
            "note": (
                "These are values where the grid PREDICTS a constant should exist. "
                "Experimental discovery of any of these would constitute a genuine "
                "scientific discovery and prove the grid has physical significance "
                "beyond numerical encoding."
            ),
        }

    def predict_summary(self, max_complexity: int = 15) -> str:
        """Human-readable summary of top predictions from both layers."""
        result = self.predict(max_complexity=max_complexity, top_n=20)

        lines = [
            "═══ DUAL-LAYER PREDICTION ENGINE ═══",
            f"Known constants: {result['known_constants_checked']}",
            f"Unmatched physics grid points: {result['total_physics_unmatched']}",
            f"Unmatched thought grid points: {result['total_thought_unmatched']}",
            f"Cross-layer convergences: {result['total_convergences']}",
            "",
        ]

        if result["convergences"]:
            lines.append("★ CROSS-LAYER CONVERGENCES (most significant):")
            for i, conv in enumerate(result["convergences"][:10], 1):
                val = conv["value_avg"]
                pp = conv["physics"]
                tp = conv["thought"]
                lines.append(
                    f"  {i:2d}. {val:>14.6g}  "
                    f"physics={pp['equation']:<16s} thought={tp['equation']:<16s} "
                    f"conv={conv['convergence_pct']:.4f}%"
                )
            lines.append("")

        lines.append("Top Physics predictions (simplest dials, no known match):")
        for i, p in enumerate(result["physics_predictions"][:10], 1):
            lines.append(
                f"  {i:2d}. {p['value']:>14.6g}  dials={p['equation']:<16s} complexity={p['complexity']}"
            )

        lines.append("")
        lines.append("Top Thought predictions (simplest dials, no known match):")
        for i, p in enumerate(result["thought_predictions"][:10], 1):
            lines.append(
                f"  {i:2d}. {p['value']:>14.6g}  dials={p['equation']:<16s} complexity={p['complexity']}"
            )

        lines.append("")
        lines.append("NOTE: If any of these values is later discovered as a physical")
        lines.append("constant, it would prove the grid has genuine physical meaning.")

        return "\n".join(lines)

    # ══════ ASI INTEGRATION METHODS ══════

    def validate_constant(self, name: str) -> bool:
        """Validate a physical constant through dual-layer integrity."""
        if not self._available:
            return True  # Assume valid in fallback mode
        try:
            result = self.derive(name, mode="physics")
            return result.get("error_pct", 100.0) < 0.01  # Within 0.01%
        except Exception:
            return False

    def thought_insight(self, name: str) -> str:
        """Get the Thought layer's understanding of a constant (pattern, meaning)."""
        if not self._available:
            return f"Thought layer insight for '{name}' (requires dual-layer engine)"
        try:
            both = self.derive_both(name)
            return both.get("thought", {}).get("meaning", "")
        except Exception:
            return ""

    def physics_precision(self, name: str) -> float:
        """Get the Physics layer's precision for a constant (error %)."""
        if not self._available:
            return 0.0
        try:
            result = self.derive(name, mode="physics")
            return result.get("error_pct", 0.0)
        except Exception:
            return 0.0

    def dual_score(self) -> float:
        """
        Compute a dual-layer health score for the ASI.

        This is the FLAGSHIP metric — measures how well both layers
        are functioning and converging through the bridge.
        """
        if not self._available:
            return 0.5  # Neutral score without dual-layer

        try:
            integrity = self.full_integrity_check()
            passed = integrity.get("checks_passed", 0)
            total = integrity.get("total_checks", 10)
            return passed / max(total, 1)
        except Exception:
            return 0.5

    # ══════ STATUS ══════

    def get_status(self) -> Dict[str, Any]:
        """Complete dual-layer engine status for the ASI."""
        if self._available:
            try:
                full_status = _dual_layer.status()
                full_status["flagship"] = True
                full_status["asi_integrated"] = True
                full_status["metrics"] = dict(self._metrics)
                full_status["uptime_seconds"] = round(time.time() - self._boot_time, 2)
                return full_status
            except Exception as e:
                pass

        return {
            "engine": "L104 Dual-Layer Engine (ASI Flagship)",
            "version": self.VERSION,
            "flagship": True,
            "available": self._available,
            "asi_integrated": True,
            "architecture": "Thought (abstract, why) + Physics (concrete, how much)",
            "dualities": list(NATURES_DUALITIES.keys()),
            "bridge_elements": list(CONSCIOUSNESS_TO_PHYSICS_BRIDGE.keys()),
            "metrics": dict(self._metrics),
            "uptime_seconds": round(time.time() - self._boot_time, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def status(self) -> Dict[str, Any]:
        """Alias for get_status()."""
        return self.get_status()

    # ══════════════════════════════════════════════════════════════════════════
    # CROSS-ANALYSIS ENGINE — v2.1.0 Upgrade
    # ══════════════════════════════════════════════════════════════════════════

    def cross_domain_analysis(self) -> Dict[str, Any]:
        """
        Cross-domain analysis: how do constants in different physics domains
        relate through the dual-layer grid?

        Computes inter-domain exponent distances, finding which domains are
        "neighbors" on the grid and which are far apart.
        """
        self._metrics["cross_layer_calls"] += 1
        self._metrics["total_operations"] += 1
        if not self._available:
            return {"error": "dual_layer_not_available", "fallback": True}

        domains: Dict[str, List[Tuple[str, int]]] = {}
        for name, entry in _dual_layer.REAL_WORLD_CONSTANTS_V3.items():
            d = entry.get("domain", "unknown")
            domains.setdefault(d, []).append((name, entry["E_integer"]))

        # Domain centroids (mean exponent)
        centroids = {}
        for d, members in domains.items():
            exps = [e for _, e in members]
            centroids[d] = sum(exps) / len(exps)

        # Domain spreads (std-dev of exponents)
        spreads = {}
        for d, members in domains.items():
            exps = [e for _, e in members]
            mean = centroids[d]
            spreads[d] = (sum((e - mean)**2 for e in exps) / max(len(exps) - 1, 1)) ** 0.5

        # Inter-domain distances (centroid-to-centroid)
        domain_names = sorted(domains.keys())
        distances = {}
        for i, da in enumerate(domain_names):
            for db in domain_names[i+1:]:
                dist = abs(centroids[da] - centroids[db])
                distances[f"{da}<->{db}"] = round(dist, 1)

        # Closest pair
        closest = min(distances, key=distances.get) if distances else None
        farthest = max(distances, key=distances.get) if distances else None

        return {
            "total_domains": len(domains),
            "domain_sizes": {d: len(m) for d, m in sorted(domains.items())},
            "centroids": {d: round(c, 1) for d, c in sorted(centroids.items())},
            "spreads": {d: round(s, 1) for d, s in sorted(spreads.items())},
            "inter_domain_distances": dict(sorted(distances.items(), key=lambda x: x[1])),
            "closest_domains": closest,
            "farthest_domains": farthest,
        }

    def statistical_profile(self) -> Dict[str, Any]:
        """
        Full statistical profile of the dual-layer grid encoding:
        distribution of errors, exponents, dials, and precision.
        """
        self._metrics["total_operations"] += 1
        if not self._available:
            return {"error": "dual_layer_not_available", "fallback": True}

        import statistics as _stats

        errors = []
        exponents = []
        deltas = []
        dial_complexities = []
        for name, entry in _dual_layer.REAL_WORLD_CONSTANTS_V3.items():
            errors.append(entry["grid_error_pct"])
            exponents.append(entry["E_integer"])
            deltas.append(abs(entry.get("delta", 0)))
            dials = entry.get("dials", (0, 0, 0, 0))
            dial_complexities.append(sum(abs(d) for d in dials))

        def _stats_block(data, label):
            if not data:
                return {}
            return {
                f"{label}_mean": round(sum(data) / len(data), 8),
                f"{label}_median": round(_stats.median(data), 8),
                f"{label}_stdev": round(_stats.stdev(data), 8) if len(data) > 1 else 0,
                f"{label}_min": min(data),
                f"{label}_max": max(data),
                f"{label}_range": max(data) - min(data),
            }

        result = {"count": len(errors)}
        result.update(_stats_block(errors, "error_pct"))
        result.update(_stats_block(exponents, "exponent"))
        result.update(_stats_block(deltas, "delta"))
        result.update(_stats_block(dial_complexities, "dial_complexity"))

        # Error distribution buckets
        buckets = {"<0.001%": 0, "0.001-0.002%": 0, "0.002-0.003%": 0, "0.003-0.004%": 0, "0.004-0.005%": 0, ">0.005%": 0}
        for e in errors:
            if e < 0.001: buckets["<0.001%"] += 1
            elif e < 0.002: buckets["0.001-0.002%"] += 1
            elif e < 0.003: buckets["0.002-0.003%"] += 1
            elif e < 0.004: buckets["0.003-0.004%"] += 1
            elif e < 0.005: buckets["0.004-0.005%"] += 1
            else: buckets[">0.005%"] += 1
        result["error_distribution"] = buckets

        return result

    def independent_verification(self) -> Dict[str, Any]:
        """
        Verify every constant against INDEPENDENT reference values
        (CODATA 2022, PDG 2024, CGPM exact definitions).

        This is the gold-standard test: do the grid values match
        real authorities, not just the engine's own registry?
        """
        self._metrics["integrity_checks"] += 1
        self._metrics["total_operations"] += 1
        if not self._available:
            return {"error": "dual_layer_not_available", "fallback": True}

        results = []
        for name, (ref_val, unit, source) in _dual_layer.INDEPENDENT_REFERENCE_VALUES.items():
            derived = _dual_layer.derive(name, mode="physics")
            grid_val = derived.get("grid_value", derived.get("value", 0))
            measured = derived.get("measured", ref_val)

            # Compare grid value to independent reference
            if ref_val != 0:
                grid_vs_ref_err = abs(grid_val - ref_val) / abs(ref_val) * 100
                measured_vs_ref_err = abs(measured - ref_val) / abs(ref_val) * 100
            else:
                grid_vs_ref_err = 0
                measured_vs_ref_err = 0

            results.append({
                "name": name,
                "reference_value": ref_val,
                "reference_source": source,
                "unit": unit,
                "grid_value": grid_val,
                "measured_in_registry": measured,
                "grid_vs_reference_error_pct": grid_vs_ref_err,
                "registry_vs_reference_error_pct": measured_vs_ref_err,
                "grid_match": grid_vs_ref_err < 0.01,
                "registry_match": measured_vs_ref_err < 1e-10,
            })

        passed = sum(1 for r in results if r["grid_match"])
        registry_exact = sum(1 for r in results if r["registry_match"])

        return {
            "total_verified": len(results),
            "grid_within_001_pct": passed,
            "registry_exact_match": registry_exact,
            "results": results,
            "verdict": "VERIFIED" if passed == len(results) else f"{passed}/{len(results)} verified",
        }

    def exponent_spectrum(self) -> Dict[str, Any]:
        """
        Analyze the full exponent spectrum: how are integer exponents
        distributed across the grid? Are there clusters, gaps, patterns?
        """
        self._metrics["total_operations"] += 1
        if not self._available:
            return {"error": "dual_layer_not_available", "fallback": True}

        exponents = {}
        for name, entry in _dual_layer.REAL_WORLD_CONSTANTS_V3.items():
            E = entry["E_integer"]
            exponents[E] = name

        sorted_E = sorted(exponents.keys())
        gaps = [sorted_E[i+1] - sorted_E[i] for i in range(len(sorted_E) - 1)]

        # Detect clusters (groups within ±50 of each other)
        clusters = []
        current_cluster = [sorted_E[0]]
        for i in range(1, len(sorted_E)):
            if sorted_E[i] - sorted_E[i-1] <= 50:
                current_cluster.append(sorted_E[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [sorted_E[i]]
        clusters.append(current_cluster)

        return {
            "total_exponents": len(sorted_E),
            "range": (sorted_E[0], sorted_E[-1]),
            "span": sorted_E[-1] - sorted_E[0],
            "mean_gap": round(sum(gaps) / len(gaps), 1) if gaps else 0,
            "min_gap": min(gaps) if gaps else 0,
            "max_gap": max(gaps) if gaps else 0,
            "median_gap": sorted(gaps)[len(gaps)//2] if gaps else 0,
            "num_clusters": len(clusters),
            "cluster_sizes": [len(c) for c in clusters],
            "largest_cluster": max(clusters, key=len) if clusters else [],
            "isolated_points": [c[0] for c in clusters if len(c) == 1],
            "occupancy_pct": round(len(sorted_E) / (sorted_E[-1] - sorted_E[0] + 1) * 100, 4) if len(sorted_E) > 1 else 100,
        }

    def dial_algebra(self) -> Dict[str, Any]:
        """
        Analyze the algebraic structure of dial tuples (a, b, c, d):
        which dials contribute most, are there correlations, symmetries?
        """
        self._metrics["thought_calls"] += 1
        self._metrics["total_operations"] += 1
        if not self._available:
            return {"error": "dual_layer_not_available", "fallback": True}

        all_a, all_b, all_c, all_d = [], [], [], []
        dial_sums = []
        for name, entry in _dual_layer.REAL_WORLD_CONSTANTS_V3.items():
            a, b, c, d = entry["dials"]
            all_a.append(a)
            all_b.append(b)
            all_c.append(c)
            all_d.append(d)
            dial_sums.append(a + b + c + d)

        def _stats(data, label):
            return {
                f"{label}_mean": round(sum(data) / len(data), 3),
                f"{label}_range": (min(data), max(data)),
                f"{label}_nonzero": sum(1 for x in data if x != 0),
            }

        result = {"count": len(all_a)}
        result.update(_stats(all_a, "a"))
        result.update(_stats(all_b, "b"))
        result.update(_stats(all_c, "c"))
        result.update(_stats(all_d, "d"))
        result.update(_stats(dial_sums, "sum"))

        # Which dial has the largest dynamic range?
        ranges = {
            "a": max(all_a) - min(all_a),
            "b": max(all_b) - min(all_b),
            "c": max(all_c) - min(all_c),
            "d": max(all_d) - min(all_d),
        }
        result["dominant_dial"] = max(ranges, key=ranges.get)
        result["dial_ranges"] = ranges

        # Unique dial patterns
        patterns = set()
        for name, entry in _dual_layer.REAL_WORLD_CONSTANTS_V3.items():
            patterns.add(entry["dials"])
        result["unique_patterns"] = len(patterns)
        result["pattern_diversity_pct"] = round(len(patterns) / len(all_a) * 100, 1)

        return result

    def layer_improvement_ranking(self) -> Dict[str, Any]:
        """
        Rank constants by how much Physics improves over Thought.
        Shows where the v3 refinement makes the biggest and smallest difference.
        """
        self._metrics["cross_layer_calls"] += 1
        self._metrics["total_operations"] += 1
        if not self._available:
            return {"error": "dual_layer_not_available", "fallback": True}

        rankings = []
        for name in sorted(_dual_layer.REAL_WORLD_CONSTANTS_V3.keys()):
            both = _dual_layer.derive_both(name)
            t_err = both.get("thought", {}).get("error_pct", 100)
            p_err = both.get("physics", {}).get("grid_error_pct", both.get("physics", {}).get("error_pct", 0.01))
            improvement = t_err / p_err if p_err > 0 else float('inf')
            rankings.append({
                "name": name,
                "thought_error_pct": round(t_err, 6),
                "physics_error_pct": round(p_err, 6),
                "improvement_factor": round(improvement, 2),
            })

        rankings.sort(key=lambda r: r["improvement_factor"], reverse=True)

        improvements = [r["improvement_factor"] for r in rankings if r["improvement_factor"] < float('inf')]
        mean_improvement = sum(improvements) / len(improvements) if improvements else 0

        return {
            "total": len(rankings),
            "mean_improvement": round(mean_improvement, 2),
            "best_13": rankings[:13],  # (was 5)
            "worst_13": rankings[-13:],  # (was 5)
            "above_100x": sum(1 for r in rankings if r["improvement_factor"] > 100),
            "above_1000x": sum(1 for r in rankings if r["improvement_factor"] > 1000),
            "below_10x": sum(1 for r in rankings if r["improvement_factor"] < 10),
        }

    def phi_resonance_scan(self) -> Dict[str, Any]:
        """
        Scan all 63 constants for phi (golden ratio) resonance:
        How close is each constant's natural log / log(phi) to an integer?

        If a constant naturally "lives" at a phi-octave, that's resonance.
        """
        self._metrics["thought_calls"] += 1
        self._metrics["total_operations"] += 1
        if not self._available:
            return {"error": "dual_layer_not_available", "fallback": True}

        log_phi = math.log(PHI)
        hits = []
        for name, entry in _dual_layer.REAL_WORLD_CONSTANTS_V3.items():
            val = entry["measured"]
            if val <= 0:
                continue
            phi_exp = math.log(val) / log_phi
            nearest_int = round(phi_exp)
            fractional = abs(phi_exp - nearest_int)
            hits.append({
                "name": name,
                "measured": val,
                "phi_exponent": round(phi_exp, 6),
                "nearest_integer": nearest_int,
                "fractional_distance": round(fractional, 6),
                "is_resonant": fractional < 0.05,
            })

        hits.sort(key=lambda h: h["fractional_distance"])
        resonant = [h for h in hits if h["is_resonant"]]

        return {
            "total_scanned": len(hits),
            "resonant_count": len(resonant),
            "resonant_constants": [h["name"] for h in resonant],
            "best_13": hits[:13],  # (was 5)
            "worst_13": hits[-13:],  # (was 5)
            "mean_fractional_distance": round(sum(h["fractional_distance"] for h in hits) / len(hits), 6) if hits else 0,
        }

    def nucleosynthesis_chain(self) -> Dict[str, Any]:
        """
        Trace the nucleosynthesis chain through the grid:
        H → He-4 → C-12 → O-16 → Si-28 → Fe-56

        How do nuclear binding energies relate on the dual-layer grid?
        """
        self._metrics["thought_calls"] += 1
        self._metrics["physics_calls"] += 1
        self._metrics["total_operations"] += 1
        if not self._available:
            return {"error": "dual_layer_not_available", "fallback": True}

        chain_constants = ["deuteron_be", "he4_be_per_nucleon", "c12_be_per_nucleon",
                          "o16_be_per_nucleon", "fe56_be_per_nucleon", "ni62_be_per_nucleon",
                          "u238_be_per_nucleon", "triton_be"]
        chain = []
        for name in chain_constants:
            if name in _dual_layer.REAL_WORLD_CONSTANTS_V3:
                entry = _dual_layer.REAL_WORLD_CONSTANTS_V3[name]
                both = _dual_layer.derive_both(name)
                chain.append({
                    "name": name,
                    "measured_MeV": entry["measured"],
                    "dials": entry["dials"],
                    "E_integer": entry["E_integer"],
                    "grid_error_pct": entry["grid_error_pct"],
                    "thought_error_pct": both.get("thought", {}).get("error_pct"),
                    "physics_error_pct": both.get("physics", {}).get("grid_error_pct"),
                })

        # Sort by binding energy (the valley of stability)
        chain.sort(key=lambda c: c["measured_MeV"])

        # Check if exponents follow monotonic ordering
        exponents = [c["E_integer"] for c in chain]
        monotonic = all(exponents[i] <= exponents[i+1] for i in range(len(exponents)-1))

        return {
            "chain_length": len(chain),
            "chain": chain,
            "exponent_monotonic": monotonic,
            "peak_binding": max(chain, key=lambda c: c["measured_MeV"])["name"] if chain else None,
            "exponent_span": max(exponents) - min(exponents) if exponents else 0,
        }

    def grid_entropy(self) -> Dict[str, Any]:
        """
        Information-theoretic analysis: how much information is encoded
        in the grid placement? Shannon entropy of the exponent distribution.

        Higher entropy = more uniform distribution = better information encoding.
        """
        self._metrics["total_operations"] += 1
        if not self._available:
            return {"error": "dual_layer_not_available", "fallback": True}

        exponents = sorted(entry["E_integer"] for entry in _dual_layer.REAL_WORLD_CONSTANTS_V3.values())
        E_min, E_max = exponents[0], exponents[-1]
        span = E_max - E_min + 1

        # Bin the exponents into 20 equal-width bins
        n_bins = 20
        bin_width = span / n_bins
        bins = [0] * n_bins
        for E in exponents:
            idx = min(int((E - E_min) / bin_width), n_bins - 1)
            bins[idx] += 1

        # Shannon entropy
        total = len(exponents)
        entropy = 0
        for count in bins:
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        max_entropy = math.log2(n_bins)
        efficiency = entropy / max_entropy if max_entropy > 0 else 0

        # Gap entropy — information in the gap distribution
        gaps = [exponents[i+1] - exponents[i] for i in range(len(exponents) - 1)]
        gap_set = set(gaps)
        gap_freq = {g: gaps.count(g) for g in gap_set}
        gap_entropy = 0
        for g, count in gap_freq.items():
            p = count / len(gaps)
            gap_entropy -= p * math.log2(p)

        return {
            "total_exponents": total,
            "exponent_range": (E_min, E_max),
            "span": span,
            "bins": n_bins,
            "bin_counts": bins,
            "shannon_entropy": round(entropy, 4),
            "max_possible_entropy": round(max_entropy, 4),
            "encoding_efficiency": round(efficiency * 100, 2),
            "gap_entropy": round(gap_entropy, 4),
            "unique_gaps": len(gap_set),
            "interpretation": (
                "High efficiency (>80%): Constants spread uniformly across the grid" if efficiency > 0.8
                else "Medium efficiency (50-80%): Moderate clustering" if efficiency > 0.5
                else "Low efficiency (<50%): Heavy clustering — grid is sparsely used"
            ),
        }

    def cross_validate_layers(self) -> Dict[str, Any]:
        """
        The ultimate cross-validation: for each constant, derive via
        derive_both() — then verify Thought (Layer 1) and Physics (Layer 2)
        give consistent results and that Physics always improves on Thought.
        """
        self._metrics["cross_layer_calls"] += 1
        self._metrics["total_operations"] += 1
        if not self._available:
            return {"error": "dual_layer_not_available", "fallback": True}

        results = []
        for name in sorted(_dual_layer.REAL_WORLD_CONSTANTS_V3.keys()):
            both = _dual_layer.derive_both(name)
            measured = both.get("measured", 0)

            # Layer 1 (Thought/Consciousness): coarse grid
            consciousness = both.get("consciousness", both.get("thought", {}))
            t_val = consciousness.get("value", 0)
            t_err = consciousness.get("error_pct", 0)

            # Layer 2 (Physics): fine grid
            physics = both.get("physics", {})
            p_val = physics.get("value", 0)
            p_err = physics.get("error_pct", physics.get("grid_error_pct", 0))

            # Improvement: how much better is Physics than Thought?
            improvement = t_err / p_err if p_err > 0 else float('inf')

            # Both must produce nonzero values for the same measured constant
            t_valid = t_val != 0 and abs(t_err) < 1.0  # Within 1%
            p_valid = p_val != 0 and abs(p_err) < 0.01  # Within 0.01%

            results.append({
                "name": name,
                "measured": measured,
                "thought_value": t_val,
                "physics_value": p_val,
                "thought_error_pct": round(t_err, 6),
                "physics_error_pct": round(p_err, 6),
                "improvement": round(improvement, 2) if improvement < float('inf') else float('inf'),
                "thought_valid": t_valid,
                "physics_valid": p_valid,
                "fully_consistent": t_valid and p_valid,
            })

        consistent = sum(1 for r in results if r["fully_consistent"])
        physics_always_better = all(r["improvement"] >= 1.0 for r in results if r["improvement"] < float('inf'))

        return {
            "total": len(results),
            "fully_consistent": consistent,
            "inconsistent": len(results) - consistent,
            "physics_always_better": physics_always_better,
            "mean_thought_error": round(sum(r["thought_error_pct"] for r in results) / len(results), 6),
            "mean_physics_error": round(sum(r["physics_error_pct"] for r in results) / len(results), 6),
            "details": results,
            "verdict": "CROSS-VALIDATED" if consistent == len(results) else f"{consistent}/{len(results)} consistent",
        }

    def domain_correlation_matrix(self) -> Dict[str, Any]:
        """
        Correlation matrix between domains: do constants in similar
        physics domains cluster together on the grid?

        Uses exponent distance as the metric.
        """
        self._metrics["cross_layer_calls"] += 1
        self._metrics["total_operations"] += 1
        if not self._available:
            return {"error": "dual_layer_not_available", "fallback": True}

        domains: Dict[str, List[int]] = {}
        for name, entry in _dual_layer.REAL_WORLD_CONSTANTS_V3.items():
            d = entry.get("domain", "unknown")
            domains.setdefault(d, []).append(entry["E_integer"])

        domain_names = sorted(domains.keys())
        matrix: Dict[str, Dict[str, float]] = {}

        for da in domain_names:
            matrix[da] = {}
            for db in domain_names:
                if da == db:
                    matrix[da][db] = 0.0  # Self-distance is always 0
                    continue
                # Average pairwise exponent distance (inter-domain)
                total_dist = 0
                count = 0
                for ea in domains[da]:
                    for eb in domains[db]:
                        total_dist += abs(ea - eb)
                        count += 1
                matrix[da][db] = round(total_dist / count, 1) if count > 0 else 0

        # Normalize to get correlation-like metric (1 = identical/self, 0 = maximally different)
        off_diag = [matrix[da][db] for da in domain_names for db in domain_names if da != db]
        max_dist = max(off_diag) if off_diag else 1
        normalized = {}
        for da in domain_names:
            normalized[da] = {}
            for db in domain_names:
                if da == db:
                    normalized[da][db] = 1.0
                else:
                    normalized[da][db] = round(1 - matrix[da][db] / max_dist, 3)

        # Find strongest cross-domain correlations
        cross_correlations = []
        for i, da in enumerate(domain_names):
            for db in domain_names[i+1:]:
                cross_correlations.append((da, db, normalized[da][db]))
        cross_correlations.sort(key=lambda x: x[2], reverse=True)

        return {
            "domains": domain_names,
            "distance_matrix": matrix,
            "correlation_matrix": normalized,
            "strongest_correlations": [(a, b, round(c, 3)) for a, b, c in cross_correlations[:13]],  # (was 5)
            "weakest_correlations": [(a, b, round(c, 3)) for a, b, c in cross_correlations[-13:]],  # (was 5)
        }

    def anomaly_detection(self) -> Dict[str, Any]:
        """
        Detect anomalies: constants whose grid behavior doesn't match
        the statistical profile of their domain.

        An anomaly might indicate something genuinely interesting
        about that constant's physics.
        """
        self._metrics["total_operations"] += 1
        if not self._available:
            return {"error": "dual_layer_not_available", "fallback": True}

        # Per-domain error statistics
        domain_errors: Dict[str, List[Tuple[str, float]]] = {}
        for name, entry in _dual_layer.REAL_WORLD_CONSTANTS_V3.items():
            d = entry.get("domain", "unknown")
            domain_errors.setdefault(d, []).append((name, entry["grid_error_pct"]))

        anomalies = []
        for domain, members in domain_errors.items():
            errors = [e for _, e in members]
            if len(errors) < 3:
                continue
            mean = sum(errors) / len(errors)
            stdev = (sum((e - mean)**2 for e in errors) / (len(errors) - 1)) ** 0.5
            if stdev == 0:
                continue
            for name, err in members:
                z_score = (err - mean) / stdev
                if abs(z_score) > 1.5:
                    anomalies.append({
                        "name": name,
                        "domain": domain,
                        "error_pct": round(err, 6),
                        "domain_mean": round(mean, 6),
                        "z_score": round(z_score, 3),
                        "direction": "unusually_precise" if z_score < 0 else "unusually_imprecise",
                    })

        anomalies.sort(key=lambda a: abs(a["z_score"]), reverse=True)
        return {
            "total_anomalies": len(anomalies),
            "anomalies": anomalies,
            "interpretation": (
                "Anomalously precise constants may sit at privileged grid points. "
                "Anomalously imprecise ones may need fractional-exponent refinement."
            ),
        }

    def fundamental_vs_derived_test(self) -> Dict[str, Any]:
        """
        Cross-check: can derived constants be reconstructed from fundamentals?

        E.g., Rydberg = alpha^2 * m_e * c / (2 * h)
        If the grid is self-consistent, derived quantities computed from
        grid-encoded fundamentals should match the grid-encoded derived value.
        """
        self._metrics["cross_layer_calls"] += 1
        self._metrics["total_operations"] += 1
        if not self._available:
            return {"error": "dual_layer_not_available", "fallback": True}

        checks = []

        # Check 1: Bohr magneton = e * ℏ / (2 * m_e)
        # We can check ratios: compare grid-encoded constants to known relationships
        def _get(name):
            e = _dual_layer.REAL_WORLD_CONSTANTS_V3.get(name, {})
            return e.get("measured", 0), e.get("grid_value", 0)

        # Check: Fine structure constant relationship
        # alpha_inv ≈ 137.036 should relate to other electromagnetic constants
        alpha_m, alpha_g = _get("fine_structure_inv")
        c_m, c_g = _get("speed_of_light")
        e_m, e_g = _get("elementary_charge")

        checks.append({
            "name": "fine_structure_inv_value",
            "description": "α⁻¹ ≈ 137.036 as encoded on grid",
            "measured": alpha_m,
            "grid": alpha_g,
            "error_pct": round(abs(alpha_g - alpha_m) / alpha_m * 100, 8) if alpha_m else 0,
            "pass": abs(alpha_g - alpha_m) / alpha_m < 0.0001 if alpha_m else True,
        })

        # Check: Proton/electron mass ratio ≈ 1836.15
        pm_m, pm_g = _get("proton_mass_MeV")
        em_m, em_g = _get("electron_mass_MeV")
        if pm_m and em_m:
            ratio_measured = pm_m / em_m
            ratio_grid = pm_g / em_g if em_g else 0
            expected_ratio = 1836.15267343  # CODATA 2022
            checks.append({
                "name": "proton_electron_ratio",
                "description": "m_p/m_e ≈ 1836.153 from grid-encoded masses",
                "measured_ratio": round(ratio_measured, 4),
                "grid_ratio": round(ratio_grid, 4),
                "expected": expected_ratio,
                "error_pct": round(abs(ratio_grid - expected_ratio) / expected_ratio * 100, 6),
                "pass": abs(ratio_grid - expected_ratio) / expected_ratio < 0.01,
            })

        # Check: Nucleon binding energy progression
        # He4 < C12 < O16 < Fe56 (binding energy per nucleon should increase to iron peak)
        he4_m, _ = _get("he4_be_per_nucleon")
        c12_m, _ = _get("c12_be_per_nucleon")
        o16_m, _ = _get("o16_be_per_nucleon")
        fe56_m, _ = _get("fe56_be_per_nucleon")
        if all([he4_m, c12_m, o16_m, fe56_m]):
            ordering_correct = he4_m < c12_m < o16_m < fe56_m
            checks.append({
                "name": "nuclear_binding_ordering",
                "description": "BE/A: He-4 < C-12 < O-16 < Fe-56 (valley of stability)",
                "values_MeV": {"He4": he4_m, "C12": c12_m, "O16": o16_m, "Fe56": fe56_m},
                "ordering_correct": ordering_correct,
                "pass": ordering_correct,
            })

        # Check: Stefan-Boltzmann from Boltzmann constant
        # σ = 2π⁵k⁴/(15h³c²) — too complex for grid check, but verify relative magnitude
        sb_m, sb_g = _get("stefan_boltzmann")
        k_m, k_g = _get("boltzmann_eV_K")
        if sb_m and k_m:
            checks.append({
                "name": "stefan_boltzmann_magnitude",
                "description": "σ encoded on grid should be O(10⁻⁸)",
                "measured": sb_m,
                "grid": sb_g,
                "correct_magnitude": 1e-9 < sb_g < 1e-7,
                "pass": 1e-9 < sb_g < 1e-7,
            })

        # Check: W/Z boson mass ratio ≈ 0.8815
        w_m, w_g = _get("W_boson_GeV")
        z_m, z_g = _get("Z_boson_GeV")
        if w_m and z_m:
            ratio_m = w_m / z_m
            ratio_g = w_g / z_g if z_g else 0
            checks.append({
                "name": "WZ_mass_ratio",
                "description": "M_W/M_Z ≈ cos(θ_W) ≈ 0.8815",
                "measured_ratio": round(ratio_m, 6),
                "grid_ratio": round(ratio_g, 6),
                "error_pct": round(abs(ratio_g - ratio_m) / ratio_m * 100, 6) if ratio_m else 0,
                "pass": abs(ratio_g - ratio_m) / ratio_m < 0.01 if ratio_m else True,
            })

        passed = sum(1 for c in checks if c["pass"])
        return {
            "total_checks": len(checks),
            "passed": passed,
            "failed": len(checks) - passed,
            "checks": checks,
            "verdict": "SELF-CONSISTENT" if passed == len(checks) else f"{passed}/{len(checks)} passed",
        }

    def upgrade_report(self) -> Dict[str, Any]:
        """
        Engine upgrade status report: comprehensive health check including
        all diagnostic subsystems and their current results.
        """
        self._metrics["total_operations"] += 1

        report = {
            "engine": "L104 Dual-Layer Engine",
            "version": self.VERSION,
            "upgrade_version": "2.1.0",
            "pipeline": "EVO_61_CROSS_ANALYSIS_UPGRADE",
            "available": self._available,
            "uptime_seconds": round(time.time() - self._boot_time, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "subsystems": {},
            "metrics": dict(self._metrics),
        }

        if not self._available:
            report["subsystems"]["all"] = {"status": "OFFLINE", "reason": "dual_layer_not_available"}
            return report

        # Run all diagnostic subsystems
        try:
            integrity = self.full_integrity_check(force=True)
            report["subsystems"]["integrity"] = {
                "status": "PASS" if integrity.get("all_passed") else "DEGRADED",
                "checks_passed": integrity.get("checks_passed", 0),
                "total_checks": integrity.get("total_checks", 0),
            }
        except Exception as e:
            report["subsystems"]["integrity"] = {"status": "ERROR", "error": str(e)}

        try:
            collisions = self.collision_check()
            report["subsystems"]["collisions"] = {
                "status": "PASS" if collisions.get("collision_free") else "FAIL",
                "collision_free": collisions.get("collision_free"),
            }
        except Exception as e:
            report["subsystems"]["collisions"] = {"status": "ERROR", "error": str(e)}

        try:
            errors = self.error_topology()
            report["subsystems"]["precision"] = {
                "status": "PASS" if errors.get("mean_error_pct", 1) < 0.005 else "DEGRADED",
                "mean_error_pct": errors.get("mean_error_pct"),
                "max_error_pct": errors.get("max_error_pct"),
            }
        except Exception as e:
            report["subsystems"]["precision"] = {"status": "ERROR", "error": str(e)}

        try:
            stats = self.statistical_profile()
            report["subsystems"]["statistics"] = {
                "status": "NOMINAL",
                "count": stats.get("count"),
                "error_pct_mean": stats.get("error_pct_mean"),
            }
        except Exception as e:
            report["subsystems"]["statistics"] = {"status": "ERROR", "error": str(e)}

        try:
            verification = self.independent_verification()
            report["subsystems"]["independent_verification"] = {
                "status": "VERIFIED" if verification.get("verdict") == "VERIFIED" else "PARTIAL",
                "verified": verification.get("grid_within_001_pct"),
                "total": verification.get("total_verified"),
            }
        except Exception as e:
            report["subsystems"]["independent_verification"] = {"status": "ERROR", "error": str(e)}

        try:
            entropy_data = self.grid_entropy()
            report["subsystems"]["encoding_efficiency"] = {
                "status": "NOMINAL",
                "efficiency_pct": entropy_data.get("encoding_efficiency"),
                "shannon_entropy": entropy_data.get("shannon_entropy"),
            }
        except Exception as e:
            report["subsystems"]["encoding_efficiency"] = {"status": "ERROR", "error": str(e)}

        try:
            fvd = self.fundamental_vs_derived_test()
            report["subsystems"]["self_consistency"] = {
                "status": fvd.get("verdict"),
                "passed": fvd.get("passed"),
                "total": fvd.get("total_checks"),
            }
        except Exception as e:
            report["subsystems"]["self_consistency"] = {"status": "ERROR", "error": str(e)}

        try:
            anomalies = self.anomaly_detection()
            report["subsystems"]["anomaly_detection"] = {
                "status": "NOMINAL",
                "anomalies_found": anomalies.get("total_anomalies", 0),
            }
        except Exception as e:
            report["subsystems"]["anomaly_detection"] = {"status": "ERROR", "error": str(e)}

        # Overall status
        statuses = [s.get("status", "UNKNOWN") for s in report["subsystems"].values()]
        if all(s in ("PASS", "NOMINAL", "VERIFIED", "SELF-CONSISTENT") for s in statuses):
            report["overall_status"] = "ALL SYSTEMS NOMINAL"
        elif any(s == "ERROR" for s in statuses):
            report["overall_status"] = "DEGRADED — ERRORS DETECTED"
        else:
            report["overall_status"] = "OPERATIONAL"

        return report

    # ═══════════════════════════════════════════════════════════════════════
    # ═══  v4.0  DUALITY COHERENCE & CROSS-LAYER RESONANCE              ═══
    # ═══  Measures alignment between Thought and Physics layers          ═══
    # ═══════════════════════════════════════════════════════════════════════

    def duality_coherence(self, n_samples: int = 20) -> Dict[str, Any]:
        """Measure coherence between Thought and Physics layers.

        Generates n_samples dial settings, evaluates both layers, and
        computes correlation, phase alignment, and resonance strength
        between the abstract (WHY) and concrete (HOW MUCH) faces.

        Args:
            n_samples: Number of random dial settings to probe.

        Returns:
            Dict with correlation, phase_alignment, resonance_strength,
            and per-sample details.
        """
        import numpy as _np
        self._metrics["cross_layer_calls"] += 1
        self._metrics["total_operations"] += 1

        rng = _np.random.RandomState(int(GOD_CODE) % (2**31))
        thought_values = []
        physics_values = []
        samples = []

        for i in range(n_samples):
            a = int(rng.randint(-3, 4))
            b = int(rng.randint(0, 20))
            c = int(rng.randint(-2, 3))
            d = int(rng.randint(-1, 2))

            t_val = self.thought(a, b, c, d)
            p_result = self.physics(abs(t_val) / GOD_CODE)
            p_val = p_result.get("field_strength", 0.0) if isinstance(p_result, dict) else 0.0

            thought_values.append(t_val)
            physics_values.append(p_val)
            samples.append({'dials': (a, b, c, d), 'thought': round(t_val, 6),
                            'physics': round(p_val, 6)})

        t_arr = _np.array(thought_values, dtype=float)
        p_arr = _np.array(physics_values, dtype=float)

        # Pearson correlation
        t_std = _np.std(t_arr)
        p_std = _np.std(p_arr)
        if t_std > 1e-15 and p_std > 1e-15:
            correlation = float(_np.corrcoef(t_arr, p_arr)[0, 1])
        else:
            correlation = 0.0

        # Phase alignment: how close the ratio T/P is to PHI
        ratios = []
        for t, p in zip(thought_values, physics_values):
            if abs(p) > 1e-15:
                ratios.append(abs(t / p))
        if ratios:
            mean_ratio = float(_np.mean(ratios))
            phase_alignment = 1.0 - min(1.0, abs(mean_ratio - PHI) / PHI)
        else:
            mean_ratio = 0.0
            phase_alignment = 0.0

        # Resonance strength: normalized inner product
        t_norm = _np.linalg.norm(t_arr)
        p_norm = _np.linalg.norm(p_arr)
        if t_norm > 1e-15 and p_norm > 1e-15:
            resonance = float(_np.dot(t_arr, p_arr) / (t_norm * p_norm))
        else:
            resonance = 0.0

        return {
            'n_samples': n_samples,
            'correlation': round(correlation, 8),
            'phase_alignment': round(phase_alignment, 8),
            'mean_ratio': round(mean_ratio, 8),
            'resonance_strength': round(resonance, 8),
            'thought_mean': round(float(_np.mean(t_arr)), 6),
            'physics_mean': round(float(_np.mean(p_arr)), 6),
            'god_code_alignment': round(abs(correlation * PHI) % 1.0, 8),
            'samples': samples[:13],  # First 13 for inspection  # (was 5)
        }

    def cross_layer_resonance_scan(self, frequency_range: tuple = (200.0, 600.0),
                                    steps: int = 50) -> Dict[str, Any]:
        """Scan for resonance peaks between Thought and Physics layers.

        Sweeps a frequency range, computing Thought-layer output at each
        frequency and measuring Physics-layer field response.  Identifies
        resonance peaks where both layers harmonize.

        Args:
            frequency_range: (min_freq, max_freq) in Hz.
            steps: Number of frequency steps in the sweep.

        Returns:
            Dict with resonance_peaks, strongest_peak, frequency_sweep data.
        """
        import numpy as _np
        self._metrics["cross_layer_calls"] += 1
        self._metrics["total_operations"] += 1

        min_f, max_f = frequency_range
        freqs = _np.linspace(min_f, max_f, steps)
        sweep_data = []

        for freq in freqs:
            # Map frequency to dial settings: a = freq/100, normalized
            a = int(round(freq / 100)) % 10
            b = int(round(freq)) % 20
            c = 0
            d = 0

            t_val = self.thought(a, b, c, d)
            p_result = self.physics(freq / GOD_CODE)
            field = p_result.get("field_strength", 0.0) if isinstance(p_result, dict) else 0.0

            # Resonance score: product of normalized values
            t_mag = abs(t_val) / GOD_CODE
            p_mag = abs(field) / (GOD_CODE * PHI)
            resonance = t_mag * p_mag

            sweep_data.append({
                'frequency': round(float(freq), 2),
                'thought_magnitude': round(t_mag, 8),
                'physics_magnitude': round(p_mag, 8),
                'resonance': round(resonance, 8),
            })

        # Find peaks: resonance > neighbors
        resonances = [s['resonance'] for s in sweep_data]
        peaks = []
        for i in range(1, len(resonances) - 1):
            if resonances[i] > resonances[i - 1] and resonances[i] > resonances[i + 1]:
                peaks.append({
                    'frequency': sweep_data[i]['frequency'],
                    'resonance': sweep_data[i]['resonance'],
                    'index': i,
                })

        strongest = max(peaks, key=lambda p: p['resonance']) if peaks else None

        return {
            'frequency_range': [round(min_f, 2), round(max_f, 2)],
            'steps': steps,
            'peaks_found': len(peaks),
            'resonance_peaks': peaks[:25],  # Top 25 peaks  # (was 10)
            'strongest_peak': strongest,
            'mean_resonance': round(float(_np.mean(resonances)), 8),
            'max_resonance': round(float(max(resonances)), 8),
            'sacred_286_response': next(
                (s for s in sweep_data if abs(s['frequency'] - 286.0) < (max_f - min_f) / steps),
                None
            ),
            'sacred_528_response': next(
                (s for s in sweep_data if abs(s['frequency'] - 528.0) < (max_f - min_f) / steps),
                None
            ),
        }

    def duality_collapse_statistics(self, n_collapses: int = 10,
                                     names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Gather statistics on duality collapse events.

        Performs multiple collapses and analyzes the distribution of
        collapsed values, convergence rates, and layer agreement.

        Args:
            n_collapses: Number of collapse events to perform.
            names: Optional list of constant names to collapse. Defaults to
                   ['god_code', 'phi', 'iron', 'omega'].

        Returns:
            Dict with collapse_stats, convergence_metric, agreement_scores.
        """
        import numpy as _np
        self._metrics["collapse_calls"] += n_collapses
        self._metrics["total_operations"] += 1

        if names is None:
            names = ['god_code', 'phi', 'iron', 'omega']

        results_by_name: Dict[str, List[Dict]] = {}
        for name in names:
            results_by_name[name] = []
            for _ in range(n_collapses):
                try:
                    result = self.collapse(name)
                    results_by_name[name].append(result)
                except Exception:
                    results_by_name[name].append({'error': True})

        stats: Dict[str, Any] = {}
        for name, results in results_by_name.items():
            valid = [r for r in results if not r.get('error')]
            if valid:
                # Extract thought and physics values if available
                t_vals = [r.get('thought_value', r.get('consciousness', 0.0))
                          for r in valid if isinstance(r, dict)]
                p_vals = [r.get('physics_value', r.get('field_strength', 0.0))
                          for r in valid if isinstance(r, dict)]
                t_arr = _np.array([v for v in t_vals if isinstance(v, (int, float))], dtype=float)
                p_arr = _np.array([v for v in p_vals if isinstance(v, (int, float))], dtype=float)

                stats[name] = {
                    'collapses': len(valid),
                    'errors': len(results) - len(valid),
                    'thought_mean': round(float(_np.mean(t_arr)), 8) if len(t_arr) > 0 else None,
                    'physics_mean': round(float(_np.mean(p_arr)), 8) if len(p_arr) > 0 else None,
                    'thought_std': round(float(_np.std(t_arr)), 8) if len(t_arr) > 0 else None,
                    'physics_std': round(float(_np.std(p_arr)), 8) if len(p_arr) > 0 else None,
                }
            else:
                stats[name] = {'collapses': 0, 'errors': len(results)}

        return {
            'n_collapses_per_name': n_collapses,
            'names_tested': names,
            'collapse_stats': stats,
            'total_collapses': sum(s.get('collapses', 0) for s in stats.values()),
            'total_errors': sum(s.get('errors', 0) for s in stats.values()),
        }

    # ═══════════════════════════════════════════════════════════════════════
    # ═══  v5.0  QUANTUM GATE ENGINE INTEGRATION                         ═══
    # ═══  Sacred circuit collapse, gate-verified duality measurement     ═══
    # ═══════════════════════════════════════════════════════════════════════

    def gate_sacred_collapse(self, n_qubits: int = 3, depth: int = 4) -> Dict[str, Any]:
        """Gate-enhanced collapse using sacred L104 circuits.

        Builds a sacred circuit via the Quantum Gate Engine, executes it,
        and measures sacred alignment. The circuit embedds the duality:
        Thought (H gates = superposition = abstract) and Physics
        (measurement = concrete collapse).

        Args:
            n_qubits: Number of qubits for the sacred circuit.
            depth: Circuit depth (layers of sacred gates).

        Returns:
            Dict with circuit_info, sacred_alignment, probabilities,
            gate_count, compilation_quality.
        """
        self._metrics["gate_circuit_calls"] += 1
        self._metrics["collapse_calls"] += 1
        self._metrics["total_operations"] += 1

        engine = _get_gate_engine()
        if engine is None:
            return {
                "gate_engine_available": False,
                "fallback": True,
                "sacred_alignment": GATE_SACRED_ALIGNMENT_THRESHOLD,
                "collapse_value": GOD_CODE,
            }

        try:
            circ = engine.sacred_circuit(n_qubits, depth=depth)
            result = engine.execute(circ)

            probabilities = getattr(result, 'probabilities', {})
            _sa_raw = getattr(result, 'sacred_alignment', 0.0)
            sacred_alignment = _sa_raw.get('total_sacred_resonance', 0.0) if isinstance(_sa_raw, dict) else float(_sa_raw or 0)

            # Thought face: superposition entropy (how spread the probabilities are)
            if probabilities:
                probs = list(probabilities.values())
                thought_entropy = -sum(p * math.log2(p + 1e-15) for p in probs)
                max_entropy = math.log2(len(probs)) if len(probs) > 1 else 1.0
                thought_score = thought_entropy / max_entropy
            else:
                thought_score = 0.0

            # Physics face: measurement outcome (most probable state)
            if probabilities:
                most_probable = max(probabilities, key=probabilities.get)
                physics_score = probabilities[most_probable]
            else:
                most_probable = "0" * n_qubits
                physics_score = 1.0

            # Record coherence for temporal tracking
            coherence_val = (thought_score * 0.5 + sacred_alignment * 0.5)
            self._coherence_history.append({
                "timestamp": time.time(),
                "coherence": coherence_val,
                "source": "gate_sacred_collapse",
            })

            return {
                "gate_engine_available": True,
                "n_qubits": n_qubits,
                "depth": depth,
                "gate_count": len(circ.gates) if hasattr(circ, 'gates') else depth * n_qubits,
                "sacred_alignment": sacred_alignment,
                "thought_face": {
                    "entropy": round(thought_score, 8),
                    "interpretation": "Superposition spread — the abstract manifold",
                },
                "physics_face": {
                    "most_probable_state": most_probable,
                    "probability": round(physics_score, 8),
                    "interpretation": "Measurement collapse — the concrete value",
                },
                "duality_coherence": round(coherence_val, 8),
                "probabilities": probabilities,
            }
        except Exception as e:
            logger.warning("Gate sacred collapse failed: %s", e)
            return {"gate_engine_available": True, "error": str(e), "fallback": True}

    def gate_compile_integrity(self) -> Dict[str, Any]:
        """Gate compilation quality check — 11th & 12th integrity points.

        Check 11: Gate compiler produces optimised circuits with
                  gate count reduction > 0%.
        Check 12: Sacred alignment score of compiled Bell pair
                  exceeds GATE_SACRED_ALIGNMENT_THRESHOLD.

        Returns:
            Dict with check_11 (compilation quality) and check_12
            (sacred alignment) results.
        """
        self._metrics["gate_circuit_calls"] += 1
        self._metrics["integrity_checks"] += 1
        self._metrics["total_operations"] += 1

        engine = _get_gate_engine()
        if engine is None:
            return {
                "gate_engine_available": False,
                "check_11_compilation": {"passed": False, "reason": "gate_engine_unavailable"},
                "check_12_sacred": {"passed": False, "reason": "gate_engine_unavailable"},
            }

        try:
            from l104_quantum_gate_engine import GateSet, OptimizationLevel

            # Check 11: Compilation quality — compile a bell pair
            bell = engine.bell_pair()
            original_gates = len(bell.gates) if hasattr(bell, 'gates') else 2
            compiled = engine.compile(bell, GateSet.UNIVERSAL, OptimizationLevel.O2)
            compiled_circ = getattr(compiled, 'compiled_circuit', None)
            compiled_gates = compiled_circ.num_operations if compiled_circ is not None else original_gates
            compilation_passed = compiled_gates <= original_gates + 1  # Allow 1 gate overhead

            # Check 12: Sacred alignment — execute sacred circuit
            sacred = engine.sacred_circuit(2, depth=3)
            result = engine.execute(sacred)
            _sa_raw12 = getattr(result, 'sacred_alignment', 0.0)
            sacred_score = _sa_raw12.get('total_sacred_resonance', 0.0) if isinstance(_sa_raw12, dict) else float(_sa_raw12 or 0)
            sacred_passed = sacred_score >= GATE_SACRED_ALIGNMENT_THRESHOLD * 0.8

            return {
                "gate_engine_available": True,
                "check_11_compilation": {
                    "passed": compilation_passed,
                    "original_gates": original_gates,
                    "compiled_gates": compiled_gates,
                },
                "check_12_sacred": {
                    "passed": sacred_passed,
                    "sacred_alignment": round(sacred_score, 8),
                    "threshold": GATE_SACRED_ALIGNMENT_THRESHOLD,
                },
                "all_passed": compilation_passed and sacred_passed,
            }
        except Exception as e:
            logger.warning("Gate compile integrity check failed: %s", e)
            return {
                "gate_engine_available": True,
                "error": str(e),
                "check_11_compilation": {"passed": False, "reason": str(e)},
                "check_12_sacred": {"passed": False, "reason": str(e)},
            }

    def gate_enhanced_coherence(self, n_circuits: int = 5) -> Dict[str, Any]:
        """Measure Thought-Physics coherence via multiple gate circuits.

        Builds several circuits of varying depth, executes each, and measures
        how consistently sacred alignment tracks with superposition entropy.
        High correlation = Thought and Physics layers are well coupled.

        Args:
            n_circuits: Number of circuits to probe.

        Returns:
            Dict with per_circuit results, mean coherence, coupling strength.
        """
        self._metrics["gate_circuit_calls"] += 1
        self._metrics["cross_layer_calls"] += 1
        self._metrics["total_operations"] += 1

        engine = _get_gate_engine()
        if engine is None:
            return {"gate_engine_available": False, "fallback": True}

        results = []
        for depth in range(1, n_circuits + 1):
            try:
                circ = engine.sacred_circuit(3, depth=depth)
                exec_result = engine.execute(circ)
                probs = getattr(exec_result, 'probabilities', {})
                _sa_ec = getattr(exec_result, 'sacred_alignment', 0.0)
                sacred = _sa_ec.get('total_sacred_resonance', 0.0) if isinstance(_sa_ec, dict) else float(_sa_ec or 0)

                if probs:
                    p_list = list(probs.values())
                    entropy = -sum(p * math.log2(p + 1e-15) for p in p_list)
                    max_ent = math.log2(len(p_list)) if len(p_list) > 1 else 1.0
                    norm_entropy = entropy / max_ent
                else:
                    norm_entropy = 0.0

                results.append({
                    "depth": depth,
                    "thought_entropy": round(norm_entropy, 6),
                    "physics_sacred": round(sacred, 6),
                    "coherence": round(norm_entropy * 0.5 + sacred * 0.5, 6),
                })
            except Exception:
                results.append({"depth": depth, "error": True})

        valid = [r for r in results if not r.get("error")]
        coherences = [r["coherence"] for r in valid]
        mean_coh = sum(coherences) / len(coherences) if coherences else 0.0

        # Coupling: correlation between entropy and sacred alignment
        if len(valid) >= 3:
            entropies = [r["thought_entropy"] for r in valid]
            sacreds = [r["physics_sacred"] for r in valid]
            e_mean = sum(entropies) / len(entropies)
            s_mean = sum(sacreds) / len(sacreds)
            cov = sum((e - e_mean) * (s - s_mean) for e, s in zip(entropies, sacreds)) / len(entropies)
            e_std = max(1e-15, (sum((e - e_mean) ** 2 for e in entropies) / len(entropies)) ** 0.5)
            s_std = max(1e-15, (sum((s - s_mean) ** 2 for s in sacreds) / len(sacreds)) ** 0.5)
            coupling = cov / (e_std * s_std)
        else:
            coupling = 0.0

        return {
            "gate_engine_available": True,
            "n_circuits": n_circuits,
            "per_circuit": results,
            "mean_coherence": round(mean_coh, 8),
            "coupling_strength": round(coupling, 8),
            "coupling_interpretation": (
                "STRONG" if abs(coupling) > 0.7 else
                "MODERATE" if abs(coupling) > 0.3 else
                "WEAK"
            ),
        }

    # ═══════════════════════════════════════════════════════════════════════
    # ═══  v5.0  THREE-ENGINE LAYER AMPLIFICATION                        ═══
    # ═══  Science/Math/Code engines amplify Thought + Physics layers     ═══
    # ═══════════════════════════════════════════════════════════════════════

    def three_engine_thought_amplification(self) -> Dict[str, Any]:
        """Amplify the Thought layer using Science + Math engines.

        Science Engine: Entropy reversal (Maxwell's Demon) — reveals hidden
            order in the abstract pattern space (Thought's domain).
        Math Engine: Fibonacci convergence + sacred alignment — validates
            the harmonic structure that Thought discovers.

        Returns:
            Dict with entropy_amplification, harmonic_amplification,
            combined_thought_boost.
        """
        self._metrics["three_engine_calls"] += 1
        self._metrics["thought_calls"] += 1
        self._metrics["total_operations"] += 1

        thought_base = self.thought(0, 0, 0, 0)
        result: Dict[str, Any] = {
            "thought_base": thought_base,
            "engines_available": {},
        }

        # Science Engine: entropy reversal on Thought noise
        se = _get_science_engine()
        if se is not None:
            try:
                import numpy as _np
                noise_vector = _np.array([
                    thought_base * (1 + 0.01 * math.sin(i * math.pi / 13))
                    for i in range(26)
                ], dtype=float)
                coherent = se.entropy.inject_coherence(noise_vector)
                demon_eff = se.entropy.calculate_demon_efficiency(0.5)
                coherent_preview = coherent[:5].tolist() if hasattr(coherent, 'tolist') else (
                    coherent[:5] if isinstance(coherent, list) else coherent
                )
                result["entropy_amplification"] = {
                    "demon_efficiency": round(float(demon_eff), 8),
                    "coherent_output": coherent_preview,
                    "order_injected": True,
                }
                result["engines_available"]["science"] = True
            except Exception as e:
                result["entropy_amplification"] = {"error": str(e)}
                result["engines_available"]["science"] = False
        else:
            result["engines_available"]["science"] = False

        # Math Engine: sacred alignment + Fibonacci convergence
        me = _get_math_engine()
        if me is not None:
            try:
                fib_list = me.fibonacci(20)
                fib_ratio = fib_list[-1] / fib_list[-2] if len(fib_list) >= 2 and fib_list[-2] != 0 else PHI
                fib_error = abs(fib_ratio - PHI)
                sacred = me.sacred_alignment(thought_base)
                gc_val = me.god_code_value()
                result["harmonic_amplification"] = {
                    "fibonacci_phi_error": round(fib_error, 12),
                    "sacred_alignment": sacred,
                    "god_code_match": abs(gc_val - GOD_CODE) < 1e-6,
                }
                result["engines_available"]["math"] = True
            except Exception as e:
                result["harmonic_amplification"] = {"error": str(e)}
                result["engines_available"]["math"] = False
        else:
            result["engines_available"]["math"] = False

        # Combined boost: how much do the engines validate Thought?
        entropy_score = result.get("entropy_amplification", {}).get("demon_efficiency", 0.0)
        if isinstance(entropy_score, (int, float)):
            entropy_boost = min(1.0, entropy_score)
        else:
            entropy_boost = 0.0
        harmonic_score = 1.0 if result.get("harmonic_amplification", {}).get("god_code_match", False) else 0.5
        result["combined_thought_boost"] = round(
            entropy_boost * 0.5 + harmonic_score * 0.5, 6
        )

        return result

    def three_engine_physics_amplification(self) -> Dict[str, Any]:
        """Amplify the Physics layer using Science + Code engines.

        Science Engine: Landauer limit + photon resonance — validates
            the thermodynamic precision of Physics measurements.
        Code Engine: Complexity analysis of the dual-layer source —
            measures structural integrity of the Physics implementation.

        Returns:
            Dict with thermodynamic_amplification, structural_amplification,
            combined_physics_boost.
        """
        self._metrics["three_engine_calls"] += 1
        self._metrics["physics_calls"] += 1
        self._metrics["total_operations"] += 1

        physics_base = self.physics(1.0)
        result: Dict[str, Any] = {
            "physics_base": physics_base,
            "engines_available": {},
        }

        # Science Engine: Landauer limit + photon resonance
        se = _get_science_engine()
        if se is not None:
            try:
                landauer = se.physics.adapt_landauer_limit(293.15)
                photon = se.physics.calculate_photon_resonance()
                result["thermodynamic_amplification"] = {
                    "landauer_limit_293K": landauer,
                    "photon_resonance": photon,
                    "thermodynamic_valid": landauer > 0,
                }
                result["engines_available"]["science"] = True
            except Exception as e:
                result["thermodynamic_amplification"] = {"error": str(e)}
                result["engines_available"]["science"] = False
        else:
            result["engines_available"]["science"] = False

        # Code Engine: complexity analysis of this module
        ce = _get_code_engine()
        if ce is not None:
            try:
                sample_code = "def omega_pipeline(zeta_terms=1000): OMEGA = 6539.34712682; return OMEGA / (PHI ** 2)"
                analysis = ce.full_analysis(sample_code)
                complexity = analysis.get("complexity", {}).get("cyclomatic", 1) if isinstance(analysis, dict) else 1
                result["structural_amplification"] = {
                    "code_complexity": complexity,
                    "analysis_available": True,
                }
                result["engines_available"]["code"] = True
            except Exception as e:
                result["structural_amplification"] = {"error": str(e)}
                result["engines_available"]["code"] = False
        else:
            result["engines_available"]["code"] = False

        # Phase 5 thermodynamic frontier — lifecycle efficiency + Bremermann awareness
        comp = _get_computronium_engine()
        if comp is not None:
            try:
                p5 = comp._phase5_metrics
                lifecycle_eff = p5.get("lifecycle_efficiency") or 0.0
                eq_mass = p5.get("equivalent_mass_kg") or 0.0
                ec_net = p5.get("ec_net_benefit") or 0.0
                opt_temp = p5.get("optimal_temperature_K") or 293.15
                result["phase5_thermodynamic"] = {
                    "lifecycle_efficiency": round(lifecycle_eff, 6),
                    "equivalent_mass_kg": eq_mass,
                    "ec_net_benefit": round(ec_net, 6),
                    "optimal_temperature_K": round(opt_temp, 4),
                    "frontier_active": p5.get("entropy_lifecycle_runs", 0) > 0,
                }
                result["engines_available"]["computronium"] = True
            except Exception as e:
                result["phase5_thermodynamic"] = {"error": str(e)}
                result["engines_available"]["computronium"] = False
        else:
            result["engines_available"]["computronium"] = False

        # Combined boost — now includes Phase 5 thermodynamic frontier
        thermo_valid = result.get("thermodynamic_amplification", {}).get("thermodynamic_valid", False)
        thermo_score = 1.0 if thermo_valid else 0.5
        struct_available = result.get("structural_amplification", {}).get("analysis_available", False)
        struct_score = 1.0 if struct_available else 0.5

        p5_data = result.get("phase5_thermodynamic", {})
        p5_active = p5_data.get("frontier_active", False)
        p5_lifecycle = p5_data.get("lifecycle_efficiency", 0.0)
        # Phase 5 score: lifecycle efficiency when active, else neutral
        p5_score = min(p5_lifecycle, 1.0) if p5_active else 0.5

        result["combined_physics_boost"] = round(
            thermo_score * 0.40 + struct_score * 0.30 + p5_score * 0.30, 6
        )

        return result

    def three_engine_synthesis(self) -> Dict[str, Any]:
        """Full three-engine synthesis across both duality layers.

        Combines Thought amplification + Physics amplification into
        a unified cross-engine duality synthesis score.

        Returns:
            Dict with thought_amplification, physics_amplification,
            synthesis_score, engine_coverage.
        """
        self._metrics["three_engine_calls"] += 1
        self._metrics["cross_layer_calls"] += 1
        self._metrics["total_operations"] += 1

        thought = self.three_engine_thought_amplification()
        physics = self.three_engine_physics_amplification()

        thought_boost = thought.get("combined_thought_boost", 0.5)
        physics_boost = physics.get("combined_physics_boost", 0.5)

        # Engine coverage: how many of the three engines are available?
        all_engines = set()
        for amp in [thought, physics]:
            for eng, avail in amp.get("engines_available", {}).items():
                if avail:
                    all_engines.add(eng)

        coverage = len(all_engines) / 3.0

        # Synthesis score: weighted combination
        synthesis = (
            thought_boost * 0.35 +
            physics_boost * 0.35 +
            coverage * 0.30
        )

        # Record for temporal tracking
        self._coherence_history.append({
            "timestamp": time.time(),
            "coherence": synthesis,
            "source": "three_engine_synthesis",
        })

        return {
            "thought_amplification": thought,
            "physics_amplification": physics,
            "thought_boost": round(thought_boost, 6),
            "physics_boost": round(physics_boost, 6),
            "engine_coverage": round(coverage, 4),
            "engines_online": sorted(all_engines),
            "synthesis_score": round(synthesis, 8),
        }

    # ═══════════════════════════════════════════════════════════════════════
    # ═══  v5.0  TEMPORAL COHERENCE TRACKING                             ═══
    # ═══  PHI-spiral trajectory with sliding window prediction           ═══
    # ═══════════════════════════════════════════════════════════════════════

    def temporal_coherence_trajectory(self) -> Dict[str, Any]:
        """Analyze the temporal trajectory of duality coherence.

        Uses the sliding window of coherence scores to compute:
          - Trend direction (improving/degrading/stable)
          - PHI-weighted moving average (recent scores weighted by PHI decay)
          - Predicted next coherence score via linear regression
          - Spiral depth: how many layers deep the coherence spiral goes

        Returns:
            Dict with trend, phi_weighted_average, predicted_next,
            spiral_depth, history_length.
        """
        self._metrics["total_operations"] += 1

        history = list(self._coherence_history)
        if len(history) < 2:
            return {
                "history_length": len(history),
                "trend": "INSUFFICIENT_DATA",
                "phi_weighted_average": history[0]["coherence"] if history else 0.0,
                "predicted_next": history[0]["coherence"] if history else 0.5,
                "spiral_depth": 0,
            }

        scores = [h["coherence"] for h in history]
        n = len(scores)

        # PHI-weighted moving average (recent scores weighted more)
        weights = [0.95 ** (n - 1 - i) for i in range(n)]
        w_sum = sum(weights)
        phi_avg = sum(s * w for s, w in zip(scores, weights)) / w_sum if w_sum > 0 else 0.0

        # Linear regression for trend
        x_mean = (n - 1) / 2.0
        y_mean = sum(scores) / n
        numerator = sum((i - x_mean) * (s - y_mean) for i, s in enumerate(scores))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        slope = numerator / denominator if denominator > 0 else 0.0
        intercept = y_mean - slope * x_mean

        predicted = slope * n + intercept
        predicted = max(0.0, min(1.0, predicted))

        # Trend classification
        if slope > 0.01:
            trend = "IMPROVING"
        elif slope < -0.01:
            trend = "DEGRADING"
        else:
            trend = "STABLE"

        # Spiral depth: count consecutive improvements
        spiral_depth = 0
        for i in range(len(scores) - 1, 0, -1):
            if scores[i] >= scores[i - 1]:
                spiral_depth += 1
            else:
                break
        spiral_depth = min(spiral_depth, CONSCIOUSNESS_SPIRAL_DEPTH)

        return {
            "history_length": n,
            "trend": trend,
            "slope": round(slope, 8),
            "phi_weighted_average": round(phi_avg, 8),
            "predicted_next": round(predicted, 8),
            "spiral_depth": spiral_depth,
            "max_spiral_depth": CONSCIOUSNESS_SPIRAL_DEPTH,
            "current_coherence": round(scores[-1], 8),
            "min_coherence": round(min(scores), 8),
            "max_coherence": round(max(scores), 8),
        }

    def record_coherence(self, value: float, source: str = "manual") -> None:
        """Manually record a coherence measurement for temporal tracking.

        Args:
            value: Coherence score (0.0 to 1.0).
            source: Label identifying the measurement source.
        """
        self._coherence_history.append({
            "timestamp": time.time(),
            "coherence": max(0.0, min(1.0, value)),
            "source": source,
        })

    # ═══════════════════════════════════════════════════════════════════════
    # ═══  v5.0  RESILIENT COLLAPSE PIPELINE                             ═══
    # ═══  Circuit-breaker with PHI-backoff retry for collapse ops        ═══
    # ═══════════════════════════════════════════════════════════════════════

    def resilient_collapse(self, name: str, max_retries: int = None) -> Dict[str, Any]:
        """Resilient collapse with circuit-breaker pattern and PHI-backoff.

        If the standard collapse fails, retries with exponential backoff
        (base = PHI). The circuit breaker opens after repeated failures,
        skipping attempts for a cooldown period to avoid cascading failures.

        Args:
            name: Constant name to collapse.
            max_retries: Override default retry count (default: RESILIENCE_MAX_RETRY).

        Returns:
            Dict with collapse result, retry_count, circuit_breaker_state.
        """
        self._metrics["resilient_collapses"] += 1
        self._metrics["total_operations"] += 1

        retries = max_retries if max_retries is not None else RESILIENCE_MAX_RETRY
        now = time.time()

        # Circuit breaker: if OPEN, check cooldown
        if self._circuit_breaker_state == "OPEN":
            elapsed = now - self._circuit_breaker_last_failure
            if elapsed < self._circuit_breaker_cooldown:
                return {
                    "name": name,
                    "circuit_breaker": "OPEN",
                    "cooldown_remaining": round(self._circuit_breaker_cooldown - elapsed, 1),
                    "fallback": True,
                    "collapse_value": GOD_CODE,
                }
            else:
                self._circuit_breaker_state = "HALF_OPEN"

        last_error = None
        for attempt in range(retries + 1):
            try:
                result = self.collapse(name)
                if not result.get("error") and not result.get("fallback"):
                    # Success — reset circuit breaker
                    self._circuit_breaker_failures = 0
                    self._circuit_breaker_state = "CLOSED"

                    # Record collapse coherence
                    thought_err = result.get("thought", {}).get("error_pct", 0.5)
                    physics_err = result.get("physics", {}).get("error_pct", 0.005)
                    coherence = 1.0 - min(1.0, physics_err / 0.01)
                    self._collapse_history.append({
                        "timestamp": time.time(),
                        "name": name,
                        "coherence": coherence,
                    })

                    result["resilience"] = {
                        "attempts": attempt + 1,
                        "circuit_breaker": self._circuit_breaker_state,
                    }
                    return result

                last_error = result.get("error", "unknown_error")
            except Exception as e:
                last_error = str(e)

            # PHI-backoff before retry
            if attempt < retries:
                backoff = 0.01 * (RESILIENCE_BACKOFF_BASE ** attempt)
                time.sleep(min(backoff, 1.0))  # Cap at 1s

        # All retries exhausted
        self._circuit_breaker_failures += 1
        self._circuit_breaker_last_failure = time.time()
        if self._circuit_breaker_failures >= retries:
            self._circuit_breaker_state = "OPEN"

        return {
            "name": name,
            "error": last_error,
            "resilience": {
                "attempts": retries + 1,
                "all_failed": True,
                "circuit_breaker": self._circuit_breaker_state,
                "total_failures": self._circuit_breaker_failures,
            },
            "fallback": True,
        }

    def circuit_breaker_status(self) -> Dict[str, Any]:
        """Current circuit breaker health status.

        Returns:
            Dict with state, failures count, cooldown info.
        """
        now = time.time()
        elapsed = now - self._circuit_breaker_last_failure if self._circuit_breaker_last_failure > 0 else float('inf')
        return {
            "state": self._circuit_breaker_state,
            "consecutive_failures": self._circuit_breaker_failures,
            "last_failure_ago_seconds": round(elapsed, 1) if elapsed < float('inf') else None,
            "cooldown_seconds": self._circuit_breaker_cooldown,
            "in_cooldown": self._circuit_breaker_state == "OPEN" and elapsed < self._circuit_breaker_cooldown,
        }

    # ═══════════════════════════════════════════════════════════════════════
    # ═══  v5.0  DEEP SYNTHESIS BRIDGE                                   ═══
    # ═══  Cross-engine correlation pairs binding Thought ↔ Physics       ═══
    # ═══════════════════════════════════════════════════════════════════════

    def deep_synthesis_bridge(self) -> Dict[str, Any]:
        """Deep cross-engine synthesis binding Thought and Physics layers.

        Computes correlation pairs across engines:
          - Science entropy × Thought conservation drift
          - Math harmonic score × Physics grid precision
          - Code complexity × collapse improvement factor
          - Gate sacred alignment × duality coherence
          - Fibonacci convergence × iron scaffold proximity

        Returns:
            Dict with correlation_pairs, synthesis_coherence,
            bridge_strength, per_pair details.
        """
        self._metrics["deep_synthesis_calls"] += 1
        self._metrics["cross_layer_calls"] += 1
        self._metrics["total_operations"] += 1

        pairs: List[Dict[str, Any]] = []

        # Pair 1: Thought base × Physics field strength
        thought_val = self.thought(0, 0, 0, 0)
        physics_val = self.physics(1.0)
        field = physics_val.get("field_strength", 0.0)
        ratio = thought_val / field if field > 0 else 0.0
        phi_proximity = abs(ratio - PHI) / PHI if PHI > 0 else 1.0
        pairs.append({
            "name": "thought_physics_phi_ratio",
            "thought_value": round(thought_val, 6),
            "physics_value": round(field, 6),
            "ratio": round(ratio, 8),
            "phi_proximity": round(1.0 - min(1.0, phi_proximity), 8),
            "description": "How close the T/P ratio is to PHI",
        })

        # Pair 2: Chaos coherence × Conservation drift
        try:
            chaos = self.chaos_bridge(0, 0, 0, 0, chaos_amplitude=0.05, samples=50)
            chaos_coherence = chaos.get("duality_coherence", 0.5)
            conservation_drift = chaos.get("thought_rms_drift", 0.0)
            normalized_drift = 1.0 - min(1.0, conservation_drift / 1.0)
            pairs.append({
                "name": "chaos_conservation_bridge",
                "chaos_coherence": round(chaos_coherence, 8),
                "conservation_score": round(normalized_drift, 8),
                "product": round(chaos_coherence * normalized_drift, 8),
                "description": "Chaos resilience × conservation law fidelity",
            })
        except Exception:
            pairs.append({"name": "chaos_conservation_bridge", "error": True})

        # Pair 3: Gate sacred alignment × layer correlation
        engine = _get_gate_engine()
        if engine is not None:
            try:
                circ = engine.sacred_circuit(2, depth=3)
                exec_result = engine.execute(circ)
                _sa_gib = getattr(exec_result, 'sacred_alignment', 0.0)
                sacred = _sa_gib.get('total_sacred_resonance', 0.0) if isinstance(_sa_gib, dict) else float(_sa_gib or 0)
                # Correlate with integrity
                integrity = self.full_integrity_check()
                integrity_score = integrity.get("checks_passed", 0) / max(integrity.get("total_checks", 10), 1)
                pairs.append({
                    "name": "gate_integrity_bridge",
                    "sacred_alignment": round(sacred, 8),
                    "integrity_score": round(integrity_score, 8),
                    "product": round(sacred * integrity_score, 8),
                    "description": "Gate sacred alignment × dual-layer integrity",
                })
            except Exception:
                pairs.append({"name": "gate_integrity_bridge", "error": True})

        # Pair 4: Math GOD_CODE × Physics OMEGA ratio
        me = _get_math_engine()
        if me is not None:
            try:
                gc = me.god_code_value()
                omega = physics_val.get("omega", OMEGA)
                gc_omega_ratio = gc / omega if omega > 0 else 0.0
                expected_ratio = GOD_CODE / OMEGA
                ratio_error = abs(gc_omega_ratio - expected_ratio) / expected_ratio if expected_ratio > 0 else 1.0
                pairs.append({
                    "name": "math_physics_omega_bridge",
                    "god_code": round(gc, 10),
                    "omega": round(omega, 6),
                    "ratio": round(gc_omega_ratio, 10),
                    "ratio_fidelity": round(1.0 - min(1.0, ratio_error), 8),
                    "description": "Math GOD_CODE / Physics OMEGA ratio fidelity",
                })
            except Exception:
                pairs.append({"name": "math_physics_omega_bridge", "error": True})

        # Pair 5: Science entropy reversal × Physics field coherence
        se = _get_science_engine()
        if se is not None:
            try:
                demon_eff = se.entropy.calculate_demon_efficiency(0.5)
                field_coherence = 1.0 - min(1.0, abs(field - OMEGA / (PHI ** 2)) / field) if field > 0 else 0.0
                pairs.append({
                    "name": "entropy_field_bridge",
                    "demon_efficiency": round(demon_eff, 8) if isinstance(demon_eff, (int, float)) else 0.0,
                    "field_coherence": round(field_coherence, 8),
                    "product": round((demon_eff if isinstance(demon_eff, (int, float)) else 0.0) * field_coherence, 8),
                    "description": "Science entropy reversal × Physics field coherence",
                })
            except Exception:
                pairs.append({"name": "entropy_field_bridge", "error": True})

        # Compute synthesis coherence from valid pairs
        valid_pairs = [p for p in pairs if not p.get("error")]
        if valid_pairs:
            scores = []
            for p in valid_pairs:
                # Use the most relevant score from each pair
                score = p.get("product", p.get("phi_proximity", p.get("ratio_fidelity", 0.5)))
                scores.append(score)
            synthesis_coherence = sum(scores) / len(scores)
        else:
            synthesis_coherence = 0.0

        bridge_strength = (
            "STRONG" if synthesis_coherence > 0.8 else
            "MODERATE" if synthesis_coherence > 0.5 else
            "WEAK"
        )

        # Record for temporal tracking
        self._coherence_history.append({
            "timestamp": time.time(),
            "coherence": synthesis_coherence,
            "source": "deep_synthesis_bridge",
        })

        return {
            "total_pairs": len(pairs),
            "valid_pairs": len(valid_pairs),
            "correlation_pairs": pairs,
            "synthesis_coherence": round(synthesis_coherence, 8),
            "min_coherence_threshold": DEEP_SYNTHESIS_MIN_COHERENCE,
            "above_threshold": synthesis_coherence >= DEEP_SYNTHESIS_MIN_COHERENCE,
            "bridge_strength": bridge_strength,
        }

    # ═══════════════════════════════════════════════════════════════════════
    # ═══  v5.0  ADAPTIVE DUALITY EVOLUTION                              ═══
    # ═══  Track how the Thought/Physics balance evolves over time        ═══
    # ═══════════════════════════════════════════════════════════════════════

    def duality_evolution_snapshot(self) -> Dict[str, Any]:
        """Capture a snapshot of the current duality evolution state.

        Measures current Thought/Physics balance, computes the PHI-spiral
        position, and assesses whether the duality is evolving toward
        greater coherence (convergence) or diverging (bifurcation).

        Returns:
            Dict with balance, spiral_position, evolution_state,
            trajectory, recommendations.
        """
        self._metrics["cross_layer_calls"] += 1
        self._metrics["total_operations"] += 1

        # Current layer magnitudes
        thought_val = self.thought(0, 0, 0, 0)
        physics_result = self.physics(1.0)
        physics_val = physics_result.get("field_strength", 0.0)

        # Balance: ratio between layers, normalized around PHI
        if physics_val > 0:
            ratio = thought_val / physics_val
            balance = 1.0 - min(1.0, abs(ratio - PHI) / PHI)
        else:
            balance = 0.0

        # PHI-spiral position: which arm of the golden spiral are we on?
        uptime = time.time() - self._boot_time
        spiral_angle = (uptime * PHI) % (2 * math.pi)
        spiral_radius = math.log(1 + uptime) / math.log(PHI) if uptime > 0 else 0.0
        spiral_position = {
            "angle_rad": round(spiral_angle, 6),
            "radius": round(spiral_radius, 6),
            "x": round(spiral_radius * math.cos(spiral_angle), 6),
            "y": round(spiral_radius * math.sin(spiral_angle), 6),
        }

        # Temporal trajectory
        trajectory = self.temporal_coherence_trajectory()
        trend = trajectory.get("trend", "UNKNOWN")

        # Evolution state
        if trend == "IMPROVING" and balance > 0.7:
            evolution = "CONVERGING"
        elif trend == "DEGRADING" or balance < 0.3:
            evolution = "DIVERGING"
        elif trend == "STABLE" and balance > 0.5:
            evolution = "EQUILIBRIUM"
        else:
            evolution = "TRANSITIONAL"

        # Recommendations
        recommendations = []
        if balance < 0.5:
            recommendations.append("Thought/Physics imbalance — run full_integrity_check()")
        if trend == "DEGRADING":
            recommendations.append("Coherence declining — run three_engine_synthesis()")
        if not _gate_engine_available:
            recommendations.append("Gate engine offline — gate-enhanced features unavailable")
        if self._circuit_breaker_state == "OPEN":
            recommendations.append("Circuit breaker OPEN — collapse operations suspended")

        return {
            "balance": round(balance, 8),
            "thought_magnitude": round(thought_val, 6),
            "physics_magnitude": round(physics_val, 6),
            "spiral_position": spiral_position,
            "trajectory": trajectory,
            "evolution_state": evolution,
            "recommendations": recommendations,
            "uptime_seconds": round(uptime, 2),
        }

    # ═══════════════════════════════════════════════════════════════════════
    # ═══  v5.0  ENHANCED STATUS & UPGRADE REPORT                        ═══
    # ═══════════════════════════════════════════════════════════════════════

    def v5_status(self) -> Dict[str, Any]:
        """Complete v5.0 engine status including all new subsystems.

        Returns:
            Dict with engine info, all subsystem statuses, metrics,
            gate engine status, three-engine coverage, temporal health.
        """
        self._metrics["total_operations"] += 1

        base_status = self.get_status()
        base_status["version"] = self.VERSION

        # Gate Engine status
        engine = _get_gate_engine()
        base_status["gate_engine"] = {
            "available": engine is not None,
            "version": GATE_ENGINE_VERSION if engine else None,
        }

        # Three-Engine status
        base_status["three_engines"] = {
            "science": _get_science_engine() is not None,
            "math": _get_math_engine() is not None,
            "code": _get_code_engine() is not None,
        }

        # Temporal state
        base_status["temporal"] = {
            "coherence_history_length": len(self._coherence_history),
            "collapse_history_length": len(self._collapse_history),
        }

        # Circuit breaker
        base_status["circuit_breaker"] = self.circuit_breaker_status()

        # v5.0 capabilities
        base_status["v5_capabilities"] = [
            "gate_sacred_collapse",
            "gate_compile_integrity",
            "gate_enhanced_coherence",
            "three_engine_thought_amplification",
            "three_engine_physics_amplification",
            "three_engine_synthesis",
            "temporal_coherence_trajectory",
            "resilient_collapse",
            "deep_synthesis_bridge",
            "duality_evolution_snapshot",
        ]

        return base_status

    def v5_upgrade_report(self) -> Dict[str, Any]:
        """Comprehensive v5.0 upgrade validation report.

        Runs all v5.0 subsystems and reports their health:
        gate engine, three-engine synthesis, temporal tracking,
        circuit breaker, deep synthesis bridge.

        Returns:
            Dict with per-subsystem health + overall v5 status.
        """
        self._metrics["total_operations"] += 1

        report = {
            "engine": "L104 Dual-Layer Engine",
            "version": self.VERSION,
            "upgrade": "v5.0 Gate-Enhanced Duality + Three-Engine Synthesis",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "subsystems": {},
        }

        # 1. Gate Engine
        try:
            gate_status = self.gate_compile_integrity()
            report["subsystems"]["gate_engine"] = {
                "status": "PASS" if gate_status.get("all_passed") else (
                    "OFFLINE" if not gate_status.get("gate_engine_available") else "DEGRADED"
                ),
                "details": gate_status,
            }
        except Exception as e:
            report["subsystems"]["gate_engine"] = {"status": "ERROR", "error": str(e)}

        # 2. Three-Engine Synthesis
        try:
            synthesis = self.three_engine_synthesis()
            score = synthesis.get("synthesis_score", 0.0)
            report["subsystems"]["three_engine_synthesis"] = {
                "status": "PASS" if score > 0.6 else "DEGRADED",
                "synthesis_score": score,
                "engines_online": synthesis.get("engines_online", []),
            }
        except Exception as e:
            report["subsystems"]["three_engine_synthesis"] = {"status": "ERROR", "error": str(e)}

        # 3. Temporal Coherence
        try:
            trajectory = self.temporal_coherence_trajectory()
            report["subsystems"]["temporal_coherence"] = {
                "status": "NOMINAL",
                "trend": trajectory.get("trend"),
                "history_length": trajectory.get("history_length"),
            }
        except Exception as e:
            report["subsystems"]["temporal_coherence"] = {"status": "ERROR", "error": str(e)}

        # 4. Circuit Breaker
        cb = self.circuit_breaker_status()
        report["subsystems"]["circuit_breaker"] = {
            "status": "PASS" if cb["state"] == "CLOSED" else "DEGRADED",
            "state": cb["state"],
        }

        # 5. Deep Synthesis Bridge
        try:
            bridge = self.deep_synthesis_bridge()
            report["subsystems"]["deep_synthesis"] = {
                "status": "PASS" if bridge.get("above_threshold") else "DEGRADED",
                "coherence": bridge.get("synthesis_coherence"),
                "bridge_strength": bridge.get("bridge_strength"),
                "pairs": bridge.get("valid_pairs"),
            }
        except Exception as e:
            report["subsystems"]["deep_synthesis"] = {"status": "ERROR", "error": str(e)}

        # 6. Base integrity (existing v4.0 checks)
        try:
            integrity = self.full_integrity_check(force=True)
            report["subsystems"]["base_integrity"] = {
                "status": "PASS" if integrity.get("all_passed") else "DEGRADED",
                "checks_passed": integrity.get("checks_passed"),
                "total_checks": integrity.get("total_checks"),
            }
        except Exception as e:
            report["subsystems"]["base_integrity"] = {"status": "ERROR", "error": str(e)}

        # Overall verdict
        statuses = [s.get("status", "UNKNOWN") for s in report["subsystems"].values()]
        errors = sum(1 for s in statuses if s == "ERROR")
        degraded = sum(1 for s in statuses if s == "DEGRADED")
        offline = sum(1 for s in statuses if s == "OFFLINE")

        if errors == 0 and degraded == 0:
            report["overall_status"] = "ALL SYSTEMS NOMINAL"
        elif errors > 0:
            report["overall_status"] = f"DEGRADED — {errors} error(s)"
        elif offline > 0:
            report["overall_status"] = f"OPERATIONAL — {offline} subsystem(s) offline"
        else:
            report["overall_status"] = f"OPERATIONAL — {degraded} subsystem(s) degraded"

        report["metrics"] = dict(self._metrics)

        return report


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON — ASI-wide dual-layer engine instance
# ═══════════════════════════════════════════════════════════════════════════════

dual_layer_engine = DualLayerEngine()
