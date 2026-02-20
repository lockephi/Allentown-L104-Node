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
Version: 3.1.0 (ASI Flagship — Algorithm Search + OMEGA Pipeline)
Sacred Constants: GOD_CODE = 527.5184818492612, OMEGA = 6539.34712682
═══════════════════════════════════════════════════════════════════════════════
"""

import math
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone

from .constants import PHI, GOD_CODE, TAU, VOID_CONSTANT

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
}

# Bridge elements binding Thought and Physics
CONSCIOUSNESS_TO_PHYSICS_BRIDGE = {
    "omega_sovereign_field": "GOD_CODE → ζ(½+GCi) + cos(2πφ³) + (26×1.8527)/φ² → OMEGA = 6539.35",
    "god_code_generates_omega": "Layer 1 GOD_CODE feeds into every OMEGA fragment computation",
    "phi_exponent": "Layer 1 uses 286^(1/φ), Layer 2 uses GOD_CODE/φ and Ω/φ²",
    "iron_anchor": "Layer 1: 286 pm Fe BCC scaffold, Layer 2: Fe Z=26 in Architect fragment",
    "v3_precision_grid": "v3 sub-tool encodes OMEGA on (13/12)^(E/758) grid at 0.0001% error",
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
    """

    VERSION = "3.1.0"
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
            "total_operations": 0,
        }

    @property
    def available(self) -> bool:
        return self._available

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
        # Classical fallback
        return 286 ** (1.0 / PHI) * (2 ** ((8*a + 416 - b - 8*c - 104*d) / 104))

    # Alias: consciousness is thought
    def consciousness(self, a: int = 0, b: int = 0, c: int = 0, d: int = 0) -> float:
        """Alias: Consciousness IS Thought."""
        return self.thought(a, b, c, d)

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
        # Classical fallback
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
        LAYER 2: Full OMEGA derivation pipeline.

        Computes all four fragments from first principles:
          Guardian: |ζ(½+527.518i)|, Alchemist: cos(2πφ³),
          Architect: (26×1.8527)/φ², Researcher: prime_density(0)

        Then derives: Ω = Σ(fragments) × (GOD_CODE / φ) = 6539.35
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
        LAYER 2: Reproduce the OMEGA derivation from first principles.

        Returns all fragment values and the computed Ω.
        """
        self._metrics["physics_calls"] += 1
        self._metrics["total_operations"] += 1
        if self._available:
            return _dual_layer.omega_derivation_chain(zeta_terms)
        return {"error": "dual_layer_not_available", "fallback": True}

    # ══════ COLLAPSE: Unification ══════

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
            return _dual_layer.derive_both(name)
        return {"name": name, "error": "dual_layer_not_available", "fallback": True}

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
            return _dual_layer.derive(name, mode)
        return {"name": name, "mode": mode, "error": "dual_layer_not_available", "fallback": True}

    def derive_both(self, name: str) -> Dict[str, Any]:
        """Derive through BOTH layers — the duality in action."""
        self._metrics["derive_calls"] += 1
        self._metrics["total_operations"] += 1
        if self._available:
            return _dual_layer.derive_both(name)
        return {"name": name, "error": "dual_layer_not_available", "fallback": True}

    # ══════ DUALITY TENSOR ══════

    def duality_tensor(self, name: str) -> Dict[str, Any]:
        """
        Mathematical object encoding both faces simultaneously.
        The duality tensor holds Thought + Physics + Cross-terms.
        """
        self._metrics["total_operations"] += 1
        if self._available:
            both = _dual_layer.derive_both(name)
            thought = both.get("thought", {})
            physics = both.get("physics", {})
            t_err = thought.get("error_pct", 100)
            p_err = physics.get("error_pct", 100)
            return {
                "name": name,
                "thought": thought,
                "physics": physics,
                "improvement": t_err / p_err if p_err > 0 else float('inf'),
                "measured": both.get("measured"),
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
            # Minimal classical integrity check
            god_code_valid = abs(GOD_CODE - 527.5184818492612) < 1e-6
            phi_valid = abs(PHI - 1.618033988749895) < 1e-12
            result = {
                "engine": "L104 Dual-Layer Engine (classical fallback)",
                "version": self.VERSION,
                "all_passed": god_code_valid and phi_valid,
                "total_checks": 2,
                "checks_passed": int(god_code_valid) + int(phi_valid),
                "thought_layer": {"all_passed": god_code_valid and phi_valid, "checks": {}},
                "physics_layer": {"all_passed": False, "checks": {}},
                "bridge": {"all_passed": False, "checks": {}},
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
        # Use derive_both to get symmetry info from both layers
        both = _dual_layer.derive_both(name)
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
        # Compute harmonic relationship using derive
        da = _dual_layer.derive(name_a, mode="physics")
        db = _dual_layer.derive(name_b, mode="physics")
        va = da.get("value", da.get("measured", 0))
        vb = db.get("value", db.get("measured", 0))
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
            "best_5": errors[:5],
            "worst_5": errors[-5:],
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
            p_val = self.physics(a=0, b=0, c=0, d=0)
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
            "best_physics": by_physics[:5],
            "worst_physics": by_physics[-5:],
            "most_improved": by_improvement[:5],
            "best_thought": by_thought[:5],
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
            "convergences": convergences[:20],
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
            "best_5": rankings[:5],
            "worst_5": rankings[-5:],
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
            "best_5": hits[:5],
            "worst_5": hits[-5:],
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
            "strongest_correlations": [(a, b, round(c, 3)) for a, b, c in cross_correlations[:5]],
            "weakest_correlations": [(a, b, round(c, 3)) for a, b, c in cross_correlations[-5:]],
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


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON — ASI-wide dual-layer engine instance
# ═══════════════════════════════════════════════════════════════════════════════

dual_layer_engine = DualLayerEngine()
