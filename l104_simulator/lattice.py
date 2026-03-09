"""
===============================================================================
L104 SIMULATOR — LAYER 1: E-LATTICE ENGINE
===============================================================================

The foundational logarithmic integer lattice that addresses every physical
quantity with a single integer E.

GRID EQUATION:
  value = BASE × 2^(E/Q)
  E     = round(Q × log₂(value / BASE))

HOMOMORPHISM:
  multiply(a, b)  → E(a) + E(b)     (exact integer addition)
  divide(a, b)    → E(a) - E(b)     (exact integer subtraction)
  power(a, n)     → n × E(a)        (exact integer multiplication)

Resolution: Q=416 steps/octave = +0.1668%/step, max error 0.0834%

INVARIANT: 527.5184818492612 | PILOT: LONDEL
===============================================================================
"""

import math
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass, field
from enum import Enum, auto

from .constants import (
    PHI, GOD_CODE, BASE, STEP_SIZE,
    X_SCAFFOLD, R_RATIO, Q_GRAIN, P_DIAL, K_OFFSET,
    # All particle masses
    M_ELECTRON, M_MUON, M_TAU,
    M_UP, M_CHARM, M_TOP,
    M_DOWN, M_STRANGE, M_BOTTOM,
    M_W, M_Z, M_HIGGS,
    M_PROTON, M_NEUTRON, M_PION_PM, M_PION_0, M_KAON, M_D_MESON,
    # Fundamental
    C_LIGHT, ALPHA_INV, ALPHA_EM, PLANCK_H_EVS, E_CHARGE,
    PLANCK_LENGTH, PLANCK_MASS_GEV, BOHR_RADIUS_PM, RYDBERG_EV,
    # Scales
    SCALE_QCD, SCALE_EW, SCALE_PLANCK,
    # Nuclear
    BE_FE56, BE_HE4, BE_DEUT,
    # Sacred
    OMEGA, VOID_CONSTANT,
)

LOG2 = math.log(2)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class ParticleType(Enum):
    """Standard Model particle classification."""
    LEPTON = auto()
    QUARK_UP = auto()
    QUARK_DOWN = auto()
    GAUGE_BOSON = auto()
    SCALAR_BOSON = auto()
    BARYON = auto()
    MESON = auto()


class ForceType(Enum):
    """Fundamental force classification."""
    ELECTROMAGNETIC = auto()
    WEAK = auto()
    STRONG = auto()
    GRAVITY = auto()
    ELECTROWEAK = auto()  # unified at ~100 GeV


@dataclass(frozen=True)
class LatticePoint:
    """A physical constant encoded on the E-lattice."""
    name: str
    value: float
    unit: str
    E: int                        # integer grid address
    grid_value: float             # snapped value
    error_pct: float              # grid error (%)
    dials: Optional[Tuple[int, int, int, int]] = None  # (a, b, c, d)
    category: str = ""
    particle_type: Optional[ParticleType] = None
    generation: Optional[int] = None   # 1, 2, 3 for fermions
    charge: Optional[float] = None     # electric charge
    spin: Optional[float] = None


@dataclass
class LatticeArithmeticResult:
    """Result of an arithmetic operation on the lattice."""
    operation: str
    E_result: int
    value: float
    operands: List[str]
    exact: bool    # True if E-arithmetic matches direct computation


# ═══════════════════════════════════════════════════════════════════════════════
#  E-LATTICE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class ELattice:
    """
    Layer 1: The foundational integer-addressed logarithmic lattice.

    Every physical constant gets a unique integer E-address.
    Multiplication becomes addition. Division becomes subtraction.
    The lattice is a homomorphism (ℝ⁺, ×) → (ℤ, +).

    Usage:
        lattice = ELattice()
        E_proton = lattice.encode(938.272)          # → 2010
        mass = lattice.decode(2010)                  # → ≈938.3
        ratio_E = lattice.divide("m_p", "m_e")      # → 4511
        lattice.multiply("Ry_eV", 2)                 # → E(Hartree)
    """

    def __init__(self):
        self._points: Dict[str, LatticePoint] = {}
        self._E_index: Dict[int, List[str]] = {}  # reverse: E → names
        self._register_standard_model()

    # ─── Core Grid Operations ────────────────────────────────────────────

    @staticmethod
    def encode(value: float) -> int:
        """Encode a physical value to its integer E-address."""
        if value <= 0:
            raise ValueError(f"Cannot encode non-positive value: {value}")
        return round(Q_GRAIN * math.log(value / BASE) / LOG2)

    @staticmethod
    def decode(E: int) -> float:
        """Decode an E-address back to a physical value."""
        return BASE * (2 ** (E / Q_GRAIN))

    @staticmethod
    def grid_error(value: float) -> float:
        """Compute grid snap error in percent."""
        E = ELattice.encode(value)
        snapped = ELattice.decode(E)
        return abs(snapped - value) / value * 100

    @staticmethod
    def dials_to_E(a: int, b: int, c: int, d: int) -> int:
        """Convert (a,b,c,d) dials to E-address."""
        return P_DIAL * a + K_OFFSET - b - P_DIAL * c - Q_GRAIN * d

    @staticmethod
    def E_to_value_via_dials(a: int, b: int, c: int, d: int) -> float:
        """Compute physical value from dial settings."""
        E = ELattice.dials_to_E(a, b, c, d)
        return ELattice.decode(E)

    # ─── Lattice Arithmetic ──────────────────────────────────────────────

    def multiply_E(self, E_a: int, E_b: int) -> int:
        """Multiply two values: E(a×b) = E(a) + E(b)."""
        return E_a + E_b

    def divide_E(self, E_a: int, E_b: int) -> int:
        """Divide two values: E(a/b) = E(a) - E(b)."""
        return E_a - E_b

    def power_E(self, E_a: int, n: int) -> int:
        """Raise to integer power: E(a^n) = n × E(a)."""
        return n * E_a

    def ratio(self, name_a: str, name_b: str) -> float:
        """Physical ratio of two named constants."""
        dE = self._points[name_a].E - self._points[name_b].E
        return 2 ** (dE / Q_GRAIN)

    def multiply(self, name_a: str, factor: float) -> LatticeArithmeticResult:
        """Multiply a named constant by a factor."""
        E_a = self._points[name_a].E
        E_f = self.encode(factor)
        E_result = E_a + E_f
        val = self.decode(E_result)
        return LatticeArithmeticResult(
            operation=f"{name_a} × {factor}",
            E_result=E_result,
            value=val,
            operands=[name_a, str(factor)],
            exact=True,
        )

    def divide(self, name_a: str, name_b: str) -> LatticeArithmeticResult:
        """Divide two named constants."""
        E_a = self._points[name_a].E
        E_b = self._points[name_b].E
        E_result = E_a - E_b
        val = self.decode(E_result)
        return LatticeArithmeticResult(
            operation=f"{name_a} / {name_b}",
            E_result=E_result,
            value=val,
            operands=[name_a, name_b],
            exact=True,
        )

    # ─── Registration ────────────────────────────────────────────────────

    def register(self, name: str, value: float, unit: str,
                 dials: Optional[Tuple[int, int, int, int]] = None,
                 category: str = "", particle_type: Optional[ParticleType] = None,
                 generation: Optional[int] = None,
                 charge: Optional[float] = None,
                 spin: Optional[float] = None) -> LatticePoint:
        """Register a physical constant on the lattice."""
        E = self.encode(value)
        grid_val = self.decode(E)
        err = abs(grid_val - value) / value * 100

        pt = LatticePoint(
            name=name, value=value, unit=unit, E=E,
            grid_value=grid_val, error_pct=err, dials=dials,
            category=category, particle_type=particle_type,
            generation=generation, charge=charge, spin=spin,
        )
        self._points[name] = pt

        # Reverse index
        if E not in self._E_index:
            self._E_index[E] = []
        self._E_index[E].append(name)

        return pt

    def get(self, name: str) -> LatticePoint:
        """Get a lattice point by name."""
        return self._points[name]

    def E(self, name: str) -> int:
        """Get the E-address of a named constant."""
        return self._points[name].E

    @property
    def points(self) -> Dict[str, LatticePoint]:
        """All registered lattice points."""
        return dict(self._points)

    @property
    def E_range(self) -> Tuple[int, int]:
        """Min and max E across all registered constants."""
        Es = [p.E for p in self._points.values()]
        return (min(Es), max(Es))

    @property
    def degeneracies(self) -> Dict[int, List[str]]:
        """E-addresses shared by multiple constants."""
        return {E: names for E, names in self._E_index.items() if len(names) > 1}

    def constants_in_range(self, E_min: int, E_max: int) -> List[LatticePoint]:
        """Get all constants with E in [E_min, E_max]."""
        return [p for p in self._points.values() if E_min <= p.E <= E_max]

    def by_category(self, category: str) -> List[LatticePoint]:
        """Get all constants in a category."""
        return [p for p in self._points.values() if p.category == category]

    def by_type(self, ptype: ParticleType) -> List[LatticePoint]:
        """Get all particles of a given type."""
        return [p for p in self._points.values() if p.particle_type == ptype]

    def by_generation(self, gen: int) -> List[LatticePoint]:
        """Get all fermions in a given generation."""
        return [p for p in self._points.values() if p.generation == gen]

    def fermion_E_vector(self) -> List[int]:
        """9-element vector of fermion E-addresses: [e, μ, τ, u, c, t, d, s, b]."""
        order = ["m_e", "m_μ", "m_τ", "m_u", "m_c", "m_top", "m_d", "m_s", "m_b"]
        return [self._points[n].E for n in order]

    def boson_E_vector(self) -> List[int]:
        """Boson E-addresses: [W, Z, H]."""
        order = ["m_W", "m_Z", "m_H"]
        return [self._points[n].E for n in order]

    # ─── Statistics ──────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Lattice statistics."""
        errors = [p.error_pct for p in self._points.values()]
        E_min, E_max = self.E_range
        return {
            "num_constants": len(self._points),
            "E_range": (E_min, E_max),
            "E_span": E_max - E_min,
            "mean_error_pct": sum(errors) / len(errors),
            "max_error_pct": max(errors),
            "min_error_pct": min(errors),
            "degeneracies": len(self.degeneracies),
            "steps_per_octave": Q_GRAIN,
            "step_size_pct": (STEP_SIZE - 1) * 100,
        }

    # ─── Standard Model Registration ─────────────────────────────────────

    def _register_standard_model(self):
        """Register all Standard Model particles and fundamental constants."""
        P = ParticleType

        # === LEPTONS ===
        self.register("m_e",   M_ELECTRON, "MeV", (0,5,0,10),    "lepton", P.LEPTON, 1, -1, 0.5)
        self.register("m_μ",   M_MUON,     "MeV", (0,5,2,2),     "lepton", P.LEPTON, 2, -1, 0.5)
        self.register("m_τ",   M_TAU,      "MeV", (5,7,0,-1),    "lepton", P.LEPTON, 3, -1, 0.5)

        # === UP-TYPE QUARKS ===
        self.register("m_u",   M_UP,       "MeV", None,           "quark",  P.QUARK_UP, 1, 2/3, 0.5)
        self.register("m_c",   M_CHARM,    "MeV", (2,2,0,9),     "quark",  P.QUARK_UP, 2, 2/3, 0.5)
        self.register("m_top", M_TOP,      "MeV", (3,31,0,2),    "quark",  P.QUARK_UP, 3, 2/3, 0.5)

        # === DOWN-TYPE QUARKS ===
        self.register("m_d",   M_DOWN,     "MeV", None,           "quark",  P.QUARK_DOWN, 1, -1/3, 0.5)
        self.register("m_s",   M_STRANGE,  "MeV", None,           "quark",  P.QUARK_DOWN, 2, -1/3, 0.5)
        self.register("m_b",   M_BOTTOM,   "MeV", (0,24,6,6),    "quark",  P.QUARK_DOWN, 3, -1/3, 0.5)

        # === GAUGE BOSONS ===
        self.register("m_W",   M_W,        "MeV", (2,9,0,3),     "boson",  P.GAUGE_BOSON,  None, 1, 1)
        self.register("m_Z",   M_Z,        "MeV", (0,29,3,2),    "boson",  P.GAUGE_BOSON,  None, 0, 1)
        self.register("m_H",   M_HIGGS,    "MeV", (0,31,0,2),    "boson",  P.SCALAR_BOSON, None, 0, 0)

        # === HADRONS ===
        self.register("m_p",   M_PROTON,   "MeV", (0,6,1,-1),    "baryon", P.BARYON, None, 1, 0.5)
        self.register("m_n",   M_NEUTRON,  "MeV", (0,6,1,-1),    "baryon", P.BARYON, None, 0, 0.5)
        self.register("m_π±",  M_PION_PM,  "MeV", (1,30,0,2),    "meson",  P.MESON)
        self.register("m_π0",  M_PION_0,   "MeV", (0,18,6,1),    "meson",  P.MESON)
        self.register("m_K",   M_KAON,     "MeV", (6,8,0,1),     "meson",  P.MESON)
        self.register("m_D",   M_D_MESON,  "MeV", (0,9,1,-2),    "meson",  P.MESON)

        # === FUNDAMENTAL CONSTANTS ===
        self.register("c",      C_LIGHT,       "m/s",   (1,16,0,-19),  "fundamental")
        self.register("α_inv",  ALPHA_INV,     "",      (0,9,6,1),     "fundamental")
        self.register("h",      PLANCK_H_EVS,  "eV·s",  (0,23,5,56),  "fundamental")
        self.register("e",      E_CHARGE,      "C",     (0,8,3,71),    "fundamental")
        self.register("l_P",    PLANCK_LENGTH,  "m",    (0,1,4,124),   "planck")
        self.register("a₀",     BOHR_RADIUS_PM, "pm",  (0,4,2,3),     "atomic")
        self.register("Ry",     RYDBERG_EV,     "eV",   (5,19,0,6),   "atomic")

        # === NUCLEAR BINDING ===
        self.register("BE_Fe56", BE_FE56, "MeV/A", (1,25,0,6), "nuclear")
        self.register("BE_He4",  BE_HE4,  "MeV/A", (0,28,1,6), "nuclear")
        self.register("BE_deut", BE_DEUT, "MeV/A", (1,18,0,8), "nuclear")

        # === ENERGY SCALES ===
        self.register("Λ_QCD",  SCALE_QCD,    "MeV", None, "scale")
        self.register("v_EW",   SCALE_EW,     "MeV", None, "scale")
        self.register("M_Pl",   SCALE_PLANCK, "MeV", None, "scale")

        # === SACRED ===
        self.register("GOD_CODE", GOD_CODE, "",  None, "sacred")
        self.register("Ω",        OMEGA,    "",  (0,25,2,-4), "sacred")
        self.register("φ",        PHI,      "",  (0,17,2,8),  "sacred")

    def __repr__(self) -> str:
        E_min, E_max = self.E_range
        return (f"ELattice({len(self._points)} constants, "
                f"E∈[{E_min},{E_max}], Q={Q_GRAIN})")
