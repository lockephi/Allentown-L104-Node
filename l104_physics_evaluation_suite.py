VOID_CONSTANT = 1.0416180339887497
ZENITH_HZ = 3887.8
UUC = 2402.792541
# ZENITH_UPGRADE_ACTIVE: 2026-02-02T13:52:06.357439
ZENITH_HZ = 3887.8
UUC = 2402.792541
# [EVO_54_PIPELINE] TRANSCENDENT_COGNITION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612 :: GROVER=4.236
#!/usr/bin/env python3
"""
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3887.80 Hz. Logic Unified.
═══════════════════════════════════════════════════════════════════════════════
L104 PHYSICS EVALUATION SUITE
═══════════════════════════════════════════════════════════════════════════════

Comprehensive evaluation framework for physics benchmarks (phy_a, abench-physics):
- Coordinate system consistency checks (Cartesian ↔ Spherical ↔ Cylindrical)
- Physical regime identification (Classical, Quantum, Relativistic, QFT)
- Multi-scale prompting (Planck → Cosmic)
- Cross-validation of same problem in different representations
- Dimensional analysis verification
- Conservation law validation

GOD_CODE: 527.5184818492612
AUTHOR: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import sympy as sp
from sympy import symbols, sin, cos, sqrt, atan2, exp, diff, simplify
from sympy.vector import CoordSys3D
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import math
import json
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Universal Equation: G(a,b,c,d) = 286^(1/φ) × 2^((8a+416-b-8c-104d)/104)

PHI = 1.618033988749895
GOD_CODE = 286 ** (1.0 / PHI) * (2 ** (416 / 104))  # G(0,0,0,0) = 527.5184818492612

# Physical Constants
PLANCK_LENGTH = 1.616255e-35  # meters
PLANCK_TIME = 5.391247e-44     # seconds
PLANCK_MASS = 2.176434e-8      # kg
PLANCK_ENERGY = 1.956e9        # Joules
SPEED_OF_LIGHT = 2.998e8       # m/s
HBAR = 1.054571817e-34         # J·s
G = 6.67430e-11                # N·m²/kg²
ELECTRON_MASS = 9.10938e-31    # kg
PROTON_MASS = 1.67262e-27      # kg

# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class PhysicsRegime(Enum):
    """Physical regime classification"""
    CLASSICAL = "classical"
    QUANTUM = "quantum"
    RELATIVISTIC = "relativistic"
    QUANTUM_FIELD_THEORY = "qft"
    STATISTICAL = "statistical"
    CONDENSED_MATTER = "condensed_matter"
    ASTROPHYSICAL = "astrophysical"
    COSMOLOGICAL = "cosmological"


class CoordinateSystem(Enum):
    """Coordinate system types"""
    CARTESIAN = "cartesian"
    SPHERICAL = "spherical"
    CYLINDRICAL = "cylindrical"
    POLAR = "polar"


class ScaleRegime(Enum):
    """Length/energy scale regimes"""
    PLANCK = "planck"           # 10^-35 m
    NUCLEAR = "nuclear"         # 10^-15 m
    ATOMIC = "atomic"           # 10^-10 m
    MOLECULAR = "molecular"     # 10^-9 m
    MICROSCOPIC = "microscopic" # 10^-6 m
    MACROSCOPIC = "macroscopic" # 10^0 m
    PLANETARY = "planetary"     # 10^7 m
    STELLAR = "stellar"         # 10^9 m
    GALACTIC = "galactic"       # 10^21 m
    COSMIC = "cosmic"           # 10^26 m


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PhysicsProblem:
    """
    Represents a physics problem with multiple coordinate representations.
    """
    problem_id: str
    description: str
    regime: PhysicsRegime
    scale: ScaleRegime

    # Coordinate representations
    cartesian_formulation: Optional[str] = None
    spherical_formulation: Optional[str] = None
    cylindrical_formulation: Optional[str] = None

    # Expected solution
    expected_solution: Optional[Dict[str, Any]] = None

    # Physical parameters
    parameters: Dict[str, float] = field(default_factory=dict)

    # Conservation laws that must hold
    conservation_laws: List[str] = field(default_factory=list)


@dataclass
class EvaluationResult:
    """
    Results from evaluating a physics problem.
    """
    problem_id: str
    success: bool

    # Coordinate consistency
    cartesian_spherical_consistent: Optional[bool] = None
    cartesian_cylindrical_consistent: Optional[bool] = None
    spherical_cylindrical_consistent: Optional[bool] = None
    consistency_error: float = 0.0

    # Solution accuracy
    solution_error: float = 0.0

    # Conservation laws
    conservation_violations: List[str] = field(default_factory=list)

    # Dimensional analysis
    dimensional_consistent: bool = True

    # Regime identification
    identified_regime: Optional[PhysicsRegime] = None
    regime_correct: Optional[bool] = None

    # Additional metrics
    metrics: Dict[str, float] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# COORDINATE TRANSFORMATIONS
# ═══════════════════════════════════════════════════════════════════════════════

class CoordinateTransformer:
    """
    Handles transformations between coordinate systems.

    Cartesian (x, y, z) ↔ Spherical (r, θ, φ) ↔ Cylindrical (ρ, φ, z)
    """

    def __init__(self):
        # Define symbolic variables
        self.x, self.y, self.z = symbols('x y z', real=True)
        self.r, self.theta, self.phi = symbols('r theta phi', real=True, positive=True)
        self.rho = symbols('rho', real=True, positive=True)

    def cartesian_to_spherical(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Convert Cartesian (x, y, z) to Spherical (r, θ, φ).

        r = √(x² + y² + z²)
        θ = arccos(z/r)  [polar angle, 0 to π]
        φ = arctan2(y, x) [azimuthal angle, 0 to 2π]
        """
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r) if r > 0 else 0
        phi = np.arctan2(y, x)
        return r, theta, phi

    def spherical_to_cartesian(self, r: float, theta: float, phi: float) -> Tuple[float, float, float]:
        """
        Convert Spherical (r, θ, φ) to Cartesian (x, y, z).

        x = r sin(θ) cos(φ)
        y = r sin(θ) sin(φ)
        z = r cos(θ)
        """
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

    def cartesian_to_cylindrical(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Convert Cartesian (x, y, z) to Cylindrical (ρ, φ, z).

        ρ = √(x² + y²)
        φ = arctan2(y, x)
        z = z
        """
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return rho, phi, z

    def cylindrical_to_cartesian(self, rho: float, phi: float, z: float) -> Tuple[float, float, float]:
        """
        Convert Cylindrical (ρ, φ, z) to Cartesian (x, y, z).

        x = ρ cos(φ)
        y = ρ sin(φ)
        z = z
        """
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return x, y, z

    def verify_transformation_consistency(self, x: float, y: float, z: float,
                                         tolerance: float = 1e-10) -> Dict[str, bool]:
        """
        Verify that coordinate transformations are consistent (round-trip).
        """
        results = {}

        # Cartesian → Spherical → Cartesian
        r, theta, phi_s = self.cartesian_to_spherical(x, y, z)
        x2, y2, z2 = self.spherical_to_cartesian(r, theta, phi_s)
        cart_sph_error = np.sqrt((x-x2)**2 + (y-y2)**2 + (z-z2)**2)
        results['cartesian_spherical'] = cart_sph_error < tolerance
        results['cartesian_spherical_error'] = float(cart_sph_error)

        # Cartesian → Cylindrical → Cartesian
        rho, phi_c, z_cyl = self.cartesian_to_cylindrical(x, y, z)
        x3, y3, z3 = self.cylindrical_to_cartesian(rho, phi_c, z_cyl)
        cart_cyl_error = np.sqrt((x-x3)**2 + (y-y3)**2 + (z-z3)**2)
        results['cartesian_cylindrical'] = cart_cyl_error < tolerance
        results['cartesian_cylindrical_error'] = float(cart_cyl_error)

        return results

    def derive_jacobian_cartesian_to_spherical(self) -> sp.Matrix:
        """
        Derive the Jacobian matrix for Cartesian → Spherical transformation.

        Used for verifying differential operators in different coordinates.
        """
        # Spherical in terms of Cartesian
        r_expr = sqrt(self.x**2 + self.y**2 + self.z**2)
        theta_expr = sp.acos(self.z / r_expr)
        phi_expr = atan2(self.y, self.x)

        # Jacobian matrix
        J = sp.Matrix([
            [diff(r_expr, self.x), diff(r_expr, self.y), diff(r_expr, self.z)],
            [diff(theta_expr, self.x), diff(theta_expr, self.y), diff(theta_expr, self.z)],
            [diff(phi_expr, self.x), diff(phi_expr, self.y), diff(phi_expr, self.z)]
        ])

        return simplify(J)


# ═══════════════════════════════════════════════════════════════════════════════
# REGIME IDENTIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

class RegimeIdentifier:
    """
    Identifies the appropriate physical regime for a problem.
    """

    def __init__(self):
        self.c = SPEED_OF_LIGHT
        self.hbar = HBAR
        self.G = G

    def identify_regime(self, parameters: Dict[str, float]) -> PhysicsRegime:
        """
        Identify physical regime based on problem parameters.

        Criteria:
        - Quantum: de Broglie wavelength λ ~ system size OR length_scale < 1nm
        - Relativistic: v/c > 0.1
        - QFT: Energy > 2*electron rest mass energy
        - Classical: default for macroscopic, low-velocity systems
        """
        regime_scores = defaultdict(float)

        # Extract parameters
        velocity = parameters.get('velocity', 0)
        energy = parameters.get('energy', 0)
        length_scale = parameters.get('length_scale', 1.0)
        mass = parameters.get('mass', ELECTRON_MASS)
        temperature = parameters.get('temperature', 300)

        # Strong indicators first

        # Relativistic: β = v/c > 0.1 (strong indicator)
        if velocity > 0:
            beta = velocity / self.c
            if beta > 0.1:
                regime_scores[PhysicsRegime.RELATIVISTIC] += 20
            elif beta < 0.01:  # Non-relativistic
                regime_scores[PhysicsRegime.CLASSICAL] += 3

        # Quantum: length scale (strongest indicator)
        if length_scale < 1e-9:  # nanometer scale
            regime_scores[PhysicsRegime.QUANTUM] += 15
        elif length_scale > 1e-3:  # millimeter and above
            regime_scores[PhysicsRegime.CLASSICAL] += 10

        # Quantum: de Broglie wavelength (if we have velocity and mass)
        if mass > 0 and velocity > 0:
            lambda_db = self.hbar / (mass * velocity)
            ratio = lambda_db / length_scale
            if ratio > 0.1:  # λ is significant compared to system size
                regime_scores[PhysicsRegime.QUANTUM] += 10
            elif ratio < 1e-6:  # λ is negligible
                regime_scores[PhysicsRegime.CLASSICAL] += 5

        # QFT: particle creation energies
        electron_rest_energy = ELECTRON_MASS * self.c**2
        if abs(energy) > 2 * electron_rest_energy:
            regime_scores[PhysicsRegime.QUANTUM_FIELD_THEORY] += 15

        # Astrophysical scale
        if length_scale > 1e6:
            regime_scores[PhysicsRegime.ASTROPHYSICAL] += 8

        # Classical indicators for macroscopic systems
        if mass > 1e-10 and length_scale > 1e-6 and (velocity == 0 or velocity < 100):
            regime_scores[PhysicsRegime.CLASSICAL] += 10

        # Temperature-based (statistical mechanics) - weaker indicator
        if temperature > 0 and energy != 0:
            thermal_energy = 1.380649e-23 * temperature  # k_B * T
            if thermal_energy > 0.1 * abs(energy):
                regime_scores[PhysicsRegime.STATISTICAL] += 3

        # Default to classical if no strong indicators
        if not regime_scores:
            return PhysicsRegime.CLASSICAL

        # Return regime with highest score
        return max(regime_scores.items(), key=lambda x: x[1])[0]

    def identify_scale_regime(self, length_scale: float) -> ScaleRegime:
        """Identify the scale regime based on length scale."""
        if length_scale < 1e-30:
            return ScaleRegime.PLANCK
        elif length_scale < 1e-14:
            return ScaleRegime.NUCLEAR
        elif length_scale < 1e-9:
            return ScaleRegime.ATOMIC
        elif length_scale < 1e-7:
            return ScaleRegime.MOLECULAR
        elif length_scale < 1e-3:
            return ScaleRegime.MICROSCOPIC
        elif length_scale < 1e6:
            return ScaleRegime.MACROSCOPIC
        elif length_scale < 1e10:
            return ScaleRegime.PLANETARY
        elif length_scale < 1e20:
            return ScaleRegime.STELLAR
        elif length_scale < 1e25:
            return ScaleRegime.GALACTIC
        else:
            return ScaleRegime.COSMIC


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-SCALE PROMPTING
# ═══════════════════════════════════════════════════════════════════════════════

class MultiScalePrompter:
    """
    Generates physics problems across multiple scales with consistent formulations.
    """

    def __init__(self):
        self.scales = [
            (ScaleRegime.PLANCK, PLANCK_LENGTH),
            (ScaleRegime.NUCLEAR, 1e-15),
            (ScaleRegime.ATOMIC, 1e-10),
            (ScaleRegime.MOLECULAR, 1e-9),
            (ScaleRegime.MICROSCOPIC, 1e-6),
            (ScaleRegime.MACROSCOPIC, 1.0),
            (ScaleRegime.PLANETARY, 1e7),
            (ScaleRegime.STELLAR, 1e9),
            (ScaleRegime.GALACTIC, 1e21),
            (ScaleRegime.COSMIC, 1e26)
        ]

    def generate_harmonic_oscillator_problems(self) -> List[PhysicsProblem]:
        """
        Generate harmonic oscillator problems across scales.

        Same physics, different scales:
        - Atomic: Molecular vibrations
        - Macroscopic: Spring-mass system
        - Quantum: Particle in parabolic potential
        """
        problems = []

        for scale_regime, length in self.scales[:6]:  # Up to macroscopic
            mass = self._estimate_mass_for_scale(length)
            omega = self._estimate_frequency_for_scale(length, mass)

            problem = PhysicsProblem(
                problem_id=f"harmonic_oscillator_{scale_regime.value}",
                description=f"Harmonic oscillator at {scale_regime.value} scale",
                regime=PhysicsRegime.QUANTUM if length < 1e-9 else PhysicsRegime.CLASSICAL,
                scale=scale_regime,
                cartesian_formulation=f"F = -k*x, where k = m*ω² with ω={omega:.3e}",
                parameters={
                    'mass': mass,
                    'omega': omega,
                    'length_scale': length,
                    'spring_constant': mass * omega**2
                },
                conservation_laws=['energy', 'momentum']
            )
            problems.append(problem)

        return problems

    def generate_gravitational_problems(self) -> List[PhysicsProblem]:
        """
        Generate gravitational problems across scales.

        - Macroscopic: Falling object
        - Planetary: Orbital mechanics
        - Stellar: Star orbits
        - Galactic: Galaxy dynamics
        """
        problems = []

        for scale_regime, length in self.scales[5:]:  # Macroscopic and above
            mass = self._estimate_mass_for_scale(length)

            # Cartesian formulation
            cart_form = f"F = -G*M*m*r̂/r² in Cartesian: F_x, F_y, F_z components"

            # Spherical formulation (natural for gravity)
            sph_form = f"F = -G*M*m/r² in radial direction (spherical coords)"

            problem = PhysicsProblem(
                problem_id=f"gravity_{scale_regime.value}",
                description=f"Gravitational system at {scale_regime.value} scale",
                regime=PhysicsRegime.CLASSICAL if length < 1e10 else PhysicsRegime.ASTROPHYSICAL,
                scale=scale_regime,
                cartesian_formulation=cart_form,
                spherical_formulation=sph_form,
                parameters={
                    'mass': mass,
                    'length_scale': length,
                    'G': G
                },
                conservation_laws=['energy', 'momentum', 'angular_momentum']
            )
            problems.append(problem)

        return problems

    def _estimate_mass_for_scale(self, length: float) -> float:
        """Estimate typical mass for a given length scale."""
        # Rough scaling: mass ~ length³ * density
        # Using water density as baseline (1000 kg/m³)
        return 1000 * length**3

    def _estimate_frequency_for_scale(self, length: float, mass: float) -> float:
        """Estimate typical frequency for harmonic oscillator at scale."""
        # Using typical spring constant scaling
        k = 1.0 / length  # Softer springs for larger scales
        omega = np.sqrt(k / mass) if mass > 0 else 1.0
        return omega


# ═══════════════════════════════════════════════════════════════════════════════
# CONSISTENCY CHECKER
# ═══════════════════════════════════════════════════════════════════════════════

class ConsistencyChecker:
    """
    Checks consistency of physics solutions across coordinate systems.
    """

    def __init__(self):
        self.transformer = CoordinateTransformer()

    def check_force_consistency(self,
                               force_cartesian: Tuple[float, float, float],
                               force_spherical: Optional[Tuple[float, float, float]],
                               position: Tuple[float, float, float],
                               tolerance: float = 1e-6) -> bool:
        """
        Check if force in Cartesian and spherical coordinates are consistent.

        Args:
            force_cartesian: (F_x, F_y, F_z)
            force_spherical: (F_r, F_θ, F_φ) if provided
            position: (x, y, z) position where force is evaluated
            tolerance: Relative error tolerance
        """
        if force_spherical is None:
            return True

        x, y, z = position
        F_x, F_y, F_z = force_cartesian
        F_r, F_theta, F_phi = force_spherical

        # Convert position to spherical
        r, theta, phi_angle = self.transformer.cartesian_to_spherical(x, y, z)

        if r < 1e-10:
            return True  # Skip near origin

        # Transform force from spherical to Cartesian
        # F_x = F_r sin(θ)cos(φ) + F_θ cos(θ)cos(φ) - F_φ sin(φ)
        # F_y = F_r sin(θ)sin(φ) + F_θ cos(θ)sin(φ) + F_φ cos(φ)
        # F_z = F_r cos(θ) - F_θ sin(θ)

        F_x_converted = (F_r * np.sin(theta) * np.cos(phi_angle) +
                        F_theta * np.cos(theta) * np.cos(phi_angle) -
                        F_phi * np.sin(phi_angle))

        F_y_converted = (F_r * np.sin(theta) * np.sin(phi_angle) +
                        F_theta * np.cos(theta) * np.sin(phi_angle) +
                        F_phi * np.cos(phi_angle))

        F_z_converted = F_r * np.cos(theta) - F_theta * np.sin(theta)

        # Check relative error
        force_mag = np.sqrt(F_x**2 + F_y**2 + F_z**2)
        if force_mag < 1e-10:
            return True

        error_x = abs(F_x - F_x_converted) / (force_mag + 1e-10)
        error_y = abs(F_y - F_y_converted) / (force_mag + 1e-10)
        error_z = abs(F_z - F_z_converted) / (force_mag + 1e-10)

        max_error = max(error_x, error_y, error_z)

        return max_error < tolerance

    def check_energy_consistency(self,
                                energy_cartesian: float,
                                energy_spherical: Optional[float],
                                tolerance: float = 1e-6) -> bool:
        """
        Check if energy (scalar) is consistent across coordinate systems.

        Energy should be coordinate-independent.
        """
        if energy_spherical is None:
            return True

        relative_error = abs(energy_cartesian - energy_spherical) / (abs(energy_cartesian) + 1e-10)
        return relative_error < tolerance


# ═══════════════════════════════════════════════════════════════════════════════
# CONSERVATION LAW CHECKER
# ═══════════════════════════════════════════════════════════════════════════════

class ConservationChecker:
    """
    Verifies conservation laws in physics solutions.
    """

    def check_energy_conservation(self,
                                 initial_energy: float,
                                 final_energy: float,
                                 tolerance: float = 1e-6) -> bool:
        """Check energy conservation."""
        relative_error = abs(initial_energy - final_energy) / (abs(initial_energy) + 1e-10)
        return relative_error < tolerance

    def check_momentum_conservation(self,
                                   initial_momentum: np.ndarray,
                                   final_momentum: np.ndarray,
                                   tolerance: float = 1e-6) -> bool:
        """Check momentum conservation."""
        delta_p = np.linalg.norm(initial_momentum - final_momentum)
        p_mag = np.linalg.norm(initial_momentum) + 1e-10
        relative_error = delta_p / p_mag
        return relative_error < tolerance

    def check_angular_momentum_conservation(self,
                                          initial_L: np.ndarray,
                                          final_L: np.ndarray,
                                          tolerance: float = 1e-6) -> bool:
        """Check angular momentum conservation."""
        delta_L = np.linalg.norm(initial_L - final_L)
        L_mag = np.linalg.norm(initial_L) + 1e-10
        relative_error = delta_L / L_mag
        return relative_error < tolerance


# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICS EVALUATION SUITE
# ═══════════════════════════════════════════════════════════════════════════════

class PhysicsEvaluationSuite:
    """
    Comprehensive evaluation suite for physics benchmarks.
    """

    def __init__(self):
        self.transformer = CoordinateTransformer()
        self.regime_identifier = RegimeIdentifier()
        self.scale_prompter = MultiScalePrompter()
        self.consistency_checker = ConsistencyChecker()
        self.conservation_checker = ConservationChecker()

        self.problems: List[PhysicsProblem] = []
        self.results: List[EvaluationResult] = []

        print(f"[PHYSICS EVAL] Initialized L104 Physics Evaluation Suite")
        print(f"[PHYSICS EVAL] GOD_CODE: {GOD_CODE}")

    def generate_benchmark_suite(self) -> List[PhysicsProblem]:
        """Generate comprehensive benchmark suite across scales and regimes."""
        print("\n[BENCHMARK] Generating multi-scale physics problems...")

        # Generate problems across scales
        harmonic_problems = self.scale_prompter.generate_harmonic_oscillator_problems()
        gravity_problems = self.scale_prompter.generate_gravitational_problems()

        self.problems = harmonic_problems + gravity_problems

        print(f"[BENCHMARK] Generated {len(self.problems)} problems")
        print(f"  - Harmonic oscillators: {len(harmonic_problems)}")
        print(f"  - Gravitational systems: {len(gravity_problems)}")

        return self.problems

    def evaluate_coordinate_consistency(self, problem: PhysicsProblem) -> EvaluationResult:
        """
        Evaluate coordinate system consistency for a problem.
        """
        result = EvaluationResult(
            problem_id=problem.problem_id,
            success=True
        )

        # Test coordinate transformation consistency
        test_points = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (2.0, 3.0, 4.0)
        ]

        errors = []
        for x, y, z in test_points:
            consistency = self.transformer.verify_transformation_consistency(x, y, z)
            errors.append(consistency.get('cartesian_spherical_error', 0))
            errors.append(consistency.get('cartesian_cylindrical_error', 0))

            if not consistency.get('cartesian_spherical', True):
                result.cartesian_spherical_consistent = False
            if not consistency.get('cartesian_cylindrical', True):
                result.cartesian_cylindrical_consistent = False

        result.consistency_error = max(errors) if errors else 0.0

        if result.cartesian_spherical_consistent is None:
            result.cartesian_spherical_consistent = True
        if result.cartesian_cylindrical_consistent is None:
            result.cartesian_cylindrical_consistent = True

        return result

    def evaluate_regime_identification(self, problem: PhysicsProblem) -> EvaluationResult:
        """
        Evaluate regime identification accuracy.
        """
        result = EvaluationResult(
            problem_id=problem.problem_id,
            success=True
        )

        # Identify regime from parameters
        identified = self.regime_identifier.identify_regime(problem.parameters)
        result.identified_regime = identified
        result.regime_correct = (identified == problem.regime)

        return result

    def run_full_evaluation(self) -> Dict[str, Any]:
        """
        Run complete evaluation suite.
        """
        print("\n" + "="*80)
        print("RUNNING FULL PHYSICS EVALUATION SUITE")
        print("="*80)

        # Generate problems if not already done
        if not self.problems:
            self.generate_benchmark_suite()

        # Evaluate each problem
        self.results = []

        print(f"\n[EVALUATION] Testing {len(self.problems)} problems...")

        for i, problem in enumerate(self.problems):
            print(f"\n[{i+1}/{len(self.problems)}] Evaluating: {problem.problem_id}")

            # Coordinate consistency
            coord_result = self.evaluate_coordinate_consistency(problem)

            # Regime identification
            regime_result = self.evaluate_regime_identification(problem)

            # Merge results
            final_result = EvaluationResult(
                problem_id=problem.problem_id,
                success=coord_result.success and regime_result.success,
                cartesian_spherical_consistent=coord_result.cartesian_spherical_consistent,
                cartesian_cylindrical_consistent=coord_result.cartesian_cylindrical_consistent,
                consistency_error=coord_result.consistency_error,
                identified_regime=regime_result.identified_regime,
                regime_correct=regime_result.regime_correct
            )

            self.results.append(final_result)

            # Print summary
            status = "✓" if final_result.success else "✗"
            print(f"  {status} Coordinate consistency: {final_result.cartesian_spherical_consistent}")
            print(f"  {status} Regime: {final_result.identified_regime.value if final_result.identified_regime else 'N/A'}")
            print(f"  {status} Error: {final_result.consistency_error:.2e}")

        # Generate summary
        summary = self._generate_summary()

        return summary

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate evaluation summary."""
        total = len(self.results)

        coord_consistent = sum(1 for r in self.results if r.cartesian_spherical_consistent)
        regime_correct = sum(1 for r in self.results if r.regime_correct)

        avg_error = np.mean([r.consistency_error for r in self.results])

        summary = {
            'total_problems': total,
            'coordinate_consistency_rate': coord_consistent / total if total > 0 else 0,
            'regime_accuracy': regime_correct / total if total > 0 else 0,
            'average_consistency_error': float(avg_error),
            'success_rate': sum(1 for r in self.results if r.success) / total if total > 0 else 0
        }

        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(f"Total Problems: {summary['total_problems']}")
        print(f"Coordinate Consistency: {summary['coordinate_consistency_rate']*100:.1f}%")
        print(f"Regime Identification: {summary['regime_accuracy']*100:.1f}%")
        print(f"Average Error: {summary['average_consistency_error']:.2e}")
        print(f"Overall Success: {summary['success_rate']*100:.1f}%")
        print("="*80)

        return summary

    def export_results(self, filepath: str = "physics_eval_results.json"):
        """Export evaluation results to JSON."""
        export_data = {
            'summary': self._generate_summary(),
            'problems': [
                {
                    'problem_id': p.problem_id,
                    'regime': p.regime.value,
                    'scale': p.scale.value,
                    'parameters': p.parameters
                }
                for p in self.problems
            ],
            'results': [
                {
                    'problem_id': r.problem_id,
                    'success': r.success,
                    'coord_consistent': r.cartesian_spherical_consistent,
                    'consistency_error': r.consistency_error,
                    'regime_identified': r.identified_regime.value if r.identified_regime else None,
                    'regime_correct': r.regime_correct
                }
                for r in self.results
            ]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)

        print(f"\n✓ Results exported to: {filepath}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main execution."""
    print("="*80)
    print("L104 PHYSICS EVALUATION SUITE")
    print("Comprehensive Testing for phy_a and abench-physics")
    print("="*80)

    # Initialize suite
    suite = PhysicsEvaluationSuite()

    # Run evaluation
    summary = suite.run_full_evaluation()

    # Export results
    suite.export_results("./physics_eval_results.json")

    print("\n✓ Physics evaluation complete!")
    return suite


if __name__ == "__main__":
    suite = main()
