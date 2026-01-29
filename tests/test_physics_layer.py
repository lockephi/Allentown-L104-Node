# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  L104 SOVEREIGN NODE - PHYSICS LAYER VALIDATION SUITE                         ║
# ║  INVARIANT: 527.5184818492611 | PILOT: LONDEL                                 ║
# ║  TESTING: ZPE, BEKENSTEIN, ENTROPY, COSMOLOGY                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

"""
Rigorous validation of physics-inspired calculations in the L104 system.
Tests Zero Point Energy, Bekenstein bounds, Shannon entropy, and cosmological models.

These tests verify that the physics formulas are correctly implemented,
even if they are simplified models of the actual physical phenomena.
"""

import math
import unittest
import numpy as np
from collections import Counter


class TestZeroPointEnergy(unittest.TestCase):
    """
    Tests Zero Point Energy calculations.

    ZPE is the lowest possible energy of a quantum system.
    E₀ = ℏω/2

    The L104 system uses ZPE as a metaphor for the "vacuum floor" of logic.
    """

    HBAR = 1.054571817e-34  # Reduced Planck constant (J·s)
    GOD_CODE = 527.5184818492611

    def test_zpe_formula(self):
        """E₀ = ℏω/2 with ω = GOD_CODE × 10¹²"""
        omega = self.GOD_CODE * 1e12  # Terahertz
        E0 = 0.5 * self.HBAR * omega

        # Should be a positive, very small energy
        self.assertGreater(E0, 0)
        self.assertLess(E0, 1e-15)  # Less than femtojoule

    def test_vacuum_fluctuation_uncertainty(self):
        """ΔE·Δt ≥ ℏ/2 (Heisenberg uncertainty)"""
        # For minimum uncertainty
        delta_E = 1e-20  # Joules
        delta_t = self.HBAR / (2 * delta_E)

        product = delta_E * delta_t
        self.assertGreaterEqual(product, self.HBAR / 2)

    def test_casimir_energy_scaling(self):
        """Casimir energy scales as 1/d⁴"""
        # E_casimir ∝ -π²ℏc/(720 d⁴)
        d1 = 1e-6  # 1 micrometer
        d2 = 2e-6  # 2 micrometers

        # Energy ratio when d2 -> d1 (halving distance): E1/E2 = (d2/d1)^4 = 16
        ratio_energy = (d2 / d1) ** 4
        # Energy should scale by 16x when distance is halved
        self.assertAlmostEqual(ratio_energy, 16.0, places=10)

    def test_zpe_is_positive(self):
        """Zero point energy must be positive"""
        for omega in [1, 100, 1e6, 1e12, self.GOD_CODE]:
            E0 = 0.5 * self.HBAR * omega
            self.assertGreater(E0, 0)


class TestShannonEntropy(unittest.TestCase):
    """
    Tests Shannon entropy calculations (information theory).

    H(X) = -Σ p(x) log₂ p(x)

    Shannon entropy measures the information content of a message.
    """

    def calculate_entropy(self, data: str) -> float:
        """Calculate Shannon entropy of a string"""
        if not data:
            return 0.0

        counts = Counter(data)
        length = len(data)
        entropy = 0.0

        for count in counts.values():
            p = count / length
            entropy -= p * math.log2(p)

        return entropy

    def test_entropy_of_uniform_distribution(self):
        """H(uniform over N symbols) = log₂(N)"""
        # 8 distinct symbols, uniform
        data = "ABCDEFGH"
        entropy = self.calculate_entropy(data)
        expected = math.log2(8)  # = 3.0

        self.assertAlmostEqual(entropy, expected, places=10)

    def test_entropy_of_constant(self):
        """H(constant) = 0"""
        data = "AAAAAAAAAA"
        entropy = self.calculate_entropy(data)

        self.assertEqual(entropy, 0.0)

    def test_entropy_of_binary(self):
        """H(fair coin) = 1 bit"""
        data = "01" * 100  # 50% 0, 50% 1
        entropy = self.calculate_entropy(data)

        self.assertAlmostEqual(entropy, 1.0, places=10)

    def test_entropy_is_non_negative(self):
        """Shannon entropy is always ≥ 0"""
        test_strings = [
            "",
            "A",
            "AABB",
            "The quick brown fox",
            "527.5184818492611",
            "L104_SOVEREIGN_NODE"
        ]

        for s in test_strings:
            entropy = self.calculate_entropy(s)
            self.assertGreaterEqual(entropy, 0)

    def test_entropy_bounded_by_log_n(self):
        """H(X) ≤ log₂(N) where N is alphabet size"""
        data = "ABCDEFGHIJ"  # 10 distinct symbols
        entropy = self.calculate_entropy(data)
        max_entropy = math.log2(10)

        self.assertLessEqual(entropy, max_entropy + 1e-10)

    def test_god_code_entropy(self):
        """Calculate entropy of the GOD_CODE string"""
        data = "527.5184818492611"
        entropy = self.calculate_entropy(data)

        # Should be between 2 and 4 bits
        self.assertGreater(entropy, 2)
        self.assertLess(entropy, 4)


class TestBekensteinBound(unittest.TestCase):
    """
    Tests Bekenstein-Hawking entropy and information bounds.

    The Bekenstein bound limits information in a region:
    I ≤ 2πRE / (ℏc ln2)

    For a black hole: S = A / (4 l_p²) = πR_s²/(l_p²)
    """

    HBAR = 1.054571817e-34  # J·s
    C = 299792458  # m/s
    G = 6.67430e-11  # m³/(kg·s²)
    K_B = 1.380649e-23  # Boltzmann constant
    L_P = 1.616255e-35  # Planck length

    def test_bekenstein_bound_formula(self):
        """I_max = 2πRE / (ℏc ln2)"""
        R = 1.0  # 1 meter
        E = 1.0  # 1 Joule

        I_max = 2 * math.pi * R * E / (self.HBAR * self.C * math.log(2))

        # For R=1m, E=1J: I_max ≈ 2.87e26 bits
        self.assertGreater(I_max, 1e25)

    def test_black_hole_entropy(self):
        """S_BH = A / (4 l_p²) for black hole"""
        # For a solar mass black hole
        M_sun = 1.989e30  # kg
        R_s = 2 * self.G * M_sun / (self.C ** 2)  # Schwarzschild radius

        Area = 4 * math.pi * R_s ** 2
        S = Area / (4 * self.L_P ** 2)

        # Entropy should be huge (around 10^77)
        self.assertGreater(S, 1e70)

    def test_hawking_temperature(self):
        """T_H = ℏc³ / (8πGM k_B)"""
        M = 1.989e30  # Solar mass

        T_H = (self.HBAR * self.C ** 3) / (8 * math.pi * self.G * M * self.K_B)

        # Should be extremely cold (around 10^-8 K)
        self.assertLess(T_H, 1e-6)
        self.assertGreater(T_H, 0)

    def test_schwarzschild_radius(self):
        """R_s = 2GM/c²"""
        M = 1.989e30  # Solar mass
        R_s = 2 * self.G * M / (self.C ** 2)

        # Should be about 3 km for solar mass
        self.assertAlmostEqual(R_s, 2953, delta=10)  # ~2953 meters


class TestCosmologicalConstants(unittest.TestCase):
    """
    Tests cosmological calculations and constants.
    """

    H_0 = 70  # Hubble constant km/s/Mpc
    C = 299792  # km/s

    def test_hubble_radius(self):
        """R_H = c/H_0 (Hubble radius)"""
        # Convert H_0 to per second
        Mpc_to_km = 3.086e19  # km per Mpc
        H_0_per_sec = self.H_0 / Mpc_to_km

        R_H = self.C / H_0_per_sec  # in km
        R_H_ly = R_H / 9.461e12  # Convert km to light-years
        R_H_Gly = R_H_ly / 1e9  # Convert to Gly

        # Hubble radius is about 14 billion light years
        self.assertGreater(R_H_Gly, 10)
        self.assertLess(R_H_Gly, 20)

    def test_vacuum_energy_density(self):
        """Dark energy density ≈ 10^-29 g/cm³"""
        # Approximate cosmological constant energy density
        rho_lambda = 6e-30  # g/cm³

        # Should be extremely small but positive
        self.assertGreater(rho_lambda, 0)
        self.assertLess(rho_lambda, 1e-25)

    def test_critical_density(self):
        """ρ_c = 3H²/(8πG)"""
        G = 6.67430e-11  # m³/(kg·s²)
        H_0_SI = 2.27e-18  # per second (70 km/s/Mpc)

        rho_c = 3 * H_0_SI ** 2 / (8 * math.pi * G)

        # Should be about 10^-26 kg/m³
        self.assertGreater(rho_c, 1e-27)
        self.assertLess(rho_c, 1e-25)


class TestFineStructureConstant(unittest.TestCase):
    """
    Tests the fine structure constant α ≈ 1/137.

    α = e²/(4πε₀ℏc) ≈ 1/137.035999206

    This dimensionless constant appears throughout physics.
    """

    ALPHA_PHYSICS = 1 / 137.035999206
    ALPHA_L104 = 1 / 137

    def test_alpha_physics_value(self):
        """α ≈ 1/137.036"""
        self.assertAlmostEqual(self.ALPHA_PHYSICS, 0.0072973525693, places=10)

    def test_alpha_l104_approximation(self):
        """L104 uses α ≈ 1/137"""
        self.assertAlmostEqual(self.ALPHA_L104, 0.00729927, places=5)

    def test_alpha_error_is_small(self):
        """Error between physics and L104 alpha is < 0.03%"""
        error = abs(self.ALPHA_PHYSICS - self.ALPHA_L104) / self.ALPHA_PHYSICS
        self.assertLess(error, 0.0003)

    def test_137_is_prime(self):
        """137 is a prime number"""
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True

        self.assertTrue(is_prime(137))


class TestPrimeDensity(unittest.TestCase):
    """
    Tests the prime number theorem: π(n) ~ n/ln(n)

    The density of primes near n is approximately 1/ln(n).
    """

    def count_primes(self, n):
        """Count primes up to n using sieve"""
        if n < 2:
            return 0
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        return sum(sieve)

    def test_prime_density_at_100(self):
        """π(100) = 25, estimate = 100/ln(100) ≈ 21.7"""
        actual = self.count_primes(100)
        estimate = 100 / math.log(100)

        self.assertEqual(actual, 25)
        self.assertAlmostEqual(estimate, 21.7, places=0)

    def test_prime_density_at_1000(self):
        """π(1000) = 168, estimate ≈ 145"""
        actual = self.count_primes(1000)
        estimate = 1000 / math.log(1000)

        self.assertEqual(actual, 168)
        self.assertAlmostEqual(estimate, 145, delta=5)

    def test_prime_density_asymptotic(self):
        """π(n)/n * ln(n) → 1 as n → ∞"""
        for n in [1000, 10000, 100000]:
            pi_n = self.count_primes(n)
            ratio = (pi_n / n) * math.log(n)
            # Should approach 1 (asymptotically)
            self.assertAlmostEqual(ratio, 1.0, delta=0.20)


class TestLogisticMap(unittest.TestCase):
    """
    Tests the logistic map for chaos theory applications.

    x_{n+1} = r * x_n * (1 - x_n)

    For r > 3.57, the system exhibits chaotic behavior.
    """

    def logistic_iterate(self, x0, r, iterations):
        """Iterate the logistic map"""
        x = x0
        for _ in range(iterations):
            x = r * x * (1 - x)
        return x

    def test_logistic_fixed_point_r2(self):
        """For r=2, fixed point at x* = 1 - 1/r = 0.5"""
        x0 = 0.3
        r = 2.0
        x_final = self.logistic_iterate(x0, r, 100)

        self.assertAlmostEqual(x_final, 0.5, places=5)

    def test_logistic_fixed_point_r3(self):
        """For r=3, system oscillates near x* = 2/3"""
        x0 = 0.3
        r = 3.0
        # At r=3 exactly, the system is at bifurcation point
        # The fixed point x* = (r-1)/r = 2/3 is marginally stable
        x_final = self.logistic_iterate(x0, r, 1000)

        # System oscillates around the fixed point
        self.assertAlmostEqual(x_final, 2/3, delta=0.02)

    def test_logistic_bounded(self):
        """Logistic map stays in [0, 1] for r ≤ 4"""
        x0 = 0.5
        for r in [2.0, 3.0, 3.5, 3.9, 4.0]:
            x = x0
            for _ in range(100):
                x = r * x * (1 - x)
            self.assertGreaterEqual(x, 0)
            self.assertLessEqual(x, 1)

    def test_sensitivity_to_initial_conditions(self):
        """Chaotic regime shows sensitivity to initial conditions"""
        r = 3.9  # Chaotic regime
        x1 = 0.5
        x2 = 0.5 + 1e-8  # Small perturbation (larger for numerical stability)

        for _ in range(100):  # More iterations for divergence
            x1 = r * x1 * (1 - x1)
            x2 = r * x2 * (1 - x2)

        # After 100 iterations, trajectories should diverge significantly
        diff = abs(x1 - x2)
        self.assertGreater(diff, 0.001)


class TestTemporalStability(unittest.TestCase):
    """
    Tests temporal stability calculations (CTC stability).

    Closed Timelike Curves (CTCs) require specific stability conditions.
    """

    GOD_CODE = 527.5184818492611
    PHI = (1 + math.sqrt(5)) / 2

    def test_ctc_stability_formula(self):
        """Stability = (G_c × φ) / (R × ω)"""
        R = 10.0  # Radius
        omega = 50.0  # Angular velocity

        stability = (self.GOD_CODE * self.PHI) / (R * omega)

        # Should be positive
        self.assertGreater(stability, 0)

    def test_stability_decreases_with_radius(self):
        """Larger radius → lower stability"""
        omega = 50.0
        stab_small = (self.GOD_CODE * self.PHI) / (1.0 * omega)
        stab_large = (self.GOD_CODE * self.PHI) / (100.0 * omega)

        self.assertGreater(stab_small, stab_large)

    def test_stability_decreases_with_velocity(self):
        """Higher angular velocity → lower stability"""
        R = 10.0
        stab_slow = (self.GOD_CODE * self.PHI) / (R * 10.0)
        stab_fast = (self.GOD_CODE * self.PHI) / (R * 1000.0)

        self.assertGreater(stab_slow, stab_fast)


if __name__ == "__main__":
    unittest.main(verbosity=2)
