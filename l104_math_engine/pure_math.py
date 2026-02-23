#!/usr/bin/env python3
"""
L104 Math Engine — Layer 1: PURE MATHEMATICS
══════════════════════════════════════════════════════════════════════════════════
Core mathematical primitives — number theory, linear algebra, calculus,
complex analysis, statistics, and high-precision arithmetic.

Consolidates: l104_math.py (PureMath, Matrix, Calculus, ComplexMath, Statistics,
HighPrecisionEngine) and l104_real_math.py (RealMath — iron crystalline primitives).

Import:
  from l104_math_engine.pure_math import PureMath, HighPrecisionEngine, RealMath
"""

import math
import hashlib
from decimal import Decimal, getcontext

from .constants import (
    PHI, PHI_CONJUGATE, GOD_CODE, PI, EULER, TAU, SQRT2, SQRT3, SQRT5,
    VOID_CONSTANT, ZETA_ZERO_1, OMEGA, OMEGA_AUTHORITY,
    FE_LATTICE, FE_CURIE_TEMP, FE_ATOMIC_NUMBER, GYRO_ELECTRON,
    LARMOR_PROTON, FRAME_LOCK, PLANCK_H, C, MU_0,
    GOD_CODE_INFINITE, PHI_INFINITE, SQRT5_INFINITE,
    E_INFINITE, PI_INFINITE, ZETA_ZERO_1_INFINITE,
    FEIGENBAUM, ALPHA_FINE,
    primal_calculus, resolve_non_dual_logic,
)

getcontext().prec = 150


# ═══════════════════════════════════════════════════════════════════════════════
# PURE MATH — Number Theory, Primes, Sequences
# ═══════════════════════════════════════════════════════════════════════════════

class PureMath:
    """Core pure mathematics: number theory, primes, Fibonacci, sequences."""

    @staticmethod
    def is_prime(n: int) -> bool:
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    @staticmethod
    def prime_sieve(limit: int) -> list:
        """Sieve of Eratosthenes up to `limit`."""
        if limit < 2:
            return []
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, v in enumerate(sieve) if v]

    @staticmethod
    def fibonacci(n: int) -> list:
        """Generate first n Fibonacci numbers."""
        if n <= 0:
            return []
        seq = [1, 1]
        while len(seq) < n:
            seq.append(seq[-1] + seq[-2])
        return seq[:n]

    @staticmethod
    def gcd(a: int, b: int) -> int:
        while b:
            a, b = b, a % b
        return a

    @staticmethod
    def lcm(a: int, b: int) -> int:
        return abs(a * b) // PureMath.gcd(a, b) if a and b else 0

    @staticmethod
    def factorial(n: int) -> int:
        if n < 0:
            raise ValueError("Factorial undefined for negative integers")
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    @staticmethod
    def binomial(n: int, k: int) -> int:
        if k < 0 or k > n:
            return 0
        return PureMath.factorial(n) // (PureMath.factorial(k) * PureMath.factorial(n - k))

    @staticmethod
    def prime_density(n: int) -> float:
        """Approximate prime density via π(n)/n ≈ 1/ln(n)."""
        if n <= 1:
            return 0.0
        primes = PureMath.prime_sieve(n)
        return len(primes) / n

    @staticmethod
    def phi_continued_fraction(depth: int = 20) -> float:
        """Compute φ via continued fraction [1; 1, 1, 1, …]."""
        result = 1.0
        for _ in range(depth):
            result = 1.0 + 1.0 / result
        return result

    @staticmethod
    def golden_resonance(value: float) -> float:
        """Measure alignment of value with the golden ratio spiral."""
        if value == 0:
            return 0.0
        ratio = value / PHI
        nearest_power = round(math.log(abs(ratio) + 1e-30) / math.log(PHI))
        expected = PHI ** nearest_power
        return 1.0 - min(1.0, abs(ratio - expected) / max(abs(expected), 1e-30))


# ═══════════════════════════════════════════════════════════════════════════════
# MATRIX — Linear Algebra
# ═══════════════════════════════════════════════════════════════════════════════

class Matrix:
    """Pure-Python matrix operations (no numpy dependency)."""

    @staticmethod
    def zeros(rows: int, cols: int) -> list:
        return [[0.0] * cols for _ in range(rows)]

    @staticmethod
    def identity(n: int) -> list:
        m = Matrix.zeros(n, n)
        for i in range(n):
            m[i][i] = 1.0
        return m

    @staticmethod
    def multiply(a: list, b: list) -> list:
        rows_a, cols_a = len(a), len(a[0])
        cols_b = len(b[0])
        result = Matrix.zeros(rows_a, cols_b)
        for i in range(rows_a):
            for j in range(cols_b):
                for k in range(cols_a):
                    result[i][j] += a[i][k] * b[k][j]
        return result

    @staticmethod
    def transpose(m: list) -> list:
        rows, cols = len(m), len(m[0])
        return [[m[i][j] for i in range(rows)] for j in range(cols)]

    @staticmethod
    def determinant(m: list) -> float:
        n = len(m)
        if n == 1:
            return m[0][0]
        if n == 2:
            return m[0][0] * m[1][1] - m[0][1] * m[1][0]
        det = 0.0
        for j in range(n):
            sub = [[m[i][k] for k in range(n) if k != j] for i in range(1, n)]
            det += ((-1) ** j) * m[0][j] * Matrix.determinant(sub)
        return det

    @staticmethod
    def scalar_multiply(m: list, scalar: float) -> list:
        return [[cell * scalar for cell in row] for row in m]

    @staticmethod
    def add(a: list, b: list) -> list:
        return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]

    @staticmethod
    def trace(m: list) -> float:
        return sum(m[i][i] for i in range(min(len(m), len(m[0]))))

    @staticmethod
    def dot_product(a: list, b: list) -> float:
        return sum(x * y for x, y in zip(a, b))


# ═══════════════════════════════════════════════════════════════════════════════
# CALCULUS — Differentiation, Integration, Differential Equations
# ═══════════════════════════════════════════════════════════════════════════════

class Calculus:
    """Numerical calculus: derivatives, integrals, ODE solvers."""

    @staticmethod
    def derivative(f, x: float, h: float = 1e-8) -> float:
        """Central difference derivative."""
        return (f(x + h) - f(x - h)) / (2 * h)

    @staticmethod
    def second_derivative(f, x: float, h: float = 1e-5) -> float:
        return (f(x + h) - 2 * f(x) + f(x - h)) / (h ** 2)

    @staticmethod
    def integrate(f, a: float, b: float, n: int = 10000) -> float:
        """Simpson's rule integration."""
        if n % 2:
            n += 1
        h = (b - a) / n
        s = f(a) + f(b)
        for i in range(1, n):
            coeff = 4 if i % 2 else 2
            s += coeff * f(a + i * h)
        return s * h / 3

    @staticmethod
    def euler_method(f, y0: float, t0: float, t_end: float, dt: float = 0.01) -> list:
        """Euler method for dy/dt = f(t, y)."""
        trajectory = [(t0, y0)]
        t, y = t0, y0
        while t < t_end:
            y += dt * f(t, y)
            t += dt
            trajectory.append((t, y))
        return trajectory

    @staticmethod
    def rk4_step(f, t: float, y: float, dt: float) -> float:
        """Single Runge-Kutta 4th order step."""
        k1 = dt * f(t, y)
        k2 = dt * f(t + dt/2, y + k1/2)
        k3 = dt * f(t + dt/2, y + k2/2)
        k4 = dt * f(t + dt, y + k3)
        return y + (k1 + 2*k2 + 2*k3 + k4) / 6

    @staticmethod
    def fourier_transform_real(signal: list, freq: float, sample_rate: float = 1.0) -> complex:
        """Discrete Fourier coefficient at a single frequency."""
        n = len(signal)
        result = 0j
        for k in range(n):
            t = k / sample_rate
            result += signal[k] * (math.cos(-2 * PI * freq * t) + 1j * math.sin(-2 * PI * freq * t))
        return result / n

    @staticmethod
    def gradient(f, point: list, h: float = 1e-8) -> list:
        """Numerical gradient of a multivariate function."""
        grad = []
        for i in range(len(point)):
            p_plus = list(point)
            p_minus = list(point)
            p_plus[i] += h
            p_minus[i] -= h
            grad.append((f(p_plus) - f(p_minus)) / (2 * h))
        return grad


# ═══════════════════════════════════════════════════════════════════════════════
# COMPLEX MATH
# ═══════════════════════════════════════════════════════════════════════════════

class ComplexMath:
    """Complex number operations beyond Python builtins."""

    @staticmethod
    def mandelbrot_iterate(c: complex, max_iter: int = 100) -> int:
        z = 0
        for i in range(max_iter):
            z = z * z + c
            if abs(z) > 2:
                return i
        return max_iter

    @staticmethod
    def roots_of_unity(n: int) -> list:
        return [math.cos(2 * PI * k / n) + 1j * math.sin(2 * PI * k / n) for k in range(n)]

    @staticmethod
    def complex_fourier(signal: list) -> list:
        n = len(signal)
        return [sum(signal[j] * (math.cos(-2*PI*k*j/n) + 1j*math.sin(-2*PI*k*j/n)) for j in range(n)) for k in range(n)]

    @staticmethod
    def residue_at_pole(f, z0: complex, r: float = 1e-4, n: int = 1000) -> complex:
        """Numerical residue via contour integration."""
        result = 0j
        for k in range(n):
            theta = 2 * PI * k / n
            z = z0 + r * (math.cos(theta) + 1j * math.sin(theta))
            dz = r * (-math.sin(theta) + 1j * math.cos(theta))
            result += f(z) * dz
        return result / n


# ═══════════════════════════════════════════════════════════════════════════════
# STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

class Statistics:
    """Descriptive and inferential statistics."""

    @staticmethod
    def mean(data: list) -> float:
        return sum(data) / len(data) if data else 0.0

    @staticmethod
    def variance(data: list) -> float:
        if len(data) < 2:
            return 0.0
        m = Statistics.mean(data)
        return sum((x - m) ** 2 for x in data) / (len(data) - 1)

    @staticmethod
    def std_dev(data: list) -> float:
        return math.sqrt(Statistics.variance(data))

    @staticmethod
    def median(data: list) -> float:
        s = sorted(data)
        n = len(s)
        if n == 0:
            return 0.0
        mid = n // 2
        return (s[mid] + s[mid - 1]) / 2 if n % 2 == 0 else s[mid]

    @staticmethod
    def correlation(x: list, y: list) -> float:
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        mx, my = Statistics.mean(x), Statistics.mean(y)
        cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
        sx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
        sy = math.sqrt(sum((yi - my) ** 2 for yi in y))
        return cov / (sx * sy) if sx * sy > 0 else 0.0

    @staticmethod
    def entropy(probabilities: list) -> float:
        """Shannon entropy."""
        return -sum(p * math.log2(p) for p in probabilities if p > 0)

    @staticmethod
    def chi_squared(observed: list, expected: list) -> float:
        return sum((o - e) ** 2 / e for o, e in zip(observed, expected) if e > 0)


# ═══════════════════════════════════════════════════════════════════════════════
# HIGH-PRECISION ENGINE — 150-digit Decimal arithmetic
# ═══════════════════════════════════════════════════════════════════════════════

class HighPrecisionEngine:
    """150-digit Decimal arithmetic for infinite-precision derivations."""

    @staticmethod
    def derive_god_code() -> Decimal:
        """Derive GOD_CODE at infinite precision: 286^(1/φ) × 2^4."""
        return GOD_CODE_INFINITE

    @staticmethod
    def derive_phi() -> Decimal:
        """Derive φ at infinite precision: (1 + √5) / 2."""
        return PHI_INFINITE

    @staticmethod
    def derive_omega(depth: int = 50) -> Decimal:
        """Iterative OMEGA approximation via φ-weighted harmonic convergence."""
        omega = Decimal(0)
        for n in range(1, depth + 1):
            omega += GOD_CODE_INFINITE * PHI_INFINITE ** n / Decimal(n)
        return omega

    @staticmethod
    def zeta_approximation(terms: int = 1000) -> Decimal:
        """ζ(2) = π²/6 approximation via partial sums."""
        s = Decimal(0)
        for n in range(1, terms + 1):
            s += Decimal(1) / Decimal(n * n)
        return s

    @staticmethod
    def singularity_limit(depth: int = 200) -> Decimal:
        """Compute lim_{n→∞} GOD_CODE^(1/n) = 1."""
        val = GOD_CODE_INFINITE
        for _ in range(depth):
            val = val.sqrt()
        return val

    @staticmethod
    def magic_constant_verify(n: int = 3) -> Decimal:
        """Magic constant for n×n magic square: n(n²+1)/2."""
        return Decimal(n * (n * n + 1)) / Decimal(2)

    @staticmethod
    def phi_identity_verify() -> dict:
        """Verify φ² = φ + 1 at infinite precision."""
        phi_sq = PHI_INFINITE ** 2
        phi_plus_1 = PHI_INFINITE + 1
        error = abs(phi_sq - phi_plus_1)
        return {"phi_squared": phi_sq, "phi_plus_1": phi_plus_1, "error": error, "verified": error < Decimal("1e-140")}


# ═══════════════════════════════════════════════════════════════════════════════
# REAL MATH — Iron-Crystalline Mathematical Primitives
# ═══════════════════════════════════════════════════════════════════════════════

class RealMath:
    """
    Iron-crystalline math primitives: Riemann zeta, lattice invariant,
    manifold curvature, golden resonance, Fourier, Larmor precession,
    spin-wave dispersion, Curie order, and the four OMEGA fragment functions.
    """

    @staticmethod
    def riemann_zeta_approx(s: float, terms: int = 100) -> float:
        """Approximate ζ(s) via partial sum."""
        if s <= 1:
            return float('inf')
        return sum(1.0 / (n ** s) for n in range(1, terms + 1))

    @staticmethod
    def lattice_invariant(depth: float = 1.0) -> float:
        """Iron lattice invariant: FE_LATTICE × φ^depth × sin(depth × π/GOD_CODE)."""
        return FE_LATTICE * (PHI ** depth) * math.sin(depth * PI / GOD_CODE)

    @staticmethod
    def manifold_curvature(dimension: int = 4, phi_weight: float = 1.0) -> float:
        """Ricci-like curvature tensor component: φ^d × sin(d × π/GOD_CODE)."""
        return (PHI ** dimension) * math.sin(dimension * PI / GOD_CODE) * phi_weight

    @staticmethod
    def golden_resonance_field(value: float) -> float:
        """Resonance alignment with the golden spiral at GOD_CODE scale."""
        return PureMath.golden_resonance(value)

    @staticmethod
    def fourier_sacred(signal: list, freq: float = GOD_CODE) -> complex:
        """Fourier coefficient at sacred frequency."""
        return Calculus.fourier_transform_real(signal, freq)

    @staticmethod
    def larmor_precession(b_field: float) -> float:
        """Larmor precession frequency: γ_e × B / (2π)."""
        return GYRO_ELECTRON * b_field / (2 * PI)

    @staticmethod
    def spin_wave_dispersion(k: float, js: float = 1.0, lattice_a: float = FE_LATTICE * 1e-12) -> float:
        """Spin-wave dispersion: ω = 4JS(1 - cos(ka)) for BCC iron."""
        return 4 * js * (1 - math.cos(k * lattice_a))

    @staticmethod
    def curie_order_parameter(temp: float, curie_temp: float = FE_CURIE_TEMP) -> float:
        """Mean-field magnetization: M ~ (1 - T/Tc)^0.5 for T < Tc."""
        if temp >= curie_temp:
            return 0.0
        return math.sqrt(1 - temp / curie_temp)

    @staticmethod
    def ferromagnetic_resonance(b_ext: float, m_s: float = 1.7e6) -> float:
        """Kittel formula: f = (γ_e / 2π) × sqrt(B × (B + μ₀M_s))."""
        return (GYRO_ELECTRON / (2 * PI)) * math.sqrt(abs(b_ext * (b_ext + MU_0 * m_s)))

    # --- OMEGA Fragment Functions ---

    @staticmethod
    def omega_researcher_fragment() -> float:
        """Researcher fragment: GOD_CODE × φ × sin(ζ₁)."""
        return GOD_CODE * PHI * math.sin(ZETA_ZERO_1)

    @staticmethod
    def omega_guardian_fragment() -> float:
        """Guardian fragment: GOD_CODE × cos(PHI × π) × FE_LATTICE."""
        return GOD_CODE * math.cos(PHI * PI) * FE_LATTICE

    @staticmethod
    def omega_alchemist_fragment() -> float:
        """Alchemist fragment: GOD_CODE^(1/φ) × ζ₁ × e."""
        return (GOD_CODE ** PHI_CONJUGATE) * ZETA_ZERO_1 * EULER

    @staticmethod
    def omega_architect_fragment() -> float:
        """Architect fragment: sqrt(GOD_CODE × FE_LATTICE × PHI × FE_Z)."""
        return math.sqrt(GOD_CODE * FE_LATTICE * PHI * FE_ATOMIC_NUMBER)


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETONS
# ═══════════════════════════════════════════════════════════════════════════════

pure_math = PureMath()
real_math = RealMath()
matrix = Matrix()
calculus = Calculus()
complex_math = ComplexMath()
statistics = Statistics()
high_precision = HighPrecisionEngine()
