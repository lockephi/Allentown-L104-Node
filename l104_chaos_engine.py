#!/usr/bin/env python3

# [L104 EVO_49] Evolved: 2026-01-24
"""
L104 Chaos Engine
Implements chaos theory dynamics, strange attractors, and fractal generation
"""
import math
from dataclasses import dataclass
from typing import List, Tuple, Generator

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


# Sacred Constants
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895

@dataclass
class ChaosState:
    """State in a chaotic system"""
    x: float
    y: float
    z: float

    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

class LogisticMap:
    """Logistic map: x_{n+1} = r * x_n * (1 - x_n)"""

    def __init__(self, r: float = 3.9):
        self.r = r
        self.x = 0.5

    def iterate(self) -> float:
        self.x = self.r * self.x * (1 - self.x)
        return self.x

    def lyapunov_exponent(self, iterations: int = 1000) -> float:
        """Calculate Lyapunov exponent"""
        x = 0.5
        lyap_sum = 0.0
        for _ in range(iterations):
            derivative = abs(self.r * (1 - 2*x))
            if derivative > 0:
                lyap_sum += math.log(derivative)
            x = self.r * x * (1 - x)
        return lyap_sum / iterations

class LorenzAttractor:
    """Lorenz strange attractor"""

    def __init__(self, sigma: float = 10.0, rho: float = 28.0, beta: float = 8/3):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.state = ChaosState(1.0, 1.0, 1.0)
        self.dt = 0.01

    def step(self) -> ChaosState:
        """Advance one time step using Euler integration"""
        x, y, z = self.state.x, self.state.y, self.state.z

        dx = self.sigma * (y - x) * self.dt
        dy = (x * (self.rho - z) - y) * self.dt
        dz = (x * y - self.beta * z) * self.dt

        self.state = ChaosState(x + dx, y + dy, z + dz)
        return self.state

    def trajectory(self, steps: int) -> List[ChaosState]:
        """Generate trajectory of states"""
        return [self.step() for _ in range(steps)]

class RosslerAttractor:
    """Rossler strange attractor"""

    def __init__(self, a: float = 0.2, b: float = 0.2, c: float = 5.7):
        self.a = a
        self.b = b
        self.c = c
        self.state = ChaosState(0.1, 0.1, 0.1)
        self.dt = 0.01

    def step(self) -> ChaosState:
        x, y, z = self.state.x, self.state.y, self.state.z

        dx = (-y - z) * self.dt
        dy = (x + self.a * y) * self.dt
        dz = (self.b + z * (x - self.c)) * self.dt

        self.state = ChaosState(x + dx, y + dy, z + dz)
        return self.state

class MandelbrotGenerator:
    """Mandelbrot set membership testing"""

    def __init__(self, max_iter: int = 100):
        self.max_iter = max_iter

    def escape_time(self, c_real: float, c_imag: float) -> int:
        """Calculate escape time for complex point c"""
        z_real, z_imag = 0.0, 0.0

        for i in range(self.max_iter):
            if z_real**2 + z_imag**2 > 4:
                return i
            z_real, z_imag = z_real**2 - z_imag**2 + c_real, 2*z_real*z_imag + c_imag

        return self.max_iter

    def scan_region(self, x_min: float, x_max: float, y_min: float, y_max: float,
                   width: int, height: int) -> List[List[int]]:
        """Scan rectangular region and return escape times"""
        result = []
        for j in range(height):
            row = []
            for i in range(width):
                x = x_min + (x_max - x_min) * i / width
                y = y_min + (y_max - y_min) * j / height
                row.append(self.escape_time(x, y))
            result.append(row)
        return result

class ChaosEngine:
    """Main chaos computation engine"""

    def __init__(self):
        self.logistic = LogisticMap(GOD_CODE / 132)  # ~3.996 (chaotic)
        self.lorenz = LorenzAttractor()
        self.rossler = RosslerAttractor()
        self.mandelbrot = MandelbrotGenerator()

    def bifurcation_diagram(self, r_min: float = 2.5, r_max: float = 4.0,
                           r_steps: int = 100, iterations: int = 100) -> List[Tuple[float, List[float]]]:
        """Generate bifurcation diagram data"""
        result = []
        for i in range(r_steps):
            r = r_min + (r_max - r_min) * i / r_steps
            logistic = LogisticMap(r)

            # Discard transient
            for _ in range(200):
                logistic.iterate()

            # Collect attractor points
            points = [logistic.iterate() for _ in range(iterations)]
            result.append((r, points))

        return result

    def feigenbaum_constant(self) -> float:
        """Estimate Feigenbaum constant from bifurcation points"""
        # Known bifurcation points for logistic map
        r1 = 3.0
        r2 = 3.449490
        r3 = 3.544090
        r4 = 3.564407

        delta = (r2 - r1) / (r3 - r2)
        return delta  # Should approach ~4.669

if __name__ == "__main__":
    print("L104 Chaos Engine")
    engine = ChaosEngine()

    # Test Lyapunov exponent
    lyap = engine.logistic.lyapunov_exponent()
    print(f"Logistic map Lyapunov exponent: {lyap:.4f}")

    # Test Lorenz attractor
    trajectory = engine.lorenz.trajectory(100)
    print(f"Lorenz attractor - 100 steps, final: {trajectory[-1].tuple()}")

    # Test Feigenbaum
    delta = engine.feigenbaum_constant()
    print(f"Feigenbaum constant estimate: {delta:.4f}")
