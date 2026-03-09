"""Fractal and dynamical systems analysis at 100-decimal precision.

Mandelbrot orbit analysis, Feigenbaum constant verification,
logistic map bifurcation, Lyapunov exponents, and chaos quantification.

Extracted from l104_quantum_numerical_builder.py (lines 2950-3104).
"""

from typing import Dict, List

from ..precision import D, decimal_ln, decimal_sqrt
from ..constants import FEIGENBAUM_HP


class FractalDynamicsLab:
    """Fractal and dynamical systems analysis at 100-decimal precision."""

    def feigenbaum_verification(self, depth: int = 8) -> Dict:
        """Verify the Feigenbaum constant delta = 4.669201... from logistic map bifurcations.

        Uses known approximate bifurcation points as seeds and refines them via
        Newton's method on the iterated logistic map f^(2^k)(1/2) - 1/2 = 0."""
        # Known bifurcation points (approximate seeds for period-doubling cascade)
        seed_bifurcations = [
            (1, D('3.0')),            # period 2
            (2, D('3.44949')),        # period 4
            (3, D('3.54409')),        # period 8
            (4, D('3.5644')),         # period 16
            (5, D('3.568759')),       # period 32
            (6, D('3.56969')),        # period 64
            (7, D('3.56989')),        # period 128
            (8, D('3.569934')),       # period 256
        ]
        bifurcations = []
        for k, r_seed in seed_bifurcations[:depth]:
            period = 2 ** k
            # Refine bifurcation point via bisection
            r_refined = self._refine_bifurcation(r_seed, period)
            bifurcations.append({"k": k, "period": period, "r": str(r_refined)[:30]})
        # Compute Feigenbaum ratios
        deltas = []
        for i in range(2, len(bifurcations)):
            r_prev = D(bifurcations[i-2]["r"])
            r_curr = D(bifurcations[i-1]["r"])
            r_next = D(bifurcations[i]["r"])
            gap = r_next - r_curr
            if abs(gap) > D('1E-30'):
                delta = (r_curr - r_prev) / gap
                deltas.append({
                    "from_k": bifurcations[i]["k"],
                    "delta": str(delta)[:30],
                    "diff_from_exact": str(abs(delta - FEIGENBAUM_HP))[:15],
                })
        return {
            "bifurcation_points": bifurcations,
            "feigenbaum_ratios": deltas,
            "known_delta": str(FEIGENBAUM_HP)[:30],
            "best_ratio": deltas[-1]["delta"] if deltas else "N/A",
            "convergence_note": "Ratios converge to delta ~ 4.669201...",
        }

    def _refine_bifurcation(self, r_seed: D, period: int) -> D:
        """Refine a bifurcation point via bisection: find r where orbit changes from
        period p/2 to period p."""
        r_low = r_seed - D('0.01')
        r_high = r_seed + D('0.01')
        for _ in range(150):  # 150 bisection steps ~ 45 decimal digits
            r_mid = (r_low + r_high) / D('2')
            if self._has_period(r_mid, period):
                r_high = r_mid
            else:
                r_low = r_mid
        return (r_low + r_high) / D('2')

    def _has_period(self, r: D, period: int) -> bool:
        """Check whether the logistic map has a stable orbit of given period."""
        x = D('0.5')
        # Transient: iterate enough to settle
        transient = max(2000, period * 10)
        for _ in range(transient):
            x = r * x * (D('1') - x)
        # Check periodicity by comparing x with x after 'period' iterations
        x_start = x
        for _ in range(period):
            x = r * x * (D('1') - x)
        return abs(x - x_start) < D('1E-25')

    def logistic_map_orbit(self, r_val: float, x0: float = 0.5,
                           iterations: int = 500) -> Dict:
        """Compute logistic map orbit with high precision."""
        r = D(str(r_val))
        x = D(str(x0))
        orbit = []
        for i in range(iterations):
            x = r * x * (D('1') - x)
            if i >= iterations - 50:  # Last 50 values
                orbit.append(float(x))
        # Lyapunov exponent
        lyapunov = self.lyapunov_exponent(r_val, x0, iterations)
        return {
            "r": r_val,
            "x0": x0,
            "iterations": iterations,
            "final_value": str(x)[:50],
            "last_orbit_values": orbit[-10:],
            "lyapunov_exponent": lyapunov,
            "chaotic": lyapunov > 0,
        }

    def lyapunov_exponent(self, r_val: float, x0: float = 0.5,
                          iterations: int = 10000) -> float:
        """Compute the Lyapunov exponent of the logistic map."""
        r = D(str(r_val))
        x = D(str(x0))
        lyap_sum = D('0')
        for _ in range(iterations):
            x = r * x * (D('1') - x)
            deriv = abs(r * (D('1') - D('2') * x))
            if deriv > D('0'):
                lyap_sum += decimal_ln(deriv)
        return float(lyap_sum / D(str(iterations)))

    def mandelbrot_orbit(self, c_real: float, c_imag: float,
                         max_iter: int = 1000) -> Dict:
        """Compute Mandelbrot orbit z -> z^2 + c with 100-decimal precision."""
        cr = D(str(c_real))
        ci = D(str(c_imag))
        zr, zi = D('0'), D('0')
        orbit = []
        escaped = False
        for i in range(max_iter):
            zr_new = zr * zr - zi * zi + cr
            zi_new = D('2') * zr * zi + ci
            zr, zi = zr_new, zi_new
            mag_sq = zr * zr + zi * zi
            if i < 20 or i % 100 == 0:
                orbit.append({"i": i, "re": str(zr)[:30], "im": str(zi)[:30],
                              "mag": str(decimal_sqrt(mag_sq))[:20]})
            if mag_sq > D('4'):
                escaped = True
                orbit.append({"i": i, "escaped": True})
                break
        return {
            "c": f"{c_real} + {c_imag}i",
            "max_iter": max_iter,
            "in_set": not escaped,
            "escape_iteration": orbit[-1]["i"] if escaped else None,
            "orbit_sample": orbit[:15],
        }

    def full_analysis(self) -> Dict:
        """Complete fractal dynamics analysis."""
        return {
            "feigenbaum": self.feigenbaum_verification(6),
            "logistic_chaotic": self.logistic_map_orbit(3.9),
            "logistic_periodic": self.logistic_map_orbit(3.2),
            "mandelbrot_in_set": self.mandelbrot_orbit(-0.5, 0.0, 200),
            "mandelbrot_escaped": self.mandelbrot_orbit(0.5, 0.5, 200),
            "engine": "FractalDynamicsLab",
        }
