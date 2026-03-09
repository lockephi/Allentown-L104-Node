"""Differential and integral calculus of the God Code equation G(X).

dG/dX, integral of G(X)dX, Taylor expansion of God Code equation,
critical point analysis -- all at 100-decimal precision.
G(X) = 286^(1/phi) x 2^((416-X)/104)

Extracted from l104_quantum_numerical_builder.py (lines 3106-3257).
"""

from typing import Dict

from ..precision import D, decimal_pow, decimal_ln, decimal_exp, decimal_factorial
from ..constants import GOD_CODE_BASE_HP, LN2_HP, LN10_HP


class GodCodeCalculusEngine:
    """Differential and integral calculus of the God Code equation G(X)."""

    def __init__(self):
        """Initialize GodCodeCalculusEngine."""
        self.base = GOD_CODE_BASE_HP  # 286^(1/phi)
        self.ln2_over_104 = LN2_HP / D('104')

    def G(self, x: D) -> D:
        """G(X) = 286^(1/phi) x 2^((416-X)/104)."""
        exponent = (D('416') - x) / D('104')
        return self.base * decimal_pow(D('2'), exponent)

    def dG_dx(self, x: D) -> D:
        """Exact derivative: dG/dX = -G(X) x ln(2)/104."""
        return -self.G(x) * self.ln2_over_104

    def d2G_dx2(self, x: D) -> D:
        """Second derivative: d^2G/dX^2 = G(X) x (ln(2)/104)^2."""
        return self.G(x) * self.ln2_over_104 * self.ln2_over_104

    def dG_dx_numerical(self, x: D, h: D = D('1E-50')) -> D:
        """Numerical derivative using central difference at extreme precision."""
        return (self.G(x + h) - self.G(x - h)) / (D('2') * h)

    def integral_analytical(self, x_low: D, x_high: D) -> D:
        """Analytical integral: integral G(X)dX = -G(X) x 104/ln(2) + C."""
        coeff = D('-104') / LN2_HP
        return coeff * (self.G(x_high) - self.G(x_low))

    def integral_numerical(self, x_low: D, x_high: D, intervals: int = 1000) -> D:
        """Simpson's 1/3 rule numerical integration at 100-decimal precision."""
        h = (x_high - x_low) / D(str(intervals))
        total = self.G(x_low) + self.G(x_high)
        for i in range(1, intervals):
            x = x_low + D(str(i)) * h
            weight = D('4') if i % 2 == 1 else D('2')
            total += weight * self.G(x)
        return total * h / D('3')

    def taylor_expansion(self, x0: D, order: int = 10) -> Dict:
        """Taylor expansion of G(X) around X0 up to given order."""
        # G(X) = G(x0) x 2^(-(X-x0)/104)
        # = G(x0) x sum (-ln2/104)^n (X-x0)^n / n!
        g_x0 = self.G(x0)
        k = -self.ln2_over_104  # = -ln(2)/104  (negative)
        coefficients = []
        k_power = D('1')  # k^0 = 1
        for n in range(order + 1):
            coeff = g_x0 * k_power / decimal_factorial(n)
            coefficients.append({
                "n": n,
                "coefficient": str(coeff)[:50],
                "term": f"[{str(coeff)[:20]}...] x (X - {x0})^{n}",
            })
            k_power *= k  # k^(n+1)
        # Verify: evaluate Taylor at x0 + 1
        x_test = x0 + D('1')
        taylor_val = D('0')
        dx = x_test - x0
        k_power = D('1')
        dx_power = D('1')
        for n in range(order + 1):
            taylor_val += g_x0 * k_power / decimal_factorial(n) * dx_power
            k_power *= k
            dx_power *= dx
        exact_val = self.G(x_test)
        return {
            "x0": str(x0),
            "G(x0)": str(g_x0)[:50],
            "order": order,
            "coefficients": coefficients[:6],
            "taylor_at_x0_plus_1": str(taylor_val)[:50],
            "exact_at_x0_plus_1": str(exact_val)[:50],
            "matching_digits": self._count_matching(taylor_val, exact_val),
        }

    def critical_analysis(self) -> Dict:
        """Analyze the God Code function properties."""
        # G(X) is monotonically decreasing, no critical points
        # But we can find where G(X) = specific values
        x_vals = [D('0'), D('104'), D('208'), D('312'), D('416')]
        analysis = {}
        for x in x_vals:
            g = self.G(x)
            dg = self.dG_dx(x)
            d2g = self.d2G_dx2(x)
            analysis[f"X={x}"] = {
                "G(X)": str(g)[:50],
                "dG/dX": str(dg)[:50],
                "d^2G/dX^2": str(d2g)[:50],
                "log10_G": str(decimal_ln(g) / LN10_HP)[:30] if g > 0 else "N/A",
            }
        return analysis

    def derivative_verification(self) -> Dict:
        """Verify analytical vs numerical derivatives."""
        x0 = D('104')
        analytic = self.dG_dx(x0)
        numerical = self.dG_dx_numerical(x0)
        diff = abs(analytic - numerical)
        return {
            "x0": "104",
            "analytical_dG_dx": str(analytic)[:60],
            "numerical_dG_dx": str(numerical)[:60],
            "difference": str(diff)[:20],
            "matching_digits": self._count_matching(analytic, numerical),
        }

    def integral_verification(self) -> Dict:
        """Verify analytical vs numerical integration."""
        x_low, x_high = D('100'), D('200')
        analytic = self.integral_analytical(x_low, x_high)
        numerical = self.integral_numerical(x_low, x_high, 10000)
        diff = abs(analytic - numerical)
        return {
            "range": "100 to 200",
            "analytical": str(analytic)[:60],
            "numerical_simpson": str(numerical)[:60],
            "difference": str(diff)[:20],
            "matching_digits": self._count_matching(analytic, numerical),
        }

    def _count_matching(self, a: D, b: D) -> int:
        """Count matching decimal digits."""
        sa, sb = str(a), str(b)
        count = 0
        for c1, c2 in zip(sa, sb):
            if c1 == c2:
                count += 1
            elif c1 != '.' and c2 != '.':
                break
        return count

    def full_analysis(self) -> Dict:
        """Complete God Code calculus analysis."""
        return {
            "critical_analysis": self.critical_analysis(),
            "derivative_verification": self.derivative_verification(),
            "integral_verification": self.integral_verification(),
            "taylor_at_X104": self.taylor_expansion(D('104'), 8),
            "taylor_at_X0": self.taylor_expansion(D('0'), 8),
            "engine": "GodCodeCalculusEngine",
        }
