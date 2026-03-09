"""Advanced number theory computations at 100-decimal precision.

Continued fractions, Pell equations, quadratic residues,
Fibonacci/Lucas identities, partition function, and modular
arithmetic.

Extracted from l104_quantum_numerical_builder.py (lines 2760-2948).
"""

from typing import Dict, List

from ..precision import D, decimal_sqrt, decimal_exp, _fibonacci_hp, lucas_number
from ..constants import PHI_HP, E_HP, PI_HP


class NumberTheoryForge:
    """Advanced number theory computations at 100-decimal precision."""

    def continued_fraction(self, value: D, depth: int = 50) -> Dict:
        """Compute the continued fraction expansion of a high-precision Decimal."""
        coefficients = []
        x = value
        for _ in range(depth):
            a = int(x)
            coefficients.append(a)
            frac = x - D(str(a))
            if frac < D('1E-100'):
                break
            x = D('1') / frac
        # Reconstruct convergents
        convergents = []
        h_prev, h_curr = D('0'), D('1')
        k_prev, k_curr = D('1'), D('0')
        for a in coefficients:
            a_d = D(str(a))
            h_prev, h_curr = h_curr, a_d * h_curr + h_prev
            k_prev, k_curr = k_curr, a_d * k_curr + k_prev
            if k_curr != D('0'):
                convergents.append(str(h_curr / k_curr)[:40])
        return {
            "value": str(value)[:50],
            "cf_coefficients": coefficients[:30],
            "depth": len(coefficients),
            "last_convergent": convergents[-1] if convergents else "",
            "convergents_sample": convergents[-5:] if len(convergents) >= 5 else convergents,
        }

    def golden_ratio_cf(self) -> Dict:
        """The golden ratio has the simplest continued fraction: [1; 1, 1, 1, ...]."""
        result = self.continued_fraction(PHI_HP)
        result["note"] = "phi = [1; 1, 1, 1, ...] -- the most irrational number"
        all_ones = all(c == 1 for c in result["cf_coefficients"])
        result["all_ones_verified"] = all_ones
        return result

    def sqrt_cf(self, n: int) -> Dict:
        """Continued fraction expansion of sqrt(n) (periodic for non-perfect-squares)."""
        val = decimal_sqrt(D(str(n)))
        result = self.continued_fraction(val)
        result["sqrt_of"] = n
        # Detect period
        coeffs = result["cf_coefficients"]
        if len(coeffs) > 3:
            a0 = coeffs[0]
            period_candidate = None
            for plen in range(1, len(coeffs) // 2):
                if coeffs[1:1+plen] == coeffs[1+plen:1+2*plen]:
                    period_candidate = coeffs[1:1+plen]
                    break
            if period_candidate:
                result["period"] = period_candidate
                result["period_length"] = len(period_candidate)
        return result

    def pell_equation(self, D_val: int, solutions: int = 5) -> Dict:
        """Solve x^2 - Dy^2 = 1 using continued fractions."""
        if int(D_val ** 0.5) ** 2 == D_val:
            return {"D": D_val, "error": "D is a perfect square"}
        # Get CF of sqrt(D)
        sqrt_d = decimal_sqrt(D(str(D_val)))
        a0 = int(sqrt_d)
        m, d_cf, a = 0, 1, a0
        cf = [a0]
        seen = set()
        for _ in range(200):
            m = d_cf * a - m
            d_cf = (D_val - m * m) // d_cf
            if d_cf == 0:
                break
            a = (a0 + m) // d_cf
            state = (m, d_cf)
            if state in seen:
                break
            seen.add(state)
            cf.append(a)
        # Find fundamental solution from convergents
        h_prev, h_curr = 0, 1
        k_prev, k_curr = 1, 0
        fund_x, fund_y = None, None
        for a_val in cf:
            h_prev, h_curr = h_curr, a_val * h_curr + h_prev
            k_prev, k_curr = k_curr, a_val * k_curr + k_prev
            if h_curr * h_curr - D_val * k_curr * k_curr == 1:
                fund_x, fund_y = h_curr, k_curr
                break
        sol_list = []
        if fund_x is not None:
            x, y = fund_x, fund_y
            for _ in range(solutions):
                sol_list.append({"x": x, "y": y, "check": x*x - D_val*y*y})
                x_new = fund_x * x + D_val * fund_y * y
                y_new = fund_x * y + fund_y * x
                x, y = x_new, y_new
        return {
            "D": D_val,
            "cf_sqrt_D": cf[:20],
            "fundamental_solution": {"x": fund_x, "y": fund_y} if fund_x else None,
            "solutions": sol_list,
        }

    def fibonacci_identities(self, n: int = 50) -> Dict:
        """Verify Fibonacci identities at high precision."""
        identities = {}
        # Cassini's identity: F(n-1)*F(n+1) - F(n)^2 = (-1)^n
        # Use raw integers (unlimited precision) for exact verification
        fn1_int = _fibonacci_hp(n - 1)
        fn_int = _fibonacci_hp(n)
        fn_plus1_int = _fibonacci_hp(n + 1)
        cassini_int = fn1_int * fn_plus1_int - fn_int * fn_int
        identities["cassini"] = {
            "n": n, "result": cassini_int,
            "expected": (-1)**n, "verified": cassini_int == (-1)**n,
        }
        # F(2n) = F(n)(2F(n+1) - F(n))
        f2n_int = _fibonacci_hp(2 * n)
        computed_int = fn_int * (2 * fn_plus1_int - fn_int)
        identities["doubling"] = {
            "F(2n)_direct": str(f2n_int)[:40],
            "F(n)(2F(n+1)-F(n))": str(computed_int)[:40],
            "verified": f2n_int == computed_int,
        }
        # Ratio F(n+1)/F(n) -> phi (this one needs Decimal for division)
        fn = D(str(fn_int))
        fn_plus1 = D(str(fn_plus1_int))
        ratio = fn_plus1 / fn if fn != 0 else D('0')
        diff = abs(ratio - PHI_HP)
        identities["golden_ratio_convergence"] = {
            "F(n+1)/F(n)": str(ratio)[:50],
            "phi": str(PHI_HP)[:50],
            "difference": str(diff)[:20],
            "matching_digits_to_phi": self._count_matching(ratio, PHI_HP),
        }
        return identities

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

    def partition_count(self, n: int) -> Dict:
        """Compute the partition function p(n) using dynamic programming."""
        p_table = [0] * (n + 1)
        p_table[0] = 1
        for k in range(1, n + 1):
            for j in range(k, n + 1):
                p_table[j] += p_table[j - k]
        return {
            "n": n,
            "p_n": p_table[n],
            "sample_values": {str(i): p_table[i] for i in range(min(n, 20) + 1)},
            "hardy_ramanujan_approx": float(
                decimal_exp(PI_HP * decimal_sqrt(D('2') * D(str(n)) / D('3')))
                / (D('4') * D(str(n)) * decimal_sqrt(D('3')))
            ) if n > 0 else 1,
        }

    def full_analysis(self) -> Dict:
        """Complete number theory analysis."""
        return {
            "golden_ratio_cf": self.golden_ratio_cf(),
            "sqrt2_cf": self.sqrt_cf(2),
            "sqrt5_cf": self.sqrt_cf(5),
            "e_cf": self.continued_fraction(E_HP),
            "pell_D61": self.pell_equation(61, 3),
            "pell_D109": self.pell_equation(109, 3),
            "fibonacci_identities": self.fibonacci_identities(300),
            "partition_100": self.partition_count(100),
            "partition_200": self.partition_count(200),
            "engine": "NumberTheoryForge v2",
        }
