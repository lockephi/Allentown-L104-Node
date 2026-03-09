"""Riemann zeta function and L-function analysis at 100-decimal precision.

Extracted from l104_quantum_numerical_builder.py (lines 2100-2321).
"""

from typing import Dict, List

from ..precision import (
    D, decimal_pow, decimal_factorial, decimal_sqrt, decimal_ln,
    decimal_cos, decimal_sin, decimal_bernoulli, decimal_gamma_lanczos,
)
from ..constants import PI_HP, LN2_HP, APERY_HP


class RiemannZetaEngine:
    """Riemann zeta function and L-function analysis at 100-decimal precision."""

    def __init__(self):
        """Initialize RiemannZetaEngine."""
        self.zeros_found: List[Dict] = []
        self.zeta_cache: Dict[str, D] = {}

    def zeta_real(self, s_val: float, terms: int = 200) -> D:
        """Compute \u03b6(s) for real s > 1 via Euler-Maclaurin with deep Bernoulli correction.

        Uses N direct terms + Euler-Maclaurin remainder with Bernoulli numbers B2..B30
        for extreme accuracy (50-100 digits depending on s and N)."""
        s = D(str(s_val))
        if s <= D('1'):
            return D('Infinity')
        cache_key = f"zeta_real_{s_val}_{terms}"
        if cache_key in self.zeta_cache:
            return self.zeta_cache[cache_key]
        # Direct summation
        total = D('0')
        N = terms
        for n in range(1, N + 1):
            total += D('1') / decimal_pow(D(str(n)), s)
        # Euler-Maclaurin remainder: N^(1-s)/(s-1) + N^(-s)/2
        ns = D(str(N))
        remainder = decimal_pow(ns, D('1') - s) / (s - D('1'))
        remainder += decimal_pow(ns, -s) / D('2')
        # Bernoulli corrections B_{2k}/(2k)! x s(s+1)...(s+2k-2) x N^{-(s+2k-1)}
        # Use Bernoulli numbers up to B30 for maximum accuracy
        bernoulli_even = [
            (2,  D('1') / D('6')),
            (4,  D('-1') / D('30')),
            (6,  D('1') / D('42')),
            (8,  D('-1') / D('30')),
            (10, D('5') / D('66')),
            (12, D('-691') / D('2730')),
            (14, D('7') / D('6')),
            (16, D('-3617') / D('510')),
            (18, D('43867') / D('798')),
            (20, D('-174611') / D('330')),
            (22, D('854513') / D('138')),
            (24, D('-236364091') / D('2730')),
            (26, D('8553103') / D('6')),
            (28, D('-23749461029') / D('870')),
            (30, D('8615841276005') / D('14322')),
        ]
        for two_k, b2k in bernoulli_even:
            # Pochhammer product: s(s+1)(s+2)...(s+2k-2)
            rising = D('1')
            for j in range(two_k - 1):
                rising *= (s + D(str(j)))
            correction = b2k / decimal_factorial(two_k) * rising * decimal_pow(ns, -(s + D(str(two_k - 1))))
            remainder += correction
        result = total + remainder
        self.zeta_cache[cache_key] = result
        return result

    def zeta_even_exact(self, n: int) -> D:
        """Exact \u03b6(2n) = (-1)^(n+1) B_{2n} (2\u03c0)^{2n} / (2(2n)!) for even positive integers."""
        if n < 1:
            return D('NaN')
        b2n = decimal_bernoulli(2 * n)
        two_pi_2n = decimal_pow(D('2') * PI_HP, D(str(2 * n)))
        sign = D('1') if (n + 1) % 2 == 0 else D('-1')
        return sign * b2n * two_pi_2n / (D('2') * decimal_factorial(2 * n))

    def dirichlet_eta(self, s_val: float, terms: int = 1000) -> D:
        """Dirichlet eta function \u03b7(s) = \u03a3 (-1)^(n-1) / n^s = (1 - 2^(1-s)) \u03b6(s)."""
        s = D(str(s_val))
        total = D('0')
        for n in range(1, terms + 1):
            sign = D('1') if n % 2 == 1 else D('-1')
            total += sign / decimal_pow(D(str(n)), s)
        return total

    def verify_known_values(self) -> Dict:
        """Verify \u03b6(s) against known closed-form values."""
        verifications = {}
        # \u03b6(2) = \u03c0\u00b2/6  -- use both Euler-Maclaurin and exact Bernoulli formula
        z2_em = self.zeta_real(2.0, 500)
        z2_exact_formula = self.zeta_even_exact(1)
        z2_exact = PI_HP * PI_HP / D('6')
        verifications["zeta_2"] = {
            "computed_euler_maclaurin": str(z2_em)[:60],
            "computed_bernoulli": str(z2_exact_formula)[:60],
            "exact": str(z2_exact)[:60],
            "matching_digits": self._count_matching(z2_em, z2_exact),
            "bernoulli_digits": self._count_matching(z2_exact_formula, z2_exact),
        }
        # \u03b6(4) = \u03c0\u2074/90
        z4_em = self.zeta_real(4.0, 300)
        z4_exact_formula = self.zeta_even_exact(2)
        z4_exact = PI_HP ** 4 / D('90')
        verifications["zeta_4"] = {
            "computed_euler_maclaurin": str(z4_em)[:60],
            "computed_bernoulli": str(z4_exact_formula)[:60],
            "exact": str(z4_exact)[:60],
            "matching_digits": self._count_matching(z4_em, z4_exact),
            "bernoulli_digits": self._count_matching(z4_exact_formula, z4_exact),
        }
        # \u03b6(6) = \u03c0\u2076/945
        z6 = self.zeta_real(6.0, 200)
        z6_exact_formula = self.zeta_even_exact(3)
        z6_exact = PI_HP ** 6 / D('945')
        verifications["zeta_6"] = {
            "computed": str(z6)[:60],
            "exact": str(z6_exact)[:60],
            "matching_digits": self._count_matching(z6, z6_exact),
            "bernoulli_digits": self._count_matching(z6_exact_formula, z6_exact),
        }
        # \u03b6(8) through \u03b6(20) via exact Bernoulli formula
        for n in range(4, 11):
            zn_exact = self.zeta_even_exact(n)
            verifications[f"zeta_{2*n}"] = {
                "bernoulli_exact": str(zn_exact)[:60],
                "precision": "100-decimal (closed-form via Bernoulli numbers)",
            }
        # Dirichlet eta \u03b7(1) = ln(2)
        eta1 = self.dirichlet_eta(1.0, 5000)
        verifications["eta_1"] = {
            "computed": str(eta1)[:60],
            "exact_ln2": str(LN2_HP)[:60],
            "matching_digits": self._count_matching(eta1, LN2_HP),
        }
        # Apery's constant \u03b6(3)
        z3 = self.zeta_real(3.0, 500)
        verifications["zeta_3_apery"] = {
            "computed": str(z3)[:60],
            "reference": str(APERY_HP)[:60],
            "matching_digits": self._count_matching(z3, APERY_HP),
        }
        return verifications

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

    def critical_strip_analysis(self, im_start: float = 14.0, im_end: float = 50.0,
                                 steps: int = 100) -> Dict:
        """Analyze the Riemann zeta function behavior near the critical line Re(s)=1/2."""
        # Approximate |\u03b6(1/2 + it)| using the Dirichlet series (truncated)
        results = []
        for i in range(steps):
            t = im_start + (im_end - im_start) * i / max(steps - 1, 1)
            magnitude = self._approx_zeta_critical(t)
            results.append({"t": round(t, 6), "magnitude": float(magnitude)})
        # Find approximate zero locations (magnitude minima)
        zeros = []
        for i in range(1, len(results) - 1):
            if (results[i]["magnitude"] < results[i-1]["magnitude"] and
                results[i]["magnitude"] < results[i+1]["magnitude"] and
                results[i]["magnitude"] < 0.5):
                zeros.append(results[i]["t"])
        return {
            "samples": len(results),
            "approx_zeros_near_critical_line": zeros,
            "known_first_zero_t": 14.134725,
            "min_magnitude": min(r["magnitude"] for r in results),
        }

    def _approx_zeta_critical(self, t: float, terms: int = 150) -> D:
        """Approximate |\u03b6(1/2 + it)| via partial Dirichlet series."""
        real_part = D('0')
        imag_part = D('0')
        for n in range(1, terms + 1):
            # n^(-1/2) * (cos(t*ln(n)) - i*sin(t*ln(n)))
            n_d = D(str(n))
            inv_sqrt_n = D('1') / decimal_sqrt(n_d)
            ln_n = decimal_ln(n_d) if n > 1 else D('0')
            angle = D(str(t)) * ln_n
            real_part += inv_sqrt_n * decimal_cos(angle)
            imag_part -= inv_sqrt_n * decimal_sin(angle)
        return decimal_sqrt(real_part * real_part + imag_part * imag_part)

    def functional_equation_test(self, s_val: float = 3.0) -> Dict:
        """Test \u03b6(s) = 2^s \u03c0^(s-1) sin(\u03c0s/2) \u0393(1-s) \u03b6(1-s) numerically."""
        s = D(str(s_val))
        lhs = self.zeta_real(s_val, 400)
        # For integer s > 1, we compute the RHS components
        two_s = decimal_pow(D('2'), s)
        pi_s1 = decimal_pow(PI_HP, s - D('1'))
        sin_val = decimal_sin(PI_HP * s / D('2'))
        gamma_arg = D('1') - s
        gamma_val = decimal_gamma_lanczos(gamma_arg)
        gamma_is_pole = str(gamma_val) == 'Infinity'
        one_minus_s = float(D('1') - s)
        zeta_1ms = self.zeta_real(one_minus_s, 400) if one_minus_s > 1 else D('NaN')
        return {
            "s": s_val,
            "zeta_s": str(lhs)[:60],
            "functional_eq_components": {
                "2^s": str(two_s)[:40],
                "pi^(s-1)": str(pi_s1)[:40],
                "sin(pi*s/2)": str(sin_val)[:40],
                "gamma(1-s)": "pole (Infinity)" if gamma_is_pole else str(gamma_val)[:40],
            },
            "note": "Full verification requires analytic continuation for Re(s)<1"
                    + (" -- \u0393(1-s) has a pole at s=" + str(s_val) if gamma_is_pole else ""),
        }

    def full_analysis(self) -> Dict:
        """Complete Riemann zeta analysis."""
        return {
            "known_value_verification": self.verify_known_values(),
            "critical_strip": self.critical_strip_analysis(),
            "functional_eq_test": self.functional_equation_test(3.0),
            "zeta_even_exact": {
                f"zeta_{2*n}": str(self.zeta_even_exact(n))[:60]
                for n in range(1, 11)
            },
            "engine": "RiemannZetaEngine v2",
            "precision": "100-decimal (Bernoulli closed-form for even \u03b6)",
        }
