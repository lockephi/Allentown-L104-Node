"""Infinite series computation and convergence analysis at 100-decimal precision.

Ramanujan 1/pi, Basel problem, Catalan constant, Leibniz formula,
Chudnovsky, Wallis product, BBP, Machin -- all verified at 100-decimal
precision with convergence rate analysis.

Extracted from l104_quantum_numerical_builder.py (lines 2542-2758).
"""

from typing import Dict

from ..precision import (
    D, decimal_sqrt, decimal_pow, decimal_factorial, decimal_ln,
    decimal_exp, decimal_pi_machin,
)
from ..constants import PI_HP, LN2_HP, CATALAN_HP


class InfiniteSeriesLab:
    """Infinite series computation and convergence analysis at 100-decimal precision."""

    def ramanujan_pi(self, terms: int = 20) -> Dict:
        """Ramanujan's 1/pi formula: extraordinary convergence (~8 digits/term)."""
        # 1/pi = (2*sqrt(2)/9801) * sum (4k)!(1103+26390k) / ((k!)^4 396^(4k))
        coeff = D('2') * decimal_sqrt(D('2')) / D('9801')
        total = D('0')
        convergence = []
        for k in range(terms):
            num = decimal_factorial(4 * k) * (D('1103') + D('26390') * D(str(k)))
            den = (decimal_factorial(k) ** 4) * (D('396') ** (4 * k))
            total += num / den
            pi_approx = D('1') / (coeff * total)
            digits = self._matching_digits(pi_approx, PI_HP)
            convergence.append({"term": k, "digits_correct": digits})
        pi_final = D('1') / (coeff * total)
        return {
            "series": "Ramanujan_1_over_pi",
            "terms": terms,
            "pi_computed": str(pi_final)[:105],
            "pi_reference": str(PI_HP)[:105],
            "digits_correct": self._matching_digits(pi_final, PI_HP),
            "convergence_rate": convergence[-5:] if len(convergence) >= 5 else convergence,
            "digits_per_term": "~8",
        }

    def basel_problem(self, terms: int = 5000) -> Dict:
        """Basel problem: sum 1/n^2 = pi^2/6."""
        total = D('0')
        for n in range(1, terms + 1):
            total += D('1') / (D(str(n)) * D(str(n)))
        exact = PI_HP * PI_HP / D('6')
        return {
            "series": "Basel_problem",
            "terms": terms,
            "computed": str(total)[:60],
            "exact_pi2_over_6": str(exact)[:60],
            "matching_digits": self._matching_digits(total, exact),
        }

    def catalan_constant_series(self, terms: int = 3000) -> Dict:
        """Catalan constant G = sum (-1)^n / (2n+1)^2."""
        total = D('0')
        for n in range(terms):
            sign = D('1') if n % 2 == 0 else D('-1')
            total += sign / (D(str(2 * n + 1)) ** 2)
        return {
            "series": "Catalan_constant",
            "terms": terms,
            "computed": str(total)[:60],
            "reference": str(CATALAN_HP)[:60],
            "matching_digits": self._matching_digits(total, CATALAN_HP),
        }

    def leibniz_pi(self, terms: int = 50000) -> Dict:
        """Leibniz formula: pi/4 = 1 - 1/3 + 1/5 - 1/7 + ... (slow convergence)."""
        total = D('0')
        for n in range(terms):
            sign = D('1') if n % 2 == 0 else D('-1')
            total += sign / D(str(2 * n + 1))
        pi_approx = total * D('4')
        return {
            "series": "Leibniz_pi_over_4",
            "terms": terms,
            "pi_computed": str(pi_approx)[:60],
            "matching_digits": self._matching_digits(pi_approx, PI_HP),
            "convergence": "O(1/n) -- very slow",
        }

    def euler_product_zeta2(self, prime_limit: int = 10000) -> Dict:
        """Euler product for zeta(2): prod 1/(1-p^(-2)) = pi^2/6."""
        # Get primes
        sieve = [True] * (prime_limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(prime_limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, prime_limit + 1, i):
                    sieve[j] = False
        primes = [i for i, v in enumerate(sieve) if v]
        product = D('1')
        for p in primes:
            factor = D('1') / (D('1') - D('1') / (D(str(p)) * D(str(p))))
            product *= factor
        exact = PI_HP ** 2 / D('6')
        return {
            "series": "Euler_product_zeta_2",
            "primes_used": len(primes),
            "computed": str(product)[:60],
            "exact": str(exact)[:60],
            "matching_digits": self._matching_digits(product, exact),
        }

    def wallis_product(self, terms: int = 50000) -> Dict:
        """Wallis product: pi/2 = prod (4n^2)/(4n^2-1)."""
        product = D('1')
        for n in range(1, terms + 1):
            n2 = D(str(n)) * D(str(n)) * D('4')
            product *= n2 / (n2 - D('1'))
        pi_approx = product * D('2')
        return {
            "series": "Wallis_product",
            "terms": terms,
            "pi_computed": str(pi_approx)[:60],
            "matching_digits": self._matching_digits(pi_approx, PI_HP),
            "convergence": "O(1/n) -- slow",
        }

    def chudnovsky_pi(self, terms: int = 8) -> Dict:
        """Chudnovsky algorithm for pi -- ~14 digits per term."""
        C = D('426880') * decimal_sqrt(D('10005'))
        total = D('0')
        convergence = []
        for k in range(terms):
            num = decimal_factorial(6*k) * (D('13591409') + D('545140134') * D(str(k)))
            den = decimal_factorial(3*k) * (decimal_factorial(k) ** 3) * (D('-262537412640768000') ** k)
            total += num / den
            if total != D('0'):
                pi_approx = C / total
                digits = self._matching_digits(pi_approx, PI_HP)
                convergence.append({"term": k, "digits_correct": digits})
        pi_final = C / total
        return {
            "series": "Chudnovsky",
            "terms": terms,
            "pi_computed": str(pi_final)[:105],
            "digits_correct": self._matching_digits(pi_final, PI_HP),
            "convergence_rate": convergence,
            "digits_per_term": "~14",
        }

    def _matching_digits(self, a: D, b: D) -> int:
        """Count matching digits between values."""
        sa, sb = str(a), str(b)
        count = 0
        for c1, c2 in zip(sa, sb):
            if c1 == c2:
                count += 1
            elif c1 != '.' and c2 != '.':
                break
        return count

    def full_analysis(self) -> Dict:
        """Run all series analyses."""
        return {
            "chudnovsky": self.chudnovsky_pi(),
            "ramanujan": self.ramanujan_pi(),
            "basel": self.basel_problem(2000),
            "catalan": self.catalan_constant_series(1000),
            "euler_product": self.euler_product_zeta2(5000),
            "wallis": self.wallis_product(10000),
            "euler_transform_ln2": self.euler_transform_ln2(),
            "bbp_pi": self.bbp_pi_hex_digits(),
            "machin_pi": self.machin_pi_verification(),
            "engine": "InfiniteSeriesLab v2",
            "precision": "100-decimal",
        }

    def euler_transform_ln2(self, terms: int = 200) -> Dict:
        """Compute ln(2) via Euler-accelerated alternating series: ln(2) = sum(-1)^{n+1}/n.
        Euler transform converges exponentially instead of O(1/n)."""
        # Standard alternating harmonic
        alt = D(0)
        for k in range(1, terms + 1):
            sign = D(1) if k % 2 == 1 else D(-1)
            alt += sign / D(k)
        # Euler transform: sum a_n -> sum 2^{-(n+1)} sum_{k=0}^{n} C(n,k) a_k
        # For a_n = 1/(n+1): euler_ln2 = sum_{n=0}^{N} 1/((n+1) 2^{n+1})
        euler_sum = D(0)
        power_of_2 = D(2)
        for n in range(120):
            euler_sum += D(1) / (D(n + 1) * power_of_2)
            power_of_2 *= D(2)
        return {
            "standard_terms": terms,
            "standard_result": str(alt)[:60],
            "standard_digits": self._matching_digits(alt, LN2_HP),
            "euler_accelerated": str(euler_sum)[:60],
            "euler_digits": self._matching_digits(euler_sum, LN2_HP),
            "known_ln2": str(LN2_HP)[:60],
            "acceleration_factor": f"{self._matching_digits(euler_sum, LN2_HP) - self._matching_digits(alt, LN2_HP)}+ extra digits",
        }

    def bbp_pi_hex_digits(self) -> Dict:
        """Bailey-Borwein-Plouffe formula: pi = sum 1/16^k [4/(8k+1) - 2/(8k+4) - 1/(8k+5) - 1/(8k+6)].
        This can extract individual hex digits of pi."""
        total = D(0)
        for k in range(80):
            pk = D(16) ** (-k)
            term = pk * (D(4) / D(8 * k + 1) - D(2) / D(8 * k + 4) - D(1) / D(8 * k + 5) - D(1) / D(8 * k + 6))
            total += term
        return {
            "series": "Bailey-Borwein-Plouffe",
            "terms": 80,
            "pi_computed": str(total)[:105],
            "matching_digits": self._matching_digits(total, PI_HP),
            "note": "BBP allows extracting hex digits of pi without computing preceding digits",
        }

    def machin_pi_verification(self) -> Dict:
        """Verify pi via Machin's formula: pi/4 = 4*atan(1/5) - atan(1/239)."""
        pi_machin = decimal_pi_machin()
        return {
            "formula": "pi/4 = 4*atan(1/5) - atan(1/239)",
            "pi_computed": str(pi_machin)[:105],
            "matching_digits": self._matching_digits(pi_machin, PI_HP),
            "independent_verification": True,
        }
