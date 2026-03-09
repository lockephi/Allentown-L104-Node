"""High-precision prime number theory computations.

Extracted from l104_quantum_numerical_builder.py (lines 2331-2540).
"""

from typing import Dict, List

from ..precision import D, decimal_ln, decimal_sqrt
from ..constants import TWIN_PRIME_CONST_HP


class PrimeNumberTheoryEngine:
    """High-precision prime number theory computations."""

    def __init__(self):
        """Initialize PrimeNumberTheoryEngine."""
        self.primes_cache: List[int] = []
        self._sieve_limit = 0

    def sieve(self, limit: int = 100000) -> List[int]:
        """Sieve of Eratosthenes up to limit."""
        if limit <= self._sieve_limit and self.primes_cache:
            return [p for p in self.primes_cache if p <= limit]
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        self.primes_cache = [i for i, v in enumerate(sieve) if v]
        self._sieve_limit = limit
        return self.primes_cache

    def miller_rabin_hp(self, n: int, witnesses: int = 25) -> bool:
        """Deterministic Miller-Rabin for small n, probabilistic for large n."""
        if n < 2:
            return False
        if n in (2, 3, 5, 7, 11, 13):
            return True
        if n % 2 == 0:
            return False
        # Write n-1 as 2^r * d
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        # Deterministic witnesses for n < 3.3e24
        det_witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in det_witnesses[:witnesses]:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    def prime_counting(self, n: int) -> Dict:
        """Compute pi(n) and compare with Li(n) and n/ln(n) approximations."""
        primes = self.sieve(n)
        pi_n = len(primes)
        # Li(n) approximation via numerical integration
        ln_n = decimal_ln(D(str(n)))
        n_over_ln = D(str(n)) / ln_n
        # Li(n) ~ n/ln(n) * (1 + 1/ln(n) + 2/ln(n)^2)
        li_n = n_over_ln * (D('1') + D('1')/ln_n + D('2')/(ln_n*ln_n))
        return {
            "n": n,
            "pi_n": pi_n,
            "n_over_ln_n": float(n_over_ln),
            "li_n_approx": float(li_n),
            "ratio_pi_to_li": float(D(str(pi_n)) / li_n) if li_n > 0 else 0,
            "relative_error_n_ln": float(abs(D(str(pi_n)) - n_over_ln) / D(str(pi_n))) if pi_n > 0 else 0,
        }

    def twin_primes(self, limit: int = 100000) -> Dict:
        """Find twin prime pairs up to limit."""
        primes = self.sieve(limit)
        twins = [(primes[i], primes[i+1])
                 for i in range(len(primes) - 1)
                 if primes[i+1] - primes[i] == 2]
        return {
            "limit": limit,
            "twin_pairs_found": len(twins),
            "first_10": twins[:10],
            "last_5": twins[-5:] if len(twins) >= 5 else twins,
            "twin_prime_constant_approx": str(TWIN_PRIME_CONST_HP)[:30],
        }

    def prime_gaps(self, limit: int = 100000) -> Dict:
        """Analyze distribution of prime gaps."""
        primes = self.sieve(limit)
        gaps = [primes[i+1] - primes[i] for i in range(len(primes) - 1)]
        gap_counts: Dict[int, int] = {}
        for g in gaps:
            gap_counts[g] = gap_counts.get(g, 0) + 1
        max_gap = max(gaps) if gaps else 0
        avg_gap = sum(gaps) / len(gaps) if gaps else 0
        return {
            "limit": limit,
            "primes_found": len(primes),
            "max_gap": max_gap,
            "avg_gap": round(avg_gap, 4),
            "gap_distribution_top10": dict(sorted(gap_counts.items(), key=lambda x: -x[1])[:10]),
            "mersenne_exponents_found": [p for p in primes[:30]
                                          if self.miller_rabin_hp(2**p - 1) and p <= 19],
        }

    def goldbach_verify(self, limit: int = 1000) -> Dict:
        """Verify Goldbach's conjecture for even numbers up to limit."""
        primes_list = self.sieve(limit)
        primes_set = set(primes_list)
        verified = 0
        violations = []
        for n in range(4, limit + 1, 2):
            found = False
            for p in primes_list:
                if p >= n:
                    break
                if n - p in primes_set:
                    found = True
                    break
            if found:
                verified += 1
            else:
                violations.append(n)
        return {
            "range": f"4 to {limit}",
            "even_numbers_tested": verified + len(violations),
            "verified": verified,
            "violations": violations,
            "conjecture_holds": len(violations) == 0,
        }

    def full_analysis(self) -> Dict:
        """Complete prime number theory analysis."""
        return {
            "prime_counting": self.prime_counting(100000),
            "twin_primes": self.twin_primes(100000),
            "prime_gaps": self.prime_gaps(100000),
            "goldbach": self.goldbach_verify(2000),
            "mertens_function": self.mertens_function(5000),
            "prime_reciprocal_sum": self.prime_reciprocal_sum(10000),
            "engine": "PrimeNumberTheoryEngine v2",
        }

    def mertens_function(self, limit: int = 5000) -> Dict:
        """Compute Mertens function M(n) = sum_{k=1}^{n} mu(k) where mu is Mobius function."""
        # Compute Mobius function via sieve
        mu = [0] * (limit + 1)
        mu[1] = 1
        is_prime = [True] * (limit + 1)
        primes = []
        for i in range(2, limit + 1):
            if is_prime[i]:
                primes.append(i)
                mu[i] = -1  # Prime: mu(p) = -1
            for p in primes:
                if i * p > limit:
                    break
                is_prime[i * p] = False
                if i % p == 0:
                    mu[i * p] = 0  # p^2 | n: mu(n) = 0
                    break
                mu[i * p] = -mu[i]
        # Accumulate M(n)
        M = [0] * (limit + 1)
        for k in range(1, limit + 1):
            M[k] = M[k - 1] + mu[k]
        # Record crossings and extrema
        extrema = []
        max_M, min_M = 0, 0
        for n in [10, 100, 500, 1000, 2000, 5000]:
            if n <= limit:
                if abs(M[n]) > max(abs(max_M), abs(min_M)):
                    pass
                extrema.append({"n": n, "M(n)": M[n]})
                max_M = max(max_M, M[n])
                min_M = min(min_M, M[n])
        return {
            "limit": limit,
            "checkpoints": extrema,
            "max_M": max_M,
            "min_M": min_M,
            "note": "Mertens conjecture (|M(n)| < sqrt(n)) disproved by Odlyzko-te Riele 1985",
        }

    def prime_reciprocal_sum(self, limit: int = 10000) -> Dict:
        """Compute sum 1/p for primes up to limit -- diverges as log(log(n)) per Mertens' theorem."""
        primes = self.sieve(limit)
        partial_sum = D(0)
        checkpoints = {}
        for p in primes:
            partial_sum += D(1) / D(p)
            if p in [10, 100, 1000, 5000, 10000]:
                ln_ln_p = decimal_ln(decimal_ln(D(p)))
                checkpoints[f"sum_to_{p}"] = {
                    "sum": str(partial_sum)[:30],
                    "log_log_p": str(ln_ln_p)[:20],
                    "mertens_M": str(partial_sum - ln_ln_p)[:20],
                }
        return {
            "total_primes": len(primes),
            "checkpoints": checkpoints,
            "mertens_constant_approx": str(partial_sum - decimal_ln(decimal_ln(D(limit))))[:30],
            "known_mertens_M": "0.2614972128...",
        }
