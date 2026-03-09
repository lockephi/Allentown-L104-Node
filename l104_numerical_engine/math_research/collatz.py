"""Statistical and computational analysis of the Collatz conjecture.

Deep statistical analysis of the 3n+1 problem: stopping times,
record-breaking sequences, total stopping time distribution,
path length analysis, and probabilistic heuristics.

Extracted from l104_quantum_numerical_builder.py (lines 3814-3941).
"""

from collections import Counter
from typing import Dict, List

from ..precision import D, decimal_ln


class CollatzConjectureAnalyzer:
    """Statistical and computational analysis of the Collatz conjecture."""

    def collatz_sequence(self, n: int) -> List[int]:
        """Compute the full Collatz sequence from n to 1."""
        seq = [n]
        while n != 1 and len(seq) < 10000:
            n = n // 2 if n % 2 == 0 else 3 * n + 1
            seq.append(n)
        return seq

    def stopping_time(self, n: int) -> int:
        """Total stopping time: steps until reaching 1."""
        steps = 0
        while n != 1 and steps < 100000:
            n = n // 2 if n % 2 == 0 else 3 * n + 1
            steps += 1
        return steps

    def stopping_time_records(self, up_to: int = 10000) -> Dict:
        """Find stopping time records up to a given value."""
        records = []
        max_time = 0
        max_value = 0
        for n in range(2, up_to + 1):
            t = self.stopping_time(n)
            if t > max_time:
                max_time = t
                max_value = n
                records.append({"n": n, "stopping_time": t})
        return {
            "up_to": up_to,
            "record_holders": records[-15:],
            "max_stopping_time": max_time,
            "max_value": max_value,
        }

    def peak_value_analysis(self, up_to: int = 5000) -> Dict:
        """Analyze peak values in Collatz sequences -- how high do sequences fly?"""
        peak_records = []
        max_peak = 0
        for n in range(2, up_to + 1):
            seq = self.collatz_sequence(n)
            peak = max(seq)
            ratio = peak / n
            if peak > max_peak:
                max_peak = peak
                peak_records.append({"n": n, "peak": peak, "ratio": round(ratio, 2)})
        return {
            "peak_records": peak_records[-10:],
            "max_peak_value": max_peak,
        }

    def path_length_distribution(self, up_to: int = 10000) -> Dict:
        """Distribution of stopping times -- histogram analysis."""
        times = [self.stopping_time(n) for n in range(2, up_to + 1)]
        hist = Counter()
        for t in times:
            bucket = (t // 10) * 10  # 10-step buckets
            hist[bucket] += 1
        return {
            "up_to": up_to,
            "mean_stopping_time": round(sum(times) / len(times), 2),
            "median_stopping_time": sorted(times)[len(times) // 2],
            "max_stopping_time": max(times),
            "distribution_buckets": dict(sorted(hist.items())[:20]),
        }

    def glide_analysis(self, start_values: List[int] = None) -> Dict:
        """Analyze glide: count steps before first drop below starting value."""
        if start_values is None:
            start_values = [27, 97, 871, 6171, 77031, 837799]
        results = {}
        for n in start_values:
            seq = self.collatz_sequence(n)
            glide = 0
            for i, v in enumerate(seq[1:], 1):
                if v < n:
                    glide = i
                    break
            results[f"n={n}"] = {
                "glide_length": glide,
                "total_stopping_time": len(seq) - 1,
                "peak": max(seq),
                "peak_ratio": round(max(seq) / n, 2),
            }
        return results

    def odd_even_ratio(self, up_to: int = 5000) -> Dict:
        """Analyze odd/even step ratios -- connection to log(3/2)/log(2) heuristic."""
        ratios = []
        for n in range(2, up_to + 1):
            seq = self.collatz_sequence(n)
            odds = sum(1 for v in seq if v % 2 == 1)
            evens = len(seq) - odds
            if odds > 0:
                ratio = evens / odds
                ratios.append(ratio)
        mean_ratio = sum(ratios) / len(ratios) if ratios else 0
        # Expected ratio: log(3)/log(4) ~ 0.792 means ~1.26 evens per odd
        expected = float(decimal_ln(D(3)) / decimal_ln(D(4)))
        return {
            "mean_even_odd_ratio": round(mean_ratio, 6),
            "expected_heuristic": round(1.0 / expected, 6),
            "samples": len(ratios),
        }

    def full_analysis(self) -> Dict:
        """Run full analysis suite and return report."""
        return {
            "stopping_time_records": self.stopping_time_records(5000),
            "path_distribution": self.path_length_distribution(5000),
            "glide_analysis": self.glide_analysis(),
            "odd_even_ratio": self.odd_even_ratio(2000),
            "engine": "CollatzConjectureAnalyzer",
        }
