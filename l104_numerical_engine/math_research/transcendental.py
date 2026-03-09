"""Numerical proofs and analysis of transcendental/irrational properties.

Irrationality measure computation, continued fraction analysis,
algebraic independence tests, and numerical Lindemann-Weierstrass.

Extracted from l104_quantum_numerical_builder.py (lines 3259-3411).
"""

import itertools
from typing import Dict, List

from ..precision import D, decimal_sqrt, decimal_exp, decimal_ln, decimal_pow, decimal_factorial
from ..constants import PI_HP, E_HP, EULER_GAMMA_HP, PHI_HP


class TranscendentalProver:
    """Numerical proofs and analysis of transcendental/irrational properties."""

    def irrationality_measure(self, value: D, name: str = "value",
                               max_q: int = 10000) -> Dict:
        """Estimate the irrationality measure mu: how well value is approximated by p/q."""
        best_approximations = []
        best_err = D('Infinity')
        for q in range(1, max_q + 1):
            p = int(value * D(str(q)) + D('0.5'))  # Nearest integer
            err = abs(value - D(str(p)) / D(str(q)))
            if err < best_err and err > D('0'):
                best_err = err
                log_err = decimal_ln(err) / decimal_ln(D(str(q))) if q > 1 else D('0')
                best_approximations.append({
                    "p": p, "q": q,
                    "error": str(err)[:30],
                    "neg_log_err_over_log_q": str(-log_err)[:15] if q > 1 else "N/A",
                })
        # The irrationality measure is related to the supremum of -log|alpha-p/q|/log(q)
        measures = []
        for approx in best_approximations:
            if approx["neg_log_err_over_log_q"] != "N/A":
                measures.append(float(D(approx["neg_log_err_over_log_q"])))
        return {
            "name": name,
            "value": str(value)[:50],
            "best_rational_approximations": best_approximations[-10:],
            "estimated_irrationality_measure": max(measures) if measures else 2.0,
            "note": "mu=2 for algebraic irrationals, mu=inf for Liouville numbers",
        }

    def algebraic_independence_test(self, values: List[D], names: List[str]) -> Dict:
        """Numerical test for algebraic independence via LLL-style analysis."""
        # Simplified: check if integer linear combinations get close to zero
        results = {}
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                # Search for small integer relation a*v_i + b*v_j ~ c
                min_residual = D('Infinity')
                best_relation = None
                for a in range(-10, 11):
                    for b in range(-10, 11):
                        if a == 0 and b == 0:
                            continue
                        combo = D(str(a)) * values[i] + D(str(b)) * values[j]
                        c = int(combo + D('0.5'))
                        residual = abs(combo - D(str(c)))
                        if residual < min_residual:
                            min_residual = residual
                            best_relation = {"a": a, "b": b, "c": c, "residual": str(residual)[:30]}
                pair_name = f"{names[i]}_vs_{names[j]}"
                results[pair_name] = {
                    "best_integer_relation": best_relation,
                    "likely_independent": float(min_residual) > 1e-30,
                }
        return results

    def verify_e_transcendence(self) -> Dict:
        """Numerical evidence for the transcendence of e."""
        # Check that e^n is not rational for small n (component of Lindemann-Weierstrass)
        results = {}
        for n in range(1, 6):
            e_n = decimal_pow(E_HP, D(str(n)))
            # Find best rational approximation p/q with q < 10000
            best_err = D('Infinity')
            best_pq = (0, 1)
            for q in range(1, 5000):
                p = int(e_n * D(str(q)) + D('0.5'))
                err = abs(e_n - D(str(p)) / D(str(q)))
                if err < best_err:
                    best_err = err
                    best_pq = (p, q)
            results[f"e^{n}"] = {
                "value": str(e_n)[:40],
                "best_p_q": best_pq,
                "min_error": str(best_err)[:25],
                "far_from_rational": float(best_err) > 1e-10,
            }
        return {"e_power_irrationality": results}

    def verify_pi_transcendence(self) -> Dict:
        """Numerical evidence for pi transcendence: integer polynomial evaluation."""
        # Test coefficients: a0 + a1*pi + a2*pi^2 + ... for small integer a_i
        results = {}
        for degree in [1, 2, 3, 4, 5]:
            min_residual = D('Infinity')
            best_coeffs = None
            # Search over small integer coefficients
            coeff_range = range(-5, 6)
            for coeffs in itertools.product(coeff_range, repeat=degree + 1):
                if all(c == 0 for c in coeffs):
                    continue
                val = D(0)
                pi_pow = D(1)
                for c in coeffs:
                    val += D(c) * pi_pow
                    pi_pow *= PI_HP
                residual = abs(val)
                if residual < min_residual:
                    min_residual = residual
                    best_coeffs = coeffs
            results[f"degree_{degree}"] = {
                "best_coeffs": list(best_coeffs),
                "min_residual": str(min_residual)[:30],
                "far_from_zero": float(min_residual) > 1e-10,
            }
        return {"pi_polynomial_irrationality": results}

    def verify_euler_mascheroni_status(self) -> Dict:
        """Test whether gamma (Euler-Mascheroni) might be rational -- unknown!"""
        # Find best rational p/q approximation
        best_err = D('Infinity')
        best_pq = (0, 1)
        for q in range(1, 20000):
            p = int(EULER_GAMMA_HP * D(q) + D('0.5'))
            err = abs(EULER_GAMMA_HP - D(p) / D(q))
            if err < best_err:
                best_err = err
                best_pq = (p, q)
        return {
            "gamma_value": str(EULER_GAMMA_HP)[:50],
            "best_rational_p": best_pq[0],
            "best_rational_q": best_pq[1],
            "min_error": str(best_err)[:30],
            "note": "Rationality of gamma is UNKNOWN -- one of math's great open problems",
            "likely_irrational": float(best_err) > 1e-15,
        }

    def full_analysis(self) -> Dict:
        """Complete transcendental prover analysis."""
        return {
            "pi_irrationality": self.irrationality_measure(PI_HP, "pi", 5000),
            "phi_irrationality": self.irrationality_measure(PHI_HP, "phi", 5000),
            "e_irrationality": self.irrationality_measure(E_HP, "e", 5000),
            "e_transcendence": self.verify_e_transcendence(),
            "pi_transcendence": self.verify_pi_transcendence(),
            "euler_mascheroni_status": self.verify_euler_mascheroni_status(),
            "algebraic_independence": self.algebraic_independence_test(
                [PI_HP, E_HP, PHI_HP, EULER_GAMMA_HP],
                ["pi", "e", "phi", "gamma"],
            ),
            "engine": "TranscendentalProver v2",
        }
