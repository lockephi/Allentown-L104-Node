"""Statistical mechanics computations applied to the token lattice.

Boltzmann partition functions, entropy landscapes, free energy
computation, and God Code thermodynamics.

Extracted from l104_quantum_numerical_builder.py (lines 3413-3558).

Note: This engine accepts a lattice object in its constructor. The lattice
must expose a ``.tokens`` dict whose values have ``.value`` and ``.tier``
attributes. No direct import of TokenLatticeEngine is required.
"""

from typing import Any, Dict, List

from ..precision import D, decimal_exp, decimal_ln, decimal_factorial, decimal_sqrt


class StatisticalMechanicsEngine:
    """Statistical mechanics computations applied to the token lattice."""

    def __init__(self, lattice: Any):
        """Initialize StatisticalMechanicsEngine."""
        self.lattice = lattice

    def partition_function(self, beta_vals: List[float] = None) -> Dict:
        """Compute the canonical partition function Z(beta) = sum exp(-beta E_i)."""
        if beta_vals is None:
            beta_vals = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        # Energy levels from token values
        energies = []
        for token in self.lattice.tokens.values():
            try:
                v = D(token.value) if isinstance(token.value, str) else token.value
                if v > D('0'):
                    energies.append(decimal_ln(v))
            except Exception:
                pass
        if not energies:
            return {"error": "No tokens in lattice"}
        results = {}
        for beta in beta_vals:
            b = D(str(beta))
            Z = D('0')
            for e in energies:
                Z += decimal_exp(-b * e)
            avg_E = D('0')
            for e in energies:
                avg_E += e * decimal_exp(-b * e) / Z
            # Entropy S = beta<E> + ln(Z)
            S = b * avg_E + decimal_ln(Z)
            # Free energy F = -ln(Z)/beta
            F = -decimal_ln(Z) / b if b > D('0') else D('0')
            results[f"beta={beta}"] = {
                "Z": str(Z)[:40],
                "avg_energy": str(avg_E)[:30],
                "entropy_S": str(S)[:30],
                "free_energy_F": str(F)[:30],
            }
        return {"partition_function": results, "energy_levels": len(energies)}

    def boltzmann_distribution(self, beta: float = 1.0) -> Dict:
        """Compute Boltzmann probability distribution over token states."""
        b = D(str(beta))
        energies = {}
        for name, token in self.lattice.tokens.items():
            try:
                v = D(token.value) if isinstance(token.value, str) else token.value
                if v > D('0'):
                    energies[name] = decimal_ln(v)
            except Exception:
                pass
        if not energies:
            return {"error": "No tokens"}
        Z = D('0')
        for e in energies.values():
            Z += decimal_exp(-b * e)
        distribution = {}
        for name, e in energies.items():
            prob = decimal_exp(-b * e) / Z
            distribution[name] = {
                "energy": str(e)[:20],
                "probability": str(prob)[:20],
            }
        # Sort by probability
        sorted_dist = dict(sorted(distribution.items(),
                                   key=lambda x: D(x[1]["probability"]),
                                   reverse=True)[:15])
        entropy = D('0')
        for d in distribution.values():
            p = D(d["probability"])
            if p > D('0'):
                entropy -= p * decimal_ln(p)
        return {
            "beta": beta,
            "tokens": len(distribution),
            "top_15_by_probability": sorted_dist,
            "boltzmann_entropy": str(entropy)[:40],
            "partition_function_Z": str(Z)[:40],
        }

    def energy_landscape(self) -> Dict:
        """Map the energy landscape of the token lattice."""
        energies = []
        for name, token in self.lattice.tokens.items():
            try:
                v = D(token.value) if isinstance(token.value, str) else token.value
                if v > D('0'):
                    e = decimal_ln(v)
                    energies.append({"name": name, "energy": float(e), "tier": token.tier})
            except Exception:
                pass
        energies.sort(key=lambda x: x["energy"])
        # Compute landscape statistics
        e_vals = [x["energy"] for x in energies]
        if not e_vals:
            return {"error": "Empty lattice"}
        mean_e = sum(e_vals) / len(e_vals)
        var_e = sum((e - mean_e)**2 for e in e_vals) / len(e_vals)
        return {
            "total_states": len(energies),
            "energy_range": [min(e_vals), max(e_vals)],
            "mean_energy": round(mean_e, 10),
            "energy_variance": round(var_e, 10),
            "lowest_5": energies[:5],
            "highest_5": energies[-5:],
            "tier_distribution": self._tier_energy_stats(energies),
        }

    def _tier_energy_stats(self, energies: List[Dict]) -> Dict:
        """Compute energy statistics per tier."""
        tier_data: Dict[str, List[float]] = {}
        for e in energies:
            t = e["tier"]
            if t not in tier_data:
                tier_data[t] = []
            tier_data[t].append(e["energy"])
        stats = {}
        for tier, vals in tier_data.items():
            stats[tier] = {
                "count": len(vals),
                "mean": round(sum(vals)/len(vals), 6),
                "min": round(min(vals), 6),
                "max": round(max(vals), 6),
            }
        return stats

    def full_analysis(self) -> Dict:
        """Complete statistical mechanics analysis."""
        return {
            "partition_function": self.partition_function(),
            "boltzmann_beta_1": self.boltzmann_distribution(1.0),
            "energy_landscape": self.energy_landscape(),
            "engine": "StatisticalMechanicsEngine",
        }
