"""L104 Numerical Engine — Research Engines.

QuantumNumericalResearchEngine, StochasticNumericalResearchLab, NumericalTestGenerator.

PART V RESEARCH — l104_runtime_infrastructure_research.py:
  F61: Lattice capacity 22T governs discovery space
  F63: Tier envelope progression constrains research drift analysis
  F64: 501 GOD_CODE spectrum tokens seed invention synthesis
  F88: Entanglement eigenvalues φ+φ⁻¹=√5 discoverable in harmonic analysis
"""
from __future__ import annotations

import json
import math
import time
import random
import statistics
from datetime import datetime, timezone
from decimal import InvalidOperation
from typing import Any, Dict, List, TYPE_CHECKING

from .precision import D, fmt100
from .constants import (
    PHI_HP, PHI_GROWTH_HP, GOD_CODE_HP, god_code_hp, WORKSPACE_ROOT,
)

if TYPE_CHECKING:
    from .lattice import TokenLatticeEngine
    from .verification import PrecisionVerificationEngine


class QuantumNumericalResearchEngine:
    """Research engine for the quantum numerical lattice."""

    def __init__(self, lattice: 'TokenLatticeEngine'):
        """Initialize QuantumNumericalResearchEngine."""
        self.lattice = lattice
        self.memory: Dict = {}
        self.memory_file = WORKSPACE_ROOT / ".l104_numerical_research_memory.json"
        self._load_memory()

    def _load_memory(self):
        """Load persistent memory from disk."""
        if self.memory_file.exists():
            try:
                self.memory = json.loads(self.memory_file.read_text())
            except Exception:
                self.memory = {}

    def _save_memory(self):
        """Save persistent memory to disk."""
        try:
            self.memory["last_updated"] = datetime.now(timezone.utc).isoformat()
            self.memory_file.write_text(json.dumps(self.memory, indent=2, default=str))
        except Exception:
            pass

    def full_research(self) -> Dict:
        """Run all research modules."""
        start = time.time()

        stability = self._stability_analysis()
        harmonics = self._harmonic_analysis()
        entropy = self._entropy_landscape()
        convergence = self._convergence_analysis()
        inventions = self._invention_synthesis()

        elapsed = time.time() - start

        result = {
            "stability": stability,
            "harmonics": harmonics,
            "entropy_landscape": entropy,
            "convergence": convergence,
            "inventions": inventions,
            "elapsed_sec": round(elapsed, 3),
            "research_health": self._compute_research_health(
                stability, harmonics, entropy, convergence
            ),
        }

        # Update memory
        self.memory["last_research"] = result
        self.memory["research_count"] = self.memory.get("research_count", 0) + 1
        self._save_memory()

        return result

    def _stability_analysis(self) -> Dict:
        """Module 1: Detect tokens with excessive drift or boundary violations."""
        unstable = []
        drift_magnitudes = []

        for tid, token in self.lattice.tokens.items():
            drift = abs(D(token.drift_velocity)) if token.drift_velocity else D(0)
            drift_magnitudes.append(float(drift))

            val = D(token.value)
            lo = D(token.min_bound)
            hi = D(token.max_bound)
            range_width = hi - lo

            if range_width > 0:
                relative_drift = float(drift / range_width) if range_width != 0 else 0
                if relative_drift > 0.1:
                    unstable.append({
                        "token_id": tid,
                        "name": token.name,
                        "relative_drift": relative_drift,
                        "origin": token.origin,
                    })

        mean_drift = statistics.mean(drift_magnitudes) if drift_magnitudes else 0
        return {
            "total_tokens": len(self.lattice.tokens),
            "unstable_count": len(unstable),
            "unstable_tokens": unstable[:10],
            "mean_drift": mean_drift,
            "stability_score": 1.0 - min(1.0, len(unstable) / max(len(self.lattice.tokens), 1)),
        }

    def _harmonic_analysis(self) -> Dict:
        """Module 2: Find φ-resonant token clusters."""
        clusters = []
        sacred_tokens = [
            t for t in self.lattice.tokens.values() if t.origin == "sacred"
        ]

        for token in sacred_tokens:
            val = D(token.value)
            if val <= 0:
                continue

            resonant_peers = []
            for other in self.lattice.tokens.values():
                if other.token_id == token.token_id:
                    continue
                other_val = D(other.value)
                if other_val <= 0:
                    continue

                ratio = other_val / val
                for n in range(-5, 6):
                    if n == 0:
                        continue
                    phi_n = PHI_GROWTH_HP ** abs(n) if n > 0 else PHI_HP ** abs(n)
                    if abs(ratio - phi_n) / phi_n < D('0.01'):
                        resonant_peers.append({
                            "peer_id": other.token_id,
                            "peer_name": other.name,
                            "phi_power": n,
                            "ratio_error": float(abs(ratio - phi_n) / phi_n),
                        })
                        break

            if resonant_peers:
                clusters.append({
                    "anchor": token.token_id,
                    "anchor_name": token.name,
                    "resonant_peers": resonant_peers[:5],
                })

        return {
            "harmonic_clusters": len(clusters),
            "clusters": clusters[:10],
            "phi_resonance_score": len(clusters) / max(len(sacred_tokens), 1),
        }

    def _entropy_landscape(self) -> Dict:
        """Module 3: Map the information landscape of the lattice."""
        values = []
        for t in self.lattice.tokens.values():
            try:
                v = float(D(t.value))
                if math.isfinite(v) and abs(v) > 0:
                    values.append(abs(v))
            except (ValueError, OverflowError, InvalidOperation):
                continue

        if len(values) < 2:
            return {"entropy": 0, "complexity": 0}

        log_values = [math.log(v) for v in values if v > 0]
        mean_log = statistics.mean(log_values) if log_values else 0
        std_log = statistics.stdev(log_values) if len(log_values) > 1 else 0

        n_bins = min(50, len(values) // 2 + 1)
        if n_bins < 2:
            return {"entropy": 0, "complexity": 0}

        min_v = min(log_values)
        max_v = max(log_values)
        bin_width = (max_v - min_v) / n_bins if max_v > min_v else 1
        bins = [0] * n_bins
        for lv in log_values:
            idx = min(int((lv - min_v) / bin_width), n_bins - 1)
            bins[idx] += 1

        total = sum(bins)
        entropy = 0.0
        for count in bins:
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        return {
            "entropy_bits": round(entropy, 6),
            "log_mean": round(mean_log, 6),
            "log_std": round(std_log, 6),
            "complexity": round(entropy * std_log, 6),
            "n_tokens": len(values),
        }

    def _convergence_analysis(self) -> Dict:
        """Module 4: Track lattice evolution toward equilibrium."""
        prev = self.memory.get("last_research", {})
        prev_coherence = prev.get("stability", {}).get("stability_score", 0)
        prev_entropy = prev.get("entropy_landscape", {}).get("entropy_bits", 0)

        curr_coherence = float(self.lattice.lattice_coherence)
        curr_entropy = float(self.lattice.lattice_entropy)

        coherence_delta = curr_coherence - prev_coherence
        entropy_delta = curr_entropy - prev_entropy

        converging = coherence_delta >= 0 and abs(entropy_delta) < 1.0

        return {
            "current_coherence": curr_coherence,
            "previous_coherence": prev_coherence,
            "coherence_delta": coherence_delta,
            "current_entropy": curr_entropy,
            "previous_entropy": prev_entropy,
            "entropy_delta": entropy_delta,
            "converging": converging,
            "convergence_score": min(1.0, max(0.0, curr_coherence + (0.1 if converging else -0.1))),
        }

    def _invention_synthesis(self) -> Dict:
        """Module 5: Discover new mathematical tokens from lattice patterns."""
        inventions = []

        existing_gx = set()
        for tid in self.lattice.tokens:
            if tid.startswith("GC_X"):
                try:
                    x = int(tid.replace("GC_X", ""))
                    existing_gx.add(x)
                except ValueError:
                    pass

        for x in range(-50, 51):
            half_x = D(x) + D('0.5')
            gx_half = god_code_hp(half_x)
            token_id = f"GC_XHALF_{x}"
            if token_id not in self.lattice.tokens:
                inventions.append({
                    "type": "half_integer_harmonic",
                    "X": float(half_x),
                    "value_preview": str(gx_half)[:40],
                    "token_id": token_id,
                })

        sacred = [(tid, D(t.value)) for tid, t in self.lattice.tokens.items()
                  if t.origin == "sacred" and D(t.value) > 0]
        for i, (tid_a, val_a) in enumerate(sacred):
            for tid_b, val_b in sacred[i + 1:]:
                ratio = val_a / val_b if val_b != 0 else D(0)
                test = abs(ratio * PHI_GROWTH_HP)
                if test > 0:
                    nearest_int = round(float(test))
                    if nearest_int > 0:
                        error = abs(test - D(nearest_int)) / D(nearest_int)
                        if error < D('0.01'):
                            inventions.append({
                                "type": "phi_bridge",
                                "token_a": tid_a,
                                "token_b": tid_b,
                                "ratio_times_phi": float(test),
                                "nearest_integer": nearest_int,
                                "error": float(error),
                            })

        return {
            "total_inventions": len(inventions),
            "half_integer_harmonics": sum(1 for i in inventions if i["type"] == "half_integer_harmonic"),
            "phi_bridges": sum(1 for i in inventions if i["type"] == "phi_bridge"),
            "inventions": inventions[:20],
        }

    def _compute_research_health(self, stability, harmonics, entropy, convergence) -> float:
        """Unified research health score: 0–1."""
        s = stability.get("stability_score", 0)
        h = harmonics.get("phi_resonance_score", 0)
        e_bits = entropy.get("entropy_bits", 0)
        e_score = min(1.0, e_bits / 6.0) if e_bits > 0 else 0
        c = convergence.get("convergence_score", 0)

        return (s * 0.3 + h * 0.2 + e_score * 0.2 + c * 0.3)


class StochasticNumericalResearchLab:
    """v3.0 — Random R&D on numerical tokens."""

    PHI = 1.618033988749895
    GOD_CODE = 286 ** (1.0 / PHI) * (2 ** (416 / 104))
    MAX_EXPERIMENTS = 200

    def __init__(self, lattice=None):
        self.lattice = lattice
        self.experiments: List[Dict] = []
        self.breakthroughs: List[Dict] = []
        self.total_runs = 0

    def run_stochastic_cycle(self, n: int = 20) -> Dict:
        """Run n random experiments on token pairs."""
        self.total_runs += 1
        cycle_experiments = []

        tokens = list(self.lattice.tokens.values()) if self.lattice else []
        if len(tokens) < 2:
            return {"experiments": 0, "breakthroughs": 0, "message": "insufficient tokens"}

        for _ in range(min(n, self.MAX_EXPERIMENTS)):
            t_a, t_b = random.sample(tokens, 2)
            try:
                val_a = float(t_a.value[:30])
                val_b = float(t_b.value[:30])
                if val_b == 0:
                    continue
            except (ValueError, TypeError):
                continue

            ratio = val_a / val_b
            phi_dist = abs(ratio - self.PHI)
            gc_dist = abs(ratio * 1000 - self.GOD_CODE)

            experiment = {
                "token_a": t_a.name,
                "token_b": t_b.name,
                "ratio": round(ratio, 8),
                "phi_distance": round(phi_dist, 8),
                "godcode_distance": round(gc_dist, 4),
                "sacred_resonance": phi_dist < 0.01 or gc_dist < 1.0,
            }
            cycle_experiments.append(experiment)

            if experiment["sacred_resonance"]:
                self.breakthroughs.append(experiment)

        self.experiments.extend(cycle_experiments[-50:])
        self.experiments = self.experiments[-self.MAX_EXPERIMENTS:]

        return {
            "experiments_run": len(cycle_experiments),
            "breakthroughs_found": sum(1 for e in cycle_experiments if e["sacred_resonance"]),
            "total_breakthroughs": len(self.breakthroughs),
            "total_runs": self.total_runs,
        }

    def status(self) -> Dict:
        return {
            "class": "StochasticNumericalResearchLab",
            "total_runs": self.total_runs,
            "experiments_stored": len(self.experiments),
            "breakthroughs": len(self.breakthroughs),
        }


class NumericalTestGenerator:
    """v3.0 — Automated test suite for numerical token verification."""

    PHI = 1.618033988749895
    GOD_CODE = 286 ** (1.0 / PHI) * (2 ** (416 / 104))

    def __init__(self, lattice=None, verifier=None):
        self.lattice = lattice
        self.verifier = verifier
        self.test_results: List[Dict] = []
        self.total_runs = 0

    def run_test_suite(self) -> Dict:
        """Run full automated test suite on the lattice."""
        self.total_runs += 1
        tests = []

        sacred_ids = ["PHI", "GOD_CODE", "PI", "E", "EULER_GAMMA", "TAU"]
        sacred_present = sum(1 for s in sacred_ids if s in (self.lattice.tokens if self.lattice else {}))
        tests.append({"test": "sacred_token_presence", "expected": len(sacred_ids),
                       "actual": sacred_present, "passed": sacred_present == len(sacred_ids)})

        if self.lattice:
            gc0 = self.lattice.tokens.get("GC_X0")
            gc1 = self.lattice.tokens.get("GC_X1")
            if gc0 and gc1:
                try:
                    v0 = float(gc0.value[:30])
                    v1 = float(gc1.value[:30])
                    inv0 = v0 * (2 ** (0 / 104))
                    inv1 = v1 * (2 ** (1 / 104))
                    err = abs(inv0 - inv1) / max(abs(inv0), 1e-30)
                    tests.append({"test": "conservation_law", "error": err, "passed": err < 1e-10})
                except Exception:
                    tests.append({"test": "conservation_law", "passed": False, "error": "computation_failed"})
            else:
                tests.append({"test": "conservation_law", "passed": False, "error": "tokens_missing"})

        if self.lattice:
            coh = float(self.lattice.lattice_coherence)
            tests.append({"test": "coherence_range", "value": coh,
                           "passed": 0 <= coh <= 2.0})

        phi_sq = self.PHI ** 2
        phi_plus_1 = self.PHI + 1
        phi_err = abs(phi_sq - phi_plus_1)
        tests.append({"test": "phi_identity", "error": phi_err, "passed": phi_err < 1e-12})

        if self.lattice:
            ent = float(self.lattice.lattice_entropy)
            tests.append({"test": "entropy_non_negative", "value": ent, "passed": ent >= 0})

        passed = sum(1 for t in tests if t.get("passed"))
        total = len(tests)
        self.test_results = tests

        return {
            "tests_run": total,
            "tests_passed": passed,
            "tests_failed": total - passed,
            "pass_rate": round(passed / max(total, 1), 4),
            "total_runs": self.total_runs,
            "details": tests,
        }

    def status(self) -> Dict:
        passed = sum(1 for t in self.test_results if t.get("passed"))
        return {
            "class": "NumericalTestGenerator",
            "total_runs": self.total_runs,
            "last_passed": passed,
            "last_total": len(self.test_results),
            "pass_rate": round(passed / max(len(self.test_results), 1), 4),
        }
