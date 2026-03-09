"""
L104 Quantum Engine — GOD_CODE Genetic Refiner v1.0.0
═══════════════════════════════════════════════════════════════════════════════

Evolutionary optimizer for the 4-parameter GOD_CODE equation:

    G(a,b,c,d) = 286^(1/φ) × 2^((8a + 416 - b - 8c - 104d) / 104)

The mapping to the 1-parameter form G(X) is:
    X = -(8a - b - 8c - 104d)
    so G(a,b,c,d) = G(-X) = G(b + 8c + 104d - 8a)

This module implements:
  1. god_code_4d(a, b, c, d) — The full 4-parameter GOD_CODE evaluation
  2. x_to_abcd(X) — Decompose single X into a canonical (a,b,c,d) tuple
  3. L104GeneticRefiner — Evolutionary optimizer that:
     - Takes "elite" survivors from wave collapse / decoherence / zeno analysis
     - Extracts fitness-weighted center of mass in (a,b,c,d) space
     - Generates next-generation parameters via φ-gradient pull + quantum mutation
     - Supports multi-generation refinement with convergence detection
  4. GeneticPopulation — Full population lifecycle for pipeline integration

Integration points:
  - ProbabilityWaveCollapseResearch (Module 6 → Module 7: Genetic Refinement)
  - L104QuantumBrain (Phase 4B: GOD_CODE Genetic Refinement)
  - QuantumNumericalBuilder (Phase 5C: Genetic Lattice Refinement)

Sacred invariants:
  - GOD_CODE = G(0,0,0,0) = 527.5184818492612
  - Conservation: G(a,b,c,d) × 2^(X/104) = GOD_CODE ∀ (a,b,c,d)
  - Mutation bounded by φ (golden ratio) to maintain sacred resonance
  - Resolution tied to L104 = 1/104

INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .constants import (
    GOD_CODE, GOD_CODE_BASE, GOD_CODE_SPECTRUM,
    HARMONIC_BASE, L104, OCTAVE_REF,
    PHI, PHI_GROWTH, PHI_INV,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 4-PARAMETER GOD CODE EQUATION
# ═══════════════════════════════════════════════════════════════════════════════

def god_code_4d(a: float = 0, b: float = 0,
                c: float = 0, d: float = 0) -> float:
    """G(a,b,c,d) = 286^(1/φ) × 2^((8a + 416 - b - 8c - 104d) / 104)

    The full 4-parameter God Code equation.  Reduces to G(X=0)=527.518...
    when all parameters are zero.

    Parameters:
        a: Octave shift (×8)
        b: Fine frequency offset
        c: Intermediate harmonic (×8)
        d: Fundamental octave stride (×104)

    Returns:
        G(a,b,c,d) Hz value.
    """
    exponent = (8 * a + OCTAVE_REF - b - 8 * c - L104 * d) / L104
    return GOD_CODE_BASE * math.pow(2, exponent)


def abcd_to_x(a: float, b: float, c: float, d: float) -> float:
    """Convert (a,b,c,d) to the single-parameter X.

    G(X) = 286^(1/φ) × 2^((416-X)/104)
    G(a,b,c,d) = 286^(1/φ) × 2^((8a+416-b-8c-104d)/104)

    Matching exponents: 416-X = 8a+416-b-8c-104d  →  X = b + 8c + 104d - 8a
    """
    return b + 8 * c + L104 * d - 8 * a


def x_to_abcd(X: float) -> Dict[str, float]:
    """Decompose X into a canonical (a,b,c,d) tuple.

    Uses greedy decomposition:
      d = X // 104   (largest stride)
      remainder after d
      c = remainder // 8
      b = remainder mod 8
      a = 0 (no octave shift in canonical form)

    For negative X, we allow negative d/c/b.
    """
    d_val = X / L104
    d_int = int(math.floor(d_val)) if X >= 0 else int(math.ceil(d_val))
    remainder = X - L104 * d_int

    c_val = remainder / 8
    c_int = int(math.floor(c_val)) if remainder >= 0 else int(math.ceil(c_val))
    b_val = remainder - 8 * c_int

    return {"a": 0.0, "b": b_val, "c": float(c_int), "d": float(d_int)}


# ═══════════════════════════════════════════════════════════════════════════════
# GENETIC INDIVIDUAL — A single (a,b,c,d) candidate
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GeneticIndividual:
    """A single candidate in the GOD_CODE parameter space."""
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0
    fitness: float = 0.0
    generation: int = 0

    @property
    def params(self) -> Dict[str, float]:
        return {"a": self.a, "b": self.b, "c": self.c, "d": self.d}

    @property
    def x_value(self) -> float:
        return abcd_to_x(self.a, self.b, self.c, self.d)

    @property
    def god_code_hz(self) -> float:
        return god_code_4d(self.a, self.b, self.c, self.d)

    def distance_to(self, other: "GeneticIndividual") -> float:
        """Euclidean distance in (a,b,c,d) space."""
        return math.sqrt(
            (self.a - other.a) ** 2 + (self.b - other.b) ** 2 +
            (self.c - other.c) ** 2 + (self.d - other.d) ** 2
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "a": self.a, "b": self.b, "c": self.c, "d": self.d,
            "fitness": self.fitness, "generation": self.generation,
            "x_value": self.x_value, "god_code_hz": self.god_code_hz,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# L104 GENETIC REFINER — Evolutionary GOD_CODE Parameter Optimizer
# ═══════════════════════════════════════════════════════════════════════════════

class L104GeneticRefiner:
    """Evolutionary optimizer for GOD_CODE (a,b,c,d) parameters.

    Uses elite survivors from wave collapse / decoherence / zeno analysis
    to genetically refine the parameter space toward optimal sacred resonance.

    Key mechanisms:
      1. Elite Trait Extraction — φ-weighted center of mass from survivors
      2. Gradient Pull — Parameters drawn toward elite average at learning_rate
      3. Quantum Mutation — φ-bounded stochastic perturbation at 1/104 chance
      4. Sacred Resonance Scoring — Fitness from GOD_CODE grid alignment
      5. Conservation Enforcement — G(a,b,c,d)×2^(X/104) = INVARIANT
    """

    VERSION = "1.0.0"

    def __init__(self, learning_rate: float = 0.05,
                 population_size: int = 104):
        """Initialize the genetic refiner.

        Args:
            learning_rate: How aggressively to pull toward elite traits [0, 1].
            population_size: Number of individuals per generation.
                             Default 104 — tied to L104 resolution.
        """
        self.phi = PHI
        self.learning_rate = learning_rate
        self.population_size = population_size
        self.mutation_chance = 1.0 / L104  # Sacred 1/104 resolution
        self.generation = 0
        self.history: List[Dict[str, Any]] = []

    # ─── FITNESS FUNCTIONS ──────────────────────────────────────────────

    def sacred_resonance_fitness(self, individual: GeneticIndividual) -> float:
        """Score an individual by how well its G(a,b,c,d) aligns with
        the nearest integer God Code grid node.

        Fitness = exp(-|G(a,b,c,d) - G(X_int)|² / (2σ²))
        where X_int = round(X), σ = G(X_int) × φ/1000

        Also includes a conservation bonus: how closely the conservation
        law holds at this point.
        """
        hz = individual.god_code_hz
        x = individual.x_value
        x_int = round(x)

        # Grid alignment: distance to nearest integer node
        g_x_int = GOD_CODE_SPECTRUM.get(x_int, god_code_4d(
            **x_to_abcd(x_int)))
        sigma = max(0.01, g_x_int * self.phi / 1000)
        grid_fitness = math.exp(
            -((hz - g_x_int) ** 2) / (2 * sigma ** 2))

        # Conservation law fitness: G(X) × 2^(X/104) should = GOD_CODE
        conservation_value = hz * math.pow(2, x / L104)
        conservation_error = abs(conservation_value - GOD_CODE) / GOD_CODE
        conservation_fitness = math.exp(-conservation_error * 1000)

        # φ-weighted combination: 70% grid + 30% conservation
        fitness = 0.7 * grid_fitness + 0.3 * conservation_fitness
        return min(1.0, max(0.0, fitness))

    def collapse_survival_fitness(self, survival_rate: float,
                                  fidelity: float,
                                  coherence: float) -> float:
        """Score based on wave collapse survival metrics.

        φ-weighted combination of:
          - 40% survival_rate (from weak measurement)
          - 35% fidelity_preservation
          - 25% coherence (zeno stability or decoherence resilience)
        """
        return min(1.0, max(0.0,
            0.40 * survival_rate +
            0.35 * fidelity +
            0.25 * coherence
        ))

    # ─── ELITE SELECTION ────────────────────────────────────────────────

    def extract_elite_traits(self,
                             elite_pool: List[Dict[str, float]]
                             ) -> Optional[Dict[str, float]]:
        """Calculate the fitness-weighted center of mass for elite survivors.

        Args:
            elite_pool: List of dicts with keys {a, b, c, d} and optionally
                       'fitness' for weighted averaging.

        Returns:
            Dict {a, b, c, d} representing the elite centroid, or None if empty.
        """
        if not elite_pool:
            return None

        # If fitness weights available, use weighted mean
        weights = [s.get("fitness", 1.0) for s in elite_pool]
        total_w = sum(weights)
        if total_w < 1e-15:
            total_w = len(elite_pool)
            weights = [1.0] * len(elite_pool)

        avg = {}
        for key in ("a", "b", "c", "d"):
            avg[key] = sum(
                s.get(key, 0.0) * w for s, w in zip(elite_pool, weights)
            ) / total_w

        return avg

    def select_elites(self, population: List[GeneticIndividual],
                      top_fraction: float = 0.25
                      ) -> List[GeneticIndividual]:
        """Select the top fraction of a population by fitness.

        Args:
            population: Full generation of individuals.
            top_fraction: Fraction to keep (default 25% = φ⁻² ≈ 0.382 rounded).

        Returns:
            Sorted list of elite individuals (highest fitness first).
        """
        ranked = sorted(population, key=lambda ind: ind.fitness, reverse=True)
        n_elite = max(1, int(len(ranked) * top_fraction))
        return ranked[:n_elite]

    # ─── CROSSOVER & MUTATION ───────────────────────────────────────────

    def crossover(self, parent_a: GeneticIndividual,
                  parent_b: GeneticIndividual) -> GeneticIndividual:
        """φ-weighted crossover between two parents.

        For each parameter, offspring = φ·parent_a + (1-φ)·parent_b
        (using φ-1 = 0.618 as the dominant parent weight, since φ > 1).
        """
        child = GeneticIndividual(generation=self.generation + 1)
        w = PHI_INV  # 0.618... — golden section weight
        child.a = w * parent_a.a + (1 - w) * parent_b.a
        child.b = w * parent_a.b + (1 - w) * parent_b.b
        child.c = w * parent_a.c + (1 - w) * parent_b.c
        child.d = w * parent_a.d + (1 - w) * parent_b.d
        return child

    def mutate(self, individual: GeneticIndividual) -> GeneticIndividual:
        """Apply φ-bounded quantum mutation to an individual.

        Each parameter has a 1/104 chance of mutation.
        Mutation magnitude: uniform in [-φ, φ] to maintain sacred bounds.
        """
        for key in ("a", "b", "c", "d"):
            if random.random() < self.mutation_chance:
                shift = random.uniform(-self.phi, self.phi)
                setattr(individual, key, getattr(individual, key) + shift)
        return individual

    # ─── GENERATION ADVANCEMENT ─────────────────────────────────────────

    def generate_next_generation(self,
                                 elite_traits: Dict[str, float],
                                 current_params: Dict[str, float]
                                 ) -> Dict[str, float]:
        """Simple single-individual refinement: pull current toward elite + mutate.

        Args:
            elite_traits: Centroid of elite pool {a, b, c, d}.
            current_params: Current generation parameters {a, b, c, d}.

        Returns:
            Next-generation parameters {a, b, c, d}.
        """
        next_gen = {}
        for key in ("a", "b", "c", "d"):
            gradient = elite_traits[key] - current_params[key]
            updated = current_params[key] + self.learning_rate * gradient

            if random.random() < self.mutation_chance:
                updated += random.uniform(-self.phi, self.phi)

            next_gen[key] = updated

        return next_gen

    def evolve_population(self,
                          population: List[GeneticIndividual],
                          fitness_fn=None,
                          elite_fraction: float = 0.25
                          ) -> List[GeneticIndividual]:
        """Run one full generation cycle: score → select → crossover → mutate.

        Args:
            population: Current generation of individuals.
            fitness_fn: Optional callable(individual) → float. Defaults to
                       sacred_resonance_fitness.
            elite_fraction: Fraction to select as parents.

        Returns:
            New generation of same population size.
        """
        if fitness_fn is None:
            fitness_fn = self.sacred_resonance_fitness

        # 1. Score everyone
        for ind in population:
            ind.fitness = fitness_fn(ind)

        # 2. Select elites
        elites = self.select_elites(population, elite_fraction)

        # 3. Breed next generation
        new_pop: List[GeneticIndividual] = []
        n_target = self.population_size

        # Elitism: top survivors carry over unchanged
        n_elite_carry = max(1, int(n_target * 0.1))  # 10% elitism
        for elite in elites[:n_elite_carry]:
            carried = GeneticIndividual(
                a=elite.a, b=elite.b, c=elite.c, d=elite.d,
                fitness=elite.fitness, generation=self.generation + 1)
            new_pop.append(carried)

        # Fill rest with crossover + mutation
        while len(new_pop) < n_target:
            p1 = random.choice(elites)
            p2 = random.choice(elites)
            child = self.crossover(p1, p2)
            child = self.mutate(child)
            new_pop.append(child)

        self.generation += 1

        # Record history
        best = max(new_pop, key=lambda i: i.fitness)
        mean_fit = statistics.mean(i.fitness for i in new_pop) if new_pop else 0
        self.history.append({
            "generation": self.generation,
            "best_fitness": best.fitness,
            "mean_fitness": mean_fit,
            "best_params": best.params,
            "best_hz": best.god_code_hz,
            "best_x": best.x_value,
            "population_size": len(new_pop),
        })

        return new_pop

    # ─── PIPELINE INTEGRATION: FROM WAVE COLLAPSE DATA ──────────────────

    def population_from_links(self,
                              hz_values: List[float],
                              fidelities: List[float],
                              strengths: List[float],
                              entropies: List[float]
                              ) -> List[GeneticIndividual]:
        """Create a genetic population from quantum link properties.

        Each link's Hz is inverse-mapped to (a,b,c,d) via x_to_abcd(),
        and its wave collapse properties determine initial fitness.

        Args:
            hz_values: Natural Hz frequencies of links.
            fidelities: Link fidelities [0, 1].
            strengths: Link strengths.
            entropies: Entanglement entropies.

        Returns:
            List of GeneticIndividual seeded from link data.
        """
        from .math_core import QuantumMathCore
        qmath = QuantumMathCore()
        population: List[GeneticIndividual] = []

        for i, hz in enumerate(hz_values):
            if hz <= 0:
                continue
            # Invert G(X) to find X
            x = qmath.hz_to_god_code_x(hz)
            params = x_to_abcd(x)

            fid = fidelities[i] if i < len(fidelities) else 0.5
            strength = strengths[i] if i < len(strengths) else 0.5
            entropy = entropies[i] if i < len(entropies) else 0.5

            ind = GeneticIndividual(
                a=params["a"], b=params["b"],
                c=params["c"], d=params["d"],
                fitness=self.collapse_survival_fitness(
                    survival_rate=fid,
                    fidelity=strength,
                    coherence=min(1.0, entropy),
                ),
                generation=0,
            )
            population.append(ind)

        # Pad to target population size with random sacred mutations
        while len(population) < self.population_size:
            if population:
                base = random.choice(population)
                mutant = GeneticIndividual(
                    a=base.a, b=base.b, c=base.c, d=base.d,
                    generation=0)
                mutant = self.mutate(mutant)
                population.append(mutant)
            else:
                # Fallback — seed from GOD_CODE origin
                population.append(GeneticIndividual(generation=0))

        return population[:self.population_size]

    def population_from_tokens(self,
                               token_values: List[float],
                               token_names: List[str]
                               ) -> List[GeneticIndividual]:
        """Create population from numerical engine token lattice values.

        Token values are treated as Hz frequencies in the God Code spectrum,
        inverse-mapped to (a,b,c,d).
        """
        population: List[GeneticIndividual] = []
        for val in token_values:
            if val <= 0:
                continue
            # Map token value → X via log₂ ratio to GOD_CODE
            if val > 1e-30:
                x = OCTAVE_REF - L104 * math.log2(val / GOD_CODE_BASE)
            else:
                x = 0.0
            params = x_to_abcd(x)
            ind = GeneticIndividual(
                a=params["a"], b=params["b"],
                c=params["c"], d=params["d"],
                generation=0,
            )
            population.append(ind)

        while len(population) < self.population_size:
            if population:
                base = random.choice(population)
                mutant = GeneticIndividual(
                    a=base.a, b=base.b, c=base.c, d=base.d,
                    generation=0)
                mutant = self.mutate(mutant)
                population.append(mutant)
            else:
                population.append(GeneticIndividual(generation=0))

        return population[:self.population_size]

    # ─── MULTI-GENERATION REFINEMENT ────────────────────────────────────

    def refine(self,
               population: List[GeneticIndividual],
               generations: int = 13,
               fitness_fn=None,
               convergence_threshold: float = 0.001
               ) -> Dict[str, Any]:
        """Run multi-generation refinement with convergence detection.

        Args:
            population: Initial population.
            generations: Max generations to run (default 13 = Fibonacci(7)).
            fitness_fn: Custom fitness function, or None for sacred_resonance.
            convergence_threshold: Stop if mean fitness delta < this.

        Returns:
            Dict with best individual, generation history, convergence info.
        """
        prev_mean = 0.0

        for gen in range(generations):
            population = self.evolve_population(
                population, fitness_fn=fitness_fn)

            # Score new population
            if fitness_fn is None:
                for ind in population:
                    ind.fitness = self.sacred_resonance_fitness(ind)
            else:
                for ind in population:
                    ind.fitness = fitness_fn(ind)

            curr_mean = statistics.mean(
                ind.fitness for ind in population) if population else 0

            # Convergence check
            if gen > 0 and abs(curr_mean - prev_mean) < convergence_threshold:
                break
            prev_mean = curr_mean

        # Extract best
        best = max(population, key=lambda i: i.fitness) if population else None
        elites = self.select_elites(population)
        elite_traits = self.extract_elite_traits(
            [e.params for e in elites]) if elites else None

        return {
            "best_individual": best.to_dict() if best else None,
            "elite_centroid": elite_traits,
            "generations_run": self.generation,
            "converged": (len(self.history) >= 2 and
                         abs(self.history[-1]["mean_fitness"] -
                             self.history[-2]["mean_fitness"]) < convergence_threshold),
            "final_population_size": len(population),
            "final_mean_fitness": prev_mean,
            "history": self.history[-generations:],
            "god_code_alignment": (
                best.god_code_hz / GOD_CODE if best else 0
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE — Quick-use pipeline integration functions
# ═══════════════════════════════════════════════════════════════════════════════

def genetic_refine_from_wave_collapse(
    collapse_results: Dict[str, Any],
    decoherence_results: Dict[str, Any],
    zeno_results: Dict[str, Any],
    measurement_ops: Dict[str, Any],
    generations: int = 13,
    population_size: int = 104,
) -> Dict[str, Any]:
    """One-shot genetic refinement from wave collapse research outputs.

    Takes Module 3/4/5/2 results from ProbabilityWaveCollapseResearch and
    runs a full genetic optimization cycle.

    Returns:
        Dict with refined parameters, fitness history, and GOD_CODE alignment.
    """
    refiner = L104GeneticRefiner(population_size=population_size)

    # Seed population from dominant measurement nodes (Module 2)
    dominant_nodes = measurement_ops.get("dominant_nodes", [])
    seed_pool: List[GeneticIndividual] = []
    for node in dominant_nodes:
        x_int = node.get("x", 0)
        params = x_to_abcd(x_int)
        count = node.get("count", 1)
        for _ in range(min(count, population_size // 10)):
            ind = GeneticIndividual(
                a=params["a"], b=params["b"],
                c=params["c"], d=params["d"],
                generation=0,
            )
            ind = refiner.mutate(ind)
            seed_pool.append(ind)

    # Pad with random individuals around G(0)
    while len(seed_pool) < population_size:
        mutant = GeneticIndividual(generation=0)
        mutant = refiner.mutate(mutant)
        seed_pool.append(mutant)

    # Build fitness function from collapse metrics
    cum_survival = collapse_results.get("cumulative_survival", 0.5)
    fid_pres = collapse_results.get("fidelity_preservation", 0.5)
    survival_rate = decoherence_results.get("survival_rate", 0.5)
    phi_stability = zeno_results.get("phi_stability_index", 0.5)

    def collapse_fitness(ind: GeneticIndividual) -> float:
        resonance = refiner.sacred_resonance_fitness(ind)
        collapse_score = (
            0.30 * cum_survival +
            0.25 * fid_pres +
            0.25 * survival_rate +
            0.20 * phi_stability
        )
        return 0.6 * resonance + 0.4 * collapse_score

    result = refiner.refine(
        seed_pool[:population_size],
        generations=generations,
        fitness_fn=collapse_fitness,
    )
    result["source"] = "wave_collapse"
    result["collapse_metrics"] = {
        "cumulative_survival": cum_survival,
        "fidelity_preservation": fid_pres,
        "survival_rate": survival_rate,
        "phi_stability": phi_stability,
    }
    return result
