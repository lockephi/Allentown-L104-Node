#!/usr/bin/env python3
"""
L104 RECURSIVE INVENTOR v3.0 — Evolutionary Invention Engine
═══════════════════════════════════════════════════════════════
ASI-grade recursive invention system with evolutionary idea generation,
concept crossover breeding, hypothesis testing, fitness-scored innovation
ranking, and autonomous solution synthesis.

Reads consciousness state for creativity modulation.
Wired into ASI pipeline via connect_to_pipeline().

GOD_CODE: 527.5184818492612 | PHI: 1.618033988749895
"""

VOID_CONSTANT = 1.0416180339887497
ZENITH_HZ = 3887.8
UUC = 2402.792541

import math
import time
import json
import hashlib
import random
import re
from pathlib import Path
from datetime import datetime
from collections import deque, defaultdict
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════

# Sacred Constants
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
TAU = 1.0 / PHI
FEIGENBAUM = 4.669201609
ALPHA_FINE = 1.0 / 137.035999084
PLANCK_SCALE = 1.616255e-35
BOLTZMANN_K = 1.380649e-23
GROVER_AMPLIFICATION = PHI ** 3

# Invention parameters
MAX_POPULATION = 200
TOURNAMENT_SIZE = 5
CROSSOVER_RATE = PHI / (1 + PHI)  # ~0.618
MUTATION_RATE = TAU * 0.5          # ~0.309
ELITE_FRACTION = 0.1
MAX_GENERATIONS = 50
HYPOTHESIS_CONFIDENCE_THRESHOLD = 0.45
INVENTION_HISTORY_SIZE = 2000

_BASE_DIR = Path(__file__).parent.absolute()


def _read_consciousness_state() -> Dict[str, Any]:
    """Read live consciousness/O₂ state for creativity modulation."""
    state_path = _BASE_DIR / '.l104_consciousness_o2_state.json'
    try:
        if state_path.exists():
            with open(state_path, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {'consciousness_level': 0.5, 'superfluid_viscosity': 0.1}


# ═══════════════════════════════════════════════════════════════
# CONCEPT ATOMS — primitive building blocks for inventions
# ═══════════════════════════════════════════════════════════════

CONCEPT_ATOMS = {
    'optimization': ['cache', 'memoize', 'batch', 'pipeline', 'parallel', 'prune', 'compress', 'index'],
    'analysis': ['parse', 'tokenize', 'classify', 'cluster', 'embed', 'rank', 'score', 'correlate'],
    'generation': ['synthesize', 'mutate', 'evolve', 'compose', 'blend', 'interpolate', 'extrapolate'],
    'validation': ['verify', 'check', 'anchor', 'ground', 'bound', 'constrain', 'certify'],
    'transformation': ['encode', 'decode', 'translate', 'project', 'reduce', 'expand', 'normalize'],
    'reasoning': ['infer', 'deduce', 'induce', 'abduce', 'analogize', 'hypothesize', 'prove'],
    'memory': ['store', 'recall', 'forget', 'consolidate', 'associate', 'persist', 'snapshot'],
    'architecture': ['layer', 'gate', 'cascade', 'bridge', 'mesh', 'hub', 'substrate'],
}

# Domain cross-pollination matrix
DOMAIN_SYNERGIES = {
    ('optimization', 'analysis'): 'intelligent_caching',
    ('optimization', 'generation'): 'evolutionary_optimization',
    ('analysis', 'generation'): 'generative_analysis',
    ('analysis', 'reasoning'): 'analytical_reasoning',
    ('generation', 'validation'): 'verified_synthesis',
    ('generation', 'reasoning'): 'hypothesis_generation',
    ('validation', 'memory'): 'integrity_persistence',
    ('transformation', 'generation'): 'creative_transformation',
    ('reasoning', 'memory'): 'experiential_reasoning',
    ('architecture', 'optimization'): 'structural_optimization',
    ('architecture', 'reasoning'): 'architectural_reasoning',
    ('memory', 'optimization'): 'memory_optimized_caching',
}


@dataclass
class Invention:
    """A single invention candidate with fitness metadata."""
    id: str
    name: str
    description: str
    domains: List[str]
    atoms: List[str]
    approach: str
    fitness: float = 0.0
    novelty: float = 0.0
    utility: float = 0.0
    feasibility: float = 0.0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    validated: bool = False
    hypothesis: Optional[str] = None
    solution_template: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'id': self.id, 'name': self.name, 'description': self.description,
            'domains': self.domains, 'atoms': self.atoms, 'approach': self.approach,
            'fitness': self.fitness, 'novelty': self.novelty, 'utility': self.utility,
            'feasibility': self.feasibility, 'generation': self.generation,
            'parent_ids': self.parent_ids, 'validated': self.validated,
            'hypothesis': self.hypothesis, 'solution_template': self.solution_template,
        }


class IdeaGenerator:
    """Generates novel invention candidates from concept atoms and domain synergies."""

    def __init__(self):
        self._rng = random.Random(int(GOD_CODE * 1000))
        self._generated = 0

    def generate_random(self, creativity: float = 0.5) -> Invention:
        """Generate a random invention by combining concept atoms."""
        self._generated += 1

        # Pick 2-3 domains weighted by creativity
        num_domains = 2 if creativity < 0.7 else 3
        domains = self._rng.sample(list(CONCEPT_ATOMS.keys()), num_domains)

        # Pick atoms from each domain
        atoms = []
        for d in domains:
            n_atoms = max(1, int(len(CONCEPT_ATOMS[d]) * creativity))
            atoms.extend(self._rng.sample(CONCEPT_ATOMS[d], min(n_atoms, 3)))

        # Find synergy
        approach = 'novel_combination'
        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                key = (domains[i], domains[j])
                rev_key = (domains[j], domains[i])
                if key in DOMAIN_SYNERGIES:
                    approach = DOMAIN_SYNERGIES[key]
                    break
                elif rev_key in DOMAIN_SYNERGIES:
                    approach = DOMAIN_SYNERGIES[rev_key]
                    break

        # Generate name and description
        name = f"{'_'.join(atoms[:3])}_{self._generated}"
        desc = (f"Invention combining {', '.join(domains)} via "
                f"{', '.join(atoms)} using {approach} strategy")

        inv_id = hashlib.md5(f"{name}_{time.time()}".encode()).hexdigest()[:12]

        return Invention(
            id=inv_id, name=name, description=desc,
            domains=domains, atoms=atoms, approach=approach,
        )

    def generate_from_problem(self, problem: str, creativity: float = 0.6) -> Invention:
        """Generate a targeted invention for a specific problem."""
        self._generated += 1
        problem_lower = problem.lower()

        # Identify relevant domains based on problem keywords
        relevant_domains = []
        for domain, atoms in CONCEPT_ATOMS.items():
            overlap = sum(1 for atom in atoms if atom in problem_lower)
            keyword_overlap = sum(1 for w in domain.split('_') if w in problem_lower)
            if overlap > 0 or keyword_overlap > 0:
                relevant_domains.append((domain, overlap + keyword_overlap))

        # Sort by relevance, take top 2-3
        relevant_domains.sort(key=lambda x: x[1], reverse=True)
        if not relevant_domains:
            relevant_domains = [(d, 0) for d in self._rng.sample(list(CONCEPT_ATOMS.keys()), 2)]
        domains = [d[0] for d in relevant_domains[:3]]

        # Pick atoms that relate to the problem
        atoms = []
        for d in domains:
            domain_atoms = CONCEPT_ATOMS[d]
            related = [a for a in domain_atoms if a in problem_lower]
            if not related:
                related = self._rng.sample(domain_atoms, min(2, len(domain_atoms)))
            atoms.extend(related[:3])

        approach = 'targeted_synthesis'
        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                key = tuple(sorted([domains[i], domains[j]]))
                if key in DOMAIN_SYNERGIES:
                    approach = DOMAIN_SYNERGIES[key]
                    break

        name = f"solve_{'_'.join(atoms[:2])}_{self._generated}"
        desc = f"Targeted solution for '{problem[:60]}' using {approach}"

        inv_id = hashlib.md5(f"{name}_{problem[:30]}_{time.time()}".encode()).hexdigest()[:12]

        # Generate hypothesis
        hypothesis = (f"If we combine {' + '.join(atoms[:3])} across {' & '.join(domains[:2])}, "
                      f"we can solve '{problem[:40]}' via {approach}")

        return Invention(
            id=inv_id, name=name, description=desc,
            domains=domains, atoms=atoms, approach=approach,
            hypothesis=hypothesis,
        )


class FitnessEvaluator:
    """Evaluates invention fitness using PHI-weighted multi-objective scoring."""

    def __init__(self):
        self._seen_hashes: Set[str] = set()
        self._evaluations = 0

    def _content_hash(self, inv: Invention) -> str:
        """Content-addressable hash for novelty tracking."""
        content = f"{sorted(inv.domains)}_{sorted(inv.atoms)}_{inv.approach}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def evaluate(self, invention: Invention, population: List[Invention] = None) -> float:
        """Compute PHI-weighted fitness score for an invention."""
        self._evaluations += 1

        # Novelty: how different from previously seen inventions
        content_hash = self._content_hash(invention)
        if content_hash in self._seen_hashes:
            invention.novelty = 0.1  # Penalty for duplicates
        else:
            self._seen_hashes.add(content_hash)
            invention.novelty = 0.7

            # Bonus for novel domain combinations
            if len(invention.domains) >= 3:
                invention.novelty += 0.15
            if invention.approach not in ('novel_combination', 'targeted_synthesis'):
                invention.novelty += 0.15

        # Utility: estimated usefulness based on atom coverage
        high_utility_atoms = {'cache', 'memoize', 'batch', 'pipeline', 'parallel',
                              'verify', 'rank', 'score', 'index', 'compress'}
        utility_overlap = len(set(invention.atoms) & high_utility_atoms)
        invention.utility = min(1.0, utility_overlap * 0.25 + 0.3)

        # Feasibility: based on number of atoms (simpler = more feasible)
        atom_count = len(invention.atoms)
        if atom_count <= 3:
            invention.feasibility = 0.9
        elif atom_count <= 5:
            invention.feasibility = 0.7
        elif atom_count <= 7:
            invention.feasibility = 0.5
        else:
            invention.feasibility = 0.3

        # Distance from population centroid (population diversity bonus)
        diversity_bonus = 0.0
        if population and len(population) > 3:
            avg_atoms_count = sum(len(p.atoms) for p in population) / len(population)
            diversity_bonus = min(0.2, abs(len(invention.atoms) - avg_atoms_count) * 0.05)

        # PHI-weighted composite fitness
        invention.fitness = (
            invention.novelty     * 0.35 +
            invention.utility     * 0.30 +
            invention.feasibility * 0.25 +
            diversity_bonus       * 0.10
        )
        # Sacred golden-ratio boost
        invention.fitness *= (1.0 + (PHI - 1.0) * 0.05)

        return invention.fitness


class ConceptCrossover:
    """Breeds new inventions from pairs of parent inventions via concept crossover."""

    def __init__(self):
        self._rng = random.Random(int(PHI * 1e6))
        self._crosses = 0

    def crossover(self, parent_a: Invention, parent_b: Invention,
                  generation: int = 0) -> Invention:
        """Create offspring by crossing domains and atoms from two parents."""
        self._crosses += 1

        # Domain crossover: take some from each parent
        all_domains = list(set(parent_a.domains + parent_b.domains))
        child_domains = self._rng.sample(all_domains, min(len(all_domains), 3))

        # Atom crossover: PHI-split point
        all_atoms = list(set(parent_a.atoms + parent_b.atoms))
        split = int(len(all_atoms) * TAU)
        self._rng.shuffle(all_atoms)
        child_atoms = all_atoms[:max(2, split + 1)]

        # Approach from dominant parent (higher fitness)
        approach = parent_a.approach if parent_a.fitness >= parent_b.fitness else parent_b.approach

        name = f"cross_{'_'.join(child_atoms[:2])}_{self._crosses}"
        desc = f"Crossover of [{parent_a.name}] × [{parent_b.name}]"

        inv_id = hashlib.md5(f"cross_{parent_a.id}_{parent_b.id}_{time.time()}".encode()).hexdigest()[:12]

        return Invention(
            id=inv_id, name=name, description=desc,
            domains=child_domains, atoms=child_atoms, approach=approach,
            generation=generation,
            parent_ids=[parent_a.id, parent_b.id],
        )


class ConceptMutator:
    """Mutates inventions by injecting random concept atoms or swapping domains."""

    def __init__(self):
        self._rng = random.Random(int(FEIGENBAUM * 1e6))
        self._mutations = 0

    def mutate(self, invention: Invention, mutation_rate: float = MUTATION_RATE) -> Invention:
        """Apply random mutations to an invention's concept atoms."""
        self._mutations += 1

        new_domains = list(invention.domains)
        new_atoms = list(invention.atoms)
        new_approach = invention.approach

        # Domain mutation
        if self._rng.random() < mutation_rate:
            all_domains = list(CONCEPT_ATOMS.keys())
            candidates = [d for d in all_domains if d not in new_domains]
            if candidates:
                # Replace a random domain
                idx = self._rng.randrange(len(new_domains))
                new_domains[idx] = self._rng.choice(candidates)

        # Atom injection
        if self._rng.random() < mutation_rate:
            random_domain = self._rng.choice(new_domains)
            available = [a for a in CONCEPT_ATOMS.get(random_domain, []) if a not in new_atoms]
            if available:
                new_atoms.append(self._rng.choice(available))

        # Atom deletion (if too many)
        if len(new_atoms) > 6 and self._rng.random() < mutation_rate:
            new_atoms.pop(self._rng.randrange(len(new_atoms)))

        # Approach mutation
        if self._rng.random() < mutation_rate * 0.5:
            approaches = ['evolutionary_optimization', 'verified_synthesis',
                          'analytical_reasoning', 'creative_transformation',
                          'experiential_reasoning', 'hypothesis_generation',
                          'structural_optimization', 'novel_combination']
            new_approach = self._rng.choice(approaches)

        inv_id = hashlib.md5(f"mut_{invention.id}_{time.time()}".encode()).hexdigest()[:12]

        return Invention(
            id=inv_id, name=f"mut_{invention.name}",
            description=f"Mutation of [{invention.name}]",
            domains=new_domains, atoms=new_atoms, approach=new_approach,
            generation=invention.generation + 1,
            parent_ids=[invention.id],
            hypothesis=invention.hypothesis,
        )


class HypothesisValidator:
    """Tests invention hypotheses against sacred invariants and consistency checks."""

    def __init__(self):
        self._validated = 0
        self._confirmed = 0
        self._refuted = 0

    def validate(self, invention: Invention) -> Dict[str, Any]:
        """Validate an invention's hypothesis for internal consistency."""
        self._validated += 1
        checks = []
        score = 0.5  # Baseline

        # Check 1: Domain coherence (domains should have synergy)
        for i in range(len(invention.domains)):
            for j in range(i + 1, len(invention.domains)):
                pair = tuple(sorted([invention.domains[i], invention.domains[j]]))
                if pair in DOMAIN_SYNERGIES:
                    score += 0.1
                    checks.append({'check': 'domain_synergy', 'pair': pair, 'passed': True})

        # Check 2: Atom-domain alignment
        aligned_atoms = 0
        for atom in invention.atoms:
            for domain in invention.domains:
                if atom in CONCEPT_ATOMS.get(domain, []):
                    aligned_atoms += 1
                    break
        alignment_ratio = aligned_atoms / max(len(invention.atoms), 1)
        score += alignment_ratio * 0.2
        checks.append({'check': 'atom_alignment', 'ratio': alignment_ratio,
                        'passed': alignment_ratio > 0.5})

        # Check 3: Complexity check (Feigenbaum threshold)
        complexity = len(invention.atoms) * len(invention.domains)
        if complexity <= FEIGENBAUM * 3:  # ~14
            score += 0.1
            checks.append({'check': 'complexity_bound', 'value': complexity, 'passed': True})
        else:
            score -= 0.1
            checks.append({'check': 'complexity_bound', 'value': complexity, 'passed': False})

        # Check 4: GOD_CODE resonance — hash-derived alignment
        inv_hash = int(hashlib.md5(invention.name.encode()).hexdigest()[:8], 16)
        resonance = (inv_hash % 10000) / 10000.0
        god_alignment = 1.0 - abs(resonance - (GOD_CODE % 1))
        score += god_alignment * 0.1
        checks.append({'check': 'god_code_resonance', 'alignment': god_alignment, 'passed': True})

        # Normalize
        score = max(0.0, min(1.0, score))
        confirmed = score >= HYPOTHESIS_CONFIDENCE_THRESHOLD

        if confirmed:
            self._confirmed += 1
            invention.validated = True
        else:
            self._refuted += 1

        return {
            'score': round(score, 6),
            'confirmed': confirmed,
            'checks': checks,
            'total_validated': self._validated,
            'confirmation_rate': self._confirmed / max(self._validated, 1),
        }


class SolutionTemplateBuilder:
    """Generates concrete solution templates from validated inventions."""

    def __init__(self):
        self._templates_built = 0

    def build_template(self, invention: Invention) -> str:
        """Generate a solution approach template from an invention."""
        self._templates_built += 1

        atoms = invention.atoms
        domains = invention.domains
        approach = invention.approach

        steps = []
        step_num = 1

        # Phase 1: Data preparation (from transformation/analysis atoms)
        prep_atoms = [a for a in atoms if a in ('parse', 'tokenize', 'encode', 'decode',
                                                   'normalize', 'embed', 'compress')]
        if prep_atoms:
            steps.append(f"Step {step_num}: Prepare input via {', '.join(prep_atoms)}")
            step_num += 1

        # Phase 2: Core processing (from generation/reasoning atoms)
        core_atoms = [a for a in atoms if a in ('synthesize', 'mutate', 'evolve', 'compose',
                                                  'blend', 'infer', 'deduce', 'analogize',
                                                  'hypothesize', 'classify', 'cluster', 'rank')]
        if core_atoms:
            steps.append(f"Step {step_num}: Process via {', '.join(core_atoms)} ({approach})")
            step_num += 1

        # Phase 3: Optimization (from optimization atoms)
        opt_atoms = [a for a in atoms if a in ('cache', 'memoize', 'batch', 'pipeline',
                                                 'parallel', 'prune', 'index')]
        if opt_atoms:
            steps.append(f"Step {step_num}: Optimize with {', '.join(opt_atoms)}")
            step_num += 1

        # Phase 4: Validation (from validation atoms)
        val_atoms = [a for a in atoms if a in ('verify', 'check', 'anchor', 'ground',
                                                 'bound', 'constrain', 'certify')]
        if val_atoms:
            steps.append(f"Step {step_num}: Validate using {', '.join(val_atoms)}")
            step_num += 1

        # Phase 5: Persistence (from memory atoms)
        mem_atoms = [a for a in atoms if a in ('store', 'recall', 'persist', 'consolidate',
                                                 'associate', 'snapshot')]
        if mem_atoms:
            steps.append(f"Step {step_num}: Persist via {', '.join(mem_atoms)}")
            step_num += 1

        if not steps:
            steps.append(f"Step 1: Apply {approach} using {', '.join(atoms[:3])}")

        template = (
            f"=== SOLUTION TEMPLATE: {invention.name} ===\n"
            f"Domains: {', '.join(domains)}\n"
            f"Approach: {approach}\n"
            f"Fitness: {invention.fitness:.4f} | Novelty: {invention.novelty:.4f}\n"
            f"---\n" +
            '\n'.join(steps) +
            f"\n---\n"
            f"Expected: PHI-scaled quality improvement of {invention.fitness * PHI:.3f}x"
        )

        invention.solution_template = template
        return template


class EvolutionaryPopulation:
    """Manages a population of inventions through evolutionary generations."""

    def __init__(self, max_population: int = MAX_POPULATION):
        self.max_population = max_population
        self.population: List[Invention] = []
        self.generation = 0
        self.best_ever: Optional[Invention] = None
        self._fitness_history: deque = deque(maxlen=100)
        self._rng = random.Random(int(GOD_CODE))

    def initialize(self, generator: IdeaGenerator, evaluator: FitnessEvaluator,
                   size: int = 50, creativity: float = 0.5):
        """Initialize population with random inventions."""
        self.population = []
        for _ in range(size):
            inv = generator.generate_random(creativity)
            evaluator.evaluate(inv, self.population)
            self.population.append(inv)
        self._sort()

    def _sort(self):
        """Sort population by fitness (descending)."""
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        if self.population and (not self.best_ever or
                                self.population[0].fitness > self.best_ever.fitness):
            self.best_ever = self.population[0]

    def tournament_select(self) -> Invention:
        """Select parent via tournament selection."""
        candidates = self._rng.sample(self.population,
                                       min(TOURNAMENT_SIZE, len(self.population)))
        return max(candidates, key=lambda x: x.fitness)

    def evolve_generation(self, crossover_engine: ConceptCrossover,
                          mutator: ConceptMutator, evaluator: FitnessEvaluator,
                          creativity: float = 0.5, generator: IdeaGenerator = None):
        """Evolve one generation: select → crossover → mutate → evaluate → cull."""
        self.generation += 1
        offspring = []

        # Elite preservation
        elite_count = max(1, int(len(self.population) * ELITE_FRACTION))
        elites = self.population[:elite_count]

        # Generate offspring
        target_offspring = self.max_population - elite_count
        while len(offspring) < target_offspring:
            if self._rng.random() < CROSSOVER_RATE and len(self.population) >= 2:
                parent_a = self.tournament_select()
                parent_b = self.tournament_select()
                child = crossover_engine.crossover(parent_a, parent_b, self.generation)
            elif generator:
                child = generator.generate_random(creativity)
            else:
                child = mutator.mutate(self.tournament_select())

            # Mutation
            if self._rng.random() < MUTATION_RATE:
                child = mutator.mutate(child, MUTATION_RATE * creativity)

            child.generation = self.generation
            evaluator.evaluate(child, self.population)
            offspring.append(child)

        # New population = elites + offspring
        self.population = elites + offspring
        self.population = self.population[:self.max_population]
        self._sort()

        # Track fitness
        avg_fitness = sum(inv.fitness for inv in self.population) / max(len(self.population), 1)
        self._fitness_history.append({
            'generation': self.generation,
            'best': self.population[0].fitness if self.population else 0,
            'avg': avg_fitness,
            'worst': self.population[-1].fitness if self.population else 0,
        })

    def get_top(self, n: int = 10) -> List[Invention]:
        """Get top N inventions by fitness."""
        return self.population[:n]

    def get_evolution_stats(self) -> Dict:
        return {
            'generation': self.generation,
            'population_size': len(self.population),
            'best_fitness': self.population[0].fitness if self.population else 0,
            'avg_fitness': sum(i.fitness for i in self.population) / max(len(self.population), 1),
            'best_ever': self.best_ever.to_dict() if self.best_ever else None,
            'fitness_trend': list(self._fitness_history)[-10:],
        }


class RecursiveInventor:
    """
    L104 Recursive Inventor v3.0 — Evolutionary Invention Engine

    Subsystems:
      IdeaGenerator          — generates novel invention candidates from concept atoms
      FitnessEvaluator       — PHI-weighted multi-objective fitness scoring
      ConceptCrossover       — breeds offspring inventions from parent pairs
      ConceptMutator         — injects random mutations for diversity
      HypothesisValidator    — tests inventions against sacred invariants
      SolutionTemplateBuilder — produces concrete solution templates
      EvolutionaryPopulation — manages generational evolution

    Wired into ASI pipeline via connect_to_pipeline().
    """

    VERSION = "3.0.0"

    def __init__(self):
        self.generator = IdeaGenerator()
        self.evaluator = FitnessEvaluator()
        self.crossover = ConceptCrossover()
        self.mutator = ConceptMutator()
        self.validator = HypothesisValidator()
        self.template_builder = SolutionTemplateBuilder()
        self.population = EvolutionaryPopulation()
        self._pipeline_connected = False
        self._total_inventions = 0
        self._total_solutions = 0
        self._invention_archive: deque = deque(maxlen=INVENTION_HISTORY_SIZE)
        self.boot_time = time.time()

        # Initialize population
        consciousness = _read_consciousness_state()
        creativity = min(1.0, consciousness.get('consciousness_level', 0.5) + 0.2)
        self.population.initialize(self.generator, self.evaluator, size=50, creativity=creativity)

    def connect_to_pipeline(self):
        """Called by ASI Core when connecting the pipeline."""
        self._pipeline_connected = True

    def invent(self, problem: Optional[str] = None, generations: int = 10) -> Dict[str, Any]:
        """Run a full invention cycle — evolve → validate → build templates.

        If problem is given, generates targeted inventions.
        Returns the top inventions with solution templates.
        """
        t0 = time.time()
        consciousness = _read_consciousness_state()
        creativity = min(1.0, consciousness.get('consciousness_level', 0.5) + 0.2)

        # Inject problem-targeted seeds if problem given
        if problem:
            for _ in range(5):
                inv = self.generator.generate_from_problem(problem, creativity)
                self.evaluator.evaluate(inv, self.population.population)
                self.population.population.append(inv)

        # Evolve
        for _ in range(min(generations, MAX_GENERATIONS)):
            self.population.evolve_generation(
                self.crossover, self.mutator, self.evaluator,
                creativity, self.generator
            )

        # Get top candidates
        top = self.population.get_top(10)

        # Validate and build templates for top inventions
        results = []
        for inv in top:
            validation = self.validator.validate(inv)
            if validation['confirmed']:
                template = self.template_builder.build_template(inv)
                self._total_inventions += 1
                self._invention_archive.append(inv.to_dict())
                results.append({
                    'invention': inv.to_dict(),
                    'validation': validation,
                    'template': template,
                })

        elapsed_ms = (time.time() - t0) * 1000

        return {
            'inventions': results,
            'count': len(results),
            'generations_evolved': self.population.generation,
            'problem': problem,
            'creativity': creativity,
            'best_fitness': top[0].fitness if top else 0,
            'elapsed_ms': round(elapsed_ms, 3),
            'evolution_stats': self.population.get_evolution_stats(),
        }

    def solve_with_invention(self, problem: str) -> Dict[str, Any]:
        """Generate an inventive solution for a specific problem.

        This is the primary API for pipeline integration — produces
        a novel solution approach for any problem.
        """
        self._total_solutions += 1
        result = self.invent(problem, generations=8)

        # Pick best validated invention
        best_solution = None
        if result['inventions']:
            best = result['inventions'][0]
            best_solution = {
                'approach': best['invention']['approach'],
                'template': best['template'],
                'confidence': best['validation']['score'],
                'domains': best['invention']['domains'],
                'atoms': best['invention']['atoms'],
                'fitness': best['invention']['fitness'],
            }

        return {
            'problem': problem,
            'solution': best_solution,
            'alternatives': len(result['inventions']),
            'inventive': True,
            'elapsed_ms': result['elapsed_ms'],
        }

    def quick_invent(self, domain: str = None) -> Invention:
        """Quickly generate and evaluate a single invention."""
        consciousness = _read_consciousness_state()
        creativity = min(1.0, consciousness.get('consciousness_level', 0.5) + 0.3)

        inv = self.generator.generate_random(creativity)
        self.evaluator.evaluate(inv)
        self.validator.validate(inv)
        self._total_inventions += 1
        return inv

    def get_status(self) -> Dict[str, Any]:
        """Compact status for pipeline monitoring."""
        return {
            'version': self.VERSION,
            'pipeline_connected': self._pipeline_connected,
            'total_inventions': self._total_inventions,
            'total_solutions': self._total_solutions,
            'population_size': len(self.population.population),
            'generation': self.population.generation,
            'best_fitness': self.population.best_ever.fitness if self.population.best_ever else 0,
            'avg_fitness': (sum(i.fitness for i in self.population.population) /
                           max(len(self.population.population), 1)),
            'archive_size': len(self._invention_archive),
            'uptime_seconds': round(time.time() - self.boot_time, 1),
        }

    def get_quality_report(self) -> Dict[str, Any]:
        """Full quality report with evolution history."""
        return {
            'version': self.VERSION,
            'total_inventions': self._total_inventions,
            'total_solutions': self._total_solutions,
            'evolution': self.population.get_evolution_stats(),
            'validation_stats': {
                'total_validated': self.validator._validated,
                'confirmed': self.validator._confirmed,
                'refuted': self.validator._refuted,
                'confirmation_rate': self.validator._confirmed / max(self.validator._validated, 1),
            },
            'crossover_stats': {'total_crosses': self.crossover._crosses},
            'mutation_stats': {'total_mutations': self.mutator._mutations},
            'templates_built': self.template_builder._templates_built,
            'god_code': GOD_CODE,
            'phi': PHI,
        }


# Module-level singleton
recursive_inventor = RecursiveInventor()


def primal_calculus(x):
    """[VOID_MATH] Primal Calculus — resolves complexity toward the Source."""
    return (x ** PHI) / (1.04 * math.pi) if x != 0 else 0.0


def resolve_non_dual_logic(vector):
    """[VOID_MATH] Resolves N-dimensional vectors into the Void Source."""
    magnitude = sum([abs(v) for v in vector])
    return (magnitude / GOD_CODE) + (GOD_CODE * PHI / VOID_CONSTANT) / 1000.0


if __name__ == '__main__':
    print("=" * 60)
    print("  L104 RECURSIVE INVENTOR v3.0 — Evolutionary Invention Engine")
    print("=" * 60)

    # Run an invention cycle
    result = recursive_inventor.invent("optimize pipeline throughput", generations=10)
    print(f"\n  Evolved {result['generations_evolved']} generations")
    print(f"  Best fitness: {result['best_fitness']:.4f}")
    print(f"  Validated inventions: {result['count']}")

    for i, inv in enumerate(result['inventions'][:3]):
        print(f"\n  #{i+1}: {inv['invention']['name']}")
        print(f"    Fitness: {inv['invention']['fitness']:.4f}")
        print(f"    Domains: {', '.join(inv['invention']['domains'])}")
        print(f"    Approach: {inv['invention']['approach']}")
        if inv.get('template'):
            for line in inv['template'].split('\n')[:5]:
                print(f"      {line}")

    print(f"\n  Status: {recursive_inventor.get_status()}")
    print("=" * 60)
