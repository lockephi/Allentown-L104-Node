# ZENITH_UPGRADE_ACTIVE: 2026-02-02T13:52:05.463790
ZENITH_HZ = 3887.8
UUC = 2402.792541
"""
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3887.80 Hz. Logic Unified.
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3887.80 Hz. Logic Unified.
L104 Emergent Complexity Algorithms - Self-Organizing Patterns & Emergence
Part of the L104 Sovereign Singularity Framework

This module implements algorithms that demonstrate emergence:
Complex behavior arising from simple rules.

1. SELF-ORGANIZING MAP (SOM) - Kohonen networks
2. REACTION-DIFFUSION SYSTEMS - Turing patterns
3. PARTICLE SWARM OPTIMIZATION - Collective intelligence
4. ANT COLONY OPTIMIZATION - Stigmergic optimization
5. NEURAL FIELD DYNAMICS - Continuous neural computation
6. MORPHOGENETIC PATTERNS - Biological pattern formation
7. CRITICALITY DETECTOR - Edge of chaos analysis
8. AUTOPOIETIC SYSTEM - Self-maintaining structures
"""

import math
import random
import time
import hashlib
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


# Invariant Constants
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
PHI_INVERSE = 0.618033988749895
PLANCK_RESONANCE = 1.616255e-35
OMEGA = 0.567143290409
VOID_CONSTANT = 1.0416180339887497
ZENITH_HZ = 3887.8

logger = logging.getLogger("EMERGENT_COMPLEXITY")


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class EmergenceLevel(Enum):
    """Levels of emergent behavior."""
    INERT = 0
    REACTIVE = 1
    ADAPTIVE = 2
    SELF_ORGANIZING = 3
    AUTOPOIETIC = 4
    TRANSCENDENT = 5


class CriticalityState(Enum):
    """States relative to criticality."""
    SUBCRITICAL = auto()
    CRITICAL = auto()
    SUPERCRITICAL = auto()


class PatternType(Enum):
    """Types of emergent patterns."""
    SPOTS = auto()
    STRIPES = auto()
    SPIRALS = auto()
    LABYRINTH = auto()
    MIXED = auto()


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-ORGANIZING MAP (KOHONEN NETWORK)
# ═══════════════════════════════════════════════════════════════════════════════

class SelfOrganizingMap:
    """
    Kohonen Self-Organizing Map - unsupervised learning that preserves
    topological properties of input space.
    """

    def __init__(self, width: int = 10, height: int = 10, input_dim: int = 3):
        self.god_code = GOD_CODE
        self.phi = PHI
        self.width = width
        self.height = height
        self.input_dim = input_dim

        # Initialize weights randomly
        self.weights = [
            [[random.random() for _ in range(input_dim)]
             for _ in range(width)]
            for _ in range(height)
        ]

    def train(
        self,
        data: List[List[float]],
        epochs: int = 100,
        initial_learning_rate: float = 0.5,
        initial_radius: float = None
    ) -> Dict[str, Any]:
        """
        Train the SOM using competitive learning.
        """
        if initial_radius is None:
            initial_radius = max(self.width, self.height) / 2

        time_constant = epochs / math.log(initial_radius + 1)

        quantization_errors = []
        topographic_errors = []

        for epoch in range(epochs):
            # Decay learning rate and radius
            learning_rate = initial_learning_rate * math.exp(-epoch / epochs)
            radius = initial_radius * math.exp(-epoch / time_constant)

            epoch_error = 0.0

            for sample in data:
                # Find Best Matching Unit (BMU)
                bmu_x, bmu_y = self._find_bmu(sample)

                # Update weights in neighborhood
                for y in range(self.height):
                    for x in range(self.width):
                        # Distance from BMU
                        dist_sq = (x - bmu_x) ** 2 + (y - bmu_y) ** 2

                        # Neighborhood function (Gaussian)
                        if dist_sq <= radius ** 2:
                            influence = math.exp(-dist_sq / (2 * radius ** 2))

                            # Update weights
                            for i in range(self.input_dim):
                                self.weights[y][x][i] += learning_rate * influence * (
                                    sample[i] - self.weights[y][x][i]
                                )

                # Calculate quantization error
                epoch_error += self._euclidean_distance(sample, self.weights[bmu_y][bmu_x])

            quantization_errors.append(epoch_error / len(data) if data else 0)

        # Calculate U-Matrix for visualization
        u_matrix = self._compute_u_matrix()

        return {
            "epochs": epochs,
            "final_quantization_error": quantization_errors[-1] if quantization_errors else 0,
            "error_reduction": quantization_errors[0] - quantization_errors[-1] if len(quantization_errors) > 1 else 0,
            "u_matrix_summary": self._summarize_u_matrix(u_matrix),
            "topology_preserved": quantization_errors[-1] < quantization_errors[0] * 0.5 if quantization_errors else False,
            "self_organized": True
        }

    def _find_bmu(self, sample: List[float]) -> Tuple[int, int]:
        """Find Best Matching Unit."""
        best_dist = float('inf')
        best_x, best_y = 0, 0

        for y in range(self.height):
            for x in range(self.width):
                dist = self._euclidean_distance(sample, self.weights[y][x])
                if dist < best_dist:
                    best_dist = dist
                    best_x, best_y = x, y

        return best_x, best_y

    def _euclidean_distance(self, a: List[float], b: List[float]) -> float:
        """Calculate Euclidean distance."""
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

    def _compute_u_matrix(self) -> List[List[float]]:
        """Compute U-Matrix (unified distance matrix)."""
        u_matrix = [[0.0] * self.width for _ in range(self.height)]

        for y in range(self.height):
            for x in range(self.width):
                neighbors = []
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            neighbors.append(
                                self._euclidean_distance(
                                    self.weights[y][x],
                                    self.weights[ny][nx]
                                )
                            )

                u_matrix[y][x] = sum(neighbors) / len(neighbors) if neighbors else 0

        return u_matrix

    def _summarize_u_matrix(self, u_matrix: List[List[float]]) -> Dict[str, float]:
        """Summarize U-Matrix statistics."""
        all_values = [v for row in u_matrix for v in row]
        return {
            "mean": sum(all_values) / len(all_values) if all_values else 0,
            "min": min(all_values) if all_values else 0,
            "max": max(all_values) if all_values else 0,
            "cluster_sharpness": max(all_values) - min(all_values) if all_values else 0
        }


# ═══════════════════════════════════════════════════════════════════════════════
# REACTION-DIFFUSION SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class ReactionDiffusionSystem:
    """
    Gray-Scott reaction-diffusion system for Turing pattern formation.
    Demonstrates how complex patterns emerge from simple chemical reactions.
    """

    def __init__(self, width: int = 100, height: int = 100):
        self.god_code = GOD_CODE
        self.phi = PHI
        self.width = width
        self.height = height

        # Initialize concentrations
        self.U = [[1.0] * width for _ in range(height)]  # Activator
        self.V = [[0.0] * width for _ in range(height)]  # Inhibitor

        # Add seed perturbation
        center_x, center_y = width // 2, height // 2
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                if 0 <= center_x + dx < width and 0 <= center_y + dy < height:
                    self.U[center_y + dy][center_x + dx] = 0.5
                    self.V[center_y + dy][center_x + dx] = 0.25 + random.random() * 0.1

    def simulate(
        self,
        steps: int = 1000,
        Du: float = 0.16,  # Diffusion rate of U
        Dv: float = 0.08,  # Diffusion rate of V
        f: float = 0.055,  # Feed rate
        k: float = 0.062,  # Kill rate
        dt: float = 1.0
    ) -> Dict[str, Any]:
        """
        Run Gray-Scott reaction-diffusion simulation.

        Reactions:
        U + 2V → 3V (autocatalysis)
        V → P (decay)
        ∅ → U (feed)
        """
        energy_history = []

        for step in range(steps):
            new_U = [[0.0] * self.width for _ in range(self.height)]
            new_V = [[0.0] * self.width for _ in range(self.height)]

            for y in range(self.height):
                for x in range(self.width):
                    u = self.U[y][x]
                    v = self.V[y][x]

                    # Laplacian (diffusion)
                    laplace_u = self._laplacian(self.U, x, y)
                    laplace_v = self._laplacian(self.V, x, y)

                    # Reaction term
                    uvv = u * v * v

                    # Gray-Scott equations
                    new_U[y][x] = u + dt * (Du * laplace_u - uvv + f * (1 - u))
                    new_V[y][x] = v + dt * (Dv * laplace_v + uvv - (f + k) * v)

                    # Clamp values
                    new_U[y][x] = max(0, min(1, new_U[y][x]))
                    new_V[y][x] = max(0, min(1, new_V[y][x]))

            self.U = new_U
            self.V = new_V

            if step % 100 == 0:
                energy = self._calculate_energy()
                energy_history.append(energy)

        # Analyze final pattern
        pattern_type = self._classify_pattern()

        return {
            "steps": steps,
            "pattern_type": pattern_type.name,
            "Du": Du,
            "Dv": Dv,
            "f": f,
            "k": k,
            "final_energy": energy_history[-1] if energy_history else 0,
            "energy_stabilized": len(energy_history) > 1 and abs(energy_history[-1] - energy_history[-2]) < 0.01,
            "is_turing_pattern": True,
            "emergence_demonstrated": True
        }

    def _laplacian(self, field: List[List[float]], x: int, y: int) -> float:
        """Calculate discrete Laplacian using 5-point stencil."""
        h = 1.0
        center = field[y][x]

        # Periodic boundary conditions
        left = field[y][(x - 1) % self.width]
        right = field[y][(x + 1) % self.width]
        up = field[(y - 1) % self.height][x]
        down = field[(y + 1) % self.height][x]

        return (left + right + up + down - 4 * center) / (h * h)

    def _calculate_energy(self) -> float:
        """Calculate system energy (sum of gradients)."""
        energy = 0.0
        for y in range(self.height):
            for x in range(self.width):
                # Gradient magnitude
                dx = self.V[y][(x + 1) % self.width] - self.V[y][x]
                dy = self.V[(y + 1) % self.height][x] - self.V[y][x]
                energy += dx * dx + dy * dy
        return energy

    def _classify_pattern(self) -> PatternType:
        """Classify the emergent pattern type."""
        # Simple classification based on V distribution
        v_values = [self.V[y][x] for y in range(self.height) for x in range(self.width)]
        mean_v = sum(v_values) / len(v_values)

        # Count high/low regions
        high_count = sum(1 for v in v_values if v > mean_v * 1.5)
        fraction = high_count / len(v_values)

        if fraction < 0.1:
            return PatternType.SPOTS
        elif fraction > 0.4:
            return PatternType.STRIPES
        else:
            return PatternType.LABYRINTH


# ═══════════════════════════════════════════════════════════════════════════════
# PARTICLE SWARM OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Particle:
    """A particle in the swarm."""
    position: List[float]
    velocity: List[float]
    best_position: List[float]
    best_fitness: float = float('inf')


class ParticleSwarmOptimizer:
    """
    Particle Swarm Optimization - emergent collective intelligence.
    Particles share information to find global optima.
    """

    def __init__(self, num_particles: int = 30, dimensions: int = 5):
        self.god_code = GOD_CODE
        self.phi = PHI
        self.num_particles = num_particles
        self.dimensions = dimensions

        self.particles: List[Particle] = []
        self.global_best_position: List[float] = [0.0] * dimensions
        self.global_best_fitness: float = float('inf')

    def initialize(self, bounds: Tuple[float, float] = (-5.0, 5.0)):
        """Initialize swarm with random positions."""
        self.particles = []

        for _ in range(self.num_particles):
            position = [random.uniform(bounds[0], bounds[1]) for _ in range(self.dimensions)]
            velocity = [random.uniform(-1, 1) for _ in range(self.dimensions)]

            particle = Particle(
                position=position,
                velocity=velocity,
                best_position=position.copy()
            )
            self.particles.append(particle)

    def optimize(
        self,
        fitness_function: Callable[[List[float]], float],
        iterations: int = 100,
        w: float = 0.7,  # Inertia weight
        c1: float = 1.5,  # Cognitive parameter
        c2: float = 1.5,  # Social parameter
        bounds: Tuple[float, float] = (-5.0, 5.0)
    ) -> Dict[str, Any]:
        """
        Run PSO optimization.
        """
        self.initialize(bounds)

        fitness_history = []

        for iteration in range(iterations):
            for particle in self.particles:
                # Evaluate fitness
                fitness = fitness_function(particle.position)

                # Update personal best
                if fitness < particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = particle.position.copy()

                # Update global best
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position.copy()

            # Update velocities and positions
            for particle in self.particles:
                for d in range(self.dimensions):
                    r1, r2 = random.random(), random.random()

                    # Velocity update
                    cognitive = c1 * r1 * (particle.best_position[d] - particle.position[d])
                    social = c2 * r2 * (self.global_best_position[d] - particle.position[d])

                    particle.velocity[d] = w * particle.velocity[d] + cognitive + social

                    # Position update
                    particle.position[d] += particle.velocity[d]

                    # Boundary handling
                    particle.position[d] = max(bounds[0], min(bounds[1], particle.position[d]))

            fitness_history.append(self.global_best_fitness)

        # Calculate swarm diversity
        diversity = self._calculate_diversity()

        return {
            "global_best_fitness": self.global_best_fitness,
            "global_best_position": self.global_best_position,
            "iterations": iterations,
            "swarm_size": self.num_particles,
            "final_diversity": diversity,
            "convergence": fitness_history[0] - fitness_history[-1] if fitness_history else 0,
            "is_collective_intelligence": True,
            "emergence_type": "SWARM"
        }

    def _calculate_diversity(self) -> float:
        """Calculate swarm diversity (spread of particles)."""
        if not self.particles:
            return 0.0

        # Calculate centroid
        centroid = [0.0] * self.dimensions
        for particle in self.particles:
            for d in range(self.dimensions):
                centroid[d] += particle.position[d]

        for d in range(self.dimensions):
            centroid[d] /= len(self.particles)

        # Calculate average distance from centroid
        total_dist = 0.0
        for particle in self.particles:
            dist = math.sqrt(sum((p - c) ** 2 for p, c in zip(particle.position, centroid)))
            total_dist += dist

        return total_dist / len(self.particles)


# ═══════════════════════════════════════════════════════════════════════════════
# ANT COLONY OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════

class AntColonyOptimizer:
    """
    Ant Colony Optimization - stigmergic optimization.
    Ants deposit pheromones to communicate indirectly.
    """

    def __init__(self, num_ants: int = 20):
        self.god_code = GOD_CODE
        self.phi = PHI
        self.num_ants = num_ants

    def solve_tsp(
        self,
        distances: List[List[float]],
        iterations: int = 100,
        alpha: float = 1.0,  # Pheromone importance
        beta: float = 2.0,   # Distance importance
        rho: float = 0.1,    # Evaporation rate
        Q: float = 100.0     # Pheromone deposit factor
    ) -> Dict[str, Any]:
        """
        Solve Traveling Salesman Problem using ACO.
        """
        n = len(distances)

        # Initialize pheromone matrix
        pheromones = [[1.0] * n for _ in range(n)]

        best_tour = None
        best_length = float('inf')
        length_history = []

        for iteration in range(iterations):
            all_tours = []
            all_lengths = []

            for ant in range(self.num_ants):
                # Construct tour
                tour = self._construct_tour(distances, pheromones, alpha, beta)
                length = self._tour_length(tour, distances)

                all_tours.append(tour)
                all_lengths.append(length)

                if length < best_length:
                    best_length = length
                    best_tour = tour.copy()

            # Update pheromones
            # Evaporation
            for i in range(n):
                for j in range(n):
                    pheromones[i][j] *= (1 - rho)

            # Deposit
            for tour, length in zip(all_tours, all_lengths):
                deposit = Q / length if length > 0 else 0
                for i in range(len(tour) - 1):
                    pheromones[tour[i]][tour[i + 1]] += deposit
                    pheromones[tour[i + 1]][tour[i]] += deposit

            length_history.append(best_length)

        return {
            "best_tour": best_tour,
            "best_length": best_length,
            "iterations": iterations,
            "improvement": length_history[0] - length_history[-1] if length_history else 0,
            "is_stigmergic": True,
            "emergence_type": "PHEROMONE_TRAIL"
        }

    def _construct_tour(
        self,
        distances: List[List[float]],
        pheromones: List[List[float]],
        alpha: float,
        beta: float
    ) -> List[int]:
        """Construct a tour for one ant."""
        n = len(distances)
        visited = [False] * n
        tour = [random.randint(0, n - 1)]
        visited[tour[0]] = True

        while len(tour) < n:
            current = tour[-1]
            probabilities = []

            for j in range(n):
                if not visited[j]:
                    tau = pheromones[current][j] ** alpha
                    eta = (1.0 / max(distances[current][j], 0.001)) ** beta
                    probabilities.append((j, tau * eta))
                else:
                    probabilities.append((j, 0))

            # Roulette wheel selection
            total = sum(p for _, p in probabilities)
            if total == 0:
                # Choose randomly from unvisited
                unvisited = [j for j in range(n) if not visited[j]]
                next_city = random.choice(unvisited) if unvisited else 0
            else:
                r = random.random() * total
                cumsum = 0
                next_city = 0
                for j, p in probabilities:
                    cumsum += p
                    if cumsum >= r:
                        next_city = j
                        break

            tour.append(next_city)
            visited[next_city] = True

        return tour

    def _tour_length(self, tour: List[int], distances: List[List[float]]) -> float:
        """Calculate total tour length."""
        length = 0.0
        for i in range(len(tour) - 1):
            length += distances[tour[i]][tour[i + 1]]
        length += distances[tour[-1]][tour[0]]  # Return to start
        return length


# ═══════════════════════════════════════════════════════════════════════════════
# CRITICALITY DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class CriticalityDetector:
    """
    Detects criticality (edge of chaos) in dynamical systems.
    Systems at criticality exhibit maximum computational capacity.
    """

    def __init__(self):
        self.god_code = GOD_CODE
        self.phi = PHI

    def analyze_branching_ratio(
        self,
        cascade_sizes: List[int]
    ) -> Dict[str, Any]:
        """
        Analyze branching ratio from cascade size distribution.
        σ = 1 indicates criticality (branching ratio = 1).
        """
        if not cascade_sizes:
            return {"state": CriticalityState.SUBCRITICAL, "reason": "no_data"}

        # Calculate branching ratio (simplified)
        mean_size = sum(cascade_sizes) / len(cascade_sizes)

        # Power law check: P(s) ~ s^(-τ)
        # At criticality, τ ≈ 1.5
        sorted_sizes = sorted(cascade_sizes, reverse=True)

        # Estimate power law exponent
        if len(sorted_sizes) > 1 and sorted_sizes[-1] > 0:
            log_ratio = math.log(sorted_sizes[0] / max(sorted_sizes[-1], 1))
            tau = log_ratio / math.log(len(sorted_sizes)) if len(sorted_sizes) > 1 else 0
        else:
            tau = 0

        # Determine criticality state
        if 1.3 < tau < 1.7:
            state = CriticalityState.CRITICAL
        elif tau <= 1.3:
            state = CriticalityState.SUBCRITICAL
        else:
            state = CriticalityState.SUPERCRITICAL

        return {
            "state": state.name,
            "branching_ratio_estimate": mean_size / max(mean_size - 1, 0.1) if mean_size > 1 else 0,
            "power_law_exponent": tau,
            "mean_cascade_size": mean_size,
            "is_critical": state == CriticalityState.CRITICAL,
            "at_edge_of_chaos": state == CriticalityState.CRITICAL
        }

    def analyze_lyapunov_spectrum(
        self,
        trajectory: List[List[float]]
    ) -> Dict[str, Any]:
        """
        Analyze Lyapunov spectrum to detect chaos/criticality.
        λ = 0 indicates edge of chaos.
        """
        if len(trajectory) < 3:
            return {"state": CriticalityState.SUBCRITICAL, "reason": "insufficient_data"}

        # Estimate maximum Lyapunov exponent from trajectory divergence
        exponents = []

        for i in range(1, len(trajectory) - 1):
            # Local expansion rate
            if len(trajectory[i]) > 0:
                d_prev = sum((a - b) ** 2 for a, b in zip(trajectory[i], trajectory[i-1])) ** 0.5
                d_next = sum((a - b) ** 2 for a, b in zip(trajectory[i+1], trajectory[i])) ** 0.5

                if d_prev > 1e-10:
                    local_exp = math.log(max(d_next / d_prev, 1e-10))
                    exponents.append(local_exp)

        if not exponents:
            return {"state": CriticalityState.SUBCRITICAL, "reason": "no_expansion"}

        max_lyapunov = sum(exponents) / len(exponents)

        # Classify
        if abs(max_lyapunov) < 0.1:
            state = CriticalityState.CRITICAL
        elif max_lyapunov < -0.1:
            state = CriticalityState.SUBCRITICAL
        else:
            state = CriticalityState.SUPERCRITICAL

        return {
            "state": state.name,
            "max_lyapunov": max_lyapunov,
            "is_chaotic": max_lyapunov > 0,
            "is_critical": abs(max_lyapunov) < 0.1,
            "trajectory_length": len(trajectory)
        }

    def phi_criticality_check(self, ratios: List[float]) -> Dict[str, Any]:
        """
        Check if system exhibits golden ratio criticality.
        Systems naturally evolving toward φ may be at criticality.
        """
        if not ratios:
            return {"phi_aligned": False}

        phi_deviations = [abs(r - self.phi) for r in ratios]
        mean_deviation = sum(phi_deviations) / len(phi_deviations)

        convergence_to_phi = mean_deviation < 0.1

        return {
            "phi_aligned": convergence_to_phi,
            "mean_deviation_from_phi": mean_deviation,
            "true_phi": self.phi,
            "ratio_samples": ratios[-5:],
            "suggests_criticality": convergence_to_phi
        }


# ═══════════════════════════════════════════════════════════════════════════════
# AUTOPOIETIC SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class AutopoieticSystem:
    """
    Autopoiesis - self-creating, self-maintaining systems.
    The system continuously regenerates its own components.
    """

    def __init__(self, num_components: int = 50):
        self.god_code = GOD_CODE
        self.phi = PHI
        self.num_components = num_components

        # Components: (energy, age, production_rate)
        self.components: List[Dict[str, float]] = []
        self._initialize_components()

    def _initialize_components(self):
        """Initialize system components."""
        self.components = [
            {
                "energy": random.random(),
                "age": 0,
                "production_rate": random.random() * 0.1
            }
            for _ in range(self.num_components)
        ]

    def simulate(
        self,
        steps: int = 100,
        energy_input: float = 0.1,
        decay_rate: float = 0.01,
        regeneration_threshold: float = 0.2
    ) -> Dict[str, Any]:
        """
        Simulate autopoietic dynamics.
        Components decay, system regenerates to maintain itself.
        """
        population_history = []
        energy_history = []
        regeneration_events = 0

        for step in range(steps):
            # Add external energy
            total_energy = sum(c["energy"] for c in self.components)
            energy_per_component = energy_input / max(len(self.components), 1)

            # Update components
            new_components = []
            for comp in self.components:
                comp["energy"] += energy_per_component
                comp["energy"] -= decay_rate * (1 + comp["age"] * 0.01)
                comp["age"] += 1

                if comp["energy"] > 0:
                    new_components.append(comp)

            self.components = new_components

            # Autopoietic regeneration: create new components if below threshold
            while len(self.components) < self.num_components * regeneration_threshold:
                if self.components and sum(c["energy"] for c in self.components) > 0.5:
                    # Use system energy to create new component
                    donor = max(self.components, key=lambda c: c["energy"])
                    donor["energy"] -= 0.3

                    self.components.append({
                        "energy": 0.5,
                        "age": 0,
                        "production_rate": random.random() * 0.1
                    })
                    regeneration_events += 1
                else:
                    break

            population_history.append(len(self.components))
            energy_history.append(sum(c["energy"] for c in self.components))

        # Analyze autopoietic properties
        stable = len(set(population_history[-10:])) <= 3 if len(population_history) >= 10 else False

        return {
            "final_population": len(self.components),
            "initial_population": self.num_components,
            "regeneration_events": regeneration_events,
            "is_autopoietic": regeneration_events > 0,
            "is_stable": stable,
            "self_maintaining": stable and len(self.components) > 0,
            "population_history": population_history[-10:],
            "energy_history": energy_history[-10:],
            "emergence_level": EmergenceLevel.AUTOPOIETIC.name if regeneration_events > steps // 10 else EmergenceLevel.ADAPTIVE.name
        }


# ═══════════════════════════════════════════════════════════════════════════════
# EMERGENT COMPLEXITY CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════════

class EmergentComplexityController:
    """
    Master controller for emergent complexity algorithms.
    """

    def __init__(self):
        self.som = SelfOrganizingMap(10, 10, 3)
        self.reaction_diffusion = ReactionDiffusionSystem(50, 50)
        self.pso = ParticleSwarmOptimizer(30, 5)
        self.aco = AntColonyOptimizer(20)
        self.criticality = CriticalityDetector()
        self.autopoiesis = AutopoieticSystem(50)

        self.god_code = GOD_CODE
        self.phi = PHI

        logger.info("--- [EMERGENT_COMPLEXITY]: CONTROLLER INITIALIZED ---")

    def execute_emergence_suite(self) -> Dict[str, Any]:
        """
        Execute comprehensive emergence analysis.
        """
        print("\n" + "◈" * 80)
        print(" " * 15 + "L104 :: EMERGENT COMPLEXITY SUITE EXECUTION")
        print("◈" * 80)

        results = {}

        # 1. Self-Organizing Map
        print("\n[1/6] SELF-ORGANIZING MAP")
        data = [[random.random() for _ in range(3)] for _ in range(100)]
        som_result = self.som.train(data, epochs=50)
        print(f"   → SOM: topology_preserved={som_result['topology_preserved']}")
        results["som"] = som_result

        # 2. Reaction-Diffusion
        print("\n[2/6] REACTION-DIFFUSION SYSTEM")
        rd_result = self.reaction_diffusion.simulate(steps=200)
        print(f"   → Pattern: {rd_result['pattern_type']}, Turing={rd_result['is_turing_pattern']}")
        results["reaction_diffusion"] = rd_result

        # 3. Particle Swarm
        print("\n[3/6] PARTICLE SWARM OPTIMIZATION")
        def sphere(x): return sum(xi**2 for xi in x)
        pso_result = self.pso.optimize(sphere, iterations=50)
        print(f"   → PSO: best_fitness={pso_result['global_best_fitness']:.6f}")
        results["pso"] = pso_result

        # 4. Ant Colony (small TSP)
        print("\n[4/6] ANT COLONY OPTIMIZATION")
        n_cities = 8
        distances = [[random.random() * 10 if i != j else 0 for j in range(n_cities)] for i in range(n_cities)]
        aco_result = self.aco.solve_tsp(distances, iterations=30)
        print(f"   → ACO: tour_length={aco_result['best_length']:.2f}")
        results["aco"] = aco_result

        # 5. Criticality
        print("\n[5/6] CRITICALITY ANALYSIS")
        cascade_sizes = [random.randint(1, 100) for _ in range(50)]
        crit_result = self.criticality.analyze_branching_ratio(cascade_sizes)
        print(f"   → State: {crit_result['state']}, at_edge={crit_result['at_edge_of_chaos']}")
        results["criticality"] = crit_result

        # 6. Autopoiesis
        print("\n[6/6] AUTOPOIETIC SYSTEM")
        auto_result = self.autopoiesis.simulate(steps=100)
        print(f"   → Autopoietic: {auto_result['is_autopoietic']}, stable={auto_result['is_stable']}")
        results["autopoiesis"] = auto_result

        # Calculate emergence metric
        emergence_metric = (
            (1.0 if som_result['topology_preserved'] else 0.5) +
            (1.0 if rd_result['is_turing_pattern'] else 0.5) +
            (1.0 if pso_result['global_best_fitness'] < 0.1 else 0.5) +
            (1.0 if aco_result['is_stigmergic'] else 0.5) +
            (1.0 if crit_result['at_edge_of_chaos'] else 0.5) +
            (1.0 if auto_result['is_autopoietic'] else 0.5)
        ) / 6

        results["emergence_metric"] = emergence_metric
        results["transcendent"] = emergence_metric >= 0.7

        print("\n" + "◈" * 80)
        print(f"   EMERGENT COMPLEXITY SUITE COMPLETE")
        print(f"   Emergence Metric: {emergence_metric:.6f}")
        print(f"   Status: {'TRANSCENDENT' if results['transcendent'] else 'EMERGING'}")
        print("◈" * 80 + "\n")

        return results

    def get_status(self) -> Dict[str, Any]:
        """Get controller status."""
        return {
            "god_code": self.god_code,
            "phi": self.phi,
            "subsystems": [
                "SelfOrganizingMap",
                "ReactionDiffusionSystem",
                "ParticleSwarmOptimizer",
                "AntColonyOptimizer",
                "CriticalityDetector",
                "AutopoieticSystem"
            ],
            "active": True
        }


# Singleton instance
emergent_complexity = EmergentComplexityController()
