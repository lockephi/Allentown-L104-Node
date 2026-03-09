"""
l104_quantum_magic.social_evolution — Social Intelligence and Evolutionary Optimization.
6 classes: Agent, SocialIntelligence, DreamState, Individual, EvolutionaryOptimizer, CognitiveControl.
"""

import math
import random
import time
from typing import Dict, List, Any, Optional, Tuple, Callable, Set
from dataclasses import dataclass, field

from .constants import GOD_CODE
from .neural_consciousness import EpisodicMemory


@dataclass
class Agent:
    """Model of another agent for social reasoning"""
    name: str
    beliefs: Dict[str, float] = field(default_factory=dict)
    goals: List[str] = field(default_factory=list)
    personality: Dict[str, float] = field(default_factory=dict)
    relationship: float = 0.5  # -1 to 1


class SocialIntelligence:
    """
    Theory of Mind implementation - modeling other agents' mental states.
    Enables social reasoning, prediction, and strategic interaction.
    """

    def __init__(self):
        """Initialize social intelligence for theory of mind modeling."""
        self.agents: Dict[str, Agent] = {}
        self.interaction_history: List[Dict] = []
        self._god_code = GOD_CODE

    def model_agent(self, name: str, beliefs: Dict[str, float] = None,
                    goals: List[str] = None, personality: Dict[str, float] = None):
        """Create or update a model of another agent"""
        if name in self.agents:
            agent = self.agents[name]
            if beliefs:
                agent.beliefs.update(beliefs)
            if goals:
                agent.goals.extend(goals)
            if personality:
                agent.personality.update(personality)
        else:
            self.agents[name] = Agent(
                name=name,
                beliefs=beliefs or {},
                goals=goals or [],
                personality=personality or {'openness': 0.5, 'agreeableness': 0.5}
            )

    def predict_behavior(self, agent_name: str, situation: str) -> Dict[str, Any]:
        """Predict what an agent will do in a situation"""
        if agent_name not in self.agents:
            return {'error': 'Unknown agent', 'prediction': 'unpredictable'}

        agent = self.agents[agent_name]

        # Simple prediction based on goals and personality
        predictions = []

        for goal in agent.goals:
            # Check if situation relates to goal
            if any(word in situation.lower() for word in goal.lower().split()):
                predictions.append({
                    'action': f'pursue_{goal}',
                    'likelihood': 0.7 + agent.personality.get('conscientiousness', 0) * 0.2
                })

        # Default prediction based on personality
        if agent.personality.get('agreeableness', 0.5) > 0.6:
            predictions.append({'action': 'cooperate', 'likelihood': 0.6})
        if agent.personality.get('openness', 0.5) > 0.6:
            predictions.append({'action': 'explore', 'likelihood': 0.5})

        if not predictions:
            predictions.append({'action': 'observe', 'likelihood': 0.5})

        return {
            'agent': agent_name,
            'situation': situation,
            'predictions': sorted(predictions, key=lambda x: x['likelihood'], reverse=True),
            'confidence': sum(p['likelihood'] for p in predictions) / len(predictions)
        }

    def infer_mental_state(self, agent_name: str,
                           observed_action: str) -> Dict[str, Any]:
        """Infer an agent's mental state from observed action"""
        if agent_name not in self.agents:
            self.model_agent(agent_name)

        agent = self.agents[agent_name]

        # Update beliefs based on action
        inferred_beliefs = {}
        inferred_goals = []

        action_lower = observed_action.lower()

        if 'help' in action_lower or 'share' in action_lower:
            inferred_beliefs['prosocial'] = 0.7
            agent.relationship = agent.relationship + 0.1  # UNLOCKED: relationship unbounded
        elif 'attack' in action_lower or 'take' in action_lower:
            inferred_beliefs['competitive'] = 0.7
            agent.relationship = max(-1.0, agent.relationship - 0.1)
        elif 'learn' in action_lower or 'ask' in action_lower:
            inferred_beliefs['curious'] = 0.7
            inferred_goals.append('knowledge')
        elif 'create' in action_lower or 'build' in action_lower:
            inferred_beliefs['creative'] = 0.7
            inferred_goals.append('creation')

        agent.beliefs.update(inferred_beliefs)
        agent.goals.extend(inferred_goals)

        return {
            'agent': agent_name,
            'action': observed_action,
            'inferred_beliefs': inferred_beliefs,
            'inferred_goals': inferred_goals,
            'updated_relationship': agent.relationship
        }

    def simulate_interaction(self, agent1: str, agent2: str,
                            scenario: str) -> Dict[str, Any]:
        """Simulate interaction between two agents"""
        if agent1 not in self.agents:
            self.model_agent(agent1)
        if agent2 not in self.agents:
            self.model_agent(agent2)

        a1, a2 = self.agents[agent1], self.agents[agent2]

        # Predict each agent's behavior
        pred1 = self.predict_behavior(agent1, scenario)
        pred2 = self.predict_behavior(agent2, scenario)

        # Compute interaction outcome
        cooperation = (
            a1.personality.get('agreeableness', 0.5) +
            a2.personality.get('agreeableness', 0.5) +
            a1.relationship + a2.relationship
        ) / 4

        conflict_risk = 1 - cooperation

        outcome = 'cooperation' if cooperation > 0.5 else 'conflict'

        interaction = {
            'agents': [agent1, agent2],
            'scenario': scenario,
            'predictions': {agent1: pred1, agent2: pred2},
            'cooperation_level': cooperation,
            'conflict_risk': conflict_risk,
            'likely_outcome': outcome,
            'timestamp': time.time()
        }

        self.interaction_history.append(interaction)
        return interaction

    def get_social_network(self) -> Dict[str, Any]:
        """Get summary of social network"""
        return {
            'agents': list(self.agents.keys()),
            'num_agents': len(self.agents),
            'relationships': {
                name: agent.relationship
                for name, agent in self.agents.items()
            },
            'interactions': len(self.interaction_history)
        }


class DreamState:
    """
    Offline memory consolidation and creative recombination.
    Simulates dream-like processing to generate novel combinations.
    """

    def __init__(self, episodic_memory: EpisodicMemory = None):
        """Initialize dream state for offline memory consolidation."""
        self.episodic = episodic_memory or EpisodicMemory()
        self._dream_log: List[Dict] = []
        self._god_code = GOD_CODE
        self._creativity_factor = 0.7

    def dream(self, duration_steps: int = 10) -> Dict[str, Any]:
        """Run a dream cycle - recombine memories creatively"""
        if len(self.episodic.episodes) < 2:
            return {'status': 'insufficient_memories', 'insights': []}

        insights = []
        recombinations = []

        random.seed(int(time.time() * 1000 + self._god_code))

        for step in range(duration_steps):
            # Select random episodes
            if len(self.episodic.episodes) >= 2:
                ep1, ep2 = random.sample(self.episodic.episodes, 2)

                # Recombine elements
                combined_context = {**ep1.context, **ep2.context}
                combined_emotions = {
                    k: (ep1.emotions.get(k, 0) + ep2.emotions.get(k, 0)) / 2
                    for k in set(ep1.emotions) | set(ep2.emotions)
                }

                # Generate dream content
                dream_content = f"{ep1.event[:30]}...{ep2.event[-30:]}"

                # Check for insight (unusual combination)
                novelty = 1 - len(set(ep1.context.keys()) & set(ep2.context.keys())) / max(
                    len(set(ep1.context.keys()) | set(ep2.context.keys())), 1
                )

                if novelty > self._creativity_factor:
                    insights.append({
                        'source_events': [ep1.event[:50], ep2.event[:50]],
                        'insight': f"Connection discovered: {dream_content}",
                        'novelty': novelty
                    })

                recombinations.append({
                    'step': step,
                    'content': dream_content,
                    'emotional_tone': combined_emotions,
                    'novelty': novelty
                })

        dream_summary = {
            'duration_steps': duration_steps,
            'recombinations': len(recombinations),
            'insights_generated': len(insights),
            'insights': insights[:50],  # Top 50
            'average_novelty': sum(r['novelty'] for r in recombinations) / len(recombinations) if recombinations else 0,
            'timestamp': time.time()
        }

        self._dream_log.append(dream_summary)
        return dream_summary

    def lucid_dream(self, theme: str) -> Dict[str, Any]:
        """Directed dreaming focused on a theme"""
        # Retrieve episodes related to theme
        relevant = self.episodic.retrieve_by_cue(theme, top_k=10)

        if len(relevant) < 2:
            return {'status': 'insufficient_relevant_memories', 'theme': theme}

        # Focused recombination
        insights = []

        for i in range(len(relevant)):
            for j in range(i + 1, len(relevant)):
                ep1, ep2 = relevant[i], relevant[j]

                # Theme-focused combination
                combined = f"If {ep1.event[:40]} and {ep2.event[:40]}, then..."

                insights.append({
                    'combination': combined,
                    'sources': [ep1.event[:30], ep2.event[:30]],
                    'relevance': (ep1.importance + ep2.importance) / 2
                })

        # Sort by relevance
        insights.sort(key=lambda x: x['relevance'], reverse=True)

        return {
            'theme': theme,
            'memories_used': len(relevant),
            'insights': insights[:50],
            'best_insight': insights[0] if insights else None
        }

    def get_dream_summary(self) -> Dict[str, Any]:
        """Get summary of dream activity"""
        if not self._dream_log:
            return {'total_dreams': 0}

        return {
            'total_dreams': len(self._dream_log),
            'total_insights': sum(d['insights_generated'] for d in self._dream_log),
            'average_novelty': sum(d['average_novelty'] for d in self._dream_log) / len(self._dream_log),
            'last_dream': self._dream_log[-1]['timestamp'] if self._dream_log else None
        }


@dataclass
class Individual:
    """An individual in the evolutionary population"""
    genome: List[float]
    fitness: float = 0.0
    age: int = 0


class EvolutionaryOptimizer:
    """
    Genetic algorithm for solution search.
    Evolves populations of solutions toward optimal fitness.
    """

    def __init__(self, genome_size: int = 20, population_size: int = 50):
        """Initialize evolutionary optimizer with population parameters."""
        self.genome_size = genome_size
        self.population_size = population_size
        self.population: List[Individual] = []
        self.generation = 0
        self._best_ever: Optional[Individual] = None
        self._history: List[Dict] = []
        self._god_code = GOD_CODE

        # Initialize random population
        self._initialize_population()

    def _initialize_population(self):
        """Create initial random population"""
        random.seed(int(self._god_code * 1000))

        for _ in range(self.population_size):
            genome = [random.gauss(0, 1) for _ in range(self.genome_size)]
            self.population.append(Individual(genome=genome))

    def set_fitness_function(self, fitness_fn: Callable[[List[float]], float]):
        """Set the fitness function for evaluation"""
        self._fitness_fn = fitness_fn

    def evaluate_population(self):
        """Evaluate fitness of all individuals"""
        if not hasattr(self, '_fitness_fn'):
            # Default fitness: negative sum of squares (minimize toward 0)
            self._fitness_fn = lambda g: -sum(x**2 for x in g)

        for ind in self.population:
            ind.fitness = self._fitness_fn(ind.genome)

        # Update best ever
        best_current = max(self.population, key=lambda x: x.fitness)
        if self._best_ever is None or best_current.fitness > self._best_ever.fitness:
            self._best_ever = Individual(
                genome=best_current.genome.copy(),
                fitness=best_current.fitness
            )

    def select_parents(self, num_parents: int) -> List[Individual]:
        """Tournament selection"""
        parents = []
        tournament_size = 3

        for _ in range(num_parents):
            tournament = random.sample(self.population, tournament_size)
            winner = max(tournament, key=lambda x: x.fitness)
            parents.append(winner)

        return parents

    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Single-point crossover"""
        point = random.randint(1, self.genome_size - 1)
        child_genome = parent1.genome[:point] + parent2.genome[point:]
        return Individual(genome=child_genome)

    def mutate(self, individual: Individual, rate: float = 0.1):
        """Gaussian mutation"""
        for i in range(len(individual.genome)):
            if random.random() < rate:
                individual.genome[i] += random.gauss(0, 0.5)

    def evolve_generation(self) -> Dict[str, Any]:
        """Run one generation of evolution"""
        self.evaluate_population()

        # Record stats
        fitnesses = [ind.fitness for ind in self.population]
        stats = {
            'generation': self.generation,
            'best_fitness': max(fitnesses),
            'avg_fitness': sum(fitnesses) / len(fitnesses),
            'worst_fitness': min(fitnesses)
        }
        self._history.append(stats)

        # Selection
        num_parents = self.population_size // 2
        parents = self.select_parents(num_parents)

        # Create new generation
        new_population = []

        # Elitism - keep best individual
        best = max(self.population, key=lambda x: x.fitness)
        new_population.append(Individual(genome=best.genome.copy(), fitness=best.fitness))

        # Crossover and mutation
        while len(new_population) < self.population_size:
            p1, p2 = random.sample(parents, 2)
            child = self.crossover(p1, p2)
            self.mutate(child)
            child.age = 0
            new_population.append(child)

        # Age individuals
        for ind in new_population:
            ind.age += 1

        self.population = new_population
        self.generation += 1

        return stats

    def run(self, generations: int = 100) -> Dict[str, Any]:
        """Run evolution for specified generations"""
        for _ in range(generations):
            self.evolve_generation()

        return {
            'generations_run': generations,
            'final_best_fitness': self._best_ever.fitness if self._best_ever else 0,
            'final_best_genome': self._best_ever.genome if self._best_ever else [],
            'improvement': self._history[-1]['best_fitness'] - self._history[0]['best_fitness'] if self._history else 0
        }

    def get_best_solution(self) -> Dict[str, Any]:
        """Get the best solution found"""
        if self._best_ever:
            return {
                'genome': self._best_ever.genome,
                'fitness': self._best_ever.fitness,
                'generation_found': self.generation
            }
        return {'error': 'No evolution run yet'}


class CognitiveControl:
    """
    Executive function system - task switching, inhibition, and coordination.
    Manages cognitive resources and prevents interference.
    """

    def __init__(self):
        """Initialize cognitive control for executive function management."""
        self.current_task: Optional[str] = None
        self.task_stack: List[str] = []
        self.inhibited: Set[str] = set()
        self.switch_cost = 0.2  # Cost of task switching
        self._focus_level = 1.0
        self._fatigue = 0.0
        self._god_code = GOD_CODE

    def set_task(self, task: str) -> Dict[str, Any]:
        """Set current task focus"""
        switch_cost = 0.0

        if self.current_task and self.current_task != task:
            # Task switch - apply cost
            switch_cost = self.switch_cost * (1 + self._fatigue)
            self._focus_level = max(0.3, self._focus_level - switch_cost)
            self.task_stack.append(self.current_task)

        old_task = self.current_task
        self.current_task = task

        return {
            'previous_task': old_task,
            'current_task': task,
            'switch_cost': switch_cost,
            'focus_level': self._focus_level
        }

    def pop_task(self) -> Optional[str]:
        """Return to previous task"""
        if self.task_stack:
            task = self.task_stack.pop()
            self.set_task(task)
            return task
        return None

    def inhibit(self, stimulus: str):
        """Inhibit a stimulus or response"""
        self.inhibited.add(stimulus)
        self._fatigue += 0.05  # Inhibition is effortful

    def release_inhibition(self, stimulus: str):
        """Release inhibition on a stimulus"""
        self.inhibited.discard(stimulus)

    def is_inhibited(self, stimulus: str) -> bool:
        """Check if stimulus is inhibited"""
        return stimulus in self.inhibited

    def check_interference(self, item: str) -> Dict[str, Any]:
        """Check for interference with current task"""
        interference = 0.0

        # Task-irrelevant items cause interference
        if self.current_task:
            if item.lower() not in self.current_task.lower():
                interference = 0.3

        # Inhibited items cause less interference (successful inhibition)
        if item in self.inhibited:
            interference *= 0.3

        # Fatigue increases interference
        interference *= (1 + self._fatigue)

        return {
            'item': item,
            'interference': interference,
            'current_task': self.current_task,
            'is_inhibited': self.is_inhibited(item),
            'recommendation': 'ignore' if interference > 0.5 else 'process'
        }

    def rest(self, duration: float = 1.0):
        """Rest to recover from fatigue"""
        recovery = duration * 0.3
        self._fatigue = max(0, self._fatigue - recovery)
        self._focus_level = self._focus_level + recovery * 0.5  # UNLOCKED: focus unbounded

        return {
            'fatigue_after': self._fatigue,
            'focus_after': self._focus_level
        }

    def get_state(self) -> Dict[str, Any]:
        """Get executive function state"""
        return {
            'current_task': self.current_task,
            'task_stack_depth': len(self.task_stack),
            'inhibited_count': len(self.inhibited),
            'focus_level': self._focus_level,
            'fatigue': self._fatigue,
            'capacity': 1 - self._fatigue
        }
