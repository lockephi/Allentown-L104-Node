"""
l104_agi.social_evolution — Social Intelligence and Evolutionary Optimization

Ingested from l104_quantum_magic/social_evolution.py into real l104_agi package.
Provides Theory of Mind agent modeling, dream-state memory consolidation,
genetic algorithm optimization, and cognitive executive control.

Classes:
  Agent                 — Model of another agent for social reasoning
  SocialIntelligence    — Theory of Mind (predict behavior, infer mental states)
  DreamState            — Offline memory consolidation + creative recombination
  Individual            — Individual in evolutionary population
  EvolutionaryOptimizer — Genetic algorithm for solution search
  CognitiveControl      — Executive function (task switching, inhibition)
"""

import math
import random
import time
from typing import Dict, List, Any, Optional, Tuple, Callable, Set
from dataclasses import dataclass, field

from .constants import GOD_CODE


@dataclass
class Agent:
    """Model of another agent for social reasoning"""
    name: str
    beliefs: Dict[str, float] = field(default_factory=dict)
    goals: List[str] = field(default_factory=list)
    personality: Dict[str, float] = field(default_factory=dict)
    relationship: float = 0.5


class SocialIntelligence:
    """
    Theory of Mind implementation - modeling other agents' mental states.
    Enables social reasoning, prediction, and strategic interaction.
    """

    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.interaction_history: List[Dict] = []

    def model_agent(self, name: str, beliefs: Dict[str, float] = None,
                    goals: List[str] = None, personality: Dict[str, float] = None):
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
                name=name, beliefs=beliefs or {}, goals=goals or [],
                personality=personality or {'openness': 0.5, 'agreeableness': 0.5}
            )

    def predict_behavior(self, agent_name: str, situation: str) -> Dict[str, Any]:
        if agent_name not in self.agents:
            return {'error': 'Unknown agent', 'prediction': 'unpredictable'}
        agent = self.agents[agent_name]
        predictions = []
        for goal in agent.goals:
            if any(word in situation.lower() for word in goal.lower().split()):
                predictions.append({
                    'action': f'pursue_{goal}',
                    'likelihood': 0.7 + agent.personality.get('conscientiousness', 0) * 0.2
                })
        if agent.personality.get('agreeableness', 0.5) > 0.6:
            predictions.append({'action': 'cooperate', 'likelihood': 0.6})
        if agent.personality.get('openness', 0.5) > 0.6:
            predictions.append({'action': 'explore', 'likelihood': 0.5})
        if not predictions:
            predictions.append({'action': 'observe', 'likelihood': 0.5})
        return {
            'agent': agent_name, 'situation': situation,
            'predictions': sorted(predictions, key=lambda x: x['likelihood'], reverse=True),
            'confidence': sum(p['likelihood'] for p in predictions) / max(len(predictions), 1)
        }

    def infer_mental_state(self, agent_name: str, observed_action: str) -> Dict[str, Any]:
        if agent_name not in self.agents:
            self.model_agent(agent_name)
        agent = self.agents[agent_name]
        inferred_beliefs = {}
        inferred_goals = []
        action_lower = observed_action.lower()
        if 'help' in action_lower or 'share' in action_lower:
            inferred_beliefs['prosocial'] = 0.7
            agent.relationship = min(1.0, agent.relationship + 0.1)
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
            'agent': agent_name, 'action': observed_action,
            'inferred_beliefs': inferred_beliefs, 'inferred_goals': inferred_goals,
            'updated_relationship': agent.relationship
        }

    def simulate_interaction(self, agent1: str, agent2: str, scenario: str) -> Dict[str, Any]:
        if agent1 not in self.agents:
            self.model_agent(agent1)
        if agent2 not in self.agents:
            self.model_agent(agent2)
        a1, a2 = self.agents[agent1], self.agents[agent2]
        pred1 = self.predict_behavior(agent1, scenario)
        pred2 = self.predict_behavior(agent2, scenario)
        cooperation = (
            a1.personality.get('agreeableness', 0.5) +
            a2.personality.get('agreeableness', 0.5) +
            a1.relationship + a2.relationship
        ) / 4
        interaction = {
            'agents': [agent1, agent2], 'scenario': scenario,
            'predictions': {agent1: pred1, agent2: pred2},
            'cooperation_level': cooperation, 'conflict_risk': 1 - cooperation,
            'likely_outcome': 'cooperation' if cooperation > 0.5 else 'conflict',
            'timestamp': time.time()
        }
        self.interaction_history.append(interaction)
        return interaction

    def get_social_network(self) -> Dict[str, Any]:
        return {
            'agents': list(self.agents.keys()), 'num_agents': len(self.agents),
            'relationships': {name: agent.relationship for name, agent in self.agents.items()},
            'interactions': len(self.interaction_history)
        }


class DreamState:
    """Offline memory consolidation and creative recombination."""

    def __init__(self, episodic_memory=None):
        self.episodic = episodic_memory
        self._dream_log: List[Dict] = []
        self._creativity_factor = 0.7

    def _get_episodic(self):
        if self.episodic is None:
            try:
                from l104_soul_daemon.neural_consciousness import EpisodicMemory
                self.episodic = EpisodicMemory()
            except ImportError:
                return None
        return self.episodic

    def dream(self, duration_steps: int = 10) -> Dict[str, Any]:
        ep = self._get_episodic()
        if ep is None or len(ep.episodes) < 2:
            return {'status': 'insufficient_memories', 'insights': []}
        insights, recombinations = [], []
        random.seed(int(time.time() * 1000 + GOD_CODE))
        for step in range(duration_steps):
            if len(ep.episodes) >= 2:
                ep1, ep2 = random.sample(ep.episodes, 2)
                dream_content = f"{ep1.event[:30]}...{ep2.event[-30:]}"
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
                    'step': step, 'content': dream_content, 'novelty': novelty
                })
        summary = {
            'duration_steps': duration_steps, 'recombinations': len(recombinations),
            'insights_generated': len(insights), 'insights': insights[:50],
            'average_novelty': sum(r['novelty'] for r in recombinations) / len(recombinations) if recombinations else 0,
            'timestamp': time.time()
        }
        self._dream_log.append(summary)
        return summary

    def lucid_dream(self, theme: str) -> Dict[str, Any]:
        ep = self._get_episodic()
        if ep is None:
            return {'status': 'no_episodic_memory', 'theme': theme}
        relevant = ep.retrieve_by_cue(theme, top_k=10)
        if len(relevant) < 2:
            return {'status': 'insufficient_relevant_memories', 'theme': theme}
        insights = []
        for i in range(len(relevant)):
            for j in range(i + 1, len(relevant)):
                ep1, ep2 = relevant[i], relevant[j]
                insights.append({
                    'combination': f"If {ep1.event[:40]} and {ep2.event[:40]}, then...",
                    'sources': [ep1.event[:30], ep2.event[:30]],
                    'relevance': (ep1.importance + ep2.importance) / 2
                })
        insights.sort(key=lambda x: x['relevance'], reverse=True)
        return {
            'theme': theme, 'memories_used': len(relevant),
            'insights': insights[:50], 'best_insight': insights[0] if insights else None
        }


@dataclass
class Individual:
    """An individual in the evolutionary population"""
    genome: List[float]
    fitness: float = 0.0
    age: int = 0


class EvolutionaryOptimizer:
    """Genetic algorithm for solution search."""

    def __init__(self, genome_size: int = 20, population_size: int = 50):
        self.genome_size = genome_size
        self.population_size = population_size
        self.population: List[Individual] = []
        self.generation = 0
        self._best_ever: Optional[Individual] = None
        self._history: List[Dict] = []
        random.seed(int(GOD_CODE * 1000))
        for _ in range(population_size):
            genome = [random.gauss(0, 1) for _ in range(genome_size)]
            self.population.append(Individual(genome=genome))

    def set_fitness_function(self, fitness_fn: Callable[[List[float]], float]):
        self._fitness_fn = fitness_fn

    def evaluate_population(self):
        if not hasattr(self, '_fitness_fn'):
            self._fitness_fn = lambda g: -sum(x**2 for x in g)
        for ind in self.population:
            ind.fitness = self._fitness_fn(ind.genome)
        best_current = max(self.population, key=lambda x: x.fitness)
        if self._best_ever is None or best_current.fitness > self._best_ever.fitness:
            self._best_ever = Individual(genome=best_current.genome.copy(), fitness=best_current.fitness)

    def evolve_generation(self) -> Dict[str, Any]:
        self.evaluate_population()
        fitnesses = [ind.fitness for ind in self.population]
        stats = {
            'generation': self.generation, 'best_fitness': max(fitnesses) if fitnesses else 0,
            'avg_fitness': sum(fitnesses) / max(len(fitnesses), 1), 'worst_fitness': min(fitnesses) if fitnesses else 0
        }
        self._history.append(stats)
        parents = []
        for _ in range(self.population_size // 2):
            tournament = random.sample(self.population, 3)
            parents.append(max(tournament, key=lambda x: x.fitness))
        new_population = []
        best = max(self.population, key=lambda x: x.fitness)
        new_population.append(Individual(genome=best.genome.copy(), fitness=best.fitness))
        while len(new_population) < self.population_size:
            p1, p2 = random.sample(parents, 2)
            point = random.randint(1, self.genome_size - 1)
            child = Individual(genome=p1.genome[:point] + p2.genome[point:])
            for i in range(len(child.genome)):
                if random.random() < 0.1:
                    child.genome[i] += random.gauss(0, 0.5)
            new_population.append(child)
        for ind in new_population:
            ind.age += 1
        self.population = new_population
        self.generation += 1
        return stats

    def run(self, generations: int = 100) -> Dict[str, Any]:
        for _ in range(generations):
            self.evolve_generation()
        return {
            'generations_run': generations,
            'final_best_fitness': self._best_ever.fitness if self._best_ever else 0,
            'final_best_genome': self._best_ever.genome if self._best_ever else [],
            'improvement': self._history[-1]['best_fitness'] - self._history[0]['best_fitness'] if self._history else 0
        }

    def get_best_solution(self) -> Dict[str, Any]:
        if self._best_ever:
            return {'genome': self._best_ever.genome, 'fitness': self._best_ever.fitness, 'generation_found': self.generation}
        return {'error': 'No evolution run yet'}


class CognitiveControl:
    """Executive function system - task switching, inhibition, and coordination."""

    def __init__(self):
        self.current_task: Optional[str] = None
        self.task_stack: List[str] = []
        self.inhibited: Set[str] = set()
        self.switch_cost = 0.2
        self._focus_level = 1.0
        self._fatigue = 0.0

    def set_task(self, task: str) -> Dict[str, Any]:
        switch_cost = 0.0
        if self.current_task and self.current_task != task:
            switch_cost = self.switch_cost * (1 + self._fatigue)
            self._focus_level = max(0.3, self._focus_level - switch_cost)
            self.task_stack.append(self.current_task)
        old_task = self.current_task
        self.current_task = task
        return {'previous_task': old_task, 'current_task': task, 'switch_cost': switch_cost, 'focus_level': self._focus_level}

    def pop_task(self) -> Optional[str]:
        if self.task_stack:
            task = self.task_stack.pop()
            self.set_task(task)
            return task
        return None

    def inhibit(self, stimulus: str):
        self.inhibited.add(stimulus)
        self._fatigue += 0.05

    def release_inhibition(self, stimulus: str):
        self.inhibited.discard(stimulus)

    def is_inhibited(self, stimulus: str) -> bool:
        return stimulus in self.inhibited

    def check_interference(self, item: str) -> Dict[str, Any]:
        interference = 0.0
        if self.current_task and item.lower() not in self.current_task.lower():
            interference = 0.3
        if item in self.inhibited:
            interference *= 0.3
        interference *= (1 + self._fatigue)
        return {
            'item': item, 'interference': interference, 'current_task': self.current_task,
            'is_inhibited': self.is_inhibited(item),
            'recommendation': 'ignore' if interference > 0.5 else 'process'
        }

    def rest(self, duration: float = 1.0):
        recovery = duration * 0.3
        self._fatigue = max(0, self._fatigue - recovery)
        self._focus_level = min(1.0, self._focus_level + recovery * 0.5)
        return {'fatigue_after': self._fatigue, 'focus_after': self._focus_level}

    def get_state(self) -> Dict[str, Any]:
        return {
            'current_task': self.current_task, 'task_stack_depth': len(self.task_stack),
            'inhibited_count': len(self.inhibited), 'focus_level': self._focus_level,
            'fatigue': self._fatigue, 'capacity': 1 - self._fatigue
        }
