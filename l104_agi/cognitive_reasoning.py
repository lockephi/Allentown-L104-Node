"""
l104_agi.cognitive_reasoning — Intelligent Reasoning Framework (EVO_52)

Ingested from l104_quantum_magic/cognitive.py into real l104_agi package.
Provides 9 classes for multi-strategy reasoning, Bayesian inference,
adaptive learning, pattern recognition, meta-cognition, and prediction.

Classes:
  ReasoningStrategy  — 15 available reasoning strategies (enum)
  Observation        — Recorded observation with metadata
  Hypothesis         — Hypothesis with Bayesian updating + quantum amplitude
  ContextualMemory   — HDC-based context-aware memory retrieval
  QuantumInferenceEngine — Bayesian inference with quantum amplitude encoding
  AdaptiveLearner    — Reinforcement-like strategy selection (UCB exploration)
  PatternRecognizer  — Statistical + HDC pattern matching
  MetaCognition      — Self-monitoring reasoning quality
  PredictiveReasoner — Markov chain + quantum evolution prediction
"""

import math
import cmath
import random
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque, defaultdict, Counter

from .constants import GOD_CODE, PHI

# ═══════════════════════════════════════════════════════════════════════════════
# SACRED ALGORITHMS INTEGRATION (EVO_72)
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from l104_sacred_algorithms import (
        derive_scoring_weight,
        derive_learning_rate_sacred,
        derive_momentum_sacred,
        derive_threshold,
        derive_priority_score,
        derive_activation_threshold,
        derive_batch_epochs,
    )
except ImportError:
    # Fallback implementations
    def derive_scoring_weight(dimension: int = 0) -> float:
        return (1 / PHI) ** max(0, dimension)

    def derive_learning_rate_sacred(iteration: int) -> float:
        return PHI / (iteration + 1)

    def derive_momentum_sacred(velocity: float) -> float:
        return (1 / PHI) * velocity

    def derive_threshold(entropy: float = 0.5, coherence: float = 0.5) -> float:
        return (1 / PHI) * (1 + entropy / 6539.34712682) * (1 + coherence * PHI)

    def derive_priority_score(urgency: float, importance: float) -> float:
        return urgency * PHI + importance / PHI

    def derive_activation_threshold(signal_strength: float = 1.0) -> float:
        return (1 / PHI) * signal_strength * (1 + PHI / 100)

    def derive_batch_epochs(data_size: int) -> int:
        return int(GOD_CODE / data_size * PHI)




# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

HBAR = 1.0545718e-34
_2PI = 2 * math.pi


# ═══════════════════════════════════════════════════════════════════════════════
# LIGHTWEIGHT HDC SUPPORT (self-contained, no external dependency)
# ═══════════════════════════════════════════════════════════════════════════════

class _Hypervector:
    """Minimal bipolar hypervector for cognitive reasoning."""
    __slots__ = ('data',)

    def __init__(self, data: List[int]):
        self.data = data


class _HypervectorFactory:
    """Generates deterministic seed vectors via hashing."""

    def __init__(self, dimension: int = 5000):
        self.dimension = dimension

    def seed_vector(self, seed_str: str) -> _Hypervector:
        rng = random.Random(hash(seed_str))
        data = [1 if rng.random() > 0.5 else -1 for _ in range(self.dimension)]
        return _Hypervector(data)


class _HDCAlgebra:
    """Bundle and similarity ops for lightweight HDC."""

    @staticmethod
    def bundle(vectors: List[_Hypervector]) -> _Hypervector:
        if not vectors:
            return _Hypervector([])
        dim = len(vectors[0].data)
        sums = [0] * dim
        for v in vectors:
            for i, x in enumerate(v.data):
                sums[i] += x
        return _Hypervector([1 if s >= 0 else -1 for s in sums])

    @staticmethod
    def similarity(a: _Hypervector, b: _Hypervector) -> float:
        if not a.data or not b.data:
            return 0.0
        dim = min(len(a.data), len(b.data))
        if dim == 0:
            return 0.0
        dot = sum(a.data[i] * b.data[i] for i in range(dim))
        return dot / dim


# ═══════════════════════════════════════════════════════════════════════════════
# INTELLIGENT REASONING FRAMEWORK
# ═══════════════════════════════════════════════════════════════════════════════

class ReasoningStrategy(Enum):
    """Available reasoning strategies"""
    BAYESIAN = auto()
    QUANTUM = auto()
    ANALOGICAL = auto()
    PATTERN = auto()
    EVOLUTIONARY = auto()
    ENSEMBLE = auto()
    CAUSAL = auto()
    COUNTERFACTUAL = auto()
    ABDUCTIVE = auto()
    CREATIVE = auto()
    TEMPORAL = auto()
    SYMBOLIC = auto()
    INTUITIVE = auto()
    SOCIAL = auto()
    DREAM = auto()


@dataclass
class Observation:
    """A recorded observation with metadata"""
    timestamp: float
    context: str
    data: Dict[str, Any]
    outcome: Optional[Any] = None
    confidence: float = 0.5
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'context': self.context,
            'data': self.data,
            'outcome': self.outcome,
            'confidence': self.confidence,
            'tags': self.tags
        }


@dataclass
class Hypothesis:
    """A hypothesis with supporting evidence"""
    statement: str
    prior_probability: float
    evidence_for: List[str] = field(default_factory=list)
    evidence_against: List[str] = field(default_factory=list)
    posterior_probability: Optional[float] = None
    quantum_amplitude: Optional[complex] = None

    def update_posterior(self, likelihood_ratio: float):
        if self.posterior_probability is None:
            self.posterior_probability = self.prior_probability
        odds = self.posterior_probability / (1 - self.posterior_probability + 1e-10)
        new_odds = odds * likelihood_ratio
        self.posterior_probability = new_odds / (1 + new_odds)
        self.quantum_amplitude = cmath.sqrt(self.posterior_probability)


class ContextualMemory:
    """
    Intelligent memory system with context-aware retrieval.
    Remembers patterns and learns from experience.
    """

    def __init__(self, max_size: int = 100000, decay_rate: float = 0.95):
        self.observations: deque = deque(maxlen=max_size)
        self.patterns: Dict[str, List[Dict]] = defaultdict(list)
        self.decay_rate = decay_rate
        self.importance_scores: Dict[int, float] = {}
        self._hdc_factory = _HypervectorFactory(5000)
        self._context_vectors: Dict[str, Any] = {}

    def store(self, observation: Observation) -> int:
        obs_id = len(self.observations)
        self.observations.append(observation)
        self.importance_scores[obs_id] = observation.confidence
        for tag in observation.tags:
            self.patterns[tag].append({
                'id': obs_id, 'data': observation.data, 'outcome': observation.outcome
            })
        context_key = f"{observation.context}_{obs_id}"
        self._context_vectors[context_key] = self._hdc_factory.seed_vector(
            observation.context + json.dumps(observation.data, default=str)[:100]
        )
        return obs_id

    def retrieve_similar(self, query_context: str, top_k: int = 5) -> List[Observation]:
        query_vec = self._hdc_factory.seed_vector(query_context)
        algebra = _HDCAlgebra()
        similarities = []
        for i, obs in enumerate(self.observations):
            context_key = f"{obs.context}_{i}"
            if context_key in self._context_vectors:
                sim = algebra.similarity(query_vec, self._context_vectors[context_key])
                recency = self.decay_rate ** (len(self.observations) - i - 1)
                importance = self.importance_scores.get(i, 0.5)
                score = sim * recency * importance
                similarities.append((i, obs, score))
        similarities.sort(key=lambda x: x[2], reverse=True)
        return [obs for _, obs, _ in similarities[:top_k]]

    def find_patterns(self, tag: str, min_occurrences: int = 3) -> Dict[str, Any]:
        if tag not in self.patterns or len(self.patterns[tag]) < min_occurrences:
            return {'found': False, 'reason': 'Insufficient data'}
        entries = self.patterns[tag]
        outcomes = [e['outcome'] for e in entries if e['outcome'] is not None]
        if not outcomes:
            return {'found': False, 'reason': 'No outcomes recorded'}
        if all(isinstance(o, (int, float)) for o in outcomes):
            mean_outcome = sum(outcomes) / max(len(outcomes), 1)
            variance = sum((o - mean_outcome)**2 for o in outcomes) / max(len(outcomes), 1)
            return {
                'found': True, 'tag': tag, 'count': len(outcomes),
                'mean': mean_outcome, 'variance': variance,
                'std': math.sqrt(variance), 'pattern_type': 'numeric'
            }
        else:
            counts = Counter(str(o) for o in outcomes)
            return {
                'found': True, 'tag': tag, 'count': len(outcomes),
                'distribution': dict(counts), 'mode': counts.most_common(1)[0][0],
                'pattern_type': 'categorical'
            }

    def decay_importance(self):
        for obs_id in self.importance_scores:
            self.importance_scores[obs_id] *= self.decay_rate


class QuantumInferenceEngine:
    """
    Bayesian reasoning with quantum amplitude encoding.
    Hypotheses exist in superposition until evidence collapses them.
    """

    def __init__(self):
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.evidence_log: list = []

    def add_hypothesis(self, name: str, statement: str, prior: float = 0.5) -> Hypothesis:
        h = Hypothesis(
            statement=statement, prior_probability=prior,
            quantum_amplitude=cmath.sqrt(prior)
        )
        self.hypotheses[name] = h
        return h

    def observe_evidence(self, evidence: str,
                        likelihood_if_true: Dict[str, float],
                        likelihood_if_false: Dict[str, float]) -> Dict[str, float]:
        posteriors = {}
        for name, hypothesis in self.hypotheses.items():
            p_e_given_h = likelihood_if_true.get(name, 0.5)
            p_e_given_not_h = likelihood_if_false.get(name, 0.5)
            lr = p_e_given_h / p_e_given_not_h if p_e_given_not_h > 0 else 100.0
            hypothesis.update_posterior(lr)
            hypothesis.evidence_for.append(f"{evidence} (LR={lr:.2f})")
            posteriors[name] = hypothesis.posterior_probability
        self.evidence_log.append((evidence, time.time()))
        return posteriors

    def get_superposition_state(self) -> Dict[str, Any]:
        amplitudes = {}
        total_prob = 0
        for name, h in self.hypotheses.items():
            prob = h.posterior_probability if h.posterior_probability else h.prior_probability
            total_prob += prob
            amplitudes[name] = h.quantum_amplitude
        if total_prob > 0:
            norm_factor = 1.0 / math.sqrt(total_prob)
            amplitudes = {k: v * norm_factor for k, v in amplitudes.items()}
        return {
            'hypotheses': {k: {
                'statement': v.statement,
                'probability': v.posterior_probability or v.prior_probability,
                'amplitude': v.quantum_amplitude
            } for k, v in self.hypotheses.items()},
            'normalized_amplitudes': amplitudes,
            'entropy': self._compute_entropy(),
            'evidence_count': len(self.evidence_log)
        }

    def collapse(self) -> Tuple[str, Hypothesis]:
        probs, names = [], []
        for name, h in self.hypotheses.items():
            probs.append(h.posterior_probability if h.posterior_probability else h.prior_probability)
            names.append(name)
        total = sum(probs)
        probs = [p / total for p in probs]
        r = random.random()
        cumulative = 0
        selected = names[0]
        for name, prob in zip(names, probs):
            cumulative += prob
            if r <= cumulative:
                selected = name
                break
        return selected, self.hypotheses[selected]

    def _compute_entropy(self) -> float:
        entropy = 0
        for h in self.hypotheses.values():
            p = h.posterior_probability if h.posterior_probability else h.prior_probability
            if 0 < p < 1:
                entropy -= p * math.log2(p)
        return entropy


class AdaptiveLearner:
    """
    Learns from experience and adapts strategies automatically.
    Uses reinforcement-like updates based on outcomes.

    EVO_72: Sacred Algorithm Integration — PHI-based learning rates
    and TAU-scaled momentum for optimal convergence.
    """

    def __init__(self, learning_rate: Optional[float] = None, iteration: int = 0):
        # EVO_72: Use sacred learning rate derivation if not provided
        self.learning_rate = learning_rate if learning_rate is not None else derive_learning_rate_sacred(iteration)
        self._iteration = iteration
        self.strategy_scores: Dict[str, float] = {s.name: 1.0 for s in ReasoningStrategy}
        self.action_history: List[Dict] = []
        self.parameter_history: Dict[str, List[float]] = defaultdict(list)
        # EVO_72: Sacred exploration rate using PHI proportion
        self._exploration_rate = PHI / (PHI + 1.0)  # ~0.618 (golden cut)
        # EVO_72: Dynamic batch sizing based on strategy count
        self._batch_epochs = derive_batch_epochs(len(ReasoningStrategy))

    def select_strategy(self, context: str) -> ReasoningStrategy:
        total_uses = sum(1 for a in self.action_history if 'strategy' in a)
        strategy_uses = defaultdict(int)
        for a in self.action_history:
            if 'strategy' in a:
                strategy_uses[a['strategy']] += 1
        best_score = -float('inf')
        best_strategy = ReasoningStrategy.ENSEMBLE
        for strategy in ReasoningStrategy:
            base_score = self.strategy_scores[strategy.name]
            uses = strategy_uses.get(strategy.name, 0)
            if uses > 0 and total_uses > 0:
                exploration_bonus = math.sqrt(2 * math.log(total_uses + 1) / uses)
            else:
                exploration_bonus = 2.0
            score = base_score + self._exploration_rate * exploration_bonus
            if score > best_score:
                best_score = score
                best_strategy = strategy
        return best_strategy

    def record_outcome(self, strategy: ReasoningStrategy, success: bool, reward: float = 1.0):
        self.action_history.append({
            'strategy': strategy.name, 'success': success,
            'reward': reward, 'timestamp': time.time()
        })
        old_score = self.strategy_scores[strategy.name]
        # EVO_72: Sacred scoring weight for dimension-aware updates
        sacred_weight = derive_scoring_weight(self._iteration)
        if success:
            # EVO_72: PHI-weighted reward adjustment
            new_score = old_score + self.learning_rate * (reward * PHI - old_score + sacred_weight)
        else:
            # EVO_72: TAU-damped penalty
            new_score = old_score - self.learning_rate * (PHI - reward) * TAU
        self.strategy_scores[strategy.name] = max(0.1, min(10.0, new_score))
        # EVO_72: Increment iteration and update learning rate
        self._iteration += 1
        self.learning_rate = derive_learning_rate_sacred(self._iteration)

    def adapt_parameter(self, param_name: str, current_value: float,
                       gradient: float, constraint: Tuple[float, float] = (0, 1)) -> float:
        history = self.parameter_history[param_name]
        # EVO_72: Sacred momentum using TAU-scaled velocity
        velocity = 0.0
        if len(history) >= 2:
            recent_deltas = [history[i] - history[i-1] for i in range(-1, -min(5, len(history)), -1)]
            if recent_deltas:
                avg_velocity = sum(recent_deltas) / max(len(recent_deltas), 1)
                velocity = derive_momentum_sacred(avg_velocity)
        # EVO_72: PHI-weighted gradient with sacred momentum
        new_value = current_value + self.learning_rate * gradient * PHI + velocity
        new_value = max(constraint[0], min(constraint[1], new_value))
        history.append(new_value)
        # EVO_72: Dynamic history size based on sacred batch epochs
        if len(history) > self._batch_epochs:
            history.pop(0)
        return new_value

    def get_learning_summary(self) -> Dict[str, Any]:
        success_rate = 0.0
        if self.action_history:
            successes = sum(1 for a in self.action_history if a.get('success', False))
            success_rate = successes / max(len(self.action_history), 1)
        return {
            'total_actions': len(self.action_history),
            'success_rate': success_rate,
            'strategy_scores': dict(self.strategy_scores),
            'best_strategy': max(self.strategy_scores, key=self.strategy_scores.get),
            'exploration_rate': self._exploration_rate,
            'parameters_tracked': len(self.parameter_history)
        }


class PatternRecognizer:
    """
    Recognizes patterns in quantum states and observations.
    Uses both statistical and HDC-based pattern matching.

    EVO_72: Sacred Algorithm Integration — PHI-based thresholds
    for optimal pattern detection sensitivity.
    """

    def __init__(self, dimension: int = 5000):
        self.dimension = dimension
        self.factory = _HypervectorFactory(dimension)
        self.algebra = _HDCAlgebra()
        self.known_patterns: Dict[str, Any] = {}
        self._pattern_vectors: Dict[str, _Hypervector] = {}
        # EVO_72: Sacred threshold for pattern recognition
        self._sacred_threshold = derive_activation_threshold(1.0)

    def learn_pattern(self, name: str, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not examples:
            return {'error': 'No examples provided'}
        feature_vectors = []
        # EVO_72: PHI-proportioned feature extraction
        max_features = int(200 * PHI / (PHI + 1.0))  # ~124 features
        for ex in examples:
            ex_str = json.dumps(ex, sort_keys=True, default=str)[:max_features]
            feature_vectors.append(self.factory.seed_vector(ex_str))
        pattern_hv = self.algebra.bundle(feature_vectors)
        self._pattern_vectors[name] = pattern_hv
        self.known_patterns[name] = {
            'num_examples': len(examples), 'learned_at': time.time(),
            'feature_count': len(examples[0]) if examples else 0
        }
        return {'pattern': name, 'examples_used': len(examples), 'status': 'learned'}

    def recognize(self, instance: Dict[str, Any], threshold: Optional[float] = None) -> List[Tuple[str, float]]:
        # EVO_72: Use sacred threshold if not provided
        if threshold is None:
            threshold = self._sacred_threshold
        instance_str = json.dumps(instance, sort_keys=True, default=str)[:200]
        instance_hv = self.factory.seed_vector(instance_str)
        matches = []
        for name, pattern_hv in self._pattern_vectors.items():
            similarity = self.algebra.similarity(instance_hv, pattern_hv)
            if similarity >= threshold:
                matches.append((name, similarity))
        return sorted(matches, key=lambda x: x[1], reverse=True)

    def find_anomalies(self, observations: List[Dict[str, Any]], threshold: Optional[float] = None) -> List[int]:
        # EVO_72: Use sacred threshold if not provided
        if threshold is None:
            threshold = derive_activation_threshold(0.5)
        return [i for i, obs in enumerate(observations) if not self.recognize(obs, threshold)]

    def detect_sequence_pattern(self, sequence: List[Any], max_period: Optional[int] = None) -> Dict[str, Any]:
        # EVO_72: Sacred batch sizing for sequence analysis
        if max_period is None:
            max_period = derive_batch_epochs(len(sequence))
        if len(sequence) < 3:
            return {'pattern_found': False, 'reason': 'Sequence too short'}
        for period in range(1, min(max_period, len(sequence) // 2) + 1):
            if all(sequence[i] == sequence[i % period] for i in range(period, len(sequence))):
                return {
                    'pattern_found': True, 'pattern_type': 'periodic',
                    'period': period, 'repeating_unit': sequence[:period]
                }
        if all(isinstance(x, (int, float)) for x in sequence):
            diffs = [sequence[i+1] - sequence[i] for i in range(len(sequence)-1)]
            if len(set(diffs)) == 1:
                return {
                    'pattern_found': True, 'pattern_type': 'arithmetic',
                    'common_difference': diffs[0], 'formula': f'a_n = {sequence[0]} + {diffs[0]} * n'
                }
        return {'pattern_found': False, 'reason': 'No simple pattern detected'}


class MetaCognition:
    """
    The system that reasons about its own reasoning.
    Monitors performance and adjusts cognitive strategies.
    """

    def __init__(self):
        self.reasoning_log: List[Dict] = []
        self.current_confidence: float = 0.5
        self.cognitive_load: float = 0.0
        self._uncertainty_threshold = 0.3
        self.performance_metrics: Dict[str, list] = {'confidence': []}

    def log_reasoning_step(self, step_type: str, input_data: Any,
                          output_data: Any, confidence: float):
        self.reasoning_log.append({
            'step_type': step_type, 'timestamp': time.time(),
            'input_summary': str(input_data)[:100],
            'output_summary': str(output_data)[:100],
            'confidence': confidence
        })
        self.performance_metrics['confidence'].append(confidence)
        self.current_confidence = confidence
        self.cognitive_load = 0.9 * self.cognitive_load + 0.1 * (1 - confidence)

    def should_reconsider(self) -> bool:
        if self.current_confidence < self._uncertainty_threshold:
            return True
        recent = self.performance_metrics['confidence'][-10:]
        if len(recent) >= 5:
            trend = (recent[-1] - recent[0]) / max(len(recent), 1)
            if trend < -0.1:
                return True
        return False

    def get_reasoning_quality(self) -> Dict[str, Any]:
        if not self.reasoning_log:
            return {'status': 'no_data', 'quality': 0.5}
        recent_confidences = self.performance_metrics['confidence'][-20:]
        return {
            'status': 'analyzed',
            'total_steps': len(self.reasoning_log),
            'mean_confidence': sum(recent_confidences) / max(len(recent_confidences), 1),
            'confidence_trend': self._compute_trend(recent_confidences),
            'cognitive_load': self.cognitive_load,
            'should_simplify': self.cognitive_load > 0.7,
            'should_reconsider': self.should_reconsider()
        }

    def suggest_improvement(self) -> str:
        quality = self.get_reasoning_quality()
        if quality['status'] == 'no_data':
            return "Gather more observations before reasoning"
        if quality['mean_confidence'] < 0.3:
            return "Confidence is low - seek more evidence or use ensemble strategies"
        if quality['confidence_trend'] < -0.05:
            return "Confidence is declining - reconsider assumptions or try different approach"
        if quality['cognitive_load'] > 0.7:
            return "Cognitive load is high - simplify the problem or break into sub-problems"
        if quality['mean_confidence'] > 0.8:
            return "Reasoning is strong - consider edge cases for robustness"
        return "Continue current approach - performance is adequate"

    def _compute_trend(self, values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / max(len(values), 1)
        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean)**2 for i in range(n))
        return numerator / max(denominator, 1e-10) if denominator != 0 else 0.0


class PredictiveReasoner:
    """
    Predicts future states based on quantum evolution and learned patterns.
    """

    def __init__(self):
        self.state_history: List[Dict[str, Any]] = []
        self.transition_matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def record_state(self, state_name: str, state_data: Dict[str, Any]):
        self.state_history.append({
            'name': state_name, 'data': state_data, 'timestamp': time.time()
        })
        if len(self.state_history) >= 2:
            prev_state = self.state_history[-2]['name']
            self.transition_matrix[prev_state][state_name] += 1

    def predict_next_state(self, current_state: str, steps: int = 1) -> List[Tuple[str, float]]:
        if current_state not in self.transition_matrix:
            return [('unknown', 1.0)]
        transitions = self.transition_matrix[current_state]
        total = sum(transitions.values())
        if total == 0:
            return [('unknown', 1.0)]
        predictions = [(state, count/total) for state, count in transitions.items()]
        predictions.sort(key=lambda x: x[1], reverse=True)
        if steps > 1:
            return self._markov_predict(current_state, steps)
        return predictions

    def _markov_predict(self, start_state: str, steps: int) -> List[Tuple[str, float]]:
        current_probs = {start_state: 1.0}
        for _ in range(steps):
            next_probs = defaultdict(float)
            for state, prob in current_probs.items():
                if state in self.transition_matrix:
                    total = sum(self.transition_matrix[state].values())
                    if total > 0:
                        for next_state, count in self.transition_matrix[state].items():
                            next_probs[next_state] += prob * (count / total)
            current_probs = dict(next_probs) if next_probs else current_probs
        return sorted(current_probs.items(), key=lambda x: x[1], reverse=True)[:50]

    def quantum_evolution(self, initial_state: Dict[str, complex],
                         hamiltonian_diag: List[float],
                         time_step: float = 0.1) -> Dict[str, complex]:
        evolved = {}
        for state_name, amplitude in initial_state.items():
            idx = hash(state_name) % len(hamiltonian_diag) if hamiltonian_diag else 0
            energy = hamiltonian_diag[idx] if hamiltonian_diag else GOD_CODE
            phase = -energy * time_step / HBAR
            evolution_factor = cmath.exp(complex(0, phase))
            evolved[state_name] = amplitude * evolution_factor
        return evolved
