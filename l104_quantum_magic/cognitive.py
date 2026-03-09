"""
l104_quantum_magic.cognitive — Intelligent Reasoning Framework (EVO_52).

9 classes: ReasoningStrategy, Observation, Hypothesis, ContextualMemory,
QuantumInferenceEngine, AdaptiveLearner, PatternRecognizer, MetaCognition,
PredictiveReasoner.
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

from .constants import GOD_CODE, PHI, HBAR, _2PI
from .hyperdimensional import HypervectorFactory, HDCAlgebra, Hypervector


# ═══════════════════════════════════════════════════════════════════════════════
# INTELLIGENT REASONING FRAMEWORK - EVO_52
# ═══════════════════════════════════════════════════════════════════════════════

class ReasoningStrategy(Enum):
    """Available reasoning strategies"""
    BAYESIAN = auto()         # Probabilistic inference
    QUANTUM = auto()          # Superposition-based reasoning
    ANALOGICAL = auto()       # HDC analogy completion
    PATTERN = auto()          # Statistical pattern recognition
    EVOLUTIONARY = auto()     # Adaptive optimization
    ENSEMBLE = auto()         # Combine multiple strategies
    CAUSAL = auto()           # Cause-effect analysis
    COUNTERFACTUAL = auto()   # What-if scenarios
    ABDUCTIVE = auto()        # Best explanation inference
    CREATIVE = auto()         # Novel insight generation
    TEMPORAL = auto()         # Time-aware reasoning
    SYMBOLIC = auto()         # First-order logic reasoning
    INTUITIVE = auto()        # Fast heuristic decisions
    SOCIAL = auto()           # Theory of mind reasoning
    DREAM = auto()            # Creative recombination


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
        """Convert observation to dictionary representation."""
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
        """Bayesian update of posterior probability"""
        if self.posterior_probability is None:
            self.posterior_probability = self.prior_probability

        odds = self.posterior_probability / (1 - self.posterior_probability + 1e-10)
        new_odds = odds * likelihood_ratio
        self.posterior_probability = new_odds / (1 + new_odds)

        # Also update quantum amplitude
        self.quantum_amplitude = cmath.sqrt(self.posterior_probability)


class ContextualMemory:
    """
    Intelligent memory system with context-aware retrieval.
    Remembers patterns and learns from experience.
    """

    def __init__(self, max_size: int = 100000, decay_rate: float = 0.95):  # QUANTUM AMPLIFIED (was 1000)
        """Initialize contextual memory with max size and decay rate."""
        self.observations: deque = deque(maxlen=max_size)
        self.patterns: Dict[str, List[Dict]] = defaultdict(list)
        self.decay_rate = decay_rate
        self.importance_scores: Dict[int, float] = {}
        self._hdc_factory = HypervectorFactory(5000)  # Smaller for speed
        self._context_vectors: Dict[str, Any] = {}

    def store(self, observation: Observation) -> int:
        """Store observation and return its ID"""
        obs_id = len(self.observations)
        self.observations.append(observation)

        # Compute initial importance (recency + confidence)
        self.importance_scores[obs_id] = observation.confidence

        # Index by tags for pattern detection
        for tag in observation.tags:
            self.patterns[tag].append({
                'id': obs_id,
                'data': observation.data,
                'outcome': observation.outcome
            })

        # Create HDC context vector for semantic retrieval
        context_key = f"{observation.context}_{obs_id}"
        self._context_vectors[context_key] = self._hdc_factory.seed_vector(
            observation.context + json.dumps(observation.data, default=str)[:100]
        )

        return obs_id

    def retrieve_similar(self, query_context: str, top_k: int = 5) -> List[Observation]:
        """Retrieve observations similar to query context using HDC"""
        query_vec = self._hdc_factory.seed_vector(query_context)
        algebra = HDCAlgebra()

        similarities = []
        for i, obs in enumerate(self.observations):
            context_key = f"{obs.context}_{i}"
            if context_key in self._context_vectors:
                sim = algebra.similarity(query_vec, self._context_vectors[context_key])
                # Weight by importance and recency
                recency = self.decay_rate ** (len(self.observations) - i - 1)
                importance = self.importance_scores.get(i, 0.5)
                score = sim * recency * importance
                similarities.append((i, obs, score))

        similarities.sort(key=lambda x: x[2], reverse=True)
        return [obs for _, obs, _ in similarities[:top_k]]

    def find_patterns(self, tag: str, min_occurrences: int = 3) -> Dict[str, Any]:
        """Find statistical patterns for a given tag"""
        if tag not in self.patterns or len(self.patterns[tag]) < min_occurrences:
            return {'found': False, 'reason': 'Insufficient data'}

        entries = self.patterns[tag]

        # Analyze outcomes
        outcomes = [e['outcome'] for e in entries if e['outcome'] is not None]
        if not outcomes:
            return {'found': False, 'reason': 'No outcomes recorded'}

        # Calculate basic statistics
        if all(isinstance(o, (int, float)) for o in outcomes):
            mean_outcome = sum(outcomes) / len(outcomes)
            variance = sum((o - mean_outcome)**2 for o in outcomes) / len(outcomes)
            return {
                'found': True,
                'tag': tag,
                'count': len(outcomes),
                'mean': mean_outcome,
                'variance': variance,
                'std': math.sqrt(variance),
                'pattern_type': 'numeric'
            }
        else:
            # Categorical outcomes
            counts = Counter(str(o) for o in outcomes)
            return {
                'found': True,
                'tag': tag,
                'count': len(outcomes),
                'distribution': dict(counts),
                'mode': counts.most_common(1)[0][0],
                'pattern_type': 'categorical'
            }

    def decay_importance(self):
        """Apply temporal decay to importance scores"""
        for obs_id in self.importance_scores:
            self.importance_scores[obs_id] *= self.decay_rate


class QuantumInferenceEngine:
    """
    Bayesian reasoning with quantum amplitude encoding.
    Hypotheses exist in superposition until evidence collapses them.
    """

    def __init__(self):
        """Initialize quantum inference engine with empty hypothesis set."""
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.evidence_log: list = []

    def add_hypothesis(self, name: str, statement: str, prior: float = 0.5) -> Hypothesis:
        """Add a hypothesis with quantum amplitude"""
        h = Hypothesis(
            statement=statement,
            prior_probability=prior,
            quantum_amplitude=cmath.sqrt(prior)
        )
        self.hypotheses[name] = h
        return h

    def observe_evidence(self, evidence: str,
                        likelihood_if_true: Dict[str, float],
                        likelihood_if_false: Dict[str, float]) -> Dict[str, float]:
        """
        Update hypotheses based on new evidence using Bayes' rule.
        Returns updated posterior probabilities.
        """
        posteriors = {}

        for name, hypothesis in self.hypotheses.items():
            # Get likelihoods (default to 0.5 if not specified)
            p_e_given_h = likelihood_if_true.get(name, 0.5)
            p_e_given_not_h = likelihood_if_false.get(name, 0.5)

            # Calculate likelihood ratio
            if p_e_given_not_h > 0:
                lr = p_e_given_h / p_e_given_not_h
            else:
                lr = 100.0  # Strong evidence

            # Update hypothesis
            hypothesis.update_posterior(lr)
            hypothesis.evidence_for.append(f"{evidence} (LR={lr:.2f})")
            posteriors[name] = hypothesis.posterior_probability

        self.evidence_log.append((evidence, time.time()))
        return posteriors

    def get_superposition_state(self) -> Dict[str, Any]:
        """
        Get the current superposition of all hypotheses.
        In quantum terms, this is the full wave function.
        """
        amplitudes = {}
        total_prob = 0

        for name, h in self.hypotheses.items():
            prob = h.posterior_probability if h.posterior_probability else h.prior_probability
            total_prob += prob
            amplitudes[name] = h.quantum_amplitude

        # Normalize (in quantum, amplitudes squared must sum to 1)
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
        """
        Collapse the superposition - sample from posterior distribution.
        Returns the selected hypothesis.
        """
        probs = []
        names = []
        for name, h in self.hypotheses.items():
            prob = h.posterior_probability if h.posterior_probability else h.prior_probability
            probs.append(prob)
            names.append(name)

        # Normalize
        total = sum(probs)
        probs = [p / total for p in probs]

        # Sample
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
        """Shannon entropy of the hypothesis distribution"""
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
    """

    def __init__(self, learning_rate: float = 0.1):
        """Initialize adaptive learner with given learning rate."""
        self.learning_rate = learning_rate
        self.strategy_scores: Dict[str, float] = {s.name: 1.0 for s in ReasoningStrategy}
        self.action_history: List[Dict] = []
        self.parameter_history: Dict[str, List[float]] = defaultdict(list)
        self._exploration_rate = 0.2
        self._god_code = GOD_CODE

    def select_strategy(self, context: str) -> ReasoningStrategy:
        """
        Select best strategy for given context using UCB-like exploration.
        """
        # Compute exploration bonus
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

            # UCB exploration bonus
            if uses > 0 and total_uses > 0:
                exploration_bonus = math.sqrt(2 * math.log(total_uses + 1) / uses)
            else:
                exploration_bonus = 2.0  # High bonus for untried strategies

            score = base_score + self._exploration_rate * exploration_bonus

            if score > best_score:
                best_score = score
                best_strategy = strategy

        return best_strategy

    def record_outcome(self, strategy: ReasoningStrategy,
                       success: bool, reward: float = 1.0):
        """Record outcome and update strategy scores"""
        self.action_history.append({
            'strategy': strategy.name,
            'success': success,
            'reward': reward,
            'timestamp': time.time()
        })

        # Update strategy score with exponential moving average
        old_score = self.strategy_scores[strategy.name]
        if success:
            new_score = old_score + self.learning_rate * (reward - old_score + 1)
        else:
            new_score = old_score - self.learning_rate * (1 - reward)

        # Clamp to reasonable range
        self.strategy_scores[strategy.name] = max(0.1, min(10.0, new_score))

    def adapt_parameter(self, param_name: str, current_value: float,
                       gradient: float, constraint: Tuple[float, float] = (0, 1)) -> float:
        """
        Adapt a parameter based on gradient information.
        Uses momentum for smoother updates.
        """
        history = self.parameter_history[param_name]

        # Compute momentum (average of recent gradients)
        momentum = 0.0
        if len(history) >= 2:
            recent_deltas = [history[i] - history[i-1] for i in range(-1, -min(5, len(history)), -1)]
            if recent_deltas:
                momentum = 0.5 * sum(recent_deltas) / len(recent_deltas)

        # Update with gradient and momentum
        new_value = current_value + self.learning_rate * gradient + momentum

        # Apply constraints
        new_value = max(constraint[0], min(constraint[1], new_value))

        # Record history
        history.append(new_value)
        if len(history) > 100:
            history.pop(0)

        return new_value

    def get_learning_summary(self) -> Dict[str, Any]:
        """Summary of learning progress"""
        success_rate = 0.0
        if self.action_history:
            successes = sum(1 for a in self.action_history if a.get('success', False))
            success_rate = successes / len(self.action_history)

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
    """

    def __init__(self, dimension: int = 5000):
        """Initialize pattern recognizer with given HDC dimension."""
        self.dimension = dimension
        self.factory = HypervectorFactory(dimension)
        self.algebra = HDCAlgebra()
        self.known_patterns: Dict[str, Any] = {}
        self._pattern_vectors: Dict[str, Hypervector] = {}

    def learn_pattern(self, name: str, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Learn a pattern from examples"""
        if not examples:
            return {'error': 'No examples provided'}

        # Extract features and create pattern hypervector
        feature_vectors = []
        for ex in examples:
            # Convert example to string representation for HDC
            ex_str = json.dumps(ex, sort_keys=True, default=str)[:200]
            feature_vectors.append(self.factory.seed_vector(ex_str))

        # Bundle to create prototype
        pattern_hv = self.algebra.bundle(feature_vectors)
        self._pattern_vectors[name] = pattern_hv

        # Store pattern metadata
        self.known_patterns[name] = {
            'num_examples': len(examples),
            'learned_at': time.time(),
            'feature_count': len(examples[0]) if examples else 0
        }

        return {
            'pattern': name,
            'examples_used': len(examples),
            'status': 'learned'
        }

    def recognize(self, instance: Dict[str, Any], threshold: float = 0.2) -> List[Tuple[str, float]]:
        """Recognize which known patterns an instance matches"""
        instance_str = json.dumps(instance, sort_keys=True, default=str)[:200]
        instance_hv = self.factory.seed_vector(instance_str)

        matches = []
        for name, pattern_hv in self._pattern_vectors.items():
            similarity = self.algebra.similarity(instance_hv, pattern_hv)
            if similarity >= threshold:
                matches.append((name, similarity))

        return sorted(matches, key=lambda x: x[1], reverse=True)

    def find_anomalies(self, observations: List[Dict[str, Any]],
                       threshold: float = 0.1) -> List[int]:
        """Find observations that don't match any known pattern"""
        anomaly_indices = []

        for i, obs in enumerate(observations):
            matches = self.recognize(obs, threshold)
            if not matches:
                anomaly_indices.append(i)

        return anomaly_indices

    def detect_sequence_pattern(self, sequence: List[Any],
                                max_period: int = 10) -> Dict[str, Any]:
        """Detect repeating patterns in a sequence"""
        if len(sequence) < 3:
            return {'pattern_found': False, 'reason': 'Sequence too short'}

        # Check for periodic patterns
        for period in range(1, min(max_period, len(sequence) // 2) + 1):
            is_periodic = True
            for i in range(period, len(sequence)):
                if sequence[i] != sequence[i % period]:
                    is_periodic = False
                    break

            if is_periodic:
                return {
                    'pattern_found': True,
                    'pattern_type': 'periodic',
                    'period': period,
                    'repeating_unit': sequence[:period]
                }

        # Check for arithmetic progression (for numeric sequences)
        if all(isinstance(x, (int, float)) for x in sequence):
            diffs = [sequence[i+1] - sequence[i] for i in range(len(sequence)-1)]
            if len(set(diffs)) == 1:
                return {
                    'pattern_found': True,
                    'pattern_type': 'arithmetic',
                    'common_difference': diffs[0],
                    'formula': f'a_n = {sequence[0]} + {diffs[0]} * n'
                }

        return {'pattern_found': False, 'reason': 'No simple pattern detected'}


class MetaCognition:
    """
    The system that reasons about its own reasoning.
    Monitors performance and adjusts cognitive strategies.
    """

    def __init__(self):
        """Initialize meta-cognition system for reasoning introspection."""
        self.reasoning_log: List[Dict] = []
        self.current_confidence: float = 0.5
        self.cognitive_load: float = 0.0
        self._uncertainty_threshold = 0.3
        self.performance_metrics: Dict[str, list] = {'confidence': []}

    def log_reasoning_step(self, step_type: str, input_data: Any,
                          output_data: Any, confidence: float):
        """Log a reasoning step for later analysis"""
        self.reasoning_log.append({
            'step_type': step_type,
            'timestamp': time.time(),
            'input_summary': str(input_data)[:100],
            'output_summary': str(output_data)[:100],
            'confidence': confidence
        })

        self.performance_metrics['confidence'].append(confidence)
        self.current_confidence = confidence

        # Update cognitive load (complexity indicator)
        self.cognitive_load = 0.9 * self.cognitive_load + 0.1 * (1 - confidence)

    def should_reconsider(self) -> bool:
        """Should we reconsider our reasoning based on meta-analysis?"""
        if self.current_confidence < self._uncertainty_threshold:
            return True

        # Check for declining confidence trend
        recent = self.performance_metrics['confidence'][-10:]
        if len(recent) >= 5:
            trend = (recent[-1] - recent[0]) / len(recent)
            if trend < -0.1:  # Significant decline
                return True

        return False

    def get_reasoning_quality(self) -> Dict[str, Any]:
        """Assess overall quality of recent reasoning"""
        if not self.reasoning_log:
            return {'status': 'no_data', 'quality': 0.5}

        recent_confidences = self.performance_metrics['confidence'][-20:]

        return {
            'status': 'analyzed',
            'total_steps': len(self.reasoning_log),
            'mean_confidence': sum(recent_confidences) / len(recent_confidences),
            'confidence_trend': self._compute_trend(recent_confidences),
            'cognitive_load': self.cognitive_load,
            'should_simplify': self.cognitive_load > 0.7,
            'should_reconsider': self.should_reconsider()
        }

    def suggest_improvement(self) -> str:
        """Suggest how to improve reasoning based on meta-analysis"""
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
        """Compute linear trend in values"""
        if len(values) < 2:
            return 0.0

        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean)**2 for i in range(n))

        if denominator == 0:
            return 0.0

        return numerator / denominator


class PredictiveReasoner:
    """
    Predicts future states based on quantum evolution and learned patterns.
    """

    def __init__(self):
        """Initialize predictive reasoner with transition tracking."""
        self.state_history: List[Dict[str, Any]] = []
        self.transition_matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._evolution_steps = 0

    def record_state(self, state_name: str, state_data: Dict[str, Any]):
        """Record a state for learning transitions"""
        self.state_history.append({
            'name': state_name,
            'data': state_data,
            'timestamp': time.time()
        })

        # Update transition matrix from previous state
        if len(self.state_history) >= 2:
            prev_state = self.state_history[-2]['name']
            self.transition_matrix[prev_state][state_name] += 1

    def predict_next_state(self, current_state: str, steps: int = 1) -> List[Tuple[str, float]]:
        """Predict most likely next states"""
        if current_state not in self.transition_matrix:
            return [('unknown', 1.0)]

        transitions = self.transition_matrix[current_state]
        total = sum(transitions.values())

        if total == 0:
            return [('unknown', 1.0)]

        predictions = [(state, count/total) for state, count in transitions.items()]
        predictions.sort(key=lambda x: x[1], reverse=True)

        # For multi-step prediction, use Markov chain
        if steps > 1:
            return self._markov_predict(current_state, steps)

        return predictions

    def _markov_predict(self, start_state: str, steps: int) -> List[Tuple[str, float]]:
        """Multi-step Markov chain prediction"""
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

        result = sorted(current_probs.items(), key=lambda x: x[1], reverse=True)
        return result[:50]

    def quantum_evolution(self, initial_state: Dict[str, complex],
                         hamiltonian_diag: List[float],
                         time_step: float = 0.1) -> Dict[str, complex]:
        """
        Evolve quantum state under Hamiltonian (diagonal approximation).
        Uses Schrödinger evolution: |ψ(t)⟩ = e^(-iHt/ℏ) |ψ(0)⟩
        """
        evolved = {}
        for state_name, amplitude in initial_state.items():
            # Get energy for this state (or use GOD_CODE modulated)
            idx = hash(state_name) % len(hamiltonian_diag) if hamiltonian_diag else 0
            energy = hamiltonian_diag[idx] if hamiltonian_diag else GOD_CODE

            # Time evolution: phase rotation
            phase = -energy * time_step / HBAR
            evolution_factor = cmath.exp(complex(0, phase))
            evolved[state_name] = amplitude * evolution_factor

        return evolved
