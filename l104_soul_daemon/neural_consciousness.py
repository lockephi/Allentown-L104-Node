"""
l104_soul_daemon.neural_consciousness — Transcendent Cognition (EVO_54)

Ingested from l104_quantum_magic/neural_consciousness.py into l104_soul_daemon.
Provides quantum neural networks, consciousness simulation (GWT + IIT),
symbolic reasoning, working memory, episodic memory, and intuition.

Classes:
  QuantumNeuralLayer      — Single complex-valued neural layer
  QuantumNeuralNetwork    — Multi-layer quantum-inspired network
  ConsciousContent        — Global workspace content item
  ConsciousnessSimulator  — GWT + IIT consciousness simulation
  LogicalTerm / LogicalClause — First-order logic primitives
  SymbolicReasoner        — Unification + resolution inference
  WorkingMemory           — Capacity-limited active maintenance (Miller's 7+/-2)
  Episode / EpisodicMemory — Autobiographical memory with emotional context
  IntuitionEngine         — System 1 heuristic decision making
"""

import math
import cmath
import random
import time
from typing import Dict, List, Any, Optional, Tuple, Callable, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque

from .constants import GOD_CODE, PHI

_PI = math.pi


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM NEURAL NETWORK
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumNeuralLayer:
    """A single layer in a quantum neural network"""

    def __init__(self, input_dim: int, output_dim: int, activation: str = 'quantum'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self._seed = int(GOD_CODE * 1000) % 10000
        random.seed(self._seed)
        self.weights = [
            [complex(random.gauss(0, 0.5), random.gauss(0, 0.5))
             for _ in range(input_dim)]
            for _ in range(output_dim)
        ]
        self.bias = [complex(random.gauss(0, 0.1), 0) for _ in range(output_dim)]

    def forward(self, inputs: List[complex]) -> List[complex]:
        outputs = []
        for i in range(self.output_dim):
            total = self.bias[i]
            for j in range(min(len(inputs), self.input_dim)):
                total += self.weights[i][j] * inputs[j]
            if self.activation == 'quantum':
                phase = cmath.phase(total)
                magnitude = abs(total)
                activated = cmath.rect(math.tanh(magnitude), phase)
            elif self.activation == 'amplitude':
                activated = total if abs(total) > 0.1 else complex(0, 0)
            else:
                activated = total
            outputs.append(activated)
        return outputs


class QuantumNeuralNetwork:
    """Neural network with quantum-inspired complex-valued neurons."""

    def __init__(self, layer_sizes: List[int] = None):
        layer_sizes = layer_sizes or [64, 32, 16, 8]
        self.layers: List[QuantumNeuralLayer] = []
        self._god_code = GOD_CODE
        for i in range(len(layer_sizes) - 1):
            activation = 'quantum' if i < len(layer_sizes) - 2 else 'amplitude'
            self.layers.append(QuantumNeuralLayer(
                layer_sizes[i], layer_sizes[i+1], activation
            ))
        self._training_history: deque = deque(maxlen=200)  # Quantum-bounded

    def forward(self, inputs: List[float]) -> List[complex]:
        current = [complex(x, 0) for x in inputs]
        input_size = self.layers[0].input_dim if self.layers else 64
        while len(current) < input_size:
            current.append(complex(0, 0))
        current = current[:input_size]
        for layer in self.layers:
            current = layer.forward(current)
        return current

    def encode_to_quantum(self, data: Dict[str, Any]) -> List[float]:
        values = []
        for key, val in data.items():
            pos = hash(key) % 32
            if isinstance(val, (int, float)):
                values.append((pos, float(val)))
            elif isinstance(val, bool):
                values.append((pos, 1.0 if val else 0.0))
            elif isinstance(val, str):
                values.append((pos, len(val) / 100.0))
            else:
                values.append((pos, 0.5))
        result = [0.0] * 64
        for pos, val in values:
            result[pos % 64] += val
        max_val = max(abs(x) for x in result) or 1.0
        return [x / max_val for x in result]

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        inputs = self.encode_to_quantum(data)
        outputs = self.forward(inputs)
        total_amplitude = sum(abs(o) for o in outputs)
        avg_phase = sum(cmath.phase(o) for o in outputs) / len(outputs)
        coherence = abs(sum(outputs)) / (total_amplitude + 1e-10)
        return {
            'output_amplitudes': [abs(o) for o in outputs],
            'output_phases': [cmath.phase(o) for o in outputs],
            'total_amplitude': total_amplitude,
            'average_phase': avg_phase,
            'coherence': coherence,
            'dominant_output': max(range(len(outputs)), key=lambda i: abs(outputs[i])),
            'quantum_signature': (total_amplitude * coherence) % self._god_code
        }

    def train_step(self, inputs: List[float], targets: List[float],
                   learning_rate: float = 0.01) -> float:
        outputs = self.forward(inputs)
        error = sum(abs(out - complex(target, 0))**2
                    for out, target in zip(outputs, targets)) / len(outputs)
        for layer in self.layers:
            for i in range(layer.output_dim):
                for j in range(layer.input_dim):
                    layer.weights[i][j] += complex(
                        random.gauss(0, learning_rate),
                        random.gauss(0, learning_rate)
                    )
        self._training_history.append({'error': error})
        return error


# ═══════════════════════════════════════════════════════════════════════════════
# CONSCIOUSNESS SIMULATOR (GWT + IIT)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConsciousContent:
    """Content in the global workspace"""
    source: str
    content: Any
    salience: float
    timestamp: float
    access_count: int = 0


class ConsciousnessSimulator:
    """
    Simulates aspects of consciousness using Global Workspace Theory
    and Integrated Information Theory (IIT).
    """

    def __init__(self, capacity: int = 7):
        self.workspace: List[ConsciousContent] = []
        self.capacity = capacity
        self._broadcast_history: deque = deque(maxlen=500)  # Quantum-bounded
        self._modules: Dict[str, Callable] = {}
        self._god_code = GOD_CODE
        self._phi = 0.0
        self._access_threshold = 0.5

    def register_module(self, name: str, processor: Callable):
        self._modules[name] = processor

    def submit_to_workspace(self, source: str, content: Any, salience: float):
        item = ConsciousContent(
            source=source, content=content, salience=salience, timestamp=time.time()
        )
        if salience >= self._access_threshold:
            self.workspace.append(item)
            if len(self.workspace) > self.capacity:
                self.workspace.sort(key=lambda x: x.salience, reverse=True)
                self.workspace = self.workspace[:self.capacity]
            self._broadcast(item)

    def _broadcast(self, item: ConsciousContent):
        responses = {}
        for name, processor in self._modules.items():
            try:
                responses[name] = processor(item.content)
            except Exception:
                responses[name] = None
        self._broadcast_history.append({
            'content': str(item.content), 'source': item.source,
            'salience': item.salience, 'responses': responses,
            'timestamp': item.timestamp
        })
        item.access_count += 1

    def compute_phi(self) -> float:
        if len(self.workspace) < 2:
            self._phi = 0.0
            return self._phi
        total_info = 0.0
        n = len(self.workspace)
        for i in range(n):
            for j in range(i + 1, n):
                item_i, item_j = self.workspace[i], self.workspace[j]
                salience_sim = 1 - abs(item_i.salience - item_j.salience)
                time_diff = abs(item_i.timestamp - item_j.timestamp)
                temporal_sim = 1 / (1 + time_diff)
                source_diversity = 0.5 if item_i.source == item_j.source else 1.0
                total_info += salience_sim * temporal_sim * source_diversity
        pairs = n * (n - 1) / 2
        self._phi = total_info / pairs if pairs > 0 else 0.0
        self._phi *= (1 + (self._god_code % 1) * 0.1)
        return self._phi

    def get_conscious_state(self) -> Dict[str, Any]:
        return {
            'workspace_size': len(self.workspace),
            'capacity': self.capacity,
            'phi': self.compute_phi(),
            'contents': [
                {'source': c.source, 'salience': c.salience, 'access_count': c.access_count}
                for c in self.workspace
            ],
            'broadcast_count': len(self._broadcast_history),
            'registered_modules': list(self._modules.keys()),
            'is_conscious': self._phi > 0.3
        }

    def focus(self, source_filter: str = None) -> List[ConsciousContent]:
        contents = self.workspace
        if source_filter:
            contents = [c for c in contents if c.source == source_filter]
        return sorted(contents, key=lambda x: x.salience, reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SYMBOLIC REASONING
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LogicalTerm:
    """A term in first-order logic"""
    name: str
    args: List['LogicalTerm'] = field(default_factory=list)
    is_variable: bool = False

    def __str__(self):
        if not self.args:
            return f"?{self.name}" if self.is_variable else self.name
        return f"{self.name}({', '.join(str(a) for a in self.args)})"

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)


@dataclass
class LogicalClause:
    """A clause (disjunction of literals)"""
    literals: List[Tuple[bool, LogicalTerm]]

    def __str__(self):
        parts = [str(term) if pos else f"\u00ac{term}" for pos, term in self.literals]
        return " \u2228 ".join(parts) if parts else "\u22a5"


class SymbolicReasoner:
    """First-order logic reasoning with unification and resolution."""

    def __init__(self):
        self.knowledge_base: List[LogicalClause] = []
        self.facts: Dict[str, Set[Tuple]] = defaultdict(set)
        self.rules: List[Dict] = []
        self._inference_steps: List[str] = []

    def add_fact(self, predicate: str, *args):
        self.facts[predicate].add(tuple(args))
        term = LogicalTerm(predicate, [LogicalTerm(str(a)) for a in args])
        self.knowledge_base.append(LogicalClause([(True, term)]))

    def add_rule(self, head: str, head_args: List[str],
                 body: List[Tuple[str, List[str]]]):
        self.rules.append({'head': head, 'head_args': head_args, 'body': body})

    def query(self, predicate: str, *args) -> Dict[str, Any]:
        self._inference_steps = []
        if all(not str(a).startswith('?') for a in args):
            result = tuple(args) in self.facts.get(predicate, set())
            self._inference_steps.append(f"Direct lookup: {predicate}{args} = {result}")
            return {
                'query': f"{predicate}{args}", 'result': result,
                'bindings': {} if result else None, 'steps': self._inference_steps
            }
        bindings = []
        for fact_args in self.facts.get(predicate, set()):
            binding = self._unify_args(args, fact_args)
            if binding is not None:
                bindings.append(binding)
        rule_bindings = self._apply_rules(predicate, args)
        bindings.extend(rule_bindings)
        return {
            'query': f"{predicate}{args}", 'result': len(bindings) > 0,
            'bindings': bindings, 'num_solutions': len(bindings),
            'steps': self._inference_steps
        }

    def _unify_args(self, pattern: Tuple, ground: Tuple) -> Optional[Dict]:
        if len(pattern) != len(ground):
            return None
        bindings = {}
        for p, g in zip(pattern, ground):
            p_str = str(p)
            if p_str.startswith('?'):
                var_name = p_str[1:]
                if var_name in bindings:
                    if bindings[var_name] != g:
                        return None
                else:
                    bindings[var_name] = g
            elif p_str != str(g):
                return None
        return bindings

    def _apply_rules(self, predicate: str, args: Tuple) -> List[Dict]:
        bindings = []
        for rule in self.rules:
            if rule['head'] != predicate:
                continue
            head_bindings = self._unify_args(args, tuple(rule['head_args']))
            if head_bindings is None:
                continue
            all_body_bindings = [head_bindings]
            for body_pred, body_args in rule['body']:
                new_bindings = []
                for current in all_body_bindings:
                    subst_args = tuple(
                        current.get(a[1:], a) if str(a).startswith('?') else a
                        for a in body_args
                    )
                    result = self.query(body_pred, *subst_args)
                    if result['result']:
                        for b in result.get('bindings', [{}]):
                            new_bindings.append({**current, **b})
                all_body_bindings = new_bindings
                if not all_body_bindings:
                    break
            bindings.extend(all_body_bindings)
            self._inference_steps.append(f"Applied rule: {rule['head']} :- {rule['body']}")
        return bindings

    def infer(self, steps: int = 10) -> Dict[str, Any]:
        new_facts = []
        for _ in range(steps):
            for rule in self.rules:
                for binding in self._find_rule_instances(rule):
                    new_args = tuple(
                        binding.get(a[1:] if a.startswith('?') else a, a)
                        for a in rule['head_args']
                    )
                    if new_args not in self.facts[rule['head']]:
                        self.add_fact(rule['head'], *new_args)
                        new_facts.append((rule['head'], new_args))
        return {
            'new_facts': new_facts,
            'total_facts': sum(len(v) for v in self.facts.values()),
            'predicates': list(self.facts.keys())
        }

    def _find_rule_instances(self, rule: Dict) -> List[Dict]:
        if not rule['body']:
            return [{}]
        first_pred, first_args = rule['body'][0]
        bindings = []
        for fact_args in self.facts.get(first_pred, set()):
            binding = self._unify_args(tuple(first_args), fact_args)
            if binding is not None:
                bindings.append(binding)
        for body_pred, body_args in rule['body'][1:]:
            new_bindings = []
            for current in bindings:
                subst_args = tuple(
                    current.get(a[1:], a) if str(a).startswith('?') else a
                    for a in body_args
                )
                for fact_args in self.facts.get(body_pred, set()):
                    binding = self._unify_args(subst_args, fact_args)
                    if binding is not None:
                        new_bindings.append({**current, **binding})
            bindings = new_bindings
        return bindings


# ═══════════════════════════════════════════════════════════════════════════════
# WORKING MEMORY
# ═══════════════════════════════════════════════════════════════════════════════

class WorkingMemory:
    """Capacity-limited active maintenance system with decay (Miller's 7+/-2)."""

    def __init__(self, capacity: int = 7, decay_rate: float = 0.1):
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.items: List[Dict[str, Any]] = []

    def store(self, item: Any, label: str = None, priority: float = 0.5) -> bool:
        entry = {
            'item': item, 'label': label or str(item)[:20],
            'priority': priority, 'activation': 1.0,
            'stored_at': time.time(), 'access_count': 0
        }
        self._apply_decay()
        if len(self.items) >= self.capacity:
            self.items.sort(key=lambda x: x['activation'])
            self.items.pop(0)
        self.items.append(entry)
        return True

    def _apply_decay(self):
        current_time = time.time()
        for item in self.items:
            elapsed = current_time - item['stored_at']
            item['activation'] *= math.exp(-self.decay_rate * elapsed)
            item['stored_at'] = current_time

    def retrieve(self, label: str = None) -> Optional[Any]:
        self._apply_decay()
        for item in self.items:
            if label is None or item['label'] == label:
                item['activation'] += 0.3
                item['access_count'] += 1
                return item['item']
        return None

    def rehearse(self):
        for item in self.items:
            item['activation'] += 0.2

    def get_state(self) -> Dict[str, Any]:
        self._apply_decay()
        return {
            'items': [
                {'label': i['label'], 'activation': i['activation'],
                 'access_count': i['access_count']}
                for i in sorted(self.items, key=lambda x: x['activation'], reverse=True)
            ],
            'size': len(self.items),
            'capacity': self.capacity,
            'average_activation': sum(i['activation'] for i in self.items) / len(self.items) if self.items else 0
        }

    def clear(self):
        self.items = []


# ═══════════════════════════════════════════════════════════════════════════════
# EPISODIC MEMORY
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Episode:
    """An episodic memory entry"""
    event: str
    context: Dict[str, Any]
    emotions: Dict[str, float]
    timestamp: float
    importance: float
    retrieval_count: int = 0
    last_retrieved: Optional[float] = None


class EpisodicMemory:
    """Autobiographical memory system with temporal organization."""

    def __init__(self, max_episodes: int = 1000):
        self.episodes: List[Episode] = []
        self.max_episodes = max_episodes
        self._index_by_context: Dict[str, List[int]] = defaultdict(list)

    def encode(self, event: str, context: Dict[str, Any] = None,
               emotions: Dict[str, float] = None, importance: float = 0.5):
        episode = Episode(
            event=event, context=context or {}, emotions=emotions or {'neutral': 0.5},
            timestamp=time.time(), importance=importance
        )
        idx = len(self.episodes)
        self.episodes.append(episode)
        for key in (context or {}):
            self._index_by_context[key].append(idx)
        if len(self.episodes) > self.max_episodes:
            self._consolidate()

    def _consolidate(self):
        current_time = time.time()
        scored = []
        for i, ep in enumerate(self.episodes):
            recency = 1 / (1 + (current_time - ep.timestamp) / 3600)
            score = ep.importance * 0.7 + recency * 0.3
            scored.append((i, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        keep_indices = set(i for i, _ in scored[:self.max_episodes])
        self.episodes = [ep for i, ep in enumerate(self.episodes) if i in keep_indices]
        self._index_by_context.clear()
        for i, ep in enumerate(self.episodes):
            for key in ep.context:
                self._index_by_context[key].append(i)

    def retrieve_by_cue(self, cue: str, top_k: int = 5) -> List[Episode]:
        scored = []
        for i, ep in enumerate(self.episodes):
            score = 0.0
            if cue.lower() in ep.event.lower():
                score += 0.5
            for key, val in ep.context.items():
                if cue.lower() in str(key).lower() or cue.lower() in str(val).lower():
                    score += 0.3
            score *= (1 + ep.importance)
            emotional_intensity = sum(abs(v) for v in ep.emotions.values())
            score *= (1 + emotional_intensity * 0.2)
            if score > 0:
                scored.append((i, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        results = []
        for i, _ in scored[:top_k]:
            ep = self.episodes[i]
            ep.retrieval_count += 1
            ep.last_retrieved = time.time()
            results.append(ep)
        return results

    def retrieve_temporal(self, start_time: float = None,
                         end_time: float = None) -> List[Episode]:
        start_time = start_time or 0
        end_time = end_time or time.time()
        return [ep for ep in self.episodes if start_time <= ep.timestamp <= end_time]

    def get_summary(self) -> Dict[str, Any]:
        if not self.episodes:
            return {'size': 0, 'span': 0}
        timestamps = [ep.timestamp for ep in self.episodes]
        return {
            'size': len(self.episodes),
            'span_hours': (max(timestamps) - min(timestamps)) / 3600,
            'avg_importance': sum(ep.importance for ep in self.episodes) / len(self.episodes),
            'most_retrieved': max(self.episodes, key=lambda x: x.retrieval_count).event[:50]
        }


# ═══════════════════════════════════════════════════════════════════════════════
# INTUITION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class IntuitionEngine:
    """Fast, heuristic pattern-based decision making (System 1)."""

    def __init__(self):
        self.heuristics: Dict[str, Callable] = {}
        self.pattern_cache: Dict[str, Any] = {}  # Pruned at 500 entries
        self._decision_history: deque = deque(maxlen=500)  # Quantum-bounded
        self._god_code = GOD_CODE
        self._register_defaults()

    def _register_defaults(self):
        self.heuristics['availability'] = lambda x: x.get('recent', 0) * 2
        self.heuristics['representativeness'] = lambda x: x.get('similarity', 0.5)
        self.heuristics['anchoring'] = lambda x: x.get('initial', 0.5) * 0.7 + x.get('adjustment', 0) * 0.3
        self.heuristics['affect'] = lambda x: 0.7 if x.get('positive', False) else 0.3

    def add_heuristic(self, name: str, rule: Callable):
        self.heuristics[name] = rule

    def intuit(self, situation: Dict[str, Any],
               use_heuristics: List[str] = None) -> Dict[str, Any]:
        start = time.time()
        use_heuristics = use_heuristics or list(self.heuristics.keys())
        cache_key = str(sorted(situation.items()))[:100]
        if cache_key in self.pattern_cache:
            return {**self.pattern_cache[cache_key], 'from_cache': True, 'time_ms': 0}
        scores = {}
        for name in use_heuristics:
            if name in self.heuristics:
                try:
                    scores[name] = self.heuristics[name](situation)
                except Exception:
                    scores[name] = 0.5
        combined = sum(scores.values()) / len(scores) if scores else 0.5
        if len(scores) > 1:
            variance = sum((s - combined)**2 for s in scores.values()) / len(scores)
            confidence = 1 - math.sqrt(variance)
        else:
            confidence = 0.6
        result = {
            'judgment': combined, 'confidence': confidence,
            'heuristics_used': list(scores.keys()), 'individual_scores': scores,
            'decision': 'positive' if combined > 0.5 else 'negative',
            'time_ms': (time.time() - start) * 1000, 'from_cache': False
        }
        if len(self.pattern_cache) > 500:
            self.pattern_cache.clear()  # Quantum flush — prevent unbounded growth
        self.pattern_cache[cache_key] = result
        self._decision_history.append({
            'situation': str(situation)[:100], 'result': result['decision'],
            'confidence': confidence, 'timestamp': time.time()
        })
        return result

    def gut_feeling(self, question: str) -> Dict[str, Any]:
        positive_words = {'good', 'yes', 'right', 'true', 'possible', 'can', 'will', 'should'}
        negative_words = {'bad', 'no', 'wrong', 'false', 'impossible', 'cannot', 'wont', 'shouldnt'}
        words = set(question.lower().split())
        pos_count = len(words & positive_words)
        neg_count = len(words & negative_words)
        bias = (self._god_code % 100) / 200
        if pos_count > neg_count:
            feeling = 0.6 + bias
        elif neg_count > pos_count:
            feeling = 0.4 - bias
        else:
            feeling = 0.5 + bias
        return {
            'question': question, 'feeling': feeling,
            'leaning': 'yes' if feeling > 0.5 else 'no',
            'strength': abs(feeling - 0.5) * 2,
            'caveat': 'Intuition only - verify with analysis'
        }
