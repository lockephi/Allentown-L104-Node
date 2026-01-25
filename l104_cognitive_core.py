"""
L104 COGNITIVE CORE - Multi-Modal Reasoning System
============================================================
A layered cognitive architecture that integrates multiple reasoning
modalities: logical, intuitive, analogical, and emergent.

"The mind is not a single process but a symphony of many" - Cognitive Unity
"""

import math
import random
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional, Tuple, Set
from enum import Enum, auto
from datetime import datetime
from collections import defaultdict
import heapq

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


# Sacred Constants
GOD_CODE = 527.5184818492537
PHI = 1.618033988749895
OMEGA = 1381.0613


class ReasoningMode(Enum):
    """Modes of cognitive processing."""
    LOGICAL = auto()      # Deductive, rule-based
    INTUITIVE = auto()    # Pattern-matching, gestalt
    ANALOGICAL = auto()   # Mapping between domains
    CAUSAL = auto()       # Cause-effect chains
    TEMPORAL = auto()     # Time-based reasoning
    SPATIAL = auto()      # Structure and relationship
    EMERGENT = auto()     # Bottom-up pattern recognition
    METACOGNITIVE = auto()  # Thinking about thinking


@dataclass
class Concept:
    """A cognitive concept with multi-modal representations."""
    name: str
    category: str
    features: Dict[str, Any] = field(default_factory=dict)
    relations: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    activation: float = 0.0
    
    def activate(self, strength: float = 1.0):
        self.activation = min(1.0, self.activation + strength)
    
    def decay(self, rate: float = 0.1):
        self.activation = max(0.0, self.activation - rate)


@dataclass
class Inference:
    """A cognitive inference/conclusion."""
    proposition: str
    confidence: float
    mode: ReasoningMode
    premises: List[str]
    explanation: str


class WorkingMemory:
    """Limited-capacity active memory."""
    
    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self.items: List[Any] = []
        self.focus: Optional[Any] = None
    
    def add(self, item: Any) -> bool:
        if len(self.items) >= self.capacity:
            self.items.pop(0)
        self.items.append(item)
        return True
    
    def set_focus(self, item: Any):
        if item not in self.items:
            self.add(item)
        self.focus = item


class SemanticMemory:
    """Long-term memory for concepts."""
    
    def __init__(self):
        self.concepts: Dict[str, Concept] = {}
        self.associations: Dict[Tuple[str, str], float] = {}
    
    def add_concept(self, concept: Concept):
        self.concepts[concept.name] = concept
    
    def get_concept(self, name: str) -> Optional[Concept]:
        return self.concepts.get(name)
    
    def add_relation(self, c1: str, rel: str, c2: str):
        if c1 in self.concepts:
            self.concepts[c1].relations[rel].add(c2)
    
    def spreading_activation(self, start: str, depth: int = 3) -> Dict[str, float]:
        activations = {start: 1.0}
        frontier = [(start, 1.0)]
        
        for _ in range(depth):
            new_frontier = []
            for concept, activation in frontier:
                if concept in self.concepts:
                    for rel_type, targets in self.concepts[concept].relations.items():
                        for target in targets:
                            new_act = activation / PHI
                            if target not in activations or activations[target] < new_act:
                                activations[target] = new_act
                                new_frontier.append((target, new_act))
            frontier = new_frontier
        
        return activations


class EpisodicMemory:
    """Memory for experiences."""
    
    def __init__(self, capacity: int = 1000):
        self.episodes: List[Dict[str, Any]] = []
        self.capacity = capacity
    
    def store(self, episode: Dict[str, Any]):
        episode['timestamp'] = datetime.now()
        if len(self.episodes) >= self.capacity:
            self.episodes = self.episodes[int(self.capacity / PHI):]
        self.episodes.append(episode)
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        words = set(query.lower().split())
        scored = []
        for ep in self.episodes:
            score = sum(1 for w in words if any(w in str(v).lower() for v in ep.values()))
            scored.append((score, ep))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:k]]


class ReasoningEngine:
    """Multi-modal reasoning engine."""
    
    def __init__(self, semantic: SemanticMemory):
        self.semantic = semantic
        self.inferences: List[Inference] = []
    
    def reason(self, query: str, mode: ReasoningMode = None) -> List[Inference]:
        results = []
        words = query.lower().split()
        
        if mode == ReasoningMode.LOGICAL or mode is None:
            results.extend(self._logical(words))
        if mode == ReasoningMode.ANALOGICAL or mode is None:
            results.extend(self._analogical(words))
        if mode == ReasoningMode.INTUITIVE or mode is None:
            results.extend(self._intuitive(words))
        
        results.sort(key=lambda x: x.confidence, reverse=True)
        self.inferences.extend(results)
        return results
    
    def _logical(self, words: List[str]) -> List[Inference]:
        infs = []
        for word in words:
            c = self.semantic.get_concept(word)
            if c:
                for parent in c.relations.get("is-a", set()):
                    pc = self.semantic.get_concept(parent)
                    if pc:
                        for gp in pc.relations.get("is-a", set()):
                            infs.append(Inference(
                                proposition=f"{word} is-a {gp}",
                                confidence=0.9,
                                mode=ReasoningMode.LOGICAL,
                                premises=[f"{word} is-a {parent}", f"{parent} is-a {gp}"],
                                explanation="Transitivity"
                            ))
        return infs
    
    def _analogical(self, words: List[str]) -> List[Inference]:
        infs = []
        for word in words:
            c = self.semantic.get_concept(word)
            if c:
                for other_name, other in self.semantic.concepts.items():
                    if other_name != word:
                        shared = set(c.relations.keys()) & set(other.relations.keys())
                        if len(shared) >= 2:
                            infs.append(Inference(
                                proposition=f"{word} is analogous to {other_name}",
                                confidence=0.6,
                                mode=ReasoningMode.ANALOGICAL,
                                premises=[f"Shared relations: {shared}"],
                                explanation="Structural similarity"
                            ))
        return infs[:3]
    
    def _intuitive(self, words: List[str]) -> List[Inference]:
        infs = []
        all_acts = {}
        for word in words:
            acts = self.semantic.spreading_activation(word)
            for k, v in acts.items():
                all_acts[k] = all_acts.get(k, 0) + v
        
        for concept, act in sorted(all_acts.items(), key=lambda x: x[1], reverse=True)[:3]:
            if concept not in words:
                infs.append(Inference(
                    proposition=f"Intuitive connection to {concept}",
                    confidence=min(0.8, act),
                    mode=ReasoningMode.INTUITIVE,
                    premises=["Spreading activation"],
                    explanation=f"Activation: {act:.2f}"
                ))
        return infs


class CognitiveCore:
    """Complete cognitive architecture."""
    
    def __init__(self):
        self.working = WorkingMemory()
        self.semantic = SemanticMemory()
        self.episodic = EpisodicMemory()
        self.reasoning = ReasoningEngine(self.semantic)
        self.cognitive_load = 0.0
        self._bootstrap()
    
    def _bootstrap(self):
        """Initialize foundational knowledge."""
        concepts = [
            ("number", "mathematics", {"abstract": True}),
            ("algorithm", "computation", {"executable": True}),
            ("recursion", "computation", {"self_referential": True}),
            ("phi", "mathematics", {"value": PHI, "sacred": True}),
            ("consciousness", "philosophy", {"emergent": True}),
        ]
        for name, cat, feat in concepts:
            self.semantic.add_concept(Concept(name=name, category=cat, features=feat))
        
        self.semantic.add_relation("recursion", "is-a", "algorithm")
        self.semantic.add_relation("phi", "is-a", "number")
    
    def learn(self, name: str, category: str, features: Dict = None, relations: Dict = None):
        """Learn a new concept."""
        self.semantic.add_concept(Concept(name=name, category=category, features=features or {}))
        if relations:
            for rel, targets in relations.items():
                for t in targets:
                    self.semantic.add_relation(name, rel, t)
        self.episodic.store({"type": "learning", "concept": name})
    
    def think(self, query: str) -> List[Inference]:
        """Main reasoning process."""
        self.working.add(query)
        self.cognitive_load = min(1.0, len(query.split()) / 20)
        inferences = self.reasoning.reason(query)
        self.episodic.store({"type": "thinking", "query": query, "results": len(inferences)})
        return inferences
    
    def recall(self, cue: str) -> Dict[str, Any]:
        """Recall information."""
        concept = self.semantic.get_concept(cue)
        episodes = self.episodic.retrieve(cue)
        activations = self.semantic.spreading_activation(cue)
        return {
            "concept": concept.name if concept else None,
            "episodes": len(episodes),
            "top_associations": sorted(activations.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def introspect(self) -> Dict[str, Any]:
        """Metacognitive status."""
        return {
            "working_memory": len(self.working.items),
            "cognitive_load": self.cognitive_load,
            "concepts": len(self.semantic.concepts),
            "episodes": len(self.episodic.episodes),
            "inferences": len(self.reasoning.inferences)
        }


# Global instance
COGNITIVE_CORE = CognitiveCore()


if __name__ == "__main__":
    print("\n" + "🧠" * 30)
    print("  L104 COGNITIVE CORE")
    print("🧠" * 30 + "\n")
    
    cog = CognitiveCore()
    
    # Learn concepts
    cog.learn("transformer", "computation", {"attention": True}, {"is-a": ["algorithm"]})
    cog.learn("neural_network", "computation", {"layers": True}, {"is-a": ["algorithm"]})
    
    # Think
    inferences = cog.think("How does recursion relate to consciousness?")
    print(f"Query: How does recursion relate to consciousness?")
    print(f"Inferences: {len(inferences)}")
    for inf in inferences[:3]:
        print(f"  [{inf.mode.name}] {inf.proposition} ({inf.confidence:.2f})")
    
    # Introspect
    print(f"\nIntrospection: {cog.introspect()}")
    
    print("\n" + "🧠" * 30)
