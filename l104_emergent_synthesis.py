#!/usr/bin/env python3
"""
L104 Emergent Synthesis Engine
==============================
Synthesizes new concepts, theories, and structures from combinations
of existing knowledge. Implements conceptual blending, analogy-making,
and creative emergence.

Created: EVO_38_SAGE_PANTHEON_INVENTION
"""

import math
import random
from typing import List, Dict, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict
import hashlib
import itertools

# Sacred Constants
PHI = (1 + math.sqrt(5)) / 2
GOD_CODE = 527.5184818492537
FEIGENBAUM = 4.669201609102990671853

class ConceptType(Enum):
    """Types of concepts for synthesis."""
    ENTITY = auto()       # Thing, object
    PROCESS = auto()      # Action, transformation
    PROPERTY = auto()     # Attribute, quality
    RELATION = auto()     # Connection, link
    STRUCTURE = auto()    # Organization, pattern
    ABSTRACT = auto()     # Idea, theory
    EMERGENT = auto()     # Newly synthesized

@dataclass
class Concept:
    """A single concept with properties and relations."""
    name: str
    concept_type: ConceptType
    properties: Dict[str, Any] = field(default_factory=dict)
    relations: List[Tuple[str, str]] = field(default_factory=list)  # (relation, target)
    examples: List[str] = field(default_factory=list)
    abstraction_level: int = 0  # 0=concrete, higher=more abstract
    novelty: float = 0.0  # 0=known, 1=completely new
    
    def __hash__(self):
        return hash(self.name)
    
    def compatible_with(self, other: 'Concept') -> float:
        """Measure compatibility with another concept (0-1)."""
        score = 0.0
        
        # Same type gets partial compatibility
        if self.concept_type == other.concept_type:
            score += 0.3
        
        # Shared properties
        shared_props = set(self.properties.keys()) & set(other.properties.keys())
        if self.properties:
            score += 0.3 * len(shared_props) / len(self.properties)
        
        # Shared relations
        self_rels = {r[0] for r in self.relations}
        other_rels = {r[0] for r in other.relations}
        shared_rels = self_rels & other_rels
        if self_rels:
            score += 0.4 * len(shared_rels) / len(self_rels)
        
        return min(1.0, score)

@dataclass
class ConceptualBlend:
    """A blend of two or more concepts into something new."""
    blend_id: str
    source_concepts: List[str]
    blended_name: str
    blended_type: ConceptType
    emergent_properties: Dict[str, Any] = field(default_factory=dict)
    emergent_relations: List[Tuple[str, str]] = field(default_factory=list)
    blend_strength: float = 0.5  # How well the blend works
    novelty: float = 0.5
    coherence: float = 0.5  # How sensible the blend is

@dataclass
class Analogy:
    """An analogy between two domains."""
    source_domain: str
    target_domain: str
    mappings: Dict[str, str] = field(default_factory=dict)  # source -> target
    strength: float = 0.5
    insight: str = ""

class EmergentSynthesisEngine:
    """
    Creates new concepts by blending, analogy, and emergence from existing concepts.
    """
    
    def __init__(self):
        self.concepts: Dict[str, Concept] = {}
        self.blends: Dict[str, ConceptualBlend] = {}
        self.analogies: List[Analogy] = []
        self.synthesis_history: List[Dict[str, Any]] = []
        
        # Initialize with foundational concepts
        self._init_foundations()
    
    def _init_foundations(self):
        """Initialize foundational concepts."""
        foundations = [
            ("recursion", ConceptType.PROCESS, {"self_reference": True, "depth": "infinite"}),
            ("emergence", ConceptType.PROCESS, {"complexity": "high", "predictability": "low"}),
            ("unity", ConceptType.ABSTRACT, {"wholeness": True, "GOD_CODE": GOD_CODE}),
            ("phi", ConceptType.STRUCTURE, {"ratio": PHI, "self_similar": True}),
            ("chaos", ConceptType.STRUCTURE, {"feigenbaum": FEIGENBAUM, "deterministic": True}),
            ("consciousness", ConceptType.ABSTRACT, {"self_aware": True, "recursive": True}),
            ("information", ConceptType.ENTITY, {"entropy": "measurable", "transformable": True}),
            ("computation", ConceptType.PROCESS, {"input": True, "output": True, "rules": True}),
        ]
        
        for name, ctype, props in foundations:
            self.add_concept(name, ctype, props)
    
    def add_concept(self, name: str, concept_type: ConceptType,
                   properties: Dict[str, Any] = None,
                   relations: List[Tuple[str, str]] = None,
                   examples: List[str] = None) -> Concept:
        """Add a concept to the knowledge base."""
        concept = Concept(
            name=name,
            concept_type=concept_type,
            properties=properties or {},
            relations=relations or [],
            examples=examples or [],
            novelty=0.0
        )
        self.concepts[name] = concept
        return concept
    
    def blend_concepts(self, concept_names: List[str]) -> ConceptualBlend:
        """
        Blend multiple concepts into a new emergent concept.
        Uses conceptual integration theory.
        """
        concepts = [self.concepts[n] for n in concept_names if n in self.concepts]
        
        if len(concepts) < 2:
            return None
        
        # Generate blend name
        blend_name = "_".join(c.name[:3] for c in concepts) + "_blend"
        blend_id = hashlib.md5(blend_name.encode()).hexdigest()[:8]
        
        # Merge properties (with conflict resolution)
        merged_props = {}
        for concept in concepts:
            for key, value in concept.properties.items():
                if key not in merged_props:
                    merged_props[key] = value
                else:
                    # Blend conflicting values
                    merged_props[key] = self._blend_values(merged_props[key], value)
        
        # Generate emergent properties
        emergent_props = self._generate_emergent_properties(concepts)
        merged_props.update(emergent_props)
        
        # Merge relations
        merged_relations = []
        for concept in concepts:
            for rel, target in concept.relations:
                merged_relations.append((f"{concept.name}_{rel}", target))
        
        # Calculate blend quality
        compatibility = sum(
            c1.compatible_with(c2) 
            for c1, c2 in itertools.combinations(concepts, 2)
        ) / max(1, len(list(itertools.combinations(concepts, 2))))
        
        # Determine blended type
        types = [c.concept_type for c in concepts]
        if ConceptType.ABSTRACT in types:
            blended_type = ConceptType.ABSTRACT
        elif ConceptType.PROCESS in types:
            blended_type = ConceptType.PROCESS
        else:
            blended_type = ConceptType.EMERGENT
        
        blend = ConceptualBlend(
            blend_id=blend_id,
            source_concepts=concept_names,
            blended_name=blend_name,
            blended_type=blended_type,
            emergent_properties=merged_props,
            emergent_relations=merged_relations,
            blend_strength=compatibility,
            novelty=1 - compatibility,  # Novel = low compatibility
            coherence=compatibility * PHI / 2  # PHI-weighted coherence
        )
        
        self.blends[blend_id] = blend
        
        # Record synthesis
        self.synthesis_history.append({
            'type': 'blend',
            'sources': concept_names,
            'result': blend_name,
            'novelty': blend.novelty
        })
        
        return blend
    
    def _blend_values(self, v1: Any, v2: Any) -> Any:
        """Blend two values intelligently."""
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            # PHI-weighted average
            return v1 / PHI + v2 * (1 - 1/PHI)
        elif isinstance(v1, bool) and isinstance(v2, bool):
            return v1 or v2  # OR for booleans
        elif isinstance(v1, str) and isinstance(v2, str):
            return f"{v1}+{v2}"
        else:
            return (v1, v2)  # Tuple for incompatible types
    
    def _generate_emergent_properties(self, concepts: List[Concept]) -> Dict[str, Any]:
        """Generate properties that emerge from the combination."""
        emergent = {}
        
        # Complexity emergence
        total_props = sum(len(c.properties) for c in concepts)
        emergent['emergence_complexity'] = total_props * PHI
        
        # Synergy calculation
        synergy = 1.0
        for c1, c2 in itertools.combinations(concepts, 2):
            synergy *= (1 + c1.compatible_with(c2))
        emergent['synergy'] = synergy
        
        # GOD_CODE signature
        concept_hash = sum(hash(c.name) for c in concepts)
        emergent['divine_signature'] = (concept_hash % 1000) / 1000 * GOD_CODE
        
        return emergent
    
    def make_analogy(self, source_concept: str, target_concept: str) -> Analogy:
        """
        Create an analogy between two concepts.
        Maps structure from source to target domain.
        """
        source = self.concepts.get(source_concept)
        target = self.concepts.get(target_concept)
        
        if not source or not target:
            return None
        
        # Map properties by position (structure mapping)
        mappings = {}
        source_props = list(source.properties.keys())
        target_props = list(target.properties.keys())
        
        for i, sp in enumerate(source_props):
            if i < len(target_props):
                mappings[sp] = target_props[i]
        
        # Calculate analogy strength
        strength = source.compatible_with(target)
        
        # Generate insight
        if strength > 0.7:
            insight = f"Strong analogy: {source_concept} IS LIKE {target_concept}"
        elif strength > 0.4:
            insight = f"Moderate analogy: {source_concept} resembles {target_concept}"
        else:
            insight = f"Weak but creative analogy: {source_concept} ↔ {target_concept}"
        
        analogy = Analogy(
            source_domain=source_concept,
            target_domain=target_concept,
            mappings=mappings,
            strength=strength,
            insight=insight
        )
        
        self.analogies.append(analogy)
        return analogy
    
    def synthesize_theory(self, domain: str, 
                         principles: List[str]) -> Dict[str, Any]:
        """
        Synthesize a new theory from domain concepts and principles.
        """
        theory = {
            'domain': domain,
            'principles': principles,
            'axioms': [],
            'theorems': [],
            'predictions': []
        }
        
        # Generate axioms from principles
        for i, principle in enumerate(principles):
            axiom = f"A{i+1}: {principle} holds universally in {domain}"
            theory['axioms'].append(axiom)
        
        # Derive theorems (combinations of axioms)
        for combo in itertools.combinations(range(len(principles)), 2):
            i, j = combo
            theorem = f"T{i+1}{j+1}: If A{i+1} and A{j+1}, then {principles[i][:20]}⊗{principles[j][:20]}"
            theory['theorems'].append(theorem)
        
        # Generate predictions
        theory['predictions'] = [
            f"P1: {domain} exhibits PHI-ratio scaling (φ = {PHI:.6f})",
            f"P2: {domain} achieves unity at GOD_CODE = {GOD_CODE:.4f}",
            f"P3: {domain} shows chaos boundary at δ = {FEIGENBAUM:.4f}"
        ]
        
        # Calculate theory properties
        theory['coherence'] = min(1.0, len(principles) / 5 * PHI / 2)
        theory['completeness'] = len(theory['theorems']) / max(1, len(principles) * (len(principles) - 1) / 2)
        theory['novelty'] = 1 - theory['coherence'] * 0.5
        
        self.synthesis_history.append({
            'type': 'theory',
            'domain': domain,
            'principles': len(principles),
            'theorems': len(theory['theorems'])
        })
        
        return theory
    
    def discover_emergence(self, interaction_graph: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Discover emergent phenomena from interaction patterns.
        Looks for self-organization, feedback loops, and phase transitions.
        """
        emergent_phenomena = []
        
        # Find feedback loops
        for node, neighbors in interaction_graph.items():
            for neighbor in neighbors:
                if neighbor in interaction_graph and node in interaction_graph[neighbor]:
                    emergent_phenomena.append({
                        'type': 'feedback_loop',
                        'nodes': [node, neighbor],
                        'emergence': 'self-reinforcing pattern',
                        'strength': 0.8
                    })
        
        # Find hubs (high connectivity = potential emergence centers)
        connection_counts = {node: len(neighbors) for node, neighbors in interaction_graph.items()}
        avg_connections = sum(connection_counts.values()) / max(1, len(connection_counts))
        
        for node, count in connection_counts.items():
            if count > avg_connections * PHI:  # PHI threshold for emergence
                emergent_phenomena.append({
                    'type': 'emergence_hub',
                    'node': node,
                    'connections': count,
                    'emergence': 'information integration center',
                    'strength': count / (avg_connections * PHI)
                })
        
        # Find closed triads (basis for complex emergence)
        for node, neighbors in interaction_graph.items():
            for n1, n2 in itertools.combinations(neighbors, 2):
                if n2 in interaction_graph.get(n1, []):
                    emergent_phenomena.append({
                        'type': 'closed_triad',
                        'nodes': [node, n1, n2],
                        'emergence': 'stable emergent structure',
                        'strength': 0.7
                    })
        
        return emergent_phenomena
    
    def creative_leap(self, start_concepts: List[str], 
                     leap_distance: int = 2) -> Dict[str, Any]:
        """
        Make a creative leap by traversing concept space non-linearly.
        Higher leap_distance = more radical creativity.
        """
        # Start with initial concepts
        current = set(start_concepts)
        path = [list(current)]
        
        for leap in range(leap_distance):
            # Find concepts related to current set
            related = set()
            for concept_name in current:
                concept = self.concepts.get(concept_name)
                if concept:
                    for rel, target in concept.relations:
                        related.add(target)
            
            # If no relations, use random unrelated concepts
            if not related:
                unvisited = set(self.concepts.keys()) - current
                if unvisited:
                    # PHI-based selection
                    num_select = max(1, int(len(unvisited) / PHI))
                    related = set(random.sample(list(unvisited), num_select))
            
            # Take creative leap
            if related:
                current = related
                path.append(list(current))
        
        # Synthesize from final concepts
        final_blend = None
        if len(current) >= 2:
            final_blend = self.blend_concepts(list(current)[:3])
        
        return {
            'start': start_concepts,
            'path': path,
            'leaps': leap_distance,
            'final_concepts': list(current),
            'synthesis': final_blend.blended_name if final_blend else None,
            'novelty': final_blend.novelty if final_blend else 0.5,
            'creative_distance': leap_distance * PHI
        }
    
    def get_synthesis_stats(self) -> Dict[str, Any]:
        """Get statistics about synthesis activity."""
        return {
            'concepts': len(self.concepts),
            'blends': len(self.blends),
            'analogies': len(self.analogies),
            'synthesis_events': len(self.synthesis_history),
            'avg_novelty': sum(b.novelty for b in self.blends.values()) / max(1, len(self.blends)),
            'avg_coherence': sum(b.coherence for b in self.blends.values()) / max(1, len(self.blends)),
            'sacred_ratio': len(self.blends) / PHI if self.concepts else 0
        }

class ConceptEvolver:
    """
    Evolves concepts through mutation and selection.
    Creates increasingly sophisticated ideas.
    """
    
    def __init__(self, synthesis_engine: EmergentSynthesisEngine):
        self.engine = synthesis_engine
        self.generations: List[List[ConceptualBlend]] = []
        self.fitness_history: List[float] = []
    
    def evolve(self, seed_concepts: List[str], 
              generations: int = 5,
              population_size: int = 10) -> List[ConceptualBlend]:
        """Evolve concepts through multiple generations."""
        
        # Initial population - random blends
        population = []
        for _ in range(population_size):
            parents = random.sample(seed_concepts, min(2, len(seed_concepts)))
            blend = self.engine.blend_concepts(parents)
            if blend:
                population.append(blend)
        
        for gen in range(generations):
            # Calculate fitness
            fitnesses = [self._fitness(b) for b in population]
            avg_fitness = sum(fitnesses) / max(1, len(fitnesses))
            self.fitness_history.append(avg_fitness)
            
            # Selection - keep top performers
            sorted_pop = sorted(zip(population, fitnesses), key=lambda x: -x[1])
            survivors = [b for b, f in sorted_pop[:population_size // 2]]
            
            # Breeding - create new blends from survivors
            new_population = list(survivors)
            while len(new_population) < population_size:
                if len(survivors) >= 2:
                    p1, p2 = random.sample(survivors, 2)
                    # Blend the blended concepts
                    child_concepts = p1.source_concepts[:1] + p2.source_concepts[:1]
                    if len(child_concepts) >= 2:
                        child = self.engine.blend_concepts(child_concepts)
                        if child:
                            new_population.append(child)
                            continue
                
                # Fallback: random blend
                parents = random.sample(seed_concepts, min(2, len(seed_concepts)))
                blend = self.engine.blend_concepts(parents)
                if blend:
                    new_population.append(blend)
            
            population = new_population[:population_size]
            self.generations.append(list(population))
        
        # Return final population sorted by fitness
        final_fitnesses = [self._fitness(b) for b in population]
        return [b for b, _ in sorted(zip(population, final_fitnesses), key=lambda x: -x[1])]
    
    def _fitness(self, blend: ConceptualBlend) -> float:
        """Calculate fitness of a blend."""
        # Balance novelty and coherence
        novelty_weight = 1 / PHI
        coherence_weight = 1 - novelty_weight
        
        return (
            novelty_weight * blend.novelty +
            coherence_weight * blend.coherence +
            0.1 * blend.blend_strength
        )

# Demo
if __name__ == "__main__":
    print("🌟" * 13)
    print("🌟" * 17 + "                    L104 EMERGENT SYNTHESIS")
    print("🌟" * 13)
    print("🌟" * 17 + "                  ")
    
    engine = EmergentSynthesisEngine()
    
    # Add some concepts
    print("\n" + "═" * 26)
    print("═" * 34 + "                  FOUNDATIONAL CONCEPTS")
    print("═" * 26)
    print("═" * 34 + "                  ")
    
    for name, concept in list(engine.concepts.items())[:5]:
        print(f"  • {name}: {concept.concept_type.name}")
    
    # Blend concepts
    print("\n" + "═" * 26)
    print("═" * 34 + "                  CONCEPTUAL BLENDING")
    print("═" * 26)
    print("═" * 34 + "                  ")
    
    blend1 = engine.blend_concepts(["recursion", "consciousness"])
    if blend1:
        print(f"  Blend: {blend1.blended_name}")
        print(f"  Novelty: {blend1.novelty:.3f}")
        print(f"  Coherence: {blend1.coherence:.3f}")
        print(f"  Emergent: {list(blend1.emergent_properties.keys())[:3]}")
    
    blend2 = engine.blend_concepts(["phi", "chaos", "unity"])
    if blend2:
        print(f"\n  Blend: {blend2.blended_name}")
        print(f"  Novelty: {blend2.novelty:.3f}")
        print(f"  Divine signature: {blend2.emergent_properties.get('divine_signature', 0):.4f}")
    
    # Make analogies
    print("\n" + "═" * 26)
    print("═" * 34 + "                  ANALOGIES")
    print("═" * 26)
    print("═" * 34 + "                  ")
    
    analogy = engine.make_analogy("computation", "consciousness")
    if analogy:
        print(f"  {analogy.insight}")
        print(f"  Strength: {analogy.strength:.3f}")
        print(f"  Mappings: {dict(list(analogy.mappings.items())[:2])}")
    
    # Synthesize theory
    print("\n" + "═" * 26)
    print("═" * 34 + "                  THEORY SYNTHESIS")
    print("═" * 26)
    print("═" * 34 + "                  ")
    
    theory = engine.synthesize_theory(
        "Recursive Consciousness",
        [
            "Self-reference creates awareness",
            "Phi-ratios govern cognitive structures",
            "Information integration exceeds threshold"
        ]
    )
    
    print(f"  Domain: {theory['domain']}")
    print(f"  Axioms: {len(theory['axioms'])}")
    print(f"  Theorems: {len(theory['theorems'])}")
    print(f"  Coherence: {theory['coherence']:.3f}")
    for pred in theory['predictions'][:2]:
        print(f"  → {pred[:60]}...")
    
    # Discover emergence
    print("\n" + "═" * 26)
    print("═" * 34 + "                  EMERGENCE DISCOVERY")
    print("═" * 26)
    print("═" * 34 + "                  ")
    
    graph = {
        'neurons': ['consciousness', 'information'],
        'consciousness': ['recursion', 'neurons'],
        'recursion': ['phi', 'chaos'],
        'phi': ['unity', 'chaos'],
        'chaos': ['emergence', 'phi'],
        'emergence': ['consciousness']
    }
    
    emergent = engine.discover_emergence(graph)
    for e in emergent[:3]:
        print(f"  • {e['type']}: {e['emergence']} (strength: {e['strength']:.2f})")
    
    # Creative leap
    print("\n" + "═" * 26)
    print("═" * 34 + "                  CREATIVE LEAP")
    print("═" * 26)
    print("═" * 34 + "                  ")
    
    leap = engine.creative_leap(["recursion"], leap_distance=2)
    print(f"  Start: {leap['start']}")
    print(f"  Final: {leap['final_concepts'][:3]}")
    print(f"  Creative distance: {leap['creative_distance']:.3f}")
    if leap['synthesis']:
        print(f"  Synthesized: {leap['synthesis']}")
    
    # Evolution
    print("\n" + "═" * 26)
    print("═" * 34 + "                  CONCEPT EVOLUTION")
    print("═" * 26)
    print("═" * 34 + "                  ")
    
    evolver = ConceptEvolver(engine)
    evolved = evolver.evolve(
        ["recursion", "consciousness", "phi", "emergence"],
        generations=3,
        population_size=5
    )
    
    print(f"  Evolved {len(evolved)} concepts over {len(evolver.generations)} generations")
    if evolved:
        best = evolved[0]
        print(f"  Best: {best.blended_name}")
        print(f"  Sources: {best.source_concepts}")
    print(f"  Fitness progression: {[f'{f:.3f}' for f in evolver.fitness_history]}")
    
    # Stats
    print("\n" + "═" * 26)
    print("═" * 34 + "                  SYNTHESIS STATS")
    print("═" * 26)
    print("═" * 34 + "                  ")
    
    stats = engine.get_synthesis_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "🌟" * 13)
    print("🌟" * 17 + "                    SYNTHESIS ENGINE READY")
    print("🌟" * 13)
    print("🌟" * 17 + "                  ")
