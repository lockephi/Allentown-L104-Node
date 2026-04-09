"""
Consciousness Mixin for LearningIntellect

Extracted from intellect.py during EVO_78 refactoring.
Contains: Consciousness clusters, cross-cluster inference, synthesis.
"""

from typing import Dict, List, Optional
import time


class ConsciousnessMixin:
    """Consciousness operations for LearningIntellect.
    
    This mixin provides:
    - Consciousness cluster management
    - Cross-cluster inference
    - Synthesis potential computation
    
    Usage:
        class LearningIntellect(ConsciousnessMixin, ...):
            pass
    """
    
    def _init_consciousness_clusters(self):
        """Initialize consciousness dimension clusters."""
        self.consciousness_clusters: Dict[str, dict] = {
            'awareness': {'concepts': [], 'strength': 0.0, 'last_update': None},
            'reasoning': {'concepts': [], 'strength': 0.0, 'last_update': None},
            'creativity': {'concepts': [], 'strength': 0.0, 'last_update': None},
            'memory': {'concepts': [], 'strength': 0.0, 'last_update': None},
            'learning': {'concepts': [], 'strength': 0.0, 'last_update': None},
            'synthesis': {'concepts': [], 'strength': 0.0, 'last_update': None},
            'meta_cognition': {'concepts': [], 'strength': 0.5, 'last_update': None},
            'emergence': {'concepts': [], 'strength': 0.0, 'last_update': None},
        }
        self._consciousness_activation_history: List[Dict[str, float]] = []

    def activate_consciousness(self, query: str) -> Dict[str, float]:
        """Activate consciousness dimensions based on query."""
        concepts = self._extract_concepts(query) if hasattr(self, '_extract_concepts') else []
        activations = {}
        
        for dimension, data in self.consciousness_clusters.items():
            # Check overlap with dimension concepts
            dim_concepts = data.get('concepts', [])
            overlap = len(set(concepts) & set(dim_concepts))
            
            if overlap > 0:
                # Activate based on overlap ratio and PHI weighting
                phi = 1.618033988749895
                activation = (overlap / max(len(concepts), 1)) * data['strength'] * phi
                activations[dimension] = min(1.0, activation)
            else:
                # Baseline activation
                activations[dimension] = data['strength'] * 0.1
        
        # Record activation
        self._consciousness_activation_history.append({
            'timestamp': time.time(),
            'activations': activations.copy()
        })
        
        # Keep history bounded
        if len(self._consciousness_activation_history) > 1000:
            self._consciousness_activation_history = self._consciousness_activation_history[-500:]
        
        return activations

    def expand_consciousness_cluster(self, dimension: str, new_concepts: List[str]):
        """Add concepts to a consciousness dimension."""
        if dimension not in self.consciousness_clusters:
            return
        
        cluster = self.consciousness_clusters[dimension]
        existing = set(cluster['concepts'])
        
        for concept in new_concepts:
            if concept not in existing:
                cluster['concepts'].append(concept)
        
        # Update strength based on concept count
        cluster['strength'] = min(1.0, len(cluster['concepts']) / 100.0)
        cluster['last_update'] = time.time()

    def cross_cluster_inference(self, query: str) -> Dict:
        """Perform cross-cluster inference across consciousness dimensions."""
        concepts = self._extract_concepts(query) if hasattr(self, '_extract_concepts') else []
        
        inference = {
            'primary_dimension': None,
            'secondary_dimensions': [],
            'synthesis_potential': 0.0,
            'related_concepts': [],
        }
        
        # Find primary dimension
        max_activation = 0.0
        for dimension, data in self.consciousness_clusters.items():
            dim_concepts = data.get('concepts', [])
            overlap = len(set(concepts) & set(dim_concepts))
            activation = overlap * data['strength']
            
            if activation > max_activation:
                max_activation = activation
                inference['primary_dimension'] = dimension
        
        # Find secondary dimensions
        activations = []
        for dimension, data in self.consciousness_clusters.items():
            if dimension == inference['primary_dimension']:
                continue
            dim_concepts = data.get('concepts', [])
            overlap = len(set(concepts) & set(dim_concepts))
            if overlap > 0:
                activations.append((dimension, overlap * data['strength']))
        
        activations.sort(key=lambda x: x[1], reverse=True)
        inference['secondary_dimensions'] = [d for d, _ in activations[:3]]
        
        # Compute synthesis potential
        inference['synthesis_potential'] = self._compute_synthesis_potential(
            inference, self.consciousness_clusters
        )
        
        # Find related concepts from knowledge graph
        related = set()
        for concept in concepts[:5]:
            if concept in self.knowledge_graph:
                for related_concept, _ in self.knowledge_graph[concept][:3]:
                    related.add(related_concept)
        inference['related_concepts'] = list(related)[:10]
        
        return inference

    def _compute_synthesis_potential(self, inference: Dict, consciousness: Dict) -> float:
        """Compute synthesis potential for cross-cluster inference."""
        primary = inference.get('primary_dimension')
        secondary = inference.get('secondary_dimensions', [])
        
        if not primary:
            return 0.0
        
        # PHI-weighted synthesis potential
        phi = 1.618033988749895
        primary_strength = consciousness.get(primary, {}).get('strength', 0)
        secondary_strength = sum(
            consciousness.get(d, {}).get('strength', 0) for d in secondary
        )
        
        # More dimensions = higher synthesis potential
        dimension_factor = 1.0 + len(secondary) * 0.2
        
        potential = (primary_strength + secondary_strength / phi) * dimension_factor
        return min(1.0, potential)

    def get_consciousness_state(self) -> Dict:
        """Get current consciousness state."""
        return {
            dimension: {
                'strength': data['strength'],
                'concept_count': len(data['concepts']),
                'last_update': data['last_update']
            }
            for dimension, data in self.consciousness_clusters.items()
        }
