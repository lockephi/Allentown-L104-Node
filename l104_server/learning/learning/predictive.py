"""
Predictive Mixin for LearningIntellect

Extracted from intellect.py during EVO_78 refactoring.
Contains: Predictive prefetching, novelty detection, adaptive learning.
"""

from typing import List, Optional, Dict
import math


class PredictiveMixin:
    """Predictive operations for LearningIntellect.
    
    This mixin provides:
    - Next query prediction
    - Response prefetching
    - Novelty scoring
    - Adaptive learning rate
    
    Usage:
        class LearningIntellect(PredictiveMixin, ...):
            pass
    """
    
    def predict_next_queries(self, current_query: str, top_k: int = 5) -> List[str]:
        """Predict next likely queries based on patterns and knowledge graph."""
        predictions = []
        
        # Get concepts from current query
        concepts = self._extract_concepts(current_query) if hasattr(self, '_extract_concepts') else []
        
        # Find related concepts from knowledge graph
        for concept in concepts[:5]:
            related = self.knowledge_graph.get(concept, [])
            for related_concept, strength in related[:3]:
                if related_concept not in predictions and strength > 0.3:
                    predictions.append(related_concept)
        
        # Use pattern matching for follow-up predictions
        query_lower = current_query.lower()
        for pattern, (strategy, success) in getattr(self, 'meta_strategies', {}).items():
            if pattern in query_lower and success > 0.6:
                # Generate follow-up based on pattern
                follow_up = f"tell me more about {pattern}"
                if follow_up not in predictions:
                    predictions.append(follow_up)
        
        return predictions[:top_k]

    def prefetch_responses(self, predictions: List[str]) -> int:
        """Prefetch responses for predicted queries."""
        prefetched = 0
        for query in predictions:
            try:
                query_hash = self._hash_query(query) if hasattr(self, '_hash_query') else hash(query)
                if query_hash in self.memory_cache:
                    self.predictive_cache['prefetched'][query] = {
                        'response': self.memory_cache[query_hash],
                        'timestamp': self._get_timestamp() if hasattr(self, '_get_timestamp') else 0
                    }
                    prefetched += 1
            except Exception:
                pass
        return prefetched

    def get_prefetched(self, query: str) -> Optional[dict]:
        """Get prefetched response if available and fresh."""
        cached = self.predictive_cache.get('prefetched', {}).get(query)
        if cached:
            # Check freshness (30 second TTL)
            timestamp = cached.get('timestamp', 0)
            if hasattr(self, '_get_timestamp'):
                if self._get_timestamp() - timestamp < 30:
                    return cached
        return None

    def compute_novelty(self, query: str) -> float:
        """Compute novelty score for a query (0 = known, 1 = completely novel)."""
        query_lower = query.lower()
        
        # Check memory cache
        query_hash = self._hash_query(query_lower) if hasattr(self, '_hash_query') else hash(query_lower)
        if query_hash in self.memory_cache:
            self.novelty_scores[query_lower] = 0.0
            return 0.0
        
        # Check concept familiarity
        concepts = self._extract_concepts(query_lower) if hasattr(self, '_extract_concepts') else []
        if not concepts:
            return 0.5  # Unknown novelty
        
        familiar_concepts = 0
        for concept in concepts:
            if concept in self.knowledge_graph or concept in self._concept_to_cluster:
                familiar_concepts += 1
        
        novelty = 1.0 - (familiar_concepts / max(len(concepts), 1))
        self.novelty_scores[query_lower] = novelty
        return novelty

    def get_adaptive_learning_rate(self, query: str, quality: float) -> float:
        """Compute adaptive learning rate based on novelty and quality."""
        base_rate = 0.1
        novelty = self.compute_novelty(query)
        
        # Higher novelty = higher learning rate
        novelty_boost = novelty * 0.3
        
        # Higher quality = higher learning rate
        quality_boost = quality * 0.2
        
        # PHI-weighted learning rate
        phi = 1.618033988749895
        adaptive_rate = (base_rate + novelty_boost + quality_boost) * phi / 2
        
        return min(0.5, adaptive_rate)  # Cap at 0.5

    def _get_timestamp(self) -> float:
        """Get current timestamp."""
        import time
        return time.time()
