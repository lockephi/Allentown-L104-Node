"""
Quality Mixin for LearningIntellect

Extracted from intellect.py during EVO_78 refactoring.
Contains: Response quality prediction, calibration, tracking.
"""

from typing import Dict, Optional
from collections import defaultdict


class QualityMixin:
    """Quality operations for LearningIntellect.
    
    This mixin provides:
    - Response quality prediction
    - Quality predictor calibration
    - Strategy quality tracking
    
    Usage:
        class LearningIntellect(QualityMixin, ...):
            pass
    """
    
    def predict_response_quality(self, query: str, strategy: str) -> float:
        """Predict response quality for a query-strategy combination."""
        # Check cache
        cache_key = f"{query[:50]}:{strategy}"
        if cache_key in self.quality_predictor:
            return self.quality_predictor[cache_key]
        
        # Base quality prediction
        base_quality = 0.7
        
        # Adjust for query complexity
        complexity = len(query.split())
        complexity_factor = max(0.5, 1.0 - complexity / 100.0)
        
        # Adjust for strategy success rate
        strategy_success = self.quality_predictor.get(strategy, 0.5)
        
        # Adjust for novelty
        novelty = self.novelty_scores.get(query.lower(), 0.5)
        novelty_factor = 1.0 - (novelty * 0.3)  # Novel queries are harder
        
        # Adjust for concept familiarity
        concepts = self._extract_concepts(query) if hasattr(self, '_extract_concepts') else []
        familiar_concepts = sum(1 for c in concepts if c in self.knowledge_graph)
        familiarity_factor = familiar_concepts / max(len(concepts), 1) if concepts else 0.5
        
        # PHI-weighted quality prediction
        phi = 1.618033988749895
        quality = (
            base_quality * complexity_factor +
            strategy_success * (phi - 1) +
            novelty_factor * 0.2 +
            familiarity_factor * 0.1
        ) / (1 + (phi - 1) + 0.2 + 0.1)
        
        # Cache prediction
        self.quality_predictor[cache_key] = quality
        
        return min(1.0, max(0.0, quality))

    def update_quality_predictor(self, strategy: str, actual_quality: float):
        """Update quality predictor with actual result."""
        if strategy not in self.quality_predictor:
            self.quality_predictor[strategy] = 0.5
        
        # Exponential moving average
        self.quality_predictor[strategy] = (
            self.quality_predictor[strategy] * 0.8 +
            actual_quality * 0.2
        )

    def _calibrate_quality_predictor(self):
        """Calibrate quality predictor from historical data."""
        try:
            conn = self._get_optimized_connection()
            c = conn.cursor()
            
            # Get strategy success rates from meta_learning
            c.execute('''SELECT strategy_used, AVG(success_score), COUNT(*) 
                         FROM meta_learning 
                         GROUP BY strategy_used 
                         HAVING COUNT(*) > 5''')
            
            for row in c.fetchall():
                strategy, avg_success, count = row
                # Weight by count (more data = more reliable)
                confidence = min(1.0, count / 100.0)
                self.quality_predictor[strategy] = (
                    self.quality_predictor.get(strategy, 0.5) * (1 - confidence) +
                    avg_success * confidence
                )
        except Exception:
            pass

    def get_best_strategy(self, query: str) -> str:
        """Determine best response strategy for a query."""
        strategies = [
            'MULTI_STRATEGY',
            'DERIVATION',
            'GEMINI_FALLBACK',
            'PATTERN_MATCH',
            'KNOWLEDGE_GRAPH',
            'SEMANTIC_SEARCH',
        ]
        
        best_strategy = 'MULTI_STRATEGY'
        best_quality = 0.0
        
        for strategy in strategies:
            quality = self.predict_response_quality(query, strategy)
            if quality > best_quality:
                best_quality = quality
                best_strategy = strategy
        
        return best_strategy

    def record_feedback(self, query: str, response: str, feedback_type: str):
        """Record user feedback for reinforcement learning."""
        try:
            query_hash = self._hash_query(query) if hasattr(self, '_hash_query') else hash(query)
            response_hash = self._hash_query(response) if hasattr(self, '_hash_query') else hash(response)
            
            conn = self._get_optimized_connection()
            c = conn.cursor()
            c.execute('''INSERT INTO feedback (query_hash, response_hash, feedback_type, timestamp)
                        VALUES (?, ?, ?, datetime('now'))''',
                     (query_hash, response_hash, feedback_type))
            conn.commit()
            
            # Update quality predictor
            quality_delta = 0.1 if feedback_type == 'positive' else -0.1
            strategy = self.get_best_strategy(query)
            self.update_quality_predictor(strategy, 
                max(0, min(1, self.quality_predictor.get(strategy, 0.5) + quality_delta)))
        except Exception:
            pass
