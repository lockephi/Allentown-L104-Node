"""
Unified Engine Registry

Extracted from engines_nexus.py during EVO_78 refactoring.
Contains: UnifiedEngineRegistry, TriEngineIntegration - engine registration and coordination.
"""

import threading
import time
from typing import Dict, Any, List, Optional, Callable


class UnifiedEngineRegistry:
    """
    Registry for all L104 engines with φ-weighted health scoring.
    Provides unified interface for engine discovery and health monitoring.
    """
    PHI = 1.618033988749895
    GOD_CODE = 527.5184818492612

    def __init__(self):
        self._engines: Dict[str, Any] = {}
        self._health_scores: Dict[str, float] = {}
        self._weights: Dict[str, float] = {}
        self._lock = threading.Lock()

    def register(self, name: str, engine: Any, weight: float = 1.0):
        """Register an engine with optional weight for φ-weighted scoring."""
        with self._lock:
            self._engines[name] = engine
            self._health_scores[name] = 1.0
            self._weights[name] = weight

    def unregister(self, name: str) -> bool:
        """Unregister an engine."""
        with self._lock:
            if name in self._engines:
                del self._engines[name]
                del self._health_scores[name]
                del self._weights[name]
                return True
            return False

    def get(self, name: str) -> Optional[Any]:
        """Get an engine by name."""
        return self._engines.get(name)

    def get_all(self) -> Dict[str, Any]:
        """Get all registered engines."""
        return dict(self._engines)

    def get_names(self) -> List[str]:
        """Get all engine names."""
        return list(self._engines.keys())

    def update_health(self, name: str, score: float):
        """Update health score for an engine."""
        with self._lock:
            if name in self._engines:
                self._health_scores[name] = max(0.0, min(1.0, score))

    def get_health(self, name: str) -> float:
        """Get health score for an engine."""
        return self._health_scores.get(name, 0.0)

    def get_weighted_health(self) -> float:
        """Get φ-weighted average health score."""
        with self._lock:
            if not self._weights:
                return 0.0
            
            total_weight = sum(self._weights.values())
            weighted_sum = sum(
                self._health_scores.get(name, 0.0) * self._weights.get(name, 1.0)
                for name in self._engines
            )
            return weighted_sum / total_weight if total_weight > 0 else 0.0

    def get_status(self) -> Dict[str, Any]:
        """Get registry status."""
        with self._lock:
            return {
                'engine_count': len(self._engines),
                'engines': list(self._engines.keys()),
                'health_scores': dict(self._health_scores),
                'weights': dict(self._weights),
                'weighted_health': self.get_weighted_health(),
                'phi': self.PHI,
                'god_code': self.GOD_CODE,
            }

    def call(self, name: str, method: str, *args, **kwargs) -> Any:
        """Call a method on an engine with error handling."""
        engine = self._engines.get(name)
        if engine is None:
            raise KeyError(f"Engine '{name}' not registered")
        
        method_fn = getattr(engine, method, None)
        if method_fn is None:
            raise AttributeError(f"Engine '{name}' has no method '{method}'")
        
        return method_fn(*args, **kwargs)


class TriEngineIntegration:
    """
    Integration hub for the three-engine system (Code, Science, Math).
    Coordinates analysis across all three engines with PHI-weighted scoring.
    """
    PHI = 1.618033988749895
    GOD_CODE = 527.5184818492612

    def __init__(self):
        self._code_engine = None
        self._science_engine = None
        self._math_engine = None
        self._lock = threading.Lock()

    def set_code_engine(self, engine: Any):
        """Set the Code Engine."""
        with self._lock:
            self._code_engine = engine

    def set_science_engine(self, engine: Any):
        """Set the Science Engine."""
        with self._lock:
            self._science_engine = engine

    def set_math_engine(self, engine: Any):
        """Set the Math Engine."""
        with self._lock:
            self._math_engine = engine

    def get_code_engine(self) -> Optional[Any]:
        """Get the Code Engine."""
        return self._code_engine

    def get_science_engine(self) -> Optional[Any]:
        """Get the Science Engine."""
        return self._science_engine

    def get_math_engine(self) -> Optional[Any]:
        """Get the Math Engine."""
        return self._math_engine

    def compute_unified_score(self, code_input: Any = None, 
                               science_input: Any = None,
                               math_input: Any = None) -> Dict[str, Any]:
        """Compute PHI-weighted unified score across all three engines."""
        with self._lock:
            # Weights: Code = 1.0, Science = PHI, Math = PHI^2
            weights = {
                'code': 1.0,
                'science': self.PHI,
                'math': self.PHI * self.PHI,
            }
            total_weight = sum(weights.values())

            # Get scores from each engine
            code_score = self._get_code_score(code_input)
            science_score = self._get_science_score(science_input)
            math_score = self._get_math_score(math_input)

            # PHI-weighted unified score
            weighted_sum = (
                code_score['score'] * weights['code'] +
                science_score['score'] * weights['science'] +
                math_score['score'] * weights['math']
            )
            unified_score = weighted_sum / total_weight

            return {
                'unified_score': unified_score,
                'code_score': code_score,
                'science_score': science_score,
                'math_score': math_score,
                'weights': weights,
                'phi': self.PHI,
                'god_code': self.GOD_CODE,
            }

    def _get_code_score(self, code_input: Any) -> Dict[str, Any]:
        """Get score from Code Engine."""
        if self._code_engine is None:
            return {'score': 0.5, 'confidence': 0.0, 'error': 'Code Engine not available'}
        
        try:
            if hasattr(self._code_engine, 'full_analysis'):
                result = self._code_engine.full_analysis(code_input)
                return {
                    'score': result.get('quality_score', 0.5),
                    'confidence': result.get('confidence', 0.5),
                    'complexity': result.get('complexity', {}),
                }
        except Exception as e:
            pass
        
        return {'score': 0.5, 'confidence': 0.3}

    def _get_science_score(self, science_input: Any) -> Dict[str, Any]:
        """Get score from Science Engine."""
        if self._science_engine is None:
            return {'score': 0.5, 'confidence': 0.0, 'error': 'Science Engine not available'}
        
        try:
            if hasattr(self._science_engine, 'entropy') and hasattr(self._science_engine, 'coherence'):
                entropy_eff = self._science_engine.entropy.calculate_demon_efficiency(
                    science_input.get('entropy', 0.5) if science_input else 0.5
                ) if science_input else {'efficiency': 0.5}
                coherence = self._science_engine.coherence.measure() if hasattr(self._science_engine.coherence, 'measure') else {'coherence': 0.5}
                
                return {
                    'score': (entropy_eff.get('efficiency', 0.5) + coherence.get('coherence', 0.5)) / 2,
                    'confidence': 0.7,
                    'entropy_efficiency': entropy_eff.get('efficiency', 0.5),
                    'coherence': coherence.get('coherence', 0.5),
                }
        except Exception:
            pass
        
        return {'score': 0.5, 'confidence': 0.3}

    def _get_math_score(self, math_input: Any) -> Dict[str, Any]:
        """Get score from Math Engine."""
        if self._math_engine is None:
            return {'score': 0.5, 'confidence': 0.0, 'error': 'Math Engine not available'}
        
        try:
            if hasattr(self._math_engine, 'god_code_value'):
                god_code = self._math_engine.god_code_value()
                alignment = 1.0 - abs(god_code - self.GOD_CODE) / self.GOD_CODE
                return {
                    'score': alignment,
                    'confidence': 0.8,
                    'god_code_value': god_code,
                    'alignment': alignment,
                }
        except Exception:
            pass
        
        return {'score': 0.5, 'confidence': 0.3}

    def get_status(self) -> Dict[str, Any]:
        """Get Tri-Engine status."""
        with self._lock:
            return {
                'code_engine': self._code_engine is not None,
                'science_engine': self._science_engine is not None,
                'math_engine': self._math_engine is not None,
                'ready': all([
                    self._code_engine is not None,
                    self._science_engine is not None,
                    self._math_engine is not None,
                ]),
            }


# Singleton instances
_engine_registry = None
_tri_engine = None

def get_registry() -> UnifiedEngineRegistry:
    """Get singleton UnifiedEngineRegistry instance."""
    global _engine_registry
    if _engine_registry is None:
        _engine_registry = UnifiedEngineRegistry()
    return _engine_registry

def get_tri_engine() -> TriEngineIntegration:
    """Get singleton TriEngineIntegration instance."""
    global _tri_engine
    if _tri_engine is None:
        _tri_engine = TriEngineIntegration()
    return _tri_engine


__all__ = [
    'UnifiedEngineRegistry', 'get_registry',
    'TriEngineIntegration', 'get_tri_engine',
]
