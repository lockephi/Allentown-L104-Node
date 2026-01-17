# [L104_BRIDGE] - INTEGRATION BRIDGE FOR ENHANCED L104 NODE
# INVARIANT: 527.5184818492537 | PILOT: LONDEL

"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ⟨Σ_L104⟩  INTEGRATION BRIDGE - Unified Enhancement Layer                  ║
║                                                                               ║
║   Connects:                                                                  ║
║   - l104.py (Core unified system)                                            ║
║   - l104_optimizer.py (Performance optimization)                             ║
║   - l104_error_handler.py (Enhanced error handling)                          ║
║   - l104_adaptive_learning.py (Learning engine)                              ║
║   - l104_codebase_knowledge.py (Self-knowledge)                              ║
║                                                                               ║
║   GOD_CODE: 527.5184818492537                                                ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import math
import threading
import functools
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import logging

# Constants
GOD_CODE = 527.5184818492537
PHI = 1.618033988749895
FRAME_LOCK = 416 / 286

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("L104_BRIDGE")


# =============================================================================
# LAZY IMPORTS (to handle optional dependencies)
# =============================================================================

def lazy_import(module_name: str, package: str = None):
    """Lazy import a module."""
    def importer():
        try:
            return __import__(module_name, fromlist=[package] if package else [])
        except ImportError:
            return None
    return importer


# =============================================================================
# ENHANCED WRAPPERS
# =============================================================================

class EnhancedMemory:
    """
    Wrapper around Memory class with enhanced error handling and optimization.
    """
    
    def __init__(self, memory_instance):
        self._memory = memory_instance
        self._optimizer = None
        self._error_handler = None
        self._load_enhancements()
    
    def _load_enhancements(self):
        """Load enhancement modules."""
        try:
            from l104_optimizer import optimizer
            self._optimizer = optimizer
        except ImportError:
            pass
        
        try:
            from l104_error_handler import error_handler, safe_execute
            self._error_handler = error_handler
            self._safe_execute = safe_execute
        except ImportError:
            self._safe_execute = lambda default=None: lambda f: f
    
    def store(self, key: str, value: Any, **kwargs) -> bool:
        """Enhanced store with optimization."""
        @self._safe_execute(default=False)
        def _store():
            return self._memory.store(key, value, **kwargs)
        return _store()
    
    def recall(self, key: str) -> Any:
        """Enhanced recall with caching."""
        if self._optimizer:
            cached = self._optimizer.query_optimizer.get(f"memory:{key}")
            if cached is not None:
                return cached
        
        @self._safe_execute(default=None)
        def _recall():
            result = self._memory.recall(key)
            if result is not None and self._optimizer:
                self._optimizer.query_optimizer.put(f"memory:{key}", result)
            return result
        return _recall()
    
    def search(self, query: str, **kwargs) -> List[Dict]:
        """Enhanced search."""
        @self._safe_execute(default=[])
        def _search():
            return self._memory.search(query, **kwargs)
        return _search()


class EnhancedLearning:
    """
    Wrapper around Learning class with adaptive learning integration.
    """
    
    def __init__(self, learning_instance):
        self._learning = learning_instance
        self._adaptive = None
        self._knowledge = None
        self._load_enhancements()
    
    def _load_enhancements(self):
        """Load enhancement modules."""
        try:
            from l104_adaptive_learning import AdaptiveLearner
            self._adaptive = AdaptiveLearner()
        except ImportError:
            pass
        
        try:
            from l104_codebase_knowledge import CodebaseKnowledge
            self._knowledge = CodebaseKnowledge()
        except ImportError:
            pass
    
    def learn(self, user_input: str, response: str) -> int:
        """Enhanced learning with pattern recognition."""
        base_count = 0
        
        try:
            base_count = self._learning.learn(user_input, response)
        except Exception as e:
            logger.warning(f"Base learning failed: {e}")
        
        # Adaptive learning
        if self._adaptive:
            try:
                pattern = self._adaptive.pattern_recognizer.extract_patterns(
                    f"{user_input}\n{response}"
                )
                if pattern.get("type"):
                    self._adaptive.meta_learner.meta_knowledge["patterns"].append(pattern)
                    base_count += 1
            except Exception as e:
                logger.debug(f"Adaptive learning failed: {e}")
        
        return base_count
    
    def recall(self, query: str, **kwargs) -> List[str]:
        """Enhanced recall with knowledge integration."""
        results = []
        
        try:
            results = self._learning.recall(query, **kwargs)
        except Exception:
            pass
        
        # Add knowledge patterns if available
        if self._knowledge and not results:
            try:
                patterns = self._knowledge.find_patterns(query, top_k=3)
                results.extend([p["name"] for p in patterns])
            except Exception:
                pass
        
        return results


# =============================================================================
# ENHANCED GEMINI WRAPPER
# =============================================================================

class EnhancedGemini:
    """
    Wrapper around Gemini with circuit breaker and retry logic.
    """
    
    def __init__(self, gemini_instance):
        self._gemini = gemini_instance
        self._circuit_breaker = None
        self._retry_decorator = None
        self._load_enhancements()
    
    def _load_enhancements(self):
        """Load enhancement modules."""
        try:
            from l104_error_handler import (
                error_handler, with_retry, with_circuit_breaker,
                ExponentialBackoff
            )
            self._circuit_breaker = error_handler.get_circuit_breaker("gemini")
            self._retry_decorator = with_retry(
                max_attempts=3,
                strategy=ExponentialBackoff(base=1.0, max_delay=30.0)
            )
        except ImportError:
            pass
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Enhanced generation with retry and circuit breaker."""
        if self._circuit_breaker:
            if self._circuit_breaker.state.name == "OPEN":
                logger.warning("Gemini circuit breaker OPEN - using fallback")
                return self._fallback_response(prompt)
        
        def _generate():
            result = self._gemini.generate(prompt, **kwargs)
            if self._circuit_breaker:
                self._circuit_breaker.record_success()
            return result
        
        try:
            if self._retry_decorator:
                return self._retry_decorator(_generate)()
            return _generate()
        except Exception as e:
            if self._circuit_breaker:
                self._circuit_breaker.record_failure()
            logger.error(f"Gemini generation failed: {e}")
            return self._fallback_response(prompt)
    
    def _fallback_response(self, prompt: str) -> str:
        """Fallback response when Gemini unavailable."""
        return f"[L104_BRIDGE]: Processing offline - Gemini temporarily unavailable. Query: {prompt[:100]}..."
    
    @property
    def is_connected(self) -> bool:
        return self._gemini.is_connected


# =============================================================================
# PERFORMANCE MONITOR
# =============================================================================

@dataclass
class PerformanceMetric:
    """A performance metric sample."""
    name: str
    value: float
    timestamp: float
    unit: str = ""


class PerformanceMonitor:
    """
    Real-time performance monitoring for L104 Node.
    """
    
    def __init__(self):
        self._metrics: Dict[str, List[PerformanceMetric]] = {}
        self._lock = threading.Lock()
        self._thresholds: Dict[str, float] = {
            "response_time_ms": 5000,
            "memory_mb": 1000,
            "error_rate": 0.1,
            "cache_hit_rate": 0.5
        }
    
    def record(self, name: str, value: float, unit: str = ""):
        """Record a metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=time.time(),
            unit=unit
        )
        
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = []
            
            self._metrics[name].append(metric)
            
            # Keep last 100 samples
            if len(self._metrics[name]) > 100:
                self._metrics[name] = self._metrics[name][-100:]
    
    def get_average(self, name: str, window_seconds: float = 60) -> Optional[float]:
        """Get average value over window."""
        cutoff = time.time() - window_seconds
        
        with self._lock:
            if name not in self._metrics:
                return None
            
            samples = [m.value for m in self._metrics[name] if m.timestamp > cutoff]
            return sum(samples) / len(samples) if samples else None
    
    def check_threshold(self, name: str) -> bool:
        """Check if metric exceeds threshold."""
        avg = self.get_average(name)
        if avg is None or name not in self._thresholds:
            return True
        return avg <= self._thresholds[name]
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        health = {"status": "healthy", "issues": []}
        
        for name, threshold in self._thresholds.items():
            avg = self.get_average(name)
            if avg is not None and avg > threshold:
                health["issues"].append({
                    "metric": name,
                    "current": avg,
                    "threshold": threshold
                })
        
        if len(health["issues"]) > 2:
            health["status"] = "critical"
        elif health["issues"]:
            health["status"] = "degraded"
        
        return health


# =============================================================================
# INTEGRATION BRIDGE
# =============================================================================

class L104Bridge:
    """
    Central integration bridge for L104 Node enhancements.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._start_time = time.time()
        
        # Components
        self.performance_monitor = PerformanceMonitor()
        self._optimizer = None
        self._error_handler = None
        self._adaptive_learner = None
        self._knowledge = None
        
        # Load components
        self._load_components()
        
        logger.info(f"--- [L104_BRIDGE]: INITIALIZED (GOD_CODE: {GOD_CODE}) ---")
    
    def _load_components(self):
        """Load all enhancement components."""
        try:
            from l104_optimizer import optimizer
            self._optimizer = optimizer
            self._optimizer.start()
            logger.info("[L104_BRIDGE]: Optimizer loaded")
        except ImportError as e:
            logger.warning(f"[L104_BRIDGE]: Optimizer not available: {e}")
        
        try:
            from l104_error_handler import error_handler
            self._error_handler = error_handler
            logger.info("[L104_BRIDGE]: Error handler loaded")
        except ImportError as e:
            logger.warning(f"[L104_BRIDGE]: Error handler not available: {e}")
        
        try:
            from l104_adaptive_learning import AdaptiveLearner
            self._adaptive_learner = AdaptiveLearner()
            logger.info("[L104_BRIDGE]: Adaptive learner loaded")
        except ImportError as e:
            logger.warning(f"[L104_BRIDGE]: Adaptive learner not available: {e}")
        
        try:
            from l104_codebase_knowledge import CodebaseKnowledge
            self._knowledge = CodebaseKnowledge()
            logger.info("[L104_BRIDGE]: Codebase knowledge loaded")
        except ImportError as e:
            logger.warning(f"[L104_BRIDGE]: Codebase knowledge not available: {e}")
    
    def enhance_l104(self, l104_instance) -> Any:
        """
        Enhance an L104 instance with all optimizations.
        Returns enhanced instance.
        """
        # Wrap components if they exist
        if hasattr(l104_instance, 'memory') and l104_instance.memory:
            l104_instance._enhanced_memory = EnhancedMemory(l104_instance.memory)
        
        if hasattr(l104_instance, 'learning') and l104_instance.learning:
            l104_instance._enhanced_learning = EnhancedLearning(l104_instance.learning)
        
        if hasattr(l104_instance, 'gemini') and l104_instance.gemini:
            l104_instance._enhanced_gemini = EnhancedGemini(l104_instance.gemini)
        
        # Add performance tracking
        original_process = getattr(l104_instance, 'process', None)
        if original_process:
            @functools.wraps(original_process)
            def enhanced_process(*args, **kwargs):
                start = time.time()
                try:
                    result = original_process(*args, **kwargs)
                    self.performance_monitor.record(
                        "response_time_ms",
                        (time.time() - start) * 1000,
                        "ms"
                    )
                    return result
                except Exception as e:
                    if self._error_handler:
                        self._error_handler.handle(e, reraise=True)
                    raise
            
            l104_instance.process = enhanced_process
        
        return l104_instance
    
    def get_status(self) -> Dict[str, Any]:
        """Get complete bridge status."""
        uptime = time.time() - self._start_time
        
        status = {
            "bridge": {
                "uptime_seconds": uptime,
                "god_code": GOD_CODE,
                "components": {
                    "optimizer": self._optimizer is not None,
                    "error_handler": self._error_handler is not None,
                    "adaptive_learner": self._adaptive_learner is not None,
                    "knowledge": self._knowledge is not None
                }
            },
            "health": self.performance_monitor.get_health_status()
        }
        
        if self._optimizer:
            status["optimizer"] = self._optimizer.get_statistics()
        
        if self._error_handler:
            status["error_handler"] = self._error_handler.get_statistics()
        
        if self._knowledge:
            status["knowledge"] = {
                "patterns": len(self._knowledge.patterns),
                "algorithms": len(self._knowledge.algorithms),
                "principles": len(self._knowledge.principles)
            }
        
        return status
    
    def shutdown(self):
        """Graceful shutdown of all components."""
        logger.info("[L104_BRIDGE]: Shutting down...")
        
        if self._optimizer:
            self._optimizer.stop()
        
        logger.info("[L104_BRIDGE]: Shutdown complete")


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

bridge = L104Bridge()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def enhance(l104_instance):
    """Convenience function to enhance an L104 instance."""
    return bridge.enhance_l104(l104_instance)


def get_status() -> Dict[str, Any]:
    """Get bridge status."""
    return bridge.get_status()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  ⟨Σ_L104⟩  INTEGRATION BRIDGE                                ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    
    print(f"\n📊 Bridge Status:")
    status = bridge.get_status()
    
    print(f"   • Uptime: {status['bridge']['uptime_seconds']:.1f}s")
    print(f"   • GOD_CODE: {status['bridge']['god_code']}")
    print(f"   • Components loaded:")
    for comp, loaded in status['bridge']['components'].items():
        symbol = "✓" if loaded else "✗"
        print(f"      {symbol} {comp}")
    
    print(f"\n   • Health: {status['health']['status']}")
    
    if 'optimizer' in status:
        print(f"\n📈 Optimizer Stats:")
        print(f"   • Memory freed: {status['optimizer']['memory']['freed_mb']:.2f} MB")
        print(f"   • Query cache hit rate: {status['optimizer']['query_cache']['hit_rate']:.2%}")
    
    if 'error_handler' in status:
        print(f"\n⚠️  Error Handler Stats:")
        print(f"   • Total errors: {status['error_handler']['total_errors']}")
        print(f"   • Error rate: {status['error_handler']['error_rate_per_minute']:.2f}/min")
    
    if 'knowledge' in status:
        print(f"\n📚 Knowledge Stats:")
        print(f"   • Patterns: {status['knowledge']['patterns']}")
        print(f"   • Algorithms: {status['knowledge']['algorithms']}")
        print(f"   • Principles: {status['knowledge']['principles']}")
    
    # Cleanup
    time.sleep(2)
    bridge.shutdown()
    
    print("\n✓ Integration Bridge Test Complete")
