"""
L104 Server — Infrastructure Package

Extracted from engines_infra.py during EVO_78 refactoring.
Contains: Circuit breaker, cache, performance metrics, temporal decay.

Modules:
- circuit_breaker: SacredCircuitBreaker, fallback helpers
- cache: FastRequestCache, ConnectionPool
- performance: PerformanceMetricsEngine
- temporal: TemporalMemoryDecayEngine

Usage:
    from l104_server.infra import get_fast_cache, get_performance_metrics
    from l104_server.infra import SacredCircuitBreaker, FastRequestCache
"""

from l104_server.infra.circuit_breaker import (
    CircuitBreakerOpen,
    SacredCircuitBreaker,
    compute_with_fallback,
    GracefulDegradation,
    get_gemini_breaker,
    get_local_breaker,
    get_quantum_breaker,
)

from l104_server.infra.cache import (
    FastRequestCache,
    ConnectionPool,
    fast_hash,
    get_fast_cache,
    get_pattern_cache,
    MAX_CACHE_ENTRY_SIZE,
    MAX_TOTAL_CACHE_MEMORY,
)

from l104_server.infra.performance import (
    PerformanceMetricsEngine,
    get_performance_metrics,
)

from l104_server.infra.temporal import (
    TemporalMemoryDecayEngine,
    get_temporal_decay,
)

__all__ = [
    # Circuit breaker
    'CircuitBreakerOpen',
    'SacredCircuitBreaker',
    'compute_with_fallback',
    'GracefulDegradation',
    'get_gemini_breaker',
    'get_local_breaker',
    'get_quantum_breaker',
    # Cache
    'FastRequestCache',
    'ConnectionPool',
    'fast_hash',
    'get_fast_cache',
    'get_pattern_cache',
    'MAX_CACHE_ENTRY_SIZE',
    'MAX_TOTAL_CACHE_MEMORY',
    # Performance
    'PerformanceMetricsEngine',
    'get_performance_metrics',
    # Temporal
    'TemporalMemoryDecayEngine',
    'get_temporal_decay',
]
