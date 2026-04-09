"""
Sacred Circuit Breaker for Fault Tolerance

Extracted from engines_infra.py during EVO_78 refactoring.
Contains: SacredCircuitBreaker, CircuitBreakerOpen, fallback helpers.
"""

import time
import logging
from typing import Callable, Any, Optional

# Sacred constants
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
OMEGA = 6539.34712682


class CircuitBreakerOpen(Exception):
    """Exception raised when circuit breaker is OPEN."""
    pass


class SacredCircuitBreaker:
    """
    Circuit breaker pattern with sacred constants.

    States:
        CLOSED: Normal operation, requests pass through
        OPEN: Failing fast, rejecting requests
        HALF_OPEN: Testing if service has recovered

    Failure threshold derived from sacred constants: GOD_CODE / PHI / 32
    """

    CB_CLOSED = 'CLOSED'
    CB_OPEN = 'OPEN'
    CB_HALF_OPEN = 'HALF_OPEN'

    def __init__(self, failure_threshold: int = None, recovery_time: float = None):
        """
        Initialize circuit breaker with sacred-derived thresholds.

        Args:
            failure_threshold: Max failures before opening (default: GOD_CODE/PHI/32)
            recovery_time: Seconds before attempting recovery (default: PHI * 10)
        """
        self.failure_threshold = failure_threshold or int(GOD_CODE / PHI / 32)  # ~10
        self.recovery_time = recovery_time or PHI * 10  # ~16.18 seconds
        self.failure_count = 0
        self.state = self.CB_CLOSED
        self._last_failure_time = 0.0
        self._half_open_successes = 0
        self._half_open_required = 2

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == self.CB_OPEN:
            if time.time() - self._last_failure_time >= self.recovery_time:
                self.state = self.CB_HALF_OPEN
                self._half_open_successes = 0
            else:
                raise CircuitBreakerOpen(f"Circuit breaker OPEN — {self.failure_count} failures recorded")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        """Record successful operation."""
        if self.state == self.CB_HALF_OPEN:
            self._half_open_successes += 1
            if self._half_open_successes >= self._half_open_required:
                self.state = self.CB_CLOSED
                self.failure_count = 0
        else:
            self.failure_count = max(0, self.failure_count - 1)

    def _on_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self._last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = self.CB_OPEN

    def get_health(self) -> dict:
        """Return circuit breaker health status."""
        return {
            'state': self.state,
            'failure_count': self.failure_count,
            'failure_threshold': self.failure_threshold,
            'recovery_time': self.recovery_time,
            'time_since_last_failure': time.time() - self._last_failure_time if self._last_failure_time else None,
            'healthy': self.state == self.CB_CLOSED
        }

    def reset(self):
        """Reset circuit breaker to closed state."""
        self.state = self.CB_CLOSED
        self.failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_successes = 0

    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        return self.state == self.CB_OPEN

    def is_closed(self) -> bool:
        """Check if circuit breaker is closed."""
        return self.state == self.CB_CLOSED


def compute_with_fallback(primary_func: Callable, fallback_func: Callable, 
                          log_degradation: bool = True) -> Any:
    """
    Execute primary function with fallback on failure.

    Args:
        primary_func: Primary function to execute
        fallback_func: Fallback function to execute if primary fails
        log_degradation: Whether to log the degradation event

    Returns:
        Result from primary_func or fallback_func
    """
    try:
        return primary_func()
    except Exception as e:
        if log_degradation:
            logging.getLogger("l104_resilience").warning(
                f"Graceful degradation triggered: {e}"
            )
        return fallback_func()


class GracefulDegradation:
    """
    Context manager for graceful degradation with circuit breaker.
    """

    def __init__(self, circuit_breaker: SacredCircuitBreaker, fallback_value: Any = None):
        self.circuit_breaker = circuit_breaker
        self.fallback_value = fallback_value

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.circuit_breaker._on_failure()
            return True  # Suppress exception
        return False

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with fallback."""
        try:
            return self.circuit_breaker.call(func, *args, **kwargs)
        except CircuitBreakerOpen:
            return self.fallback_value


# Singleton circuit breakers for common services
_gemini_breaker = None
_local_breaker = None
_quantum_breaker = None


def get_gemini_breaker() -> SacredCircuitBreaker:
    """Get singleton circuit breaker for Gemini API."""
    global _gemini_breaker
    if _gemini_breaker is None:
        _gemini_breaker = SacredCircuitBreaker(failure_threshold=5, recovery_time=30.0)
    return _gemini_breaker


def get_local_breaker() -> SacredCircuitBreaker:
    """Get singleton circuit breaker for local operations."""
    global _local_breaker
    if _local_breaker is None:
        _local_breaker = SacredCircuitBreaker(failure_threshold=10, recovery_time=10.0)
    return _local_breaker


def get_quantum_breaker() -> SacredCircuitBreaker:
    """Get singleton circuit breaker for quantum operations."""
    global _quantum_breaker
    if _quantum_breaker is None:
        _quantum_breaker = SacredCircuitBreaker(failure_threshold=3, recovery_time=60.0)
    return _quantum_breaker


__all__ = [
    'CircuitBreakerOpen',
    'SacredCircuitBreaker',
    'compute_with_fallback',
    'GracefulDegradation',
    'get_gemini_breaker',
    'get_local_breaker',
    'get_quantum_breaker',
]
