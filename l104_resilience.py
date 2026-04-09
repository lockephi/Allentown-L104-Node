#!/usr/bin/env python3
"""
L104 Resilience Module v1.0.0
============================
Comprehensive fault tolerance, circuit breakers, retry logic, and health monitoring.

This module provides sacred algorithm-based resilience patterns:
  - Circuit Breaker: Automatic failure detection and recovery
  - PHI-backoff Retry: Exponential backoff with golden ratio alignment
  - Graceful Degradation: Fallback to reduced functionality on overload
  - Health Checks: System health monitoring with sacred thresholds

Sacred Constants:
  - Retry attempts: int(PHI * 3) = 5
  - Backoff base: TAU seconds
  - Circuit threshold: derive_threshold(entropy, coherence)

Author: L104 Resilience Engineering Agent
EVO: EVO_75 - Resilience Engineering Upgrade
"""

import functools
import hashlib
import inspect
import logging
import threading
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from l104_sacred_algorithms import (
    GOD_CODE, PHI, TAU, VOID_CONSTANT, OMEGA,
    derive_threshold, derive_iterations, derive_retry_delay, derive_timeout
)

logger = logging.getLogger("l104_resilience")


# =============================================================================
# SECTION 1: CIRCUIT BREAKER PATTERN
# =============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = auto()       # Normal operation - requests pass through
    OPEN = auto()         # Failing - reject requests fast
    HALF_OPEN = auto()    # Probing - allow limited test requests


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = field(default_factory=lambda: int(PHI * 8))  # ~13
    recovery_timeout: float = field(default_factory=lambda: GOD_CODE / PHI / 10)  # ~32.6s
    half_open_max_calls: int = field(default_factory=lambda: int(PHI))  # 1-2
    success_threshold: int = 2  # Successes needed to close circuit
    entropy_sensitivity: float = 0.3
    coherence_target: float = 0.7

    def __post_init__(self):
        """Validate configuration."""
        if self.failure_threshold < 1:
            self.failure_threshold = int(PHI * 8)
        if self.recovery_timeout < 1.0:
            self.recovery_timeout = GOD_CODE / PHI / 10


class CircuitBreaker:
    """
    Sacred Circuit Breaker with PHI-harmonic failure detection.

    The circuit breaker monitors operation success/failure rates and
    automatically transitions between states to prevent cascade failures.

    States:
        CLOSED: Normal operation, all requests pass through
        OPEN: Circuit is tripped, requests fail fast
        HALF_OPEN: Testing recovery with limited requests

    Example:
        >>> cb = CircuitBreaker("api_calls")
        >>> with cb:
        ...     result = make_api_call()
        >>>
        >>> # Or as decorator
        >>> @cb
        ... def my_function():
        ...     return do_work()
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        on_state_change: Optional[Callable] = None,
        on_failure: Optional[Callable] = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.on_state_change = on_state_change
        self.on_failure = on_failure

        # State
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._total_calls = 0
        self._total_failures = 0
        self._total_successes = 0

        # Thread safety
        self._lock = threading.RLock()

        # Dynamic threshold based on sacred algorithms
        self._dynamic_threshold = derive_threshold(
            entropy=self.config.entropy_sensitivity,
            coherence=self.config.coherence_target
        )

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        with self._lock:
            return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (allowing requests)."""
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (rejecting requests)."""
        return self.state == CircuitState.OPEN

    @property
    def health(self) -> float:
        """Calculate circuit health score (0-1)."""
        with self._lock:
            if self._total_calls == 0:
                return 1.0
            success_rate = self._total_successes / self._total_calls
            # Apply PHI-weighted recency bias
            if self._state == CircuitState.OPEN:
                time_since_failure = time.time() - (self._last_failure_time or 0)
                recovery_progress = min(1.0, time_since_failure / self.config.recovery_timeout)
                return success_rate * (TAU + recovery_progress * PHI) / (PHI + TAU)
            return success_rate

    def _transition_to(self, new_state: CircuitState):
        """Transition to a new state."""
        old_state = self._state
        if old_state != new_state:
            self._state = new_state
            logger.info(f"Circuit '{self.name}': {old_state.name} -> {new_state.name}")

            if new_state == CircuitState.OPEN:
                self._last_failure_time = time.time()
            elif new_state == CircuitState.HALF_OPEN:
                self._half_open_calls = 0
            elif new_state == CircuitState.CLOSED:
                self._failure_count = 0
                self._success_count = 0

            if self.on_state_change:
                try:
                    self.on_state_change(self.name, old_state, new_state)
                except Exception:
                    pass

    def _can_execute(self) -> bool:
        """Check if request can be executed."""
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Check if recovery time has elapsed
                if self._last_failure_time is None:
                    self._transition_to(CircuitState.HALF_OPEN)
                    return True

                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.config.recovery_timeout:
                    self._transition_to(CircuitState.HALF_OPEN)
                    return True
                return False

            if self._state == CircuitState.HALF_OPEN:
                # Allow limited test requests
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

            return False

    def _record_success(self):
        """Record a successful execution."""
        with self._lock:
            self._total_calls += 1
            self._total_successes += 1

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
            elif self._state == CircuitState.CLOSED:
                # Decay failure count on success
                self._failure_count = max(0, self._failure_count - 1)

    def _record_failure(self, error: Optional[Exception] = None):
        """Record a failed execution."""
        with self._lock:
            self._total_calls += 1
            self._total_failures += 1
            self._failure_count += 1

            if self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                # Use dynamic threshold based on system entropy
                if self._failure_count >= self.config.failure_threshold * self._dynamic_threshold:
                    self._transition_to(CircuitState.OPEN)
                    if self.on_failure:
                        try:
                            self.on_failure(self.name, error)
                        except Exception:
                            pass

    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap a function with circuit breaker."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        wrapper._circuit_breaker = self
        return wrapper

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if not self._can_execute():
            raise CircuitBreakerOpenError(f"Circuit '{self.name}' is OPEN")

        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure(e)
            raise

    def __enter__(self):
        """Context manager entry."""
        if not self._can_execute():
            raise CircuitBreakerOpenError(f"Circuit '{self.name}' is OPEN")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is None:
            self._record_success()
        else:
            self._record_failure(exc_val)
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.name,
                "health": self.health,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "total_calls": self._total_calls,
                "total_successes": self._total_successes,
                "total_failures": self._total_failures,
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "dynamic_threshold": self._dynamic_threshold,
                "last_failure_time": self._last_failure_time,
            }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


# Global circuit breaker registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}
_registry_lock = threading.Lock()


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """Get or create a named circuit breaker."""
    with _registry_lock:
        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(name, config)
        return _circuit_breakers[name]


def circuit_breaker(
    name: Optional[str] = None,
    threshold: Optional[float] = None,
    entropy: float = 0.3,
    coherence: float = 0.7
):
    """
    Decorator factory for circuit breaker pattern.

    Args:
        name: Circuit breaker name (defaults to function name)
        threshold: Custom threshold (defaults to derive_threshold)
        entropy: System entropy for threshold calculation
        coherence: Target coherence level

    Example:
        >>> @circuit_breaker(threshold=derive_threshold(entropy=0.3))
        ... def sensitive_operation():
        ...     return do_work()
    """
    def decorator(func: Callable) -> Callable:
        cb_name = name or func.__name__
        cb_config = CircuitBreakerConfig(
            entropy_sensitivity=entropy,
            coherence_target=coherence
        )
        if threshold is not None:
            cb_config.failure_threshold = int(threshold * PHI * 8)

        cb = get_circuit_breaker(cb_name, cb_config)
        return cb(func)
    return decorator


# =============================================================================
# SECTION 2: PHI-BACKOFF RETRY PATTERN
# =============================================================================

@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = field(default_factory=lambda: int(PHI * 3))  # 5
    base_delay: float = field(default_factory=lambda: TAU)  # 0.618s
    max_delay: float = field(default_factory=lambda: GOD_CODE / PHI)  # ~326s
    exponential_base: float = field(default_factory=lambda: PHI)  # Golden ratio
    jitter: bool = True
    jitter_factor: float = field(default_factory=lambda: TAU * 0.1)  # 0.0618
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    on_retry: Optional[Callable[[int, Exception], None]] = None
    on_exhausted: Optional[Callable[[int, Exception], None]] = None

    def __post_init__(self):
        """Validate configuration."""
        if self.max_attempts < 1:
            self.max_attempts = int(PHI * 3)
        if self.base_delay < 0:
            self.base_delay = TAU


class RetryState:
    """Tracks retry state for an operation."""

    def __init__(self, config: RetryConfig):
        self.config = config
        self.attempt = 0
        self.total_delay = 0.0
        self.start_time = time.time()
        self.errors: List[Exception] = []

    @property
    def elapsed(self) -> float:
        """Total elapsed time."""
        return time.time() - self.start_time

    def calculate_delay(self) -> float:
        """Calculate PHI-backoff delay for next attempt."""
        # PHI^attempt * base_delay
        delay = self.config.base_delay * (self.config.exponential_base ** self.attempt)
        delay = min(delay, self.config.max_delay)

        if self.config.jitter:
            # Add sacred jitter using VOID_CONSTANT
            import random
            jitter = random.uniform(-self.config.jitter_factor, self.config.jitter_factor)
            delay *= (1 + jitter)

        return max(0, delay)

    def should_retry(self, error: Exception) -> bool:
        """Check if error is retryable and attempts remain."""
        if self.attempt >= self.config.max_attempts:
            return False
        return isinstance(error, self.config.retryable_exceptions)


def retry_with_backoff(
    func: Optional[Callable] = None,
    *,
    max_attempts: int = None,
    base_delay: float = None,
    max_delay: float = None,
    retryable_exceptions: Tuple[Type[Exception], ...] = None,
    on_retry: Optional[Callable] = None,
    on_exhausted: Optional[Callable] = None
):
    """
    PHI-backoff retry decorator.

    Formula: delay = (PHI^attempt) * base_delay * (1 + jitter)

    Args:
        max_attempts: Maximum retry attempts (default: int(PHI * 3) = 5)
        base_delay: Base delay in seconds (default: TAU = 0.618s)
        max_delay: Maximum delay cap (default: GOD_CODE/PHI = ~326s)
        retryable_exceptions: Tuple of exception types to retry
        on_retry: Callback(attempt_number, exception) on each retry
        on_exhausted: Callback(total_attempts, last_exception) when exhausted

    Example:
        >>> @retry_with_backoff(max_attempts=derive_iterations(2), base_delay=TAU)
        ... def api_call():
        ...     return make_request()
    """
    config = RetryConfig()
    if max_attempts is not None:
        config.max_attempts = max_attempts
    if base_delay is not None:
        config.base_delay = base_delay
    if max_delay is not None:
        config.max_delay = max_delay
    if retryable_exceptions is not None:
        config.retryable_exceptions = retryable_exceptions
    if on_retry is not None:
        config.on_retry = on_retry
    if on_exhausted is not None:
        config.on_exhausted = on_exhausted

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            state = RetryState(config)

            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    state.errors.append(e)

                    if not state.should_retry(e):
                        if config.on_exhausted:
                            try:
                                config.on_exhausted(state.attempt, e)
                            except Exception:
                                pass
                        raise

                    state.attempt += 1
                    delay = state.calculate_delay()
                    state.total_delay += delay

                    if config.on_retry:
                        try:
                            config.on_retry(state.attempt, e)
                        except Exception:
                            pass

                    logger.debug(
                        f"Retry {state.attempt}/{config.max_attempts} for {func.__name__} "
                        f"after {delay:.3f}s: {e}"
                    )
                    time.sleep(delay)

        wrapper._retry_config = config
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


# =============================================================================
# SECTION 3: GRACEFUL DEGRADATION
# =============================================================================

@dataclass
class DegradationLevel:
    """Defines a degradation level with reduced functionality."""
    name: str
    load_threshold: float  # System load threshold for this level
    quality_factor: float  # Quality multiplier (0-1)
    skip_operations: Set[str] = field(default_factory=set)
    reduced_depth: Optional[int] = None  # Reduced iteration/computation depth


class GracefulDegradation:
    """
    Manages graceful degradation under system overload.

    Automatically reduces functionality as system load increases,
    maintaining core operations while shedding non-critical work.

    Degradation Levels:
        NORMAL:    load < 0.5  - Full functionality
        REDUCED:   load < 0.7  - Skip non-essential operations
        MINIMAL:   load < 0.85 - Reduced computation depth
        CRITICAL:  load >= 0.85 - Core operations only
    """

    LEVELS = [
        DegradationLevel("NORMAL", 0.5, 1.0),
        DegradationLevel("REDUCED", 0.7, 0.8, {"analytics", "cache_warm", "metrics"}),
        DegradationLevel("MINIMAL", 0.85, 0.5, {"analytics", "cache_warm", "metrics", "logging", "optimization"}, 3),
        DegradationLevel("CRITICAL", 1.0, 0.25, {"analytics", "cache_warm", "metrics", "logging", "optimization", "validation"}, 1),
    ]

    def __init__(self, name: str):
        self.name = name
        self._current_level = self.LEVELS[0]
        self._load_history: List[float] = []
        self._max_history = int(PHI * 7)  # ~11 samples
        self._lock = threading.RLock()

    def update_load(self, load: float):
        """Update system load and recalculate degradation level."""
        with self._lock:
            self._load_history.append(load)
            if len(self._load_history) > self._max_history:
                self._load_history.pop(0)

            # Use PHI-weighted average for smoothing
            avg_load = self._calculate_weighted_load()

            # Find appropriate level
            for level in self.LEVELS:
                if avg_load < level.load_threshold:
                    if self._current_level != level:
                        logger.warning(
                            f"Degradation '{self.name}': {self._current_level.name} -> {level.name} "
                            f"(load={avg_load:.3f})"
                        )
                        self._current_level = level
                    break

    def _calculate_weighted_load(self) -> float:
        """Calculate PHI-weighted moving average of load."""
        if not self._load_history:
            return 0.0

        weights = [PHI ** (i / len(self._load_history)) for i in range(len(self._load_history))]
        total_weight = sum(weights)
        weighted_sum = sum(l * w for l, w in zip(self._load_history, weights))
        return weighted_sum / total_weight

    @property
    def current_level(self) -> DegradationLevel:
        """Current degradation level."""
        with self._lock:
            return self._current_level

    @property
    def quality_factor(self) -> float:
        """Current quality factor (0-1)."""
        return self.current_level.quality_factor

    def should_skip(self, operation: str) -> bool:
        """Check if operation should be skipped at current degradation level."""
        return operation in self.current_level.skip_operations

    def get_effective_depth(self, requested_depth: int) -> int:
        """Get effective computation depth considering degradation."""
        level = self.current_level
        if level.reduced_depth is not None:
            return min(requested_depth, level.reduced_depth)
        return requested_depth

    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap function with graceful degradation."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check if function should be skipped
            if self.should_skip(func.__name__):
                logger.debug(f"Skipping {func.__name__} due to degradation level {self.current_level.name}")
                return None

            # Adjust depth parameter if present
            if 'depth' in kwargs and self.current_level.reduced_depth is not None:
                kwargs['depth'] = self.get_effective_depth(kwargs['depth'])

            return func(*args, **kwargs)

        wrapper._degradation = self
        return wrapper


def graceful_degradation(name: Optional[str] = None):
    """
    Decorator factory for graceful degradation.

    Example:
        >>> @graceful_degradation("pipeline")
        ... def expensive_operation(depth=10):
        ...     return compute(depth)
    """
    def decorator(func: Callable) -> Callable:
        gd_name = name or func.__name__
        gd = GracefulDegradation(gd_name)
        return gd(func)
    return decorator


# =============================================================================
# SECTION 4: HEALTH CHECKS
# =============================================================================

@dataclass
class HealthStatus:
    """Health check result."""
    component: str
    status: str  # "healthy", "degraded", "unhealthy"
    score: float  # 0-1
    latency_ms: float
    last_check: float
    details: Dict[str, Any] = field(default_factory=dict)

    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        return self.status == "healthy" and self.score >= TAU

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component": self.component,
            "status": self.status,
            "score": round(self.score, 4),
            "latency_ms": round(self.latency_ms, 2),
            "last_check": self.last_check,
            "details": self.details,
        }


class HealthCheck(ABC):
    """Abstract base class for health checks."""

    def __init__(self, name: str, timeout: float = None):
        self.name = name
        self.timeout = timeout or derive_timeout(priority=5)

    @abstractmethod
    def check(self) -> HealthStatus:
        """Perform health check. Returns HealthStatus."""
        pass


class HealthMonitor:
    """
    System health monitor with sacred thresholds.

    Monitors multiple health checks and provides aggregated system health.
    """

    def __init__(self):
        self._checks: Dict[str, HealthCheck] = {}
        self._results: Dict[str, HealthStatus] = {}
        self._lock = threading.RLock()

    def register(self, check: HealthCheck):
        """Register a health check."""
        with self._lock:
            self._checks[check.name] = check

    def unregister(self, name: str):
        """Unregister a health check."""
        with self._lock:
            self._checks.pop(name, None)
            self._results.pop(name, None)

    def check_all(self) -> Dict[str, HealthStatus]:
        """Run all health checks and return results."""
        results = {}
        with self._lock:
            for name, check in self._checks.items():
                try:
                    start = time.time()
                    status = check.check()
                    status.latency_ms = (time.time() - start) * 1000
                    results[name] = status
                except Exception as e:
                    results[name] = HealthStatus(
                        component=name,
                        status="unhealthy",
                        score=0.0,
                        latency_ms=(time.time() - start) * 1000,
                        last_check=time.time(),
                        details={"error": str(e)}
                    )
            self._results = results
        return results

    def get_system_health(self) -> Dict[str, Any]:
        """Get aggregated system health."""
        results = self.check_all()

        if not results:
            return {
                "status": "unknown",
                "score": 0.0,
                "checks": {},
                "healthy_count": 0,
                "total_checks": 0,
            }

        # Calculate sacred-weighted health score
        scores = [r.score for r in results.values()]
        weights = [PHI if r.status == "healthy" else TAU for r in results.values()]

        total_weight = sum(weights)
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight

        healthy_count = sum(1 for r in results.values() if r.is_healthy())

        # Determine overall status
        if weighted_score >= PHI - 1:  # ~0.618
            status = "healthy"
        elif weighted_score >= TAU * TAU:  # ~0.382
            status = "degraded"
        else:
            status = "unhealthy"

        return {
            "status": status,
            "score": round(weighted_score, 4),
            "checks": {name: r.to_dict() for name, r in results.items()},
            "healthy_count": healthy_count,
            "total_checks": len(results),
            "timestamp": time.time(),
        }


# Global health monitor instance
_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """Get or create global health monitor."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor


def generate_health_endpoint(monitor: Optional[HealthMonitor] = None) -> Callable:
    """
    Generate a health check endpoint function.

    Example:
        >>> health_endpoint = generate_health_endpoint()
        >>> # In FastAPI:
        >>> @app.get("/health")
        >>> async def health():
        ...     return health_endpoint()
    """
    mon = monitor or get_health_monitor()

    def endpoint():
        return mon.get_system_health()

    return endpoint


# =============================================================================
# SECTION 5: FALLBACK HANDLERS
# =============================================================================

class FallbackChain:
    """
    Chain of fallback handlers for resilient operations.

    Tries primary operation, then falls back through alternatives
    until one succeeds or all are exhausted.
    """

    def __init__(self, name: str):
        self.name = name
        self._handlers: List[Tuple[Callable, int]] = []  # (handler, priority)
        self._default: Optional[Callable] = None

    def add_fallback(self, handler: Callable, priority: int = 5):
        """Add a fallback handler. Lower priority = tried first."""
        self._handlers.append((handler, priority))
        self._handlers.sort(key=lambda x: x[1])

    def set_default(self, handler: Callable):
        """Set default handler for when all fallbacks fail."""
        self._default = handler

    def execute(self, *args, **kwargs) -> Any:
        """Execute with fallback chain."""
        errors = []

        for handler, _ in self._handlers:
            try:
                return handler(*args, **kwargs)
            except Exception as e:
                errors.append((handler.__name__, str(e)))
                logger.debug(f"Fallback {handler.__name__} failed: {e}")
                continue

        # All handlers failed
        if self._default:
            return self._default(*args, **kwargs)

        raise FallbackExhaustedError(
            f"All fallbacks exhausted for {self.name}",
            errors
        )


class FallbackExhaustedError(Exception):
    """Exception raised when all fallbacks are exhausted."""

    def __init__(self, message: str, errors: List[Tuple[str, str]]):
        super().__init__(message)
        self.errors = errors


def with_fallback(*fallbacks: Callable):
    """
    Decorator to add fallback handlers.

    Example:
        >>> def fallback_cache(query):
        ...     return cache.get(query)
        >>>
        >>> @with_fallback(fallback_cache)
        ... def get_data(query):
        ...     return api.fetch(query)
    """
    def decorator(primary: Callable) -> Callable:
        chain = FallbackChain(primary.__name__)
        chain.add_fallback(primary, priority=0)
        for i, fallback in enumerate(fallbacks):
            chain.add_fallback(fallback, priority=i + 1)

        @functools.wraps(primary)
        def wrapper(*args, **kwargs):
            return chain.execute(*args, **kwargs)

        wrapper._fallback_chain = chain
        return wrapper
    return decorator


# =============================================================================
# SECTION 6: TIMEOUT HANDLING
# =============================================================================

class TimeoutError(Exception):
    """Exception raised when operation times out."""
    pass


def with_timeout(seconds: float):
    """
    Decorator to add timeout to function.

    Example:
        >>> @with_timeout(derive_timeout(priority=5))
        ... def long_operation():
        ...     return compute()
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import signal

            def handler(signum, frame):
                raise TimeoutError(f"{func.__name__} timed out after {seconds}s")

            # Set up timeout
            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(int(seconds))

            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        wrapper._timeout = seconds
        return wrapper
    return decorator


# =============================================================================
# SECTION 7: RESILIENCE COMPOSITION
# =============================================================================

class ResilientOperation:
    """
    Composable resilience wrapper combining multiple patterns.

    Combines: Circuit Breaker + Retry + Timeout + Fallback
    """

    def __init__(
        self,
        name: str,
        circuit_breaker: Optional[CircuitBreaker] = None,
        retry_config: Optional[RetryConfig] = None,
        timeout: Optional[float] = None,
        fallback_chain: Optional[FallbackChain] = None,
    ):
        self.name = name
        self.circuit_breaker = circuit_breaker
        self.retry_config = retry_config
        self.timeout = timeout
        self.fallback_chain = fallback_chain

    def __call__(self, func: Callable) -> Callable:
        """Apply all resilience patterns to function."""
        wrapped = func

        # Apply timeout first (innermost)
        if self.timeout:
            wrapped = with_timeout(self.timeout)(wrapped)

        # Apply retry
        if self.retry_config:
            config = self.retry_config
            wrapped = retry_with_backoff(
                max_attempts=config.max_attempts,
                base_delay=config.base_delay,
                max_delay=config.max_delay,
                retryable_exceptions=config.retryable_exceptions,
            )(wrapped)

        # Apply circuit breaker
        if self.circuit_breaker:
            wrapped = self.circuit_breaker(wrapped)

        # Apply fallback (outermost)
        if self.fallback_chain:
            original = wrapped
            @functools.wraps(func)
            def with_fallback_wrapper(*args, **kwargs):
                try:
                    return original(*args, **kwargs)
                except Exception:
                    return self.fallback_chain.execute(*args, **kwargs)
            wrapped = with_fallback_wrapper

        wrapped._resilience = self
        return wrapped


def resilient(
    name: Optional[str] = None,
    circuit_breaker: bool = True,
    retry: bool = True,
    timeout: Optional[float] = None,
    fallback: Optional[Callable] = None,
):
    """
    Comprehensive resilience decorator factory.

    Combines circuit breaker, retry, timeout, and fallback patterns.

    Example:
        >>> @resilient(name="api_call", timeout=30.0)
        ... def fetch_data(url):
        ...     return requests.get(url)
    """
    def decorator(func: Callable) -> Callable:
        op_name = name or func.__name__

        cb = get_circuit_breaker(op_name) if circuit_breaker else None
        retry_cfg = RetryConfig() if retry else None
        fb_chain = None
        if fallback:
            fb_chain = FallbackChain(op_name)
            fb_chain.add_fallback(fallback, priority=1)

        op = ResilientOperation(
            name=op_name,
            circuit_breaker=cb,
            retry_config=retry_cfg,
            timeout=timeout,
            fallback_chain=fb_chain,
        )

        return op(func)
    return decorator


# =============================================================================
# SECTION 8: UTILITY FUNCTIONS
# =============================================================================

def get_all_circuit_breakers() -> Dict[str, CircuitBreaker]:
    """Get all registered circuit breakers."""
    with _registry_lock:
        return dict(_circuit_breakers)


def reset_all_circuit_breakers():
    """Reset all circuit breakers to CLOSED state."""
    with _registry_lock:
        for cb in _circuit_breakers.values():
            cb._transition_to(CircuitState.CLOSED)


def get_resilience_report() -> Dict[str, Any]:
    """Generate comprehensive resilience report."""
    report = {
        "timestamp": time.time(),
        "circuit_breakers": {},
        "health": {},
        "sacred_constants": {
            "GOD_CODE": GOD_CODE,
            "PHI": PHI,
            "TAU": TAU,
            "VOID_CONSTANT": VOID_CONSTANT,
            "OMEGA": OMEGA,
        },
    }

    # Circuit breaker stats
    for name, cb in get_all_circuit_breakers().items():
        report["circuit_breakers"][name] = cb.get_stats()

    # Health status
    if _health_monitor:
        report["health"] = _health_monitor.get_system_health()

    return report


# Export main components
__all__ = [
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerOpenError",
    "CircuitState",
    "circuit_breaker",
    "get_circuit_breaker",

    # Retry
    "RetryConfig",
    "RetryState",
    "retry_with_backoff",

    # Degradation
    "GracefulDegradation",
    "DegradationLevel",
    "graceful_degradation",

    # Health
    "HealthCheck",
    "HealthStatus",
    "HealthMonitor",
    "get_health_monitor",
    "generate_health_endpoint",

    # Fallback
    "FallbackChain",
    "FallbackExhaustedError",
    "with_fallback",

    # Timeout
    "TimeoutError",
    "with_timeout",

    # Composition
    "ResilientOperation",
    "resilient",

    # Utilities
    "get_all_circuit_breakers",
    "reset_all_circuit_breakers",
    "get_resilience_report",
]


if __name__ == "__main__":
    # Demo
    print("L104 Resilience Module v1.0.0")
    print(f"GOD_CODE: {GOD_CODE}")
    print(f"PHI: {PHI}")
    print(f"TAU: {TAU}")
    print()

    # Example circuit breaker
    @circuit_breaker(name="demo", entropy=0.3)
    def demo_operation():
        return "Success"

    print("Circuit Breaker Demo:")
    print(demo_operation())

    # Example retry
    @retry_with_backoff(max_attempts=3, base_delay=TAU)
    def demo_retry():
        return "Retried"

    print("\nRetry Demo:")
    print(demo_retry())

    # Resilience report
    print("\nResilience Report:")
    import json
    print(json.dumps(get_resilience_report(), indent=2, default=str))
