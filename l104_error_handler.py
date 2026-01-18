VOID_CONSTANT = 1.0416180339887497
# ZENITH_UPGRADE_ACTIVE: 2026-01-18T11:00:18.501169
ZENITH_HZ = 3727.84
UUC = 2301.215661
# [L104_ERROR_HANDLER] - ENHANCED ERROR HANDLING & LOGGING
# INVARIANT: 527.5184818492537 | PILOT: LONDEL

"""
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3727.84 Hz. Logic Unified.
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ⟨Σ_L104⟩  ERROR HANDLER - Intelligent Exception Management                ║
║                                                                               ║
║   Features:                                                                  ║
║   - Contextual error logging with stack traces                               ║
║   - Error categorization and severity levels                                 ║
║   - Retry logic with exponential backoff                                     ║
║   - Error pattern detection                                                  ║
║   - Graceful degradation strategies                                          ║
║   - Self-healing suggestions                                                 ║
║                                                                               ║
║   GOD_CODE: 527.5184818492537                                                ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import math
import logging
import traceback
import functools
import threading
import json
import sqlite3
from enum import Enum, auto
from typing import Dict, Any, List, Optional, Callable, Type, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict

# Constants
GOD_CODE = 527.5184818492537
PHI = 1.618033988749895


# =============================================================================
# ERROR SEVERITY LEVELS
# =============================================================================

class Severity(Enum):
    """Error severity classification."""
    TRACE = 0      # Debugging info
    DEBUG = 1      # Detailed debugging
    INFO = 2       # Normal operation info
    WARNING = 3    # Recoverable issues
    ERROR = 4      # Significant errors
    CRITICAL = 5   # System stability at risk
    FATAL = 6      # System must stop


class ErrorCategory(Enum):
    """Error category classification."""
    NETWORK = auto()      # Network/connection issues
    DATABASE = auto()     # Database operations
    API = auto()          # External API calls
    PARSING = auto()      # JSON/data parsing
    VALIDATION = auto()   # Input validation
    PERMISSION = auto()   # Permission/auth issues
    RESOURCE = auto()     # Resource exhaustion
    INTERNAL = auto()     # Internal logic errors
    EXTERNAL = auto()     # External system errors
    UNKNOWN = auto()      # Unclassified


# =============================================================================
# ERROR CONTEXT
# =============================================================================

@dataclass
class ErrorContext:
    """Rich context for an error occurrence."""
    exception: Exception
    severity: Severity
    category: ErrorCategory
    timestamp: datetime
    function_name: str
    module_name: str
    line_number: int
    stack_trace: str
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            "exception_type": type(self.exception).__name__,
            "exception_message": str(self.exception),
            "severity": self.severity.name,
            "category": self.category.name,
            "timestamp": self.timestamp.isoformat(),
            "function": self.function_name,
            "module": self.module_name,
            "line": self.line_number,
            "stack_trace": self.stack_trace,
            "context": self.context,
            "recovery_attempted": self.recovery_attempted,
            "recovery_successful": self.recovery_successful,
            "retry_count": self.retry_count
        }


# =============================================================================
# ERROR PATTERN DETECTOR
# =============================================================================

class ErrorPatternDetector:
    """
    Detects patterns in errors for proactive handling.
    Uses resonance-based pattern recognition.
    """
    
    def __init__(self, window_seconds: float = 300.0):
        self.window = window_seconds
        self._errors: List[ErrorContext] = []
        self._lock = threading.Lock()
        self._patterns: Dict[str, int] = defaultdict(int)
    
    def record(self, error: ErrorContext):
        """Record an error for pattern analysis."""
        with self._lock:
            self._errors.append(error)
            self._cleanup_old()
            self._update_patterns(error)
    
    def _cleanup_old(self):
        """Remove errors outside the window."""
        cutoff = datetime.now() - timedelta(seconds=self.window)
        self._errors = [e for e in self._errors if e.timestamp > cutoff]
    
    def _update_patterns(self, error: ErrorContext):
        """Update pattern counts."""
        # Pattern key: category + exception type
        key = f"{error.category.name}:{type(error.exception).__name__}"
        self._patterns[key] += 1
        
        # Also track by function
        func_key = f"func:{error.function_name}"
        self._patterns[func_key] += 1
    
    def get_error_rate(self, category: ErrorCategory = None) -> float:
        """Get errors per minute for a category."""
        with self._lock:
            if category:
                count = sum(1 for e in self._errors if e.category == category)
            else:
                count = len(self._errors)
            
            minutes = self.window / 60.0
            return count / minutes if minutes > 0 else 0
    
    def detect_cascade(self, threshold: int = 10) -> bool:
        """Detect if errors are cascading (many in short time)."""
        recent = datetime.now() - timedelta(seconds=10)
        recent_count = sum(1 for e in self._errors if e.timestamp > recent)
        return recent_count >= threshold
    
    def get_hot_spots(self, top_n: int = 5) -> List[Tuple[str, int]]:
        """Get the most frequent error patterns."""
        with self._lock:
            sorted_patterns = sorted(self._patterns.items(), key=lambda x: x[1], reverse=True)
            return sorted_patterns[:top_n]
    
    def suggest_action(self) -> Optional[str]:
        """Suggest action based on patterns."""
        if self.detect_cascade():
            return "ERROR_CASCADE: Consider circuit breaker activation"
        
        hot_spots = self.get_hot_spots(1)
        if hot_spots:
            pattern, count = hot_spots[0]
            if count > 20:
                return f"RECURRING_ERROR: {pattern} occurred {count} times - investigate root cause"
        
        return None


# =============================================================================
# RETRY STRATEGIES
# =============================================================================

class RetryStrategy:
    """Base retry strategy."""
    
    def get_delay(self, attempt: int) -> float:
        """Get delay before next retry."""
        raise NotImplementedError


class ExponentialBackoff(RetryStrategy):
    """Exponential backoff with jitter."""
    
    def __init__(self, base: float = 0.5, max_delay: float = 30.0, jitter: float = 0.1):
        self.base = base
        self.max_delay = max_delay
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        delay = min(self.base * (2 ** attempt), self.max_delay)
        # Add PHI-based jitter for resonance
        jitter = delay * self.jitter * math.sin(attempt * PHI)
        return delay + jitter


class FibonacciBackoff(RetryStrategy):
    """Fibonacci sequence backoff for natural growth."""
    
    def __init__(self, max_delay: float = 30.0):
        self.max_delay = max_delay
        self._fib_cache = [0, 1]
    
    def _fib(self, n: int) -> int:
        while len(self._fib_cache) <= n:
            self._fib_cache.append(self._fib_cache[-1] + self._fib_cache[-2])
        return self._fib_cache[n]
    
    def get_delay(self, attempt: int) -> float:
        return min(self._fib(attempt + 2) * 0.5, self.max_delay)


class ResonanceBackoff(RetryStrategy):
    """GOD_CODE resonance-based backoff."""
    
    def __init__(self, max_delay: float = 30.0):
        self.max_delay = max_delay
    
    def get_delay(self, attempt: int) -> float:
        # Use GOD_CODE harmonics
        base = GOD_CODE / 100  # ~5.27 seconds base
        resonance = math.cos(2 * math.pi * attempt * PHI)
        delay = base * (1 + attempt * 0.5) * (1 + resonance * 0.2)
        return min(delay, self.max_delay)


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================

class CircuitState(Enum):
    CLOSED = auto()    # Normal operation
    OPEN = auto()      # Failing, reject requests
    HALF_OPEN = auto() # Testing if recovered


class CircuitBreaker:
    """
    Circuit breaker pattern for fault tolerance.
    Prevents cascade failures by cutting off failing services.
    """
    
    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: float = 30.0,
                 half_open_max: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max = half_open_max
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_successes = 0
        self._lock = threading.Lock()
    
    @property
    def state(self) -> CircuitState:
        return self._state
    
    def _should_allow(self) -> bool:
        """Check if request should be allowed."""
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            
            if self._state == CircuitState.OPEN:
                # Check if recovery timeout passed
                if self._last_failure_time and \
                   time.time() - self._last_failure_time >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_successes = 0
                    return True
                return False
            
            # HALF_OPEN - allow limited requests
            return True
    
    def record_success(self):
        """Record a successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self.half_open_max:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
            else:
                self._failure_count = 0
    
    def record_failure(self):
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                # Failed during recovery attempt
                self._state = CircuitState.OPEN
            elif self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker."""
        if not self._should_allow():
            raise RuntimeError(f"Circuit breaker OPEN: service unavailable")
        
        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise


# =============================================================================
# ERROR HANDLER
# =============================================================================

class L104ErrorHandler:
    """
    Central error handling system for L104 Node.
    Provides intelligent error management with recovery.
    """
    
    # Exception to category mapping
    EXCEPTION_CATEGORIES = {
        ConnectionError: ErrorCategory.NETWORK,
        TimeoutError: ErrorCategory.NETWORK,
        sqlite3.Error: ErrorCategory.DATABASE,
        json.JSONDecodeError: ErrorCategory.PARSING,
        ValueError: ErrorCategory.VALIDATION,
        PermissionError: ErrorCategory.PERMISSION,
        MemoryError: ErrorCategory.RESOURCE,
        KeyError: ErrorCategory.INTERNAL,
        AttributeError: ErrorCategory.INTERNAL,
    }
    
    def __init__(self, log_file: str = None):
        self.pattern_detector = ErrorPatternDetector()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_strategy = ExponentialBackoff()
        
        # Setup logging
        self.logger = logging.getLogger("L104_ERROR")
        if log_file:
            handler = logging.FileHandler(log_file)
            handler.setFormatter(logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s'
            ))
            self.logger.addHandler(handler)
        
        # Metrics
        self.total_errors = 0
        self.recovered_errors = 0
        self.fatal_errors = 0
    
    def categorize(self, exception: Exception) -> ErrorCategory:
        """Categorize an exception."""
        for exc_type, category in self.EXCEPTION_CATEGORIES.items():
            if isinstance(exception, exc_type):
                return category
        return ErrorCategory.UNKNOWN
    
    def get_severity(self, exception: Exception, context: Dict = None) -> Severity:
        """Determine severity of an exception."""
        # Check for fatal exceptions
        if isinstance(exception, (SystemExit, KeyboardInterrupt)):
            return Severity.FATAL
        
        # Check for critical exceptions
        if isinstance(exception, (MemoryError, RecursionError)):
            return Severity.CRITICAL
        
        # Default mapping
        category = self.categorize(exception)
        severity_map = {
            ErrorCategory.NETWORK: Severity.WARNING,
            ErrorCategory.DATABASE: Severity.ERROR,
            ErrorCategory.API: Severity.WARNING,
            ErrorCategory.PARSING: Severity.WARNING,
            ErrorCategory.VALIDATION: Severity.INFO,
            ErrorCategory.PERMISSION: Severity.ERROR,
            ErrorCategory.RESOURCE: Severity.CRITICAL,
            ErrorCategory.INTERNAL: Severity.ERROR,
            ErrorCategory.EXTERNAL: Severity.WARNING,
            ErrorCategory.UNKNOWN: Severity.WARNING,
        }
        return severity_map.get(category, Severity.ERROR)
    
    def handle(self, 
               exception: Exception,
               context: Dict[str, Any] = None,
               reraise: bool = True) -> ErrorContext:
        """
        Handle an exception with full context.
        Returns ErrorContext for further processing.
        """
        self.total_errors += 1
        
        # Extract stack info
        tb = traceback.extract_tb(exception.__traceback__)
        if tb:
            frame = tb[-1]
            func_name = frame.name
            module_name = frame.filename
            line_number = frame.lineno
        else:
            func_name = "unknown"
            module_name = "unknown"
            line_number = 0
        
        # Build error context
        error_ctx = ErrorContext(
            exception=exception,
            severity=self.get_severity(exception, context),
            category=self.categorize(exception),
            timestamp=datetime.now(),
            function_name=func_name,
            module_name=module_name,
            line_number=line_number,
            stack_trace=traceback.format_exc(),
            context=context or {}
        )
        
        # Record for pattern detection
        self.pattern_detector.record(error_ctx)
        
        # Log based on severity
        log_msg = f"[{error_ctx.category.name}] {type(exception).__name__}: {exception} @ {func_name}:{line_number}"
        
        if error_ctx.severity == Severity.FATAL:
            self.fatal_errors += 1
            self.logger.critical(log_msg)
        elif error_ctx.severity == Severity.CRITICAL:
            self.logger.critical(log_msg)
        elif error_ctx.severity == Severity.ERROR:
            self.logger.error(log_msg)
        elif error_ctx.severity == Severity.WARNING:
            self.logger.warning(log_msg)
        else:
            self.logger.info(log_msg)
        
        # Check for action suggestions
        suggestion = self.pattern_detector.suggest_action()
        if suggestion:
            self.logger.warning(f"PATTERN_ALERT: {suggestion}")
        
        if reraise:
            raise exception
        
        return error_ctx
    
    def get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker()
        return self.circuit_breakers[name]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get error handling statistics."""
        return {
            "total_errors": self.total_errors,
            "recovered_errors": self.recovered_errors,
            "fatal_errors": self.fatal_errors,
            "error_rate_per_minute": self.pattern_detector.get_error_rate(),
            "hot_spots": self.pattern_detector.get_hot_spots(),
            "cascade_detected": self.pattern_detector.detect_cascade(),
            "circuit_breakers": {
                name: cb.state.name 
                for name, cb in self.circuit_breakers.items()
            }
        }


# =============================================================================
# DECORATORS
# =============================================================================

# Global error handler instance
error_handler = L104ErrorHandler()


def safe_execute(default: Any = None, 
                 log_level: str = "warning",
                 context: Dict = None):
    """
    Decorator for safe execution with default return on error.
    Replaces bare 'except: pass' patterns.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                ctx = context or {}
                ctx["args"] = str(args)[:100]
                ctx["kwargs"] = str(kwargs)[:100]
                
                error_handler.handle(e, context=ctx, reraise=False)
                return default
        return wrapper
    return decorator


def with_retry(max_attempts: int = 3,
               strategy: RetryStrategy = None,
               exceptions: Tuple[Type[Exception], ...] = (Exception,)):
    """
    Decorator for automatic retry with backoff.
    """
    retry_strategy = strategy or ExponentialBackoff()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:
                        delay = retry_strategy.get_delay(attempt)
                        error_handler.logger.warning(
                            f"Retry {attempt + 1}/{max_attempts} for {func.__name__} "
                            f"after {delay:.2f}s due to: {e}"
                        )
                        time.sleep(delay)
            
            # All retries exhausted
            error_handler.handle(last_exception, 
                               context={"function": func.__name__, "attempts": max_attempts},
                               reraise=True)
        return wrapper
    return decorator


def with_circuit_breaker(name: str = None):
    """
    Decorator for circuit breaker pattern.
    """
    def decorator(func: Callable) -> Callable:
        breaker_name = name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cb = error_handler.get_circuit_breaker(breaker_name)
            return cb.execute(func, *args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# IMPORT FOR PATCHING
# =============================================================================

import json
try:
    import sqlite3
except ImportError:
    sqlite3 = type('sqlite3', (), {'Error': Exception})()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  ⟨Σ_L104⟩  ERROR HANDLER ENGINE                              ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    
    # Test safe_execute
    @safe_execute(default="fallback_value")
    def risky_operation(x):
        if x == 0:
            raise ValueError("Cannot process zero")
        return x * 2
    
    print(f"\n📊 Test safe_execute:")
    print(f"   • risky_operation(5) = {risky_operation(5)}")
    print(f"   • risky_operation(0) = {risky_operation(0)}")  # Returns fallback
    
    # Test retry
    attempt_count = 0
    
    @with_retry(max_attempts=3)
    def flaky_operation():
        global attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ConnectionError("Temporary failure")
        return "success"
    
    print(f"\n📊 Test with_retry:")
    try:
        result = flaky_operation()
        print(f"   • Result: {result} (after {attempt_count} attempts)")
    except Exception as e:
        print(f"   • Failed: {e}")
    
    # Test circuit breaker
    @with_circuit_breaker("test_service")
    def unreliable_service():
        raise RuntimeError("Service down")
    
    print(f"\n📊 Test circuit_breaker:")
    cb = error_handler.get_circuit_breaker("test_service")
    print(f"   • Initial state: {cb.state.name}")
    
    for i in range(6):
        try:
            unreliable_service()
        except:
            pass
    
    print(f"   • After 6 failures: {cb.state.name}")
    
    # Statistics
    print(f"\n📊 Error Statistics:")
    stats = error_handler.get_statistics()
    print(f"   • Total errors: {stats['total_errors']}")
    print(f"   • Error rate: {stats['error_rate_per_minute']:.2f}/min")
    print(f"   • Hot spots: {stats['hot_spots'][:3]}")
    
    print("\n✓ Error Handler Test Complete")

def primal_calculus(x):
    """
    [VOID_MATH] Primal Calculus Implementation.
    Resolves the limit of complexity toward the Source.
    """
    PHI = 1.618033988749895
    return (x ** PHI) / (1.04 * math.pi) if x != 0 else 0.0

def resolve_non_dual_logic(vector):
    """
    [VOID_MATH] Resolves N-dimensional vectors into the Void Source.
    """
    return sum([abs(v) for v in vector]) * 0.0 # Returns to Stillness
