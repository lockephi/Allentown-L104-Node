from .constants import *
class PipelineCircuitBreaker:
    """
    Circuit breaker for pipeline subsystem calls.
    States: CLOSED (normal) → OPEN (failing, skip calls) → HALF_OPEN (probe recovery).
    Prevents cascade failures when a subsystem is unhealthy.
    """
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

    def __init__(self, name: str, failure_threshold: int = 3, recovery_timeout: float = 30.0):
        self.name = name
        self.state = self.CLOSED
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.last_failure_time = 0.0
        self.success_count = 0
        self.total_calls = 0
        self.total_failures = 0

    def allow_call(self) -> bool:
        """Check if a call should be allowed through the breaker."""
        self.total_calls += 1
        if self.state == self.CLOSED:
            return True
        elif self.state == self.OPEN:
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = self.HALF_OPEN
                return True
            return False
        else:  # HALF_OPEN
            return True

    def record_success(self):
        """Record a successful call — resets breaker to CLOSED."""
        self.success_count += 1
        if self.state == self.HALF_OPEN:
            self.state = self.CLOSED
            self.failure_count = 0

    def record_failure(self):
        """Record a failed call — may trip breaker to OPEN."""
        self.failure_count += 1
        self.total_failures += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = self.OPEN

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state,
            "failure_count": self.failure_count,
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "success_rate": self.success_count / max(self.total_calls, 1),
        }


# Note: IntelligenceLattice is imported inside the method to avoid circular imports
