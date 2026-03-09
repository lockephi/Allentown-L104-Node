from .constants import *
import functools
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════════
# EVO_57 — PIPELINE CIRCUIT BREAKER v2.0
# Upgraded: Exponential backoff, sliding-window failure tracking,
#   graduated half-open recovery, async support, decorator pattern,
#   state-transition callbacks, GOD_CODE health alignment, three-engine metrics.
# Backward-compatible: all v1 methods & properties preserved.
# ═══════════════════════════════════════════════════════════════════════════════


class BreakerState(str, Enum):
    """Circuit breaker states as an Enum for richer comparison / pattern matching."""
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"
    FORCED_OPEN = "FORCED_OPEN"


class PipelineCircuitBreaker:
    """
    Circuit breaker for pipeline subsystem calls — **v2.0**.

    States
    ------
    CLOSED      → Normal operation.
    OPEN        → Failing — calls skipped until recovery timeout.
    HALF_OPEN   → Probing recovery — requires *half_open_successes_required*
                  consecutive successes before returning to CLOSED.
    FORCED_OPEN → Manually tripped — ignores recovery timeout until force_close().

    New in v2.0
    -----------
    * Sliding-window failure detection (failures within *window_seconds*).
    * Exponential backoff on repeated trips (capped at *max_recovery_timeout*).
    * Graduated half-open recovery (configurable consecutive-success gate).
    * State-transition callback list (*on_state_change*).
    * Manual controls: force_open / force_close / reset.
    * Async wrapper: *allow_call_async*, *record_success_async*, *record_failure_async*.
    * Decorator: *@breaker.protect* for wrapping sync / async callables.
    * Health score with GOD_CODE sacred alignment.
    * Three-engine integration hooks (entropy, coherence, harmonic).
    """

    # Legacy class-level constants kept for backward compat
    CLOSED = BreakerState.CLOSED.value
    OPEN = BreakerState.OPEN.value
    HALF_OPEN = BreakerState.HALF_OPEN.value
    FORCED_OPEN = BreakerState.FORCED_OPEN.value

    # ── Constructor ──────────────────────────────────────────────

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        *,
        window_seconds: float = 60.0,
        half_open_successes_required: int = 2,
        backoff_multiplier: float = 1.5,
        max_recovery_timeout: float = 300.0,
        on_state_change: Optional[List] = None,
    ):
        self.name = name

        # State machine
        self._state: BreakerState = BreakerState.CLOSED
        self.failure_count = 0
        self.failure_threshold = failure_threshold

        # Recovery & backoff
        self.recovery_timeout = recovery_timeout
        self._base_recovery_timeout = recovery_timeout
        self.backoff_multiplier = backoff_multiplier
        self.max_recovery_timeout = max_recovery_timeout
        self._consecutive_trips = 0

        # Sliding-window failure tracking
        self.window_seconds = window_seconds
        self._failure_timestamps: List[float] = []

        # Half-open graduated recovery
        self.half_open_successes_required = half_open_successes_required
        self._half_open_consecutive_successes = 0

        # Metrics
        self.last_failure_time = 0.0
        self.last_state_change_time: float = time.time()
        self.success_count = 0
        self.total_calls = 0
        self.total_failures = 0
        self._state_history: deque = deque(maxlen=200)

        # Callbacks — list[Callable[[str, str, str], None]]  (name, old, new)
        self._on_state_change: List = on_state_change or []

        # Three-engine integration cache
        self._entropy_score: float = 0.0
        self._harmonic_score: float = 0.0
        self._coherence_score: float = 0.0

    # ── State property (fires callbacks) ─────────────────────────

    @property
    def state(self) -> str:
        """Return state as plain string for backward compat."""
        return self._state.value

    @state.setter
    def state(self, new_value: str):
        """Set state from a plain string (backward compat) or BreakerState."""
        new_bs = BreakerState(new_value) if isinstance(new_value, str) else new_value
        self._transition(new_bs)

    def _transition(self, new_state: BreakerState):
        """Internal state transition — records history + fires callbacks."""
        old = self._state
        if old == new_state:
            return
        self._state = new_state
        now = time.time()
        self.last_state_change_time = now
        self._state_history.append({
            "from": old.value,
            "to": new_state.value,
            "time": now,
        })
        for cb in self._on_state_change:
            try:
                cb(self.name, old.value, new_state.value)
            except Exception:
                pass  # Never let a callback crash the breaker

    # ── Core API (backward-compatible) ───────────────────────────

    def allow_call(self) -> bool:
        """Check if a call should be allowed through the breaker."""
        self.total_calls += 1

        if self._state == BreakerState.CLOSED:
            return True

        if self._state == BreakerState.FORCED_OPEN:
            return False

        if self._state == BreakerState.OPEN:
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self._transition(BreakerState.HALF_OPEN)
                self._half_open_consecutive_successes = 0
                return True
            return False

        # HALF_OPEN — allow probing call
        return True

    def record_success(self):
        """Record a successful call — graduated HALF_OPEN → CLOSED recovery."""
        self.success_count += 1

        if self._state == BreakerState.HALF_OPEN:
            self._half_open_consecutive_successes += 1
            if self._half_open_consecutive_successes >= self.half_open_successes_required:
                self._transition(BreakerState.CLOSED)
                self.failure_count = 0
                self._failure_timestamps.clear()
                self._consecutive_trips = 0
                self.recovery_timeout = self._base_recovery_timeout
        elif self._state == BreakerState.CLOSED:
            # Gradual decay of failure count on sustained success
            self.failure_count = max(0, self.failure_count - 1)

    def record_failure(self):
        """Record a failed call — sliding-window check, may trip breaker to OPEN."""
        now = time.time()
        self.failure_count += 1
        self.total_failures += 1
        self.last_failure_time = now

        # Sliding-window tracking
        self._failure_timestamps.append(now)
        cutoff = now - self.window_seconds
        self._failure_timestamps = [t for t in self._failure_timestamps if t > cutoff]

        if self._state == BreakerState.HALF_OPEN:
            # Any failure in half-open instantly re-opens
            self._trip()
        elif len(self._failure_timestamps) >= self.failure_threshold:
            self._trip()

    def _trip(self):
        """Trip the breaker to OPEN with exponential backoff."""
        self._consecutive_trips += 1
        self.recovery_timeout = min(
            self._base_recovery_timeout * (self.backoff_multiplier ** (self._consecutive_trips - 1)),
            self.max_recovery_timeout,
        )
        self._transition(BreakerState.OPEN)

    # ── Manual Controls ──────────────────────────────────────────

    def force_open(self):
        """Manually trip the breaker — stays open until force_close() or reset()."""
        self._transition(BreakerState.FORCED_OPEN)

    def force_close(self):
        """Manually close the breaker, resuming normal operation."""
        self.failure_count = 0
        self._failure_timestamps.clear()
        self._half_open_consecutive_successes = 0
        self._consecutive_trips = 0
        self.recovery_timeout = self._base_recovery_timeout
        self._transition(BreakerState.CLOSED)

    def reset(self):
        """Full reset — alias for force_close(), clears all counters."""
        self.success_count = 0
        self.total_calls = 0
        self.total_failures = 0
        self.force_close()

    # ── Async Wrappers ───────────────────────────────────────────

    async def allow_call_async(self) -> bool:
        """Async-compatible version of allow_call."""
        return self.allow_call()

    async def record_success_async(self):
        """Async-compatible version of record_success."""
        self.record_success()

    async def record_failure_async(self):
        """Async-compatible version of record_failure."""
        self.record_failure()

    # ── Decorator ────────────────────────────────────────────────

    def protect(self, func=None, *, fallback=None):
        """
        Decorator that wraps a sync or async callable in this circuit breaker.

        Usage::

            @breaker.protect
            def risky_call():
                ...

            @breaker.protect(fallback={"error": "unavailable"})
            async def risky_async_call():
                ...
        """
        def decorator(fn):
            if asyncio.iscoroutinefunction(fn):
                @functools.wraps(fn)
                async def async_wrapper(*a, **kw):
                    if not self.allow_call():
                        return fallback
                    try:
                        result = await fn(*a, **kw)
                        self.record_success()
                        return result
                    except Exception:
                        self.record_failure()
                        if fallback is not None:
                            return fallback
                        raise
                return async_wrapper
            else:
                @functools.wraps(fn)
                def sync_wrapper(*a, **kw):
                    if not self.allow_call():
                        return fallback
                    try:
                        result = fn(*a, **kw)
                        self.record_success()
                        return result
                    except Exception:
                        self.record_failure()
                        if fallback is not None:
                            return fallback
                        raise
                return sync_wrapper
        # Allow @breaker.protect  or  @breaker.protect(fallback=...)
        if func is not None:
            return decorator(func)
        return decorator

    # ── Three-Engine Integration ─────────────────────────────────

    def update_engine_scores(
        self,
        entropy_score: float = 0.0,
        harmonic_score: float = 0.0,
        coherence_score: float = 0.0,
    ):
        """Inject three-engine metrics for GOD_CODE-aligned health scoring."""
        self._entropy_score = entropy_score
        self._harmonic_score = harmonic_score
        self._coherence_score = coherence_score

    def compute_health_score(self) -> float:
        """
        Compute a [0.0, 1.0] health score incorporating:
          - Success rate (weighted 0.4)
          - State penalty (weighted 0.3)
          - Three-engine coherence (weighted 0.2)
          - GOD_CODE sacred alignment (weighted 0.1)
        """
        # Base success rate
        sr = self.success_count / max(self.total_calls, 1)

        # State penalty
        state_scores = {
            BreakerState.CLOSED: 1.0,
            BreakerState.HALF_OPEN: 0.5,
            BreakerState.OPEN: 0.1,
            BreakerState.FORCED_OPEN: 0.0,
        }
        state_val = state_scores.get(self._state, 0.0)

        # Three-engine coherence blend
        engine_blend = (
            self._entropy_score * 0.4
            + self._harmonic_score * 0.35
            + self._coherence_score * 0.25
        )

        # GOD_CODE alignment — success_count mod-ratio to GOD_CODE
        gc_ratio = (self.success_count % GOD_CODE) / GOD_CODE if GOD_CODE > 0 else 0.0
        gc_alignment = 1.0 - abs(gc_ratio - PHI + int(PHI))  # φ-proximity resonance

        health = (
            0.4 * sr
            + 0.3 * state_val
            + 0.2 * min(engine_blend, 1.0)
            + 0.1 * max(gc_alignment, 0.0)
        )
        return round(max(0.0, min(1.0, health)), 6)

    # ── Status / Serialization ───────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Extended status report — superset of v1 fields."""
        return {
            "name": self.name,
            "state": self.state,
            "failure_count": self.failure_count,
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "success_rate": round(self.success_count / max(self.total_calls, 1), 6),
            "health_score": self.compute_health_score(),
            "recovery_timeout": round(self.recovery_timeout, 2),
            "consecutive_trips": self._consecutive_trips,
            "half_open_progress": f"{self._half_open_consecutive_successes}/{self.half_open_successes_required}",
            "window_failures": len(self._failure_timestamps),
            "window_seconds": self.window_seconds,
            "engine_scores": {
                "entropy": round(self._entropy_score, 4),
                "harmonic": round(self._harmonic_score, 4),
                "coherence": round(self._coherence_score, 4),
            },
            "last_state_change": self.last_state_change_time,
            "state_transitions": len(self._state_history),
        }

    def __repr__(self) -> str:
        return (
            f"<PipelineCircuitBreaker '{self.name}' state={self.state} "
            f"health={self.compute_health_score():.3f} "
            f"calls={self.total_calls} failures={self.total_failures}>"
        )


# Note: IntelligenceLattice is imported inside the method to avoid circular imports
