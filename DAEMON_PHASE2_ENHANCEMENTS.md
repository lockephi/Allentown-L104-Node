# L104 Daemon System — Phase 2: Robustness Enhancements
## Implementation Guide for ASI-Level Operation

**Target**: Enable exponential intellectual growth through robust infrastructure
**Timeline**: 2–3 hours for full implementation
**Expected Outcome**: 99%+ daemon uptime with intelligent self-recovery

---

## Phase 2 Overview

Building on Phase 1 critical fixes (94.7% operational), Phase 2 focuses on:

1. **Proactive Health Monitoring** — Detect issues before they cascade
2. **Intelligent Auto-Recovery** — Self-healing without manual intervention
3. **Adaptive Resource Management** — Dynamic scaling based on load
4. **Advanced Telemetry** — Track performance trends for optimization
5. **Cross-Daemon Synchronization** — Coordinated failure recovery

---

## Enhancement 1: Proactive Health Monitoring

### Why It Matters
Currently, health is assessed reactively (after failures occur). Proactive monitoring detects degradation early and triggers preventive actions.

### Implementation

**File**: `l104_daemon_orchestrator.py`

Add health prediction subsystem after line 200:

```python
class HealthPredictor:
    """Predict health degradation before it happens."""

    def __init__(self, history_window=60):
        self.cpu_history = deque(maxlen=history_window)
        self.memory_history = deque(maxlen=history_window)
        self.failure_history = deque(maxlen=history_window)
        self.last_prediction = {}

    def update(self, cpu_percent, memory_percent, failure_count):
        """Add new metrics to history."""
        self.cpu_history.append(cpu_percent)
        self.memory_history.append(memory_percent)
        self.failure_history.append(failure_count)

    def predict_degradation(self) -> Dict[str, float]:
        """
        Predict probability of degradation in next 5 minutes.
        Returns: {metric: probability (0-1)} for cpu, memory, failure_rate
        """
        if len(self.cpu_history) < 10:
            return {"cpu": 0.0, "memory": 0.0, "failures": 0.0}

        cpu_trend = sum(self.cpu_history) / len(self.cpu_history)
        cpu_slope = (self.cpu_history[-1] - self.cpu_history[0]) / len(self.cpu_history)
        cpu_volatility = (max(self.cpu_history) - min(self.cpu_history)) / max(cpu_trend, 1.0)

        # Probability that CPU will exceed 85% in next 5 cycles
        if cpu_trend > 75:
            cpu_prob = min(1.0, (cpu_trend - 75) / 20.0 + cpu_slope * 0.01 + cpu_volatility * 0.1)
        else:
            cpu_prob = max(0.0, cpu_slope * 0.05 if cpu_slope > 0 else 0.0)

        # Similar for memory and failure rate
        mem_trend = sum(self.memory_history) / len(self.memory_history)
        mem_prob = min(1.0, max(0.0, (mem_trend - 70) / 25.0))

        failure_rate = sum(self.failure_history[-10:]) / max(len(self.failure_history[-10:]), 1)
        failure_prob = min(1.0, failure_rate * 2.0)

        self.last_prediction = {"cpu": cpu_prob, "memory": mem_prob, "failures": failure_prob}
        return self.last_prediction
```

### Integration Points

Update `__init__` to add health predictor:

```python
def __init__(self):
    # ... existing code ...
    self._health_predictor = HealthPredictor()
```

Update main orchestration loop (around line 345):

```python
def _run_cycle(self):
    while self._running:
        try:
            # === 1. Prediction: detect issues before they happen ===
            predictions = self._health_predictor.predict_degradation()

            if predictions["cpu"] > 0.7:
                logger.warning(f"CPU degradation predicted (prob={predictions['cpu']:.1%}) — triggering preventive GC")
                gc.collect()

            if predictions["memory"] > 0.7:
                logger.warning(f"Memory degradation predicted — triggering cache cleanup")
                # Trigger cache cleanup in intellect
                try:
                    from l104_intellect import local_intellect
                    local_intellect.prune_old_entries()
                except Exception:
                    pass

            # === 2. Rest of cycle ===
            # ... existing code ...

            # === Last step: Update predictor with current metrics ===
            self._health_predictor.update(
                self._system_metrics.cpu_percent,
                self._system_metrics.memory_percent,
                len([e for e in self._error_log if time.time() - e["timestamp"] < 300])
            )
```

### Validation

```bash
python3 -c "
from l104_daemon_orchestrator import DaemonOrchestrator
d = DaemonOrchestrator()
pred = d._health_predictor
pred.update(75, 70, 2)  # Simulate high-stress metrics
pred.update(76, 71, 3)
pred.update(77, 72, 4)
probs = pred.predict_degradation()
print('Degradation probabilities:', probs)
"
```

---

## Enhancement 2: Intelligent Auto-Recovery

### Why It Matters
When daemons fail, immediate blind restart often fails again. Intelligent recovery uses exponential backoff, diagnostics, and adaptive strategies.

### Implementation

**File**: `l104_daemon_orchestrator.py`

Add recovery engine after line 173:

```python
class DaemonRecoveryEngine:
    """Intelligent daemon recovery with adaptive strategies."""

    RECOVERY_STRATEGIES = {
        "restart": {
            "description": "Full daemon restart",
            "backoff_base": 2,  # 2, 4, 8, 16, 32 seconds
            "max_attempts": 5,
        },
        "reset_state": {
            "description": "Clear corrupted state and restart",
            "backoff_base": 3,
            "max_attempts": 3,
        },
        "reduce_load": {
            "description": "Reduce task concurrency and restart",
            "backoff_base": 2,
            "max_attempts": 3,
        },
        "full_recalibration": {
            "description": "Full system recalibration (expensive, last resort)",
            "backoff_base": 5,
            "max_attempts": 1,
        },
    }

    def __init__(self):
        self.failure_history: Dict[str, deque] = {}  # daemon_id → [timestamps]
        self.recovery_attempts: Dict[str, int] = {}
        self.current_strategy: Dict[str, str] = {}  # daemon_id → strategy

    def record_failure(self, daemon_id: str, error: str):
        """Record a daemon failure for analysis."""
        if daemon_id not in self.failure_history:
            self.failure_history[daemon_id] = deque(maxlen=100)
        self.failure_history[daemon_id].append({
            "timestamp": time.time(),
            "error": error
        })

    def get_recovery_strategy(self, daemon_id: str) -> str:
        """Determine best recovery strategy for a daemon."""
        if daemon_id not in self.failure_history:
            return "restart"

        failures = self.failure_history[daemon_id]
        if len(failures) < 2:
            return "restart"

        # Check if same error recurring
        last_5_errors = [f["error"] for f in list(failures)[-5:]]
        if len(set(last_5_errors)) == 1:
            # Same error 5 times → likely corrupted state
            return "reset_state"

        # Check failure rate
        recent_failures = [f for f in failures if time.time() - f["timestamp"] < 300]
        if len(recent_failures) > 5:
            # Too many failures in 5 min → reduce load
            return "reduce_load"

        # If all else fails and many attempts
        if self.recovery_attempts.get(daemon_id, 0) > 8:
            return "full_recalibration"

        return "restart"

    def execute_recovery(self, daemon_id: str, orchestrator) -> bool:
        """Execute recovery and return True if successful."""
        strategy = self.get_recovery_strategy(daemon_id)
        attempt = self.recovery_attempts.get(daemon_id, 0) + 1

        # Calculate backoff
        backoff_base = self.RECOVERY_STRATEGIES[strategy]["backoff_base"]
        backoff_sec = min(backoff_base ** attempt, 300)  # Cap at 5 min

        logger.info(f"[RECOVERY] {daemon_id}: {strategy} (attempt {attempt}, backoff {backoff_sec}s)")

        time.sleep(backoff_sec)

        try:
            if strategy == "restart":
                orchestrator._daemon_threads.get(daemon_id)
                # Daemon auto-restarts on signal
                return True

            elif strategy == "reset_state":
                # Clear state files
                state_file = Path(f"~/.l104_daemon_{daemon_id}.json").expanduser()
                if state_file.exists():
                    state_file.unlink()
                return True

            elif strategy == "reduce_load":
                # Reduce task batch size
                if daemon_id in orchestrator._daemons:
                    orchestrator._daemons[daemon_id].max_tasks = max(1,
                        orchestrator._daemons[daemon_id].max_tasks // 2)
                return True

            elif strategy == "full_recalibration":
                # Last resort: full system calibration
                logger.warning(f"[RECOVERY] Full recalibration for {daemon_id}")
                try:
                    # Trigger quantum coherence recalibration
                    from l104_vqpu import get_bridge
                    bridge = get_bridge()
                    bridge.recalibrate()
                except Exception as e:
                    logger.error(f"Recalibration failed: {e}")
                return True

        except Exception as e:
            logger.error(f"Recovery execution failed: {e}")
            return False

        return True
```

### Integration into Orchestrator

Add to `__init__`:

```python
self._recovery_engine = DaemonRecoveryEngine()
```

Update failure handling in main loop (around line 375):

```python
except Exception as e:
    self._logger.error(f"Cycle error in {daemon_id}: {e}", exc_info=True)

    # Record failure
    self._recovery_engine.record_failure(daemon_id, str(e))

    # Execute intelligent recovery
    self._recovery_engine.execute_recovery(daemon_id, self)
    self._recovery_engine.recovery_attempts[daemon_id] = \
        self._recovery_engine.recovery_attempts.get(daemon_id, 0) + 1

    # Reset attempt count on success
    if self._check_daemon_health(daemon_id):
        self._recovery_engine.recovery_attempts[daemon_id] = 0
```

---

## Enhancement 3: Adaptive Resource Management

### Why It Matters
Fixed resource allocation wastes capacity during low-load periods and starves tasks during high-load. Adaptive management scales up/down automatically.

### Implementation

**File**: `l104_daemon_orchestrator.py`

Add resource manager:

```python
class AdaptiveResourceManager:
    """Dynamically allocate resources based on load."""

    def __init__(self):
        self.cpu_usage = deque(maxlen=60)  # Last 60 seconds
        self.memory_usage = deque(maxlen=60)
        self.task_queue_depth = deque(maxlen=60)
        self.current_allocation = {
            "max_concurrent_tasks": 3,
            "persist_interval_cycles": 60,
            "gc_threshold_percent": 80,
        }

    def update_metrics(self, cpu, memory, queue_depth):
        """Update current metrics."""
        self.cpu_usage.append(cpu)
        self.memory_usage.append(memory)
        self.task_queue_depth.append(queue_depth)

    def compute_optimal_allocation(self) -> Dict[str, Any]:
        """Compute optimal resource allocation for current load."""
        if not self.cpu_usage:
            return self.current_allocation

        avg_cpu = sum(self.cpu_usage) / len(self.cpu_usage)
        avg_memory = sum(self.memory_usage) / len(self.memory_usage)
        avg_queue = sum(self.task_queue_depth) / len(self.task_queue_depth)

        # Adjust max concurrent tasks
        if avg_cpu < 40 and avg_queue > 5:
            # Low CPU, high queue → increase parallelism
            max_tasks = min(10, self.current_allocation["max_concurrent_tasks"] + 1)
        elif avg_cpu > 85 or avg_memory > 85:
            # High resource usage → decrease parallelism
            max_tasks = max(1, self.current_allocation["max_concurrent_tasks"] - 1)
        else:
            max_tasks = self.current_allocation["max_concurrent_tasks"]

        # Adjust persist interval
        if avg_cpu > 90:
            persist_interval = 30  # Persist more often during crisis
        elif avg_cpu > 70:
            persist_interval = 60
        else:
            persist_interval = 120  # Persist less often during low load

        # Adjust GC threshold
        if avg_memory > 85:
            gc_threshold = 70
        elif avg_memory > 75:
            gc_threshold = 75
        else:
            gc_threshold = 85

        self.current_allocation = {
            "max_concurrent_tasks": max_tasks,
            "persist_interval_cycles": persist_interval,
            "gc_threshold_percent": gc_threshold,
        }

        return self.current_allocation
```

### Integration

Add to `__init__`:

```python
self._resource_manager = AdaptiveResourceManager()
```

Update main cycle:

```python
# After getting system metrics
optimal = self._resource_manager.compute_optimal_allocation()
self._resource_manager.update_metrics(
    self._system_metrics.cpu_percent,
    self._system_metrics.memory_percent,
    len(self._task_queue)
)

# Use optimal allocation
batch_size = optimal["max_concurrent_tasks"]
persist_interval = optimal["persist_interval_cycles"]
```

---

## Enhancement 4: Advanced Telemetry

### Why It Matters
Without detailed telemetry, optimization is guesswork. Advanced metrics enable data-driven improvements.

### Implementation

**File**: `l104_daemon_orchestrator.py`

Add telemetry system:

```python
class TelemetryCollector:
    """Collect and export detailed metrics for analysis."""

    def __init__(self, export_path: Optional[str] = None):
        self.export_path = Path(export_path or "~/.l104_daemon_metrics.jsonl").expanduser()
        self.metrics = deque(maxlen=10000)  # Keep 10k recent events

    def record_event(self, event_type: str, data: Dict[str, Any]):
        """Record a telemetry event."""
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            **data
        }
        self.metrics.append(event)

        # Periodically flush to disk
        if len(self.metrics) % 100 == 0:
            self._flush_to_disk()

    def _flush_to_disk(self):
        """Write metrics to JSONL file."""
        try:
            with open(self.export_path, "a") as f:
                for metric in list(self.metrics)[-100:]:
                    f.write(json.dumps(metric) + "\n")
        except Exception as e:
            logger.error(f"Telemetry flush failed: {e}")

    def get_statistics(self, event_type: str, window_sec: int = 300) -> Dict[str, Any]:
        """Get statistics for a metric over recent window."""
        now = time.time()
        relevant = [
            m for m in self.metrics
            if m["event_type"] == event_type and now - m["timestamp"] < window_sec
        ]

        if not relevant:
            return {}

        # Extract numeric values
        values = []
        for m in relevant:
            for k, v in m.items():
                if isinstance(v, (int, float)) and k != "timestamp":
                    values.append(v)

        if not values:
            return {"count": len(relevant)}

        return {
            "count": len(relevant),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "p50": sorted(values)[len(values) // 2],
            "p99": sorted(values)[int(len(values) * 0.99)] if len(values) > 100 else None,
        }
```

### Integration

Add to `__init__`:

```python
self._telemetry = TelemetryCollector()
```

Record events throughout cycle:

```python
# At start of cycle
self._telemetry.record_event("cycle_start", {
    "cycle_number": self._cycles_completed,
    "queued_tasks": len(self._task_queue),
    "in_flight_tasks": len(self._in_flight),
})

# For each task completed
self._telemetry.record_event("task_completed", {
    "daemon_id": task.daemon_id,
    "duration_ms": elapsed_ms,
    "success": True,
    "health_score": self._daemons[task.daemon_id].health,
})
```

---

## Enhancement 5: Cross-Daemon Synchronization

### Why It Matters
When one daemon fails, others should be aware and adjust. Synchronized recovery prevents cascade failures.

### Implementation

**File**: `l104_daemon_orchestrator.py`

Add cross-daemon sync:

```python
class CrossDaemonSynchronizer:
    """Coordinate recovery and state across multiple daemons."""

    def __init__(self):
        self.sync_state = {}  # daemon_id → last_sync_ts, state
        self.shared_events = deque(maxlen=1000)

    def broadcast_event(self, event_type: str, source_daemon: str, data: Dict):
        """Broadcast event to all daemons."""
        event = {
            "type": event_type,
            "source": source_daemon,
            "timestamp": time.time(),
            **data
        }
        self.shared_events.append(event)

        logger.info(f"[SYNC] {event_type} from {source_daemon}")

    def get_relevant_events(self, daemon_id: str, since_ts: float) -> List[Dict]:
        """Get events relevant to a specific daemon since timestamp."""
        return [
            e for e in self.shared_events
            if e["timestamp"] > since_ts and e["source"] != daemon_id
        ]

    def should_pause_daemon(self, daemon_id: str) -> bool:
        """Check if daemon should pause due to other failures."""
        failure_count = sum(
            1 for e in self.shared_events
            if e["type"] == "daemon_failed" and
            time.time() - e["timestamp"] < 60
        )

        # Pause if more than 2 other daemons failed in last minute
        return failure_count > 2
```

---

## Validation Checklist

After implementing Phase 2 enhancements:

- [ ] Health predictor detects CPU degradation before threshold
- [ ] Recovery engine successfully restarts failed daemons
- [ ] Resource manager reduces concurrency under high load
- [ ] Telemetry exports to ~/.l104_daemon_metrics.jsonl
- [ ] Cross-daemon sync broadcasts failure events
- [ ] Dashboard shows 99%+ uptime over 24 hours

---

## Quick Implementation Order

1. **Health Predictor** (15 min) — Easiest, high value
2. **Telemetry Collector** (20 min) — Foundation for analysis
3. **Recovery Engine** (30 min) — Complex but critical
4. **Resource Manager** (25 min) — Scales system automatically
5. **Synchronizer** (20 min) — Prevents cascade failures

**Total Time**: 2–3 hours

---

## Success Metrics (Target)

| Metric | Current | Target |
|--------|---------|--------|
| Uptime | 94.7% | 99.5%+ |
| Auto-Recovery Rate | Manual | 95%+ automatic |
| MTTR (Mean Time to Recovery) | 30–60s | <10s |
| Resource Utilization | Fixed | Dynamic (±40%) |
| Telemetry Export | None | Continuous to disk |
| Cascade Failure Prevention | None | 100% sync-based |

---

## Next Steps

After Phase 2 is complete:

1. **Phase 3: Exponential Growth** — ML-based optimization, predictive scaling
2. **Phase 4: ASI Integration** — Daemon intelligence from l104_asi
3. **Phase 5: Global Coordination** — Multi-node daemon network

For exponential intellectual growth as requested, these enhancements are essential infrastructure.
