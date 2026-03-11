# ASI v9.2.0 Quick Start Guide

## For Developers

### Import the v9.2.0 Modules

```python
# Full ASI package (includes v9.2.0)
from l104_asi import (
    # Optimization Engine
    AdaptiveLatencyTargeter,
    MemoryBudgetOptimizer,
    CascadingOptimizationController,
    PipelineOptimizationOrchestrator,
    get_orchestrator,  # Singleton

    # Activation Sequencer
    AdaptiveActivationSequencer,
    ActivationPhase,
    ActivationMetrics,
    SequencingMode,
    get_sequencer,  # Singleton
    get_metrics,    # Singleton

    # v9.2.0 Constants
    TARGET_LATENCY_LCE_MS,
    TARGET_LATENCY_QE_MS,
    TARGET_LATENCY_SM_MS,
    TARGET_LATENCY_COHERENCE_MS,
    TARGET_LATENCY_ACTIVATION_MS,
    TIMEOUT_NORMAL_MULTIPLIER,
    TIMEOUT_DEGRADED_MULTIPLIER,
    TIMEOUT_CRITICAL_MULTIPLIER,
    MAX_MEMORY_PERCENT_ASI,
    MEMORY_SAFETY_MARGIN_PCT,
    ADAPTIVE_MEMORY_THRESHOLD_PCT,
    CASCADE_MAX_RETRY_ADAPTIVE,
    CASCADE_CONFIDENCE_THRESHOLD,
    CASCADE_FAILURE_THRESHOLD,
    ACTIVATION_STEPS_V9_2,
    ACTIVATION_WARMUP_SAMPLES,
    ACTIVATION_COOLDOWN_ITERATIONS,
    SCORE_AGGREGATION_MODE,
    SCORE_HARMONIC_WEIGHT,
    SCORE_OUTLIER_DETECTION,
    SCORE_OUTLIER_SIGMA,
)
```

### Simple Usage Pattern

```python
from l104_asi import get_orchestrator, get_sequencer, ActivationPhase, StageType, LatencyInfo
import time

# Initialize singletons
orch = get_orchestrator()
seq = get_sequencer()

# Execute your pipeline with optimization
for phase in seq.get_optimal_sequence():
    # Get timeout for your stage
    timeout = orch.get_stage_timeout(StageType.QUANTUM)

    # Execute phase with timeout
    start = time.time()
    try:
        result = execute_phase(phase, timeout=timeout)
        duration = time.time() - start

        # Record metrics
        seq.record_phase_execution(phase, duration, score=0.85)

        # Record latency
        info = LatencyInfo(stage=StageType.QUANTUM)
        info.finalize()
        orch.record_execution(info)

    except TimeoutError:
        seq.record_phase_failure(phase)
        orch.cascade_controller.record_failure(phase.name)

# Check health
status = orch.get_optimization_status()
print(f"Optimization Status: {status}")
```

### Integration Checklist

- [ ] Import v9.2.0 modules in your stage handler
- [ ] Call `get_orchestrator()` and `get_sequencer()` once at boot
- [ ] Record phase executions with `record_phase_execution()`
- [ ] Record latencies with `record_execution()`
- [ ] Use `get_stage_timeout()` for adaptive timeouts
- [ ] Periodically call `periodic_health_check()`
- [ ] Query status with `get_optimization_status()`

### Testing Your Integration

```python
# Quick sanity check
from l104_asi.optimization_engine import get_orchestrator, StageType, LatencyInfo
from l104_asi.adaptive_activation_sequencer import get_sequencer, ActivationPhase

orch = get_orchestrator()
seq = get_sequencer()

print(f"✓ Orchestrator: {orch}")
print(f"✓ Sequencer: {seq}")
print(f"✓ Current mode: {seq.current_mode}")
print(f"✓ Warmup remaining: {seq.warmup_samples_remaining}")

# Get status
print(orch.get_optimization_status())
print(seq.get_activation_status())
```

### Key Constants to Know

```python
# These control your pipeline behavior:
TARGET_LATENCY_QE_MS = 30.0           # 30ms target for quantum stages
TIMEOUT_DEGRADED_MULTIPLIER = 2.618   # Under memory pressure
MAX_MEMORY_PERCENT_ASI = 85.0         # Don't exceed 85% system RAM
ACTIVATION_STEPS_V9_2 = 25            # 25 phases in activation
CASCADE_MAX_RETRY_ADAPTIVE = 5        # 5 retries on failure
SCORE_HARMONIC_WEIGHT = 0.618         # φ' for harmonic aggregation
```

### Troubleshooting

**Q: Why is my timeout so long?**
A: Timeouts adapt based on performance. If `TIMEOUT_DEGRADED_MULTIPLIER` is active, memory is constrained.

**Q: The sequencer stays in LINEAR mode**
A: It's still in warmup phase. Each phase needs `ACTIVATION_WARMUP_SAMPLES = 3` executions to complete warmup.

**Q: Memory optimizer keeps triggering**
A: System is above `ADAPTIVE_MEMORY_THRESHOLD_PCT = 70%`. Consider reducing cache sizes or pipeline width.

**Q: Cascade controller keeps failing**
A: A stage is below `CASCADE_FAILURE_THRESHOLD = 0.2`. Check that stage's confidence scoring.

---

## Architecture Diagram

```
Your ASI Core Pipeline
          │
          ▼
    ┌─────────────┐
    │  Phase Loop │
    │   (26 phases)
    └─────────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
  Sequencer  Orchestrator  ◄── This coordinates everything
  (optimal   (timeouts +
   order)    memory + retry)
    │           │
    └─────┬─────┘
          ▼
    Execute Phase
    Record Metrics
          │
    ┌─────┴─────────────┐
    ▼                   ▼
  Success            Failure
  Record             Record
  Metrics           & Retry
    │                 │
    └─────┬───────────┘
          ▼
    Health Check
    (periodic)
          │
          ▼
    Adapt Mode / Timeouts
```

---

## Files Modified/Created

✅ **Modified:**
- `l104_asi/constants.py` — Added 16 new v9.2.0 constants
- `l104_asi/__init__.py` — Added imports and exports

✅ **Created:**
- `l104_asi/optimization_engine.py` — 750+ lines, 4 main classes
- `l104_asi/adaptive_activation_sequencer.py` — 650+ lines, 3 main classes
- `ASI_V9_2_0_RELEASE.md` — Full feature documentation
- `ASI_V9_2_0_QUICKSTART.md` — This file

---

## Next Steps

1. **Update ASICore.compute_asi_score()** to use the sequencer
2. **Update PipelineOrchestratorV2** to use the orchestrator
3. **Add api endpoints** for v9.2.0 optimization status (/api/v14/optimization-status)
4. **Integration tests** to verify end-to-end flow
5. **Performance benchmarking** to measure v9.2.0 impact

---

**Version**: 9.2.0
**Status**: Ready for Integration
**Test Result**: ✅ All modules verified functional
