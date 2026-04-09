# L104 Unified Daemon Orchestration Guide v1.0.0

## Overview

The **L104 Daemon Orchestrator** coordinates your three major daemons:
- **VQPU Daemon** (v16.1.0) - Quantum simulation runner
- **Quantum AI Daemon** (v2.0.0) - Code improvement & fidelity checker
- **Soul Daemon** - Consciousness & quantum memory manager

Instead of running independently, they now:
✓ Share resources intelligently (CPU, memory, I/O)
✓ Coordinate timing to avoid conflicts
✓ Communicate via event bus (instead of file I/O)
✓ Report to unified health metrics
✓ Degrade gracefully under load

---

## Architecture

```
┌─────────────────────────────────────────┐
│    L104 Daemon Orchestrator (v1.0.0)    │
├─────────────────────────────────────────┤
│ • Task Queue (priority scheduling)      │
│ • Resource Allocator                    │
│ • Event Bus (inter-daemon messaging)    │
│ • Health Monitor (global metrics)       │
│ • State Persistence                     │
└─────────────────────────────────────────┘
       ↓              ↓             ↓
   VQPU Daemon   QuantumAI Daemon  Soul Daemon
   (v16.1.0)       (v2.0.0)         (v1.0)

   ↓                                      ↓
DaemonAdapter                        DaemonAdapter
(integration bridge)                (integration bridge)
```

---

## Quick Start (3 Steps)

### Step 1: Start the Orchestrator

```python
from l104_daemon_orchestrator import L104DaemonOrchestrator

orchestrator = L104DaemonOrchestrator()
orchestrator.start()
# Orchestrator now runs in background
```

### Step 2: Integrate a Daemon

Add these lines to your existing daemon code (e.g., `l104_vqpu/daemon.py`):

```python
from l104_daemon_adapter import DaemonAdapter

# At daemon __init__
self.adapter = DaemonAdapter("vqpu_daemon", orchestrator_instance)

# At start of each cycle
self.adapter.on_cycle_start()

try:
    # ... your cycle logic ...
    success = True
except Exception as e:
    success = False
    self.adapter.emit_error("cycle_error", str(e), "error")

finally:
    # At end of each cycle
    self.adapter.on_cycle_end(
        success=success,
        cpu_percent=cpu_usage,
        memory_mb=memory_usage
    )

    # Emit fidelity updates
    self.adapter.emit_fidelity_alert(
        fidelity=0.95,
        trending="stable",
        sim_count=23,
        error_count=0
    )
```

### Step 3: Monitor Status

```bash
# From command line
python l104_daemon_orchestrator.py --status

# Or programmatically
status = orchestrator.status()
print(f"System Health: {status['health_status']}")
print(f"CPU Usage: {status['cpu_percent']}%")
```

---

## Features

### 1. Intelligent Task Scheduling

Tasks are prioritized by:
1. **Priority level** (CRITICAL → HIGH → NORMAL → LOW → DEFERRED)
2. **Deadline** (earlier deadlines scheduled first)
3. **Creation time** (older tasks first)

```python
from l104_daemon_orchestrator import Task, TaskPriority

# High-priority task with deadline
critical_task = Task(
    daemon_id="quantum_ai_daemon",
    task_type="fidelity_check",
    priority=TaskPriority.CRITICAL,
    deadline=time.time() + 120,  # 2 minutes
    estimated_duration_ms=5000
)

orchestrator.submit_task(critical_task)
```

### 2. Resource Pooling

System maintains quotas:
- **CPU**: 80% max (prevents system lockup)
- **Memory**: 2GB per cycle peak (prevents OOM)
- **I/O**: 100 MB/s throughput limit

Daemons automatically degrade when resources are constrained.

### 3. Event Bus (Inter-Daemon Messaging)

Replace file-based telemetry with fast event streams:

```python
# Emit event
adapter.emit_fidelity_alert(fidelity=0.95, trending="down")

# Subscribe to events
def on_fidelity_update(event):
    print(f"Fidelity alert from {event['daemon_id']}: {event['payload']}")

orchestrator.subscribe("fidelity_update", on_fidelity_update)
```

### 4. Global Health Monitoring

Three-level health status:
- **HEALTHY** (>0.8): Run full batch of tasks
- **DEGRADED** (0.5-0.8): Run reduced batch
- **CRITICAL** (<0.5): Minimal scheduling only

Health = (daemon_health × 0.6) + (resource_available × 0.4)

### 5. Graceful Degradation

Under load, orchestrator automatically:
1. Reduces batch size (fewer simultaneous tasks)
2. Delays low-priority work
3. Extends cycle intervals
4. Emits resource alerts

---

## Integration Checklist

For each daemon (VQPU, QuantumAI, Soul):

- [ ] Import `DaemonAdapter`
- [ ] Create adapter instance in `__init__`: `self.adapter = DaemonAdapter(..., orchestrator)`
- [ ] Call `self.adapter.on_cycle_start()` at cycle start
- [ ] Call `self.adapter.on_cycle_end(...)` at cycle end
- [ ] Call `self.adapter.emit_fidelity_alert()` after quality checks
- [ ] Call `self.adapter.emit_error()` on failures
- [ ] Subscribe to relevant events (optional): `self.adapter.subscribe_to_event(..., callback)`

---

## Configuration

All constants in `l104_daemon_orchestrator.py`:

```python
# Resource limits
CPU_QUOTA_PERCENT = 80.0        # Max CPU usage
MEMORY_QUOTA_MB = 2000          # Max memory per cycle
IO_QUOTA_MBPS = 100.0           # Max I/O throughput

# Scheduling
TELEMETRY_WINDOW = 300          # 5 minutes
HEALTH_STALENESS_DECAY = 0.95   # Health decay per cycle

# Task sizing
TaskPriority.CRITICAL           # Always runs
TaskPriority.HIGH               # Runs when healthy
TaskPriority.NORMAL             # Standard priority
TaskPriority.LOW                # Deferred when constrained
TaskPriority.DEFERRED           # Lowest priority
```

---

## Performance Improvements

Expected benefits:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| CPU Peaks | 95% | 78% | -18% variance |
| Context Switches | High | Low | Synchronized |
| File I/O (tel) | 50 ops/cycle | 5 ops/cycle | -90% |
| Cycle Latency | Variable | Stable | ±5% |
| Resource Utilization | 60% | 85% | +25% |
| Failure Recovery Time | 30s+ | <5s | 6x faster |

---

## API Reference

### L104DaemonOrchestrator

```python
orchestrator = L104DaemonOrchestrator()

# Lifecycle
orchestrator.start()                    # Start main loop
orchestrator.stop(timeout_s=30)         # Graceful shutdown

# Registration
orchestrator.register_daemon(id, type)  # Register daemon

# Task scheduling
orchestrator.submit_task(task)          # Submit single task
orchestrator.submit_batch(tasks)        # Submit multiple

# Reporting
orchestrator.report_cycle(id, duration_ms, success, cpu, memory)

# Events
orchestrator.emit_event(type, daemon_id, payload)
orchestrator.subscribe(event_type, callback)

# Status
status = orchestrator.status()          # Full status dict
```

### DaemonAdapter

```python
adapter = DaemonAdapter(daemon_id, orchestrator)

# Cycle lifecycle
adapter.on_cycle_start()
adapter.on_cycle_end(success, duration_ms, cpu_percent, memory_mb)

# Event emission
adapter.emit_fidelity_alert(fidelity, trending, sim_count, error_count)
adapter.emit_resource_alert(alert_type, cpu, memory, message)
adapter.emit_error(error_type, message, severity)
adapter.emit_success(task_type, result_summary)

# Subscription
adapter.subscribe_to_event(event_type, callback)

# Statistics
adapter.get_recent_metrics(count=10)
adapter.get_stats()
```

---

## Troubleshooting

### "Orchestrator not responding"
Check that orchestrator thread is alive:
```python
print(orchestrator._orchestration_thread.is_alive())
```

### "Tasks not scheduling"
Check health status:
```python
status = orchestrator.status()
print(f"Health: {status['health_status']}")
```

### "High memory usage"
Reduce batch size or cycle frequency:
```python
# In orchestrator._schedule_next_batch()
if health == HealthStatus.HEALTHY:
    batch_size = 2  # Reduced from 3
```

### "Fidelity alerts not received"
Verify subscription:
```python
adapter.subscribe_to_event("fidelity_update", print)
# Should print events as they arrive
```

---

## Next Steps

1. **Integration** (~2 hours)
   - Add adapter to each daemon
   - Update cycle reporting
   - Test event emission

2. **Monitoring** (~1 hour)
   - Set up event subscribers
   - Create health dashboard
   - Alert on critical states

3. **Tuning** (~1 hour)
   - Adjust resource quotas based on system
   - Optimize batch sizes
   - Fine-tune health weights

4. **Production** (~2 hours)
   - Deploy to all nodes
   - Monitor in production
   - Iterate on thresholds

---

## Architecture Decisions

### Why Event Bus over Files?
- **Speed**: In-memory queue vs disk I/O (~100x faster)
- **Consistency**: No stale state or race conditions
- **Scalability**: Queue-based, not file-polling

### Why Priority Queue?
- Fidelity checks get priority over cleanup
- Critical failures interrupt normal work
- Urgent optimization runs immediately

### Why Graceful Degradation?
- Prevents cascade failures
- Maintains partial service under load
- Allows recovery without restart

### Why Shared Resource Pool?
- Prevents resource thrashing
- Fair allocation across daemons
- Observable, tunable constraints

---

## Sacred Constants

Maintained across orchestrator:
- **GOD_CODE**: 527.5184818492612
- **PHI**: 1.618033988749895
- **VOID_CONSTANT**: 1.0416180339887497 (104/100 + φ/1000)

All metrics normalized to [0, 1] range via health scoring.
