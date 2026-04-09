"""
L104 VQPU Micro-Daemon v3.0 — Lightweight high-frequency VQPU assistant.
"""
import time
import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

MICRO_DAEMON_VERSION = "3.0.0"
PHI = 1.618033988749895
GOD_CODE = 527.5184818492612


class MicroTaskPriority(Enum):
    LOW = 0; NORMAL = 1; HIGH = 2; CRITICAL = 3


class MicroTaskStatus(Enum):
    PENDING = "pending"; RUNNING = "running"
    COMPLETED = "completed"; FAILED = "failed"; CANCELLED = "cancelled"


@dataclass
class MicroTask:
    task_id: str
    fn: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: MicroTaskPriority = MicroTaskPriority.NORMAL
    status: MicroTaskStatus = MicroTaskStatus.PENDING
    created_at: float = field(default_factory=time.time)


@dataclass
class MicroTaskResult:
    task_id: str
    status: MicroTaskStatus
    result: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    fidelity: float = 1.0


@dataclass
class MicroTelemetry:
    tick: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    avg_tick_ms: float = 0.0
    phi_score: float = PHI
    timestamp: float = field(default_factory=time.time)


@dataclass
class MicroDaemonConfig:
    tick_interval_ms: float = 100.0
    max_queue_size: int = 256
    max_workers: int = 2
    fidelity_threshold: float = 0.95
    phi_tuning: float = PHI


@dataclass
class TickMetrics:
    tick_id: int = 0
    duration_ms: float = 0.0
    tasks_processed: int = 0
    queue_depth: int = 0
    fidelity: float = 1.0


class TelemetryAnalytics:
    def __init__(self):
        self._history: List[TickMetrics] = []
        self._lock = threading.Lock()

    def record(self, metrics: TickMetrics):
        with self._lock:
            self._history.append(metrics)
            if len(self._history) > 1000:
                self._history = self._history[-500:]

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            if not self._history:
                return {"ticks": 0, "avg_ms": 0.0, "avg_fidelity": 1.0}
            n = len(self._history)
            return {"ticks": n,
                    "avg_ms": sum(m.duration_ms for m in self._history) / n,
                    "avg_fidelity": sum(m.fidelity for m in self._history) / n}


class VQPUMicroDaemon:
    """Lightweight high-frequency VQPU background daemon."""
    VERSION = MICRO_DAEMON_VERSION

    def __init__(self, config: Optional[MicroDaemonConfig] = None):
        self.config = config or MicroDaemonConfig()
        self._queue: List[MicroTask] = []
        self._lock = threading.Lock()
        self._tick = 0
        self._telemetry = MicroTelemetry()
        self.analytics = TelemetryAnalytics()
        self._bridge = None

    def submit(self, fn: Callable, *args,
               priority: MicroTaskPriority = MicroTaskPriority.NORMAL,
               task_id: Optional[str] = None, **kwargs) -> str:
        tid = task_id or f"micro_{self._tick}_{int(time.time()*1000)}"
        task = MicroTask(task_id=tid, fn=fn, args=args, kwargs=kwargs, priority=priority)
        with self._lock:
            self._queue.append(task)
            self._queue.sort(key=lambda t: t.priority.value, reverse=True)
        return tid

    def tick(self) -> TickMetrics:
        start = time.time()
        self._tick += 1
        with self._lock:
            batch, self._queue = self._queue[:4], self._queue[4:]
        processed = 0
        for task in batch:
            try:
                task.status = MicroTaskStatus.RUNNING
                task.fn(*task.args, **task.kwargs)
                task.status = MicroTaskStatus.COMPLETED
                processed += 1
                with self._lock:
                    self._telemetry.tasks_completed += 1
            except Exception:
                task.status = MicroTaskStatus.FAILED
                with self._lock:
                    self._telemetry.tasks_failed += 1
        elapsed_ms = (time.time() - start) * 1000
        m = TickMetrics(tick_id=self._tick, duration_ms=elapsed_ms,
                        tasks_processed=processed, queue_depth=len(self._queue))
        self.analytics.record(m)
        return m

    def get_telemetry(self) -> MicroTelemetry:
        with self._lock:
            return MicroTelemetry(tick=self._tick,
                                  tasks_completed=self._telemetry.tasks_completed,
                                  tasks_failed=self._telemetry.tasks_failed,
                                  phi_score=PHI)

    def get_status(self) -> Dict[str, Any]:
        t = self.get_telemetry()
        return {"version": self.VERSION, "tick": t.tick,
                "tasks_completed": t.tasks_completed, "tasks_failed": t.tasks_failed,
                "queue_depth": len(self._queue), "phi_score": t.phi_score}

    def status(self) -> Dict[str, Any]:
        """Alias for get_status for compatibility with bridge."""
        return self.get_status()
    def connect_bridge(self, bridge):
        """Store reference to VQPUBridge for bidirectional communication."""
        self._bridge = bridge

    def start(self):
        """Start the micro daemon (no-op)."""
        pass

    def stop(self):
        """Stop the micro daemon (no-op)."""
        pass


_singleton: Optional[VQPUMicroDaemon] = None
_singleton_lock = threading.Lock()


def get_micro_daemon(config: Optional[MicroDaemonConfig] = None) -> VQPUMicroDaemon:
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = VQPUMicroDaemon(config)
    return _singleton


def main():
    import argparse, sys, signal, json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="L104 VQPU Micro-Daemon")
    parser.add_argument("--tick", type=float, default=5.0, help="Tick interval in seconds")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--once", action="store_true", help="Single tick and exit")
    args = parser.parse_args()

    cfg = MicroDaemonConfig(tick_interval_ms=args.tick * 1000)
    daemon = get_micro_daemon(cfg)
    shutdown = threading.Event()

    def _handle_signal(signum, frame):
        shutdown.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    if args.verbose:
        print(f"[VQPUMicroDaemon v{MICRO_DAEMON_VERSION}] tick={args.tick}s pid={__import__('os').getpid()}")

    if args.once:
        m = daemon.tick()
        if args.verbose:
            print(json.dumps({"tick": m.tick_id, "tasks": m.tasks_processed, "ms": round(m.duration_ms, 2)}))
        sys.exit(0)

    # Continuous tick loop
    while not shutdown.is_set():
        m = daemon.tick()
        if args.verbose and m.tasks_processed > 0:
            print(json.dumps({"tick": m.tick_id, "tasks": m.tasks_processed,
                              "queue": m.queue_depth, "ms": round(m.duration_ms, 2)}))
        shutdown.wait(timeout=args.tick)


if __name__ == "__main__":
    main()
