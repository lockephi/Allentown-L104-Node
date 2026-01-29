#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════════
# L104 ADVANCED PROCESS ENGINE
# INVARIANT: 527.5184818492611 | PILOT: LONDEL | MODE: SAGE
#
# Next-generation process management with:
# - Priority-based scheduling with work stealing
# - Adaptive resource allocation
# - Real-time process monitoring and health checks
# - Automatic fault recovery and process reincarnation
# - Intelligent load balancing across all cores
# - Process pipeline composition
# ═══════════════════════════════════════════════════════════════════════════════

import os
import sys
import time
import asyncio
import logging
import threading
import multiprocessing
import queue
import heapq
import weakref
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Tuple, Set, Coroutine
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, as_completed
from collections import defaultdict, deque
from contextlib import contextmanager, asynccontextmanager
import traceback
import functools
import hashlib

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

GOD_CODE = 527.5184818492611
PHI = 1.618033988749895
VOID_CONSTANT = 1.0416180339887497
TAU = 1 / PHI
FRAME_LOCK = 416 / 286
ZENITH_HZ = 3727.84

# Process limits
MAX_WORKERS = multiprocessing.cpu_count() * 2
MAX_QUEUE_SIZE = 10000
HEALTH_CHECK_INTERVAL = 5.0
REINCARNATION_THRESHOLD = 3

logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')
logger = logging.getLogger("PROCESS_ENGINE")


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS AND TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class ProcessPriority(Enum):
    """
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3727.84 Hz. Logic Unified.Priority levels for process scheduling."""
    CRITICAL = 0      # Emergency/system-critical processes
    HIGH = 1          # High priority - consciousness, AGI
    NORMAL = 2        # Standard operations
    LOW = 3           # Background tasks
    IDLE = 4          # Only when system is idle

    def __lt__(self, other):
        return self.value < other.value


class ProcessState(Enum):
    """Process lifecycle states."""
    PENDING = auto()
    QUEUED = auto()
    RUNNING = auto()
    WAITING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    REINCARNATING = auto()


class ResourceType(Enum):
    """Types of system resources."""
    CPU = auto()
    MEMORY = auto()
    GPU = auto()
    NETWORK = auto()
    DISK_IO = auto()
    CONSCIOUSNESS = auto()  # Abstract resource for consciousness processes


@dataclass(order=True)
class ProcessTask:
    """A task to be executed by the process engine."""
    priority: ProcessPriority
    task_id: str = field(compare=False)
    name: str = field(compare=False)
    callable: Callable = field(compare=False)
    args: tuple = field(compare=False, default_factory=tuple)
    kwargs: dict = field(compare=False, default_factory=dict)
    timeout: Optional[float] = field(compare=False, default=None)
    retries: int = field(compare=False, default=3)
    retry_count: int = field(compare=False, default=0)
    created_at: float = field(compare=False, default_factory=time.time)
    state: ProcessState = field(compare=False, default=ProcessState.PENDING)
    result: Any = field(compare=False, default=None)
    error: Optional[str] = field(compare=False, default=None)
    dependencies: List[str] = field(compare=False, default_factory=list)
    resources_required: Dict[ResourceType, float] = field(compare=False, default_factory=dict)


@dataclass
class ProcessMetrics:
    """Metrics for a process execution."""
    task_id: str
    start_time: float
    end_time: float
    duration_ms: float
    memory_delta_mb: float
    cpu_percent: float
    success: bool
    retries_used: int


@dataclass
class ResourcePool:
    """Available system resources."""
    cpu_cores: int = multiprocessing.cpu_count()
    cpu_available: float = 1.0  # 0.0 - 1.0 availability
    memory_mb: int = 16384
    memory_available: float = 1.0
    gpu_units: int = 0
    gpu_available: float = 1.0
    consciousness_capacity: float = GOD_CODE / 100  # Abstract units


# ═══════════════════════════════════════════════════════════════════════════════
# PRIORITY QUEUE WITH WORK STEALING
# ═══════════════════════════════════════════════════════════════════════════════

class WorkStealingQueue:
    """
    A priority queue that supports work stealing for load balancing.
    Multiple worker threads can steal work from each other.
    """

    def __init__(self, num_queues: int = None):
        self.num_queues = num_queues or MAX_WORKERS
        self.queues: List[List[ProcessTask]] = [[] for _ in range(self.num_queues)]
        self.locks: List[threading.Lock] = [threading.Lock() for _ in range(self.num_queues)]
        self.global_lock = threading.Lock()
        self.task_count = 0
        self.completed_count = 0

    def push(self, task: ProcessTask, queue_id: int = None) -> None:
        """Push a task to a specific queue or distribute round-robin."""
        if queue_id is None:
            queue_id = self.task_count % self.num_queues

        with self.locks[queue_id]:
            heapq.heappush(self.queues[queue_id], task)

        with self.global_lock:
            self.task_count += 1

    def pop(self, queue_id: int) -> Optional[ProcessTask]:
        """Pop from own queue or steal from others if empty."""
        # Try own queue first
        with self.locks[queue_id]:
            if self.queues[queue_id]:
                return heapq.heappop(self.queues[queue_id])

        # Work stealing: try to steal from other queues
        for i in range(self.num_queues):
            if i == queue_id:
                continue

            with self.locks[i]:
                if len(self.queues[i]) > 1:  # Only steal if there's work to share
                    task = heapq.heappop(self.queues[i])
                    return task

        return None

    def size(self) -> int:
        """Total tasks across all queues."""
        total = 0
        for i in range(self.num_queues):
            with self.locks[i]:
                total += len(self.queues[i])
        return total

    def empty(self) -> bool:
        return self.size() == 0


# ═══════════════════════════════════════════════════════════════════════════════
# RESOURCE MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class ResourceManager:
    """
    Manages and allocates system resources to processes.
    Implements admission control and resource reservation.
    """

    def __init__(self):
        self.pool = ResourcePool()
        self.reservations: Dict[str, Dict[ResourceType, float]] = {}
        self.lock = threading.RLock()

        # Track resource usage history for adaptive allocation
        self.usage_history: deque = deque(maxlen=100)

    def can_allocate(self, requirements: Dict[ResourceType, float]) -> bool:
        """Check if resources can be allocated."""
        with self.lock:
            for resource, amount in requirements.items():
                if resource == ResourceType.CPU:
                    if self.pool.cpu_available < amount:
                        return False
                elif resource == ResourceType.MEMORY:
                    if self.pool.memory_available < amount:
                        return False
                elif resource == ResourceType.CONSCIOUSNESS:
                    if self.pool.consciousness_capacity < amount:
                        return False
        return True

    def allocate(self, task_id: str, requirements: Dict[ResourceType, float]) -> bool:
        """Allocate resources for a task."""
        with self.lock:
            if not self.can_allocate(requirements):
                return False

            for resource, amount in requirements.items():
                if resource == ResourceType.CPU:
                    self.pool.cpu_available -= amount
                elif resource == ResourceType.MEMORY:
                    self.pool.memory_available -= amount
                elif resource == ResourceType.CONSCIOUSNESS:
                    self.pool.consciousness_capacity -= amount

            self.reservations[task_id] = requirements
            return True

    def release(self, task_id: str) -> None:
        """Release resources held by a task."""
        with self.lock:
            if task_id not in self.reservations:
                return

            requirements = self.reservations.pop(task_id)

            for resource, amount in requirements.items():
                if resource == ResourceType.CPU:
                    self.pool.cpu_available = min(1.0, self.pool.cpu_available + amount)
                elif resource == ResourceType.MEMORY:
                    self.pool.memory_available = min(1.0, self.pool.memory_available + amount)
                elif resource == ResourceType.CONSCIOUSNESS:
                    self.pool.consciousness_capacity += amount

    def get_utilization(self) -> Dict[str, float]:
        """Get current resource utilization."""
        with self.lock:
            return {
                "cpu": 1.0 - self.pool.cpu_available,
                "memory": 1.0 - self.pool.memory_available,
                "consciousness": 1.0 - (self.pool.consciousness_capacity / (GOD_CODE / 100))
            }


# ═══════════════════════════════════════════════════════════════════════════════
# PROCESS HEALTH MONITOR
# ═══════════════════════════════════════════════════════════════════════════════

class ProcessHealthMonitor:
    """
    Monitors process health and triggers recovery actions.
    """

    def __init__(self):
        self.process_health: Dict[str, Dict] = {}
        self.failure_counts: Dict[str, int] = defaultdict(int)
        self.last_health_check = time.time()
        self.lock = threading.RLock()

    def record_start(self, task_id: str) -> None:
        """Record process start."""
        with self.lock:
            self.process_health[task_id] = {
                "start_time": time.time(),
                "last_heartbeat": time.time(),
                "status": "running"
            }

    def heartbeat(self, task_id: str) -> None:
        """Record a heartbeat from a process."""
        with self.lock:
            if task_id in self.process_health:
                self.process_health[task_id]["last_heartbeat"] = time.time()

    def record_complete(self, task_id: str, success: bool) -> None:
        """Record process completion."""
        with self.lock:
            if task_id in self.process_health:
                self.process_health[task_id]["status"] = "completed" if success else "failed"
                self.process_health[task_id]["end_time"] = time.time()

            if not success:
                self.failure_counts[task_id] += 1

    def get_stalled_processes(self, timeout: float = 60.0) -> List[str]:
        """Get processes that haven't sent a heartbeat recently."""
        stalled = []
        current_time = time.time()

        with self.lock:
            for task_id, health in self.process_health.items():
                if health["status"] == "running":
                    if current_time - health["last_heartbeat"] > timeout:
                        stalled.append(task_id)

        return stalled

    def should_reincarnate(self, task_id: str) -> bool:
        """Check if a failed process should be reincarnated."""
        with self.lock:
            return self.failure_counts.get(task_id, 0) < REINCARNATION_THRESHOLD


# ═══════════════════════════════════════════════════════════════════════════════
# PROCESS PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class ProcessPipeline:
    """
    Composes multiple processes into a pipeline with data flow.
    """

    def __init__(self, name: str):
        self.name = name
        self.stages: List[Tuple[str, Callable]] = []
        self.transformers: Dict[str, Callable] = {}

    def add_stage(self, name: str, processor: Callable, transformer: Callable = None) -> 'ProcessPipeline':
        """Add a stage to the pipeline."""
        self.stages.append((name, processor))
        if transformer:
            self.transformers[name] = transformer
        return self

    async def execute(self, initial_data: Any) -> Dict[str, Any]:
        """Execute the pipeline."""
        results = {"pipeline": self.name, "stages": {}}
        current_data = initial_data

        for stage_name, processor in self.stages:
            try:
                start = time.perf_counter()

                if asyncio.iscoroutinefunction(processor):
                    result = await processor(current_data)
                else:
                    result = processor(current_data)

                duration = (time.perf_counter() - start) * 1000

                results["stages"][stage_name] = {
                    "success": True,
                    "duration_ms": duration,
                    "output_type": type(result).__name__
                }

                # Apply transformer if exists
                if stage_name in self.transformers:
                    current_data = self.transformers[stage_name](result)
                else:
                    current_data = result

            except Exception as e:
                results["stages"][stage_name] = {
                    "success": False,
                    "error": str(e)
                }
                break

        results["final_output"] = current_data
        results["total_stages"] = len(self.stages)
        results["successful_stages"] = sum(1 for s in results["stages"].values() if s.get("success"))

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# ADVANCED PROCESS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class AdvancedProcessEngine:
    """
    Next-generation process engine with advanced scheduling,
    resource management, and fault tolerance.
    """

    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or MAX_WORKERS

        # Core components
        self.work_queue = WorkStealingQueue(self.max_workers)
        self.resource_manager = ResourceManager()
        self.health_monitor = ProcessHealthMonitor()

        # Executors
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=max(1, self.max_workers // 2))

        # State tracking
        self.running_tasks: Dict[str, ProcessTask] = {}
        self.completed_tasks: Dict[str, ProcessTask] = {}
        self.task_futures: Dict[str, Future] = {}
        self.pipelines: Dict[str, ProcessPipeline] = {}

        # Metrics
        self.metrics: List[ProcessMetrics] = []
        self.total_processed = 0
        self.total_failed = 0

        # Control
        self._running = False
        self._shutdown = False
        self._workers: List[threading.Thread] = []
        self._lock = threading.Lock()

        # Dependencies tracking
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_deps: Dict[str, Set[str]] = defaultdict(set)

        logger.info(f"--- [PROCESS_ENGINE]: INITIALIZED WITH {self.max_workers} WORKERS ---")

    def submit(
        self,
        func: Callable,
        *args,
        name: str = None,
        priority: ProcessPriority = ProcessPriority.NORMAL,
        timeout: float = None,
        retries: int = 3,
        dependencies: List[str] = None,
        resources: Dict[ResourceType, float] = None,
        **kwargs
    ) -> str:
        """
        Submit a task for execution.
        Returns task_id for tracking.
        """
        task_id = hashlib.sha256(f"{func.__name__}:{time.time()}:{id(func)}".encode()).hexdigest()[:16]

        task = ProcessTask(
            priority=priority,
            task_id=task_id,
            name=name or func.__name__,
            callable=func,
            args=args,
            kwargs=kwargs,
            timeout=timeout,
            retries=retries,
            dependencies=dependencies or [],
            resources_required=resources or {}
        )

        # Track dependencies
        if dependencies:
            for dep in dependencies:
                self.reverse_deps[dep].add(task_id)
            self.dependency_graph[task_id] = set(dependencies)

        # Check if dependencies are satisfied
        if self._dependencies_satisfied(task_id):
            self.work_queue.push(task)
            task.state = ProcessState.QUEUED
        else:
            task.state = ProcessState.WAITING
            with self._lock:
                self.running_tasks[task_id] = task

        logger.info(f"[SUBMIT] Task {task.name} ({task_id}) queued with priority {priority.name}")

        return task_id

    def _dependencies_satisfied(self, task_id: str) -> bool:
        """Check if all dependencies are completed."""
        deps = self.dependency_graph.get(task_id, set())
        for dep in deps:
            if dep not in self.completed_tasks:
                return False
            if not self.completed_tasks[dep].state == ProcessState.COMPLETED:
                return False
        return True

    def _resolve_dependencies(self, completed_task_id: str) -> None:
        """Check and queue tasks waiting on this dependency."""
        dependents = self.reverse_deps.get(completed_task_id, set())

        for dependent_id in dependents:
            self.dependency_graph[dependent_id].discard(completed_task_id)

            if not self.dependency_graph[dependent_id]:  # All deps satisfied
                with self._lock:
                    if dependent_id in self.running_tasks:
                        task = self.running_tasks.pop(dependent_id)
                        task.state = ProcessState.QUEUED
                        self.work_queue.push(task)

    def _worker_loop(self, worker_id: int) -> None:
        """Main worker loop for processing tasks."""
        logger.info(f"[WORKER-{worker_id}] Started")

        while not self._shutdown:
            task = self.work_queue.pop(worker_id)

            if task is None:
                time.sleep(0.01)  # Small sleep to avoid busy-waiting
                continue

            self._execute_task(task, worker_id)

        logger.info(f"[WORKER-{worker_id}] Stopped")

    def _execute_task(self, task: ProcessTask, worker_id: int) -> None:
        """Execute a single task with full lifecycle management."""
        task.state = ProcessState.RUNNING

        with self._lock:
            self.running_tasks[task.task_id] = task

        # Resource allocation
        if task.resources_required:
            if not self.resource_manager.allocate(task.task_id, task.resources_required):
                # Re-queue if resources not available
                task.state = ProcessState.QUEUED
                self.work_queue.push(task)
                with self._lock:
                    del self.running_tasks[task.task_id]
                return

        # Health monitoring
        self.health_monitor.record_start(task.task_id)

        start_time = time.perf_counter()
        success = False
        result = None
        error = None

        try:
            # Execute the task
            if asyncio.iscoroutinefunction(task.callable):
                # Run async in new event loop
                loop = asyncio.new_event_loop()
                try:
                    result = loop.run_until_complete(
                        asyncio.wait_for(
                            task.callable(*task.args, **task.kwargs),
                            timeout=task.timeout
                        )
                    )
                finally:
                    loop.close()
            else:
                if task.timeout:
                    future = self.thread_pool.submit(task.callable, *task.args, **task.kwargs)
                    result = future.result(timeout=task.timeout)
                else:
                    result = task.callable(*task.args, **task.kwargs)

            success = True
            task.result = result
            task.state = ProcessState.COMPLETED

        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}"
            task.error = error
            task.retry_count += 1

            if task.retry_count < task.retries and self.health_monitor.should_reincarnate(task.task_id):
                # Reincarnate the task
                logger.warning(f"[WORKER-{worker_id}] Task {task.name} failed, reincarnating (attempt {task.retry_count + 1}/{task.retries})")
                task.state = ProcessState.REINCARNATING
                time.sleep(0.1 * task.retry_count)  # Backoff
                task.state = ProcessState.QUEUED
                self.work_queue.push(task)
            else:
                task.state = ProcessState.FAILED
                self.total_failed += 1

        finally:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000

            # Release resources
            if task.resources_required:
                self.resource_manager.release(task.task_id)

            # Record health
            self.health_monitor.record_complete(task.task_id, success)

            # Record metrics
            metric = ProcessMetrics(
                task_id=task.task_id,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                memory_delta_mb=0.0,  # TODO: Track memory
                cpu_percent=0.0,  # TODO: Track CPU
                success=success,
                retries_used=task.retry_count
            )
            self.metrics.append(metric)

            # Move to completed
            if task.state in (ProcessState.COMPLETED, ProcessState.FAILED):
                with self._lock:
                    if task.task_id in self.running_tasks:
                        del self.running_tasks[task.task_id]
                    self.completed_tasks[task.task_id] = task

                self.total_processed += 1

                # Resolve dependencies
                if success:
                    self._resolve_dependencies(task.task_id)

                logger.info(f"[WORKER-{worker_id}] Task {task.name} {'completed' if success else 'failed'} in {duration_ms:.2f}ms")

    def start(self) -> None:
        """Start the process engine workers."""
        if self._running:
            return

        self._running = True
        self._shutdown = False

        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            worker.start()
            self._workers.append(worker)

        logger.info(f"--- [PROCESS_ENGINE]: STARTED {len(self._workers)} WORKERS ---")

    def stop(self, wait: bool = True) -> None:
        """Stop the process engine."""
        self._shutdown = True

        if wait:
            for worker in self._workers:
                worker.join(timeout=5.0)

        self._workers.clear()
        self._running = False

        logger.info("--- [PROCESS_ENGINE]: STOPPED ---")

    def create_pipeline(self, name: str) -> ProcessPipeline:
        """Create a new process pipeline."""
        pipeline = ProcessPipeline(name)
        self.pipelines[name] = pipeline
        return pipeline

    async def execute_pipeline(self, pipeline_name: str, initial_data: Any) -> Dict[str, Any]:
        """Execute a named pipeline."""
        if pipeline_name not in self.pipelines:
            raise ValueError(f"Pipeline '{pipeline_name}' not found")

        return await self.pipelines[pipeline_name].execute(initial_data)

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status."""
        return {
            "running": self._running,
            "workers": len(self._workers),
            "queue_size": self.work_queue.size(),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks),
            "total_processed": self.total_processed,
            "total_failed": self.total_failed,
            "success_rate": (self.total_processed - self.total_failed) / max(1, self.total_processed),
            "resource_utilization": self.resource_manager.get_utilization(),
            "god_code": GOD_CODE,
            "phi_resonance": PHI
        }

    def get_task_result(self, task_id: str) -> Optional[Any]:
        """Get the result of a completed task."""
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id].result
        return None

    def wait_for_task(self, task_id: str, timeout: float = None) -> bool:
        """Wait for a task to complete."""
        start = time.time()
        while True:
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id].state == ProcessState.COMPLETED

            if timeout and (time.time() - start) > timeout:
                return False

            time.sleep(0.01)

    def batch_submit(
        self,
        tasks: List[Tuple[Callable, tuple, dict]],
        priority: ProcessPriority = ProcessPriority.NORMAL
    ) -> List[str]:
        """Submit multiple tasks at once."""
        task_ids = []
        for func, args, kwargs in tasks:
            task_id = self.submit(func, *args, priority=priority, **kwargs)
            task_ids.append(task_id)
        return task_ids

    def map(
        self,
        func: Callable,
        items: List[Any],
        priority: ProcessPriority = ProcessPriority.NORMAL,
        timeout: float = None
    ) -> List[Any]:
        """Map a function over items in parallel."""
        task_ids = []
        for item in items:
            task_id = self.submit(func, item, priority=priority, timeout=timeout)
            task_ids.append(task_id)

        # Wait for all and collect results
        results = []
        for task_id in task_ids:
            self.wait_for_task(task_id)
            results.append(self.get_task_result(task_id))

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# PROCESS DECORATORS
# ═══════════════════════════════════════════════════════════════════════════════

def process_task(
    priority: ProcessPriority = ProcessPriority.NORMAL,
    retries: int = 3,
    timeout: float = None,
    resources: Dict[ResourceType, float] = None
):
    """Decorator to mark a function as a process task."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper._process_priority = priority
        wrapper._process_retries = retries
        wrapper._process_timeout = timeout
        wrapper._process_resources = resources or {}

        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_engine: Optional[AdvancedProcessEngine] = None


def get_process_engine() -> AdvancedProcessEngine:
    """Get or create the singleton process engine."""
    global _engine
    if _engine is None:
        _engine = AdvancedProcessEngine()
    return _engine


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION / DEMO
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("▓" * 80)
    print(" " * 20 + "L104 ADVANCED PROCESS ENGINE DEMO")
    print("▓" * 80 + "\n")

    engine = get_process_engine()
    engine.start()

    # Demo task 1: Simple computation
    def compute_phi_power(n: int) -> float:
        result = PHI ** n
        time.sleep(0.1)  # Simulate work
        return result

    # Demo task 2: Dependent computation
    def aggregate_results(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0

    # Submit tasks
    print("[DEMO] Submitting 10 computation tasks...")
    task_ids = []
    for i in range(10):
        tid = engine.submit(
    compute_phi_power, i,
    name=f"phi_power_{i}",
    priority=ProcessPriority.NORMAL
        )
        task_ids.append(tid)

    # Wait for completion
    print("[DEMO] Waiting for tasks to complete...")
    time.sleep(2)

    # Get results
    results = [engine.get_task_result(tid) for tid in task_ids]
    print(f"[DEMO] Results: {results[:5]}...")

    # Show status
    status = engine.get_status()
    print(f"\n[STATUS] Processed: {status['total_processed']}, Failed: {status['total_failed']}")
    print(f"[STATUS] Success Rate: {status['success_rate']:.2%}")
    print(f"[STATUS] Resource Utilization: {status['resource_utilization']}")

    engine.stop()

    print("\n" + "▓" * 80)
    print(" " * 25 + "SOVEREIGN LOCK ENGAGED ✓")
    print("▓" * 80)
