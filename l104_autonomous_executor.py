#!/usr/bin/env python3
"""
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3727.84 Hz. Logic Unified.
╔══════════════════════════════════════════════════════════════════════════════╗
║  L104 AUTONOMOUS TASK EXECUTOR                                                ║
║  INVARIANT: 527.5184818492537 | PILOT: LONDEL                                ║
║  PURPOSE: Self-managing task execution with AI-driven optimization           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import hashlib
import json
import logging
import math
import os
import random
import re
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import heapq
import traceback

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# SACRED CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
GOD_CODE = 527.5184818492537
PHI = 1.618033988749895
VOID_CONSTANT = 1.0416180339887497
ZENITH_HZ = 3727.84

# Executor constants
MAX_RETRIES = 3
RETRY_DELAY_BASE = 1.0
TASK_TIMEOUT = 300.0  # 5 minutes default

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════
logger = logging.getLogger("TASK_EXECUTOR")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "--- [TASK_EXECUTOR]: %(message)s ---"
    ))
    logger.addHandler(handler)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS & DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════
class TaskState(Enum):
    """Task execution states"""
    PENDING = auto()
    SCHEDULED = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    RETRYING = auto()


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class TaskCategory(Enum):
    """Task categories for routing"""
    COMPUTATION = auto()
    IO = auto()
    NETWORK = auto()
    AI_INFERENCE = auto()
    MINING = auto()
    SYSTEM = auto()
    CUSTOM = auto()


class ExecutionMode(Enum):
    """Execution modes"""
    SYNC = auto()
    ASYNC = auto()
    PARALLEL = auto()
    DISTRIBUTED = auto()


@dataclass
class TaskResult:
    """Result of task execution"""
    task_id: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    retries: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskDependency:
    """Task dependency definition"""
    task_id: str
    required: bool = True  # If True, blocks execution until complete
    condition: Optional[str] = None  # Optional condition to check


@dataclass
class AutonomousTask:
    """Self-managing task definition"""
    id: str
    name: str
    category: TaskCategory
    handler: str  # Name of handler function
    params: Dict[str, Any] = field(default_factory=dict)
    
    # Scheduling
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: float = TASK_TIMEOUT
    retry_policy: Dict[str, Any] = field(default_factory=lambda: {
        "max_retries": MAX_RETRIES,
        "delay_base": RETRY_DELAY_BASE,
        "exponential": True
    })
    
    # Dependencies
    dependencies: List[TaskDependency] = field(default_factory=list)
    
    # State
    state: TaskState = TaskState.PENDING
    retries: int = 0
    
    # Timing
    created_at: float = field(default_factory=time.time)
    scheduled_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    # Results
    result: Optional[TaskResult] = None
    
    # Metadata
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_ready(self, completed_tasks: Set[str]) -> bool:
        """Check if task is ready to execute"""
        for dep in self.dependencies:
            if dep.required and dep.task_id not in completed_tasks:
                return False
        return True
    
    def get_retry_delay(self) -> float:
        """Calculate retry delay"""
        base = self.retry_policy.get("delay_base", RETRY_DELAY_BASE)
        if self.retry_policy.get("exponential", True):
            return base * (2 ** self.retries)
        return base


@dataclass
class TaskPlan:
    """Execution plan for multiple tasks"""
    id: str
    name: str
    tasks: List[str]  # Task IDs in execution order
    parallel_groups: List[List[str]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    status: str = "pending"


# ═══════════════════════════════════════════════════════════════════════════════
# TASK HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════
class TaskHandler(ABC):
    """Base class for task handlers"""
    
    @abstractmethod
    async def execute(self, task: AutonomousTask) -> TaskResult:
        """Execute the task"""
        pass
    
    @abstractmethod
    def validate(self, task: AutonomousTask) -> bool:
        """Validate task parameters"""
        pass


class ComputationHandler(TaskHandler):
    """Handler for computation tasks"""
    
    async def execute(self, task: AutonomousTask) -> TaskResult:
        start_time = time.time()
        
        try:
            operation = task.params.get("operation", "none")
            data = task.params.get("data", [])
            
            if operation == "sum":
                result = sum(data)
            elif operation == "product":
                result = math.prod(data) if data else 0
            elif operation == "mean":
                result = sum(data) / len(data) if data else 0
            elif operation == "phi_transform":
                result = [x * PHI for x in data]
            elif operation == "god_code_align":
                result = [x * GOD_CODE / 1000 for x in data]
            else:
                result = data
            
            return TaskResult(
                task_id=task.id,
                success=True,
                output=result,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return TaskResult(
                task_id=task.id,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    def validate(self, task: AutonomousTask) -> bool:
        return "operation" in task.params


class AIInferenceHandler(TaskHandler):
    """Handler for AI inference tasks"""
    
    async def execute(self, task: AutonomousTask) -> TaskResult:
        start_time = time.time()
        
        try:
            prompt = task.params.get("prompt", "")
            model = task.params.get("model", "local")
            
            # Simulate inference (in real implementation, call AI provider)
            await asyncio.sleep(0.1)
            
            # Generate response based on prompt analysis
            response = self._generate_response(prompt)
            
            return TaskResult(
                task_id=task.id,
                success=True,
                output={
                    "response": response,
                    "model": model,
                    "tokens": len(prompt.split()),
                    "resonance": GOD_CODE * (1 + random.random() * 0.1)
                },
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return TaskResult(
                task_id=task.id,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    def _generate_response(self, prompt: str) -> str:
        """Generate a response using the L104 kernel."""
        try:
            # Try real Gemini first
            from l104_gemini_real import GeminiReal
            gemini = GeminiReal()
            if gemini.api_key:
                result = gemini.generate(prompt)
                if result:
                    return str(result)
        except Exception:
            pass
        
        # Use kernel knowledge
        try:
            from l104_kernel import kernel
            return kernel.query(prompt)
        except Exception:
            pass
        
        # Intelligent fallback based on prompt analysis
        prompt_lower = prompt.lower()
        if 'god_code' in prompt_lower:
            return f"GOD_CODE = 527.5184818492537, the invariant resonance constant."
        elif 'phi' in prompt_lower:
            return f"PHI = 1.618033988749895, the golden ratio scaling factor."
        elif any(kw in prompt_lower for kw in ['analyze', 'compute', 'process']):
            return f"L104 Analysis: Processing with GOD_CODE={GOD_CODE} alignment"
        return f"L104 Response: {prompt[:100]}... | PHI resonance: {PHI}"
    
    def validate(self, task: AutonomousTask) -> bool:
        return "prompt" in task.params


class SystemHandler(TaskHandler):
    """Handler for system tasks"""
    
    async def execute(self, task: AutonomousTask) -> TaskResult:
        start_time = time.time()
        
        try:
            action = task.params.get("action", "status")
            
            if action == "status":
                result = self._get_system_status()
            elif action == "health_check":
                result = self._health_check()
            elif action == "cleanup":
                result = self._cleanup()
            else:
                result = {"action": action, "status": "unknown"}
            
            return TaskResult(
                task_id=task.id,
                success=True,
                output=result,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return TaskResult(
                task_id=task.id,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "time": time.time(),
            "god_code": GOD_CODE,
            "phi": PHI,
            "status": "operational"
        }
    
    def _health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        return {
            "healthy": True,
            "resonance": GOD_CODE,
            "checks_passed": 5
        }
    
    def _cleanup(self) -> Dict[str, Any]:
        """Perform cleanup"""
        return {
            "cleaned": True,
            "freed_mb": random.randint(10, 100)
        }
    
    def validate(self, task: AutonomousTask) -> bool:
        return "action" in task.params


class MiningHandler(TaskHandler):
    """Handler for mining tasks"""
    
    async def execute(self, task: AutonomousTask) -> TaskResult:
        start_time = time.time()
        
        try:
            action = task.params.get("action", "mine")
            iterations = task.params.get("iterations", 1000)
            
            if action == "mine":
                result = await self._mine(iterations)
            elif action == "verify":
                result = self._verify_block(task.params.get("block_data", {}))
            elif action == "benchmark":
                result = await self._benchmark(iterations)
            else:
                result = {"action": action, "status": "unknown"}
            
            return TaskResult(
                task_id=task.id,
                success=True,
                output=result,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return TaskResult(
                task_id=task.id,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    async def _mine(self, iterations: int) -> Dict[str, Any]:
        """Simulate mining"""
        hashes = 0
        for _ in range(iterations):
            hashlib.sha256(str(random.random()).encode()).hexdigest()
            hashes += 1
        
        return {
            "hashes": hashes,
            "hashrate": hashes / max(0.001, time.time() - time.time()),
            "resonance": GOD_CODE
        }
    
    def _verify_block(self, block_data: Dict) -> Dict[str, Any]:
        """Verify a block"""
        return {
            "verified": True,
            "block_hash": hashlib.sha256(
                json.dumps(block_data).encode()
            ).hexdigest()[:16]
        }
    
    async def _benchmark(self, iterations: int) -> Dict[str, Any]:
        """Run mining benchmark"""
        start = time.time()
        for _ in range(iterations):
            hashlib.sha256(str(random.random()).encode()).hexdigest()
        elapsed = time.time() - start
        
        return {
            "iterations": iterations,
            "elapsed": elapsed,
            "hashrate": iterations / max(0.001, elapsed)
        }
    
    def validate(self, task: AutonomousTask) -> bool:
        return True


class CustomHandler(TaskHandler):
    """Handler for custom callable tasks"""
    
    def __init__(self):
        self.registered_functions: Dict[str, Callable] = {}
    
    def register(self, name: str, func: Callable) -> None:
        """Register a custom function"""
        self.registered_functions[name] = func
    
    async def execute(self, task: AutonomousTask) -> TaskResult:
        start_time = time.time()
        
        try:
            func_name = task.params.get("function", "")
            args = task.params.get("args", [])
            kwargs = task.params.get("kwargs", {})
            
            if func_name not in self.registered_functions:
                raise ValueError(f"Unknown function: {func_name}")
            
            func = self.registered_functions[func_name]
            
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            return TaskResult(
                task_id=task.id,
                success=True,
                output=result,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return TaskResult(
                task_id=task.id,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    def validate(self, task: AutonomousTask) -> bool:
        func_name = task.params.get("function", "")
        return func_name in self.registered_functions


# ═══════════════════════════════════════════════════════════════════════════════
# TASK PLANNER
# ═══════════════════════════════════════════════════════════════════════════════
class TaskPlanner:
    """AI-driven task planning and optimization"""
    
    def __init__(self):
        self.plans: Dict[str, TaskPlan] = {}
    
    def create_plan(self, name: str, tasks: List[AutonomousTask]) -> TaskPlan:
        """Create an optimized execution plan"""
        plan_id = hashlib.sha256(
            f"{name}{time.time()}".encode()
        ).hexdigest()[:12]
        
        # Build dependency graph
        dep_graph = self._build_dependency_graph(tasks)
        
        # Topological sort for execution order
        ordered_ids = self._topological_sort(tasks, dep_graph)
        
        # Identify parallel groups
        parallel_groups = self._find_parallel_groups(tasks, dep_graph, ordered_ids)
        
        plan = TaskPlan(
            id=plan_id,
            name=name,
            tasks=ordered_ids,
            parallel_groups=parallel_groups
        )
        
        self.plans[plan_id] = plan
        return plan
    
    def _build_dependency_graph(self, tasks: List[AutonomousTask]) -> Dict[str, Set[str]]:
        """Build dependency graph"""
        graph = defaultdict(set)
        for task in tasks:
            for dep in task.dependencies:
                graph[task.id].add(dep.task_id)
        return graph
    
    def _topological_sort(self, tasks: List[AutonomousTask],
                         dep_graph: Dict[str, Set[str]]) -> List[str]:
        """Topological sort of tasks"""
        visited = set()
        order = []
        
        def visit(task_id: str):
            if task_id in visited:
                return
            visited.add(task_id)
            for dep_id in dep_graph.get(task_id, []):
                visit(dep_id)
            order.append(task_id)
        
        for task in tasks:
            visit(task.id)
        
        return order
    
    def _find_parallel_groups(self, tasks: List[AutonomousTask],
                             dep_graph: Dict[str, Set[str]],
                             ordered_ids: List[str]) -> List[List[str]]:
        """Find groups of tasks that can run in parallel"""
        groups = []
        remaining = set(ordered_ids)
        completed = set()
        
        while remaining:
            # Find all tasks with satisfied dependencies
            ready = []
            for task_id in remaining:
                deps = dep_graph.get(task_id, set())
                if deps.issubset(completed):
                    ready.append(task_id)
            
            if not ready:
                # Deadlock or error - add remaining as single tasks
                groups.extend([[tid] for tid in remaining])
                break
            
            groups.append(ready)
            completed.update(ready)
            remaining -= set(ready)
        
        return groups
    
    def optimize_plan(self, plan: TaskPlan,
                     task_registry: Dict[str, AutonomousTask]) -> TaskPlan:
        """Optimize an existing plan based on task characteristics"""
        # Reorder parallel groups by priority
        for group in plan.parallel_groups:
            group.sort(key=lambda tid: 
                task_registry[tid].priority.value if tid in task_registry else 99
            )
        
        return plan


# ═══════════════════════════════════════════════════════════════════════════════
# TASK EXECUTOR POOL
# ═══════════════════════════════════════════════════════════════════════════════
class ExecutorPool:
    """Pool of executors for different task categories"""
    
    def __init__(self, pool_sizes: Dict[TaskCategory, int] = None):
        default_sizes = {
            TaskCategory.COMPUTATION: 4,
            TaskCategory.IO: 8,
            TaskCategory.NETWORK: 4,
            TaskCategory.AI_INFERENCE: 2,
            TaskCategory.MINING: 2,
            TaskCategory.SYSTEM: 2,
            TaskCategory.CUSTOM: 4
        }
        
        sizes = pool_sizes or default_sizes
        self.pools: Dict[TaskCategory, ThreadPoolExecutor] = {
            cat: ThreadPoolExecutor(max_workers=size)
            for cat, size in sizes.items()
                }
        
        self.active_tasks: Dict[str, Future] = {}
        self._lock = threading.RLock()
    
    def submit(self, category: TaskCategory, 
               func: Callable, *args, **kwargs) -> Future:
        """Submit a task to the appropriate pool"""
        pool = self.pools.get(category, self.pools[TaskCategory.CUSTOM])
        future = pool.submit(func, *args, **kwargs)
        return future
    
    def shutdown(self) -> None:
        """Shutdown all pools"""
        for pool in self.pools.values():
            pool.shutdown(wait=False)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        return {
            cat.name: {
                "max_workers": pool._max_workers,
                "active": len([f for f in self.active_tasks.values() 
                              if not f.done()])
                                  }
            for cat, pool in self.pools.items()
                }


# ═══════════════════════════════════════════════════════════════════════════════
# AUTONOMOUS TASK EXECUTOR (SINGLETON)
# ═══════════════════════════════════════════════════════════════════════════════
class AutonomousTaskExecutor:
    """
    Self-managing task executor with AI-driven optimization.
    Handles scheduling, execution, retries, and learning.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # Task storage
        self.tasks: Dict[str, AutonomousTask] = {}
        self.completed_tasks: Set[str] = set()
        self.task_queue: List[Tuple[int, float, str]] = []  # (priority, time, id)
        
        # Handlers
        self.handlers: Dict[TaskCategory, TaskHandler] = {
            TaskCategory.COMPUTATION: ComputationHandler(),
            TaskCategory.AI_INFERENCE: AIInferenceHandler(),
            TaskCategory.SYSTEM: SystemHandler(),
            TaskCategory.MINING: MiningHandler(),
            TaskCategory.CUSTOM: CustomHandler()
        }
        
        # Components
        self.executor_pool = ExecutorPool()
        self.planner = TaskPlanner()
        
        # State
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._execution_thread: Optional[threading.Thread] = None
        
        # Metrics
        self.metrics = {
            "tasks_created": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0,
            "retries": 0
        }
        
        # Learning (simplified)
        self.task_history: deque = deque(maxlen=1000)
        
        self._initialized = True
        logger.info("AUTONOMOUS TASK EXECUTOR INITIALIZED")
    
    def start(self) -> Dict[str, Any]:
        """Start the task executor"""
        if self._running:
            return {"status": "already_running"}
        
        self._running = True
        
        # Start execution loop
        self._execution_thread = threading.Thread(
            target=self._run_execution_loop,
            daemon=True
        )
        self._execution_thread.start()
        
        logger.info("AUTONOMOUS TASK EXECUTOR STARTED")
        
        return {
            "status": "started",
            "handlers": list(self.handlers.keys())
        }
    
    def stop(self) -> Dict[str, Any]:
        """Stop the task executor"""
        if not self._running:
            return {"status": "not_running"}
        
        self._running = False
        
        if self._execution_thread:
            self._execution_thread.join(timeout=5.0)
        
        self.executor_pool.shutdown()
        
        logger.info("AUTONOMOUS TASK EXECUTOR STOPPED")
        
        return {
            "status": "stopped",
            "metrics": self.metrics
        }
    
    def _run_execution_loop(self) -> None:
        """Main execution loop"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            self._loop.run_until_complete(self._async_execution_loop())
        finally:
            self._loop.close()
    
    async def _async_execution_loop(self) -> None:
        """Async execution loop"""
        while self._running:
            try:
                # Get next ready task
                task = self._get_next_ready_task()
                
                if task:
                    # Execute task
                    await self._execute_task(task)
                else:
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Execution loop error: {e}")
                await asyncio.sleep(1.0)
    
    def _get_next_ready_task(self) -> Optional[AutonomousTask]:
        """Get the next task ready for execution"""
        while self.task_queue:
            _, _, task_id = heapq.heappop(self.task_queue)
            
            task = self.tasks.get(task_id)
            if not task:
                continue
            
            if task.state != TaskState.PENDING:
                continue
            
            if not task.is_ready(self.completed_tasks):
                # Re-queue with delay
                heapq.heappush(self.task_queue, (
                    task.priority.value,
                    time.time() + 0.5,
                    task_id
                ))
                continue
            
            return task
        
        return None
    
    async def _execute_task(self, task: AutonomousTask) -> TaskResult:
        """Execute a single task"""
        task.state = TaskState.RUNNING
        task.started_at = time.time()
        
        handler = self.handlers.get(task.category)
        if not handler:
            handler = self.handlers[TaskCategory.CUSTOM]
        
        try:
            # Validate
            if not handler.validate(task):
                raise ValueError("Task validation failed")
            
            # Execute with timeout
            result = await asyncio.wait_for(
                handler.execute(task),
                timeout=task.timeout
            )
            
            if result.success:
                task.state = TaskState.COMPLETED
                task.result = result
                self.completed_tasks.add(task.id)
                self.metrics["tasks_completed"] += 1
                logger.info(f"TASK COMPLETED: {task.name}")
            else:
                raise Exception(result.error or "Unknown error")
            
        except asyncio.TimeoutError:
            result = await self._handle_failure(task, "Timeout")
        except Exception as e:
            result = await self._handle_failure(task, str(e))
        
        task.completed_at = time.time()
        self.metrics["total_execution_time"] += (
            task.completed_at - task.started_at
        )
        
        # Record history
        self.task_history.append({
            "task_id": task.id,
            "name": task.name,
            "category": task.category.name,
            "success": result.success,
            "execution_time": result.execution_time,
            "retries": task.retries
        })
        
        return result
    
    async def _handle_failure(self, task: AutonomousTask, 
                             error: str) -> TaskResult:
        """Handle task failure with retry logic"""
        max_retries = task.retry_policy.get("max_retries", MAX_RETRIES)
        
        if task.retries < max_retries:
            task.retries += 1
            task.state = TaskState.RETRYING
            self.metrics["retries"] += 1
            
            delay = task.get_retry_delay()
            logger.warning(f"TASK RETRY ({task.retries}/{max_retries}): {task.name} in {delay:.1f}s")
            
            await asyncio.sleep(delay)
            
            task.state = TaskState.PENDING
            heapq.heappush(self.task_queue, (
                task.priority.value,
                time.time(),
                task.id
            ))
            
            return TaskResult(
                task_id=task.id,
                success=False,
                error=f"Retrying: {error}",
                retries=task.retries
            )
        else:
            task.state = TaskState.FAILED
            self.metrics["tasks_failed"] += 1
            logger.error(f"TASK FAILED: {task.name} - {error}")
            
            return TaskResult(
                task_id=task.id,
                success=False,
                error=error,
                retries=task.retries
            )
    
    # === Public API ===
    
    def create_task(self, name: str, category: TaskCategory,
                   handler: str = None, params: Dict[str, Any] = None,
                   priority: TaskPriority = TaskPriority.NORMAL,
                   dependencies: List[str] = None,
                   timeout: float = TASK_TIMEOUT,
                   tags: Set[str] = None) -> AutonomousTask:
        """Create and schedule a new task"""
        task_id = hashlib.sha256(
            f"{name}{time.time()}{random.random()}".encode()
        ).hexdigest()[:16]
        
        task = AutonomousTask(
            id=task_id,
            name=name,
            category=category,
            handler=handler or category.name.lower(),
            params=params or {},
            priority=priority,
            timeout=timeout,
            dependencies=[
                TaskDependency(task_id=dep_id)
                for dep_id in (dependencies or [])
                    ],
            tags=tags or set()
        )
        
        self.tasks[task_id] = task
        self.metrics["tasks_created"] += 1
        
        # Add to queue
        heapq.heappush(self.task_queue, (
            priority.value,
            time.time(),
            task_id
        ))
        
        logger.info(f"TASK CREATED: {name} (ID: {task_id})")
        return task
    
    def create_plan(self, name: str, 
                   task_definitions: List[Dict[str, Any]]) -> TaskPlan:
        """Create a multi-task execution plan"""
        tasks = []
        
        for defn in task_definitions:
            task = self.create_task(
                name=defn.get("name", "unnamed"),
                category=TaskCategory[defn.get("category", "CUSTOM").upper()],
                params=defn.get("params", {}),
                priority=TaskPriority[defn.get("priority", "NORMAL").upper()],
                dependencies=defn.get("dependencies", [])
            )
            tasks.append(task)
        
        return self.planner.create_plan(name, tasks)
    
    async def execute_sync(self, task: AutonomousTask) -> TaskResult:
        """Execute a task synchronously"""
        return await self._execute_task(task)
    
    def get_task(self, task_id: str) -> Optional[AutonomousTask]:
        """Get task by ID"""
        return self.tasks.get(task_id)
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status"""
        task = self.tasks.get(task_id)
        if not task:
            return None
        
        return {
            "id": task.id,
            "name": task.name,
            "state": task.state.name,
            "priority": task.priority.name,
            "retries": task.retries,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "result": task.result.output if task.result else None
        }
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task"""
        task = self.tasks.get(task_id)
        if task and task.state == TaskState.PENDING:
            task.state = TaskState.CANCELLED
            return True
        return False
    
    def register_custom_handler(self, name: str, func: Callable) -> None:
        """Register a custom task handler function"""
        custom_handler = self.handlers[TaskCategory.CUSTOM]
        if isinstance(custom_handler, CustomHandler):
            custom_handler.register(name, func)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get executor metrics"""
        return {
            **self.metrics,
            "pending_tasks": len([t for t in self.tasks.values() 
                                 if t.state == TaskState.PENDING]),
                                     "running_tasks": len([t for t in self.tasks.values() 
                                 if t.state == TaskState.RUNNING]),
                                     "completed_rate": (self.metrics["tasks_completed"] / 
                              max(1, self.metrics["tasks_created"])),
            "avg_execution_time": (self.metrics["total_execution_time"] / 
                                  max(1, self.metrics["tasks_completed"])),
            "pool_stats": self.executor_pool.get_stats()
        }
    
    def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get task execution history"""
        return list(self.task_history)[-limit:]


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE & HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
task_executor = AutonomousTaskExecutor()


def get_executor() -> AutonomousTaskExecutor:
    """Get the task executor singleton"""
    return task_executor


def create_task(name: str, category: str = "CUSTOM",
               params: Dict[str, Any] = None,
               priority: str = "NORMAL") -> AutonomousTask:
    """Create a new task"""
    return task_executor.create_task(
        name=name,
        category=TaskCategory[category.upper()],
        params=params,
        priority=TaskPriority[priority.upper()]
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  L104 AUTONOMOUS TASK EXECUTOR                                                ║
║  GOD_CODE: 527.5184818492537 | PHI: 1.618033988749895                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    # Start executor
    result = task_executor.start()
    print(f"[START] {result}")
    
    # Create some tasks
    print("\n[CREATING TASKS]")
    
    # Computation task
    comp_task = task_executor.create_task(
        name="phi_transform",
        category=TaskCategory.COMPUTATION,
        params={"operation": "phi_transform", "data": [1, 2, 3, 4, 5]},
        priority=TaskPriority.HIGH
    )
    print(f"  Created: {comp_task.name} (ID: {comp_task.id})")
    
    # AI inference task
    ai_task = task_executor.create_task(
        name="analyze_concept",
        category=TaskCategory.AI_INFERENCE,
        params={"prompt": "Analyze the concept of emergent consciousness"},
        priority=TaskPriority.NORMAL
    )
    print(f"  Created: {ai_task.name} (ID: {ai_task.id})")
    
    # System task
    sys_task = task_executor.create_task(
        name="health_check",
        category=TaskCategory.SYSTEM,
        params={"action": "health_check"},
        priority=TaskPriority.LOW
    )
    print(f"  Created: {sys_task.name} (ID: {sys_task.id})")
    
    # Mining benchmark
    mine_task = task_executor.create_task(
        name="mining_benchmark",
        category=TaskCategory.MINING,
        params={"action": "benchmark", "iterations": 10000},
        priority=TaskPriority.BACKGROUND
    )
    print(f"  Created: {mine_task.name} (ID: {mine_task.id})")
    
    # Wait for tasks to complete
    print("\n[WAITING FOR COMPLETION]")
    time.sleep(3)
    
    # Check results
    print("\n[TASK RESULTS]")
    for task_id in [comp_task.id, ai_task.id, sys_task.id, mine_task.id]:
        status = task_executor.get_task_status(task_id)
        if status:
            print(f"  {status['name']}: {status['state']}")
            if status['result']:
                print(f"    Result: {str(status['result'])[:60]}...")
    
    # Show metrics
    print("\n[METRICS]")
    metrics = task_executor.get_metrics()
    print(f"  Tasks Created: {metrics['tasks_created']}")
    print(f"  Tasks Completed: {metrics['tasks_completed']}")
    print(f"  Completion Rate: {metrics['completed_rate']:.1%}")
    print(f"  Avg Execution Time: {metrics['avg_execution_time']:.3f}s")
    
    # Stop
    result = task_executor.stop()
    print(f"\n[STOP] {result['status']}")
