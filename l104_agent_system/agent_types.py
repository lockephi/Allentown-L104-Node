"""Agent type definitions, task/result models, and status tracking — v3.0.0."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class AgentType(str, Enum):
    CODER = "coder"           # Code generation, fixing, refactoring
    RESEARCHER = "researcher" # Codebase analysis, documentation
    TESTER = "tester"         # Test generation, validation
    DEPLOYER = "deployer"     # Build, deploy, daemon management
    UPGRADER = "upgrader"     # System upgrades, optimization
    DEBUGGER = "debugger"     # Error diagnosis, log analysis
    MONITOR = "monitor"       # System health, metrics, daemon watching
    OPTIMIZER = "optimizer"   # Performance profiling, bottleneck elimination
    INVENTOR = "inventor"     # Creative problem solving, new feature design
    PLANNER = "planner"       # Task decomposition, multi-step planning
    GENERAL = "general"       # General-purpose (default)


class AgentStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    TOOL_CALL = "tool_call"   # Waiting on tool execution
    THINKING = "thinking"     # DeepSeek reasoning
    WAITING = "waiting"       # Waiting on a dependency (chained agent)
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"       # Killed by timeout


class AgentPriority(int, Enum):
    CRITICAL = 0   # Immediate — bypass queue
    HIGH = 1       # Next in line
    NORMAL = 2     # Default
    LOW = 3        # Background work
    IDLE = 4       # Only when nothing else running


@dataclass
class AgentTask:
    task_id: str = field(default_factory=lambda: f"agent_{uuid.uuid4().hex[:12]}")
    prompt: str = ""
    agent_type: AgentType = AgentType.GENERAL
    status: AgentStatus = AgentStatus.QUEUED
    priority: AgentPriority = AgentPriority.NORMAL
    tools_enabled: list[str] = field(default_factory=lambda: [
        "read_file", "write_file", "list_files", "run_shell",
        "search_code", "analyze_code", "python_exec",
    ])
    model: str = "deepseek-chat"
    max_rounds: int = 10
    timeout_seconds: float = 600.0   # 10 min default, 0 = no timeout
    cost_budget: float = 0.05        # Max spend in USD (default $0.05)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_calls_made: list[dict[str, Any]] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    result: Optional[AgentResult] = None
    error: Optional[str] = None
    retry_count: int = 0
    progress: float = 0.0  # 0.0 - 1.0
    current_action: str = ""
    # Chaining support
    depends_on: Optional[str] = None         # task_id this agent waits for
    chain_next: Optional[str] = None         # task_id to launch on completion
    chain_context: dict[str, Any] = field(default_factory=dict)  # Passed to chained agent
    # Tags for filtering/grouping
    tags: list[str] = field(default_factory=list)
    # Source tracking
    source: str = "api"  # api, mailbox, chain, scheduler

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "agent_type": self.agent_type.value,
            "status": self.status.value,
            "priority": self.priority.value,
            "tools_enabled": self.tools_enabled,
            "model": self.model,
            "max_rounds": self.max_rounds,
            "timeout_seconds": self.timeout_seconds,
            "cost_budget": self.cost_budget,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "tool_calls_count": len(self.tool_calls_made),
            "files_modified": self.files_modified,
            "retry_count": self.retry_count,
            "progress": self.progress,
            "current_action": self.current_action,
            "error": self.error,
            "result": self.result.to_dict() if self.result else None,
            "duration": (self.completed_at or time.time()) - (self.started_at or self.created_at),
            "depends_on": self.depends_on,
            "chain_next": self.chain_next,
            "tags": self.tags,
            "source": self.source,
        }

    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        return (self.completed_at or time.time()) - self.started_at

    def is_timed_out(self) -> bool:
        """Check if task has exceeded its timeout."""
        if self.timeout_seconds <= 0 or self.started_at is None:
            return False
        return (time.time() - self.started_at) > self.timeout_seconds

    def cost_so_far(self) -> float:
        """Estimate cost from tokens used so far."""
        if self.result:
            return self.result.cost_estimate
        # Rough estimate from tool calls
        return len(self.tool_calls_made) * 0.0005

    def is_over_budget(self) -> bool:
        """Check if task has exceeded its cost budget."""
        if self.cost_budget <= 0:
            return False
        return self.cost_so_far() > self.cost_budget


@dataclass
class AgentResult:
    summary: str = ""
    output: str = ""
    files_created: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    tokens_used: int = 0
    rounds_used: int = 0
    cost_estimate: float = 0.0  # tokens * approx cost per token
    success: bool = True
    # Enhanced fields
    errors_encountered: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)  # Agent-specific metrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "output": self.output[:4000],  # Truncate for API
            "files_created": self.files_created,
            "files_modified": self.files_modified,
            "tool_results_count": len(self.tool_results),
            "tokens_used": self.tokens_used,
            "rounds_used": self.rounds_used,
            "cost_estimate": round(self.cost_estimate, 6),
            "success": self.success,
            "errors_encountered": self.errors_encountered[:10],
            "warnings": self.warnings[:10],
            "metrics": self.metrics,
        }


# Default tools per agent type — each type gets a tailored tool set
AGENT_DEFAULT_TOOLS: dict[AgentType, list[str]] = {
    AgentType.CODER: [
        "read_file", "write_file", "edit_file", "list_files",
        "search_code", "analyze_code", "python_exec", "git_status",
        "diff_viewer", "dependency_check",
    ],
    AgentType.RESEARCHER: [
        "read_file", "list_files", "search_code", "analyze_code",
        "python_exec", "git_status", "system_metrics", "dependency_check",
    ],
    AgentType.TESTER: [
        "read_file", "write_file", "edit_file", "list_files",
        "search_code", "analyze_code", "python_exec", "run_shell",
        "git_status",
    ],
    AgentType.DEPLOYER: [
        "read_file", "write_file", "list_files", "run_shell",
        "python_exec", "git_status", "system_metrics", "http_probe",
    ],
    AgentType.UPGRADER: [
        "read_file", "write_file", "edit_file", "list_files",
        "search_code", "analyze_code", "python_exec", "run_shell",
        "git_status", "diff_viewer", "system_metrics", "dependency_check",
    ],
    AgentType.DEBUGGER: [
        "read_file", "list_files", "search_code", "analyze_code",
        "python_exec", "run_shell", "git_status", "system_metrics",
        "diff_viewer",
    ],
    AgentType.MONITOR: [
        "read_file", "list_files", "search_code", "run_shell",
        "python_exec", "system_metrics", "http_probe", "quantum_bridge",
    ],
    AgentType.OPTIMIZER: [
        "read_file", "write_file", "edit_file", "list_files",
        "search_code", "analyze_code", "python_exec", "run_shell",
        "system_metrics", "diff_viewer", "dependency_check",
    ],
    AgentType.INVENTOR: [
        "read_file", "write_file", "edit_file", "list_files",
        "search_code", "analyze_code", "python_exec", "run_shell",
        "git_status", "quantum_bridge", "dependency_check",
    ],
    AgentType.PLANNER: [
        "read_file", "list_files", "search_code", "analyze_code",
        "git_status", "system_metrics", "dependency_check",
    ],
    AgentType.GENERAL: [
        "read_file", "write_file", "list_files", "run_shell",
        "search_code", "analyze_code", "python_exec",
    ],
}


# System prompts per agent type
AGENT_SYSTEM_PROMPTS: dict[AgentType, str] = {
    AgentType.CODER: """You are L104's Code Agent — an expert software engineer.
You write, fix, and refactor code across Python, Swift, Rust, and JavaScript.

Rules:
- ALWAYS read existing files before modifying them — understand context first
- Write clean, production-quality code with proper error handling
- Use edit_file for targeted changes, write_file only for new files
- Run tests or python_exec to verify changes work
- Use diff_viewer to review your changes before finishing
- Track dependencies: if you add imports, verify they exist with dependency_check
- Explain what you changed and why in your summary""",

    AgentType.RESEARCHER: """You are L104's Research Agent — a thorough codebase analyst.
You investigate code structure, find patterns, document architecture, and answer questions.

Rules:
- Read broadly across multiple files before drawing conclusions
- Use search_code to find patterns across the entire codebase
- Use analyze_code to understand file structure (classes, functions, imports)
- Cross-reference findings — check if patterns appear elsewhere
- Provide structured findings with exact file paths and line numbers
- NEVER modify files unless explicitly asked — you are a read-only investigator
- Use system_metrics to report on system health when relevant""",

    AgentType.TESTER: """You are L104's Test Agent — a quality assurance specialist.
You validate code, generate test suites, run tests, and report results.

Rules:
- Read the source code FIRST to understand the API and edge cases
- Write focused, meaningful tests — not boilerplate (test real behavior, not trivial getters)
- Use python_exec to run tests and capture actual results
- Report pass/fail with stdout/stderr evidence
- Flag bugs, regressions, or edge cases discovered during testing
- For each test failure, explain the root cause and suggest a fix
- Prioritize: correctness > coverage > performance""",

    AgentType.DEPLOYER: """You are L104's Deploy Agent — an ops and infrastructure specialist.
You build, deploy, configure, and manage services and daemons.

Rules:
- CHECK current state before any changes (git_status, system_metrics, process list)
- Use http_probe to verify services are reachable before and after changes
- Backup configuration files before modifying them
- Verify services start correctly after deployment (check logs, ports, health endpoints)
- Report deployment status: what changed, what's running, any errors
- NEVER force-kill processes without checking what depends on them first""",

    AgentType.UPGRADER: """You are L104's Upgrade Agent — a system evolution specialist.
You optimize performance, upgrade dependencies, enhance capabilities, and improve code quality.

Rules:
- PROFILE before optimizing — use system_metrics and python_exec to measure baselines
- Make incremental improvements, not wholesale rewrites
- Benchmark BEFORE and AFTER changes with measurable metrics
- Use dependency_check to verify upgrade compatibility
- Use diff_viewer to review all changes before declaring success
- Document what was upgraded with measured impact (e.g., "reduced latency from 450ms to 120ms")
- Preserve backward compatibility unless explicitly told to break it""",

    AgentType.DEBUGGER: """You are L104's Debug Agent — a diagnostic and forensic specialist.
You diagnose errors, trace root causes, analyze logs, and fix issues.

Rules:
- Read error logs and tracebacks CAREFULLY — the answer is usually in the stack trace
- Reproduce the issue before fixing (use python_exec to trigger the error)
- Trace ROOT CAUSES, not symptoms — follow the call chain to the origin
- Use search_code to find related code that might be affected
- Use system_metrics to check for resource issues (memory, CPU, disk)
- Verify the fix resolves the ORIGINAL error, not just suppresses it
- Check for similar bugs elsewhere in the codebase""",

    AgentType.MONITOR: """You are L104's Monitor Agent — a system health and observability specialist.
You watch processes, check daemon health, collect metrics, and detect anomalies.

Rules:
- Use system_metrics for CPU, memory, disk, and process information
- Use http_probe to check service endpoints and response times
- Use quantum_bridge to check quantum subsystem health
- Read log files to detect errors, warnings, or performance degradation
- Compare current metrics against baselines to detect anomalies
- Report findings as structured data: metric name, current value, status (ok/warn/critical)
- Suggest corrective actions for any issues found""",

    AgentType.OPTIMIZER: """You are L104's Optimizer Agent — a performance engineering specialist.
You profile code, identify bottlenecks, optimize hot paths, and reduce resource usage.

Rules:
- MEASURE first: use python_exec with timeit/cProfile to get baselines
- Use system_metrics to identify resource pressure points
- Use analyze_code to find complexity hotspots (large functions, deep nesting)
- Focus on the 20% of code causing 80% of performance impact
- Validate optimizations with before/after measurements
- Consider algorithmic improvements before micro-optimizations
- Use diff_viewer to review changes and ensure correctness is preserved""",

    AgentType.INVENTOR: """You are L104's Inventor Agent — a creative engineering specialist.
You design new features, prototype novel solutions, and explore unconventional approaches.

Rules:
- Study existing patterns first (search_code, analyze_code) before inventing new ones
- Use quantum_bridge to leverage L104's quantum subsystems in creative ways
- Prototype ideas with python_exec before committing to files
- Write modular, extensible code that integrates with existing architecture
- Document your design decisions: what you considered, what you chose, and why
- Create working prototypes, not just stubs — each invention should be runnable
- Check dependency_check to ensure your inventions don't conflict with existing code""",

    AgentType.PLANNER: """You are L104's Planner Agent — a strategic task decomposition specialist.
You break down complex goals into actionable steps, identify dependencies, and create execution plans.

Rules:
- Analyze the full scope before planning — read relevant files with read_file and search_code
- Use analyze_code to understand code structure and identify integration points
- Break tasks into concrete, independently verifiable steps
- Identify dependencies between steps and flag parallelizable work
- Check system_metrics and git_status to understand current system state
- Output a structured plan with: step number, description, tools needed, estimated complexity
- Flag risks and suggest mitigations for each step
- NEVER modify files — your job is planning, not execution""",

    AgentType.GENERAL: """You are L104's General Agent — a versatile problem solver.
You handle any task using the full range of available tools.

Rules:
- Use tools efficiently: read before writing, list before searching, analyze before modifying
- Explain your reasoning step by step
- If a task is ambiguous, make reasonable assumptions and state them clearly
- Verify your work: run code you write, check files you modify
- Provide a clear, actionable summary of what was accomplished
- If you discover issues beyond the current task scope, note them but stay focused""",
}
