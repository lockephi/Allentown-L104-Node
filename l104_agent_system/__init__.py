"""
l104_agent_system v3.0.0 — Unified Agent Orchestration System.

Centralized agent lifecycle management with DeepSeek-powered execution,
persistent state, tool sandboxing, priority queues, agent chaining,
cost budgets, timeout enforcement, and real-time monitoring.

Agent types: coder, researcher, tester, deployer, upgrader, debugger,
             monitor, optimizer, inventor, planner, general

Tools (14): read_file, write_file, edit_file, list_files, run_shell,
            search_code, analyze_code, python_exec, git_status,
            system_metrics, dependency_check, quantum_bridge, diff_viewer, http_probe
"""

from .orchestrator import AgentOrchestrator, get_orchestrator
from .agent_types import (
    AgentType, AgentStatus, AgentPriority,
    AgentTask, AgentResult,
    AGENT_SYSTEM_PROMPTS, AGENT_DEFAULT_TOOLS,
)
from .deepseek_executor import DeepSeekExecutor
from .tools import ToolRegistry, ToolResult

__all__ = [
    "AgentOrchestrator", "get_orchestrator",
    "AgentType", "AgentStatus", "AgentPriority",
    "AgentTask", "AgentResult",
    "AGENT_SYSTEM_PROMPTS", "AGENT_DEFAULT_TOOLS",
    "DeepSeekExecutor",
    "ToolRegistry", "ToolResult",
]
__version__ = "3.0.0"
