"""Central agent orchestrator v3.0.0 — manages lifecycle, queuing, chaining, and concurrent execution."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from .agent_types import (
    AgentResult, AgentStatus, AgentTask, AgentType,
    AgentPriority, AGENT_DEFAULT_TOOLS,
)
from .deepseek_executor import DeepSeekExecutor

log = logging.getLogger("l104.agent_orchestrator")

WORKSPACE = Path(os.environ.get(
    "L104_WORKSPACE",
    "/Users/carolalvarez/Applications/Allentown-L104-Node",
))
MAILBOX_DIR = WORKSPACE / ".l104_mailbox"
STATE_FILE = WORKSPACE / ".l104_agent_state.json"
MAX_CONCURRENT = 4
MAX_HISTORY = 100
# Global rate limit: max API calls per minute across all agents
RATE_LIMIT_PER_MINUTE = 30


class AgentOrchestrator:
    """Manages agent lifecycle: create, queue, execute, monitor, chain, cancel."""

    def __init__(self, max_concurrent: int = MAX_CONCURRENT):
        self._lock = threading.Lock()
        self._agents: dict[str, AgentTask] = {}
        self._history: list[dict[str, Any]] = []
        self._executor = DeepSeekExecutor()
        self._pool = ThreadPoolExecutor(max_workers=max_concurrent, thread_name_prefix="agent")
        self._max_concurrent = max_concurrent
        self._running_count = 0
        self._total_completed = 0
        self._total_failed = 0
        self._total_tokens = 0
        self._total_cost = 0.0
        self._boot_time = time.time()
        # Rate limiting
        self._api_call_times: list[float] = []
        self._rate_limit = RATE_LIMIT_PER_MINUTE
        # Pending chains: task_id -> task to launch when dependency completes
        self._pending_chains: dict[str, AgentTask] = {}
        self._load_state()

    def _load_state(self):
        """Load persisted agent history."""
        try:
            if STATE_FILE.exists():
                data = json.loads(STATE_FILE.read_text())
                self._history = data.get("history", [])[-MAX_HISTORY:]
                self._total_completed = data.get("total_completed", 0)
                self._total_failed = data.get("total_failed", 0)
                self._total_tokens = data.get("total_tokens", 0)
                self._total_cost = data.get("total_cost", 0.0)
        except Exception as e:
            log.warning("Failed to load agent state: %s", e)

    def _save_state(self):
        """Persist agent history."""
        try:
            STATE_FILE.write_text(json.dumps({
                "history": self._history[-MAX_HISTORY:],
                "total_completed": self._total_completed,
                "total_failed": self._total_failed,
                "total_tokens": self._total_tokens,
                "total_cost": round(self._total_cost, 6),
                "saved_at": time.time(),
                "version": "3.0.0",
            }, indent=2))
        except Exception as e:
            log.warning("Failed to save agent state: %s", e)

    def _write_mailbox_response(self, task: AgentTask):
        """Write completion to mailbox for Swift app consumption."""
        try:
            resp_dir = MAILBOX_DIR / "responses"
            resp_dir.mkdir(parents=True, exist_ok=True)
            resp_file = resp_dir / f"{task.task_id}.json"
            resp_file.write_text(json.dumps({
                "task_id": task.task_id,
                "task": task.prompt,
                "agent_type": task.agent_type.value,
                "status": task.status.value,
                "priority": task.priority.value,
                "output": task.result.output[:4000] if task.result else (task.error or ""),
                "summary": task.result.summary if task.result else "",
                "files_modified": task.files_modified,
                "tool_calls_count": len(task.tool_calls_made),
                "tokens_used": task.result.tokens_used if task.result else 0,
                "cost_estimate": task.result.cost_estimate if task.result else 0.0,
                "duration": task.elapsed(),
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "errors": task.result.errors_encountered if task.result else [],
                "warnings": task.result.warnings if task.result else [],
                "chain_next": task.chain_next,
                "tags": task.tags,
                "source": task.source,
            }, indent=2))
        except Exception as e:
            log.warning("Failed to write mailbox response: %s", e)

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits. Returns True if OK to proceed."""
        now = time.time()
        with self._lock:
            # Prune old entries
            self._api_call_times = [t for t in self._api_call_times if now - t < 60]
            if len(self._api_call_times) >= self._rate_limit:
                return False
            self._api_call_times.append(now)
            return True

    def _on_progress(self, task: AgentTask):
        """Called by executor on each round."""
        with self._lock:
            self._agents[task.task_id] = task

    def _handle_chain(self, completed_task: AgentTask):
        """If completed task has chain_next, launch the chained agent."""
        if not completed_task.chain_next:
            return

        with self._lock:
            chain_id = completed_task.chain_next
            # Check if there's a pre-registered chain task
            chained = self._pending_chains.pop(chain_id, None)

        if chained:
            # Inject parent result as context
            chained.chain_context = {
                "parent_task_id": completed_task.task_id,
                "parent_summary": completed_task.result.summary if completed_task.result else "",
                "parent_files_modified": completed_task.files_modified,
                "parent_success": completed_task.result.success if completed_task.result else False,
            }
            # Prepend parent context to prompt
            ctx_str = json.dumps(chained.chain_context, indent=2)
            chained.prompt = (
                f"[Context from parent agent {completed_task.task_id}]:\n{ctx_str}\n\n"
                f"[Your task]:\n{chained.prompt}"
            )
            chained.source = "chain"
            self._submit_internal(chained)
            log.info("Chained agent %s launched from %s", chained.task_id, completed_task.task_id)

    def _run_agent(self, task: AgentTask):
        """Run agent in thread pool."""
        # Wait for dependency if needed
        if task.depends_on:
            task.status = AgentStatus.WAITING
            wait_start = time.time()
            while True:
                with self._lock:
                    dep = self._agents.get(task.depends_on)
                if dep and dep.status in (AgentStatus.COMPLETED, AgentStatus.FAILED,
                                           AgentStatus.CANCELLED, AgentStatus.TIMEOUT):
                    # Dependency finished — inject its result
                    if dep.result:
                        task.chain_context = {
                            "dependency_task_id": dep.task_id,
                            "dependency_summary": dep.result.summary,
                            "dependency_success": dep.result.success,
                            "dependency_files": dep.files_modified,
                        }
                    break
                if time.time() - wait_start > task.timeout_seconds:
                    task.status = AgentStatus.TIMEOUT
                    task.completed_at = time.time()
                    task.error = f"Timed out waiting for dependency {task.depends_on}"
                    log.warning("Agent %s timed out waiting for dep %s", task.task_id, task.depends_on)
                    self._write_mailbox_response(task)
                    return
                time.sleep(2)

        # Rate limit check
        if not self._check_rate_limit():
            log.info("Agent %s waiting for rate limit cooldown", task.task_id)
            time.sleep(5)

        with self._lock:
            self._running_count += 1

        try:
            result = self._executor.execute(task, on_progress=self._on_progress)

            with self._lock:
                self._running_count -= 1
                if result.success:
                    self._total_completed += 1
                else:
                    self._total_failed += 1
                self._total_tokens += result.tokens_used
                self._total_cost += result.cost_estimate

                # Add to history
                self._history.append(task.to_dict())
                if len(self._history) > MAX_HISTORY:
                    self._history = self._history[-MAX_HISTORY:]

            # Write mailbox response
            self._write_mailbox_response(task)
            self._save_state()

            # Handle chaining
            self._handle_chain(task)

            log.info("Agent %s completed: %s (%.1fs, %d tools, %d tokens, $%.4f%s)",
                     task.task_id,
                     "OK" if result.success else "FAILED",
                     task.elapsed(),
                     len(task.tool_calls_made),
                     result.tokens_used,
                     result.cost_estimate,
                     f", {task.retry_count} retries" if task.retry_count > 0 else "")

        except Exception as e:
            log.error("Agent %s crashed: %s", task.task_id, e)
            with self._lock:
                self._running_count -= 1
                self._total_failed += 1
                task.status = AgentStatus.FAILED
                task.error = str(e)
                task.completed_at = time.time()
                self._history.append(task.to_dict())

            self._write_mailbox_response(task)
            self._save_state()

    def _submit_internal(self, task: AgentTask) -> AgentTask:
        """Internal submit that doesn't create a new task."""
        with self._lock:
            self._agents[task.task_id] = task
        self._pool.submit(self._run_agent, task)
        return task

    def submit(self, prompt: str, agent_type: str = "general",
               tools: list[str] | None = None, model: str = "deepseek-chat",
               max_rounds: int = 25, priority: str = "normal",
               timeout: float = 900.0, cost_budget: float = 0.25,
               depends_on: str = "", chain_next: str = "",
               tags: list[str] | None = None, source: str = "api") -> AgentTask:
        """Submit a new agent task. Returns immediately."""
        try:
            atype = AgentType(agent_type)
        except ValueError:
            atype = AgentType.GENERAL

        try:
            apriority = AgentPriority[priority.upper()]
        except (KeyError, AttributeError):
            apriority = AgentPriority.NORMAL

        task = AgentTask(
            prompt=prompt,
            agent_type=atype,
            model=model,
            max_rounds=max_rounds,
            priority=apriority,
            timeout_seconds=timeout,
            cost_budget=cost_budget,
            depends_on=depends_on or None,
            chain_next=chain_next or None,
            tags=tags or [],
            source=source,
        )
        if tools:
            task.tools_enabled = tools
        else:
            # Use agent-type defaults
            task.tools_enabled = AGENT_DEFAULT_TOOLS.get(atype, AGENT_DEFAULT_TOOLS[AgentType.GENERAL])

        with self._lock:
            self._agents[task.task_id] = task

        # Submit to thread pool
        self._pool.submit(self._run_agent, task)
        log.info("Agent %s submitted: type=%s, priority=%s, prompt=%.60s...",
                 task.task_id, atype.value, apriority.name, prompt)

        return task

    def submit_chain(self, tasks: list[dict[str, Any]]) -> list[AgentTask]:
        """Submit a chain of agents that execute sequentially.

        Each task dict has same params as submit(). Tasks run in order,
        each receiving the previous task's results as context.

        Returns list of AgentTask objects.
        """
        if not tasks:
            return []

        created: list[AgentTask] = []
        prev_id: Optional[str] = None

        for i, t in enumerate(tasks):
            is_last = (i == len(tasks) - 1)
            task = self.submit(
                prompt=t.get("prompt", ""),
                agent_type=t.get("agent_type", "general"),
                tools=t.get("tools"),
                model=t.get("model", "deepseek-chat"),
                max_rounds=t.get("max_rounds", 25),
                priority=t.get("priority", "normal"),
                timeout=t.get("timeout", 900.0),
                cost_budget=t.get("cost_budget", 0.05),
                depends_on=prev_id or "",
                tags=t.get("tags", []) + ["chain"],
                source="chain",
            )
            # Link previous task to this one
            if prev_id:
                with self._lock:
                    prev_task = self._agents.get(prev_id)
                    if prev_task:
                        prev_task.chain_next = task.task_id

            prev_id = task.task_id
            created.append(task)

        log.info("Agent chain submitted: %d tasks, IDs: %s",
                 len(created), [t.task_id for t in created])
        return created

    def get_task(self, task_id: str) -> Optional[AgentTask]:
        """Get a task by ID."""
        with self._lock:
            return self._agents.get(task_id)

    def cancel(self, task_id: str) -> bool:
        """Cancel a running task (best-effort)."""
        with self._lock:
            task = self._agents.get(task_id)
            if task and task.status in (AgentStatus.QUEUED, AgentStatus.RUNNING,
                                         AgentStatus.TOOL_CALL, AgentStatus.THINKING,
                                         AgentStatus.WAITING):
                task.status = AgentStatus.CANCELLED
                task.completed_at = time.time()
                return True
        return False

    def list_active(self) -> list[dict[str, Any]]:
        """List all active (non-completed) agents."""
        active_statuses = (AgentStatus.QUEUED, AgentStatus.RUNNING,
                          AgentStatus.TOOL_CALL, AgentStatus.THINKING, AgentStatus.WAITING)
        with self._lock:
            return [
                t.to_dict() for t in self._agents.values()
                if t.status in active_statuses
            ]

    def list_completed(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recently completed agents."""
        done_statuses = (AgentStatus.COMPLETED, AgentStatus.FAILED,
                        AgentStatus.CANCELLED, AgentStatus.TIMEOUT)
        with self._lock:
            completed = [
                t.to_dict() for t in self._agents.values()
                if t.status in done_statuses
            ]
            seen = {c["task_id"] for c in completed}
            for h in reversed(self._history):
                if h["task_id"] not in seen:
                    completed.append(h)
                    seen.add(h["task_id"])
            return sorted(completed, key=lambda x: x.get("completed_at") or 0, reverse=True)[:limit]

    def list_by_tag(self, tag: str) -> list[dict[str, Any]]:
        """List agents matching a tag."""
        with self._lock:
            return [
                t.to_dict() for t in self._agents.values()
                if tag in t.tags
            ]

    def get_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return completed agent history with full results."""
        done_statuses = (AgentStatus.COMPLETED, AgentStatus.FAILED,
                        AgentStatus.CANCELLED, AgentStatus.TIMEOUT)
        with self._lock:
            entries: list[dict[str, Any]] = []
            seen: set[str] = set()
            for t in self._agents.values():
                if t.status in done_statuses:
                    entries.append(t.to_dict())
                    seen.add(t.task_id)
            for h in reversed(self._history):
                if h.get("task_id") not in seen:
                    entries.append(h)
                    seen.add(h["task_id"])
            return sorted(entries, key=lambda x: x.get("completed_at") or 0, reverse=True)[:limit]

    def stats(self) -> dict[str, Any]:
        """Return aggregate statistics."""
        with self._lock:
            all_entries = list(self._history)
            done_statuses = (AgentStatus.COMPLETED, AgentStatus.FAILED,
                            AgentStatus.CANCELLED, AgentStatus.TIMEOUT)
            seen = {h.get("task_id") for h in all_entries}
            for t in self._agents.values():
                if t.status in done_statuses:
                    if t.task_id not in seen:
                        all_entries.append(t.to_dict())

        if not all_entries:
            return {"total": 0, "avg_duration": 0, "avg_tokens": 0, "success_rate": 0,
                    "most_used_tools": [], "most_used_agent_types": [],
                    "total_tokens": 0, "total_cost": 0}

        durations = [e.get("duration", 0) for e in all_entries]
        tokens = [e.get("result", {}).get("tokens_used", 0) if isinstance(e.get("result"), dict) else 0
                  for e in all_entries]
        successes = sum(1 for e in all_entries if e.get("status") == "completed")

        from collections import Counter
        tool_counter: Counter = Counter()
        type_counter: Counter = Counter()
        for e in all_entries:
            type_counter[e.get("agent_type", "general")] += 1
            result = e.get("result")
            if isinstance(result, dict):
                for tr in result.get("tool_results", []):
                    if isinstance(tr, dict):
                        tool_counter[tr.get("name", "unknown")] += 1

        return {
            "total": len(all_entries),
            "avg_duration": round(sum(durations) / len(durations), 2) if durations else 0,
            "avg_tokens": round(sum(tokens) / len(tokens), 1) if tokens else 0,
            "success_rate": round(successes / len(all_entries), 3) if all_entries else 0,
            "most_used_tools": tool_counter.most_common(10),
            "most_used_agent_types": type_counter.most_common(10),
            "total_completed": self._total_completed,
            "total_failed": self._total_failed,
            "total_tokens": self._total_tokens,
            "total_cost": round(self._total_cost, 6),
        }

    def cleanup(self, max_age_hours: int = 24) -> int:
        """Purge old agent state from memory. Returns number of entries removed."""
        done_statuses = (AgentStatus.COMPLETED, AgentStatus.FAILED,
                        AgentStatus.CANCELLED, AgentStatus.TIMEOUT)
        cutoff = time.time() - (max_age_hours * 3600)
        removed = 0
        with self._lock:
            old_ids = [
                tid for tid, t in self._agents.items()
                if t.status in done_statuses
                and (t.completed_at or t.created_at) < cutoff
            ]
            for tid in old_ids:
                del self._agents[tid]
                removed += 1

            before = len(self._history)
            self._history = [
                h for h in self._history
                if (h.get("completed_at") or h.get("created_at", 0)) >= cutoff
            ]
            removed += before - len(self._history)

        if removed > 0:
            self._save_state()
            log.info("Cleanup: removed %d entries older than %dh", removed, max_age_hours)

        # Also clean old mailbox responses
        try:
            resp_dir = MAILBOX_DIR / "responses"
            if resp_dir.exists():
                for f in resp_dir.glob("*.json"):
                    if f.stat().st_mtime < cutoff:
                        f.unlink(missing_ok=True)
                        removed += 1
        except Exception:
            pass

        return removed

    def status(self) -> dict[str, Any]:
        """Full orchestrator status."""
        active_statuses = (AgentStatus.QUEUED, AgentStatus.RUNNING,
                          AgentStatus.TOOL_CALL, AgentStatus.THINKING, AgentStatus.WAITING)
        with self._lock:
            active = [t for t in self._agents.values() if t.status in active_statuses]
            return {
                "status": "ONLINE",
                "version": "3.0.0",
                "max_concurrent": self._max_concurrent,
                "running": self._running_count,
                "queued": sum(1 for t in active if t.status == AgentStatus.QUEUED),
                "waiting": sum(1 for t in active if t.status == AgentStatus.WAITING),
                "active_agents": [t.to_dict() for t in active],
                "total_completed": self._total_completed,
                "total_failed": self._total_failed,
                "total_tokens": self._total_tokens,
                "total_cost": round(self._total_cost, 6),
                "history_size": len(self._history),
                "uptime": time.time() - self._boot_time,
                "has_api_key": bool(self._executor.api_key),
                "executor_stats": self._executor.get_stats(),
                "rate_limit": self._rate_limit,
                "pending_chains": len(self._pending_chains),
                "tools_available": [t["name"] for t in ToolRegistry.list_tools()],
                "agent_types": [t.value for t in AgentType],
            }

    def process_mailbox(self):
        """Check mailbox for new requests and submit them."""
        req_dir = MAILBOX_DIR / "requests"
        if not req_dir.exists():
            return []

        submitted = []
        for f in sorted(req_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text())
                task_id = data.get("agent_id", f.stem)

                # Skip if already known
                with self._lock:
                    if task_id in self._agents:
                        continue

                task = self.submit(
                    prompt=data.get("task", ""),
                    agent_type=data.get("agent_type", "general"),
                    tools=data.get("tools"),
                    model=data.get("model", "deepseek-chat"),
                    max_rounds=data.get("max_rounds", 10),
                    priority=data.get("priority", "normal"),
                    timeout=data.get("timeout", 600.0),
                    cost_budget=data.get("cost_budget", 0.05),
                    tags=data.get("tags", []),
                    source="mailbox",
                )
                # Override task_id to match request
                with self._lock:
                    old_id = task.task_id
                    task.task_id = task_id
                    self._agents[task_id] = task
                    if old_id in self._agents:
                        del self._agents[old_id]

                submitted.append(task_id)
                # Remove request file
                f.unlink(missing_ok=True)
            except Exception as e:
                log.warning("Failed to process mailbox request %s: %s", f.name, e)

        return submitted


# Import here to avoid circular import at module level
from .tools import ToolRegistry

# Singleton
_orchestrator: Optional[AgentOrchestrator] = None
_lock = threading.Lock()


def get_orchestrator() -> AgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        with _lock:
            if _orchestrator is None:
                _orchestrator = AgentOrchestrator()
    return _orchestrator
