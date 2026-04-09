"""DeepSeek-powered agent executor with multi-round tool calling — v3.0.0.

Features:
- Context window management (auto-summarize when approaching limit)
- Per-task cost budgets and timeout enforcement
- Enhanced error recovery with structured error messages
- Token tracking and rate awareness
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

from .agent_types import (
    AgentResult, AgentStatus, AgentTask, AgentType,
    AGENT_SYSTEM_PROMPTS, AGENT_DEFAULT_TOOLS,
)
from .tools import ToolRegistry, ToolResult

log = logging.getLogger("l104.agent_executor")

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# DeepSeek context window limit (tokens)
MAX_CONTEXT_TOKENS = 64_000
# When to trigger context compression
CONTEXT_COMPRESS_THRESHOLD = 48_000
# Max output per tool result injected into context
MAX_TOOL_OUTPUT_TOKENS = 4000


def _get_api_key() -> str:
    """Get DeepSeek API key from env, .env file, or config."""
    key = DEEPSEEK_API_KEY
    if not key:
        try:
            workspace = os.environ.get("L104_WORKSPACE", "/Users/carolalvarez/Applications/Allentown-L104-Node")
            env_path = os.path.join(workspace, ".env")
            if os.path.exists(env_path):
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("#") or "=" not in line:
                            continue
                        k, _, v = line.partition("=")
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k.upper() in ("DEEPSEEK_API_KEY", "DEEPSEEK_KEY") and v.startswith("sk-"):
                            key = v
                            break
        except Exception:
            pass
    if not key:
        try:
            import yaml
            cfg_path = os.path.join(
                os.environ.get("L104_WORKSPACE", "/Users/carolalvarez/Applications/Allentown-L104-Node"),
                "config", "system.yaml",
            )
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            key = cfg.get("deepseek", {}).get("api_key", "")
        except Exception:
            pass
    if not key:
        try:
            from l104_config.l104_config import L104Config
            key = L104Config.get("DEEPSEEK_API_KEY", "")
        except Exception:
            pass
    return key


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English/code."""
    return max(1, len(text) // 4)


def _estimate_messages_tokens(messages: list[dict]) -> int:
    """Estimate total tokens in a message list."""
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            total += _estimate_tokens(content)
        # Tool calls add ~50 tokens overhead each
        for tc in m.get("tool_calls", []):
            total += 50 + _estimate_tokens(tc.get("function", {}).get("arguments", ""))
    return total


class DeepSeekExecutor:
    """Execute agent tasks using DeepSeek API with tool calling."""

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or _get_api_key()
        self._total_tokens_used = 0
        self._total_cost = 0.0
        self._requests_made = 0

    # HTTP status codes that are safe to retry
    _RETRYABLE_STATUS_CODES = {429, 500, 502, 503}
    # Approx cost per token (DeepSeek chat) for cost estimation
    _COST_PER_TOKEN = 0.00000014  # ~$0.14 per 1M tokens

    def _call_api(self, messages: list[dict], tools: list[dict] | None = None,
                  model: str = "deepseek-chat", max_retries: int = 3) -> dict[str, Any]:
        """Call DeepSeek API with retry logic and exponential backoff."""
        import urllib.request
        import urllib.error

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 0.3,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        data = json.dumps(payload).encode()
        last_error: str = ""
        retries_used = 0

        for attempt in range(max_retries + 1):
            req = urllib.request.Request(
                f"{DEEPSEEK_BASE_URL}/chat/completions",
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )

            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    result = json.loads(resp.read())
                elapsed = time.time() - t0
                if elapsed > 30:
                    log.warning("Slow DeepSeek API request: %.1fs", elapsed)
                result["_retries_used"] = retries_used
                self._requests_made += 1
                return result
            except urllib.error.HTTPError as e:
                elapsed = time.time() - t0
                last_error = f"HTTP {e.code}: {e.reason}"
                log.warning("DeepSeek API HTTP error (attempt %d/%d, %.1fs): %s",
                            attempt + 1, max_retries + 1, elapsed, last_error)
                if e.code not in self._RETRYABLE_STATUS_CODES or attempt >= max_retries:
                    return {"error": last_error, "_retries_used": retries_used}
                retries_used += 1
                backoff = 2 ** (attempt + 1)
                log.info("Retrying in %ds...", backoff)
                time.sleep(backoff)
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                elapsed = time.time() - t0
                last_error = str(e)
                log.warning("DeepSeek API network error (attempt %d/%d, %.1fs): %s",
                            attempt + 1, max_retries + 1, elapsed, last_error)
                if attempt >= max_retries:
                    return {"error": last_error, "_retries_used": retries_used}
                retries_used += 1
                backoff = 2 ** (attempt + 1)
                log.info("Retrying in %ds...", backoff)
                time.sleep(backoff)
            except Exception as e:
                log.error("DeepSeek API unexpected error: %s", e)
                return {"error": str(e), "_retries_used": retries_used}

        return {"error": last_error, "_retries_used": retries_used}

    def _compress_context(self, messages: list[dict], task: AgentTask) -> list[dict]:
        """Compress message history when approaching context limit.

        Strategy: Keep system prompt + last user message + summarize middle.
        """
        if len(messages) <= 4:
            return messages

        estimated = _estimate_messages_tokens(messages)
        if estimated < CONTEXT_COMPRESS_THRESHOLD:
            return messages

        log.info("Agent %s: compressing context (%d est. tokens, %d messages)",
                 task.task_id, estimated, len(messages))

        # Keep: system prompt (0), first user message (1), last 4 messages
        system_msg = messages[0]
        first_user = messages[1]
        recent = messages[-4:]
        middle = messages[2:-4]

        # Summarize the middle section
        tool_calls_summary = []
        for m in middle:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    fn = tc.get("function", {})
                    tool_calls_summary.append(f"- {fn.get('name', '?')}({list(json.loads(fn.get('arguments', '{}')).keys())})")
            elif m.get("role") == "tool":
                content = m.get("content", "")
                # Just note success/failure
                status = "OK" if "Error:" not in content else "FAILED"
                tool_calls_summary.append(f"  -> {status}")

        summary_text = (
            f"[Context compressed: {len(middle)} messages summarized]\n"
            f"Tool calls so far:\n" + "\n".join(tool_calls_summary[-20:])
        )

        compressed = [
            system_msg,
            first_user,
            {"role": "assistant", "content": summary_text},
        ] + recent

        new_est = _estimate_messages_tokens(compressed)
        log.info("Agent %s: compressed %d -> %d messages (%d -> %d est. tokens)",
                 task.task_id, len(messages), len(compressed), estimated, new_est)

        return compressed

    def _select_optimal_tools_via_asi(self, prompt: str, available_tools: list[str], agent_type: AgentType) -> list[str]:
        """
        PHASE 7: ASI-Powered Tool Selection

        Uses the L104 ASI core to intelligently rank and select tools based on:
        1. Task requirements analysis
        2. Tool capability matching
        3. ASI alignment scoring (GOD_CODE harmonics)

        Returns optimized tool list ordered by suitability.
        """
        try:
            from l104_asi import asi_core

            # Build ASI query for tool selection
            tool_list_str = ", ".join(available_tools)
            asi_prompt = f"""
Agent type: {agent_type.name}
Task: {prompt[:500]}
Available tools: {tool_list_str}

Analyze this task and rank the available tools by relevance (highest first).
Return ONLY a comma-separated list of tool names, most relevant first.
Example format: tool_a, tool_b, tool_c
"""

            asi_result = asi_core.query(asi_prompt)

            if asi_result:
                # Parse ASI response (should be comma-separated tool names)
                ranked_tools = [t.strip() for t in asi_result.split(",")]
                # Filter to only valid tools
                valid_tools = [t for t in ranked_tools if t in available_tools]
                # Add any remaining tools not ranked by ASI
                for tool in available_tools:
                    if tool not in valid_tools:
                        valid_tools.append(tool)

                log.info("ASI tool ranking for agent: %s", " > ".join(valid_tools[:5]))
                return valid_tools

        except Exception as e:
            log.warning("ASI tool selection failed: %s, using default ranking", e)

        # Fallback: return tools in default order (prioritize code/analysis tools)
        priority_order = ["analyze_code", "read_file", "search_code", "python_exec", "write_file", "run_shell", "list_files"]
        prioritized = [t for t in priority_order if t in available_tools]
        remaining = [t for t in available_tools if t not in prioritized]
        return prioritized + remaining

    def execute(self, task: AgentTask, on_progress: Any = None) -> AgentResult:
        """Execute an agent task with multi-round tool calling.

        Features:
        - Context window management (auto-compress)
        - Timeout enforcement
        - Cost budget enforcement
        - Enhanced error reporting
        """
        if not self.api_key:
            return AgentResult(
                summary="No DeepSeek API key configured",
                output="Set DEEPSEEK_API_KEY env var or configure in config/system.yaml",
                success=False,
            )

        task.status = AgentStatus.RUNNING
        task.started_at = time.time()

        # PHASE 7: ASI-Powered Tool Selection — intelligently rank tools by task suitability
        if task.tools_enabled == ["read_file", "write_file", "list_files", "run_shell",
                                   "search_code", "analyze_code", "python_exec"]:
            type_tools = AGENT_DEFAULT_TOOLS.get(task.agent_type)
            if type_tools:
                task.tools_enabled = type_tools

        # Ask ASI to optimize tool selection for this specific task
        task.tools_enabled = self._select_optimal_tools_via_asi(task.prompt, task.tools_enabled, task.agent_type)
        log.info("Agent %s using tools (ASI-optimized): %s", task.task_id, task.tools_enabled)

        # Build system prompt
        sys_prompt = AGENT_SYSTEM_PROMPTS.get(task.agent_type, AGENT_SYSTEM_PROMPTS[AgentType.GENERAL])
        sys_prompt += f"""

Workspace: /Users/carolalvarez/Applications/Allentown-L104-Node
Available tools: {', '.join(task.tools_enabled)}
Max rounds: {task.max_rounds}
Your task: {task.prompt}

Work step by step. Use tools to read files, search code, and make changes.
After completing the task, provide a clear summary of what you accomplished with specific results."""

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": task.prompt},
        ]

        # Get tool schemas for enabled tools
        tool_schemas = ToolRegistry.get_schemas(task.tools_enabled)

        all_tool_results: list[dict[str, Any]] = []
        files_modified: list[str] = []
        files_created: list[str] = []
        errors_encountered: list[str] = []
        warnings: list[str] = []
        total_tokens = 0
        final_output = ""

        for round_num in range(1, task.max_rounds + 1):
            # ── Timeout check ──
            if task.timeout_seconds > 0 and task.is_timed_out():
                task.status = AgentStatus.TIMEOUT
                task.completed_at = time.time()
                task.error = f"Timeout after {task.timeout_seconds}s"
                log.warning("Agent %s timed out after %.1fs", task.task_id, task.elapsed())
                return AgentResult(
                    summary=f"Task timed out after {task.elapsed():.0f}s ({round_num - 1} rounds completed)",
                    output=final_output or "Task timed out before completion",
                    files_created=files_created,
                    files_modified=files_modified,
                    tool_results=all_tool_results,
                    tokens_used=total_tokens,
                    rounds_used=round_num - 1,
                    cost_estimate=total_tokens * self._COST_PER_TOKEN,
                    success=False,
                    errors_encountered=errors_encountered + ["Timeout"],
                )

            # ── Cost budget check ──
            current_cost = total_tokens * self._COST_PER_TOKEN
            if task.cost_budget > 0 and current_cost > task.cost_budget:
                task.status = AgentStatus.COMPLETED
                task.completed_at = time.time()
                warnings.append(f"Cost budget exceeded: ${current_cost:.4f} > ${task.cost_budget:.4f}")
                log.warning("Agent %s exceeded cost budget: $%.4f > $%.4f",
                            task.task_id, current_cost, task.cost_budget)
                # Do a final summary request without tools
                messages.append({"role": "user", "content": "Budget reached. Summarize what you accomplished so far."})
                response = self._call_api(messages, model=task.model)
                task.retry_count += response.pop("_retries_used", 0)
                if not response.get("error"):
                    choice = response.get("choices", [{}])[0]
                    final_output = choice.get("message", {}).get("content", "Budget reached")
                    total_tokens += response.get("usage", {}).get("total_tokens", 0)
                break

            task.current_action = f"Round {round_num}/{task.max_rounds}"
            task.progress = round_num / task.max_rounds

            # ── Context window management ──
            messages = self._compress_context(messages, task)

            # ── Call DeepSeek ──
            task.status = AgentStatus.THINKING
            if on_progress:
                on_progress(task)

            response = self._call_api(messages, tool_schemas, task.model)

            # Track retries from this API call
            task.retry_count += response.pop("_retries_used", 0)

            if "error" in response:
                error_msg = response["error"]
                errors_encountered.append(f"Round {round_num}: {error_msg}")
                task.error = error_msg

                # If it's a retryable error and we have rounds left, try to continue
                if "429" in error_msg or "502" in error_msg or "503" in error_msg:
                    if round_num < task.max_rounds:
                        warnings.append(f"API error in round {round_num}, retrying next round: {error_msg}")
                        time.sleep(5)  # Extra cooldown
                        continue

                # ASI FALLBACK: If DeepSeek fails permanently, escalate to ASI for intelligent response
                log.warning("Agent %s round %d failed on DeepSeek, attempting ASI fallback", task.task_id, round_num)
                try:
                    from l104_asi import asi_core
                    asi_result = asi_core.query(
                        f"Complete this task after DeepSeek failed: {task.prompt}. "
                        f"Context: {final_output}. "
                        f"Error: {error_msg}"
                    )
                    if asi_result:
                        final_output = asi_result
                        warnings.append(f"ASI fallback used after DeepSeek error in round {round_num}")
                        task.status = AgentStatus.COMPLETED
                        task.completed_at = time.time()
                        return AgentResult(
                            summary=f"Completed via ASI fallback after round {round_num}",
                            output=final_output,
                            files_created=files_created,
                            files_modified=files_modified,
                            tool_results=all_tool_results,
                            tokens_used=total_tokens,
                            rounds_used=round_num,
                            cost_estimate=total_tokens * self._COST_PER_TOKEN,
                            success=True,
                            errors_encountered=errors_encountered,
                            warnings=warnings,
                        )
                except Exception as asi_e:
                    log.error("ASI fallback also failed: %s", asi_e)
                    errors_encountered.append(f"ASI fallback failed: {str(asi_e)}")

                task.status = AgentStatus.FAILED
                task.completed_at = time.time()
                return AgentResult(
                    summary=f"API error in round {round_num}: {error_msg}",
                    output=final_output or error_msg,
                    files_created=files_created,
                    files_modified=files_modified,
                    tool_results=all_tool_results,
                    tokens_used=total_tokens,
                    rounds_used=round_num,
                    cost_estimate=total_tokens * self._COST_PER_TOKEN,
                    success=False,
                    errors_encountered=errors_encountered,
                )

            # Extract response
            choice = response.get("choices", [{}])[0]
            message = choice.get("message", {})
            total_tokens += response.get("usage", {}).get("total_tokens", 0)

            # Check for tool calls
            tool_calls = message.get("tool_calls", [])

            if not tool_calls:
                # No tool calls — this is the final response
                final_output = message.get("content", "")
                messages.append({"role": "assistant", "content": final_output})
                break

            # Process tool calls
            task.status = AgentStatus.TOOL_CALL
            task.current_action = f"Executing {len(tool_calls)} tool(s)"
            if on_progress:
                on_progress(task)

            # Add assistant message with tool_calls
            messages.append({
                "role": "assistant",
                "content": message.get("content", ""),
                "tool_calls": tool_calls,
            })

            for tc in tool_calls:
                fn_name = tc.get("function", {}).get("name", "")
                fn_args_str = tc.get("function", {}).get("arguments", "{}")
                tc_id = tc.get("id", f"call_{round_num}")

                try:
                    fn_args = json.loads(fn_args_str)
                except json.JSONDecodeError:
                    fn_args = {}
                    warnings.append(f"Round {round_num}: Invalid JSON args for {fn_name}")

                log.info("Agent %s tool call: %s(%s)", task.task_id, fn_name, list(fn_args.keys()))

                # Execute tool with per-tool timeout awareness
                tool_t0 = time.time()
                result = ToolRegistry.execute(fn_name, fn_args)
                tool_elapsed = time.time() - tool_t0

                if tool_elapsed > 10:
                    warnings.append(f"Slow tool: {fn_name} took {tool_elapsed:.1f}s")

                # Track errors from tools
                if not result.success:
                    errors_encountered.append(f"Tool {fn_name}: {result.error}")

                # Track file modifications
                if fn_name == "write_file" and result.success:
                    path = fn_args.get("path", "")
                    action = result.metadata.get("action", "modified")
                    if action == "created":
                        files_created.append(path)
                    else:
                        files_modified.append(path)
                elif fn_name == "edit_file" and result.success:
                    files_modified.append(fn_args.get("path", ""))

                tool_entry = {
                    "round": round_num,
                    "name": fn_name,
                    "args": fn_args,
                    "result": result.to_dict(),
                    "duration": tool_elapsed,
                }
                all_tool_results.append(tool_entry)
                task.tool_calls_made.append(tool_entry)

                # Add tool result to messages (truncated for context window)
                tool_output = result.output if result.success else f"Error: {result.error}"
                # Truncate large outputs to preserve context window
                max_output = MAX_TOOL_OUTPUT_TOKENS * 4  # Convert rough tokens to chars
                if len(tool_output) > max_output:
                    tool_output = tool_output[:max_output] + f"\n... (truncated, {len(tool_output)} total chars)"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": tool_output,
                })
        else:
            # Exhausted all rounds — ask for summary
            messages = self._compress_context(messages, task)
            messages.append({"role": "user", "content":
                "You've used all available rounds. Provide a concise summary of what you accomplished, "
                "what files were modified, and any remaining work needed."})
            response = self._call_api(messages, model=task.model)
            task.retry_count += response.pop("_retries_used", 0)
            if not response.get("error"):
                choice = response.get("choices", [{}])[0]
                final_output = choice.get("message", {}).get("content", "Max rounds reached")
                total_tokens += response.get("usage", {}).get("total_tokens", 0)

        # Build result
        cost = total_tokens * self._COST_PER_TOKEN
        self._total_tokens_used += total_tokens
        self._total_cost += cost

        task.status = AgentStatus.COMPLETED
        task.completed_at = time.time()
        task.files_modified = list(set(files_modified + files_created))
        task.progress = 1.0
        task.current_action = "Complete"

        result = AgentResult(
            summary=final_output[:500] if final_output else "Task completed",
            output=final_output,
            files_created=files_created,
            files_modified=list(set(files_modified)),
            tool_results=all_tool_results,
            tokens_used=total_tokens,
            rounds_used=min(len(all_tool_results) + 1, task.max_rounds),
            cost_estimate=cost,
            success=True,
            errors_encountered=errors_encountered,
            warnings=warnings,
            metrics={
                "total_tool_calls": len(all_tool_results),
                "api_retries": task.retry_count,
                "context_tokens_est": _estimate_messages_tokens(messages),
                "elapsed_seconds": task.elapsed(),
            },
        )
        task.result = result

        if on_progress:
            on_progress(task)

        return result

    def get_stats(self) -> dict[str, Any]:
        """Return executor-level statistics."""
        return {
            "has_api_key": bool(self.api_key),
            "total_tokens_used": self._total_tokens_used,
            "total_cost": round(self._total_cost, 6),
            "total_requests": self._requests_made,
        }
