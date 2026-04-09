"""Extended tool registry with sandboxed execution for DeepSeek agents — v3.0.0.

14 tools: read_file, write_file, edit_file, list_files, run_shell, search_code,
analyze_code, python_exec, git_status, system_metrics, dependency_check,
quantum_bridge, diff_viewer, http_probe.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

WORKSPACE = Path(os.environ.get(
    "L104_WORKSPACE",
    "/Users/carolalvarez/Applications/Allentown-L104-Node",
))

# Paths that must never be read or written
BLOCKED_PATTERNS = [
    r"\.env$", r"\.env\.", r"wallet", r"credential", r"secret",
    r"api_key", r"private_key", r"\.ssh/", r"\.gnupg/",
    r"\.git/objects", r"\.git/refs",
]
BLOCKED_RE = re.compile("|".join(BLOCKED_PATTERNS), re.IGNORECASE)

# Commands that must never be run
BLOCKED_COMMANDS = [
    r"\brm\s+-rf\s+/", r"\bsudo\b", r"\bmkfs\b", r"\bdd\s+if=",
    r"\bkill\s+-9\s+1\b", r"\bshutdown\b", r"\breboot\b",
    r"\blaunchctl\s+remove\b", r"\bgit\s+push\s+--force\b",
]
BLOCKED_CMD_RE = re.compile("|".join(BLOCKED_COMMANDS), re.IGNORECASE)


@dataclass
class ToolResult:
    name: str
    success: bool
    output: str
    error: Optional[str] = None
    duration: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "success": self.success,
            "output": self.output[:4000],
            "error": self.error,
            "duration": round(self.duration, 3),
            "metadata": self.metadata,
        }


def _resolve_path(path: str) -> Path:
    """Resolve path relative to workspace, block escapes."""
    p = Path(path)
    if not p.is_absolute():
        p = WORKSPACE / p
    p = p.resolve()
    if not str(p).startswith(str(WORKSPACE)):
        raise PermissionError(f"Path escapes workspace: {path}")
    if BLOCKED_RE.search(str(p)):
        raise PermissionError(f"Blocked path pattern: {path}")
    return p


# ─── Core Tools ────────────────────────────────────────────────

def tool_read_file(path: str, offset: int = 0, limit: int = 500) -> ToolResult:
    """Read a file from the workspace."""
    t0 = time.time()
    try:
        p = _resolve_path(path)
        if not p.exists():
            return ToolResult("read_file", False, "", f"File not found: {path}", time.time() - t0)
        if p.stat().st_size > 5_000_000:
            return ToolResult("read_file", False, "", f"File too large: {p.stat().st_size} bytes", time.time() - t0)
        lines = p.read_text(errors="replace").splitlines()
        chunk = lines[offset:offset + limit]
        content = "\n".join(f"{i + offset + 1:>5} | {line}" for i, line in enumerate(chunk))
        return ToolResult("read_file", True, content, None, time.time() - t0,
                          {"total_lines": len(lines), "offset": offset, "limit": limit})
    except Exception as e:
        return ToolResult("read_file", False, "", str(e), time.time() - t0)


def tool_write_file(path: str, content: str) -> ToolResult:
    """Write content to a file in the workspace."""
    t0 = time.time()
    try:
        p = _resolve_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        existed = p.exists()
        old_size = p.stat().st_size if existed else 0
        p.write_text(content)
        action = "modified" if existed else "created"
        return ToolResult("write_file", True,
                          f"File {action}: {path} ({len(content)} bytes)",
                          None, time.time() - t0,
                          {"action": action, "old_size": old_size, "new_size": len(content), "path": str(p)})
    except Exception as e:
        return ToolResult("write_file", False, "", str(e), time.time() - t0)


def tool_edit_file(path: str, old_text: str, new_text: str) -> ToolResult:
    """Replace exact text in a file (like sed)."""
    t0 = time.time()
    try:
        p = _resolve_path(path)
        if not p.exists():
            return ToolResult("edit_file", False, "", f"File not found: {path}", time.time() - t0)
        content = p.read_text(errors="replace")
        count = content.count(old_text)
        if count == 0:
            return ToolResult("edit_file", False, "", "old_text not found in file", time.time() - t0)
        new_content = content.replace(old_text, new_text, 1)
        p.write_text(new_content)
        return ToolResult("edit_file", True, f"Replaced {count} occurrence(s) in {path}", None, time.time() - t0,
                          {"path": str(p), "replacements": 1})
    except Exception as e:
        return ToolResult("edit_file", False, "", str(e), time.time() - t0)


def tool_list_files(path: str = ".", pattern: str = "*", max_results: int = 200) -> ToolResult:
    """List files in a directory with optional glob pattern."""
    t0 = time.time()
    try:
        p = _resolve_path(path)
        if not p.is_dir():
            return ToolResult("list_files", False, "", f"Not a directory: {path}", time.time() - t0)
        files = sorted(p.glob(pattern))[:max_results]
        entries = []
        for f in files:
            rel = f.relative_to(WORKSPACE)
            kind = "dir" if f.is_dir() else "file"
            size = f.stat().st_size if f.is_file() else 0
            entries.append(f"{kind:4s} {size:>10,d}  {rel}")
        return ToolResult("list_files", True, "\n".join(entries), None, time.time() - t0,
                          {"count": len(entries), "pattern": pattern})
    except Exception as e:
        return ToolResult("list_files", False, "", str(e), time.time() - t0)


def tool_run_shell(command: str, timeout: int = 30) -> ToolResult:
    """Run a shell command with sandboxing."""
    t0 = time.time()
    try:
        if BLOCKED_CMD_RE.search(command):
            return ToolResult("run_shell", False, "", f"Blocked command: {command}", time.time() - t0)
        venv_bin = str(WORKSPACE / ".venv" / "bin")
        current_path = os.environ.get("PATH", "/usr/bin:/bin")
        enhanced_path = f"{venv_bin}:{current_path}" if os.path.isdir(venv_bin) else current_path
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=min(timeout, 60), cwd=str(WORKSPACE),
            env={**os.environ, "PATH": enhanced_path},
        )
        output = result.stdout[-4000:] if result.stdout else ""
        err = result.stderr[-2000:] if result.stderr else ""
        combined = output
        if err and result.returncode != 0:
            combined += f"\n--- stderr ---\n{err}"
        return ToolResult("run_shell", result.returncode == 0, combined,
                          err if result.returncode != 0 else None, time.time() - t0,
                          {"exit_code": result.returncode})
    except subprocess.TimeoutExpired:
        return ToolResult("run_shell", False, "", f"Timeout after {timeout}s", time.time() - t0)
    except Exception as e:
        return ToolResult("run_shell", False, "", str(e), time.time() - t0)


def tool_search_code(query: str, path: str = ".", file_type: str = "", max_results: int = 30) -> ToolResult:
    """Search code using grep (recursive)."""
    t0 = time.time()
    try:
        p = _resolve_path(path)
        cmd = ["grep", "-rn", "--include=*.py", "--include=*.swift", "--include=*.js",
               "--include=*.ts", "--include=*.yaml", "--include=*.json", "--include=*.md",
               "--include=*.rs", "--include=*.toml",
               "-m", str(max_results)]
        if file_type:
            cmd = ["grep", "-rn", f"--include=*.{file_type}", "-m", str(max_results)]
        cmd.extend(["--", query, str(p)])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=str(WORKSPACE))
        lines = result.stdout.strip().splitlines()[:max_results]
        output_lines = []
        for line in lines:
            line = line.replace(str(WORKSPACE) + "/", "")
            output_lines.append(line)
        return ToolResult("search_code", True, "\n".join(output_lines), None, time.time() - t0,
                          {"matches": len(output_lines), "query": query})
    except Exception as e:
        return ToolResult("search_code", False, "", str(e), time.time() - t0)


def tool_analyze_code(path: str) -> ToolResult:
    """Analyze a Python/Swift file for structure, classes, functions, complexity."""
    t0 = time.time()
    try:
        p = _resolve_path(path)
        if not p.exists():
            return ToolResult("analyze_code", False, "", f"File not found: {path}", time.time() - t0)
        content = p.read_text(errors="replace")
        lines = content.splitlines()
        ext = p.suffix.lower()

        classes = []
        functions = []
        imports = []
        todos = []
        max_indent = 0
        blank_lines = 0
        comment_lines = 0

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                blank_lines += 1
                continue
            indent = len(line) - len(line.lstrip())
            max_indent = max(max_indent, indent)

            if ext == ".py":
                if stripped.startswith("class "):
                    classes.append(f"  L{i+1}: {stripped}")
                elif stripped.startswith("def "):
                    functions.append(f"  L{i+1}: {stripped}")
                elif stripped.startswith(("import ", "from ")):
                    imports.append(stripped)
                elif stripped.startswith("#"):
                    comment_lines += 1
                if "TODO" in stripped or "FIXME" in stripped or "HACK" in stripped:
                    todos.append(f"  L{i+1}: {stripped[:100]}")
            elif ext == ".swift":
                if stripped.startswith("class ") or stripped.startswith("struct ") or stripped.startswith("enum "):
                    classes.append(f"  L{i+1}: {stripped}")
                elif stripped.startswith("func "):
                    functions.append(f"  L{i+1}: {stripped}")
                elif stripped.startswith("import "):
                    imports.append(stripped)
                elif stripped.startswith("//"):
                    comment_lines += 1
            else:
                if stripped.startswith("class "):
                    classes.append(f"  L{i+1}: {stripped}")
                elif "function " in stripped or stripped.startswith("def "):
                    functions.append(f"  L{i+1}: {stripped}")

        code_lines = len(lines) - blank_lines - comment_lines
        complexity = "low"
        if code_lines > 500 or len(functions) > 30:
            complexity = "high"
        elif code_lines > 200 or len(functions) > 15:
            complexity = "medium"

        analysis = f"File: {path} ({ext})\n"
        analysis += f"Lines: {len(lines)} (code: {code_lines}, blank: {blank_lines}, comment: {comment_lines})\n"
        analysis += f"Max indent depth: {max_indent // 4} levels\n"
        analysis += f"Complexity: {complexity}\n"
        analysis += f"\nClasses ({len(classes)}):\n" + "\n".join(classes[:20]) + "\n" if classes else ""
        analysis += f"\nFunctions ({len(functions)}):\n" + "\n".join(functions[:40]) + "\n" if functions else ""
        analysis += f"\nImports ({len(imports)}):\n" + "\n".join(f"  {i}" for i in imports[:25]) + "\n" if imports else ""
        if todos:
            analysis += f"\nTODOs/FIXMEs ({len(todos)}):\n" + "\n".join(todos[:10]) + "\n"

        return ToolResult("analyze_code", True, analysis, None, time.time() - t0,
                          {"lines": len(lines), "code_lines": code_lines, "classes": len(classes),
                           "functions": len(functions), "complexity": complexity, "ext": ext})
    except Exception as e:
        return ToolResult("analyze_code", False, "", str(e), time.time() - t0)


def tool_python_exec(code: str, timeout: int = 30) -> ToolResult:
    """Execute Python code in the project venv."""
    t0 = time.time()
    try:
        venv_python = str(WORKSPACE / ".venv" / "bin" / "python")
        if not Path(venv_python).exists():
            venv_python = "python3"
        result = subprocess.run(
            [venv_python, "-c", code],
            capture_output=True, text=True, timeout=min(timeout, 60),
            cwd=str(WORKSPACE),
        )
        output = result.stdout[-4000:] if result.stdout else ""
        err = result.stderr[-2000:] if result.stderr else ""
        combined = output
        if err and result.returncode != 0:
            combined += f"\n--- stderr ---\n{err}"
        return ToolResult("python_exec", result.returncode == 0, combined,
                          err if result.returncode != 0 else None, time.time() - t0,
                          {"exit_code": result.returncode})
    except subprocess.TimeoutExpired:
        return ToolResult("python_exec", False, "", f"Timeout after {timeout}s", time.time() - t0)
    except Exception as e:
        return ToolResult("python_exec", False, "", str(e), time.time() - t0)


def tool_git_status() -> ToolResult:
    """Get git status (porcelain) and recent commit log."""
    t0 = time.time()
    try:
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=15, cwd=str(WORKSPACE),
        )
        log_result = subprocess.run(
            ["git", "log", "--oneline", "-5"],
            capture_output=True, text=True, timeout=15, cwd=str(WORKSPACE),
        )
        branch_result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, timeout=5, cwd=str(WORKSPACE),
        )
        output = f"=== Branch: {branch_result.stdout.strip()} ===\n"
        output += "=== Git Status ===\n"
        status_lines = status_result.stdout.strip().splitlines() if status_result.stdout else []
        if len(status_lines) > 50:
            output += "\n".join(status_lines[:50])
            output += f"\n... and {len(status_lines) - 50} more files\n"
        else:
            output += status_result.stdout if status_result.stdout else "(clean)\n"
        output += "\n=== Recent Commits ===\n"
        output += log_result.stdout[-1000:] if log_result.stdout else "(no commits)\n"
        return ToolResult("git_status", True, output, None, time.time() - t0,
                          {"changed_files": len(status_lines),
                           "branch": branch_result.stdout.strip()})
    except Exception as e:
        return ToolResult("git_status", False, "", str(e), time.time() - t0)


# ─── New Tools v3.0.0 ──────────────────────────────────────────

def tool_system_metrics() -> ToolResult:
    """Collect system health metrics: CPU, memory, disk, processes, daemons."""
    t0 = time.time()
    try:
        metrics: dict[str, Any] = {}

        # CPU load
        try:
            load_result = subprocess.run(
                ["sysctl", "-n", "vm.loadavg"],
                capture_output=True, text=True, timeout=5,
            )
            metrics["load_avg"] = load_result.stdout.strip()
        except Exception:
            metrics["load_avg"] = "unavailable"

        # Memory (macOS)
        try:
            vm_result = subprocess.run(
                ["vm_stat"],
                capture_output=True, text=True, timeout=5,
            )
            lines = vm_result.stdout.splitlines()
            page_size = 16384  # Default macOS page size
            for line in lines:
                if "page size" in line.lower():
                    try:
                        page_size = int(re.search(r"\d+", line).group())
                    except Exception:
                        pass
            free_pages = 0
            active_pages = 0
            for line in lines:
                if "Pages free:" in line:
                    free_pages = int(re.search(r"\d+", line.split(":")[1]).group())
                elif "Pages active:" in line:
                    active_pages = int(re.search(r"\d+", line.split(":")[1]).group())
            metrics["memory_free_mb"] = round(free_pages * page_size / 1024 / 1024)
            metrics["memory_active_mb"] = round(active_pages * page_size / 1024 / 1024)
        except Exception:
            metrics["memory"] = "unavailable"

        # Disk
        try:
            df_result = subprocess.run(
                ["df", "-h", str(WORKSPACE)],
                capture_output=True, text=True, timeout=5,
            )
            df_lines = df_result.stdout.strip().splitlines()
            if len(df_lines) >= 2:
                metrics["disk"] = df_lines[1]
        except Exception:
            metrics["disk"] = "unavailable"

        # Python processes
        try:
            ps_result = subprocess.run(
                ["pgrep", "-lf", "python.*l104"],
                capture_output=True, text=True, timeout=5,
            )
            procs = [l.strip() for l in ps_result.stdout.strip().splitlines() if l.strip()]
            metrics["l104_python_processes"] = len(procs)
            metrics["processes"] = procs[:10]
        except Exception:
            metrics["l104_python_processes"] = 0

        # Server health
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2.0)
            s.connect(("127.0.0.1", 8081))
            s.sendall(b"GET /health HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n")
            resp = s.recv(512)
            s.close()
            metrics["server_8081"] = "UP" if b"200" in resp or b"HTTP" in resp else "DEGRADED"
        except Exception:
            metrics["server_8081"] = "DOWN"

        # LaunchAgents
        try:
            la_result = subprocess.run(
                ["launchctl", "list"],
                capture_output=True, text=True, timeout=5,
            )
            l104_agents = [l for l in la_result.stdout.splitlines() if "l104" in l.lower()]
            metrics["launchd_agents"] = l104_agents[:10]
        except Exception:
            metrics["launchd_agents"] = []

        # State files freshness
        state_files = list(WORKSPACE.glob(".l104_*.json"))
        recent = []
        stale = []
        now = time.time()
        for sf in sorted(state_files, key=lambda f: f.stat().st_mtime, reverse=True)[:20]:
            age_min = (now - sf.stat().st_mtime) / 60
            if age_min < 60:
                recent.append(f"{sf.name} ({age_min:.0f}m ago)")
            else:
                stale.append(f"{sf.name} ({age_min / 60:.1f}h ago)")
        metrics["recent_state_files"] = recent[:5]
        metrics["stale_state_files"] = stale[:5]

        output = json.dumps(metrics, indent=2, default=str)
        return ToolResult("system_metrics", True, output, None, time.time() - t0, metrics)
    except Exception as e:
        return ToolResult("system_metrics", False, "", str(e), time.time() - t0)


def tool_dependency_check(module: str = "", path: str = "") -> ToolResult:
    """Check Python import availability and dependency health.

    If `module` given: checks if that module can be imported.
    If `path` given: scans a file's imports and checks each one.
    If neither: reports all installed packages.
    """
    t0 = time.time()
    try:
        venv_python = str(WORKSPACE / ".venv" / "bin" / "python")
        if not Path(venv_python).exists():
            venv_python = "python3"

        if module:
            # Check single module
            result = subprocess.run(
                [venv_python, "-c", f"import {module}; print(getattr({module}, '__version__', 'installed'))"],
                capture_output=True, text=True, timeout=10, cwd=str(WORKSPACE),
            )
            if result.returncode == 0:
                return ToolResult("dependency_check", True,
                                  f"{module}: {result.stdout.strip()}", None, time.time() - t0,
                                  {"module": module, "available": True, "version": result.stdout.strip()})
            else:
                return ToolResult("dependency_check", True,
                                  f"{module}: NOT INSTALLED\n{result.stderr.strip()[:500]}", None, time.time() - t0,
                                  {"module": module, "available": False})

        if path:
            # Scan file imports
            p = _resolve_path(path)
            if not p.exists():
                return ToolResult("dependency_check", False, "", f"File not found: {path}", time.time() - t0)
            content = p.read_text(errors="replace")
            import_lines = [l.strip() for l in content.splitlines()
                            if l.strip().startswith(("import ", "from "))]
            # Extract module names
            modules = set()
            for line in import_lines:
                if line.startswith("from "):
                    mod = line.split()[1].split(".")[0]
                else:
                    mod = line.split()[1].split(".")[0].rstrip(",")
                if mod.startswith("_") or mod.startswith("l104_"):
                    continue  # Skip private/local
                modules.add(mod)

            results_list = []
            for mod in sorted(modules):
                r = subprocess.run(
                    [venv_python, "-c", f"import {mod}"],
                    capture_output=True, text=True, timeout=5, cwd=str(WORKSPACE),
                )
                status = "OK" if r.returncode == 0 else "MISSING"
                results_list.append(f"  {mod}: {status}")

            output = f"Dependency check for {path}:\n" + "\n".join(results_list)
            missing = [r for r in results_list if "MISSING" in r]
            return ToolResult("dependency_check", True, output, None, time.time() - t0,
                              {"file": path, "total": len(modules), "missing": len(missing)})

        # Default: list installed packages
        result = subprocess.run(
            [venv_python, "-m", "pip", "list", "--format=columns"],
            capture_output=True, text=True, timeout=15, cwd=str(WORKSPACE),
        )
        return ToolResult("dependency_check", True,
                          result.stdout[-4000:] if result.stdout else "No packages found",
                          None, time.time() - t0, {"mode": "list_all"})
    except Exception as e:
        return ToolResult("dependency_check", False, "", str(e), time.time() - t0)


def tool_quantum_bridge(operation: str = "status", params: str = "{}") -> ToolResult:
    """Bridge to L104 quantum subsystems: check status, run circuits, query state.

    Operations:
    - status: Overall quantum subsystem health
    - vqpu: VQPU daemon state
    - network: Quantum network status
    - daemon: Quantum AI daemon status
    - score: Run ASI/AGI scoring
    """
    t0 = time.time()
    try:
        parsed_params = json.loads(params) if params else {}
    except json.JSONDecodeError:
        parsed_params = {}

    try:
        if operation == "status":
            # Aggregate quantum health from state files
            health: dict[str, Any] = {"operation": "status"}
            state_files = {
                "vqpu_daemon": ".l104_vqpu_daemon_state.json",
                "quantum_ai_daemon": ".l104_quantum_ai_daemon.json",
                "quantum_mesh": ".l104_quantum_mesh_state.json",
                "consciousness": ".l104_consciousness_state.json",
            }
            for name, fname in state_files.items():
                fpath = WORKSPACE / fname
                if fpath.exists():
                    try:
                        data = json.loads(fpath.read_text())
                        age = time.time() - fpath.stat().st_mtime
                        health[name] = {
                            "exists": True,
                            "age_minutes": round(age / 60, 1),
                            "keys": list(data.keys())[:8],
                            "status": data.get("status", data.get("state", "unknown")),
                        }
                    except Exception:
                        health[name] = {"exists": True, "readable": False}
                else:
                    health[name] = {"exists": False}

            return ToolResult("quantum_bridge", True, json.dumps(health, indent=2),
                              None, time.time() - t0, health)

        elif operation == "vqpu":
            fpath = WORKSPACE / ".l104_vqpu_daemon_state.json"
            if not fpath.exists():
                return ToolResult("quantum_bridge", True, "VQPU daemon state not found",
                                  None, time.time() - t0, {"vqpu": "not_found"})
            data = json.loads(fpath.read_text())
            return ToolResult("quantum_bridge", True, json.dumps(data, indent=2, default=str)[:4000],
                              None, time.time() - t0, {"operation": "vqpu"})

        elif operation == "network":
            fpath = WORKSPACE / ".l104_quantum_mesh_state.json"
            if not fpath.exists():
                return ToolResult("quantum_bridge", True, "Quantum network state not found",
                                  None, time.time() - t0, {"network": "not_found"})
            data = json.loads(fpath.read_text())
            return ToolResult("quantum_bridge", True, json.dumps(data, indent=2, default=str)[:4000],
                              None, time.time() - t0, {"operation": "network"})

        elif operation == "daemon":
            fpath = WORKSPACE / ".l104_quantum_ai_daemon.json"
            if not fpath.exists():
                return ToolResult("quantum_bridge", True, "Quantum AI daemon state not found",
                                  None, time.time() - t0, {"daemon": "not_found"})
            data = json.loads(fpath.read_text())
            return ToolResult("quantum_bridge", True, json.dumps(data, indent=2, default=str)[:4000],
                              None, time.time() - t0, {"operation": "daemon"})

        elif operation == "score":
            # Run a quick ASI/AGI scoring probe
            venv_python = str(WORKSPACE / ".venv" / "bin" / "python")
            if not Path(venv_python).exists():
                venv_python = "python3"
            code = """
import sys, json
try:
    from l104_agi import agi_core
    score = agi_core.compute_10d_agi_score()
    print(json.dumps({"agi_score": score, "status": "ok"}))
except Exception as e:
    print(json.dumps({"error": str(e), "status": "failed"}))
"""
            result = subprocess.run(
                [venv_python, "-c", code],
                capture_output=True, text=True, timeout=15, cwd=str(WORKSPACE),
            )
            return ToolResult("quantum_bridge", True,
                              result.stdout.strip() if result.stdout else result.stderr.strip(),
                              None, time.time() - t0, {"operation": "score"})

        else:
            return ToolResult("quantum_bridge", False, "",
                              f"Unknown operation: {operation}. Use: status, vqpu, network, daemon, score",
                              time.time() - t0)

    except Exception as e:
        return ToolResult("quantum_bridge", False, "", str(e), time.time() - t0)


def tool_diff_viewer(path: str = "", staged: bool = False) -> ToolResult:
    """View git diffs — unstaged changes by default, or staged with staged=true.

    If `path` is given, shows diff for that specific file only.
    """
    t0 = time.time()
    try:
        cmd = ["git", "diff"]
        if staged:
            cmd.append("--staged")
        cmd.append("--stat")

        # First get summary
        stat_result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10, cwd=str(WORKSPACE),
        )

        # Then get actual diff (limited)
        diff_cmd = ["git", "diff"]
        if staged:
            diff_cmd.append("--staged")
        if path:
            diff_cmd.extend(["--", path])
        diff_cmd.extend(["-U3"])  # 3 lines context

        diff_result = subprocess.run(
            diff_cmd, capture_output=True, text=True, timeout=15, cwd=str(WORKSPACE),
        )

        output = "=== Diff Summary ===\n"
        output += stat_result.stdout[-2000:] if stat_result.stdout else "(no changes)\n"
        output += "\n=== Diff Detail ===\n"
        diff_text = diff_result.stdout if diff_result.stdout else "(no changes)\n"
        if len(diff_text) > 3000:
            output += diff_text[:3000] + f"\n... (truncated, {len(diff_text)} total chars)"
        else:
            output += diff_text

        return ToolResult("diff_viewer", True, output, None, time.time() - t0,
                          {"staged": staged, "path": path or "(all)",
                           "diff_size": len(diff_result.stdout or "")})
    except Exception as e:
        return ToolResult("diff_viewer", False, "", str(e), time.time() - t0)


def tool_http_probe(url: str = "http://127.0.0.1:8081/health",
                    method: str = "GET", timeout: int = 5) -> ToolResult:
    """Probe an HTTP endpoint and report status, latency, and response body.

    Restricted to localhost and internal URLs for safety.
    """
    t0 = time.time()
    try:
        import urllib.request
        import urllib.error

        # Safety: only allow localhost and internal URLs
        allowed_prefixes = ("http://127.0.0.1", "http://localhost", "http://0.0.0.0")
        if not any(url.startswith(p) for p in allowed_prefixes):
            return ToolResult("http_probe", False, "",
                              f"Blocked: only localhost URLs allowed. Got: {url}", time.time() - t0)

        req = urllib.request.Request(url, method=method.upper())
        req.add_header("User-Agent", "L104-Agent/3.0")

        try:
            with urllib.request.urlopen(req, timeout=min(timeout, 10)) as resp:
                status = resp.getcode()
                body = resp.read(4000).decode("utf-8", errors="replace")
                headers = dict(resp.headers)
                latency = time.time() - t0

                output = f"Status: {status}\n"
                output += f"Latency: {latency*1000:.0f}ms\n"
                output += f"Content-Type: {headers.get('Content-Type', 'unknown')}\n"
                output += f"Content-Length: {headers.get('Content-Length', 'unknown')}\n"
                output += f"\nBody:\n{body[:2000]}"

                return ToolResult("http_probe", True, output, None, latency,
                                  {"status": status, "latency_ms": round(latency * 1000),
                                   "content_type": headers.get("Content-Type", "")})
        except urllib.error.HTTPError as e:
            latency = time.time() - t0
            body = e.read(2000).decode("utf-8", errors="replace") if e.fp else ""
            return ToolResult("http_probe", True,
                              f"Status: {e.code} {e.reason}\nLatency: {latency*1000:.0f}ms\nBody:\n{body}",
                              None, latency,
                              {"status": e.code, "latency_ms": round(latency * 1000)})
        except urllib.error.URLError as e:
            return ToolResult("http_probe", False, "",
                              f"Connection failed: {e.reason}", time.time() - t0)
    except Exception as e:
        return ToolResult("http_probe", False, "", str(e), time.time() - t0)


# ─── Tool Registry ─────────────────────────────────────────────

class ToolRegistry:
    """Central registry of all available tools with DeepSeek function schemas."""

    TOOLS: dict[str, Callable] = {
        "read_file": tool_read_file,
        "write_file": tool_write_file,
        "edit_file": tool_edit_file,
        "list_files": tool_list_files,
        "run_shell": tool_run_shell,
        "search_code": tool_search_code,
        "analyze_code": tool_analyze_code,
        "python_exec": tool_python_exec,
        "git_status": tool_git_status,
        "system_metrics": tool_system_metrics,
        "dependency_check": tool_dependency_check,
        "quantum_bridge": tool_quantum_bridge,
        "diff_viewer": tool_diff_viewer,
        "http_probe": tool_http_probe,
    }

    SCHEMAS: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file from the workspace. Returns numbered lines.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path relative to workspace"},
                        "offset": {"type": "integer", "description": "Starting line (0-based)", "default": 0},
                        "limit": {"type": "integer", "description": "Max lines to read", "default": 500},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write content to a file. Creates parent dirs if needed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path relative to workspace"},
                        "content": {"type": "string", "description": "Full file content to write"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Replace exact text in a file. Use for targeted edits.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path relative to workspace"},
                        "old_text": {"type": "string", "description": "Exact text to find and replace"},
                        "new_text": {"type": "string", "description": "Replacement text"},
                    },
                    "required": ["path", "old_text", "new_text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files in a directory with optional glob pattern.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Directory path", "default": "."},
                        "pattern": {"type": "string", "description": "Glob pattern (e.g. '*.py')", "default": "*"},
                        "max_results": {"type": "integer", "description": "Max files", "default": 200},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_shell",
                "description": "Run a shell command. Timeout 60s. Some commands blocked for safety.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to execute"},
                        "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 30},
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_code",
                "description": "Search code files recursively. Returns matching lines with file paths and line numbers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Regex or literal search pattern"},
                        "path": {"type": "string", "description": "Directory to search", "default": "."},
                        "file_type": {"type": "string", "description": "File extension filter (py, swift, js, rs)", "default": ""},
                        "max_results": {"type": "integer", "description": "Max results", "default": 30},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "analyze_code",
                "description": "Analyze a source file: classes, functions, imports, complexity, TODOs. Works with Python and Swift.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Source file path to analyze"},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "python_exec",
                "description": "Execute Python code in the project's virtual environment. Use for testing, profiling, or computation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python code to execute"},
                        "timeout": {"type": "integer", "description": "Timeout seconds", "default": 30},
                    },
                    "required": ["code"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "git_status",
                "description": "Get current branch, git status (changed files), and the 5 most recent commits.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "system_metrics",
                "description": "Collect system health metrics: CPU load, memory, disk, L104 processes, server status, launchd agents, state file freshness.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "dependency_check",
                "description": "Check Python dependencies. Pass 'module' to check one import, 'path' to scan a file's imports, or neither to list all packages.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "module": {"type": "string", "description": "Single module to check (e.g. 'numpy')", "default": ""},
                        "path": {"type": "string", "description": "File path to scan imports from", "default": ""},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "quantum_bridge",
                "description": "Bridge to L104 quantum subsystems. Operations: 'status' (health overview), 'vqpu' (VQPU state), 'network' (quantum network), 'daemon' (AI daemon), 'score' (AGI scoring).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string", "description": "Operation: status, vqpu, network, daemon, score", "default": "status"},
                        "params": {"type": "string", "description": "JSON params for the operation", "default": "{}"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "diff_viewer",
                "description": "View git diffs with summary and detail. Shows unstaged changes by default, or staged changes with staged=true.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Specific file to diff (blank for all)", "default": ""},
                        "staged": {"type": "boolean", "description": "Show staged changes instead of unstaged", "default": False},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "http_probe",
                "description": "Probe a localhost HTTP endpoint. Returns status code, latency, headers, and response body. Only localhost URLs allowed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to probe (localhost only)", "default": "http://127.0.0.1:8081/health"},
                        "method": {"type": "string", "description": "HTTP method (GET, POST, HEAD)", "default": "GET"},
                        "timeout": {"type": "integer", "description": "Timeout seconds", "default": 5},
                    },
                },
            },
        },
    ]

    @classmethod
    def execute(cls, name: str, args: dict[str, Any]) -> ToolResult:
        fn = cls.TOOLS.get(name)
        if fn is None:
            return ToolResult(name, False, "", f"Unknown tool: {name}")
        try:
            return fn(**args)
        except TypeError as e:
            return ToolResult(name, False, "", f"Invalid args for {name}: {e}")

    @classmethod
    def get_schemas(cls, enabled: list[str] | None = None) -> list[dict[str, Any]]:
        if enabled is None:
            return cls.SCHEMAS
        return [s for s in cls.SCHEMAS if s["function"]["name"] in enabled]

    @classmethod
    def list_tools(cls) -> list[dict[str, str]]:
        """List all available tools with descriptions."""
        return [
            {"name": s["function"]["name"], "description": s["function"]["description"]}
            for s in cls.SCHEMAS
        ]
