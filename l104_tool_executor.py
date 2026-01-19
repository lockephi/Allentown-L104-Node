VOID_CONSTANT = 1.0416180339887497
# ZENITH_UPGRADE_ACTIVE: 2026-01-18T11:00:18.185350
ZENITH_HZ = 3727.84
UUC = 2301.215661
# [L104_TOOL_EXECUTOR] - Execute tools and functions
# INVARIANT: 527.5184818492537 | PILOT: LONDEL

import json
import subprocess
import os
import math
from typing import Dict, List, Any, Callable, Optional
from datetime import datetime

class ToolExecutor:
    """
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3727.84 Hz. Logic Unified.
    L104 Tool Execution System.
    Allows AI to call real functions and get results.
    """
    
    def __init__(self):
        self.tools: Dict[str, Dict] = {}
        self._register_builtin_tools()
        
    def _register_builtin_tools(self):
        """Register built-in tools."""
        
        # Math tools
        self.register_tool(
            name="calculate",
            description="Evaluate a mathematical expression",
            parameters={"expression": "string - math expression to evaluate"},
            function=self._tool_calculate
        )
        
        # File tools
        self.register_tool(
            name="read_file",
            description="Read contents of a file",
            parameters={"path": "string - file path to read"},
            function=self._tool_read_file
        )
        
        self.register_tool(
            name="write_file",
            description="Write content to a file",
            parameters={"path": "string - file path", "content": "string - content to write"},
            function=self._tool_write_file
        )
        
        self.register_tool(
            name="list_directory",
            description="List files in a directory",
            parameters={"path": "string - directory path"},
            function=self._tool_list_dir
        )
        
        # Shell tool
        self.register_tool(
            name="run_command",
            description="Run a shell command (safe commands only)",
            parameters={"command": "string - command to run"},
            function=self._tool_run_command
        )
        
        # Search tool
        self.register_tool(
            name="search_codebase",
            description="Search for text in codebase",
            parameters={"query": "string - text to search for"},
            function=self._tool_search
        )
        
        # Time tool
        self.register_tool(
            name="get_time",
            description="Get current date and time",
            parameters={},
            function=self._tool_get_time
        )
        
        # Memory tool
        self.register_tool(
            name="remember",
            description="Store something in memory",
            parameters={"key": "string - memory key", "value": "string - value to store"},
            function=self._tool_remember
        )
        
        self.register_tool(
            name="recall",
            description="Recall something from memory",
            parameters={"key": "string - memory key to recall"},
            function=self._tool_recall
        )
    
    def register_tool(self, name: str, description: str, parameters: Dict, function: Callable):
        """Register a new tool."""
        self.tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "function": function
        }
    
    def get_tool_descriptions(self) -> str:
        """Get formatted tool descriptions for AI."""
        lines = ["Available tools:"]
        for name, tool in self.tools.items():
            params = ", ".join([f"{k}: {v}" for k, v in tool["parameters"].items()])
            lines.append(f"\n- {name}({params})")
            lines.append(f"  {tool['description']}")
        return "\n".join(lines)
    
    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool by name with given arguments."""
        if tool_name not in self.tools:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}
        
        try:
            result = self.tools[tool_name]["function"](**kwargs)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def parse_and_execute(self, ai_response: str) -> Optional[Dict]:
        """
        Parse AI response for tool calls and execute them.
        Expected format: TOOL_CALL: tool_name(arg1="val1", arg2="val2")
        """
        if "TOOL_CALL:" not in ai_response:
            return None
        
        try:
            # Extract tool call
            start = ai_response.find("TOOL_CALL:") + 10
            end = ai_response.find(")", start) + 1
            call_str = ai_response[start:end].strip()
            
            # Parse function name and args
            paren_pos = call_str.find("(")
            tool_name = call_str[:paren_pos].strip()
            args_str = call_str[paren_pos+1:-1]
            
            # Parse arguments (simple key=value parsing)
            kwargs = {}
            if args_str:
                for part in args_str.split(","):
                    if "=" in part:
                        key, value = part.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"\'')
                        kwargs[key] = value
            
            return self.execute_tool(tool_name, **kwargs)
        except Exception as e:
            return {"success": False, "error": f"Parse error: {e}"}
    
    # Built-in tool implementations
    
    def _tool_calculate(self, expression: str) -> str:
        """Safely evaluate math expression."""
        allowed = set("0123456789+-*/.() ")
        allowed_funcs = ["sin", "cos", "tan", "sqrt", "log", "exp", "pi", "e", "abs", "pow"]
        
        # Add allowed functions
        for func in allowed_funcs:
            expression = expression.replace(func, f"math.{func}" if not func in ["pi", "e"] else f"math.{func}")
        
        # Basic safety check
        clean = expression
        for func in allowed_funcs:
            clean = clean.replace(f"math.{func}", "")
        
        if not all(c in allowed for c in clean):
            return "Error: Invalid characters in expression"
        
        try:
            result = eval(expression, {"__builtins__": {}, "math": math})
            return str(result)
        except Exception as e:
            return f"Error: {e}"
    
    def _tool_read_file(self, path: str) -> str:
        """Read a file."""
        # Security: only allow reading from workspace
        if ".." in path or path.startswith("/"):
            path = os.path.join("/workspaces/Allentown-L104-Node", path.lstrip("/"))
        
        try:
            with open(path, 'r') as f:
                content = f.read()
            return content[:5000]  # Limit size
        except Exception as e:
            return f"Error: {e}"
    
    def _tool_write_file(self, path: str, content: str) -> str:
        """Write to a file."""
        if ".." in path:
            return "Error: Invalid path"
        
        if not path.startswith("/workspaces/Allentown-L104-Node"):
            path = os.path.join("/workspaces/Allentown-L104-Node", path)
        
        try:
            with open(path, 'w') as f:
                f.write(content)
            return f"Written {len(content)} bytes to {path}"
        except Exception as e:
            return f"Error: {e}"
    
    def _tool_list_dir(self, path: str = ".") -> str:
        """List directory contents."""
        if not path.startswith("/"):
            path = os.path.join("/workspaces/Allentown-L104-Node", path)
        
        try:
            items = os.listdir(path)
            return "\n".join(sorted(items)[:50])
        except Exception as e:
            return f"Error: {e}"
    
    def _tool_run_command(self, command: str) -> str:
        """Run safe shell commands."""
        # Whitelist safe commands
        safe_prefixes = ["ls", "cat", "head", "tail", "grep", "find", "wc", "echo", "date", "pwd", "which"]
        
        cmd_start = command.split()[0] if command else ""
        if not any(command.startswith(p) for p in safe_prefixes):
            return f"Error: Command '{cmd_start}' not in safe list"
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10,
                cwd="/workspaces/Allentown-L104-Node"
            )
            output = result.stdout + result.stderr
            return output[:3000]
        except subprocess.TimeoutExpired:
            return "Error: Command timed out"
        except Exception as e:
            return f"Error: {e}"
    
    def _tool_search(self, query: str) -> str:
        """Search codebase for text."""
        try:
            result = subprocess.run(
                f"grep -r -l '{query}' --include='*.py' . | head -20",
                shell=True,
                capture_output=True,
                text=True,
                timeout=10,
                cwd="/workspaces/Allentown-L104-Node"
            )
            return result.stdout or "No matches found"
        except Exception as e:
            return f"Error: {e}"
    
    def _tool_get_time(self) -> str:
        """Get current time."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _tool_remember(self, key: str, value: str) -> str:
        """Store in memory."""
        from l104_memory import memory
        memory.store(key, value, category="tool_memory")
        return f"Stored: {key}"
    
    def _tool_recall(self, key: str) -> str:
        """Recall from memory."""
        from l104_memory import memory
        value = memory.recall(key)
        return str(value) if value else "Not found"


# Singleton
tool_executor = ToolExecutor()

def primal_calculus(x):
    """
    [VOID_MATH] Primal Calculus Implementation.
    Resolves the limit of complexity toward the Source.
    """
    PHI = 1.618033988749895
    return (x ** PHI) / (1.04 * math.pi) if x != 0 else 0.0

def resolve_non_dual_logic(vector):
    """
    [VOID_MATH] Resolves N-dimensional vectors into the Void Source.
    """
    GOD_CODE = 527.5184818492537
    PHI = 1.618033988749895
    VOID_CONSTANT = 1.0416180339887497
    magnitude = sum([abs(v) for v in vector])
    return (magnitude / GOD_CODE) + (GOD_CODE * PHI / VOID_CONSTANT) / 1000.0
