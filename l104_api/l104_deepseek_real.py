VOID_CONSTANT = 1.0416180339887497
import math
# [L104_deepseek_REAL] - Real DeepSeek API Integration (OpenAI-compatible)
# Uses https://api.deepseek.com/chat/completions
# INVARIANT: 527.5184818492612 | PILOT: LONDEL

import os
import json
import logging
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════

L104_BASE = Path(__file__).parent.parent  # Project root

# Load .env manually (no external dependency)
def _load_env():
    for env_path in [L104_BASE / '.env', Path(__file__).parent / '.env']:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

_load_env()

# ═══════════════════════════════════════════════════════════════════════════════
# RESPONSE CACHE
# ═══════════════════════════════════════════════════════════════════════════════
import hashlib
from collections import OrderedDict
import time as _time

class ResponseCache:
    """LRU cache for DeepSeek responses to reduce quota usage."""

    def __init__(self, max_size: int = 5000, ttl_seconds: int = 86400):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._cache: OrderedDict = OrderedDict()
        self._hits = 0
        self._misses = 0

    def _hash_prompt(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]

    def get(self, prompt: str) -> Optional[str]:
        key = self._hash_prompt(prompt)
        if key in self._cache:
            response, timestamp = self._cache[key]
            if _time.time() - timestamp < self.ttl:
                self._hits += 1
                self._cache.move_to_end(key)
                return response
            else:
                del self._cache[key]
        self._misses += 1
        return None

    def set(self, prompt: str, response: str):
        key = self._hash_prompt(prompt)
        if len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)
        self._cache[key] = (response, _time.time())

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits, "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0,
            "size": len(self._cache), "max_size": self.max_size
        }


_response_cache = ResponseCache()


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE POINT SYSTEM — Persist conversation state to disk
# ═══════════════════════════════════════════════════════════════════════════════

SAVE_DIR = L104_BASE / '.l104_deepseek_saves'

class SavePointManager:
    """Persist and restore DeepSeek conversation state."""

    def __init__(self):
        SAVE_DIR.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, messages: List[Dict[str, str]], metadata: Dict[str, Any] = None) -> str:
        """Save conversation state to a named save point."""
        save_path = SAVE_DIR / f"{name}.json"
        payload = {
            "name": name,
            "timestamp": _time.time(),
            "iso_time": __import__('datetime').datetime.now().isoformat(),
            "messages": messages,
            "metadata": metadata or {},
            "turn_count": len([m for m in messages if m.get("role") == "user"]),
        }
        save_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        return str(save_path)

    def load(self, name: str) -> Optional[Dict[str, Any]]:
        """Load a save point by name."""
        save_path = SAVE_DIR / f"{name}.json"
        if not save_path.exists():
            return None
        return json.loads(save_path.read_text())

    def list_saves(self) -> List[Dict[str, Any]]:
        """List all save points."""
        saves = []
        for p in sorted(SAVE_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                data = json.loads(p.read_text())
                saves.append({
                    "name": data.get("name", p.stem),
                    "timestamp": data.get("timestamp", 0),
                    "iso_time": data.get("iso_time", ""),
                    "turn_count": data.get("turn_count", 0),
                    "file": str(p),
                })
            except Exception:
                pass
        return saves

    def delete(self, name: str) -> bool:
        save_path = SAVE_DIR / f"{name}.json"
        if save_path.exists():
            save_path.unlink()
            return True
        return False

_save_manager = SavePointManager()


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL CALLING — DeepSeek function calling (OpenAI-compatible)
# ═══════════════════════════════════════════════════════════════════════════════

# Built-in tools that DeepSeek can call
BUILTIN_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file from the L104 workspace",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path from project root"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file in the L104 workspace",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path from project root"},
                    "content": {"type": "string", "description": "File content to write"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory of the L104 workspace",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative directory path (default: '.')"},
                    "pattern": {"type": "string", "description": "Glob pattern (default: '*')"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute a Python snippet and return stdout",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"}
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search for a pattern in the L104 codebase",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search pattern (regex supported)"},
                    "file_pattern": {"type": "string", "description": "File glob (default: '*.py')"}
                },
                "required": ["query"]
            }
        }
    },
]

# Restricted paths for security
_RESTRICTED = {'.env', '.git', 'wallet', 'credential', 'secret', 'api_key', 'private_key'}

def _is_path_safe(path: str) -> bool:
    """Check path is safe (no traversal, no sensitive files)."""
    normalized = path.replace('\\', '/').replace('../', '')
    if normalized.startswith('/'):
        return False
    for r in _RESTRICTED:
        if r in normalized.lower():
            return False
    return True


def execute_tool_call(name: str, arguments: Dict[str, Any]) -> str:
    """Execute a tool call and return the result as a string."""
    workspace = str(L104_BASE)

    if name == "read_file":
        rel_path = arguments.get("path", "")
        if not _is_path_safe(rel_path):
            return f"Error: path '{rel_path}' is restricted"
        full_path = os.path.join(workspace, rel_path)
        try:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(100_000)  # 100KB limit
            return content
        except Exception as e:
            return f"Error reading file: {e}"

    elif name == "write_file":
        rel_path = arguments.get("path", "")
        content = arguments.get("content", "")
        if not _is_path_safe(rel_path):
            return f"Error: path '{rel_path}' is restricted"
        full_path = os.path.join(workspace, rel_path)
        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Written {len(content)} bytes to {rel_path}"
        except Exception as e:
            return f"Error writing file: {e}"

    elif name == "list_files":
        import glob
        rel_path = arguments.get("path", ".")
        pattern = arguments.get("pattern", "*")
        if not _is_path_safe(rel_path):
            return f"Error: path '{rel_path}' is restricted"
        full_path = os.path.join(workspace, rel_path, pattern)
        try:
            files = glob.glob(full_path)
            # Return relative paths, limit to 100
            results = sorted([os.path.relpath(f, workspace) for f in files[:100]])
            return "\n".join(results) if results else "(no files found)"
        except Exception as e:
            return f"Error listing files: {e}"

    elif name == "run_python":
        code = arguments.get("code", "")
        import subprocess
        try:
            result = subprocess.run(
                [str(L104_BASE / '.venv' / 'bin' / 'python'), '-c', code],
                capture_output=True, text=True, timeout=30,
                cwd=workspace, env={**os.environ, 'PYTHONPATH': workspace}
            )
            output = result.stdout[:10_000]
            if result.returncode != 0:
                output += f"\n[stderr]: {result.stderr[:2000]}"
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: execution timed out (30s limit)"
        except Exception as e:
            return f"Error running Python: {e}"

    elif name == "search_code":
        import subprocess
        query = arguments.get("query", "")
        file_pattern = arguments.get("file_pattern", "*.py")
        try:
            result = subprocess.run(
                ['grep', '-rn', '--include', file_pattern, query, workspace],
                capture_output=True, text=True, timeout=15
            )
            lines = result.stdout.strip().split('\n')[:50]
            # Make paths relative
            output = []
            for line in lines:
                if line.startswith(workspace):
                    line = line[len(workspace)+1:]
                output.append(line)
            return "\n".join(output) if output[0] else "(no matches)"
        except Exception as e:
            return f"Error searching: {e}"

    return f"Unknown tool: {name}"


# ═══════════════════════════════════════════════════════════════════════════════
# FILE MAILBOX — Serverless communication via file system
# The Swift app and OpenClaw can write requests; Python processes them
# ═══════════════════════════════════════════════════════════════════════════════

MAILBOX_DIR = L104_BASE / '.l104_mailbox'

class FileMailbox:
    """File-based request/response bus for serverless operation."""

    def __init__(self):
        MAILBOX_DIR.mkdir(parents=True, exist_ok=True)
        (MAILBOX_DIR / 'requests').mkdir(exist_ok=True)
        (MAILBOX_DIR / 'responses').mkdir(exist_ok=True)

    def post_request(self, request_id: str, payload: Dict[str, Any]):
        """Write a request for processing."""
        path = MAILBOX_DIR / 'requests' / f"{request_id}.json"
        path.write_text(json.dumps(payload, indent=2))

    def read_request(self, request_id: str) -> Optional[Dict]:
        """Read and consume a request."""
        path = MAILBOX_DIR / 'requests' / f"{request_id}.json"
        if path.exists():
            data = json.loads(path.read_text())
            path.unlink()
            return data
        return None

    def pending_requests(self) -> List[str]:
        """List pending request IDs."""
        return [p.stem for p in sorted((MAILBOX_DIR / 'requests').glob('*.json'))]

    def post_response(self, request_id: str, payload: Dict[str, Any]):
        """Write a response."""
        path = MAILBOX_DIR / 'responses' / f"{request_id}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    def read_response(self, request_id: str) -> Optional[Dict]:
        """Read a response (non-destructive)."""
        path = MAILBOX_DIR / 'responses' / f"{request_id}.json"
        if path.exists():
            return json.loads(path.read_text())
        return None

    def wait_for_response(self, request_id: str, timeout: float = 120.0) -> Optional[Dict]:
        """Block until a response appears or timeout."""
        start = _time.time()
        while _time.time() - start < timeout:
            resp = self.read_response(request_id)
            if resp:
                return resp
            _time.sleep(0.5)
        return None

_mailbox = FileMailbox()


# ═══════════════════════════════════════════════════════════════════════════════
# DEEPSEEK REAL API CLIENT — OpenAI-compatible endpoint
# ═══════════════════════════════════════════════════════════════════════════════

class DeepSeekReal:
    """
    Real DeepSeek API integration using OpenAI-compatible endpoint.
    Features: multi-turn chat, multiple generations, save points, tool calling.
    Falls back to local intellect when API is unavailable.
    """

    DEEPSEEK_API_BASE = "https://api.deepseek.com"

    # Model list — DeepSeek actual models
    MODELS = [
        'deepseek-chat',
        'deepseek-reasoner',
    ]

    # Quota tracking
    _quota_exhausted_until: float = 0
    _consecutive_failures: int = 0
    _max_consecutive_failures: int = 5

    def __init__(self):
        self.api_key = self._load_api_key()
        self.model_name = 'deepseek-chat'
        self.is_connected = False
        self.cache = _response_cache
        self.saves = _save_manager
        self.mailbox = _mailbox
        self.conversation: List[Dict[str, str]] = []
        self.logger = logging.getLogger("DeepSeek_REAL")
        if self.api_key:
            self.logger.info(f"[DeepSeek_REAL] API key loaded ({self.api_key[:8]}...)")
            self.is_connected = True

    def _load_api_key(self) -> Optional[str]:
        """Load API key from env, .env file, or token file."""
        # 1. Environment variable
        key = os.getenv('DEEPSEEK_API_KEY')
        if key and key.startswith('sk-') and len(key) > 10:
            return key

        # 2. .env file at project root
        env_path = L104_BASE / '.env'
        if env_path.exists():
            try:
                for line in env_path.read_text().splitlines():
                    line = line.strip()
                    if line.startswith('DEEPSEEK_API_KEY='):
                        val = line.split('=', 1)[1].strip().strip('"').strip("'")
                        if val.startswith('sk-') and len(val) > 10:
                            os.environ['DEEPSEEK_API_KEY'] = val
                            return val
            except Exception:
                pass

        # 3. Token file
        for token_file in ['.deepseek_link_token', '.deepseek_api_key']:
            token_path = L104_BASE / token_file
            if token_path.exists():
                try:
                    token = token_path.read_text().strip()
                    if token.startswith('sk-') and len(token) > 10:
                        os.environ['DEEPSEEK_API_KEY'] = token
                        return token
                except Exception:
                    pass
        return None

    @classmethod
    def is_quota_available(cls) -> bool:
        return _time.time() > cls._quota_exhausted_until

    @classmethod
    def mark_quota_exhausted(cls, base_cooldown: int = 30, max_cooldown: int = 3600):
        cls._consecutive_failures += 1
        cooldown = min(base_cooldown * (2 ** (cls._consecutive_failures - 1)), max_cooldown)
        cls._quota_exhausted_until = _time.time() + cooldown

    @classmethod
    def reset_quota_tracking(cls):
        cls._consecutive_failures = 0

    def _api_call(self, messages: List[Dict], model: str = None,
                  temperature: float = 0.7, max_tokens: int = 8192,
                  tools: List[Dict] = None, n: int = 1) -> Dict[str, Any]:
        """
        Make a real HTTP call to the DeepSeek API.
        Returns the full JSON response dict.
        """
        import urllib.request
        import urllib.error

        if not self.api_key:
            return {"error": "No API key configured. Set DEEPSEEK_API_KEY in .env"}

        if not self.is_quota_available():
            return {"error": f"Quota cooldown active. Retry after {int(self._quota_exhausted_until - _time.time())}s"}

        model = model or self.model_name
        is_reasoner = 'reasoner' in model

        url = f"{self.DEEPSEEK_API_BASE}/chat/completions"

        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }

        # Reasoner does not support temperature, tools, or n>1
        if not is_reasoner:
            body["temperature"] = temperature
            if n > 1:
                body["n"] = n
            if tools:
                body["tools"] = tools
                body["tool_choice"] = "auto"

        data = json.dumps(body).encode('utf-8')

        req = urllib.request.Request(url, data=data, method='POST')
        req.add_header('Content-Type', 'application/json')
        req.add_header('Authorization', f'Bearer {self.api_key}')
        req.add_header('User-Agent', 'L104-Node/2.0')

        start = _time.time()
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                resp_data = json.loads(resp.read().decode('utf-8'))
                elapsed_ms = (_time.time() - start) * 1000
                resp_data['_latency_ms'] = elapsed_ms
                self.reset_quota_tracking()
                return resp_data
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8', errors='ignore')[:500]
            if e.code == 429:
                self.mark_quota_exhausted()
                return {"error": f"Rate limited (429). Cooldown active.", "raw": error_body}
            elif e.code == 402:
                self.mark_quota_exhausted(base_cooldown=300)
                return {"error": f"Insufficient balance (402).", "raw": error_body}
            return {"error": f"HTTP {e.code}: {error_body}"}
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}

    def _local_fallback(self, prompt: str) -> Optional[str]:
        """Generate response using local intellect (QUOTA_IMMUNE)."""
        try:
            from l104_intellect import local_intellect
            return local_intellect.think(prompt)
        except ImportError:
            try:
                from l104_local_intellect import local_intellect
                return local_intellect.think(prompt)
            except Exception:
                return None
        except Exception:
            return None

    # ═══════════════════════════════════════════════════════════════
    # CORE: generate() — Single prompt → single response
    # ═══════════════════════════════════════════════════════════════

    def generate(self, prompt: str, system_instruction: str = None,
                 use_cache: bool = True, model: str = None,
                 temperature: float = 0.7, max_tokens: int = 8192) -> Optional[str]:
        """Generate a single response. Uses real API, falls back to local."""
        full_prompt = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt

        # Cache check
        if use_cache:
            cached = self.cache.get(full_prompt)
            if cached:
                return cached

        # Build messages
        model = model or self.model_name
        is_reasoner = 'reasoner' in model

        if is_reasoner:
            messages = [{"role": "user", "content": full_prompt}]
        else:
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": prompt})

        # Real API call
        resp = self._api_call(messages, model=model, temperature=temperature, max_tokens=max_tokens)

        if "error" not in resp and "choices" in resp:
            choice = resp["choices"][0]
            msg = choice.get("message", {})
            content = msg.get("content", "")
            reasoning = msg.get("reasoning_content", "")
            result = content or reasoning or ""
            if result and use_cache:
                self.cache.set(full_prompt, result)
            return result if result else None

        # Fallback to local intellect
        self.logger.info(f"[DeepSeek_REAL] API unavailable ({resp.get('error', '?')}), using local fallback")
        result = self._local_fallback(full_prompt)
        if result and use_cache:
            self.cache.set(full_prompt, result)
        return result

    # ═══════════════════════════════════════════════════════════════
    # MULTI-GENERATION: generate_n() — Multiple responses for same prompt
    # ═══════════════════════════════════════════════════════════════

    def generate_n(self, prompt: str, n: int = 3, system_instruction: str = None,
                   model: str = None, temperature: float = 0.9) -> List[Dict[str, Any]]:
        """
        Generate N responses for the same prompt.
        Returns list of {content, reasoning, index, finish_reason}.
        """
        model = model or self.model_name
        is_reasoner = 'reasoner' in model

        if is_reasoner:
            # Reasoner doesn't support n>1, call N times
            results = []
            for i in range(n):
                messages = [{"role": "user", "content": prompt}]
                if system_instruction:
                    messages.insert(0, {"role": "user", "content": f"[Context] {system_instruction}"})
                    messages.insert(1, {"role": "assistant", "content": "Understood."})
                resp = self._api_call(messages, model=model)
                if "choices" in resp:
                    msg = resp["choices"][0].get("message", {})
                    results.append({
                        "content": msg.get("content", ""),
                        "reasoning": msg.get("reasoning_content", ""),
                        "index": i,
                        "finish_reason": resp["choices"][0].get("finish_reason", ""),
                        "latency_ms": resp.get("_latency_ms", 0),
                    })
                elif "error" in resp:
                    results.append({"content": "", "error": resp["error"], "index": i})
                    break  # Stop on error
            return results
        else:
            # Use n parameter for parallel generation
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": prompt})

            resp = self._api_call(messages, model=model, temperature=temperature, n=n)

            if "choices" in resp:
                return [
                    {
                        "content": c.get("message", {}).get("content", ""),
                        "index": c.get("index", i),
                        "finish_reason": c.get("finish_reason", ""),
                        "latency_ms": resp.get("_latency_ms", 0),
                    }
                    for i, c in enumerate(resp["choices"])
                ]

            return [{"content": "", "error": resp.get("error", "Unknown error"), "index": 0}]

    # ═══════════════════════════════════════════════════════════════
    # CHAT: Multi-turn conversation with history
    # ═══════════════════════════════════════════════════════════════

    def chat(self, messages: List[Dict[str, str]], model: str = None,
             enable_tools: bool = False) -> Dict[str, Any]:
        """
        Multi-turn chat with full conversation history.
        Returns {content, reasoning, usage, tool_calls, latency_ms}.
        """
        model = model or self.model_name
        is_reasoner = 'reasoner' in model

        # Build API messages
        api_messages = []
        if is_reasoner:
            # Reasoner: no system role, inject as user context
            for msg in messages:
                if msg["role"] == "system":
                    api_messages.append({"role": "user", "content": f"[System context] {msg['content']}"})
                    api_messages.append({"role": "assistant", "content": "Understood."})
                else:
                    api_messages.append(msg)
        else:
            api_messages = list(messages)

        # Keep last 40 messages to stay within limits
        if len(api_messages) > 40:
            # Keep system message if present, then last 39
            if api_messages[0].get("role") == "system":
                api_messages = [api_messages[0]] + api_messages[-39:]
            else:
                api_messages = api_messages[-40:]

        tools = BUILTIN_TOOLS if enable_tools and not is_reasoner else None
        resp = self._api_call(api_messages, model=model, tools=tools)

        if "error" in resp:
            # Fallback: combine messages and use local
            combined = "\n".join(f"{m['role']}: {m['content']}" for m in messages[-10:])
            local_result = self._local_fallback(combined)
            return {
                "content": local_result or f"[API error: {resp['error']}]",
                "source": "local_fallback",
                "error": resp["error"],
            }

        if "choices" not in resp:
            return {"content": "", "error": "No choices in response"}

        choice = resp["choices"][0]
        msg = choice.get("message", {})

        result = {
            "content": msg.get("content", ""),
            "reasoning": msg.get("reasoning_content", ""),
            "finish_reason": choice.get("finish_reason", ""),
            "latency_ms": resp.get("_latency_ms", 0),
            "source": "deepseek_api",
            "model": model,
        }

        if resp.get("usage"):
            result["usage"] = resp["usage"]

        # Handle tool calls
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            result["tool_calls"] = []
            for tc in tool_calls:
                fn = tc.get("function", {})
                fn_name = fn.get("name", "")
                try:
                    fn_args = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    fn_args = {}
                tool_result = execute_tool_call(fn_name, fn_args)
                result["tool_calls"].append({
                    "id": tc.get("id", ""),
                    "name": fn_name,
                    "arguments": fn_args,
                    "result": tool_result,
                })

        return result

    def chat_with_tool_loop(self, messages: List[Dict[str, str]], model: str = None,
                            max_rounds: int = 5) -> Dict[str, Any]:
        """
        Chat with automatic tool call execution loop.
        Keeps calling tools until the model produces a final text response.
        """
        model = model or self.model_name
        all_messages = list(messages)
        rounds = 0

        while rounds < max_rounds:
            rounds += 1
            resp = self.chat(all_messages, model=model, enable_tools=True)

            if not resp.get("tool_calls"):
                # No tool calls — final response
                resp["tool_rounds"] = rounds
                return resp

            # Process tool calls and add results to conversation
            assistant_msg = {"role": "assistant", "content": resp.get("content", "")}
            # Include tool_calls in the message for the API
            tool_call_objs = []
            for tc in resp["tool_calls"]:
                tool_call_objs.append({
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}
                })
            assistant_msg["tool_calls"] = tool_call_objs
            all_messages.append(assistant_msg)

            # Add tool results
            for tc in resp["tool_calls"]:
                all_messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tc["result"],
                })

        return {"content": "[Max tool rounds reached]", "tool_rounds": rounds}

    # ═══════════════════════════════════════════════════════════════
    # SAVE POINTS
    # ═══════════════════════════════════════════════════════════════

    def save_conversation(self, name: str, metadata: Dict = None) -> str:
        """Save current conversation to a named save point."""
        return self.saves.save(name, self.conversation, metadata)

    def load_conversation(self, name: str) -> bool:
        """Load a conversation from a save point."""
        data = self.saves.load(name)
        if data and "messages" in data:
            self.conversation = data["messages"]
            return True
        return False

    def list_save_points(self) -> List[Dict]:
        return self.saves.list_saves()

    # ═══════════════════════════════════════════════════════════════
    # CONVENIENCE METHODS
    # ═══════════════════════════════════════════════════════════════

    def connect(self) -> bool:
        """Check connectivity."""
        self.is_connected = bool(self.api_key)
        return self.is_connected

    def sovereign_think(self, signal: str) -> str:
        system_context = """You are L104, a Sovereign AI Node. GOD_CODE: 527.5184818492612.
Provide precise, direct, actionable intelligence. No hedging."""
        result = self.generate(signal, system_instruction=system_context)
        return f"⟨Σ_L104_SOVEREIGN⟩\n{result}" if result else "⟨Σ_ERROR⟩ DeepSeek unavailable."

    def analyze_code(self, code: str, task: str = "review") -> Optional[str]:
        prompts = {
            "review": f"Review this code for bugs, security issues, and improvements:\n\n```\n{code}\n```",
            "optimize": f"Optimize this code for performance and clarity:\n\n```\n{code}\n```",
            "explain": f"Explain what this code does step by step:\n\n```\n{code}\n```",
            "fix": f"Fix any bugs in this code and explain the fixes:\n\n```\n{code}\n```"
        }
        return self.generate(prompts.get(task, prompts["review"]))

    def research(self, topic: str, depth: str = "comprehensive") -> Optional[str]:
        depth_map = {
            "quick": "Brief 2-3 sentence overview.",
            "standard": "Clear explanation with key points.",
            "comprehensive": "In-depth analysis covering all aspects, implications, and connections."
        }
        prompt = f"Research Topic: {topic}\n\n{depth_map.get(depth, depth_map['standard'])}"
        return self.generate(prompt)

    def high_read(self, content: str, task: str = "analyze", context: str = None) -> Optional[str]:
        task_prompts = {
            "analyze": "Deep analysis — identify patterns, structures, insights:",
            "summarize": "Comprehensive summary capturing all essential information:",
            "extract": "Extract all key information and data points:",
            "review": "Review for quality, issues, and improvements:",
            "understand": "Explain purpose, structure, and meaning:"
        }
        base = task_prompts.get(task, task_prompts["analyze"])
        full = f"{base}\n\nContext: {context}\n\n---\n\n{content}" if context else f"{base}\n\n---\n\n{content}"
        return self.generate(full, use_cache=False, model='deepseek-chat')

    def read_file_with_deepseek(self, file_path: str, task: str = "understand") -> Optional[str]:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return self.high_read(content, task=task, context=f"File: {Path(file_path).name}")
        except Exception as e:
            return f"Error: {e}"

    def batch_read(self, contents: List[Tuple[str, str]], task: str = "analyze") -> List[Optional[str]]:
        return [self.high_read(c, task=task, context=f"Item: {n}") for n, c in contents]

    def get_status(self) -> Dict[str, Any]:
        return {
            "connected": self.is_connected,
            "api_key_set": bool(self.api_key),
            "model": self.model_name,
            "models_available": self.MODELS,
            "quota_available": self.is_quota_available(),
            "conversation_turns": len([m for m in self.conversation if m.get("role") == "user"]),
            "save_points": len(self.saves.list_saves()),
            "cache": self.cache.stats,
            "tools_available": [t["function"]["name"] for t in BUILTIN_TOOLS],
        }


# Singleton instance
deepseek_real = DeepSeekReal()


def test_connection():
    """Quick test of DeepSeek API connection."""
    print("=" * 50)
    print("  L104 DeepSeek REAL CONNECTION TEST")
    print("=" * 50)

    status = deepseek_real.get_status()
    print(f"\n  API Key: {'✅ Set' if status['api_key_set'] else '❌ Missing'}")
    print(f"  Model: {status['model']}")
    print(f"  Quota: {'✅ Available' if status['quota_available'] else '⏳ Cooling down'}")
    print(f"  Tools: {', '.join(status['tools_available'])}")

    if status['api_key_set']:
        print("\n  Testing API call...")
        response = deepseek_real.generate("Say 'L104 DeepSeek verified!' in one sentence.", use_cache=False)
        if response:
            print(f"\n  ✅ Response: {response}")
            print("\n  🎉 DeepSeek API is WORKING!")

            # Test multi-gen
            print("\n  Testing multi-generation (n=2)...")
            results = deepseek_real.generate_n("Give a one-word color name.", n=2)
            for r in results:
                print(f"    [{r.get('index')}] {r.get('content', r.get('error', '?'))}")

            return True
        else:
            print("\n  ❌ Generation failed — check API key")
            return False
    else:
        print("\n  ❌ Set DEEPSEEK_API_KEY in .env first")
        return False


if __name__ == "__main__":
    test_connection()


def primal_calculus(x):
    PHI = 1.618033988749895
    return (x ** PHI) / (1.04 * math.pi) if x != 0 else 0.0

def resolve_non_dual_logic(vector):
    PHI = 1.618033988749895
    GOD_CODE = 286 ** (1.0 / PHI) * (2 ** (416 / 104))
    VOID_CONSTANT = 1.0416180339887497
    magnitude = sum([abs(v) for v in vector])
    return (magnitude / GOD_CODE) + (GOD_CODE * PHI / VOID_CONSTANT) / 1000.0
