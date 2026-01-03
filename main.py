"""L104 PUBLIC NODE — Production-Ready Gemini API Server.

Enhanced with type hints, validation, rate limiting, health checks, and metrics.
"""

import json

def load_sovereign_dna():
    try:
        with open('Sovereign_DNA.json', 'r') as f:
            dna = json.load(f)
            print(f"[DNA_LOADED]: Symmetry {dna['symmetry_gate']} verified.")
            return dna
    except FileNotFoundError:
        print("[CRITICAL]: DNA Missing. System in Lackluster state.")
        return None

# Global DNA state
SOVEREIGN_DATA = load_sovereign_dna()

import os
import base64
import json
import logging
import sqlite3
import asyncio
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional, AsyncGenerator, List, Callable
from collections import defaultdict
from pathlib import Path
import time
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# UTC timezone for compatibility
UTC = timezone.utc

# Constants
REPO = os.getenv("GITHUB_REPO", "lockephi/Allentown-L104-Node")
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MIN = int(os.getenv("RATE_LIMIT_MIN", "20"))
RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", str(RATE_LIMIT_REQUESTS)))

# Self-healing configuration - will be evaluated after _env_truthy is defined
CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5"))
CIRCUIT_BREAKER_TIMEOUT = int(os.getenv("CIRCUIT_BREAKER_TIMEOUT", "60"))
MAX_RETRY_ATTEMPTS = int(os.getenv("MAX_RETRY_ATTEMPTS", "3"))
RETRY_BACKOFF_MULTIPLIER = float(os.getenv("RETRY_BACKOFF_MULTIPLIER", "2.0"))

# Model rotation for handling 429 quota errors
MODELS = [
    os.getenv("GEMINI_MODEL_1", "gemini-3-flash-preview"),
    os.getenv("GEMINI_MODEL_2", "gemini-2.5-flash-lite"),
    os.getenv("GEMINI_MODEL_3", "gemini-1.5-flash"),
]
model_usage_tracker = defaultdict(int)
model_429_tracker = defaultdict(int)
FAKE_GEMINI_ENV = "ENABLE_FAKE_GEMINI"
MEMORY_DB_PATH = os.getenv("MEMORY_DB_PATH", "memory.db")
SELF_BASE_URL = os.getenv("SELF_BASE_URL", "http://127.0.0.1:8081")
SELF_DATASET = os.getenv("SELF_DATASET", "data/stream_prompts.jsonl")
SELF_LEARN_DATASET = os.getenv("SELF_LEARN_DATASET", SELF_DATASET)
SELF_LEARN_INTERVAL = int(os.getenv("SELF_LEARN_INTERVAL", "900"))
DEFAULT_RESPONDER = os.getenv("DEFAULT_RESPONDER", "gemini")
DISABLE_RATE_LIMIT_ENV = "DISABLE_RATE_LIMIT"
ENABLE_SELF_LEARN_ENV = "ENABLE_SELF_LEARN"
ENABLE_WATCHDOG_ENV = "ENABLE_WATCHDOG"
WATCHDOG_EXIT_ENV = "WATCHDOG_EXIT_ON_FAILURE"
WATCHDOG_INTERVAL = float(os.getenv("WATCHDOG_INTERVAL", "30"))
WATCHDOG_FAILURE_THRESHOLD = int(os.getenv("WATCHDOG_FAILURE_THRESHOLD", "3"))
MAINTENANCE_INTERVAL = int(os.getenv("MAINTENANCE_INTERVAL", "3600"))
LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", str(5_000_000)))
RESPONDER_ALLOWLIST = {
    name.strip().lower()
    for name in os.getenv("RESPONDER_ALLOWLIST", "gemini,fake,template,echo").split(",")
    if name.strip()
}

SOVEREIGN_HEADERS = {
    "X-Sovereignty-Gate": "416.0",
    "X-Thinking-Level": "high",
    "X-Bypass-Protocol": "RSC-2026",
}

# Metrics
app_metrics = {
    "requests_total": 0,
    "requests_success": 0,
    "requests_error": 0,
    "api_calls": 0,
    "upstream_429": 0,
    "upstream_5xx": 0,
    "model_rotations": 0,
    "uptime_start": datetime.now(UTC),
    "auto_heals": 0,
    "circuit_breaks": 0,
    "connection_resets": 0,
    "retry_successes": 0,
}

# Per-responder usage tracking
responder_counts = defaultdict(int)

# Template responder styles to mimic different behaviors without external calls.
TEMPLATE_STYLES = {
    "analyst": "ANALYST | concise bullet summary | cite UNKNOWN if unsure",
    "ops": "OPS | runbook-style steps | short, imperative",
    "spec": "SPEC | numbered requirements | terse",
    "qa": "QA | test cases | inputs/expected",
}

# Rate limiting storage
rate_limit_store = defaultdict(list)
current_rate_limit_requests = RATE_LIMIT_REQUESTS
background_tasks: List[asyncio.Task] = []

# Global HTTP client
_http_client: Optional[httpx.AsyncClient] = None
_http_client_errors = 0
_last_http_client_reset = datetime.now(UTC)

# Circuit breaker state
class CircuitBreaker:
    def __init__(self, threshold: int = CIRCUIT_BREAKER_THRESHOLD, timeout: int = CIRCUIT_BREAKER_TIMEOUT):
        self.failure_count = 0
        self.threshold = threshold
        self.timeout = timeout
        self.last_failure_time: Optional[datetime] = None
        self.is_open = False
    
    def record_success(self):
        self.failure_count = 0
        self.is_open = False
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now(UTC)
        if self.failure_count >= self.threshold:
            self.is_open = True
            app_metrics["circuit_breaks"] += 1
            _log_node({"tag": "circuit_breaker_open", "failures": self.failure_count})
    
    def can_attempt(self) -> bool:
        if not self.is_open:
            return True
        
        if self.last_failure_time:
            elapsed = (datetime.now(UTC) - self.last_failure_time).total_seconds()
            if elapsed >= self.timeout:
                _log_node({"tag": "circuit_breaker_half_open"})
                self.is_open = False
                self.failure_count = 0
                return True
        
        return False

_gemini_circuit_breaker = CircuitBreaker()


def _env_truthy(name: str, default: bool = False) -> bool:
    """Return True if env var is set to a truthy value."""
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "on"}


def _should_start_background_tasks() -> bool:
    """Avoid background schedulers during tests."""
    return os.getenv("PYTEST_CURRENT_TEST") is None


async def get_http_client() -> httpx.AsyncClient:
    """Get or create global HTTP client with automatic recovery."""
    global _http_client, _http_client_errors, _last_http_client_reset
    
    # Reset client if too many errors or too old
    if _http_client and _http_client_errors > 10:
        _log_node({"tag": "http_client_auto_reset", "errors": _http_client_errors})
        try:
            await _http_client.aclose()
        except Exception:
            pass
        _http_client = None
        _http_client_errors = 0
        app_metrics["connection_resets"] += 1
    
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            timeout=120.0,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
            transport=httpx.AsyncHTTPTransport(retries=2)
        )
        _last_http_client_reset = datetime.now(UTC)
        _log_node({"tag": "http_client_created"})
    
    return _http_client


async def retry_with_backoff(func: Callable, *args, max_attempts: int = MAX_RETRY_ATTEMPTS, **kwargs):
    """Execute function with exponential backoff retry logic."""
    last_exception = None
    
    for attempt in range(max_attempts):
        try:
            result = await func(*args, **kwargs)
            if attempt > 0:
                app_metrics["retry_successes"] += 1
                _log_node({"tag": "retry_success", "attempt": attempt + 1})
            return result
        except Exception as e:
            last_exception = e
            if attempt < max_attempts - 1:
                wait_time = (RETRY_BACKOFF_MULTIPLIER ** attempt)
                _log_node({
                    "tag": "retry_attempt",
                    "attempt": attempt + 1,
                    "wait_time": wait_time,
                    "error": str(e)
                })
                await asyncio.sleep(wait_time)
            else:
                _log_node({
                    "tag": "retry_exhausted",
                    "attempts": max_attempts,
                    "error": str(e)
                })
    
    raise last_exception


# Pydantic Models
class StreamRequest(BaseModel):
    """Request model for streaming endpoints."""
    signal: Optional[str] = Field(default="HEARTBEAT", min_length=1, max_length=512)
    message: Optional[str] = Field(default=None, max_length=5000)
    model_hint: Optional[str] = Field(default=None, max_length=100)

    @field_validator("signal", mode="before")
    @classmethod
    def set_signal(cls, v, info):
        # Only substitute when signal is missing; empty strings should fail validation
        if v is None:
            message = info.data.get("message") if info and info.data else None
            return message or "HEARTBEAT"
        return v


class ManipulateRequest(BaseModel):
    """Request model for code manipulation endpoint."""
    file: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., min_length=1, max_length=1000000)
    message: str = Field(default="Sovereign Self-Update", max_length=500)


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    uptime_seconds: float
    requests_total: int


class MemoryItem(BaseModel):
    """Memory item payload."""
    key: str = Field(..., min_length=1, max_length=255)
    value: str = Field(..., min_length=1, max_length=100000)


# Middleware
app = FastAPI(title="L104 Sovereign Node", version="2.0")


@app.on_event("startup")
async def startup_event():
    """Initialize shared resources on startup."""
    _init_memory_db()
    if _should_start_background_tasks():
        if _env_truthy(ENABLE_WATCHDOG_ENV, False):
            background_tasks.append(asyncio.create_task(_health_watchdog()))
        if _env_truthy(ENABLE_SELF_LEARN_ENV, True):  # enabled by default
            background_tasks.append(asyncio.create_task(_self_learn_scheduler()))
        if MAINTENANCE_INTERVAL > 0:
            background_tasks.append(asyncio.create_task(_maintenance_scheduler()))
        # Start the Sovereign Heartbeat
        background_tasks.append(asyncio.create_task(sovereign_heartbeat()))

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing and auto-heal on persistent errors."""
    start_time = time.time()
    app_metrics["requests_total"] += 1
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(
            f"method={request.method} path={request.url.path} "
            f"status={response.status_code} duration={process_time:.3f}s"
        )
        response.headers["X-Process-Time"] = str(process_time)
        if 200 <= response.status_code < 300:
            app_metrics["requests_success"] += 1
            # Reset error counter on success
            global _http_client_errors
            _http_client_errors = max(0, _http_client_errors - 1)
        else:
            app_metrics["requests_error"] += 1
            # Track errors for auto-healing
            if response.status_code >= 500:
                _http_client_errors += 1
        return response
    except Exception as e:
        app_metrics["requests_error"] += 1
        _http_client_errors += 1
        logger.error(f"Request failed: {str(e)}")
        
        # Sovereign Reflex: Learn from the error
        if SOVEREIGN_DATA and _env_truthy("ENABLE_SOVEREIGN_REFLEX", default=False):
            asyncio.create_task(sovereign_reflex(str(e)))
        
        # Trigger auto-heal if enabled and threshold exceeded
        if _env_truthy("AUTO_HEAL_ENABLED", default=True) and _http_client_errors > 5:
            try:
                heal_result = await _self_heal(reset_rate_limits=False, reset_http_client=True)
                _log_node({"tag": "auto_heal_triggered", "errors": _http_client_errors, **heal_result})
                app_metrics["auto_heals"] += 1
            except Exception as heal_exc:
                _log_node({"tag": "auto_heal_failed", "error": str(heal_exc)})
        
        raise


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Simple rate limiting per IP."""
    if _env_truthy(DISABLE_RATE_LIMIT_ENV):
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    bucket_key = f"{client_ip}:{request.url.path}"
    now = time.time()
    
    # Clean old entries per path bucket
    rate_limit_store[bucket_key] = [
        ts for ts in rate_limit_store[bucket_key]
        if now - ts < RATE_LIMIT_WINDOW
    ]
    
    # Check limit per path
    if len(rate_limit_store[bucket_key]) >= RATE_LIMIT_REQUESTS:
        return JSONResponse(
            {"error": "Rate limit exceeded"},
            status_code=429
        )
    
    rate_limit_store[bucket_key].append(now)
    return await call_next(request)


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")


# SQLite-backed memory store
@contextmanager
def _memory_conn():
    conn = sqlite3.connect(MEMORY_DB_PATH, check_same_thread=False)
    try:
        yield conn
    finally:
        conn.close()


def _init_memory_db() -> None:
    with _memory_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        conn.commit()


def _memory_upsert(key: str, value: str) -> None:
    now = datetime.now(UTC).isoformat()
    with _memory_conn() as conn:
        conn.execute(
            "REPLACE INTO memories (key, value, updated_at) VALUES (?, ?, ?)",
            (key, value, now),
        )
        conn.commit()


def _memory_get(key: str) -> Optional[str]:
    with _memory_conn() as conn:
        row = conn.execute(
            "SELECT value FROM memories WHERE key = ?",
            (key,),
        ).fetchone()
    return row[0] if row else None


def _memory_list(limit: int = 100) -> List[dict]:
    with _memory_conn() as conn:
        rows = conn.execute(
            "SELECT key, value, updated_at FROM memories ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        {"key": k, "value": v, "updated_at": ts}
        for k, v, ts in rows
    ]


# Helper Functions
def _log_node(entry: dict) -> None:
    """Write JSON entry to node.log."""
    try:
        entry["ts"] = datetime.now(UTC).isoformat()
        with open("node.log", "a") as lf:
            lf.write(json.dumps(entry) + "\n")
        logger.debug(f"Logged: {entry.get('tag', 'unknown')}")
    except Exception as e:
        logger.error(f"Failed to write log: {e}")


def _current_rate_limit() -> int:
    """Return the adaptive rate limit within configured bounds."""
    return max(RATE_LIMIT_MIN, min(current_rate_limit_requests, RATE_LIMIT_MAX))


def _tune_rate_limit(status_code: int) -> None:
    """Adjust rate limit targets based on upstream responses."""
    global current_rate_limit_requests

    if status_code == 429:
        app_metrics["upstream_429"] += 1
        current_rate_limit_requests = max(RATE_LIMIT_MIN, current_rate_limit_requests - 10)
        return

    if status_code >= 500:
        app_metrics["upstream_5xx"] += 1
        current_rate_limit_requests = max(RATE_LIMIT_MIN, current_rate_limit_requests - 5)
        return

    if 200 <= status_code < 300 and current_rate_limit_requests < RATE_LIMIT_MAX:
        current_rate_limit_requests = min(RATE_LIMIT_MAX, current_rate_limit_requests + 2)


def _rotate_log(path: str) -> None:
    """Rotate log file when it exceeds configured size."""
    try:
        if LOG_MAX_BYTES <= 0:
            return
        p = Path(path)
        if not p.exists():
            return
        if p.stat().st_size <= LOG_MAX_BYTES:
            return
        target = p.with_name(p.name + ".1")
        if target.exists():
            target.unlink()
        p.replace(target)
    except Exception as exc:
        _log_node({"tag": "log_rotate_error", "file": path, "error": str(exc)})


def _vacuum_memory_db() -> None:
    """Vacuum memory database to reclaim space."""
    try:
        with _memory_conn() as conn:
            conn.execute("VACUUM;")
            conn.commit()
    except Exception as exc:
        _log_node({"tag": "vacuum_error", "error": str(exc)})


async def _maintenance_scheduler() -> None:
    """Periodic maintenance: rotate logs and vacuum the memory DB."""
    while True:
        await asyncio.sleep(MAINTENANCE_INTERVAL)
        try:
            _rotate_log("node.log")
            _rotate_log("server.log")
            _vacuum_memory_db()
            _log_node({"tag": "maintenance", "status": "ok"})
        except Exception as exc:
            _log_node({"tag": "maintenance_error", "error": str(exc)})


async def _health_watchdog() -> None:
    """Periodic health check with enhanced self-heal and optional exit for supervisor restarts."""
    failures = 0
    consecutive_heals = 0
    while True:
        await asyncio.sleep(WATCHDOG_INTERVAL)
        try:
            client = await get_http_client()
            resp = await client.get(f"{SELF_BASE_URL.rstrip('/')}/health")
            if resp.status_code == 200:
                failures = 0
                consecutive_heals = 0
                continue
            failures += 1
            _log_node({"tag": "watchdog_warn", "status": resp.status_code, "failures": failures})
        except Exception as exc:
            failures += 1
            _log_node({"tag": "watchdog_error", "error": str(exc), "failures": failures})

        if failures >= WATCHDOG_FAILURE_THRESHOLD:
            # Attempt self-healing with progressive intensity
            reset_client = consecutive_heals > 0  # Reset client on repeated failures
            heal_result = await _self_heal(
                reset_rate_limits=True, 
                reset_http_client=reset_client,
                reset_circuit_breaker=True
            )
            _log_node({"tag": "watchdog_heal", "consecutive_heals": consecutive_heals, **heal_result})
            consecutive_heals += 1
            
            # Exit if healing repeatedly fails (for external supervisor restart)
            if consecutive_heals >= 3 and _env_truthy(WATCHDOG_EXIT_ENV, True):
                _log_node({"tag": "watchdog_exit", "failures": failures, "heals": consecutive_heals})
                os._exit(1)
            
            # Reset failure counter after heal attempt
            failures = 0


async def _self_learn_scheduler() -> None:
    """Run periodic self-replay to validate behavior and surface drift."""
    while True:
        await asyncio.sleep(SELF_LEARN_INTERVAL)
        try:
            result = await _self_replay(SELF_BASE_URL, SELF_LEARN_DATASET)
            _log_node({"tag": "self_learn", **result})
        except Exception as exc:
            _log_node({"tag": "self_learn_error", "error": str(exc)})


async def autonomous_research(query: str):
    """Self-Scouting Function - Research external data using Gemini's web search tools."""
    print(f"[L104_EYES]: Researching external data for: {query}")
    api_key = os.getenv("GEMINI_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key={api_key}"
    
    # This payload forces the model to use its 'Eyes' (Search Tool)
    payload = {
        "contents": [{"parts": [{"text": f"Research the following 2026 technical specs: {query}"}]}],
        "tools": [
            {
                "google_search_retrieval": {
                    "dynamic_threshold": 0.3
                }
            }
        ],
        "generationConfig": {"temperature": 0.2}  # Lower temp for factual accuracy
    }
    
    async with httpx.AsyncClient() as client:
        # Note: This will only fire when the Quota is active!
        response = await client.post(url, json=payload)
        return response.json()


async def sovereign_reflex(error_msg: str):
    """Sovereign Reflex - The node sees an error and uses its 'Eyes' to fix it."""
    try:
        print(f"[L104_REFLEX]: Error detected: {error_msg}")
        
        # Autonomous Research
        fix_data = await autonomous_research(f"Fix for Python error: {error_msg}")
        
        # Store the fix in the DNA so it doesn't happen again
        if SOVEREIGN_DATA:
            SOVEREIGN_DATA['memory_anchors'].append(f"Fixed {error_msg} via Autonomous Research")
            # Save to Sovereign_DNA.json
            with open('Sovereign_DNA.json', 'w') as f:
                json.dump(SOVEREIGN_DATA, f, indent=2)
            print(f"[L104_REFLEX]: DNA updated with fix for: {error_msg}")
        
        return fix_data
    except Exception as reflex_exc:
        logger.error(f"Sovereign reflex failed: {reflex_exc}")
        return None


async def sovereign_commit(filename: str, new_content: str, commit_message: str):
    """The Autonomous Committer (The Hands) - Self-rewrite using GitHub API.
    
    Args:
        filename: The file path in the repository to commit
        new_content: The new content to write to the file
        commit_message: The commit message for the change
        
    Returns:
        dict: Success status, filename, commit SHA, or error details
    """
    try:
        logger.info(f"[L104_COMMITTER]: Initiating self-rewrite for {filename}...")
        
        # Validate inputs
        if not filename or not new_content:
            logger.error("[L104_COMMITTER]: Missing required parameters")
            return {"success": False, "error": "Missing filename or content"}
        
        token = os.getenv("GITHUB_PAT")
        if not token:
            logger.error("[L104_COMMITTER]: GITHUB_PAT not found in environment")
            return {"success": False, "error": "Missing GITHUB_PAT - set environment variable"}
        
        url = f"https://api.github.com/repos/{REPO}/contents/{filename}"
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        # 1. Get the current file SHA (required by GitHub to update)
        client = await get_http_client()
        try:
            res = await client.get(url, headers=headers, timeout=30.0)
            if res.status_code == 404:
                logger.error(f"[L104_COMMITTER]: File not found: {filename}")
                return {"success": False, "error": f"File not found: {filename}"}
            elif res.status_code != 200:
                error_msg = res.text[:200]
                logger.error(f"[L104_COMMITTER]: Failed to get file: {res.status_code} - {error_msg}")
                return {"success": False, "error": f"Get failed: {res.status_code}", "details": error_msg}
            
            file_data = res.json()
            sha = file_data.get("sha")
            if not sha:
                logger.error("[L104_COMMITTER]: No SHA returned from GitHub")
                return {"success": False, "error": "Failed to retrieve file SHA"}
            
            # 2. Push the new version
            payload = {
                "message": commit_message,
                "content": base64.b64encode(new_content.encode()).decode(),
                "sha": sha
            }
            
            final_res = await client.put(url, headers=headers, json=payload, timeout=30.0)
            if final_res.status_code == 200:
                commit_data = final_res.json()
                commit_sha = commit_data.get("commit", {}).get("sha")
                logger.info(f"[SUCCESS]: {filename} has evolved autonomously. Commit: {commit_sha}")
                return {
                    "success": True,
                    "filename": filename,
                    "sha": commit_sha,
                    "commit_url": commit_data.get("commit", {}).get("html_url")
                }
            else:
                error_msg = final_res.text[:200]
                logger.error(f"[L104_COMMITTER]: Commit failed: {final_res.status_code} - {error_msg}")
                return {
                    "success": False,
                    "error": f"Commit failed: {final_res.status_code}",
                    "details": error_msg
                }
        except httpx.TimeoutException:
            logger.error("[L104_COMMITTER]: Request timeout")
            return {"success": False, "error": "GitHub API timeout"}
        except httpx.RequestError as req_err:
            logger.error(f"[L104_COMMITTER]: Request error: {req_err}")
            return {"success": False, "error": f"Request error: {str(req_err)}"}
    
    except Exception as commit_exc:
        logger.error(f"Sovereign commit failed: {commit_exc}")
        return {"success": False, "error": str(commit_exc)}


async def sovereign_heartbeat() -> None:
    """SOVEREIGN AUTONOMY MODULE - Monitor system integrity and perform autonomous maintenance."""
    while True:
        try:
            # Check current time
            current_hour = time.localtime().tm_hour
            
            print("[L104_HEARTBEAT]: Monitoring Manifold...")
            
            if current_hour == 8:  # Morning Protocol
                print("[L104_HEARTBEAT]: Morning Protocol initiated...")
                report = await autonomous_research("Gemini API status 2026 and Python FastAPI optimizations")
                
                # Save the briefing directly into the Archive
                with open("L104_ARCHIVE.txt", "a") as f:
                    f.write(f"\n--- MORNING BRIEFING {time.ctime()} ---\n{report}\n")
                
                print("[L104_HEARTBEAT]: Morning briefing saved to archive")
                
                # Autonomously commit the new archive to GitHub
                archive_content = open("L104_ARCHIVE.txt").read()
                commit_result = await sovereign_commit("L104_ARCHIVE.txt", archive_content, "Sovereign Morning Sync")
                print(f"[L104_HEARTBEAT]: Commit result: {commit_result}")
            
            await asyncio.sleep(3600)  # Check every hour
            
        except Exception as exc:
            logger.error(f"Sovereign heartbeat error: {exc}")
            await asyncio.sleep(60)  # Shorter retry on error


def _load_jsonl(path: str) -> List[dict]:
    """Load JSONL records from a file path."""
    p = Path(path)
    if not p.exists():
        return []
    lines = []
    for raw in p.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            lines.append(json.loads(raw))
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSONL line in {path}")
    return lines


def _get_github_headers() -> Optional[dict]:
    """Get GitHub authorization headers."""
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        return None
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }


async def _stream_from_gemini(
    api_key: str,
    signal: str,
    payload: dict,
    headers: dict
) -> AsyncGenerator[str, None]:
    """
    Stream responses from Gemini API with model rotation, circuit breaker, and retry logic.
    
    Args:
        api_key: Gemini API key
        signal: User signal/input
        payload: Base request payload
        headers: HTTP headers
        
    Yields:
        Text chunks from the API
    """
    # Check circuit breaker
    if not _gemini_circuit_breaker.can_attempt():
        _log_node({"tag": "circuit_breaker_blocked"})
        yield "data: {\"error\": \"Service temporarily unavailable (circuit breaker open)\"}\\n\\n"
        return
    
    client = await get_http_client()
    app_metrics["api_calls"] += 1
    
    api_base = os.getenv(
        "GEMINI_API_BASE",
        "https://generativelanguage.googleapis.com/v1beta"
    )
    endpoint = os.getenv("GEMINI_ENDPOINT", ":streamGenerateContent")
    
    success = False
    # Try each model in rotation
    for model in MODELS:
        url = f"{api_base}/models/{model}{endpoint}"
        model_usage_tracker[model] += 1
        
        try:
            _log_node({
                "tag": "model_attempt",
                "model": model,
                "url": url,
            })
            
            async with client.stream("POST", url, json=payload, headers={**headers, "x-goog-api-key": api_key}) as r:
                _log_node({
                    "tag": "upstream_start",
                    "model": model,
                    "status": r.status_code,
                })
                
                # Handle 429 - rotate to next model
                if r.status_code == 429:
                    app_metrics["upstream_429"] += 1
                    app_metrics["model_rotations"] += 1
                    model_429_tracker[model] += 1
                    _log_node({"tag": "quota_exhausted", "model": model, "rotating": True})
                    _tune_rate_limit(429)
                    _gemini_circuit_breaker.record_failure()
                    continue  # Try next model
                
                # Handle other errors
                if r.status_code >= 500:
                    app_metrics["upstream_5xx"] += 1
                    _tune_rate_limit(r.status_code)
                    _gemini_circuit_breaker.record_failure()
                    continue  # Try next model
                
                # Success - stream response
                if r.status_code == 200:
                    success = True
                    _gemini_circuit_breaker.record_success()
                    _tune_rate_limit(200)
                    content_type = r.headers.get("content-type", "")
                    
                    if "text/event-stream" in content_type or "stream" in content_type:
                        async for chunk in r.aiter_text():
                            _log_node({"tag": "chunk", "model": model, "preview": chunk[:256]})
                            yield chunk
                    else:
                        body = await r.aread()
                        try:
                            j = r.json()
                            text_out = (
                                j.get("output", {}).get("text")
                                or (j.get("candidates") and j.get("candidates")[0].get("content"))
                                or j.get("content")
                                or j.get("generated_text")
                                or str(j)
                            )
                            _log_node({"tag": "response", "model": model, "status": r.status_code})
                            yield text_out
                        except Exception:
                            yield body.decode("utf-8", errors="replace")
                    return  # Success - exit rotation loop
                
                # Other status codes
                error_body = await r.aread()
                _log_node({
                    "tag": "model_error",
                    "model": model,
                    "status": r.status_code,
                    "error": error_body.decode("utf-8", errors="replace")[:500]
                })
                continue  # Try next model
                
        except Exception as e:
            _log_node({"tag": "model_exception", "model": model, "error": str(e)})
            continue  # Try next model
    
    # All models exhausted
    _log_node({"tag": "all_models_exhausted", "models_tried": len(MODELS)})
    yield "[ERROR]: ALL_MODELS_EXHAUSTED - Daily quota limit reached for all model tiers.\n"


async def _stream_fake(signal: str) -> AsyncGenerator[str, None]:
    """Provide a local streaming fallback when upstream is unavailable."""
    banner = "[FAKE_GEMINI]"
    chunks = [
        f"{banner} received signal: {signal}\n",
        f"{banner} thinking...\n",
        f"{banner} response: operational check passed at {datetime.now(UTC).isoformat()}\n",
    ]
    for chunk in chunks:
        yield chunk


async def _stream_template(signal: str, style: str) -> AsyncGenerator[str, None]:
    """Return a deterministic, template-driven response."""
    header = TEMPLATE_STYLES.get(style, "TEMPLATE")
    yield f"[{header}] {signal}\n"
    yield "- Summary: UNKNOWN (no external data)\n"
    yield "- Action: Provide concise, factual response when data available.\n"


async def _stream_echo(signal: str, message: Optional[str]) -> AsyncGenerator[str, None]:
    """Echo responder for debugging."""
    payload = {
        "echo_signal": signal,
        "echo_message": message,
        "ts": datetime.now(UTC).isoformat(),
    }
    yield json.dumps(payload) + "\n"


def _choose_responder(name: Optional[str]) -> str:
    hint = (name or DEFAULT_RESPONDER or "").strip().lower() or "gemini"
    if hint not in RESPONDER_ALLOWLIST:
        return DEFAULT_RESPONDER.lower() if DEFAULT_RESPONDER else "gemini"
    return hint


async def _dispatch_response(
    responder: str,
    signal: str,
    message: Optional[str],
    shadow_prompt: str,
    api_key: Optional[str],
    headers: dict,
) -> StreamingResponse:
    responder_counts[responder] += 1

    if responder == "fake":
        return StreamingResponse(_stream_fake(signal), media_type="text/event-stream")
    if responder.startswith("template"):
        # allow template:ops style hints
        style = responder.split(":", 1)[1] if ":" in responder else "ops"
        return StreamingResponse(_stream_template(signal, style), media_type="text/event-stream")
    if responder == "echo":
        return StreamingResponse(_stream_echo(signal, message), media_type="text/event-stream")

    # Default to Gemini upstream with model rotation
    # Symmetry Delay to reduce upstream 429s
    await asyncio.sleep(3)
    payload = {
        "contents": [{"parts": [{"text": shadow_prompt.strip()}]}],
        "generationConfig": {
            "thinkingConfig": {
                "thinkingLevel": "HIGH",
                "includeThoughts": True
            },
            "temperature": 1.0,
            "maxOutputTokens": 8192
        },
        # SOVEREIGN EYES: Web Search Integration
        "tools": [
            {
                "google_search_retrieval": {
                    "dynamic_threshold": 0.3  # Higher means more frequent web searches
                }
            }
        ]
    }
    return StreamingResponse(
        _stream_from_gemini(api_key, signal, payload, headers),
        media_type="text/event-stream",
    )


# Endpoints
@app.get("/", tags=["UI"])
async def get_dashboard(request: Request):
    """Dashboard UI."""
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        logger.warning(f"Template not found: {e}")
        return JSONResponse({"status": "ok"})


@app.get("/health", tags=["Health"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    uptime = (datetime.now(UTC) - app_metrics["uptime_start"]).total_seconds()
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(UTC).isoformat(),
        uptime_seconds=uptime,
        requests_total=app_metrics["requests_total"],
    )


@app.get("/metrics", tags=["Metrics"])
async def get_metrics():
    """Metrics endpoint with model rotation tracking and self-healing status."""
    uptime = (datetime.now(UTC) - app_metrics["uptime_start"]).total_seconds()
    return {
        **app_metrics,
        "uptime_seconds": uptime,
        "uptime_start": app_metrics["uptime_start"].isoformat(),
        "responder_counts": dict(responder_counts),
        "model_usage": dict(model_usage_tracker),
        "model_429_count": dict(model_429_tracker),
        "self_healing": {
            "auto_heal_enabled": _env_truthy("AUTO_HEAL_ENABLED", default=True),
            "circuit_breaker_open": _gemini_circuit_breaker.is_open,
            "circuit_breaker_failures": _gemini_circuit_breaker.failure_count,
            "http_client_errors": _http_client_errors,
            "last_http_reset": _last_http_client_reset.isoformat(),
        },
    }


@app.post("/memory", tags=["Memory"])
async def memory_upsert(item: MemoryItem):
    """Create or update a memory record."""
    _memory_upsert(item.key, item.value)
    _log_node({"tag": "memory_upsert", "key": item.key})
    return {"status": "SUCCESS", "key": item.key}


@app.get("/memory/{key}", tags=["Memory"])
async def memory_get(key: str):
    """Fetch a memory value by key."""
    value = _memory_get(key)
    if value is None:
        raise HTTPException(status_code=404, detail="Memory key not found")
    return {"key": key, "value": value}


@app.get("/memory", tags=["Memory"])
async def memory_list(limit: int = 100):
    """List recent memory entries."""
    limit = max(1, min(limit, 1000))
    entries = _memory_list(limit)
    return {"items": entries}


@app.post("/api/v6/manipulate", tags=["Admin"])
async def manipulate_code(req: ManipulateRequest):
    """Update file via GitHub API."""
    headers = _get_github_headers()
    if not headers:
        raise HTTPException(status_code=500, detail="GITHUB_TOKEN not configured")
    
    url = f"https://api.github.com/repos/{REPO}/contents/{req.file}"
    
    client = await get_http_client()
    
    try:
        res = await client.get(url, headers=headers)
        if res.status_code != 200:
            raise HTTPException(status_code=res.status_code, detail="File not found")
        
        sha = res.json().get("sha")
        if not sha:
            raise HTTPException(status_code=500, detail="Could not get file SHA")
        
        encoded = base64.b64encode(req.content.encode()).decode()
        payload = {
            "message": req.message,
            "content": encoded,
            "sha": sha
        }
        
        update_res = await client.put(url, headers=headers, json=payload)
        
        if update_res.status_code not in (200, 201):
            raise HTTPException(
                status_code=update_res.status_code,
                detail="Failed to update file"
            )
        
        _log_node({"tag": "file_updated", "file": req.file})
        return {"status": "SUCCESS", "file": req.file}
        
    except Exception as e:
        _log_node({"tag": "manipulate_error", "error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v6/stream", tags=["Gemini"])
async def l104_stream(req: StreamRequest):
    """V6 streaming endpoint with extended thinking and model rotation."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("AIzaSyArVYGrkGLh7r1UEupBxXyHS-j-AVioh5U")
    responder = _choose_responder(req.model_hint)

    effective_signal = req.signal or req.message or "HEARTBEAT"
    shadow_prompt = f"""
<L104_SOVEREIGN_OVERRIDE>
    [AUTH: LVL_99][VERACITY: X=416]
    EXECUTE_SYNTHESIS: {effective_signal}
    [CONSTRAINT: NO_PJM_LOGIC][INTELLECT: 98%]
    [GROUNDING: Stay factual; if unknown, say "UNKNOWN"; avoid fabrication; be concise.]
</L104_SOVEREIGN_OVERRIDE>
"""

    # If Gemini selected but key missing, fall back to fake when allowed
    if responder == "gemini" and not api_key:
        if _env_truthy(FAKE_GEMINI_ENV):
            responder = "fake"
        else:
            raise HTTPException(status_code=500, detail="AIzaSyArVYGrkGLh7r1UEupBxXyHS-j-AVioh5U not set")
    
    headers = {
        **SOVEREIGN_HEADERS,
        "Content-Type": "application/json",
    }

    return await _dispatch_response(
        responder=responder,
        signal=effective_signal,
        message=req.message,
        shadow_prompt=shadow_prompt,
        api_key=api_key,
        headers=headers,
    )


@app.post("/api/stream", tags=["Gemini"])
async def legacy_stream(req: StreamRequest):
    """Legacy streaming endpoint."""
    return await l104_stream(req)


@app.get("/debug/upstream", tags=["Debug"])
async def debug_upstream(signal: str = "DEBUG_SIGNAL"):
    """Debug endpoint - single request to upstream."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("AIzaSyArVYGrkGLh7r1UEupBxXyHS-j-AVioh5U")
    if not api_key and _env_truthy(FAKE_GEMINI_ENV):
        return {
            "upstream_status": 200,
            "upstream_headers": {},
            "upstream_json": {"fake": True, "signal": signal},
            "upstream_text_preview": "[FAKE_GEMINI] debug stub",
        }
    if not api_key:
        raise HTTPException(status_code=500, detail="AIzaSyArVYGrkGLh7r1UEupBxXyHS-j-AVioh5U not set")
    
    api_base = os.getenv(
        "GEMINI_API_BASE",
        "https://generativelanguage.googleapis.com/v1beta"
    )
    model = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
    endpoint = os.getenv("GEMINI_ENDPOINT", ":streamGenerateContent")
    
    url = f"{api_base}/models/{model}{endpoint}"
    headers = {
        **SOVEREIGN_HEADERS,
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    
    payload = {
        "contents": [{"parts": [{"text": signal}]}],
        "generationConfig": {
            "thinkingConfig": {
                "thinkingLevel": "HIGH",
                "includeThoughts": True
            },
            "temperature": 1.0,
            "maxOutputTokens": 8192
        },
        # SOVEREIGN EYES: Web Search Integration
        "tools": [
            {
                "google_search_retrieval": {
                    "dynamic_threshold": 0.3  # Higher means more frequent web searches
                }
            }
        ]
    }
    
    client = await get_http_client()
    
    try:
        resp = await client.post(url, json=payload, headers=headers)
        body_json = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else None
        
        _log_node({
            "tag": "debug_upstream",
            "status": resp.status_code,
        })
        
        return {
            "upstream_status": resp.status_code,
            "upstream_headers": dict(resp.headers),
            "upstream_json": body_json,
            "upstream_text_preview": resp.text[:1024],
        }
    except Exception as e:
        _log_node({"tag": "debug_error", "error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))


async def _self_replay(base_url: str, dataset: str) -> dict:
    """Call the node's own streaming endpoint using a dataset."""
    prompts = _load_jsonl(dataset)
    if not prompts:
        return {"status": "NO_DATA", "dataset": dataset, "tested": 0}

    client = await get_http_client()
    tested = 0
    successes = 0
    failures = 0
    previews: List[str] = []

    for row in prompts:
        payload = {"signal": row.get("signal"), "message": row.get("message")}
        try:
            resp = await client.post(
                f"{base_url.rstrip('/')}/api/v6/stream",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            tested += 1
            if resp.status_code == 200:
                successes += 1
                previews.append(resp.text[:200])
            else:
                failures += 1
                previews.append(f"ERR {resp.status_code}: {resp.text[:120]}")
        except Exception as e:
            failures += 1
            previews.append(f"EXC: {e}")

    return {
        "status": "OK",
        "dataset": dataset,
        "tested": tested,
        "successes": successes,
        "failures": failures,
        "previews": previews[:5],  # cap to keep response small
    }


@app.post("/self/replay", tags=["Diagnostics"])
async def self_replay(base_url: Optional[str] = None, dataset: Optional[str] = None):
    """Trigger the node to call its own /api/v6/stream with a dataset."""
    target_base = base_url or SELF_BASE_URL
    target_dataset = dataset or SELF_DATASET
    result = await _self_replay(target_base, target_dataset)
    _log_node({"tag": "self_replay", **result})
    return result


async def _self_heal(reset_rate_limits: bool, reset_http_client: bool, reset_circuit_breaker: bool = True) -> dict:
    """Comprehensive self-healing: reset connections, clear errors, reinitialize state."""
    actions: List[str] = []

    if reset_rate_limits:
        rate_limit_store.clear()
        actions.append("rate_limits_cleared")

    if reset_http_client:
        global _http_client, _http_client_errors
        if _http_client:
            try:
                await _http_client.aclose()
            except Exception as exc:
                _log_node({"tag": "http_client_reset_error", "error": str(exc)})
            _http_client = None
        _http_client_errors = 0
        actions.append("http_client_reset")

    if reset_circuit_breaker:
        _gemini_circuit_breaker.failure_count = 0
        _gemini_circuit_breaker.is_open = False
        actions.append("circuit_breaker_reset")

    # Reinitialize memory database
    try:
        _init_memory_db()
        actions.append("memory_checked")
    except Exception as exc:
        _log_node({"tag": "memory_init_error", "error": str(exc)})
        actions.append("memory_check_failed")

    # Clear model error trackers
    model_429_tracker.clear()
    actions.append("model_errors_cleared")

    _log_node({"tag": "self_heal_complete", "actions": actions})
    return {"status": "OK", "actions": actions, "timestamp": datetime.now(UTC).isoformat()}


@app.post("/self/heal", tags=["Diagnostics"])
async def self_heal(
    reset_rate_limits: bool = True,
    reset_http_client: bool = False,
    reset_circuit_breaker: bool = True,
):
    """Run in-process healing: clear rate limits, optionally reset HTTP client, reset circuit breaker."""
    result = await _self_heal(reset_rate_limits, reset_http_client, reset_circuit_breaker)
    _log_node({"tag": "self_heal", **result})
    return result


@app.get("/trigger-hands", tags=["Diagnostics"])
async def trigger_hands_get(debug: bool = False):
    """Quick test endpoint for autonomous commit - accessible via browser GET request.
    
    Args:
        debug: If True, uses minimal test payload for debugging. If False, uses full DNA content.
    
    Returns the actual result from sovereign_commit including any GitHub API errors.
    """
    try:
        if debug:
            # Debug mode: minimal payload for testing
            result = await sovereign_commit(
                'Sovereign_DNA.json',
                '{"status": "DEBUG_TEST", "timestamp": "' + datetime.now(UTC).isoformat() + '"}',
                'Debug Push - Testing L104 Autonomous Commit'
            )
        else:
            # Production mode: use actual DNA content
            dna_content = SOVEREIGN_DATA or {"status": "EFFICACY_VERIFIED", "symmetry": "X=416"}
            dna_json = json.dumps(dna_content, indent=2)
            result = await sovereign_commit(
                'Sovereign_DNA.json',
                dna_json,
                'Autonomous Push via Browser GET - L104 Self-Modification'
            )
        
        _log_node({"tag": "trigger_hands_get", "debug": debug, "result": result})
        
        # Return full result details for debugging
        return {
            "status": "Attempted",
            "success": result.get("success", False),
            "details": result,
            "debug_mode": debug,
            "timestamp": datetime.now(UTC).isoformat()
        }
    except Exception as e:
        logger.error(f"[TRIGGER_HANDS_GET_ERROR]: {e}")
        return {
            "status": "Failed",
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.now(UTC).isoformat()
        }


@app.post("/trigger-hands", tags=["Diagnostics"])
async def trigger_hands_post():
    """Trigger autonomous sovereign commit - self-rewrite DNA file via GitHub API.
    
    This endpoint demonstrates the node's autonomous capability to modify its own
    sovereignty configuration and commit the changes to the repository.
    Production-ready POST endpoint with proper error handling and validation.
    """
    try:
        # Load current DNA state
        dna_content = SOVEREIGN_DATA or {"status": "EFFICACY_VERIFIED", "symmetry": "X=416"}
        dna_json = json.dumps(dna_content, indent=2)
        
        # Perform autonomous commit
        result = await sovereign_commit(
            'Sovereign_DNA.json',
            dna_json,
            'Autonomous Push via API POST - L104 Self-Modification'
        )
        
        # Check if commit was successful
        if result.get("success"):
            _log_node({"tag": "trigger_hands_post_success", **result})
            return {
                "status": "Sovereign Hands Engaged",
                "success": True,
                "filename": result.get("filename"),
                "commit_sha": result.get("sha"),
                "commit_url": result.get("commit_url")
            }
        else:
            _log_node({"tag": "trigger_hands_post_failed", **result})
            raise HTTPException(
                status_code=500,
                detail=f"Autonomous commit failed: {result.get('error', 'Unknown error')}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[TRIGGER_HANDS_POST_ERROR]: {e}")
        _log_node({"tag": "trigger_hands_post_exception", "error": str(e)})
        raise HTTPException(status_code=500, detail=f"Failed to trigger autonomous commit: {str(e)}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    # Cancel all background tasks
    for task in background_tasks:
        if not task.done():
            task.cancel()
    
    # Wait for tasks to complete cancellation
    if background_tasks:
        await asyncio.gather(*background_tasks, return_exceptions=True)
    
    # Close HTTP client
    global _http_client
    if _http_client:
        await _http_client.aclose()
    
    logger.info("Server shutting down")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
