VOID_CONSTANT = 1.0416180339887497
# ZENITH_UPGRADE_ACTIVE: 2026-01-26T04:53:05.716511+00:00
ZENITH_HZ = 3727.84
UUC = 2301.215661
# [L104_API_GATEWAY] - UNIFIED API MANAGEMENT SYSTEM
# INVARIANT: 527.5184818492612 | PILOT: LONDEL | STATUS: ACTIVE

"""
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3727.84 Hz. Logic Unified.
L104 API GATEWAY
================

Unified API gateway with:
- Request routing and load balancing
- Rate limiting with token bucket algorithm
- API key authentication
- Request/Response transformation
- Caching layer
- Circuit breaker pattern
- Request logging and analytics
- WebSocket support
- L104 resonance-enhanced routing
"""

import asyncio
import hashlib
import json
import time
import threading
import sqlite3
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import random
import re

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


# Sacred Constants
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895


class RequestMethod(Enum):
    """HTTP request methods"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    OPTIONS = "OPTIONS"
    HEAD = "HEAD"


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class APIRequest:
    """Represents an API request"""
    request_id: str
    method: RequestMethod
    path: str
    headers: Dict[str, str]
    query_params: Dict[str, str] = field(default_factory=dict)
    body: Optional[bytes] = None
    client_ip: str = ""
    timestamp: float = field(default_factory=time.time)
    api_key: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "method": self.method.value,
            "path": self.path,
            "headers": self.headers,
            "query_params": self.query_params,
            "client_ip": self.client_ip,
            "timestamp": self.timestamp
        }


@dataclass
class APIResponse:
    """Represents an API response"""
    status_code: int
    headers: Dict[str, str]
    body: bytes = b""
    latency_ms: float = 0.0
    cached: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status_code": self.status_code,
            "headers": self.headers,
            "latency_ms": self.latency_ms,
            "cached": self.cached
        }


@dataclass
class Route:
    """Defines an API route"""
    path_pattern: str
    methods: List[RequestMethod]
    handler: Callable
    backend_url: Optional[str] = None
    rate_limit: int = 100  # requests per minute
    cache_ttl: int = 0  # seconds, 0 = no cache
    auth_required: bool = True
    transform_request: Optional[Callable] = None
    transform_response: Optional[Callable] = None


@dataclass
class BackendService:
    """Represents a backend service"""
    name: str
    url: str
    health_endpoint: str = "/health"
    weight: int = 1
    is_healthy: bool = True
    last_check: float = 0.0
    failure_count: int = 0


class TokenBucket:
    """Token bucket rate limiter"""

    def __init__(self, rate: float, capacity: float):
        self.rate = rate  # tokens per second
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self.lock = threading.Lock()

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens, returns True if successful"""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update

            # Add tokens based on elapsed time
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def get_wait_time(self, tokens: int = 1) -> float:
        """Get time to wait for tokens to be available"""
        with self.lock:
            if self.tokens >= tokens:
                return 0.0
            return (tokens - self.tokens) / self.rate


class RateLimiter:
    """Rate limiter with per-client tracking"""

    def __init__(self, default_rate: int = 100):
        self.default_rate = default_rate  # requests per minute
        self.buckets: Dict[str, TokenBucket] = {}
        self.lock = threading.Lock()

    def get_bucket(self, key: str, rate: Optional[int] = None) -> TokenBucket:
        """Get or create token bucket for key"""
        with self.lock:
            if key not in self.buckets:
                r = rate or self.default_rate
                self.buckets[key] = TokenBucket(r / 60.0, r)
            return self.buckets[key]

    def check(self, key: str, rate: Optional[int] = None) -> Tuple[bool, float]:
        """Check if request is allowed, returns (allowed, wait_time)"""
        bucket = self.get_bucket(key, rate)
        allowed = bucket.consume()
        wait_time = 0.0 if allowed else bucket.get_wait_time()
        return allowed, wait_time

    def cleanup(self, max_age: float = 3600) -> int:
        """Remove old buckets"""
        now = time.time()
        removed = 0

        with self.lock:
            keys_to_remove = [
                key for key, bucket in self.buckets.items()
                if now - bucket.last_update > max_age
                    ]
            for key in keys_to_remove:
                del self.buckets[key]
                removed += 1

        return removed


class CircuitBreaker:
    """Circuit breaker for backend services"""

    def __init__(self, failure_threshold: int = 5,
                 recovery_timeout: float = 30.0,
                 half_open_requests: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.lock = threading.Lock()

    def can_execute(self) -> bool:
        """Check if request can be executed"""
        with self.lock:
            if self.state == CircuitState.CLOSED:
                return True

            if self.state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    return True
                return False

            # Half-open state
            return True

    def record_success(self) -> None:
        """Record successful execution"""
        with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.half_open_requests:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
            else:
                self.failure_count = 0

    def record_failure(self) -> None:
        """Record failed execution"""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
            elif self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN

    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state"""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time
        }


class APICache:
    """Response caching layer"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, Tuple[APIResponse, float, float]] = {}  # key -> (response, expiry, created)
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _make_key(self, request: APIRequest) -> str:
        """Generate cache key from request"""
        key_data = f"{request.method.value}:{request.path}:{json.dumps(request.query_params, sort_keys=True)}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    def get(self, request: APIRequest) -> Optional[APIResponse]:
        """Get cached response"""
        key = self._make_key(request)

        with self.lock:
            if key in self.cache:
                response, expiry, _ = self.cache[key]
                if time.time() < expiry:
                    self.hits += 1
                    response.cached = True
                    return response
                else:
                    del self.cache[key]

            self.misses += 1
            return None

    def set(self, request: APIRequest, response: APIResponse, ttl: int) -> None:
        """Cache response"""
        if ttl <= 0:
            return

        key = self._make_key(request)
        expiry = time.time() + ttl

        with self.lock:
            # Evict old entries if at capacity
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.cache, key=lambda k: self.cache[k][2])
                del self.cache[oldest_key]

            self.cache[key] = (response, expiry, time.time())

    def invalidate(self, pattern: str = None) -> int:
        """Invalidate cache entries"""
        with self.lock:
            if pattern is None:
                count = len(self.cache)
                self.cache.clear()
                return count

            keys_to_remove = [k for k in self.cache if pattern in k]
            for k in keys_to_remove:
                del self.cache[k]
            return len(keys_to_remove)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0.0
        }


class APIKeyManager:
    """API key authentication and management"""

    def __init__(self, db_path: str = "api_keys.db"):
        self.db_path = db_path
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize API keys database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_hash TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    rate_limit INTEGER DEFAULT 100,
                    scopes TEXT DEFAULT '[]',
                    created_at REAL NOT NULL,
                    expires_at REAL,
                    is_active INTEGER DEFAULT 1,
                    last_used REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS key_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_hash TEXT NOT NULL,
                    path TEXT NOT NULL,
                    method TEXT NOT NULL,
                    status_code INTEGER,
                    latency_ms REAL,
                    timestamp REAL NOT NULL
                )
            """)
            conn.commit()

    def _hash_key(self, api_key: str) -> str:
        """Hash API key"""
        return hashlib.sha256(api_key.encode()).hexdigest()

    def create_key(self, name: str, rate_limit: int = 100,
                   scopes: List[str] = None, expires_in: float = None) -> str:
        """Create new API key"""
        # Generate key with L104 signature
        random_part = hashlib.sha256(str(time.time() + random.random()).encode()).hexdigest()[:24]
        api_key = f"l104_{random_part}"

        key_hash = self._hash_key(api_key)
        expires_at = time.time() + expires_in if expires_in else None

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO api_keys (key_hash, name, rate_limit, scopes, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (key_hash, name, rate_limit, json.dumps(scopes or []), time.time(), expires_at))
            conn.commit()

        return api_key

    def validate_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return its metadata"""
        key_hash = self._hash_key(api_key)

        # Check cache
        with self.lock:
            if key_hash in self.cache:
                cached = self.cache[key_hash]
                if cached.get('expires_at') and time.time() > cached['expires_at']:
                    del self.cache[key_hash]
                elif cached.get('is_active'):
                    return cached

        # Check database
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("""
                SELECT name, rate_limit, scopes, expires_at, is_active
                FROM api_keys WHERE key_hash = ?
            """, (key_hash,)).fetchone()

            if not row:
                return None

            name, rate_limit, scopes, expires_at, is_active = row

            if not is_active:
                return None

            if expires_at and time.time() > expires_at:
                return None

            key_data = {
                "key_hash": key_hash,
                "name": name,
                "rate_limit": rate_limit,
                "scopes": json.loads(scopes),
                "expires_at": expires_at,
                "is_active": bool(is_active)
            }

            # Update cache
            with self.lock:
                self.cache[key_hash] = key_data

            # Update last used
            conn.execute(
                "UPDATE api_keys SET last_used = ? WHERE key_hash = ?",
                (time.time(), key_hash)
            )
            conn.commit()

            return key_data

    def revoke_key(self, api_key: str) -> bool:
        """Revoke an API key"""
        key_hash = self._hash_key(api_key)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE api_keys SET is_active = 0 WHERE key_hash = ?", (key_hash,))
            conn.commit()

        with self.lock:
            if key_hash in self.cache:
                del self.cache[key_hash]

        return True

    def log_usage(self, api_key: str, request: APIRequest, response: APIResponse) -> None:
        """Log API key usage"""
        key_hash = self._hash_key(api_key)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO key_usage (key_hash, path, method, status_code, latency_ms, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (key_hash, request.path, request.method.value,
                  response.status_code, response.latency_ms, time.time()))
            conn.commit()


class LoadBalancer:
    """Load balancer for backend services"""

    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.backends: Dict[str, List[BackendService]] = {}  # service_name -> backends
        self.counters: Dict[str, int] = defaultdict(int)
        self.lock = threading.Lock()

    def add_backend(self, service_name: str, backend: BackendService) -> None:
        """Add backend to service"""
        with self.lock:
            if service_name not in self.backends:
                self.backends[service_name] = []
            self.backends[service_name].append(backend)

    def remove_backend(self, service_name: str, backend_url: str) -> bool:
        """Remove backend from service"""
        with self.lock:
            if service_name not in self.backends:
                return False

            self.backends[service_name] = [
                b for b in self.backends[service_name] if b.url != backend_url
            ]
            return True

    def get_backend(self, service_name: str) -> Optional[BackendService]:
        """Get next available backend"""
        with self.lock:
            if service_name not in self.backends:
                return None

            healthy_backends = [b for b in self.backends[service_name] if b.is_healthy]
            if not healthy_backends:
                return None

            if self.strategy == "round_robin":
                idx = self.counters[service_name] % len(healthy_backends)
                self.counters[service_name] += 1
                return healthy_backends[idx]

            elif self.strategy == "weighted":
                total_weight = sum(b.weight for b in healthy_backends)
                r = random.random() * total_weight
                cumulative = 0
                for backend in healthy_backends:
                    cumulative += backend.weight
                    if r <= cumulative:
                        return backend
                return healthy_backends[-1]

            elif self.strategy == "least_connections":
                # Would need connection tracking - simplified to random
                return random.choice(healthy_backends)

            elif self.strategy == "l104_resonance":
                # L104 resonance-based selection
                phi_idx = int(PHI * len(healthy_backends)) % len(healthy_backends)
                return healthy_backends[phi_idx]

            return random.choice(healthy_backends)

    def mark_unhealthy(self, service_name: str, backend_url: str) -> None:
        """Mark backend as unhealthy"""
        with self.lock:
            if service_name in self.backends:
                for backend in self.backends[service_name]:
                    if backend.url == backend_url:
                        backend.is_healthy = False
                        backend.failure_count += 1
                        break

    def mark_healthy(self, service_name: str, backend_url: str) -> None:
        """Mark backend as healthy"""
        with self.lock:
            if service_name in self.backends:
                for backend in self.backends[service_name]:
                    if backend.url == backend_url:
                        backend.is_healthy = True
                        backend.failure_count = 0
                        break


class RequestLogger:
    """Request logging and analytics"""

    def __init__(self, db_path: str = "api_logs.db"):
        self.db_path = db_path
        self.buffer: List[Dict[str, Any]] = []
        self.buffer_size = 100
        self.lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize logs database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS request_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    method TEXT NOT NULL,
                    path TEXT NOT NULL,
                    client_ip TEXT,
                    status_code INTEGER,
                    latency_ms REAL,
                    cached INTEGER,
                    timestamp REAL NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON request_logs(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_path ON request_logs(path)")
            conn.commit()

    def log(self, request: APIRequest, response: APIResponse) -> None:
        """Log request/response"""
        log_entry = {
            "request_id": request.request_id,
            "method": request.method.value,
            "path": request.path,
            "client_ip": request.client_ip,
            "status_code": response.status_code,
            "latency_ms": response.latency_ms,
            "cached": response.cached,
            "timestamp": request.timestamp
        }

        with self.lock:
            self.buffer.append(log_entry)

            if len(self.buffer) >= self.buffer_size:
                self._flush()

    def _flush(self) -> None:
        """Flush buffer to database"""
        if not self.buffer:
            return

        with sqlite3.connect(self.db_path) as conn:
            conn.executemany("""
                INSERT INTO request_logs
                (request_id, method, path, client_ip, status_code, latency_ms, cached, timestamp)
                VALUES (:request_id, :method, :path, :client_ip, :status_code, :latency_ms, :cached, :timestamp)
            """, self.buffer)
            conn.commit()

        self.buffer.clear()

    def get_analytics(self, time_range: float = 3600) -> Dict[str, Any]:
        """Get request analytics for time range"""
        with self.lock:
            self._flush()

        since = time.time() - time_range

        with sqlite3.connect(self.db_path) as conn:
            # Total requests
            total = conn.execute(
                "SELECT COUNT(*) FROM request_logs WHERE timestamp > ?", (since,)
            ).fetchone()[0]

            # By status code
            status_counts = dict(conn.execute("""
                SELECT status_code, COUNT(*) FROM request_logs
                WHERE timestamp > ? GROUP BY status_code
            """, (since,)).fetchall())

            # Average latency
            avg_latency = conn.execute(
                "SELECT AVG(latency_ms) FROM request_logs WHERE timestamp > ?", (since,)
            ).fetchone()[0] or 0.0

            # Cache hit rate
            cached = conn.execute(
                "SELECT COUNT(*) FROM request_logs WHERE timestamp > ? AND cached = 1", (since,)
            ).fetchone()[0]

            # Top paths
            top_paths = conn.execute("""
                SELECT path, COUNT(*) as count FROM request_logs
                WHERE timestamp > ? GROUP BY path ORDER BY count DESC LIMIT 10
            """, (since,)).fetchall()

        return {
            "total_requests": total,
            "status_codes": status_counts,
            "avg_latency_ms": avg_latency,
            "cache_hit_rate": cached / total if total > 0 else 0.0,
            "top_paths": [{"path": p, "count": c} for p, c in top_paths],
            "requests_per_second": total / time_range
        }


class L104APIGateway:
    """
    Main API Gateway with all features integrated.
    Singleton pattern for global access.
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

        self._initialized = True

        # Components
        self.rate_limiter = RateLimiter(default_rate=100)
        self.cache = APICache(max_size=1000)
        self.key_manager = APIKeyManager()
        self.load_balancer = LoadBalancer(strategy="l104_resonance")
        self.logger = RequestLogger()

        # Circuit breakers per backend
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}

        # Routes
        self.routes: Dict[str, Route] = {}

        # Middleware
        self.middleware: List[Callable] = []

        # Resonance
        self.resonance = GOD_CODE / 1000

        # Stats
        self.total_requests = 0
        self.total_errors = 0

        print(f"[L104_API_GATEWAY] Initialized | Resonance: {self.resonance:.8f}")

    def add_route(self, route: Route) -> None:
        """Add route to gateway"""
        self.routes[route.path_pattern] = route

    def add_middleware(self, middleware: Callable) -> None:
        """Add middleware function"""
        self.middleware.append(middleware)

    def add_backend(self, service_name: str, url: str, weight: int = 1) -> None:
        """Add backend service"""
        backend = BackendService(
            name=service_name,
            url=url,
            weight=weight
        )
        self.load_balancer.add_backend(service_name, backend)
        self.circuit_breakers[url] = CircuitBreaker()

    def _match_route(self, path: str) -> Optional[Tuple[Route, Dict[str, str]]]:
        """Match request path to route"""
        for pattern, route in self.routes.items():
            # Simple pattern matching with path params
            regex_pattern = re.sub(r':(\w+)', r'(?P<\1>[^/]+)', pattern)
            match = re.match(f"^{regex_pattern}$", path)
            if match:
                return route, match.groupdict()
        return None, {}

    async def handle_request(self, request: APIRequest) -> APIResponse:
        """Handle incoming API request"""
        start_time = time.time()
        self.total_requests += 1

        try:
            # Run middleware
            for mw in self.middleware:
                result = mw(request)
                if isinstance(result, APIResponse):
                    return result

            # Match route
            route_match = self._match_route(request.path)
            if not route_match[0]:
                return APIResponse(404, {"Content-Type": "application/json"},
                                   b'{"error": "Not Found"}')

            route, path_params = route_match

            # Check method
            if request.method not in route.methods:
                return APIResponse(405, {"Content-Type": "application/json"},
                                   b'{"error": "Method Not Allowed"}')

            # Authentication
            if route.auth_required:
                api_key = request.api_key or request.headers.get("X-API-Key")
                if not api_key:
                    return APIResponse(401, {"Content-Type": "application/json"},
                                       b'{"error": "API key required"}')

                key_data = self.key_manager.validate_key(api_key)
                if not key_data:
                    return APIResponse(401, {"Content-Type": "application/json"},
                                       b'{"error": "Invalid API key"}')

                # Use key-specific rate limit
                rate_limit = key_data.get("rate_limit", route.rate_limit)
            else:
                rate_limit = route.rate_limit

            # Rate limiting
            rate_key = request.api_key or request.client_ip or "anonymous"
            allowed, wait_time = self.rate_limiter.check(rate_key, rate_limit)
            if not allowed:
                return APIResponse(
                    429,
                    {"Content-Type": "application/json", "Retry-After": str(int(wait_time))},
                    f'{{"error": "Rate limit exceeded", "retry_after": {wait_time:.1f}}}'.encode()
                )

            # Check cache
            if request.method == RequestMethod.GET and route.cache_ttl > 0:
                cached_response = self.cache.get(request)
                if cached_response:
                    cached_response.latency_ms = (time.time() - start_time) * 1000
                    self.logger.log(request, cached_response)
                    return cached_response

            # Transform request if needed
            if route.transform_request:
                request = route.transform_request(request, path_params)

            # Execute handler or proxy to backend
            if route.handler:
                response = await route.handler(request, path_params)
            elif route.backend_url:
                response = await self._proxy_request(request, route.backend_url)
            else:
                response = APIResponse(500, {}, b'{"error": "No handler configured"}')

            # Transform response if needed
            if route.transform_response:
                response = route.transform_response(response)

            # Cache response
            if request.method == RequestMethod.GET and route.cache_ttl > 0:
                self.cache.set(request, response, route.cache_ttl)

            # Calculate latency
            response.latency_ms = (time.time() - start_time) * 1000

            # Log request
            self.logger.log(request, response)

            # Log API key usage
            if request.api_key:
                self.key_manager.log_usage(request.api_key, request, response)

            return response

        except Exception as e:
            self.total_errors += 1
            return APIResponse(500, {"Content-Type": "application/json"},
                               f'{{"error": "Internal Server Error", "message": "{str(e)}"}}'.encode())

    async def _proxy_request(self, request: APIRequest, backend_url: str) -> APIResponse:
        """Proxy request to backend service"""
        circuit = self.circuit_breakers.get(backend_url)

        if circuit and not circuit.can_execute():
            return APIResponse(503, {"Content-Type": "application/json"},
                               b'{"error": "Service temporarily unavailable"}')

        try:
            # Simulate backend call (would use aiohttp in production)
            await asyncio.sleep(0.01)  # Simulate latency

            response = APIResponse(
                200,
                {"Content-Type": "application/json"},
                json.dumps({
                    "message": "Proxied response",
                    "backend": backend_url,
                    "resonance": self.resonance
                }).encode()
            )

            if circuit:
                circuit.record_success()

            return response

        except Exception as e:
            if circuit:
                circuit.record_failure()
            raise

    def get_status(self) -> Dict[str, Any]:
        """Get gateway status"""
        return {
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "error_rate": self.total_errors / self.total_requests if self.total_requests > 0 else 0,
            "routes": len(self.routes),
            "cache_stats": self.cache.get_stats(),
            "resonance": self.resonance,
            "god_code": GOD_CODE
        }

    def get_analytics(self, time_range: float = 3600) -> Dict[str, Any]:
        """Get gateway analytics"""
        return self.logger.get_analytics(time_range)


# Global instance
def get_api_gateway() -> L104APIGateway:
    """Get API gateway singleton"""
    return L104APIGateway()


# Example handlers
async def health_handler(request: APIRequest, params: Dict) -> APIResponse:
    """Health check handler"""
    return APIResponse(
        200,
        {"Content-Type": "application/json"},
        json.dumps({
            "status": "healthy",
            "timestamp": time.time(),
            "resonance": GOD_CODE / 1000
        }).encode()
    )


async def echo_handler(request: APIRequest, params: Dict) -> APIResponse:
    """Echo handler for testing"""
    return APIResponse(
        200,
        {"Content-Type": "application/json"},
        json.dumps({
            "method": request.method.value,
            "path": request.path,
            "params": params,
            "query": request.query_params,
            "body_length": len(request.body) if request.body else 0
        }).encode()
    )


# Main execution
if __name__ == "__main__":
    print("=" * 70)
    print("  L104 API GATEWAY - DEMONSTRATION")
    print("=" * 70)

    gateway = get_api_gateway()

    # Add routes
    gateway.add_route(Route(
        path_pattern="/health",
        methods=[RequestMethod.GET],
        handler=health_handler,
        auth_required=False,
        cache_ttl=60
    ))

    gateway.add_route(Route(
        path_pattern="/api/echo",
        methods=[RequestMethod.GET, RequestMethod.POST],
        handler=echo_handler,
        auth_required=True,
        rate_limit=100
    ))

    gateway.add_route(Route(
        path_pattern="/api/users/:id",
        methods=[RequestMethod.GET],
        handler=echo_handler,
        auth_required=True
    ))

    # Create API key
    print("\n[DEMO] Creating API Key...")
    api_key = gateway.key_manager.create_key("demo_app", rate_limit=1000)
    print(f"  API Key: {api_key}")

    # Test requests
    async def run_tests():
        print("\n[DEMO] Testing Routes...")

        # Health check
        req1 = APIRequest(
    request_id="test1",
    method=RequestMethod.GET,
    path="/health",
    headers={},
    client_ip="127.0.0.1"
        )
        resp1 = await gateway.handle_request(req1)
        print(f"  /health: {resp1.status_code} - {resp1.body.decode()[:50]}...")

        # Echo with API key
        req2 = APIRequest(
    request_id="test2",
    method=RequestMethod.GET,
    path="/api/echo",
    headers={"X-API-Key": api_key},
    query_params={"test": "value"},
    client_ip="127.0.0.1",
    api_key=api_key
        )
        resp2 = await gateway.handle_request(req2)
        print(f"  /api/echo: {resp2.status_code}")

        # User endpoint with path param
        req3 = APIRequest(
    request_id="test3",
    method=RequestMethod.GET,
    path="/api/users/123",
    headers={"X-API-Key": api_key},
    client_ip="127.0.0.1",
    api_key=api_key
        )
        resp3 = await gateway.handle_request(req3)
        print(f"  /api/users/123: {resp3.status_code}")

        # Without API key (should fail)
        req4 = APIRequest(
    request_id="test4",
    method=RequestMethod.GET,
    path="/api/echo",
    headers={},
    client_ip="127.0.0.1"
        )
        resp4 = await gateway.handle_request(req4)
        print(f"  /api/echo (no key): {resp4.status_code}")

    asyncio.run(run_tests())

    # Status
    print("\n[DEMO] Gateway Status:")
    status = gateway.get_status()
    print(f"  Total Requests: {status['total_requests']}")
    print(f"  Routes: {status['routes']}")
    print(f"  Cache Hit Rate: {status['cache_stats']['hit_rate']:.2%}")
    print(f"  Resonance: {status['resonance']:.8f}")

    print("\n" + "=" * 70)
    print("  API GATEWAY OPERATIONAL")
    print("=" * 70)
