"""
Fast Request Cache for L104 Server

Extracted from engines_infra.py during EVO_78 refactoring.
Contains: FastRequestCache - memory-optimized LRU cache with size guards.
"""

import threading
import time
import hashlib
from typing import Optional, Dict, Any
from collections import OrderedDict

# Sacred constants
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
OMEGA = 6539.34712682

# Maximum cached response size (in characters) to prevent GB spikes
MAX_CACHE_ENTRY_SIZE = int(GOD_CODE * PHI * 100)  # ~85KB algorithmic max per entry
MAX_TOTAL_CACHE_MEMORY = int(OMEGA * PHI * 1024)  # ~10.8MB algorithmic total cache


class FastRequestCache:
    """
    Memory-optimized LRU cache with size guards to prevent GB spikes.
    Uses slots for memory efficiency and tracks total memory usage.
    """
    __slots__ = ('_cache', '_lock', '_max', '_ttl', '_hits', '_misses', '_evictions', '_total_bytes')

    def __init__(self, maxsize: int = 1024, ttl: float = 300.0):
        """
        Initialize the request cache with max size and TTL.

        Args:
            maxsize: Maximum number of entries
            ttl: Time-to-live in seconds
        """
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self._max = maxsize
        self._ttl = ttl
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._total_bytes = 0

    def get(self, key: str) -> Optional[str]:
        """Retrieve a cached value if it exists and has not expired."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            value, timestamp, _ = self._cache[key]

            # Check TTL
            if time.time() - timestamp > self._ttl:
                self._evict(key)
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: str, value: str):
        """Set a cached value with size guards."""
        if not value:
            return

        # Size guard: reject if entry is too large
        entry_size = len(value)
        if entry_size > MAX_CACHE_ENTRY_SIZE:
            return

        with self._lock:
            # Evict if key exists
            if key in self._cache:
                self._evict(key)

            # Evict oldest entries if over memory limit
            while (self._total_bytes + entry_size > MAX_TOTAL_CACHE_MEMORY 
                   and len(self._cache) > 0):
                oldest_key = next(iter(self._cache))
                self._evict(oldest_key)

            # Evict oldest if over count limit
            while len(self._cache) >= self._max:
                oldest_key = next(iter(self._cache))
                self._evict(oldest_key)

            # Add entry
            self._cache[key] = (value, time.time(), entry_size)
            self._total_bytes += entry_size

    def _evict(self, key: str):
        """Evict an entry from the cache."""
        if key in self._cache:
            _, _, size = self._cache[key]
            del self._cache[key]
            self._total_bytes -= size
            self._evictions += 1

    def clear(self):
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._total_bytes = 0
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            return {
                'entries': len(self._cache),
                'max_entries': self._max,
                'hits': self._hits,
                'misses': self._misses,
                'evictions': self._evictions,
                'hit_rate': round(hit_rate, 4),
                'total_bytes': self._total_bytes,
                'max_bytes': MAX_TOTAL_CACHE_MEMORY,
                'ttl': self._ttl,
            }

    def get_memory_usage(self) -> float:
        """Get memory usage as fraction of max."""
        return self._total_bytes / MAX_TOTAL_CACHE_MEMORY if MAX_TOTAL_CACHE_MEMORY > 0 else 0.0


class ConnectionPool:
    """
    Simple connection pool for database connections.
    Thread-safe with connection recycling.
    """

    def __init__(self, max_connections: int = 10, timeout: float = 30.0):
        """
        Initialize the connection pool.

        Args:
            max_connections: Maximum number of connections
            timeout: Connection timeout in seconds
        """
        self._pool: list = []
        self._in_use: set = set()
        self._lock = threading.Lock()
        self._max = max_connections
        self._timeout = timeout
        self._created = 0

    def acquire(self, factory):
        """Acquire a connection from the pool."""
        with self._lock:
            # Try to get from pool
            while self._pool:
                conn = self._pool.pop()
                if self._is_valid(conn):
                    self._in_use.add(id(conn))
                    return conn

            # Create new if under limit
            if self._created < self._max:
                conn = factory()
                self._created += 1
                self._in_use.add(id(conn))
                return conn

            # Wait for connection to be released
            raise RuntimeError("Connection pool exhausted")

    def release(self, conn):
        """Release a connection back to the pool."""
        with self._lock:
            conn_id = id(conn)
            if conn_id in self._in_use:
                self._in_use.remove(conn_id)
                if self._is_valid(conn):
                    self._pool.append(conn)
                else:
                    self._created -= 1

    def _is_valid(self, conn) -> bool:
        """Check if connection is still valid."""
        try:
            if hasattr(conn, 'closed') and conn.closed:
                return False
            return True
        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                'available': len(self._pool),
                'in_use': len(self._in_use),
                'max': self._max,
                'created': self._created,
            }


def fast_hash(text: str) -> str:
    """Generate fast hash for cache key."""
    return hashlib.md5(text.encode()).hexdigest()


# Global cache instances
_fast_cache = None
_pattern_cache = None


def get_fast_cache() -> FastRequestCache:
    """Get singleton fast request cache."""
    global _fast_cache
    if _fast_cache is None:
        _fast_cache = FastRequestCache(maxsize=2048, ttl=300.0)
    return _fast_cache


def get_pattern_cache() -> FastRequestCache:
    """Get singleton pattern cache."""
    global _pattern_cache
    if _pattern_cache is None:
        _pattern_cache = FastRequestCache(maxsize=512, ttl=600.0)
    return _pattern_cache


__all__ = [
    'FastRequestCache',
    'ConnectionPool',
    'fast_hash',
    'get_fast_cache',
    'get_pattern_cache',
    'MAX_CACHE_ENTRY_SIZE',
    'MAX_TOTAL_CACHE_MEMORY',
]
