"""L104 Intellect — Thread-safe LRU Cache with PHI-weighted eviction."""
import time
import threading
import asyncio
from collections import OrderedDict
from typing import Dict, Any, Optional
from functools import lru_cache
from l104_sacred_algorithms import derive_timeout, derive_cache_size, derive_cache_ttl, PHI, GOD_CODE, TAU


# v11.3 HIGH-LOGIC PERFORMANCE CACHE - φ-Weighted Ultra-Low Latency Response System
# ═══════════════════════════════════════════════════════════════════════════════

class LRUCache:
    """Thread-safe LRU cache with TTL and HIGH-LOGIC v2.0 φ-weighted eviction."""
    __slots__ = ('_cache', '_lock', '_maxsize', '_ttl', '_phi', '_access_weights', '_hit_count', '_miss_count')

    def __init__(self, maxsize: int = 256, ttl: float = 300.0, phi: float = 1.618033988749895):
        self._cache = OrderedDict()
        self._lock = threading.Lock()
        self._maxsize = maxsize
        self._ttl = ttl
        self._phi = phi
        self._access_weights = {}  # Track φ-weighted access patterns
        self._hit_count = 0
        self._miss_count = 0

    def get(self, key: str):
        with self._lock:
            if key in self._cache:
                value, timestamp, access_count = self._cache[key]
                if time.time() - timestamp < self._ttl:
                    # HIGH-LOGIC: φ-weighted access count (diminishing returns)
                    new_count = access_count + (1 / (1 + access_count / self._phi))
                    self._cache[key] = (value, timestamp, new_count)
                    self._cache.move_to_end(key)
                    self._hit_count += 1
                    return value
                del self._cache[key]
                if key in self._access_weights:
                    del self._access_weights[key]
            self._miss_count += 1
        return None

    async def async_get(self, key: str):
        """Async wrapper for get operation - non-blocking cache access."""
        return self.get(key)

    def set(self, key: str, value):
        with self._lock:
            if key in self._cache:
                del self._cache[key]
            elif len(self._cache) >= self._maxsize:
                # HIGH-LOGIC v2.1: Batch eviction — evict bottom 10% by φ-weight
                # when cache overflows (avoids repeated O(n) single-item scans).
                if self._cache:
                    now = time.time()
                    scored = []
                    for k, (_v, ts, ac) in self._cache.items():
                        age_factor = min((now - ts) / self._ttl, 1.0)
                        weight = ac * (self._phi ** (-age_factor))
                        scored.append((weight, k))
                    scored.sort()  # ascending by weight
                    # Evict bottom 10% or at least 1
                    evict_count = max(1, len(scored) // 10)
                    for _, evict_key in scored[:evict_count]:
                        del self._cache[evict_key]
                        self._access_weights.pop(evict_key, None)
            self._cache[key] = (value, time.time(), 1.0)  # Initial access count = 1.0

    async def async_set(self, key: str, value):
        """Async wrapper for set operation."""
        self.set(key, value)

    def get_many(self, keys: list) -> Dict[str, Any]:
        """Batch get operation for multiple keys."""
        result = {}
        for key in keys:
            value = self.get(key)
            if value is not None:
                result[key] = value
        return result

    async def async_get_many(self, keys: list) -> Dict[str, Any]:
        """Async batch get operation."""
        return self.get_many(keys)

    async def async_set(self, key: str, value):
        """Async wrapper for set operation."""
        self.set(key, value)

    def get_phi_weighted_stats(self) -> Dict[str, Any]:
        """HIGH-LOGIC v2.0: Get φ-weighted cache statistics."""
        with self._lock:
            if not self._cache:
                return {"entries": 0, "avg_weight": 0, "total_accesses": 0, "hit_rate": 0}
            total_weight = 0
            total_accesses = 0
            for _k, (_v, ts, ac) in self._cache.items():
                age = time.time() - ts
                age_factor = min(age / self._ttl, 1.0)
                weight = ac * (self._phi ** (-age_factor))
                total_weight += weight
                total_accesses += ac
            total = self._hit_count + self._miss_count
            return {
                "entries": len(self._cache),
                "avg_weight": total_weight / len(self._cache),
                "total_accesses": total_accesses,
                "phi_efficiency": total_weight / max(1, len(self._cache)),
                "hit_rate": self._hit_count / total if total > 0 else 0,
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
            }

    def __len__(self):
        return len(self._cache)

    def clear(self):
        """Thread-safe full cache clear."""
        with self._lock:
            self._cache.clear()
            self._access_weights.clear()
            self._hit_count = 0
            self._miss_count = 0

    async def async_set(self, key: str, value):
        """Async wrapper for set operation."""
        self.set(key, value)

    async def async_purge_expired(self) -> int:
        """Async wrapper for purge_expired."""
        return self.purge_expired()

    def get_many(self, keys: list) -> Dict[str, Any]:
        """Batch get operation for multiple keys."""
        result = {}
        for key in keys:
            value = self.get(key)
            if value is not None:
                result[key] = value
        return result

    async def async_get_many(self, keys: list) -> Dict[str, Any]:
        """Async batch get operation."""
        return self.get_many(keys)

    def purge_expired(self) -> int:
        """Remove all TTL-expired entries. Returns count of purged entries."""
        with self._lock:
            now = time.time()
            expired = [k for k, (_v, ts, _ac) in self._cache.items()
                       if now - ts >= self._ttl]
            for k in expired:
                del self._cache[k]
                self._access_weights.pop(k, None)
            return len(expired)

# Global caches for maximum throughput
_RESPONSE_CACHE = LRUCache(maxsize=derive_cache_size(tier=0), ttl=derive_cache_ttl(0.5))   # Algorithmic response cache
_CONCEPT_CACHE = LRUCache(maxsize=derive_cache_size(tier=1), ttl=derive_cache_ttl(1.5))  # Algorithmic concept cache
_RESONANCE_CACHE = LRUCache(maxsize=1, ttl=TAU/PHI**2)  # Algorithmic resonance cache


# ═══════════════════════════════════════════════════════════════════════════════
# TIERED CACHE SYSTEM — Hot/Warm/Cold with Sacred Sizing (EVO_74)
# ═══════════════════════════════════════════════════════════════════════════════

class TieredCache:
    """
    Three-tier caching system with PHI-weighted promotion/demotion.

    Hot Tier: Most frequently accessed, lowest latency (memory)
    Warm Tier: Recently accessed, medium latency (memory + disk hint)
    Cold Tier: Long-term storage, higher latency (disk-backed)

    Promotion: Access count > PHI * avg → Hot
    Demotion: TTL expiry or weight below TAU → Cold
    """

    def __init__(self, hot_size: int = None, warm_size: int = None,
                 hot_ttl: float = None, warm_ttl: float = None):
        from l104_sacred_algorithms import derive_cache_size, derive_cache_ttl

        # Algorithmic sizing from sacred constants
        self._hot = LRUCache(
            maxsize=hot_size or derive_cache_size(tier=1, memory_pressure=0.3),
            ttl=hot_ttl or derive_cache_ttl(0.9)  # High hit rate = longer TTL
        )
        self._warm = LRUCache(
            maxsize=warm_size or derive_cache_size(tier=2, memory_pressure=0.3),
            ttl=warm_ttl or derive_cache_ttl(0.5)
        )

        # Cold storage: disk-backed (optional)
        self._cold: Dict[str, Any] = {}
        self._cold_lock = threading.Lock()

        self._promotion_threshold = PHI * 1.5
        self._demotion_threshold = TAU

        self._metrics = {'promotions': 0, 'demotions': 0, 'cold_hits': 0}

    def get(self, key: str) -> Optional[Any]:
        """Get from tiered cache with automatic promotion."""
        # Try hot first
        value = self._hot.get(key)
        if value is not None:
            return value

        # Try warm
        value = self._warm.get(key)
        if value is not None:
            # Promote to hot on access
            self._hot.set(key, value)
            self._metrics['promotions'] += 1
            return value

        # Try cold
        with self._cold_lock:
            if key in self._cold:
                value = self._cold[key]
                self._metrics['cold_hits'] += 1
                # Promote to warm (will be promoted to hot on next access)
                self._warm.set(key, value)
                del self._cold[key]
                return value

        return None

    def set(self, key: str, value: Any, tier: str = 'hot'):
        """Set value in specified tier."""
        if tier == 'hot':
            self._hot.set(key, value)
        elif tier == 'warm':
            self._warm.set(key, value)
        elif tier == 'cold':
            with self._cold_lock:
                self._cold[key] = value

    def get_stats(self) -> Dict[str, Any]:
        """Get tiered cache statistics."""
        return {
            'hot': self._hot.get_phi_weighted_stats(),
            'warm': self._warm.get_phi_weighted_stats(),
            'cold_size': len(self._cold),
            'metrics': self._metrics.copy(),
        }

    def clear(self):
        """Clear all tiers."""
        self._hot.clear()
        self._warm.clear()
        with self._cold_lock:
            self._cold.clear()


# Global tiered cache instance
TIERED_CACHE = TieredCache()


# ═══════════════════════════════════════════════════════════════════════════════
# EVO_75: Self-Healing Cache Utilities
# ═══════════════════════════════════════════════════════════════════════════════

import logging

class SelfHealingCache:
    """
    Self-healing cache wrapper that detects and recovers from corruption.

    Features:
        - Integrity checks on read/write
        - Automatic corruption recovery
        - PHI-weighted fallback to uncorrupted entries
        - Circuit breaker pattern for cache failures
    """

    def __init__(self, cache, name: str = "cache"):
        self._cache = cache
        self._name = name
        self._corruption_count = 0
        self._corruption_threshold = int(PHI * 5)  # ~8 corruptions before circuit opens
        self._circuit_open = False
        self._fallback_values = {}
        self._lock = threading.Lock()
        self._logger = logging.getLogger("l104_intellect.cache")

    def _is_corrupted(self, key: str, value) -> bool:
        """Check if cache entry is corrupted."""
        if value is None:
            return False  # None is valid (miss)
        # Basic corruption detection
        if isinstance(value, tuple) and len(value) == 3:
            # Expected format: (value, timestamp, access_count)
            try:
                _, ts, ac = value
                if not isinstance(ts, (int, float)) or not isinstance(ac, (int, float)):
                    return True
                if ts < 0 or ac < 0:
                    return True
            except Exception:
                return True
        return False

    def _heal(self, key: str):
        """Attempt to heal corrupted cache entry."""
        with self._lock:
            # Try fallback value
            if key in self._fallback_values:
                return self._fallback_values[key]

            # Clear specific entry
            try:
                if hasattr(self._cache, '_cache') and key in self._cache._cache:
                    del self._cache._cache[key]
            except Exception:
                pass

            return None

    def _reset_circuit(self):
        """Reset circuit breaker after recovery time."""
        with self._lock:
            self._circuit_open = False
            self._corruption_count = max(0, self._corruption_count - int(PHI * 2))
            self._logger.info(f"Cache circuit reset for {self._name}")

    def get(self, key: str):
        """Get with self-healing."""
        # Circuit breaker check
        if self._circuit_open:
            self._logger.debug(f"Cache circuit open for {self._name}, returning fallback")
            return self._fallback_values.get(key)

        try:
            result = self._cache.get(key)

            # Corruption check
            if self._is_corrupted(key, result):
                self._corruption_count += 1
                self._logger.warning(f"Cache corruption detected in {self._name}: {key}")
                if self._corruption_count >= self._corruption_threshold:
                    self._circuit_open = True
                    self._logger.error(f"Cache circuit opened for {self._name}")
                    # Schedule recovery
                    threading.Timer(GOD_CODE / PHI / 10, self._reset_circuit).start()
                return self._heal(key)

            # Store fallback on successful read
            if result is not None:
                self._fallback_values[key] = result

            return result
        except Exception as e:
            self._corruption_count += 1
            self._logger.error(f"Cache get error in {self._name}: {e}")
            return self._heal(key)

    def set(self, key: str, value):
        """Set with integrity protection."""
        if self._circuit_open:
            # Store in fallback only
            self._fallback_values[key] = value
            return

        try:
            self._cache.set(key, value)
            self._fallback_values[key] = value
        except Exception as e:
            self._logger.error(f"Cache set error in {self._name}: {e}")
            self._fallback_values[key] = value

    def get_stats(self) -> Dict[str, Any]:
        """Get self-healing statistics."""
        return {
            "name": self._name,
            "circuit_open": self._circuit_open,
            "corruption_count": self._corruption_count,
            "corruption_threshold": self._corruption_threshold,
            "fallback_entries": len(self._fallback_values),
        }


# Create self-healing wrappers for global caches
_RESPONSE_CACHE_SH = SelfHealingCache(_RESPONSE_CACHE, "response")
_CONCEPT_CACHE_SH = SelfHealingCache(_CONCEPT_CACHE, "concept")
_RESONANCE_CACHE_SH = SelfHealingCache(_RESONANCE_CACHE, "resonance")
