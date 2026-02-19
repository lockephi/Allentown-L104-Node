"""L104 Intellect — Thread-safe LRU Cache with PHI-weighted eviction."""
import time
import threading
from collections import OrderedDict
from typing import Dict, Any


# v11.3 HIGH-LOGIC PERFORMANCE CACHE - φ-Weighted Ultra-Low Latency Response System
# ═══════════════════════════════════════════════════════════════════════════════

class LRUCache:
    """Thread-safe LRU cache with TTL and HIGH-LOGIC v2.0 φ-weighted eviction."""
    __slots__ = ('_cache', '_lock', '_maxsize', '_ttl', '_phi', '_access_weights')

    def __init__(self, maxsize: int = 256, ttl: float = 300.0, phi: float = 1.618033988749895):
        self._cache = OrderedDict()
        self._lock = threading.Lock()
        self._maxsize = maxsize
        self._ttl = ttl
        self._phi = phi
        self._access_weights = {}  # Track φ-weighted access patterns

    def get(self, key: str):
        with self._lock:
            if key in self._cache:
                value, timestamp, access_count = self._cache[key]
                if time.time() - timestamp < self._ttl:
                    # HIGH-LOGIC: φ-weighted access count (diminishing returns)
                    new_count = access_count + (1 / (1 + access_count / self._phi))
                    self._cache[key] = (value, timestamp, new_count)
                    self._cache.move_to_end(key)
                    return value
                del self._cache[key]
                if key in self._access_weights:
                    del self._access_weights[key]
        return None

    def set(self, key: str, value):
        with self._lock:
            if key in self._cache:
                del self._cache[key]
            elif len(self._cache) >= self._maxsize:
                # HIGH-LOGIC: φ-weighted eviction (evict lowest weighted entry)
                if self._cache:
                    min_key = None
                    min_weight = float('inf')
                    for k, (_v, ts, ac) in self._cache.items():
                        # Weight = access_count × φ^(-age_factor)
                        age = time.time() - ts
                        age_factor = min(age / self._ttl, 1.0)
                        weight = ac * (self._phi ** (-age_factor))
                        if weight < min_weight:
                            min_weight = weight
                            min_key = k
                    if min_key:
                        del self._cache[min_key]
                        if min_key in self._access_weights:
                            del self._access_weights[min_key]
                    else:
                        self._cache.popitem(last=False)
            self._cache[key] = (value, time.time(), 1.0)  # Initial access count = 1.0

    def get_phi_weighted_stats(self) -> Dict[str, Any]:
        """HIGH-LOGIC v2.0: Get φ-weighted cache statistics."""
        with self._lock:
            if not self._cache:
                return {"entries": 0, "avg_weight": 0, "total_accesses": 0}
            total_weight = 0
            total_accesses = 0
            for _k, (_v, ts, ac) in self._cache.items():
                age = time.time() - ts
                age_factor = min(age / self._ttl, 1.0)
                weight = ac * (self._phi ** (-age_factor))
                total_weight += weight
                total_accesses += ac
            return {
                "entries": len(self._cache),
                "avg_weight": total_weight / len(self._cache),
                "total_accesses": total_accesses,
                "phi_efficiency": total_weight / max(1, len(self._cache))
            }

    def __len__(self):
        return len(self._cache)

# Global caches for maximum throughput
_RESPONSE_CACHE = LRUCache(maxsize=512, ttl=600.0)   # 10-min response cache
_CONCEPT_CACHE = LRUCache(maxsize=1024, ttl=1800.0)  # 30-min concept cache
_RESONANCE_CACHE = {'value': None, 'time': 0, 'ttl': 0.5}  # 500ms resonance cache
