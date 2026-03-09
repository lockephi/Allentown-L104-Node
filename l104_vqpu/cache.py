"""L104 VQPU v14.0.0 — Circuit Cache (LRU) + Scoring Cache (LRU + TTL).

v14.0.0 QUANTUM FIDELITY ARCHITECTURE:
  - TTL-based expiration for ASI/AGI/SC/entropy caches
  - ASI cache TTL: CACHE_ASI_TTL_S (600s = 10 min)
  - AGI cache TTL: CACHE_AGI_TTL_S (600s = 10 min)
  - SC cache TTL:  CACHE_SC_TTL_S  (300s = 5 min)
  - Entropy cache TTL: CACHE_ENTROPY_TTL_S (1800s = 30 min)
  - Automatic stale-entry eviction on access

v13.0 (retained): OrderedDict LRU, streaming hash fingerprint
"""

import time
import hashlib
from collections import OrderedDict

from .constants import (
    GOD_CODE, PHI, VOID_CONSTANT,
    CACHE_ASI_TTL_S, CACHE_AGI_TTL_S, CACHE_SC_TTL_S, CACHE_ENTROPY_TTL_S,
)

__all__ = ["CircuitCache", "ScoringCache"]

import threading


class CircuitCache:
    """
    LRU cache for circuit execution results.

    v12.2: Replaced fake bloom filter (set) with OrderedDict for true
    O(1) LRU eviction. Eliminated expensive PHI-weighted min() scan
    (O(n) with pow() per entry) in favor of O(1) popitem(last=False).
    Fingerprint uses streaming hash (hashlib.update) to avoid building
    a large concatenated string in memory.
    """

    def __init__(self, max_size: int = 1024):
        from collections import OrderedDict
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()
        self._total_hits = 0
        self._total_misses = 0

    @staticmethod
    def fingerprint(operations: list, num_qubits: int, shots: int) -> str:
        """Compute a canonical fingerprint for a circuit.

        v12.2: Streaming hash via hashlib.update — avoids building a
        large concatenated string for circuits with thousands of gates.
        """
        import hashlib as _hl
        h = _hl.sha256()
        h.update(f"{num_qubits}:{shots}:".encode())
        for op in operations:
            gate = op.get("gate", "") if isinstance(op, dict) else op.gate
            qubits = op.get("qubits", []) if isinstance(op, dict) else op.qubits
            params = op.get("parameters", None) if isinstance(op, dict) else op.parameters
            h.update(f"{gate}:{qubits}".encode())
            if params:
                h.update(f":{[round(p, 10) for p in params]}".encode())
            h.update(b"|")
        return h.hexdigest()[:24]

    def get(self, fp: str) -> dict:
        """Look up a cached result. v12.2: O(1) OrderedDict LRU."""
        with self._lock:
            entry = self._cache.get(fp)
            if entry is not None:
                entry["hits"] += 1
                # Move to end (most recently used)
                self._cache.move_to_end(fp)
                self._total_hits += 1
                return entry["result"]
            self._total_misses += 1
            return None

    def put(self, fp: str, result: dict):
        """Store a result. v12.2: O(1) LRU eviction via OrderedDict.popitem."""
        with self._lock:
            if fp in self._cache:
                self._cache[fp]["result"] = result
                self._cache[fp]["hits"] += 1
                self._cache.move_to_end(fp)
                return
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)  # Evict oldest (LRU)
            self._cache[fp] = {
                "result": result, "hits": 1,
                "last_access": time.monotonic(), "created": time.monotonic(),
            }

    def stats(self) -> dict:
        """Cache performance statistics."""
        with self._lock:
            return {
                "size": len(self._cache), "max_size": self._max_size,
                "total_hits": self._total_hits, "total_misses": self._total_misses,
                "hit_rate": round(self._total_hits / max(self._total_hits + self._total_misses, 1), 4),
            }

    def clear(self):
        """Clear all cached results."""
        with self._lock:
            self._cache.clear()
            self._total_hits = 0
            self._total_misses = 0


# ═══════════════════════════════════════════════════════════════════
# SCORING CACHE (v8.0) — Fix 96% Pipeline Bottleneck
# ═══════════════════════════════════════════════════════════════════

class ScoringCache:
    """
    Dedicated cache for expensive scoring operations.

    v8.0 OPTIMIZATION — Addresses the scoring bottleneck identified in
    benchmarks (96% of pipeline time = 44ms of 46ms).

    The three-engine harmonic and wave scores are DETERMINISTIC for
    fixed GOD_CODE/PHI/VOID_CONSTANT — computed once then cached.
    ASI/AGI scoring is cached per (num_qubits, entropy_bucket).
    Entropy score varies per-circuit but is fast to recompute.

    Cache strategy:
    - Three-engine harmonic/wave: eternal (constants never change)
    - Three-engine entropy: bucketed to 0.1 resolution
    - ASI/AGI scores: bucketed by (num_qubits, entropy_tenth)
    - Sacred alignment: per-circuit (not cached — already fast)
    """

    _harmonic_cached = None
    _wave_cached = None
    _sc_cached = None              # v9.0: SC score cache
    _sc_cached_ts = 0.0            # v14.0: SC cache timestamp
    _entropy_cache = {}                  # bucket → (score, timestamp)
    _asi_cache = OrderedDict()            # (nq, bucket) → (score, timestamp) — LRU eviction
    _agi_cache = OrderedDict()            # (nq, bucket) → (score, timestamp) — LRU eviction
    _ASI_AGI_MAX = 4096                   # max entries before LRU eviction
    _lock = threading.Lock()
    _stats = {"hits": 0, "misses": 0, "harmonic_hits": 0, "wave_hits": 0, "sc_hits": 0, "ttl_evictions": 0}

    @classmethod
    def get_harmonic(cls, scorer_fn) -> float:
        """Get cached harmonic score (deterministic — computed once)."""
        if cls._harmonic_cached is not None:
            with cls._lock:
                cls._stats["harmonic_hits"] += 1
                cls._stats["hits"] += 1
            return cls._harmonic_cached
        with cls._lock:
            cls._stats["misses"] += 1
        val = scorer_fn()
        cls._harmonic_cached = val
        return val

    @classmethod
    def get_wave(cls, scorer_fn) -> float:
        """Get cached wave score (deterministic — computed once)."""
        if cls._wave_cached is not None:
            with cls._lock:
                cls._stats["wave_hits"] += 1
                cls._stats["hits"] += 1
            return cls._wave_cached
        with cls._lock:
            cls._stats["misses"] += 1
        val = scorer_fn()
        cls._wave_cached = val
        return val

    @classmethod
    def get_sc(cls, scorer_fn) -> float:
        """v14.0: Get cached SC score with TTL expiration (CACHE_SC_TTL_S).
        Does NOT cache fallback values — the background thread will populate
        the real score asynchronously."""
        now = time.monotonic()
        with cls._lock:
            # Ensure _stats is properly initialized
            if not isinstance(cls._stats, dict):
                cls._stats = {"hits": 0, "misses": 0, "harmonic_hits": 0, "wave_hits": 0, "sc_hits": 0, "ttl_evictions": 0}
            if cls._sc_cached is not None:
                if (now - cls._sc_cached_ts) > CACHE_SC_TTL_S:
                    # TTL expired — evict
                    cls._sc_cached = None
                    cls._sc_cached_ts = 0.0
                    cls._stats["ttl_evictions"] = cls._stats.get("ttl_evictions", 0) + 1
                else:
                    cls._stats["sc_hits"] = cls._stats.get("sc_hits", 0) + 1
                    cls._stats["hits"] = cls._stats.get("hits", 0) + 1
                    return cls._sc_cached
            cls._stats["misses"] = cls._stats.get("misses", 0) + 1
            val = scorer_fn()
            # v12.3: Only cache if it's a real score (not fallback 0.5/0.75)
            from .three_engine import ThreeEngineQuantumScorer
            if ThreeEngineQuantumScorer._sc_bg_result is not None:
                cls._sc_cached = val
                cls._sc_cached_ts = now
            return val

    @classmethod
    def get_entropy(cls, measurement_entropy: float, scorer_fn) -> float:
        """Get cached entropy score (bucketed to 0.1 resolution).

        v14.0: TTL-based expiration (CACHE_ENTROPY_TTL_S = 1800s).
        v13.1: Single lock acquisition to fix TOCTOU race.
        """
        bucket = round(measurement_entropy, 1)
        now = time.monotonic()
        with cls._lock:
            entry = cls._entropy_cache.get(bucket)
            if entry is not None:
                val, ts = entry
                if (now - ts) > CACHE_ENTROPY_TTL_S:
                    # TTL expired
                    del cls._entropy_cache[bucket]
                    cls._stats["ttl_evictions"] = cls._stats.get("ttl_evictions", 0) + 1
                else:
                    cls._stats["hits"] += 1
                    return val
        # Compute outside lock
        val = scorer_fn(measurement_entropy)
        with cls._lock:
            if bucket not in cls._entropy_cache:
                cls._entropy_cache[bucket] = (val, now)
            else:
                val = cls._entropy_cache[bucket][0]
            cls._stats["misses"] += 1
        return val

    @classmethod
    def get_asi_score(cls, probs, num_qubits, entropy_bucket, scorer_fn) -> dict:
        """Get cached ASI score (bucketed by nq + entropy).

        v14.0: TTL-based expiration (CACHE_ASI_TTL_S = 600s).
        v13.0: OrderedDict LRU — evicts oldest entry when full.
        """
        key = (num_qubits, round(entropy_bucket, 1))
        now = time.monotonic()
        with cls._lock:
            if key in cls._asi_cache:
                val, ts = cls._asi_cache[key]
                if (now - ts) > CACHE_ASI_TTL_S:
                    del cls._asi_cache[key]
                    cls._stats["ttl_evictions"] = cls._stats.get("ttl_evictions", 0) + 1
                else:
                    cls._stats["hits"] += 1
                    cls._asi_cache.move_to_end(key)
                    return val
        cls._stats["misses"] += 1
        val = scorer_fn(probs, num_qubits)
        with cls._lock:
            if key in cls._asi_cache:
                cls._asi_cache.move_to_end(key)
            else:
                if len(cls._asi_cache) >= cls._ASI_AGI_MAX:
                    cls._asi_cache.popitem(last=False)
                cls._asi_cache[key] = (val, now)
        return val

    @classmethod
    def get_agi_score(cls, probs, num_qubits, entropy_bucket, scorer_fn) -> dict:
        """Get cached AGI score (bucketed by nq + entropy).

        v14.0: TTL-based expiration (CACHE_AGI_TTL_S = 600s).
        v13.0: OrderedDict LRU — evicts oldest entry when full.
        """
        key = (num_qubits, round(entropy_bucket, 1))
        now = time.monotonic()
        with cls._lock:
            if key in cls._agi_cache:
                val, ts = cls._agi_cache[key]
                if (now - ts) > CACHE_AGI_TTL_S:
                    del cls._agi_cache[key]
                    cls._stats["ttl_evictions"] = cls._stats.get("ttl_evictions", 0) + 1
                else:
                    cls._stats["hits"] += 1
                    cls._agi_cache.move_to_end(key)
                    return val
        cls._stats["misses"] += 1
        val = scorer_fn(probs, num_qubits)
        with cls._lock:
            if key in cls._agi_cache:
                cls._agi_cache.move_to_end(key)
            else:
                if len(cls._agi_cache) >= cls._ASI_AGI_MAX:
                    cls._agi_cache.popitem(last=False)
                cls._agi_cache[key] = (val, now)
        return val

    @classmethod
    def stats(cls) -> dict:
        """Cache performance statistics."""
        total = cls._stats["hits"] + cls._stats["misses"]
        return {
            "total_hits": cls._stats["hits"],
            "total_misses": cls._stats["misses"],
            "hit_rate": round(cls._stats["hits"] / max(total, 1), 4),
            "harmonic_cached": cls._harmonic_cached is not None,
            "wave_cached": cls._wave_cached is not None,
            "sc_cached": cls._sc_cached is not None,
            "sc_hits": cls._stats.get("sc_hits", 0),
            "ttl_evictions": cls._stats.get("ttl_evictions", 0),
            "entropy_buckets": len(cls._entropy_cache),
            "asi_entries": len(cls._asi_cache),
            "agi_entries": len(cls._agi_cache),
        }

    @classmethod
    def clear(cls):
        """Clear all scoring caches.

        v15.2: Uses .clear() on existing dicts/OrderedDicts instead of
        replacing the reference.  Prevents a race where another thread
        still holds a reference to the old object and either reads stale
        data or writes into a detached container.
        """
        with cls._lock:
            cls._harmonic_cached = None
            cls._wave_cached = None
            cls._sc_cached = None
            cls._sc_cached_ts = 0.0
            cls._entropy_cache.clear()
            cls._asi_cache.clear()
            cls._agi_cache.clear()
            cls._stats.update({"hits": 0, "misses": 0, "harmonic_hits": 0,
                               "wave_hits": 0, "sc_hits": 0, "ttl_evictions": 0})
