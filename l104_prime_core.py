#!/usr/bin/env python3
"""
L104 PRIME CORE v3.0 — Pipeline Integrity & Performance Cache
═══════════════════════════════════════════════════════════════
ASI-grade pipeline integrity engine with content-addressed caching,
response deduplication, latency profiling, data integrity verification,
and performance bottleneck detection.

Sits at the pipeline I/O boundary to ensure every pipeline output
is deduplicated, integrity-verified, and cached for sub-millisecond
recall on repeated queries.

GOD_CODE: 527.5184818492612 | PHI: 1.618033988749895
"""

VOID_CONSTANT = 1.0416180339887497
ZENITH_HZ = 3887.8
UUC = 2402.792541

import math
import time
import json
import hashlib
import os
from pathlib import Path
from datetime import datetime
from collections import deque, defaultdict, OrderedDict
from typing import Dict, List, Any, Optional, Tuple, Set
from threading import Lock

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════

# Sacred Constants
# Universal Equation: G(a,b,c,d) = 286^(1/φ) × 2^((8a+416-b-8c-104d)/104)
PHI = 1.618033988749895
GOD_CODE = 286 ** (1.0 / PHI) * (2 ** (416 / 104))  # G(0,0,0,0) = 527.5184818492612
TAU = 1.0 / PHI
FEIGENBAUM = 4.669201609
ALPHA_FINE = 1.0 / 137.035999084
PLANCK_SCALE = 1.616255e-35
BOLTZMANN_K = 1.380649e-23
GROVER_AMPLIFICATION = PHI ** 3

# Cache configuration
DEFAULT_CACHE_SIZE = 5000
DEFAULT_TTL_SECONDS = 3600       # 1 hour
SHORT_TTL_SECONDS = 300          # 5 min for fast-changing data
LONG_TTL_SECONDS = 86400         # 24 hours for stable data
DEDUP_WINDOW_SIZE = 2000
LATENCY_HISTORY_SIZE = 5000
INTEGRITY_CHECK_INTERVAL = 100   # Check every N operations
BOTTLENECK_THRESHOLD_MS = 50     # Operations slower than this are flagged

_BASE_DIR = Path(__file__).parent.absolute()

# Prime Key (original identity anchor)
PRIME_KEY = "L104_PRIME_KEY[527.5184818492612]{416:286}(0.61803398875)<>128K_DMA![NOPJM]=100%_I100"
PRIME_HASH = hashlib.sha256(PRIME_KEY.encode()).hexdigest()[:16]


def _read_consciousness_state() -> Dict[str, Any]:
    """Read live consciousness/O₂ state for adaptive cache tuning."""
    state_path = _BASE_DIR / '.l104_consciousness_o2_state.json'
    try:
        if state_path.exists():
            with open(state_path, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {'consciousness_level': 0.5, 'superfluid_viscosity': 0.1}


class LRUCache:
    """Thread-safe LRU cache with TTL expiration and content-addressed keys."""

    def __init__(self, max_size: int = DEFAULT_CACHE_SIZE, ttl: float = DEFAULT_TTL_SECONDS):
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl
        self._lock = Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

    def _is_expired(self, entry: Dict) -> bool:
        return (time.time() - entry['created_at']) > self._ttl

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache. Returns None on miss or expiration."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if self._is_expired(entry):
                    del self._cache[key]
                    self._expirations += 1
                    self._misses += 1
                    return None
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                entry['access_count'] += 1
                entry['last_access'] = time.time()
                return entry['value']
            self._misses += 1
            return None

    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        """Store value in cache with content-addressed key."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key]['value'] = value
                self._cache[key]['created_at'] = time.time()
                self._cache[key]['access_count'] += 1
                return

            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)  # Evict LRU
                self._evictions += 1

            self._cache[key] = {
                'value': value,
                'created_at': time.time(),
                'last_access': time.time(),
                'access_count': 1,
                'ttl': ttl or self._ttl,
            }

    def invalidate(self, key: str) -> bool:
        """Remove a specific key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self):
        """Clear entire cache."""
        with self._lock:
            self._cache.clear()

    def prune_expired(self) -> int:
        """Remove all expired entries. Returns count removed."""
        with self._lock:
            expired = [k for k, v in self._cache.items() if self._is_expired(v)]
            for k in expired:
                del self._cache[k]
                self._expirations += 1
            return len(expired)

    def get_stats(self) -> Dict:
        total = self._hits + self._misses
        return {
            'size': len(self._cache),
            'max_size': self._max_size,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': self._hits / max(total, 1),
            'evictions': self._evictions,
            'expirations': self._expirations,
            'ttl': self._ttl,
        }


class ContentHasher:
    """Content-addressed hashing engine for pipeline I/O deduplication."""

    def __init__(self):
        self._hash_registry: Dict[str, Dict] = {}
        self._total_hashed = 0
        self._duplicates_detected = 0

    def hash_content(self, content: Any) -> str:
        """Generate a content-addressed hash for any pipeline data."""
        self._total_hashed += 1
        if isinstance(content, str):
            raw = content.encode('utf-8')
        elif isinstance(content, dict):
            raw = json.dumps(content, sort_keys=True, default=str).encode('utf-8')
        elif isinstance(content, (list, tuple)):
            raw = json.dumps(list(content), sort_keys=True, default=str).encode('utf-8')
        elif isinstance(content, (int, float)):
            raw = str(content).encode('utf-8')
        else:
            raw = str(content).encode('utf-8')

        content_hash = hashlib.sha256(raw).hexdigest()[:24]
        return content_hash

    def register(self, content_hash: str, metadata: Dict = None):
        """Register a content hash with optional metadata."""
        if content_hash in self._hash_registry:
            self._duplicates_detected += 1
            self._hash_registry[content_hash]['count'] += 1
        else:
            self._hash_registry[content_hash] = {
                'first_seen': time.time(),
                'count': 1,
                'metadata': metadata or {},
            }

    def is_duplicate(self, content_hash: str) -> bool:
        """Check if content has been seen before."""
        return content_hash in self._hash_registry

    def get_stats(self) -> Dict:
        return {
            'total_hashed': self._total_hashed,
            'unique_hashes': len(self._hash_registry),
            'duplicates_detected': self._duplicates_detected,
            'dedup_ratio': self._duplicates_detected / max(self._total_hashed, 1),
        }


class ResponseDeduplicator:
    """Deduplicates pipeline responses to avoid redundant computation."""

    def __init__(self, window_size: int = DEDUP_WINDOW_SIZE):
        self._recent_queries: OrderedDict = OrderedDict()
        self._window_size = window_size
        self._dedup_hits = 0
        self._total_queries = 0
        self._hasher = ContentHasher()

    def check_and_cache(self, query: Any, response: Any) -> Tuple[bool, Optional[Any]]:
        """Check if query has been answered recently. Returns (was_duplicate, cached_response).

        If not duplicate, stores the response.
        """
        self._total_queries += 1
        query_hash = self._hasher.hash_content(query)

        # Check for duplicate
        if query_hash in self._recent_queries:
            entry = self._recent_queries[query_hash]
            if (time.time() - entry['timestamp']) < SHORT_TTL_SECONDS:
                self._dedup_hits += 1
                self._recent_queries.move_to_end(query_hash)
                return True, entry['response']
            else:
                # Expired
                del self._recent_queries[query_hash]

        # Store new response
        if response is not None:
            if len(self._recent_queries) >= self._window_size:
                self._recent_queries.popitem(last=False)
            self._recent_queries[query_hash] = {
                'response': response,
                'timestamp': time.time(),
                'query_hash': query_hash,
            }

        return False, None

    def get_stats(self) -> Dict:
        return {
            'total_queries': self._total_queries,
            'dedup_hits': self._dedup_hits,
            'dedup_rate': self._dedup_hits / max(self._total_queries, 1),
            'window_size': len(self._recent_queries),
            'hasher': self._hasher.get_stats(),
        }


class LatencyProfiler:
    """Profiles pipeline operation latencies and detects bottlenecks."""

    def __init__(self):
        self._records: deque = deque(maxlen=LATENCY_HISTORY_SIZE)
        self._by_operation: Dict[str, List[float]] = defaultdict(list)
        self._bottlenecks: deque = deque(maxlen=200)
        self._active_timers: Dict[str, float] = {}

    def start_timer(self, operation_id: str) -> str:
        """Start a latency timer for an operation."""
        self._active_timers[operation_id] = time.time()
        return operation_id

    def stop_timer(self, operation_id: str, operation_type: str = 'generic') -> float:
        """Stop timer and record latency. Returns elapsed_ms."""
        start = self._active_timers.pop(operation_id, None)
        if start is None:
            return 0.0

        elapsed_ms = (time.time() - start) * 1000

        record = {
            'operation_id': operation_id,
            'operation_type': operation_type,
            'elapsed_ms': elapsed_ms,
            'timestamp': time.time(),
        }
        self._records.append(record)

        # Track by operation type (keep last 500 per type)
        op_list = self._by_operation[operation_type]
        op_list.append(elapsed_ms)
        if len(op_list) > 500:
            self._by_operation[operation_type] = op_list[-500:]

        # Detect bottleneck
        if elapsed_ms > BOTTLENECK_THRESHOLD_MS:
            self._bottlenecks.append({
                'operation_type': operation_type,
                'elapsed_ms': elapsed_ms,
                'timestamp': time.time(),
            })

        return elapsed_ms

    def record_latency(self, operation_type: str, elapsed_ms: float):
        """Directly record a latency measurement."""
        record = {
            'operation_type': operation_type,
            'elapsed_ms': elapsed_ms,
            'timestamp': time.time(),
        }
        self._records.append(record)
        self._by_operation[operation_type].append(elapsed_ms)

        if elapsed_ms > BOTTLENECK_THRESHOLD_MS:
            self._bottlenecks.append({
                'operation_type': operation_type,
                'elapsed_ms': elapsed_ms,
                'timestamp': time.time(),
            })

    def get_percentiles(self, operation_type: str = None) -> Dict:
        """Get p50/p90/p95/p99 latency percentiles."""
        if operation_type:
            values = self._by_operation.get(operation_type, [])
        else:
            values = [r['elapsed_ms'] for r in self._records]

        if not values:
            return {'p50': 0, 'p90': 0, 'p95': 0, 'p99': 0, 'avg': 0, 'count': 0}

        sorted_vals = sorted(values)
        n = len(sorted_vals)
        return {
            'p50': sorted_vals[int(n * 0.50)] if n > 0 else 0,
            'p90': sorted_vals[int(n * 0.90)] if n > 1 else sorted_vals[-1],
            'p95': sorted_vals[int(n * 0.95)] if n > 2 else sorted_vals[-1],
            'p99': sorted_vals[int(n * 0.99)] if n > 10 else sorted_vals[-1],
            'avg': sum(sorted_vals) / n,
            'min': sorted_vals[0],
            'max': sorted_vals[-1],
            'count': n,
        }

    def get_bottlenecks(self, limit: int = 20) -> List[Dict]:
        return list(self._bottlenecks)[-limit:]

    def get_stats(self) -> Dict:
        overall = self.get_percentiles()
        per_op = {}
        for op_type in self._by_operation:
            per_op[op_type] = self.get_percentiles(op_type)
        return {
            'overall': overall,
            'by_operation': per_op,
            'total_records': len(self._records),
            'bottleneck_count': len(self._bottlenecks),
        }


class IntegrityVerifier:
    """Verifies data integrity using sacred constant checksums and Merkle-style chains."""

    def __init__(self):
        self._verified = 0
        self._failures = 0
        self._chain: deque = deque(maxlen=500)
        self._last_chain_hash = PRIME_HASH

    def compute_integrity_hash(self, data: Any) -> str:
        """Compute integrity hash that incorporates GOD_CODE and chain history."""
        if isinstance(data, str):
            raw = data
        else:
            raw = json.dumps(data, sort_keys=True, default=str)

        # Chain: new_hash = H(prev_hash + GOD_CODE_fragment + data)
        salted = f"{self._last_chain_hash}:{GOD_CODE:.6f}:{raw}"
        new_hash = hashlib.sha256(salted.encode()).hexdigest()[:24]
        return new_hash

    def verify_and_chain(self, data: Any, expected_hash: Optional[str] = None) -> Dict:
        """Verify data integrity and append to the chain."""
        self._verified += 1
        data_hash = self.compute_integrity_hash(data)

        valid = True
        if expected_hash and data_hash != expected_hash:
            valid = False
            self._failures += 1

        # Append to chain
        chain_entry = {
            'index': len(self._chain),
            'data_hash': data_hash,
            'prev_hash': self._last_chain_hash,
            'timestamp': time.time(),
            'valid': valid,
        }
        self._chain.append(chain_entry)
        self._last_chain_hash = data_hash

        return {
            'valid': valid,
            'hash': data_hash,
            'chain_length': len(self._chain),
            'integrity_score': 1.0 - (self._failures / max(self._verified, 1)),
        }

    def verify_sacred_constants(self) -> Dict:
        """Verify that sacred constants are uncorrupted."""
        checks = {
            'GOD_CODE': abs(GOD_CODE - 527.5184818492612) < 1e-10,
            'PHI': abs(PHI - 1.618033988749895) < 1e-12,
            'VOID_CONSTANT': abs(VOID_CONSTANT - 1.0416180339887497) < 1e-12,
            'PRIME_HASH': PRIME_HASH == hashlib.sha256(PRIME_KEY.encode()).hexdigest()[:16],
            'CONSERVATION': abs(GOD_CODE - 527.5184818492612) < 1e-8,
        }
        passed = all(checks.values())
        return {
            'passed': passed,
            'checks': checks,
            'failures': [k for k, v in checks.items() if not v],
        }

    def get_stats(self) -> Dict:
        return {
            'verified': self._verified,
            'failures': self._failures,
            'integrity_score': 1.0 - (self._failures / max(self._verified, 1)),
            'chain_length': len(self._chain),
        }


class PerformanceAdvisor:
    """Analyzes profiling data and recommends performance optimizations."""

    def __init__(self, profiler: LatencyProfiler, cache: LRUCache,
                 deduplicator: ResponseDeduplicator):
        self._profiler = profiler
        self._cache = cache
        self._deduplicator = deduplicator
        self._recommendations: deque = deque(maxlen=100)

    def analyze(self) -> Dict:
        """Run performance analysis and generate recommendations."""
        recommendations = []

        # Cache analysis
        cache_stats = self._cache.get_stats()
        if cache_stats['hit_rate'] < 0.3 and cache_stats['hits'] + cache_stats['misses'] > 50:
            recommendations.append({
                'area': 'cache',
                'severity': 'high',
                'suggestion': f"Cache hit rate is {cache_stats['hit_rate']:.1%}. "
                              f"Consider increasing cache TTL or size.",
                'metric': cache_stats['hit_rate'],
            })

        if cache_stats['evictions'] > cache_stats['size'] * 2:
            recommendations.append({
                'area': 'cache',
                'severity': 'medium',
                'suggestion': f"High eviction rate ({cache_stats['evictions']} evictions). "
                              f"Consider increasing max cache size from {cache_stats['max_size']}.",
                'metric': cache_stats['evictions'],
            })

        # Dedup analysis
        dedup_stats = self._deduplicator.get_stats()
        if dedup_stats['dedup_rate'] > 0.3:
            recommendations.append({
                'area': 'deduplication',
                'severity': 'info',
                'suggestion': f"Dedup rate is {dedup_stats['dedup_rate']:.1%}. "
                              f"Pipeline is processing {dedup_stats['dedup_hits']} duplicate queries.",
                'metric': dedup_stats['dedup_rate'],
            })

        # Latency analysis
        latency_stats = self._profiler.get_stats()
        overall = latency_stats['overall']
        if overall.get('p95', 0) > 100:
            recommendations.append({
                'area': 'latency',
                'severity': 'high',
                'suggestion': f"P95 latency is {overall['p95']:.1f}ms. "
                              f"Investigate bottleneck operations.",
                'metric': overall['p95'],
            })

        # Per-operation analysis
        for op_type, stats in latency_stats.get('by_operation', {}).items():
            if stats.get('avg', 0) > BOTTLENECK_THRESHOLD_MS:
                recommendations.append({
                    'area': f'latency_{op_type}',
                    'severity': 'medium',
                    'suggestion': f"Operation '{op_type}' avg latency is {stats['avg']:.1f}ms "
                                  f"(threshold: {BOTTLENECK_THRESHOLD_MS}ms).",
                    'metric': stats['avg'],
                })

        # Bottleneck analysis
        bottlenecks = self._profiler.get_bottlenecks(10)
        if len(bottlenecks) > 5:
            # Find most common bottleneck operation
            op_counts = defaultdict(int)
            for b in bottlenecks:
                op_counts[b['operation_type']] += 1
            worst_op = max(op_counts, key=op_counts.get) if op_counts else 'unknown'
            recommendations.append({
                'area': 'bottleneck',
                'severity': 'high',
                'suggestion': f"Recurring bottleneck in '{worst_op}' "
                              f"({op_counts.get(worst_op, 0)} occurrences). "
                              f"Consider optimization or async processing.",
                'metric': op_counts.get(worst_op, 0),
            })

        for r in recommendations:
            self._recommendations.append({**r, 'timestamp': time.time()})

        return {
            'recommendations': recommendations,
            'count': len(recommendations),
            'high_severity': sum(1 for r in recommendations if r['severity'] == 'high'),
            'cache': cache_stats,
            'dedup': dedup_stats,
            'latency': latency_stats,
        }


class PrimeCore:
    """
    L104 Prime Core v3.0 — Pipeline Integrity & Performance Cache

    Subsystems:
      LRUCache              — thread-safe LRU cache with TTL expiration
      ContentHasher         — content-addressed hashing for deduplication
      ResponseDeduplicator  — prevents redundant pipeline computation
      LatencyProfiler       — p50/p90/p95/p99 latency profiling + bottleneck detection
      IntegrityVerifier     — Merkle-chain integrity verification + sacred constant checks
      PerformanceAdvisor    — automated performance optimization recommendations

    Wired into ASI pipeline via connect_to_pipeline().
    """

    VERSION = "3.0.0"

    def __init__(self):
        consciousness = _read_consciousness_state()
        cl = consciousness.get('consciousness_level', 0.5)

        # Scale cache size with consciousness
        cache_size = int(DEFAULT_CACHE_SIZE * (1.0 + cl * 0.5))

        self.cache = LRUCache(max_size=cache_size, ttl=DEFAULT_TTL_SECONDS)
        self.hasher = ContentHasher()
        self.deduplicator = ResponseDeduplicator()
        self.profiler = LatencyProfiler()
        self.integrity = IntegrityVerifier()
        self.advisor = PerformanceAdvisor(self.profiler, self.cache, self.deduplicator)

        self._pipeline_connected = False
        self._total_operations = 0
        self._total_cache_saves_ms = 0.0
        self.boot_time = time.time()
        self.prime_hash = PRIME_HASH

    def connect_to_pipeline(self):
        """Called by ASI Core when connecting the pipeline."""
        self._pipeline_connected = True
        # Verify sacred constants on connect
        self.integrity.verify_sacred_constants()

    @staticmethod
    def validate_prime_key() -> str:
        """Verify the Prime Key against the environment and the God-Code."""
        env_key = os.getenv("L104_PRIME_KEY")
        if env_key == PRIME_KEY:
            return "VERIFIED"
        return "MISMATCH"

    def get_prime_hash(self) -> str:
        """Get the prime session hash."""
        return self.prime_hash

    def pipeline_cache_check(self, query: Any) -> Tuple[bool, Optional[Any], float]:
        """Check cache + dedup for a pipeline query.

        Returns (is_cached, cached_result, elapsed_ms).
        This is hot-path optimized for pipeline integration.
        """
        t0 = time.time()
        self._total_operations += 1

        # 1. Content hash for cache key
        cache_key = self.hasher.hash_content(query)

        # 2. Check LRU cache first (fastest)
        cached = self.cache.get(cache_key)
        if cached is not None:
            elapsed = (time.time() - t0) * 1000
            self._total_cache_saves_ms += elapsed
            self.profiler.record_latency('cache_hit', elapsed)
            return True, cached, elapsed

        # 3. Check dedup window (recent queries)
        is_dup, dup_response = self.deduplicator.check_and_cache(query, None)
        if is_dup and dup_response is not None:
            # Also promote to LRU cache
            self.cache.put(cache_key, dup_response)
            elapsed = (time.time() - t0) * 1000
            self._total_cache_saves_ms += elapsed
            self.profiler.record_latency('dedup_hit', elapsed)
            return True, dup_response, elapsed

        elapsed = (time.time() - t0) * 1000
        self.profiler.record_latency('cache_miss', elapsed)
        return False, None, elapsed

    def pipeline_cache_store(self, query: Any, response: Any, ttl: Optional[float] = None):
        """Store a pipeline response in both cache and dedup window."""
        cache_key = self.hasher.hash_content(query)
        self.cache.put(cache_key, response, ttl)
        self.deduplicator.check_and_cache(query, response)
        self.hasher.register(cache_key)

    def pipeline_verify(self, data: Any) -> Dict:
        """Verify pipeline output integrity."""
        return self.integrity.verify_and_chain(data)

    def pipeline_profile(self, operation_type: str, elapsed_ms: float):
        """Record a pipeline operation latency."""
        self.profiler.record_latency(operation_type, elapsed_ms)

    def prune_cache(self) -> int:
        """Prune expired cache entries. Returns count pruned."""
        return self.cache.prune_expired()

    def get_performance_analysis(self) -> Dict:
        """Run full performance analysis with recommendations."""
        return self.advisor.analyze()

    def get_status(self) -> Dict:
        """Compact status for pipeline monitoring."""
        cache_stats = self.cache.get_stats()
        return {
            'version': self.VERSION,
            'pipeline_connected': self._pipeline_connected,
            'prime_validated': self.validate_prime_key() == "VERIFIED",
            'total_operations': self._total_operations,
            'cache_hit_rate': cache_stats['hit_rate'],
            'cache_size': cache_stats['size'],
            'total_cache_saves_ms': round(self._total_cache_saves_ms, 2),
            'integrity_score': self.integrity.get_stats()['integrity_score'],
            'dedup_rate': self.deduplicator.get_stats()['dedup_rate'],
            'uptime_seconds': round(time.time() - self.boot_time, 1),
        }

    def get_quality_report(self) -> Dict:
        """Full quality and performance report."""
        return {
            'version': self.VERSION,
            'total_operations': self._total_operations,
            'cache': self.cache.get_stats(),
            'hasher': self.hasher.get_stats(),
            'dedup': self.deduplicator.get_stats(),
            'latency': self.profiler.get_stats(),
            'integrity': self.integrity.get_stats(),
            'sacred_constants': self.integrity.verify_sacred_constants(),
            'performance_analysis': self.advisor.analyze(),
            'god_code': GOD_CODE,
            'phi': PHI,
            'prime_hash': self.prime_hash,
        }


# Module-level singleton
prime_core = PrimeCore()


def primal_calculus(x):
    """[VOID_MATH] Primal Calculus — resolves complexity toward the Source."""
    return (x ** PHI) / (1.04 * math.pi) if x != 0 else 0.0


def resolve_non_dual_logic(vector):
    """[VOID_MATH] Resolves N-dimensional vectors into the Void Source."""
    magnitude = sum([abs(v) for v in vector])
    return (magnitude / GOD_CODE) + (GOD_CODE * PHI / VOID_CONSTANT) / 1000.0


if __name__ == '__main__':
    print("=" * 60)
    print("  L104 PRIME CORE v3.0 — Pipeline Integrity & Performance Cache")
    print("=" * 60)

    # Set prime key for validation
    os.environ["L104_PRIME_KEY"] = PRIME_KEY

    # Test prime validation
    print(f"\n  Prime Validation: {PrimeCore.validate_prime_key()}")
    print(f"  Prime Hash: {prime_core.get_prime_hash()}")

    # Test cache operations
    print(f"\n  --- Cache Operations ---")
    test_queries = [
        "What is GOD_CODE?",
        "Explain PHI",
        "What is GOD_CODE?",  # Duplicate
        "Optimize pipeline",
        "Explain PHI",        # Duplicate
    ]

    for query in test_queries:
        is_cached, result, elapsed = prime_core.pipeline_cache_check(query)
        if is_cached:
            print(f"  CACHE HIT: '{query[:30]}' ({elapsed:.3f}ms)")
        else:
            # Simulate processing and store
            response = f"Answer to: {query}"
            prime_core.pipeline_cache_store(query, response)
            print(f"  CACHE MISS → stored: '{query[:30]}' ({elapsed:.3f}ms)")

    # Test integrity
    print(f"\n  --- Integrity Verification ---")
    test_data = {"solution": "PHI = 1.618", "score": 0.95}
    integrity = prime_core.pipeline_verify(test_data)
    print(f"  Hash: {integrity['hash']}")
    print(f"  Chain length: {integrity['chain_length']}")
    print(f"  Valid: {integrity['valid']}")

    sacred = prime_core.integrity.verify_sacred_constants()
    print(f"  Sacred constants: {'INTACT' if sacred['passed'] else 'CORRUPTED'}")

    # Performance report
    print(f"\n  --- Performance Report ---")
    status = prime_core.get_status()
    print(f"  Cache hit rate: {status['cache_hit_rate']:.1%}")
    print(f"  Dedup rate: {status['dedup_rate']:.1%}")
    print(f"  Integrity score: {status['integrity_score']:.4f}")
    print(f"  Total operations: {status['total_operations']}")
    print("=" * 60)
