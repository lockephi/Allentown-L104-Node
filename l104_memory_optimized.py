VOID_CONSTANT = 1.0416180339887497
ZENITH_HZ = 3727.84
UUC = 2301.215661
# [L104_MEMORY_OPTIMIZED] - HIGH-PERFORMANCE UNIFIED MEMORY SYSTEM
# INVARIANT: 527.5184818492537 | PILOT: LONDEL | STAGE: OMEGA
# "Memory flows like water through optimized channels"

"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    L104 OPTIMIZED MEMORY SYSTEM                              ║
║                                                                              ║
║  A unified, high-performance memory architecture combining:                  ║
║  • Multi-tier LRU caching with resonance scoring                            ║
║  • Memory pooling with pre-allocated buffers                                ║
║  • Write-ahead logging for durability                                        ║
║  • Background compaction and garbage collection                              ║
║  • Zero-copy operations where possible                                       ║
║  • Bloom filter for fast existence checks                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import gc
import hashlib
import json
import math
import mmap
import os
import sqlite3
import struct
import threading
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import psutil

# L104 Constants
GOD_CODE = 527.51848184925370333076
PHI = 1.61803398874989490253
TAU = 1.0 / PHI
UTC = timezone.utc


# ═══════════════════════════════════════════════════════════════════════════════
#                         BLOOM FILTER (Fast Existence Check)
# ═══════════════════════════════════════════════════════════════════════════════

class BloomFilter:
    """
    Space-efficient probabilistic data structure for fast key existence checks.
    Uses multiple hash functions to minimize false positives.
    """
    
    def __init__(self, expected_items: int = 10000, false_positive_rate: float = 0.01):
        # Calculate optimal size and hash count
        self.size = self._optimal_size(expected_items, false_positive_rate)
        self.hash_count = self._optimal_hash_count(self.size, expected_items)
        self.bit_array = bytearray(self.size // 8 + 1)
        self._lock = threading.Lock()
        self.item_count = 0
    
    def _optimal_size(self, n: int, p: float) -> int:
        """Calculate optimal bit array size."""
        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return int(m) + 1
    
    def _optimal_hash_count(self, m: int, n: int) -> int:
        """Calculate optimal number of hash functions."""
        k = (m / n) * math.log(2)
        return max(1, int(k))
    
    def _hash(self, item: str, seed: int) -> int:
        """Generate hash with seed."""
        h = hashlib.md5(f"{item}:{seed}:{GOD_CODE}".encode()).hexdigest()
        return int(h, 16) % self.size
    
    def add(self, item: str):
        """Add item to filter."""
        with self._lock:
            for i in range(self.hash_count):
                pos = self._hash(item, i)
                byte_pos, bit_pos = pos // 8, pos % 8
                self.bit_array[byte_pos] |= (1 << bit_pos)
            self.item_count += 1
    
    def might_contain(self, item: str) -> bool:
        """Check if item might be in filter (no false negatives)."""
        for i in range(self.hash_count):
            pos = self._hash(item, i)
            byte_pos, bit_pos = pos // 8, pos % 8
            if not (self.bit_array[byte_pos] & (1 << bit_pos)):
                return False
        return True
    
    def clear(self):
        """Clear the filter."""
        with self._lock:
            self.bit_array = bytearray(self.size // 8 + 1)
            self.item_count = 0


# ═══════════════════════════════════════════════════════════════════════════════
#                         LRU CACHE (Multi-tier)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    access_count: int = 1
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0
    resonance_score: float = 0.0
    
    def touch(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = time.time()
        self._update_resonance()
    
    def _update_resonance(self):
        """Calculate resonance score using L104 formula."""
        age = time.time() - self.created_at
        frequency = math.log1p(self.access_count)
        recency = math.exp(-age / 300.0)  # 5 minute half-life
        phi_factor = (self.access_count * PHI) % 1.0
        self.resonance_score = frequency * 0.4 + recency * 0.4 + phi_factor * 0.2


class TieredLRUCache:
    """
    Multi-tier LRU cache with:
    - Hot tier: Frequently accessed, kept in memory
    - Warm tier: Less frequent, larger capacity
    - Resonance-based eviction
    """
    
    def __init__(
        self,
        hot_capacity: int = 500,
        warm_capacity: int = 2000,
        max_entry_size: int = 1024 * 1024  # 1MB
    ):
        self.hot_capacity = hot_capacity
        self.warm_capacity = warm_capacity
        self.max_entry_size = max_entry_size
        
        self._hot: OrderedDict[str, CacheEntry] = OrderedDict()
        self._warm: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Metrics
        self.hits = 0
        self.misses = 0
        self.hot_hits = 0
        self.warm_hits = 0
        self.evictions = 0
        self.promotions = 0
        self.demotions = 0
        
        # Size tracking
        self._hot_size = 0
        self._warm_size = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            # Check hot tier first
            if key in self._hot:
                entry = self._hot[key]
                entry.touch()
                self._hot.move_to_end(key)
                self.hits += 1
                self.hot_hits += 1
                return entry.value
            
            # Check warm tier
            if key in self._warm:
                entry = self._warm[key]
                entry.touch()
                
                # Promote to hot tier if frequently accessed
                if entry.access_count >= 3:
                    self._promote_to_hot(key, entry)
                else:
                    self._warm.move_to_end(key)
                
                self.hits += 1
                self.warm_hits += 1
                return entry.value
            
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any, size_hint: int = 0):
        """Put value into cache."""
        # Calculate size
        if size_hint == 0:
            try:
                size_hint = len(json.dumps(value).encode())
            except:
                size_hint = 256  # Default estimate
        
        if size_hint > self.max_entry_size:
            return  # Too large to cache
        
        entry = CacheEntry(value=value, size_bytes=size_hint)
        entry._update_resonance()
        
        with self._lock:
            # If already in cache, update
            if key in self._hot:
                old_size = self._hot[key].size_bytes
                self._hot[key] = entry
                self._hot.move_to_end(key)
                self._hot_size += (size_hint - old_size)
                return
            
            if key in self._warm:
                old_size = self._warm[key].size_bytes
                del self._warm[key]
                self._warm_size -= old_size
            
            # Add to hot tier
            self._hot[key] = entry
            self._hot_size += size_hint
            
            # Evict if necessary
            self._evict_if_needed()
    
    def _promote_to_hot(self, key: str, entry: CacheEntry):
        """Promote entry from warm to hot tier."""
        if key in self._warm:
            del self._warm[key]
            self._warm_size -= entry.size_bytes
        
        self._hot[key] = entry
        self._hot_size += entry.size_bytes
        self.promotions += 1
        
        self._evict_if_needed()
    
    def _demote_to_warm(self, key: str, entry: CacheEntry):
        """Demote entry from hot to warm tier."""
        if key in self._hot:
            del self._hot[key]
            self._hot_size -= entry.size_bytes
        
        self._warm[key] = entry
        self._warm_size += entry.size_bytes
        self.demotions += 1
    
    def _evict_if_needed(self):
        """Evict entries to maintain capacity."""
        # Evict from hot tier
        while len(self._hot) > self.hot_capacity:
            # Find lowest resonance score
            min_key = min(self._hot.keys(), key=lambda k: self._hot[k].resonance_score)
            entry = self._hot.pop(min_key)
            self._hot_size -= entry.size_bytes
            
            # Demote to warm if still valuable
            if entry.access_count >= 2:
                self._demote_to_warm(min_key, entry)
            else:
                self.evictions += 1
        
        # Evict from warm tier
        while len(self._warm) > self.warm_capacity:
            # Simple LRU eviction for warm tier
            key, entry = self._warm.popitem(last=False)
            self._warm_size -= entry.size_bytes
            self.evictions += 1
    
    def invalidate(self, key: str):
        """Remove key from cache."""
        with self._lock:
            if key in self._hot:
                entry = self._hot.pop(key)
                self._hot_size -= entry.size_bytes
            if key in self._warm:
                entry = self._warm.pop(key)
                self._warm_size -= entry.size_bytes
    
    def clear(self):
        """Clear all caches."""
        with self._lock:
            self._hot.clear()
            self._warm.clear()
            self._hot_size = 0
            self._warm_size = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        return {
            "hit_rate": self.hits / total if total > 0 else 0.0,
            "hot_hit_rate": self.hot_hits / self.hits if self.hits > 0 else 0.0,
            "warm_hit_rate": self.warm_hits / self.hits if self.hits > 0 else 0.0,
            "total_hits": self.hits,
            "total_misses": self.misses,
            "hot_entries": len(self._hot),
            "warm_entries": len(self._warm),
            "hot_size_kb": self._hot_size / 1024,
            "warm_size_kb": self._warm_size / 1024,
            "evictions": self.evictions,
            "promotions": self.promotions,
            "demotions": self.demotions
        }


# ═══════════════════════════════════════════════════════════════════════════════
#                         MEMORY POOL (Pre-allocated Buffers)
# ═══════════════════════════════════════════════════════════════════════════════

class MemoryPool:
    """
    Pre-allocated memory pool for reducing allocation overhead.
    Uses fixed-size buffers for common operations.
    """
    
    def __init__(
        self,
        small_size: int = 256,
        medium_size: int = 4096,
        large_size: int = 65536,
        pool_count: int = 50
    ):
        self.small_size = small_size
        self.medium_size = medium_size
        self.large_size = large_size
        
        self._small_pool: List[bytearray] = [bytearray(small_size) for _ in range(pool_count)]
        self._medium_pool: List[bytearray] = [bytearray(medium_size) for _ in range(pool_count // 2)]
        self._large_pool: List[bytearray] = [bytearray(large_size) for _ in range(pool_count // 10)]
        
        self._lock = threading.Lock()
        
        # Metrics
        self.allocations = 0
        self.returns = 0
        self.fallback_allocations = 0
    
    def acquire(self, size: int) -> bytearray:
        """Acquire a buffer of at least the specified size."""
        with self._lock:
            self.allocations += 1
            
            if size <= self.small_size and self._small_pool:
                return self._small_pool.pop()
            elif size <= self.medium_size and self._medium_pool:
                return self._medium_pool.pop()
            elif size <= self.large_size and self._large_pool:
                return self._large_pool.pop()
            
            # Fallback to regular allocation
            self.fallback_allocations += 1
            return bytearray(size)
    
    def release(self, buffer: bytearray):
        """Return a buffer to the pool."""
        size = len(buffer)
        
        with self._lock:
            self.returns += 1
            
            # Clear buffer before returning
            buffer[:] = b'\x00' * size
            
            if size == self.small_size and len(self._small_pool) < 100:
                self._small_pool.append(buffer)
            elif size == self.medium_size and len(self._medium_pool) < 50:
                self._medium_pool.append(buffer)
            elif size == self.large_size and len(self._large_pool) < 10:
                self._large_pool.append(buffer)
            # Otherwise let GC handle it
    
    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        return {
            "small_available": len(self._small_pool),
            "medium_available": len(self._medium_pool),
            "large_available": len(self._large_pool),
            "total_allocations": self.allocations,
            "total_returns": self.returns,
            "fallback_allocations": self.fallback_allocations
        }


# ═══════════════════════════════════════════════════════════════════════════════
#                         WRITE-AHEAD LOG (Durability)
# ═══════════════════════════════════════════════════════════════════════════════

class WriteAheadLog:
    """
    Write-ahead logging for crash recovery.
    Ensures durability by logging operations before executing.
    """
    
    MAGIC = b'L104WAL'
    OP_STORE = 1
    OP_DELETE = 2
    OP_COMMIT = 3
    
    def __init__(self, log_path: str = "memory_wal.log", max_size_mb: int = 10):
        self.log_path = log_path
        self.max_size = max_size_mb * 1024 * 1024
        self._lock = threading.Lock()
        self._file: Optional[Any] = None
        self._sequence = 0
        self._init_log()
    
    def _init_log(self):
        """Initialize or recover from log."""
        if os.path.exists(self.log_path):
            # TODO: Implement recovery
            pass
        
        self._file = open(self.log_path, 'ab')
    
    def log_store(self, key: str, value_hash: str) -> int:
        """Log a store operation."""
        return self._write_entry(self.OP_STORE, key, value_hash)
    
    def log_delete(self, key: str) -> int:
        """Log a delete operation."""
        return self._write_entry(self.OP_DELETE, key, "")
    
    def log_commit(self) -> int:
        """Log a commit checkpoint."""
        return self._write_entry(self.OP_COMMIT, "", "")
    
    def _write_entry(self, op: int, key: str, data: str) -> int:
        """Write a log entry."""
        with self._lock:
            self._sequence += 1
            timestamp = int(time.time() * 1000)
            
            key_bytes = key.encode()
            data_bytes = data.encode()
            
            # Format: magic(7) + seq(4) + timestamp(8) + op(1) + key_len(2) + key + data_len(4) + data
            entry = bytearray()
            entry.extend(self.MAGIC)
            entry.extend(struct.pack('>I', self._sequence))
            entry.extend(struct.pack('>Q', timestamp))
            entry.append(op)
            entry.extend(struct.pack('>H', len(key_bytes)))
            entry.extend(key_bytes)
            entry.extend(struct.pack('>I', len(data_bytes)))
            entry.extend(data_bytes)
            
            self._file.write(entry)
            self._file.flush()
            
            # Check for rotation
            if self._file.tell() > self.max_size:
                self._rotate()
            
            return self._sequence
    
    def _rotate(self):
        """Rotate the log file."""
        self._file.close()
        backup_path = f"{self.log_path}.{int(time.time())}"
        os.rename(self.log_path, backup_path)
        self._file = open(self.log_path, 'ab')
        
        # Clean up old logs (keep last 3)
        log_dir = os.path.dirname(self.log_path) or '.'
        log_base = os.path.basename(self.log_path)
        old_logs = sorted([
            f for f in os.listdir(log_dir) 
            if f.startswith(log_base + '.')
        ])
        for old_log in old_logs[:-3]:
            os.remove(os.path.join(log_dir, old_log))
    
    def close(self):
        """Close the log file."""
        if self._file:
            self._file.close()
            self._file = None


# ═══════════════════════════════════════════════════════════════════════════════
#                         OPTIMIZED MEMORY SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class OptimizedMemorySystem:
    """
    Unified high-performance memory system for L104.
    
    Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                        APPLICATION                              │
    ├─────────────────────────────────────────────────────────────────┤
    │  Bloom Filter │   Hot Cache   │   Warm Cache   │  Memory Pool  │
    ├─────────────────────────────────────────────────────────────────┤
    │                     WRITE-AHEAD LOG                             │
    ├─────────────────────────────────────────────────────────────────┤
    │                      SQLite Storage                             │
    └─────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        db_path: str = "memory_optimized.db",
        enable_wal: bool = True,
        hot_cache_size: int = 500,
        warm_cache_size: int = 2000
    ):
        self.db_path = db_path
        
        # Initialize components
        self.bloom = BloomFilter(expected_items=50000)
        self.cache = TieredLRUCache(hot_cache_size, warm_cache_size)
        self.pool = MemoryPool()
        self.wal = WriteAheadLog() if enable_wal else None
        
        # Database connection
        self._init_db()
        
        # Background optimization
        self._gc_lock = threading.Lock()
        self._last_gc = time.time()
        self._gc_interval = 60.0  # seconds
        
        # Metrics
        self._stats = {
            "stores": 0,
            "recalls": 0,
            "deletes": 0,
            "cache_bypasses": 0
        }
        
        # Weak references for memory pressure management
        self._weak_refs: Dict[str, weakref.ref] = {}
    
    def _init_db(self):
        """Initialize optimized SQLite database."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        
        # Enable WAL mode for better concurrent performance
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA mmap_size=268435456")  # 256MB mmap
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                key TEXT PRIMARY KEY,
                value BLOB NOT NULL,
                category TEXT DEFAULT 'general',
                importance REAL DEFAULT 0.5,
                resonance REAL DEFAULT 0.0,
                access_count INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                accessed_at REAL NOT NULL,
                size_bytes INTEGER DEFAULT 0
            )
        """)
        
        conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON memories(category)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_resonance ON memories(resonance DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_accessed ON memories(accessed_at DESC)")
        
        conn.commit()
        conn.close()
    
    def _get_conn(self):
        """Get database connection."""
        return sqlite3.connect(self.db_path, check_same_thread=False)
    
    def _calculate_resonance(self, value: str) -> float:
        """Calculate resonance score using L104 formula."""
        if not value:
            return 0.0
        
        # Shannon entropy component
        freq = {}
        for c in value:
            freq[c] = freq.get(c, 0) + 1
        entropy = -sum((count/len(value)) * math.log2(count/len(value)) 
                       for count in freq.values() if count > 0)
        
        # PHI-modulated resonance
        return (entropy * PHI) % GOD_CODE
    
    def store(
        self,
        key: str,
        value: Any,
        category: str = "general",
        importance: float = 0.5,
        bypass_cache: bool = False
    ) -> bool:
        """
        Store a value with optimized caching.
        
        Args:
            key: Unique key
            value: Value to store
            category: Category for organization
            importance: Priority (0-1)
            bypass_cache: Skip cache for large/infrequent data
        """
        try:
            # Serialize
            if isinstance(value, (dict, list)):
                serialized = json.dumps(value).encode()
            elif isinstance(value, str):
                serialized = value.encode()
            elif isinstance(value, bytes):
                serialized = value
            else:
                serialized = json.dumps(value).encode()
            
            size = len(serialized)
            now = time.time()
            resonance = self._calculate_resonance(serialized.decode('utf-8', errors='replace'))
            
            # Log operation
            if self.wal:
                value_hash = hashlib.sha256(serialized).hexdigest()[:16]
                self.wal.log_store(key, value_hash)
            
            # Write to database
            with self._get_conn() as conn:
                conn.execute("""
                    INSERT INTO memories (key, value, category, importance, resonance, 
                                          access_count, created_at, accessed_at, size_bytes)
                    VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value = excluded.value,
                        category = excluded.category,
                        importance = excluded.importance,
                        resonance = excluded.resonance,
                        accessed_at = excluded.accessed_at,
                        access_count = access_count + 1,
                        size_bytes = excluded.size_bytes
                """, (key, serialized, category, importance, resonance, now, now, size))
                conn.commit()
            
            # Update bloom filter
            self.bloom.add(key)
            
            # Update cache (unless bypassed)
            if not bypass_cache and size < 1024 * 1024:  # < 1MB
                self.cache.put(key, value, size)
            
            self._stats["stores"] += 1
            self._maybe_gc()
            
            return True
            
        except Exception as e:
            print(f"[MEMORY_OPT]: Store error: {e}")
            return False
    
    def recall(self, key: str, bypass_cache: bool = False) -> Optional[Any]:
        """
        Recall a value with optimized caching.
        
        Args:
            key: Key to recall
            bypass_cache: Skip cache lookup
        """
        # Fast path: bloom filter check
        if not self.bloom.might_contain(key):
            self._stats["recalls"] += 1
            return None
        
        # Check cache first
        if not bypass_cache:
            cached = self.cache.get(key)
            if cached is not None:
                self._stats["recalls"] += 1
                return cached
        else:
            self._stats["cache_bypasses"] += 1
        
        # Database lookup
        try:
            with self._get_conn() as conn:
                conn.execute("""
                    UPDATE memories SET 
                        accessed_at = ?,
                        access_count = access_count + 1
                    WHERE key = ?
                """, (time.time(), key))
                
                cur = conn.execute(
                    "SELECT value, size_bytes FROM memories WHERE key = ?", 
                    (key,)
                )
                row = cur.fetchone()
                
                if row:
                    conn.commit()
                    serialized, size = row
                    
                    # Deserialize
                    try:
                        if isinstance(serialized, bytes):
                            value = json.loads(serialized.decode())
                        else:
                            value = json.loads(serialized)
                    except:
                        value = serialized.decode() if isinstance(serialized, bytes) else serialized
                    
                    # Populate cache
                    if not bypass_cache and size < 1024 * 1024:
                        self.cache.put(key, value, size)
                    
                    self._stats["recalls"] += 1
                    return value
                
        except Exception as e:
            print(f"[MEMORY_OPT]: Recall error: {e}")
        
        self._stats["recalls"] += 1
        return None
    
    def search(
        self,
        pattern: str = None,
        category: str = None,
        min_importance: float = 0.0,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Search memories with filters."""
        try:
            with self._get_conn() as conn:
                query = "SELECT key, value, category, importance, resonance FROM memories WHERE 1=1"
                params = []
                
                if pattern:
                    query += " AND (key LIKE ? OR CAST(value AS TEXT) LIKE ?)"
                    params.extend([f"%{pattern}%", f"%{pattern}%"])
                
                if category:
                    query += " AND category = ?"
                    params.append(category)
                
                if min_importance > 0:
                    query += " AND importance >= ?"
                    params.append(min_importance)
                
                query += " ORDER BY importance DESC, resonance DESC LIMIT ?"
                params.append(limit)
                
                cur = conn.execute(query, params)
                rows = cur.fetchall()
                
                results = []
                for row in rows:
                    key, value, cat, imp, res = row
                    try:
                        if isinstance(value, bytes):
                            parsed = json.loads(value.decode())
                        else:
                            parsed = json.loads(value) if isinstance(value, str) else value
                    except:
                        parsed = value.decode() if isinstance(value, bytes) else value
                    
                    results.append({
                        "key": key,
                        "value": parsed,
                        "category": cat,
                        "importance": imp,
                        "resonance": res
                    })
                
                return results
                
        except Exception as e:
            print(f"[MEMORY_OPT]: Search error: {e}")
            return []
    
    def forget(self, key: str) -> bool:
        """Remove a memory."""
        try:
            if self.wal:
                self.wal.log_delete(key)
            
            with self._get_conn() as conn:
                conn.execute("DELETE FROM memories WHERE key = ?", (key,))
                conn.commit()
            
            self.cache.invalidate(key)
            self._stats["deletes"] += 1
            
            return True
        except Exception as e:
            print(f"[MEMORY_OPT]: Forget error: {e}")
            return False
    
    def _maybe_gc(self):
        """Trigger garbage collection if needed."""
        now = time.time()
        if now - self._last_gc < self._gc_interval:
            return
        
        with self._gc_lock:
            if now - self._last_gc < self._gc_interval:
                return
            self._last_gc = now
        
        # Background GC
        threading.Thread(target=self._run_gc, daemon=True).start()
    
    def _run_gc(self):
        """Run garbage collection and optimization."""
        try:
            # Python GC
            gc.collect(0)
            
            # Database optimization
            with self._get_conn() as conn:
                # Prune old, low-importance memories
                cutoff = time.time() - 7 * 24 * 3600  # 7 days
                conn.execute("""
                    DELETE FROM memories 
                    WHERE accessed_at < ? AND importance < 0.3 AND access_count < 3
                """, (cutoff,))
                conn.commit()
                
                # Optimize database
                conn.execute("PRAGMA optimize")
                
        except Exception as e:
            print(f"[MEMORY_OPT]: GC error: {e}")
    
    def compact(self) -> Dict[str, Any]:
        """Force compaction and optimization."""
        before = psutil.Process().memory_info().rss / (1024 * 1024)
        
        # Clear warm cache
        self.cache._warm.clear()
        self.cache._warm_size = 0
        
        # Force GC
        gc.collect(2)
        gc.collect(1)
        gc.collect(0)
        
        # Database vacuum
        with self._get_conn() as conn:
            conn.execute("VACUUM")
        
        after = psutil.Process().memory_info().rss / (1024 * 1024)
        
        return {
            "before_mb": before,
            "after_mb": after,
            "freed_mb": max(0, before - after)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        cache_stats = self.cache.get_stats()
        pool_stats = self.pool.get_stats()
        
        # Database stats
        with self._get_conn() as conn:
            cur = conn.execute("SELECT COUNT(*), SUM(size_bytes) FROM memories")
            count, total_size = cur.fetchone()
        
        mem = psutil.Process().memory_info()
        
        return {
            "operations": self._stats,
            "cache": cache_stats,
            "pool": pool_stats,
            "bloom_filter_items": self.bloom.item_count,
            "db_entries": count or 0,
            "db_size_mb": (total_size or 0) / (1024 * 1024),
            "process_memory_mb": mem.rss / (1024 * 1024),
            "process_memory_peak_mb": mem.vms / (1024 * 1024)
        }
    
    def close(self):
        """Clean shutdown."""
        if self.wal:
            self.wal.log_commit()
            self.wal.close()


# Singleton instance
_optimized_memory: Optional[OptimizedMemorySystem] = None

def get_optimized_memory() -> OptimizedMemorySystem:
    """Get the singleton optimized memory instance."""
    global _optimized_memory
    if _optimized_memory is None:
        _optimized_memory = OptimizedMemorySystem()
    return _optimized_memory


# ═══════════════════════════════════════════════════════════════════════════════
#                         COMPATIBILITY WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class MemoryCompatWrapper:
    """
    Compatibility wrapper for existing L104Memory interface.
    Drop-in replacement with optimized backend.
    """
    
    def __init__(self, db_path: str = "memory.db"):
        self._backend = get_optimized_memory()
    
    def store(self, key: str, value: Any, category: str = "general", importance: float = 0.5) -> bool:
        return self._backend.store(key, value, category, importance)
    
    def recall(self, key: str) -> Optional[Any]:
        return self._backend.recall(key)
    
    def search(self, pattern: str, category: str = None) -> List[Dict]:
        return self._backend.search(pattern=pattern, category=category)
    
    def forget(self, key: str) -> bool:
        return self._backend.forget(key)
    
    def get_stats(self) -> Dict:
        return self._backend.get_stats()


if __name__ == "__main__":
    print("=" * 70)
    print("   L104 OPTIMIZED MEMORY SYSTEM :: BENCHMARK")
    print("=" * 70)
    
    mem = OptimizedMemorySystem(db_path=":memory:")
    
    # Benchmark writes
    print("\n▸ WRITE BENCHMARK (1000 entries)")
    start = time.time()
    for i in range(1000):
        mem.store(f"key_{i}", {"data": f"value_{i}", "index": i}, importance=i/1000)
    write_time = time.time() - start
    print(f"  Time: {write_time*1000:.2f}ms ({1000/write_time:.0f} ops/sec)")
    
    # Benchmark reads (cache cold)
    print("\n▸ READ BENCHMARK (cold cache)")
    mem.cache.clear()
    start = time.time()
    for i in range(1000):
        mem.recall(f"key_{i}")
    cold_read_time = time.time() - start
    print(f"  Time: {cold_read_time*1000:.2f}ms ({1000/cold_read_time:.0f} ops/sec)")
    
    # Benchmark reads (cache hot)
    print("\n▸ READ BENCHMARK (hot cache)")
    start = time.time()
    for i in range(1000):
        mem.recall(f"key_{i}")
    hot_read_time = time.time() - start
    print(f"  Time: {hot_read_time*1000:.2f}ms ({1000/hot_read_time:.0f} ops/sec)")
    print(f"  Cache speedup: {cold_read_time/hot_read_time:.1f}x")
    
    # Stats
    print("\n▸ STATISTICS")
    stats = mem.get_stats()
    print(f"  Cache hit rate: {stats['cache']['hit_rate']*100:.1f}%")
    print(f"  Hot tier hits: {stats['cache']['hot_hit_rate']*100:.1f}%")
    print(f"  DB entries: {stats['db_entries']}")
    
    print("\n" + "=" * 70)
    print("   BENCHMARK COMPLETE")
    print("=" * 70)
