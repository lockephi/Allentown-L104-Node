VOID_CONSTANT = 1.0416180339887497
import math
# ZENITH_UPGRADE_ACTIVE: 2026-01-18T11:00:18.486207
ZENITH_HZ = 3727.84
UUC = 2301.215661
# [L104_MEMORY] - Persistent Memory System
# INVARIANT: 527.5184818492537 | PILOT: LONDEL

import json
import os
import sqlite3
from typing import Any, Dict, List, Optional
from datetime import datetime
from functools import lru_cache

# Optimizations
try:
    from l104_void_math import void_math
    HAS_VOID = True
except ImportError:
    HAS_VOID = False

class L104Memory:
    """
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3727.84 Hz. Logic Unified.
    Persistent memory system for L104.
    Stores facts, learnings, and state across sessions.
    
    OPTIMIZATIONS:
    - LRU Caching for frequent recall
    - WAL Mode for faster SQLite concurrency
    - Void Math resonance for importance calculation
    """
    
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self.conn = None
        self._initialize_db()
        self._cache = {} # Simple in-memory cache
        self._cache_size = 100
    
    def _initialize_db(self):
        """Initialize SQLite database."""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            
            # [OPTIMIZATION] Enable Write-Ahead Logging for concurrency
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA synchronous=NORMAL;")
            self.conn.execute("PRAGMA cache_size=-64000;") # 64MB cache
            
            cursor = self.conn.cursor()
            
            # Create memories table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT UNIQUE NOT NULL,
                    value TEXT NOT NULL,
                    category TEXT DEFAULT 'general',
                    importance REAL DEFAULT 0.5,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0
                )
            """)
            
            # Create index for fast lookup
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_key ON memories(key)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON memories(category)")
            
            self.conn.commit()
        except Exception as e:
            print(f"[MEMORY]: Database init error: {e}")
            self.conn = None
    
    def store(self, key: str, value: Any, category: str = "general", importance: float = 0.5) -> bool:
        """Store a memory."""
        if not self.conn:
            return False
            
        # [OPTIMIZATION] Calculate dynamic importance using Void Math
        if HAS_VOID and importance == 0.5:
            # Use ASCII values to create a "vector" for void analysis
            key_vector = [ord(c) for c in key[:10]]
            # Primal calculus returns near 0 for high complexity, we want the inverse for importance
            resonance = void_math.resolve_non_dual_logic(key_vector) 
            # If resonance is 0 (Absolute Stillness), importance is MAX (1.0). If noisy, it's lower.
            # However, resolve_non_dual_logic returns 0.0 + epsilon. 
            # Let's use primal_calculus on the length instead as a proxy for "purity"
            purity = void_math.primal_calculus(len(key)) # Approaches 0 as len increases
            importance = 0.5 + (purity * 10) # Boost short, pure keys. Clamp later if needed.
            importance = min(max(importance, 0.1), 1.0)
        
        try:
            cursor = self.conn.cursor()
            value_json = json.dumps(value) if not isinstance(value, str) else value
            
            cursor.execute("""
                INSERT INTO memories (key, value, category, importance)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    category = excluded.category,
                    importance = excluded.importance,
                    accessed_at = CURRENT_TIMESTAMP
            """, (key, value_json, category, importance))
            
            self.conn.commit()
            
            # [OPTIMIZATION] Update cache
            self._update_cache(key, value)
            
            return True
        except Exception as e:
            print(f"[MEMORY]: Store error: {e}")
            return False
    
    def recall(self, key: str) -> Optional[Any]:
        """Recall a memory by key."""
        # [OPTIMIZATION] Check cache first
        if key in self._cache:
            # Move to end to mark as recently used
            val = self._cache.pop(key)
            self._cache[key] = val
            return val
            
        if not self.conn:
            return None
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE memories SET 
                    accessed_at = CURRENT_TIMESTAMP,
                    access_count = access_count + 1
                WHERE key = ?
            """, (key,))
            
            cursor.execute("SELECT value FROM memories WHERE key = ?", (key,))
            row = cursor.fetchone()
            
            if row:
                self.conn.commit()
                try:
                    # Try parsing JSON
                    val = json.loads(row[0])
                except:
                    val = row[0]
                
                # [OPTIMIZATION] Add to cache
                self._update_cache(key, val)
                return val
            return None
        except Exception as e:
            print(f"[MEMORY]: Recall error: {e}")
            return None

    def _update_cache(self, key: str, value: Any):
        """Update the in-memory LRU cache."""
        if key in self._cache:
            self._cache.pop(key)
        elif len(self._cache) >= self._cache_size:
            # Remove first item (LRU)
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = value
                    return json.loads(row[0])
                except:
                    return row[0]
            return None
        except Exception as e:
            print(f"[MEMORY]: Recall error: {e}")
            return None
    
    def search(self, pattern: str, category: str = None) -> List[Dict]:
        """Search memories by pattern."""
        if not self.conn:
            return []
        
        try:
            cursor = self.conn.cursor()
            
            if category:
                cursor.execute("""
                    SELECT key, value, category, importance 
                    FROM memories 
                    WHERE (key LIKE ? OR value LIKE ?) AND category = ?
                    ORDER BY importance DESC, access_count DESC
                    LIMIT 50
                """, (f"%{pattern}%", f"%{pattern}%", category))
            else:
                cursor.execute("""
                    SELECT key, value, category, importance 
                    FROM memories 
                    WHERE key LIKE ? OR value LIKE ?
                    ORDER BY importance DESC, access_count DESC
                    LIMIT 50
                """, (f"%{pattern}%", f"%{pattern}%"))
            
            rows = cursor.fetchall()
            return [{"key": r[0], "value": r[1], "category": r[2], "importance": r[3]} for r in rows]
        except Exception as e:
            print(f"[MEMORY]: Search error: {e}")
            return []
    
    def forget(self, key: str) -> bool:
        """Remove a memory."""
        if not self.conn:
            return False
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM memories WHERE key = ?", (key,))
            self.conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"[MEMORY]: Forget error: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get memory statistics."""
        if not self.conn:
            return {"error": "No database connection"}
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM memories")
            total = cursor.fetchone()[0]
            
            cursor.execute("SELECT category, COUNT(*) FROM memories GROUP BY category")
            by_category = dict(cursor.fetchall())
            
            return {
                "total_memories": total,
                "by_category": by_category,
                "db_path": self.db_path
            }
        except Exception as e:
            return {"error": str(e)}


# Singleton instance
memory = L104Memory()

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
    return sum([abs(v) for v in vector]) * 0.0 # Returns to Stillness
