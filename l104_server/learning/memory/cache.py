"""
Memory Cache Mixin for LearningIntellect

Extracted from intellect.py during EVO_78 refactoring.
Contains: Database initialization, cache loading, persistence operations.
"""

import sqlite3
import threading
import time
import pickle
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict


def optimize_sqlite_connection(conn: sqlite3.Connection):
    """Optimize SQLite connection for performance."""
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')
    conn.execute('PRAGMA cache_size=-64000')  # 64MB cache
    conn.execute('PRAGMA temp_store=MEMORY')
    conn.execute('PRAGMA mmap_size=268435456')  # 256MB mmap
    conn.execute('PRAGMA page_size=4096')
    conn.execute('PRAGMA busy_timeout=30000')


class MemoryCacheMixin:
    """Memory and cache operations for LearningIntellect.
    
    This mixin provides:
    - Database initialization and schema management
    - Cache loading and optimization
    - Memory persistence and compression
    - Thread-local connection management
    
    Usage:
        class LearningIntellect(MemoryCacheMixin, ...):
            pass
    """
    
    _thread_local_conn = threading.local()
    
    def _init_db(self, db_path: str):
        """Initialize persistent memory database."""
        conn = sqlite3.connect(db_path)
        try:
            c = conn.cursor()

            # Core memory table - stores learned Q&A pairs
            c.execute('''CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY,
                query_hash TEXT UNIQUE,
                query TEXT,
                response TEXT,
                source TEXT,
                quality_score REAL DEFAULT 1.0,
                access_count INTEGER DEFAULT 0,
                created_at TEXT,
                updated_at TEXT
            )''')

            # Pattern table - learned linguistic patterns
            c.execute('''CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY,
                pattern TEXT UNIQUE,
                response_template TEXT,
                weight REAL DEFAULT 1.0,
                success_count INTEGER DEFAULT 0
            )''')

            # Knowledge graph - concept associations
            c.execute('''CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY,
                concept TEXT,
                related_concept TEXT,
                strength REAL DEFAULT 1.0,
                UNIQUE(concept, related_concept)
            )''')

            # Conversation log - full learning history
            c.execute('''CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                user_message TEXT,
                response TEXT,
                model_used TEXT,
                quality_indicator REAL
            )''')

            # Theorems table - high-level synthesized insights
            c.execute('''CREATE TABLE IF NOT EXISTS theorems (
                id INTEGER PRIMARY KEY,
                title TEXT UNIQUE,
                content TEXT,
                resonance_level REAL,
                created_at TEXT
            )''')

            # Meta-learning table - tracks what response strategies work best
            c.execute('''CREATE TABLE IF NOT EXISTS meta_learning (
                id INTEGER PRIMARY KEY,
                query_pattern TEXT UNIQUE,
                strategy_used TEXT,
                success_score REAL DEFAULT 0.5,
                usage_count INTEGER DEFAULT 1,
                last_used TEXT
            )''')

            # Feedback table - user response signals
            c.execute('''CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY,
                query_hash TEXT,
                response_hash TEXT,
                feedback_type TEXT,
                timestamp TEXT
            )''')

            # Query rewrites table
            c.execute('''CREATE TABLE IF NOT EXISTS query_rewrites (
                id INTEGER PRIMARY KEY,
                original_pattern TEXT UNIQUE,
                improved_pattern TEXT,
                success_rate REAL DEFAULT 0.5
            )''')

            # Concept clusters table
            c.execute('''CREATE TABLE IF NOT EXISTS concept_clusters (
                id INTEGER PRIMARY KEY,
                cluster_name TEXT UNIQUE,
                members BLOB,
                representative TEXT,
                member_count INTEGER,
                created_at TEXT,
                updated_at TEXT
            )''')

            # Consciousness state table
            c.execute('''CREATE TABLE IF NOT EXISTS consciousness_state (
                id INTEGER PRIMARY KEY,
                dimension TEXT UNIQUE,
                concepts BLOB,
                strength REAL DEFAULT 0.5,
                activation_count INTEGER DEFAULT 0,
                last_update TEXT
            )''')

            # Skills table
            c.execute('''CREATE TABLE IF NOT EXISTS skills (
                id INTEGER PRIMARY KEY,
                skill_name TEXT UNIQUE,
                level REAL DEFAULT 0.5,
                experience INTEGER DEFAULT 0,
                last_used TEXT
            )''')

            # Create indexes
            c.execute('CREATE INDEX IF NOT EXISTS idx_memory_quality ON memory(quality_score DESC)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_memory_access ON memory(access_count DESC)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_knowledge_concept ON knowledge(concept)')
            
            # Embeddings table
            c.execute('''CREATE TABLE IF NOT EXISTS embeddings (
                query_hash TEXT PRIMARY KEY,
                embedding BLOB,
                created_at TEXT
            )''')
            c.execute('CREATE INDEX IF NOT EXISTS idx_embeddings_hash ON embeddings(query_hash)')
            
            # Additional performance indexes
            c.execute('CREATE INDEX IF NOT EXISTS idx_memory_hash ON memory(query_hash)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_knowledge_related ON knowledge(related_concept)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_patterns_pattern ON patterns(pattern)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_memory_created ON memory(created_at DESC)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_knowledge_strength ON knowledge(strength DESC)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_memory_quality_access ON memory(quality_score DESC, access_count DESC)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_knowledge_concept_strength ON knowledge(concept, strength DESC)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_memory_source ON memory(source)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_knowledge_related_strength ON knowledge(related_concept, strength DESC)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_memory_hash_quality ON memory(query_hash, quality_score DESC)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_knowledge_both_concepts ON knowledge(concept, related_concept)')

            conn.commit()
        finally:
            conn.close()

    def _get_optimized_connection(self) -> sqlite3.Connection:
        """Get a thread-local cached connection."""
        conn = getattr(self._thread_local_conn, 'conn', None)
        if conn is not None:
            try:
                conn.execute('SELECT 1')  # Liveness check
                return conn
            except Exception:
                conn = None  # Stale — re-open

        for attempt in range(5):
            try:
                conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
                optimize_sqlite_connection(conn)
                self._thread_local_conn.conn = conn
                return conn
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < 4:
                    time.sleep((2 ** attempt) * 0.1)
                    continue
                raise
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        self._thread_local_conn.conn = conn
        return conn

    def _load_cache(self):
        """Load top memories into cache."""
        try:
            conn = self._get_optimized_connection()
            c = conn.cursor()
            
            # Load top memories
            c.execute('SELECT query_hash, response FROM memory ORDER BY access_count DESC LIMIT 10000')
            for row in c:
                self.memory_cache[row[0]] = row[1]

            # Load pattern weights
            c.execute('SELECT pattern, weight FROM patterns ORDER BY weight DESC LIMIT 5000')
            for row in c:
                self.pattern_weights[row[0]] = row[1]

            # Load knowledge graph
            c.execute('SELECT concept, related_concept, strength FROM knowledge ORDER BY strength DESC LIMIT 20000')
            for row in c:
                self.knowledge_graph[row[0]].append((row[1], row[2]))

            # Load meta-learning strategies
            c.execute('SELECT query_pattern, strategy_used, success_score FROM meta_learning ORDER BY success_score DESC')
            self.meta_strategies = {row[0]: (row[1], row[2]) for row in c.fetchall()}

            # Load query rewrite patterns
            c.execute('SELECT original_pattern, improved_pattern FROM query_rewrites WHERE success_rate > 0.6')
            self.query_rewrites = {row[0]: row[1] for row in c.fetchall()}

        except Exception:
            pass  # Start fresh if loading fails

    def persist_clusters(self) -> Dict[str, int]:
        """Persist concept clusters to database."""
        persisted = {}
        try:
            conn = self._get_optimized_connection()
            c = conn.cursor()
            
            for cluster_name, members in self.concept_clusters.items():
                if not members:
                    continue
                    
                members_blob = pickle.dumps(list(members))
                representative = members[0] if members else ""
                member_count = len(members)
                
                c.execute('''INSERT OR REPLACE INTO concept_clusters 
                            (cluster_name, members, representative, member_count, created_at, updated_at)
                            VALUES (?, ?, ?, ?, datetime('now'), datetime('now'))''',
                         (cluster_name, members_blob, representative, member_count))
                persisted[cluster_name] = member_count
            
            conn.commit()
        except Exception:
            pass
        
        return persisted

    def optimize_storage(self) -> Dict[str, Any]:
        """Optimize storage by removing duplicates and compressing."""
        stats = {'removed_duplicates': 0, 'compressed': 0}
        try:
            conn = self._get_optimized_connection()
            c = conn.cursor()
            
            # Remove duplicate patterns
            c.execute('DELETE FROM patterns WHERE rowid NOT IN (SELECT MAX(rowid) FROM patterns GROUP BY pattern)')
            stats['removed_duplicates'] = c.rowcount
            
            # Vacuum to reclaim space
            c.execute('VACUUM')
            stats['compressed'] = 1
            
            conn.commit()
        except Exception:
            pass
        
        return stats

    def temporal_decay(self):
        """Apply temporal decay to memory weights."""
        try:
            conn = self._get_optimized_connection()
            c = conn.cursor()
            
            # Decay old patterns
            c.execute('''UPDATE patterns SET weight = weight * 0.99 
                        WHERE last_used < datetime('now', '-7 days')''')
            
            # Decay old knowledge connections
            c.execute('''UPDATE knowledge SET strength = strength * 0.99 
                        WHERE strength > 0.1''')
            
            conn.commit()
        except Exception:
            pass
