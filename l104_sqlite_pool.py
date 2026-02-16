VOID_CONSTANT = 1.0416180339887497
ZENITH_HZ = 3887.8
UUC = 2402.792541
#!/usr/bin/env python3
# [EVO_54_PIPELINE] TRANSCENDENT_COGNITION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612 :: GROVER=4.236
"""
L104 SQLite Connection Pool — Thread-safe, async-compatible connection management.

Replaces ad-hoc sqlite3.connect() calls with a bounded pool that:
- Reuses connections instead of creating new ones per operation
- Enforces WAL mode and optimal PRAGMAs on every connection
- Provides context managers for both sync and async usage
- Thread-safe with proper locking

Usage:
    pool = SQLitePool("mydb.db", max_connections=8)

    # Sync context manager
    with pool.connection() as conn:
        conn.execute("SELECT 1")

    # Async wrapper (runs in thread executor)
    async def query():
        result = await pool.async_execute("SELECT 1")
"""

import os
import sqlite3
import asyncio
import threading
import queue
import functools
from contextlib import contextmanager
from typing import Any, Optional, List, Tuple

from l104_logging import get_logger

logger = get_logger("SQLITE_POOL")


class SQLitePool:
    """Thread-safe SQLite connection pool with WAL mode and configurable PRAGMAs."""

    def __init__(
        self,
        db_path: str,
        max_connections: int = 8,
        wal_mode: bool = True,
        journal_size_limit: int = 67_108_864,  # 64 MB
        cache_size_pages: int = -8000,          # ~32 MB negative = KB
        busy_timeout_ms: int = 5000,
    ):
        self.db_path = db_path
        self.max_connections = max_connections
        self._wal_mode = wal_mode
        self._journal_size_limit = journal_size_limit
        self._cache_size = cache_size_pages
        self._busy_timeout = busy_timeout_ms
        self._pool: queue.Queue[sqlite3.Connection] = queue.Queue(maxsize=max_connections)
        self._lock = threading.Lock()
        self._created = 0
        self._closed = False

        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

        logger.info("pool_created", db=db_path, max_conn=max_connections)

    def _make_connection(self) -> sqlite3.Connection:
        """Create a new connection with optimal PRAGMAs."""
        conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=self._busy_timeout / 1000,
        )
        conn.row_factory = sqlite3.Row
        if self._wal_mode:
            conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"PRAGMA journal_size_limit={self._journal_size_limit}")
        conn.execute(f"PRAGMA cache_size={self._cache_size}")
        conn.execute(f"PRAGMA busy_timeout={self._busy_timeout}")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA mmap_size=268435456")  # 256 MB mmap
        return conn

    @contextmanager
    def connection(self):
        """Acquire a connection from the pool (context manager).

        Yields:
            sqlite3.Connection configured with WAL + PRAGMAs.
        """
        conn = self._acquire()
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        else:
            conn.commit()
        finally:
            self._release(conn)

    def _acquire(self) -> sqlite3.Connection:
        """Get a connection: reuse from pool or create new if under limit."""
        try:
            return self._pool.get_nowait()
        except queue.Empty:
            pass

        with self._lock:
            if self._created < self.max_connections:
                conn = self._make_connection()
                self._created += 1
                return conn

        # Pool exhausted — block until one is returned
        return self._pool.get(timeout=self._busy_timeout / 1000)

    def _release(self, conn: sqlite3.Connection):
        """Return a connection to the pool."""
        if self._closed:
            conn.close()
            return
        try:
            self._pool.put_nowait(conn)
        except queue.Full:
            conn.close()
            with self._lock:
                self._created -= 1

    async def async_execute(
        self,
        sql: str,
        params: tuple = (),
        fetch: str = "all",
    ) -> Any:
        """Execute SQL in a thread executor (non-blocking for async code).

        Args:
            sql: SQL statement.
            params: Bind parameters.
            fetch: "all", "one", "none", or "rowcount".

        Returns:
            Query results based on fetch mode.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(
            self._sync_execute, sql, params, fetch
        ))

    def _sync_execute(self, sql: str, params: tuple, fetch: str) -> Any:
        with self.connection() as conn:
            cursor = conn.execute(sql, params)
            if fetch == "one":
                return cursor.fetchone()
            elif fetch == "all":
                return cursor.fetchall()
            elif fetch == "rowcount":
                return cursor.rowcount
            return None

    async def async_executemany(self, sql: str, param_seq: List[tuple]) -> int:
        """Execute many in a thread executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(
            self._sync_executemany, sql, param_seq
        ))

    def _sync_executemany(self, sql: str, param_seq: List[tuple]) -> int:
        with self.connection() as conn:
            cursor = conn.executemany(sql, param_seq)
            return cursor.rowcount

    def close(self):
        """Close all pooled connections."""
        self._closed = True
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except queue.Empty:
                break
        logger.info("pool_closed", db=self.db_path)

    def __del__(self):
        if not self._closed:
            self.close()
