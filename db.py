# L104 Sovereign Node — Database Helpers
# SQLite utility functions for the memory and ramnode databases.

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import MEMORY_DB_PATH, RAMNODE_DB_PATH, UTC

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# MEMORY DB
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def _memory_conn():
    """Context manager for SQLite memory database connections."""
    conn = sqlite3.connect(MEMORY_DB_PATH, check_same_thread=False)
    try:
        yield conn
    finally:
        conn.close()


def _init_memory_db() -> None:
    """Ensure the memory table exists."""
    with _memory_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def _memory_upsert(key: str, value: str) -> None:
    """Insert or update a key-value pair in the memory database."""
    _init_memory_db()
    with _memory_conn() as conn:
        conn.execute(
            """
            INSERT INTO memory(key, value, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value=excluded.value,
                created_at=excluded.created_at
            """,
            (key, value, datetime.now(UTC).isoformat()),
        )
        conn.commit()


def _memory_get(key: str) -> Optional[str]:
    """Retrieve a value from the memory database by key."""
    _init_memory_db()
    with _memory_conn() as conn:
        cur = conn.execute(
            "SELECT value FROM memory WHERE key = ? ORDER BY created_at DESC LIMIT 1",
            (key,),
        )
        row = cur.fetchone()
        return row[0] if row else None


def _memory_list(limit: int = 100) -> List[Dict[str, Any]]:
    """List recent memory entries ordered by creation time."""
    _init_memory_db()
    with _memory_conn() as conn:
        cur = conn.execute(
            "SELECT key, value, created_at FROM memory ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [{"key": r[0], "value": r[1], "created_at": r[2]} for r in cur.fetchall()]


# ─────────────────────────────────────────────────────────────────────────────
# RAMNODE DB
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def _ramnode_conn():
    """Context manager for SQLite ramnode database connections."""
    conn = sqlite3.connect(RAMNODE_DB_PATH, check_same_thread=False)
    try:
        yield conn
    finally:
        conn.close()


def _init_ramnode_db() -> None:
    """Ensure the ramnode table exists."""
    with _ramnode_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ramnode (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def _ramnode_upsert(key: str, value: str) -> None:
    """Insert or update a key-value pair in the ramnode database."""
    with _ramnode_conn() as conn:
        conn.execute(
            """
            INSERT INTO ramnode(key, value, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value=excluded.value,
                created_at=excluded.created_at
            """,
            (key, value, datetime.now(UTC).isoformat()),
        )
        conn.commit()


def _ramnode_get(key: str) -> Optional[str]:
    """Retrieve a value from the ramnode database by key."""
    with _ramnode_conn() as conn:
        cur = conn.execute(
            "SELECT value FROM ramnode WHERE key = ? ORDER BY created_at DESC LIMIT 1",
            (key,),
        )
        row = cur.fetchone()
        return row[0] if row else None


def _ramnode_list(limit: int = 100) -> List[Dict[str, Any]]:
    """List recent ramnode entries ordered by creation time."""
    with _ramnode_conn() as conn:
        cur = conn.execute(
            "SELECT key, value, created_at FROM ramnode ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [{"key": r[0], "value": r[1], "created_at": r[2]} for r in cur.fetchall()]


# ─────────────────────────────────────────────────────────────────────────────
# NODE LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def _log_node(entry: Dict[str, Any]) -> None:
    """Append a timestamped JSON log entry to node.log."""
    try:
        entry["ts"] = datetime.now(UTC).isoformat()
        with open("node.log", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# JSONL LOADER
# ─────────────────────────────────────────────────────────────────────────────

def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load and parse a JSONL file into a list of dicts."""
    p = Path(path)
    if not p.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for raw in p.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            rows.append(json.loads(raw))
        except json.JSONDecodeError:
            _log_node({"tag": "jsonl_error", "path": path})
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# BACKWARD-COMPATIBILITY PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def memory_init() -> None:
    """Backward-compatible public initializer for memory database."""
    _init_memory_db()


def memory_upsert(key: str, value: str) -> None:
    """Backward-compatible public upsert for memory database."""
    _memory_upsert(key, value)


def memory_get(key: str) -> Optional[str]:
    """Backward-compatible public getter for memory database."""
    return _memory_get(key)


def memory_list(limit: int = 100) -> List[Dict[str, Any]]:
    """Backward-compatible public lister for memory database."""
    return _memory_list(limit)


def ramnode_init() -> None:
    """Backward-compatible public initializer for ramnode database."""
    _init_ramnode_db()


def ramnode_upsert(key: str, value: str) -> None:
    """Backward-compatible public upsert for ramnode database."""
    _init_ramnode_db()
    _ramnode_upsert(key, value)


def ramnode_get(key: str) -> Optional[str]:
    """Backward-compatible public getter for ramnode database."""
    _init_ramnode_db()
    return _ramnode_get(key)


def ramnode_list(limit: int = 100) -> List[Dict[str, Any]]:
    """Backward-compatible public lister for ramnode database."""
    _init_ramnode_db()
    return _ramnode_list(limit)
